#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <csignal>
#include <atomic>
#include <iostream>
#include <unistd.h>
#include <queue>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <cmath>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "awi_track.hpp"

/*--------------------------------------*/
// #define PIC_FULL_WIDTH 4608
// #define PIC_FULL_HEIGHT 1440
#define PIC_FULL_WIDTH 2560
#define PIC_FULL_HEIGHT 1440
#define ALG_CROP_WIDTH 1280
#define ALG_CROP_HEIGHT 736
#define VALID_TOP 205  //最小155
#define VALID_BOTTOM 1085  // 1440 - 155 = 1285

// ===== 全局退出标志 =====
static std::atomic<bool> g_exit{false};

// ===== 信号处理（Ctrl+C）=====
void signal_handler(int signo)
{
    if (signo == SIGINT) {
        g_exit.store(true);
    }
}

int endswith(const char *str, const char *suffix)
{
    if (!str || !suffix)
        return 0;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr)
        return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

/*------------------------------------------------*/
// 阻塞式 Buffer Pool：无可用 buffer 时等待
class ImageBufferPool {
public:
    ImageBufferPool(size_t count, size_t buf_size)
        : buf_size_(buf_size), total_count_(count)
    {
        for (size_t i = 0; i < count; ++i) {
            image_buffer_t buf{};
            buf.size = buf_size_;
            buf.virt_addr = (unsigned char*)malloc(buf_size_);
            buf.fd = -1;
            free_queue_.push(buf);
        }
    }

    ~ImageBufferPool()
    {
        while (!free_queue_.empty()) {
            auto& buf = free_queue_.front();
            free_queue_.pop();
            if (buf.virt_addr) {
                free(buf.virt_addr);
            }
        }
    }

    ImageBufferPool(const ImageBufferPool&) = delete;
    ImageBufferPool& operator=(const ImageBufferPool&) = delete;

    bool acquire(image_buffer_t& out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return !free_queue_.empty() || stop_;
        });

        if (stop_ && free_queue_.empty()) {
            return false;
        }

        out = free_queue_.front();
        free_queue_.pop();
        return true;
    }

    bool try_acquire(image_buffer_t& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_queue_.empty())
            return false;

        out = free_queue_.front();
        free_queue_.pop();
        return true;
    }

    void release(const image_buffer_t& buf)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_queue_.push(buf);
        cv_.notify_one();
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        cv_.notify_all();
    }

    size_t available() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_queue_.size();
    }

private:
    size_t buf_size_;
    size_t total_count_;
    std::queue<image_buffer_t> free_queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_{false};
};

/*------------------------------------------------*/
/**
 * 检测区域管理器 - 管理裁剪检测和全图检测的切换
 */
class DetectionAreaManager {
public:
    enum class Mode {
        CROP_DETECT,    // 裁剪区域检测（正常模式）
        FULLFRAME_DETECT // 全图检测（搜索模式）
    };

    DetectionAreaManager(int lost_threshold = 15, int found_threshold = 3)
        : lost_threshold_(lost_threshold),    // 连续丢失多少帧后切换到全图
          found_threshold_(found_threshold),  // 全图模式下连续找到多少帧后切回裁剪
          consecutive_lost_(0),
          consecutive_found_(0),
          current_mode_(Mode::CROP_DETECT)
    {
    }

    // 报告本帧是否检测到目标
    void report_detection(bool found)
    {
        if (found) {
            consecutive_lost_ = 0;
            consecutive_found_++;
            
            // 全图模式下，连续找到目标，切回裁剪模式
            if (current_mode_ == Mode::FULLFRAME_DETECT && 
                consecutive_found_ >= found_threshold_) {
                current_mode_ = Mode::CROP_DETECT;
                consecutive_found_ = 0;
                printf("[DetectionArea] Switch to CROP_DETECT mode\n");
            }
        } else {
            consecutive_found_ = 0;
            consecutive_lost_++;
            
            // 裁剪模式下，连续丢失目标，切到全图模式
            if (current_mode_ == Mode::CROP_DETECT && 
                consecutive_lost_ >= lost_threshold_) {
                current_mode_ = Mode::FULLFRAME_DETECT;
                consecutive_lost_ = 0;
                printf("[DetectionArea] Switch to FULLFRAME_DETECT mode (lost %d frames)\n", 
                       lost_threshold_);
            }
        }
    }

    Mode get_mode() const { return current_mode_; }
    
    bool is_fullframe_mode() const { 
        return current_mode_ == Mode::FULLFRAME_DETECT; 
    }
    
    int get_consecutive_lost() const { return consecutive_lost_; }

    // 强制切换模式（调试用）
    void force_mode(Mode mode)
    {
        current_mode_ = mode;
        consecutive_lost_ = 0;
        consecutive_found_ = 0;
    }

private:
    int lost_threshold_;       // 丢失阈值
    int found_threshold_;      // 找回阈值
    int consecutive_lost_;     // 连续丢失帧数
    int consecutive_found_;    // 连续找到帧数
    Mode current_mode_;
};

/*------------------------------------------------*/
/**
 * 丝滑运镜控制器 - 支持全图检测模式
 */
class SmoothCameraController {
public:
    SmoothCameraController(int img_w, int img_h, int crop_w, int crop_h,
                           int valid_top = 0, int valid_bottom = -1)
        : img_w_(img_w),
          img_h_(img_h),
          crop_w_(crop_w),
          crop_h_(crop_h)
    {
        valid_top_ = valid_top;
        valid_bottom_ = (valid_bottom < 0) ? img_h : valid_bottom;
        valid_height_ = valid_bottom_ - valid_top_;
        
        cx_ = img_w_ * 0.5f;
        cy_ = valid_top_ + valid_height_ * 0.5f;
        
        vx_ = 0.0f;
        vy_ = 0.0f;
        
        target_cx_ = cx_;
        target_cy_ = cy_;
        
        target_vx_ = 0.0f;
        target_vy_ = 0.0f;
        last_target_cx_ = cx_;
        last_target_cy_ = cy_;
        
        update_rect();
        
        printf("[Camera] Valid region: y=[%d, %d], height=%d\n", 
               valid_top_, valid_bottom_, valid_height_);
    }

    const image_rect_t& get_rect() const
    {
        return rect_;
    }

    // 每帧调用：更新相机位置 + 裁剪
    void get_crop_window(image_buffer_t* src, image_buffer_t* dst)
    {
        update_camera_position();
        draw_crop_rect(src);
        crop_image(src, dst);
    }

    // 仅更新相机位置和绘制，不做裁剪（全图检测模式下使用）
    void update_and_draw_only(image_buffer_t* src)
    {
        update_camera_position();
        draw_crop_rect(src);
    }

    // 根据当前裁剪框对全图进行裁剪输出
    void crop_current_window(image_buffer_t* src, image_buffer_t* dst)
    {
        crop_image(src, dst);
    }

    // 收到跟踪目标时调用（原图坐标）
    void update_by_target(int xmin, int ymin, int xmax, int ymax)
    {
        float new_target_cx = 0.5f * (xmin + xmax);
        float new_target_cy = 0.5f * (ymin + ymax);
        
        target_vx_ = velocity_smooth_ * (new_target_cx - last_target_cx_) 
                   + (1.0f - velocity_smooth_) * target_vx_;
        target_vy_ = velocity_smooth_ * (new_target_cy - last_target_cy_) 
                   + (1.0f - velocity_smooth_) * target_vy_;
        
        last_target_cx_ = target_cx_;
        last_target_cy_ = target_cy_;
        
        target_cx_ = new_target_cx + target_vx_ * prediction_frames_;
        target_cy_ = new_target_cy + target_vy_ * prediction_frames_;
        
        limit_target();
        
        frames_without_target_ = 0;
        has_target_ = true;
    }

    // 全图检测结果转换为原图坐标后更新（全图模式下使用）
    // 参数已经是原图坐标，直接调用 update_by_target
    void update_by_fullframe_target(int xmin, int ymin, int xmax, int ymax)
    {
        update_by_target(xmin, ymin, xmax, ymax);
    }

    // 裁剪图坐标转原图坐标
    void crop_to_fullframe(int crop_x, int crop_y, int& full_x, int& full_y) const
    {
        full_x = rect_.left + crop_x;
        full_y = rect_.top + crop_y;
    }

    // 原图坐标转裁剪图坐标
    bool fullframe_to_crop(int full_x, int full_y, int& crop_x, int& crop_y) const
    {
        crop_x = full_x - rect_.left;
        crop_y = full_y - rect_.top;
        
        // 检查是否在裁剪框内
        return (crop_x >= 0 && crop_x < crop_w_ && 
                crop_y >= 0 && crop_y < crop_h_);
    }

    void mark_no_target()
    {
        frames_without_target_++;
        
        if (frames_without_target_ > return_to_center_delay_) {
            float center_x = img_w_ * 0.5f;
            float center_y = valid_top_ + valid_height_ * 0.5f;
            
            target_cx_ = target_cx_ + (center_x - target_cx_) * return_to_center_speed_;
            target_cy_ = target_cy_ + (center_y - target_cy_) * return_to_center_speed_;
        }
    }

    // 获取图像尺寸信息
    int get_img_width() const { return img_w_; }
    int get_img_height() const { return img_h_; }
    int get_crop_width() const { return crop_w_; }
    int get_crop_height() const { return crop_h_; }

private:
    int img_w_, img_h_;
    int crop_w_, crop_h_;
    image_rect_t rect_;
    
    int valid_top_;
    int valid_bottom_;
    int valid_height_;

    float cx_, cy_;
    float vx_, vy_;
    
    float target_cx_, target_cy_;
    float target_vx_, target_vy_;
    float last_target_cx_, last_target_cy_;
    
    bool has_target_{false};
    int frames_without_target_{0};

    const float stiffness_ = 0.02f;
    const float damping_ = 0.5f;
    const float max_speed_ = 20.0f;
    const float max_accel_ = 1.5f;
    const float dead_zone_ratio_ = 0.20f;
    const float prediction_frames_ = 2.0f;
    const float velocity_smooth_ = 0.25f;
    const int return_to_center_delay_ = 60;
    const float return_to_center_speed_ = 0.01f;

private:
    void update_camera_position()
    {
        float dx = target_cx_ - cx_;
        float dy = target_cy_ - cy_;
        
        float dead_zone_x = crop_w_ * dead_zone_ratio_;
        float dead_zone_y = crop_h_ * dead_zone_ratio_;
        
        if (std::fabs(dx) < dead_zone_x && std::fabs(dy) < dead_zone_y) {
            vx_ *= (1.0f - damping_ * 0.5f);
            vy_ *= (1.0f - damping_ * 0.5f);
        } else {
            float fx = stiffness_ * dx;
            float fy = stiffness_ * dy;
            
            fx -= damping_ * vx_;
            fy -= damping_ * vy_;
            
            float accel = std::sqrt(fx * fx + fy * fy);
            if (accel > max_accel_) {
                float scale = max_accel_ / accel;
                fx *= scale;
                fy *= scale;
            }
            
            vx_ += fx;
            vy_ += fy;
        }
        
        float speed = std::sqrt(vx_ * vx_ + vy_ * vy_);
        if (speed > max_speed_) {
            float scale = max_speed_ / speed;
            vx_ *= scale;
            vy_ *= scale;
        }
        
        cx_ += vx_;
        cy_ += vy_;
        
        limit_center();
        update_rect();
    }

    void limit_target()
    {
        target_cx_ = std::max(crop_w_ * 0.5f,
                     std::min(target_cx_, img_w_ - crop_w_ * 0.5f));
        
        float min_cy = valid_top_ + crop_h_ * 0.5f;
        float max_cy = valid_bottom_ - crop_h_ * 0.5f;
        target_cy_ = std::max(min_cy, std::min(target_cy_, max_cy));
    }

    void limit_center()
    {
        cx_ = std::max(crop_w_ * 0.5f,
              std::min(cx_, img_w_ - crop_w_ * 0.5f));
        
        float min_cy = valid_top_ + crop_h_ * 0.5f;
        float max_cy = valid_bottom_ - crop_h_ * 0.5f;
        cy_ = std::max(min_cy, std::min(cy_, max_cy));
    }

    void update_rect()
    {
        rect_.left   = static_cast<int>(cx_ - crop_w_ * 0.5f);
        rect_.top    = static_cast<int>(cy_ - crop_h_ * 0.5f);
        rect_.right  = rect_.left + crop_w_;
        rect_.bottom = rect_.top + crop_h_;

        limit_rect();
    }

    void limit_rect()
    {
        if (rect_.left < 0) {
            rect_.left = 0;
            rect_.right = crop_w_;
        }
        if (rect_.right > img_w_) {
            rect_.right = img_w_;
            rect_.left = img_w_ - crop_w_;
        }
        
        if (rect_.top < valid_top_) {
            rect_.top = valid_top_;
            rect_.bottom = valid_top_ + crop_h_;
        }
        if (rect_.bottom > valid_bottom_) {
            rect_.bottom = valid_bottom_;
            rect_.top = valid_bottom_ - crop_h_;
        }
    }

    void draw_crop_rect(image_buffer_t* img)
    {
        draw_rectangle(img,
            rect_.left,
            rect_.top,
            crop_w_,
            crop_h_,
            COLOR_RED,
            3);
    }

    void crop_image(image_buffer_t *src, image_buffer_t *dst)
    {
        if (!src || !dst)
            return;

        image_rect_t box = rect_;
        image_rect_t real_crop_rect;

        int ret = crop_alg_image(
            src,
            dst,
            box,
            &real_crop_rect,
            ALG_CROP_WIDTH,
            ALG_CROP_HEIGHT);

        if (ret != 0) {
            printf("crop_alg_image failed, ret=%d\n", ret);
        }
    }
};

/*---------------------------------------------------*/
// 阻塞式帧队列
class FrameQueue {
public:
    FrameQueue(size_t max_size)
        : max_size_(max_size) {}

    void push(image_buffer_t& buf)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [&] {
            return queue_.size() < max_size_ || stop_;
        });

        if (stop_) {
            return;
        }

        queue_.push_back(buf);
        cv_not_empty_.notify_one();
    }

    bool pop(image_buffer_t& out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [&] {
            return !queue_.empty() || stop_;
        });

        if (queue_.empty()) {
            return false;
        }

        out = queue_.front();
        queue_.pop_front();
        cv_not_full_.notify_one();
        return true;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    size_t max_size_;
    std::deque<image_buffer_t> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    bool stop_{false};
};

/*--------------------------------------------- */
void producer_thread(FrameQueue& fq, ImageBufferPool& pool, const char *frames_dir)
{
    std::vector<std::string> file_list;
    
    DIR *dir = opendir(frames_dir);
    if (dir == nullptr) {
        printf("[Producer] Failed to open directory: %s\n", frames_dir);
        return;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (endswith(entry->d_name, ".jpg") || endswith(entry->d_name, ".png")) {
            file_list.push_back(entry->d_name);
        }
    }
    closedir(dir);

    if (file_list.empty()) {
        printf("[Producer] No image files found in directory: %s\n", frames_dir);
        return;
    }

    std::sort(file_list.begin(), file_list.end());
    printf("[Producer] Found %zu frames to process\n", file_list.size());

    size_t i = 0;
    size_t total_frames = file_list.size();

    while (!g_exit.load() && i < total_frames) {
        image_buffer_t buf = {0};
        if (!pool.acquire(buf)) {
            printf("[Producer] Pool stopped, exiting\n");
            break;
        }

        char img_path[256];
        snprintf(img_path, sizeof(img_path), "%s/%s", frames_dir, file_list[i].c_str());

        if (read_image(img_path, &buf) != 0) {
            printf("[Producer] read_image failed: %s\n", img_path);
            pool.release(buf);
            i++;
            continue;
        }

        fq.push(buf);
        i++;
        
        if (i % 100 == 0 || i == total_frames) {
            printf("[Producer] Progress: %zu / %zu frames read (%.1f%%)\n", 
                   i, total_frames, 100.0 * i / total_frames);
        }
    }

    printf("[Producer] Finished reading all %zu frames\n", i);
}

/*------------------------------------------------*/
/**
 * 球目标筛选器 - 排除干扰球，锁定目标球
 */
class BallSelector {
public:
    struct BallCandidate {
        T_DetectObject det;
        float distance_to_tracker;  // 与当前跟踪的距离
        bool is_static;             // 是否为静止球
        int static_frames;          // 静止帧数
        float priority_score;       // 综合优先级分数
    };

    BallSelector(int valid_left, int valid_right, int valid_top, int valid_bottom)
        : valid_left_(valid_left),
          valid_right_(valid_right),
          valid_top_(valid_top),
          valid_bottom_(valid_bottom)
    {
    }

    /**
     * 从多个检测结果中选择目标球
     * @param detections 所有球的检测结果
     * @param tracker_prediction 跟踪器预测位置（如果有）
     * @param has_tracker 是否有活跃的跟踪器
     * @return 选中的目标球，如果没有合适的返回 nullptr
     */
    const T_DetectObject* select_target_ball(
        const std::vector<T_DetectObject>& detections,
        const T_TrackObject* tracker_prediction,
        bool has_tracker)
    {
        if (detections.empty()) {
            return nullptr;
        }

        // 如果只有一个检测，直接返回（如果在有效区域内）
        if (detections.size() == 1) {
            if (is_in_valid_region(detections[0])) {
                update_history(detections[0]);
                return &detections[0];
            }
            return nullptr;
        }

        // 多个检测，需要筛选
        std::vector<BallCandidate> candidates;
        
        for (const auto& det : detections) {
            BallCandidate candidate;
            candidate.det = det;
            candidate.distance_to_tracker = 1e9f;
            candidate.is_static = false;
            candidate.static_frames = 0;
            candidate.priority_score = 0.0f;

            // 第1层：区域过滤
            if (!is_in_valid_region(det)) {
                continue;  // 跳过有效区域外的球
            }

            // 第2层：静止球检测
            check_static_ball(det, candidate);
            if (candidate.is_static && candidate.static_frames > static_threshold_) {
                continue;  // 跳过静止超过阈值的球
            }

            // 第3层：计算与跟踪器的距离
            if (has_tracker && tracker_prediction != nullptr) {
                candidate.distance_to_tracker = calculate_distance(det, *tracker_prediction);
            }

            // 计算综合优先级分数
            candidate.priority_score = calculate_priority(candidate, has_tracker);

            candidates.push_back(candidate);
        }

        if (candidates.empty()) {
            return nullptr;
        }

        // 按优先级排序，选择最高的
        std::sort(candidates.begin(), candidates.end(),
                  [](const BallCandidate& a, const BallCandidate& b) {
                      return a.priority_score > b.priority_score;
                  });

        // 更新历史记录
        update_history(candidates[0].det);

        // 返回原始 detections 中对应的指针
        for (const auto& det : detections) {
            if (det.xmin == candidates[0].det.xmin &&
                det.ymin == candidates[0].det.ymin) {
                return &det;
            }
        }

        return nullptr;
    }

    /**
     * 重置历史记录（场景切换时调用）
     */
    void reset() {
        ball_history_.clear();
    }

private:
    // 有效区域边界
    int valid_left_;
    int valid_right_;
    int valid_top_;
    int valid_bottom_;

    // 静止检测阈值
    static constexpr int static_threshold_ = 30;      // 静止超过30帧视为干扰球
    static constexpr float static_move_threshold_ = 10.0f;  // 移动小于10像素视为静止

    // 历史检测记录（用于静止球检测）
    struct BallHistory {
        float cx, cy;           // 中心位置
        int static_count;       // 静止计数
        int last_seen_frame;    // 最后出现帧
    };
    std::vector<BallHistory> ball_history_;
    int current_frame_ = 0;

    /**
     * 检查是否在有效区域内
     */
    bool is_in_valid_region(const T_DetectObject& det) {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        return (cx >= valid_left_ && cx <= valid_right_ &&
                cy >= valid_top_ && cy <= valid_bottom_);
    }

    /**
     * 检查是否为静止球
     */
    void check_static_ball(const T_DetectObject& det, BallCandidate& candidate) {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;

        // 查找历史记录中是否有相近位置的球
        for (auto& hist : ball_history_) {
            float dist = std::sqrt((cx - hist.cx) * (cx - hist.cx) + 
                                   (cy - hist.cy) * (cy - hist.cy));
            
            if (dist < static_move_threshold_) {
                // 位置相近，增加静止计数
                hist.static_count++;
                hist.last_seen_frame = current_frame_;
                candidate.is_static = true;
                candidate.static_frames = hist.static_count;
                return;
            }
        }

        // 没有找到匹配的历史，这是新位置
        candidate.is_static = false;
        candidate.static_frames = 0;
    }

    /**
     * 更新历史记录
     */
    void update_history(const T_DetectObject& det) {
        current_frame_++;
        
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;

        // 清理过期的历史记录
        ball_history_.erase(
            std::remove_if(ball_history_.begin(), ball_history_.end(),
                [this](const BallHistory& h) {
                    return (current_frame_ - h.last_seen_frame) > 60;  // 60帧未出现则移除
                }),
            ball_history_.end());

        // 查找并更新或添加新记录
        for (auto& hist : ball_history_) {
            float dist = std::sqrt((cx - hist.cx) * (cx - hist.cx) + 
                                   (cy - hist.cy) * (cy - hist.cy));
            if (dist < static_move_threshold_) {
                hist.cx = cx;
                hist.cy = cy;
                hist.last_seen_frame = current_frame_;
                return;
            }
        }

        // 添加新记录
        if (ball_history_.size() < 20) {  // 最多记录20个位置
            ball_history_.push_back({cx, cy, 0, current_frame_});
        }
    }

    /**
     * 计算与跟踪器的距离
     */
    float calculate_distance(const T_DetectObject& det, const T_TrackObject& tracker) {
        float det_cx = (det.xmin + det.xmax) / 2.0f;
        float det_cy = (det.ymin + det.ymax) / 2.0f;
        float trk_cx = (tracker.xmin + tracker.xmax) / 2.0f;
        float trk_cy = (tracker.ymin + tracker.ymax) / 2.0f;
        
        return std::sqrt((det_cx - trk_cx) * (det_cx - trk_cx) + 
                         (det_cy - trk_cy) * (det_cy - trk_cy));
    }

    /**
     * 计算综合优先级分数
     */
    float calculate_priority(const BallCandidate& candidate, bool has_tracker) {
        float score = 0.0f;

        // 置信度分数（0~100）
        score += candidate.det.score * 100.0f;

        // 跟踪器距离分数（有跟踪器时，距离越近分数越高）
        if (has_tracker && candidate.distance_to_tracker < 1e8f) {
            // 距离在 500 像素内给予额外加分
            if (candidate.distance_to_tracker < 500.0f) {
                score += (500.0f - candidate.distance_to_tracker) * 0.5f;  // 最多加 250 分
            }
        }

        // 静止惩罚（静止越久分数越低）
        score -= candidate.static_frames * 2.0f;

        // 位置偏好（靠近画面中心的球优先）
        float cx = (candidate.det.xmin + candidate.det.xmax) / 2.0f;
        float cy = (candidate.det.ymin + candidate.det.ymax) / 2.0f;
        float center_x = (valid_left_ + valid_right_) / 2.0f;
        float center_y = (valid_top_ + valid_bottom_) / 2.0f;
        float dist_to_center = std::sqrt((cx - center_x) * (cx - center_x) + 
                                          (cy - center_y) * (cy - center_y));
        score -= dist_to_center * 0.01f;  // 轻微惩罚远离中心的球

        return score;
    }
};

/*------------------------------------------------*/
/**
 * 图像分割检测器 - 将大图分割成多块分别检测
 */
/*------------------------------------------------*/
/**
 * 图像分割检测器 - 将大图分割成多块分别检测
 */
class SplitDetector {
public:
    struct SplitRegion {
        int x_offset;      // 在原图中的 X 偏移
        int y_offset;      // 在原图中的 Y 偏移
        int width;         // 区域宽度
        int height;        // 区域高度
    };

    // 分割配置
    static constexpr int SPLIT_WIDTH = 2560;       // 每块宽度
    static constexpr int OVERLAP = 256;            // 重叠区域

    /**
     * 检查是否需要分割检测
     */
    static bool needs_split(int img_width) {
        return img_width > SPLIT_WIDTH;
    }

    /**
     * 计算分割区域
     */
    static std::vector<SplitRegion> calculate_split_regions(int img_width, int img_height) {
        std::vector<SplitRegion> regions;

        if (img_width <= SPLIT_WIDTH) {
            regions.push_back({0, 0, img_width, img_height});
            return regions;
        }

        int step = SPLIT_WIDTH - OVERLAP;
        int x = 0;
        
        while (x < img_width) {
            int region_width = std::min(SPLIT_WIDTH, img_width - x);
            
            if (img_width - x < SPLIT_WIDTH && x > 0) {
                x = img_width - SPLIT_WIDTH;
                region_width = SPLIT_WIDTH;
            }
            
            regions.push_back({x, 0, region_width, img_height});
            
            if (x + region_width >= img_width) break;
            x += step;
        }

        return regions;
    }

    /**
     * 裁剪图像区域
     */
    static int crop_region(const image_buffer_t* src, image_buffer_t* dst,
                           const SplitRegion& region) {
        if (!src || !src->virt_addr || !dst) {
            return -1;
        }

        int src_w = src->width;
        int src_h = src->height;
        int channels = 3;

        size_t dst_size = region.width * region.height * channels;
        
        memset(dst, 0, sizeof(image_buffer_t));
        dst->width = region.width;
        dst->height = region.height;
        dst->size = dst_size;
        dst->format = src->format;
        dst->fd = -1;
        dst->virt_addr = (unsigned char*)malloc(dst_size);
        if (!dst->virt_addr) {
            return -1;
        }

        for (int y = 0; y < region.height; y++) {
            int src_y = region.y_offset + y;
            if (src_y >= src_h) break;

            unsigned char* src_row = src->virt_addr + (src_y * src_w + region.x_offset) * channels;
            unsigned char* dst_row = dst->virt_addr + (y * region.width) * channels;
            
            int copy_width = std::min(region.width, src_w - region.x_offset);
            memcpy(dst_row, src_row, copy_width * channels);
        }

        return 0;
    }

    /**
     * 将检测结果坐标还原到原图（加上区域偏移）
     */
    static void restore_to_original(const SplitRegion& region,
                                    int& xmin, int& ymin, int& xmax, int& ymax) {
        xmin += region.x_offset;
        xmax += region.x_offset;
        ymin += region.y_offset;
        ymax += region.y_offset;
    }

    /**
     * 合并多个区域的检测结果（去重）
     */
    static void merge_detections(std::vector<T_DetectObject>& all_detections,
                                  const std::vector<T_DetectObject>& new_detections,
                                  float iou_threshold = 0.5f) {
        for (const auto& new_det : new_detections) {
            bool is_duplicate = false;
            
            for (auto& existing : all_detections) {
                float iou = calculate_iou(existing, new_det);
                if (iou > iou_threshold) {
                    if (new_det.score > existing.score) {
                        existing = new_det;
                    }
                    is_duplicate = true;
                    break;
                }
            }
            
            if (!is_duplicate) {
                all_detections.push_back(new_det);
            }
        }
    }

private:
    static float calculate_iou(const T_DetectObject& a, const T_DetectObject& b) {
        float inter_left = std::max(a.xmin, b.xmin);
        float inter_top = std::max(a.ymin, b.ymin);
        float inter_right = std::min(a.xmax, b.xmax);
        float inter_bottom = std::min(a.ymax, b.ymax);

        if (inter_right <= inter_left || inter_bottom <= inter_top) {
            return 0.0f;
        }

        float inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
        float area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        float area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        float union_area = area_a + area_b - inter_area;

        return inter_area / union_area;
    }
};

//裁切输出版本
void consumer_thread(FrameQueue& fq, ImageBufferPool& pool, const char *model_path, const char *out_dir)
{
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();
    int ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret) {
        printf("[Consumer] init_yolov8_model failed!\n");
        return;
    }

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    system(cmd);

    int frame_count = 0;
    int frame_track_count = 0;
    image_buffer_t src_image = {0};
    image_buffer_t crop_image_buf = {0};
    
    TrackFrame tracker;
    tracker.Init(50);
    
    SmoothCameraController camera(
        PIC_FULL_WIDTH, 
        PIC_FULL_HEIGHT, 
        ALG_CROP_WIDTH, 
        ALG_CROP_HEIGHT,
        VALID_TOP,
        VALID_BOTTOM
    );

    DetectionAreaManager detect_manager(15, 3);

    // 球筛选器
    BallSelector ball_selector(0, PIC_FULL_WIDTH, VALID_TOP, VALID_BOTTOM);

    printf("[Consumer] Detection mode: CROP=%dx%d, FULLFRAME=%dx%d\n",
           ALG_CROP_WIDTH, ALG_CROP_HEIGHT, PIC_FULL_WIDTH, PIC_FULL_HEIGHT);

    while (true) {
        frame_track_count++;
        
        if (src_image.width == 0 && src_image.height == 0) {
            memset(&src_image, 0, sizeof(image_buffer_t));
            if (!fq.pop(src_image)) {
                printf("[Consumer] Queue stopped, exiting\n");
                break;
            }
            
            if (frame_count == 0) {
                printf("[Consumer] Input resolution: %dx%d\n", src_image.width, src_image.height);
                if (SplitDetector::needs_split(src_image.width)) {
                    printf("[Consumer] Split detection enabled (width > %d)\n", SplitDetector::SPLIT_WIDTH);
                }
            }
        }
        
        memset(&crop_image_buf, 0, sizeof(image_buffer_t));

        bool is_fullframe_mode = detect_manager.is_fullframe_mode();
        bool found_ball = false;
        
        std::vector<T_DetectObject> all_ball_detections;
        std::vector<T_DetectObject> ball_detections;
        std::vector<T_TrackObject> track_results;

        if (is_fullframe_mode) {
            // ===== 全图检测模式 =====
            camera.update_and_draw_only(&src_image);
            
            if (SplitDetector::needs_split(src_image.width)) {
                // ===== 分割检测模式（大图）=====
                auto regions = SplitDetector::calculate_split_regions(
                    src_image.width, src_image.height);
                
                printf("[Consumer][FULLFRAME] Split into %zu regions\n", regions.size());

                for (size_t i = 0; i < regions.size(); i++) {
                    const auto& region = regions[i];
                    
                    // 裁剪区域
                    image_buffer_t region_img = {0};
                    ret = SplitDetector::crop_region(&src_image, &region_img, region);
                    if (ret != 0) {
                        printf("[Consumer] Crop region %zu failed\n", i);
                        continue;
                    }

                    // 直接对裁剪区域推理（2560x1440 或更小）
                    object_detect_result_list od_results;
                    inference_yolov8_model(&rknn_app_ctx, &region_img, &od_results);

                    // 处理检测结果
                    std::vector<T_DetectObject> region_detections;
                    for (int j = 0; j < od_results.count; j++) {
                        object_detect_result *det = &(od_results.results[j]);
                        char text[64];
                        snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                        
                        if (strncmp(text, "ball", 4) == 0) {
                            int xmin = det->box.left;
                            int ymin = det->box.top;
                            int xmax = det->box.right;
                            int ymax = det->box.bottom;
                            
                            // 还原到原图坐标（加上区域偏移）
                            SplitDetector::restore_to_original(region, xmin, ymin, xmax, ymax);
                            
                            T_DetectObject obj;
                            obj.cls_id = det->cls_id;
                            obj.score = det->prop;
                            obj.xmin = xmin;
                            obj.ymin = ymin;
                            obj.xmax = xmax;
                            obj.ymax = ymax;
                            region_detections.push_back(obj);
                            
                            printf("[Consumer][FULLFRAME][Region%zu] Found ball @ orig(%d,%d,%d,%d) %.3f\n",
                                   i, xmin, ymin, xmax, ymax, det->prop);
                        }
                    }

                    // 合并检测结果（去重）
                    SplitDetector::merge_detections(all_ball_detections, region_detections);

                    // 释放区域图像
                    if (region_img.virt_addr) {
                        free(region_img.virt_addr);
                        region_img.virt_addr = NULL;
                    }
                }
            } else {
                // ===== 普通全图检测（不需要分割）=====
                object_detect_result_list od_results;
                inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
                
                for (int j = 0; j < od_results.count; j++) {
                    object_detect_result *det = &(od_results.results[j]);
                    char text[64];
                    snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                    
                    if (strncmp(text, "ball", 4) == 0) {
                        T_DetectObject obj;
                        obj.cls_id = det->cls_id;
                        obj.score = det->prop;
                        obj.xmin = det->box.left;
                        obj.ymin = det->box.top;
                        obj.xmax = det->box.right;
                        obj.ymax = det->box.bottom;
                        all_ball_detections.push_back(obj);
                        
                        printf("[Consumer][FULLFRAME] Found ball @ (%d,%d,%d,%d) %.3f\n",
                               det->box.left, det->box.top, det->box.right, det->box.bottom, det->prop);
                    }
                }
            }
            
            // 裁剪输出图像
            camera.crop_current_window(&src_image, &crop_image_buf);

        } else {
            // ===== 裁剪区域检测模式 =====
            camera.get_crop_window(&src_image, &crop_image_buf);

            if (crop_image_buf.virt_addr == NULL) {
                printf("[Consumer] crop_image is NULL\n");
                pool.release(src_image);
                memset(&src_image, 0, sizeof(image_buffer_t));
                continue;
            }

            object_detect_result_list od_results;
            inference_yolov8_model(&rknn_app_ctx, &crop_image_buf, &od_results);

            for (int j = 0; j < od_results.count; j++) {
                object_detect_result *det = &(od_results.results[j]);
                char text[64];
                snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                
                if (strncmp(text, "ball", 4) == 0) {
                    T_DetectObject obj;
                    obj.cls_id = det->cls_id;
                    obj.score = det->prop;
                    // 裁剪图坐标 → 原图坐标
                    obj.xmin = camera.get_rect().left + det->box.left;
                    obj.ymin = camera.get_rect().top + det->box.top;
                    obj.xmax = camera.get_rect().left + det->box.right;
                    obj.ymax = camera.get_rect().top + det->box.bottom;
                    all_ball_detections.push_back(obj);
                }
            }
        }

        // ===== 球筛选：从多个检测中选择目标球 =====
        if (!all_ball_detections.empty()) {
            T_TrackObject tracker_pred;
            bool has_tracker = tracker.HasActiveTrack();
            
            const T_DetectObject* target = ball_selector.select_target_ball(
                all_ball_detections,
                has_tracker ? &tracker_pred : nullptr,
                has_tracker
            );
            
            if (target != nullptr) {
                ball_detections.push_back(*target);
                found_ball = true;
                
                printf("[Consumer] Selected target ball @ (%d,%d,%d,%d) from %zu candidates\n",
                       (int)target->xmin, (int)target->ymin,
                       (int)target->xmax, (int)target->ymax,
                       all_ball_detections.size());
            }
        }

        // ===== 统一的跟踪处理 =====
        if (!ball_detections.empty() || tracker.HasActiveTrack()) {
            tracker.ProcessFrame(frame_track_count, ball_detections, track_results);
        }

        // ===== 统一的绘图处理 =====
        if (crop_image_buf.virt_addr != NULL) {
            // 绘制所有检测到的球（白色细框，调试用）
            for (const auto& det : all_ball_detections) {
                int crop_x1 = det.xmin - camera.get_rect().left;
                int crop_y1 = det.ymin - camera.get_rect().top;
                int crop_x2 = det.xmax - camera.get_rect().left;
                int crop_y2 = det.ymax - camera.get_rect().top;
                
                if (crop_x1 >= 0 && crop_y1 >= 0 && 
                    crop_x2 <= ALG_CROP_WIDTH && crop_y2 <= ALG_CROP_HEIGHT) {
                    draw_rectangle(&crop_image_buf, 
                                   crop_x1, crop_y1,
                                   crop_x2 - crop_x1, crop_y2 - crop_y1,
                                   COLOR_WHITE, 1);
                }
            }

            // 绘制选中的目标球（蓝色粗框）
            for (const auto& det : ball_detections) {
                int crop_x1 = det.xmin - camera.get_rect().left;
                int crop_y1 = det.ymin - camera.get_rect().top;
                int crop_x2 = det.xmax - camera.get_rect().left;
                int crop_y2 = det.ymax - camera.get_rect().top;
                
                if (crop_x1 >= 0 && crop_y1 >= 0 && 
                    crop_x2 <= ALG_CROP_WIDTH && crop_y2 <= ALG_CROP_HEIGHT) {
                    draw_rectangle(&crop_image_buf, 
                                   crop_x1, crop_y1,
                                   crop_x2 - crop_x1, crop_y2 - crop_y1,
                                   COLOR_BLUE, 3);
                    
                    char text[64];
                    snprintf(text, sizeof(text), "TARGET %.1f%%", det.score * 100);
                    draw_text(&crop_image_buf, text, crop_x1, crop_y1 - 20, COLOR_RED, 10);
                }
            }
            
            // 绘制跟踪框（绿色）
            for (const auto& trk : track_results) {
                int crop_x1 = trk.xmin - camera.get_rect().left;
                int crop_y1 = trk.ymin - camera.get_rect().top;
                int crop_x2 = trk.xmax - camera.get_rect().left;
                int crop_y2 = trk.ymax - camera.get_rect().top;
                
                if (crop_x1 >= 0 && crop_y1 >= 0 && 
                    crop_x2 <= ALG_CROP_WIDTH && crop_y2 <= ALG_CROP_HEIGHT) {
                    draw_rectangle(&crop_image_buf, 
                                   crop_x1, crop_y1,
                                   crop_x2 - crop_x1, crop_y2 - crop_y1,
                                   COLOR_GREEN, 3);
                    
                    if (trk.is_predicted) {
                        draw_text(&crop_image_buf, "[PRED]", crop_x1, crop_y1 - 40, COLOR_YELLOW, 10);
                    }
                }
            }
            
            // 调试信息
            char debug_text[128];
            snprintf(debug_text, sizeof(debug_text), "Mode:%s Balls:%zu->%zu Dist:%.1f",
                     is_fullframe_mode ? "FULL" : "CROP",
                     all_ball_detections.size(),
                     ball_detections.size(),
                     tracker.GetLastGatingDistance());
            draw_text(&crop_image_buf, debug_text, 10, 20, COLOR_YELLOW, 12);
        }

        // ===== 更新运镜 =====
        if (!track_results.empty()) {
            auto &t = track_results[0];
            camera.update_by_target(t.xmin, t.ymin, t.xmax, t.ymax);
        }

        // ===== 更新检测模式 =====
        detect_manager.report_detection(found_ball);
        
        if (!found_ball) {
            camera.mark_no_target();
        }

        // ===== 保存输出 =====
        if (crop_image_buf.virt_addr != NULL) {
            char out_path[256];
            snprintf(out_path, sizeof(out_path), "%s/%06d.jpg", out_dir, frame_count);
            write_image(out_path, &crop_image_buf);
        }
        
        frame_count++;
        
        if (frame_count % 100 == 0) {
            printf("[Consumer] Processed %d frames, mode: %s\n", 
                   frame_count,
                   detect_manager.is_fullframe_mode() ? "FULLFRAME" : "CROP");
        }
        
        if (crop_image_buf.virt_addr) {
            free(crop_image_buf.virt_addr);
            crop_image_buf.virt_addr = NULL;
        }
        
        pool.release(src_image);
        memset(&src_image, 0, sizeof(image_buffer_t));
    }

    printf("[Consumer] Finished processing %d frames\n", frame_count);

    deinit_post_process();
    release_yolov8_model(&rknn_app_ctx);
}

//调试版本
// void consumer_thread(FrameQueue& fq, ImageBufferPool& pool, const char *model_path, const char *out_dir)
// {
//     rknn_app_context_t rknn_app_ctx;
//     memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

//     init_post_process();
//     int ret = init_yolov8_model(model_path, &rknn_app_ctx);
//     if (ret) {
//         printf("[Consumer] init_yolov8_model failed!\n");
//         return;
//     }

//     char cmd[256];
//     snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
//     system(cmd);

//     int frame_count = 0;
//     int frame_track_count = 0;
//     image_buffer_t src_image = {0};
//     image_buffer_t crop_image_buf = {0};
    
//     TrackFrame tracker;
//     tracker.Init(50);
    
//     SmoothCameraController camera(
//         PIC_FULL_WIDTH, 
//         PIC_FULL_HEIGHT, 
//         ALG_CROP_WIDTH, 
//         ALG_CROP_HEIGHT,
//         VALID_TOP,
//         VALID_BOTTOM
//     );

//     DetectionAreaManager detect_manager(15, 3);

//     // 球筛选器
//     BallSelector ball_selector(0, PIC_FULL_WIDTH, VALID_TOP, VALID_BOTTOM);

//     printf("[Consumer][DEBUG] Output: FULL IMAGE with crop window\n");
//     printf("[Consumer] Detection mode: CROP=%dx%d, FULLFRAME=%dx%d\n",
//            ALG_CROP_WIDTH, ALG_CROP_HEIGHT, PIC_FULL_WIDTH, PIC_FULL_HEIGHT);

//     while (true) {
//         frame_track_count++;
        
//         if (src_image.width == 0 && src_image.height == 0) {
//             memset(&src_image, 0, sizeof(image_buffer_t));
//             if (!fq.pop(src_image)) {
//                 printf("[Consumer] Queue stopped, exiting\n");
//                 break;
//             }
            
//             if (frame_count == 0) {
//                 printf("[Consumer] Input resolution: %dx%d\n", src_image.width, src_image.height);
//                 if (SplitDetector::needs_split(src_image.width)) {
//                     printf("[Consumer] Split detection enabled (width > %d)\n", SplitDetector::SPLIT_WIDTH);
//                 }
//             }
//         }
        
//         memset(&crop_image_buf, 0, sizeof(image_buffer_t));

//         bool is_fullframe_mode = detect_manager.is_fullframe_mode();
//         bool found_ball = false;
        
//         std::vector<T_DetectObject> all_ball_detections;
//         std::vector<T_DetectObject> ball_detections;
//         std::vector<T_TrackObject> track_results;

//         if (is_fullframe_mode) {
//             // ===== 全图检测模式 =====
//             camera.update_and_draw_only(&src_image);
            
//             if (SplitDetector::needs_split(src_image.width)) {
//                 // ===== 分割检测模式（大图）=====
//                 auto regions = SplitDetector::calculate_split_regions(
//                     src_image.width, src_image.height);

//                 for (size_t i = 0; i < regions.size(); i++) {
//                     const auto& region = regions[i];
                    
//                     image_buffer_t region_img = {0};
//                     ret = SplitDetector::crop_region(&src_image, &region_img, region);
//                     if (ret != 0) {
//                         continue;
//                     }

//                     object_detect_result_list od_results;
//                     inference_yolov8_model(&rknn_app_ctx, &region_img, &od_results);

//                     std::vector<T_DetectObject> region_detections;
//                     for (int j = 0; j < od_results.count; j++) {
//                         object_detect_result *det = &(od_results.results[j]);
//                         char text[64];
//                         snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                        
//                         if (strncmp(text, "ball", 4) == 0) {
//                             int xmin = det->box.left;
//                             int ymin = det->box.top;
//                             int xmax = det->box.right;
//                             int ymax = det->box.bottom;
                            
//                             SplitDetector::restore_to_original(region, xmin, ymin, xmax, ymax);
                            
//                             T_DetectObject obj;
//                             obj.cls_id = det->cls_id;
//                             obj.score = det->prop;
//                             obj.xmin = xmin;
//                             obj.ymin = ymin;
//                             obj.xmax = xmax;
//                             obj.ymax = ymax;
//                             region_detections.push_back(obj);
//                         }
//                     }

//                     SplitDetector::merge_detections(all_ball_detections, region_detections);

//                     if (region_img.virt_addr) {
//                         free(region_img.virt_addr);
//                         region_img.virt_addr = NULL;
//                     }
//                 }
//             } else {
//                 // ===== 普通全图检测 =====
//                 object_detect_result_list od_results;
//                 inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
                
//                 for (int j = 0; j < od_results.count; j++) {
//                     object_detect_result *det = &(od_results.results[j]);
//                     char text[64];
//                     snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                    
//                     if (strncmp(text, "ball", 4) == 0) {
//                         T_DetectObject obj;
//                         obj.cls_id = det->cls_id;
//                         obj.score = det->prop;
//                         obj.xmin = det->box.left;
//                         obj.ymin = det->box.top;
//                         obj.xmax = det->box.right;
//                         obj.ymax = det->box.bottom;
//                         all_ball_detections.push_back(obj);
//                     }
//                 }
//             }
            
//             // 仍然需要裁剪图用于检测（但不输出）
//             camera.crop_current_window(&src_image, &crop_image_buf);

//         } else {
//             // ===== 裁剪区域检测模式 =====
//             camera.get_crop_window(&src_image, &crop_image_buf);

//             if (crop_image_buf.virt_addr == NULL) {
//                 printf("[Consumer] crop_image is NULL\n");
//                 pool.release(src_image);
//                 memset(&src_image, 0, sizeof(image_buffer_t));
//                 continue;
//             }

//             object_detect_result_list od_results;
//             inference_yolov8_model(&rknn_app_ctx, &crop_image_buf, &od_results);

//             for (int j = 0; j < od_results.count; j++) {
//                 object_detect_result *det = &(od_results.results[j]);
//                 char text[64];
//                 snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                
//                 if (strncmp(text, "ball", 4) == 0) {
//                     T_DetectObject obj;
//                     obj.cls_id = det->cls_id;
//                     obj.score = det->prop;
//                     obj.xmin = camera.get_rect().left + det->box.left;
//                     obj.ymin = camera.get_rect().top + det->box.top;
//                     obj.xmax = camera.get_rect().left + det->box.right;
//                     obj.ymax = camera.get_rect().top + det->box.bottom;
//                     all_ball_detections.push_back(obj);
//                 }
//             }
//         }

//         // ===== 球筛选 =====
//         if (!all_ball_detections.empty()) {
//             T_TrackObject tracker_pred;
//             bool has_tracker = tracker.HasActiveTrack();
            
//             const T_DetectObject* target = ball_selector.select_target_ball(
//                 all_ball_detections,
//                 has_tracker ? &tracker_pred : nullptr,
//                 has_tracker
//             );
            
//             if (target != nullptr) {
//                 ball_detections.push_back(*target);
//                 found_ball = true;
//             }
//         }

//         // ===== 跟踪处理 =====
//         if (!ball_detections.empty() || tracker.HasActiveTrack()) {
//             tracker.ProcessFrame(frame_track_count, ball_detections, track_results);
//         }

//         // ===== 在原图上绘制（调试模式）=====
//         {
//             // 1. 绘制裁剪窗口（红色粗框）
//             image_rect_t crop_rect = camera.get_rect();
//             draw_rectangle(&src_image, 
//                            crop_rect.left, crop_rect.top,
//                            crop_rect.right - crop_rect.left, 
//                            crop_rect.bottom - crop_rect.top,
//                            COLOR_RED, 4);

//             // 2. 绘制所有检测到的球（白色细框）
//             for (const auto& det : all_ball_detections) {
//                 draw_rectangle(&src_image, 
//                                det.xmin, det.ymin,
//                                det.xmax - det.xmin, det.ymax - det.ymin,
//                                COLOR_WHITE, 1);
//             }

//             // 3. 绘制选中的目标球（蓝色粗框）
//             for (const auto& det : ball_detections) {
//                 draw_rectangle(&src_image, 
//                                det.xmin, det.ymin,
//                                det.xmax - det.xmin, det.ymax - det.ymin,
//                                COLOR_BLUE, 3);
                
//                 char text[64];
//                 snprintf(text, sizeof(text), "TARGET %.1f%%", det.score * 100);
//                 draw_text(&src_image, text, det.xmin, det.ymin - 20, COLOR_RED, 10);
//             }
            
//             // 4. 绘制跟踪框（绿色）
//             for (const auto& trk : track_results) {
//                 draw_rectangle(&src_image, 
//                                trk.xmin, trk.ymin,
//                                trk.xmax - trk.xmin, trk.ymax - trk.ymin,
//                                COLOR_GREEN, 3);
                
//                 if (trk.is_predicted) {
//                     draw_text(&src_image, "[PRED]", trk.xmin, trk.ymin - 40, COLOR_YELLOW, 10);
//                 }
//             }
            
//             // 5. 调试信息（左上角）
//             char debug_text[128];
//             snprintf(debug_text, sizeof(debug_text), "Mode:%s Balls:%zu->%zu Dist:%.1f",
//                      is_fullframe_mode ? "FULL" : "CROP",
//                      all_ball_detections.size(),
//                      ball_detections.size(),
//                      tracker.GetLastGatingDistance());
//             draw_text(&src_image, debug_text, 10, 30, COLOR_YELLOW, 15);
            
//             // 6. 裁剪窗口位置信息
//             char crop_info[64];
//             snprintf(crop_info, sizeof(crop_info), "Crop:[%d,%d]-[%d,%d]",
//                      crop_rect.left, crop_rect.top, crop_rect.right, crop_rect.bottom);
//             draw_text(&src_image, crop_info, 10, 60, COLOR_RED, 12);
//         }

//         // ===== 更新运镜 =====
//         if (!track_results.empty()) {
//             auto &t = track_results[0];
//             camera.update_by_target(t.xmin, t.ymin, t.xmax, t.ymax);
//         }

//         // ===== 更新检测模式 =====
//         detect_manager.report_detection(found_ball);
        
//         if (!found_ball) {
//             camera.mark_no_target();
//         }

//         // ===== 保存原图（调试模式）=====
//         {
//             char out_path[256];
//             snprintf(out_path, sizeof(out_path), "%s/%06d.jpg", out_dir, frame_count);
//             write_image(out_path, &src_image);  // 输出原图
//         }
        
//         frame_count++;
        
//         if (frame_count % 100 == 0) {
//             printf("[Consumer] Processed %d frames, mode: %s\n", 
//                    frame_count,
//                    detect_manager.is_fullframe_mode() ? "FULLFRAME" : "CROP");
//         }
        
//         // 释放裁剪图内存（仍然需要释放）
//         if (crop_image_buf.virt_addr) {
//             free(crop_image_buf.virt_addr);
//             crop_image_buf.virt_addr = NULL;
//         }
        
//         pool.release(src_image);
//         memset(&src_image, 0, sizeof(image_buffer_t));
//     }

//     printf("[Consumer] Finished processing %d frames\n", frame_count);

//     deinit_post_process();
//     release_yolov8_model(&rknn_app_ctx);
// }
/*-----------------------------------------------*/
int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Usage: %s <model_path> <frames_dir> <out_dir>\n", argv[0]);
        printf("例如： ffmpeg -i test.mp4 frames/frame_%%06d.jpg\n");
        return -1;
    }

    const char *model_path = argv[1];
    const char *frames_dir = argv[2];
    const char *out_dir = argv[3];

    constexpr size_t QUEUE_SIZE = 12;
    constexpr size_t POOL_SIZE  = 16;
    
    // ✅ 修改这一行：使用宏定义，自动适配不同分辨率
    constexpr size_t IMAGE_SIZE = PIC_FULL_WIDTH * PIC_FULL_HEIGHT * 3;
    
    printf("[Main] Image buffer size: %zu bytes (%.2f MB) for %dx%d\n",
           IMAGE_SIZE, IMAGE_SIZE / 1024.0 / 1024.0, PIC_FULL_WIDTH, PIC_FULL_HEIGHT);

    signal(SIGINT, signal_handler);

    ImageBufferPool buffer_pool(POOL_SIZE, IMAGE_SIZE);
    FrameQueue frame_queue(QUEUE_SIZE);

    std::atomic<bool> producer_done{false};

    std::thread producer([&] {
        producer_thread(frame_queue, buffer_pool, frames_dir);
        producer_done.store(true);
        printf("[Producer] Thread exiting, notifying consumer...\n");
        frame_queue.stop();
    });

    std::thread consumer([&] {
        consumer_thread(frame_queue, buffer_pool, model_path, out_dir);
    });

    std::cout << "[Main] Pipeline started (smooth camera mode)\n";
    std::cout << "[Main] Queue size: " << QUEUE_SIZE << ", Pool size: " << POOL_SIZE << "\n";

    int heartbeat_count = 0;
    while (!g_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        heartbeat_count++;
        
        if (heartbeat_count >= 5) {
            heartbeat_count = 0;
            printf("[Main] Heartbeat - Queue: %zu, Pool available: %zu\n", 
                   frame_queue.size(), buffer_pool.available());
        }

        if (producer_done.load() && frame_queue.empty()) {
            printf("[Main] Producer done and queue empty, waiting for consumer...\n");
            break;
        }
    }

    if (g_exit.load()) {
        std::cout << "[Main] Received SIGINT, stopping...\n";
        frame_queue.stop();
        buffer_pool.stop();
    }

    if (producer.joinable()) {
        producer.join();
        std::cout << "[Main] Producer joined\n";
    }

    if (consumer.joinable()) {
        consumer.join();
        std::cout << "[Main] Consumer joined\n";
    }

    std::cout << "[Main] Exit clean\n";
    return 0;
}