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
/*-------------------------------------------------*/
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
        }
        
        memset(&crop_image_buf, 0, sizeof(image_buffer_t));

        bool is_fullframe_mode = detect_manager.is_fullframe_mode();
        bool found_ball = false;
        object_detect_result_list od_results;

        // ===== 公共变量：存储检测和跟踪结果（原图坐标）=====
        std::vector<T_DetectObject> ball_detections;  // 球的检测结果（原图坐标）
        std::vector<T_TrackObject> track_results;     // 跟踪结果（原图坐标）

        if (is_fullframe_mode) {
            // ===== 全图检测模式 =====
            camera.update_and_draw_only(&src_image);
            inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
            
            // 提取球的检测结果（已经是原图坐标）
            for (int j = 0; j < od_results.count; j++) {
                object_detect_result *det = &(od_results.results[j]);
                char text[64];
                snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                
                if (strncmp(text, "ball", 4) == 0) {
                    found_ball = true;
                    
                    T_DetectObject obj;
                    obj.cls_id = det->cls_id;
                    obj.score = det->prop;
                    obj.xmin = det->box.left;
                    obj.ymin = det->box.top;
                    obj.xmax = det->box.right;
                    obj.ymax = det->box.bottom;
                    ball_detections.push_back(obj);
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

            inference_yolov8_model(&rknn_app_ctx, &crop_image_buf, &od_results);

            // 提取球的检测结果，并转换为原图坐标
            for (int j = 0; j < od_results.count; j++) {
                object_detect_result *det = &(od_results.results[j]);
                char text[64];
                snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                
                if (strncmp(text, "ball", 4) == 0) {
                    found_ball = true;
                    
                    T_DetectObject obj;
                    obj.cls_id = det->cls_id;
                    obj.score = det->prop;
                    // 裁剪图坐标 → 原图坐标
                    obj.xmin = camera.get_rect().left + det->box.left;
                    obj.ymin = camera.get_rect().top + det->box.top;
                    obj.xmax = camera.get_rect().left + det->box.right;
                    obj.ymax = camera.get_rect().top + det->box.bottom;
                    ball_detections.push_back(obj);
                }
            }
        }

        // ===== 统一的跟踪处理（始终使用原图坐标）=====
        if (!ball_detections.empty() || tracker.HasActiveTrack()) {
            tracker.ProcessFrame(frame_track_count, ball_detections, track_results);
        }

        // ===== 统一的绘图处理（在裁剪图上绘制）=====
        if (crop_image_buf.virt_addr != NULL) {
            // 绘制检测框（蓝色）
            for (const auto& det : ball_detections) {
                // 原图坐标 → 裁剪图坐标
                int crop_x1 = det.xmin - camera.get_rect().left;
                int crop_y1 = det.ymin - camera.get_rect().top;
                int crop_x2 = det.xmax - camera.get_rect().left;
                int crop_y2 = det.ymax - camera.get_rect().top;
                
                // 检查是否在裁剪框内
                if (crop_x1 >= 0 && crop_y1 >= 0 && 
                    crop_x2 <= ALG_CROP_WIDTH && crop_y2 <= ALG_CROP_HEIGHT) {
                    
                    draw_rectangle(&crop_image_buf, 
                                   crop_x1, crop_y1,
                                   crop_x2 - crop_x1, crop_y2 - crop_y1,
                                   COLOR_BLUE, 3);
                    
                    char text[64];
                    snprintf(text, sizeof(text), "ball %.1f%%", det.score * 100);
                    draw_text(&crop_image_buf, text, crop_x1, crop_y1 - 20, COLOR_RED, 10);
                }
            }
            
            // 绘制跟踪框（绿色）
            for (const auto& trk : track_results) {
                // 原图坐标 → 裁剪图坐标
                int crop_x1 = trk.xmin - camera.get_rect().left;
                int crop_y1 = trk.ymin - camera.get_rect().top;
                int crop_x2 = trk.xmax - camera.get_rect().left;
                int crop_y2 = trk.ymax - camera.get_rect().top;
                
                // 检查是否在裁剪框内
                if (crop_x1 >= 0 && crop_y1 >= 0 && 
                    crop_x2 <= ALG_CROP_WIDTH && crop_y2 <= ALG_CROP_HEIGHT) {
                    
                    draw_rectangle(&crop_image_buf, 
                                   crop_x1, crop_y1,
                                   crop_x2 - crop_x1, crop_y2 - crop_y1,
                                   COLOR_GREEN, 3);
                    
                    // 标记是否为预测框
                    if (trk.is_predicted) {
                        draw_text(&crop_image_buf, "[PRED]", crop_x1, crop_y1 - 40, COLOR_YELLOW, 10);
                    }
                }
            }
            
            // 绘制调试信息
            char debug_text[128];
            snprintf(debug_text, sizeof(debug_text), "Mode:%s Dist:%.1f Miss:%d",
                     is_fullframe_mode ? "FULL" : "CROP",
                     tracker.GetLastGatingDistance(),
                     tracker.GetMissCount());
            draw_text(&crop_image_buf, debug_text, 10, 20, COLOR_YELLOW, 12);
        }

        // ===== 更新运镜 =====
        if (!track_results.empty()) {
            auto &t = track_results[0];
            camera.update_by_target(t.xmin, t.ymin, t.xmax, t.ymax);
            
            printf("[Consumer] Track @ full(%.0f %.0f %.0f %.0f) %s\n",
                   t.xmin, t.ymin, t.xmax, t.ymax,
                   t.is_predicted ? "[PREDICTED]" : "[MATCHED]");
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
    constexpr size_t IMAGE_SIZE = 2560 * 1440 * 3;

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