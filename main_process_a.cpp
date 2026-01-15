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
#define PIC_FULL_WIDTH 4608
#define PIC_FULL_HEIGHT 1440
// #define PIC_FULL_WIDTH 2560
// #define PIC_FULL_HEIGHT 1440
#define ALG_CROP_WIDTH 1920
#define ALG_CROP_HEIGHT 860
#define VALID_TOP 476  //最小155
#define VALID_BOTTOM (ALG_CROP_HEIGHT + VALID_TOP)  // 1440 - 155 = 1285

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
 * 丝滑运镜控制器 - 篮球快速跟踪优化版
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
        
        prev_target_vx_ = 0.0f;
        prev_target_vy_ = 0.0f;
        
        update_rect();
        
        printf("[Camera] Valid region: y=[%d, %d], height=%d\n", 
               valid_top_, valid_bottom_, valid_height_);
        printf("[Camera] Tracking params: stiffness=%.3f, damping=%.2f, max_speed=%.1f\n",
               stiffness_, damping_, max_speed_);
    }

    void update_position_only()
    {
        update_camera_position();
    }

    const image_rect_t& get_rect() const
    {
        return rect_;
    }

    void get_crop_window(image_buffer_t* src, image_buffer_t* dst)
    {
        update_camera_position();
        draw_crop_rect(src);
        crop_image(src, dst);
    }

    void update_and_draw_only(image_buffer_t* src)
    {
        update_camera_position();
        draw_crop_rect(src);
    }

    void crop_current_window(image_buffer_t* src, image_buffer_t* dst)
    {
        crop_image(src, dst);
    }

    void update_by_target(int xmin, int ymin, int xmax, int ymax)
    {
        float new_target_cx = 0.5f * (xmin + xmax);
        float new_target_cy = 0.5f * (ymin + ymax);
        
        float instant_vx = new_target_cx - last_target_cx_;
        float instant_vy = new_target_cy - last_target_cy_;
        
        prev_target_vx_ = target_vx_;
        prev_target_vy_ = target_vy_;
        
        target_vx_ = velocity_smooth_ * instant_vx + (1.0f - velocity_smooth_) * target_vx_;
        target_vy_ = velocity_smooth_ * instant_vy + (1.0f - velocity_smooth_) * target_vy_;
        
        float prev_speed = std::sqrt(prev_target_vx_ * prev_target_vx_ + prev_target_vy_ * prev_target_vy_);
        float curr_speed = std::sqrt(target_vx_ * target_vx_ + target_vy_ * target_vy_);
        bool is_decelerating = (curr_speed < prev_speed * 0.8f);
        
        float dynamic_prediction = prediction_frames_;
        
        if (is_decelerating || curr_speed < 10.0f) {
            dynamic_prediction = prediction_frames_ * 0.3f;
        } else if (curr_speed > 40.0f) {
            dynamic_prediction = prediction_frames_ + 1.0f;
        }
        
        last_target_cx_ = new_target_cx;
        last_target_cy_ = new_target_cy;
        
        target_cx_ = new_target_cx + target_vx_ * dynamic_prediction;
        target_cy_ = new_target_cy + target_vy_ * dynamic_prediction;
        
        limit_target();
        
        frames_without_target_ = 0;
        has_target_ = true;
    }

    void update_by_fullframe_target(int xmin, int ymin, int xmax, int ymax)
    {
        update_by_target(xmin, ymin, xmax, ymax);
    }

    void crop_to_fullframe(int crop_x, int crop_y, int& full_x, int& full_y) const
    {
        full_x = rect_.left + crop_x;
        full_y = rect_.top + crop_y;
    }

    bool fullframe_to_crop(int full_x, int full_y, int& crop_x, int& crop_y) const
    {
        crop_x = full_x - rect_.left;
        crop_y = full_y - rect_.top;
        
        return (crop_x >= 0 && crop_x < crop_w_ && 
                crop_y >= 0 && crop_y < crop_h_);
    }

    /**
     * 标记没有目标 - 保持当前位置，不回中
     */
    void mark_no_target()
    {
        frames_without_target_++;
        
        // 丢失目标时，快速衰减目标速度
        target_vx_ *= 0.7f;
        target_vy_ *= 0.7f;
        
        // 不再回到中心，保持当前位置
    }

    void set_center(float cx, float cy)
    {
        cx = std::max(crop_w_ * 0.5f, std::min(cx, img_w_ - crop_w_ * 0.5f));
        
        float min_cy = valid_top_ + crop_h_ * 0.5f;
        float max_cy = valid_bottom_ - crop_h_ * 0.5f;
        cy = std::max(min_cy, std::min(cy, max_cy));
        
        cx_ = cx;
        cy_ = cy;
        
        vx_ = 0.0f;
        vy_ = 0.0f;
        
        target_cx_ = cx;
        target_cy_ = cy;
        target_vx_ = 0.0f;
        target_vy_ = 0.0f;
        last_target_cx_ = cx;
        last_target_cy_ = cy;
        prev_target_vx_ = 0.0f;
        prev_target_vy_ = 0.0f;
        
        update_rect();
        
        printf("[Camera] Center set to (%.1f, %.1f)\n", cx_, cy_);
    }

    int get_img_width() const { return img_w_; }
    int get_img_height() const { return img_h_; }
    int get_crop_width() const { return crop_w_; }
    int get_crop_height() const { return crop_h_; }
    float get_center_x() const { return cx_; }
    float get_center_y() const { return cy_; }
    int get_frames_without_target() const { return frames_without_target_; }

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
    
    float prev_target_vx_, prev_target_vy_;
    
    bool has_target_{false};
    int frames_without_target_{0};

    // ================= 运镜参数 =================
    const float stiffness_ = 0.025f;
    const float damping_ = 0.55f;
    const float max_speed_ = 28.0f;
    const float max_accel_ = 2.0f;
    const float dead_zone_ratio_ = 0.15f;
    const float prediction_frames_ = 1.5f;
    const float velocity_smooth_ = 0.2f;

private:
    void update_camera_position()
    {
        float dx = target_cx_ - cx_;
        float dy = target_cy_ - cy_;
        
        float dead_zone_x = crop_w_ * dead_zone_ratio_;
        float dead_zone_y = crop_h_ * dead_zone_ratio_;
        
        float distance = std::sqrt(dx * dx + dy * dy);
        
        float dot_product = vx_ * dx + vy_ * dy;
        bool is_overshooting = (dot_product < 0) && (std::sqrt(vx_*vx_ + vy_*vy_) > 5.0f);
        
        if (is_overshooting) {
            vx_ *= 0.6f;
            vy_ *= 0.6f;
        } else if (std::fabs(dx) < dead_zone_x && std::fabs(dy) < dead_zone_y) {
            vx_ *= (1.0f - damping_ * 0.7f);
            vy_ *= (1.0f - damping_ * 0.7f);
        } else {
            float dynamic_stiffness = stiffness_;
            
            if (distance < 100.0f) {
                dynamic_stiffness = stiffness_ * 0.6f;
            } else if (distance > 300.0f) {
                dynamic_stiffness = stiffness_ * 1.3f;
            }
            
            float fx = dynamic_stiffness * dx;
            float fy = dynamic_stiffness * dy;
            
            fx -= damping_ * vx_;
            fy -= damping_ * vy_;
            
            float dynamic_max_accel = max_accel_;
            if (distance < 150.0f) {
                dynamic_max_accel = max_accel_ * 0.5f;
            } else if (distance > 300.0f) {
                dynamic_max_accel = max_accel_ * 1.3f;
            }
            
            float accel = std::sqrt(fx * fx + fy * fy);
            if (accel > dynamic_max_accel) {
                float scale = dynamic_max_accel / accel;
                fx *= scale;
                fy *= scale;
            }
            
            vx_ += fx;
            vy_ += fy;
        }
        
        float dynamic_max_speed = max_speed_;
        if (distance < 100.0f) {
            dynamic_max_speed = max_speed_ * 0.5f;
        } else if (distance < 200.0f) {
            dynamic_max_speed = max_speed_ * 0.7f;
        } else if (distance > 400.0f) {
            dynamic_max_speed = max_speed_ * 1.4f;
        }
        
        float speed = std::sqrt(vx_ * vx_ + vy_ * vy_);
        if (speed > dynamic_max_speed) {
            float scale = dynamic_max_speed / speed;
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
 * 球目标筛选器 - 增强版（支持区域限制）
 */
class BallSelector {
public:
    struct BallCandidate {
        T_DetectObject det;
        float distance_to_tracker;
        float distance_to_crop_center;
        float distance_to_last_target;
        bool is_static;
        int static_frames;
        bool ever_moved;
        bool is_in_crop_region;      // 新增：是否在裁剪框内
        float priority_score;
    };

    BallSelector(int valid_left, int valid_right, int valid_top, int valid_bottom)
        : valid_left_(valid_left),
          valid_right_(valid_right),
          valid_top_(valid_top),
          valid_bottom_(valid_bottom),
          current_frame_(0),
          has_locked_target_(false),
          frames_since_target_lost_(0),
          last_target_cx_(0),
          last_target_cy_(0),
          last_target_vx_(0),
          last_target_vy_(0),
          crop_center_x_(0),
          crop_center_y_(0),
          crop_half_w_(640),
          crop_half_h_(368),
          use_crop_center_bias_(false),
          target_was_moving_(false),
          consecutive_track_frames_(0)
    {
    }

    /**
     * 设置当前裁剪框信息
     */
    void set_crop_region(float cx, float cy, float half_w, float half_h)
    {
        crop_center_x_ = cx;
        crop_center_y_ = cy;
        crop_half_w_ = half_w;
        crop_half_h_ = half_h;
        use_crop_center_bias_ = true;
    }

    /**
     * 设置当前裁剪框中心（简化版）
     */
    void set_crop_center(float cx, float cy)
    {
        crop_center_x_ = cx;
        crop_center_y_ = cy;
        use_crop_center_bias_ = true;
    }

    void clear_crop_center_bias()
    {
        use_crop_center_bias_ = false;
    }

    /**
     * 从多个检测结果中选择目标球
     */
    const T_DetectObject* select_target_ball(
        const std::vector<T_DetectObject>& detections,
        const T_TrackObject* tracker_prediction,
        bool has_tracker)
    {
        current_frame_++;
        
        if (detections.empty()) {
            handle_no_detection();
            return nullptr;
        }

        update_all_ball_history(detections);

        // 单个检测时也要严格检查
        if (detections.size() == 1) {
            const auto& det = detections[0];
            if (is_in_valid_region(det) && is_acceptable_target(det)) {
                // 额外检查：如果之前有目标且是运动的，新目标不能是静止的
                if (target_was_moving_ && frames_since_target_lost_ < strict_protection_frames_) {
                    BallHistory* hist = find_ball_history(det);
                    if (hist != nullptr && !hist->ever_moved && hist->static_count > 5) {
                        // 拒绝静止球
                        handle_no_detection();
                        return nullptr;
                    }
                }
                update_locked_target(det);
                return &detections[0];
            }
            handle_no_detection();
            return nullptr;
        }

        std::vector<BallCandidate> candidates;
        
        for (const auto& det : detections) {
            BallCandidate candidate;
            candidate.det = det;
            candidate.distance_to_tracker = 1e9f;
            candidate.distance_to_crop_center = 1e9f;
            candidate.distance_to_last_target = 1e9f;
            candidate.is_static = false;
            candidate.static_frames = 0;
            candidate.ever_moved = false;
            candidate.is_in_crop_region = false;
            candidate.priority_score = 0.0f;

            if (!is_in_valid_region(det)) {
                continue;
            }

            // 检查是否在裁剪框内
            if (use_crop_center_bias_) {
                candidate.is_in_crop_region = is_in_crop_region(det);
                candidate.distance_to_crop_center = calculate_distance_to_crop_center(det);
            }

            BallHistory* hist = find_ball_history(det);
            if (hist != nullptr) {
                candidate.is_static = hist->is_static;
                candidate.static_frames = hist->static_count;
                candidate.ever_moved = hist->ever_moved;
            }

            // ===== 严格的静止球过滤 =====
            // 条件1：从未运动过且静止时间长
            if (!candidate.ever_moved && candidate.static_frames > never_moved_threshold_) {
                continue;
            }
            
            // 条件2：如果之前跟踪的是运动球，且目标刚丢失不久，拒绝所有静止球
            if (target_was_moving_ && 
                frames_since_target_lost_ < strict_protection_frames_ &&
                candidate.is_static && 
                !candidate.ever_moved) {
                continue;
            }

            // 条件3：不在裁剪框内的静止球，直接跳过
            if (!candidate.is_in_crop_region && 
                candidate.is_static && 
                candidate.static_frames > 10) {
                continue;
            }

            if (has_tracker && tracker_prediction != nullptr) {
                candidate.distance_to_tracker = calculate_distance(det, *tracker_prediction);
            }

            if (has_locked_target_) {
                candidate.distance_to_last_target = calculate_distance_to_last_target(det);
                
                // 如果距离上一个目标位置太远，且目标刚丢失不久，大幅降低优先级
                if (candidate.distance_to_last_target > target_switch_threshold_ && 
                    frames_since_target_lost_ < target_search_grace_period_) {
                    candidate.priority_score -= 800.0f;  // 加大惩罚
                }
            }

            candidate.priority_score += calculate_priority(candidate, has_tracker);

            candidates.push_back(candidate);
        }

        if (candidates.empty()) {
            handle_no_detection();
            return nullptr;
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const BallCandidate& a, const BallCandidate& b) {
                      return a.priority_score > b.priority_score;
                  });

        const auto& best = candidates[0];
        
        // 更严格的接受条件
        if (best.priority_score < min_accept_score_) {
            // 如果有锁定目标且刚丢失，拒绝低分候选
            if (has_locked_target_ && frames_since_target_lost_ < strict_protection_frames_) {
                handle_no_detection();
                return nullptr;
            }
        }

        // 额外检查：如果最佳候选是静止球且不在裁剪框内，拒绝
        if (best.is_static && !best.is_in_crop_region && !best.ever_moved) {
            // 只有在已经丢失很久后才接受
            if (frames_since_target_lost_ < extended_search_frames_) {
                handle_no_detection();
                return nullptr;
            }
        }

        for (const auto& det : detections) {
            if (det.xmin == best.det.xmin && det.ymin == best.det.ymin) {
                update_locked_target(det);
                // 记录目标是否在运动
                BallHistory* hist = find_ball_history(det);
                if (hist != nullptr && hist->ever_moved) {
                    target_was_moving_ = true;
                }
                return &det;
            }
        }

        handle_no_detection();
        return nullptr;
    }

    bool should_trigger_fullframe_search() const
    {
        return has_locked_target_ && 
               frames_since_target_lost_ >= target_search_grace_period_;
    }

    void reset()
    {
        ball_history_.clear();
        has_locked_target_ = false;
        frames_since_target_lost_ = 0;
        last_target_cx_ = 0;
        last_target_cy_ = 0;
        last_target_vx_ = 0;
        last_target_vy_ = 0;
        current_frame_ = 0;
        use_crop_center_bias_ = false;
        target_was_moving_ = false;
        consecutive_track_frames_ = 0;
    }

    bool has_locked_target() const { return has_locked_target_; }
    int get_frames_since_lost() const { return frames_since_target_lost_; }

private:
    int valid_left_;
    int valid_right_;
    int valid_top_;
    int valid_bottom_;

    // ===== 参数配置（调整后）=====
    static constexpr int static_threshold_ = 20;           // 静止判定阈值（降低，更快判定为静止）
    static constexpr int never_moved_threshold_ = 45;      // 从未运动阈值（降低）
    static constexpr float static_move_threshold_ = 12.0f; // 静止移动阈值（降低，更严格）
    static constexpr float move_detection_threshold_ = 30.0f; // 运动检测阈值
    static constexpr float target_switch_threshold_ = 250.0f; // 目标切换距离阈值（降低）
    static constexpr int target_search_grace_period_ = 15;    // 目标丢失宽限期（增加）
    static constexpr int strict_protection_frames_ = 25;      // 严格保护期（新增）
    static constexpr int extended_search_frames_ = 40;        // 扩展搜索帧数（新增）
    static constexpr float min_accept_score_ = -50.0f;        // 最低可接受分数（提高）

    struct BallHistory {
        float cx, cy;
        float initial_cx, initial_cy;
        float total_movement;
        int static_count;
        int last_seen_frame;
        int first_seen_frame;
        bool is_static;
        bool ever_moved;
    };
    std::vector<BallHistory> ball_history_;
    int current_frame_;

    bool has_locked_target_;
    int frames_since_target_lost_;
    float last_target_cx_, last_target_cy_;
    float last_target_vx_, last_target_vy_;

    // 裁剪框信息
    float crop_center_x_, crop_center_y_;
    float crop_half_w_, crop_half_h_;
    bool use_crop_center_bias_;
    
    // 目标运动状态
    bool target_was_moving_;
    int consecutive_track_frames_;

    bool is_in_valid_region(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        return (cx >= valid_left_ && cx <= valid_right_ &&
                cy >= valid_top_ && cy <= valid_bottom_);
    }

    /**
     * 检查球是否在裁剪框区域内
     */
    bool is_in_crop_region(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        return (cx >= crop_center_x_ - crop_half_w_ &&
                cx <= crop_center_x_ + crop_half_w_ &&
                cy >= crop_center_y_ - crop_half_h_ &&
                cy <= crop_center_y_ + crop_half_h_);
    }

    BallHistory* find_ball_history(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;

        for (auto& hist : ball_history_) {
            float dist = std::sqrt((cx - hist.cx) * (cx - hist.cx) + 
                                   (cy - hist.cy) * (cy - hist.cy));
            
            if (dist < static_move_threshold_ * 2.5f) {
                return &hist;
            }
        }
        return nullptr;
    }

    void update_all_ball_history(const std::vector<T_DetectObject>& detections)
    {
        ball_history_.erase(
            std::remove_if(ball_history_.begin(), ball_history_.end(),
                [this](const BallHistory& h) {
                    return (current_frame_ - h.last_seen_frame) > 150;
                }),
            ball_history_.end());

        for (const auto& det : detections) {
            float cx = (det.xmin + det.xmax) / 2.0f;
            float cy = (det.ymin + det.ymax) / 2.0f;

            BallHistory* hist = find_ball_history(det);
            
            if (hist != nullptr) {
                float move_dist = std::sqrt((cx - hist->cx) * (cx - hist->cx) + 
                                            (cy - hist->cy) * (cy - hist->cy));
                
                hist->total_movement += move_dist;
                
                float dist_from_initial = std::sqrt(
                    (cx - hist->initial_cx) * (cx - hist->initial_cx) + 
                    (cy - hist->initial_cy) * (cy - hist->initial_cy));
                
                if (dist_from_initial > move_detection_threshold_) {
                    hist->ever_moved = true;
                }
                
                if (move_dist < static_move_threshold_) {
                    hist->static_count++;
                    hist->is_static = (hist->static_count > 3);  // 更快判定为静止
                } else {
                    hist->static_count = 0;
                    hist->is_static = false;
                }
                
                hist->cx = cx;
                hist->cy = cy;
                hist->last_seen_frame = current_frame_;
            } else {
                if (ball_history_.size() < 30) {
                    BallHistory new_hist;
                    new_hist.cx = cx;
                    new_hist.cy = cy;
                    new_hist.initial_cx = cx;
                    new_hist.initial_cy = cy;
                    new_hist.total_movement = 0.0f;
                    new_hist.static_count = 0;
                    new_hist.last_seen_frame = current_frame_;
                    new_hist.first_seen_frame = current_frame_;
                    new_hist.is_static = false;
                    new_hist.ever_moved = false;
                    ball_history_.push_back(new_hist);
                }
            }
        }
    }

    bool is_acceptable_target(const T_DetectObject& det)
    {
        BallHistory* hist = find_ball_history(det);
        
        if (hist == nullptr) {
            return true;
        }
        
        // 从未运动过且已经出现很久
        if (!hist->ever_moved && 
            (current_frame_ - hist->first_seen_frame) > never_moved_threshold_) {
            return false;
        }
        
        return true;
    }

    void update_locked_target(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        if (has_locked_target_) {
            last_target_vx_ = cx - last_target_cx_;
            last_target_vy_ = cy - last_target_cy_;
        }
        
        last_target_cx_ = cx;
        last_target_cy_ = cy;
        has_locked_target_ = true;
        frames_since_target_lost_ = 0;
        consecutive_track_frames_++;
    }

    void handle_no_detection()
    {
        if (has_locked_target_) {
            frames_since_target_lost_++;
            
            // 预测目标位置
            last_target_cx_ += last_target_vx_;
            last_target_cy_ += last_target_vy_;
            
            // 速度衰减
            last_target_vx_ *= 0.85f;
            last_target_vy_ *= 0.85f;
            
            // 丢失太久后重置运动状态
            if (frames_since_target_lost_ > extended_search_frames_) {
                target_was_moving_ = false;
            }
        }
        consecutive_track_frames_ = 0;
    }

    float calculate_distance_to_last_target(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        return std::sqrt((cx - last_target_cx_) * (cx - last_target_cx_) + 
                         (cy - last_target_cy_) * (cy - last_target_cy_));
    }

    float calculate_distance_to_crop_center(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        return std::sqrt((cx - crop_center_x_) * (cx - crop_center_x_) + 
                         (cy - crop_center_y_) * (cy - crop_center_y_));
    }

    float calculate_distance(const T_DetectObject& det, const T_TrackObject& tracker)
    {
        float det_cx = (det.xmin + det.xmax) / 2.0f;
        float det_cy = (det.ymin + det.ymax) / 2.0f;
        float trk_cx = (tracker.xmin + tracker.xmax) / 2.0f;
        float trk_cy = (tracker.ymin + tracker.ymax) / 2.0f;
        
        return std::sqrt((det_cx - trk_cx) * (det_cx - trk_cx) + 
                         (det_cy - trk_cy) * (det_cy - trk_cy));
    }

    float calculate_priority(const BallCandidate& candidate, bool has_tracker)
    {
        float score = 0.0f;

        // 1. 置信度分数
        score += candidate.det.score * 100.0f;

        // 2. 跟踪器距离分数
        if (has_tracker && candidate.distance_to_tracker < 1e8f) {
            if (candidate.distance_to_tracker < 500.0f) {
                score += (500.0f - candidate.distance_to_tracker) * 0.5f;
            }
        }

        // 3. 运动历史奖励（大幅增加）
        if (candidate.ever_moved) {
            score += 350.0f;  // 增加运动球的优势
        }

        // 4. 静止惩罚（加重）
        if (candidate.is_static) {
            score -= candidate.static_frames * 5.0f;  // 加重惩罚
        }

        // 5. 从未运动过的球额外惩罚（加重）
        if (!candidate.ever_moved) {
            if (candidate.static_frames > 5) {
                score -= 400.0f;  // 大幅降分
            }
            if (candidate.static_frames > 15) {
                score -= 300.0f;  // 继续降分
            }
        }

        // 6. 与上次目标位置的距离（目标连续性）
        if (has_locked_target_ && candidate.distance_to_last_target < 1e8f) {
            if (candidate.distance_to_last_target < 150.0f) {
                score += (150.0f - candidate.distance_to_last_target) * 2.0f;  // 增加权重
            }
            // 距离远的惩罚
            if (candidate.distance_to_last_target > 300.0f) {
                score -= (candidate.distance_to_last_target - 300.0f) * 0.5f;
            }
        }

        // 7. 裁剪框内优先（大幅增加权重）
        if (use_crop_center_bias_) {
            if (candidate.is_in_crop_region) {
                score += 300.0f;  // 裁剪框内的球大幅加分
            } else {
                // 裁剪框外的球降分，距离越远降得越多
                if (candidate.distance_to_crop_center > 400.0f) {
                    score -= 150.0f;
                }
                if (candidate.distance_to_crop_center > 800.0f) {
                    score -= 250.0f;
                }
                if (candidate.distance_to_crop_center > 1200.0f) {
                    score -= 350.0f;
                }
            }
        }

        return score;
    }
};

//调试版本
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

    // 球筛选器
    BallSelector ball_selector(0, PIC_FULL_WIDTH, VALID_TOP, VALID_BOTTOM);

    // 初始化状态
    bool is_initialized = false;
    int init_frames = 0;
    static constexpr int MAX_INIT_FRAMES = 30;

    // 上一帧的跟踪结果
    T_TrackObject last_track_result;
    bool has_last_track = false;

    // 分区检测参数
    const int VALID_HEIGHT = VALID_BOTTOM - VALID_TOP;
    const int HALF_WIDTH = PIC_FULL_WIDTH / 2;  // 720

    printf("[Consumer][DEBUG] === SPLIT DETECTION MODE ===\n");
    printf("[Consumer] Image size: %dx%d, Crop size: %dx%d\n",
           PIC_FULL_WIDTH, PIC_FULL_HEIGHT, ALG_CROP_WIDTH, ALG_CROP_HEIGHT);
    printf("[Consumer] Detection regions: Left[0-%d], Right[%d-%d], Height[%d-%d]\n",
           HALF_WIDTH, HALF_WIDTH, PIC_FULL_WIDTH, VALID_TOP, VALID_BOTTOM);

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
            }
        }
        
        memset(&crop_image_buf, 0, sizeof(image_buffer_t));

        bool found_ball = false;
        
        std::vector<T_DetectObject> all_ball_detections;
        std::vector<T_DetectObject> ball_detections;
        std::vector<T_TrackObject> track_results;

        // 获取当前裁剪框信息
        image_rect_t crop_rect = camera.get_rect();
        float crop_cx = (crop_rect.left + crop_rect.right) / 2.0f;
        float crop_cy = (crop_rect.top + crop_rect.bottom) / 2.0f;

        // 更新相机位置（不绘制）
        camera.update_position_only();

        // ===== 左右分区检测（减少误检）=====
        
        // 左半区域检测
        {
            image_buffer_t left_region_image = {0};
            image_rect_t left_box = {0, VALID_TOP, HALF_WIDTH, VALID_BOTTOM};
            image_rect_t real_left_rect;
            
            int crop_ret = crop_alg_image(
                &src_image,
                &left_region_image,
                left_box,
                &real_left_rect,
                HALF_WIDTH,
                VALID_HEIGHT
            );
            
            if (crop_ret == 0) {
                object_detect_result_list left_results;
                inference_yolov8_model(&rknn_app_ctx, &left_region_image, &left_results);
                
                // 坐标映射：左半区域 -> 全图
                for (int j = 0; j < left_results.count; j++) {
                    object_detect_result *det = &(left_results.results[j]);
                    char text[64];
                    snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                    
                    if (strncmp(text, "ball", 4) == 0) {
                        T_DetectObject obj;
                        obj.cls_id = det->cls_id;
                        obj.score = det->prop;
                        obj.xmin = det->box.left;               // X不变
                        obj.ymin = det->box.top + VALID_TOP;    // Y加偏移
                        obj.xmax = det->box.right;              // X不变
                        obj.ymax = det->box.bottom + VALID_TOP;
                        all_ball_detections.push_back(obj);
                    }
                }
                
                if (left_region_image.virt_addr != nullptr) {
                    free(left_region_image.virt_addr);
                }
            } else {
                printf("[Consumer][WARN] crop left region failed, ret=%d\n", crop_ret);
            }
        }

        // 右半区域检测
        {
            image_buffer_t right_region_image = {0};
            image_rect_t right_box = {HALF_WIDTH, VALID_TOP, PIC_FULL_WIDTH, VALID_BOTTOM};
            image_rect_t real_right_rect;
            
            int crop_ret = crop_alg_image(
                &src_image,
                &right_region_image,
                right_box,
                &real_right_rect,
                HALF_WIDTH,
                VALID_HEIGHT
            );
            
            if (crop_ret == 0) {
                object_detect_result_list right_results;
                inference_yolov8_model(&rknn_app_ctx, &right_region_image, &right_results);
                
                // 坐标映射：右半区域 -> 全图
                for (int j = 0; j < right_results.count; j++) {
                    object_detect_result *det = &(right_results.results[j]);
                    char text[64];
                    snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
                    
                    if (strncmp(text, "ball", 4) == 0) {
                        T_DetectObject obj;
                        obj.cls_id = det->cls_id;
                        obj.score = det->prop;
                        obj.xmin = det->box.left + HALF_WIDTH;    // X加偏移
                        obj.ymin = det->box.top + VALID_TOP;      // Y加偏移
                        obj.xmax = det->box.right + HALF_WIDTH;   // X加偏移
                        obj.ymax = det->box.bottom + VALID_TOP;
                        all_ball_detections.push_back(obj);
                    }
                }
                
                if (right_region_image.virt_addr != nullptr) {
                    free(right_region_image.virt_addr);
                }
            } else {
                printf("[Consumer][WARN] crop right region failed, ret=%d\n", crop_ret);
            }
        }

        // 去除中线附近的重复检测
        if (all_ball_detections.size() > 1) {
            std::vector<T_DetectObject> deduped;
            for (const auto& det : all_ball_detections) {
                bool is_duplicate = false;
                float det_cx = (det.xmin + det.xmax) / 2.0f;
                float det_cy = (det.ymin + det.ymax) / 2.0f;
                
                for (const auto& existing : deduped) {
                    float ex_cx = (existing.xmin + existing.xmax) / 2.0f;
                    float ex_cy = (existing.ymin + existing.ymax) / 2.0f;
                    float dist = std::sqrt((det_cx - ex_cx) * (det_cx - ex_cx) + 
                                           (det_cy - ex_cy) * (det_cy - ex_cy));
                    
                    // 距离小于50像素认为是重复检测，保留置信度高的
                    if (dist < 50.0f) {
                        is_duplicate = true;
                        // 如果当前检测置信度更高，替换已有的
                        if (det.score > existing.score) {
                            for (auto& e : deduped) {
                                if (&e == &existing) {
                                    e = det;
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
                
                if (!is_duplicate) {
                    deduped.push_back(det);
                }
            }
            all_ball_detections = std::move(deduped);
        }

        // 裁剪输出图（用于内部处理，但不输出）
        camera.crop_current_window(&src_image, &crop_image_buf);

        // 设置裁剪框中心偏好
        ball_selector.set_crop_center(crop_cx, crop_cy);

        // ===== 球筛选 =====
        if (!all_ball_detections.empty()) {
            const T_DetectObject *target = ball_selector.select_target_ball(
                all_ball_detections,
                has_last_track ? &last_track_result : nullptr,
                has_last_track);

            if (target != nullptr) {
                ball_detections.push_back(*target);
                found_ball = true;
            }
        }

        // ===== 初始化阶段处理 =====
        if (!is_initialized) {
            init_frames++;
            if (found_ball) {
                const auto& target = ball_detections[0];
                float target_cx = (target.xmin + target.xmax) / 2.0f;
                float target_cy = (target.ymin + target.ymax) / 2.0f;
                
                camera.set_center(target_cx, target_cy);
                
                is_initialized = true;
                printf("[Consumer][INIT] Target acquired at (%.0f, %.0f)\n", target_cx, target_cy);
            } else {
                if (init_frames >= MAX_INIT_FRAMES) {
                    printf("[Consumer][INIT] Max init frames reached\n");
                    is_initialized = true;
                }
            }
        }

        // ===== 跟踪处理 =====
        if (!ball_detections.empty() || tracker.HasActiveTrack()) {
            tracker.ProcessFrame(frame_track_count, ball_detections, track_results);
        }

        // ===== 保存当前帧跟踪结果 =====
        if (!track_results.empty()) {
            last_track_result = track_results[0];
            has_last_track = true;
        } else if (!tracker.HasActiveTrack()) {
            has_last_track = false;
        }

        // ===== 在原图上绘制 =====
        {
            // 绘制左右分区线（黄色虚线效果用实线代替）
            draw_rectangle(&src_image, 
                           HALF_WIDTH - 1, VALID_TOP,
                           2, VALID_HEIGHT,
                           COLOR_YELLOW, 1);

            // 裁剪窗口（红色粗框）
            draw_rectangle(&src_image, 
                           crop_rect.left, crop_rect.top,
                           crop_rect.right - crop_rect.left, 
                           crop_rect.bottom - crop_rect.top,
                           COLOR_RED, 4);

            // 所有检测到的球（白色细框）
            for (const auto& det : all_ball_detections) {
                draw_rectangle(&src_image, 
                               det.xmin, det.ymin,
                               det.xmax - det.xmin, det.ymax - det.ymin,
                               COLOR_WHITE, 1);
            }

            // 选中的目标球（蓝色粗框）
            for (const auto& det : ball_detections) {
                draw_rectangle(&src_image, 
                               det.xmin, det.ymin,
                               det.xmax - det.xmin, det.ymax - det.ymin,
                               COLOR_BLUE, 3);
                
                char text[64];
                snprintf(text, sizeof(text), "TARGET %.1f%%", det.score * 100);
                draw_text(&src_image, text, det.xmin, det.ymin - 20, COLOR_RED, 10);
            }
            
            // 跟踪框（绿色）
            for (const auto& trk : track_results) {
                draw_rectangle(&src_image, 
                               trk.xmin, trk.ymin,
                               trk.xmax - trk.xmin, trk.ymax - trk.ymin,
                               COLOR_GREEN, 3);
                
                if (trk.is_predicted) {
                    draw_text(&src_image, "[PRED]", trk.xmin, trk.ymin - 40, COLOR_YELLOW, 10);
                }
            }
            
            // 调试信息
            char debug_text[128];
            snprintf(debug_text, sizeof(debug_text), "[SPLIT] Balls:%zu->%zu Lost:%d",
                     all_ball_detections.size(),
                     ball_detections.size(),
                     ball_selector.get_frames_since_lost());
            draw_text(&src_image, debug_text, 10, 30, COLOR_YELLOW, 15);
            
            char crop_info[64];
            snprintf(crop_info, sizeof(crop_info), "Crop:[%d,%d]-[%d,%d]",
                     crop_rect.left, crop_rect.top, crop_rect.right, crop_rect.bottom);
            draw_text(&src_image, crop_info, 10, 60, COLOR_RED, 12);
        }

        // ===== 更新运镜 =====
        if (is_initialized && !track_results.empty()) {
            auto &t = track_results[0];
            camera.update_by_target(t.xmin, t.ymin, t.xmax, t.ymax);
        }

        // ===== 无目标时标记 =====
        if (!found_ball) {
            camera.mark_no_target();
        }

        // ===== 保存原图 =====
        {
            char out_path[256];
            snprintf(out_path, sizeof(out_path), "%s/%06d.jpg", out_dir, frame_count);
            write_image(out_path, &src_image);
        }
        
        frame_count++;
        
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