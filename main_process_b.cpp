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
#include <functional>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "awi_track.hpp"

/*--------------------------------------*/
// 颜色定义
#define COLOR_PURPLE  0xFF00FF
#define COLOR_GRAY    0x808080
#define COLOR_CYAN    0x00FFFF
//#define COLOR_ORANGE  0xFFA500

// 图像尺寸配置
#define PIC_FULL_WIDTH    4608
#define PIC_FULL_HEIGHT   1440
#define ALG_CROP_WIDTH    1920
#define ALG_CROP_HEIGHT   860
#define VALID_TOP         476
#define VALID_BOTTOM      (ALG_CROP_HEIGHT + VALID_TOP)

// 检测区域配置
#define EXCLUDE_RIGHT_WIDTH  408  // 右侧排除宽度

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
// 阻塞式 Buffer Pool
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
 * 丝滑运镜控制器
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
        
        vx_ = vy_ = 0.0f;
        target_cx_ = cx_;
        target_cy_ = cy_;
        target_vx_ = target_vy_ = 0.0f;
        last_target_cx_ = cx_;
        last_target_cy_ = cy_;
        prev_target_vx_ = prev_target_vy_ = 0.0f;
        
        update_rect();
        
        printf("[Camera] Valid region: y=[%d, %d], height=%d\n", 
               valid_top_, valid_bottom_, valid_height_);
    }

    void update_position_only() { update_camera_position(); }
    const image_rect_t& get_rect() const { return rect_; }

    void crop_current_window(image_buffer_t* src, image_buffer_t* dst)
    {
        if (!src || !dst) return;

        image_rect_t box = rect_;
        image_rect_t real_crop_rect;

        int ret = crop_alg_image(src, dst, box, &real_crop_rect,
                                 ALG_CROP_WIDTH, ALG_CROP_HEIGHT);
        if (ret != 0) {
            printf("crop_alg_image failed, ret=%d\n", ret);
        }
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
    }

    void mark_no_target()
    {
        frames_without_target_++;
        target_vx_ *= 0.7f;
        target_vy_ *= 0.7f;
    }

    void set_center(float cx, float cy)
    {
        cx = std::max(crop_w_ * 0.5f, std::min(cx, img_w_ - crop_w_ * 0.5f));
        float min_cy = valid_top_ + crop_h_ * 0.5f;
        float max_cy = valid_bottom_ - crop_h_ * 0.5f;
        cy = std::max(min_cy, std::min(cy, max_cy));
        
        cx_ = cx;
        cy_ = cy;
        vx_ = vy_ = 0.0f;
        target_cx_ = cx;
        target_cy_ = cy;
        target_vx_ = target_vy_ = 0.0f;
        last_target_cx_ = cx;
        last_target_cy_ = cy;
        prev_target_vx_ = prev_target_vy_ = 0.0f;
        
        update_rect();
        printf("[Camera] Center set to (%.1f, %.1f)\n", cx_, cy_);
    }

    int get_crop_width() const { return crop_w_; }
    int get_crop_height() const { return crop_h_; }

private:
    int img_w_, img_h_;
    int crop_w_, crop_h_;
    image_rect_t rect_;
    
    int valid_top_, valid_bottom_, valid_height_;
    float cx_, cy_;
    float vx_, vy_;
    float target_cx_, target_cy_;
    float target_vx_, target_vy_;
    float last_target_cx_, last_target_cy_;
    float prev_target_vx_, prev_target_vy_;
    int frames_without_target_{0};

    // 运镜参数
    /* 参数调整场景建议
    | 场景 | 调整方向 |
    |------|----------|
    | 画面抖动/晃动 | ↑ damping, ↓ stiffness, ↑ dead_zone |
    | 跟随太慢/滞后 | ↑ stiffness, ↑ max_speed, ↑ prediction |
    | 目标停止时过冲 | ↓ prediction, ↑ damping |
    | 快速运动丢失目标 | ↑ max_speed, ↑ max_accel, ↑ prediction |
    | 画面不够平滑 | ↓ velocity_smooth, ↑ damping |
    */
    const float stiffness_ = 0.025f;
    const float damping_ = 0.55f;
    const float max_speed_ = 28.0f;
    const float max_accel_ = 2.0f;
    const float dead_zone_ratio_ = 0.15f;
    const float prediction_frames_ = 1.5f;
    const float velocity_smooth_ = 0.2f;

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
            
            float fx = dynamic_stiffness * dx - damping_ * vx_;
            float fy = dynamic_stiffness * dy - damping_ * vy_;
            
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
        target_cx_ = std::max(crop_w_ * 0.5f, std::min(target_cx_, img_w_ - crop_w_ * 0.5f));
        float min_cy = valid_top_ + crop_h_ * 0.5f;
        float max_cy = valid_bottom_ - crop_h_ * 0.5f;
        target_cy_ = std::max(min_cy, std::min(target_cy_, max_cy));
    }

    void limit_center()
    {
        cx_ = std::max(crop_w_ * 0.5f, std::min(cx_, img_w_ - crop_w_ * 0.5f));
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
};

/*---------------------------------------------------*/
// 阻塞式帧队列
class FrameQueue {
public:
    FrameQueue(size_t max_size) : max_size_(max_size) {}

    void push(image_buffer_t& buf)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [&] {
            return queue_.size() < max_size_ || stop_;
        });
        if (stop_) return;
        queue_.push_back(buf);
        cv_not_empty_.notify_one();
    }

    bool pop(image_buffer_t& out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [&] {
            return !queue_.empty() || stop_;
        });
        if (queue_.empty()) return false;
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

/*------------------------------------------------*/
/**
 * 分区检测器 - 管理左右分区的YOLO检测
 */
class SplitRegionDetector {
public:
    struct DetectionConfig {
        int full_width;
        int full_height;
        int valid_top;
        int valid_bottom;
        int exclude_right_width;
        int num_splits;  // 分区数量（2 = 左右两半）
    };

    struct DetectionResult {
        std::vector<T_DetectObject> all_detections;
        std::vector<T_DetectObject> valid_detections;
        std::vector<T_DetectObject> excluded_detections;
    };

    SplitRegionDetector(const DetectionConfig& config)
        : config_(config)
    {
        valid_height_ = config_.valid_bottom - config_.valid_top;
        split_width_ = config_.full_width / config_.num_splits;
        exclude_right_start_ = config_.full_width - config_.exclude_right_width;
        
        printf("[Detector] Split config: %d regions, each %dx%d\n",
               config_.num_splits, split_width_, valid_height_);
        printf("[Detector] Exclude zone: x >= %d\n", exclude_right_start_);
    }

    DetectionResult detect(image_buffer_t* src_image, rknn_app_context_t* rknn_ctx)
    {
        DetectionResult result;
        
        // 对每个分区进行检测
        for (int i = 0; i < config_.num_splits; i++) {
            detect_region(src_image, rknn_ctx, i, result.all_detections);
        }
        
        // 去重
        deduplicate(result.all_detections);
        
        // 按排除区域分类
        classify_detections(result);
        
        return result;
    }

    int get_exclude_right_start() const { return exclude_right_start_; }
    int get_split_width() const { return split_width_; }
    int get_valid_height() const { return valid_height_; }

private:
    DetectionConfig config_;
    int valid_height_;
    int split_width_;
    int exclude_right_start_;

    void detect_region(image_buffer_t* src_image, rknn_app_context_t* rknn_ctx,
                       int region_index, std::vector<T_DetectObject>& detections)
    {
        int x_offset = region_index * split_width_;
        
        image_buffer_t region_image = {0};
        image_rect_t region_box = {
            x_offset, 
            config_.valid_top, 
            x_offset + split_width_, 
            config_.valid_bottom
        };
        image_rect_t real_rect;
        
        int crop_ret = crop_alg_image(src_image, &region_image, region_box,
                                      &real_rect, split_width_, valid_height_);
        
        if (crop_ret != 0) {
            printf("[Detector][WARN] crop region %d failed, ret=%d\n", region_index, crop_ret);
            return;
        }
        
        object_detect_result_list results;
        inference_yolov8_model(rknn_ctx, &region_image, &results);
        
        // 坐标映射回全图
        for (int j = 0; j < results.count; j++) {
            object_detect_result* det = &results.results[j];
            char text[64];
            snprintf(text, sizeof(text), "%s", coco_cls_to_name(det->cls_id));
            
            if (strncmp(text, "ball", 4) == 0) {
                T_DetectObject obj;
                obj.cls_id = det->cls_id;
                obj.score = det->prop;
                obj.xmin = det->box.left + x_offset;
                obj.ymin = det->box.top + config_.valid_top;
                obj.xmax = det->box.right + x_offset;
                obj.ymax = det->box.bottom + config_.valid_top;
                detections.push_back(obj);
            }
        }
        
        if (region_image.virt_addr != nullptr) {
            free(region_image.virt_addr);
        }
    }

    void deduplicate(std::vector<T_DetectObject>& detections)
    {
        if (detections.size() <= 1) return;
        
        std::vector<T_DetectObject> deduped;
        const float merge_threshold = 50.0f;
        
        for (const auto& det : detections) {
            float det_cx = (det.xmin + det.xmax) / 2.0f;
            float det_cy = (det.ymin + det.ymax) / 2.0f;
            
            bool merged = false;
            for (auto& existing : deduped) {
                float ex_cx = (existing.xmin + existing.xmax) / 2.0f;
                float ex_cy = (existing.ymin + existing.ymax) / 2.0f;
                float dist = std::sqrt((det_cx - ex_cx) * (det_cx - ex_cx) + 
                                       (det_cy - ex_cy) * (det_cy - ex_cy));
                
                if (dist < merge_threshold) {
                    // 保留置信度高的
                    if (det.score > existing.score) {
                        existing = det;
                    }
                    merged = true;
                    break;
                }
            }
            
            if (!merged) {
                deduped.push_back(det);
            }
        }
        
        detections = std::move(deduped);
    }

    void classify_detections(DetectionResult& result)
    {
        for (const auto& det : result.all_detections) {
            float det_cx = (det.xmin + det.xmax) / 2.0f;
            
            if (det_cx >= exclude_right_start_) {
                result.excluded_detections.push_back(det);
            } else {
                result.valid_detections.push_back(det);
            }
        }
    }
};

/*------------------------------------------------*/
/**
 * 球目标筛选器 - 增强版 V2
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
        bool is_in_crop_region;
        bool is_blacklisted;
        bool is_likely_other_court;
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
          crop_half_w_(ALG_CROP_WIDTH / 2),
          crop_half_h_(ALG_CROP_HEIGHT / 2),
          use_crop_center_bias_(false),
          target_was_moving_(false),
          consecutive_track_frames_(0)
    {
    }

    void set_crop_center(float cx, float cy)
    {
        crop_center_x_ = cx;
        crop_center_y_ = cy;
        use_crop_center_bias_ = true;
    }

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
        update_static_blacklist();
        update_other_court_balls();

        // 单个检测时的处理
        if (detections.size() == 1) {
            return process_single_detection(detections[0]);
        }

        // 多个检测时的处理
        return process_multiple_detections(detections, tracker_prediction, has_tracker);
    }

    void get_debug_info(int& blacklist_count, int& other_court_count) const
    {
        blacklist_count = static_cast<int>(static_blacklist_.size());
        other_court_count = static_cast<int>(other_court_balls_.size());
    }

    bool has_locked_target() const { return has_locked_target_; }
    int get_frames_since_lost() const { return frames_since_target_lost_; }
    
    std::vector<std::pair<float, float>> get_blacklist_positions() const
    {
        std::vector<std::pair<float, float>> positions;
        for (const auto& entry : static_blacklist_) {
            positions.push_back({entry.cx, entry.cy});
        }
        return positions;
    }
    
    std::vector<std::pair<float, float>> get_other_court_positions() const
    {
        std::vector<std::pair<float, float>> positions;
        for (const auto& ball : other_court_balls_) {
            if (ball.is_confirmed) {
                positions.push_back({ball.cx, ball.cy});
            }
        }
        return positions;
    }

private:
    // 配置参数
    int valid_left_, valid_right_, valid_top_, valid_bottom_;
    
    static constexpr int static_threshold_ = 20;
    static constexpr int never_moved_threshold_ = 45;
    static constexpr float static_move_threshold_ = 12.0f;
    static constexpr float move_detection_threshold_ = 30.0f;
    static constexpr float target_switch_threshold_ = 250.0f;
    static constexpr int target_search_grace_period_ = 15;
    static constexpr int strict_protection_frames_ = 25;
    static constexpr int extended_search_frames_ = 40;
    static constexpr float min_accept_score_ = -50.0f;
    
    static constexpr int blacklist_static_threshold_ = 60;
    static constexpr float blacklist_match_radius_ = 60.0f;
    static constexpr int blacklist_expire_frames_ = 900;
    static constexpr float blacklist_reconfirm_radius_ = 40.0f;
    
    static constexpr float other_court_distance_threshold_ = 400.0f;
    static constexpr int other_court_confirm_frames_ = 30;
    static constexpr float other_court_direction_threshold_ = 0.7f;
    static constexpr int other_court_expire_frames_ = 300;

    // 球历史记录
    struct BallHistory {
        float cx, cy;
        float initial_cx, initial_cy;
        float total_movement;
        int static_count;
        int last_seen_frame;
        int first_seen_frame;
        bool is_static;
        bool ever_moved;
        int id;
        float avg_vx, avg_vy;
        int velocity_samples;
    };
    std::vector<BallHistory> ball_history_;
    int current_frame_;
    int next_ball_id_ = 0;
    int main_target_id_ = -1;

    // 静止球黑名单
    struct StaticBlacklistEntry {
        float cx, cy;
        float radius;
        int added_frame;
        int last_seen_frame;
        float confidence;
    };
    std::vector<StaticBlacklistEntry> static_blacklist_;
    
    // 其他球场干扰球
    struct OtherCourtBall {
        float cx, cy;
        float vx, vy;
        int first_seen_frame;
        int last_seen_frame;
        int confirm_count;
        bool is_confirmed;
        int id;
    };
    std::vector<OtherCourtBall> other_court_balls_;

    // 状态变量
    bool has_locked_target_;
    int frames_since_target_lost_;
    float last_target_cx_, last_target_cy_;
    float last_target_vx_, last_target_vy_;
    float crop_center_x_, crop_center_y_;
    float crop_half_w_, crop_half_h_;
    bool use_crop_center_bias_;
    bool target_was_moving_;
    int consecutive_track_frames_;

    // ===== 核心处理方法 =====
    
    const T_DetectObject* process_single_detection(const T_DetectObject& det)
    {
        if (is_in_static_blacklist(det) || is_other_court_ball(det)) {
            handle_no_detection();
            return nullptr;
        }
        
        if (!is_in_valid_region(det) || !is_acceptable_target(det)) {
            handle_no_detection();
            return nullptr;
        }
        
        if (target_was_moving_ && frames_since_target_lost_ < strict_protection_frames_) {
            BallHistory* hist = find_ball_history(det);
            if (hist != nullptr && !hist->ever_moved && hist->static_count > 5) {
                handle_no_detection();
                return nullptr;
            }
        }
        
        update_locked_target(det);
        return &det;
    }

    const T_DetectObject* process_multiple_detections(
        const std::vector<T_DetectObject>& detections,
        const T_TrackObject* tracker_prediction,
        bool has_tracker)
    {
        std::vector<BallCandidate> candidates;
        
        for (const auto& det : detections) {
            BallCandidate candidate = create_candidate(det, tracker_prediction, has_tracker);
            
            if (!is_valid_candidate(candidate)) {
                continue;
            }
            
            candidate.priority_score = calculate_priority(candidate, has_tracker);
            candidates.push_back(candidate);
        }

        if (candidates.empty()) {
            handle_no_detection();
            return nullptr;
        }

        // 按优先级排序
        std::sort(candidates.begin(), candidates.end(),
                  [](const BallCandidate& a, const BallCandidate& b) {
                      return a.priority_score > b.priority_score;
                  });

        const auto& best = candidates[0];
        
        if (!is_acceptable_best_candidate(best)) {
            handle_no_detection();
            return nullptr;
        }

        // 找到原始检测对象并返回
        for (const auto& det : detections) {
            if (det.xmin == best.det.xmin && det.ymin == best.det.ymin) {
                update_locked_target(det);
                mark_as_main_target(det);
                return &det;
            }
        }

        handle_no_detection();
        return nullptr;
    }

    BallCandidate create_candidate(const T_DetectObject& det,
                                   const T_TrackObject* tracker_prediction,
                                   bool has_tracker)
    {
        BallCandidate candidate;
        candidate.det = det;
        candidate.distance_to_tracker = 1e9f;
        candidate.distance_to_crop_center = 1e9f;
        candidate.distance_to_last_target = 1e9f;
        candidate.is_static = false;
        candidate.static_frames = 0;
        candidate.ever_moved = false;
        candidate.is_in_crop_region = false;
        candidate.is_blacklisted = is_in_static_blacklist(det);
        candidate.is_likely_other_court = is_other_court_ball(det);
        candidate.priority_score = 0.0f;

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

        if (has_tracker && tracker_prediction != nullptr) {
            candidate.distance_to_tracker = calculate_distance(det, *tracker_prediction);
        }

        if (has_locked_target_) {
            candidate.distance_to_last_target = calculate_distance_to_last_target(det);
        }

        return candidate;
    }

    bool is_valid_candidate(const BallCandidate& candidate)
    {
        if (!is_in_valid_region(candidate.det)) return false;
        if (candidate.is_blacklisted) return false;
        if (candidate.is_likely_other_court && has_locked_target_ && consecutive_track_frames_ > 10) return false;
        if (!candidate.ever_moved && candidate.static_frames > never_moved_threshold_) return false;
        if (target_was_moving_ && frames_since_target_lost_ < strict_protection_frames_ &&
            candidate.is_static && !candidate.ever_moved) return false;
        if (!candidate.is_in_crop_region && candidate.is_static && candidate.static_frames > 10) return false;
        
        return true;
    }

    bool is_acceptable_best_candidate(const BallCandidate& best)
    {
        if (best.priority_score < min_accept_score_) {
            if (has_locked_target_ && frames_since_target_lost_ < strict_protection_frames_) {
                return false;
            }
        }

        if (best.is_static && !best.is_in_crop_region && !best.ever_moved) {
            if (frames_since_target_lost_ < extended_search_frames_) {
                return false;
            }
        }

        return true;
    }

    // ===== 辅助方法 =====

    bool is_in_valid_region(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        return (cx >= valid_left_ && cx <= valid_right_ &&
                cy >= valid_top_ && cy <= valid_bottom_);
    }

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
        // 清理过期记录
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
                update_existing_history(hist, cx, cy);
            } else if (ball_history_.size() < 30) {
                create_new_history(cx, cy);
            }
        }
    }

    void update_existing_history(BallHistory* hist, float cx, float cy)
    {
        float move_dist = std::sqrt((cx - hist->cx) * (cx - hist->cx) + 
                                    (cy - hist->cy) * (cy - hist->cy));
        
        // 更新速度
        float inst_vx = cx - hist->cx;
        float inst_vy = cy - hist->cy;
        const float velocity_alpha = 0.3f;
        hist->avg_vx = velocity_alpha * inst_vx + (1.0f - velocity_alpha) * hist->avg_vx;
        hist->avg_vy = velocity_alpha * inst_vy + (1.0f - velocity_alpha) * hist->avg_vy;
        hist->velocity_samples++;
        
        hist->total_movement += move_dist;
        
        float dist_from_initial = std::sqrt(
            (cx - hist->initial_cx) * (cx - hist->initial_cx) + 
            (cy - hist->initial_cy) * (cy - hist->initial_cy));
        
        if (dist_from_initial > move_detection_threshold_) {
            hist->ever_moved = true;
        }
        
        if (move_dist < static_move_threshold_) {
            hist->static_count++;
            hist->is_static = (hist->static_count > 3);
        } else {
            hist->static_count = 0;
            hist->is_static = false;
        }
        
        hist->cx = cx;
        hist->cy = cy;
        hist->last_seen_frame = current_frame_;
    }

    void create_new_history(float cx, float cy)
    {
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
        new_hist.id = next_ball_id_++;
        new_hist.avg_vx = 0.0f;
        new_hist.avg_vy = 0.0f;
        new_hist.velocity_samples = 0;
        ball_history_.push_back(new_hist);
    }

    bool is_acceptable_target(const T_DetectObject& det)
    {
        BallHistory* hist = find_ball_history(det);
        if (hist == nullptr) return true;
        
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
        
        BallHistory* hist = find_ball_history(det);
        if (hist != nullptr && hist->ever_moved) {
            target_was_moving_ = true;
        }
    }

    void handle_no_detection()
    {
        if (has_locked_target_) {
            frames_since_target_lost_++;
            last_target_cx_ += last_target_vx_;
            last_target_cy_ += last_target_vy_;
            last_target_vx_ *= 0.85f;
            last_target_vy_ *= 0.85f;
            
            if (frames_since_target_lost_ > extended_search_frames_) {
                target_was_moving_ = false;
            }
        }
        consecutive_track_frames_ = 0;
    }

    void mark_as_main_target(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        for (auto& hist : ball_history_) {
            float dist = std::sqrt((cx - hist.cx) * (cx - hist.cx) + 
                                   (cy - hist.cy) * (cy - hist.cy));
            if (dist < 30.0f) {
                main_target_id_ = hist.id;
                
                other_court_balls_.erase(
                    std::remove_if(other_court_balls_.begin(), other_court_balls_.end(),
                        [&hist](const OtherCourtBall& ball) {
                            return ball.id == hist.id;
                        }),
                    other_court_balls_.end());
                return;
            }
        }
    }

    // ===== 黑名单相关 =====
    
    void update_static_blacklist()
    {
        // 清理过期
        static_blacklist_.erase(
            std::remove_if(static_blacklist_.begin(), static_blacklist_.end(),
                [this](const StaticBlacklistEntry& entry) {
                    return (current_frame_ - entry.last_seen_frame) > blacklist_expire_frames_;
                }),
            static_blacklist_.end());
        
        // 添加新的静止球到黑名单
        for (const auto& hist : ball_history_) {
            if (!hist.ever_moved && hist.static_count > blacklist_static_threshold_) {
                add_to_blacklist_if_new(hist.cx, hist.cy);
            }
        }
    }

    void add_to_blacklist_if_new(float cx, float cy)
    {
        for (auto& entry : static_blacklist_) {
            float dist = std::sqrt((cx - entry.cx) * (cx - entry.cx) + 
                                   (cy - entry.cy) * (cy - entry.cy));
            if (dist < blacklist_reconfirm_radius_) {
                entry.confidence = std::min(entry.confidence + 0.1f, 2.0f);
                entry.last_seen_frame = current_frame_;
                return;
            }
        }
        
        if (static_blacklist_.size() < 20) {
            StaticBlacklistEntry new_entry;
            new_entry.cx = cx;
            new_entry.cy = cy;
            new_entry.radius = blacklist_match_radius_;
            new_entry.added_frame = current_frame_;
            new_entry.last_seen_frame = current_frame_;
            new_entry.confidence = 1.0f;
            static_blacklist_.push_back(new_entry);
            printf("[BallSelector][Frame %d] Static ball blacklisted at (%.0f, %.0f)\n",
                   current_frame_, cx, cy);
        }
    }
    
    bool is_in_static_blacklist(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        for (auto& entry : static_blacklist_) {
            float dist = std::sqrt((cx - entry.cx) * (cx - entry.cx) + 
                                   (cy - entry.cy) * (cy - entry.cy));
            float effective_radius = entry.radius * entry.confidence;
            
            if (dist < effective_radius) {
                entry.last_seen_frame = current_frame_;
                return true;
            }
        }
        return false;
    }

    // ===== 其他球场球相关 =====
    
    void update_other_court_balls()
    {
        // 清理过期
        other_court_balls_.erase(
            std::remove_if(other_court_balls_.begin(), other_court_balls_.end(),
                [this](const OtherCourtBall& ball) {
                    return (current_frame_ - ball.last_seen_frame) > other_court_expire_frames_;
                }),
            other_court_balls_.end());
        
        if (!has_locked_target_ || consecutive_track_frames_ < 5) return;
        
        for (const auto& hist : ball_history_) {
            if (hist.id == main_target_id_) continue;
            
            check_and_add_other_court_ball(hist);
        }
    }

    void check_and_add_other_court_ball(const BallHistory& hist)
    {
        float dist_to_target = std::sqrt(
            (hist.cx - last_target_cx_) * (hist.cx - last_target_cx_) + 
            (hist.cy - last_target_cy_) * (hist.cy - last_target_cy_));
        
        if (dist_to_target < other_court_distance_threshold_) return;
        if (!hist.ever_moved) return;
        
        bool direction_mismatch = check_direction_mismatch(hist);
        
        OtherCourtBall* existing = find_other_court_ball(hist.cx, hist.cy);
        
        if (existing != nullptr) {
            existing->cx = hist.cx;
            existing->cy = hist.cy;
            existing->vx = hist.avg_vx;
            existing->vy = hist.avg_vy;
            existing->last_seen_frame = current_frame_;
            
            if (direction_mismatch) {
                existing->confirm_count++;
            }
            
            if (existing->confirm_count > other_court_confirm_frames_ && !existing->is_confirmed) {
                existing->is_confirmed = true;
                printf("[BallSelector][Frame %d] Other court ball confirmed at (%.0f, %.0f)\n",
                       current_frame_, existing->cx, existing->cy);
            }
        } else if (direction_mismatch && other_court_balls_.size() < 10) {
            OtherCourtBall new_ball;
            new_ball.cx = hist.cx;
            new_ball.cy = hist.cy;
            new_ball.vx = hist.avg_vx;
            new_ball.vy = hist.avg_vy;
            new_ball.first_seen_frame = current_frame_;
            new_ball.last_seen_frame = current_frame_;
            new_ball.confirm_count = 1;
            new_ball.is_confirmed = false;
            new_ball.id = hist.id;
            other_court_balls_.push_back(new_ball);
        }
    }

    bool check_direction_mismatch(const BallHistory& hist)
    {
        if (hist.velocity_samples <= 5) return false;
        if (std::fabs(last_target_vx_) <= 5.0f && std::fabs(last_target_vy_) <= 5.0f) return false;
        
        float target_speed = std::sqrt(last_target_vx_ * last_target_vx_ + 
                                       last_target_vy_ * last_target_vy_);
        float hist_speed = std::sqrt(hist.avg_vx * hist.avg_vx + 
                                     hist.avg_vy * hist.avg_vy);
        
        if (target_speed <= 5.0f || hist_speed <= 5.0f) return false;
        
        float dot = (last_target_vx_ * hist.avg_vx + last_target_vy_ * hist.avg_vy) / 
                    (target_speed * hist_speed);
        
        return dot < other_court_direction_threshold_;
    }

    OtherCourtBall* find_other_court_ball(float cx, float cy)
    {
        for (auto& ball : other_court_balls_) {
            float d = std::sqrt((cx - ball.cx) * (cx - ball.cx) + 
                                (cy - ball.cy) * (cy - ball.cy));
            if (d < 100.0f) {
                return &ball;
            }
        }
        return nullptr;
    }
    
    bool is_other_court_ball(const T_DetectObject& det)
    {
        float cx = (det.xmin + det.xmax) / 2.0f;
        float cy = (det.ymin + det.ymax) / 2.0f;
        
        for (const auto& ball : other_court_balls_) {
            if (!ball.is_confirmed) continue;
            
            float dist = std::sqrt((cx - ball.cx) * (cx - ball.cx) + 
                                   (cy - ball.cy) * (cy - ball.cy));
            
            float predicted_cx = ball.cx + ball.vx * (current_frame_ - ball.last_seen_frame);
            float predicted_cy = ball.cy + ball.vy * (current_frame_ - ball.last_seen_frame);
            float dist_predicted = std::sqrt((cx - predicted_cx) * (cx - predicted_cx) + 
                                             (cy - predicted_cy) * (cy - predicted_cy));
            
            if (dist < 80.0f || dist_predicted < 120.0f) {
                return true;
            }
        }
        return false;
    }

    // ===== 距离计算 =====
    
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

        // 置信度
        score += candidate.det.score * 100.0f;

        // 跟踪器距离
        if (has_tracker && candidate.distance_to_tracker < 1e8f) {
            if (candidate.distance_to_tracker < 500.0f) {
                score += (500.0f - candidate.distance_to_tracker) * 0.5f;
            }
        }

        // 运动历史奖励
        if (candidate.ever_moved) {
            score += 350.0f;
        }

        // 静止惩罚
        if (candidate.is_static) {
            score -= candidate.static_frames * 5.0f;
        }

        // 从未运动过的球额外惩罚
        if (!candidate.ever_moved) {
            if (candidate.static_frames > 5) score -= 400.0f;
            if (candidate.static_frames > 15) score -= 300.0f;
        }

        // 与上次目标位置的距离
        if (has_locked_target_ && candidate.distance_to_last_target < 1e8f) {
            if (candidate.distance_to_last_target < 150.0f) {
                score += (150.0f - candidate.distance_to_last_target) * 2.0f;
            }
            if (candidate.distance_to_last_target > 300.0f) {
                score -= (candidate.distance_to_last_target - 300.0f) * 0.5f;
            }
            if (candidate.distance_to_last_target > target_switch_threshold_ && 
                frames_since_target_lost_ < target_search_grace_period_) {
                score -= 800.0f;
            }
        }

        // 裁剪框内优先
        if (use_crop_center_bias_) {
            if (candidate.is_in_crop_region) {
                score += 300.0f;
            } else {
                if (candidate.distance_to_crop_center > 400.0f) score -= 150.0f;
                if (candidate.distance_to_crop_center > 800.0f) score -= 250.0f;
                if (candidate.distance_to_crop_center > 1200.0f) score -= 350.0f;
            }
        }
        
        // 疑似其他球场的球降分
        if (candidate.is_likely_other_court) {
            score -= 500.0f;
        }

        return score;
    }
};

/*------------------------------------------------*/
/**
 * 调试绘制器
 */
class DebugDrawer {
public:
    struct DrawContext {
        image_buffer_t* image;
        image_rect_t crop_rect;
        int crop_width;
        int crop_height;
    };

    static void draw_all(const DrawContext& ctx,
                         const std::vector<T_DetectObject>& valid_detections,
                         const std::vector<T_DetectObject>& selected_balls,
                         const std::vector<T_TrackObject>& track_results,
                         const std::vector<std::pair<float, float>>& blacklist_positions,
                         const std::vector<std::pair<float, float>>& other_court_positions,
                         const BallSelector& selector,
                         const TrackFrame& tracker,
                         int frame_count,
                         size_t all_count, size_t valid_count, size_t excluded_count)
    {
        // 绘制黑名单
        for (const auto& pos : blacklist_positions) {
            draw_blacklist_marker(ctx, pos.first, pos.second);
        }

        // 绘制其他球场球
        for (const auto& pos : other_court_positions) {
            draw_other_court_marker(ctx, pos.first, pos.second);
        }

        // 绘制检测框
        for (const auto& det : valid_detections) {
            draw_detection(ctx, det, COLOR_WHITE, 1);
        }

        // 绘制选中的目标
        for (const auto& det : selected_balls) {
            draw_detection(ctx, det, COLOR_BLUE, 3);
            draw_target_label(ctx, det);
        }

        // 绘制跟踪框
        for (const auto& trk : track_results) {
            draw_track(ctx, trk);
        }

        // 绘制调试信息
        draw_debug_info(ctx, selector, tracker, frame_count, 
                        all_count, valid_count, excluded_count);
    }

private:
    static int to_crop_x(const DrawContext& ctx, int full_x) {
        return full_x - ctx.crop_rect.left;
    }
    
    static int to_crop_y(const DrawContext& ctx, int full_y) {
        return full_y - ctx.crop_rect.top;
    }
    
    static bool is_in_crop(const DrawContext& ctx, int full_x, int full_y) {
        return full_x >= ctx.crop_rect.left && full_x < ctx.crop_rect.right &&
               full_y >= ctx.crop_rect.top && full_y < ctx.crop_rect.bottom;
    }

    static void draw_blacklist_marker(const DrawContext& ctx, float fx, float fy)
    {
        int cx = static_cast<int>(fx);
        int cy = static_cast<int>(fy);
        
        if (!is_in_crop(ctx, cx, cy)) return;
        
        int crop_cx = to_crop_x(ctx, cx);
        int crop_cy = to_crop_y(ctx, cy);
        int radius = 45;
        
        draw_rectangle(ctx.image, crop_cx - radius, crop_cy - radius,
                       radius * 2, radius * 2, COLOR_PURPLE, 3);
        
        // 画X
        for (int i = -radius + 5; i < radius - 5; i++) {
            int px = crop_cx + i;
            int py1 = crop_cy + i;
            int py2 = crop_cy - i;
            if (px >= 0 && px < ctx.crop_width) {
                if (py1 >= 0 && py1 < ctx.crop_height) {
                    draw_rectangle(ctx.image, px, py1, 3, 3, COLOR_PURPLE, 1);
                }
                if (py2 >= 0 && py2 < ctx.crop_height) {
                    draw_rectangle(ctx.image, px, py2, 3, 3, COLOR_PURPLE, 1);
                }
            }
        }
        
        if (crop_cy - radius - 25 > 0) {
            draw_text(ctx.image, "[BLACKLIST]", crop_cx - 40, crop_cy - radius - 25, COLOR_PURPLE, 12);
        }
    }

    static void draw_other_court_marker(const DrawContext& ctx, float fx, float fy)
    {
        int cx = static_cast<int>(fx);
        int cy = static_cast<int>(fy);
        
        if (!is_in_crop(ctx, cx, cy)) return;
        
        int crop_cx = to_crop_x(ctx, cx);
        int crop_cy = to_crop_y(ctx, cy);
        int radius = 40;
        
        draw_rectangle(ctx.image, crop_cx - radius, crop_cy - radius,
                       radius * 2, radius * 2, COLOR_ORANGE, 3);
        
        if (crop_cy - radius - 25 > 0) {
            draw_text(ctx.image, "[OTHER]", crop_cx - 30, crop_cy - radius - 25, COLOR_ORANGE, 12);
        }
    }

    static void draw_detection(const DrawContext& ctx, const T_DetectObject& det, 
                               unsigned int color, int thickness)
    {
        int x1 = to_crop_x(ctx, det.xmin);
        int y1 = to_crop_y(ctx, det.ymin);
        int w = det.xmax - det.xmin;
        int h = det.ymax - det.ymin;
        
        if (x1 + w < 0 || x1 >= ctx.crop_width || y1 + h < 0 || y1 >= ctx.crop_height) {
            return;
        }
        
        draw_rectangle(ctx.image, x1, y1, w, h, color, thickness);
    }

    static void draw_target_label(const DrawContext& ctx, const T_DetectObject& det)
    {
        int x1 = to_crop_x(ctx, det.xmin);
        int y1 = to_crop_y(ctx, det.ymin);
        
        char text[64];
        snprintf(text, sizeof(text), "TARGET %.1f%%", det.score * 100);
        if (y1 - 25 > 0) {
            draw_text(ctx.image, text, x1, y1 - 25, COLOR_RED, 12);
        }
    }

    static void draw_track(const DrawContext& ctx, const T_TrackObject& trk)
    {
        int x1 = to_crop_x(ctx, trk.xmin);
        int y1 = to_crop_y(ctx, trk.ymin);
        int w = trk.xmax - trk.xmin;
        int h = trk.ymax - trk.ymin;
        
        if (x1 + w < 0 || x1 >= ctx.crop_width || y1 + h < 0 || y1 >= ctx.crop_height) {
            return;
        }
        
        draw_rectangle(ctx.image, x1, y1, w, h, COLOR_GREEN, 3);
        
        if (trk.is_predicted && y1 - 50 > 0) {
            draw_text(ctx.image, "[PRED]", x1, y1 - 50, COLOR_YELLOW, 12);
        }
    }

    static void draw_debug_info(const DrawContext& ctx,
                                const BallSelector& selector,
                                const TrackFrame& tracker,
                                int frame_count,
                                size_t all_count, size_t valid_count, size_t excluded_count)
    {
        int text_y = 40;
        int line_height = 45;
        int font_size = 20;
        
        int blacklist_count, other_court_count;
        selector.get_debug_info(blacklist_count, other_court_count);
        
        char text[128];
        
        snprintf(text, sizeof(text), "Detect: All=%zu Valid=%zu Excl=%zu",
                 all_count, valid_count, excluded_count);
        draw_text(ctx.image, text, 10, text_y, COLOR_YELLOW, font_size);
        text_y += line_height;
        
        snprintf(text, sizeof(text), "Select: Lost=%d Track=%d",
                 selector.get_frames_since_lost(), tracker.HasActiveTrack() ? 1 : 0);
        draw_text(ctx.image, text, 10, text_y, COLOR_YELLOW, font_size);
        text_y += line_height;
        
        snprintf(text, sizeof(text), "Filter: BL=%d Other=%d",
                 blacklist_count, other_court_count);
        draw_text(ctx.image, text, 10, text_y, COLOR_PURPLE, font_size);
        text_y += line_height;
        
        snprintf(text, sizeof(text), "Crop: [%d,%d]-[%d,%d]",
                 ctx.crop_rect.left, ctx.crop_rect.top, 
                 ctx.crop_rect.right, ctx.crop_rect.bottom);
        draw_text(ctx.image, text, 10, text_y, COLOR_RED, font_size);
        text_y += line_height;
        
        snprintf(text, sizeof(text), "Frame: %d", frame_count);
        draw_text(ctx.image, text, 10, text_y, COLOR_WHITE, font_size);
    }
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
            printf("[Producer] Progress: %zu / %zu frames (%.1f%%)\n", 
                   i, total_frames, 100.0 * i / total_frames);
        }
    }

    printf("[Producer] Finished reading %zu frames\n", i);
}

/*------------------------------------------------*/
void consumer_thread(FrameQueue& fq, ImageBufferPool& pool, const char *model_path, const char *out_dir)
{
    // 初始化模型
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();
    int ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret) {
        printf("[Consumer] init_yolov8_model failed!\n");
        return;
    }

    // 创建输出目录
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    system(cmd);

    // 初始化组件
    SmoothCameraController camera(PIC_FULL_WIDTH, PIC_FULL_HEIGHT, 
                                  ALG_CROP_WIDTH, ALG_CROP_HEIGHT,
                                  VALID_TOP, VALID_BOTTOM);

    SplitRegionDetector::DetectionConfig det_config = {
        .full_width = PIC_FULL_WIDTH,
        .full_height = PIC_FULL_HEIGHT,
        .valid_top = VALID_TOP,
        .valid_bottom = VALID_BOTTOM,
        .exclude_right_width = EXCLUDE_RIGHT_WIDTH,
        .num_splits = 2
    };
    SplitRegionDetector detector(det_config);
    
    BallSelector ball_selector(0, detector.get_exclude_right_start(), VALID_TOP, VALID_BOTTOM);
    
    TrackFrame tracker;
    tracker.Init(50);

    // 状态变量
    int frame_count = 0;
    int frame_track_count = 0;
    bool is_initialized = false;
    int init_frames = 0;
    const int MAX_INIT_FRAMES = 30;
    
    T_TrackObject last_track_result;
    bool has_last_track = false;

    printf("[Consumer] === SPLIT DETECTION MODE ===\n");
    printf("[Consumer] Image: %dx%d, Crop: %dx%d\n",
           PIC_FULL_WIDTH, PIC_FULL_HEIGHT, ALG_CROP_WIDTH, ALG_CROP_HEIGHT);
    printf("[Consumer] Output: CROPPED (%dx%d)\n", ALG_CROP_WIDTH, ALG_CROP_HEIGHT);

    // 主处理循环
    image_buffer_t src_image = {0};
    image_buffer_t crop_image_buf = {0};

    while (true) {
        frame_track_count++;
        
        // 获取输入帧
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

        // 更新相机位置
        camera.update_position_only();
        image_rect_t crop_rect = camera.get_rect();
        float crop_cx = (crop_rect.left + crop_rect.right) / 2.0f;
        float crop_cy = (crop_rect.top + crop_rect.bottom) / 2.0f;

        // 执行检测
        auto det_result = detector.detect(&src_image, &rknn_app_ctx);

        // 设置裁剪框中心偏好，让球筛选器知道裁剪框在哪，从而优先选择裁剪框内或附近的球，避免镜头突然跳到远处的干扰球。
        ball_selector.set_crop_center(crop_cx, crop_cy);

        // 球筛选
        std::vector<T_DetectObject> ball_detections;
        bool found_ball = false;
        
        if (!det_result.valid_detections.empty()) {
            const T_DetectObject* target = ball_selector.select_target_ball(
                det_result.valid_detections,
                has_last_track ? &last_track_result : nullptr,
                has_last_track);

            if (target != nullptr) {
                ball_detections.push_back(*target);
                found_ball = true;
            }
        }

        // 初始化阶段
        if (!is_initialized) {
            init_frames++;
            if (found_ball) {
                const auto& target = ball_detections[0];
                float target_cx = (target.xmin + target.xmax) / 2.0f;
                float target_cy = (target.ymin + target.ymax) / 2.0f;
                camera.set_center(target_cx, target_cy);
                is_initialized = true;
                printf("[Consumer][INIT] Target at (%.0f, %.0f)\n", target_cx, target_cy);
            } else if (init_frames >= MAX_INIT_FRAMES) {
                printf("[Consumer][INIT] Max init frames reached\n");
                is_initialized = true;
            }
        }

        // 跟踪处理
        std::vector<T_TrackObject> track_results;
        if (!ball_detections.empty() || tracker.HasActiveTrack()) {
            tracker.ProcessFrame(frame_track_count, ball_detections, track_results);
        }

        // 保存跟踪结果
        if (!track_results.empty()) {
            last_track_result = track_results[0];
            has_last_track = true;
        } else if (!tracker.HasActiveTrack()) {
            has_last_track = false;
        }

        // 更新运镜
        if (is_initialized && !track_results.empty()) {
            auto& t = track_results[0];
            camera.update_by_target(t.xmin, t.ymin, t.xmax, t.ymax);
        }

        if (!found_ball) {
            camera.mark_no_target();
        }

        // 裁剪输出图
        camera.crop_current_window(&src_image, &crop_image_buf);

        // 绘制调试信息
        DebugDrawer::DrawContext draw_ctx = {
            .image = &crop_image_buf,
            .crop_rect = crop_rect,
            .crop_width = ALG_CROP_WIDTH,
            .crop_height = ALG_CROP_HEIGHT
        };
        
        DebugDrawer::draw_all(draw_ctx,
                              det_result.valid_detections,
                              ball_detections,
                              track_results,
                              ball_selector.get_blacklist_positions(),
                              ball_selector.get_other_court_positions(),
                              ball_selector,
                              tracker,
                              frame_count,
                              det_result.all_detections.size(),
                              det_result.valid_detections.size(),
                              det_result.excluded_detections.size());

        // 保存裁切图
        char out_path[256];
        snprintf(out_path, sizeof(out_path), "%s/%06d.jpg", out_dir, frame_count);
        write_image(out_path, &crop_image_buf);
        
        frame_count++;
        
        // 释放资源
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
        return -1;
    }

    const char *model_path = argv[1];
    const char *frames_dir = argv[2];
    const char *out_dir = argv[3];

    constexpr size_t QUEUE_SIZE = 12;
    constexpr size_t POOL_SIZE  = 16;
    constexpr size_t IMAGE_SIZE = PIC_FULL_WIDTH * PIC_FULL_HEIGHT * 3;
    
    printf("[Main] Image buffer: %zu bytes (%.2f MB) for %dx%d\n",
           IMAGE_SIZE, IMAGE_SIZE / 1024.0 / 1024.0, PIC_FULL_WIDTH, PIC_FULL_HEIGHT);

    signal(SIGINT, signal_handler);

    ImageBufferPool buffer_pool(POOL_SIZE, IMAGE_SIZE);
    FrameQueue frame_queue(QUEUE_SIZE);

    std::atomic<bool> producer_done{false};

    std::thread producer([&] {
        producer_thread(frame_queue, buffer_pool, frames_dir);
        producer_done.store(true);
        printf("[Producer] Thread exiting\n");
        frame_queue.stop();
    });

    std::thread consumer([&] {
        consumer_thread(frame_queue, buffer_pool, model_path, out_dir);
    });

    printf("[Main] Pipeline started\n");

    while (!g_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        printf("[Main] Queue: %zu, Pool: %zu\n", frame_queue.size(), buffer_pool.available());

        if (producer_done.load() && frame_queue.empty()) {
            break;
        }
    }

    if (g_exit.load()) {
        printf("[Main] SIGINT received, stopping...\n");
        frame_queue.stop();
        buffer_pool.stop();
    }

    producer.join();
    consumer.join();

    printf("[Main] Exit clean\n");
    return 0;
}