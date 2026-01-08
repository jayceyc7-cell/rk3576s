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


/**
 * 丝滑运镜控制器 - 基于物理模型的平滑相机运动
 * 
 * 有效显示区域：2560 × 1130（中间区域）
 * 顶部留白：155 像素
 * 底部留白：155 像素
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
        // 设置有效显示区域
        valid_top_ = valid_top;
        valid_bottom_ = (valid_bottom < 0) ? img_h : valid_bottom;
        valid_height_ = valid_bottom_ - valid_top_;
        
        // 初始位置：有效区域中心
        cx_ = img_w_ * 0.5f;
        cy_ = valid_top_ + valid_height_ * 0.5f;
        
        // 初始速度为 0
        vx_ = 0.0f;
        vy_ = 0.0f;
        
        // 目标位置初始化
        target_cx_ = cx_;
        target_cy_ = cy_;
        
        // 目标速度估计（用于预测）
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

    // 收到跟踪目标时调用
    void update_by_target(int xmin, int ymin, int xmax, int ymax)
    {
        float new_target_cx = 0.5f * (xmin + xmax);
        float new_target_cy = 0.5f * (ymin + ymax);
        
        // 估计目标速度
        target_vx_ = velocity_smooth_ * (new_target_cx - last_target_cx_) 
                   + (1.0f - velocity_smooth_) * target_vx_;
        target_vy_ = velocity_smooth_ * (new_target_cy - last_target_cy_) 
                   + (1.0f - velocity_smooth_) * target_vy_;
        
        last_target_cx_ = target_cx_;
        last_target_cy_ = target_cy_;
        
        // 预测性跟踪
        target_cx_ = new_target_cx + target_vx_ * prediction_frames_;
        target_cy_ = new_target_cy + target_vy_ * prediction_frames_;
        
        limit_target();
        
        frames_without_target_ = 0;
        has_target_ = true;
    }

    // 没有检测到目标时调用
    void mark_no_target()
    {
        frames_without_target_++;
        
        if (frames_without_target_ > return_to_center_delay_) {
            // 缓慢将目标设为有效区域中心
            float center_x = img_w_ * 0.5f;
            float center_y = valid_top_ + valid_height_ * 0.5f;
            
            target_cx_ = target_cx_ + (center_x - target_cx_) * return_to_center_speed_;
            target_cy_ = target_cy_ + (center_y - target_cy_) * return_to_center_speed_;
        }
    }

private:
    // ================= 图像/裁剪参数 =================
    int img_w_, img_h_;
    int crop_w_, crop_h_;
    image_rect_t rect_;
    
    // ================= 有效显示区域 =================
    int valid_top_;       // 有效区域顶部 Y 坐标
    int valid_bottom_;    // 有效区域底部 Y 坐标
    int valid_height_;    // 有效区域高度

    // ================= 相机状态 =================
    float cx_, cy_;
    float vx_, vy_;
    
    // ================= 目标状态 =================
    float target_cx_, target_cy_;
    float target_vx_, target_vy_;
    float last_target_cx_, last_target_cy_;
    
    bool has_target_{false};
    int frames_without_target_{0};

    // ================= 运镜参数 =================
    const float stiffness_ = 0.04f;
    const float damping_ = 0.4f;
    const float max_speed_ = 30.0f;
    const float max_accel_ = 3.0f;
    const float dead_zone_ratio_ = 0.15f;
    const float prediction_frames_ = 3.0f;
    const float velocity_smooth_ = 0.3f;
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

    // 限制目标位置在有效区域内
    void limit_target()
    {
        // X 方向：整个图像宽度
        target_cx_ = std::max(crop_w_ * 0.5f,
                     std::min(target_cx_, img_w_ - crop_w_ * 0.5f));
        
        // Y 方向：仅在有效区域内
        float min_cy = valid_top_ + crop_h_ * 0.5f;
        float max_cy = valid_bottom_ - crop_h_ * 0.5f;
        target_cy_ = std::max(min_cy, std::min(target_cy_, max_cy));
    }

    // 限制相机中心在有效区域内
    void limit_center()
    {
        // X 方向
        cx_ = std::max(crop_w_ * 0.5f,
              std::min(cx_, img_w_ - crop_w_ * 0.5f));
        
        // Y 方向：仅在有效区域内
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

    // 限制裁剪框在有效区域内
    void limit_rect()
    {
        // X 方向限制
        if (rect_.left < 0) {
            rect_.left = 0;
            rect_.right = crop_w_;
        }
        if (rect_.right > img_w_) {
            rect_.right = img_w_;
            rect_.left = img_w_ - crop_w_;
        }
        
        // Y 方向限制：在有效区域内
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
    
    // 有效区域：顶部 155 到底部 1440-155=1285
    // 即 y ∈ [155, 1285]，高度 1130

    
    SmoothCameraController camera(
        PIC_FULL_WIDTH, 
        PIC_FULL_HEIGHT, 
        ALG_CROP_WIDTH, 
        ALG_CROP_HEIGHT,
        VALID_TOP,      // 有效区域顶部 Y
        VALID_BOTTOM    // 有效区域底部 Y
    );

    while (true) {
        frame_track_count++;
        
        if (src_image.width == 0 && src_image.height == 0) {
            memset(&src_image, 0, sizeof(image_buffer_t));
            if (!fq.pop(src_image)) {
                printf("[Consumer] Queue stopped,  exiting\n");
                break;
            }
        }
        
        memset(&crop_image_buf, 0, sizeof(image_buffer_t));

        // 获取动态裁切窗口（每帧都会更新相机位置）
        camera.get_crop_window(&src_image, &crop_image_buf);

        object_detect_result_list od_results; 
        if (crop_image_buf.virt_addr == NULL) {
            printf("[Consumer] crop_image is NULL\n");
            pool.release(src_image);
            memset(&src_image, 0, sizeof(image_buffer_t));
            continue;
        }

        inference_yolov8_model(&rknn_app_ctx, &crop_image_buf, &od_results);

        char text[256];
        bool found_ball = false;
        
        for (int j = 0; j < od_results.count; j++) {
            object_detect_result *det = &(od_results.results[j]);

            int x1 = det->box.left;
            int y1 = det->box.top;
            int x2 = det->box.right;
            int y2 = det->box.bottom;

            snprintf(text, sizeof(text), "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
            
            if (strncmp(text, "ball", 4) == 0) {
                found_ball = true;
                
                printf("[Consumer] cls_id:%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det->cls_id),
                       det->box.left, det->box.top,
                       det->box.right, det->box.bottom,
                       det->prop);

                draw_rectangle(&crop_image_buf, x1, y1, (x2 - x1), (y2 - y1), COLOR_BLUE, 3);
                draw_text(&crop_image_buf, text, x1, y1 - 20, COLOR_RED, 10);

                std::vector<T_DetectObject> detections;
                for (int k = 0; k < od_results.count; k++) {
                    auto &det_obj = od_results.results[k];

                    T_DetectObject obj;
                    obj.cls_id = det_obj.cls_id;
                    obj.score = det_obj.prop;
                    obj.xmin = det_obj.box.left;
                    obj.ymin = det_obj.box.top;
                    obj.xmax = det_obj.box.right;
                    obj.ymax = det_obj.box.bottom;

                    detections.push_back(obj);
                }
                
                std::vector<T_TrackObject> track_results;
                tracker.ProcessFrame(frame_track_count, crop_image_buf, detections, track_results);

                printf("[Consumer] track_results.size() = %zu\n", track_results.size());
                if (!track_results.empty()) {
                    printf("[Consumer] draw_rectangle @ (%d %d %d %d)\n", 
                           track_results[0].xmin, track_results[0].ymin, 
                           track_results[0].xmax - track_results[0].xmin, 
                           track_results[0].ymax - track_results[0].ymin);
                    draw_rectangle(&crop_image_buf, track_results[0].xmin, track_results[0].ymin,
                                   track_results[0].xmax - track_results[0].xmin, 
                                   track_results[0].ymax - track_results[0].ymin, COLOR_GREEN, 3);

                    auto &t = track_results[0];

                    // 裁剪图坐标 → 原图坐标
                    int oxmin = camera.get_rect().left + t.xmin;
                    int oymin = camera.get_rect().top + t.ymin;
                    int oxmax = camera.get_rect().left + t.xmax;
                    int oymax = camera.get_rect().top + t.ymax;

                    // 更新目标位置（运镜控制器会平滑处理）
                    camera.update_by_target(oxmin, oymin, oxmax, oymax);
                }

                break;
            } 
        }

        // 没有检测到球时，通知运镜控制器
        if (!found_ball) {
            camera.mark_no_target();
        }

        char out_path[256];
        snprintf(out_path, sizeof(out_path), "%s/%06d.jpg", out_dir, frame_count);
        write_image(out_path, &crop_image_buf);
        frame_count++;
        
        if (frame_count % 100 == 0) {
            printf("[Consumer] Processed %d frames\n", frame_count);
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