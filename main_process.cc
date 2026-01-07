
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <string>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "awi_track.hpp"

#include <thread>
#include <csignal>
#include <atomic>
#include <iostream>
#include <unistd.h> 

/*--------------------------------------*/
#define PIC_FULL_WIDTH 2560
#define PIC_FULL_HEIGHT 1440
#define ALG_CROP_WIDTH 1280
#define ALG_CROP_HEIGHT 736

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
    int lenstr = strlen(str);
    int lensuffix = strlen(suffix);
    if (lensuffix > lenstr)
        return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

/*------------------------------------------------*/
#include <vector>
#include <queue>
#include <mutex>

class ImageBufferPool {
public:
    ImageBufferPool(size_t count, int buf_size)
        : buf_size_(buf_size)
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
            free(buf.virt_addr);
        }
    }

    bool acquire(image_buffer_t& out)
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
    }

private:
    size_t buf_size_;
    std::queue<image_buffer_t> free_queue_;
    std::mutex mutex_;
};

template <typename T>
inline T clamp(T v, T lo, T hi)
{
    return (v < lo) ? lo : (v > hi) ? hi : v;
}


class crop_window {
public:
    crop_window(int img_w, int img_h, int crop_w, int crop_h)
        : img_w_(img_w),
          img_h_(img_h),
          crop_w_(crop_w),
          crop_h_(crop_h)
    {
        // 初始裁剪中心：整图中心
        cx_ = img_w_ * 0.5f;
        cy_ = img_h_ * 0.5f;

        target_cx_ = cx_;
        target_cy_ = cy_;

        vx_ = vy_ = 0.0f;
        
        // 预测状态初始化
        pred_cx_ = cx_;
        pred_cy_ = cy_;
        pred_vx_ = 0.0f;
        pred_vy_ = 0.0f;
        
        miss_frames_ = 0;
        confidence_ = 1.0f;

        update_rect();
    }

    const image_rect_t &get_rect() const
    {
        return rect_;
    }

    // 每一帧调用：画框 + 裁剪
    void get_crop_window(image_buffer_t* src, image_buffer_t* dst)
    {
        draw_crop_rect(src);
        crop_image(src, dst);
    }

    void reset_to_center()
    {
        vx_ = vy_ = 0.0f;
        cx_ = img_w_ * 0.5f;
        cy_ = img_h_ * 0.5f;
        target_cx_ = cx_;
        target_cy_ = cy_;
        
        pred_cx_ = cx_;
        pred_cy_ = cy_;
        pred_vx_ = 0.0f;
        pred_vy_ = 0.0f;
        
        miss_frames_ = 0;
        confidence_ = 1.0f;
        
        update_rect();
    }

    // ============ 核心改进：舒适区运镜逻辑 ============
    void update_by_target(int xmin, int ymin, int xmax, int ymax, bool is_detection = true)
    {
        // ===== 调试日志 =====
        printf("\n========== update_by_target DEBUG ==========\n");
        printf("输入目标框（原图坐标）: (%d, %d, %d, %d)\n", xmin, ymin, xmax, ymax);
        printf("当前窗口中心（原图坐标）: (%.1f, %.1f)\n", cx_, cy_);
        printf("裁剪窗口大小: %d x %d\n", crop_w_, crop_h_);

        // 输入参数是原图坐标系的目标框
        // cx_, cy_ 是当前裁剪窗口中心在原图坐标系的位置

        // 1. 目标中心（原图坐标系）
        float target_cx_origin = 0.5f * (xmin + xmax);
        float target_cy_origin = 0.5f * (ymin + ymax);

        printf("目标中心（原图坐标）: (%.1f, %.1f)\n", target_cx_origin, target_cy_origin);

        // 2. 目标相对于裁剪窗口中心的偏移（原图坐标系）
        float offset_x = target_cx_origin - cx_;
        float offset_y = target_cy_origin - cy_;

        printf("目标偏移量: (%.1f, %.1f)\n", offset_x, offset_y);

        // ============ 舒适区判断 ============
        float comfort_zone_half_w = crop_w_ * comfort_zone_ratio_w_;
        float comfort_zone_half_h = crop_h_ * comfort_zone_ratio_h_;

        printf("舒适区半宽高: (%.1f, %.1f)\n", comfort_zone_half_w, comfort_zone_half_h);

        // 计算目标超出舒适区的距离
        float exceed_x = 0.0f;
        float exceed_y = 0.0f;

        if (fabs(offset_x) > comfort_zone_half_w)
        {
            exceed_x = offset_x - (offset_x > 0 ? comfort_zone_half_w : -comfort_zone_half_w);
        }

        if (fabs(offset_y) > comfort_zone_half_h)
        {
            exceed_y = offset_y - (offset_y > 0 ? comfort_zone_half_h : -comfort_zone_half_h);
        }

        printf("超出舒适区距离: (%.1f, %.1f)\n", exceed_x, exceed_y);

        // 4. 根据来源调整置信度
        if (is_detection)
        {
            confidence_ = 1.0f;
            miss_frames_ = 0;

            if (exceed_x != 0.0f || exceed_y != 0.0f)
            {
                float dt = 1.0f;
                pred_vx_ = exceed_x / dt;
                pred_vy_ = exceed_y / dt;
            }
            pred_cx_ = cx_ + exceed_x;
            pred_cy_ = cy_ + exceed_y;

            printf("检测模式: 置信度=1.0\n");
        }
        else
        {
            miss_frames_++;
            float decay_factor = miss_frames_ * 0.1f;
            confidence_ = std::max(0.3f, 1.0f - decay_factor);

            exceed_x *= 0.8f;
            exceed_y *= 0.8f;

            pred_cx_ = 0.7f * pred_cx_ + 0.3f * (cx_ + exceed_x);
            pred_cy_ = 0.7f * pred_cy_ + 0.3f * (cy_ + exceed_y);

            printf("跟踪模式: 置信度=%.2f, 丢失帧数=%d\n", confidence_, miss_frames_);
        }

        // 5. 在舒适区内：不触发运镜
        if (fabs(exceed_x) < 1.0f && fabs(exceed_y) < 1.0f)
        {
            printf(">>> 目标在舒适区内，镜头保持静止\n");
            printf("============================================\n\n");

            vx_ *= 0.85f;
            vy_ *= 0.85f;

            cx_ += vx_;
            cy_ += vy_;

            limit_center();
            update_rect();
            return;
        }

        printf(">>> 目标超出舒适区，开始运镜\n");

        // ============ 超出舒适区：开始运镜 ============
        // 6. 计算目标位置（原图坐标系）
        float new_target_cx = cx_ + exceed_x;
        float new_target_cy = cy_ + exceed_y;

        target_cx_ = new_target_cx;
        target_cy_ = new_target_cy;

        printf("运镜目标位置: (%.1f, %.1f)\n", target_cx_, target_cy_);

        // 7. 位置误差
        float ex = target_cx_ - cx_;
        float ey = target_cy_ - cy_;

        printf("位置误差: (%.1f, %.1f)\n", ex, ey);

        // 8. 微小死区
        const float micro_dead_zone = 2.0f;
        if (fabs(ex) < micro_dead_zone)
            ex = 0.0f;
        if (fabs(ey) < micro_dead_zone)
            ey = 0.0f;

        // 9. 动态参数
        float ratio_x = fabs(exceed_x) / comfort_zone_half_w;
        float ratio_y = fabs(exceed_y) / comfort_zone_half_h;
        float exceed_ratio = std::min(1.0f, std::max(ratio_x, ratio_y));

        float k_p = 0.05f + 0.03f * exceed_ratio * confidence_;
        float k_d = 0.93f + 0.02f * (1.0f - confidence_);
        float max_v = 50.0f + 30.0f * exceed_ratio * confidence_;

        printf("控制参数: k_p=%.3f, k_d=%.3f, max_v=%.1f\n", k_p, k_d, max_v);

        // 10. 加速度
        float ax = k_p * ex;
        float ay = k_p * ey;

        // 11. 更新速度
        vx_ = k_d * vx_ + ax;
        vy_ = k_d * vy_ + ay;

        // 12. 限速
        vx_ = clamp(vx_, -max_v, max_v);
        vy_ = clamp(vy_, -max_v, max_v);

        printf("速度: (%.2f, %.2f)\n", vx_, vy_);

        // 13. 积分得到位置
        cx_ += vx_;
        cy_ += vy_;

        printf("新窗口中心: (%.1f, %.1f)\n", cx_, cy_);

        // 14. 边界限制
        limit_center();
        update_rect();

        printf("最终窗口: left=%d, top=%d, right=%d, bottom=%d\n",
               rect_.left, rect_.top, rect_.right, rect_.bottom);
        printf("============================================\n\n");
    }

    // ============ 完全丢失时的惯性更新 ============
    void update_with_prediction()
    {
        miss_frames_++;
        
        // 置信度快速衰减
        float decay_factor = miss_frames_ * 0.15f;
        confidence_ = std::max(0.0f, 1.0f - decay_factor);
        
        // 使用预测位置 + 速度衰减
        float decay_multiplier = miss_frames_ * 0.1f;
        float decay = std::exp(-decay_multiplier);
        
        pred_cx_ += pred_vx_ * decay;
        pred_cy_ += pred_vy_ * decay;
        
        // 预测速度也衰减
        pred_vx_ *= 0.95f;
        pred_vy_ *= 0.95f;
        
        // 使用预测位置作为目标
        target_cx_ = pred_cx_;
        target_cy_ = pred_cy_;
        
        // 保持平滑运动（高阻尼）
        float ex = target_cx_ - cx_;
        float ey = target_cy_ - cy_;
        
        const float k_p = 0.02f;  // 惯性模式：极低增益
        const float k_d = 0.97f;  // 高阻尼
        const float max_v = 30.0f; // 低速度上限
        
        float ax = k_p * ex;
        float ay = k_p * ey;
        
        vx_ = k_d * vx_ + ax;
        vy_ = k_d * vy_ + ay;
        
        vx_ = clamp(vx_, -max_v, max_v);
        vy_ = clamp(vy_, -max_v, max_v);
        
        cx_ += vx_;
        cy_ += vy_;
        
        limit_center();
        update_rect();
    }
    
    // 获取当前置信度（用于外部判断）
    float get_confidence() const { return confidence_; }
    int get_miss_frames() const { return miss_frames_; }
    
    // ============ 新增：设置舒适区大小 ============
    void set_comfort_zone(float ratio_w, float ratio_h) {
        comfort_zone_ratio_w_ = clamp(ratio_w, 0.1f, 0.5f);
        comfort_zone_ratio_h_ = clamp(ratio_h, 0.1f, 0.5f);
    }
    
    // 获取舒适区参数（用于可视化）
    void get_comfort_zone_rect(int& left, int& top, int& width, int& height) const {
        float half_w = crop_w_ * comfort_zone_ratio_w_;
        float half_h = crop_h_ * comfort_zone_ratio_h_;
        
        left = static_cast<int>(crop_w_ * 0.5f - half_w);
        top = static_cast<int>(crop_h_ * 0.5f - half_h);
        width = static_cast<int>(half_w * 2);
        height = static_cast<int>(half_h * 2);
    }

private:
    // ================= 图像 / 裁剪参数 =================
    int img_w_, img_h_;
    int crop_w_, crop_h_;

    // ================= 裁剪中心（连续） =================
    float cx_, cy_;
    float target_cx_, target_cy_;

    image_rect_t rect_;

    // ================= 舒适区参数（核心调参区）=================
    // 舒适区大小 = 裁剪窗口大小 × ratio
    // 推荐值：0.15-0.25（即裁剪窗口中心的15%-25%区域）
    float comfort_zone_ratio_w_ = 0.50f;  // 宽度比例：推荐0.15-0.30
    float comfort_zone_ratio_h_ = 0.50f;  // 高度比例：推荐0.15-0.30
    
    // ===== 二阶模型状态 =====
    float vx_ = 0.0f;
    float vy_ = 0.0f;
    
    // ===== 预测状态 =====
    float pred_cx_, pred_cy_;
    float pred_vx_, pred_vy_;
    int miss_frames_;
    float confidence_;

private:
    void update_rect()
    {
        rect_.left   = static_cast<int>(cx_ - crop_w_ * 0.5f);
        rect_.top    = static_cast<int>(cy_ - crop_h_ * 0.5f);
        rect_.right  = rect_.left + crop_w_;
        rect_.bottom = rect_.top  + crop_h_;

        limit_rect();
    }

    void limit_center()
    {
        cx_ = std::max(crop_w_ * 0.5f,
              std::min(cx_, img_w_ - crop_w_ * 0.5f));

        cy_ = std::max(crop_h_ * 0.5f,
              std::min(cy_, img_h_ - crop_h_ * 0.5f));
    }

    void limit_rect()
    {
        if (rect_.left < 0) {
            rect_.left = 0;
            rect_.right = crop_w_;
        }
        if (rect_.top < 0) {
            rect_.top = 0;
            rect_.bottom = crop_h_;
        }
        if (rect_.right > img_w_) {
            rect_.right = img_w_;
            rect_.left = img_w_ - crop_w_;
        }
        if (rect_.bottom > img_h_) {
            rect_.bottom = img_h_;
            rect_.top = img_h_ - crop_h_;
        }
    }

    float clamp(float val, float min_val, float max_val)
    {
        return std::max(min_val, std::min(val, max_val));
    }

    // 红色裁剪框（始终画）
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

        if (ret != 0)
        {
            printf("crop_alg_image failed, ret=%d\n", ret);
        }
    }
};



/*---------------------------------------------------*/
#include <deque>
#include <condition_variable>

class FrameQueue {
public:
    FrameQueue(size_t max_size, ImageBufferPool& pool)
        : max_size_(max_size), pool_(pool) {}

    void push(image_buffer_t& buf)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (queue_.size() >= max_size_) {
            // === 丢最旧帧 ===
            auto old = queue_.front();
            queue_.pop_front();
            pool_.release(old);
            printf("~~~~~~~~~~FrameQueue full, dropping oldest frame 丢帧!!! \n");
        }

        queue_.push_back(buf);
        cv_.notify_one();
    }

    bool pop(image_buffer_t& out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return !queue_.empty() || stop_;
        });

        if (queue_.empty())
            return false;

        out = queue_.front();
        queue_.pop_front();
        return true;
    }

    void stop()
    {
        stop_ = true;
        cv_.notify_all();
    }

private:
    size_t max_size_;
    std::deque<image_buffer_t> queue_;
    ImageBufferPool& pool_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_{false};
};
/*--------------------------------------------- */
void producer_thread(FrameQueue& fq, ImageBufferPool& pool, const char *frames_dir)
{
    // 读取目录中的所有 frame_XXXXXX.jpg 文件
    std::vector<std::string> file_list;
    DIR *dir = opendir(frames_dir);
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL)
    {
        if (endswith(entry->d_name, ".jpg") || endswith(entry->d_name, ".png"))
        {
            file_list.push_back(entry->d_name);
        }
    }
    closedir(dir);

    // 按文件名排序（frame_000001.jpg → frame_000002.jpg）
    std::sort(file_list.begin(), file_list.end());

    size_t i = 0;
    constexpr int interval_ms = 100;
    auto next_tick = std::chrono::steady_clock::now();

    while (!g_exit.load()) {
        next_tick += std::chrono::milliseconds(interval_ms);

        image_buffer_t buf = {0};
        if (!pool.acquire(buf)) {
            // pool 空，直接跳过这一帧
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        // 读取当前帧
        char img_path[256];
        sprintf(img_path, "%s/%s", frames_dir, file_list[i].c_str());
        if(i >= file_list.size()){
            break;
        }
        i++;

        //printf("Processing frame: %s\n", img_path);
        if (read_image(img_path, &buf) != 0)
        {
            printf("read_image failed: %s\n", img_path);
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        fq.push(buf);// 推送到队列
        
        std::this_thread::sleep_until(next_tick);
    }
}

enum DetectMode {
    MODE_ROI,      // 只检测裁剪区域（快）
    MODE_FULL      // 全图检测（慢，但兜底）
};

/*
    裁剪窗口类
*/



/*
第一层（检测层）：yolo检测层
第二层（筛选层）：筛选跟踪目标——解决“同一帧多个球，选哪个？”
第三层（追踪层）：把“时序噪声观测”变成“连续稳定状态”
第四层（运镜层）：把“目标点”变成“丝滑的镜头运动”
*/
/*-------------------------------------------------*/
void consumer_thread(FrameQueue& fq, ImageBufferPool& pool, const char *model_path, const char *out_dir)
{
    // 初始化 YOLO 模型
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();
    int ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret)
    {
        printf("init_yolov8_model failed!\n");
        return;
    }

    char cmd[256];
    sprintf(cmd, "mkdir -p %s", out_dir);
    system(cmd);

    int frame_count = 0;
    int frame_track_count = 0;
    image_buffer_t src_image = {0};
    image_buffer_t crop_image = {0};
    
    // 初始化跟踪器
    TrackFrame tracker;
    tracker.Init(50);
    
    // === 改进的检测模式参数 ===
    DetectMode detect_mode = MODE_ROI;
    int roi_miss_count = 0;
    const int ROI_MISS_THRESHOLD = 5;  // 提高阈值，减少模式切换
    const int FULL_FIND_THRESHOLD = 2; // 整图模式连续检测到2次才切换回ROI
    int full_find_count = 0;
    
    bool found_target = false;
    bool crop_allocated = false;
    bool has_track_result = false;  // 新增：标记是否有跟踪结果
    
    // 初始化裁切窗口
    crop_window crop_win(PIC_FULL_WIDTH, PIC_FULL_HEIGHT, ALG_CROP_WIDTH, ALG_CROP_HEIGHT);
    
    // ============ 设置舒适区大小（核心调参） ============
    // 参数含义：舒适区占裁剪窗口的比例
    // 0.20 表示中心40%区域（左右各20%）
    // 推荐范围：0.15-0.30
    // - 0.15：较小舒适区，运镜更积极
    // - 0.20：平衡（推荐）
    // - 0.25：较大舒适区，运镜更保守
    crop_win.set_comfort_zone(0.20f, 0.20f);  // 宽度20%, 高度20%

    while (true) {
        found_target = false;
        has_track_result = false;
        frame_track_count++;
        
        if(src_image.width == 0 && src_image.height == 0){
            memset(&src_image, 0, sizeof(image_buffer_t));
            if (!fq.pop(src_image)){
                usleep(5000);
                continue;
            }
        }
        
        memset(&crop_image, 0, sizeof(image_buffer_t));

        //获取动态裁切接口
        crop_win.get_crop_window(&src_image, &crop_image);
        crop_allocated = true;

        // === 推理 / 处理 ===
        object_detect_result_list od_results; 
        if(crop_image.virt_addr == NULL){
            printf("------crop_image is NULL\n");
            pool.release(src_image);
            memset(&src_image, 0, sizeof(image_buffer_t));
            continue;
        }
        
        //推理前判断需要选择什么图去检测
        image_buffer_t *detect_image = nullptr;
        if (detect_mode == MODE_ROI)
        {
            detect_image = &crop_image;
        }
        else
        {
            detect_image = &src_image;
        }
        inference_yolov8_model(&rknn_app_ctx, detect_image, &od_results);

        // ============ 核心改进：统一的检测+跟踪逻辑 ============
        printf("检测模式: %s, 置信度: %.2f\n", 
               detect_mode == MODE_ROI ? "ROI" : "FULL",
               crop_win.get_confidence());

        // 1. 准备检测结果供跟踪使用
        std::vector<T_DetectObject> detections;
        for (int k = 0; k < od_results.count; k++)
        {
            auto &det = od_results.results[k];
            char text[256];
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det.cls_id), det.prop * 100);
            
            if (strncmp(text, "ball", 4) == 0)
            {
                T_DetectObject obj;
                obj.cls_id = det.cls_id;
                obj.score = det.prop;
                obj.xmin = det.box.left;
                obj.ymin = det.box.top;
                obj.xmax = det.box.right;
                obj.ymax = det.box.bottom;
                detections.push_back(obj);
                
                found_target = true;
                
                printf("*检测到球: @ (%d %d %d %d) %.3f\n",
                       det.box.left, det.box.top,
                       det.box.right, det.box.bottom,
                       det.prop);
                
                draw_rectangle(&crop_image, det.box.left, det.box.top,
                             det.box.right - det.box.left,
                             det.box.bottom - det.box.top,
                             COLOR_BLUE, 3);
                draw_text(&crop_image, text, det.box.left, det.box.top - 20, COLOR_RED, 10);
            }
        }

        // 2. 调用跟踪（不管有没有检测结果都调用）
        std::vector<T_TrackObject> track_results;
        tracker.ProcessFrame(frame_track_count, crop_image, detections, track_results);

        // 3. 根据检测和跟踪结果更新裁剪窗口
        if (detect_mode == MODE_ROI)
        {
            if (found_target && track_results.size() > 0)
            {
                // 场景A：检测+跟踪都成功 → 用跟踪结果，高置信度
                auto &t = track_results[0];
                
                draw_rectangle(&crop_image, t.xmin, t.ymin,
                             t.xmax - t.xmin, t.ymax - t.ymin,
                             COLOR_GREEN, 3);
                
                // ============ 可选：绘制舒适区（调试用）============
                int cz_left, cz_top, cz_width, cz_height;
                crop_win.get_comfort_zone_rect(cz_left, cz_top, cz_width, cz_height);
                draw_rectangle(&crop_image, cz_left, cz_top, cz_width, cz_height,
                             COLOR_YELLOW, 2);  // 黄色虚线表示舒适区
                
                // 转换到原图坐标
                int oxmin = crop_win.get_rect().left + t.xmin;
                int oymin = crop_win.get_rect().top + t.ymin;
                int oxmax = crop_win.get_rect().left + t.xmax;
                int oymax = crop_win.get_rect().top + t.ymax;
                
                crop_win.update_by_target(oxmin, oymin, oxmax, oymax, true);
                roi_miss_count = 0;
                has_track_result = true;
            }
            else if (!found_target && track_results.size() > 0)
            {
                // 场景B：检测失败但跟踪成功 → 用跟踪结果，中置信度
                auto &t = track_results[0];
                
                draw_rectangle(&crop_image, t.xmin, t.ymin,
                             t.xmax - t.xmin, t.ymax - t.ymin,
                             COLOR_YELLOW, 3);  // 黄色表示仅跟踪
                
                int oxmin = crop_win.get_rect().left + t.xmin;
                int oymin = crop_win.get_rect().top + t.ymin;
                int oxmax = crop_win.get_rect().left + t.xmax;
                int oymax = crop_win.get_rect().top + t.ymax;
                
                crop_win.update_by_target(oxmin, oymin, oxmax, oymax, false);
                roi_miss_count++;  // 仍然计数，但允许跟踪维持运镜
                has_track_result = true;
                
                printf("[INFO] 仅跟踪模式，丢失计数: %d/%d\n", 
                       roi_miss_count, ROI_MISS_THRESHOLD);
            }
            else
            {
                // 场景C：检测和跟踪都失败 → 使用惯性预测
                printf("[WARN] 检测和跟踪都丢失，使用惯性预测\n");
                crop_win.update_with_prediction();
                roi_miss_count++;
            }
            
            // 连续丢失过多才切换到全图
            if (roi_miss_count >= ROI_MISS_THRESHOLD)
            {
                printf("[INFO] ROI连续丢失%d帧，切换到FULL模式\n", roi_miss_count);
                detect_mode = MODE_FULL;
                roi_miss_count = 0;
                full_find_count = 0;
                // 不重置裁剪窗口，保持当前位置继续搜索
            }
        }
        else // MODE_FULL
        {
            if (found_target)
            {
                // 整图模式检测到目标
                for (int j = 0; j < od_results.count; j++)
                {
                    object_detect_result *det = &(od_results.results[j]);
                    char text[256];
                    sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
                    
                    if (strncmp(text, "ball", 4) == 0)
                    {
                        draw_rectangle(&crop_image, det->box.left, det->box.top,
                                     det->box.right - det->box.left,
                                     det->box.bottom - det->box.top,
                                     COLOR_BLUE, 3);
                        draw_text(&crop_image, text, det->box.left, det->box.top - 20, COLOR_RED, 10);
                        
                        // 用检测结果更新裁剪窗口
                        crop_win.update_by_target(
                            det->box.left,
                            det->box.top,
                            det->box.right,
                            det->box.bottom,
                            true);  // 高置信度
                        
                        full_find_count++;
                        break;
                    }
                }
                
                // 连续检测到才切回ROI，避免抖动
                if (full_find_count >= FULL_FIND_THRESHOLD)
                {
                    printf("[INFO] FULL模式连续检测到%d次，切换回ROI\n", full_find_count);
                    detect_mode = MODE_ROI;
                    roi_miss_count = 0;
                    full_find_count = 0;
                }
            }
            else
            {
                // 整图模式也没找到，继续用惯性
                crop_win.update_with_prediction();
                full_find_count = 0;
            }
        }

        // 保存结果输出
        char out_path[256];
        sprintf(out_path, "%s/%05d.jpg", out_dir, frame_count);
        write_image(out_path, &crop_image);
        frame_count++;
        
        // 释放裁剪图内存
        if (crop_allocated && crop_image.virt_addr)
        {
            free(crop_image.virt_addr);
            crop_image.virt_addr = NULL;
        }
        
        // 回收原图
        pool.release(src_image);
        memset(&src_image, 0, sizeof(image_buffer_t));
    }

    // 清理
    deinit_post_process();
    release_yolov8_model(&rknn_app_ctx);
}
/*-----------------------------------------------*/
int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <model_path> <frames_dir> <out_dir>\n", argv[0]);
        printf("例如： ffmpeg -i test.mp4 frames/frame_%%06d.jpg\n");
        return -1;
    }

    const char *model_path = argv[1];
    const char *frames_dir = argv[2];
    const char *out_dir = argv[3];

    // ===== 参数配置（可按需改成配置文件）=====
    constexpr size_t QUEUE_SIZE = 12;
    constexpr size_t POOL_SIZE  = 12;      // queue + pipeline + margin
    constexpr int    IMAGE_SIZE = 2560*1440*3; //grb888

    // ===== 信号注册 =====
    signal(SIGINT,  signal_handler);

    // ===== 初始化 Buffer Pool =====
    ImageBufferPool buffer_pool(POOL_SIZE, IMAGE_SIZE);

    // ===== 初始化帧队列（丢旧帧）=====
    FrameQueue frame_queue(QUEUE_SIZE, buffer_pool);

    // ===== 启动 Producer / Consumer =====
    std::thread producer([&] {
        producer_thread(frame_queue, buffer_pool, frames_dir);
    });

    std::thread consumer([&] {
        consumer_thread(frame_queue, buffer_pool, model_path, out_dir);
    });

    std::cout << "[Main] pipeline started\n";

    // ===== 主线程心跳 / 监控 =====
    int count = 0;
    while (!g_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (count > 20)
        {
            // 重置计数器
            count = 0;
            printf("*** 主线程心跳 / 监控 20 次\n");
        }
        count++;
    }

    std::cout << "[Main] stopping...\n";

    // ===== 通知队列退出 =====
    frame_queue.stop();

    // ===== 等待线程结束 =====
    if (producer.joinable())
        producer.join();

    if (consumer.joinable())
        consumer.join();

    std::cout << "[Main] exit clean\n";
    return 0;
}