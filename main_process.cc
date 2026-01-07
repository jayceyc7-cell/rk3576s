
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
        update_rect();
    }

    // 仅在“检测 / 跟踪到目标”时调用
    // void update_by_target(int xmin, int ymin, int xmax, int ymax)
    // {
    //     target_cx_ = 0.5f * (xmin + xmax);
    //     target_cy_ = 0.5f * (ymin + ymax);

    //     // // ===== 死区判断（小抖动不更新）=====
    //     // if (fabs(target_cx_ - cx_) < dead_zone_px_ &&
    //     //     fabs(target_cy_ - cy_) < dead_zone_px_)
    //     // {
    //     //     //return;
    //     // }

    //     // // ===== EMA 平滑 =====
    //     // cx_ = alpha_ * target_cx_ + (1.0f - alpha_) * cx_;
    //     // cy_ = alpha_ * target_cy_ + (1.0f - alpha_) * cy_;

    //     float dx = fabs(target_cx_ - cx_);
    //     float dy = fabs(target_cy_ - cy_);

    //     float scale = std::max(dx, dy) / dead_zone_px_;
    //     scale = clamp(scale, 0.0f, 1.0f);

    //     float adaptive_alpha = alpha_ * scale;

    //     cx_ = adaptive_alpha * target_cx_ + (1 - adaptive_alpha) * cx_;
    //     cy_ = adaptive_alpha * target_cy_ + (1 - adaptive_alpha) * cy_;

    //     limit_center();
    //     update_rect();
    // }

    void update_by_target(int xmin, int ymin, int xmax, int ymax)
    {
        // 1. 目标中心
        target_cx_ = 0.5f * (xmin + xmax);
        target_cy_ = 0.5f * (ymin + ymax);

        // 2. 位置误差
        float ex = target_cx_ - cx_;
        float ey = target_cy_ - cy_;

        // 3. 死区（抑制微抖）
        if (fabs(ex) < dead_zone_px_)
            ex = 0.0f;
        if (fabs(ey) < dead_zone_px_)
            ey = 0.0f;

        // ================= 二阶系统参数 =================
        const float k_p = 0.08f;   // 位置增益（越大越跟手，0.05~0.15）
        const float k_d = 0.85f;   // 阻尼（0.7~0.95，越大越稳）
        const float max_v = 80.0f; // 最大速度（像素/帧，防止拉扯）

        // 4. 加速度（比例项）
        float ax = k_p * ex;
        float ay = k_p * ey;

        // 5. 更新速度（带阻尼）
        vx_ = k_d * vx_ + ax;
        vy_ = k_d * vy_ + ay;

        // 6. 限速（防止大跳）
        vx_ = clamp(vx_, -max_v, max_v);
        vy_ = clamp(vy_, -max_v, max_v);

        // 7. 积分得到位置
        cx_ += vx_;
        cy_ += vy_;

        // 8. 边界限制
        limit_center();
        update_rect();
    }

private:
    // ================= 图像 / 裁剪参数 =================
    int img_w_, img_h_;
    int crop_w_, crop_h_;

    // ================= 裁剪中心（连续） =================
    float cx_, cy_;
    float target_cx_, target_cy_;

    image_rect_t rect_;

    // ================= 调参区（非常重要） =================
    //const float alpha_ = 0.2f;       // 平滑系数（0.15~0.3 推荐）
    const float dead_zone_px_ = 500.0f; // 死区像素（5~15）
    // ===== 二阶模型状态 =====
    float vx_ = 0.0f;
    float vy_ = 0.0f;

private:
    // 根据中心更新裁剪框
    void update_rect()
    {
        rect_.left   = static_cast<int>(cx_ - crop_w_ * 0.5f);
        rect_.top    = static_cast<int>(cy_ - crop_h_ * 0.5f);
        rect_.right  = rect_.left + crop_w_;
        rect_.bottom = rect_.top  + crop_h_;

        limit_rect();
    }

    // 中心点限幅（保证裁剪框不出界）
    void limit_center()
    {
        cx_ = std::max(crop_w_ * 0.5f,
              std::min(cx_, img_w_ - crop_w_ * 0.5f));

        cy_ = std::max(crop_h_ * 0.5f,
              std::min(cy_, img_h_ - crop_h_ * 0.5f));
    }

    // 裁剪框边界修正（双保险）
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

    // 实际裁剪（你工程里换回 image_crop 即可）
    // void crop_image(image_buffer_t* src, image_buffer_t* dst)
    // {
    //     // 正式版本：
    //     // image_crop(src, dst, rect_);

    //     // 临时占位（调试）
    //     *dst = *src;
    // }
    void crop_image(image_buffer_t *src, image_buffer_t *dst)
    {
        if (!src || !dst)
            return;

        // 1. 使用跟踪得到的裁剪框
        image_rect_t box = rect_;

        // 2. 实际裁剪后在 src 中使用的区域（调试用）
        image_rect_t real_crop_rect;

        // 3. 调用你已经实现好的裁剪算法
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

    // 创建输出目录（假如不存在）
    char cmd[256];
    sprintf(cmd, "mkdir -p %s", out_dir);
    system(cmd);

    struct timeval start, end;
    double time_use = 0;
    double total_time_use = 0;
    int frame_count = 0;
    int frame_track_count = 0;
    image_rect_t obj_rect = {0};
    image_rect_t last_obj_rect = {0};
    bool first_frame = true;
    image_rect_t crop_rect = {0};
    image_buffer_t src_image = {0};
    image_buffer_t save_image = {0};
    image_buffer_t crop_image = {0};
    // 初始化跟踪器
    TrackFrame tracker;
    tracker.Init(50);
    // === 裁剪检测模式参数 ===
    DetectMode detect_mode = MODE_ROI;
    int roi_miss_count = 0;
    const int ROI_MISS_THRESHOLD = 2; // 连续帧丢失 → 全图
    bool found_target = false;
    bool crop_allocated = false;
    // 初始化裁切窗口
    crop_window crop_win(PIC_FULL_WIDTH, PIC_FULL_HEIGHT, ALG_CROP_WIDTH, ALG_CROP_HEIGHT);

    while (true) {
        found_target = false;
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
            // === 回收 ===
            pool.release(src_image);
            memset(&src_image, 0, sizeof(image_buffer_t));
            continue;
        }
        //推理前判断需要选择什么图去检测
        image_buffer_t *detect_image = nullptr;
        if (detect_mode == MODE_ROI)
        {
            detect_image = &crop_image; // ROI：小图检测
        }
        else
        {
            detect_image = &src_image; // FULL：整图检测
        }
        inference_yolov8_model(&rknn_app_ctx, detect_image, &od_results);

        // === 检测模式 ===
        printf("检测模式\n");
        if (detect_mode == MODE_ROI)
        {
            printf("ROI小图模式\n");

            char text[256];
            int j = 0;
            for (j = 0; j < od_results.count; j++)
            {
                object_detect_result *det = &(od_results.results[j]);

                int x1 = det->box.left;
                int y1 = det->box.top;
                int x2 = det->box.right;
                int y2 = det->box.bottom;

                sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
                // 过滤球目标
                if (strncmp(text, "ball", 4) == 0)
                {
                    found_target = true;
                    printf("*cls_id:%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det->cls_id),
                           det->box.left, det->box.top,
                           det->box.right, det->box.bottom,
                           det->prop);

                    draw_rectangle(&crop_image, x1, y1, (x2 - x1), (y2 - y1), COLOR_BLUE, 3);
                    draw_text(&crop_image, text, x1, y1 - 20, COLOR_RED, 10);

                    /******************球的跟踪预测入口*********************/

                    // 将 YOLO的输出结果作为跟踪的输入放入结构体DetectObject中
                    std::vector<T_DetectObject> detections;
                    for (int k = 0; k < od_results.count; k++)
                    {
                        auto &det = od_results.results[k];

                        T_DetectObject obj;
                        obj.cls_id = det.cls_id;
                        obj.score = det.prop;
                        obj.xmin = det.box.left;
                        obj.ymin = det.box.top;
                        obj.xmax = det.box.right;
                        obj.ymax = det.box.bottom;

                        detections.push_back(obj);
                    }
                    // 调用跟踪算法
                    std::vector<T_TrackObject> track_results;
                    tracker.ProcessFrame(frame_track_count, crop_image, detections, track_results);

                    /******************球的跟踪预测出口*********************/
                    printf("track_results.size() = %d\n", track_results.size());
                    if (track_results.size() > 0)
                    {
                        printf("draw_rectangle @ (%d %d %d %d)\n", track_results[0].xmin, track_results[0].ymin, track_results[0].xmax - track_results[0].xmin, track_results[0].ymax - track_results[0].ymin);
                        draw_rectangle(&crop_image, track_results[0].xmin, track_results[0].ymin,
                                       track_results[0].xmax - track_results[0].xmin, track_results[0].ymax - track_results[0].ymin, COLOR_GREEN, 3);

                        /*跟踪预测结果不能直接作为画面裁剪的输入，需要先经过一个过滤器来判断是否跟新裁剪窗口*/
                        // 更新裁切窗口
                        auto &t = track_results[0];

                        // 裁剪图 → 原图坐标
                        int oxmin = crop_win.get_rect().left + t.xmin;
                        int oymin = crop_win.get_rect().top + t.ymin;
                        int oxmax = crop_win.get_rect().left + t.xmax;
                        int oymax = crop_win.get_rect().top + t.ymax;

                        // 用“原图坐标”更新裁剪窗口
                        crop_win.update_by_target(oxmin, oymin, oxmax, oymax);
                    }

                    // 这个break是测试下用的, 只检测第一个目标,多目标情况下还需要其他处理
                    break;
                }
            }
            if (found_target)
            {
                roi_miss_count = 0;
            }
            else
            {
                roi_miss_count++;
                if (roi_miss_count >= ROI_MISS_THRESHOLD)
                {
                    printf("[INFO] ROI lost, switch to FULL detect\n");
                    detect_mode = MODE_FULL;
                    roi_miss_count = 0;

                    // ★ 重置裁剪窗口到整图中心（非常关键）
                    //crop_win.reset_to_center();
                }
            }
        }
        else // MODE_FULL
        {
            printf("FULL整图模式\n");

            char text[256];
            int j = 0;
            for (j = 0; j < od_results.count; j++)
            {
                object_detect_result *det = &(od_results.results[j]);

                int x1 = det->box.left;
                int y1 = det->box.top;
                int x2 = det->box.right;
                int y2 = det->box.bottom;

                sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
                // 过滤球目标
                if (strncmp(text, "ball", 4) == 0)
                {
                    found_target = true;
                    printf("*cls_id:%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det->cls_id),
                           det->box.left, det->box.top,
                           det->box.right, det->box.bottom,
                           det->prop);

                    draw_rectangle(&crop_image, x1, y1, (x2 - x1), (y2 - y1), COLOR_BLUE, 3);
                    draw_text(&crop_image, text, x1, y1 - 20, COLOR_RED, 10);

                    // === 用检测结果直接初始化裁剪窗口 ===
                    crop_win.update_by_target(
                        det->box.left,
                        det->box.top,
                        det->box.right,
                        det->box.bottom);

                }
            }
                    
                    

            if (found_target)
            {
                printf("[INFO] Target re-found, switch to ROI detect\n");
                detect_mode = MODE_ROI;
                roi_miss_count = 0;
            }
        }

        //保存结果输出
        printf("*保存结果输出\n");
        char out_path[256];
        sprintf(out_path, "%s/%s.%s", out_dir, std::to_string(frame_count).c_str(), "jpg");
        write_image(out_path, &crop_image);
        frame_count++;
        // ===== 释放裁剪图内存=====
        if (crop_allocated && crop_image.virt_addr)
        {
            free(crop_image.virt_addr);
            crop_image.virt_addr = NULL;
        }
        // === 回收 ===
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