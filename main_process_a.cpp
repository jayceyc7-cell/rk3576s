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
        // 注意：这里只释放队列中的 buffer
        // 如果有 buffer 被借出未归还，会泄漏
        while (!free_queue_.empty()) {
            auto& buf = free_queue_.front();
            free_queue_.pop();
            if (buf.virt_addr) {
                free(buf.virt_addr);
            }
        }
    }

    // 禁用拷贝
    ImageBufferPool(const ImageBufferPool&) = delete;
    ImageBufferPool& operator=(const ImageBufferPool&) = delete;

    // 阻塞式获取：无可用 buffer 时等待
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

    // 非阻塞式获取（可选）
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
        cv_.notify_one();  // 通知等待的生产者
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

    // 仅在"检测 / 跟踪到目标"时调用
    void update_by_target(int xmin, int ymin, int xmax, int ymax)
    {
        target_cx_ = 0.5f * (xmin + xmax);
        target_cy_ = 0.5f * (ymin + ymax);

        // ===== 死区判断（小抖动不更新）=====
        if (std::fabs(target_cx_ - cx_) < dead_zone_px_ &&
            std::fabs(target_cy_ - cy_) < dead_zone_px_)
        {
            return;
        }

        // ===== EMA 平滑 =====
        cx_ = alpha_ * target_cx_ + (1.0f - alpha_) * cx_;
        cy_ = alpha_ * target_cy_ + (1.0f - alpha_) * cy_;

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
    const float alpha_ = 0.05f;         // 平滑系数（0.15~0.3 推荐）
    const float dead_zone_px_ = 200.0f; // 死区像素（5~15）

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

    // 实际裁剪
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
// 阻塞式帧队列：队列满时生产者等待，不丢帧
class FrameQueue {
public:
    FrameQueue(size_t max_size)
        : max_size_(max_size) {}

    // 阻塞式 push：队列满时等待
    void push(image_buffer_t& buf)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 等待队列有空位或收到停止信号
        cv_not_full_.wait(lock, [&] {
            return queue_.size() < max_size_ || stop_;
        });

        if (stop_) {
            return;  // 收到停止信号，直接返回
        }

        queue_.push_back(buf);
        cv_not_empty_.notify_one();  // 通知消费者
    }

    // 阻塞式 pop：队列空时等待
    bool pop(image_buffer_t& out)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        
        cv_not_empty_.wait(lock, [&] {
            return !queue_.empty() || stop_;
        });

        if (queue_.empty()) {
            return false;  // 队列空且收到停止信号
        }

        out = queue_.front();
        queue_.pop_front();
        cv_not_full_.notify_one();  // 通知生产者
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
    std::condition_variable cv_not_empty_;  // 队列非空条件
    std::condition_variable cv_not_full_;   // 队列未满条件
    bool stop_{false};
};

/*--------------------------------------------- */
void producer_thread(FrameQueue& fq, ImageBufferPool& pool, const char *frames_dir)
{
    // 读取目录中的所有 frame_XXXXXX.jpg 文件
    std::vector<std::string> file_list;
    
    DIR *dir = opendir(frames_dir);
    if (dir == nullptr) {
        printf("[Producer] Failed to open directory: %s\n", frames_dir);
        return;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (endswith(entry->d_name, ".jpg") || endswith(entry->d_name, ".png"))
        {
            file_list.push_back(entry->d_name);
        }
    }
    closedir(dir);

    if (file_list.empty()) {
        printf("[Producer] No image files found in directory: %s\n", frames_dir);
        return;
    }

    // 按文件名排序（frame_000001.jpg → frame_000002.jpg）
    std::sort(file_list.begin(), file_list.end());

    printf("[Producer] Found %zu frames to process\n", file_list.size());

    size_t i = 0;
    size_t total_frames = file_list.size();

    while (!g_exit.load() && i < total_frames) {
        
        // 阻塞式获取 buffer（无可用时等待）
        image_buffer_t buf = {0};
        if (!pool.acquire(buf)) {
            // 收到停止信号
            printf("[Producer] Pool stopped, exiting\n");
            break;
        }

        // 读取当前帧
        char img_path[256];
        snprintf(img_path, sizeof(img_path), "%s/%s", frames_dir, file_list[i].c_str());

        if (read_image(img_path, &buf) != 0)
        {
            printf("[Producer] read_image failed: %s\n", img_path);
            pool.release(buf);  // 读取失败时归还 buffer
            i++;
            continue;
        }

        // 阻塞式推送到队列（队列满时等待）
        fq.push(buf);
        
        i++;
        
        // 每处理 100 帧打印一次进度
        if (i % 100 == 0 || i == total_frames) {
            printf("[Producer] Progress: %zu / %zu frames read (%.1f%%)\n", 
                   i, total_frames, 100.0 * i / total_frames);
        }
    }

    printf("[Producer] Finished reading all %zu frames\n", i);
}

/*
第一层（检测层）：yolo检测层
第二层（筛选层）：筛选跟踪目标——解决"同一帧多个球，选哪个？"
第三层（追踪层）：把"时序噪声观测"变成"连续稳定状态"
第四层（运镜层）：把"目标点"变成"丝滑的镜头运动"
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
        printf("[Consumer] init_yolov8_model failed!\n");
        return;
    }

    // 创建输出目录（假如不存在）
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
    system(cmd);

    int frame_count = 0;
    int frame_track_count = 0;
    image_buffer_t src_image = {0};
    image_buffer_t crop_image_buf = {0};
    
    // 初始化跟踪器
    TrackFrame tracker;
    tracker.Init(50);
    // 初始化裁切窗口
    crop_window crop_win(PIC_FULL_WIDTH, PIC_FULL_HEIGHT, ALG_CROP_WIDTH, ALG_CROP_HEIGHT);

    while (true) {
        frame_track_count++;
        
        // 获取新帧
        if (src_image.width == 0 && src_image.height == 0) {
            memset(&src_image, 0, sizeof(image_buffer_t));
            if (!fq.pop(src_image)) {
                // 队列空且收到停止信号，退出
                printf("[Consumer] Queue stopped, exiting\n");
                break;
            }
        }
        
        memset(&crop_image_buf, 0, sizeof(image_buffer_t));

        // 获取动态裁切接口
        crop_win.get_crop_window(&src_image, &crop_image_buf);

        // === 推理 / 处理 ===
        object_detect_result_list od_results; 
        if (crop_image_buf.virt_addr == NULL) {
            printf("[Consumer] crop_image is NULL\n");
            // === 回收 ===
            pool.release(src_image);
            memset(&src_image, 0, sizeof(image_buffer_t));
            continue;
        }

        inference_yolov8_model(&rknn_app_ctx, &crop_image_buf, &od_results);

        char text[256];
        for (int j = 0; j < od_results.count; j++) {
            object_detect_result *det = &(od_results.results[j]);

            int x1 = det->box.left;
            int y1 = det->box.top;
            int x2 = det->box.right;
            int y2 = det->box.bottom;

            snprintf(text, sizeof(text), "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
            // 过滤球目标
            if (strncmp(text, "ball", 4) == 0)
            {
                printf("[Consumer] cls_id:%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det->cls_id),
                det->box.left, det->box.top,
                det->box.right, det->box.bottom,
                det->prop);

                draw_rectangle(&crop_image_buf, x1, y1, (x2 - x1), (y2 - y1), COLOR_BLUE, 3);
                draw_text(&crop_image_buf, text, x1, y1 - 20, COLOR_RED, 10);

                /******************球的跟踪预测入口*********************/

                // 将 YOLO的输出结果作为跟踪的输入放入结构体DetectObject中
                std::vector<T_DetectObject> detections;
                for (int k = 0; k < od_results.count; k++)
                {
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
                // 调用跟踪算法
                std::vector<T_TrackObject> track_results;
                tracker.ProcessFrame(frame_track_count, crop_image_buf, detections, track_results);

                /******************球的跟踪预测出口*********************/
                printf("[Consumer] track_results.size() = %zu\n", track_results.size());
                if (!track_results.empty())
                {
                    printf("[Consumer] draw_rectangle @ (%d %d %d %d)\n", 
                           track_results[0].xmin, track_results[0].ymin, 
                           track_results[0].xmax - track_results[0].xmin, 
                           track_results[0].ymax - track_results[0].ymin);
                    draw_rectangle(&crop_image_buf, track_results[0].xmin, track_results[0].ymin,
                                   track_results[0].xmax - track_results[0].xmin, 
                                   track_results[0].ymax - track_results[0].ymin, COLOR_GREEN, 3);

                    /*跟踪预测结果不能直接作为画面裁剪的输入，需要先经过一个过滤器来判断是否跟新裁剪窗口*/
                    // 更新裁切窗口
                    auto &t = track_results[0];

                    // 裁剪图 → 原图坐标
                    int oxmin = crop_win.get_rect().left + t.xmin;
                    int oymin = crop_win.get_rect().top + t.ymin;
                    int oxmax = crop_win.get_rect().left + t.xmax;
                    int oymax = crop_win.get_rect().top + t.ymax;

                    // 用"原图坐标"更新裁剪窗口
                    crop_win.update_by_target(oxmin, oymin, oxmax, oymax);
                }

                // 这个break是测试下用的
                break;
            } 
        }

        // 保存结果输出
        char out_path[256];
        snprintf(out_path, sizeof(out_path), "%s/%06d.jpg", out_dir, frame_count);
        write_image(out_path, &crop_image_buf);
        frame_count++;
        
        // 每处理 100 帧打印一次进度
        if (frame_count % 100 == 0) {
            printf("[Consumer] Processed %d frames\n", frame_count);
        }
        
        // ===== 释放裁剪图内存 =====
        if (crop_image_buf.virt_addr)
        {
            free(crop_image_buf.virt_addr);
            crop_image_buf.virt_addr = NULL;
        }
        // === 回收 buffer 到 pool ===
        pool.release(src_image);
        memset(&src_image, 0, sizeof(image_buffer_t));
    }

    printf("[Consumer] Finished processing %d frames\n", frame_count);

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
    constexpr size_t POOL_SIZE  = 16;      // 略大于队列，留一些余量
    constexpr size_t IMAGE_SIZE = 2560 * 1440 * 3; // rgb888

    // ===== 信号注册 =====
    signal(SIGINT,  signal_handler);

    // ===== 初始化 Buffer Pool =====
    ImageBufferPool buffer_pool(POOL_SIZE, IMAGE_SIZE);

    // ===== 初始化帧队列（阻塞式，不丢帧）=====
    FrameQueue frame_queue(QUEUE_SIZE);

    // ===== 用于标记生产者是否完成 =====
    std::atomic<bool> producer_done{false};

    // ===== 启动 Producer =====
    std::thread producer([&] {
        producer_thread(frame_queue, buffer_pool, frames_dir);
        producer_done.store(true);
        printf("[Producer] Thread exiting, notifying consumer...\n");
        // 生产者结束后，通知队列停止（让消费者能退出）
        frame_queue.stop();
    });

    // ===== 启动 Consumer =====
    std::thread consumer([&] {
        consumer_thread(frame_queue, buffer_pool, model_path, out_dir);
    });

    std::cout << "[Main] Pipeline started (no-drop mode)\n";
    std::cout << "[Main] Queue size: " << QUEUE_SIZE << ", Pool size: " << POOL_SIZE << "\n";

    // ===== 主线程心跳 / 监控 =====
    int heartbeat_count = 0;
    while (!g_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        heartbeat_count++;
        
        if (heartbeat_count >= 5) {
            heartbeat_count = 0;
            printf("[Main] Heartbeat - Queue: %zu, Pool available: %zu\n", 
                   frame_queue.size(), buffer_pool.available());
        }

        // 如果生产者完成且队列为空，可以提前退出等待
        if (producer_done.load() && frame_queue.empty()) {
            printf("[Main] Producer done and queue empty, waiting for consumer...\n");
            break;
        }
    }

    // ===== 如果是 Ctrl+C 触发的退出 =====
    if (g_exit.load()) {
        std::cout << "[Main] Received SIGINT, stopping...\n";
        frame_queue.stop();
        buffer_pool.stop();
    }

    // ===== 等待线程结束 =====
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