
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

class crop_window
{

public:
    // 初始化裁切窗口信息
    crop_window(int x, int y, int w, int h)
        : default_x_(x), default_y_(y), default_w_(w), default_h_(h) {}

    // 裁切窗口
    void get_crop_window(image_buffer_t* src_image, image_buffer_t* crop_image)
    {
        printf("get_crop_window\n");
        if (is_cropping)
        {
            // 裁剪
            printf("*裁剪\n");
            crop_alg_image(src_image, crop_image, obj_rect, &crop_rect, ALG_CROP_WIDTH, ALG_CROP_HEIGHT); 
        }
        else
        {
            // 不裁剪，使用全图
            printf("*不裁剪，使用全图\n");
            *crop_image = *src_image;
        }
        // 重置裁切标志
        is_cropping = false;
    }
    // 更新裁切窗口信息
    void update_crop_window(image_buffer_t* crop_image, float xmin, float ymin, float xmax, float ymax)
    {
        printf("update_crop_window\n");
        //测试阶段都置false
        is_cropping = false;
        // 更新裁切窗口
        obj_rect.left = xmin;
        obj_rect.top = ymin;
        obj_rect.right = xmax - xmin;
        obj_rect.bottom = ymax - ymin;



        //画红色裁剪框
        draw_rectangle(crop_image, obj_rect.left, obj_rect.top, ALG_CROP_WIDTH, ALG_CROP_HEIGHT, COLOR_RED, 3);


        /// 检测不到目标处理
        // if (!first_frame && (od_results.count == 0 || j == od_results.count))
        // {
        //     need_crop = false;
        //     if (crop_image.width != PIC_FULL_WIDTH || crop_image.height != PIC_FULL_HEIGHT)
        //     {
        //         // 逐步放大裁剪区域
        //         printf("***扩大裁剪区域: %d x %d\n", PIC_FULL_WIDTH, PIC_FULL_HEIGHT);
        //         if (crop_image.virt_addr != NULL)
        //         {
        //             free(crop_image.virt_addr);
        //             memset(&crop_image, 0, sizeof(image_buffer_t));
        //         }
        //         continue;
        //     }
        //     else
        //     {
        //         printf("***全图也检测不到\n");
        //         // 保存上次写入的输出帧（后续可以连续前三帧检测不到使用之前的数据，否则从全景图裁剪）
        //         crop_width = ALG_CROP_WIDTH;
        //         crop_height = ALG_CROP_HEIGHT;
        //         crop_alg_image(&src_image, &save_image, last_obj_rect, &crop_rect, crop_width, crop_height);
        //         char out_path[256];
        //         sprintf(out_path, "%s/%s.%s", out_dir, std::to_string(frame_count).c_str(), "jpg");
        //         write_image(out_path, &save_image);
        //         frame_count++;
        //         if (save_image.virt_addr != NULL)
        //         {
        //             free(save_image.virt_addr);
        //             memset(&save_image, 0, sizeof(image_buffer_t));
        //         }

        //         // === 回收 ===
        //         pool.release(src_image);
        //         memset(&src_image, 0, sizeof(image_buffer_t));
        //         continue;
        //     }
        // }
        /// 检测目标处理
        // if (first_frame || need_crop)
        // {
        //     printf("*save to file...\n");
        //     // 保存输出帧

        //     if (need_crop)
        //     {
        //         if (crop_image.width == PIC_FULL_WIDTH && crop_image.height == PIC_FULL_HEIGHT)
        //         {
        //             crop_width = ALG_CROP_WIDTH;
        //             crop_height = ALG_CROP_HEIGHT;
        //             crop_alg_image(&crop_image, &save_image, obj_rect, &crop_rect, crop_width, crop_height); // 差不多要10ms;
        //             crop_image = save_image;
        //         }
        //         else
        //         {
        //             save_image = crop_image;
        //         }
        //         last_obj_rect = obj_rect;
        //     }

        //     char out_path[256];
        //     sprintf(out_path, "%s/%s.%s", out_dir, std::to_string(frame_count).c_str(), "jpg");
        //     write_image(out_path, &crop_image);
        //     frame_count++;

        //     if (save_image.virt_addr != NULL)
        //     {
        //         free(save_image.virt_addr);
        //         memset(&save_image, 0, sizeof(image_buffer_t));
        //     }

        //     // === 回收 ===
        //     pool.release(src_image);
        //     memset(&src_image, 0, sizeof(image_buffer_t));

        //     if (first_frame && need_crop)
        //     {
        //         first_frame = false;
        //     }
        // }
    }
    // 重置裁切窗口信息
    void reset_crop_window()
    {
        // x_ = default_x_;
        // y_ = default_y_;
        // w_ = default_w_;
        // h_ = default_h_;
    }
private:
    bool is_cropping = false;

    int default_x_;
    int default_y_;
    int default_w_;
    int default_h_;

    image_buffer_t crop_image = {0};   //存放裁剪后的图像信息
    image_buffer_t src_image = {0};    //存放图像信息
    image_rect_t obj_rect = {0};       //存放检测到的目标框
    image_rect_t crop_rect = {0};      //存放裁剪框

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

    int crop_width = ALG_CROP_WIDTH;
    int crop_height = ALG_CROP_HEIGHT;

    TrackFrame tracker;
    tracker.Init(50);

    crop_window crop_win(0, 0, PIC_FULL_WIDTH, PIC_FULL_HEIGHT);

    while (true) {
        frame_track_count++;
        if(src_image.width == 0 && src_image.height == 0){
            memset(&src_image, 0, sizeof(image_buffer_t));
            if (!fq.pop(src_image)){
                usleep(5000);
                continue;
            }
        }
        
        memset(&crop_image, 0, sizeof(image_buffer_t));

        //动态裁切接口
        crop_win.get_crop_window(&src_image, &crop_image);

        // === 推理 / 处理 ===
        object_detect_result_list od_results; 
        if(crop_image.virt_addr == NULL){
            printf("------crop_image is NULL\n");
            // === 回收 ===
            pool.release(src_image);
            memset(&src_image, 0, sizeof(image_buffer_t));
            continue;
        }

        inference_yolov8_model(&rknn_app_ctx, &crop_image, &od_results);

        char text[256];  
        int j = 0;     
        for (j = 0; j < od_results.count; j++){
            object_detect_result *det = &(od_results.results[j]);

            int x1 = det->box.left;
            int y1 = det->box.top;
            int x2 = det->box.right;
            int y2 = det->box.bottom;

            sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
            //过滤球目标
            if (strncmp(text, "ball", 4) == 0)
            {
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
                    crop_win.update_crop_window(&crop_image, track_results[0].xmin, track_results[0].ymin, track_results[0].xmax, track_results[0].ymax);
                }

                // 这个break是测试下用的
                break;
            } 
        }
        printf("*保存结果输出\n");
        //保存结果输出
        char out_path[256];
        sprintf(out_path, "%s/%s.%s", out_dir, std::to_string(frame_count).c_str(), "jpg");
        write_image(out_path, &crop_image);
        frame_count++;
        // === 回收 ===
        src_image = crop_image;
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