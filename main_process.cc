

/*-------------------------------------------
                图片检测
-------------------------------------------*/
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <sys/time.h>

// #include <vector>
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <algorithm>
// #include <cmath>

// #include "yolov8.h"
// #include "image_utils.h"
// #include "file_utils.h"
// #include "image_drawing.h"
// #include "rknn_api.h"
// #include "dirent.h"

// #if defined(RV1106_1103) 
//     #include "dma_alloc.hpp"
// #endif

// /*-------------------------------------------
//                   Main Function
// -------------------------------------------*/
// int main(int argc, char **argv)
// {
//     //setbuf(stdout, NULL); // 禁用 stdout 缓冲

//     struct timeval start, end;
//     double time_use = 0;
//     printf("example for yolov8 main!!\n");
//     if (argc != 3)
//     {
//         printf("%s <model_path> <image_path>\n", argv[0]);
//         return -1;
//     }

//     const char *model_path = argv[1];
//     const char *image_path = argv[2];

//     int ret;
//     rknn_app_context_t rknn_app_ctx;
//     memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

//     init_post_process();

//     ret = init_yolov8_model(model_path, &rknn_app_ctx);
//     if (ret != 0)
//     {
//         printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
//         deinit_post_process();

//         ret = release_yolov8_model(&rknn_app_ctx);
//         if (ret != 0)
//         {
//             printf("release_yolov8_model fail! ret=%d\n", ret);
//         }
//         // goto out;
//     }
//     //图片检测
//     image_buffer_t src_image;
//     image_buffer_t crop_image;
//     image_rect_t crop_rect = {0};
//     memset(&src_image, 0, sizeof(image_buffer_t));
//     memset(&crop_image, 0, sizeof(image_buffer_t));
//     ret = read_image(image_path, &src_image);

//     if (ret != 0)
//     {
//         printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
//         goto out;
//     }

//     // 开始计时
//     gettimeofday(&start, NULL);

//     crop_alg_image(&src_image, &crop_image, {658, 644, 701, 691}, &crop_rect); //差不多要10ms

//     gettimeofday(&end, NULL);

//     // 计算耗时（微秒→毫秒）
//     time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
//                (end.tv_usec - start.tv_usec) / 1000.0;

//     printf("crop_alg_image 耗时：%.3f ms\n", time_use);


//     gettimeofday(&start, NULL);

//     object_detect_result_list od_results;
    
//     ret = inference_yolov8_model(&rknn_app_ctx, &crop_image, &od_results);
//     if (ret != 0)
//     {
//         printf("init_yolov8_model fail! ret=%d\n", ret);
//         goto out;
//     }
//     gettimeofday(&end, NULL);

//     // 计算耗时（微秒→毫秒）
//     time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
//                (end.tv_usec - start.tv_usec) / 1000.0;

//     printf("inference_yolov8_model 耗时：%.3f ms, count:%d\n", time_use, od_results.count);

//     // 画框和概率
//     char text[256];
//     for (int i = 0; i < od_results.count; i++)
//     {
//         object_detect_result *det_result = &(od_results.results[i]);
//         printf("cls_id:%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
//                det_result->box.left, det_result->box.top,
//                det_result->box.right, det_result->box.bottom,
//                det_result->prop);

//         int x1 = det_result->box.left;
//         int y1 = det_result->box.top;
//         int x2 = det_result->box.right;
//         int y2 = det_result->box.bottom;

//         draw_rectangle(&crop_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

//         sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
//         draw_text(&crop_image, text, x1, y1 - 20, COLOR_RED, 10);
//         // int x1 = det_result->box.left + crop_rect.left;
//         // int y1 = det_result->box.top + crop_rect.top;
//         // int x2 = det_result->box.right + crop_rect.left;
//         // int y2 = det_result->box.bottom + crop_rect.top;

//         // draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

//         // sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
//         // draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
//     }

//     write_image("out1.jpg", &crop_image);

// out:
//     deinit_post_process();

//     ret = release_yolov8_model(&rknn_app_ctx);
//     if (ret != 0)
//     {
//         printf("release_yolov8_model fail! ret=%d\n", ret);
//     }

//     if (src_image.virt_addr != NULL)
//     {
//         free(src_image.virt_addr);
//     }
//     if (crop_image.virt_addr != NULL)
//     {
//         free(crop_image.virt_addr);
//     }

//     return 0;
// }

/*-------------------------------------------
                模型精度评估
-------------------------------------------*/
// struct BBox {
//     float x1, y1, x2, y2;
//     int cls;
//     float score;
//     bool operator==(const BBox &other) const
//     {
//         return fabs(x1 - other.x1) < 1e-6 &&
//                fabs(y1 - other.y1) < 1e-6 &&
//                fabs(x2 - other.x2) < 1e-6 &&
//                fabs(y2 - other.y2) < 1e-6 &&
//                cls == other.cls &&
//                fabs(score - other.score) < 1e-6;
//     }
// };
// // 计算 IoU
// float iou(BBox a, BBox b) {
//     float xx1 = std::max(a.x1, b.x1);
//     float yy1 = std::max(a.y1, b.y1);
//     float xx2 = std::min(a.x2, b.x2);
//     float yy2 = std::min(a.y2, b.y2);

//     float w = std::max(0.0f, xx2 - xx1);
//     float h = std::max(0.0f, yy2 - yy1);

//     float inter = w * h;
//     float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
//     float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

//     return inter / (areaA + areaB - inter + 1e-6);
// }

// // 读取 YOLO 标签（txt）
// std::vector<BBox> load_gt_boxes(const std::string& txt_path, int img_w, int img_h) {
//     std::vector<BBox> gts;
//     std::ifstream ifs(txt_path);
//     if (!ifs.is_open()) return gts;

//     float cls, cx, cy, w, h;
//     while (ifs >> cls >> cx >> cy >> w >> h) {
//         BBox b;
//         b.cls = (int)cls;
//         b.x1 = (cx - w/2) * img_w;
//         b.y1 = (cy - h/2) * img_h;
//         b.x2 = (cx + w/2) * img_w;
//         b.y2 = (cy + h/2) * img_h;
//         b.score = 1.0; // GT 没分数
//         gts.push_back(b);
//     }
//     return gts;
// }
// float compute_ap(std::vector<float>& precisions, std::vector<float>& recalls) {
//     // COCO 官方风格插值
//     float ap = 0.0;
//     for (float t = 0.0; t < 1.01; t += 0.01) {
//         float p = 0;
//         for (int i = 0; i < recalls.size(); i++) {
//             if (recalls[i] >= t)
//                 p = std::max(p, precisions[i]);
//         }
//         ap += p;
//     }
//     return ap / 101.0;
// }

// int main(int argc, char **argv)
// {
//     // ==========================================================
//     //      模型评估：Precision，Recall，F1，mAP50/75/50-95
//     // ==========================================================

//     std::string img_dir = "/root/special_test_delete/images/test/";
//     std::string label_dir = "/root/special_test_delete/labels/test/";

//     DIR *dir = opendir(img_dir.c_str());
//     if (!dir)
//     {
//         printf("无法打开图片目录\n");
//         return 0;
//     }

//     const char *model_path = argv[1];
//     const char *image_path = argv[2];

//     std::vector<BBox> all_dets;
//     std::vector<BBox> all_gts;

//     int ret;
//     rknn_app_context_t rknn_app_ctx;
//     memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

//     init_post_process();

//     ret = init_yolov8_model(model_path, &rknn_app_ctx);
//     if (ret != 0)
//     {
//         printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
//         deinit_post_process();

//         ret = release_yolov8_model(&rknn_app_ctx);
//         if (ret != 0)
//         {
//             printf("release_yolov8_model fail! ret=%d\n", ret);
//         }
//         // goto out;
//     }

//     struct dirent *ptr;
//     while ((ptr = readdir(dir)) != NULL)
//     {
//         std::string name = ptr->d_name;
//         if (name.size() < 4)
//             continue;
//         if (name.substr(name.size() - 4) != ".jpg" &&
//             name.substr(name.size() - 4) != ".png")
//             continue;

//         std::string img_path = img_dir + name;
//         std::string txt_path = label_dir + name.substr(0, name.size() - 4) + ".txt";

//         // 读取图片
//         image_buffer_t src;
//         memset(&src, 0, sizeof(src));
//         if (read_image(img_path.c_str(), &src) != 0)
//             continue;

//         // 推理
//         object_detect_result_list od_res;
//         inference_yolov8_model(&rknn_app_ctx, &src, &od_res);

//         // 处理预测框
//         for (int i = 0; i < od_res.count; i++)
//         {
//             object_detect_result *r = &(od_res.results[i]);
//             BBox b;
//             b.x1 = r->box.left;
//             b.y1 = r->box.top;
//             b.x2 = r->box.right;
//             b.y2 = r->box.bottom;
//             b.cls = r->cls_id;
//             b.score = r->prop;
//             all_dets.push_back(b);
//         }

//         // 加载 GT
//         auto gts = load_gt_boxes(txt_path, src.width, src.height);
//         all_gts.insert(all_gts.end(), gts.begin(), gts.end());
//         // ------------------------------
//         // 🔴 必须释放图片内存，否则 OOM
//         // ------------------------------
//         if (src.virt_addr)
//             free(src.virt_addr);

//     }

//     closedir(dir);

// int num_classes = 3;

// // 为每个类别保存结果
// std::vector<float> precision_c(num_classes);
// std::vector<float> recall_c(num_classes);
// std::vector<float> f1_c(num_classes);

// std::vector<std::vector<float>> APs_c(num_classes); // 每类10个AP

// // -----------------------------------------
// // 按类别统计 Precision / Recall / F1
// // -----------------------------------------
// for (int cls = 0; cls < num_classes; cls++)
// {
//     int TP50 = 0, FP50 = 0, FN50 = 0;

//     // 收集该类别的 GT / DET
//     std::vector<BBox> gts, dets;
//     for (auto &g : all_gts) if (g.cls == cls) gts.push_back(g);
//     for (auto &d : all_dets) if (d.cls == cls) dets.push_back(d);
//     int dets_num = dets.size();

//     // GT - 计算 TP / FN
//     for (auto &gt : gts)
//     {
//         float best_iou = 0;
//         BBox best_det;
//         for (auto &det : dets){
//             if (best_iou < iou(det, gt)){
//                 best_det = det;
//             }
//             best_iou = std::max(best_iou, iou(det, gt));
//         }
//         if (best_iou >= 0.5){
//             TP50++;
//             dets.erase(std::remove(dets.begin(), dets.end(), best_det), dets.end());
//         }
//         else FN50++;
//     }

//     FP50 = dets_num - TP50;

//     precision_c[cls] = TP50 * 1.0 / (TP50 + FP50 + 1e-6);
//     recall_c[cls] = TP50 * 1.0 / (TP50 + FN50 + 1e-6);
//     f1_c[cls] = 2 * precision_c[cls] * recall_c[cls] /
//                 (precision_c[cls] + recall_c[cls] + 1e-6);
// }

// // -----------------------------------------
// // 按类别计算 mAP (COCO 标准)
// // -----------------------------------------
// float thresholds[10] = {0.50, 0.55, 0.60, 0.65, 0.70,
//                         0.75, 0.80, 0.85, 0.90, 0.95};

// for (int cls = 0; cls < num_classes; cls++)
// {
//     // 按类别筛选
//     std::vector<BBox> gts, dets;
//     for (auto &g : all_gts) if (g.cls == cls) gts.push_back(g);
//     for (auto &d : all_dets) if (d.cls == cls) dets.push_back(d);

//     for (float iou_th : thresholds)
//     {
//         std::vector<std::pair<float, int>> score_TPFP;

//         // 按 score 降序
//         std::vector<BBox> dets_sorted = dets;
//         std::sort(dets_sorted.begin(), dets_sorted.end(),
//                   [](const BBox &a, const BBox &b)
//                   { return a.score > b.score; });

//         std::vector<int> used(gts.size(), 0);

//         for (auto &det : dets_sorted)
//         {
//             float best_iou = 0;
//             int best_gt = -1;

//             for (int i = 0; i < gts.size(); i++)
//             {
//                 if (used[i]) continue;
//                 float ov = iou(det, gts[i]);
//                 if (ov > best_iou)
//                 {
//                     best_iou = ov;
//                     best_gt = i;
//                 }
//             }

//             if (best_iou >= iou_th)
//             {
//                 used[best_gt] = 1;
//                 score_TPFP.push_back({det.score, 1});
//             }
//             else
//             {
//                 score_TPFP.push_back({det.score, 0});
//             }
//         }

//         std::sort(score_TPFP.begin(), score_TPFP.end(),
//                   [](auto &a, auto &b)
//                   { return a.first > b.first; });

//         int TP = 0, FP = 0;
//         int total_gt = gts.size();

//         std::vector<float> precisions, recalls;

//         for (auto &p : score_TPFP)
//         {
//             if (p.second == 1) TP++;
//             else FP++;

//             precisions.push_back(TP * 1.0 / (TP + FP + 1e-6));
//             recalls.push_back(TP * 1.0 / (total_gt + 1e-6));
//         }

//         float ap = compute_ap(precisions, recalls);
//         APs_c[cls].push_back(ap);
//     }
// }

// // -----------------------------------------
// // 输出每个类别结果
// // -----------------------------------------
// for (int cls = 0; cls < num_classes; cls++)
// {
//     printf("\n===== Class %d =====\n", cls);

//     printf("Precision: %.4f\n", precision_c[cls]);
//     printf("Recall   : %.4f\n", recall_c[cls]);
//     printf("F1       : %.4f\n", f1_c[cls]);

//     printf("mAP@50   : %.4f\n", APs_c[cls][0]);
//     printf("mAP@75   : %.4f\n", APs_c[cls][5]);

//     float sum = 0;
//     for (float v : APs_c[cls]) sum += v;
//     printf("mAP@50-95: %.4f\n", sum / 10);
// }

//     return 0;
// }

/*-------------------------------------------
                视频检测
-------------------------------------------*/
// int main(int argc, char **argv)
// {
//     printf("YOLOv8 video! detection!!\n");

//     if (argc != 4)
//     {
//         printf("Usage: %s <model_path> <frames_dir> <out_dir>\n", argv[0]);
//         printf("例如： ffmpeg -i test.mp4 frames/frame_%%06d.jpg\n");
//         return -1;
//     }

//     const char *model_path = argv[1];
//     const char *frames_dir = argv[2];
//     const char *out_dir = argv[3];

//     // 初始化 YOLO 模型
//     rknn_app_context_t rknn_app_ctx;
//     memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

//     init_post_process();
//     int ret = init_yolov8_model(model_path, &rknn_app_ctx);
//     if (ret)
//     {
//         printf("init_yolov8_model failed!\n");
//         return -1;
//     }

//     // 创建输出目录（假如不存在）
//     char cmd[256];
//     sprintf(cmd, "mkdir -p %s", out_dir);
//     system(cmd);

//     // 读取目录中的所有 frame_XXXXXX.jpg 文件
//     std::vector<std::string> file_list;
//     DIR *dir = opendir(frames_dir);
//     struct dirent *entry;

//     while ((entry = readdir(dir)) != NULL)
//     {
//         if (endswith(entry->d_name, ".jpg") || endswith(entry->d_name, ".png"))
//         {
//             file_list.push_back(entry->d_name);
//         }
//     }
//     closedir(dir);

//     // 按文件名排序（frame_000001.jpg → frame_000002.jpg）
//     std::sort(file_list.begin(), file_list.end());

//     struct timeval start, end;
//     double time_use = 0;

//     // 开始逐帧推理
//     for (size_t i = 0; i < file_list.size(); i++)
//     {
//         char img_path[256];
//         sprintf(img_path, "%s/%s", frames_dir, file_list[i].c_str());

//         printf("Processing frame: %s\n", img_path);

//         image_buffer_t src_image;
//         memset(&src_image, 0, sizeof(src_image));

//         // 读取当前帧
//         if (read_image(img_path, &src_image) != 0)
//         {
//             printf("read_image failed: %s\n", img_path);
//             continue;
//         }

//         object_detect_result_list od_results; 

//         gettimeofday(&start, NULL);
//         inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
//         gettimeofday(&end, NULL);

//         time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
//                    (end.tv_usec - start.tv_usec) / 1000.0;

//         printf("Frame %s   time: %.2f ms\n", file_list[i].c_str(), time_use);

//         // draw detection results
//         char text[256];
//         for (int j = 0; j < od_results.count; j++)
//         {
//             object_detect_result *det = &(od_results.results[j]);

//             int x1 = det->box.left;
//             int y1 = det->box.top;
//             int x2 = det->box.right;
//             int y2 = det->box.bottom;

//             draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
            

//             sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
//             if (strncmp(text, "ball", 4) == 0)
//             {
//                 draw_rectangle(&src_image, x1, y1, (x2 - x1) * 2, (y2 - y1) * 2, COLOR_GREEN, 3);
//             }
            
//             draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
//         }

//         // 保存输出帧
//         char out_path[256];
//         sprintf(out_path, "%s/%s", out_dir, file_list[i].c_str());
//         write_image(out_path, &src_image);

//         free(src_image.virt_addr);
//     }

//     // 清理
//     deinit_post_process();
//     release_yolov8_model(&rknn_app_ctx);

//     printf("All frames processed. Now use ffmpeg to combine them:\n");
//     printf("ffmpeg -r 30 -i %s/frame_%%06d.jpg -vcodec libx264 -pix_fmt yuv420p result.mp4\n", out_dir);

//     return 0;
// }


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

#include <thread>
#include <csignal>
#include <atomic>
#include <iostream>
#include <unistd.h> 

/*--------------------------------------*/
#define PIC_FULL_WIDTH 2560
#define PIC_FULL_HEIGHT 1440
#define ALG_CROP_WIDTH 1280
#define ALG_CROP_HEIGHT 720

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
            printf("~~~~~~~~~~FrameQueue full, dropping oldest frame\n");
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
    constexpr int interval_ms = 45;
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
    bool need_crop = false;
    image_rect_t obj_rect = {0};
    image_rect_t last_obj_rect = {0};
    bool first_frame = true;
    image_rect_t crop_rect = {0};
    image_buffer_t src_image = {0};
    image_buffer_t save_image = {0};
    image_buffer_t crop_image = {0};

    int crop_width = ALG_CROP_WIDTH;;
    int crop_height = ALG_CROP_HEIGHT;

    while (true) {
        if(src_image.width == 0 && src_image.height == 0){
            memset(&src_image, 0, sizeof(image_buffer_t));
            if (!fq.pop(src_image)){
                usleep(5000);
                continue;
            }
        }
        
        memset(&crop_image, 0, sizeof(image_buffer_t));

        if(need_crop){     
            //裁剪  
            gettimeofday(&start, NULL);    
            crop_alg_image(&src_image, &crop_image, obj_rect, &crop_rect, crop_width, crop_height); //差不多要10ms;
            gettimeofday(&end, NULL);

            time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
                       (end.tv_usec - start.tv_usec) / 1000.0;
    
            total_time_use += time_use;      
            printf("*裁剪: %d x %d, use time: %.2f ms\n", crop_width, crop_height, time_use);
        }else{
            //不裁剪，使用全图
            crop_image = src_image;
        }
        // === 推理 / 处理 ===
        object_detect_result_list od_results; 
        if(crop_image.virt_addr == NULL){
            printf("------crop_image is NULL\n");
            // === 回收 ===
            pool.release(src_image);
            memset(&src_image, 0, sizeof(image_buffer_t));
            continue;
        }
        gettimeofday(&start, NULL);
        inference_yolov8_model(&rknn_app_ctx, &crop_image, &od_results);
        gettimeofday(&end, NULL);

        time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
                   (end.tv_usec - start.tv_usec) / 1000.0;

        total_time_use += time_use;      
        printf("*Frame %d inference time: %.2f ms\n", frame_count, time_use);

        // draw detection results
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

                draw_rectangle(&crop_image, x1, y1, (x2 - x1) * 2, (y2 - y1) * 2, COLOR_GREEN, 3);
                draw_text(&crop_image, text, x1, y1 - 20, COLOR_RED, 10);

                //转换到全图坐标
                if(crop_image.width == PIC_FULL_WIDTH && crop_image.height == PIC_FULL_HEIGHT){
                    obj_rect = det->box;
                }else{
                    obj_rect.left = det->box.left + crop_rect.left;
                    obj_rect.top = det->box.top + crop_rect.top;
                    obj_rect.right = det->box.right + crop_rect.left;
                    obj_rect.bottom = det->box.bottom + crop_rect.top;
                }               
                need_crop = true;
                break;
            } 
            printf("need crop = %d\n", need_crop);
        }
        ///检测不到目标处理
        if(!first_frame && (od_results.count == 0 || j == od_results.count)){
            need_crop = false;
            if(crop_image.width != PIC_FULL_WIDTH || crop_image.height != PIC_FULL_HEIGHT){
                //逐步放大裁剪区域
                printf("***扩大裁剪区域: %d x %d\n", PIC_FULL_WIDTH, PIC_FULL_HEIGHT);
                if(crop_image.virt_addr != NULL){
                    free(crop_image.virt_addr);
                    memset(&crop_image, 0, sizeof(image_buffer_t));
                }
                continue;

            }else{
                printf("***全图也检测不到\n");
                // 保存上次写入的输出帧（后续可以连续前三帧检测不到使用之前的数据，否则从全景图裁剪）
                crop_width = ALG_CROP_WIDTH;
                crop_height = ALG_CROP_HEIGHT;     
                crop_alg_image(&src_image, &save_image, last_obj_rect, &crop_rect, crop_width, crop_height);
                char out_path[256];
                sprintf(out_path, "%s/%s.%s", out_dir, std::to_string(frame_count).c_str(),"jpg");

                write_image(out_path, &save_image); 
                frame_count++;
                if(save_image.virt_addr != NULL){
                    free(save_image.virt_addr);
                    memset(&save_image, 0, sizeof(image_buffer_t));
                }

                // === 回收 ===
                pool.release(src_image);
                memset(&src_image, 0, sizeof(image_buffer_t));
                continue;
            }
        }
        ///检测目标处理
        if(first_frame || need_crop){
            printf("*save to file...\n");
            // 保存输出帧
            
            if(need_crop){
                if(crop_image.width == PIC_FULL_WIDTH && crop_image.height == PIC_FULL_HEIGHT){
                    crop_width = ALG_CROP_WIDTH;
                    crop_height = ALG_CROP_HEIGHT;     
                    crop_alg_image(&crop_image, &save_image, obj_rect, &crop_rect, crop_width, crop_height); //差不多要10ms;
                    crop_image = save_image;
                }else{
                    save_image = crop_image;
                }
                last_obj_rect = obj_rect;
            }
           
            char out_path[256];
            sprintf(out_path, "%s/%s.%s", out_dir, std::to_string(frame_count).c_str(),"jpg");
            write_image(out_path, &crop_image);
            frame_count++;
            
            if(save_image.virt_addr != NULL){
                free(save_image.virt_addr);
                memset(&save_image, 0, sizeof(image_buffer_t));
            }
  
            // === 回收 ===
            pool.release(src_image);
            memset(&src_image, 0, sizeof(image_buffer_t));

            if(first_frame && need_crop){
                first_frame = false;
            }
        }
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
    constexpr size_t QUEUE_SIZE = 6;
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
    while (!g_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

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