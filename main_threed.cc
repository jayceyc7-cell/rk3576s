

/*-------------------------------------------
                图片检测
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "rknn_api.h"
#include "dirent.h"
#include "rknn_pool.h"
#include <thread>
#include <chrono>

#if defined(RV1106_1103) 
    #include "dma_alloc.hpp"
#endif

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
// int main(int argc, char **argv)
// {
//     //setbuf(stdout, NULL); // 禁用 stdout 缓冲

//     struct timeval start, end;
//     double time_use = 0;
//     printf("example for yolov8 wyc!!\n");
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
//     memset(&src_image, 0, sizeof(image_buffer_t));
//     ret = read_image(image_path, &src_image);

//     if (ret != 0)
//     {
//         printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
//         goto out;
//     }

//     object_detect_result_list od_results;
//     // 开始计时
//     gettimeofday(&start, NULL);
//     ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
//     if (ret != 0)
//     {
//         printf("init_yolov8_model fail! ret=%d\n", ret);
//         goto out;
//     }
//     gettimeofday(&end, NULL);

//     // 计算耗时（微秒→毫秒）
//     time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
//                (end.tv_usec - start.tv_usec) / 1000.0;

//     printf("inference_yolov8_model 耗时：%.3f ms\n", time_use);

//     // 画框和概率
//     char text[256];
//     for (int i = 0; i < od_results.count; i++)
//     {
//         object_detect_result *det_result = &(od_results.results[i]);
//         printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
//                det_result->box.left, det_result->box.top,
//                det_result->box.right, det_result->box.bottom,
//                det_result->prop);
//         int x1 = det_result->box.left;
//         int y1 = det_result->box.top;
//         int x2 = det_result->box.right;
//         int y2 = det_result->box.bottom;

//         draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

//         sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
//         draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
//     }

//     write_image("out1.jpg", &src_image);

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

//     return 0;
// }

/*-------------------------------------------
                模型精度评估
-------------------------------------------*/
struct BBox {
    float x1, y1, x2, y2;
    int cls;
    float score;
    bool operator==(const BBox &other) const
    {
        return fabs(x1 - other.x1) < 1e-6 &&
               fabs(y1 - other.y1) < 1e-6 &&
               fabs(x2 - other.x2) < 1e-6 &&
               fabs(y2 - other.y2) < 1e-6 &&
               cls == other.cls &&
               fabs(score - other.score) < 1e-6;
    }
    void print_b(){
        printf("\nx1:%d, y1:%d, x2:%d, y2:%d\n", x1, y1, x2, y2);
    }
};
// 计算 IoU
float iou(BBox a, BBox b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);

    float inter = w * h;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (areaA + areaB - inter + 1e-6);
}

// 读取 YOLO 标签（txt）
std::vector<BBox> load_gt_boxes(const std::string& txt_path, int img_w, int img_h) {
    std::vector<BBox> gts;
    std::ifstream ifs(txt_path);
    if (!ifs.is_open()) return gts;

    float cls, cx, cy, w, h;
    while (ifs >> cls >> cx >> cy >> w >> h) {
        BBox b;
        b.cls = (int)cls;
        b.x1 = (cx - w/2) * img_w;
        b.y1 = (cy - h/2) * img_h;
        b.x2 = (cx + w/2) * img_w;
        b.y2 = (cy + h/2) * img_h;
        b.score = 1.0; // GT 没分数
        gts.push_back(b);
    }
    return gts;
}
float compute_ap(std::vector<float>& precisions, std::vector<float>& recalls) {
    // COCO 官方风格插值
    float ap = 0.0;
    for (float t = 0.0; t < 1.01; t += 0.01) {
        float p = 0;
        for (int i = 0; i < recalls.size(); i++) {
            if (recalls[i] >= t)
                p = std::max(p, precisions[i]);
        }
        ap += p;
    }
    return ap / 101.0;
}

int main(int argc, char **argv)
{
    // ==========================================================
    //      模型评估：Precision，Recall，F1，mAP50/75/50-95
    // ==========================================================

    std::string img_dir = "/root/special_test_delete/images/test1/";
    std::string label_dir = "/root/special_test_delete/labels/test1/";

    struct timeval start, end;
    struct timeval one_start, one_end;
    double time_use = 0;

    DIR *dir = opendir(img_dir.c_str());
    if (!dir)
    {
        printf("无法打开图片目录\n");
        return 0;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const int thread_count = 2;
    const char* label_path = nullptr;

    int output_num = 0;   //在队列中的未取出的结果
    int input_num = 0;
    int min_time = 20;  //最少的运行时间
    int delay = 0;

    std::vector<BBox> all_dets;
    std::vector<BBox> all_gts;
    std::shared_ptr<Output> od_res;

    init_post_process();

    auto rknn_pool = std::make_unique<RknnPool>(
        model_path, thread_count, label_path);
    
    gettimeofday(&start, NULL);
    struct dirent *ptr;
    bool is_read = 1;
    while(is_read or input_num > output_num)
    {
        gettimeofday(&one_start, NULL);
        if(is_read and (ptr = readdir(dir)) != NULL){
            std::string name = ptr->d_name;
            if (name.size() < 4)
                continue;
            if (name.substr(name.size() - 4) != ".jpg" &&
                name.substr(name.size() - 4) != ".png")
                continue;

            std::string img_path = img_dir + name;
            printf("img_path: %s\n", img_path.c_str());
            std::string txt_path = label_dir + name.substr(0, name.size() - 4) + ".txt";

            // 读取图片
            auto src = std::make_unique<image_buffer_t>(image_buffer_t{});
            if (read_image(img_path.c_str(), src.get()) != 0)
                continue;

            // 送入推理
            rknn_pool->AddInferenceTask(std::move(src), txt_path);
            input_num++;
        }else{
            is_read = 0;
        }
        
        printf("\n input: %d\n", input_num);
        printf("\n output: %d\n", output_num);
        od_res = rknn_pool->GetImageResultFromQueue();
        if(od_res != nullptr){
            output_num++;
            std::shared_ptr<object_detect_result_list> result = od_res.get()->result;
            std::shared_ptr<image_buffer_t> img = od_res.get()->img;
            // 处理预测框
            for (int i = 0; i < result.get()->count; i++)
            {
                object_detect_result *r = &(result.get()->results[i]);
                BBox b;
                b.x1 = r->box.left;
                b.y1 = r->box.top;
                b.x2 = r->box.right;
                b.y2 = r->box.bottom;
                b.cls = r->cls_id;
                b.score = r->prop;
                // b.print_b();
                all_dets.push_back(b);
            }

            // 加载 GT
            auto gts = load_gt_boxes(od_res.get()->txt_path, img.get()->width, img.get()->height);
            all_gts.insert(all_gts.end(), gts.begin(), gts.end());
        }
        gettimeofday(&one_end, NULL);
        time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
            (end.tv_usec - start.tv_usec) / 1000.0;
        if(time_use < min_time){
            delay = int(min_time - time_use);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
    }

    gettimeofday(&end, NULL);

    closedir(dir);
    // 计算耗时（微秒→毫秒）
    time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
               (end.tv_usec - start.tv_usec) / 1000.0;
    printf("inference_yolov8_model 耗时：%.3f ms\n", time_use);

int num_classes = 3;

// 为每个类别保存结果
std::vector<float> precision_c(num_classes);
std::vector<float> recall_c(num_classes);
std::vector<float> f1_c(num_classes);

std::vector<std::vector<float>> APs_c(num_classes); // 每类10个AP

// -----------------------------------------
// 按类别统计 Precision / Recall / F1
// -----------------------------------------
for (int cls = 0; cls < num_classes; cls++)
{
    int TP50 = 0, FP50 = 0, FN50 = 0;

    // 收集该类别的 GT / DET
    std::vector<BBox> gts, dets;
    for (auto &g : all_gts) if (g.cls == cls) gts.push_back(g);
    for (auto &d : all_dets) if (d.cls == cls) dets.push_back(d);
    int dets_num = dets.size();

    // GT - 计算 TP / FN
    for (auto &gt : gts)
    {
        float best_iou = 0;
        BBox best_det;
        for (auto &det : dets){
            if (best_iou < iou(det, gt)){
                best_det = det;
            }
            best_iou = std::max(best_iou, iou(det, gt));
        }
        if (best_iou >= 0.5){
            TP50++;
            dets.erase(std::remove(dets.begin(), dets.end(), best_det), dets.end());
        }
        else FN50++;
    }

    FP50 = dets_num - TP50;

    precision_c[cls] = TP50 * 1.0 / (TP50 + FP50 + 1e-6);
    recall_c[cls] = TP50 * 1.0 / (TP50 + FN50 + 1e-6);
    f1_c[cls] = 2 * precision_c[cls] * recall_c[cls] /
                (precision_c[cls] + recall_c[cls] + 1e-6);
}

// -----------------------------------------
// 按类别计算 mAP (COCO 标准)
// -----------------------------------------
float thresholds[10] = {0.50, 0.55, 0.60, 0.65, 0.70,
                        0.75, 0.80, 0.85, 0.90, 0.95};

for (int cls = 0; cls < num_classes; cls++)
{
    // 按类别筛选
    std::vector<BBox> gts, dets;
    for (auto &g : all_gts) if (g.cls == cls) gts.push_back(g);
    for (auto &d : all_dets) if (d.cls == cls) dets.push_back(d);

    for (float iou_th : thresholds)
    {
        std::vector<std::pair<float, int>> score_TPFP;

        // 按 score 降序
        std::vector<BBox> dets_sorted = dets;
        std::sort(dets_sorted.begin(), dets_sorted.end(),
                  [](const BBox &a, const BBox &b)
                  { return a.score > b.score; });

        std::vector<int> used(gts.size(), 0);

        for (auto &det : dets_sorted)
        {
            float best_iou = 0;
            int best_gt = -1;

            for (int i = 0; i < gts.size(); i++)
            {
                if (used[i]) continue;
                float ov = iou(det, gts[i]);
                if (ov > best_iou)
                {
                    best_iou = ov;
                    best_gt = i;
                }
            }

            if (best_iou >= iou_th)
            {
                used[best_gt] = 1;
                score_TPFP.push_back({det.score, 1});
            }
            else
            {
                score_TPFP.push_back({det.score, 0});
            }
        }

        std::sort(score_TPFP.begin(), score_TPFP.end(),
                  [](auto &a, auto &b)
                  { return a.first > b.first; });

        int TP = 0, FP = 0;
        int total_gt = gts.size();

        std::vector<float> precisions, recalls;

        for (auto &p : score_TPFP)
        {
            if (p.second == 1) TP++;
            else FP++;

            precisions.push_back(TP * 1.0 / (TP + FP + 1e-6));
            recalls.push_back(TP * 1.0 / (total_gt + 1e-6));
        }

        float ap = compute_ap(precisions, recalls);
        APs_c[cls].push_back(ap);
    }
}

// -----------------------------------------
// 输出每个类别结果
// -----------------------------------------
for (int cls = 0; cls < num_classes; cls++)
{
    printf("\n===== Class %d =====\n", cls);

    printf("Precision: %.4f\n", precision_c[cls]);
    printf("Recall   : %.4f\n", recall_c[cls]);
    printf("F1       : %.4f\n", f1_c[cls]);

    printf("mAP@50   : %.4f\n", APs_c[cls][0]);
    printf("mAP@75   : %.4f\n", APs_c[cls][5]);

    float sum = 0;
    for (float v : APs_c[cls]) sum += v;
    printf("mAP@50-95: %.4f\n", sum / 10);
}

    return 0;
}

/*-------------------------------------------
                视频检测
-------------------------------------------*/
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <sys/time.h>
// #include <dirent.h>
// #include <algorithm>
// #include <vector>
// #include <string>

// #include "yolov8.h"
// #include "image_utils.h"
// #include "file_utils.h"
// #include "image_drawing.h"

// int endswith(const char *str, const char *suffix)
// {
//     if (!str || !suffix)
//         return 0;
//     int lenstr = strlen(str);
//     int lensuffix = strlen(suffix);
//     if (lensuffix > lenstr)
//         return 0;
//     return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
// }

// int main(int argc, char **argv)
// {
//     printf("YOLOv8 video! detection by wyc!!\n");

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

//     // 🚀 开始逐帧推理
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

//         printf("Frame %s inference time: %.2f ms\n", file_list[i].c_str(), time_use);

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
