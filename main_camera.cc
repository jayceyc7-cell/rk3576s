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
#include "rknn_api.h"
#include "dirent.h"
#include "rknn_pool.h"
#include <thread>
#include <chrono>
#include "awi_track.hpp"

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

int main(int argc, char **argv)
{
    printf("YOLOv8 camera! detection by wyc!!!!!!\n");

    if (argc != 4)
    {
        printf("Usage: %s <model_path> <frames_dir> <out_dir>\n", argv[0]);
        printf("例如： ffmpeg -i test.mp4 frames/frame_%%06d.jpg\n");
        return -1;
    }

    const char *model_path = argv[1];
    const char *frames_dir = argv[2];
    const char *out_dir = argv[3];
    const int thread_count = 2;
    const char* label_path = nullptr;

    // 初始化 YOLO 模型
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();
    int ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret)
    {
        printf("init_yolov8_model failed!\n");
        return -1;
    }

    // 创建输出目录（假如不存在）
    char cmd[256];
    sprintf(cmd, "mkdir -p %s", out_dir);
    system(cmd);

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

    struct timeval start, end;
    double time_use = 0;

    //线程池
    // auto rknn_pool = std::make_unique<RknnPool>(
    //     model_path, thread_count, label_path);

    //图像加载
    // for (size_t i = 0; i < file_list.size(); i++)
    // {
    //     char img_path[256];
    //     sprintf(img_path, "%s/%s", frames_dir, file_list[i].c_str());
    //     printf("Processing frame: %s\n", img_path);
    //     image_buffer_t src_image;
    //     memset(&src_image, 0, sizeof(src_image));

    //     // 读取当前帧
    //     if (read_image(img_path, &src_image) != 0)
    //     {
    //         printf("read_image failed: %s\n", img_path);
    //         continue;
    //     }

    //     rknn_pool->AddInferenceTask(, );



    // }

    /*  1.先获取到待处理的图像(来自视频或者来自摄像头)
            前处理（原图像的裁切到检测的尺寸）
        2. 1）将图像添加到RknnPool线程池的任务队列中
           2）直接将图像放到等待推理的队列中
        3. 1）最后等待线程池中的线程处理完任务队列中的图像
           2）等待推理完成的队列中的图像
    */



    TrackFrame tracker;
    tracker.Init(50);

    // 🚀 开始逐帧推理
    for (size_t i = 0; i < file_list.size(); i++)
    {
        char img_path[256];
        sprintf(img_path, "%s/%s", frames_dir, file_list[i].c_str());

        printf("Processing frame: %s\n", img_path);

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(src_image));

        // 读取当前帧
        if (read_image(img_path, &src_image) != 0)
        {
            printf("read_image failed: %s\n", img_path);
            continue;
        }

        object_detect_result_list od_results; 

        gettimeofday(&start, NULL);
        inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
        gettimeofday(&end, NULL);
        time_use = (end.tv_sec - start.tv_sec) * 1000.0 +
                   (end.tv_usec - start.tv_usec) / 1000.0;

        printf("Frame %s inference time: %.2f ms\n", file_list[i].c_str(), time_use);

        // draw detection results
        char text[256];
        for (int j = 0; j < od_results.count; j++)
        {
            object_detect_result *det = &(od_results.results[j]);

            int x1 = det->box.left;
            int y1 = det->box.top;
            int x2 = det->box.right;
            int y2 = det->box.bottom;

            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);            
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        }

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
        tracker.ProcessFrame(i, src_image, detections, track_results);

        /******************球的跟踪预测出口*********************/

        // 保存输出帧
        char out_path[256];
        sprintf(out_path, "%s/%s", out_dir, file_list[i].c_str());
        write_image(out_path, &src_image);

        //释放内存
        free(src_image.virt_addr);
    }




    // 清理
    deinit_post_process();
    release_yolov8_model(&rknn_app_ctx);

    printf("All frames processed. Now use ffmpeg to combine them:\n");
    printf("ffmpeg -r 30 -i %s/frame_%%06d.jpg -vcodec libx264 -pix_fmt yuv420p result.mp4\n", out_dir);

    return 0;
}
