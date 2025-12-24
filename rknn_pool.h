//
// Created by kaylor on 3/6/24.
//

#pragma once
#include "queue"
#include "threadpool.h"
#include "yolov8.h"

class Output {
  public:
    Output(std::shared_ptr<image_buffer_t> im, std::shared_ptr<object_detect_result_list> re, std::string txt_pa);
    std::shared_ptr<image_buffer_t> img;
    std::shared_ptr<object_detect_result_list> result;
    std::string txt_path;
    ~Output();
};

class RknnPool {
 public:
  RknnPool(const char* model_path, const int thread_num,
           const char* label_path);
  ~RknnPool();
  void Init();
  void DeInit();
  void AddInferenceTask(std::shared_ptr<image_buffer_t> src, std::string txt_path);
  int get_model_id();
  std::shared_ptr<Output> GetImageResultFromQueue();
  int GetTasksSize();

 private:
  int thread_num_{1};
  const char* model_path_{"null"};
  const char* label_path_{"null"};
  uint32_t id{0};
  std::unique_ptr<ThreadPool> pool_;
  std::queue<std::shared_ptr<Output>> image_results_;
  std::vector<std::shared_ptr<rknn_app_context_t>> models_;
  std::mutex id_mutex_;
  std::mutex image_results_mutex_;
};
