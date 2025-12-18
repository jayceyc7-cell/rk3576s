//
// Created by kaylor on 3/6/24.
//

#include "rknn_pool.h"
#include "postprocess.h"

RknnPool::RknnPool(const char* model_path, const int thread_num,
                   const char* lable_path) {
  this->thread_num_ = thread_num;
  this->model_path_ = model_path;
  this->label_path_ = lable_path;
  this->Init();
}

RknnPool::~RknnPool() { this->DeInit(); }

void RknnPool::Init() {
  try {
    // 配置线程池
    this->pool_ = std::make_unique<ThreadPool>(this->thread_num_);
    // 这里每一个线程需要加载一个模型
    for (int i = 0; i < this->thread_num_; ++i) {
        auto ctx = std::make_shared<rknn_app_context_t>(rknn_app_context_t{});
        models_.push_back(ctx);
    }
  } catch (const std::bad_alloc &e) {
    printf("Out of memory: {}", e.what());
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < this->thread_num_; ++i) {
    auto ret = init_yolov8_model(this->model_path_, this->models_[i].get());
    if (ret != 0) {
      printf("Init rknn model failed!");
      exit(EXIT_FAILURE);
    }
  }
}

void RknnPool::DeInit() { deinit_post_process(); }

void RknnPool::AddInferenceTask(std::shared_ptr<image_buffer_t> src, std::string txt_path) {
  pool_->enqueue(
      [&](std::shared_ptr<image_buffer_t> original_img) {
        auto od_results = std::make_shared<object_detect_result_list>();
        auto mode_id = get_model_id();
        int ret = inference_yolov8_model(
            this->models_[mode_id].get(), 
            original_img.get(), 
            od_results.get()
        );
        if (ret != 0){
            printf("init_yolov8_model fail! ret=%d\n", ret);
        }

        auto output = std::make_shared<Output>(
            original_img,
            od_results,
            txt_path
        );
        std::lock_guard<std::mutex> lock_guard(this->image_results_mutex_);
        this->image_results_.push(output);
      },
      std::move(src));
}

int RknnPool::get_model_id() {
  std::lock_guard<std::mutex> lock(id_mutex_);
  int mode_id = id;
  id++;
  if (id == thread_num_) {
    id = 0;
  }
  //  KAYLORDUT_LOG_INFO("id = {}, num = {}, mode id = {}", id, thread_num_,
  //  mode_id);
  return mode_id;
}

std::shared_ptr<Output> RknnPool::GetImageResultFromQueue() {
  std::lock_guard<std::mutex> lock_guard(this->image_results_mutex_);
  if (this->image_results_.empty()) {
    return nullptr;
  } else {
    auto res = this->image_results_.front();
    this->image_results_.pop();
    return std::move(res);
  }
}

int RknnPool::GetTasksSize() { return pool_->TasksSize(); }

Output::Output(std::shared_ptr<image_buffer_t> im, std::shared_ptr<object_detect_result_list> re, std::string txt_pa){
    this->img = im;
    this->result = re;
    this->txt_path = txt_pa;
}
