#include "awi_track.hpp"
#include <iostream>
#include "BallTrack.h"

TrackFrame::TrackFrame()
    : max_track_num_(50),
      next_track_id_(0),
      current_frame_id_(0),
      ball_tracker_(nullptr),
      miss_count_(0),
      last_gating_distance_(0.0f),
      max_miss_count_(2),        // 可配置：最大丢失帧数
      gate_threshold_(200.0f)    // 可配置：马氏距离阈值
{
    std::cout << "TrackFrame created" << std::endl;
}

TrackFrame::~TrackFrame()
{
    DeInit();
    std::cout << "TrackFrame destroyed" << std::endl;
}

bool TrackFrame::Init(int max_track_num)
{
    max_track_num_ = max_track_num;
    next_track_id_ = 0;
    current_frame_id_ = 0;
    miss_count_ = 0;
    last_gating_distance_ = 0.0f;
    
    // 清理旧的跟踪器
    if (ball_tracker_ != nullptr) {
        delete ball_tracker_;
        ball_tracker_ = nullptr;
    }
    
    return true;
}

void TrackFrame::Reset()
{
    if (ball_tracker_ != nullptr) {
        delete ball_tracker_;
        ball_tracker_ = nullptr;
    }
    
    next_track_id_ = 0;
    current_frame_id_ = 0;
    miss_count_ = 0;
    last_gating_distance_ = 0.0f;
}

void TrackFrame::DeInit()
{
    if (ball_tracker_ != nullptr) {
        delete ball_tracker_;
        ball_tracker_ = nullptr;
    }
    
    max_track_num_ = 0;
    miss_count_ = 0;
}

void TrackFrame::ProcessFrame(
    uint64_t frame_id,
    const std::vector<T_DetectObject>& detections,
    std::vector<T_TrackObject>& track_results)
{
    current_frame_id_ = frame_id;
    track_results.clear();
    last_gating_distance_ = -1.0f;

    printf("[TrackFrame] frame_id=%lu, det_num=%zu, miss_count=%d\n",
           frame_id, detections.size(), miss_count_);

    // =====================================================
    // 1. 筛选 cls_id == 0（篮球），选择最佳候选
    // =====================================================
    const T_DetectObject* ball_det = nullptr;
    
    if (ball_tracker_ == nullptr) {
        // 没有跟踪器时，选置信度最高的球
        float best_score = 0.0f;
        for (const auto& det : detections) {
            if (det.cls_id == 0 && det.score > best_score) {
                best_score = det.score;
                ball_det = &det;
            }
        }
    } else {
        // 有跟踪器时，选马氏距离最小的球
        float best_dist = 1e9f;
        for (const auto& det : detections) {
            if (det.cls_id != 0) continue;
            
            float w = det.xmax - det.xmin;
            float h = det.ymax - det.ymin;
            float cx = det.xmin + w * 0.5f;
            float cy = det.ymin + h * 0.5f;
            float a = w / h;
            
            std::vector<float> xyah = {cx, cy, a, h};
            std::vector<DETECTBOX> dets = {
                Eigen::Map<DETECTBOX>(xyah.data())
            };
            
            auto maha = kf_.gating_distance(
                ball_tracker_->get_mean(),
                ball_tracker_->get_covariance(),
                dets,
                false
            );
            
            float dist = maha(0);
            if (dist < best_dist) {
                best_dist = dist;
                ball_det = &det;
                last_gating_distance_ = dist;  // 记录最佳距离
            }
        }
        
        // 如果最佳距离也超过阈值，说明没有合适的匹配
        if (best_dist > gate_threshold_) {
            printf("[TrackFrame] Best distance %.2f > threshold %.2f, treating as no detection\n",
                   best_dist, gate_threshold_);
            ball_det = nullptr;  // 当作没有检测
        }
    }

    // =====================================================
    // 2. 若本帧没有合适的篮球检测
    // =====================================================
    if (ball_det == nullptr) {
        printf("[TrackFrame] No suitable ball detected this frame\n");

        if (ball_tracker_ != nullptr) {
            miss_count_++;

            ball_tracker_->predict(kf_);
            auto pred = ball_tracker_->get_tlwh();

            T_TrackObject track;
            track.track_id = 0;
            track.cls_id = 0;
            track.xmin = pred[0];
            track.ymin = pred[1];
            track.xmax = pred[0] + pred[2];
            track.ymax = pred[1] + pred[3];
            track.is_predicted = true;
            track_results.push_back(track);

            printf("[TrackFrame] Predict only: (%.1f, %.1f, %.1f, %.1f), miss=%d/%d\n",
                   pred[0], pred[1], pred[2], pred[3], miss_count_, max_miss_count_);

            if (miss_count_ >= max_miss_count_) {
                printf("[TrackFrame] Reset tracker (max miss reached)\n");
                delete ball_tracker_;
                ball_tracker_ = nullptr;
                miss_count_ = 0;
            }
        }
        return;
    }

    // =====================================================
    // 3. 有合适的检测，提取检测框
    // =====================================================
    const auto& det = *ball_det;
    float xmin = det.xmin;
    float ymin = det.ymin;
    float xmax = det.xmax;
    float ymax = det.ymax;
    float w = xmax - xmin;
    float h = ymax - ymin;

    std::vector<float> tlwh = {xmin, ymin, w, h};

    printf("[TrackFrame] Selected detection: (%.1f, %.1f, %.1f, %.1f) score=%.2f\n", 
           xmin, ymin, w, h, det.score);

    // =====================================================
    // 4. 初始化跟踪器
    // =====================================================
    if (ball_tracker_ == nullptr) {
        ball_tracker_ = new BallTrack(tlwh);
        ball_tracker_->init(kf_);
        miss_count_ = 0;
        last_gating_distance_ = 0.0f;
        
        printf("[TrackFrame] Tracker initialized\n");
        
        T_TrackObject track;
        track.track_id = 0;
        track.cls_id = 0;
        track.xmin = xmin;
        track.ymin = ymin;
        track.xmax = xmax;
        track.ymax = ymax;
        track.is_predicted = false;
        track_results.push_back(track);
        return;
    }

    // =====================================================
    // 5. 更新（已经在上面选择时计算过距离，且通过了阈值检查）
    // =====================================================
    ball_tracker_->update(kf_, tlwh);
    miss_count_ = 0;
    printf("[TrackFrame] Update: MATCHED (dist=%.2f)\n", last_gating_distance_);

    // =====================================================
    // 6. 预测
    // =====================================================
    ball_tracker_->predict(kf_);

    auto pred = ball_tracker_->get_tlwh();
    
    T_TrackObject track;
    track.track_id = 0;
    track.cls_id = 0;
    track.xmin = pred[0];
    track.ymin = pred[1];
    track.xmax = pred[0] + pred[2];
    track.ymax = pred[1] + pred[3];
    track.is_predicted = false;
    track_results.push_back(track);

    printf("[TrackFrame] Output: (%.1f, %.1f, %.1f, %.1f)\n",
           pred[0], pred[1], pred[2], pred[3]);
}