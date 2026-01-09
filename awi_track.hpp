#pragma once
#include <stdint.h>
#include <vector>
#include "kalmanFilter.h"
#include <memory>
#include "common.h"

// yolo检测结果结构体
struct T_DetectObject {
    int cls_id;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

// 轨迹预测结果结构体
struct T_TrackObject {
    int track_id;
    int cls_id;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    bool is_predicted;  // 是否为纯预测框（无检测匹配）
};

// 前向声明
class BallTrack;

class TrackFrame {
public:
    TrackFrame();
    ~TrackFrame();

    // ========== 生命周期 ==========
    bool Init(int max_track_num = 50);
    void Reset();
    void DeInit();

    // ========== 核心接口（每帧调用）==========
    // 注意：不再传入 image，跟踪器不负责绘图
    void ProcessFrame(
        uint64_t frame_id,
        const std::vector<T_DetectObject>& detections,
        std::vector<T_TrackObject>& track_results
    );

    // ========== 状态查询 ==========
    bool HasActiveTrack() const { return ball_tracker_ != nullptr; }
    int GetMissCount() const { return miss_count_; }
    float GetLastGatingDistance() const { return last_gating_distance_; }

private:
    // ========== 成员变量 ==========
    int max_track_num_;
    int next_track_id_;
    uint64_t current_frame_id_;

    // 卡尔曼滤波器（类成员，非全局）
    byte_kalman::KalmanFilter kf_;
    
    // 球跟踪器（类成员，非全局）
    BallTrack* ball_tracker_;
    int miss_count_;
    
    // 调试信息
    float last_gating_distance_;

    // ========== 参数配置 ==========
    int max_miss_count_;        // 最大丢失帧数
    float gate_threshold_;      // 马氏距离阈值
};