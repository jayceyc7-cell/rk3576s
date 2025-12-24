#pragma once
#include <stdint.h>
#include <vector>
#include "kalmanFilter.h"
#include <memory>
#include "common.h"


//yolo检测结果结构体
struct T_DetectObject {
    int cls_id;        // 类别ID（篮球、足球等）
    float score;       // 置信度
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};
// 轨迹预测结果结构体
struct T_TrackObject {
    int track_id;      // 轨迹ID（全局唯一）
    int cls_id;        // 类别
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    bool is_predicted; // 是否为预测框（非当前检测）
};

// 跟踪器接口
struct BBox {
    float x;
    float y;
    float w;
    float h;
};

class ITracker {
public:
    virtual ~ITracker() = default;

    virtual void Init() = 0;
    virtual void Predict() = 0;
    virtual void Update(const BBox& bbox) = 0;

    virtual BBox GetBBox() const = 0;
    virtual bool IsConfirmed() const = 0;
};
// 跟踪状态结构体
struct TrackState
{
    int track_id;
    int cls_id;

    float xmin;
    float ymin;
    float xmax;
    float ymax;

    int lost_frames;
    uint64_t last_frame;

    std::unique_ptr<ITracker> tracker;
};

class TrackFrame {
public:
    TrackFrame();
    ~TrackFrame();

    // ========== 生命周期 ==========
    bool Init(int max_track_num = 50);
    void Reset();
    void DeInit();

    // ========== 核心接口（每帧调用） ==========
void ProcessFrame(
    uint64_t frame_id,
    image_buffer_t& src_image,
    const std::vector<T_DetectObject>& detections,
    std::vector<T_TrackObject>& track_results
);

private:
    // ========== 内部方法（算法相关，后续实现） ==========
    void PredictTracks();
    void AssociateDetections(
        const std::vector<T_DetectObject>& detections
    );
    void UpdateTracks();
    void CreateNewTracks(
        const std::vector<T_DetectObject>& unmatched_dets
    );
    void RemoveLostTracks();

private:
    // ========== 成员变量 ==========
    int max_track_num_;
    int next_track_id_;

    uint64_t current_frame_id_;

    byte_kalman::KalmanFilter g_kf;   // 你已有的 KF

    std::vector<TrackState> tracks_;
    std::vector<T_TrackObject> output_tracks_;
};


