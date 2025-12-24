#pragma once
#include <vector>
#include "kalmanFilter.h"
#include "dataType.h"

class BallTrack
{
public:
    BallTrack(const std::vector<float>& tlwh_init);
    ~BallTrack();

    // 使用 KalmanFilter 初始化、预测、更新
    void init(byte_kalman::KalmanFilter& kf);
    void predict(byte_kalman::KalmanFilter& kf);
    void update(byte_kalman::KalmanFilter& kf, const std::vector<float>& new_tlwh);

    std::vector<float> get_tlwh() const;  // 返回当前预测框 [x, y, w, h]
    bool isInitialized() const { return is_initialized; }
    
    const KAL_MEAN& get_mean() const;
    const KAL_COVA& get_covariance() const;

private:
    std::vector<float> tlwh;  // 初始输入框 [x, y, w, h]
    KAL_MEAN mean;       // 状态均值
    KAL_COVA covariance; // 协方差矩阵
    bool is_initialized = false;


};
