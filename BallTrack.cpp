#include "BallTrack.h"
#include <Eigen/Core>

using namespace byte_kalman;

BallTrack::BallTrack(const std::vector<float>& tlwh_init)
{
    tlwh = tlwh_init;
}

BallTrack::~BallTrack() {}

// 将 [x, y, w, h] 转换为 [cx, cy, a, h]
static DETECTBOX tlwh_to_xyah(const std::vector<float>& tlwh)
{
    DETECTBOX xyah;
    xyah(0) = tlwh[0] + tlwh[2] / 2.0f; // center x
    xyah(1) = tlwh[1] + tlwh[3] / 2.0f; // center y
    xyah(2) = tlwh[2] / tlwh[3];        // aspect ratio
    xyah(3) = tlwh[3];                  // height
    return xyah;
}

// 将 [cx, cy, a, h] 转换回 [x, y, w, h]
static std::vector<float> xyah_to_tlwh(const KAL_MEAN& mean)
{
    std::vector<float> tlwh(4);
    float cx = mean(0);
    float cy = mean(1);
    float a = mean(2);
    float h = mean(3);

    float w = a * h;
    tlwh[0] = cx - w / 2.0f;
    tlwh[1] = cy - h / 2.0f;
    tlwh[2] = w;
    tlwh[3] = h;
    return tlwh;
}

void BallTrack::init(KalmanFilter& kf)
{
    DETECTBOX measurement = tlwh_to_xyah(tlwh);
    auto mc = kf.initiate(measurement);
    mean = mc.first;
    covariance = mc.second;
    is_initialized = true;
}

void BallTrack::predict(KalmanFilter& kf)
{
    if (!is_initialized) return;
    kf.predict(mean, covariance);
}

void BallTrack::update(KalmanFilter& kf, const std::vector<float>& new_tlwh)
{
    if (!is_initialized) return;
    DETECTBOX measurement = tlwh_to_xyah(new_tlwh);
    auto mc = kf.update(mean, covariance, measurement);
    mean = mc.first;
    covariance = mc.second;
}

std::vector<float> BallTrack::get_tlwh() const
{
    if (!is_initialized)
        return tlwh; // 若未初始化，则返回初始框
    return xyah_to_tlwh(mean);
}

// BallTrack.cpp (新增实现)
const KAL_MEAN& BallTrack::get_mean() const {
    return mean;
}

const KAL_COVA& BallTrack::get_covariance() const {
    return covariance;
}
