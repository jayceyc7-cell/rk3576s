#pragma once
#include "BallTrack.h"
#include "awi_track.hpp"

// class BallTrackAdapter : public ITracker {
// public:
//     explicit BallTrackAdapter(const std::vector<float>& tlwh)
//         : impl_(tlwh) {}

//     void Init(byte_kalman::KalmanFilter& kf) override {
//         impl_.init(kf);
//     }

//     void Predict(byte_kalman::KalmanFilter& kf) override {
//         impl_.predict(kf);
//     }

//     void Update(byte_kalman::KalmanFilter& kf,
//                 const std::vector<float>& tlwh) override {
//         impl_.update(kf, tlwh);
//     }

//     std::vector<float> GetTlwh() const override {
//         return impl_.get_tlwh();
//     }

//     const KAL_MEAN& GetMean() const override {
//         return impl_.get_mean();
//     }

//     const KAL_COVA& GetCovariance() const override {
//         return impl_.get_covariance();
//     }

// private:
//     BallTrack impl_;
// };
