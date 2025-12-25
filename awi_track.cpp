#include "awi_track.hpp"
#include <iostream>
#include "queue"
#include "threadpool.h"
#include "BallTrack.h"
#include "common.h"
#include "image_drawing.h"


// ======================= 参数配置 =======================
static constexpr int   MAX_MISS_COUNT = 2;
static const float GATE_THRESHOLD = 200.0f;
BallTrack *g_ball_ = nullptr; // 使用指针保持全局状态（多帧间保持滤波状态）
int miss_count_ = 0;             // 连续未匹配帧数

TrackFrame::TrackFrame() {
    std::cout << "TrackFrame!!!" << std::endl;
}

TrackFrame::~TrackFrame()
{
    std::cout << "~TrackFrame!!!" << std::endl;
    DeInit();
}

// ======================= Init / Reset / DeInit =======================
bool TrackFrame::Init(int max_track_num)
{
    max_track_num_   = max_track_num;
    next_track_id_   = 0;
    current_frame_id_ = 0;

    tracks_.clear();
    output_tracks_.clear();

    return true;
}

void TrackFrame::Reset()
{
    tracks_.clear();
    output_tracks_.clear();

    next_track_id_    = 0;
    current_frame_id_ = 0;
}

void TrackFrame::DeInit()
{
    tracks_.clear();
    output_tracks_.clear();
    max_track_num_ = 0;
}


void TrackFrame::ProcessFrame(
    uint64_t frame_id,
    image_buffer_t& src_image,
    const std::vector<T_DetectObject>& detections,
    std::vector<T_TrackObject>& track_results)
{
    printf("[ProcessFrame] frame_id=%lu, det_num=%zu\n",
           frame_id, detections.size());

    track_results.clear();

    // =====================================================
    // 1. 只筛选 cls_id == 0（篮球） 
    // 2025-12-25:在篮球的筛选中，可能检测到多个篮球，此时为每一个篮球都创建一个轨迹，而后对每一个轨迹放入筛选器筛选，筛选器主要作用是筛选出需要跟踪的主篮球轨迹，干扰篮球轨迹需要删除。
    // =====================================================
    const T_DetectObject* ball_det = nullptr;
    for (const auto& det : detections)
    {
        if (det.cls_id == 0)
        {
            ball_det = &det;
            break;   // 单目标，取第一个即可
        }
    }

    // =====================================================
    // 2. 若本帧没有篮球检测
    // =====================================================
    if (ball_det == nullptr)
    {
        printf("[INFO] No cls_id == 0 detected\n");

        if (g_ball_ != nullptr)
        {
            miss_count_++;

            // 仅预测
            g_ball_->predict(g_kf);

            auto pred = g_ball_->get_tlwh();
            draw_rectangle(&src_image,
                           pred[0], pred[1], pred[2], pred[3],
                           COLOR_GREEN, 3);

            printf("[PREDICT ONLY] miss_count=%d\n", miss_count_);

            // 超过最大丢失帧，重置 tracker
            if (miss_count_ >= MAX_MISS_COUNT)
            {
                printf("[RESET] tracker reset (no detection)\n");
                delete g_ball_;
                g_ball_ = nullptr;
                miss_count_ = 0;
            }
        }
        return;
    }

    // =====================================================
    // 3. 取篮球检测框
    // =====================================================
    const auto& det = *ball_det;

    float xmin = det.xmin;
    float ymin = det.ymin;
    float xmax = det.xmax;
    float ymax = det.ymax;

    float w = xmax - xmin;
    float h = ymax - ymin;

    std::vector<float> tlwh = {xmin, ymin, w, h};

    // =====================================================
    // 4. 初始化 tracker（第一帧）
    // =====================================================
    if (g_ball_ == nullptr)
    {
        g_ball_ = new BallTrack(tlwh);
        g_ball_->init(g_kf);
        miss_count_ = 0;

        printf("[INIT] BallTrack initialized (cls_id=0)\n");
        return;
    }

    // =====================================================
    // 5. gating 距离计算（Mahalanobis）
    // =====================================================
    float cx = xmin + w * 0.5f;
    float cy = ymin + h * 0.5f;
    float a  = w / h;

    std::vector<float> xyah = {cx, cy, a, h};
    std::vector<DETECTBOX> dets = {
        Eigen::Map<DETECTBOX>(xyah.data())
    };

    auto maha = g_kf.gating_distance(
        g_ball_->get_mean(),
        g_ball_->get_covariance(),
        dets,
        false
    );

    float dist = maha(0);
    printf("[GATE] Mahalanobis distance = %.2f\n", dist);
    // 构造要显示的文本
    char text_buf[128];
    snprintf(text_buf, sizeof(text_buf),
             "Dist: %.2f  Miss: %d",
             dist, miss_count_);

    // =====================================================
    // 6. update 或 miss
    // =====================================================
    if (dist <= GATE_THRESHOLD)
    {
        g_ball_->update(g_kf, tlwh);
        miss_count_ = 0;
        draw_text(&src_image,
                  text_buf,
                  10, 40,
                  COLOR_YELLOW, 20);

        draw_text(&src_image,
                  "[Update MATCH]",
                  10, 80,
                  COLOR_YELLOW, 20);
    }
    else
    {
        miss_count_++;
        snprintf(text_buf, sizeof(text_buf),
                 "Dist: %.2f  Miss: %d",
                 dist, miss_count_);
        draw_text(&src_image,
                  text_buf,
                  10, 40,
                  COLOR_YELLOW, 20);

        draw_text(&src_image,
                  "[Update MATCH]",
                  10, 80,
                  COLOR_YELLOW, 20);
    }

    // =====================================================
    // 7. predict
    // =====================================================
    g_ball_->predict(g_kf);

    auto pred = g_ball_->get_tlwh();
    float pred_xmin = pred[0];
    float pred_ymin = pred[1];
    float pred_w    = pred[2];
    float pred_h    = pred[3];
    //更新轨迹
    T_TrackObject track;
    track.track_id = 0; //篮球轨迹id，暂时取第一个，多个轨迹时需要修改id号
    track.cls_id = 0; // 篮球类别的id为0
    track.xmin = pred[0];
    track.ymin = pred[1];
    track.xmax = pred[0] + pred[2];
    track.ymax = pred[1] + pred[3];
    track_results.push_back(track);

    draw_rectangle(&src_image,
                   pred_xmin, pred_ymin,
                   pred_w, pred_h,
                   COLOR_GREEN, 3);

    printf("[PREDICT] x=%.2f y=%.2f w=%.2f h=%.2f\n",
           pred_xmin, pred_ymin, pred_w, pred_h);
    
    draw_text(&src_image,
              "BallTrack info:",
              10, 10,
              COLOR_YELLOW, 20);

    // =====================================================
    // 8. 连续丢失 → reset
    // =====================================================
    if (miss_count_ >= MAX_MISS_COUNT)
    {
        printf("[RESET] tracker reset (miss overflow)\n");
        delete g_ball_;
        g_ball_ = nullptr;
        miss_count_ = 0;
    }
}


