#include "histogram_stage.h"
#include "../profiling/scoped_timer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

struct HistogramStage::Impl {};

HistogramStage::HistogramStage() : impl_(std::make_unique<Impl>()) {}
HistogramStage::~HistogramStage() = default;

void HistogramStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    ScopedTimer timer(name());
    cv::Mat view(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat clone = view.clone();

    std::vector<cv::Mat> bgr_planes;
    cv::split(clone, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = range;
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange);

    int hist_w = 256, hist_h = 100;
    cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(b_hist, b_hist, 0, hist_image.rows, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, hist_image.rows, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, hist_image.rows, cv::NORM_MINMAX);

    for (int i = 1; i < histSize; i++) {
        cv::line(hist_image,
                 cv::Point(i - 1, hist_h - cvRound(b_hist.at<float>(i - 1))),
                 cv::Point(i,     hist_h - cvRound(b_hist.at<float>(i))),
                 cv::Scalar(255, 0, 0), 1);
        cv::line(hist_image,
                 cv::Point(i - 1, hist_h - cvRound(g_hist.at<float>(i - 1))),
                 cv::Point(i,     hist_h - cvRound(g_hist.at<float>(i))),
                 cv::Scalar(0, 255, 0), 1);
        cv::line(hist_image,
                 cv::Point(i - 1, hist_h - cvRound(r_hist.at<float>(i - 1))),
                 cv::Point(i,     hist_h - cvRound(r_hist.at<float>(i))),
                 cv::Scalar(0, 0, 255), 1);
    }

    int hist_x = std::max(0, frame.width - hist_w);
    cv::Mat roi = clone(cv::Rect(hist_x, 0, hist_w, hist_h));
    hist_image.copyTo(roi);

    if (clone.isContinuous() && clone.data) {
        std::memcpy(frame.buffer.data(), clone.data, size);
    }
    if (out_latency_us) *out_latency_us = timer.elapsed_us();
}

const char* HistogramStage::name() const {
    return "Histogram";
}
