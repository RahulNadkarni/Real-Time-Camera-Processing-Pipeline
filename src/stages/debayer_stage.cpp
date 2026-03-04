#include "debayer_stage.h"
#include "../profiling/scoped_timer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <vector>

struct DebayerStage::Impl {};

DebayerStage::DebayerStage() : impl_(std::make_unique<Impl>()) {}

DebayerStage::~DebayerStage() = default;

void DebayerStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    ScopedTimer timer(name());

    if (frame.channels == 3) {
        // Webcam is already BGR; run demosaic path anyway so timing reflects real cost.
        // Synthetic Bayer from green channel -> demosaic -> BGR (shows ~8–15ms at full res).
        cv::Mat bgr(frame.height, frame.width, CV_8UC3, frame.buffer.data());
        std::vector<cv::Mat> planes(3);
        cv::split(bgr, planes);
        cv::Mat bayer = planes[1];  // green as fake Bayer plane
        cv::Mat dst;
        cv::cvtColor(bayer, dst, cv::COLOR_BayerBG2BGR);
        if (dst.isContinuous() && dst.data) {
            std::memcpy(frame.buffer.data(), dst.data, size);
        }
    } else {
        cv::Mat src(frame.height, frame.width, CV_8UC1, frame.buffer.data());
        cv::Mat dst(frame.height, frame.width, CV_8UC3);
        cv::cvtColor(src, dst, cv::COLOR_BayerBG2BGR);
        frame.buffer.resize(static_cast<size_t>(frame.width) * frame.height * 3);
        frame.channels = 3;
        if (dst.isContinuous() && dst.data) {
            std::memcpy(frame.buffer.data(), dst.data, frame.buffer.size());
        }
    }

    if (out_latency_us) *out_latency_us = timer.elapsed_us();
}

const char* DebayerStage::name() const {
    return "Debayer";
}
