#include "tone_mapping_stage.h"
#include "../profiling/scoped_timer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <vector>

struct ToneMappingStage::Impl {};

ToneMappingStage::ToneMappingStage() : impl_(std::make_unique<Impl>()) {}
ToneMappingStage::~ToneMappingStage() = default;

void ToneMappingStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    ScopedTimer timer(name());
    cv::Mat src(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.6, cv::Size(8, 8));
    cv::Mat l_enhanced;
    clahe->apply(lab_channels[0], l_enhanced);
    lab_channels[0] = l_enhanced;

    cv::merge(lab_channels, lab);
    cv::Mat locally_toned;
    cv::cvtColor(lab, locally_toned, cv::COLOR_Lab2BGR);

    cv::Mat dst(frame.height, frame.width, CV_8UC3);
    cv::addWeighted(src, 0.6, locally_toned, 0.4, 0.0, dst);
    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, size);
    }
    if (out_latency_us) *out_latency_us = timer.elapsed_us();
}

const char* ToneMappingStage::name() const {
    return "ToneMapping";
}
