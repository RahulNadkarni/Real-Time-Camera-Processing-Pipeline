#include "tone_mapping_stage.h"
#include "../profiling/scoped_timer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>

struct ToneMappingStage::Impl {};

ToneMappingStage::ToneMappingStage() : impl_(std::make_unique<Impl>()) {}

ToneMappingStage::~ToneMappingStage() = default;

void ToneMappingStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    ScopedTimer timer(name());
    cv::Mat src(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat dst(frame.height, frame.width, CV_8UC3);
    cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX);
    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, size);
    }
    if (out_latency_us) *out_latency_us = timer.elapsed_us();
}

const char* ToneMappingStage::name() const {
    return "ToneMapping";
}
