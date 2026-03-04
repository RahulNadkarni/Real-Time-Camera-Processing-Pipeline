#include "noise_reduction_stage.h"
#include "../profiling/scoped_timer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>

struct NoiseReductionStage::Impl {};

NoiseReductionStage::NoiseReductionStage() : impl_(std::make_unique<Impl>()) {}

NoiseReductionStage::~NoiseReductionStage() = default;

void NoiseReductionStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    ScopedTimer timer(name());
    cv::Mat src(frame.height, frame.width, CV_8UC3, frame.buffer.data());

    const double scale = 0.5;
    cv::Mat half;
    cv::resize(src, half, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::Mat half_filtered;
    cv::bilateralFilter(half, half_filtered, 5, 50.0, 50.0);
    cv::Mat dst;
    cv::resize(half_filtered, dst, src.size(), 0, 0, cv::INTER_LINEAR);

    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, size);
    }
    if (out_latency_us) *out_latency_us = timer.elapsed_us();
}

const char* NoiseReductionStage::name() const {
    return "NoiseReduction";
}
