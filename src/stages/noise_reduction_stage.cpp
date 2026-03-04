#include "noise_reduction_stage.h"
#include <opencv2/opencv.hpp>
#include <cstring>

struct NoiseReductionStage::Impl {};

NoiseReductionStage::NoiseReductionStage() : impl_(std::make_unique<Impl>()) {}

NoiseReductionStage::~NoiseReductionStage() = default;

void NoiseReductionStage::process(Frame& frame) {
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    cv::Mat src(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat dst(frame.height, frame.width, CV_8UC3);
    cv::bilateralFilter(src, dst, 9, 75, 75);
    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, size);
    }
}

const char* NoiseReductionStage::name() const {
    return "NoiseReduction";
}
