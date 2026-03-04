#include "tone_mapping_stage.h"
#include <opencv2/opencv.hpp>
#include <cstring>

struct ToneMappingStage::Impl {};

ToneMappingStage::ToneMappingStage() : impl_(std::make_unique<Impl>()) {}

ToneMappingStage::~ToneMappingStage() = default;

void ToneMappingStage::process(Frame& frame) {
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    cv::Mat src(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat dst(frame.height, frame.width, CV_8UC3);
    cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX);
    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, size);
    }
}

const char* ToneMappingStage::name() const {
    return "ToneMapping";
}
