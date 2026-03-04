#include "debayer_stage.h"
#include <opencv2/opencv.hpp>
#include <cstring>

struct DebayerStage::Impl {};

DebayerStage::DebayerStage() : impl_(std::make_unique<Impl>()) {}

DebayerStage::~DebayerStage() = default;

void DebayerStage::process(Frame& frame) {
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    if (frame.channels == 3) {
        return;
    }
    cv::Mat src(frame.height, frame.width, CV_8UC1, frame.buffer.data());
    cv::Mat dst(frame.height, frame.width, CV_8UC3);
    cv::cvtColor(src, dst, cv::COLOR_BayerBG2BGR);
    frame.buffer.resize(static_cast<size_t>(frame.width) * frame.height * 3);
    frame.channels = 3;
    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, frame.buffer.size());
    }
}

const char* DebayerStage::name() const {
    return "Debayer";
}
