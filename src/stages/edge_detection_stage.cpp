#include "edge_detection_stage.h"
#include <opencv2/opencv.hpp>
#include <cstring>

struct EdgeDetectionStage::Impl {};

EdgeDetectionStage::EdgeDetectionStage() : impl_(std::make_unique<Impl>()) {}

EdgeDetectionStage::~EdgeDetectionStage() = default;

void EdgeDetectionStage::process(Frame& frame) {
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    cv::Mat src(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat gray, edges, dst;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200);
    cv::cvtColor(edges, dst, cv::COLOR_GRAY2BGR);
    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, size);
    }
}

const char* EdgeDetectionStage::name() const {
    return "EdgeDetection";
}
