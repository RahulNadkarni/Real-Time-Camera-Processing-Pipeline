#include "edge_detection_stage.h"
#include "../profiling/scoped_timer.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>

struct EdgeDetectionStage::Impl {};

EdgeDetectionStage::EdgeDetectionStage() : impl_(std::make_unique<Impl>()) {}

EdgeDetectionStage::~EdgeDetectionStage() = default;

void EdgeDetectionStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    const size_t size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < size) return;

    ScopedTimer timer(name());
    cv::Mat src(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat gray, edges, dst;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200);
    cv::cvtColor(edges, dst, cv::COLOR_GRAY2BGR);
    if (dst.isContinuous() && dst.data) {
        std::memcpy(frame.buffer.data(), dst.data, size);
    }
    if (out_latency_us) *out_latency_us = timer.elapsed_us();
}

const char* EdgeDetectionStage::name() const {
    return "EdgeDetection";
}
