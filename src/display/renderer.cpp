#include "renderer.h"
#include "../profiling/pipeline_stats.h"
#include "../controls/stage_controller.h"
#include <opencv2/opencv.hpp>
#include <sstream>

struct Renderer::Impl {
    int width = 0;
    int height = 0;
    std::string window_title = "Camera Pipeline";
    std::atomic<bool> open{true};
};

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}

Renderer::Renderer(int width, int height) : impl_(std::make_unique<Impl>()) {
    impl_->width = width;
    impl_->height = height;
}

Renderer::~Renderer() {
    close();
}

void Renderer::set_window_title(const std::string& title) {
    impl_->window_title = title;
}

int Renderer::render(const Frame& frame,
                     const PipelineStats* stats,
                     const StageController* controller) {
    const size_t expected_size = static_cast<size_t>(frame.width) * frame.height * frame.channels;
    if (frame.buffer.size() < expected_size) {
        return -1;
    }
    cv::Mat src(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.buffer.data()));
    cv::Mat view = src.clone();
    std::stringstream ss;
    if (stats) {
        ss << impl_->window_title << " (FPS: " << stats->get_fps() << ")";
    } else {
        ss << impl_->window_title;
    }
    const cv::Scalar text_color(200, 200, 200);  // light grey (BGR)
    cv::putText(view, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
    if (controller && stats) {
        for (size_t i = 0; i < StageController::kNumStages; i++) {
            ss.str("");
            ss.clear();
            ss << StageController::stage_name(i) << ": "
               << (controller->is_enabled(i) ? "ON " : "OFF ");
            ss << stats->get_avg_latency_us(i) << " us";
            cv::putText(view, ss.str(), cv::Point(10, 60 + static_cast<int>(i + 1) * 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        }
    }
    cv::imshow(impl_->window_title, view);
    return cv::waitKey(1);
}

bool Renderer::is_open() const {
    return impl_->open.load(std::memory_order_relaxed);
}

void Renderer::close() {
    impl_->open.store(false, std::memory_order_relaxed);
    cv::destroyWindow(impl_->window_title);
}
