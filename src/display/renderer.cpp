#include "renderer.h"
#include "../profiling/pipeline_stats.h"
#include "../controls/stage_controller.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <atomic>
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
    if (frame.buffer.size() < expected_size) return -1;

    cv::Mat src(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.buffer.data()));
    cv::Mat view = src.clone();

    const cv::Scalar text_color(200, 200, 200);
    const double font_scale = 0.45;
    const int thickness = 1;

    std::stringstream ss;
    if (stats) {
        ss << impl_->window_title
           << "  FPS: " << static_cast<int>(stats->get_fps())
           << "  E2E mean: " << stats->get_mean_e2e_us() << "us"
           << "  p99: " << stats->get_p99_e2e_us() << "us"
           << "  drops: " << stats->get_drop_count();
    } else {
        ss << impl_->window_title;
    }
    cv::putText(view, ss.str(), cv::Point(10, 18), cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);

    if (controller && stats) {
        for (size_t i = 0; i < StageController::kNumStages; i++) {
            ss.str("");
            ss.clear();
            ss << StageController::stage_name(i)
               << ": " << (controller->is_enabled(i) ? "ON " : "OFF")
               << "  mean: " << stats->get_mean_latency_us(i) << "us"
               << "  p99: " << stats->get_p99_latency_us(i) << "us";
            cv::putText(view, ss.str(), cv::Point(10, 36 + static_cast<int>(i) * 18),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
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
