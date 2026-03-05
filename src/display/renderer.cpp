#include "renderer.h"
#include "../profiling/pipeline_stats.h"
#include "../controls/stage_controller.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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

void Renderer::overlaySceneLabels(Frame& frame, const SceneResult& scene_result) {
    if (!scene_result.valid || scene_result.top_k_labels.empty()) return;
    cv::Mat view(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    const std::string& label = scene_result.top_k_labels[0].first;
    const float confidence = scene_result.top_k_labels[0].second;
    cv::putText(view, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << confidence;
    cv::putText(view, oss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
}

void Renderer::overlaySaliencyMap(Frame& frame, const cv::Mat& saliency_map, double alpha) {
    if (saliency_map.empty()) return;
    cv::Mat view(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    cv::Mat saliency_resized;
    cv::resize(saliency_map, saliency_resized, cv::Size(frame.width, frame.height));
    if (saliency_resized.type() == CV_32FC1) {
        cv::Mat u8;
        saliency_resized.convertTo(u8, CV_8UC1, 255.0, 0.0);
        saliency_resized = u8;
    }
    cv::Mat saliency_colored;
    cv::applyColorMap(saliency_resized, saliency_colored, cv::COLORMAP_JET);
    cv::addWeighted(view, 1.0 - alpha, saliency_colored, alpha, 0.0, view);
}

void Renderer::overlayNeuralMetrics(Frame& frame, float psnr, float ssim, const cv::Point& position) {
    cv::Mat view(frame.height, frame.width, CV_8UC3, frame.buffer.data());
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << "PSNR: " << psnr << " dB, SSIM: " << ssim;
    cv::putText(view, oss.str(), position, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
}
