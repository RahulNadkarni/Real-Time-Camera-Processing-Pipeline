#include "renderer.h"
#include "../profiling/pipeline_stats.h"
#include "../controls/stage_controller.h"

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
    (void)title;
    // TODO: store title in impl_; optionally update OpenCV window title
}

int Renderer::render(const Frame& frame,
                     const PipelineStats* stats,
                     const StageController* controller) {
    (void)frame;
    (void)stats;
    (void)controller;
    // TODO: convert frame.buffer to cv::Mat; draw overlay (active stages + per-stage latency from stats); imshow; return waitKey(1)
    return -1;
}

bool Renderer::is_open() const {
    return impl_->open.load(std::memory_order_relaxed);
}

void Renderer::close() {
    // TODO: set impl_->open = false; destroy OpenCV window if created
}
