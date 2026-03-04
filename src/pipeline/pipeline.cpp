#include "pipeline.h"

Pipeline::Pipeline(const Config& config) {
    // TODO: construct FramePool from config (resolution, queue max size or capacity); create queues (N+1 for N stages); instantiate each stage and stats/controller; store capture and renderer handles if needed
}

Pipeline::~Pipeline() {
    stop();
    // TODO: release capture_handle_ (e.g., delete VideoCapture); renderer_ is unique_ptr, auto-released
}

void Pipeline::start() {
    // TODO: launch capture_loop in one thread, run_stage(i) for each stage in separate threads, display_loop in one thread; store thread handles in threads_
}

void Pipeline::stop() {
    // TODO: set shutdown_requested_ = true; call shutdown() on all queues; join all threads in threads_
}

bool Pipeline::is_shutdown_requested() const {
    return shutdown_requested_.load(std::memory_order_relaxed);
}

PipelineStats& Pipeline::stats() {
    return *stats_;
}

StageController& Pipeline::controller() {
    return *controller_;
}

void Pipeline::capture_loop() {
    // TODO: while !is_shutdown_requested(): acquire frame from pool_; read from webcam into frame buffer; set timestamp and frame_id; push frame to queues_[0]; respect target fps / drop if queue full
}

void Pipeline::run_stage(size_t stage_index) {
    // TODO: while (auto frame = queues_[stage_index]->pop()): if frame is nullopt break; if stages_[stage_index] is enabled (via controller_), call process(*frame) and push to queues_[stage_index+1]; else push frame unchanged to next queue; release or pass ownership as per design
}

void Pipeline::display_loop() {
    // TODO: while (auto frame = queues_.back()->pop()): if nullopt break; render frame via renderer_->render(); release frame to pool_; update stats (fps, latency) if needed
}
