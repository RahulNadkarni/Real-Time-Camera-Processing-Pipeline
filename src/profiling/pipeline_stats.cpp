#include "pipeline_stats.h"
#include <chrono>

PipelineStats::PipelineStats() {
    fps_window_start_ = std::chrono::steady_clock::now();
}

PipelineStats::PipelineStats(size_t num_stages) : num_stages_(num_stages) {
    stage_rings_.resize(num_stages);
    fps_window_start_ = std::chrono::steady_clock::now();
}

void PipelineStats::set_num_stages(size_t num_stages) {
    num_stages_ = num_stages;
    stage_rings_.resize(num_stages);
}

void PipelineStats::record_stage_latency_us(size_t stage_index, int64_t latency_us) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (stage_index < stage_rings_.size()) {
        stage_rings_[stage_index].push(latency_us);
    }
}

void PipelineStats::record_e2e_latency_us(int64_t latency_us) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    e2e_ring_.push(latency_us);
}

void PipelineStats::record_drop() {
    drops_.fetch_add(1, std::memory_order_relaxed);
}

void PipelineStats::record_frame_displayed() {
    frames_displayed_.fetch_add(1, std::memory_order_relaxed);
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(fps_mutex_);
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fps_window_start_).count();
    if (elapsed >= 1) {
        fps_window_start_ = now;
        fps_.store(static_cast<double>(frames_displayed_.load(std::memory_order_relaxed)));
        frames_displayed_.store(0, std::memory_order_relaxed);
    }
}

int64_t PipelineStats::get_mean_latency_us(size_t stage_index) const {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (stage_index >= stage_rings_.size()) return 0;
    return stage_rings_[stage_index].mean();
}

int64_t PipelineStats::get_p99_latency_us(size_t stage_index) const {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (stage_index >= stage_rings_.size()) return 0;
    return stage_rings_[stage_index].p99();
}

int64_t PipelineStats::get_mean_e2e_us() const {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return e2e_ring_.mean();
}

int64_t PipelineStats::get_p99_e2e_us() const {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return e2e_ring_.p99();
}

uint64_t PipelineStats::get_drop_count() const {
    return drops_.load(std::memory_order_relaxed);
}

double PipelineStats::get_fps() const {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(fps_mutex_);
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fps_window_start_).count();
    if (elapsed >= 1) {
        fps_window_start_ = now;
        fps_.store(static_cast<double>(frames_displayed_.load(std::memory_order_relaxed)));
        frames_displayed_.store(0, std::memory_order_relaxed);
    }
    return fps_.load(std::memory_order_relaxed);
}

void PipelineStats::reset() {
    drops_.store(0, std::memory_order_relaxed);
    frames_displayed_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(fps_mutex_);
        fps_.store(0.0, std::memory_order_relaxed);
        fps_window_start_ = std::chrono::steady_clock::now();
    }
    std::lock_guard<std::mutex> lock(latency_mutex_);
    for (auto& ring : stage_rings_) ring = Ring{};
    e2e_ring_ = Ring{};
}
