#include "pipeline_stats.h"
#include <chrono>
#include <algorithm>

PipelineStats::PipelineStats() {
    fps_window_start_ = std::chrono::steady_clock::now();
}

PipelineStats::PipelineStats(size_t num_stages) : num_stages_(num_stages) {
    avg_latency_us_.resize(num_stages, 0);
    fps_window_start_ = std::chrono::steady_clock::now();
}

void PipelineStats::set_num_stages(size_t num_stages) {
    num_stages_ = num_stages;
    avg_latency_us_.resize(num_stages, 0);
}

void PipelineStats::record_stage_latency_us(size_t stage_index, int64_t latency_us) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (stage_index < avg_latency_us_.size()) {
        avg_latency_us_[stage_index] = latency_us;
    }
}

void PipelineStats::record_drop() {
    drops_.fetch_add(1, std::memory_order_relaxed);
}

void PipelineStats::record_frame_displayed() {
    frames_displayed_.fetch_add(1, std::memory_order_relaxed);
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(fps_mutex_);
    auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - fps_window_start_).count();
    if (elapsed_sec >= 1) {
        fps_window_start_ = now;
        fps_.store(static_cast<double>(frames_displayed_.load(std::memory_order_relaxed)));
        frames_displayed_.store(0, std::memory_order_relaxed);
    }
}

int64_t PipelineStats::get_avg_latency_us(size_t stage_index) const {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (stage_index >= avg_latency_us_.size()) {
        return 0;
    }
    return avg_latency_us_[stage_index];
}

uint64_t PipelineStats::get_drop_count() const {
    return drops_.load(std::memory_order_relaxed);
}

double PipelineStats::get_fps() const {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(fps_mutex_);
        auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - fps_window_start_).count();
        if (elapsed_sec >= 1) {
            fps_window_start_ = now;
            fps_.store(static_cast<double>(frames_displayed_.load(std::memory_order_relaxed)));
            frames_displayed_.store(0, std::memory_order_relaxed);
        }
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
    std::fill(avg_latency_us_.begin(), avg_latency_us_.end(), 0);
}
