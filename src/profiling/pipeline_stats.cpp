#include "pipeline_stats.h"
#include <chrono>

PipelineStats::PipelineStats() {
    // TODO: init fps_window_start_ to now(); set_num_stages can be called later
}

PipelineStats::PipelineStats(size_t num_stages) : num_stages_(num_stages) {
    // TODO: resize avg_latency_us_ to num_stages, init to 0; init fps_window_start_
}

void PipelineStats::set_num_stages(size_t num_stages) {
    // TODO: set num_stages_, resize avg_latency_us_
}

void PipelineStats::record_stage_latency_us(size_t stage_index, int64_t latency_us) {
    (void)stage_index;
    (void)latency_us;
    // TODO: update exponential moving average (or sliding window) for this stage; keep lock-free
}

void PipelineStats::record_drop() {
    // TODO: drops_.fetch_add(1, std::memory_order_relaxed)
}

void PipelineStats::record_frame_displayed() {
    // TODO: frames_displayed_.fetch_add(1); update fps window if needed
}

int64_t PipelineStats::get_avg_latency_us(size_t stage_index) const {
    (void)stage_index;
    // TODO: return current average for stage_index (from avg_latency_us_ or internal EMA state)
    return 0;
}

uint64_t PipelineStats::get_drop_count() const {
    // TODO: return drops_.load(std::memory_order_relaxed)
    return 0;
}

double PipelineStats::get_fps() const {
    // TODO: compute fps from frames_displayed_ over last N seconds (sliding window)
    return 0.0;
}

void PipelineStats::reset() {
    // TODO: set drops_ and frames_displayed_ to 0; reset per-stage averages; reset fps window
}
