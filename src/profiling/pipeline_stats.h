#pragma once

#include <chrono>
#include <vector>
#include <atomic>
#include <mutex>
#include <cstdint>

/**
 * Tracks per-stage average latency, frame drop rate, and throughput (fps).
 * Uses std::atomic for drop/frame counts; mutex for per-stage latency arrays.
 * Single responsibility: aggregate and expose pipeline performance metrics.
 */
class PipelineStats {
public:
    PipelineStats();
    explicit PipelineStats(size_t num_stages);

    /**
     * Sets the number of stages (for per-stage latency). Should be called before
     * recording if default constructor was used. Not thread-safe if called while
     * other threads are calling record_*.
     */
    void set_num_stages(size_t num_stages);

    /**
     * Records latency for a stage (in microseconds). Thread-safe (protected by mutex).
     */
    void record_stage_latency_us(size_t stage_index, int64_t latency_us);

    /**
     * Records one dropped frame (e.g., capture skipped because queue was full).
     * Lock-free. Thread-safe.
     */
    void record_drop();

    /**
     * Records one successfully displayed frame for throughput calculation.
     * Lock-free. Thread-safe.
     */
    void record_frame_displayed();

    /**
     * Returns average latency in microseconds for the given stage. Thread-safe for read.
     */
    int64_t get_avg_latency_us(size_t stage_index) const;

    /**
     * Returns total number of dropped frames since start. Lock-free. Thread-safe.
     */
    uint64_t get_drop_count() const;

    /**
     * Returns current throughput (frames per second) over a recent window.
     * Thread-safe for read.
     */
    double get_fps() const;

    /**
     * Resets all counters and averages. Not thread-safe if other threads are
     * recording; call when pipeline is stopped.
     */
    void reset();

private:
    size_t num_stages_{0};
    mutable std::mutex latency_mutex_;
    std::vector<int64_t> avg_latency_us_;
    std::atomic<uint64_t> drops_{0};
    std::atomic<uint64_t> frames_displayed_{0};
    std::chrono::steady_clock::time_point fps_window_start_;
    // TODO: add members for fps sliding window if needed
};
