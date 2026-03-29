#pragma once

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

class PipelineStats {
public:
    static constexpr size_t kRingSize = 128;

    PipelineStats();
    explicit PipelineStats(size_t num_stages);

    void set_num_stages(size_t num_stages);

    void record_stage_latency_us(size_t stage_index, int64_t latency_us);
    void record_e2e_latency_us(int64_t latency_us);
    void record_drop();
    void record_frame_displayed();

    int64_t get_mean_latency_us(size_t stage_index) const;
    int64_t get_p99_latency_us(size_t stage_index) const;
    int64_t get_mean_e2e_us() const;
    int64_t get_p99_e2e_us() const;
    uint64_t get_drop_count() const;
    double get_fps() const;

    void reset();

private:
    struct Ring {
        std::array<int64_t, kRingSize> samples{};
        size_t head{0};
        size_t count{0};

        void push(int64_t v) {
            samples[head] = v;
            head = (head + 1) % kRingSize;
            if (count < kRingSize) ++count;
        }

        int64_t mean() const {
            if (count == 0) return 0;
            int64_t sum = 0;
            for (size_t i = 0; i < count; ++i) sum += samples[i];
            return sum / static_cast<int64_t>(count);
        }

        int64_t p99() const {
            if (count == 0) return 0;
            std::array<int64_t, kRingSize> sorted{};
            std::copy(samples.begin(), samples.begin() + count, sorted.begin());
            std::sort(sorted.begin(), sorted.begin() + count);
            const size_t idx = count > 1 ? (count - 1) * 99 / 100 : 0;
            return sorted[idx];
        }
    };

    size_t num_stages_{0};
    mutable std::mutex latency_mutex_;
    std::vector<Ring> stage_rings_;
    Ring e2e_ring_;

    std::atomic<uint64_t> drops_{0};
    mutable std::atomic<uint64_t> frames_displayed_{0};
    mutable std::chrono::steady_clock::time_point fps_window_start_;
    mutable std::atomic<double> fps_{0.0};
    mutable std::mutex fps_mutex_;
};
