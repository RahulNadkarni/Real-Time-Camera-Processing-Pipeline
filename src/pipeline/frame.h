#pragma once

#include <cstdint>
#include <vector>
#include <chrono>

/**
 * Represents a single video frame in the pipeline.
 * Holds pixel buffer and metadata; used by all stages.
 * Not thread-safe — each Frame is owned by one stage at a time.
 */
struct Frame {
    /// Raw pixel buffer (e.g., BGR or RGB, 8-bit per channel). Size = width * height * channels.
    std::vector<uint8_t> buffer;

    /// Capture timestamp (monotonic clock) for latency and ordering.
    std::chrono::steady_clock::time_point timestamp;

    /// Monotonically increasing frame ID for debugging and drop detection.
    uint64_t frame_id;

    /// Frame width in pixels.
    int width;

    /// Frame height in pixels.
    int height;

    /// Number of color channels (e.g., 3 for BGR).
    int channels;
};
