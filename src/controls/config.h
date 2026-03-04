#pragma once

#include <cstddef>
#include <cstdint>

/**
 * Pipeline configuration. Single responsibility: hold all user- and build-time
 * settings for the pipeline (target fps, queue size, resolution, default stage toggles).
 */
struct Config {
    /// Target capture/display frames per second (0 = uncapped).
    int target_fps{30};

    /// Maximum number of frames in each stage queue (backpressure).
    size_t queue_max_size{4};

    /// Capture and processing width in pixels.
    int width{640};

    /// Capture and processing height in pixels.
    int height{480};

    /// Webcam device index (e.g., 0 for default).
    int camera_index{0};

    /// Default enabled state for each stage (Debayer, NoiseReduction, ToneMapping, Histogram, EdgeDetection).
    /// Order matches pipeline stage order. true = enabled by default.
    bool debayer_enabled{true};
    bool noise_reduction_enabled{true};
    bool tone_mapping_enabled{true};
    bool histogram_enabled{false};
    bool edge_detection_enabled{false};
};
