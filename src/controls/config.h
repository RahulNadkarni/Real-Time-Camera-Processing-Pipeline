#pragma once

#include <cstddef>
#include <cstdint>

struct Config {
    int target_fps{30};
    size_t queue_max_size{4};
    int width{640};
    int height{480};
    int camera_index{0};
    bool debayer_enabled{true};
    bool noise_reduction_enabled{true};
    bool tone_mapping_enabled{true};
    bool histogram_enabled{false};
    bool edge_detection_enabled{false};
};
