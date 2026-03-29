#pragma once

#include <cstdint>
#include <vector>
#include <chrono>

struct Frame {
    std::vector<uint8_t> buffer;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t frame_id;
    int width;
    int height;
    int channels;
};
