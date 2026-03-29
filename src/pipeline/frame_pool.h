#pragma once

#include "frame.h"
#include <memory>
#include <vector>
#include <mutex>

class FramePool {
public:
    FramePool(int capacity, int width, int height, int channels);

    std::unique_ptr<Frame> acquire();
    void release(std::unique_ptr<Frame> frame);
    size_t available() const;

private:
    std::vector<std::unique_ptr<Frame>> pool_;
    mutable std::mutex mutex_;
    int width_;
    int height_;
    int channels_;
};
