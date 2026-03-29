#include "frame_pool.h"

FramePool::FramePool(int capacity, int width, int height, int channels)
    : width_(width), height_(height), channels_(channels) {
    pool_.reserve(static_cast<size_t>(capacity));
    for (int i = 0; i < capacity; i++) {
        auto frame = std::make_unique<Frame>();
        frame->width = width;
        frame->height = height;
        frame->channels = channels;
        frame->buffer.resize(static_cast<size_t>(width) * height * channels);
        pool_.push_back(std::move(frame));
    }
}

std::unique_ptr<Frame> FramePool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.empty()) return nullptr;
    auto frame = std::move(pool_.back());
    pool_.pop_back();
    return frame;
}

void FramePool::release(std::unique_ptr<Frame> frame) {
    if (!frame) return;
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(std::move(frame));
}

size_t FramePool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
}
