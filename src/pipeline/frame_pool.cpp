#include "frame_pool.h"

FramePool::FramePool(int capacity, int width, int height, int channels)
    : width_(width), height_(height), channels_(channels) {
    // TODO: pre-allocate capacity Frames; set each frame's width/height/channels and buffer.resize(width*height*channels)
}

std::unique_ptr<Frame> FramePool::acquire() {
    // TODO: lock mutex_; if pool_ has an available frame, pop and return it; else return nullptr
    return nullptr;
}

void FramePool::release(std::unique_ptr<Frame> frame) {
    (void)frame;
    // TODO: lock mutex_; reset frame buffer if needed; push frame back onto pool_
}

size_t FramePool::available() const {
    // TODO: lock mutex_; return pool_.size()
    return 0;
}
