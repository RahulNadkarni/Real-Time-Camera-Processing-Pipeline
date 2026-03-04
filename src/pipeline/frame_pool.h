#pragma once

#include "frame.h"
#include <memory>
#include <vector>
#include <mutex>

/**
 * Memory pool allocator for Frame objects to avoid per-frame heap allocation.
 * Pre-allocates a fixed number of Frames; acquire() returns one (or null if exhausted),
 * release() returns it to the pool. Thread-safe.
 */
class FramePool {
public:
    /**
     * Constructs pool with given capacity. Pre-allocates that many Frame objects
     * with buffers sized for width*height*channels. Does not block.
     */
    FramePool(int capacity, int width, int height, int channels);

    /**
     * Returns a Frame from the pool, or nullptr if all are in use. Caller owns
     * the returned pointer until release() is called. Blocks only on internal lock.
     * Thread-safe.
     */
    std::unique_ptr<Frame> acquire();

    /**
     * Returns a Frame to the pool. Must have been obtained via acquire().
     * Blocks only on internal lock. Thread-safe.
     */
    void release(std::unique_ptr<Frame> frame);

    /**
     * Returns the number of Frames currently available in the pool. Approximate
     * under concurrent access. Thread-safe.
     */
    size_t available() const;

private:
    std::vector<std::unique_ptr<Frame>> pool_;
    mutable std::mutex mutex_;
    int width_;
    int height_;
    int channels_;
};
