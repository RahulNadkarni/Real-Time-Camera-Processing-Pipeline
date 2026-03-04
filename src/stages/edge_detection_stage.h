#pragma once

#include "stage_base.h"
#include <memory>

/**
 * Canny edge detection pass. Single responsibility: compute edges and
 * optionally overlay or replace frame content with edge map.
 */
class EdgeDetectionStage : public StageBase {
public:
    EdgeDetectionStage();
    ~EdgeDetectionStage() override;

    /**
     * Applies Canny edge detection to frame.buffer; may overlay edges on
     * the image or output an edge map. Modifies frame in-place. Blocks only
     * on computation; thread-safe if no shared mutable state.
     */
    void process(Frame& frame) override;

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
