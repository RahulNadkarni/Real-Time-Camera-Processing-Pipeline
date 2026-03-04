#pragma once

#include "stage_base.h"
#include <memory>
#include <vector>

/**
 * Per-channel RGB histogram computation and overlay on frame. Single responsibility:
 * compute histograms and draw them (or metadata) onto the frame for display.
 */
class HistogramStage : public StageBase {
public:
    HistogramStage();
    ~HistogramStage() override;

    /**
     * Computes per-channel histograms from frame.buffer and overlays a visualization
     * (e.g., small graph in corner) onto the frame. Modifies frame in-place. Blocks
     * only on computation; thread-safe if no shared mutable state.
     */
    void process(Frame& frame) override;

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
