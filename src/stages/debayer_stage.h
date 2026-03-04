#pragma once

#include "stage_base.h"
#include <memory>

/**
 * Simulates Bayer pattern demosaicing (raw sensor -> RGB). Single responsibility:
 * convert Bayer-like or raw input to full RGB in the frame buffer.
 */
class DebayerStage : public StageBase {
public:
    DebayerStage();
    ~DebayerStage() override;

    /**
     * Applies demosaicing to frame.buffer (assumes Bayer pattern or passthrough
     * for already-RGB). Writes result back into frame.buffer. Blocks only on
     * internal work; thread-safe for concurrent process() calls if implementation
     * uses no shared mutable state.
     */
    void process(Frame& frame) override;

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
