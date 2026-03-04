#pragma once

#include "stage_base.h"
#include <memory>

/**
 * Bilateral filter for noise reduction. Single responsibility: reduce noise
 * while preserving edges in the frame buffer.
 */
class NoiseReductionStage : public StageBase {
public:
    NoiseReductionStage();
    ~NoiseReductionStage() override;

    /**
     * Applies bilateral filter to frame.buffer in-place (or to a copy then swap).
     * Blocks only on computation; thread-safe if no shared mutable state.
     */
    void process(Frame& frame) override;

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
