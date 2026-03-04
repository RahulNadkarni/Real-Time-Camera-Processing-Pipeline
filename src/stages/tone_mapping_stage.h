#pragma once

#include "stage_base.h"
#include <memory>

/**
 * HDR-to-SDR tone curve (or SDR tone mapping). Single responsibility: apply
 * a configurable tone curve to map luminance to display range.
 */
class ToneMappingStage : public StageBase {
public:
    ToneMappingStage();
    ~ToneMappingStage() override;

    /**
     * Applies tone mapping curve to frame.buffer (e.g., Reinhard or ACES-style).
     * Modifies frame in-place. Blocks only on computation; thread-safe if no
     * shared mutable state.
     */
    void process(Frame& frame) override;

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
