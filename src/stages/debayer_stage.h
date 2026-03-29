#pragma once

#include "stage_base.h"
#include <memory>

class DebayerStage : public StageBase {
public:
    DebayerStage();
    ~DebayerStage() override;

    void process(Frame& frame, int64_t* out_latency_us = nullptr) override;
    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
