#pragma once

#include "../pipeline/frame.h"

class StageBase {
public:
    virtual ~StageBase() = default;
    virtual void process(Frame& frame, int64_t* out_latency_us = nullptr) = 0;
    virtual const char* name() const = 0;
};
