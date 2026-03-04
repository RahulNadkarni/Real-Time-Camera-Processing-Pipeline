#include "noise_reduction_stage.h"

struct NoiseReductionStage::Impl {};

NoiseReductionStage::NoiseReductionStage() : impl_(std::make_unique<Impl>()) {}

NoiseReductionStage::~NoiseReductionStage() = default;

void NoiseReductionStage::process(Frame& frame) {
    (void)frame;
    // TODO: apply bilateral filter to frame.buffer (e.g., via OpenCV or custom kernel)
}

const char* NoiseReductionStage::name() const {
    return "NoiseReduction";
}
