#include "tone_mapping_stage.h"

struct ToneMappingStage::Impl {};

ToneMappingStage::ToneMappingStage() : impl_(std::make_unique<Impl>()) {}

ToneMappingStage::~ToneMappingStage() = default;

void ToneMappingStage::process(Frame& frame) {
    (void)frame;
    // TODO: apply HDR-to-SDR tone curve to frame.buffer (e.g., per-channel or luminance-based)
}

const char* ToneMappingStage::name() const {
    return "ToneMapping";
}
