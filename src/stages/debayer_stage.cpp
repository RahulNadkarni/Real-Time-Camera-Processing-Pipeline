#include "debayer_stage.h"

struct DebayerStage::Impl {};

DebayerStage::DebayerStage() : impl_(std::make_unique<Impl>()) {}

DebayerStage::~DebayerStage() = default;

void DebayerStage::process(Frame& frame) {
    (void)frame;
    // TODO: apply Bayer demosaicing (e.g., bilinear or better) to frame.buffer, write RGB back
}

const char* DebayerStage::name() const {
    return "Debayer";
}
