#include "edge_detection_stage.h"

struct EdgeDetectionStage::Impl {};

EdgeDetectionStage::EdgeDetectionStage() : impl_(std::make_unique<Impl>()) {}

EdgeDetectionStage::~EdgeDetectionStage() = default;

void EdgeDetectionStage::process(Frame& frame) {
    (void)frame;
    // TODO: apply Canny edge detection to frame.buffer, write result (overlay or edge map) back
}

const char* EdgeDetectionStage::name() const {
    return "EdgeDetection";
}
