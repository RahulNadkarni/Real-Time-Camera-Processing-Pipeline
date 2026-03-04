#include "histogram_stage.h"

struct HistogramStage::Impl {};

HistogramStage::HistogramStage() : impl_(std::make_unique<Impl>()) {}

HistogramStage::~HistogramStage() = default;

void HistogramStage::process(Frame& frame) {
    (void)frame;
    // TODO: compute RGB histograms from frame.buffer, draw overlay (e.g., in corner) onto frame
}

const char* HistogramStage::name() const {
    return "Histogram";
}
