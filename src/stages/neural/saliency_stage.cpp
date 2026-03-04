#include "saliency_stage.h"
#include "../../controls/neural_config.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

struct SaliencyStage::Impl {
    cv::dnn::Net net;
    int run_every_n_frames{NeuralConfig::kSaliencyRunEveryNFrames};
    uint64_t frame_count{0};
};

SaliencyStage::SaliencyStage() : impl_(std::make_unique<Impl>()) {}

SaliencyStage::~SaliencyStage() = default;

bool SaliencyStage::loadModel(const std::string& path) {
    // TODO: implement — cv::dnn::readNetFromONNX(path), store in impl_->net
    return false;
}

cv::Mat SaliencyStage::preprocess(const cv::Mat& frame) {
    // TODO: implement — resize to 224x224 (or kSaliencyInput*), normalize, NCHW blob
    return cv::Mat();
}

void SaliencyStage::runInference(const cv::Mat& frame) {
    // TODO: implement — preprocess, setInput, forward, postprocess to frame size, lock mutex, cache heatmap, result_ready_ = true
}

cv::Mat SaliencyStage::postprocess(const cv::Mat& output_blob, const cv::Size& original_size) {
    // TODO: implement — reshape/squeeze blob to 2D, resize to original_size, normalize to [0,1]
    return cv::Mat();
}

void SaliencyStage::process(Frame& frame, int64_t* out_latency_us) {
    // TODO: implement — every 5 frames optionally trigger inference (via dispatcher); else no-op; out_latency_us = 0
    if (out_latency_us) *out_latency_us = 0;
}

cv::Mat SaliencyStage::getCachedResult() {
    // TODO: implement — lock result_mutex_, copy cached cv::Mat heatmap, return; empty Mat if !result_ready_
    return cv::Mat();
}

const char* SaliencyStage::name() const {
    return "Saliency";
}
