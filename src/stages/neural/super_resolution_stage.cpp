#include "super_resolution_stage.h"
#include "../../controls/neural_config.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

struct SuperResolutionStage::Impl {
    cv::dnn::Net net;
    int run_every_n_frames{NeuralConfig::kSuperResRunEveryNFrames};
    int scale_factor{NeuralConfig::kSuperResScaleFactor};
    uint64_t frame_count{0};
};

SuperResolutionStage::SuperResolutionStage() : impl_(std::make_unique<Impl>()) {}

SuperResolutionStage::~SuperResolutionStage() = default;

bool SuperResolutionStage::loadModel(const std::string& path) {
    // TODO: implement — cv::dnn::readNetFromONNX(path), store in impl_->net
    return false;
}

cv::Mat SuperResolutionStage::preprocess(const cv::Mat& frame) {
    // TODO: implement — optionally downscale by scale_factor for LR input; convert to blob
    return cv::Mat();
}

void SuperResolutionStage::runInference(const cv::Mat& frame) {
    // TODO: implement — preprocess, setInput, forward, convert output blob to cv::Mat upscaled,
    //       compute PSNR/SSIM vs original (if same size), lock mutex, cache SuperResResult, result_ready_ = true
}

float SuperResolutionStage::computePSNR(const cv::Mat& original, const cv::Mat& upscaled) {
    // TODO: implement — MSE over overlapping region, then 10*log10(max^2/MSE) in dB
    return 0.f;
}

float SuperResolutionStage::computeSSIM(const cv::Mat& original, const cv::Mat& upscaled) {
    // TODO: implement — structural similarity (e.g. opencv or manual); return 0–1
    return 0.f;
}

void SuperResolutionStage::process(Frame& frame, int64_t* out_latency_us) {
    // TODO: implement — every frame if latency allows else use cache; do not block; out_latency_us = 0 when using cache
    if (out_latency_us) *out_latency_us = 0;
}

SuperResResult SuperResolutionStage::getCachedResult() {
    // TODO: implement — lock result_mutex_, copy SuperResResult, return; invalid if !result_ready_
    return SuperResResult{};
}

const char* SuperResolutionStage::name() const {
    return "SuperResolution";
}
