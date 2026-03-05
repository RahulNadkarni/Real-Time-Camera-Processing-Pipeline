#include "saliency_stage.h"
#include "../../controls/neural_config.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

namespace {
const float kMean[] = {0.485f, 0.456f, 0.406f};
const float kStd[]  = {0.229f, 0.224f, 0.225f};
const int kSaliencyH = NeuralConfig::kSaliencyInputHeight;
const int kSaliencyW = NeuralConfig::kSaliencyInputWidth;
}  // namespace

struct SaliencyStage::Impl {
    cv::dnn::Net net;
    int run_every_n_frames{NeuralConfig::kSaliencyRunEveryNFrames};
    uint64_t frame_count{0};
    cv::Mat cached_heatmap;
};

SaliencyStage::SaliencyStage() : impl_(std::make_unique<Impl>()) {}

SaliencyStage::~SaliencyStage() = default;

bool SaliencyStage::loadModel(const std::string& path) {
    impl_->net = cv::dnn::readNetFromONNX(path);
    return !impl_->net.empty();
}

cv::Mat SaliencyStage::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(kSaliencyW, kSaliencyH), 0, 0, cv::INTER_LINEAR);
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(kSaliencyW, kSaliencyH), cv::Scalar(0, 0, 0), true, false);
    const int C = 3;
    float* ptr = blob.ptr<float>();
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < kSaliencyH * kSaliencyW; ++i)
            ptr[c * kSaliencyH * kSaliencyW + i] = (ptr[c * kSaliencyH * kSaliencyW + i] - kMean[c]) / kStd[c];
    }
    return blob;
}

cv::Mat SaliencyStage::postprocess(const cv::Mat& output_blob, const cv::Size& original_size) {
    // Output shape: (1, 1, kSaliencyH, kSaliencyW), values in [0,1] from sigmoid
    if (output_blob.empty() || output_blob.type() != CV_32F) return cv::Mat();
    const size_t total = output_blob.total();
    if (total < static_cast<size_t>(kSaliencyH * kSaliencyW)) return cv::Mat();
    const float* src = output_blob.ptr<float>();
    cv::Mat heatmap(kSaliencyH, kSaliencyW, CV_32FC1);
    for (int y = 0; y < kSaliencyH; ++y)
        for (int x = 0; x < kSaliencyW; ++x)
            heatmap.at<float>(y, x) = src[y * kSaliencyW + x];
    cv::Mat heatmap_resized;
    cv::resize(heatmap, heatmap_resized, original_size, 0, 0, cv::INTER_LINEAR);
    return heatmap_resized;  // already CV_32FC1, [0,1]
}

void SaliencyStage::runInference(const cv::Mat& frame) {
    if (frame.empty()) return;
    cv::Mat blob = preprocess(frame);
    impl_->net.setInput(blob);
    cv::Mat output = impl_->net.forward();
    cv::Mat heatmap = postprocess(output, frame.size());
    if (heatmap.empty()) return;
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        impl_->cached_heatmap = heatmap.clone();
        result_ready_ = true;
    }
}

void SaliencyStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    impl_->frame_count++;
    (void)frame;
}

cv::Mat SaliencyStage::getCachedResult() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (!result_ready_ || impl_->cached_heatmap.empty()) return cv::Mat();
    return impl_->cached_heatmap.clone();
}

const char* SaliencyStage::name() const {
    return "Saliency";
}
