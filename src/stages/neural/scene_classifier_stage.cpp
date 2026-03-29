/**
 * @file scene_classifier_stage.cpp
 * @brief ONNX scene classifier: `cv::dnn` load, ImageNet-style preprocess, softmax top-k, CIFAR-10 labels.
 *
 * Implements `SceneClassifierStage`: `loadModel`, `preprocess` (resize + blob + mean/std), `runInference`
 * (forward + label mapping), `getTopK` (stable softmax + partial sort), no-op `process`, and
 * mutex-protected `getCachedResult`. Class name table matches training export order. Pair with
 * `scene_classifier_stage.h`.
 */

#include "scene_classifier_stage.h"
#include "../../controls/neural_config.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
const char* kClassNames[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
};
constexpr int kNumClasses = static_cast<int>(sizeof(kClassNames) / sizeof(kClassNames[0]));

const float kMean[] = {0.485f, 0.456f, 0.406f};
const float kStd[]  = {0.229f, 0.224f, 0.225f};
}

struct SceneClassifierStage::Impl {
    cv::dnn::Net net;
    int run_every_n_frames{NeuralConfig::kSceneClassifierRunEveryNFrames};
    uint64_t frame_count{0};
    SceneResult result;
};

SceneClassifierStage::SceneClassifierStage() : impl_(std::make_unique<Impl>()) {}

SceneClassifierStage::~SceneClassifierStage() = default;

/**
 * @brief ONNX load wrapped in try/catch — OpenCV throws on missing file / parse error.
 */
bool SceneClassifierStage::loadModel(const std::string& path) {
    try {
        impl_->net = cv::dnn::readNetFromONNX(path);
    } catch (const cv::Exception&) {
        // Model file doesn't exist or is invalid - stage will be disabled
        return false;
    }
    return !impl_->net.empty();
}

/**
 * @brief Builds NCHW float blob and applies `(x-mean)/std` per channel in planar layout.
 */
cv::Mat SceneClassifierStage::preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(NeuralConfig::kClassifierInputWidth, NeuralConfig::kClassifierInputHeight), 0, 0, cv::INTER_LINEAR);
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(NeuralConfig::kClassifierInputWidth, NeuralConfig::kClassifierInputHeight), cv::Scalar(0, 0, 0), true, false);
    // Apply ImageNet normalization per channel (blob is 1 x 3 x H x W)
    CV_Assert(blob.isContinuous() && blob.type() == CV_32F && blob.total() >= 3u * resized.total());
    const int C = 3;
    const int H = NeuralConfig::kClassifierInputHeight, W = NeuralConfig::kClassifierInputWidth;
    float* ptr = blob.ptr<float>();
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < H * W; ++i) {
            ptr[c * H * W + i] = (ptr[c * H * W + i] - kMean[c]) / kStd[c];
        }
    }
    return blob;
}

/**
 * @brief Full inference + cache update; guards empty frames/nets.
 */
void SceneClassifierStage::runInference(const cv::Mat& frame) {
    if (frame.empty() || impl_->net.empty()) return;
    cv::Mat blob = preprocess(frame);
    impl_->net.setInput(blob);
    cv::Mat output = impl_->net.forward();
    // Classifier output shape: (1, num_classes) logits
    const int num_classes = static_cast<int>(output.total());
    if (num_classes <= 0) return;
    const int k = std::min(NeuralConfig::kClassifierTopK, num_classes);
    std::vector<std::pair<int, float>> top_k = getTopK(output, k);
    SceneResult result;
    result.top_k_labels.reserve(top_k.size());
    for (const auto& p : top_k) {
        const char* name = (p.first >= 0 && p.first < kNumClasses) ? kClassNames[p.first] : "?";
        result.top_k_labels.emplace_back(name, p.second);
    }
    result.valid = true;
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        impl_->result = std::move(result);
        result_ready_ = true;
    }
}

/**
 * @brief Converts logits → probabilities via stable softmax; selects top-k by partial_sort.
 */
std::vector<std::pair<int, float>> SceneClassifierStage::getTopK(const cv::Mat& output_blob, int k) {
    // output_blob: (1, num_classes) or (num_classes,) logits; apply softmax then take top-k
    if (output_blob.empty() || !output_blob.isContinuous() || output_blob.type() != CV_32F) return {};
    const int num_classes = static_cast<int>(output_blob.total());
    if (num_classes <= 0 || k <= 0) return {};
    k = std::min(k, num_classes);
    const float* logits = output_blob.ptr<float>();
    std::vector<float> probs(logits, logits + num_classes);
    // Softmax (subtract max for numerical stability)
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum = 0.f;
    for (float& p : probs) {
        p = std::exp(p - max_logit);
        sum += p;
    }
    if (sum > 1e-10f) { for (float& p : probs) p /= sum; }
    // Build (value, index) and partial sort by value descending
    std::vector<std::pair<float, int>> val_idx;
    val_idx.reserve(static_cast<size_t>(num_classes));
    for (int i = 0; i < num_classes; ++i) val_idx.emplace_back(probs[i], i);
    std::partial_sort(val_idx.begin(), val_idx.begin() + k, val_idx.end(),
        [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
    std::vector<std::pair<int, float>> top_k;
    top_k.reserve(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) top_k.emplace_back(val_idx[i].second, val_idx[i].first);
    return top_k;
}

/**
 * @brief Intentionally does not touch pixels — async path owns compute.
 */
void SceneClassifierStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    // Inference is run by NeuralDispatcher on a background thread.
    // This stage does not modify frame.buffer; the renderer uses getCachedResult() for overlay.
    (void)frame;
}

/**
 * @brief Mutex copy of cached struct — cheap for small top-k vectors.
 */
SceneResult SceneClassifierStage::getCachedResult() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (!result_ready_) return SceneResult();
    return impl_->result;
}

const char* SceneClassifierStage::name() const {
    return "SceneClassifier";
}
