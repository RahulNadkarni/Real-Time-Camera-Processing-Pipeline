#include "scene_classifier_stage.h"
#include "../../controls/neural_config.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <cstring>

struct SceneClassifierStage::Impl {
    cv::dnn::Net net;
    int run_every_n_frames{NeuralConfig::kSceneClassifierRunEveryNFrames};
    uint64_t frame_count{0};
};

SceneClassifierStage::SceneClassifierStage() : impl_(std::make_unique<Impl>()) {}

SceneClassifierStage::~SceneClassifierStage() = default;

bool SceneClassifierStage::loadModel(const std::string& path) {
    // TODO: implement — cv::dnn::readNetFromONNX(path), store in impl_->net, return true if success
    return false;
}

cv::Mat SceneClassifierStage::preprocess(const cv::Mat& frame) {
    // TODO: implement — resize to 224x224, convert to float, normalize (e.g. ImageNet mean/std), NCHW blob
    return cv::Mat();
}

void SceneClassifierStage::runInference(const cv::Mat& frame) {
    // TODO: implement — preprocess(frame), setInput, net.forward(), getTopK, fill SceneResult with class names,
    //       lock result_mutex_, write cached result, result_ready_ = true
}

std::vector<std::pair<int, float>> SceneClassifierStage::getTopK(const cv::Mat& output_blob, int k) {
    // TODO: implement — parse blob (1 x num_classes), partial sort or argpartition for top-k indices and values
    return {};
}

void SceneClassifierStage::process(Frame& frame, int64_t* out_latency_us) {
    // TODO: implement — if (frame.frame_id % run_every_n == 0) optionally submit to dispatcher or run inference;
    //       else only read getCachedResult() and attach to frame if Frame is extended, or leave for renderer;
    //       set out_latency_us to 0 when not running inference so pipeline never blocks
    if (out_latency_us) *out_latency_us = 0;
}

SceneResult SceneClassifierStage::getCachedResult() {
    // TODO: implement — lock result_mutex_, copy SceneResult from cache, return; if !result_ready_ return invalid
    return SceneResult{};
}

const char* SceneClassifierStage::name() const {
    return "SceneClassifier";
}
