#pragma once

#include "neural_stage_base.h"
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <utility>
#include <string>

/** Single prediction: class label and confidence in [0,1]. */
struct SceneResult {
    std::vector<std::pair<std::string, float>> top_k_labels;
    bool valid{false};
};

/**
 * Neural stage that runs the scene classifier ONNX model (MobileNetV3 / CIFAR-10).
 * Loads scene_classifier.onnx via cv::dnn::readNetFromONNX. Runs inference every
 * 15 frames and caches the last result; process() overrides base to pass frame
 * through and optionally trigger inference or attach cached result. All inference
 * is intended to be run from NeuralDispatcher so the pipeline never blocks.
 */
class SceneClassifierStage : public NeuralStageBase {
public:
    SceneClassifierStage();
    ~SceneClassifierStage() override;

    /**
     * Load ONNX model from path. Blocks on I/O and net load. Thread-safe only if
     * not called concurrently with runInference or getCachedResult.
     */
    bool loadModel(const std::string& path) override;

    /**
     * Resize frame to 224x224 and normalize for the classifier. Does not block.
     * Returns preprocessed blob (1x3x224x224). Const ref input not modified.
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * Run classifier forward pass on the given frame; updates cached result with
     * top-k labels and confidences. Blocks on inference. Call from dispatcher only.
     */
    void runInference(const cv::Mat& frame) override;

    /**
     * Extract top-k class indices and scores from the network output blob.
     * Does not block. Returns pairs of (index or label, score) for caller to map to names.
     */
    std::vector<std::pair<int, float>> getTopK(const cv::Mat& output_blob, int k);

    /**
     * Override StageBase: process frame in-place. Runs inference every 15 frames
     * (or delegates to dispatcher) and caches result; does not block pipeline.
     * May set out_latency_us to 0 when only reading cache.
     */
    void process(Frame& frame, int64_t* out_latency_us = nullptr) override;

    /**
     * Return the last cached scene result (top-3 labels + confidences). Non-blocking;
     * reads under result_mutex_. Thread-safe.
     */
    SceneResult getCachedResult();

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
