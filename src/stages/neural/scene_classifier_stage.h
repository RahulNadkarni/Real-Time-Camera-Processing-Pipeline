#pragma once

#include "neural_stage_base.h"
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <utility>
#include <string>

/**
 * @struct SceneResult
 * @brief Serializable HUD payload: top-k string labels with softmax probabilities.
 */
struct SceneResult {
    std::vector<std::pair<std::string, float>> top_k_labels;
    bool valid{false};
};

/**
 * @class SceneClassifierStage
 * @brief CIFAR-10 / MobileNet-style classifier exported as ONNX for `cv::dnn`.
 *
 * Project role: async label overlay; training/export lives under `ml/`. Classical `process` is a
 * no-op because inference is dispatcher-driven only.
 */
class SceneClassifierStage : public NeuralStageBase {
public:
    SceneClassifierStage();
    ~SceneClassifierStage() override;

    /**
     * @brief Reads ONNX; empty net on failure so `runInference` early-exits.
     */
    bool loadModel(const std::string& path) override;

    /**
     * @brief Resize + `blobFromImage` + per-channel ImageNet mean/std normalization manually.
     *
     * Why manual norm after blobFromImage: matches PyTorch export expectations precisely; alternative
     * is embedding normalization in ONNX graph (fewer lines here, more export complexity).
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * @brief Forward net, softmax logits, map indices to `kClassNames`, store `SceneResult` under mutex.
     */
    void runInference(const cv::Mat& frame) override;

    /**
     * @brief Numerically stable softmax + `partial_sort` for top-k (faster than full sort).
     */
    std::vector<std::pair<int, float>> getTopK(const cv::Mat& output_blob, int k);

    /**
     * @brief `StageBase` hook: no buffer mutation — HUD uses `getCachedResult` from display thread.
     */
    void process(Frame& frame, int64_t* out_latency_us = nullptr) override;

    /**
     * @brief Returns copy of last result (mutex); empty if never succeeded.
     */
    SceneResult getCachedResult();

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
