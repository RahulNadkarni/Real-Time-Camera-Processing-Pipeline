#pragma once

#include "neural_stage_base.h"
#include <opencv2/core.hpp>
#include <memory>

/**
 * Neural stage that runs the saliency U-Net ONNX model. Loads saliency.onnx.
 * Produces a heatmap (cv::Mat) normalized 0–1. Runs inference every 5 frames
 * and caches the heatmap; process() runs every 5 frames and caches. Inference
 * is run from NeuralDispatcher so the pipeline never blocks.
 */
class SaliencyStage : public NeuralStageBase {
public:
    SaliencyStage();
    ~SaliencyStage() override;

    /**
     * Load ONNX model from path. Blocks on I/O. Call before starting pipeline.
     */
    bool loadModel(const std::string& path) override;

    /**
     * Preprocess frame for saliency model (resize, normalize). Const ref not modified.
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * Run saliency model forward; updates cached heatmap. Blocks on inference.
     * Call from dispatcher only.
     */
    void runInference(const cv::Mat& frame) override;

    /**
     * Upsample the model output blob back to original_size (frame dimensions).
     * Does not block.
     */
    cv::Mat postprocess(const cv::Mat& output_blob, const cv::Size& original_size);

    /**
     * Override StageBase: process frame; run inference every 5 frames and cache
     * heatmap. Does not block pipeline; inference should be in dispatcher.
     */
    void process(Frame& frame, int64_t* out_latency_us = nullptr) override;

    /**
     * Return the last cached heatmap (0–1 normalized). Non-blocking; thread-safe.
     */
    cv::Mat getCachedResult();

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
