#pragma once

#include "neural_stage_base.h"
#include <opencv2/core.hpp>
#include <memory>

/** Result of super-resolution: upscaled image and optional quality metrics. */
struct SuperResResult {
    cv::Mat upscaled_frame;
    float psnr_db{0.f};
    float ssim{0.f};
    bool valid{false};
};

/**
 * Neural stage that runs the super-resolution ONNX model (SRCNN or ESRGAN-tiny).
 * Loads superres.onnx. Preprocess downsamples by scale factor; inference returns
 * upscaled image. process() runs every frame if latency allows, else uses cached
 * result. Inference run from NeuralDispatcher so pipeline never blocks.
 */
class SuperResolutionStage : public NeuralStageBase {
public:
    SuperResolutionStage();
    ~SuperResolutionStage() override;

    /**
     * Load ONNX model from path. Blocks on I/O. Call before starting pipeline.
     */
    bool loadModel(const std::string& path) override;

    /**
     * Preprocess: optionally downscale frame by scale factor for LR input.
     * Const ref not modified.
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * Run SR model forward; return upscaled cv::Mat. Blocks on inference.
     */
    void runInference(const cv::Mat& frame) override;

    /**
     * Compute PSNR in dB between original and upscaled (same size region).
     * Const refs not modified. Does not block.
     */
    float computePSNR(const cv::Mat& original, const cv::Mat& upscaled);

    /**
     * Compute SSIM between original and upscaled (0–1). Const refs not modified.
     */
    float computeSSIM(const cv::Mat& original, const cv::Mat& upscaled);

    /**
     * Override StageBase: run every frame if latency allows; else use cached result.
     * Does not block pipeline.
     */
    void process(Frame& frame, int64_t* out_latency_us = nullptr) override;

    /**
     * Return last cached SuperResResult (upscaled frame + PSNR/SSIM). Non-blocking; thread-safe.
     */
    SuperResResult getCachedResult();

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
