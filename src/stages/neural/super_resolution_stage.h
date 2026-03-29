#pragma once

#include "neural_stage_base.h"
#include <opencv2/core.hpp>
#include <memory>

/**
 * @struct SuperResResult
 * @brief Upscaled BGR image plus quality metrics vs original-sized comparison.
 */
struct SuperResResult {
    cv::Mat upscaled_frame;
    float psnr_db{0.f};
    float ssim{0.f};
    bool valid{false};
};

/**
 * @class SuperResolutionStage
 * @brief ONNX SR model (256² tensor contract) with PSNR/SSIM for HUD validation.
 *
 * Note: `process` can splat cached upscale into a `Frame`, but the classical pipeline does not
 * invoke `process` today — only `NeuralDispatcher` calls `runInference`; display reads metrics via
 * `getCachedResult`.
 */
class SuperResolutionStage : public NeuralStageBase {
public:
    SuperResolutionStage();
    ~SuperResolutionStage() override;

    /**
     * @brief Loads SR ONNX; empty net disables inference.
     */
    bool loadModel(const std::string& path) override;

    /**
     * @brief Downscales by `scale_factor` then `blobFromImage` to fixed 256² (must match export).
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * @brief Forward, NCHW→HWC, uint8 upscale to `scale_factor` × input dims, compute PSNR/SSIM.
     */
    void runInference(const cv::Mat& frame) override;

    /**
     * @brief Classic MSE-based PSNR on 8-bit BGR after float conversion (cap at 60 dB near-equal).
     */
    float computePSNR(const cv::Mat& original, const cv::Mat& upscaled);

    /**
     * @brief Windowed SSIM on grayscale (standard constants C1, C2 for 8-bit range).
     */
    float computeSSIM(const cv::Mat& original, const cv::Mat& upscaled);

    /**
     * @brief If cache valid, replaces `frame.buffer` with upscaled pixels (unused in current wiring).
     */
    void process(Frame& frame, int64_t* out_latency_us = nullptr) override;

    /**
     * @brief Returns struct copy holding `cv::Mat` upscaled + metrics (mutex during shallow copy).
     */
    SuperResResult getCachedResult();

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
