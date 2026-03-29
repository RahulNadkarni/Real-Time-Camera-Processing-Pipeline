#pragma once

#include "neural_stage_base.h"
#include <opencv2/core.hpp>
#include <memory>

/**
 * @class SaliencyStage
 * @brief U-Net style saliency ONNX producing a single-channel importance map.
 *
 * Project role: async thumbnail HUD; export contract assumes 224² bilinear input and 32F output
 * reinterpreted in `postprocess`.
 */
class SaliencyStage : public NeuralStageBase {
public:
    SaliencyStage();
    ~SaliencyStage() override;

    /**
     * @brief Load ONNX; false on OpenCV exception.
     */
    bool loadModel(const std::string& path) override;

    /**
     * @brief Resize + blobFromImage + ImageNet normalization loops (duplicated from classifier for clarity).
     */
    cv::Mat preprocess(const cv::Mat& frame);

    /**
     * @brief Runs net; maps NCHW output plane to `CV_32FC1` heatmap resized to frame size.
     */
    void runInference(const cv::Mat& frame) override;

    /**
     * @brief Flattens model blob to H×W float map, then linear upsample to camera resolution.
     */
    cv::Mat postprocess(const cv::Mat& output_blob, const cv::Size& original_size);

    /**
     * @brief No classical mutate — increments unused frame counter in impl (legacy hook).
     */
    void process(Frame& frame, int64_t* out_latency_us = nullptr) override;

    /**
     * @brief Returns **clone** of heatmap so display thread owns independent memory.
     */
    cv::Mat getCachedResult();

    const char* name() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
