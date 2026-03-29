#pragma once

#include <string>

/**
 * @file neural_config.h
 * @brief Central list of ONNX file paths, model input sizes, and per-model run-every-N-frame cadence.
 *
 * The `NeuralConfig` namespace holds string paths to ONNX files under `ml/models/`, spatial dimensions that
 * must match exported networks (e.g. 224×224 classifier, 256×256 SR blob), scale factor for
 * super-resolution, and throttling integers consumed by `NeuralDispatcher::run`. Training and
 * export live under `ml/`; this header is the C++ side’s contract. No implementation file.
 */

/**
 * @brief Namespace grouping to avoid polluting global preprocessor with model macros.
 *
 * Project role: single source of truth read by `NeuralDispatcher::start` and stage preprocessors.
 */
namespace NeuralConfig {

/** @brief Relative path from cwd when running the binary (typically repo root or `build/`). */
const std::string kSceneClassifierOnnxPath = "ml/models/scene_classifier.onnx";

/** @brief U-Net saliency ONNX from training/export scripts. */
const std::string kSaliencyOnnxPath = "ml/models/saliency.onnx";

/** @brief Super-resolution network ONNX. */
const std::string kSuperResOnnxPath = "ml/models/superres.onnx";

/**
 * @brief Run scene classifier forward only every N neural-queue pops.
 *
 * Why: classifier is often heavier than needed for HUD; throttling saves CPU. Trade-off: label
 * lag grows with N. Alternatives: deadline scheduler or adaptive throttling from `PipelineStats`.
 */
constexpr int kSceneClassifierRunEveryNFrames = 15;

/// @brief Saliency refresh rate vs neural queue; more frequent than classifier, still bounded.
constexpr int kSaliencyRunEveryNFrames = 5;

/// @brief SR runs every processed neural frame (often 1:1 with dispatcher dequeue count).
constexpr int kSuperResRunEveryNFrames = 1;

/// @brief Must match exported ONNX spatial input for classifier (MobileNet-style 224²).
constexpr int kClassifierInputHeight = 224;
constexpr int kClassifierInputWidth = 224;

/// @brief Saliency model spatial input (exported training assumption).
constexpr int kSaliencyInputHeight = 224;
constexpr int kSaliencyInputWidth = 224;

/**
 * @brief Integer upscale factor used when interpreting SR output size vs capture resolution.
 *
 * Must stay consistent with `SuperResolutionStage::runInference` resize math and training export.
 */
constexpr int kSuperResScaleFactor = 2;

/// @brief How many softmax classes to keep for HUD / `SceneResult`.
constexpr int kClassifierTopK = 3;

}  // namespace NeuralConfig
