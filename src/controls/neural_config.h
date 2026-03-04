#pragma once

#include <string>

/**
 * Central configuration for neural inference stages.
 * All ONNX model paths and neural-stage settings are loaded from here.
 * Single responsibility: hold paths and constants for scene classifier,
 * saliency, and super-resolution models so the pipeline never blocks on inference.
 */
namespace NeuralConfig {

/** Path to the exported scene classifier ONNX model (CIFAR-10 / MobileNetV3). */
const std::string kSceneClassifierOnnxPath = "ml/models/scene_classifier.onnx";

/** Path to the exported saliency U-Net ONNX model. */
const std::string kSaliencyOnnxPath = "ml/models/saliency.onnx";

/** Path to the exported super-resolution ONNX model (SRCNN or ESRGAN-tiny). */
const std::string kSuperResOnnxPath = "ml/models/superres.onnx";

/** Scene classifier runs inference every N frames (e.g. 15) to reduce load. */
constexpr int kSceneClassifierRunEveryNFrames = 15;

/** Saliency model runs inference every N frames (e.g. 5). */
constexpr int kSaliencyRunEveryNFrames = 5;

/** Super-resolution runs every frame if latency allows; otherwise use cached result. */
constexpr int kSuperResRunEveryNFrames = 1;

/** Classifier input size (height x width). Must match exported ONNX. */
constexpr int kClassifierInputHeight = 224;
constexpr int kClassifierInputWidth = 224;

/** Saliency model input size. */
constexpr int kSaliencyInputHeight = 224;
constexpr int kSaliencyInputWidth = 224;

/** Super-resolution scale factor (e.g. 2 for 2x upscale). */
constexpr int kSuperResScaleFactor = 2;

/** Top-k class labels to return from classifier (e.g. 3). */
constexpr int kClassifierTopK = 3;

}  // namespace NeuralConfig
