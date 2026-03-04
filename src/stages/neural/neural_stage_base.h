#pragma once

#include "../stage_base.h"
#include <opencv2/core.hpp>
#include <atomic>
#include <mutex>
#include <string>

/**
 * Abstract base class for all neural inference stages. Inherits from StageBase.
 * Adds loadModel(), runInference(), and getCachedResult() so that inference
 * can be run asynchronously (e.g. by NeuralDispatcher) and the main pipeline
 * never blocks waiting for inference — it only reads the cached result.
 * result_ready flag and a mutex protect the cached result for thread-safe
 * non-blocking reads and single-writer updates.
 */
class NeuralStageBase : public StageBase {
public:
    NeuralStageBase() = default;
    virtual ~NeuralStageBase() override = default;

    /**
     * Load the ONNX model from the given path. Called once at startup or when
     * model path changes. Blocks on file I/O and net loading; not thread-safe
     * if called concurrently with runInference or getCachedResult. Call from
     * dispatcher thread or before starting the pipeline.
     */
    virtual bool loadModel(const std::string& model_path) = 0;

    /**
     * Run inference on the given frame (OpenCV Mat). Implementations resize/normalize
     * as needed and run the net forward. Blocks on inference; intended to be
     * called from the dispatcher background thread only, not from the main
     * pipeline stage thread.
     */
    virtual void runInference(const cv::Mat& frame) = 0;

    /**
     * Derived classes implement getCachedResult() returning their result type
     * (SceneResult, cv::Mat, SuperResResult). Non-blocking; reads cached result
     * under mutex. Thread-safe for call from display thread.
     */

protected:
    /** Set when a new result has been written to cache; cleared when cache is stale. */
    std::atomic<bool> result_ready_{false};

    /** Protects the cached result (e.g. SceneResult, cv::Mat heatmap, SuperResResult). */
    std::mutex result_mutex_;
};
