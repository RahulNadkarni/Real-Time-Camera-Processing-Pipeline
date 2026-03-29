#pragma once

#include "../stage_base.h"
#include <opencv2/core.hpp>
#include <atomic>
#include <mutex>
#include <string>

/**
 * @class NeuralStageBase
 * @brief Extends `StageBase` with ONNX load + async `runInference` + cached result pattern.
 *
 * Project role: unify how `NeuralDispatcher` drives DNN while `Pipeline::run_stage` **does not**
 * call heavy `process` for neural workloads (some `process` overrides are no-ops or cache consumers).
 * Alternatives: pure interface unrelated to `StageBase` (would duplicate `name()` / future hooks);
 * Coroutine-based inference (complex on portable C++17).
 */
class NeuralStageBase : public StageBase {
public:
    NeuralStageBase() = default;
    virtual ~NeuralStageBase() override = default;

    /**
     * @brief Loads weights/graph from ONNX path via `cv::dnn::readNetFromONNX`.
     *
     * Typically called once on dispatcher thread before `run` loop. Not thread-safe against
     * concurrent `runInference` — `start()` sequencing guarantees safety here.
     */
    virtual bool loadModel(const std::string& model_path) = 0;

    /**
     * @brief Executes forward pass(es) synchronously **on dispatcher thread**.
     *
     * Must update caches under `result_mutex_`. Caller passes `cv::Mat` view of BGR uint8 data.
     */
    virtual void runInference(const cv::Mat& frame) = 0;

    /**
     * @brief Derived concrete types implement typed `getCachedResult()` (not virtual here to avoid
     * return-type covariance pain); see `SceneClassifierStage`, etc.
     */

protected:
    /// @brief Optional fast flag (not always cleared — accuracy depends on derived class usage).
    std::atomic<bool> result_ready_{false};

    /// @brief Serializes cache writes (inference thread) vs reads (display thread).
    std::mutex result_mutex_;
};
