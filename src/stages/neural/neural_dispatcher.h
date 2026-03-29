#pragma once

#include "scene_classifier_stage.h"
#include "saliency_stage.h"
#include "super_resolution_stage.h"
#include "../../pipeline/frame.h"
#include <chrono>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * @class NeuralDispatcher
 * @brief Single background thread that runs all ONNX stages on copied frames from capture.
 *
 * Project role: decouples expensive `cv::dnn` work from classical `Pipeline` latency. Display
 * reads **cached** outputs via lock + clone pattern inside each stage. Alternatives: one OS thread
 * per model (more parallel, more contention on CPU cache), or process pool for batching.
 */
class NeuralDispatcher {
public:
    /**
     * @brief Constructs three stage objects (models not loaded until `start`).
     */
    NeuralDispatcher();

    /**
     * @brief Ensures `stop()` joins worker (RAII safety if caller forgets stop).
     */
    ~NeuralDispatcher();

    NeuralDispatcher(const NeuralDispatcher&) = delete;
    NeuralDispatcher& operator=(const NeuralDispatcher&) = delete;

    /**
     * @brief Loads ONNX from `NeuralConfig` paths (failures leave nets empty), spawns `run` thread.
     *
     * Blocking: file IO + net parse + thread creation. Call before pipeline begins producing work
     * or guard externally — today `Pipeline::start` invokes after camera opens.
     */
    void start();

    /**
     * @brief Sets shutdown, wakes condition variable, joins worker.
     *
     * Idempotent: `joinable` guard handles double stop. Must complete before destroying stages if
     * future shared state requires it — current unique_ptrs last until dispatcher destructor after join.
     */
    void stop();

    /**
     * @brief Enqueues a **copy** of `frame` for async processing; may drop oldest if queue full.
     *
     * What it does: mutex + push `Frame` value (pixel buffer memcpy via `Frame` copy); if depth
     * exceeds cap, pop front — **latest-biased** policy for live preview (old frames discarded).
     * Notifies one waiter. Never calls `runInference` inline — capture thread stays cheap.
     */
    void submitFrame(const Frame& frame);

    /**
     * @brief Reads cached scene classification from `SceneClassifierStage` (mutex inside stage).
     */
    SceneResult getSceneResult();

    /**
     * @brief Reads cached saliency `cv::Mat` (may be empty).
     */
    cv::Mat getSaliencyResult();

    /**
     * @brief Reads cached SR metrics + optional upscaled image buffer in `SuperResResult`.
     */
    SuperResResult getSuperResResult();

    /**
     * @brief Worker loop: wait on queue, pop one frame, run throttled inferences, update stats.
     *
     * `cv::Mat` header wraps copied neural-queue buffer data — pointer valid for this iteration only.
     * Throttle uses `frames_processed % NeuralConfig::k*RunEveryNFrames` gated calls per stage.
     */
    void run();

    /**
     * @brief Updates instantaneous neural FPS estimate from inter-dequeue times.
     *
     * Not integrated with `PipelineStats` — separate concern for future neural HUD line.
     */
    void updateStats();

    /**
     * @brief Returns last computed `neural_fps_` under mutex.
     */
    double getNeuralFps() const;

private:
    std::unique_ptr<SceneClassifierStage> scene_stage_;
    std::unique_ptr<SaliencyStage> saliency_stage_;
    std::unique_ptr<SuperResolutionStage> superres_stage_;

    std::queue<Frame> neural_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_{false};
    std::thread worker_thread_;

    mutable std::mutex stats_mutex_;
    double neural_fps_{0.0};
    std::chrono::steady_clock::time_point last_inference_time_;
    uint64_t inference_count_{0};
};
