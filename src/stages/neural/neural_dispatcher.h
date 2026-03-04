#pragma once

#include "scene_classifier_stage.h"
#include "saliency_stage.h"
#include "super_resolution_stage.h"
#include "pipeline/frame.h"
#include <chrono>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * Orchestrates all three neural stages on a single background thread. Submits
 * frames to a neural queue; run() loop pops frames and dispatches to each stage's
 * runInference(), so the main pipeline never blocks. Tracks neural inference FPS
 * separately from the main pipeline FPS via updateStats(). All get*Result() methods
 * are non-blocking and return the latest cached result from each stage.
 */
class NeuralDispatcher {
public:
    NeuralDispatcher();
    ~NeuralDispatcher();

    /** Non-copyable, non-movable. */
    NeuralDispatcher(const NeuralDispatcher&) = delete;
    NeuralDispatcher& operator=(const NeuralDispatcher&) = delete;

    /**
     * Load all three ONNX models (paths from NeuralConfig) and start the
     * background thread. Blocks on load and thread start. Call once after construction.
     */
    void start();

    /**
     * Signal shutdown, wake the worker, and join the background thread. Blocks
     * until the thread exits. Idempotent after first call. Thread-safe.
     */
    void stop();

    /**
     * Add a frame to the neural queue for processing. Non-blocking if queue is
     * not full; may copy or take ownership of frame data so the pipeline can
     * continue. Thread-safe. Does not block on inference.
     */
    void submitFrame(const Frame& frame);

    /**
     * Return the latest cached scene classifier result. Non-blocking; thread-safe.
     */
    SceneResult getSceneResult();

    /**
     * Return the latest cached saliency heatmap. Non-blocking; thread-safe.
     */
    cv::Mat getSaliencyResult();

    /**
     * Return the latest cached super-resolution result. Non-blocking; thread-safe.
     */
    SuperResResult getSuperResResult();

    /**
     * Main loop for the background thread: pop frame from queue, run each neural
     * stage's runInference() (every N frames per stage), then updateStats().
     * Blocks on queue pop when idle. Called only from the dispatcher thread.
     */
    void run();

    /**
     * Update neural inference FPS and any latency stats. Call from run() after
     * processing a frame. Uses a separate latency tracker distinct from the main
     * pipeline profiler (PipelineStats). Thread-safe if only called from run().
     */
    void updateStats();

    /**
     * Return current neural inference FPS (frames processed per second by the
     * dispatcher). Thread-safe for read.
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

    /** Separate latency tracker for neural inference (not PipelineStats). */
    mutable std::mutex stats_mutex_;
    double neural_fps_{0.0};
    std::chrono::steady_clock::time_point last_inference_time_;
    uint64_t inference_count_{0};
};
