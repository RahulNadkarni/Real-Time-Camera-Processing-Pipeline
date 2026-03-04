#pragma once

#include "frame.h"
#include "frame_pool.h"
#include "thread_safe_queue.h"
#include "../controls/config.h"
#include "../stages/stage_base.h"
#include "../profiling/pipeline_stats.h"
#include "../controls/stage_controller.h"
#include "../display/renderer.h"
#include <memory>
#include <vector>
#include <thread>
#include <atomic>

/**
 * Master pipeline orchestrator. Chains capture -> stages -> display,
 * starts/stops all worker threads, and handles graceful shutdown using atomics.
 * Single responsibility: lifecycle and wiring of the pipeline.
 */
class Pipeline {
public:
    /**
     * Builds pipeline from config: creates frame pool, queues between stages,
     * stage instances, and wires them. Does not start threads. May throw if
     * config is invalid or resources unavailable.
     */
    explicit Pipeline(const Config& config);

    ~Pipeline();

    /** Non-copyable, non-movable. */
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /**
     * Opens the camera, then starts the capture thread, all stage threads, and display thread.
     * Returns true if the camera opened and all threads were started; false otherwise.
     * Thread-safe to call once; calling after start() or after stop() is undefined.
     */
    bool start();

    /**
     * Signals graceful shutdown: shuts down all queues (so workers unblock),
     * then joins all threads. Blocks until every thread has exited. Idempotent
     * after first call. Thread-safe.
     */
    void stop();

    /**
     * Returns true if stop() has been called or shutdown was requested.
     * Lock-free. Thread-safe.
     */
    bool is_shutdown_requested() const;

    /**
     * Runs one iteration of the display loop on the current thread (must be main
     * thread on macOS for OpenCV window). Pops a frame from the last queue with
     * timeout, renders it, releases to pool, handles key. Returns the key pressed
     * (e.g. 27 for ESC), or -1 if no frame was displayed.
     */
    int run_display_iteration();

    /**
     * Returns the pipeline statistics aggregator for latency/fps/drops.
     * Valid after construction. Thread-safe to read from PipelineStats.
     */
    PipelineStats& stats();

    /**
     * Returns the stage controller for toggling stages at runtime.
     * Valid after construction. Thread-safe per StageController contract.
     */
    StageController& controller();

private:
    /**
     * Capture loop: read from webcam, acquire frame from pool, fill buffer,
     * push to first stage queue. Exits when input queue is shut down or
     * is_shutdown_requested() is true.
     */
    void capture_loop();

    /**
     * Runs a single stage: pop from input queue, process (if stage enabled),
     * push to output queue. Exits when input queue returns nullopt.
     */
    void run_stage(size_t stage_index);

    std::unique_ptr<FramePool> pool_;
    std::vector<std::unique_ptr<ThreadSafeQueue<std::unique_ptr<Frame>>>> queues_;
    std::vector<std::unique_ptr<StageBase>> stages_;
    std::unique_ptr<PipelineStats> stats_;
    std::unique_ptr<StageController> controller_;

    std::vector<std::thread> threads_;
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<uint64_t> frame_id_counter_{0};
    Config config_;

    std::unique_ptr<Renderer> renderer_;
    void* capture_handle_{nullptr};  // e.g., cv::VideoCapture*; lifecycle managed here
};
