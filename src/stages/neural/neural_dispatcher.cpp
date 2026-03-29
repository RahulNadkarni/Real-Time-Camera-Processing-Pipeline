/**
 * @file neural_dispatcher.cpp
 * @brief Async neural path: loads three ONNX models, runs one worker thread, bounded frame queue.
 *
 * Implements `NeuralDispatcher::start/stop/submitFrame`, the `run` loop that pops copied `Frame`
 * values and calls throttled `runInference` on scene/saliency/super-resolution stages, delegates
 * `get*Result` to each stage’s cache, and maintains optional neural FPS stats. Owns the mutex,
 * condition variable, and `std::queue<Frame>` used between capture and the neural worker.
 */

#include "neural_dispatcher.h"
#include "../../controls/neural_config.h"
#include <opencv2/core.hpp>
#include <chrono>
#include <thread>

namespace {
/// @brief Caps memory/latency of async queue — drop-oldest policy when full.
constexpr size_t kMaxNeuralQueueSize = 4;
}

/**
 * @brief Allocates stage unique_ptrs with default constructors (nets empty).
 */
NeuralDispatcher::NeuralDispatcher()
    : scene_stage_(std::make_unique<SceneClassifierStage>()),
      saliency_stage_(std::make_unique<SaliencyStage>()),
      superres_stage_(std::make_unique<SuperResolutionStage>()) {}

/**
 * @brief Joins worker so destructor never leaves a running thread accessing `this`.
 */
NeuralDispatcher::~NeuralDispatcher() {
    stop();
}

/**
 * @brief Loads models (exceptions swallowed inside stages returning empty nets), resets stats, starts thread.
 *
 * Why reset stats in critical section: avoids torn reads if HUD ever queried mid-init.
 */
void NeuralDispatcher::start() {
    if (scene_stage_) scene_stage_->loadModel(NeuralConfig::kSceneClassifierOnnxPath);
    if (saliency_stage_) saliency_stage_->loadModel(NeuralConfig::kSaliencyOnnxPath);
    if (superres_stage_) superres_stage_->loadModel(NeuralConfig::kSuperResOnnxPath);
    shutdown_.store(false);
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        inference_count_ = 0;
        last_inference_time_ = std::chrono::steady_clock::now();
        neural_fps_ = 0.0;
    }
    worker_thread_ = std::thread(&NeuralDispatcher::run, this);
}

/**
 * @brief Cooperative shutdown: atomic flag + notify_all + join.
 */
void NeuralDispatcher::stop() {
    shutdown_.store(true);
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

/**
 * @brief Push path for neural work; drop oldest on overflow (keeps queue fresh).
 */
void NeuralDispatcher::submitFrame(const Frame& frame) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (neural_queue_.size() >= kMaxNeuralQueueSize) {
        neural_queue_.pop();
    }
    neural_queue_.push(frame);
    queue_cv_.notify_one();
}

/**
 * @brief Thin delegate to scene stage cache read.
 */
SceneResult NeuralDispatcher::getSceneResult() {
    return scene_stage_ ? scene_stage_->getCachedResult() : SceneResult{};
}

/**
 * @brief Thin delegate to saliency stage cache read.
 */
cv::Mat NeuralDispatcher::getSaliencyResult() {
    return saliency_stage_ ? saliency_stage_->getCachedResult() : cv::Mat();
}

/**
 * @brief Thin delegate to super-resolution cache read.
 */
SuperResResult NeuralDispatcher::getSuperResResult() {
    return superres_stage_ ? superres_stage_->getCachedResult() : SuperResResult{};
}

/**
 * @brief Core worker loop — pops `Frame` copies, wraps buffer as `cv::Mat`, runs gated `runInference`.
 *
 * Note: `mat` points into `frame.buffer` of the **queue copy**, not the live pool frame in classical
 * pipeline — safe because no other thread references this `Frame` after pop until loop iter ends.
 */
void NeuralDispatcher::run() {
    uint64_t frames_processed = 0;
    while (!shutdown_.load()) {
        Frame frame;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return shutdown_.load() || !neural_queue_.empty(); });
            if (shutdown_.load()) break;
            if (neural_queue_.empty()) continue;
            frame = neural_queue_.front();
            neural_queue_.pop();
        }
        frames_processed++;
        cv::Mat mat(frame.height, frame.width, CV_8UC3, frame.buffer.data());

        if (scene_stage_ && (frames_processed % NeuralConfig::kSceneClassifierRunEveryNFrames == 0)) {
            scene_stage_->runInference(mat);
        }
        if (saliency_stage_ && (frames_processed % NeuralConfig::kSaliencyRunEveryNFrames == 0)) {
            saliency_stage_->runInference(mat);
        }
        if (superres_stage_ && (frames_processed % NeuralConfig::kSuperResRunEveryNFrames == 0)) {
            superres_stage_->runInference(mat);
        }
        updateStats();
    }
}

/**
 * @brief Inverse of time delta since last processed neural frame — crude instantaneous FPS.
 */
void NeuralDispatcher::updateStats() {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(stats_mutex_);
    inference_count_++;
    using Sec = std::chrono::duration<double>;
    Sec elapsed = std::chrono::duration_cast<Sec>(now - last_inference_time_);
    if (inference_count_ > 1 && elapsed.count() > 0.0) {
        neural_fps_ = 1.0 / elapsed.count();
    }
    last_inference_time_ = now;
}

/**
 * @brief Returns `neural_fps_` snapshot for diagnostics / future overlay.
 */
double NeuralDispatcher::getNeuralFps() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return neural_fps_;
}
