#include "neural_dispatcher.h"
#include "../../controls/neural_config.h"
#include <opencv2/core.hpp>
#include <chrono>
#include <thread>

namespace {
constexpr size_t kMaxNeuralQueueSize = 4;
}

NeuralDispatcher::NeuralDispatcher()
    : scene_stage_(std::make_unique<SceneClassifierStage>()),
      saliency_stage_(std::make_unique<SaliencyStage>()),
      superres_stage_(std::make_unique<SuperResolutionStage>()) {}

NeuralDispatcher::~NeuralDispatcher() {
    stop();
}

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

void NeuralDispatcher::stop() {
    shutdown_.store(true);
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

void NeuralDispatcher::submitFrame(const Frame& frame) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (neural_queue_.size() >= kMaxNeuralQueueSize) {
        neural_queue_.pop();
    }
    neural_queue_.push(frame);
    queue_cv_.notify_one();
}

SceneResult NeuralDispatcher::getSceneResult() {
    return scene_stage_ ? scene_stage_->getCachedResult() : SceneResult{};
}

cv::Mat NeuralDispatcher::getSaliencyResult() {
    return saliency_stage_ ? saliency_stage_->getCachedResult() : cv::Mat();
}

SuperResResult NeuralDispatcher::getSuperResResult() {
    return superres_stage_ ? superres_stage_->getCachedResult() : SuperResResult{};
}

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

double NeuralDispatcher::getNeuralFps() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return neural_fps_;
}
