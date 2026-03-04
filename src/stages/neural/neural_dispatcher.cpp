#include "neural_dispatcher.h"
#include "../../controls/neural_config.h"
#include <opencv2/core.hpp>
#include <chrono>
#include <thread>

NeuralDispatcher::NeuralDispatcher()
    : scene_stage_(std::make_unique<SceneClassifierStage>()),
      saliency_stage_(std::make_unique<SaliencyStage>()),
      superres_stage_(std::make_unique<SuperResolutionStage>()) {}

NeuralDispatcher::~NeuralDispatcher() {
    stop();
}

void NeuralDispatcher::start() {
    // TODO: implement — loadModel for each stage using NeuralConfig paths; start worker_thread_(run, this)
    if (scene_stage_) scene_stage_->loadModel(NeuralConfig::kSceneClassifierOnnxPath);
    if (saliency_stage_) saliency_stage_->loadModel(NeuralConfig::kSaliencyOnnxPath);
    if (superres_stage_) superres_stage_->loadModel(NeuralConfig::kSuperResOnnxPath);
    shutdown_.store(false);
    worker_thread_ = std::thread(&NeuralDispatcher::run, this);
}

void NeuralDispatcher::stop() {
    // TODO: implement — shutdown_.store(true), queue_cv_.notify_all(), if (worker_thread_.joinable()) join
    shutdown_.store(true);
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

void NeuralDispatcher::submitFrame(const Frame& frame) {
    // TODO: implement — copy Frame into neural_queue_ under queue_mutex_, then queue_cv_.notify_one();
    //       drop frame if queue size exceeds a limit to avoid backlog (pipeline never blocks)
    std::lock_guard<std::mutex> lock(queue_mutex_);
    neural_queue_.push(frame);
    queue_cv_.notify_one();
}

SceneResult NeuralDispatcher::getSceneResult() {
    // TODO: implement — return scene_stage_->getCachedResult(); non-blocking
    return scene_stage_ ? scene_stage_->getCachedResult() : SceneResult{};
}

cv::Mat NeuralDispatcher::getSaliencyResult() {
    // TODO: implement — return saliency_stage_->getCachedResult()
    return saliency_stage_ ? saliency_stage_->getCachedResult() : cv::Mat();
}

SuperResResult NeuralDispatcher::getSuperResResult() {
    // TODO: implement — return superres_stage_->getCachedResult()
    return superres_stage_ ? superres_stage_->getCachedResult() : SuperResResult{};
}

void NeuralDispatcher::run() {
    // TODO: implement — while (!shutdown_): wait on queue_cv_ with pop; build cv::Mat from Frame;
    //       call scene_stage_->runInference(mat) every kSceneClassifierRunEveryNFrames,
    //       saliency_stage_->runInference(mat) every kSaliencyRunEveryNFrames,
    //       superres_stage_->runInference(mat) every kSuperResRunEveryNFrames; updateStats()
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
        cv::Mat mat(frame.height, frame.width, CV_8UC3, frame.buffer.data());
        // TODO: run each stage at its designated interval (every N frames) to match NeuralConfig
        if (scene_stage_) scene_stage_->runInference(mat);
        if (saliency_stage_) saliency_stage_->runInference(mat);
        if (superres_stage_) superres_stage_->runInference(mat);
        updateStats();
    }
}

void NeuralDispatcher::updateStats() {
    // TODO: implement — increment inference_count_, compute time since last_inference_time_, update neural_fps_
    //       (rolling window or exponential moving average); hold stats_mutex_
    std::lock_guard<std::mutex> lock(stats_mutex_);
    inference_count_++;
    auto now = std::chrono::steady_clock::now();
    // compute fps from inference_count_ and elapsed time
    last_inference_time_ = now;
}

double NeuralDispatcher::getNeuralFps() const {
    // TODO: implement — read neural_fps_ under stats_mutex_ (or use atomic); return value
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return neural_fps_;
}
