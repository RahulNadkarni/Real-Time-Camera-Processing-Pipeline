#include "pipeline.h"
#include "../stages/debayer_stage.h"
#include "../stages/noise_reduction_stage.h"
#include "../stages/tone_mapping_stage.h"
#include "../stages/histogram_stage.h"
#include "../stages/edge_detection_stage.h"
#include <opencv2/videoio.hpp>
#include <thread>
#include <chrono>

namespace {
constexpr size_t kNumStages = StageController::kNumStages;
constexpr int kChannels = 3;
}  // namespace

Pipeline::Pipeline(const Config& config) : config_(config) {
    const int pool_capacity = static_cast<int>(config.queue_max_size * (kNumStages + 2));
    pool_ = std::make_unique<FramePool>(pool_capacity, config.width, config.height, kChannels);

    queues_.resize(kNumStages + 1);
    for (size_t i = 0; i < queues_.size(); i++) {
        queues_[i] = std::make_unique<ThreadSafeQueue<std::unique_ptr<Frame>>>();
    }

    stages_.push_back(std::make_unique<DebayerStage>());
    stages_.push_back(std::make_unique<NoiseReductionStage>());
    stages_.push_back(std::make_unique<ToneMappingStage>());
    stages_.push_back(std::make_unique<HistogramStage>());
    stages_.push_back(std::make_unique<EdgeDetectionStage>());

    stats_ = std::make_unique<PipelineStats>(kNumStages);

    bool default_enabled[kNumStages] = {
        config.debayer_enabled,
        config.noise_reduction_enabled,
        config.tone_mapping_enabled,
        config.histogram_enabled,
        config.edge_detection_enabled,
    };
    controller_ = std::make_unique<StageController>(default_enabled);

    renderer_ = std::make_unique<Renderer>(config.width, config.height);
    capture_handle_ = nullptr;
    shutdown_requested_.store(false, std::memory_order_relaxed);
}

Pipeline::~Pipeline() {
    stop();
    if (capture_handle_) {
        delete static_cast<cv::VideoCapture*>(capture_handle_);
        capture_handle_ = nullptr;
    }
    renderer_->close();
}

bool Pipeline::start() {
    cv::VideoCapture* cap = new cv::VideoCapture(config_.camera_index);
#if defined(__APPLE__)
    if (!cap->isOpened()) {
        delete cap;
        cap = new cv::VideoCapture(config_.camera_index, cv::CAP_AVFOUNDATION);
    }
#endif
    if (!cap->isOpened()) {
        delete cap;
        capture_handle_ = nullptr;
        return false;
    }
    cap->set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
    cap->set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
    capture_handle_ = cap;

    threads_.push_back(std::thread([this] { capture_loop(); }));
    for (size_t i = 0; i < kNumStages; i++) {
        threads_.push_back(std::thread([this, i] { run_stage(i); }));
    }
    return true;
}

void Pipeline::stop() {
    shutdown_requested_.store(true, std::memory_order_relaxed);
    for (auto& queue : queues_) {
        queue->shutdown();
    }
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads_.clear();
}

bool Pipeline::is_shutdown_requested() const {
    return shutdown_requested_.load(std::memory_order_relaxed);
}

PipelineStats& Pipeline::stats() {
    return *stats_;
}

StageController& Pipeline::controller() {
    return *controller_;
}

void Pipeline::capture_loop() {
    auto* cap = static_cast<cv::VideoCapture*>(capture_handle_);
    if (!cap || !cap->isOpened()) {
        return;
    }
    const int target_fps = config_.target_fps > 0 ? config_.target_fps : 30;
    const auto frame_interval = std::chrono::milliseconds(1000 / target_fps);

    while (!is_shutdown_requested()) {
        std::unique_ptr<Frame> frame = pool_->acquire();
        if (!frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        cv::Mat view(frame->height, frame->width, CV_8UC3, frame->buffer.data());
        if (!cap->read(view)) {
            pool_->release(std::move(frame));
            continue;
        }
        frame->timestamp = std::chrono::steady_clock::now();
        frame->frame_id = frame_id_counter_.fetch_add(1, std::memory_order_relaxed);
        queues_[0]->push(std::move(frame));
        std::this_thread::sleep_for(frame_interval);
    }
}

void Pipeline::run_stage(size_t stage_index) {
    if (stage_index >= kNumStages || stage_index + 1 >= queues_.size()) {
        return;
    }
    while (auto opt = queues_[stage_index]->pop()) {
        std::unique_ptr<Frame> frame = std::move(*opt);
        if (controller_->is_enabled(stage_index)) {
            int64_t latency_us = 0;
            stages_[stage_index]->process(*frame, &latency_us);
            stats_->record_stage_latency_us(stage_index, latency_us);
        }
        queues_[stage_index + 1]->push(std::move(frame));
    }
}

int Pipeline::run_display_iteration() {
    constexpr int kEscKey = 27;
    auto opt = queues_.back()->pop_for(std::chrono::milliseconds(16));
    if (!opt) {
        return -1;
    }
    std::unique_ptr<Frame> frame = std::move(*opt);
    int key = renderer_->render(*frame, stats_.get(), controller_.get());
    controller_->handle_key(key);
    if (key == kEscKey) {
        shutdown_requested_.store(true, std::memory_order_relaxed);
    }
    pool_->release(std::move(frame));
    stats_->record_frame_displayed();
    return key;
}
