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

class Pipeline {
public:
    explicit Pipeline(const Config& config);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    bool start();
    void stop();

    bool is_shutdown_requested() const;
    int run_display_iteration();

    PipelineStats& stats();
    StageController& controller();

private:
    void capture_loop();
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
    void* capture_handle_{nullptr};
};
