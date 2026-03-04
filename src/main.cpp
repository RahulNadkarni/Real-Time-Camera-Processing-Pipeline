/**
 * Entry point for the real-time webcam processing pipeline.
 * Sets up the pipeline, opens webcam, starts all threads, runs the event loop,
 * and handles clean shutdown on ESC or Ctrl+C.
 */

#include "pipeline/pipeline.h"
#include "controls/config.h"
#include "controls/stage_controller.h"
#include <iostream>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <chrono>
#include <sstream>

static std::atomic<bool> g_signal_received{false};

extern "C" void on_signal(int) {
    g_signal_received.store(true);
}

int main(int argc, char* argv[]) {
    int camera_index = 0; 
    int width = 640; 
    int height = 480; 
    int target_fps = 30; 

    for (int i = 1; i < argc; i++) { 
        if (std::strcmp(argv[i], "--camera") == 0) { 
            if (i + 1 < argc) {
                camera_index = std::atoi(argv[i + 1]);
                i++;
            }
        } else if (std::strcmp(argv[i], "--width") == 0) {
            if (i + 1 < argc) {
                width = std::atoi(argv[i + 1]);
                i++;
            }
        } else if (std::strcmp(argv[i], "--height") == 0) {
            if (i + 1 < argc) {
                height = std::atoi(argv[i + 1]);
                i++;
            }
        } else if (std::strcmp(argv[i], "--fps") == 0) {
            if (i + 1 < argc) {
                target_fps = std::atoi(argv[i + 1]);
                i++;
            }
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [--camera <index>] [--width <width>] [--height <height>] [--fps <fps>]" << std::endl;
            return 0;
        }
    }

    Config config;
    config.camera_index = camera_index;
    config.width = width;
    config.height = height;
    config.target_fps = target_fps;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    Pipeline pipeline(config);
    if (!pipeline.start()) {
        std::cerr << "Error: Could not open camera (index " << config.camera_index << ").\n"
                  << "  - Check that the camera is connected and not in use by another app.\n"
                  << "  - On macOS: grant Camera permission in System Settings > Privacy & Security > Camera.\n"
                  << "  - Try a different index: " << argv[0] << " --camera 1\n";
        return 1;
    }

    constexpr auto kStatsInterval = std::chrono::seconds(15);
    auto last_stats_time = std::chrono::steady_clock::now();
    const int target_fps_cap = (config.target_fps > 0) ? config.target_fps : 30;
    const auto frame_interval = std::chrono::microseconds(1000000 / target_fps_cap);
    auto last_frame_time = std::chrono::steady_clock::now();

    while (!g_signal_received.load(std::memory_order_relaxed) &&
           !pipeline.is_shutdown_requested()) {
        pipeline.run_display_iteration();

        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - last_frame_time;
        if (elapsed < frame_interval) {
            std::this_thread::sleep_for(frame_interval - elapsed);
        }
        last_frame_time = std::chrono::steady_clock::now();

        now = std::chrono::steady_clock::now();
        if (now - last_stats_time >= kStatsInterval) {
            last_stats_time = now;
            PipelineStats& s = pipeline.stats();
            std::ostringstream out;
            out << "[stats] FPS: " << s.get_fps() << " | Drops: " << s.get_drop_count();
            for (size_t i = 0; i < StageController::kNumStages; i++) {
                out << " | " << StageController::stage_name(i) << ": " << s.get_avg_latency_us(i) << " us";
            }
            std::cout << out.str() << std::endl;
        }
    }

    pipeline.stop();
    std::cout << "Pipeline stopped.\n";
    return 0;
}
