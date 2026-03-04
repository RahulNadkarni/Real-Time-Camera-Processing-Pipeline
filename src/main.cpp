/**
 * Entry point for the real-time webcam processing pipeline.
 * Sets up the pipeline, opens webcam, starts all threads, runs the event loop,
 * and handles clean shutdown on ESC or Ctrl+C.
 */

#include "pipeline/pipeline.h"
#include "controls/config.h"
#include <iostream>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <chrono>

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
    pipeline.start();

    while (!g_signal_received.load(std::memory_order_relaxed) &&
           !pipeline.is_shutdown_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    pipeline.stop();
    std::cout << "Pipeline stopped.\n";
    return 0;
}
