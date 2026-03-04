/**
 * Entry point for the real-time webcam processing pipeline.
 * Sets up the pipeline, opens webcam, starts all threads, runs the event loop,
 * and handles clean shutdown on ESC or Ctrl+C.
 */

#include "pipeline/pipeline.h"
#include "controls/config.h"
#include <iostream>
#include <csignal>
#include <atomic>

static std::atomic<bool> g_signal_received{false};

extern "C" void on_signal(int) {
    g_signal_received.store(true);
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    // TODO: parse optional argv for camera index, resolution, target_fps

    Config config;
    // TODO: override config from argv if provided

    // TODO: register signal handler for SIGINT (Ctrl+C): set g_signal_received = true

    Pipeline pipeline(config);
    // TODO: open webcam (OpenCV VideoCapture) and store in pipeline or pass to start(); fail fast if camera not available

    pipeline.start();

    // Event loop: poll keyboard (e.g., pipeline.controller().handle_key()) and check g_signal_received;
    // when ESC or signal received, call pipeline.stop() and break
    // TODO: in a loop: get key from display (e.g., renderer returns key or poll separately); handle_key(key); if key == ESC or g_signal_received, break; optional sleep to throttle polling

    pipeline.stop();
    std::cout << "Pipeline stopped.\n";
    return 0;
}
