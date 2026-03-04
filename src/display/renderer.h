#pragma once

#include "../pipeline/frame.h"
#include <memory>
#include <string>
#include <vector>
#include <atomic>

class PipelineStats;
class StageController;

/**
 * OpenCV-based display. Runs on the display thread; draws the processed frame
 * and an overlay showing active stages and per-stage latency. Single responsibility:
 * present the final frame to the user and show pipeline state.
 */
class Renderer {
public:
    Renderer();
    explicit Renderer(int width, int height);
    ~Renderer();

    /** Non-copyable, non-movable. */
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /**
     * Sets the window title. Safe to call from any thread; may copy string.
     */
    void set_window_title(const std::string& title);

    /**
     * Draws the frame to the window and overlays active stages and per-stage
     * latency (from stats and controller). Blocks until OpenCV imshow/waitKey.
     * Call from display thread only. Returns the key pressed (e.g., for ESC) or -1.
     */
    int render(const Frame& frame,
               const PipelineStats* stats,
               const StageController* controller);

    /**
     * Returns true if the display window is still open and should continue.
     * Thread-safe if implemented with atomic or mutex.
     */
    bool is_open() const;

    /**
     * Closes the window and releases OpenCV resources. Safe to call from display thread.
     */
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
