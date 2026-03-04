#pragma once

#include "../pipeline/frame.h"
#include "../stages/neural/scene_classifier_stage.h"
#include "../stages/neural/super_resolution_stage.h"
#include <opencv2/core.hpp>
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

    /**
     * Overlay scene classifier top-k labels and confidences on the frame (e.g. top-left).
     * Modifies frame.buffer via OpenCV draw. Call from display thread only. Non-blocking.
     */
    void overlaySceneLabels(Frame& frame, const SceneResult& scene_result);

    /**
     * Overlay saliency heatmap on the frame with given alpha blend. Modifies frame.buffer.
     * Call from display thread only. Non-blocking. saliency_map is normalized 0–1, same size or resized.
     */
    void overlaySaliencyMap(Frame& frame, const cv::Mat& saliency_map, double alpha);

    /**
     * Overlay PSNR and SSIM metrics (e.g. for super-resolution) on the frame at position.
     * Modifies frame.buffer. Call from display thread only. Non-blocking.
     */
    void overlayNeuralMetrics(Frame& frame, float psnr, float ssim, const cv::Point& position);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
