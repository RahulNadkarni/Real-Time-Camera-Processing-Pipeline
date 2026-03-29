#pragma once

#include "../pipeline/frame.h"
#include <memory>
#include <string>

class PipelineStats;
class StageController;

class Renderer {
public:
    Renderer();
    explicit Renderer(int width, int height);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void set_window_title(const std::string& title);

    int render(const Frame& frame,
               const PipelineStats* stats,
               const StageController* controller);

    bool is_open() const;
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
