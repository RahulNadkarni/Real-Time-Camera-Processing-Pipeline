#pragma once

#include "../pipeline/frame.h"

/**
 * Abstract base class for all pipeline stages. Each stage runs on its own
 * thread, reads from one queue and writes to the next. Single responsibility:
 * define the process() contract for a single frame.
 */
class StageBase {
public:
    virtual ~StageBase() = default;

    /**
     * Processes the frame in-place (or produces a new frame). Called from
     * the stage's worker thread; must not block on other stages. May modify
     * frame.buffer and metadata. Pure virtual — implement in derived classes.
     */
    virtual void process(Frame& frame) = 0;

    /**
     * Human-readable name for this stage (e.g., "Debayer", "NoiseReduction").
     * Used for profiling output and UI overlay. Thread-safe if not modified after construction.
     */
    virtual const char* name() const = 0;
};
