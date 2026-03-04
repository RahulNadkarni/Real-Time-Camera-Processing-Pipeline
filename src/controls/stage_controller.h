#pragma once

#include <atomic>
#include <cstddef>
#include <string>

/**
 * Keyboard input handler and stage toggles. Holds atomic flags for each stage;
 * pipeline stages read these to decide whether to process or pass-through.
 * Single responsibility: runtime enable/disable of stages and reporting current state.
 */
class StageController {
public:
    /**
     * Number of pipeline stages (Debayer, NoiseReduction, ToneMapping, Histogram, EdgeDetection).
     */
    static constexpr size_t kNumStages = 5;

    enum class StageId : size_t {
        Debayer = 0,
        NoiseReduction = 1,
        ToneMapping = 2,
        Histogram = 3,
        EdgeDetection = 4
    };

    StageController();
    explicit StageController(const bool default_enabled[kNumStages]);

    /**
     * Returns true if the given stage should run (process); false = pass-through.
     * Lock-free. Thread-safe.
     */
    bool is_enabled(size_t stage_index) const;

    /**
     * Returns true if the given stage should run. Lock-free. Thread-safe.
     */
    bool is_enabled(StageId stage) const;

    /**
     * Toggles the stage at stage_index. Lock-free. Thread-safe.
     */
    void toggle(size_t stage_index);

    /**
     * Toggles the stage. Lock-free. Thread-safe.
     */
    void toggle(StageId stage);

    /**
     * Sets enabled state for a stage. Lock-free. Thread-safe.
     */
    void set_enabled(size_t stage_index, bool enabled);

    /**
     * Handles a key press (e.g., '1'–'5' for stages 0–4). Returns true if the key
     * was consumed (mapped to a stage toggle). Does not block.
     */
    bool handle_key(int key);

    /**
     * Returns a string describing current pipeline state (which stages are on/off).
     * Thread-safe for read.
     */
    std::string get_state_string() const;

    /**
     * Returns human-readable name for stage index. Not thread-safe if stage names
     * are modified (they are not in skeleton).
     */
    static const char* stage_name(size_t stage_index);

private:
    std::atomic<bool> enabled_[kNumStages];
};
