#pragma once

#include <atomic>
#include <cstddef>
#include <string>

class StageController {
public:
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

    bool is_enabled(size_t stage_index) const;
    bool is_enabled(StageId stage) const;

    void toggle(size_t stage_index);
    void toggle(StageId stage);
    void set_enabled(size_t stage_index, bool enabled);

    bool handle_key(int key);
    std::string get_state_string() const;

    static const char* stage_name(size_t stage_index);

private:
    std::atomic<bool> enabled_[kNumStages];
};
