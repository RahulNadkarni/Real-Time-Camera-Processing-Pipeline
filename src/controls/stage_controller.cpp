#include "stage_controller.h"
#include <sstream>

StageController::StageController() {
    for (size_t i = 0; i < kNumStages; i++) {
        enabled_[i].store(true, std::memory_order_relaxed);
    }
}

StageController::StageController(const bool default_enabled[kNumStages]) {
    for (size_t i = 0; i < kNumStages; i++) {
        enabled_[i].store(default_enabled[i], std::memory_order_relaxed);
    }
}

bool StageController::is_enabled(size_t stage_index) const {
    if (stage_index >= kNumStages) return false;
    return enabled_[stage_index].load(std::memory_order_relaxed);
}

bool StageController::is_enabled(StageId stage) const {
    return is_enabled(static_cast<size_t>(stage));
}

void StageController::toggle(size_t stage_index) {
    if (stage_index >= kNumStages) return;
    bool current = enabled_[stage_index].load(std::memory_order_relaxed);
    enabled_[stage_index].store(!current, std::memory_order_relaxed);
}

void StageController::toggle(StageId stage) {
    toggle(static_cast<size_t>(stage));
}

void StageController::set_enabled(size_t stage_index, bool enabled) {
    if (stage_index >= kNumStages) return;
    enabled_[stage_index].store(enabled, std::memory_order_relaxed);
}

bool StageController::handle_key(int key) {
    if (key == '1') { toggle(0); return true; }
    if (key == '2') { toggle(1); return true; }
    if (key == '3') { toggle(2); return true; }
    if (key == '4') { toggle(3); return true; }
    if (key == '5') { toggle(4); return true; }
    return false;
}

std::string StageController::get_state_string() const {
    std::stringstream ss;
    for (size_t i = 0; i < kNumStages; i++) {
        ss << stage_name(i) << ":" << (enabled_[i].load(std::memory_order_relaxed) ? "ON" : "OFF") << " ";
    }
    return ss.str();
}

const char* StageController::stage_name(size_t stage_index) {
    switch (stage_index) {
        case 0: return "Debayer";
        case 1: return "NoiseReduction";
        case 2: return "ToneMapping";
        case 3: return "Histogram";
        case 4: return "EdgeDetection";
        default: return "";
    }
}
