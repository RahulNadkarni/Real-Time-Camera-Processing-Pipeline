#include "stage_controller.h"

StageController::StageController() {
    // TODO: set all enabled_[i] to true (or from a default)
}

StageController::StageController(const bool default_enabled[kNumStages]) {
    // TODO: copy default_enabled into enabled_[]
}

bool StageController::is_enabled(size_t stage_index) const {
    (void)stage_index;
    // TODO: return enabled_[stage_index].load(std::memory_order_relaxed)
    return true;
}

bool StageController::is_enabled(StageId stage) const {
    return is_enabled(static_cast<size_t>(stage));
}

void StageController::toggle(size_t stage_index) {
    (void)stage_index;
    // TODO: flip enabled_[stage_index] (compare_exchange or load + store)
}

void StageController::toggle(StageId stage) {
    toggle(static_cast<size_t>(stage));
}

void StageController::set_enabled(size_t stage_index, bool enabled) {
    (void)stage_index;
    (void)enabled;
    // TODO: enabled_[stage_index].store(enabled, std::memory_order_relaxed)
}

bool StageController::handle_key(int key) {
    (void)key;
    // TODO: map key '1'..'5' to toggle(0)..toggle(4); return true if mapped, else false
    return false;
}

std::string StageController::get_state_string() const {
    // TODO: build string like "Debayer:ON NoiseReduction:ON ..." from enabled_[]
    return "";
}

const char* StageController::stage_name(size_t stage_index) {
    (void)stage_index;
    // TODO: return "Debayer", "NoiseReduction", etc. for index 0..4
    return "";
}
