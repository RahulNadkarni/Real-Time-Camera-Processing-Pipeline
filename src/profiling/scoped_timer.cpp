#include "scoped_timer.h"

ScopedTimer::ScopedTimer(const std::string& stage_name)
    : stage_name_(stage_name),
      start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    (void)stage_name_;
}

int64_t ScopedTimer::elapsed_us() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count();
}
