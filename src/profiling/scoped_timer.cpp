#include "scoped_timer.h"
#include <iostream>

ScopedTimer::ScopedTimer(const std::string& stage_name)
    : stage_name_(stage_name),
      start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    int64_t us = elapsed_us();
    std::cout << stage_name_ << ": " << us << " us" << std::endl;
}

int64_t ScopedTimer::elapsed_us() const {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<int64_t, std::micro> duration = now - start_;
    return duration.count();
}
