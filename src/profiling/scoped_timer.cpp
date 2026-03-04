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
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count();
}
