#include "scoped_timer.h"
#include <iostream>

ScopedTimer::ScopedTimer(const std::string& stage_name)
    : stage_name_(stage_name),
      start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    // TODO: compute elapsed_us(), print "stage_name: N us" to stdout (or use a logger)
}

int64_t ScopedTimer::elapsed_us() const {
    // TODO: return duration in microseconds from start_ to now
    return 0;
}
