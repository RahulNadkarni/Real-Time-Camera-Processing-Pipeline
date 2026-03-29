#pragma once

#include <chrono>
#include <string>

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& stage_name);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    int64_t elapsed_us() const;

private:
    std::string stage_name_;
    std::chrono::steady_clock::time_point start_;
};
