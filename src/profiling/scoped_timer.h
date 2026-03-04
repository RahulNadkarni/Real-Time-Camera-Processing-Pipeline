#pragma once

#include <chrono>
#include <string>

/**
 * RAII scoped timer using std::chrono. On destruction, prints stage name and
 * elapsed time in microseconds to stdout. Single responsibility: measure and
 * log a time interval for a named operation.
 */
class ScopedTimer {
public:
    /**
     * Starts the timer. stage_name is copied and printed on destruction.
     * Does not block.
     */
    explicit ScopedTimer(const std::string& stage_name);

    ~ScopedTimer();

    /** Non-copyable, non-movable. */
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    /**
     * Returns elapsed time in microseconds since construction. Does not block.
     */
    int64_t elapsed_us() const;

private:
    std::string stage_name_;
    std::chrono::steady_clock::time_point start_;
};
