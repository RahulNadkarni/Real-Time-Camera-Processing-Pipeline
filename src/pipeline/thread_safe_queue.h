#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>
#include <cstddef>

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t capacity = 0) : capacity_(capacity) {}

    bool push(T& item);

    std::optional<T> pop();

    template <typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout);

    std::optional<T> try_pop();

    void shutdown();

    size_t size() const;
    bool is_shutdown() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::queue<T> queue_;
    size_t capacity_;
    bool shutdown_{false};
};

template <typename T>
bool ThreadSafeQueue<T>::push(T& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_) return false;
    if (capacity_ > 0 && queue_.size() >= capacity_) return false;
    queue_.push(std::move(item));
    cond_.notify_one();
    return true;
}

template <typename T>
std::optional<T> ThreadSafeQueue<T>::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (shutdown_) return std::nullopt;
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
}

template <typename T>
template <typename Rep, typename Period>
std::optional<T> ThreadSafeQueue<T>::pop_for(const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; })) {
        return std::nullopt;
    }
    if (shutdown_) return std::nullopt;
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
}

template <typename T>
std::optional<T> ThreadSafeQueue<T>::try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_ || queue_.empty()) return std::nullopt;
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
}

template <typename T>
void ThreadSafeQueue<T>::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    cond_.notify_all();
}

template <typename T>
size_t ThreadSafeQueue<T>::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

template <typename T>
bool ThreadSafeQueue<T>::is_shutdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
}
