#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

/**
 * Thread-safe FIFO queue for passing items between pipeline stages.
 * All public methods are thread-safe. Blocks on pop when empty until
 * an item is pushed or shutdown() is called.
 */
template <typename T>
class ThreadSafeQueue {
public:
    /**
     * Pushes item onto the back of the queue. Wakes one waiting consumer.
     * Does not block. Thread-safe.
     */
    void push(T item);

    /**
     * Pops and returns the front item. Blocks if queue is empty until
     * push() or shutdown() is called. Returns std::nullopt after shutdown.
     * Thread-safe.
     */
    std::optional<T> pop();

    /**
     * Pops and returns the front item if available within the timeout.
     * Returns std::nullopt if timeout expires with queue empty or after shutdown.
     * Thread-safe.
     */
    template <typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout);

    /**
     * Signals all waiting pop() callers to wake and return std::nullopt.
     * Idempotent. Thread-safe.
     */
    void shutdown();

    /**
     * Returns current number of elements in the queue. Approximate under
     * concurrent access. Thread-safe.
     */
    size_t size() const;

    /**
     * Returns true if shutdown() has been called. Thread-safe.
     */
    bool is_shutdown() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::queue<T> queue_;
    bool shutdown_{false};
};

template <typename T>
void ThreadSafeQueue<T>::push(T item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    cond_.notify_one();
}

template <typename T>
std::optional<T> ThreadSafeQueue<T>::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (shutdown_) {
        return std::nullopt;
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
}

template <typename T>
template <typename Rep, typename Period>
std::optional<T> ThreadSafeQueue<T>::pop_for(const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; })) {
        return std::nullopt;  // timeout, queue still empty
    }
    if (shutdown_) {
        return std::nullopt;
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
}

template <typename T>
void ThreadSafeQueue<T>::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_); 
    shutdown_ = true; 
    cond_.notify_all(); 
    return; 
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
