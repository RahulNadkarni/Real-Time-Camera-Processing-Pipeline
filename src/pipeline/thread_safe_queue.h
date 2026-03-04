#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

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

// Template implementation in header (skeleton only — implement logic in place)
template <typename T>
void ThreadSafeQueue<T>::push(T item) {
    (void)item;
    // TODO: lock mutex, push item onto queue_, notify one waiter via cond_
}

template <typename T>
std::optional<T> ThreadSafeQueue<T>::pop() {
    // TODO: lock mutex, wait on cond_ until !queue_.empty() or shutdown_; then pop and return front, or return nullopt if shutdown
    return std::nullopt;
}

template <typename T>
void ThreadSafeQueue<T>::shutdown() {
    // TODO: set shutdown_ = true under lock, call cond_.notify_all()
}

template <typename T>
size_t ThreadSafeQueue<T>::size() const {
    // TODO: lock mutex, return queue_.size()
    return 0;
}

template <typename T>
bool ThreadSafeQueue<T>::is_shutdown() const {
    // TODO: lock mutex, return shutdown_
    return false;
}
