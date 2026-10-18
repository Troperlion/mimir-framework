#pragma once
#include <atomic>

// Shared by the training model and all UI backends. UI changes are sampled
// once per forward; backward uses the saved forward state.
struct SkipConnectionControl {
    std::atomic<bool> available{false};
    std::atomic<bool> enabled{true};
    void toggle() {
        if (!available.load()) return;
        bool previous = enabled.load();
        while (!enabled.compare_exchange_weak(previous, !previous)) {}
    }
};
