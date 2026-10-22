#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <mutex>

// One observation per optimizer update, independent of UI refresh frequency.
// No gradient scan is requested until a monitor attaches to the model.
class GradientDiagnostics {
public:
    static constexpr size_t window_size = 256;
    struct Snapshot {
        size_t samples = 0;
        size_t nonfinite = 0;
        double clip_rate = 0; // Percent of actual clipping decisions, including disabled steps.
        double grad_norm_median = 0;
        double grad_norm_p95 = 0;
        double grad_norm_unclipped_max = 0;
    };
    void request() { requested_.store(true, std::memory_order_relaxed); }
    bool requested() const { return requested_.load(std::memory_order_relaxed); }
    void record(double norm, bool clipped) {
        std::lock_guard<std::mutex> lock(mutex_);
        norms_[next_] = norm;
        clipped_[next_] = clipped;
        next_ = (next_ + 1) % window_size;
        count_ = std::min(count_ + 1, window_size);
    }
    Snapshot snapshot() const {
        std::array<double, window_size> sorted{};
        Snapshot result;
        size_t finite = 0, clipped = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            result.samples = count_;
            for (size_t i = 0; i < count_; ++i) {
                clipped += clipped_[i];
                if (std::isfinite(norms_[i])) sorted[finite++] = norms_[i];
                else ++result.nonfinite;
            }
        }
        if (result.samples) result.clip_rate = 100.0 * clipped / result.samples;
        std::sort(sorted.begin(), sorted.begin() + finite);
        if (finite) {
            result.grad_norm_median = sorted[finite / 2];
            if (finite % 2 == 0)
                result.grad_norm_median = (sorted[finite / 2 - 1] + sorted[finite / 2]) / 2;
            // Nearest-rank percentile, including very short windows.
            result.grad_norm_p95 = sorted[(95 * finite + 99) / 100 - 1];
            result.grad_norm_unclipped_max = sorted[finite - 1];
        }
        return result;
    }
private:
    std::atomic<bool> requested_{false};
    mutable std::mutex mutex_;
    std::array<double, window_size> norms_{};
    std::array<bool, window_size> clipped_{};
    size_t next_ = 0, count_ = 0;
};
