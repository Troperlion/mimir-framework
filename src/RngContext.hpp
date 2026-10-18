#pragma once

#include <cstdint>
#include <random>

// Small RNG context used to make stochastic ops deterministic when a seed is provided.
// Thread-local so it works with OpenMP threads and avoids cross-call interference.
namespace MimirRng {

inline bool& seededFlag() {
    static thread_local bool value = false;
    return value;
}

inline bool& initializedFlag() {
    static thread_local bool value = false;
    return value;
}

inline std::mt19937& generatorStorage() {
    static thread_local std::mt19937 gen;
    return gen;
}

inline void setSeed(uint32_t seed) {
    auto& gen = generatorStorage();
    gen.seed(seed);
    seededFlag() = true;
    initializedFlag() = true;
}

inline void clearSeed() {
    seededFlag() = false;
}

inline bool hasSeed() {
    return seededFlag();
}

inline std::mt19937& generator() {
    if (!initializedFlag()) {
        std::random_device rd;
        auto& gen = generatorStorage();
        gen.seed(rd());
        initializedFlag() = true;
        return gen;
    }
    return generatorStorage();
}

class ScopedSeed {
public:
    explicit ScopedSeed(uint32_t seed) : prev_seeded_(seededFlag()) {
        // Note: we don't restore the previous generator state (only whether a seed was active).
        // This keeps behavior predictable across calls.
        setSeed(seed);
    }

    ~ScopedSeed() {
        seededFlag() = prev_seeded_;
    }

    ScopedSeed(const ScopedSeed&) = delete;
    ScopedSeed& operator=(const ScopedSeed&) = delete;

private:
    bool prev_seeded_;
};

} // namespace MimirRng
