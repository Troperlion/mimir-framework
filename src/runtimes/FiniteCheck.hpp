#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Mimir::RuntimeChecks {
inline bool allFinite(const float* values, size_t count) {
    if (!values) return count == 0;
    size_t i = 0;
#if defined(__AVX2__)
    if constexpr (sizeof(float) == 4 && std::numeric_limits<float>::is_iec559) {
        const __m256i exponent = _mm256_set1_epi32(0x7f800000);
        // Classify 32 values per branch; loads do not require alignment.
        for (; count - i >= 32; i += 32) {
            __m256i invalid = _mm256_setzero_si256();
            for (size_t offset = 0; offset < 32; offset += 8) {
                const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(values + i + offset));
                invalid = _mm256_or_si256(invalid,
                    _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent));
            }
            if (_mm256_movemask_epi8(invalid)) return false;
        }
    }
#endif
    for (; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}
}
