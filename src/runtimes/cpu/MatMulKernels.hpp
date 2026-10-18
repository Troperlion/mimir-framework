#pragma once
#include "SIMD_Ops.hpp"
#include "../LayerOps.hpp"

namespace RuntimeMatMul {
// Each batch-row is independent: one parallel region, no nested GEMMs.
inline void forwardBatched(const float* a, const float* b, float* y,
                           size_t batches, size_t m, size_t k, size_t n) {
    const bool accelerated = RuntimeLayerOps::hardwareAccelerationEnabled();
    #pragma omp parallel for if(accelerated && batches*m*k*n > 262144) schedule(static)
    for (size_t row = 0; row < batches*m; ++row) {
        const float* ar = a + row*k;
        const float* bb = b + (row/m)*k*n;
        float* yr = y + row*n;
        size_t j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        if (accelerated) {
            for (; j + 24 <= n; j += 24) {
                __m256 x = _mm256_setzero_ps(), z = x, w = x;
                for (size_t c = 0; c < k; ++c) {
                    const __m256 v = _mm256_set1_ps(ar[c]);
                    const float* br = bb + c*n+j;
                    x = _mm256_fmadd_ps(v, _mm256_loadu_ps(br), x);
                    z = _mm256_fmadd_ps(v, _mm256_loadu_ps(br+8), z);
                    w = _mm256_fmadd_ps(v, _mm256_loadu_ps(br+16), w);
                }
                _mm256_storeu_ps(yr+j, x);
                _mm256_storeu_ps(yr+j+8, z);
                _mm256_storeu_ps(yr+j+16, w);
            }
            for (; j + 8 <= n; j += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (size_t c = 0; c < k; ++c)
                    sum = _mm256_fmadd_ps(_mm256_set1_ps(ar[c]),
                        _mm256_loadu_ps(bb+c*n+j),sum);
                _mm256_storeu_ps(yr+j,sum);
            }
        }
#endif
        for (; j < n; ++j) {
            float sum = 0;
            for (size_t c = 0; c < k; ++c) sum += ar[c] * bb[c*n+j];
            yr[j] = sum;
        }
    }
}

// dA = dY B^T; dB = A^T dY. No transposition buffers are needed.
inline void backward(const float* a, const float* b, const float* dy,
                     float* da, float* db, size_t m, size_t k, size_t n) {
    const bool accelerated = RuntimeLayerOps::hardwareAccelerationEnabled();
    if (accelerated) SIMD::matmul_transpose_avx2(da, dy, b, m, k, n);
    else {
        for (size_t i = 0; i < m; ++i)
            for (size_t c = 0; c < k; ++c) {
                float sum = 0;
                for (size_t j = 0; j < n; ++j) sum += dy[i*n+j] * b[c*n+j];
                da[i*k+c] = sum;
            }
    }
    #pragma omp parallel for if(accelerated && m*k*n > 262144) schedule(static)
    for (size_t c = 0; c < k; ++c) {
        size_t j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        if (accelerated) {
            for (; j + 24 <= n; j += 24) {
                __m256 x = _mm256_setzero_ps(), y = x, z = x;
                for (size_t i = 0; i < m; ++i) {
                    const __m256 v = _mm256_set1_ps(a[i*k+c]);
                    const float* row = dy + i*n+j;
                    x = _mm256_fmadd_ps(v, _mm256_loadu_ps(row), x);
                    y = _mm256_fmadd_ps(v, _mm256_loadu_ps(row+8), y);
                    z = _mm256_fmadd_ps(v, _mm256_loadu_ps(row+16), z);
                }
                _mm256_storeu_ps(db+c*n+j, x);
                _mm256_storeu_ps(db+c*n+j+8, y);
                _mm256_storeu_ps(db+c*n+j+16, z);
            }
        }
#endif
        for (; j < n; ++j) {
            float sum = 0;
            for (size_t i = 0; i < m; ++i) sum += a[i*k+c] * dy[i*n+j];
            db[c*n+j] = sum;
        }
    }
}
}
