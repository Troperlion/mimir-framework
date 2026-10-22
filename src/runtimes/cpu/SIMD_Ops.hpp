#ifndef __SIMD_OPS_HPP__
#define __SIMD_OPS_HPP__

#include <immintrin.h>  // AVX2
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "HardwareOpt.hpp"  // Optimisations hardware avancées

namespace SIMD {

inline float hsum256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Matrix multiplication optimisée avec FMA SATURÉ (utilise HardwareOpt)
// A: [M x K], B: [K x N], C: [M x N]
inline void matmul_avx2(float* C, const float* A, const float* B, 
                        size_t M, size_t N, size_t K) {
    // Utiliser la version FMA saturée pour meilleures performances
    HardwareOpt::matmul_fma_saturated(C, A, B, M, N, K);
}

// Matrix transpose multiplication optimisée: C = A @ B^T
// A: [M x K], B: [N x K], C: [M x N]
inline void matmul_transpose_avx2(float* C, const float* A, const float* B,
                                   size_t M, size_t N, size_t K) {
    
    const int workers=HardwareOpt::matmul_workers(M,N,K);
    #pragma omp parallel for if(workers>1) num_threads(workers) schedule(static)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __m256 sum = _mm256_setzero_ps();
            
            size_t k = 0;
            // Process 8 elements at a time
            for (; k + 8 <= K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
            }

            float result = hsum256_ps(sum);
            
            // Process remaining elements
            for (; k < K; ++k) {
                result += A[i * K + k] * B[j * K + k];
            }
            
            C[i * N + j] = result;
        }
    }
}

// Element-wise operations avec AVX2
inline void add_vectors_avx2(float* C, const float* A, const float* B, size_t N) {
    const size_t vecN = N & ~static_cast<size_t>(7);
    for (size_t i = 0; i < vecN; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        __m256 c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(&C[i], c);
    }
    for (size_t i = vecN; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

inline void mul_vectors_avx2(float* C, const float* A, const float* B, size_t N) {
    const size_t vecN = N & ~static_cast<size_t>(7);
    for (size_t i = 0; i < vecN; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        __m256 c = _mm256_mul_ps(a, b);
        _mm256_storeu_ps(&C[i], c);
    }
    for (size_t i = vecN; i < N; ++i) {
        C[i] = A[i] * B[i];
    }
}

// GELU activation avec AVX2
inline void gelu_forward_avx2(float* out, const float* in, size_t N) {
    const float c1 = 0.7978845608f;  // sqrt(2/pi)
    const float c2 = 0.044715f;
    
    const size_t vecN = N & ~static_cast<size_t>(7);
    for (size_t i = 0; i < vecN; i += 8) {
            __m256 x = _mm256_loadu_ps(&in[i]);
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 c1_vec = _mm256_set1_ps(c1);
            __m256 c2_vec = _mm256_set1_ps(c2);
            
            // x^3
            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);
            
            // 0.044715 * x^3
            __m256 term = _mm256_mul_ps(c2_vec, x3);
            
            // x + 0.044715 * x^3
            __m256 sum = _mm256_add_ps(x, term);
            
            // 0.7978... * (x + 0.044715 * x^3)
            __m256 arg = _mm256_mul_ps(c1_vec, sum);
            
            // tanh approximation (pas parfait mais rapide)
            // tanh(x) ≈ x pour |x| < 1
            // Pour une vraie impl, utiliser _mm256_tanh_ps si disponible
            // Ici on fait une approximation rapide
            __m256 tanh_arg = arg;
            __m256 tanh_approx = _mm256_max_ps(_mm256_set1_ps(-1.0f), 
                                _mm256_min_ps(_mm256_set1_ps(1.0f), tanh_arg));
            
            // 0.5 * x * (1 + tanh(...))
            __m256 one_plus_tanh = _mm256_add_ps(one, tanh_approx);
            __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_tanh));
            
            _mm256_storeu_ps(&out[i], result);
    }
    for (size_t i = vecN; i < N; ++i) {
        float x = in[i];
        out[i] = x * 0.5f * (1.0f + std::tanh(c1 * (x + c2 * x * x * x)));
    }
}

// Softmax avec AVX2
inline void softmax_avx2(float* out, const float* in, size_t N) {
    // Find max for numerical stability avec OpenMP reduction
    float max_val = in[0];
    #pragma omp simd reduction(max:max_val)
    for (size_t i = 1; i < N; ++i) {
        if (in[i] > max_val) max_val = in[i];
    }
    
    // Compute exp(x - max)
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    const size_t vecN = N & ~static_cast<size_t>(7);
    for (size_t i = 0; i < vecN; i += 8) {
        __m256 vals = _mm256_loadu_ps(&out[i]);
        __m256 scale = _mm256_set1_ps(inv_sum);
        __m256 result = _mm256_mul_ps(vals, scale);
        _mm256_storeu_ps(&out[i], result);
    }
    for (size_t i = vecN; i < N; ++i) {
        out[i] *= inv_sum;
    }
}

// Dot product optimisé
inline float dot_product_avx2(const float* A, const float* B, size_t N) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 16 <= N; i += 16) {
        __m256 a0 = _mm256_loadu_ps(&A[i]);
        __m256 b0 = _mm256_loadu_ps(&B[i]);
        __m256 a1 = _mm256_loadu_ps(&A[i + 8]);
        __m256 b1 = _mm256_loadu_ps(&B[i + 8]);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }
    for (; i + 8 <= N; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        sum0 = _mm256_fmadd_ps(a, b, sum0);
    }

    float result = hsum256_ps(_mm256_add_ps(sum0, sum1));
    
    // Remaining elements
    for (; i < N; ++i) {
        result += A[i] * B[i];
    }
    
    return result;
}

} // namespace SIMD

#endif // __SIMD_OPS_HPP__
