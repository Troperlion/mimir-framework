#ifndef __HARDWARE_OPT_HPP__
#define __HARDWARE_OPT_HPP__

#include <immintrin.h>  // AVX2, FMA, F16C, BMI
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#if !defined(_WIN32)
#include <sys/mman.h>   // madvise, HugePages
#include <unistd.h>
#endif

// ===== FP16 STORAGE + F16C =====

namespace HardwareOpt {

// Keep small tiled GEMMs inside one worker. A large global thread setting
// otherwise launches a full team for every small convolution tile.
inline int matmul_workers(size_t rows,size_t columns,size_t reduction) {
    if(!rows || !columns || !reduction)return 1;
#ifdef _OPENMP
    const size_t workers=std::min(rows,std::max(size_t(1),rows*columns*reduction/(1024*1024)));
    return static_cast<int>(std::min(size_t(omp_get_max_threads()),workers));
#else
    return 1;
#endif
}

#if defined(_MSC_VER)
#define MIMIR_RESTRICT __restrict
#else
#define MIMIR_RESTRICT __restrict__
#endif

// Conversion FP32 -> FP16 avec F16C
inline void fp32_to_fp16_f16c(uint16_t* dst, const float* src, size_t count) {
#ifdef __F16C__
    // Parallélisation OpenMP pour grandes conversions (>1024 éléments)
    size_t vec_count = (count / 8) * 8; // Nombre d'éléments vectorisés
    for (size_t i = 0; i < vec_count; i += 8) {
        __m256 f32 = _mm256_loadu_ps(&src[i]);
        __m128i f16 = _mm256_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i*)&dst[i], f16);
    }
    // Remaining elements (séquentiel)
    for (size_t i = vec_count; i < count; ++i) {
        __m128 f32 = _mm_set_ss(src[i]);
        __m128i f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);
        dst[i] = _mm_extract_epi16(f16, 0);
    }
#else
    // Fallback software avec OpenMP
    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        union { float f; uint32_t u; } v;
        v.f = src[i];
        uint32_t sign = (v.u >> 16) & 0x8000;
        uint32_t exp = ((v.u >> 23) & 0xFF) - 112;
        uint32_t mant = (v.u >> 13) & 0x3FF;
        
        if (exp > 30) exp = 31, mant = 0; // Inf/NaN
        else if (exp < 0) exp = 0, mant = 0; // Denorm -> 0
        
        dst[i] = sign | (exp << 10) | mant;
    }
#endif
}

// Conversion FP16 -> FP32 avec F16C
inline void fp16_to_fp32_f16c(float* dst, const uint16_t* src, size_t count) {
#ifdef __F16C__
    // Parallélisation OpenMP pour grandes conversions
    size_t vec_count = (count / 8) * 8; // Nombre d'éléments vectorisés
    for (size_t i = 0; i < vec_count; i += 8) {
        __m128i f16 = _mm_loadu_si128((__m128i*)&src[i]);
        __m256 f32 = _mm256_cvtph_ps(f16);
        _mm256_storeu_ps(&dst[i], f32);
    }
    // Remaining elements (séquentiel)
    for (size_t i = vec_count; i < count; ++i) {
        __m128i f16 = _mm_set1_epi16(src[i]);
        __m128 f32 = _mm_cvtph_ps(f16);
        dst[i] = _mm_cvtss_f32(f32);
    }
#else
    // Fallback software avec OpenMP
    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        uint16_t h = src[i];
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = ((h >> 10) & 0x1F);
        uint32_t mant = (h & 0x3FF);
        
        if (exp == 0) {
            exp = 0; mant = 0; // Denorm -> 0
        } else if (exp == 31) {
            exp = 255; // Inf/NaN
        } else {
            exp += 112; // Rebias
        }
        
        union { float f; uint32_t u; } v;
        v.u = sign | (exp << 23) | (mant << 13);
        dst[i] = v.f;
    }
#endif
}

// ===== BF16 STORAGE (software, portable) =====

inline void fp32_to_bf16(uint16_t* dst, const float* src, size_t count) {
    // Rounding-to-nearest-even on truncation of lower 16 bits.
    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        union { float f; uint32_t u; } v;
        v.f = src[i];
        const uint32_t lsb = (v.u >> 16) & 1u;
        const uint32_t rounding_bias = 0x7FFFu + lsb;
        dst[i] = static_cast<uint16_t>((v.u + rounding_bias) >> 16);
    }
}

inline void bf16_to_fp32(float* dst, const uint16_t* src, size_t count) {
    #pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        union { float f; uint32_t u; } v;
        v.u = (static_cast<uint32_t>(src[i]) << 16);
        dst[i] = v.f;
    }
}

// ===== FMA SATURÉ (3 opérations par cycle) =====

// Matmul optimisé avec FMA complètement saturé
// Pipeline: 3 accumulateurs indépendants pour saturer les 2 ports FMA
template <bool Accumulate>
inline void matmul_fma_saturated_impl(float* MIMIR_RESTRICT C,
                                  const float* MIMIR_RESTRICT A,
                                  const float* MIMIR_RESTRICT B,
                                  size_t M, size_t N, size_t K) {

    const size_t vecN8 = N & ~static_cast<size_t>(7);
    const size_t vecN24 = (N / 24) * 24;

    const int workers=matmul_workers(M,N,K);
    #pragma omp parallel for if(workers>1) num_threads(workers) schedule(static)
    for (size_t i = 0; i < M; ++i) {
        // 24-wide unroll (3x8) for FMA saturation
        for (size_t j = 0; j < vecN24; j += 24) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();

            for (size_t k = 0; k < K; ++k) {
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);

                __m256 b0 = _mm256_loadu_ps(&B[k * N + j]);
                __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
                __m256 b2 = _mm256_loadu_ps(&B[k * N + j + 16]);

                acc0 = _mm256_fmadd_ps(a_broadcast, b0, acc0);
                acc1 = _mm256_fmadd_ps(a_broadcast, b1, acc1);
                acc2 = _mm256_fmadd_ps(a_broadcast, b2, acc2);
            }

            if constexpr (Accumulate) acc0 = _mm256_add_ps(_mm256_loadu_ps(&C[i * N + j]), acc0);
            _mm256_storeu_ps(&C[i * N + j], acc0);
            if constexpr (Accumulate) acc1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * N + j + 8]), acc1);
            _mm256_storeu_ps(&C[i * N + j + 8], acc1);
            if constexpr (Accumulate) acc2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * N + j + 16]), acc2);
            _mm256_storeu_ps(&C[i * N + j + 16], acc2);
        }

        // Remaining full 8-wide vectors (after vecN24)
        for (size_t j = vecN24; j < vecN8; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t k = 0; k < K; ++k) {
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);
                __m256 b = _mm256_loadu_ps(&B[k * N + j]);
                acc = _mm256_fmadd_ps(a_broadcast, b, acc);
            }
            if constexpr (Accumulate) acc = _mm256_add_ps(_mm256_loadu_ps(&C[i * N + j]), acc);
            _mm256_storeu_ps(&C[i * N + j], acc);
        }

        // Scalar tail (N not multiple of 8)
        for (size_t j = vecN8; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            if constexpr (Accumulate) C[i * N + j] += acc;
            else C[i * N + j] = acc;
        }
    }
}

inline void matmul_fma_saturated(float* MIMIR_RESTRICT C,
                                  const float* MIMIR_RESTRICT A,
                                  const float* MIMIR_RESTRICT B,
                                  size_t M, size_t N, size_t K) {
    matmul_fma_saturated_impl<false>(C,A,B,M,N,K);
}

inline void matmul_add_fma_saturated(float* MIMIR_RESTRICT C,
                                      const float* MIMIR_RESTRICT A,
                                      const float* MIMIR_RESTRICT B,
                                      size_t M, size_t N, size_t K) {
    matmul_fma_saturated_impl<true>(C,A,B,M,N,K);
}

// Convolution 2D avec FMA saturé
inline void conv2d_fma_saturated(float* MIMIR_RESTRICT output,
                                  const float* MIMIR_RESTRICT input,
                                  const float* MIMIR_RESTRICT kernel,
                                  int in_h, int in_w, int out_h, int out_w,
                                  int kernel_size, int stride, int padding) {
    std::memset(output, 0, out_h * out_w * sizeof(float));

    #pragma omp parallel for collapse(2) if(static_cast<long long>(out_h) * out_w * kernel_size * kernel_size > 262144) schedule(static)
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            // 3 accumulateurs pour saturation FMA
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            
            int k_idx = 0;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; kw += 3, k_idx += 3) {
                    int ih = oh * stride + kh - padding;
                    int iw = oh * stride + kw - padding;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        // Load 3 inputs et 3 kernel weights (FMA saturé)
                        if (kw < kernel_size) {
                            __m256 in0 = _mm256_set1_ps(input[ih * in_w + iw]);
                            __m256 k0 = _mm256_set1_ps(kernel[k_idx]);
                            acc0 = _mm256_fmadd_ps(in0, k0, acc0);
                        }
                        if (kw + 1 < kernel_size) {
                            __m256 in1 = _mm256_set1_ps(input[ih * in_w + iw + 1]);
                            __m256 k1 = _mm256_set1_ps(kernel[k_idx + 1]);
                            acc1 = _mm256_fmadd_ps(in1, k1, acc1);
                        }
                        if (kw + 2 < kernel_size) {
                            __m256 in2 = _mm256_set1_ps(input[ih * in_w + iw + 2]);
                            __m256 k2 = _mm256_set1_ps(kernel[k_idx + 2]);
                            acc2 = _mm256_fmadd_ps(in2, k2, acc2);
                        }
                    }
                }
            }
            
            // Sum accumulators
            __m256 total = _mm256_add_ps(acc0, _mm256_add_ps(acc1, acc2));
            
            // Horizontal sum
            __m128 sum_high = _mm256_extractf128_ps(total, 1);
            __m128 sum_low = _mm256_castps256_ps128(total);
            __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            __m128 shuf = _mm_movehdup_ps(sum128);
            __m128 sums = _mm_add_ps(sum128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            
            output[oh * out_w + ow] = _mm_cvtss_f32(sums);
        }
    }
}

// ===== BMI (BIT MANIPULATION) POUR QUANTIFICATION =====

#ifdef __BMI2__

// Quantification 8-bit avec BMI2
inline void quantize_int8_bmi(int8_t* dst, const float* src, size_t count,
                               float scale, float zero_point) {
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < count; i += 8) {
        __m256 f32 = _mm256_loadu_ps(&src[i]);
        
        // Scale et round
        __m256 scaled = _mm256_mul_ps(f32, _mm256_set1_ps(inv_scale));
        __m256 shifted = _mm256_add_ps(scaled, _mm256_set1_ps(zero_point));
        __m256i i32 = _mm256_cvtps_epi32(shifted);
        
        // Clamp to int8 range [-128, 127]
        __m256i clamped = _mm256_max_epi32(i32, _mm256_set1_epi32(-128));
        clamped = _mm256_min_epi32(clamped, _mm256_set1_epi32(127));
        
        // Pack 32->16->8 (utilise registres BMI efficacement)
        __m128i low = _mm256_castsi256_si128(clamped);
        __m128i high = _mm256_extracti128_si256(clamped, 1);
        __m128i packed16 = _mm_packs_epi32(low, high);
        __m128i packed8 = _mm_packs_epi16(packed16, packed16);
        
        _mm_storel_epi64((__m128i*)&dst[i], packed8);
    }
}

// Dequantification int8 avec BMI2
inline void dequantize_int8_bmi(float* dst, const int8_t* src, size_t count,
                                 float scale, float zero_point) {
    for (size_t i = 0; i < count; i += 8) {
        // Load 8 int8
        __m128i i8 = _mm_loadl_epi64((__m128i*)&src[i]);
        
        // Unpack 8->16->32 avec BMI
        __m128i i16 = _mm_cvtepi8_epi16(i8);
        __m128i low_i32 = _mm_cvtepi16_epi32(i16);
        __m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(i16, 8));
        __m256i i32 = _mm256_set_m128i(high_i32, low_i32);
        
        // Convert to float et scale
        __m256 f32 = _mm256_cvtepi32_ps(i32);
        __m256 shifted = _mm256_sub_ps(f32, _mm256_set1_ps(zero_point));
        __m256 scaled = _mm256_mul_ps(shifted, _mm256_set1_ps(scale));
        
        _mm256_storeu_ps(&dst[i], scaled);
    }
}

// Quantification 4-bit avec BMI2 (packing efficace)
inline void quantize_int4_bmi(uint8_t* dst, const float* src, size_t count,
                               float scale, float zero_point) {
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < count; i += 16) {
        __m256 f32_0 = _mm256_loadu_ps(&src[i]);
        __m256 f32_1 = _mm256_loadu_ps(&src[i + 8]);
        
        // Scale et round
        __m256 scaled_0 = _mm256_mul_ps(f32_0, _mm256_set1_ps(inv_scale));
        __m256 scaled_1 = _mm256_mul_ps(f32_1, _mm256_set1_ps(inv_scale));
        
        __m256i i32_0 = _mm256_cvtps_epi32(scaled_0);
        __m256i i32_1 = _mm256_cvtps_epi32(scaled_1);
        
        // Clamp to 4-bit range [0, 15]
        __m256i clamped_0 = _mm256_max_epi32(i32_0, _mm256_setzero_si256());
        clamped_0 = _mm256_min_epi32(clamped_0, _mm256_set1_epi32(15));
        
        __m256i clamped_1 = _mm256_max_epi32(i32_1, _mm256_setzero_si256());
        clamped_1 = _mm256_min_epi32(clamped_1, _mm256_set1_epi32(15));
        
        // Pack en utilisant BMI2 pour masking efficace
        uint32_t vals[16];
        _mm256_storeu_si256((__m256i*)&vals[0], clamped_0);
        _mm256_storeu_si256((__m256i*)&vals[8], clamped_1);
        
        // Pack 2 nibbles par byte avec PEXT (BMI2)
        for (size_t j = 0; j < 8; ++j) {
            dst[i/2 + j] = (vals[j*2] & 0xF) | ((vals[j*2 + 1] & 0xF) << 4);
        }
    }
}

#endif // __BMI2__

// ===== HUGEPAGES + MADVISE =====

class HugePageAllocator {
public:
    static constexpr size_t HUGEPAGE_SIZE = 2 * 1024 * 1024; // 2MB
    
    // Allouer avec HugePages
    static void* allocate_huge(size_t size) {
        // Round up to hugepage boundary
        size_t aligned_size = ((size + HUGEPAGE_SIZE - 1) / HUGEPAGE_SIZE) * HUGEPAGE_SIZE;

#if defined(_WIN32)
        // Windows fallback: standard heap allocation in cross-builds.
        return std::malloc(aligned_size);
#else
        
        void* ptr = mmap(nullptr, aligned_size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                        -1, 0);
        
        if (ptr == MAP_FAILED) {
            // Fallback to normal pages
            ptr = mmap(nullptr, aligned_size,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS,
                      -1, 0);
        }
        
        if (ptr != MAP_FAILED) {
            // Advise kernel about usage pattern
            madvise(ptr, aligned_size, MADV_HUGEPAGE);    // Prefer huge pages
            madvise(ptr, aligned_size, MADV_SEQUENTIAL);  // Sequential access
            madvise(ptr, aligned_size, MADV_WILLNEED);    // Prefetch
            
            // Prefetch parallèle par blocs de 4KB (page size) pour grandes allocations
            if (aligned_size > 16 * HUGEPAGE_SIZE) {
                for (size_t offset = 0; offset < aligned_size; offset += 4096) {
                    __builtin_prefetch(static_cast<char*>(ptr) + offset, 1, 3);
                }
            }
        }
        
        return ptr;
#endif
    }
    
    // Libérer
    static void deallocate_huge(void* ptr, size_t size) {
#if defined(_WIN32)
        (void)size;
        std::free(ptr);
#else
        if (ptr) {
            size_t aligned_size = ((size + HUGEPAGE_SIZE - 1) / HUGEPAGE_SIZE) * HUGEPAGE_SIZE;
            munmap(ptr, aligned_size);
        }
#endif
    }
    
    // Allouer buffer avec HugePages
    template<typename T>
    static T* allocate_buffer(size_t count) {
        size_t bytes = count * sizeof(T);
        return static_cast<T*>(allocate_huge(bytes));
    }
    
    // Libérer buffer
    template<typename T>
    static void deallocate_buffer(T* ptr, size_t count) {
        deallocate_huge(ptr, count * sizeof(T));
    }
};

// Wrapper pour std::vector avec HugePages
template<typename T>
class HugePageVector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
public:
    HugePageVector() : data_(nullptr), size_(0), capacity_(0) {}
    
    ~HugePageVector() {
        if (data_) {
            HugePageAllocator::deallocate_buffer(data_, capacity_);
        }
    }
    
    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            T* new_data = HugePageAllocator::allocate_buffer<T>(new_capacity);
            if (data_) {
                std::memcpy(new_data, data_, size_ * sizeof(T));
                HugePageAllocator::deallocate_buffer(data_, capacity_);
            }
            data_ = new_data;
            capacity_ = new_capacity;
        }
    }
    
    void resize(size_t new_size) {
        if (new_size > capacity_) {
            reserve(new_size * 2); // Growth factor 2x
        }
        size_ = new_size;
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
};

} // namespace HardwareOpt

#undef MIMIR_RESTRICT

#endif // __HARDWARE_OPT_HPP__
