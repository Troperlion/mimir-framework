#include "Model.hpp"
#include "HardwareOpt.hpp"
#include "SIMD_Ops.hpp"
#include "Layers.hpp"
#include "LayerTypes.hpp"
#include "LayerOps.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#ifdef ENABLE_VULKAN
#include "VulkanCompute.hpp"
#endif
#ifdef ENABLE_OPENCL
#include "OpenCLCompute.hpp"
#endif
#include "RngContext.hpp"
#include "RuntimeAllocator.hpp"
#include "LayerOpsExt.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <iostream>
#include <cmath>
#include <sstream>
#include <cstdlib>

#include <unordered_map>
#include <mutex>


namespace {
struct DistMoments {
    double mean = 0.0;
    double var = 0.0;
    double skew = 0.0;
};

static inline DistMoments compute_moments(const std::vector<float>& v) {
    DistMoments m;
    if (v.empty()) return m;
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x);
    m.mean = sum / static_cast<double>(v.size());

    double m2 = 0.0;
    double m3 = 0.0;
    for (float x : v) {
        const double d = static_cast<double>(x) - m.mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m.var = m2 / static_cast<double>(v.size());
    const double std = std::sqrt(std::max(0.0, m.var));
    if (std > 1e-12) {
        m.skew = (m3 / static_cast<double>(v.size())) / (std * std * std);
    } else {
        m.skew = 0.0;
    }
    return m;
}

static inline float sigmoid_scalar_f(float x) {
    // Stable-ish sigmoid
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

struct GlobalSSIM {
    double loss = 0.0;
    std::vector<float> grad; // d(loss)/d(pred), same size as pred
};

static GlobalSSIM ssim_global_hwc(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    int w,
    int h,
    int c,
    float k1,
    float k2,
    float L
) {
    GlobalSSIM out;
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int C = std::max(1, c);
    const size_t N = static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C);
    out.grad.assign(N, 0.0f);
    if (pred.size() < N || target.size() < N) {
        out.loss = 0.0;
        return out;
    }

    const double C1 = static_cast<double>(k1) * static_cast<double>(L);
    const double C2 = static_cast<double>(k2) * static_cast<double>(L);
    const double c1 = C1 * C1;
    const double c2 = C2 * C2;

    // Per-channel SSIM, averaged.
    double loss_sum = 0.0;

    const size_t HW = static_cast<size_t>(W) * static_cast<size_t>(H);
    const double inv_hw = 1.0 / static_cast<double>(std::max<size_t>(1, HW));
    for (int ch = 0; ch < C; ++ch) {
        // Compute means and second moments for channel
        double sum_x = 0.0, sum_y = 0.0;
        double sum_x2 = 0.0, sum_y2 = 0.0;
        double sum_xy = 0.0;

        for (int yy = 0; yy < H; ++yy) {
            const size_t row = static_cast<size_t>(yy) * static_cast<size_t>(W);
            for (int xx = 0; xx < W; ++xx) {
                const size_t idx = (row + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(ch);
                const double x = static_cast<double>(pred[idx]);
                const double y = static_cast<double>(target[idx]);
                sum_x += x;
                sum_y += y;
                sum_x2 += x * x;
                sum_y2 += y * y;
                sum_xy += x * y;
            }
        }

        const double mu_x = sum_x * inv_hw;
        const double mu_y = sum_y * inv_hw;
        const double ex2 = sum_x2 * inv_hw;
        const double ey2 = sum_y2 * inv_hw;
        const double exy = sum_xy * inv_hw;
        const double sigma_x2 = std::max(0.0, ex2 - mu_x * mu_x);
        const double sigma_y2 = std::max(0.0, ey2 - mu_y * mu_y);
        const double sigma_xy = exy - mu_x * mu_y;

        const double A = 2.0 * mu_x * mu_y + c1;
        const double B = 2.0 * sigma_xy + c2;
        const double Cc = mu_x * mu_x + mu_y * mu_y + c1;
        const double D = sigma_x2 + sigma_y2 + c2;

        const double eps = 1e-12;
        const double denom = (Cc * D);
        const double ssim = (denom != 0.0) ? ((A * B) / (denom + eps)) : 0.0;
        const double loss_ch = 1.0 - ssim;
        loss_sum += loss_ch;

        // dSSIM/dx_i (global version)
        const double invA = 1.0 / (A + eps);
        const double invB = 1.0 / (B + eps);
        const double invC = 1.0 / (Cc + eps);
        const double invD = 1.0 / (D + eps);
        const double ssim_safe = ssim;

        for (int yy = 0; yy < H; ++yy) {
            const size_t row = static_cast<size_t>(yy) * static_cast<size_t>(W);
            for (int xx = 0; xx < W; ++xx) {
                const size_t idx = (row + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(ch);
                const double x = static_cast<double>(pred[idx]);
                const double y = static_cast<double>(target[idx]);
                const double dmu_x = inv_hw;
                const double dA = 2.0 * mu_y * dmu_x;
                const double dC = 2.0 * mu_x * dmu_x;
                const double dsigma_xy = (static_cast<double>(y) - mu_y) * inv_hw;
                const double dB = 2.0 * dsigma_xy;
                const double dsigma_x2 = 2.0 * (x - mu_x) * inv_hw;
                const double dD = dsigma_x2;

                const double dssim = ssim_safe * (dA * invA + dB * invB - dC * invC - dD * invD);
                const double dloss = -dssim;
                out.grad[idx] += static_cast<float>(dloss / static_cast<double>(C));
            }
        }
    }

    out.loss = loss_sum / static_cast<double>(std::max(1, c));
    return out;
}

static inline void avgpool2x2_hwc(
    const std::vector<float>& in,
    int w,
    int h,
    int c,
    std::vector<float>& out,
    int& out_w,
    int& out_h
) {
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int C = std::max(1, c);
    out_w = std::max(1, W / 2);
    out_h = std::max(1, H / 2);
    out.assign(static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * static_cast<size_t>(C), 0.0f);
    for (int yy = 0; yy < out_h; ++yy) {
        for (int xx = 0; xx < out_w; ++xx) {
            const int in_y = yy * 2;
            const int in_x = xx * 2;
            for (int ch = 0; ch < C; ++ch) {
                double sum = 0.0;
                int count = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int sy = in_y + dy;
                        const int sx = in_x + dx;
                        if (sy >= H || sx >= W) continue;
                        const size_t sidx = (static_cast<size_t>(sy) * static_cast<size_t>(W) + static_cast<size_t>(sx)) * static_cast<size_t>(C) + static_cast<size_t>(ch);
                        sum += static_cast<double>(in[sidx]);
                        ++count;
                    }
                }
                const size_t didx = (static_cast<size_t>(yy) * static_cast<size_t>(out_w) + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(ch);
                out[didx] = static_cast<float>(sum / static_cast<double>(std::max(1, count)));
            }
        }
    }
}

static inline void avgpool2x2_back_hwc(
    const std::vector<float>& grad_out,
    int in_w,
    int in_h,
    int c,
    std::vector<float>& grad_in
) {
    const int W = std::max(1, in_w);
    const int H = std::max(1, in_h);
    const int C = std::max(1, c);
    const int out_w = std::max(1, W / 2);
    const int out_h = std::max(1, H / 2);
    grad_in.assign(static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C), 0.0f);
    if (grad_out.size() < static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * static_cast<size_t>(C)) return;

    for (int yy = 0; yy < out_h; ++yy) {
        for (int xx = 0; xx < out_w; ++xx) {
            const int in_y = yy * 2;
            const int in_x = xx * 2;
            int count = 0;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int sy = in_y + dy;
                    const int sx = in_x + dx;
                    if (sy >= H || sx >= W) continue;
                    ++count;
                }
            }
            const float inv = 1.0f / static_cast<float>(std::max(1, count));
            for (int ch = 0; ch < C; ++ch) {
                const size_t oidx = (static_cast<size_t>(yy) * static_cast<size_t>(out_w) + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(ch);
                const float g = grad_out[oidx] * inv;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int sy = in_y + dy;
                        const int sx = in_x + dx;
                        if (sy >= H || sx >= W) continue;
                        const size_t iidx = (static_cast<size_t>(sy) * static_cast<size_t>(W) + static_cast<size_t>(sx)) * static_cast<size_t>(C) + static_cast<size_t>(ch);
                        grad_in[iidx] += g;
                    }
                }
            }
        }
    }
}

struct LossWithGrad {
    double loss = 0.0;
    std::vector<float> grad;
};

struct Dct1DCache {
    int n = 0;
    std::vector<float> cos_nk;   // [n*N + k] = cos(pi*(n+0.5)*k/N)
    std::vector<float> alpha_k;  // orthonormal scaling
};

static inline const Dct1DCache& get_dct1d_cache(int N) {
    static std::mutex m;
    static std::unordered_map<int, Dct1DCache> caches;
    N = std::max(1, N);

    {
        std::lock_guard<std::mutex> lock(m);
        auto it = caches.find(N);
        if (it != caches.end()) return it->second;

        Dct1DCache c;
        c.n = N;
        c.cos_nk.assign(static_cast<size_t>(N) * static_cast<size_t>(N), 0.0f);
        c.alpha_k.assign(static_cast<size_t>(N), 0.0f);

        const double invN = 1.0 / static_cast<double>(N);
        for (int k = 0; k < N; ++k) {
            const double a = (k == 0) ? std::sqrt(invN) : std::sqrt(2.0 * invN);
            c.alpha_k[static_cast<size_t>(k)] = static_cast<float>(a);
        }

        const double pi = 3.1415926535897932384626433832795;
        for (int n = 0; n < N; ++n) {
            const double nn = static_cast<double>(n) + 0.5;
            for (int k = 0; k < N; ++k) {
                const double ang = (pi * nn * static_cast<double>(k)) * invN;
                c.cos_nk[static_cast<size_t>(n) * static_cast<size_t>(N) + static_cast<size_t>(k)] = static_cast<float>(std::cos(ang));
            }
        }

        auto [insIt, _] = caches.emplace(N, std::move(c));
        return insIt->second;
    }
}

static inline void dct2d_ortho_hwc(
    const std::vector<float>& in,
    int w,
    int h,
    int c,
    std::vector<float>& out
) {
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int C = std::max(1, c);
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C);
    out.assign(n, 0.0f);
    if (in.size() < n) return;

    const auto& cw = get_dct1d_cache(W);
    const auto& ch = get_dct1d_cache(H);

    std::vector<float> tmp_row(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0f);
    std::vector<float> tmp_col(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0f);

    for (int cc = 0; cc < C; ++cc) {
        // Row DCT
        for (int yy = 0; yy < H; ++yy) {
            for (int kk = 0; kk < W; ++kk) {
                double sum = 0.0;
                const float alpha = cw.alpha_k[static_cast<size_t>(kk)];
                for (int xx = 0; xx < W; ++xx) {
                    const size_t iidx = (static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(cc);
                    const float x = in[iidx];
                    const float co = cw.cos_nk[static_cast<size_t>(xx) * static_cast<size_t>(W) + static_cast<size_t>(kk)];
                    sum += static_cast<double>(x) * static_cast<double>(co);
                }
                tmp_row[static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(kk)] = static_cast<float>(static_cast<double>(alpha) * sum);
            }
        }

        // Col DCT
        for (int kkx = 0; kkx < W; ++kkx) {
            for (int kky = 0; kky < H; ++kky) {
                double sum = 0.0;
                const float alpha = ch.alpha_k[static_cast<size_t>(kky)];
                for (int yy = 0; yy < H; ++yy) {
                    const float v = tmp_row[static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(kkx)];
                    const float co = ch.cos_nk[static_cast<size_t>(yy) * static_cast<size_t>(H) + static_cast<size_t>(kky)];
                    sum += static_cast<double>(v) * static_cast<double>(co);
                }
                tmp_col[static_cast<size_t>(kky) * static_cast<size_t>(W) + static_cast<size_t>(kkx)] = static_cast<float>(static_cast<double>(alpha) * sum);
            }
        }

        // Store
        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                const size_t oidx = (static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(cc);
                out[oidx] = tmp_col[static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)];
            }
        }
    }
}

static inline void idct2d_ortho_hwc(
    const std::vector<float>& in,
    int w,
    int h,
    int c,
    std::vector<float>& out
) {
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int C = std::max(1, c);
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C);
    out.assign(n, 0.0f);
    if (in.size() < n) return;

    const auto& cw = get_dct1d_cache(W);
    const auto& ch = get_dct1d_cache(H);

    std::vector<float> tmp_row(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0f);
    std::vector<float> tmp_col(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0f);

    for (int cc = 0; cc < C; ++cc) {
        // Gather coefficients for this channel into tmp_col
        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                const size_t iidx = (static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(cc);
                tmp_col[static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)] = in[iidx];
            }
        }

        // Inverse Col (DCT-III)
        for (int kkx = 0; kkx < W; ++kkx) {
            for (int yy = 0; yy < H; ++yy) {
                double sum = 0.0;
                for (int kky = 0; kky < H; ++kky) {
                    const float alpha = ch.alpha_k[static_cast<size_t>(kky)];
                    const float v = tmp_col[static_cast<size_t>(kky) * static_cast<size_t>(W) + static_cast<size_t>(kkx)];
                    const float co = ch.cos_nk[static_cast<size_t>(yy) * static_cast<size_t>(H) + static_cast<size_t>(kky)];
                    sum += static_cast<double>(alpha) * static_cast<double>(v) * static_cast<double>(co);
                }
                tmp_row[static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(kkx)] = static_cast<float>(sum);
            }
        }

        // Inverse Row (DCT-III)
        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                double sum = 0.0;
                for (int kk = 0; kk < W; ++kk) {
                    const float alpha = cw.alpha_k[static_cast<size_t>(kk)];
                    const float v = tmp_row[static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(kk)];
                    const float co = cw.cos_nk[static_cast<size_t>(xx) * static_cast<size_t>(W) + static_cast<size_t>(kk)];
                    sum += static_cast<double>(alpha) * static_cast<double>(v) * static_cast<double>(co);
                }
                const size_t oidx = (static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)) * static_cast<size_t>(C) + static_cast<size_t>(cc);
                out[oidx] = static_cast<float>(sum);
            }
        }
    }
}

static inline LossWithGrad spectral_dct_l1_hwc(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    int w,
    int h,
    int c
) {
    LossWithGrad out;
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int C = std::max(1, c);
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C);
    out.grad.assign(n, 0.0f);
    if (pred.size() < n || target.size() < n) return out;

    std::vector<float> coeff_p;
    std::vector<float> coeff_t;
    dct2d_ortho_hwc(pred, W, H, C, coeff_p);
    dct2d_ortho_hwc(target, W, H, C, coeff_t);

    std::vector<float> grad_coeff(n, 0.0f);
    const double inv_n = 1.0 / static_cast<double>(std::max<size_t>(1, n));
    double sum_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = coeff_p[i] - coeff_t[i];
        sum_abs += static_cast<double>(std::abs(d));
        const float s = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
        grad_coeff[i] = static_cast<float>(inv_n) * s;
    }
    out.loss = sum_abs * inv_n;

    // Backprop to pixel space
    idct2d_ortho_hwc(grad_coeff, W, H, C, out.grad);
    return out;
}

static inline DistMoments compute_moments_prefix(const std::vector<float>& v, size_t off, int n) {
    DistMoments m;
    if (n <= 0) return m;
    const size_t vn = v.size();
    const size_t end = std::min(vn, off + static_cast<size_t>(n));
    if (end <= off) return m;
    const size_t count = end - off;

    double sum = 0.0;
    for (size_t i = off; i < end; ++i) sum += static_cast<double>(v[i]);
    m.mean = sum / static_cast<double>(count);

    double m2 = 0.0;
    double m3 = 0.0;
    for (size_t i = off; i < end; ++i) {
        const double d = static_cast<double>(v[i]) - m.mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m.var = m2 / static_cast<double>(count);
    const double std = std::sqrt(std::max(0.0, m.var));
    if (std > 1e-12) {
        m.skew = (m3 / static_cast<double>(count)) / (std * std * std);
    } else {
        m.skew = 0.0;
    }
    return m;
}

static inline double pearson_corr_prefix(const std::vector<float>& a, size_t a_off,
                                         const std::vector<float>& b, size_t b_off,
                                         int n) {
    if (n < 2) return 0.0;
    const size_t an = a.size();
    const size_t bn = b.size();
    const size_t a_end = std::min(an, a_off + static_cast<size_t>(n));
    const size_t b_end = std::min(bn, b_off + static_cast<size_t>(n));
    const size_t count = std::min(a_end - std::min(a_end, a_off), b_end - std::min(b_end, b_off));
    if (count < 2) return 0.0;

    double sa = 0.0, sb = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sa += static_cast<double>(a[a_off + i]);
        sb += static_cast<double>(b[b_off + i]);
    }
    const double ma = sa / static_cast<double>(count);
    const double mb = sb / static_cast<double>(count);

    double num = 0.0;
    double da = 0.0;
    double db = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double xa = static_cast<double>(a[a_off + i]) - ma;
        const double xb = static_cast<double>(b[b_off + i]) - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    const double den = std::sqrt(da) * std::sqrt(db);
    if (den <= 1e-18) return 0.0;
    const double r = num / den;
    return std::clamp(r, -1.0, 1.0);
}

static inline double mean_abs_adjacent_diff(const std::vector<float>& v) {
    if (v.size() < 2) return 0.0;
    double acc = 0.0;
    for (size_t i = 1; i < v.size(); ++i) {
        acc += std::abs(static_cast<double>(v[i]) - static_cast<double>(v[i - 1]));
    }
    return acc / static_cast<double>(v.size() - 1);
}

static inline double pearson_corr(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0;
    double sa = 0.0, sb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sa += static_cast<double>(a[i]);
        sb += static_cast<double>(b[i]);
    }
    const double ma = sa / static_cast<double>(n);
    const double mb = sb / static_cast<double>(n);
    double num = 0.0;
    double da = 0.0;
    double db = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double xa = static_cast<double>(a[i]) - ma;
        const double xb = static_cast<double>(b[i]) - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    const double den = std::sqrt(da) * std::sqrt(db);
    if (den <= 1e-18) return 0.0;
    const double r = num / den;
    return std::clamp(r, -1.0, 1.0);
}
} // namespace
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#include <cpuid.h>
#include <atomic>
#include <mutex>
#include <unordered_set>

// ============================================================================
// Registry centralisé (via LayerTypes.hpp)
// ============================================================================

using namespace LayerRegistry;

// ============================================================================
// Implémentation des méthodes Layer
// ============================================================================

// ============================================================================
// Détection des capacités CPU au runtime
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
static inline bool mimir_os_supports_avx_state() {
    // Vérifie que l'OS a activé le sauvegarde/restauration XMM/YMM (AVX).
    // Nécessaire pour utiliser AVX/AVX2/FMA/F16C sans #UD.
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;

    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx_hw = (ecx & (1u << 28)) != 0;
    if (!osxsave || !avx_hw) return false;

    // XGETBV(0): bits 1 (XMM) et 2 (YMM) doivent être à 1
    uint32_t xcr0_lo = 0;
    uint32_t xcr0_hi = 0;
    // GCC/Clang: xgetbv via asm
    __asm__ volatile ("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    (void)xcr0_hi;
    return (xcr0_lo & 0x6u) == 0x6u;
}
#endif

bool Model::hasAVX2() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            const bool os_ok = mimir_os_supports_avx_state();
            if (os_ok && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                result = (ebx & (1u << 5)) != 0; // EBX bit 5 = AVX2
            } else {
                result = false;
            }
        #endif
        detected = true;
    }
    
    return result;
}

bool Model::hasFMA() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            const bool os_ok = mimir_os_supports_avx_state();
            if (os_ok && __get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                result = (ecx & (1u << 12)) != 0; // ECX bit 12 = FMA
            } else {
                result = false;
            }
        #endif
        detected = true;
    }
    
    return result;
}

bool Model::hasF16C() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            const bool os_ok = mimir_os_supports_avx_state();
            if (os_ok && __get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                result = (ecx & (1u << 29)) != 0; // ECX bit 29 = F16C
            } else {
                result = false;
            }
        #endif
        detected = true;
    }
    
    return result;
}

bool Model::hasBMI2() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                result = (ebx & (1 << 8)) != 0; // EBX bit 8 = BMI2
            }
        #endif
        detected = true;
    }
    
    return result;
}

// Global compute engine (initialized on demand)
#ifdef ENABLE_VULKAN
static std::unique_ptr<VulkanCompute::ComputeEngine> g_compute_engine = nullptr;
#endif
static bool g_compute_available = false;

// OpenCL compute engine (initialized on demand)
#ifdef ENABLE_OPENCL
static std::unique_ptr<OpenCLCompute::ComputeEngine> g_opencl_engine = nullptr;
#endif
static bool g_opencl_available = false;

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
static inline bool env_flag_true(const char* name, bool default_value) {
    if (!name) return default_value;
    const char* v = std::getenv(name);
    if (!v) return default_value;
    if (v[0] == '\0') return default_value;
    if (v[0] == '0' && v[1] == '\0') return false;
    // tout autre valeur non vide => true
    return true;
}

static inline int env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return default_value;
    try {
        return std::stoi(v);
    } catch (...) {
        return default_value;
    }
}
}

// === constructeurs / destructeurs (déjà présents) ===
Model::Model()
    : tokenizer(20000), encoder(256, 20000), hasTokenizer(true), hasEncoder(true),
    max_ram_mb_(0)
{
    tw = 64; th = 64;
    // Encoder toujours présent + embeddings spéciaux (SEQ/MOD/MAG) disponibles.
    encoder.ensureSpecialEmbeddings();
    // Tenter d'initialiser le compute engine
    initializeComputeEngine();
    initializeOpenCLComputeEngine();
}

Model::~Model() {
    shutdownComputeEngine();
    shutdownOpenCLComputeEngine();
}

// ===== Hardware Acceleration =====

bool Model::hasVulkanCompute() const {
    return g_compute_available;
}

bool Model::hasOpenCLCompute() const {
    return g_opencl_available;
}

bool Model::initializeComputeEngine() {
#ifndef ENABLE_VULKAN
    g_compute_available = false;
    return false;
#else
    // Pattern init_once thread-safe avec atomic
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    // Permet de forcer le mode CPU pour diagnostic/stabilité.
    // Toute valeur non vide et différente de "0" désactive Vulkan.
    if (const char* v = std::getenv("MIMIR_DISABLE_VULKAN")) {
        if (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0')) {
            g_compute_available = false;
            g_compute_engine.reset();
            initialized.store(true, std::memory_order_release);
            std::cout << "⚠ Vulkan Compute disabled via MIMIR_DISABLE_VULKAN" << std::endl;
            return false;
        }
    }
    
    if (initialized.load(std::memory_order_acquire)) {
        return g_compute_available;
    }
    
    std::lock_guard<std::mutex> lock(init_mutex);
    
    // Double-check après lock
    if (initialized.load(std::memory_order_relaxed)) {
        return g_compute_available;
    }
    
    try {
        g_compute_engine = std::make_unique<VulkanCompute::ComputeEngine>();
        g_compute_available = g_compute_engine->initialize();
        
        if (g_compute_available) {
            std::cout << "✓ Vulkan Compute initialized" << std::endl;
        } else {
            std::cout << "⚠ Vulkan Compute initialization failed, using CPU fallback" << std::endl;
            g_compute_engine.reset();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠ Vulkan Compute unavailable: " << e.what() << std::endl;
        g_compute_available = false;
        g_compute_engine.reset();
    }
    
    initialized.store(true, std::memory_order_release);
    return g_compute_available;
#endif
}

bool Model::initializeOpenCLComputeEngine() {
#ifndef ENABLE_OPENCL
    g_opencl_available = false;
    return false;
#else
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    // Toute valeur non vide et différente de "0" désactive OpenCL.
    if (const char* v = std::getenv("MIMIR_DISABLE_OPENCL")) {
        if (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0')) {
            g_opencl_available = false;
            g_opencl_engine.reset();
            initialized.store(true, std::memory_order_release);
            std::cout << "⚠ OpenCL Compute disabled via MIMIR_DISABLE_OPENCL" << std::endl;
            return false;
        }
    }

    if (initialized.load(std::memory_order_acquire)) {
        return g_opencl_available;
    }

    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized.load(std::memory_order_relaxed)) {
        return g_opencl_available;
    }

    try {
        g_opencl_engine = std::make_unique<OpenCLCompute::ComputeEngine>();
        g_opencl_available = g_opencl_engine->initialize();
        if (g_opencl_available) {
            std::cout << "✓ OpenCL Compute initialized" << std::endl;
        } else {
            if (env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
                std::cout << "⚠ OpenCL Compute unavailable, using CPU fallback" << std::endl;
            }
            g_opencl_engine.reset();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠ OpenCL Compute unavailable: " << e.what() << std::endl;
        g_opencl_available = false;
        g_opencl_engine.reset();
    }

    initialized.store(true, std::memory_order_release);
    return g_opencl_available;
#endif
}

void Model::shutdownComputeEngine() {
#ifdef ENABLE_VULKAN
    if (g_compute_engine) {
        g_compute_engine->cleanup();
        g_compute_engine.reset();
        g_compute_available = false;
    }
#else
    g_compute_available = false;
#endif
}

void Model::shutdownOpenCLComputeEngine() {
#ifdef ENABLE_OPENCL
    if (g_opencl_engine) {
        g_opencl_engine->cleanup();
        g_opencl_engine.reset();
        g_opencl_available = false;
    }
#else
    g_opencl_available = false;
#endif
}

void Model::zeroGradients() {
    if (params_frozen_) {
        throw std::runtime_error("Model::zeroGradients: parameters are frozen");
    }
    // Réinitialiser tous les gradients des layers à zéro
    for (auto& layer : layers) {
        std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), 0.0f);
        std::fill(layer.grad_bias.begin(), layer.grad_bias.end(), 0.0f);
    }
    
    // Réinitialiser l'état du forward pour le prochain backward
    forward_state.clear();
}

Gradients Model::getGradients() const {
    Gradients grads;
    
    // Collecter tous les gradients des layers
    size_t param_idx = 0;
    for (const auto& layer : layers) {
        // Ajouter les gradients de poids
        for (const auto& grad : layer.grad_weights) {
            grads.param_grads[param_idx++] = grad;
        }
        
        // Ajouter les gradients de biais
        for (const auto& grad : layer.grad_bias) {
            grads.param_grads[param_idx++] = grad;
        }
    }
    
    return grads;
}

// ============================================================================
// TENSOR STORE (Multi-input/Branch Support)
// ============================================================================

const std::vector<float>& Model::getTensor(const std::string& name) const {
    auto it = tensor_store.find(name);
    if (it == tensor_store.end()) {
        std::cerr << "❌ ERROR: Tensor '" << name << "' not found in TensorStore" << std::endl;
        std::cerr << "Available tensors: ";
        for (const auto& kv : tensor_store) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

bool Model::hasTensor(const std::string& name) const {
    return tensor_store.find(name) != tensor_store.end();
}

const std::vector<int>& Model::getTensorInt(const std::string& name) const {
    auto it = tensor_store_int.find(name);
    if (it == tensor_store_int.end()) {
        std::cerr << "❌ ERROR: IntTensor '" << name << "' not found in IntTensorStore" << std::endl;
        std::cerr << "Available int tensors: ";
        for (const auto& kv : tensor_store_int) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Int tensor not found: " + name);
    }
    return it->second;
}

bool Model::hasTensorInt(const std::string& name) const {
    return tensor_store_int.find(name) != tensor_store_int.end();
}

std::vector<int>& Model::getTensorIntMutable(const std::string& name) {
    auto it = tensor_store_int.find(name);
    if (it == tensor_store_int.end()) {
        std::cerr << "❌ ERROR: IntTensor '" << name << "' not found in IntTensorStore" << std::endl;
        std::cerr << "Available int tensors: ";
        for (const auto& kv : tensor_store_int) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Int tensor not found: " + name);
    }
    return it->second;
}

std::vector<float>& Model::getTensorMutable(const std::string& name) {
    auto it = tensor_store.find(name);
    if (it == tensor_store.end()) {
        std::cerr << "❌ ERROR: Tensor '" << name << "' not found in TensorStore" << std::endl;
        std::cerr << "Available tensors: ";
        for (const auto& kv : tensor_store) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

void Model::storeTensor(const std::string& name, const std::vector<float>& data) {
    tensor_store[name] = data;
}

void Model::storeTensorInt(const std::string& name, const std::vector<int>& data) {
    tensor_store_int[name] = data;
}

void Model::storeTensor(const std::string& name, std::vector<float>&& data) {
    tensor_store[name] = std::move(data);
}

void Model::storeTensorInt(const std::string& name, std::vector<int>&& data) {
    tensor_store_int[name] = std::move(data);
}

std::vector<std::string> Model::getAvailableTensors() const {
    std::vector<std::string> names;
    names.reserve(tensor_store.size());
    for (const auto& kv : tensor_store) {
        names.push_back(kv.first);
    }
    return names;
}

std::vector<std::string> Model::getAvailableIntTensors() const {
    std::vector<std::string> names;
    names.reserve(tensor_store_int.size());
    for (const auto& kv : tensor_store_int) {
        names.push_back(kv.first);
    }
    return names;
}

void Model::clearTensorStore() {
    tensor_store.clear();
}

void Model::clearTensorStoreInt() {
    tensor_store_int.clear();
}

// === Forward pass (tokens int) ===

std::vector<float> Model::forwardPass(const std::vector<int> &input_ids, bool training) {
    return forwardPassView(input_ids, training);
}

const std::vector<float>& Model::forwardPassView(const std::vector<int> &input_ids, bool training) {
    // Stocker les ids int dans le store dédié, puis exécuter le même graphe.
    // Les layers Embedding liront tensor_store_int, les autres tensor_store (float).
    if (layers.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: no layers defined" << std::endl;
        static const std::vector<float> empty;
        return empty;
    }

    if (layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: weights not allocated" << std::endl;
        std::cerr << "    Call allocate_params() and init_weights() first" << std::endl;
        static const std::vector<float> empty;
        return empty;
    }

    clearTensorStore();
    clearTensorStoreInt();
    storeTensorInt("x", input_ids);
    storeTensorInt("__input__", input_ids);

    // Convention: encoder fournit mag/mod (float) pour conditionnement.
    if (hasEncoder) {
        const auto& mag = encoder.getMagEmbedding();
        const auto& mod = encoder.getModEmbedding();

        auto graph_uses_mag_mod = [&]() -> bool {
            for (const auto& lyr : layers) {
                for (const auto& in : lyr.inputs) {
                    if (in == "mag" || in == "mod") return true;
                }
            }
            return false;
        };

        if (graph_uses_mag_mod()) {
            int expected = 0;
            if (modelConfig.contains("d_model")) expected = std::max(0, modelConfig["d_model"].get<int>());
            else if (modelConfig.contains("text_d_model")) expected = std::max(0, modelConfig["text_d_model"].get<int>());
            if (expected > 0) {
                if (!mag.empty() && static_cast<int>(mag.size()) != expected) {
                    throw std::runtime_error("Encoder mag dim mismatch: have=" + std::to_string(mag.size()) + ", expected=" + std::to_string(expected));
                }
                if (!mod.empty() && static_cast<int>(mod.size()) != expected) {
                    throw std::runtime_error("Encoder mod dim mismatch: have=" + std::to_string(mod.size()) + ", expected=" + std::to_string(expected));
                }
            }
        }

        if (!mag.empty()) storeTensor("mag", mag);
        if (!mod.empty()) storeTensor("mod", mod);
    }

    if (training) {
        forward_state.clear();
        forward_state.is_valid = true;
    }

    if (training) {
        forward_state.layer_outputs.clear();
        forward_state.layer_outputs.reserve(layers.size());
        forward_state.layer_output_masks.clear();
        forward_state.layer_output_masks.reserve(layers.size());
        forward_state.layer_inputs_multi.clear();
        forward_state.layer_inputs_multi.reserve(layers.size());
        forward_state.layer_input_names.clear();
        forward_state.layer_input_names.reserve(layers.size());
        forward_state.layer_input_sizes_multi.clear();
        forward_state.layer_input_sizes_multi.reserve(layers.size());
    }

    auto needs_input_value_snapshot = [](LayerType t) -> bool {
        switch (t) {
            case LayerType::Linear:
            case LayerType::LayerNorm:
            case LayerType::GELU:
            case LayerType::MultiHeadAttention:
            case LayerType::SelfAttention:
            case LayerType::Embedding:
                return true;
            case LayerType::Add:
            case LayerType::Concat:
            case LayerType::Dropout:
            case LayerType::Dropout2d:
            case LayerType::ReLU:
            case LayerType::Tanh:
            case LayerType::Sigmoid:
            case LayerType::Softmax:
            case LayerType::LogSoftmax:
            default:
                return false;
        }
    };

    auto needs_output_mask = [](LayerType t) -> bool {
        switch (t) {
            case LayerType::Dropout:
            case LayerType::Dropout2d:
                return true;
            default:
                return false;
        }
    };

    MemoryGuard& guard = MemoryGuard::instance();
    const size_t guard_mb = guard.getLimit() / (1024ULL * 1024ULL);
    const size_t cap_mb = (max_ram_mb_ > 0) ? max_ram_mb_ : guard_mb;
    RuntimeAllocator allocator(guard, cap_mb);

    static const std::vector<std::string> kDefaultInputNameX = {"x"};

    // VIZ: dernier HxW "connu" pendant ce forward, utile pour les layers
    // qui ne renseignent pas output_width/output_height (ex: activations).
    int viz_last_w = 0;
    int viz_last_h = 0;

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];

        std::vector<float> layer_output;

        try {
            if (layer.type_enum == LayerType::Embedding) {
                const std::vector<std::string>& input_names = layer.inputs.empty() ? kDefaultInputNameX : layer.inputs;
                const std::vector<int>& ids = getTensorInt(input_names[0]);

                // Snapshot inputs for backward (ids stockés en float)
                if (training) {
                    forward_state.layer_input_names.push_back(input_names);

                    std::vector<size_t> sizes;
                    sizes.reserve(1);
                    sizes.push_back(ids.size());
                    forward_state.layer_input_sizes_multi.push_back(std::move(sizes));

                    std::vector<std::vector<float>> snap;
                    snap.reserve(1);
                    std::vector<float> ids_as_float;
                    ids_as_float.reserve(ids.size());
                    for (int v : ids) ids_as_float.push_back(static_cast<float>(v));
                    snap.push_back(std::move(ids_as_float));
                    forward_state.layer_inputs_multi.push_back(std::move(snap));

                    forward_state.layer_outputs.emplace_back();
                    forward_state.layer_output_masks.emplace_back();
                }

                const int vocab = std::max(1, layer.vocab_size);
                const int dim = std::max(1, layer.embed_dim);
                const int pad = layer.padding_idx;

                RUNTIME_CHECK(layer.getWeights() != nullptr, "Embedding: weights not initialized");
                RUNTIME_CHECK(static_cast<int>(layer.getWeightsSize()) >= vocab * dim, "Embedding: invalid weight size");

                const float* w = layer.getWeights();
                const size_t outN = static_cast<size_t>(ids.size()) * static_cast<size_t>(dim);
                auto output_handle = allocator.allocate_tensor(
                    {static_cast<int>(outN)},
                    "float32",
                    layer.name + "_output"
                );
                std::vector<float>& out = output_handle.data();
                out.assign(outN, 0.0f);

                for (size_t t = 0; t < ids.size(); ++t) {
                    const int id = ids[t];
                    if (pad >= 0 && id == pad) continue;
                    if (id < 0 || id >= vocab) continue;
                    const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                    const size_t base_o = t * static_cast<size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        out[base_o + static_cast<size_t>(d)] = w[base_w + static_cast<size_t>(d)];
                    }
                }

                layer_output = std::move(out);
            } else {
                // Pour les autres layers, reprendre le chemin float habituel:
                // récupérer les inputs float depuis tensor_store.
                const std::vector<std::string>& input_names = layer.inputs.empty() ? kDefaultInputNameX : layer.inputs;

                auto& inputs = scratch_input_ptrs_;
                inputs.clear();
                inputs.reserve(input_names.size());
                for (const auto& name : input_names) {
                    inputs.push_back(&getTensor(name));
                }

                // Snapshot inputs for backward
                if (training) {
                    forward_state.layer_input_names.push_back(input_names);

                    std::vector<size_t> sizes;
                    sizes.reserve(inputs.size());
                    for (auto* p : inputs) sizes.push_back(p ? p->size() : 0ULL);
                    forward_state.layer_input_sizes_multi.push_back(std::move(sizes));

                    std::vector<std::vector<float>> snap;
                    if (needs_input_value_snapshot(layer.type_enum)) {
                        snap.reserve(inputs.size());
                        for (auto* p : inputs) snap.push_back(*p);
                    }
                    forward_state.layer_inputs_multi.push_back(std::move(snap));

                    forward_state.layer_outputs.emplace_back();
                    forward_state.layer_output_masks.emplace_back();
                }

                const std::vector<float>& x = *inputs[0];

                // Réutiliser le switch-case existant en appelant une petite lambda locale
                // en se basant sur le même dispatch que forwardPass(float).
                switch (layer.type_enum) {
                    case LayerType::Identity: {
                        layer_output = x;
                        break;
                    }
                    case LayerType::Linear: {
                        layer_output = LayerOps::linear_forward(x, layer, training);
                        break;
                    }
                    case LayerType::LayerNorm: {
                        layer_output = LayerOps::layernorm_forward(x, layer, training);
                        break;
                    }
                    case LayerType::GELU: {
                        layer_output = LayerOps::gelu_forward(x);
                        break;
                    }
                    case LayerType::ReLU: {
                        layer_output = LayerOps::relu_forward(x);
                        break;
                    }
                    case LayerType::Tanh: {
                        layer_output = LayerOps::tanh_forward(x);
                        break;
                    }
                    case LayerType::Sigmoid: {
                        layer_output = LayerOps::sigmoid_forward(x);
                        break;
                    }
                    case LayerType::Softmax:
                    case LayerType::LogSoftmax: {
                        layer_output = LayerOps::softmax_forward(x, layer);
                        break;
                    }
                    case LayerType::Dropout:
                    case LayerType::Dropout2d: {
                        layer_output = LayerOps::dropout_forward(x, layer, training);
                        break;
                    }
                    case LayerType::Add: {
                        RUNTIME_CHECK(inputs.size() >= 2, "Add requires 2 inputs");
                        layer_output = LayerOps::add_forward(*inputs[0], *inputs[1]);
                        break;
                    }
                    case LayerType::Concat: {
                        RUNTIME_CHECK(inputs.size() >= 2, "Concat requires >=2 inputs");
                        std::vector<std::vector<float>> inputs_vec;
                        inputs_vec.reserve(inputs.size());
                        for (auto* p : inputs) inputs_vec.push_back(*p);
                        layer_output = LayerOps::concat_forward(inputs_vec, layer.concat_axis);
                        break;
                    }
                    case LayerType::MultiHeadAttention:
                    case LayerType::SelfAttention: {
                        RUNTIME_CHECK(layer.getWeights() != nullptr, "Attention: weights not initialized");
                        int seq_len = layer.seq_len > 0 ? layer.seq_len : 1;
                        int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : static_cast<int>(x.size());
                        int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
                        bool causal = layer.causal;

                        const float* weights = layer.getWeights();
                        int qkv_size = embed_dim * embed_dim * 3;
                        int out_size = embed_dim * embed_dim;
                        std::vector<float> qkv_weight(weights, weights + qkv_size);
                        std::vector<float> out_weight(weights + qkv_size, weights + qkv_size + out_size);

                        if (layer.type_enum == LayerType::SelfAttention) {
                            layer_output = LayerOps::self_attention_forward(x, qkv_weight, out_weight, seq_len, embed_dim, num_heads, causal);
                        } else {
                            layer_output = LayerOps::multihead_attention_forward(x, qkv_weight, out_weight, seq_len, embed_dim, num_heads, causal);
                        }
                        break;
                    }
                    default: {
                        // Fallback: appeler le forward float standard en utilisant le tensor_store "x".
                        // NOTE: on ne veut pas dupliquer tout le switch ici.
                        // Stratégie: exécuter un forward float complet si ce layer n'est pas dans la liste.
                        // Pour éviter un coût élevé, on force l'utilisateur à ne pas mélanger trop de types ici.
                        RUNTIME_ERROR_STRICT(
                            "forwardPass(int): layer type not supported in int path: " + type_to_string(layer.type_enum)
                        );
                        break;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ ERROR in layer " << layer_idx << " (" << layer.name
                      << ", type: " << type_to_string(layer.type_enum) << "): "
                      << e.what() << std::endl;
            throw;
        }

        std::string output_name = layer.output.empty() ? "x" : layer.output;

        // Masques/snapshots output pour le backward (sans copier toutes les sorties)
        if (training) {
            // On a déjà poussé des placeholders (outputs/masks) ci-dessus.
            if (needs_output_mask(layer.type_enum)) {
                std::vector<uint8_t> mask;
                mask.resize(layer_output.size());
                for (size_t i = 0; i < layer_output.size(); ++i) {
                    mask[i] = (layer_output[i] != 0.0f) ? 1 : 0;
                }
                forward_state.layer_output_masks.back() = std::move(mask);
            }
        }

        storeTensor(output_name, std::move(layer_output));
    }

    return getTensor("x");
}

// === Forward pass (multi-entrées: floats + tokens int) ===

std::vector<float> Model::forwardPassNamed(
    const std::unordered_map<std::string, std::vector<float>>& float_inputs,
    const std::unordered_map<std::string, std::vector<int>>& int_inputs,
    bool training
) {
    return forwardPassNamedView(float_inputs, int_inputs, training);
}

void Model::addVizTapFrame(VizFrame vf) {
    if (!viz_taps_enabled_) return;
    if (viz_taps_max_frames_ <= 0) return;
    if (vf.w <= 0 || vf.h <= 0 || vf.channels <= 0) return;
    if (vf.pixels.empty()) return;

    // Dedup by label (keep last)
    auto it = std::find_if(viz_taps_.begin(), viz_taps_.end(), [&](const VizFrame& existing) {
        return existing.label == vf.label;
    });
    if (it != viz_taps_.end()) {
        *it = std::move(vf);
        return;
    }

    if (static_cast<int>(viz_taps_.size()) < viz_taps_max_frames_) {
        viz_taps_.push_back(std::move(vf));
        return;
    }

    // Evict last (best-effort) to guarantee key frames can be shown.
    viz_taps_.back() = std::move(vf);
}

const std::vector<float>& Model::forwardPassNamedView(
    const std::unordered_map<std::string, std::vector<float>>& float_inputs,
    const std::unordered_map<std::string, std::vector<int>>& int_inputs,
    bool training
) {
    // On injecte les entrées supplémentaires via un canal interne, puis on
    // réutilise le forward float principal (qui exécute le graphe complet).
    pending_float_inputs_ = float_inputs;
    pending_int_inputs_ = int_inputs;

    auto itx = float_inputs.find("x");
    if (itx != float_inputs.end()) {
        return forwardPassView(itx->second, training);
    }

    // Fallback: si "x" absent, utiliser __input__ si présent, sinon vecteur vide.
    auto iti = float_inputs.find("__input__");
    if (iti != float_inputs.end()) {
        return forwardPassView(iti->second, training);
    }

    static const std::vector<float> empty;
    return forwardPassView(empty, training);
}

Model::StepStats Model::trainStepNamed(
    const std::unordered_map<std::string, std::vector<float>>& float_inputs,
    const std::unordered_map<std::string, std::vector<int>>& int_inputs,
    const std::vector<float>& target,
    Optimizer& opt,
    float learning_rate
) {
    if (layers.empty()) {
        throw std::runtime_error("Model::trainStepNamed: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("Model::trainStepNamed: weights not allocated (call allocateParams/initWeights)");
    }

    zeroGradients();

    const std::vector<float>& prediction = forwardPassNamedView(float_inputs, int_inputs, true);

    StepStats stats;
    stats.loss = computeLoss(prediction, target, "mse");

    // Métriques supplémentaires: best-effort sur les distributions globales.
    // Ces métriques servent au monitoring (Htop/Viz) et ne modifient pas l'entraînement.
    {
        const auto mp = compute_moments(prediction);
        const auto mt = compute_moments(target);
        const double vp = std::max(mp.var, 1e-12);
        const double vt = std::max(mt.var, 1e-12);

        // KL(Nt || Np)
        const double kl = 0.5 * (std::log(vp / vt) + (vt + (mt.mean - mp.mean) * (mt.mean - mp.mean)) / vp - 1.0);
        stats.kl_divergence = static_cast<float>(std::max(0.0, kl));

        // Wasserstein-2 entre gaussiennes (1D)
        const double w2 = (mt.mean - mp.mean) * (mt.mean - mp.mean) + (std::sqrt(vt) - std::sqrt(vp)) * (std::sqrt(vt) - std::sqrt(vp));
        stats.wasserstein = static_cast<float>(std::sqrt(std::max(0.0, w2)));

        // Entropie gaussienne: 0.5 * log(2πeσ²)
        const double Hp = 0.5 * std::log(2.0 * M_PI * M_E * vp);
        const double Ht = 0.5 * std::log(2.0 * M_PI * M_E * vt);
        stats.entropy_diff = static_cast<float>(Hp - Ht);

        // Mismatch de moments: skewness difference (|skew_p - skew_t|)
        stats.moment_mismatch = static_cast<float>(std::abs(mp.skew - mt.skew));

        // Cohérence spatiale: différence de "total variation" 1D (adjacent diffs)
        const double tvp = mean_abs_adjacent_diff(prediction);
        const double tvt = mean_abs_adjacent_diff(target);
        stats.spatial_coherence = static_cast<float>(std::abs(tvp - tvt));

        // Consistency: corrélation prediction/target (Pearson)
        stats.temporal_consistency = static_cast<float>(pearson_corr(prediction, target));
    }

    scratch_loss_grad_.clear();
    computeLossGradientInto(prediction, target, scratch_loss_grad_, "mse");
    backwardPass(scratch_loss_grad_);

    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (const auto& layer : layers) {
        for (float g : layer.grad_weights) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
    }
    stats.grad_norm = static_cast<float>(std::sqrt(sum_sq));
    stats.grad_max_abs = max_abs;

    optimizerStep(opt, learning_rate);
    return stats;
}

Model::VAEStepStats Model::trainStepVAE(const std::vector<float>& x, Optimizer& opt, float learning_rate) {
    if (params_frozen_) {
        throw std::runtime_error("Model::trainStepVAE: parameters are frozen");
    }
    if (layers.empty()) {
        throw std::runtime_error("Model::trainStepVAE: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("Model::trainStepVAE: weights not allocated (call allocateParams/initWeights)");
    }

    // Read config
    int image_dim = 0;
    if (modelConfig.contains("image_dim")) {
        image_dim = std::max(0, modelConfig["image_dim"].get<int>());
    }
    if (image_dim <= 0) image_dim = static_cast<int>(x.size());

    int latent_dim = 0;
    if (modelConfig.contains("latent_dim")) {
        latent_dim = std::max(0, modelConfig["latent_dim"].get<int>());
    }

    float kl_beta = 1.0f;
    if (modelConfig.contains("kl_beta")) {
        kl_beta = modelConfig["kl_beta"].get<float>();
    } else if (modelConfig.contains("vae_kl_beta")) {
        kl_beta = modelConfig["vae_kl_beta"].get<float>();
    }
    kl_beta = std::max(0.0f, kl_beta);

    int kl_warmup_steps = 0;
    if (modelConfig.contains("kl_warmup_steps")) {
        kl_warmup_steps = std::max(0, modelConfig["kl_warmup_steps"].get<int>());
    }

    // Optional marker-driven scaling of reconstruction loss.
    // These are intended as training "marqueurs" (KL/Wass/Temp) to modulate the recon signal.
    float marker_wass_scale = 0.0f;
    float marker_temp_scale = 0.0f;
    float marker_scale_max = 10.0f;
    int marker_warmup_steps = 0;
    if (modelConfig.contains("marker_wass_scale")) marker_wass_scale = modelConfig["marker_wass_scale"].get<float>();
    if (modelConfig.contains("marker_temp_scale")) marker_temp_scale = modelConfig["marker_temp_scale"].get<float>();
    if (modelConfig.contains("marker_scale_max")) marker_scale_max = modelConfig["marker_scale_max"].get<float>();
    if (modelConfig.contains("marker_warmup_steps")) marker_warmup_steps = std::max(0, modelConfig["marker_warmup_steps"].get<int>());
    marker_wass_scale = std::max(0.0f, marker_wass_scale);
    marker_temp_scale = std::max(0.0f, marker_temp_scale);
    marker_scale_max = std::max(1.0f, marker_scale_max);

    float logvar_min = -10.0f;
    float logvar_max = 10.0f;
    if (modelConfig.contains("logvar_clip_min")) {
        logvar_min = modelConfig["logvar_clip_min"].get<float>();
    }
    if (modelConfig.contains("logvar_clip_max")) {
        logvar_max = modelConfig["logvar_clip_max"].get<float>();
    }
    if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

    // grad clipping is handled in optimizerStep (supports gradient accumulation)

    const float progress = (kl_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(kl_warmup_steps))
        : 1.0f;
    const float beta_eff = kl_beta * progress;

    const float marker_progress = (marker_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(marker_warmup_steps))
        : 1.0f;
    const float marker_wass_eff = marker_wass_scale * marker_progress;
    const float marker_temp_eff = marker_temp_scale * marker_progress;

    // Forward
    zeroGradients();
    const std::vector<float>& pred = forwardPassView(x, true);
    const int out_dim = static_cast<int>(pred.size());

    // Infer latent_dim if missing
    if (latent_dim <= 0) {
        if (out_dim > image_dim + 2 && ((out_dim - image_dim) % 2) == 0) {
            latent_dim = std::max(1, (out_dim - image_dim) / 2);
        }
    }

    if (image_dim <= 0 || out_dim < image_dim + 2) {
        throw std::runtime_error("Model::trainStepVAE: invalid output/image_dim (out_dim=" + std::to_string(out_dim) + ", image_dim=" + std::to_string(image_dim) + ")");
    }
    if (latent_dim <= 0 || out_dim < image_dim + 2 * latent_dim) {
        // Fallback robust: use whatever tail we have.
        const int tail = out_dim - image_dim;
        if (tail < 2 || (tail % 2) != 0) {
            throw std::runtime_error("Model::trainStepVAE: cannot infer latent_dim from output tail (tail=" + std::to_string(tail) + ")");
        }
        latent_dim = std::max(1, tail / 2);
    }

    const int recon_n = std::min(image_dim, static_cast<int>(x.size()));
    std::string recon_loss = "mse";
    if (modelConfig.contains("recon_loss")) {
        try {
            recon_loss = modelConfig["recon_loss"].get<std::string>();
        } catch (...) {
        }
    }

    // Optional additive recon components
    float ssim_weight = 0.0f;
    float spectral_weight = 0.0f;
    float perceptual_weight = 0.0f;
    float adv_weight = 0.0f;
    if (modelConfig.contains("ssim_weight")) ssim_weight = std::max(0.0f, modelConfig["ssim_weight"].get<float>());
    if (modelConfig.contains("spectral_weight")) spectral_weight = std::max(0.0f, modelConfig["spectral_weight"].get<float>());
    if (modelConfig.contains("perceptual_weight")) perceptual_weight = std::max(0.0f, modelConfig["perceptual_weight"].get<float>());
    // Adversarial (discriminateur) retiré pour VAEConv. On ignore adv_weight si présent.
    (void)adv_weight;

    // Image shape for image-specific losses
    int img_w = 0, img_h = 0, img_c = 0;
    if (modelConfig.contains("image_w")) img_w = std::max(0, modelConfig["image_w"].get<int>());
    if (modelConfig.contains("image_h")) img_h = std::max(0, modelConfig["image_h"].get<int>());
    if (modelConfig.contains("image_c")) img_c = std::max(0, modelConfig["image_c"].get<int>());
    if (img_w <= 0 || img_h <= 0 || img_c <= 0) {
        img_c = 1;
        img_w = recon_n;
        img_h = 1;
    }
    const bool recon_is_hwc = (recon_n == img_w * img_h * img_c);

    // Loss: recon (avg) + beta * KL (avg)
    // Recon = base pixel loss + optional additive losses.
    double recon = 0.0;

    // Parameters for some losses
    float huber_delta = 1.0f;
    if (modelConfig.contains("huber_delta")) huber_delta = std::max(1e-6f, modelConfig["huber_delta"].get<float>());
    if (modelConfig.contains("smoothl1_delta")) huber_delta = std::max(1e-6f, modelConfig["smoothl1_delta"].get<float>());
    if (modelConfig.contains("smoothl1_beta")) huber_delta = std::max(1e-6f, modelConfig["smoothl1_beta"].get<float>());

    float charbonnier_eps = 1e-3f;
    if (modelConfig.contains("charbonnier_eps")) charbonnier_eps = std::max(1e-12f, modelConfig["charbonnier_eps"].get<float>());

    float nll_sigma = 1.0f;
    if (modelConfig.contains("nll_sigma")) nll_sigma = std::max(1e-6f, modelConfig["nll_sigma"].get<float>());
    if (modelConfig.contains("gaussian_nll_sigma")) nll_sigma = std::max(1e-6f, modelConfig["gaussian_nll_sigma"].get<float>());

    std::vector<float> grad_recon(static_cast<size_t>(recon_n), 0.0f);

    auto add_pixel_loss_and_grad = [&](const std::string& type, double weight) {
        if (weight <= 0.0) return;
        const double inv_n = 1.0 / static_cast<double>(std::max(1, recon_n));
        double lsum = 0.0;
        if (type == "l1" || type == "mae") {
            for (int i = 0; i < recon_n; ++i) {
                const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>(x[static_cast<size_t>(i)]);
                lsum += std::abs(d);
                const float df = static_cast<float>(d);
                const float s = (df > 0.0f) ? 1.0f : (df < 0.0f ? -1.0f : 0.0f);
                grad_recon[static_cast<size_t>(i)] += static_cast<float>(weight * inv_n) * s;
            }
        } else if (type == "huber" || type == "smoothl1") {
            const float dlt = huber_delta;
            for (int i = 0; i < recon_n; ++i) {
                const float diff = pred[static_cast<size_t>(i)] - x[static_cast<size_t>(i)];
                const float ad = std::abs(diff);
                if (ad <= dlt) {
                    lsum += 0.5 * static_cast<double>(diff) * static_cast<double>(diff);
                    grad_recon[static_cast<size_t>(i)] += static_cast<float>(weight * inv_n) * diff;
                } else {
                    lsum += static_cast<double>(dlt) * (static_cast<double>(ad) - 0.5 * static_cast<double>(dlt));
                    const float s = (diff > 0.0f) ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                    grad_recon[static_cast<size_t>(i)] += static_cast<float>(weight * inv_n) * dlt * s;
                }
            }
        } else if (type == "charbonnier") {
            const float eps = charbonnier_eps;
            for (int i = 0; i < recon_n; ++i) {
                const float diff = pred[static_cast<size_t>(i)] - x[static_cast<size_t>(i)];
                const float denom = std::sqrt(diff * diff + eps * eps);
                lsum += static_cast<double>(denom);
                grad_recon[static_cast<size_t>(i)] += (denom > 0.0f)
                    ? (static_cast<float>(weight * inv_n) * (diff / denom))
                    : 0.0f;
            }
        } else if (type == "gaussian_nll" || type == "nll_gaussian") {
            const double inv_var = 1.0 / (static_cast<double>(nll_sigma) * static_cast<double>(nll_sigma));
            const double log_var = std::log(static_cast<double>(nll_sigma) * static_cast<double>(nll_sigma));
            for (int i = 0; i < recon_n; ++i) {
                const double diff = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>(x[static_cast<size_t>(i)]);
                lsum += 0.5 * (diff * diff * inv_var + log_var);
                grad_recon[static_cast<size_t>(i)] += static_cast<float>(weight * inv_n) * static_cast<float>(diff * inv_var);
            }
        } else {
            // default: mse
            for (int i = 0; i < recon_n; ++i) {
                const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>(x[static_cast<size_t>(i)]);
                lsum += d * d;
                grad_recon[static_cast<size_t>(i)] += static_cast<float>(weight * (2.0 * inv_n)) * static_cast<float>(d);
            }
        }
        recon += weight * (lsum * inv_n);
    };

    // Base recon loss
    add_pixel_loss_and_grad(recon_loss, 1.0);

    // SSIM / MS-SSIM (global, differentiable)
    if (ssim_weight > 0.0f && recon_is_hwc) {
        float k1 = 0.01f, k2 = 0.03f, L = 2.0f;
        if (modelConfig.contains("ssim_k1")) k1 = modelConfig["ssim_k1"].get<float>();
        if (modelConfig.contains("ssim_k2")) k2 = modelConfig["ssim_k2"].get<float>();
        if (modelConfig.contains("ssim_L")) L = modelConfig["ssim_L"].get<float>();

        std::string ssim_mode = "ssim";
        if (modelConfig.contains("ssim_mode")) {
            try { ssim_mode = modelConfig["ssim_mode"].get<std::string>(); } catch (...) {}
        }

        if (ssim_mode == "ms_ssim" || ssim_mode == "ms-ssim" || recon_loss == "ms_ssim" || recon_loss == "ms-ssim") {
            // Build scales
            std::vector<std::vector<float>> x_scales;
            std::vector<std::vector<float>> t_scales;
            std::vector<int> ws;
            std::vector<int> hs;
            x_scales.emplace_back(pred.begin(), pred.begin() + recon_n);
            t_scales.emplace_back(x.begin(), x.begin() + recon_n);
            ws.push_back(img_w);
            hs.push_back(img_h);
            int cur_w = img_w, cur_h = img_h;
            for (int s = 1; s < 5; ++s) {
                if (cur_w < 8 || cur_h < 8) break;
                std::vector<float> xd, td;
                int nw = 0, nh = 0;
                avgpool2x2_hwc(x_scales.back(), cur_w, cur_h, img_c, xd, nw, nh);
                avgpool2x2_hwc(t_scales.back(), cur_w, cur_h, img_c, td, nw, nh);
                cur_w = nw;
                cur_h = nh;
                x_scales.push_back(std::move(xd));
                t_scales.push_back(std::move(td));
                ws.push_back(cur_w);
                hs.push_back(cur_h);
            }

            // Weights (MS-SSIM paper defaults, truncated/renorm)
            static const double w_default[5] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
            const int S = static_cast<int>(x_scales.size());
            double wsum = 0.0;
            for (int s = 0; s < S; ++s) wsum += w_default[s];
            if (wsum <= 0.0) wsum = 1.0;

            // Accumulate loss and gradients
            double ms_loss = 0.0;
            std::vector<float> grad_full(static_cast<size_t>(recon_n), 0.0f);
            for (int s = 0; s < S; ++s) {
                const double ws_norm = w_default[s] / wsum;
                const auto r = ssim_global_hwc(x_scales[s], t_scales[s], ws[s], hs[s], img_c, k1, k2, L);
                ms_loss += ws_norm * r.loss;

                // Backprop grad to full resolution via avgpool adjoint
                std::vector<float> g = r.grad;
                for (int back = s - 1; back >= 0; --back) {
                    std::vector<float> up;
                    avgpool2x2_back_hwc(g, ws[back], hs[back], img_c, up);
                    g.swap(up);
                }
                if (g.size() == grad_full.size()) {
                    for (size_t i = 0; i < grad_full.size(); ++i) {
                        grad_full[i] += static_cast<float>(ws_norm) * g[i];
                    }
                }
            }
            recon += static_cast<double>(ssim_weight) * ms_loss;
            for (int i = 0; i < recon_n; ++i) {
                grad_recon[static_cast<size_t>(i)] += ssim_weight * grad_full[static_cast<size_t>(i)];
            }
        } else {
            const std::vector<float> pred_recon(pred.begin(), pred.begin() + recon_n);
            const std::vector<float> tgt_recon(x.begin(), x.begin() + recon_n);
            const auto r = ssim_global_hwc(pred_recon, tgt_recon, img_w, img_h, img_c, k1, k2, L);
            recon += static_cast<double>(ssim_weight) * r.loss;
            for (int i = 0; i < recon_n; ++i) {
                grad_recon[static_cast<size_t>(i)] += ssim_weight * r.grad[static_cast<size_t>(i)];
            }
        }
    }

    // Spectral / frequency-domain loss (DCT L1, multi-scale via avgpool)
    if (spectral_weight > 0.0f && recon_is_hwc) {
        int spectral_scales = 1;
        if (modelConfig.contains("spectral_scales")) {
            spectral_scales = std::max(1, modelConfig["spectral_scales"].get<int>());
        }

        std::vector<std::vector<float>> x_scales;
        std::vector<std::vector<float>> t_scales;
        std::vector<int> ws;
        std::vector<int> hs;
        x_scales.emplace_back(pred.begin(), pred.begin() + recon_n);
        t_scales.emplace_back(x.begin(), x.begin() + recon_n);
        ws.push_back(img_w);
        hs.push_back(img_h);

        int cur_w = img_w, cur_h = img_h;
        for (int s = 1; s < spectral_scales; ++s) {
            if (cur_w < 8 || cur_h < 8) break;
            std::vector<float> xd, td;
            int nw = 0, nh = 0;
            avgpool2x2_hwc(x_scales.back(), cur_w, cur_h, img_c, xd, nw, nh);
            avgpool2x2_hwc(t_scales.back(), cur_w, cur_h, img_c, td, nw, nh);
            cur_w = nw;
            cur_h = nh;
            x_scales.push_back(std::move(xd));
            t_scales.push_back(std::move(td));
            ws.push_back(cur_w);
            hs.push_back(cur_h);
        }

        const int S = static_cast<int>(x_scales.size());
        double wsum = 0.0;
        for (int s = 0; s < S; ++s) wsum += std::pow(0.5, static_cast<double>(s));
        if (wsum <= 0.0) wsum = 1.0;

        double spec_loss = 0.0;
        std::vector<float> grad_full(static_cast<size_t>(recon_n), 0.0f);
        for (int s = 0; s < S; ++s) {
            const double ws_norm = std::pow(0.5, static_cast<double>(s)) / wsum;
            const auto r = spectral_dct_l1_hwc(x_scales[s], t_scales[s], ws[s], hs[s], img_c);
            spec_loss += ws_norm * r.loss;

            // Backprop grad to full resolution via avgpool adjoint
            std::vector<float> g = r.grad;
            for (int back = s - 1; back >= 0; --back) {
                std::vector<float> up;
                avgpool2x2_back_hwc(g, ws[back], hs[back], img_c, up);
                g.swap(up);
            }
            if (g.size() == grad_full.size()) {
                for (size_t i = 0; i < grad_full.size(); ++i) {
                    grad_full[i] += static_cast<float>(ws_norm) * g[i];
                }
            }
        }

        recon += static_cast<double>(spectral_weight) * spec_loss;
        for (int i = 0; i < recon_n; ++i) {
            grad_recon[static_cast<size_t>(i)] += spectral_weight * grad_full[static_cast<size_t>(i)];
        }
    }

    // Perceptual loss (requires a dedicated architecture with feature output)
    if (perceptual_weight > 0.0f && recon_is_hwc) {
        std::string p_arch = "vgg16_feat";
        if (modelConfig.contains("perceptual_arch")) {
            try { p_arch = modelConfig["perceptual_arch"].get<std::string>(); } catch (...) {}
        }

        // Lazy init aux model
        if (!aux_perceptual_) {
            json pcfg = ModelArchitectures::defaultConfig(p_arch);
            pcfg["image_w"] = img_w;
            pcfg["image_h"] = img_h;
            pcfg["image_c"] = img_c;
            if (modelConfig.contains("perceptual_base_channels")) {
                int bc = std::max(1, modelConfig["perceptual_base_channels"].get<int>());
                // Compat: vgg16_feat force base_channels>=4.
                if (p_arch == "vgg16_feat") bc = std::max(4, bc);
                pcfg["base_channels"] = bc;
            }
            aux_perceptual_ = ModelArchitectures::create(p_arch, pcfg);
            aux_perceptual_->allocateParams();

            // IMPORTANT: si aucun checkpoint n'est fourni, il faut au minimum
            // initialiser les poids; sinon le réseau peut produire des features
            // dégénérées (ex: zéros) et la loss perceptuelle n'apporte aucun
            // signal => pas de "punition"/"récompense".
            // On utilise un seed fixe pour garder un comportement déterministe.
            try {
                aux_perceptual_->initializeWeights("xavier", 1337u);
            } catch (...) {
                // Best-effort: si l'init échoue, on continue (checkpoint peut encore charger).
            }

            std::string ckpt;
            if (modelConfig.contains("perceptual_checkpoint")) {
                try { ckpt = modelConfig["perceptual_checkpoint"].get<std::string>(); } catch (...) {}
            }
            if (!ckpt.empty()) {
                Mimir::Serialization::LoadOptions opts;
                opts.format = Mimir::Serialization::detect_format(ckpt);
                opts.load_tokenizer = false;
                opts.load_encoder = false;
                opts.load_optimizer = false;
                opts.strict_mode = false;
                opts.validate_checksums = false;
                std::string err;
                if (!Mimir::Serialization::load_checkpoint(*aux_perceptual_, ckpt, opts, &err)) {
                    // Helpful fallback: for raw_folder checkpoints, infer the base_channels from
                    // the checkpoint architecture (e.g. base=4 vs base=8 mismatch) and retry.
                    bool retried = false;
                    if (opts.format == Mimir::Serialization::CheckpointFormat::RawFolder) {
                        try {
                            namespace fs = std::filesystem;
                            const fs::path arch_path = fs::path(ckpt) / "model" / "architecture.json";
                            if (fs::exists(arch_path)) {
                                std::ifstream f(arch_path);
                                json arch;
                                f >> arch;

                                int inferred_base = 0;
                                if (arch.is_object() && arch.contains("layers") && arch["layers"].is_array()) {
                                    for (const auto& L : arch["layers"]) {
                                        if (!L.is_object()) continue;
                                        const std::string name = L.value("name", std::string());
                                        if (name == "vgg16_feat/b1/c1") {
                                            inferred_base = L.value("out_channels", 0);
                                            break;
                                        }
                                    }
                                }

                                const int cfg_base = pcfg.contains("base_channels") ? (int)pcfg["base_channels"].get<int>() : 0;
                                if (inferred_base > 0 && inferred_base != cfg_base) {
                                    std::cerr << "⚠️  Perceptual checkpoint base_channels mismatch (cfg=" << cfg_base
                                              << ", ckpt=" << inferred_base << "). Rebuilding aux model and retrying load..." << std::endl;

                                    json pcfg2 = pcfg;
                                    pcfg2["base_channels"] = std::max(1, inferred_base);

                                    aux_perceptual_.reset();
                                    aux_perceptual_ = ModelArchitectures::create(p_arch, pcfg2);
                                    aux_perceptual_->allocateParams();
                                    try { aux_perceptual_->initializeWeights("xavier", 1337u); } catch (...) {}

                                    std::string err2;
                                    if (Mimir::Serialization::load_checkpoint(*aux_perceptual_, ckpt, opts, &err2)) {
                                        retried = true;
                                    } else {
                                        err = err2;
                                    }
                                }
                            }
                        } catch (...) {
                            // best-effort: keep original err
                        }
                    }

                    if (!retried) {
                        std::cerr << "⚠️  Perceptual checkpoint load failed: " << ckpt << " | " << err << std::endl;
                    }
                }
            } else {
                std::cerr << "⚠️  Perceptual loss active but perceptual_checkpoint is empty: using fixed random Xavier init (seed=1337)." << std::endl;
            }
        }

        const std::vector<float> pred_recon(pred.begin(), pred.begin() + recon_n);
        const std::vector<float> tgt_recon(x.begin(), x.begin() + recon_n);

        // Forward real first (copy features), then forward fake (keeps activations for backward)
        aux_perceptual_->zeroGradients();
        const std::vector<float>& f_real_view = aux_perceptual_->forwardPassView(tgt_recon, true);
        std::vector<float> f_real(f_real_view.begin(), f_real_view.end());

        aux_perceptual_->zeroGradients();
        const std::vector<float>& f_fake_view = aux_perceptual_->forwardPassView(pred_recon, true);
        const size_t fn = std::min(f_fake_view.size(), f_real.size());
        if (fn > 0) {
            double pl = 0.0;
            std::vector<float> gfeat(fn, 0.0f);
            const float scale = 2.0f / static_cast<float>(fn);
            for (size_t i = 0; i < fn; ++i) {
                const double d = static_cast<double>(f_fake_view[i]) - static_cast<double>(f_real[i]);
                pl += d * d;
                gfeat[i] = scale * static_cast<float>(d);
            }
            pl /= static_cast<double>(fn);
            recon += static_cast<double>(perceptual_weight) * pl;

            aux_perceptual_->backwardPass(gfeat);
            if (aux_perceptual_->hasLastInputGradient()) {
                const auto& gin = aux_perceptual_->getLastInputGradient();
                if (gin.size() >= static_cast<size_t>(recon_n)) {
                    for (int i = 0; i < recon_n; ++i) {
                        grad_recon[static_cast<size_t>(i)] += perceptual_weight * gin[static_cast<size_t>(i)];
                    }
                }
            }
        }
    }

    // Adversarial (discriminateur) retiré.
    const int mu_off = image_dim;
    const int lv_off = image_dim + latent_dim;
    double kl = 0.0;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu_f = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const double mu = static_cast<double>(mu_f);
        const double ev = std::exp(static_cast<double>(lv));
        kl += 0.5 * (mu * mu + ev - 1.0 - static_cast<double>(lv));
    }
    kl /= static_cast<double>(std::max(1, latent_dim));

    const double total_loss = recon + static_cast<double>(beta_eff) * kl;

    // Marker metrics computed on recon vs target image (prefix).
    const auto mp = compute_moments_prefix(pred, 0, recon_n);
    const auto mt = compute_moments_prefix(x, 0, recon_n);
    const double vp = std::max(mp.var, 1e-12);
    const double vt = std::max(mt.var, 1e-12);
    const double w2 = (mt.mean - mp.mean) * (mt.mean - mp.mean) + (std::sqrt(vt) - std::sqrt(vp)) * (std::sqrt(vt) - std::sqrt(vp));
    const float wass = static_cast<float>(std::sqrt(std::max(0.0, w2)));
    const float temp = static_cast<float>(pearson_corr_prefix(pred, 0, x, 0, recon_n));
    const float temp_pen = 1.0f - std::clamp(temp, -1.0f, 1.0f);

    // Stop-grad marker scaling (treat marker as constant for the step).
    float marker_scale = 1.0f;
    if (marker_wass_eff > 0.0f || marker_temp_eff > 0.0f) {
        marker_scale = 1.0f + marker_wass_eff * wass + marker_temp_eff * temp_pen;
        marker_scale = std::clamp(marker_scale, 0.1f, marker_scale_max);
    }

    const double total_loss_marked = static_cast<double>(marker_scale) * recon + static_cast<double>(beta_eff) * kl;

    // Gradient on packed output [recon|mu|logvar]
    scratch_packed_grad_.resize(static_cast<size_t>(out_dim));
    std::fill(scratch_packed_grad_.begin(), scratch_packed_grad_.end(), 0.0f);
    auto& grad = scratch_packed_grad_;
    // recon grad: already normalized by n in each component, just apply marker_scale.
    for (int i = 0; i < recon_n; ++i) {
        grad[static_cast<size_t>(i)] = marker_scale * grad_recon[static_cast<size_t>(i)];
    }

    const float kl_scale = (latent_dim > 0) ? (beta_eff / static_cast<float>(latent_dim)) : 0.0f;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const float in_range = (lv_raw >= logvar_min && lv_raw <= logvar_max) ? 1.0f : 0.0f;

        grad[static_cast<size_t>(mu_off + i)] = kl_scale * mu;
        grad[static_cast<size_t>(lv_off + i)] = kl_scale * (0.5f * (std::exp(lv) - 1.0f)) * in_range;
    }

    backwardPass(grad);

    // Gradient norm (monitoring)
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (const auto& layer : layers) {
        for (float g : layer.grad_weights) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
        for (float g : layer.grad_bias) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
    }
    const float grad_norm = static_cast<float>(std::sqrt(sum_sq));

    optimizerStep(opt, learning_rate);

    VAEStepStats stats;
    stats.loss = static_cast<float>(total_loss_marked);
    stats.mse = static_cast<float>(recon);
    stats.kl = static_cast<float>(kl);
    stats.wass = wass;
    stats.temp = temp;
    stats.kl_beta_effective = beta_eff;
    stats.latent_dim = latent_dim;
    stats.grad_norm = grad_norm;
    stats.grad_max_abs = max_abs;
    return stats;
}

Model::VAEStepStats Model::backwardStepVAE(const std::vector<float>& x, Optimizer& opt, float grad_scale) {
    if (params_frozen_) {
        throw std::runtime_error("Model::backwardStepVAE: parameters are frozen");
    }
    if (layers.empty()) {
        throw std::runtime_error("Model::backwardStepVAE: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("Model::backwardStepVAE: weights not allocated (call allocateParams/initWeights)");
    }
    if (!std::isfinite(grad_scale) || grad_scale <= 0.0f) grad_scale = 1.0f;

    int image_dim = 0;
    if (modelConfig.contains("image_dim")) {
        image_dim = std::max(0, modelConfig["image_dim"].get<int>());
    }
    if (image_dim <= 0) image_dim = static_cast<int>(x.size());

    int latent_dim = 0;
    if (modelConfig.contains("latent_dim")) {
        latent_dim = std::max(0, modelConfig["latent_dim"].get<int>());
    }

    float kl_beta = 1.0f;
    if (modelConfig.contains("kl_beta")) {
        kl_beta = modelConfig["kl_beta"].get<float>();
    } else if (modelConfig.contains("vae_kl_beta")) {
        kl_beta = modelConfig["vae_kl_beta"].get<float>();
    }
    kl_beta = std::max(0.0f, kl_beta);

    int kl_warmup_steps = 0;
    if (modelConfig.contains("kl_warmup_steps")) {
        kl_warmup_steps = std::max(0, modelConfig["kl_warmup_steps"].get<int>());
    }

    float marker_wass_scale = 0.0f;
    float marker_temp_scale = 0.0f;
    float marker_scale_max = 10.0f;
    int marker_warmup_steps = 0;
    if (modelConfig.contains("marker_wass_scale")) marker_wass_scale = modelConfig["marker_wass_scale"].get<float>();
    if (modelConfig.contains("marker_temp_scale")) marker_temp_scale = modelConfig["marker_temp_scale"].get<float>();
    if (modelConfig.contains("marker_scale_max")) marker_scale_max = modelConfig["marker_scale_max"].get<float>();
    if (modelConfig.contains("marker_warmup_steps")) marker_warmup_steps = std::max(0, modelConfig["marker_warmup_steps"].get<int>());
    marker_wass_scale = std::max(0.0f, marker_wass_scale);
    marker_temp_scale = std::max(0.0f, marker_temp_scale);
    marker_scale_max = std::max(1.0f, marker_scale_max);

    float logvar_min = -10.0f;
    float logvar_max = 10.0f;
    if (modelConfig.contains("logvar_clip_min")) {
        logvar_min = modelConfig["logvar_clip_min"].get<float>();
    }
    if (modelConfig.contains("logvar_clip_max")) {
        logvar_max = modelConfig["logvar_clip_max"].get<float>();
    }
    if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

    const float progress = (kl_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(kl_warmup_steps))
        : 1.0f;
    const float beta_eff = kl_beta * progress;

    const float marker_progress = (marker_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(marker_warmup_steps))
        : 1.0f;
    const float marker_wass_eff = marker_wass_scale * marker_progress;
    const float marker_temp_eff = marker_temp_scale * marker_progress;

    const std::vector<float>& pred = forwardPassView(x, true);
    const int out_dim = static_cast<int>(pred.size());

    if (latent_dim <= 0) {
        if (out_dim > image_dim + 2 && ((out_dim - image_dim) % 2) == 0) {
            latent_dim = std::max(1, (out_dim - image_dim) / 2);
        }
    }
    if (image_dim <= 0 || out_dim < image_dim + 2) {
        throw std::runtime_error("Model::backwardStepVAE: invalid output/image_dim (out_dim=" + std::to_string(out_dim) + ", image_dim=" + std::to_string(image_dim) + ")");
    }
    if (latent_dim <= 0 || out_dim < image_dim + 2 * latent_dim) {
        const int tail = out_dim - image_dim;
        if (tail < 2 || (tail % 2) != 0) {
            throw std::runtime_error("Model::backwardStepVAE: cannot infer latent_dim from output tail (tail=" + std::to_string(tail) + ")");
        }
        latent_dim = std::max(1, tail / 2);
    }

    const int recon_n = std::min(image_dim, static_cast<int>(x.size()));
    std::string recon_loss = "mse";
    if (modelConfig.contains("recon_loss")) {
        try {
            recon_loss = modelConfig["recon_loss"].get<std::string>();
        } catch (...) {
        }
    }
    // Loss: recon (avg) + beta * KL (avg)
    double recon = 0.0;
    if (recon_loss == "l1" || recon_loss == "mae") {
        for (int i = 0; i < recon_n; ++i) {
            const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>(x[static_cast<size_t>(i)]);
            recon += std::abs(d);
        }
        recon /= static_cast<double>(std::max(1, recon_n));
    } else {
        for (int i = 0; i < recon_n; ++i) {
            const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>(x[static_cast<size_t>(i)]);
            recon += d * d;
        }
        recon /= static_cast<double>(std::max(1, recon_n));
    }
    const int mu_off = image_dim;
    const int lv_off = image_dim + latent_dim;
    double kl = 0.0;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu_f = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const double mu = static_cast<double>(mu_f);
        const double ev = std::exp(static_cast<double>(lv));
        kl += 0.5 * (mu * mu + ev - 1.0 - static_cast<double>(lv));
    }
    kl /= static_cast<double>(std::max(1, latent_dim));

    const double total_loss = recon + static_cast<double>(beta_eff) * kl;

    const auto mp = compute_moments_prefix(pred, 0, recon_n);
    const auto mt = compute_moments_prefix(x, 0, recon_n);
    const double vp = std::max(mp.var, 1e-12);
    const double vt = std::max(mt.var, 1e-12);
    const double w2 = (mt.mean - mp.mean) * (mt.mean - mp.mean) + (std::sqrt(vt) - std::sqrt(vp)) * (std::sqrt(vt) - std::sqrt(vp));
    const float wass = static_cast<float>(std::sqrt(std::max(0.0, w2)));
    const float temp = static_cast<float>(pearson_corr_prefix(pred, 0, x, 0, recon_n));
    const float temp_pen = 1.0f - std::clamp(temp, -1.0f, 1.0f);

    float marker_scale = 1.0f;
    if (marker_wass_eff > 0.0f || marker_temp_eff > 0.0f) {
        marker_scale = 1.0f + marker_wass_eff * wass + marker_temp_eff * temp_pen;
        marker_scale = std::clamp(marker_scale, 0.1f, marker_scale_max);
    }

    const double total_loss_marked = static_cast<double>(marker_scale) * recon + static_cast<double>(beta_eff) * kl;

    scratch_packed_grad_.resize(static_cast<size_t>(out_dim));
    std::fill(scratch_packed_grad_.begin(), scratch_packed_grad_.end(), 0.0f);
    auto& grad = scratch_packed_grad_;
    if (recon_loss == "l1" || recon_loss == "mae") {
        const float recon_scale = (1.0f / static_cast<float>(std::max(1, recon_n))) * grad_scale * marker_scale;
        for (int i = 0; i < recon_n; ++i) {
            const float d = pred[static_cast<size_t>(i)] - x[static_cast<size_t>(i)];
            const float s = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
            grad[static_cast<size_t>(i)] = recon_scale * s;
        }
    } else {
        const float recon_scale = (2.0f / static_cast<float>(std::max(1, recon_n))) * grad_scale * marker_scale;
        for (int i = 0; i < recon_n; ++i) {
            grad[static_cast<size_t>(i)] = recon_scale * (pred[static_cast<size_t>(i)] - x[static_cast<size_t>(i)]);
        }
    }

    const float kl_scale = (latent_dim > 0) ? ((beta_eff / static_cast<float>(latent_dim)) * grad_scale) : 0.0f;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const float in_range = (lv_raw >= logvar_min && lv_raw <= logvar_max) ? 1.0f : 0.0f;
        grad[static_cast<size_t>(mu_off + i)] = kl_scale * mu;
        grad[static_cast<size_t>(lv_off + i)] = kl_scale * (0.5f * (std::exp(lv) - 1.0f)) * in_range;
    }

    backwardPass(grad);

    double sum_sq2 = 0.0;
    float max_abs2 = 0.0f;
    for (const auto& layer : layers) {
        for (float g : layer.grad_weights) {
            sum_sq2 += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs2) max_abs2 = a;
        }
        for (float g : layer.grad_bias) {
            sum_sq2 += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs2) max_abs2 = a;
        }
    }

    VAEStepStats stats;
    stats.loss = static_cast<float>(total_loss_marked);
    stats.mse = static_cast<float>(recon);
    stats.kl = static_cast<float>(kl);
    stats.wass = wass;
    stats.temp = temp;
    stats.kl_beta_effective = beta_eff;
    stats.latent_dim = latent_dim;
    stats.grad_norm = static_cast<float>(std::sqrt(sum_sq2));
    stats.grad_max_abs = max_abs2;
    return stats;
}

static inline float dot_f(const std::vector<float>& a, size_t a_off, const std::vector<float>& b, size_t b_off, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += static_cast<double>(a[a_off + static_cast<size_t>(i)]) * static_cast<double>(b[b_off + static_cast<size_t>(i)]);
    }
    return static_cast<float>(s);
}

static inline float norm2_f(const std::vector<float>& a, size_t a_off, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(a[a_off + static_cast<size_t>(i)]);
        s += v * v;
    }
    return static_cast<float>(std::sqrt(s));
}

Model::VAEStepStats Model::trainStepVAEText(const std::vector<float>& x,
                                           const std::vector<int>& text_ids,
                                           Optimizer& opt,
                                           float learning_rate) {
    if (params_frozen_) {
        throw std::runtime_error("Model::trainStepVAEText: parameters are frozen");
    }
    if (layers.empty()) {
        throw std::runtime_error("Model::trainStepVAEText: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("Model::trainStepVAEText: weights not allocated (call allocateParams/initWeights)");
    }

    // Reconstruction loss type
    std::string recon_loss = "mse";
    if (modelConfig.contains("recon_loss")) {
        try { recon_loss = modelConfig["recon_loss"].get<std::string>(); } catch (...) {}
    }

    const bool recon_is_ce = (recon_loss == "ce" || recon_loss == "cross_entropy" || recon_loss == "xent");

    // Target: for MSE/L1 we use x or an internal tap tensor. For CE we use text_ids.
    const std::vector<float>* target = &x;
    bool using_internal_target = false;

    int image_dim = 0;
    if (modelConfig.contains("image_dim")) {
        image_dim = std::max(0, modelConfig["image_dim"].get<int>());
    }

    int seq_len_cfg = 0;
    int vocab_cfg = 0;
    int pad_id = -1;
    if (modelConfig.contains("seq_len")) seq_len_cfg = std::max(0, modelConfig["seq_len"].get<int>());
    if (modelConfig.contains("vocab_size")) vocab_cfg = std::max(0, modelConfig["vocab_size"].get<int>());
    if (modelConfig.contains("padding_idx")) pad_id = modelConfig["padding_idx"].get<int>();

    if (recon_is_ce) {
        const int sl = std::max(1, seq_len_cfg);
        const int vv = std::max(2, vocab_cfg);
        image_dim = std::max(1, sl * vv);
    } else {
        if (x.empty()) {
            std::string tname;
            if (modelConfig.contains("target_tensor")) {
                try { tname = modelConfig["target_tensor"].get<std::string>(); } catch (...) {}
            }
            if (!tname.empty() && hasTensor(tname)) {
                target = &getTensor(tname);
            } else if (hasTensor("vae_text/target")) {
                target = &getTensor("vae_text/target");
            }
        }

        using_internal_target = x.empty() && (target != &x) && (!target->empty());
        if (using_internal_target) {
            image_dim = static_cast<int>(target->size());
        } else if (image_dim <= 0) {
            image_dim = static_cast<int>(target->size());
        }
    }

    int latent_dim = 0;
    if (modelConfig.contains("latent_dim")) {
        latent_dim = std::max(0, modelConfig["latent_dim"].get<int>());
    }
    if (modelConfig.contains("latent_tokens") && modelConfig.contains("d_model")) {
        const int lt = std::max(1, modelConfig["latent_tokens"].get<int>());
        const int dm = std::max(1, modelConfig["d_model"].get<int>());
        const int ld = std::max(1, lt * dm);
        if (latent_dim <= 0 || latent_dim != ld) latent_dim = ld;
    }

    int proj_dim = 0;
    if (modelConfig.contains("proj_dim")) {
        proj_dim = std::max(0, modelConfig["proj_dim"].get<int>());
    }

    float align_weight = 0.1f;
    if (modelConfig.contains("align_weight")) {
        align_weight = modelConfig["align_weight"].get<float>();
    }
    align_weight = std::max(0.0f, align_weight);

    float kl_beta = 1.0f;
    if (modelConfig.contains("kl_beta")) {
        kl_beta = modelConfig["kl_beta"].get<float>();
    } else if (modelConfig.contains("vae_kl_beta")) {
        kl_beta = modelConfig["vae_kl_beta"].get<float>();
    }
    kl_beta = std::max(0.0f, kl_beta);

    int kl_warmup_steps = 0;
    if (modelConfig.contains("kl_warmup_steps")) {
        kl_warmup_steps = std::max(0, modelConfig["kl_warmup_steps"].get<int>());
    }

    float marker_wass_scale = 0.0f;
    float marker_temp_scale = 0.0f;
    float marker_scale_max = 10.0f;
    int marker_warmup_steps = 0;
    if (modelConfig.contains("marker_wass_scale")) marker_wass_scale = modelConfig["marker_wass_scale"].get<float>();
    if (modelConfig.contains("marker_temp_scale")) marker_temp_scale = modelConfig["marker_temp_scale"].get<float>();
    if (modelConfig.contains("marker_scale_max")) marker_scale_max = modelConfig["marker_scale_max"].get<float>();
    if (modelConfig.contains("marker_warmup_steps")) marker_warmup_steps = std::max(0, modelConfig["marker_warmup_steps"].get<int>());
    marker_wass_scale = std::max(0.0f, marker_wass_scale);
    marker_temp_scale = std::max(0.0f, marker_temp_scale);
    marker_scale_max = std::max(1.0f, marker_scale_max);

    float logvar_min = -10.0f;
    float logvar_max = 10.0f;
    if (modelConfig.contains("logvar_clip_min")) {
        logvar_min = modelConfig["logvar_clip_min"].get<float>();
    }
    if (modelConfig.contains("logvar_clip_max")) {
        logvar_max = modelConfig["logvar_clip_max"].get<float>();
    }
    if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

    const float progress = (kl_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(kl_warmup_steps))
        : 1.0f;
    const float beta_eff = kl_beta * progress;

    const float marker_progress = (marker_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(marker_warmup_steps))
        : 1.0f;
    const float marker_wass_eff = marker_wass_scale * marker_progress;
    const float marker_temp_eff = marker_temp_scale * marker_progress;

    // Forward (named inputs)
    zeroGradients();
    std::unordered_map<std::string, std::vector<float>> fin;
    std::unordered_map<std::string, std::vector<int>> iin;
    fin["__input__"] = x;
    iin["text_ids"] = text_ids;
    const std::vector<float>& pred = forwardPassNamedView(fin, iin, true);
    const int out_dim = static_cast<int>(pred.size());

    // Resolve internal target AFTER forward (it is produced by the graph) for MSE/L1 modes.
    if (!recon_is_ce && x.empty()) {
        std::string tname;
        if (modelConfig.contains("target_tensor")) {
            try { tname = modelConfig["target_tensor"].get<std::string>(); } catch (...) {}
        }
        if (!tname.empty() && hasTensor(tname)) {
            target = &getTensor(tname);
        } else if (hasTensor("vae_text/target")) {
            target = &getTensor("vae_text/target");
        }
        using_internal_target = x.empty() && (target != &x) && (!target->empty());
        if (using_internal_target) {
            image_dim = static_cast<int>(target->size());
        } else if (x.empty() && modelConfig.contains("seq_len") && modelConfig.contains("d_model")) {
            const int sl = std::max(1, modelConfig["seq_len"].get<int>());
            const int dm = std::max(1, modelConfig["d_model"].get<int>());
            image_dim = std::max(1, sl * dm);
        }
    }

    // Infer latent_dim / proj_dim if missing
    if (latent_dim <= 0) {
        // We need at least recon + mu + logvar
        const int tail = out_dim - image_dim;
        if (tail >= 4 && (tail % 2) == 0) {
            // ambiguous when proj heads exist; prefer config. fallback: split in 2 equal halves for mu/logvar and ignore proj.
            latent_dim = std::max(1, tail / 2);
        }
    }

    if (image_dim <= 0 || latent_dim <= 0 || out_dim < image_dim + 2 * latent_dim + 2) {
        throw std::runtime_error("Model::trainStepVAEText: invalid dims (out_dim=" + std::to_string(out_dim) + ", image_dim=" + std::to_string(image_dim) + ", latent_dim=" + std::to_string(latent_dim) + ")");
    }

    const int proj_tail = out_dim - (image_dim + 2 * latent_dim);
    if (proj_dim <= 0) {
        if (proj_tail > 0 && (proj_tail % 2) == 0) proj_dim = std::max(1, proj_tail / 2);
    }
    if (proj_dim <= 0 || out_dim < image_dim + 2 * latent_dim + 2 * proj_dim) {
        throw std::runtime_error("Model::trainStepVAEText: missing/invalid proj_dim (proj_tail=" + std::to_string(proj_tail) + ")");
    }

    const int recon_n = std::min(image_dim, static_cast<int>(target->size()));

    // Recon
    double recon = 0.0;
    float wass = 0.0f;
    float temp = 0.0f;
    float marker_scale = 1.0f;

    if (recon_is_ce) {
        const int sl = std::max(1, seq_len_cfg);
        const int vv = std::max(2, vocab_cfg);
        const int valid_n = std::min(static_cast<int>(text_ids.size()), sl);

        int count = 0;
        for (int t = 0; t < valid_n; ++t) {
            const int y = text_ids[static_cast<size_t>(t)];
            if (pad_id >= 0 && y == pad_id) continue;
            if (y < 0 || y >= vv) continue;
            const size_t base = static_cast<size_t>(t) * static_cast<size_t>(vv);

            float m = -1e30f;
            for (int j = 0; j < vv; ++j) {
                const float v = pred[base + static_cast<size_t>(j)];
                if (v > m) m = v;
            }
            double sum = 0.0;
            for (int j = 0; j < vv; ++j) {
                sum += std::exp(static_cast<double>(pred[base + static_cast<size_t>(j)] - m));
            }
            const double lse = static_cast<double>(m) + std::log(std::max(1e-30, sum));
            recon += lse - static_cast<double>(pred[base + static_cast<size_t>(y)]);
            count += 1;
        }
        if (count > 0) recon /= static_cast<double>(count);
    } else {
        if (!using_internal_target && x.empty()) {
            // x empty without internal target: fallback to seq_len*d_model if available
            if (modelConfig.contains("seq_len") && modelConfig.contains("d_model")) {
                const int sl = std::max(1, modelConfig["seq_len"].get<int>());
                const int dm = std::max(1, modelConfig["d_model"].get<int>());
                image_dim = std::max(1, sl * dm);
            }
        }

        const int recon_n = std::min(image_dim, static_cast<int>(target->size()));

        if (recon_loss == "l1" || recon_loss == "mae") {
            for (int i = 0; i < recon_n; ++i) {
                const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>((*target)[static_cast<size_t>(i)]);
                recon += std::abs(d);
            }
            recon /= static_cast<double>(std::max(1, recon_n));
        } else {
            for (int i = 0; i < recon_n; ++i) {
                const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>((*target)[static_cast<size_t>(i)]);
                recon += d * d;
            }
            recon /= static_cast<double>(std::max(1, recon_n));
        }

        const auto mp = compute_moments_prefix(pred, 0, recon_n);
        const auto mt = compute_moments_prefix(*target, 0, recon_n);
        const double vp = std::max(mp.var, 1e-12);
        const double vt = std::max(mt.var, 1e-12);
        const double w2 = (mt.mean - mp.mean) * (mt.mean - mp.mean) + (std::sqrt(vt) - std::sqrt(vp)) * (std::sqrt(vt) - std::sqrt(vp));
        wass = static_cast<float>(std::sqrt(std::max(0.0, w2)));
        temp = static_cast<float>(pearson_corr_prefix(pred, 0, *target, 0, recon_n));
        const float temp_pen = 1.0f - std::clamp(temp, -1.0f, 1.0f);

        if (marker_wass_eff > 0.0f || marker_temp_eff > 0.0f) {
            marker_scale = 1.0f + marker_wass_eff * wass + marker_temp_eff * temp_pen;
            marker_scale = std::clamp(marker_scale, 0.1f, marker_scale_max);
        }
    }

    // KL
    const int mu_off = image_dim;
    const int lv_off = image_dim + latent_dim;
    double kl = 0.0;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu_f = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const double mu = static_cast<double>(mu_f);
        const double ev = std::exp(static_cast<double>(lv));
        kl += 0.5 * (mu * mu + ev - 1.0 - static_cast<double>(lv));
    }
    kl /= static_cast<double>(std::max(1, latent_dim));

    // Alignment (1 - cosine)
    const size_t imgp_off = static_cast<size_t>(image_dim + 2 * latent_dim);
    const size_t txtp_off = imgp_off + static_cast<size_t>(proj_dim);
    const float eps = 1e-8f;
    const float na = std::max(eps, norm2_f(pred, imgp_off, proj_dim));
    const float nb = std::max(eps, norm2_f(pred, txtp_off, proj_dim));
    const float dp = dot_f(pred, imgp_off, pred, txtp_off, proj_dim);
    const float cos_sim = dp / (na * nb);
    const float align = (align_weight > 0.0f) ? (align_weight * (1.0f - cos_sim)) : 0.0f;

    const double total_loss = static_cast<double>(marker_scale) * recon + static_cast<double>(beta_eff) * kl + static_cast<double>(align);

    // Gradient on packed output
    scratch_packed_grad_.resize(static_cast<size_t>(out_dim));
    std::fill(scratch_packed_grad_.begin(), scratch_packed_grad_.end(), 0.0f);
    auto& grad = scratch_packed_grad_;

    // Recon grad
    if (recon_is_ce) {
        const int sl = std::max(1, seq_len_cfg);
        const int vv = std::max(2, vocab_cfg);
        const int valid_n = std::min(static_cast<int>(text_ids.size()), sl);

        int count = 0;
        for (int t = 0; t < valid_n; ++t) {
            const int y = text_ids[static_cast<size_t>(t)];
            if (pad_id >= 0 && y == pad_id) continue;
            if (y < 0 || y >= vv) continue;
            count += 1;
        }
        const float scale = (count > 0) ? (marker_scale / static_cast<float>(count)) : 0.0f;

        for (int t = 0; t < valid_n; ++t) {
            const int y = text_ids[static_cast<size_t>(t)];
            if (pad_id >= 0 && y == pad_id) continue;
            if (y < 0 || y >= vv) continue;
            const size_t base = static_cast<size_t>(t) * static_cast<size_t>(vv);

            float m = -1e30f;
            for (int j = 0; j < vv; ++j) {
                const float v = pred[base + static_cast<size_t>(j)];
                if (v > m) m = v;
            }
            double sum = 0.0;
            for (int j = 0; j < vv; ++j) {
                sum += std::exp(static_cast<double>(pred[base + static_cast<size_t>(j)] - m));
            }
            const double inv_denom = 1.0 / std::max(1e-30, sum);
            for (int j = 0; j < vv; ++j) {
                const double p = std::exp(static_cast<double>(pred[base + static_cast<size_t>(j)] - m)) * inv_denom;
                grad[base + static_cast<size_t>(j)] = scale * static_cast<float>(p);
            }
            grad[base + static_cast<size_t>(y)] -= scale;
        }
    } else {
        const int recon_n = std::min(image_dim, static_cast<int>(target->size()));
        if (recon_loss == "l1" || recon_loss == "mae") {
            const float recon_scale = (1.0f / static_cast<float>(std::max(1, recon_n))) * marker_scale;
            for (int i = 0; i < recon_n; ++i) {
                const float d = pred[static_cast<size_t>(i)] - (*target)[static_cast<size_t>(i)];
                const float s = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
                grad[static_cast<size_t>(i)] = recon_scale * s;
            }
        } else {
            const float recon_scale = (2.0f / static_cast<float>(std::max(1, recon_n))) * marker_scale;
            for (int i = 0; i < recon_n; ++i) {
                grad[static_cast<size_t>(i)] = recon_scale * (pred[static_cast<size_t>(i)] - (*target)[static_cast<size_t>(i)]);
            }
        }
    }

    // KL grad
    const float kl_scale = (latent_dim > 0) ? (beta_eff / static_cast<float>(latent_dim)) : 0.0f;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const float in_range = (lv_raw >= logvar_min && lv_raw <= logvar_max) ? 1.0f : 0.0f;
        grad[static_cast<size_t>(mu_off + i)] = kl_scale * mu;
        grad[static_cast<size_t>(lv_off + i)] = kl_scale * (0.5f * (std::exp(lv) - 1.0f)) * in_range;
    }

    // Align grad (cosine)
    if (align_weight > 0.0f) {
        // u=a/||a||, v=b/||b||, cos=u·v
        // dcos/da = (v - cos*u)/||a||, dL/da = -w*dcos/da
        const float inv_na = 1.0f / na;
        const float inv_nb = 1.0f / nb;
        for (int i = 0; i < proj_dim; ++i) {
            const float a = pred[imgp_off + static_cast<size_t>(i)];
            const float b = pred[txtp_off + static_cast<size_t>(i)];
            const float u = a * inv_na;
            const float v = b * inv_nb;
            const float dcos_da = (v - cos_sim * u) * inv_na;
            const float dcos_db = (u - cos_sim * v) * inv_nb;
            grad[imgp_off + static_cast<size_t>(i)] = -align_weight * dcos_da;
            grad[txtp_off + static_cast<size_t>(i)] = -align_weight * dcos_db;
        }
    }

    backwardPass(grad);

    // Gradient norm (monitoring)
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (const auto& layer : layers) {
        for (float g : layer.grad_weights) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
        for (float g : layer.grad_bias) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
    }
    const float grad_norm = static_cast<float>(std::sqrt(sum_sq));

    optimizerStep(opt, learning_rate);

    VAEStepStats stats;
    stats.loss = static_cast<float>(total_loss);
    stats.mse = static_cast<float>(recon);
    stats.kl = static_cast<float>(kl);
    stats.wass = wass;
    stats.temp = temp;
    stats.align = align;
    stats.kl_beta_effective = beta_eff;
    stats.latent_dim = latent_dim;
    stats.grad_norm = grad_norm;
    stats.grad_max_abs = max_abs;
    return stats;
}

Model::VAEStepStats Model::backwardStepVAEText(const std::vector<float>& x,
                                               const std::vector<int>& text_ids,
                                               Optimizer& opt,
                                               float grad_scale) {
    if (params_frozen_) {
        throw std::runtime_error("Model::backwardStepVAEText: parameters are frozen");
    }
    if (layers.empty()) {
        throw std::runtime_error("Model::backwardStepVAEText: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("Model::backwardStepVAEText: weights not allocated (call allocateParams/initWeights)");
    }
    if (!std::isfinite(grad_scale) || grad_scale <= 0.0f) grad_scale = 1.0f;

    std::string recon_loss = "mse";
    if (modelConfig.contains("recon_loss")) {
        try { recon_loss = modelConfig["recon_loss"].get<std::string>(); } catch (...) {}
    }
    const bool recon_is_ce = (recon_loss == "ce" || recon_loss == "cross_entropy" || recon_loss == "xent");

    const std::vector<float>* target = &x;
    bool using_internal_target = false;

    int image_dim = 0;
    if (modelConfig.contains("image_dim")) {
        image_dim = std::max(0, modelConfig["image_dim"].get<int>());
    }
    int seq_len_cfg = 0;
    int vocab_cfg = 0;
    int pad_id = -1;
    if (modelConfig.contains("seq_len")) seq_len_cfg = std::max(0, modelConfig["seq_len"].get<int>());
    if (modelConfig.contains("vocab_size")) vocab_cfg = std::max(0, modelConfig["vocab_size"].get<int>());
    if (modelConfig.contains("padding_idx")) pad_id = modelConfig["padding_idx"].get<int>();

    if (recon_is_ce) {
        const int sl = std::max(1, seq_len_cfg);
        const int vv = std::max(2, vocab_cfg);
        image_dim = std::max(1, sl * vv);
    } else {
        if (x.empty()) {
            std::string tname;
            if (modelConfig.contains("target_tensor")) {
                try { tname = modelConfig["target_tensor"].get<std::string>(); } catch (...) {}
            }
            if (!tname.empty() && hasTensor(tname)) {
                target = &getTensor(tname);
            } else if (hasTensor("vae_text/target")) {
                target = &getTensor("vae_text/target");
            }
        }
        using_internal_target = x.empty() && (target != &x) && (!target->empty());
        if (using_internal_target) {
            image_dim = static_cast<int>(target->size());
        } else if (image_dim <= 0) {
            image_dim = static_cast<int>(target->size());
        }
    }

    int latent_dim = 0;
    if (modelConfig.contains("latent_dim")) {
        latent_dim = std::max(0, modelConfig["latent_dim"].get<int>());
    }
    if (modelConfig.contains("latent_tokens") && modelConfig.contains("d_model")) {
        const int lt = std::max(1, modelConfig["latent_tokens"].get<int>());
        const int dm = std::max(1, modelConfig["d_model"].get<int>());
        const int ld = std::max(1, lt * dm);
        if (latent_dim <= 0 || latent_dim != ld) latent_dim = ld;
    }

    int proj_dim = 0;
    if (modelConfig.contains("proj_dim")) {
        proj_dim = std::max(0, modelConfig["proj_dim"].get<int>());
    }

    float align_weight = 0.1f;
    if (modelConfig.contains("align_weight")) {
        align_weight = modelConfig["align_weight"].get<float>();
    }
    align_weight = std::max(0.0f, align_weight);

    float kl_beta = 1.0f;
    if (modelConfig.contains("kl_beta")) {
        kl_beta = modelConfig["kl_beta"].get<float>();
    } else if (modelConfig.contains("vae_kl_beta")) {
        kl_beta = modelConfig["vae_kl_beta"].get<float>();
    }
    kl_beta = std::max(0.0f, kl_beta);

    int kl_warmup_steps = 0;
    if (modelConfig.contains("kl_warmup_steps")) {
        kl_warmup_steps = std::max(0, modelConfig["kl_warmup_steps"].get<int>());
    }

    float marker_wass_scale = 0.0f;
    float marker_temp_scale = 0.0f;
    float marker_scale_max = 10.0f;
    int marker_warmup_steps = 0;
    if (modelConfig.contains("marker_wass_scale")) marker_wass_scale = modelConfig["marker_wass_scale"].get<float>();
    if (modelConfig.contains("marker_temp_scale")) marker_temp_scale = modelConfig["marker_temp_scale"].get<float>();
    if (modelConfig.contains("marker_scale_max")) marker_scale_max = modelConfig["marker_scale_max"].get<float>();
    if (modelConfig.contains("marker_warmup_steps")) marker_warmup_steps = std::max(0, modelConfig["marker_warmup_steps"].get<int>());
    marker_wass_scale = std::max(0.0f, marker_wass_scale);
    marker_temp_scale = std::max(0.0f, marker_temp_scale);
    marker_scale_max = std::max(1.0f, marker_scale_max);

    float logvar_min = -10.0f;
    float logvar_max = 10.0f;
    if (modelConfig.contains("logvar_clip_min")) {
        logvar_min = modelConfig["logvar_clip_min"].get<float>();
    }
    if (modelConfig.contains("logvar_clip_max")) {
        logvar_max = modelConfig["logvar_clip_max"].get<float>();
    }
    if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

    const float progress = (kl_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(kl_warmup_steps))
        : 1.0f;
    const float beta_eff = kl_beta * progress;

    const float marker_progress = (marker_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(marker_warmup_steps))
        : 1.0f;
    const float marker_wass_eff = marker_wass_scale * marker_progress;
    const float marker_temp_eff = marker_temp_scale * marker_progress;

    std::unordered_map<std::string, std::vector<float>> fin;
    std::unordered_map<std::string, std::vector<int>> iin;
    fin["__input__"] = x;
    iin["text_ids"] = text_ids;
    const std::vector<float>& pred = forwardPassNamedView(fin, iin, true);
    const int out_dim = static_cast<int>(pred.size());

    // Resolve internal target AFTER forward (it is produced by the graph) for MSE/L1 modes.
    if (!recon_is_ce && x.empty()) {
        std::string tname;
        if (modelConfig.contains("target_tensor")) {
            try { tname = modelConfig["target_tensor"].get<std::string>(); } catch (...) {}
        }
        if (!tname.empty() && hasTensor(tname)) {
            target = &getTensor(tname);
        } else if (hasTensor("vae_text/target")) {
            target = &getTensor("vae_text/target");
        }
        using_internal_target = x.empty() && (target != &x) && (!target->empty());
        if (using_internal_target) {
            image_dim = static_cast<int>(target->size());
        } else if (x.empty() && modelConfig.contains("seq_len") && modelConfig.contains("d_model")) {
            const int sl = std::max(1, modelConfig["seq_len"].get<int>());
            const int dm = std::max(1, modelConfig["d_model"].get<int>());
            image_dim = std::max(1, sl * dm);
        }
    }

    if (latent_dim <= 0) {
        const int tail = out_dim - image_dim;
        if (tail > 2 && (tail % 2) == 0) latent_dim = std::max(1, tail / 2);
    }
    const int proj_tail = out_dim - (image_dim + 2 * latent_dim);
    if (proj_dim <= 0) {
        if (proj_tail > 0 && (proj_tail % 2) == 0) proj_dim = std::max(1, proj_tail / 2);
    }
    if (image_dim <= 0 || latent_dim <= 0 || proj_dim <= 0 || out_dim < image_dim + 2 * latent_dim + 2 * proj_dim) {
        throw std::runtime_error("Model::backwardStepVAEText: invalid dims");
    }

    const int recon_n = (!recon_is_ce) ? std::min(image_dim, static_cast<int>(target->size())) : 0;

    const int mu_off = image_dim;
    const int lv_off = image_dim + latent_dim;
    const size_t imgp_off = static_cast<size_t>(image_dim + 2 * latent_dim);
    const size_t txtp_off = imgp_off + static_cast<size_t>(proj_dim);
    const float eps = 1e-8f;
    const float na = std::max(eps, norm2_f(pred, imgp_off, proj_dim));
    const float nb = std::max(eps, norm2_f(pred, txtp_off, proj_dim));
    const float dp = dot_f(pred, imgp_off, pred, txtp_off, proj_dim);
    const float cos_sim = dp / (na * nb);
    const float align = (align_weight > 0.0f) ? (align_weight * (1.0f - cos_sim)) : 0.0f;

    // Metrics (unscaled)
    double recon = 0.0;
    float wass = 0.0f;
    float temp = 0.0f;

    if (recon_is_ce) {
        const int sl = std::max(1, seq_len_cfg);
        const int vv = std::max(2, vocab_cfg);
        const int valid_n = std::min(static_cast<int>(text_ids.size()), sl);
        int count = 0;
        for (int t = 0; t < valid_n; ++t) {
            const int y = text_ids[static_cast<size_t>(t)];
            if (pad_id >= 0 && y == pad_id) continue;
            if (y < 0 || y >= vv) continue;
            const size_t base = static_cast<size_t>(t) * static_cast<size_t>(vv);

            float m = -1e30f;
            for (int j = 0; j < vv; ++j) {
                const float v = pred[base + static_cast<size_t>(j)];
                if (v > m) m = v;
            }
            double sum = 0.0;
            for (int j = 0; j < vv; ++j) {
                sum += std::exp(static_cast<double>(pred[base + static_cast<size_t>(j)] - m));
            }
            const double lse = static_cast<double>(m) + std::log(std::max(1e-30, sum));
            recon += lse - static_cast<double>(pred[base + static_cast<size_t>(y)]);
            count += 1;
        }
        if (count > 0) recon /= static_cast<double>(count);
    } else {
        if (recon_loss == "l1" || recon_loss == "mae") {
            for (int i = 0; i < recon_n; ++i) {
                const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>((*target)[static_cast<size_t>(i)]);
                recon += std::abs(d);
            }
            recon /= static_cast<double>(std::max(1, recon_n));
        } else {
            for (int i = 0; i < recon_n; ++i) {
                const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>((*target)[static_cast<size_t>(i)]);
                recon += d * d;
            }
            recon /= static_cast<double>(std::max(1, recon_n));
        }

        const auto mp = compute_moments_prefix(pred, 0, recon_n);
        const auto mt = compute_moments_prefix(*target, 0, recon_n);
        const double vp = std::max(mp.var, 1e-12);
        const double vt = std::max(mt.var, 1e-12);
        const double w2 = (mt.mean - mp.mean) * (mt.mean - mp.mean) + (std::sqrt(vt) - std::sqrt(vp)) * (std::sqrt(vt) - std::sqrt(vp));
        wass = static_cast<float>(std::sqrt(std::max(0.0, w2)));
        temp = static_cast<float>(pearson_corr_prefix(pred, 0, *target, 0, recon_n));
    }

    double kl = 0.0;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu_f = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const double mu = static_cast<double>(mu_f);
        const double ev = std::exp(static_cast<double>(lv));
        kl += 0.5 * (mu * mu + ev - 1.0 - static_cast<double>(lv));
    }
    kl /= static_cast<double>(std::max(1, latent_dim));

    float marker_scale = 1.0f;
    if (!recon_is_ce && (marker_wass_eff > 0.0f || marker_temp_eff > 0.0f)) {
        const float temp_pen = 1.0f - std::clamp(temp, -1.0f, 1.0f);
        marker_scale = 1.0f + marker_wass_eff * wass + marker_temp_eff * temp_pen;
        marker_scale = std::clamp(marker_scale, 0.1f, marker_scale_max);
    }

    const double total_loss = static_cast<double>(marker_scale) * recon + static_cast<double>(beta_eff) * kl + static_cast<double>(align);

    // Build grad
    scratch_packed_grad_.resize(static_cast<size_t>(out_dim));
    std::fill(scratch_packed_grad_.begin(), scratch_packed_grad_.end(), 0.0f);
    auto& grad = scratch_packed_grad_;
    if (recon_is_ce) {
        const int sl = std::max(1, seq_len_cfg);
        const int vv = std::max(2, vocab_cfg);
        const int valid_n = std::min(static_cast<int>(text_ids.size()), sl);

        int count = 0;
        for (int t = 0; t < valid_n; ++t) {
            const int y = text_ids[static_cast<size_t>(t)];
            if (pad_id >= 0 && y == pad_id) continue;
            if (y < 0 || y >= vv) continue;
            count += 1;
        }
        const float scale = (count > 0) ? ((grad_scale * marker_scale) / static_cast<float>(count)) : 0.0f;

        for (int t = 0; t < valid_n; ++t) {
            const int y = text_ids[static_cast<size_t>(t)];
            if (pad_id >= 0 && y == pad_id) continue;
            if (y < 0 || y >= vv) continue;
            const size_t base = static_cast<size_t>(t) * static_cast<size_t>(vv);

            float m = -1e30f;
            for (int j = 0; j < vv; ++j) {
                const float v = pred[base + static_cast<size_t>(j)];
                if (v > m) m = v;
            }
            double sum = 0.0;
            for (int j = 0; j < vv; ++j) {
                sum += std::exp(static_cast<double>(pred[base + static_cast<size_t>(j)] - m));
            }
            const double inv_denom = 1.0 / std::max(1e-30, sum);
            for (int j = 0; j < vv; ++j) {
                const double p = std::exp(static_cast<double>(pred[base + static_cast<size_t>(j)] - m)) * inv_denom;
                grad[base + static_cast<size_t>(j)] = scale * static_cast<float>(p);
            }
            grad[base + static_cast<size_t>(y)] -= scale;
        }
    } else {
        if (recon_loss == "l1" || recon_loss == "mae") {
            const float recon_scale = (1.0f / static_cast<float>(std::max(1, recon_n))) * grad_scale * marker_scale;
            for (int i = 0; i < recon_n; ++i) {
                const float d = pred[static_cast<size_t>(i)] - (*target)[static_cast<size_t>(i)];
                const float s = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
                grad[static_cast<size_t>(i)] = recon_scale * s;
            }
        } else {
            const float recon_scale = (2.0f / static_cast<float>(std::max(1, recon_n))) * grad_scale * marker_scale;
            for (int i = 0; i < recon_n; ++i) {
                grad[static_cast<size_t>(i)] = recon_scale * (pred[static_cast<size_t>(i)] - (*target)[static_cast<size_t>(i)]);
            }
        }
    }

    const float kl_scale = ((latent_dim > 0) ? (beta_eff / static_cast<float>(latent_dim)) : 0.0f) * grad_scale;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = pred[static_cast<size_t>(mu_off + i)];
        const float lv_raw = pred[static_cast<size_t>(lv_off + i)];
        const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
        const float in_range = (lv_raw >= logvar_min && lv_raw <= logvar_max) ? 1.0f : 0.0f;
        grad[static_cast<size_t>(mu_off + i)] = kl_scale * mu;
        grad[static_cast<size_t>(lv_off + i)] = kl_scale * (0.5f * (std::exp(lv) - 1.0f)) * in_range;
    }

    if (align_weight > 0.0f) {
        const float inv_na = 1.0f / na;
        const float inv_nb = 1.0f / nb;
        const float w = align_weight * grad_scale;
        for (int i = 0; i < proj_dim; ++i) {
            const float a = pred[imgp_off + static_cast<size_t>(i)];
            const float b = pred[txtp_off + static_cast<size_t>(i)];
            const float u = a * inv_na;
            const float v = b * inv_nb;
            const float dcos_da = (v - cos_sim * u) * inv_na;
            const float dcos_db = (u - cos_sim * v) * inv_nb;
            grad[imgp_off + static_cast<size_t>(i)] = -w * dcos_da;
            grad[txtp_off + static_cast<size_t>(i)] = -w * dcos_db;
        }
    }

    backwardPass(grad);

    // Gradient norm (monitoring)
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (const auto& layer : layers) {
        for (float g : layer.grad_weights) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
        for (float g : layer.grad_bias) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
    }

    VAEStepStats stats;
    stats.loss = static_cast<float>(total_loss);
    stats.mse = static_cast<float>(recon);
    stats.kl = static_cast<float>(kl);
    stats.wass = wass;
    stats.temp = temp;
    stats.align = align;
    stats.kl_beta_effective = beta_eff;
    stats.latent_dim = latent_dim;
    stats.grad_norm = static_cast<float>(std::sqrt(sum_sq));
    stats.grad_max_abs = max_abs;
    return stats;
}

Layer* Model::getLayerByName(const std::string& name) {
    for (auto& layer : layers) {
        if (layer.name == name) {
            return &layer;
        }
    }
    return nullptr;  // Layer not found
}

// === méthodes utilitaires simples (déjà présentes) ===
void Model::setDensity(double d) { densityFactor = (d > 0.0 ? d : 1.0); }
double Model::getDensity() const { return densityFactor; }

void Model::push(const std::string &name, const std::string &type, size_t params_count) {
    // Normaliser le type et créer le layer avec enum
    std::string normalized_type = normalize_type(type);
    Layer layer(name, normalized_type, params_count);
    
    // Le constructeur Layer a déjà converti string -> enum
    // Vérifier que c'est supporté
    if (layer.type_enum == LayerType::UNKNOWN) {
        std::cerr << "❌ ERROR: Unknown layer type '" << type << "' (normalized: '" 
                  << normalized_type << "')" << std::endl;
        log_supported_types();
        throw std::runtime_error("Unknown layer type: " + type);
    }
    
    // Si des dimensions sont configurées dans modelConfig, les appliquer
    if (modelConfig.contains("in_channels")) {
        layer.in_channels = modelConfig["in_channels"];
    }
    if (modelConfig.contains("out_channels")) {
        layer.out_channels = modelConfig["out_channels"];
    }
    if (modelConfig.contains("height")) {
        layer.input_height = modelConfig["height"];
    }
    if (modelConfig.contains("width")) {
        layer.input_width = modelConfig["width"];
    }
    if (modelConfig.contains("kernel")) {
        layer.kernel_size = modelConfig["kernel"];
    }
    if (modelConfig.contains("stride")) {
        layer.stride = modelConfig["stride"];
    }
    if (modelConfig.contains("padding")) {
        layer.padding = modelConfig["padding"];
    }
    
    // Calculer les dimensions de sortie pour Conv2D
    if ((normalized_type == "Conv2d" || normalized_type == "ConvTranspose2d") && layer.kernel_size > 0) {
        if (normalized_type == "Conv2d") {
            layer.output_height = (layer.input_height + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
            layer.output_width = (layer.input_width + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        } else { // ConvTranspose2d
            layer.output_height = (layer.input_height - 1) * layer.stride - 2 * layer.padding + layer.kernel_size;
            layer.output_width = (layer.input_width - 1) * layer.stride - 2 * layer.padding + layer.kernel_size;
        }
    }
    
    // Détecter automatiquement le type de branche basé sur le nom du layer
    layer.detectBranchType();
    
    layers.push_back(layer);
}

size_t Model::totalParamCount() const {
    size_t s = 0;
    for (const auto &L : layers) s += L.params_count;
    return s;
}

void Model::allocateParams() {
    size_t tot = totalParamCount();
    
    auto& allocator = DynamicTensorAllocator::instance();
    
    std::cout << "📦 Allocation de " << layers.size() << " blocs de poids (" << tot << " paramètres au total)..." << std::endl;
    
    // NOUVEAU: Allouer un tensor par layer au lieu d'un tensor par paramètre
    layer_weight_blocks.clear();
    layer_weight_blocks.resize(layers.size());
    
    for (size_t i = 0; i < layers.size(); ++i) {
        size_t layer_param_count = layers[i].params_count;
        
        if (layer_param_count > 0) {
            // ⚠️ CRITIQUE: Allocation dynamique via MemoryGuard (passe par DynamicTensorAllocator)
            // Le flag 'true' force l'allocation à passer par requestAllocation()
            layer_weight_blocks[i] = tensor(layer_param_count, true);
            
            // Lier le tensor au layer
            layers[i].weight_block = &layer_weight_blocks[i];
            
            std::cout << "  Layer " << i << " (" << layers[i].name << "): " 
                      << layer_param_count << " paramètres dans 1 tensor" << std::endl;
        }
    }
    
    std::cout << "✓ " << layers.size() << " blocs de poids créés (1 tensor par layer)" << std::endl;
}

void Model::initializeWeights(const std::string &method, unsigned int seed) {
    if (params_frozen_) {
        throw std::runtime_error("Model::initializeWeights: parameters are frozen");
    }
    if (layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot initialize weights: weight blocks not allocated" << std::endl;
        return;
    }
    
    auto& allocator = DynamicTensorAllocator::instance();
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
    std::cout << "🎲 Initializing weights using " << method << " method (bloc par layer)..." << std::endl;
    
    auto frozen_prefixes = [&]() -> std::vector<std::string> {
        std::vector<std::string> out;
        if (modelConfig.contains("frozen_layer_prefixes") && modelConfig["frozen_layer_prefixes"].is_array()) {
            for (const auto& v : modelConfig["frozen_layer_prefixes"]) {
                if (v.is_string()) out.push_back(v.get<std::string>());
            }
        }
        return out;
    }();

    auto is_frozen_layer = [&](const Layer& l) -> bool {
        if (frozen_prefixes.empty()) return false;
        for (const auto& p : frozen_prefixes) {
            if (p.empty()) continue;
            if (l.name.rfind(p, 0) == 0) return true;
        }
        return false;
    };

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];
        
        if (layer.params_count == 0 || !layer.weight_block) continue;
        if (is_frozen_layer(layer)) continue;
        
        // Afficher progression tous les 10 layers
        if (layer_idx % 10 == 0) {
            std::cout << "  Initializing layer " << layer_idx << "/" << layers.size() 
                      << " (" << layer.name << ")..." << std::endl;
        }
        
        // fan_in/fan_out: utiliser les dimensions réelles quand disponibles
        int fan_in = 0;
        int fan_out = 0;

        if (layer.type_enum == LayerType::Linear && layer.in_features > 0 && layer.out_features > 0) {
            fan_in = layer.in_features;
            fan_out = layer.out_features;
        } else {
            // Estimation fan_in/fan_out depuis params_count
            int fan_estimate = static_cast<int>(std::sqrt(static_cast<float>(layer.params_count)));
            fan_in = std::max(fan_estimate, 32);
            fan_out = std::max(fan_estimate, 32);
        }
        
        float std_dev = 0.01f;
        
        if (method == "xavier" || method == "glorot") {
            std_dev = std::sqrt(2.0f / (fan_in + fan_out));
        }
        else if (method == "he" || method == "kaiming") {
            std_dev = 1.5f * std::sqrt(2.0f / fan_in);
        }
        else if (method == "normal") {
            std_dev = 0.05f;
        }
        
        std::normal_distribution<float> dist(0.0f, std_dev);
        
        // Déterminer précisément la zone bias quand possible
        const size_t num_weights = layer.params_count;
        size_t num_pure_weights = num_weights;

        if (layer.type_enum == LayerType::Linear && layer.in_features > 0 && layer.out_features > 0) {
            const size_t expected_w = static_cast<size_t>(layer.in_features) * static_cast<size_t>(layer.out_features);
            const size_t expected_b = layer.use_bias ? static_cast<size_t>(layer.out_features) : 0;
            if (expected_w + expected_b == num_weights) {
                num_pure_weights = expected_w;
            } else {
                // Fallback si le comptage ne correspond pas exactement
                size_t estimated_bias = std::min(expected_b, num_weights / 10);
                num_pure_weights = num_weights - estimated_bias;
            }
        } else {
            // Heuristique générique
            size_t estimated_bias = static_cast<size_t>(fan_out);
            if (estimated_bias > num_weights / 10) {
                estimated_bias = num_weights / 10;
            }
            num_pure_weights = num_weights - estimated_bias;
        }
        
        // NOUVEAU: Initialiser directement le weight_block du layer
        float* weights_data = layer.weight_block->getData();
        
        for (size_t i = 0; i < num_weights; ++i) {
            float value;
            
            // Bias initialisé à 0, weights ~ N(0,std²)
            value = (i >= num_pure_weights) ? 0.0f : dist(gen);
            
            // Clip direct sans tanh (préserve magnitude)
            value = std::clamp(value, -3.0f, 3.0f);  // ±3σ capture 99.7%
            
            weights_data[i] = value;
        }
    }
    
    std::cout << "✓ Weights initialized (" << layers.size() << " layers, " << totalParamCount() << " parameters)" << std::endl;
}

void Model::updateWeightsWithNoise(float learning_rate, float noise_std) {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    std::cerr << "⚠️ updateWeightsWithNoise() est obsolète" << std::endl;
}

std::vector<uint16_t> Model::getWeights() const {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    return std::vector<uint16_t>();
}

void Model::setTokenizer(const Tokenizer &t) {
    tokenizer = t;
    hasTokenizer = true;
    // Garder l'encoder compatible avec la taille vocab du tokenizer.
    // Utile lorsque le tokenizer est chargé après construction.
    encoder.ensureVocabSize(tokenizer.getVocabSize());
    encoder.ensureSpecialEmbeddings();
    hasEncoder = true;
}

void Model::setEncoder(const Encoder &e) {
    encoder = e;
    encoder.ensureSpecialEmbeddings();
    hasEncoder = true;
}

void Model::forward(std::vector<uint8_t> &out_uint8) const {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    // Utilisez forwardPass() à la place
    out_uint8.clear();
}

void Model::setOutputTarget(const std::vector<uint8_t> &target) {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
}

void Model::setSerializedOptimizer(Optimizer opt) {
    // If runtime moments exist (mv_by_param_ptr), pack them into flat m/v vectors
    // in the deterministic order of layers. This keeps checkpoint save/load working.
    // IMPORTANT: we clear mv_by_param_ptr after packing to avoid holding moments twice.
    if (!opt.mv_by_param_ptr.empty()) {
        size_t total = 0;
        for (const auto& layer : layers) {
            if (!layer.weight_block || layer.params_count == 0) continue;
            total += layer.getWeightsSize();
        }

        opt.m.assign(total, 0.0f);
        opt.v.assign(total, 0.0f);

        size_t off = 0;
        for (const auto& layer : layers) {
            if (!layer.weight_block || layer.params_count == 0) continue;
            const float* p = layer.weight_block->getData();
            const size_t n = layer.getWeightsSize();
            const std::uintptr_t key = reinterpret_cast<std::uintptr_t>(p);
            auto it = opt.mv_by_param_ptr.find(key);
            if (it != opt.mv_by_param_ptr.end()) {
                const auto& blk = it->second;
                const size_t nn = std::min(n, blk.m.size());
                for (size_t i = 0; i < nn; ++i) opt.m[off + i] = blk.m[i];
                const size_t nv = std::min(n, blk.v.size());
                for (size_t i = 0; i < nv; ++i) opt.v[off + i] = blk.v[i];
            }
            off += n;
        }

        opt.mv_by_param_ptr.clear();
    }

    serialized_optimizer_ = std::move(opt);
}

void Model::applyParamUpdate(float learning_rate) {
    // NOTE: Fonction obsolète - utilisez optimizerStep() pour l'entraînement moderne avec layer_weight_blocks
    std::cerr << "[DEPRECATED] applyParamUpdate() est obsolète. Utilisez optimizerStep() à la place.\n";
    return;
}

// Multi-optimizer step (SGD, Adam, AdamW)
void Model::optimizerStep(Optimizer &opt, float learning_rate, const Gradients* gradients) {
    if (params_frozen_) {
        throw std::runtime_error("Model::optimizerStep: parameters are frozen");
    }
    // NOUVEAU: Utiliser les weight_blocks au lieu de params
    if (layer_weight_blocks.empty()) return;

    auto frozen_prefixes = [&]() -> std::vector<std::string> {
        std::vector<std::string> out;
        if (modelConfig.contains("frozen_layer_prefixes") && modelConfig["frozen_layer_prefixes"].is_array()) {
            for (const auto& v : modelConfig["frozen_layer_prefixes"]) {
                if (v.is_string()) out.push_back(v.get<std::string>());
            }
        }
        return out;
    }();

    auto is_frozen_layer = [&](const Layer& l) -> bool {
        if (frozen_prefixes.empty()) return false;
        for (const auto& p : frozen_prefixes) {
            if (p.empty()) continue;
            if (l.name.rfind(p, 0) == 0) return true;
        }
        return false;
    };

    // Sécurité numérique: eps doit être strictement positif et fini.
    // Certains checkpoints/configs peuvent contenir NaN/0 ou 0, ce qui casse Adam/AdamW.
    if (!std::isfinite(opt.eps) || opt.eps <= 0.0f) {
        opt.eps = 1e-8f;
    }

    // Optional global grad clipping (L2 norm), configured via modelConfig.
    // This makes clipping work for any training loop (including grad accumulation).
    float grad_clip_norm = 0.0f;
    if (modelConfig.contains("grad_clip_norm")) {
        grad_clip_norm = std::max(0.0f, modelConfig["grad_clip_norm"].get<float>());
    } else if (modelConfig.contains("clip_norm")) {
        grad_clip_norm = std::max(0.0f, modelConfig["clip_norm"].get<float>());
    }
    if (grad_clip_norm > 0.0f) {
        double sum_sq = 0.0;
        for (const auto& layer : layers) {
            if (is_frozen_layer(layer)) continue;
            for (float g : layer.grad_weights) sum_sq += static_cast<double>(g) * static_cast<double>(g);
            for (float g : layer.grad_bias) sum_sq += static_cast<double>(g) * static_cast<double>(g);
        }
        const float norm = static_cast<float>(std::sqrt(sum_sq));
        if (norm > grad_clip_norm && norm > 1e-12f) {
            const float scale = grad_clip_norm / norm;
            for (auto& layer : layers) {
                if (is_frozen_layer(layer)) continue;
                for (auto& g : layer.grad_weights) g *= scale;
                for (auto& g : layer.grad_bias) g *= scale;
            }
        }
    }
    
    // Warmup doit fonctionner même si decay_strategy=NONE.
    float effective_lr = learning_rate;
    const size_t wu = opt.warmup_steps > 0 ? static_cast<size_t>(opt.warmup_steps) : 0ULL;
    if (wu > 0 && opt.step < wu) {
        effective_lr = opt.getCurrentLR();
    } else if (opt.decay_strategy != LRDecayStrategy::NONE) {
        effective_lr = opt.getCurrentLR();
    }
    
    // If optimizer was loaded from checkpoint, it may only have flat m/v.
    // Import them once into mv_by_param_ptr so moments match each parameter block.
    if (opt.mv_by_param_ptr.empty() && !opt.m.empty() && opt.m.size() == opt.v.size()) {
        size_t total = 0;
        for (const auto& layer : layers) {
            if (!layer.weight_block || layer.params_count == 0) continue;
            total += layer.getWeightsSize();
        }
        if (total > 0 && opt.m.size() == total) {
            size_t off = 0;
            for (const auto& layer : layers) {
                if (!layer.weight_block || layer.params_count == 0) continue;
                float* p = layer.weight_block->getData();
                const size_t n = layer.getWeightsSize();
                auto& blk = opt.ensureMomentsFor(p, n);
                for (size_t i = 0; i < n; ++i) {
                    blk.m[i] = opt.m[off + i];
                    blk.v[i] = opt.v[off + i];
                }
                off += n;
            }
        }
    }

    opt.step += 1;
    
    // NOUVEAU: Appliquer l'optimiseur sur chaque weight_block du layer
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        auto &layer = layers[layer_idx];
        
        if (!layer.weight_block || layer.params_count == 0) continue;
        if (is_frozen_layer(layer)) continue;
        if (layer.grad_weights.empty()) continue;
        
        float* weights = layer.weight_block->getData();
        size_t weight_count = layer.getWeightsSize();
        const size_t n = std::min(weight_count, layer.grad_weights.size());
        
        Optimizer::MomentBlock* moments = nullptr;
        if (opt.type == OptimizerType::ADAM || opt.type == OptimizerType::ADAMW) {
            moments = &opt.ensureMomentsFor(weights, weight_count);
        }
        
        switch (opt.type) {
            case OptimizerType::SGD: {
                // SGD simple
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    float grad = layer.grad_weights[i];
                    weights[i] -= effective_lr * grad;
                    weights[i] = std::clamp(weights[i], -3.0f, 3.0f);
                }
                break;
            }
            
            case OptimizerType::ADAM: {
                // Adam standard
                const float b1 = opt.beta1, b2 = opt.beta2;
                float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
                float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
                if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
                if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
                
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    float grad = layer.grad_weights[i];

                    float& mi = (*moments).m[i];
                    float& vi = (*moments).v[i];
                    mi = b1 * mi + (1.0f - b1) * grad;
                    vi = b2 * vi + (1.0f - b2) * grad * grad;

                    float m_hat = mi / bias_correction1;
                    float v_hat = vi / bias_correction2;
                    
                    float denom = std::sqrt(v_hat) + opt.eps;
                    weights[i] -= effective_lr * (m_hat / denom);
                    weights[i] = std::clamp(weights[i], -3.0f, 3.0f);
                }
                break;
            }
            
            case OptimizerType::ADAMW: {
                // AdamW avec weight decay découplé
                const float b1 = opt.beta1, b2 = opt.beta2;
                float bias_correction1 = 1.0f - std::pow(b1, static_cast<float>(opt.step));
                float bias_correction2 = 1.0f - std::pow(b2, static_cast<float>(opt.step));
                if (bias_correction1 <= 0.0f) bias_correction1 = 1e-8f;
                if (bias_correction2 <= 0.0f) bias_correction2 = 1e-8f;
                
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    float grad = layer.grad_weights[i];
                    float current = weights[i];

                    float& mi = (*moments).m[i];
                    float& vi = (*moments).v[i];
                    mi = b1 * mi + (1.0f - b1) * grad;
                    vi = b2 * vi + (1.0f - b2) * grad * grad;

                    float m_hat = mi / bias_correction1;
                    float v_hat = vi / bias_correction2;
                    
                    float denom = std::sqrt(v_hat) + opt.eps;
                    float weight_decay_term = opt.weight_decay * current;
                    float adam_update = effective_lr * (m_hat / denom);
                    
                    weights[i] = current - adam_update - effective_lr * weight_decay_term;
                    weights[i] = std::clamp(weights[i], -3.0f, 3.0f);
                }
                break;
            }
        }
        
        // NOTE: ne pas réinitialiser ici.
        // Les boucles d'entraînement modernes appellent zeroGradients() en début de step.
        // Garder le dernier gradient permet la sérialisation/debug (include_gradients).
    }
}

Model::DecoderOutput Model::eval(const std::vector<uint8_t> &target) const {
    DecoderOutput out;
    std::vector<uint8_t> gen;
    forward(gen);
    if (gen.size() != target.size() || gen.empty()) { out.mse = -1.0; return out; }
    double s = 0.0;
    for (size_t i = 0; i < gen.size(); ++i) {
        double d = double(gen[i]) - double(target[i]);
        s += d * d;
    }
    out.mse = s / double(gen.size());

    if (!hasTokenizer) return out;
    size_t vs = tokenizer.getVocabSize();
    if (vs == 0) return out;
    // produce trivial logits from generated image
    out.logits.assign(vs, 0.0f);
    for (size_t i = 0; i < out.logits.size(); ++i) out.logits[i] = 1.0f / float(out.logits.size());
    // top-k tokens
    for (size_t i = 0; i < std::min<size_t>(8, out.logits.size()); ++i) out.tokens.push_back(int(i));
    return out;
}

void Model::setLastEncoding(const std::vector<float> &e) { lastEncoding = e; }

// ---------------- file helpers ----------------
// convert MagicToken vector to JSON
[[maybe_unused]] static json magic_tokens_to_json(const std::vector<MagicToken> &mvec) {
    json a = json::array();
    for (const auto &m : mvec) {
        json mj;
        mj["modality_mask"] = m.modality_mask;
        mj["seed"] = m.seed;
        mj["embed"] = json::array();
        for (int i = 0; i < 8; ++i) mj["embed"].push_back(m.embed[i]);
        a.push_back(mj);
    }
    return a;
}

// read magic tokens from JSON
static void json_to_magic_tokens(const json &j, std::vector<MagicToken> &outMagic) {
    if (!j.is_array()) return;
    for (const auto &m : j) {
        MagicToken mt{};
        mt.modality_mask = m.value("modality_mask", 0u);
        mt.seed = m.value("seed", 0u);
        if (m.contains("embed") && m["embed"].is_array()) {
            for (size_t i = 0; i < 8 && i < m["embed"].size(); ++i) mt.embed[i] = m["embed"][i].get<float>();
        }
        outMagic.push_back(mt);
    }
}

// ---------------- static persistence helpers ----------------
// helper: sanitize strings in id2token array (replace control chars by '<NL>' or space)
[[maybe_unused]] static void sanitize_id2token_json(json &tokj) {
    if (!tokj.is_object() || !tokj.contains("id2token")) return;
    try {
        auto &arr = tokj["id2token"];
        if (!arr.is_array()) return;
        for (auto &el : arr) {
            if (!el.is_string()) continue;
            std::string s = el.get<std::string>();
            bool changed = false;
            for (char &c : s) {
                if (static_cast<unsigned char>(c) <= 0x1F) { // control chars
                    changed = true;
                    c = ' '; // replace with space to avoid embedded newlines
                }
            }
            if (changed) el = s;
        }
    } catch (...) { /* best-effort */ }
}

bool Model::saveCheckpoint(const Tokenizer &tokenizer, const std::vector<MagicToken> &magic_tokens, const fs::path &dir, int epoch) {
    // NOTE: Cette fonction est obsolète et a été remplacée par le module Serialization
    // Utilisez maintenant checkpoint.save() depuis Lua ou Mimir::Serialization::save_checkpoint() depuis C++
    std::cerr << "⚠️ Model::saveCheckpoint() est obsolète! Utilisez Mimir::Serialization::save_checkpoint()" << std::endl;
    std::cerr << "   Depuis Lua: checkpoint.save(model, path, {format='safetensors'})" << std::endl;
    return false;
}

static void write_u64_le(std::ofstream &f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    f.write(reinterpret_cast<char*>(b), 8);
}

// writer for a set of float32 tensors into a safetensors-like file.
// Format written:
// [8 bytes little-endian u64] header_length
// [header_length bytes UTF-8 JSON header]
// [binary blob of tensors concatenated as raw little-endian float32]
//
// Header format (JSON object) follows safetensors style:
// { "metadata": {}, "tensors": { "name": { "dtype":"f32", "shape":[N], "data":[offset, length] }, ... } }
static bool write_safetensors_file(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors, std::string *err = nullptr) {
    try {
        // prepare metadata and compute offsets
        json header;
        header["metadata"] = json::object();
        json tensors_meta = json::object();

        uint64_t offset = 0; // data offset after header
        std::vector<std::pair<const std::string*, const std::vector<float>*>> order;
        order.reserve(tensors.size());
        for (const auto &kv : tensors) order.emplace_back(&kv.first, &kv.second);

        // compute total data size to help building header offsets (we don't need it here)
        for (const auto &p : order) {
            const std::string &name = *p.first;
            const std::vector<float> &buf = *p.second;
            uint64_t byte_len = static_cast<uint64_t>(buf.size()) * sizeof(float);
            // record meta: data = [offset, length]
            json m;
            m["dtype"] = "f32";
            m["shape"] = json::array({ static_cast<uint64_t>(buf.size()) });
            m["data"] = json::array({ offset, byte_len });
            tensors_meta[name] = m;
            offset += byte_len;
        }
        header["tensors"] = tensors_meta;

        std::string header_str = header.dump();
        uint64_t header_len = static_cast<uint64_t>(header_str.size());

        // open file and write header length + header
        std::ofstream ofs(outpath.string(), std::ios::binary);
        if (!ofs) {
            if (err) *err = "failed to open output file";
            return false;
        }

        write_u64_le(ofs, header_len);
        ofs.write(header_str.data(), static_cast<std::streamsize>(header_len));

        // now write tensor data in the same order as header (order vector)
        for (const auto &p : order) {
            const std::vector<float> &buf = *p.second;
            if (!buf.empty()) {
                // write raw floats (assume host is little-endian; if not, convert)
                ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(float)));
            }
        }

        ofs.close();
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    } catch (...) {
        if (err) *err = "unknown error";
        return false;
    }
}

// Model::packToSafetensor implementation that delegates to writer above.
// Utilise une map fournie par l'appelant (nom -> float buffer).
bool Model::packToSafetensor(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors) const {
    // create parent dir
    try {
        if (outpath.has_parent_path()) fs::create_directories(outpath.parent_path());
    } catch (...) { /* ignore */ }

    std::string err;
    if (!write_safetensors_file(outpath, tensors, &err)) {
        std::cerr << "packToSafetensor: failed to write " << outpath << " : " << err << "\n";
        return false;
    }
    return true;
}

bool Model::tryLoadExistingModel(const fs::path &ckdir, const fs::path &safep, Tokenizer &outTok, Encoder &outEnc, std::vector<MagicToken> &outMagic) {
    bool loaded_any = false;
    try {
        fs::path sjson = safep; sjson += ".json";
        if (fs::exists(sjson) && fs::is_regular_file(sjson)) {
            try {
                std::ifstream f(sjson);
                if (f) {
                    json full; f >> full;
                    if (full.contains("tokenizer")) { try { outTok.from_json(full["tokenizer"]); loaded_any = true; } catch(...) {} }
                    else if (full.contains("id2token")) { json tj; tj["id2token"] = full["id2token"]; try { outTok.from_json(tj); loaded_any = true; } catch(...) {} }
                    if (full.contains("magic_tokens")) { try { json_to_magic_tokens(full["magic_tokens"], outMagic); loaded_any = true; } catch(...) {} }
                    if (full.contains("encoder")) {
                        try {
                            auto ej = full["encoder"];
                            outEnc.dim = ej.value("dim", outEnc.dim);
                            if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                                auto &rows = ej["embeddings"];
                                outEnc.vocab_size = (int)rows.size();
                                outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                                for (size_t r = 0; r < rows.size(); ++r)
                                    for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                        outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                                loaded_any = true;
                            }
                        } catch (...) {}
                    }
                    if (loaded_any) return true;
                }
            } catch (...) {
                // fallback: safep json is invalid/corrupted, ignore and continue to checkpoint folders
            }
        }
    } catch (...) {}

    try {
        if (fs::exists(ckdir) && fs::is_directory(ckdir)) {
            int best_epoch = -1; fs::path best_dir;
            for (auto &p : fs::directory_iterator(ckdir)) {
                if (!p.is_directory()) continue;
                std::string n = p.path().filename().string();
                if (n.rfind("epoch_", 0) == 0) {
                    try { int e = std::stoi(n.substr(6)); if (e > best_epoch) { best_epoch = e; best_dir = p.path(); } } catch(...) {}
                }
            }
            if (best_epoch >= 0 && !best_dir.empty()) {
                fs::path tokp = best_dir / "tokenizer.json";
                fs::path encp = best_dir / "encoder.json";
                fs::path mp = best_dir / "metadata.json";
                fs::path layersp = best_dir / "layers.json";
                fs::path embp = best_dir / "embeddings.bin";
                fs::path paramsp = best_dir / "params_data.bin";
                
                if (fs::exists(tokp)) {
                    try {
                        std::ifstream tf(tokp);
                        json tj;
                        tf >> tj;
                        outTok.from_json(tj);
                        loaded_any = true;
                    } catch(...) {
                        // tokenizer.json is invalid -> fallback to minimal tokenizer
                        try {
                            json minimal;
                            minimal["id2token"] = json::array({ "<PAD>", "<UNK>", "<SEQ>", "<MOD>", "<MAG>", "<NL>" });
                            outTok.from_json(minimal);
                            loaded_any = true;
                        } catch(...) {}
                    }
                }
                if (fs::exists(encp)) {
                    try { std::ifstream ef(encp); json ej; ef >> ej;
                        outEnc.dim = ej.value("dim", outEnc.dim);
                        if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                            auto &rows = ej["embeddings"];
                            outEnc.vocab_size = (int)rows.size();
                            outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                            for (size_t r = 0; r < rows.size(); ++r)
                                for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                    outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                            loaded_any = true;
                        }
                    } catch(...) {}
                }
                if (fs::exists(mp)) {
                    try { std::ifstream mf(mp); json mj; mf >> mj; if (mj.contains("magic_tokens")) { json_to_magic_tokens(mj["magic_tokens"], outMagic); loaded_any = true; } } catch(...) {}
                }
                
                // NOTE: Anciennes méthodes de sauvegarde/chargement supprimées
                // Utiliser maintenant le module Serialization avec checkpoint.load()
                /*
                // Charger la structure des layers
                if (fs::exists(layersp)) {
                    try {
                        if (loadLayersStructure(layersp)) {
                            std::cout << "✓ Structure des layers chargée depuis " << layersp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement de layers.json" << std::endl;
                    }
                }
                
                // Charger les embeddings
                if (fs::exists(embp)) {
                    try {
                        if (loadEmbeddings(embp)) {
                            std::cout << "✓ Embeddings chargés depuis " << embp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement des embeddings" << std::endl;
                    }
                }
                
                // Charger les données des paramètres
                if (fs::exists(paramsp)) {
                    try {
                        if (loadParamsData(paramsp)) {
                            std::cout << "✓ Données des paramètres chargées depuis " << paramsp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement des params_data" << std::endl;
                    }
                }
                */
                
                if (loaded_any) return true;
            }
        }
    } catch (...) {}

    return loaded_any;
}

// ===== Implémentation des opérations de layer =====

void Model::computeConv2D(const std::vector<float>& input, std::vector<float>& output,
                         const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                         bool use_hardware) {
    // Utilise directement l'implémentation optimisée de Conv::conv2d
    // qui gère automatiquement SIMD et CPU selon la compilation
    Conv::conv2d(input, output, params.weights, params.bias,
                in_h, in_w, in_c, out_c, params.kernel_size,
                params.stride, params.padding, params.dilation);
}

void Model::computeLinear(const std::vector<float>& input, std::vector<float>& output,
                         const LayerParams& params, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    output.resize(params.out_features, 0.0f);
    
    if (use_hardware && hasFMA()) {
        // Version hardware avec FMA saturé
        SIMD::matmul_avx2(output.data(), input.data(), params.weights.data(),
                         1, params.out_features, params.in_features);
        
        // Ajouter bias
        if (!params.bias.empty()) {
            SIMD::add_vectors_avx2(output.data(), output.data(), params.bias.data(), params.out_features);
        }
    } else {
        // Version software
        for (int o = 0; o < params.out_features; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < params.in_features; ++i) {
                sum += input[i] * params.weights[o * params.in_features + i];
            }
            if (!params.bias.empty()) sum += params.bias[o];
            output[o] = sum;
        }
    }
}

void Model::computeMaxPool2D(const std::vector<float>& input, std::vector<float>& output,
                            int in_h, int in_w, int channels, int kernel_size, int stride,
                            bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (stride < 0) stride = kernel_size;
    int out_h = (in_h - kernel_size) / stride + 1;
    int out_w = (in_w - kernel_size) / stride + 1;
    
    output.resize(out_h * out_w * channels);
    
    if (use_hardware) {
        // Version hardware avec AVX2
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; kw += 8) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (kw + 8 <= kernel_size) {
                                __m256 vals = _mm256_loadu_ps(&input[(c * in_h + ih) * in_w + iw]);
                                max_vec = _mm256_max_ps(max_vec, vals);
                            } else {
                                // Scalar fallback pour derniers éléments
                                for (int k = kw; k < kernel_size; ++k) {
                                    float val = input[(c * in_h + ih) * in_w + (ow * stride + k)];
                                    float temp[8];
                                    _mm256_storeu_ps(temp, max_vec);
                                    temp[0] = std::max(temp[0], val);
                                    max_vec = _mm256_loadu_ps(temp);
                                }
                            }
                        }
                    }
                    
                    // Horizontal max
                    float temp[8];
                    _mm256_storeu_ps(temp, max_vec);
                    float max_val = temp[0];
                    for (int i = 1; i < 8; ++i) max_val = std::max(max_val, temp[i]);
                    
                    output[(c * out_h + oh) * out_w + ow] = max_val;
                }
            }
        }
    } else {
        // Version software
        Pooling::maxpool2d(input, output, in_h, in_w, channels, kernel_size, stride);
    }
}

void Model::computeAvgPool2D(const std::vector<float>& input, std::vector<float>& output,
                            int in_h, int in_w, int channels, int kernel_size, int stride,
                            bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (stride < 0) stride = kernel_size;
    int out_h = (in_h - kernel_size) / stride + 1;
    int out_w = (in_w - kernel_size) / stride + 1;
    
    output.resize(out_h * out_w * channels);
    
    if (use_hardware) {
        // Version hardware avec AVX2
        float inv_area = 1.0f / (kernel_size * kernel_size);
        __m256 inv_vec = _mm256_set1_ps(inv_area);

        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    __m256 sum_vec = _mm256_setzero_ps();
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; kw += 8) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (kw + 8 <= kernel_size) {
                                __m256 vals = _mm256_loadu_ps(&input[(c * in_h + ih) * in_w + iw]);
                                sum_vec = _mm256_add_ps(sum_vec, vals);
                            }
                        }
                    }
                    
                    // Horizontal sum
                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                    __m128 shuf = _mm_movehdup_ps(sum128);
                    __m128 sums = _mm_add_ps(sum128, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    
                    float sum = _mm_cvtss_f32(sums) * inv_area;
                    output[(c * out_h + oh) * out_w + ow] = sum;
                }
            }
        }
    } else {
        // Version software
        Pooling::avgpool2d(input, output, in_h, in_w, channels, kernel_size, stride);
    }
}

void Model::computeActivation(std::vector<float>& data, const std::string& activation_type,
                             float param, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (activation_type == "gelu" && use_hardware) {
        SIMD::gelu_forward_avx2(data.data(), data.data(), data.size());
    } else if (activation_type == "relu") {
        if (use_hardware) {
            size_t n = data.size();
            __m256 zero = _mm256_setzero_ps();

            const size_t vecN = n & ~static_cast<size_t>(7);
            for (size_t i = 0; i < vecN; i += 8) {
                __m256 vals = _mm256_loadu_ps(&data[i]);
                vals = _mm256_max_ps(vals, zero);
                _mm256_storeu_ps(&data[i], vals);
            }
            for (size_t i = vecN; i < n; ++i) {
                data[i] = std::max(0.0f, data[i]);
            }
        } else {
            relu_inplace(data);
        }
    } else if (activation_type == "leaky_relu") {
        leaky_relu_inplace(data, param);
    } else if (activation_type == "tanh") {
        tanh_inplace(data);
    } else if (activation_type == "sigmoid") {
        for (auto& v : data) v = sigmoidf(v);
    } else if (activation_type == "softmax") {
        if (use_hardware) {
            SIMD::softmax_avx2(data.data(), data.data(), data.size());
        } else {
            softmax_inplace(data);
        }
    } else if (activation_type == "elu") {
        elu_inplace(data, param);
    }
}

void Model::computeBatchNorm(std::vector<float>& data, const std::vector<float>& gamma,
                            const std::vector<float>& beta, const std::vector<float>& running_mean,
                            const std::vector<float>& running_var, int batch_size, int channels,
                            int spatial_size, float eps, bool training, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (use_hardware) {
        // Version hardware avec AVX2
        __m256 eps_vec = _mm256_set1_ps(eps);

        for (int c = 0; c < channels; ++c) {
            float mean = running_mean[c];
            float var = running_var[c];
            
            if (training) {
                // Calculer mean (AVX2)
                __m256 mean_vec = _mm256_setzero_ps();
                int total_size = batch_size * spatial_size;
                int count = 0;
                
                for (int b = 0; b < batch_size; ++b) {
                    for (int s = 0; s < spatial_size; s += 8) {
                        if (s + 8 <= spatial_size) {
                            __m256 vals = _mm256_loadu_ps(&data[b * channels * spatial_size + c * spatial_size + s]);
                            mean_vec = _mm256_add_ps(mean_vec, vals);
                            count += 8;
                        }
                    }
                }
                
                // Horizontal sum pour mean
                float temp[8];
                _mm256_storeu_ps(temp, mean_vec);
                mean = 0.0f;
                for (int i = 0; i < 8; ++i) mean += temp[i];
                for (int b = 0; b < batch_size; ++b) {
                    for (int s = count; s < spatial_size; ++s) {
                        mean += data[b * channels * spatial_size + c * spatial_size + s];
                    }
                }
                mean /= total_size;
                
                // Calculer variance
                var = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    for (int s = 0; s < spatial_size; ++s) {
                        float diff = data[b * channels * spatial_size + c * spatial_size + s] - mean;
                        var += diff * diff;
                    }
                }
                var /= total_size;
            }
            
            __m256 mean_vec = _mm256_set1_ps(mean);
            __m256 inv_std_vec = _mm256_set1_ps(1.0f / std::sqrt(var + eps));
            __m256 gamma_vec = _mm256_set1_ps(gamma[c]);
            __m256 beta_vec = _mm256_set1_ps(beta[c]);
            
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < spatial_size; s += 8) {
                    int idx = b * channels * spatial_size + c * spatial_size + s;
                    if (s + 8 <= spatial_size) {
                        __m256 vals = _mm256_loadu_ps(&data[idx]);
                        vals = _mm256_sub_ps(vals, mean_vec);
                        vals = _mm256_mul_ps(vals, inv_std_vec);
                        vals = _mm256_mul_ps(vals, gamma_vec);
                        vals = _mm256_add_ps(vals, beta_vec);
                        _mm256_storeu_ps(&data[idx], vals);
                    } else {
                        for (int i = s; i < spatial_size; ++i) {
                            int idx2 = b * channels * spatial_size + c * spatial_size + i;
                            data[idx2] = (data[idx2] - mean) * (1.0f / std::sqrt(var + eps)) * gamma[c] + beta[c];
                        }
                    }
                }
            }
        }
    } else {
        // Version software
        Normalization::batch_norm(data, gamma, beta, running_mean, running_var,
                                 batch_size, channels, spatial_size, eps, training);
    }
}

void Model::computeLayerNorm(std::vector<float>& data, const std::vector<float>& gamma,
                            const std::vector<float>& beta, int normalized_size,
                            float eps, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    if (use_hardware) {
        // Version hardware avec AVX2
        int num_groups = data.size() / normalized_size;
        __m256 eps_vec = _mm256_set1_ps(eps);

        for (int g = 0; g < num_groups; ++g) {
            // Calculer mean
            __m256 mean_vec = _mm256_setzero_ps();
            int base = g * normalized_size;
            
            int i = 0;
            for (; i + 8 <= normalized_size; i += 8) {
                __m256 vals = _mm256_loadu_ps(&data[base + i]);
                mean_vec = _mm256_add_ps(mean_vec, vals);
            }
            
            float temp[8];
            _mm256_storeu_ps(temp, mean_vec);
            float mean = 0.0f;
            for (int j = 0; j < 8; ++j) mean += temp[j];
            for (; i < normalized_size; ++i) mean += data[base + i];
            mean /= normalized_size;
            
            // Calculer variance
            float var = 0.0f;
            for (int i = 0; i < normalized_size; ++i) {
                float diff = data[base + i] - mean;
                var += diff * diff;
            }
            var /= normalized_size;
            
            __m256 mean_vec_bc = _mm256_set1_ps(mean);
            __m256 inv_std = _mm256_set1_ps(1.0f / std::sqrt(var + eps));
            
            // Normaliser
            for (int i = 0; i + 8 <= normalized_size; i += 8) {
                __m256 vals = _mm256_loadu_ps(&data[base + i]);
                __m256 gamma_vec = _mm256_loadu_ps(&gamma[i]);
                __m256 beta_vec = _mm256_loadu_ps(&beta[i]);
                
                vals = _mm256_sub_ps(vals, mean_vec_bc);
                vals = _mm256_mul_ps(vals, inv_std);
                vals = _mm256_mul_ps(vals, gamma_vec);
                vals = _mm256_add_ps(vals, beta_vec);
                
                _mm256_storeu_ps(&data[base + i], vals);
            }
            
            // Remaining elements
            for (int i = (normalized_size / 8) * 8; i < normalized_size; ++i) {
                data[base + i] = (data[base + i] - mean) * (1.0f / std::sqrt(var + eps)) * gamma[i] + beta[i];
            }
        }
    } else {
        // Version software
        Normalization::layer_norm(data, gamma, beta, normalized_size, eps);
    }
}

void Model::computeConvTranspose2D(const std::vector<float>& input, std::vector<float>& output,
                                  const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                                  bool use_hardware) {
    // ConvTranspose est complexe, utiliser version software
    Conv::conv_transpose2d(input, output, params.weights, params.bias,
                          in_h, in_w, in_c, out_c, params.kernel_size,
                          params.stride, params.padding);
}

void Model::computeAttention(const std::vector<float>& query, const std::vector<float>& key,
                            const std::vector<float>& value, std::vector<float>& output,
                            int seq_len, int d_model, int num_heads, bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    int head_dim = d_model / num_heads;
    output.resize(seq_len * d_model, 0.0f);
    
    std::vector<float> attention_scores(seq_len * seq_len);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    for (int h = 0; h < num_heads; ++h) {
        // Q * K^T avec scaling
        if (use_hardware) {
            SIMD::matmul_transpose_avx2(attention_scores.data(),
                                       &query[h * head_dim], &key[h * head_dim],
                                       seq_len, seq_len, head_dim);
        } else {
            // Scalar version
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < head_dim; ++k) {
                        sum += query[(i * d_model) + h * head_dim + k] * key[(j * d_model) + h * head_dim + k];
                    }
                    attention_scores[i * seq_len + j] = sum * scale;
                }
            }
        }
        
        // Softmax sur chaque ligne
        for (int i = 0; i < seq_len; ++i) {
            std::vector<float> row(attention_scores.begin() + i * seq_len,
                                 attention_scores.begin() + (i + 1) * seq_len);
            if (use_hardware) {
                SIMD::softmax_avx2(row.data(), row.data(), seq_len);
            } else {
                softmax_inplace(row);
            }
            std::copy(row.begin(), row.end(), attention_scores.begin() + i * seq_len);
        }
        
        // Attention * V
        if (use_hardware) {
            std::vector<float> head_output(seq_len * head_dim);
            SIMD::matmul_avx2(head_output.data(), attention_scores.data(),
                            &value[h * head_dim], seq_len, head_dim, seq_len);
            
            // Copier dans output
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < head_dim; ++j) {
                    output[i * d_model + h * head_dim + j] = head_output[i * head_dim + j];
                }
            }
        } else {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < head_dim; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < seq_len; ++k) {
                        sum += attention_scores[i * seq_len + k] * value[(k * d_model) + h * head_dim + j];
                    }
                    output[i * d_model + h * head_dim + j] = sum;
                }
            }
        }
    }
}

void Model::conv2d_same(const std::vector<float> &in, std::vector<float> &out, int W, int H, const std::vector<float> &kernel, int ksize)
{

    out.assign(W * H, 0.0f);
    // Utiliser la version optimis\u00e9e si disponible
    if (global_use_hardware && hasAVX2() && hasFMA()) {
        LayerParams params;
        params.weights = kernel;
        params.kernel_size = ksize;
        params.stride = 1;
        params.padding = ksize / 2;
        
        computeConv2D(in, out, params, H, W, 1, 1, true);
        return;
    }
    
    // Fallback software
    const int khalf = ksize / 2;
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            float sum = 0.0f;
            for (int ky = 0; ky < ksize; ++ky)
            {
                const int iy = y + ky - khalf;
                if (iy < 0 || iy >= H)
                    continue;
                for (int kx = 0; kx < ksize; ++kx)
                {
                    const int ix = x + kx - khalf;
                    if (ix < 0 || ix >= W)
                        continue;
                    sum += in[iy * W + ix] * kernel[ky * ksize + kx];
                }
            }
            out[y * W + x] = sum;
        }
    }
}

// --- Définitions vides pour méthodes virtuelles afin de fournir la vtable ---
void Model::buildBackboneUNet(int /*stages*/, int /*blocks_per_stage*/, int /*bottleneck_depth*/) { /* noop */ }
void Model::injectMagicToken(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildTextBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildAudioBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildImageBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildVideoBranch(const MagicToken & /*tok*/) { /* noop */ }

// ============================= 
// Branch Operations Implementation
// =============================

void Model::computeBranchMerge(const std::vector<float>& branch1, 
                               const std::vector<float>& branch2,
                               std::vector<float>& output,
                               MergeOperation merge_op,
                               bool use_hardware) {
    use_hardware = use_hardware && global_use_hardware && hasAVX2();
    
    size_t size = branch1.size();
    output.resize(size);
    
    switch (merge_op) {
        case MergeOperation::ADD: {
            if (use_hardware) {
                #ifdef __AVX2__
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 result = _mm256_add_ps(a, b);
                    _mm256_storeu_ps(&output[i], result);
                }
                // Éléments restants
                for (; i < size; ++i) {
                    output[i] = branch1[i] + branch2[i];
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] + branch2[i];
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] + branch2[i];
                }
            }
            break;
        }
        
        case MergeOperation::MULTIPLY: {
            if (use_hardware) {
                #ifdef __AVX2__
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 result = _mm256_mul_ps(a, b);
                    _mm256_storeu_ps(&output[i], result);
                }
                for (; i < size; ++i) {
                    output[i] = branch1[i] * branch2[i];
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] * branch2[i];
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = branch1[i] * branch2[i];
                }
            }
            break;
        }
        
        case MergeOperation::MAX: {
            if (use_hardware) {
                #ifdef __AVX2__
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 result = _mm256_max_ps(a, b);
                    _mm256_storeu_ps(&output[i], result);
                }
                for (; i < size; ++i) {
                    output[i] = std::max(branch1[i], branch2[i]);
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = std::max(branch1[i], branch2[i]);
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = std::max(branch1[i], branch2[i]);
                }
            }
            break;
        }
        
        case MergeOperation::AVERAGE: {
            if (use_hardware) {
                #ifdef __AVX2__
                __m256 half = _mm256_set1_ps(0.5f);
                size_t i = 0;
                for (; i + 8 <= size; i += 8) {
                    __m256 a = _mm256_loadu_ps(&branch1[i]);
                    __m256 b = _mm256_loadu_ps(&branch2[i]);
                    __m256 sum = _mm256_add_ps(a, b);
                    __m256 result = _mm256_mul_ps(sum, half);
                    _mm256_storeu_ps(&output[i], result);
                }
                for (; i < size; ++i) {
                    output[i] = (branch1[i] + branch2[i]) * 0.5f;
                }
                #else
                for (size_t i = 0; i < size; ++i) {
                    output[i] = (branch1[i] + branch2[i]) * 0.5f;
                }
                #endif
            } else {
                for (size_t i = 0; i < size; ++i) {
                    output[i] = (branch1[i] + branch2[i]) * 0.5f;
                }
            }
            break;
        }
        
        case MergeOperation::CONCATENATE: {
            // Concaténation simple
            output.resize(branch1.size() + branch2.size());
            std::copy(branch1.begin(), branch1.end(), output.begin());
            std::copy(branch2.begin(), branch2.end(), output.begin() + branch1.size());
            break;
        }
        
        default: {
            // Par défaut, addition
            std::copy(branch1.begin(), branch1.end(), output.begin());
            for (size_t i = 0; i < size; ++i) {
                output[i] += branch2[i];
            }
            break;
        }
    }
}

void Model::computeBranchSplit(const std::vector<float>& input,
                               std::vector<std::vector<float>>& outputs,
                               const std::vector<int>& split_sizes) {
    outputs.resize(split_sizes.size());
    size_t offset = 0;
    
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        outputs[i].resize(split_sizes[i]);
        std::copy(input.begin() + offset, 
                  input.begin() + offset + split_sizes[i], 
                  outputs[i].begin());
        offset += split_sizes[i];
    }
}

void Model::detectAndSetupBranches() {
    // Parcourir tous les layers et détecter automatiquement les types de branches
    for (auto& layer : layers) {
        layer.detectBranchType();
    }
    
    // Analyser la structure pour identifier les connexions entre branches
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        
        // Si c'est un layer résiduel, chercher le layer source
        if (layer.branch_type == BranchType::RESIDUAL) {
            // Par convention, le shortcut se connecte généralement plusieurs layers en arrière
            // Chercher un layer avec un nom similaire mais sans "shortcut" ou "residual"
            std::string base_name = layer.name;
            size_t pos = base_name.find("_shortcut");
            if (pos == std::string::npos) {
                pos = base_name.find("_residual");
            }
            
            if (pos != std::string::npos) {
                base_name = base_name.substr(0, pos);
                
                // Chercher le layer de base correspondant
                for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                    if (layers[j].name.find(base_name) != std::string::npos && 
                        j != static_cast<int>(i)) {
                        layer.branch_sources.push_back(j);
                        layers[j].is_branch_point = true;
                        break;
                    }
                }
            }
        }
    }
    
    std::cout << "✓ Détection des branches terminée. Trouvé:" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].requiresBranchComputation()) {
            std::cout << "  - Layer " << i << " (" << layers[i].name << "): ";
            if (layers[i].branch_type == BranchType::RESIDUAL) {
                std::cout << "RESIDUAL";
            } else if (layers[i].branch_type == BranchType::SKIP_CONNECTION) {
                std::cout << "SKIP_CONNECTION";
            } else if (layers[i].is_branch_point) {
                std::cout << "BRANCH_POINT";
            } else if (layers[i].is_merge_point) {
                std::cout << "MERGE_POINT";
            }
            std::cout << std::endl;
        }
    }
}

void Model::executeBranchComputation(int layer_idx, 
                                    std::vector<std::vector<float>>& layer_outputs,
                                    bool training) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
        return;
    }
    
    auto& layer = layers[layer_idx];
    
    if (!layer.requiresBranchComputation()) {
        return;
    }
    
    // Si c'est un point de fusion (residual, skip connection, etc.)
    if (layer.branch_type == BranchType::RESIDUAL && !layer.branch_sources.empty()) {
        // Récupérer la sortie du layer source
        int source_idx = layer.branch_sources[0];
        if (source_idx >= 0 && source_idx < static_cast<int>(layer_outputs.size())) {
            // Fusionner avec l'opération spécifiée
            std::vector<float> merged_output;
            computeBranchMerge(layer_outputs[layer_idx], 
                             layer_outputs[source_idx],
                             merged_output,
                             layer.merge_op,
                             true);
            layer_outputs[layer_idx] = std::move(merged_output);
        }
    }
    else if (layer.branch_type == BranchType::SPLIT) {
        // Pour les splits, on doit diviser la sortie
        // Ceci sera géré au niveau du forward pass principal
    }
}

void Model::backpropThroughBranch(int layer_idx,
                                 const std::vector<float>& grad_output,
                                 std::vector<std::vector<float>>& layer_gradients) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
        return;
    }
    
    auto& layer = layers[layer_idx];
    
    if (!layer.requiresBranchComputation()) {
        return;
    }
    
    // Backprop à travers les connexions de branche
    if (layer.branch_type == BranchType::RESIDUAL && !layer.branch_sources.empty()) {
        // Pour une connexion résiduelle, le gradient se propage vers les deux branches
        int source_idx = layer.branch_sources[0];
        if (source_idx >= 0 && source_idx < static_cast<int>(layer_gradients.size())) {
            // Le gradient du résidual se propage tel quel vers la branche source
            if (layer_gradients[source_idx].empty()) {
                layer_gradients[source_idx] = grad_output;
            } else {
                // Accumuler les gradients
                for (size_t i = 0; i < grad_output.size() && i < layer_gradients[source_idx].size(); ++i) {
                    layer_gradients[source_idx][i] += grad_output[i];
                }
            }
        }
    }
}

// === Forward/Backward Pass Complet ===

std::vector<float> Model::forwardPass(const std::vector<float> &input, bool training) {
    return forwardPassView(input, training);
}

const std::vector<float>& Model::forwardPassView(const std::vector<float> &input, bool training) {
    // Vérifications préliminaires
    if (layers.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: no layers defined" << std::endl;
        return input;
    }
    
    if (layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: weights not allocated" << std::endl;
        std::cerr << "    Call allocate_params() and init_weights() first" << std::endl;
        return input;
    }
    
    // ========================================================================
    // INITIALIZATION: TensorStore + Validation
    // ========================================================================
    
    // Clear et initialiser TensorStore avec l'input principal
    clearTensorStore();
    storeTensor("x", input);
    // Alias immuable de l'entrée (utile si le graphe réutilise le nom "x" pour la sortie avec une taille différente)
    storeTensor("__input__", input);

    // Injection optionnelle d'entrées nommées (forwardPassNamed)
    if (pending_int_inputs_.has_value()) {
        clearTensorStoreInt();
        for (const auto& kv : *pending_int_inputs_) {
            storeTensorInt(kv.first, kv.second);
        }

        // Convention: "seq" = sortie tokenizer (ids). Alias utile pour modèles existants.
        // PonyXL et d'autres utilisent "text_ids".
        const auto it_seq = pending_int_inputs_->find("seq");
        if (it_seq != pending_int_inputs_->end()) {
            // store under both keys (safe overwrite ok)
            storeTensorInt("text_ids", it_seq->second);
            // Compat: beaucoup de graphes lisent "__input__" (et parfois "x") comme entrée ids.
            // Permet d'utiliser forwardPassNamed({seq=...}) avec ces architectures.
            storeTensorInt("__input__", it_seq->second);
            storeTensorInt("x", it_seq->second);
        }
    }
    if (pending_float_inputs_.has_value()) {
        for (const auto& kv : *pending_float_inputs_) {
            storeTensor(kv.first, kv.second);
        }
    }

    // Convention: "mag" et "mod" = embeddings float (médias + liaison).
    // On les injecte par défaut depuis l'Encoder si non fournis explicitement.
    if (hasEncoder) {
        const auto& mag = encoder.getMagEmbedding();
        const auto& mod = encoder.getModEmbedding();

        auto graph_uses_mag_mod = [&]() -> bool {
            for (const auto& lyr : layers) {
                for (const auto& in : lyr.inputs) {
                    if (in == "mag" || in == "mod") return true;
                }
            }
            return false;
        };

        if (graph_uses_mag_mod()) {
            int expected = 0;
            if (modelConfig.contains("d_model")) expected = std::max(0, modelConfig["d_model"].get<int>());
            else if (modelConfig.contains("text_d_model")) expected = std::max(0, modelConfig["text_d_model"].get<int>());
            if (expected > 0) {
                if (!mag.empty() && static_cast<int>(mag.size()) != expected) {
                    throw std::runtime_error("Encoder mag dim mismatch: have=" + std::to_string(mag.size()) + ", expected=" + std::to_string(expected));
                }
                if (!mod.empty() && static_cast<int>(mod.size()) != expected) {
                    throw std::runtime_error("Encoder mod dim mismatch: have=" + std::to_string(mod.size()) + ", expected=" + std::to_string(expected));
                }
            }
        }

        if (!mag.empty() && tensor_store.find("mag") == tensor_store.end()) {
            storeTensor("mag", mag);
        }
        if (!mod.empty() && tensor_store.find("mod") == tensor_store.end()) {
            storeTensor("mod", mod);
        }
    }
    pending_float_inputs_.reset();
    pending_int_inputs_.reset();
    
    // VALIDATION: Vérifier que tous les layers sont supportés (une seule fois)
    static bool validated = false;
    if (!validated) {
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].type_enum == LayerType::UNKNOWN) {
                std::cerr << "❌ ERROR: Unsupported layer type '" << layers[i].type 
                          << "' at index " << i << " (" << layers[i].name << ")" << std::endl;
                log_supported_types();
                throw std::runtime_error("Unsupported layer type: " + layers[i].type);
            }
        }
        validated = true;
        std::cerr << "✓ All " << layers.size() << " layers validated" << std::endl;
    }
    
    // État du forward
    if (training) {
        forward_state.clear();
        forward_state.is_valid = true;
    }

    const bool has_branches = training && std::any_of(
        layers.begin(), layers.end(),
        [](const Layer& l) { return l.requiresBranchComputation(); }
    );

    std::vector<std::vector<float>> all_layer_outputs;
    if (training && has_branches) {
        all_layer_outputs.reserve(layers.size());
    }
    
    // Conservation pour backward (à migrer vers TensorStore)
    if (training) {
        forward_state.layer_outputs.clear();
        forward_state.layer_outputs.reserve(layers.size());
        forward_state.layer_output_masks.clear();
        forward_state.layer_output_masks.reserve(layers.size());
        forward_state.layer_inputs_multi.clear();
        forward_state.layer_inputs_multi.reserve(layers.size());
        forward_state.layer_input_names.clear();
        forward_state.layer_input_names.reserve(layers.size());
        forward_state.layer_input_sizes_multi.clear();
        forward_state.layer_input_sizes_multi.reserve(layers.size());
    }

    auto needs_input_value_snapshot = [](const Layer& layer) -> bool {
        // Par défaut: snapshot valeurs (sécurité), sauf si le backward n'utilise que des tailles.
        // Important: pour Add/Concat/Split/Subtract/TokenMeanPool/Upsample/Identity, on évite la copie.
        if (layer.type == "Add" || layer.type == "Concat" || layer.type == "Split" || layer.type == "Subtract" ||
            layer.type == "TokenMeanPool" || layer.type == "UpsampleNearest" || layer.type == "Identity") {
            return false;
        }
        // Dropout backward se base sur un masque (output), pas sur l'input.
        if (layer.type == "Dropout" || layer.type == "Dropout2d" || layer.type == "AlphaDropout") {
            return false;
        }
        return true;
    };

    auto needs_output_mask = [](const Layer& layer) -> bool {
        if (layer.type == "Dropout" || layer.type == "Dropout2d" || layer.type == "AlphaDropout") return true;
        if ((layer.type == "Conv2d" || layer.type == "ConvTranspose2d") && layer.activation != ActivationType::NONE) return true;
        return false;
    };

    auto needs_output_snapshot = [](const Layer& layer) -> bool {
        // Reparameterize backward a besoin de z (output) pour reconstruire eps.
        if (layer.type == "Reparameterize") return true;
        return false;
    };
    
    // ========================================================================
    // FORWARD PASS: Routing via TensorStore
    // ========================================================================

    MemoryGuard& guard = MemoryGuard::instance();
    const size_t guard_mb = guard.getLimit() / (1024ULL * 1024ULL);
    const size_t cap_mb = (max_ram_mb_ > 0) ? max_ram_mb_ : guard_mb;
    RuntimeAllocator allocator(guard, cap_mb);

    static const std::vector<std::string> kDefaultInputNameX = {"x"};

    // VIZ: dernier HxW "connu" pendant ce forward, utile pour les layers
    // qui ne renseignent pas output_width/output_height (ex: activations).
    int viz_last_w = 0;
    int viz_last_h = 0;
    
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];
        
        // ====================================================================
        // RETRIEVE INPUTS (multi-input support)
        // ====================================================================
        
        const std::vector<std::string>& input_names = layer.inputs.empty() ? kDefaultInputNameX : layer.inputs;

        auto& inputs = scratch_input_ptrs_;
        inputs.clear();
        inputs.reserve(input_names.size());

        // Pour Embedding: accepter un input fourni en int (tensor_store_int) et le projeter en float ids.
        // IMPORTANT: on évite d'appeler getTensor() en premier (qui loggue une erreur) car
        // le chemin normal pour Embedding peut être via tensor_store_int.
        auto& embedding_ids_tmp = scratch_embedding_ids_tmp_;
        auto& embedding_ids_fallback = scratch_embedding_ids_fallback_;

        for (size_t in_i = 0; in_i < input_names.size(); ++in_i) {
            const auto& name = input_names[in_i];

            // 1) Chemin standard: float TensorStore
            auto itf = tensor_store.find(name);
            if (itf != tensor_store.end()) {
                inputs.push_back(&itf->second);
                continue;
            }

            // 2) Cas spécial Embedding: ids int -> float
            if (layer.type_enum == LayerType::Embedding && in_i == 0) {
                auto iti = tensor_store_int.find(name);
                if (iti == tensor_store_int.end()) {
                    // Fallback: si text_ids n'est pas fourni (ex: smoke tests / entraînement sans prompt),
                    // on génère des ids de padding de longueur seq_len pour stabiliser les shapes.
                    const int L = (layer.seq_len > 0) ? layer.seq_len : 1;
                    const int pad = (layer.padding_idx >= 0) ? layer.padding_idx : 0;
                    embedding_ids_fallback.assign(static_cast<size_t>(L), pad);
                    iti = tensor_store_int.emplace(name, embedding_ids_fallback).first;
                }

                const std::vector<int>& ids = iti->second;
                embedding_ids_tmp.clear();
                embedding_ids_tmp.reserve(ids.size());
                for (int v : ids) embedding_ids_tmp.push_back(static_cast<float>(v));
                inputs.push_back(&embedding_ids_tmp);
                continue;
            }

            // 3) Erreur: input manquant
            std::cerr << "❌ ERROR in layer " << layer_idx << " (" << layer.name
                      << "): Cannot find input tensor '" << name << "'" << std::endl;
            std::cerr << "Available tensors: ";
            auto available = getAvailableTensors();
            for (const auto& t : available) std::cerr << "'" << t << "' ";
            std::cerr << std::endl;
            std::cerr << "Available int tensors: ";
            auto available_i = getAvailableIntTensors();
            for (const auto& t : available_i) std::cerr << "'" << t << "' ";
            std::cerr << std::endl;
            throw std::runtime_error("Missing input tensor: " + name);
        }
        
        // Snapshot inputs for backward (multi-input)
        if (training) {
            forward_state.layer_input_names.push_back(input_names);

            std::vector<size_t> sizes;
            sizes.reserve(inputs.size());
            for (const auto* inp : inputs) sizes.push_back(inp ? inp->size() : 0ULL);
            forward_state.layer_input_sizes_multi.push_back(std::move(sizes));

            std::vector<std::vector<float>> snap;
            if (needs_input_value_snapshot(layer)) {
                snap.reserve(inputs.size());
                for (const auto* inp : inputs) {
                    snap.push_back(*inp);
                }
            }
            forward_state.layer_inputs_multi.push_back(std::move(snap));

            // placeholders alignés par layer
            forward_state.layer_outputs.emplace_back();
            forward_state.layer_output_masks.emplace_back();
        }

        // Pour compatibilité: x est toujours inputs[0]
        const std::vector<float>& x = *inputs[0];
        
        std::vector<float> layer_output;
        
        // ====================================================================
        // DISPATCH PRINCIPAL VIA SWITCH/CASE SUR LayerType (MODE STRICT)
        // ====================================================================
        
        try {
            switch (layer.type_enum) {
    
    // ====================================================================
    // CONVOLUTION
    // ====================================================================
    
    case LayerType::Conv2d:
    case LayerType::ConvTranspose2d: {
        static bool accel_logged_conv2d = false;
        RUNTIME_CHECK(
            layer.in_channels > 0 && layer.out_channels > 0,
            "Conv2d: in_channels and out_channels must be set"
        );
        RUNTIME_CHECK(
            layer.get_kernel_h() > 0,
            "Conv2d: kernel_size must be set"
        );
        
        const int kernel_size = layer.get_kernel_h();
        const int in_channels = layer.in_channels;
        const int out_channels = layer.out_channels;
        int height = layer.input_height > 0 ? layer.input_height : 64;
        int width = layer.input_width > 0 ? layer.input_width : 64;
        const int stride = layer.get_stride_h();
        const int padding = layer.get_pad_h();

        // Robustesse: si H/W configurés ne matchent pas la taille réelle, on tente d'inférer.
        if (in_channels > 0) {
            const size_t ic = static_cast<size_t>(in_channels);
            if (ic > 0 && (x.size() % ic) == 0) {
                const size_t hw = x.size() / ic;
                const size_t cfg_hw = static_cast<size_t>(std::max(1, height)) * static_cast<size_t>(std::max(1, width));
                if (cfg_hw != hw) {
                    const int cfg_h = layer.input_height;
                    const int cfg_w = layer.input_width;
                    bool fixed = false;
                    if (cfg_h > 0 && (hw % static_cast<size_t>(cfg_h)) == 0) {
                        height = cfg_h;
                        width = static_cast<int>(hw / static_cast<size_t>(cfg_h));
                        fixed = true;
                    } else if (cfg_w > 0 && (hw % static_cast<size_t>(cfg_w)) == 0) {
                        width = cfg_w;
                        height = static_cast<int>(hw / static_cast<size_t>(cfg_w));
                        fixed = true;
                    }
                    if (!fixed) {
                        const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                        if (s > 0 && s * s == hw) {
                            height = static_cast<int>(s);
                            width = static_cast<int>(s);
                        }
                    }
                }
            }
        }
        
        int out_height, out_width;
        if (layer.type_enum == LayerType::Conv2d) {
            out_height = (height + 2 * padding - kernel_size) / stride + 1;
            out_width = (width + 2 * padding - kernel_size) / stride + 1;
        } else {
            out_height = (height - 1) * stride - 2 * padding + kernel_size;
            out_width = (width - 1) * stride - 2 * padding + kernel_size;
        }
        
        // ✅ Allocation gérée
        auto output_handle = allocator.allocate_tensor(
            {out_channels, out_height, out_width},
            "float32",
            layer.name + "_output"
        );
        layer_output = output_handle.data();
        
        const float* layer_weights = layer.getWeights();

        // Fast path: im2col + GEMM (tuilé) via HardwareOpt::matmul_fma_saturated
        // NOTE: Pour ConvTranspose2d, on garde le chemin naïf (à optimiser ensuite).
        const bool can_fast = (layer.type_enum == LayerType::Conv2d) && global_use_hardware && hasAVX2() && hasFMA();
        const int out_spatial = out_height * out_width;
        const int K = in_channels * kernel_size * kernel_size;

        if (!accel_logged_conv2d && env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
            accel_logged_conv2d = true;
            #if defined(__AVX2__)
                std::cerr << "Conv2d: compiled_with_avx2=1";
            #else
                std::cerr << "Conv2d: compiled_with_avx2=0";
            #endif

            #if defined(__FMA__)
                std::cerr << " compiled_with_fma=1";
            #else
                std::cerr << " compiled_with_fma=0";
            #endif

            std::cerr << " runtime_avx2=" << (hasAVX2() ? 1 : 0)
                      << " runtime_fma=" << (hasFMA() ? 1 : 0)
                      << " global_use_hardware=" << (global_use_hardware ? 1 : 0)
                      << " will_use_fast_path=" << (can_fast ? 1 : 0)
                      << std::endl;
        }

        if (can_fast && out_spatial > 0 && K > 0 && out_channels > 0) {
            // Transpose des poids: [out_c x K] -> [K x out_c]
            const size_t w_need = static_cast<size_t>(out_channels) * static_cast<size_t>(K);
            if (layer.getWeights() == nullptr || layer.getWeightsSize() < w_need) {
                throw std::runtime_error("Conv2d: weights invalid");
            }

            // Choix d'une tuile M pour limiter la mémoire (X_col et C_tile).
            // Cible ~32MB pour X_col.
            const size_t target_bytes = 32ULL * 1024ULL * 1024ULL;
            const size_t floats_budget = target_bytes / sizeof(float);
            int tile_m = static_cast<int>(std::max<size_t>(256, std::min<size_t>(8192, floats_budget / static_cast<size_t>(K))));
            if (tile_m > out_spatial) tile_m = out_spatial;

            auto wT_buf = allocator.get_scratchpad(w_need * sizeof(float), layer.name + "/conv_wT");
            float* wT = wT_buf.data();
            for (int oc = 0; oc < out_channels; ++oc) {
                const float* w_oc = layer_weights + static_cast<size_t>(oc) * static_cast<size_t>(K);
                for (int k = 0; k < K; ++k) {
                    wT[static_cast<size_t>(k) * static_cast<size_t>(out_channels) + static_cast<size_t>(oc)] = w_oc[static_cast<size_t>(k)];
                }
            }

            const size_t xcol_max = static_cast<size_t>(tile_m) * static_cast<size_t>(K);
            const size_t c_max = static_cast<size_t>(tile_m) * static_cast<size_t>(out_channels);
            auto xcol_buf = allocator.get_scratchpad(xcol_max * sizeof(float), layer.name + "/im2col");
            auto c_buf = allocator.get_scratchpad(c_max * sizeof(float), layer.name + "/conv_gemm_out");
            float* Xcol = xcol_buf.data();
            float* Ctmp = c_buf.data();

            for (int m0 = 0; m0 < out_spatial; m0 += tile_m) {
                const int m1 = std::min(out_spatial, m0 + tile_m);
                const int tm = m1 - m0;

                // im2col: Xcol[tm x K]
                for (int r = 0; r < tm; ++r) {
                    const int m = m0 + r;
                    const int oh = m / out_width;
                    const int ow = m - oh * out_width;
                    float* row = Xcol + static_cast<size_t>(r) * static_cast<size_t>(K);
                    int col = 0;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int in_base_c = ic * (height * width);
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            const int ih = oh * stride + kh - padding;
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                const int iw = ow * stride + kw - padding;
                                float v = 0.0f;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    const int in_idx = in_base_c + ih * width + iw;
                                    if (in_idx >= 0 && static_cast<size_t>(in_idx) < x.size()) {
                                        v = x[static_cast<size_t>(in_idx)];
                                    }
                                }
                                row[col++] = v;
                            }
                        }
                    }
                }

                // GEMM: Ctmp[tm x out_c] = Xcol[tm x K] @ wT[K x out_c]
                HardwareOpt::matmul_fma_saturated(Ctmp, Xcol, wT, static_cast<size_t>(tm), static_cast<size_t>(out_channels), static_cast<size_t>(K));

                // Scatter vers layout output [out_c, out_h, out_w]
                for (int r = 0; r < tm; ++r) {
                    const int m = m0 + r;
                    const size_t base_c = static_cast<size_t>(r) * static_cast<size_t>(out_channels);
                    for (int oc = 0; oc < out_channels; ++oc) {
                        layer_output[static_cast<size_t>(oc) * static_cast<size_t>(out_spatial) + static_cast<size_t>(m)] = Ctmp[base_c + static_cast<size_t>(oc)];
                    }
                }
            }

            allocator.return_scratchpad(std::move(c_buf));
            allocator.return_scratchpad(std::move(xcol_buf));
            allocator.return_scratchpad(std::move(wT_buf));
        } else {
            // Fallback: chemin naïf
            #pragma omp parallel for schedule(static) collapse(2) if(static_cast<long long>(out_channels) * out_height * out_width * in_channels * kernel_size * kernel_size > 262144)
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        float sum = 0.0f;
                        
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        int in_idx = ic * (height * width) + ih * width + iw;
                                        int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                        
                                        if (in_idx < static_cast<int>(x.size()) && 
                                            w_idx < static_cast<int>(layer.getWeightsSize())) {
                                            sum += x[in_idx] * layer_weights[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        
                        int out_idx = oc * (out_height * out_width) + oh * out_width + ow;
                        layer_output[out_idx] = sum;
                    }
                }
            }
        }
        
        // ReLU activation if specified
        if (layer.activation != ActivationType::NONE) {
            for (auto &val : layer_output) {
                val = std::max(0.0f, val);
            }
        }
        
        break;
    }
    
    case LayerType::Conv1d: {
        layer_output = LayerOpsExt::conv1d_forward(x, layer);
        break;
    }
    
    case LayerType::DepthwiseConv2d: {
        layer_output = LayerOpsExt::depthwise_conv2d_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // LINEAR
    // ====================================================================
    
    case LayerType::Linear: {
        bool did_vulkan = false;
        bool did_opencl = false;
#ifdef ENABLE_VULKAN
        // Politique: par défaut off. Activer avec MIMIR_VULKAN_LINEAR=1.
        const bool vulkan_linear = env_flag_true("MIMIR_VULKAN_LINEAR", false);
        const int vk_min_ops = env_int("MIMIR_VULKAN_LINEAR_MIN_OPS", 1 << 20);
        if (!training && vulkan_linear && g_compute_available && g_compute_engine) {
            const int in_f = layer.in_features > 0 ? layer.in_features : static_cast<int>(x.size());
            const int out_f = layer.out_features;
            int batch = 1;
            if (layer.seq_len > 0 && static_cast<int>(x.size()) == layer.seq_len * in_f) {
                batch = layer.seq_len;
            }

            // Garde-fous: éviter tout OOB côté GPU (peut crasher le driver => SIGSEGV).
            const size_t expected_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
            const size_t expected_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
            const size_t expected_b = static_cast<size_t>(out_f);
            const size_t expected_total_w = expected_w + ((layer.use_bias) ? expected_b : 0u);

            const long long ops = static_cast<long long>(batch) * static_cast<long long>(in_f) * static_cast<long long>(out_f);
            if (out_f > 0 && in_f > 0 && ops >= vk_min_ops && x.size() == expected_in) {
                const float* weights = layer.getWeights();
                const float* bias = nullptr;
                const size_t weights_n = static_cast<size_t>(layer.getWeightsSize());
                if (weights && weights_n >= expected_total_w) {
                    if (layer.use_bias) {
                        bias = weights + expected_w;
                    }
                    layer_output.assign(static_cast<size_t>(batch) * static_cast<size_t>(out_f), 0.0f);
                    did_vulkan = g_compute_engine->linearForward(
                        x.data(),
                        weights,
                        bias,
                        layer_output.data(),
                        batch,
                        in_f,
                        out_f
                    );
                    if (did_vulkan && env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
                        std::cout << "[accel] Linear via Vulkan (batch=" << batch << ", in=" << in_f << ", out=" << out_f << ")\n";
                    }
                } else if (env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
                    std::cout << "[accel] Vulkan Linear skipped (shape/weights mismatch)."
                              << " x=" << x.size() << " expected=" << expected_in
                              << " weights=" << weights_n << " expected>=" << expected_total_w << "\n";
                }
            }
        }
#endif
#ifdef ENABLE_OPENCL
        // Politique: par défaut off (sécurité). Activer avec MIMIR_OPENCL_LINEAR=1.
        // Permet d'utiliser OpenCL et Vulkan simultanément (backends indépendants).
        const bool opencl_linear = env_flag_true("MIMIR_OPENCL_LINEAR", false);
        const int min_ops = env_int("MIMIR_OPENCL_LINEAR_MIN_OPS", 1 << 20);
        if (!did_vulkan && !training && opencl_linear && g_opencl_available && g_opencl_engine) {
            const int in_f = layer.in_features > 0 ? layer.in_features : static_cast<int>(x.size());
            const int out_f = layer.out_features;
            int batch = 1;
            if (layer.seq_len > 0 && static_cast<int>(x.size()) == layer.seq_len * in_f) {
                batch = layer.seq_len;
            }

            const size_t expected_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
            const size_t expected_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
            const size_t expected_b = static_cast<size_t>(out_f);
            const size_t expected_total_w = expected_w + ((layer.use_bias) ? expected_b : 0u);

            const long long ops = static_cast<long long>(batch) * static_cast<long long>(in_f) * static_cast<long long>(out_f);
            if (out_f > 0 && in_f > 0 && ops >= min_ops && x.size() == expected_in) {
                const float* weights = layer.getWeights();
                const float* bias = nullptr;
                const size_t weights_n = static_cast<size_t>(layer.getWeightsSize());
                if (weights && weights_n >= expected_total_w) {
                    if (layer.use_bias) {
                        bias = weights + expected_w;
                    }
                    layer_output.assign(static_cast<size_t>(batch) * static_cast<size_t>(out_f), 0.0f);
                    did_opencl = g_opencl_engine->linearForward(
                        x.data(),
                        weights,
                        bias,
                        layer_output.data(),
                        batch,
                        in_f,
                        out_f
                    );
                }
            }
        }
#endif
        if (!did_vulkan && !did_opencl) {
            static bool accel_logged_linear_cpu = false;
            if (!accel_logged_linear_cpu && env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
                accel_logged_linear_cpu = true;
                #if defined(__AVX2__)
                    std::cerr << "Linear(CPU): compiled_with_avx2=1";
                #else
                    std::cerr << "Linear(CPU): compiled_with_avx2=0";
                #endif

                #if defined(__FMA__)
                    std::cerr << " compiled_with_fma=1";
                #else
                    std::cerr << " compiled_with_fma=0";
                #endif

                std::cerr << " runtime_avx2=" << (hasAVX2() ? 1 : 0)
                          << " runtime_fma=" << (hasFMA() ? 1 : 0)
                          << " global_use_hardware=" << (global_use_hardware ? 1 : 0)
                          << " vulkan_linear=" << (env_flag_true("MIMIR_VULKAN_LINEAR", false) ? 1 : 0)
                          << " opencl_linear=" << (env_flag_true("MIMIR_OPENCL_LINEAR", false) ? 1 : 0)
                          << std::endl;
            }
            layer_output = LayerOps::linear_forward(x, layer, training);
        }
        break;
    }

    case LayerType::PatchEmbed: {
        // Input layout: [text_tokens((seq_text+1)*d_model), patch_raw(num_patches*patch_dim)]
        const int d_model = layer.embed_dim > 0 ? layer.embed_dim : layer.out_features;
        const int seq_text = std::max(1, layer.seq_text);
        const int num_patches = std::max(1, layer.num_patches);
        const int patch_dim = std::max(1, layer.patch_dim);

        const int text_dim = (seq_text + 1) * d_model;  // +1 pour le time token
        const int in_dim = text_dim + num_patches * patch_dim;
        const int out_dim = (seq_text + 1 + num_patches) * d_model;

        RUNTIME_CHECK(
            static_cast<int>(x.size()) == in_dim,
            "PatchEmbed: input size mismatch"
        );
        RUNTIME_CHECK(
            layer.getWeights() != nullptr && static_cast<int>(layer.getWeightsSize()) == (patch_dim * d_model + d_model),
            "PatchEmbed: weights not initialized or invalid size"
        );

        layer_output.assign(static_cast<size_t>(out_dim), 0.0f);

        // Copier texte + time token sans modification
        std::copy(x.begin(), x.begin() + text_dim, layer_output.begin());

        const float* w = layer.getWeights();
        const float* b = w + patch_dim * d_model;
        const float inv = 1.0f / std::sqrt(static_cast<float>(patch_dim));

        // Projeter chaque patch: y = (x_patch * inv) @ W + b
        for (int p = 0; p < num_patches; ++p) {
            const int in_off = text_dim + p * patch_dim;
            const int out_off = (seq_text + 1 + p) * d_model;
            for (int d = 0; d < d_model; ++d) {
                float sum = b[d];
                for (int k = 0; k < patch_dim; ++k) {
                    sum += (x[static_cast<size_t>(in_off + k)] * inv) * w[static_cast<size_t>(k * d_model + d)];
                }
                layer_output[static_cast<size_t>(out_off + d)] = sum;
            }
        }
        break;
    }
    
    case LayerType::Bilinear: {
        RUNTIME_CHECK(inputs.size() >= 2, "Bilinear requires 2 inputs");
        const std::vector<float>& x1 = *inputs[0];
        const std::vector<float>& x2 = *inputs[1];

        // Convention:
        // - in_features  = in1
        // - out_features = in2
        // - embed_dim    = out_features (output dim)
        const int in1 = std::max(1, layer.in_features);
        const int in2 = std::max(1, layer.out_features);
        const int out_f = (layer.embed_dim > 0) ? layer.embed_dim : (layer.target_shape.empty() ? 0 : layer.target_shape[0]);
        RUNTIME_CHECK(out_f > 0, "Bilinear: set embed_dim (output dim) or target_shape[0]");

        RUNTIME_CHECK((static_cast<int>(x1.size()) % in1) == 0, "Bilinear: input0 size not divisible by in_features");
        RUNTIME_CHECK((static_cast<int>(x2.size()) % in2) == 0, "Bilinear: input1 size not divisible by out_features");
        const int B1 = static_cast<int>(x1.size()) / in1;
        const int B2 = static_cast<int>(x2.size()) / in2;
        RUNTIME_CHECK(B1 == B2, "Bilinear: batch mismatch between inputs");
        const int B = B1;

        const bool use_bias = layer.use_bias;
        const size_t Wsz = static_cast<size_t>(out_f) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
        const size_t need = Wsz + (use_bias ? static_cast<size_t>(out_f) : 0ULL);
        RUNTIME_CHECK(layer.getWeights() != nullptr && layer.getWeightsSize() >= need, "Bilinear: weights not initialized/invalid size");

        const float* W = layer.getWeights();
        const float* bias = use_bias ? (W + Wsz) : nullptr;

        layer_output.assign(static_cast<size_t>(B) * static_cast<size_t>(out_f), 0.0f);
        for (int b = 0; b < B; ++b) {
            const float* arow = &x1[static_cast<size_t>(b) * static_cast<size_t>(in1)];
            const float* brow = &x2[static_cast<size_t>(b) * static_cast<size_t>(in2)];
            float* yrow = &layer_output[static_cast<size_t>(b) * static_cast<size_t>(out_f)];
            for (int o = 0; o < out_f; ++o) {
                float sum = bias ? bias[static_cast<size_t>(o)] : 0.0f;
                const size_t base_o = static_cast<size_t>(o) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                for (int i = 0; i < in1; ++i) {
                    const float ai = arow[static_cast<size_t>(i)];
                    const size_t base_i = base_o + static_cast<size_t>(i) * static_cast<size_t>(in2);
                    for (int j = 0; j < in2; ++j) {
                        sum += ai * W[base_i + static_cast<size_t>(j)] * brow[static_cast<size_t>(j)];
                    }
                }
                yrow[static_cast<size_t>(o)] = sum;
            }
        }
        break;
    }
    
    // ====================================================================
    // EMBEDDING
    // ====================================================================
    
    case LayerType::Embedding: {
        // Mode float: interpréter x comme des ids (arrondis).
        // Recommandé: utiliser forwardPass(std::vector<int>) pour les modèles NLP.
        const int vocab = std::max(1, layer.vocab_size);
        const int dim = std::max(1, layer.embed_dim);
        const int pad = layer.padding_idx;

        RUNTIME_CHECK(layer.getWeights() != nullptr, "Embedding: weights not initialized");
        RUNTIME_CHECK(static_cast<int>(layer.getWeightsSize()) >= vocab * dim, "Embedding: invalid weight size");

        const float* w = layer.getWeights();
        const size_t outN = x.size() * static_cast<size_t>(dim);
        auto output_handle = allocator.allocate_tensor(
            {static_cast<int>(outN)},
            "float32",
            layer.name + "_output"
        );
        std::vector<float>& out = output_handle.data();
        out.assign(outN, 0.0f);

        for (size_t t = 0; t < x.size(); ++t) {
            const int id = static_cast<int>(std::llround(static_cast<double>(x[t])));
            if (pad >= 0 && id == pad) continue;
            if (id < 0 || id >= vocab) continue;
            const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
            const size_t base_o = t * static_cast<size_t>(dim);
            for (int d = 0; d < dim; ++d) {
                out[base_o + static_cast<size_t>(d)] = w[base_w + static_cast<size_t>(d)];
            }
        }

        layer_output = std::move(out);
        break;
    }
    
    case LayerType::EmbeddingBag: {
        // Convention:
        // - inputs[0] = ids (float -> arrondi)
        // - inputs[1] optionnel = offsets (float -> arrondi), taille = num_bags+1
        // Pooling = SUM
        RUNTIME_CHECK(inputs.size() >= 1, "EmbeddingBag requires at least 1 input (ids)");
        const std::vector<float>& ids_f = *inputs[0];
        const std::vector<float>* offsets_f = (inputs.size() >= 2) ? inputs[1] : nullptr;

        const int vocab = std::max(1, layer.vocab_size);
        const int dim = std::max(1, layer.embed_dim);
        const int pad = layer.padding_idx;

        RUNTIME_CHECK(layer.getWeights() != nullptr, "EmbeddingBag: weights not initialized");
        RUNTIME_CHECK(static_cast<int>(layer.getWeightsSize()) >= vocab * dim, "EmbeddingBag: invalid weight size");
        const float* w = layer.getWeights();

        std::vector<int> offsets;
        int num_bags = 1;
        if (offsets_f && !offsets_f->empty()) {
            offsets.reserve(offsets_f->size());
            for (float v : *offsets_f) offsets.push_back(static_cast<int>(std::llround(static_cast<double>(v))));
            RUNTIME_CHECK(offsets.size() >= 2, "EmbeddingBag: offsets must have at least 2 entries");
            num_bags = static_cast<int>(offsets.size()) - 1;
        } else {
            offsets = {0, static_cast<int>(ids_f.size())};
            num_bags = 1;
        }

        layer_output.assign(static_cast<size_t>(num_bags) * static_cast<size_t>(dim), 0.0f);
        for (int b = 0; b < num_bags; ++b) {
            const int start = std::clamp(offsets[static_cast<size_t>(b)], 0, static_cast<int>(ids_f.size()));
            const int end = std::clamp(offsets[static_cast<size_t>(b + 1)], start, static_cast<int>(ids_f.size()));
            float* out = &layer_output[static_cast<size_t>(b) * static_cast<size_t>(dim)];
            for (int t = start; t < end; ++t) {
                const int id = static_cast<int>(std::llround(static_cast<double>(ids_f[static_cast<size_t>(t)])));
                if (pad >= 0 && id == pad) continue;
                if (id < 0 || id >= vocab) continue;
                const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                for (int d = 0; d < dim; ++d) {
                    out[static_cast<size_t>(d)] += w[base_w + static_cast<size_t>(d)];
                }
            }
        }
        break;
    }
    
    // ====================================================================
    // NORMALIZATION
    // ====================================================================
    
    case LayerType::BatchNorm2d:
    case LayerType::BatchNorm1d: {
        auto output_handle = allocator.allocate_tensor(
            {static_cast<int>(x.size())},
            "float32",
            layer.name + "_output"
        );
        std::vector<float>& out = output_handle.data();
        out = x;
        
        float mean = 0.0f;
        for (float val : x) mean += val;
        mean /= x.size();
        
        float var = 0.0f;
        for (float val : x) {
            float diff = val - mean;
            var += diff * diff;
        }
        var /= x.size();
        float std = std::sqrt(var + layer.eps);
        
        const float* layer_weights = layer.getWeights();
        
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = (out[i] - mean) / std;
            if (layer.affine && i < layer.getWeightsSize()) {
                out[i] *= layer_weights[i];
            }
        }

        layer_output = out;
        
        break;
    }
    
    case LayerType::LayerNorm: {
        layer_output = LayerOps::layernorm_forward(x, layer, training);
        break;
    }
    
    case LayerType::GroupNorm: {
        layer_output = LayerOps::groupnorm_forward(x, layer, training);
        break;
    }
    
    case LayerType::InstanceNorm2d: {
        layer_output = LayerOpsExt::instance_norm2d_forward(x, layer);
        break;
    }
    
    case LayerType::RMSNorm: {
        layer_output = LayerOpsExt::rms_norm_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // ACTIVATION
    // ====================================================================
    
    case LayerType::ReLU: {
        layer_output = LayerOps::relu_forward(x);
        break;
    }
    
    case LayerType::LeakyReLU: {
        float alpha = layer.leaky_relu_alpha > 0 ? layer.leaky_relu_alpha : 0.01f;
        layer_output = LayerOpsExt::leaky_relu_forward(x, alpha);
        break;
    }
    
    case LayerType::GELU: {
        layer_output = LayerOps::gelu_forward(x);
        break;
    }
    
    case LayerType::SiLU: {
        layer_output = LayerOps::silu_forward(x);
        break;
    }
    
    case LayerType::Tanh: {
        layer_output = LayerOps::tanh_forward(x);
        break;
    }
    
    case LayerType::Sigmoid: {
        layer_output = LayerOps::sigmoid_forward(x);
        break;
    }
    
    case LayerType::Softmax:
    case LayerType::LogSoftmax: {
        layer_output = LayerOps::softmax_forward(x, layer);
        break;
    }
    
    case LayerType::Softplus: {
        layer_output = LayerOpsExt::softplus_forward(x);
        break;
    }
    
    case LayerType::Mish: {
        layer_output = LayerOpsExt::mish_forward(x);
        break;
    }
    
    case LayerType::HardSigmoid: {
        layer_output = LayerOpsExt::hard_sigmoid_forward(x);
        break;
    }
    
    case LayerType::HardSwish: {
        layer_output = LayerOpsExt::hard_swish_forward(x);
        break;
    }
    
    // ====================================================================
    // POOLING
    // ====================================================================
    
    case LayerType::MaxPool2d: {
        const int kernel_size = layer.get_kernel_h();
        const int in_channels = layer.in_channels > 0 ? layer.in_channels : 64;
        const int height = layer.input_height > 0 ? layer.input_height : 64;
        const int width = layer.input_width > 0 ? layer.input_width : 64;
        const int stride = layer.get_stride_h();
        const int padding = layer.get_pad_h();

        RUNTIME_CHECK(kernel_size > 0, "MaxPool2d: kernel_size must be > 0");
        RUNTIME_CHECK(stride > 0, "MaxPool2d: stride must be > 0");

        const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
        const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
        RUNTIME_CHECK(out_height > 0 && out_width > 0, "MaxPool2d: output dimensions must be > 0");

        auto output_handle = allocator.allocate_tensor(
            {in_channels, out_height, out_width},
            "float32",
            layer.name + "_output"
        );
        std::vector<float>& out = output_handle.data();
        std::fill(out.begin(), out.end(), -std::numeric_limits<float>::infinity());

        for (int c = 0; c < in_channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int in_idx = c * (height * width) + ih * width + iw;
                                if (in_idx < static_cast<int>(x.size())) {
                                    max_val = std::max(max_val, x[in_idx]);
                                }
                            }
                        }
                    }
                    
                    int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                    out[out_idx] = max_val;
                }
            }
        }

        // Copy into the returned buffer for consistency with other ops.
        layer_output = out;
        break;
    }
    
    case LayerType::AvgPool2d: {
        layer_output = LayerOps::avgpool2d_forward(x, layer);
        break;
    }
    
    case LayerType::AvgPool1d: {
        layer_output = LayerOpsExt::avgpool1d_forward(x, layer);
        break;
    }

    case LayerType::TokenMeanPool: {
        const int seq_len = layer.seq_len > 0 ? layer.seq_len : 0;
        const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : 0;
        RUNTIME_CHECK(seq_len > 0 && embed_dim > 0, "TokenMeanPool: seq_len and embed_dim must be set");
        RUNTIME_CHECK(static_cast<int>(x.size()) == seq_len * embed_dim, "TokenMeanPool: input size mismatch");

        layer_output.assign(static_cast<size_t>(embed_dim), 0.0f);
        for (int t = 0; t < seq_len; ++t) {
            const int base = t * embed_dim;
            for (int d = 0; d < embed_dim; ++d) {
                layer_output[static_cast<size_t>(d)] += x[static_cast<size_t>(base + d)];
            }
        }
        const float inv = 1.0f / static_cast<float>(seq_len);
        for (int d = 0; d < embed_dim; ++d) {
            layer_output[static_cast<size_t>(d)] *= inv;
        }
        break;
    }
    
    case LayerType::GlobalAvgPool2d:
    case LayerType::AdaptiveAvgPool2d: {
        layer_output = LayerOps::global_avgpool2d_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // DROPOUT
    // ====================================================================
    
    case LayerType::Dropout:
    case LayerType::Dropout2d: {
        layer_output = LayerOps::dropout_forward(x, layer, training);
        break;
    }
    
    case LayerType::AlphaDropout: {
        if (!training) {
            layer_output = x;
            break;
        }

        const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
        // SELU constants
        const float alpha = 1.6732632423543772848170429916717f;
        const float scale = 1.0507009873554804934193349852946f;
        const float alpha_p = -alpha * scale;

        // Affine correction (PyTorch-style) to preserve mean/var
        const float a = 1.0f / std::sqrt((1.0f - p) * (1.0f + p * alpha_p * alpha_p));
        const float b = -a * alpha_p * p;

        layer_output.resize(x.size());
        auto& gen = MimirRng::generator();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < x.size(); ++i) {
            const bool keep = (dist(gen) > p);
            const float v = keep ? x[i] : alpha_p;
            layer_output[i] = a * v + b;
        }
        break;
    }
    
    // ====================================================================
    // SHAPE OPERATIONS
    // ====================================================================
    
    case LayerType::Flatten: {
        layer_output = LayerOps::flatten_forward(x, layer);
        break;
    }
    
    case LayerType::Reshape:
    case LayerType::View: {
        layer_output = LayerOps::reshape_forward(x, layer);
        break;
    }
    
    case LayerType::Transpose: {
        RUNTIME_CHECK(
            layer.in_features > 0 && layer.out_features > 0,
            "Transpose: in_features and out_features must be set"
        );
        
        layer_output = LayerOps::transpose_forward(
            x, layer.in_features, layer.out_features
        );
        break;
    }
    
    case LayerType::Permute: {
        RUNTIME_CHECK(
            !layer.permute_dims.empty(),
            "Permute: permute_dims must be configured"
        );
        
        std::vector<int> shape = layer.shape;
        if (shape.empty()) {
            shape = {1, static_cast<int>(x.size())};
        }
        
        layer_output = LayerOps::permute_forward(x, layer.permute_dims, shape);
        break;
    }
    
    case LayerType::Squeeze: {
        std::vector<int> input_shape = {static_cast<int>(x.size())};
        std::vector<int> output_shape;
        layer_output = LayerOpsExt::squeeze_forward(
            x, input_shape, output_shape, layer.squeeze_dim
        );
        break;
    }
    
    case LayerType::Unsqueeze: {
        std::vector<int> input_shape = {static_cast<int>(x.size())};
        std::vector<int> output_shape;
        RUNTIME_CHECK(
            layer.unsqueeze_dim >= -10 && layer.unsqueeze_dim < 10,
            "Unsqueeze: unsqueeze_dim must be set (valid range: -10 to 10)"
        );
        layer_output = LayerOpsExt::unsqueeze_forward(
            x, input_shape, output_shape, layer.unsqueeze_dim
        );
        break;
    }
    
    case LayerType::Identity: {
        layer_output = LayerOps::identity_forward(x);
        break;
    }
    
    case LayerType::Lambda: {
        RUNTIME_ERROR_STRICT(
            "Lambda layer with Lua callbacks is unsafe in strict mode. "
            "Use C++ layer implementations instead."
        );
        break;
    }
    
    // ====================================================================
    // ELEMENT-WISE OPERATIONS
    // ====================================================================
    
    case LayerType::Add: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Add layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOps::add_forward(*inputs[0], *inputs[1]);
        break;
    }
    
    case LayerType::Subtract: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Subtract layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOpsExt::subtract_forward(*inputs[0], *inputs[1]);
        break;
    }
    
    case LayerType::Multiply: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Multiply layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOps::multiply_forward(*inputs[0], *inputs[1]);
        break;
    }
    
    case LayerType::Divide: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Divide layer requires 2 inputs, got " + std::to_string(inputs.size())
        );
        layer_output = LayerOpsExt::divide_forward(*inputs[0], *inputs[1]);
        break;
    }

    case LayerType::Reparameterize: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Reparameterize requires 2 inputs (mu, logvar), got " + std::to_string(inputs.size())
        );
        const std::vector<float>& mu = *inputs[0];
        const std::vector<float>& logvar = *inputs[1];
        RUNTIME_CHECK(mu.size() == logvar.size(), "Reparameterize: mu/logvar size mismatch");

        layer_output.resize(mu.size());
        bool stochastic_latent = true;
        if (modelConfig.contains("stochastic_latent")) {
            try { stochastic_latent = modelConfig["stochastic_latent"].get<bool>(); } catch (...) {}
        } else if (modelConfig.contains("vae_stochastic_latent")) {
            try { stochastic_latent = modelConfig["vae_stochastic_latent"].get<bool>(); } catch (...) {}
        }

        if (!training || !stochastic_latent) {
            std::copy(mu.begin(), mu.end(), layer_output.begin());
            break;
        }

        static thread_local std::mt19937 rng(1337);
        std::normal_distribution<float> n01(0.0f, 1.0f);
        for (size_t i = 0; i < mu.size(); ++i) {
            const float lv = std::clamp(logvar[i], -20.0f, 20.0f);
            const float stdv = std::exp(0.5f * lv);
            layer_output[i] = mu[i] + stdv * n01(rng);
        }
        break;
    }
    
    // ====================================================================
    // TENSOR OPERATIONS
    // ====================================================================
    
    case LayerType::Concat: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Concat requires at least 2 inputs, got " + std::to_string(inputs.size())
        );
        
        std::vector<std::vector<float>> inputs_vec;
        inputs_vec.reserve(inputs.size());
        for (const auto* inp : inputs) {
            inputs_vec.push_back(*inp);
        }
        layer_output = LayerOps::concat_forward(inputs_vec, layer.concat_axis);
        break;
    }
    
    case LayerType::Split: {
        RUNTIME_CHECK(
            !layer.split_sizes.empty(),
            "Split: split_sizes must be configured"
        );
        
        auto splits = LayerOps::split_forward(x, layer.split_sizes, layer.split_axis);
        const std::string output_base = layer.output.empty() ? "x" : layer.output;
        for (size_t i = 0; i < splits.size(); ++i) {
            storeTensor(output_base + "_" + std::to_string(i), std::move(splits[i]));
        }
        layer_output = getTensor(output_base + "_0");
        break;
    }
    
    case LayerType::Chunk: {
        RUNTIME_CHECK(
            layer.num_chunks > 0,
            "Chunk: num_chunks must be set"
        );
        
        auto chunks = LayerOpsExt::chunk_forward(x, layer.num_chunks, layer.split_axis);
        
        std::string output_base = layer.output.empty() ? "x" : layer.output;
        for (size_t i = 0; i < chunks.size(); ++i) {
            storeTensor(output_base + "_" + std::to_string(i), std::move(chunks[i]));
        }
        
        layer_output = getTensor(output_base + "_0");
        break;
    }
    
    case LayerType::Stack: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "Stack requires at least 2 inputs, got " + std::to_string(inputs.size())
        );
        
        std::vector<std::vector<float>> inputs_vec;
        for (const auto* inp : inputs) inputs_vec.push_back(*inp);
        
        layer_output = LayerOpsExt::stack_forward(inputs_vec, layer.stack_axis);
        break;
    }
    
    case LayerType::MatMul: {
        RUNTIME_CHECK(
            inputs.size() >= 2,
            "MatMul requires 2 matrix inputs, got " + std::to_string(inputs.size())
        );
        RUNTIME_CHECK(
            layer.in_features > 0 && layer.out_features > 0 && layer.embed_dim > 0,
            "MatMul: dimensions (in_features, out_features, embed_dim) must be configured"
        );
        
        int M = layer.in_features;
        int K = layer.out_features;
        int N = layer.embed_dim;
        
        layer_output = LayerOps::matmul_forward(*inputs[0], *inputs[1], M, K, N);
        break;
    }
    
    case LayerType::BatchMatMul: {
        RUNTIME_CHECK(inputs.size() >= 2, "BatchMatMul requires 2 inputs");
        RUNTIME_CHECK(layer.seq_len > 0 && layer.in_features > 0 && layer.out_features > 0 && layer.embed_dim > 0,
                      "BatchMatMul: configure seq_len (batch), in_features (M), out_features (K), embed_dim (N)");

        const int B = layer.seq_len;
        const int M = layer.in_features;
        const int K = layer.out_features;
        const int N = layer.embed_dim;

        const std::vector<float>& A = *inputs[0];
        const std::vector<float>& Bm = *inputs[1];
        RUNTIME_CHECK(static_cast<int>(A.size()) == B * M * K, "BatchMatMul: A size mismatch");
        RUNTIME_CHECK(static_cast<int>(Bm.size()) == B * K * N, "BatchMatMul: B size mismatch");

        layer_output.assign(static_cast<size_t>(B) * static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
        #pragma omp parallel for schedule(static) if(static_cast<long long>(B) * M * N * K > 262144)
        for (int b = 0; b < B; ++b) {
            const float* Ap = &A[static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(K)];
            const float* Bp = &Bm[static_cast<size_t>(b) * static_cast<size_t>(K) * static_cast<size_t>(N)];
            float* Cp = &layer_output[static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(N)];
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += Ap[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
                               Bp[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)];
                    }
                    Cp[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] = sum;
                }
            }
        }
        break;
    }
    
    // ====================================================================
    // ATTENTION
    // ====================================================================
    
    case LayerType::SelfAttention:
    case LayerType::MultiHeadAttention: {
        static bool accel_logged_attention = false;
        if (!accel_logged_attention && env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
            accel_logged_attention = true;
            #if defined(__AVX2__)
                std::cerr << "Attention: compiled_with_avx2=1";
            #else
                std::cerr << "Attention: compiled_with_avx2=0";
            #endif

            #if defined(__FMA__)
                std::cerr << " compiled_with_fma=1";
            #else
                std::cerr << " compiled_with_fma=0";
            #endif

            std::cerr << " runtime_avx2=" << (hasAVX2() ? 1 : 0)
                      << " runtime_fma=" << (hasFMA() ? 1 : 0)
                      << " global_use_hardware=" << (global_use_hardware ? 1 : 0)
                      << std::endl;
        }

        RUNTIME_CHECK(
            layer.getWeights() != nullptr,
            "Attention: weights not initialized. Call allocateParams() first."
        );
        
        int seq_len = layer.seq_len > 0 ? layer.seq_len : 1;
        int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : x.size();
        int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
        bool causal = layer.causal;
        
        const float* weights = layer.getWeights();
        int qkv_size = embed_dim * embed_dim * 3;
        int out_size = embed_dim * embed_dim;
        
        std::vector<float> qkv_weight(weights, weights + qkv_size);
        std::vector<float> out_weight(weights + qkv_size, weights + qkv_size + out_size);
        
        if (layer.type_enum == LayerType::SelfAttention) {
            layer_output = LayerOps::self_attention_forward(
                x, qkv_weight, out_weight, seq_len, embed_dim, num_heads, causal
            );
        } else {
            layer_output = LayerOps::multihead_attention_forward(
                x, qkv_weight, out_weight, seq_len, embed_dim, num_heads, causal
            );
        }
        break;
    }
    
    case LayerType::CrossAttention: {
        static bool accel_logged_cross_attention = false;
        if (!accel_logged_cross_attention && env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
            accel_logged_cross_attention = true;
            #if defined(__AVX2__)
                std::cerr << "CrossAttention: compiled_with_avx2=1";
            #else
                std::cerr << "CrossAttention: compiled_with_avx2=0";
            #endif

            #if defined(__FMA__)
                std::cerr << " compiled_with_fma=1";
            #else
                std::cerr << " compiled_with_fma=0";
            #endif

            std::cerr << " runtime_avx2=" << (hasAVX2() ? 1 : 0)
                      << " runtime_fma=" << (hasFMA() ? 1 : 0)
                      << " global_use_hardware=" << (global_use_hardware ? 1 : 0)
                      << std::endl;
        }

        RUNTIME_CHECK(
            inputs.size() >= 2,
            "CrossAttention requires 2 inputs (query, key_value), got " +
            std::to_string(inputs.size())
        );

        RUNTIME_CHECK(
            layer.getWeights() != nullptr,
            "CrossAttention: weights not initialized. Call allocateParams() first."
        );

        const std::vector<float>& q_in = *inputs[0];
        const std::vector<float>& kv_in = *inputs[1];

        int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
        bool causal = layer.causal;

        int embed_dim = layer.embed_dim;
        if (embed_dim <= 0 && layer.head_dim > 0 && num_heads > 0) {
            embed_dim = layer.head_dim * num_heads;
        }
        RUNTIME_CHECK(
            embed_dim > 0,
            "CrossAttention: embed_dim must be configured (set layer.embed_dim or head_dim*num_heads)"
        );
        RUNTIME_CHECK(
            (q_in.size() % static_cast<size_t>(embed_dim)) == 0,
            "CrossAttention: query input size must be divisible by embed_dim"
        );
        RUNTIME_CHECK(
            (kv_in.size() % static_cast<size_t>(embed_dim)) == 0,
            "CrossAttention: key/value input size must be divisible by embed_dim"
        );

        const int query_len = static_cast<int>(q_in.size() / static_cast<size_t>(embed_dim));
        const int kv_len = static_cast<int>(kv_in.size() / static_cast<size_t>(embed_dim));
        RUNTIME_CHECK(query_len > 0 && kv_len > 0, "CrossAttention: invalid sequence lengths");

        const float* weights = layer.getWeights();
        const int q_size = embed_dim * embed_dim;
        const int kv_size = embed_dim * (2 * embed_dim);
        const int out_size = embed_dim * embed_dim;

        std::vector<float> q_weight(weights, weights + q_size);
        std::vector<float> kv_weight(weights + q_size, weights + q_size + kv_size);
        std::vector<float> out_weight(weights + q_size + kv_size, weights + q_size + kv_size + out_size);

        layer_output = LayerOps::cross_attention_forward(
            q_in,
            kv_in,
            q_weight,
            kv_weight,
            out_weight,
            query_len,
            kv_len,
            embed_dim,
            num_heads,
            causal
        );
        break;
    }
    
    // ====================================================================
    // UPSAMPLING
    // ====================================================================
    
    case LayerType::UpsampleNearest: {
        RUNTIME_CHECK(
            layer.out_h > 0 && layer.out_w > 0 && layer.in_channels > 0,
            "UpsampleNearest: dimensions (out_h, out_w, in_channels) must be set"
        );
        
        int in_h = layer.out_h;
        int in_w = layer.out_w;
        int channels = layer.in_channels;
        int scale_h = layer.scale_h > 0 ? layer.scale_h : 2;
        int scale_w = layer.scale_w > 0 ? layer.scale_w : 2;
        
        layer_output = LayerOps::upsample_nearest_forward(
            x, in_h, in_w, channels, scale_h, scale_w
        );
        break;
    }
    
    case LayerType::UpsampleBilinear: {
        RUNTIME_CHECK(
            layer.out_h > 0 && layer.out_w > 0 && layer.in_channels > 0,
            "UpsampleBilinear: dimensions must be set"
        );
        
        int in_h = layer.out_h;
        int in_w = layer.out_w;
        int channels = layer.in_channels;
        int out_h = in_h * 2;
        int out_w = in_w * 2;
        
        layer_output = LayerOps::upsample_bilinear_forward(
            x, in_h, in_w, channels, out_h, out_w
        );
        break;
    }
    
    case LayerType::UpsampleBicubic: {
        RUNTIME_CHECK(
            layer.out_h > 0 && layer.out_w > 0,
            "UpsampleBicubic: dimensions must be set"
        );
        
        int in_h = layer.out_h;
        int in_w = layer.out_w;
        int channels = layer.in_channels > 0 ? layer.in_channels : 3;
        int out_h = in_h * 2;
        int out_w = in_w * 2;
        
        layer_output = LayerOpsExt::upsample_bicubic_forward(
            x, in_h, in_w, channels, out_h, out_w
        );
        break;
    }
    
    case LayerType::PixelShuffle: {
        layer_output = LayerOpsExt::pixel_shuffle_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // PADDING
    // ====================================================================
    
    case LayerType::ZeroPad2d: {
        layer_output = LayerOpsExt::zero_pad2d_forward(x, layer);
        break;
    }
    
    case LayerType::ReflectionPad2d: {
        layer_output = LayerOpsExt::reflection_pad2d_forward(x, layer);
        break;
    }
    
    case LayerType::ReplicationPad2d: {
        layer_output = LayerOpsExt::replication_pad2d_forward(x, layer);
        break;
    }
    
    // ====================================================================
    // RECURRENT (Hors scope - À SUPPRIMER de l'enum si non implémenté)
    // ====================================================================
    
    case LayerType::LSTM:
    case LayerType::GRU:
    case LayerType::RNN: {
        RUNTIME_CHECK(layer.seq_len > 0, "Recurrent: seq_len must be set");
        RUNTIME_CHECK(layer.in_features > 0 && layer.out_features > 0, "Recurrent: in_features/out_features must be set");
        RUNTIME_CHECK(layer.getWeights() != nullptr, "Recurrent: weights not initialized");

        const int T = layer.seq_len;
        const int I = layer.in_features;
        const int H = layer.out_features;
        RUNTIME_CHECK(static_cast<int>(x.size()) == T * I, "Recurrent: input size mismatch (expected seq_len*in_features)");

        auto sigmoid_scalar = [](float v) -> float { return 1.0f / (1.0f + std::exp(-v)); };

        if (layer.type_enum == LayerType::RNN) {
            // Params (PyTorch-like): W_ih[H,I], W_hh[H,H], b_ih[H], b_hh[H]
            const bool use_bias = layer.use_bias;
            const size_t Wih_sz = static_cast<size_t>(H) * static_cast<size_t>(I);
            const size_t Whh_sz = static_cast<size_t>(H) * static_cast<size_t>(H);
            const size_t bih_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
            const size_t bhh_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
            const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
            RUNTIME_CHECK(layer.getWeightsSize() >= need, "RNN: invalid weight size");

            const float* Wih = layer.getWeights();
            const float* Whh = Wih + Wih_sz;
            const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
            const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

            std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
            layer_output.assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
            for (int t = 0; t < T; ++t) {
                const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                float* ht = &layer_output[static_cast<size_t>(t) * static_cast<size_t>(H)];
                for (int h = 0; h < H; ++h) {
                    const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                    const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                    float sum = 0.0f;
                    for (int i = 0; i < I; ++i) sum += wih[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                    for (int k = 0; k < H; ++k) sum += whh[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                    if (bih) sum += bih[static_cast<size_t>(h)];
                    if (bhh) sum += bhh[static_cast<size_t>(h)];
                    ht[static_cast<size_t>(h)] = std::tanh(sum);
                }
                std::copy(ht, ht + H, h_prev.begin());
            }
            break;
        }

        if (layer.type_enum == LayerType::GRU) {
            // Params: W_ih[3H,I], W_hh[3H,H], b_ih[3H], b_hh[3H] ; order (r,z,n)
            const bool use_bias = layer.use_bias;
            const size_t Wih_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(I);
            const size_t Whh_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(H);
            const size_t bih_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
            const size_t bhh_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
            const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
            RUNTIME_CHECK(layer.getWeightsSize() >= need, "GRU: invalid weight size");

            const float* Wih = layer.getWeights();
            const float* Whh = Wih + Wih_sz;
            const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
            const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

            std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
            std::vector<float> r(static_cast<size_t>(H), 0.0f);
            std::vector<float> z(static_cast<size_t>(H), 0.0f);
            std::vector<float> n(static_cast<size_t>(H), 0.0f);

            layer_output.assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
            for (int t = 0; t < T; ++t) {
                const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];

                for (int h = 0; h < H; ++h) {
                    // r
                    const float* w_ir = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                    const float* w_hr = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                    float sr = 0.0f;
                    for (int i = 0; i < I; ++i) sr += w_ir[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                    for (int k = 0; k < H; ++k) sr += w_hr[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                    if (bih) sr += bih[static_cast<size_t>(h)];
                    if (bhh) sr += bhh[static_cast<size_t>(h)];
                    r[static_cast<size_t>(h)] = sigmoid_scalar(sr);

                    // z
                    const float* w_iz = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                    const float* w_hz = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                    float sz = 0.0f;
                    for (int i = 0; i < I; ++i) sz += w_iz[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                    for (int k = 0; k < H; ++k) sz += w_hz[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                    if (bih) sz += bih[static_cast<size_t>(H + h)];
                    if (bhh) sz += bhh[static_cast<size_t>(H + h)];
                    z[static_cast<size_t>(h)] = sigmoid_scalar(sz);
                }

                for (int h = 0; h < H; ++h) {
                    // n
                    const float* w_in = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                    const float* w_hn = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                    float sn = 0.0f;
                    for (int i = 0; i < I; ++i) sn += w_in[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                    float hn = 0.0f;
                    for (int k = 0; k < H; ++k) hn += w_hn[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                    if (bih) sn += bih[static_cast<size_t>(2 * H + h)];
                    if (bhh) sn += bhh[static_cast<size_t>(2 * H + h)];
                    n[static_cast<size_t>(h)] = std::tanh(sn + r[static_cast<size_t>(h)] * hn);
                }

                float* ht = &layer_output[static_cast<size_t>(t) * static_cast<size_t>(H)];
                for (int h = 0; h < H; ++h) {
                    ht[static_cast<size_t>(h)] = (1.0f - z[static_cast<size_t>(h)]) * n[static_cast<size_t>(h)] + z[static_cast<size_t>(h)] * h_prev[static_cast<size_t>(h)];
                }
                std::copy(ht, ht + H, h_prev.begin());
            }
            break;
        }

        // LSTM
        {
            // Params: W_ih[4H,I], W_hh[4H,H], b_ih[4H], b_hh[4H] ; order (i,f,g,o)
            const bool use_bias = layer.use_bias;
            const size_t Wih_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(I);
            const size_t Whh_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(H);
            const size_t bih_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
            const size_t bhh_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
            const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
            RUNTIME_CHECK(layer.getWeightsSize() >= need, "LSTM: invalid weight size");

            const float* Wih = layer.getWeights();
            const float* Whh = Wih + Wih_sz;
            const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
            const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

            std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
            std::vector<float> c_prev(static_cast<size_t>(H), 0.0f);
            layer_output.assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);

            for (int t = 0; t < T; ++t) {
                const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                float* ht = &layer_output[static_cast<size_t>(t) * static_cast<size_t>(H)];
                for (int h = 0; h < H; ++h) {
                    auto dot_in = [&](const float* wrow) -> float {
                        float s = 0.0f;
                        for (int i = 0; i < I; ++i) s += wrow[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        return s;
                    };
                    auto dot_h = [&](const float* wrow) -> float {
                        float s = 0.0f;
                        for (int k = 0; k < H; ++k) s += wrow[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                        return s;
                    };

                    const float* wii = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                    const float* wif = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                    const float* wig = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                    const float* wio = Wih + static_cast<size_t>(3 * H + h) * static_cast<size_t>(I);

                    const float* whi = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                    const float* whf = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                    const float* whg = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                    const float* who = Whh + static_cast<size_t>(3 * H + h) * static_cast<size_t>(H);

                    float si = dot_in(wii) + dot_h(whi);
                    float sf = dot_in(wif) + dot_h(whf);
                    float sg = dot_in(wig) + dot_h(whg);
                    float so = dot_in(wio) + dot_h(who);
                    if (bih) {
                        si += bih[static_cast<size_t>(h)];
                        sf += bih[static_cast<size_t>(H + h)];
                        sg += bih[static_cast<size_t>(2 * H + h)];
                        so += bih[static_cast<size_t>(3 * H + h)];
                    }
                    if (bhh) {
                        si += bhh[static_cast<size_t>(h)];
                        sf += bhh[static_cast<size_t>(H + h)];
                        sg += bhh[static_cast<size_t>(2 * H + h)];
                        so += bhh[static_cast<size_t>(3 * H + h)];
                    }

                    const float i_gate = sigmoid_scalar(si);
                    const float f_gate = sigmoid_scalar(sf);
                    const float g_gate = std::tanh(sg);
                    const float o_gate = sigmoid_scalar(so);

                    const float c = f_gate * c_prev[static_cast<size_t>(h)] + i_gate * g_gate;
                    c_prev[static_cast<size_t>(h)] = c;
                    ht[static_cast<size_t>(h)] = o_gate * std::tanh(c);
                }
                std::copy(ht, ht + H, h_prev.begin());
            }
            break;
        }
    }
    
    // ====================================================================
    // UNKNOWN / DEFAULT
    // ====================================================================
    
    case LayerType::UNKNOWN:
    default: {
        throw std::runtime_error(
            "Layer '" + layer.name + "' type '" + 
            type_to_string(layer.type_enum) + "' is UNKNOWN. " +
            "This should never happen in strict mode. Check LayerType registry."
        );
        break;
    }
}
        
        } catch (const std::exception& e) {
            std::cerr << "❌ ERROR in layer " << layer_idx << " (" << layer.name 
                      << ", type: " << type_to_string(layer.type_enum) << "): " 
                      << e.what() << std::endl;
            throw;
        }

        // ====================================================================
        // VIZ TAPS (best-effort): capturer des vignettes intermédiaires par bloc/layer
        // ====================================================================
        // NOTE: Beaucoup d'archis ne renseignent pas output_w/output_h pour tous les layers
        // (ex: activations). On garde un "dernier HxW connu" pour afficher plus de blocs
        // sans modifier les builders.
        if (viz_taps_enabled_ && viz_taps_max_frames_ > 0) {
            auto is_viz_candidate_layer = [&](LayerType t) {
                switch (t) {
                    // Spatial ops (déjà supportés)
                    case LayerType::Conv2d:
                    case LayerType::ConvTranspose2d:
                    case LayerType::DepthwiseConv2d:
                    case LayerType::Reparameterize:
                    case LayerType::MaxPool2d:
                    case LayerType::AvgPool2d:
                    case LayerType::AdaptiveAvgPool2d:
                    case LayerType::GlobalAvgPool2d:
                    case LayerType::UpsampleNearest:
                    case LayerType::UpsampleBilinear:
                    case LayerType::UpsampleBicubic:
                    case LayerType::PixelShuffle:
                    case LayerType::ZeroPad2d:
                    case LayerType::ReflectionPad2d:
                    case LayerType::ReplicationPad2d:
                        return true;

                    // Layers qui conservent souvent la forme spatiale (permet d'afficher + de blocs)
                    case LayerType::BatchNorm2d:
                    case LayerType::InstanceNorm2d:
                    case LayerType::GroupNorm:
                    case LayerType::LayerNorm:
                    case LayerType::RMSNorm:
                    case LayerType::Dropout2d:
                    case LayerType::Dropout:
                    case LayerType::AlphaDropout:
                    case LayerType::Identity:
                    case LayerType::Permute:
                    case LayerType::Transpose:
                    case LayerType::Reshape:
                    case LayerType::View:
                    case LayerType::Squeeze:
                    case LayerType::Unsqueeze:
                    case LayerType::Concat:
                    case LayerType::Add:
                    case LayerType::Subtract:
                    case LayerType::Multiply:
                    case LayerType::Divide:

                    // Attention (affichage best-effort; utile pour diagnostiquer la mémoire K/V)
                    case LayerType::SelfAttention:
                    case LayerType::MultiHeadAttention:
                    case LayerType::CrossAttention:

                    // Activations (souvent omettent output_w/output_h dans les builders)
                    case LayerType::ReLU:
                    case LayerType::LeakyReLU:
                    case LayerType::GELU:
                    case LayerType::SiLU:
                    case LayerType::Tanh:
                    case LayerType::Sigmoid:
                    case LayerType::Softmax:
                    case LayerType::LogSoftmax:
                    case LayerType::Softplus:
                    case LayerType::Mish:
                    case LayerType::HardSigmoid:
                    case LayerType::HardSwish:
                        return true;

                    default:
                        return false;
                }
            };

            auto split_path = [](const std::string& s, char sep) {
                std::vector<std::string> parts;
                size_t start = 0;
                while (start < s.size()) {
                    size_t end = s.find(sep, start);
                    if (end == std::string::npos) end = s.size();
                    if (end > start) parts.push_back(s.substr(start, end - start));
                    start = end + 1;
                }
                return parts;
            };

            auto block_label_for = [&](const Layer& lyr) -> std::string {
                // Convention de nommage (pour le visualizer):
                //   <model>/blocks/<path>/<LayerType>
                // - stable et lisible (hiérarchique)
                // - évite le suffixe "/activation" (sinon masqué si hide_activation_blocks=true)
                if (lyr.name.empty()) return {};

                const auto parts = split_path(lyr.name, '/');
                if (parts.size() < 1) return {};
                const std::string& model = parts[0];

                std::string path;
                // On saute le préfixe modèle, puis on garde le chemin complet du layer.
                for (size_t i = 1; i < parts.size(); ++i) {
                    if (parts[i].empty()) continue;
                    if (!path.empty()) path += "/";
                    path += parts[i];
                }
                if (path.empty()) path = "root";

                std::string type = type_to_string(lyr.type_enum);
                if (type.empty()) type = "Layer";

                // IMPORTANT: ne jamais finir par "/activation".
                if (type == "activation") type = "act";

                return model + "/blocks/" + path + "/" + type;
            };

            if (is_viz_candidate_layer(layer.type_enum)) {
                auto infer_hw = [&](const Layer& lyr, size_t out_size, int& ow, int& oh, int& c) -> bool {
                    auto ok = [&](int w, int h) -> bool {
                        if (w <= 0 || h <= 0) return false;
                        const size_t spatial = static_cast<size_t>(w) * static_cast<size_t>(h);
                        if (spatial == 0) return false;
                        if (out_size < spatial) return false;
                        if ((out_size % spatial) != 0) return false;
                        c = static_cast<int>(out_size / spatial);
                        ow = w;
                        oh = h;
                        return true;
                    };

                    // 1) Métadonnées explicites
                    if (ok(lyr.output_width, lyr.output_height)) return true;
                    if (ok(lyr.out_w, lyr.out_h)) return true;
                    if (ok(lyr.input_width, lyr.input_height)) return true;

                    // 2) Shapes connues (si elles matchent exactement la taille)
                    auto try_shape = [&](const std::vector<int>& s) -> bool {
                        if (s.size() != 3) return false;
                        const int a = s[0];
                        const int b = s[1];
                        const int d = s[2];
                        if (a <= 0 || b <= 0 || d <= 0) return false;
                        const size_t n = static_cast<size_t>(a) * static_cast<size_t>(b) * static_cast<size_t>(d);
                        if (n != out_size) return false;

                        // Deux interprétations courantes: HWC (H=a,W=b,C=d) ou CHW (C=a,H=b,W=d).
                        const int out_c = lyr.out_channels;

                        // Si out_channels est renseigné, on choisit celle qui match.
                        if (out_c > 0) {
                            if (out_c == d) {
                                c = d;
                                ow = b;
                                oh = a;
                                return true;
                            }
                            if (out_c == a) {
                                c = a;
                                ow = d;
                                oh = b;
                                return true;
                            }
                        }

                        // Sinon, essayer de coller aux dims input si dispo.
                        if (lyr.input_height == a && lyr.input_width == b) {
                            c = d;
                            ow = b;
                            oh = a;
                            return true;
                        }
                        if (lyr.input_height == b && lyr.input_width == d) {
                            c = a;
                            ow = d;
                            oh = b;
                            return true;
                        }

                        // Default: HWC (c'est ce que les builders Mimir utilisent le plus souvent).
                        c = d;
                        ow = b;
                        oh = a;
                        return true;
                    };
                    if (try_shape(lyr.shape)) return true;
                    if (try_shape(lyr.target_shape)) return true;

                    // 3) Fallback: dernier HxW connu dans ce thread (souvent valable pour activations/norm)
                    if (ok(viz_last_w, viz_last_h)) return true;

                    return false;
                };

                int ow = 0;
                int oh = 0;
                int c = 0;
                const bool has_hw = infer_hw(layer, layer_output.size(), ow, oh, c);

                if (has_hw && ow > 0 && oh > 0) {
                    const size_t spatial = static_cast<size_t>(ow) * static_cast<size_t>(oh);
                    if (spatial > 0 && layer_output.size() >= spatial) {
                        // Mémoriser pour les layers suivants (activations, etc.).
                        viz_last_w = ow;
                        viz_last_h = oh;

                        const int max_side = std::max(1, viz_taps_max_side_);
                        const int sx = (ow > max_side) ? static_cast<int>((ow + max_side - 1) / max_side) : 1;
                        const int sy = (oh > max_side) ? static_cast<int>((oh + max_side - 1) / max_side) : 1;
                        const int vw = std::max(1, ow / sx);
                        const int vh = std::max(1, oh / sy);

                        VizFrame vf;

                        // Si le tenseur a >= 3 canaux, on affiche un aperçu RGB (3 premiers canaux)
                        // pour mieux visualiser les "modifications" spatiales.
                        if (c >= 3) {
                            std::vector<float> map;
                            map.resize(static_cast<size_t>(vw) * static_cast<size_t>(vh) * 3, 0.0f);

                            for (int y = 0; y < vh; ++y) {
                                const int yy = y * sy;
                                for (int x = 0; x < vw; ++x) {
                                    const int xx = x * sx;
                                    const size_t base = (static_cast<size_t>(yy) * static_cast<size_t>(ow) + static_cast<size_t>(xx));
                                    for (int cc = 0; cc < 3; ++cc) {
                                        const size_t idx = static_cast<size_t>(cc) * spatial + base;
                                        const float v = (idx < layer_output.size()) ? layer_output[idx] : 0.0f;
                                        map[(static_cast<size_t>(y) * static_cast<size_t>(vw) + static_cast<size_t>(x)) * 3ULL + static_cast<size_t>(cc)] = v;
                                    }
                                }
                            }

                            float max_abs = 0.0f;
                            for (float v : map) max_abs = std::max(max_abs, std::fabs(v));
                            const float inv = 1.0f / (max_abs + 1e-6f);

                            std::vector<uint8_t> px;
                            px.resize(map.size());
                            for (size_t i = 0; i < map.size(); ++i) {
                                const float s = map[i] * inv;
                                const float t = 0.5f + 0.5f * std::tanh(s);
                                const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                                px[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
                            }

                            vf.pixels = std::move(px);
                            vf.w = vw;
                            vf.h = vh;
                            vf.channels = 3;
                        } else {
                            // Fallback: heatmap 1 canal (moyenne de quelques canaux)
                            std::vector<float> map;
                            map.resize(static_cast<size_t>(vw) * static_cast<size_t>(vh), 0.0f);

                            const int c_take = std::max(1, std::min(c, 16));
                            for (int y = 0; y < vh; ++y) {
                                const int yy = y * sy;
                                for (int x = 0; x < vw; ++x) {
                                    const int xx = x * sx;
                                    const size_t base = (static_cast<size_t>(yy) * static_cast<size_t>(ow) + static_cast<size_t>(xx));
                                    double acc = 0.0;
                                    for (int cc = 0; cc < c_take; ++cc) {
                                        const size_t idx = static_cast<size_t>(cc) * spatial + base;
                                        if (idx < layer_output.size()) acc += static_cast<double>(layer_output[idx]);
                                    }
                                    map[static_cast<size_t>(y) * static_cast<size_t>(vw) + static_cast<size_t>(x)] = static_cast<float>(acc / static_cast<double>(c_take));
                                }
                            }

                            float max_abs = 0.0f;
                            for (float v : map) max_abs = std::max(max_abs, std::fabs(v));
                            const float inv = 1.0f / (max_abs + 1e-6f);

                            std::vector<uint8_t> px;
                            px.resize(map.size());
                            for (size_t i = 0; i < map.size(); ++i) {
                                const float s = map[i] * inv;
                                const float t = 0.5f + 0.5f * std::tanh(s);
                                const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                                px[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
                            }

                            vf.pixels = std::move(px);
                            vf.w = vw;
                            vf.h = vh;
                            vf.channels = 1;
                        }

                        const std::string block_label = block_label_for(layer);
                        vf.label = !block_label.empty()
                            ? block_label
                            : (layer.name.empty()
                                ? (type_to_string(layer.type_enum) + "#" + std::to_string(layer_idx))
                                : layer.name);

                        // Déduplication: 1 vignette par label (on garde la dernière vue).
                        // Si on est plein, on évince (comme addVizTapFrame) pour laisser apparaître d'autres layers.
                        auto it = std::find_if(viz_taps_.begin(), viz_taps_.end(), [&](const VizFrame& existing) {
                            return existing.label == vf.label;
                        });
                        if (it != viz_taps_.end()) {
                            *it = std::move(vf);
                        } else if (static_cast<int>(viz_taps_.size()) < viz_taps_max_frames_) {
                            viz_taps_.push_back(std::move(vf));
                        } else if (!viz_taps_.empty()) {
                            viz_taps_.back() = std::move(vf);
                        }
                    }
                } else {
                    // Fallback minimal: vecteur 1D -> bande (1 x max_side) pour ne pas rater
                    // complètement les layers dont la forme n'est pas connue.
                    const int max_side = std::max(1, viz_taps_max_side_);
                    const int vw = std::max(1, std::min<int>(max_side, static_cast<int>(layer_output.size())));

                    std::vector<float> map;
                    map.resize(static_cast<size_t>(vw), 0.0f);
                    for (int i = 0; i < vw; ++i) {
                        map[static_cast<size_t>(i)] = layer_output[static_cast<size_t>(i)];
                    }

                    float max_abs = 0.0f;
                    for (float v : map) max_abs = std::max(max_abs, std::fabs(v));
                    const float inv = 1.0f / (max_abs + 1e-6f);

                    VizFrame vf;
                    vf.pixels.resize(static_cast<size_t>(vw));
                    for (int i = 0; i < vw; ++i) {
                        const float s = map[static_cast<size_t>(i)] * inv;
                        const float t = 0.5f + 0.5f * std::tanh(s);
                        const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                        vf.pixels[static_cast<size_t>(i)] = static_cast<uint8_t>(std::clamp(p, 0, 255));
                    }
                    vf.w = vw;
                    vf.h = 1;
                    vf.channels = 1;

                    const std::string block_label = block_label_for(layer);
                    vf.label = !block_label.empty()
                        ? (block_label + "/vec")
                        : (layer.name.empty()
                            ? (type_to_string(layer.type_enum) + "#" + std::to_string(layer_idx) + "/vec")
                            : (layer.name + "/vec"));

                    auto it = std::find_if(viz_taps_.begin(), viz_taps_.end(), [&](const VizFrame& existing) {
                        return existing.label == vf.label;
                    });
                    if (it != viz_taps_.end()) {
                        *it = std::move(vf);
                    } else if (static_cast<int>(viz_taps_.size()) < viz_taps_max_frames_) {
                        viz_taps_.push_back(std::move(vf));
                    } else if (!viz_taps_.empty()) {
                        viz_taps_.back() = std::move(vf);
                    }
                }
            }
        }
        
        // ====================================================================
        // STORE OUTPUT (multi-output support)
        // ====================================================================
        
        std::string output_name = layer.output.empty() ? "x" : layer.output;

        // Masque (ReLU/Dropout) ou snapshot output (Reparameterize) selon besoins.
        if (training) {
            if (needs_output_mask(layer)) {
                std::vector<uint8_t> mask;
                mask.resize(layer_output.size());
                if ((layer.type == "Conv2d" || layer.type == "ConvTranspose2d") && layer.activation != ActivationType::NONE) {
                    for (size_t i = 0; i < layer_output.size(); ++i) mask[i] = (layer_output[i] > 0.0f) ? 1 : 0;
                } else {
                    for (size_t i = 0; i < layer_output.size(); ++i) mask[i] = (layer_output[i] != 0.0f) ? 1 : 0;
                }
                forward_state.layer_output_masks.back() = std::move(mask);
            }
            if (needs_output_snapshot(layer)) {
                forward_state.layer_outputs.back() = layer_output;
            }
        }

        storeTensor(output_name, std::move(layer_output));

        if (training && has_branches) {
            all_layer_outputs.push_back(getTensor(output_name));
        }
        
        // Gestion des branches
        if (layer.requiresBranchComputation() && training) {
            executeBranchComputation(layer_idx, all_layer_outputs, training);
            // Update tensor store avec le résultat mergé
            storeTensor(output_name, all_layer_outputs[layer_idx]);
        }
    }
    
    // Le résultat final est toujours dans "x" (ou dernier output)

    // ====================================================================
    // VAEConv extra viz frames: MU + resdiff (best-effort)
    // ====================================================================
    // Objectif UX: dans la Viz VAE_conv, toujours montrer le latent MU et
    // une heatmap d'erreur de reconstruction (|recon - input|).
    if (viz_taps_enabled_ && viz_taps_max_frames_ > 0) {
        auto downsample_mean_chw_to_gray = [&](
            const std::vector<float>& tensor,
            int W,
            int H,
            int C,
            int max_side,
            bool take_abs,
            bool symmetric
        ) -> VizFrame {
            VizFrame vf;
            if (W <= 0 || H <= 0 || C <= 0) return vf;
            const size_t expected = static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C);
            if (tensor.size() != expected) return vf;

            const int vw = std::max(1, std::min(max_side, W));
            const int vh = std::max(1, std::min(max_side, H));

            std::vector<float> map;
            map.resize(static_cast<size_t>(vw) * static_cast<size_t>(vh), 0.0f);

            for (int y = 0; y < vh; ++y) {
                const int sy = (vh > 1) ? (y * H) / vh : 0;
                for (int x = 0; x < vw; ++x) {
                    const int sx = (vw > 1) ? (x * W) / vw : 0;
                    double acc = 0.0;
                    const size_t base = (static_cast<size_t>(sy) * static_cast<size_t>(W) + static_cast<size_t>(sx)) * static_cast<size_t>(C);
                    for (int c = 0; c < C; ++c) {
                        float v = tensor[base + static_cast<size_t>(c)];
                        if (take_abs) v = std::fabs(v);
                        acc += static_cast<double>(v);
                    }
                    map[static_cast<size_t>(y) * static_cast<size_t>(vw) + static_cast<size_t>(x)] = static_cast<float>(acc / static_cast<double>(C));
                }
            }

            float max_abs = 0.0f;
            for (float v : map) {
                max_abs = std::max(max_abs, std::fabs(v));
            }
            const float inv = 1.0f / (max_abs + 1e-6f);

            vf.pixels.resize(map.size());
            for (size_t i = 0; i < map.size(); ++i) {
                const float s = map[i] * inv;
                float t = 0.0f;
                if (symmetric) {
                    // [-1, 1] -> [0, 1]
                    t = 0.5f + 0.5f * std::tanh(s);
                } else {
                    // [0, +] -> [0, 1]
                    t = std::clamp(s, 0.0f, 1.0f);
                }
                const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                vf.pixels[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
            }
            vf.w = vw;
            vf.h = vh;
            vf.channels = 1;
            return vf;
        };

        auto add_mu_frame = [&](const char* prefix) {
            const std::string mu_name = std::string(prefix) + "/mu";
            const std::string mu_layer = std::string(prefix) + "/enc/mu";
            if (!hasTensor(mu_name)) return;
            const auto& mu = getTensor(mu_name);
            int muW = 0, muH = 0, muC = 0;
            if (Layer* L = getLayerByName(mu_layer)) {
                muC = std::max(0, L->out_channels);
                muH = std::max(0, L->input_height);
                muW = std::max(0, L->input_width);
            }
            if (muC > 0 && (muW <= 0 || muH <= 0)) {
                const size_t hw = mu.size() / static_cast<size_t>(muC);
                const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                if (s > 0 && s * s == hw) {
                    muH = static_cast<int>(s);
                    muW = static_cast<int>(s);
                }
            }
            if (muW > 0 && muH > 0 && muC > 0 && mu.size() == static_cast<size_t>(muW) * static_cast<size_t>(muH) * static_cast<size_t>(muC)) {
                VizFrame vf = downsample_mean_chw_to_gray(mu, muW, muH, muC, std::max(1, viz_taps_max_side_), /*take_abs*/false, /*symmetric*/true);
                if (!vf.pixels.empty()) {
                    vf.label = std::string(prefix) + "/latent/mu";
                    addVizTapFrame(std::move(vf));
                }
            }
        };

        // MU: tensor spatial CHW (best-effort via layer config)
        add_mu_frame("vae_conv");

        auto add_resdiff_frame = [&](const char* prefix) {
            const std::string recon_name = std::string(prefix) + "/recon";
            const std::string in_name = std::string(prefix) + "/in_hwc";
            const std::string recon_to_hwc = std::string(prefix) + "/recon_to_hwc";
            if (!hasTensor(recon_name) || !hasTensor(in_name)) return;
            const auto& recon = getTensor(recon_name);
            const auto& in_hwc = getTensor(in_name);
            if (recon.size() != in_hwc.size() || recon.empty()) return;

            int W = 0, H = 0, C = 0;
            if (Layer* P = getLayerByName(recon_to_hwc)) {
                if (P->shape.size() == 3) {
                    C = std::max(0, P->shape[0]);
                    H = std::max(0, P->shape[1]);
                    W = std::max(0, P->shape[2]);
                }
            }
            if ((W <= 0 || H <= 0 || C <= 0) && recon.size() % 3 == 0) {
                C = 3;
                const size_t hw = recon.size() / 3ULL;
                const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                if (s > 0 && s * s == hw) {
                    H = static_cast<int>(s);
                    W = static_cast<int>(s);
                }
            }
            if (W <= 0 || H <= 0 || C <= 0) return;
            if (recon.size() != static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C)) return;

            std::vector<float> diff;
            diff.resize(recon.size());
            #pragma omp simd
            for (size_t i = 0; i < diff.size(); ++i) {
                diff[i] = std::fabs(recon[i] - in_hwc[i]);
            }
            VizFrame vf = downsample_mean_chw_to_gray(diff, W, H, C, std::max(1, viz_taps_max_side_), /*take_abs*/false, /*symmetric*/false);
            if (!vf.pixels.empty()) {
                vf.label = std::string(prefix) + "/err/resdiff_abs";
                addVizTapFrame(std::move(vf));
            }
        };

        // resdiff: |recon - input| en image-space (HWC)
        add_resdiff_frame("vae_conv");
    }

    return getTensor("x");
}

std::vector<float> Model::forwardPromptImageSeed(const std::vector<float>& prompt_vec,
                                                 const std::vector<float>& image_vec,
                                                 uint32_t seed,
                                                 bool training) {
    std::vector<float> packed;
    packed.reserve(prompt_vec.size() + image_vec.size());
    packed.insert(packed.end(), prompt_vec.begin(), prompt_vec.end());
    packed.insert(packed.end(), image_vec.begin(), image_vec.end());

    MimirRng::ScopedSeed scoped(seed);
    return forwardPass(packed, training);
}

namespace {

static inline float softmax_row_dot(const float* p, const float* dp, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += dp[i] * p[i];
    return s;
}

static bool backward_self_attention(
    const std::vector<float>& x,
    const std::vector<float>& grad_out,
    int seq_len,
    int embed_dim,
    int num_heads,
    bool causal,
    const float* Wqkv, // [embed, 3*embed]
    const float* Wout, // [embed, embed]
    float* grad_Wqkv,  // same layout as Wqkv
    float* grad_Wout,  // same layout as Wout
    std::vector<float>& grad_x
) {
    if (seq_len <= 0 || embed_dim <= 0 || num_heads <= 0) return false;
    if ((embed_dim % num_heads) != 0) return false;
    if (static_cast<int>(x.size()) != seq_len * embed_dim) return false;
    if (static_cast<int>(grad_out.size()) != seq_len * embed_dim) return false;

    const int head_dim = embed_dim / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // qkv = x @ Wqkv
    std::vector<float> qkv(static_cast<size_t>(seq_len) * static_cast<size_t>(3 * embed_dim), 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        const float* xrow = &x[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        float* yrow = &qkv[static_cast<size_t>(i) * static_cast<size_t>(3 * embed_dim)];
        for (int n = 0; n < 3 * embed_dim; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) {
                sum += xrow[static_cast<size_t>(k)] * Wqkv[static_cast<size_t>(k) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(n)];
            }
            yrow[static_cast<size_t>(n)] = sum;
        }
    }

    auto q_at = [&](int t, int d) -> float {
        return qkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(d)];
    };
    auto k_at = [&](int t, int d) -> float {
        return qkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(embed_dim + d)];
    };
    auto v_at = [&](int t, int d) -> float {
        return qkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(2 * embed_dim + d)];
    };

    // Attention probabilities P[h,i,j]
    std::vector<float> P(static_cast<size_t>(num_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len), 0.0f);
    std::vector<float> attended(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        const int hoff = h * head_dim;
        for (int i = 0; i < seq_len; ++i) {
            float maxv = -1e30f;
            for (int j = 0; j < seq_len; ++j) {
                float s = -1e9f;
                if (!(causal && j > i)) {
                    float dot = 0.0f;
                    for (int k = 0; k < head_dim; ++k) dot += q_at(i, hoff + k) * k_at(j, hoff + k);
                    s = dot * scale;
                }
                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                P[idx] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                const float e = std::exp(P[idx] - maxv);
                P[idx] = e;
                sum += e;
            }
            const float inv = 1.0f / (sum + 1e-9f);
            for (int j = 0; j < seq_len; ++j) {
                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                P[idx] *= inv;
            }

            for (int k = 0; k < head_dim; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                    acc += P[idx] * v_at(j, hoff + k);
                }
                attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] = acc;
            }
        }
    }

    // dAtt = dY @ Wout^T
    std::vector<float> dAtt(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        const float* dy = &grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        float* da = &dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        for (int k = 0; k < embed_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < embed_dim; ++n) {
                sum += dy[static_cast<size_t>(n)] * Wout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
            }
            da[static_cast<size_t>(k)] = sum;
        }
    }

    // dWout = attended^T @ dY
    for (int k = 0; k < embed_dim; ++k) {
        for (int n = 0; n < embed_dim; ++n) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                sum += attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)] * grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
            }
            grad_Wout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)] += sum;
        }
    }

    std::vector<float> dQ(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
    std::vector<float> dK(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
    std::vector<float> dV(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        const int hoff = h * head_dim;

        // dV = P^T @ dAtt
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < head_dim; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < seq_len; ++i) {
                    const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                    sum += P[idx] * dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)];
                }
                dV[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum;
            }
        }

        // dP = dAtt @ V^T
        std::vector<float> dP(static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len), 0.0f);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < head_dim; ++k) {
                    sum += dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] * v_at(j, hoff + k);
                }
                dP[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] = sum;
            }
        }

        // softmax backward -> dS
        std::vector<float> dS(static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len), 0.0f);
        for (int i = 0; i < seq_len; ++i) {
            const float* p_row = &P[(static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len)];
            const float* dp_row = &dP[static_cast<size_t>(i) * static_cast<size_t>(seq_len)];
            const float dot = softmax_row_dot(p_row, dp_row, seq_len);

            for (int j = 0; j < seq_len; ++j) {
                if (causal && j > i) {
                    dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] = 0.0f;
                    continue;
                }
                dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] = (dp_row[j] - dot) * p_row[j];
            }
        }

        // dQ = dS @ K * scale
        for (int i = 0; i < seq_len; ++i) {
            for (int k = 0; k < head_dim; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    sum += dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] * k_at(j, hoff + k);
                }
                dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
            }
        }

        // dK = dS^T @ Q * scale
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < head_dim; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < seq_len; ++i) {
                    sum += dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] * q_at(i, hoff + k);
                }
                dK[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
            }
        }
    }

    // dqkv and grads
    std::vector<float> dqkv(static_cast<size_t>(seq_len) * static_cast<size_t>(3 * embed_dim), 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        float* row = &dqkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim)];
        for (int d = 0; d < embed_dim; ++d) {
            row[static_cast<size_t>(d)] = dQ[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
            row[static_cast<size_t>(embed_dim + d)] = dK[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
            row[static_cast<size_t>(2 * embed_dim + d)] = dV[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
        }
    }

    // dWqkv = x^T @ dqkv
    for (int k = 0; k < embed_dim; ++k) {
        for (int n = 0; n < 3 * embed_dim; ++n) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                sum += x[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)] * dqkv[static_cast<size_t>(i) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(n)];
            }
            grad_Wqkv[static_cast<size_t>(k) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(n)] += sum;
        }
    }

    // dx = dqkv @ Wqkv^T
    grad_x.assign(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        const float* drow = &dqkv[static_cast<size_t>(i) * static_cast<size_t>(3 * embed_dim)];
        float* xrow = &grad_x[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        for (int k = 0; k < embed_dim; ++k) {
            float sum = 0.0f;
            const float* wrow = &Wqkv[static_cast<size_t>(k) * static_cast<size_t>(3 * embed_dim)];
            for (int n = 0; n < 3 * embed_dim; ++n) sum += drow[n] * wrow[n];
            xrow[static_cast<size_t>(k)] = sum;
        }
    }

    return true;
}

static bool backward_cross_attention(
    const std::vector<float>& q_in,
    const std::vector<float>& kv_in,
    const std::vector<float>& grad_out,
    int query_len,
    int kv_len,
    int embed_dim,
    int num_heads,
    bool causal,
    const float* Wq,
    const float* Wkv,
    const float* Wout,
    float* grad_Wq,
    float* grad_Wkv,
    float* grad_Wout,
    std::vector<float>& grad_q,
    std::vector<float>& grad_kv
) {
    if (query_len <= 0 || kv_len <= 0 || embed_dim <= 0 || num_heads <= 0) return false;
    if ((embed_dim % num_heads) != 0) return false;
    if (static_cast<int>(q_in.size()) != query_len * embed_dim) return false;
    if (static_cast<int>(kv_in.size()) != kv_len * embed_dim) return false;
    if (static_cast<int>(grad_out.size()) != query_len * embed_dim) return false;

    const int head_dim = embed_dim / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Q = q_in @ Wq
    std::vector<float> Q(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
    for (int i = 0; i < query_len; ++i) {
        const float* xrow = &q_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        float* yrow = &Q[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        for (int n = 0; n < embed_dim; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) sum += xrow[k] * Wq[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
            yrow[n] = sum;
        }
    }
    // KV = kv_in @ Wkv -> [kv_len, 2*embed]
    std::vector<float> KV(static_cast<size_t>(kv_len) * static_cast<size_t>(2 * embed_dim), 0.0f);
    for (int i = 0; i < kv_len; ++i) {
        const float* xrow = &kv_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        float* yrow = &KV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim)];
        for (int n = 0; n < 2 * embed_dim; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < embed_dim; ++k) sum += xrow[k] * Wkv[static_cast<size_t>(k) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)];
            yrow[n] = sum;
        }
    }

    auto q_at = [&](int t, int d) -> float { return Q[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)]; };
    auto k_at = [&](int t, int d) -> float { return KV[static_cast<size_t>(t) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(d)]; };
    auto v_at = [&](int t, int d) -> float { return KV[static_cast<size_t>(t) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(embed_dim + d)]; };

    // P and attended
    std::vector<float> P(static_cast<size_t>(num_heads) * static_cast<size_t>(query_len) * static_cast<size_t>(kv_len), 0.0f);
    std::vector<float> attended(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
    for (int h = 0; h < num_heads; ++h) {
        const int hoff = h * head_dim;
        for (int i = 0; i < query_len; ++i) {
            float maxv = -1e30f;
            for (int j = 0; j < kv_len; ++j) {
                float s = -1e9f;
                if (!(causal && j > i)) {
                    float dot = 0.0f;
                    for (int k = 0; k < head_dim; ++k) dot += q_at(i, hoff + k) * k_at(j, hoff + k);
                    s = dot * scale;
                }
                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                P[idx] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_len; ++j) {
                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                const float e = std::exp(P[idx] - maxv);
                P[idx] = e;
                sum += e;
            }
            const float inv = 1.0f / (sum + 1e-9f);
            for (int j = 0; j < kv_len; ++j) {
                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                P[idx] *= inv;
            }
            for (int k = 0; k < head_dim; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < kv_len; ++j) {
                    const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                    acc += P[idx] * v_at(j, hoff + k);
                }
                attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] = acc;
            }
        }
    }

    // dAtt = dY @ Wout^T
    std::vector<float> dAtt(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
    for (int i = 0; i < query_len; ++i) {
        const float* dy = &grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        float* da = &dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        for (int k = 0; k < embed_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < embed_dim; ++n) sum += dy[n] * Wout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
            da[k] = sum;
        }
    }

    // dWout = attended^T @ dY
    for (int k = 0; k < embed_dim; ++k) {
        for (int n = 0; n < embed_dim; ++n) {
            float sum = 0.0f;
            for (int i = 0; i < query_len; ++i) {
                sum += attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)] * grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
            }
            grad_Wout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)] += sum;
        }
    }

    std::vector<float> dQ(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
    std::vector<float> dK(static_cast<size_t>(kv_len) * static_cast<size_t>(embed_dim), 0.0f);
    std::vector<float> dV(static_cast<size_t>(kv_len) * static_cast<size_t>(embed_dim), 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        const int hoff = h * head_dim;
        // dV = P^T @ dAtt
        for (int j = 0; j < kv_len; ++j) {
            for (int k = 0; k < head_dim; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < query_len; ++i) {
                    const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                    sum += P[idx] * dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)];
                }
                dV[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum;
            }
        }

        // dP = dAtt @ V^T
        std::vector<float> dP(static_cast<size_t>(query_len) * static_cast<size_t>(kv_len), 0.0f);
        for (int i = 0; i < query_len; ++i) {
            for (int j = 0; j < kv_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < head_dim; ++k) {
                    sum += dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] * v_at(j, hoff + k);
                }
                dP[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] = sum;
            }
        }

        // softmax backward -> dS
        std::vector<float> dS(static_cast<size_t>(query_len) * static_cast<size_t>(kv_len), 0.0f);
        for (int i = 0; i < query_len; ++i) {
            const float* p_row = &P[(static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len)];
            const float* dp_row = &dP[static_cast<size_t>(i) * static_cast<size_t>(kv_len)];
            const float dot = softmax_row_dot(p_row, dp_row, kv_len);
            for (int j = 0; j < kv_len; ++j) {
                if (causal && j > i) {
                    dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] = 0.0f;
                    continue;
                }
                dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] = (dp_row[j] - dot) * p_row[j];
            }
        }

        // dQ
        for (int i = 0; i < query_len; ++i) {
            for (int k = 0; k < head_dim; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < kv_len; ++j) sum += dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] * k_at(j, hoff + k);
                dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
            }
        }
        // dK
        for (int j = 0; j < kv_len; ++j) {
            for (int k = 0; k < head_dim; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < query_len; ++i) sum += dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] * q_at(i, hoff + k);
                dK[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
            }
        }
    }

    // dWq = q_in^T @ dQ ; dq_in = dQ @ Wq^T
    for (int k = 0; k < embed_dim; ++k) {
        for (int n = 0; n < embed_dim; ++n) {
            float sum = 0.0f;
            for (int i = 0; i < query_len; ++i) {
                sum += q_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)] * dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
            }
            grad_Wq[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)] += sum;
        }
    }
    grad_q.assign(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
    for (int i = 0; i < query_len; ++i) {
        const float* drow = &dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        float* xrow = &grad_q[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        for (int k = 0; k < embed_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < embed_dim; ++n) sum += drow[n] * Wq[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
            xrow[k] = sum;
        }
    }

    // dKV = concat(dK,dV)
    std::vector<float> dKV(static_cast<size_t>(kv_len) * static_cast<size_t>(2 * embed_dim), 0.0f);
    for (int i = 0; i < kv_len; ++i) {
        float* row = &dKV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim)];
        for (int d = 0; d < embed_dim; ++d) {
            row[d] = dK[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
            row[embed_dim + d] = dV[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
        }
    }
    // dWkv = kv_in^T @ dKV
    for (int k = 0; k < embed_dim; ++k) {
        for (int n = 0; n < 2 * embed_dim; ++n) {
            float sum = 0.0f;
            for (int i = 0; i < kv_len; ++i) {
                sum += kv_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)] * dKV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)];
            }
            grad_Wkv[static_cast<size_t>(k) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)] += sum;
        }
    }

    // dkv_in = dKV @ Wkv^T
    grad_kv.assign(static_cast<size_t>(kv_len) * static_cast<size_t>(embed_dim), 0.0f);
    for (int i = 0; i < kv_len; ++i) {
        const float* drow = &dKV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim)];
        float* xrow = &grad_kv[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
        for (int k = 0; k < embed_dim; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < 2 * embed_dim; ++n) sum += drow[n] * Wkv[static_cast<size_t>(k) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)];
            xrow[k] = sum;
        }
    }

    return true;
}

} // namespace

Gradients Model::backwardPass(const std::vector<float> &loss_gradient) {
    if (params_frozen_) {
        throw std::runtime_error("Model::backwardPass: parameters are frozen");
    }
    Gradients grads;
    
    if (!forward_state.is_valid) {
        std::cerr << "⚠️  Cannot perform backward pass: no valid forward state" << std::endl;
        std::cerr << "    Call forwardPass() in training mode first" << std::endl;
        return grads;
    }
    
    if (layers.empty() || layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot perform backward pass: layers or weights not initialized" << std::endl;
        return grads;
    }

    MemoryGuard& guard = MemoryGuard::instance();
    const size_t guard_mb = guard.getLimit() / (1024ULL * 1024ULL);
    const size_t cap_mb = (max_ram_mb_ > 0) ? max_ram_mb_ : guard_mb;
    RuntimeAllocator allocator(guard, cap_mb);

    auto accumulate_grad = [](std::vector<float>& dst, const std::vector<float>& src) {
        if (dst.empty()) {
            dst = src;
            return;
        }
        if (dst.size() != src.size()) {
            throw std::runtime_error("Gradient accumulate: size mismatch");
        }
        #pragma omp simd
        for (size_t i = 0; i < dst.size(); ++i) {
            dst[i] += src[i];
        }
    };

    // Gradients routés par nom de tensor (TensorStore)
    std::unordered_map<std::string, std::vector<float>> grad_store;
    grad_store["x"] = loss_gradient;

    // Backward pass à travers chaque layer (en ordre inverse)
    for (int layer_idx = static_cast<int>(layers.size()) - 1; layer_idx >= 0; --layer_idx) {
        auto &layer = layers[static_cast<size_t>(layer_idx)];

        const std::string output_name = layer.output.empty() ? "x" : layer.output;
        auto it_go = grad_store.find(output_name);
        if (it_go == grad_store.end()) {
            continue;
        }
        const std::vector<float>& grad_out = it_go->second;

        const std::vector<float>* layer_out_snap = nullptr;
        if (static_cast<size_t>(layer_idx) < forward_state.layer_outputs.size()) {
            layer_out_snap = &forward_state.layer_outputs[static_cast<size_t>(layer_idx)];
        }

        static const std::vector<std::string> kDefaultInputNameX = {"x"};
        const std::vector<std::string>& input_names = layer.inputs.empty() ? kDefaultInputNameX : layer.inputs;

        // Tailles des inputs snapshotées (utile quand les valeurs n'ont pas été copiées)
        std::vector<size_t> input_sizes;
        bool have_input_sizes = false;
        if (static_cast<size_t>(layer_idx) < forward_state.layer_input_sizes_multi.size()) {
            const auto& sz = forward_state.layer_input_sizes_multi[static_cast<size_t>(layer_idx)];
            if (sz.size() == input_names.size()) {
                input_sizes = sz;
                have_input_sizes = true;
            }
        }

        auto get_fallback_sizes_from_store = [&]() {
            input_sizes.clear();
            input_sizes.reserve(input_names.size());
            for (const auto& nm : input_names) {
                try {
                    input_sizes.push_back(getTensor(nm).size());
                } catch (...) {
                    input_sizes.push_back(0ULL);
                }
            }
            have_input_sizes = (input_sizes.size() == input_names.size());
        };
        if (!have_input_sizes) {
            get_fallback_sizes_from_store();
        }

        auto needs_input_values_for_backward = [&](const Layer& l) -> bool {
            if (l.type == "Add" || l.type == "Concat" || l.type == "Split" || l.type == "Subtract" ||
                l.type == "TokenMeanPool" || l.type == "UpsampleNearest" || l.type == "Identity") {
                return false;
            }
            if (l.type == "Dropout" || l.type == "Dropout2d" || l.type == "AlphaDropout") {
                return false;
            }
            return true;
        };

        // Récupérer les inputs tels qu'ils étaient pendant le forward (sinon "x" peut être écrasé)
        std::vector<const std::vector<float>*> inputs;
        if (needs_input_values_for_backward(layer)) {
            inputs.reserve(input_names.size());
            if (static_cast<size_t>(layer_idx) < forward_state.layer_inputs_multi.size()) {
                const auto& snap = forward_state.layer_inputs_multi[static_cast<size_t>(layer_idx)];
                if (snap.size() == input_names.size()) {
                    for (size_t i = 0; i < snap.size(); ++i) {
                        inputs.push_back(&snap[i]);
                    }
                }
            }
            if (inputs.size() != input_names.size()) {
                // Fallback: relire TensorStore (moins fiable si noms réutilisés)
                inputs.clear();
                inputs.reserve(input_names.size());
                for (const auto& nm : input_names) {
                    inputs.push_back(&getTensor(nm));
                }
            }
        }

        const size_t input0_size = (have_input_sizes && !input_sizes.empty()) ? input_sizes[0] : (inputs.empty() ? 0ULL : inputs[0]->size());
        std::vector<float> grad_input0(input0_size, 0.0f);

        if (layer.type == "Embedding") {
            if (inputs.empty()) {
                // Embedding backward nécessite les ids.
                continue;
            }
            const int vocab = std::max(1, layer.vocab_size);
            const int dim = std::max(1, layer.embed_dim);
            const int pad = layer.padding_idx;

            const std::vector<float>& layer_input0 = *inputs[0];
            if (grad_out.size() != layer_input0.size() * static_cast<size_t>(dim)) {
                std::cerr << "⚠️  Embedding backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            const size_t w_expected = static_cast<size_t>(vocab) * static_cast<size_t>(dim);
            if (layer.getWeights() == nullptr || layer.getWeightsSize() < w_expected) {
                std::cerr << "⚠️  Embedding backward skipped: weights invalid (" << layer.name << ")" << std::endl;
                continue;
            }
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }

            for (size_t t = 0; t < layer_input0.size(); ++t) {
                const int id = static_cast<int>(std::llround(static_cast<double>(layer_input0[t])));
                if (pad >= 0 && id == pad) continue;
                if (id < 0 || id >= vocab) continue;
                const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                const size_t base_g = t * static_cast<size_t>(dim);
                for (int d = 0; d < dim; ++d) {
                    layer.grad_weights[base_w + static_cast<size_t>(d)] += grad_out[base_g + static_cast<size_t>(d)];
                }
            }

        } else if (layer.type == "Identity") {
            if (grad_out.size() != input0_size) {
                std::cerr << "⚠️  Identity backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            accumulate_grad(grad_store[input_names[0]], grad_out);

        } else if (layer.type == "Concat") {
            if (input_names.size() < 2) {
                std::cerr << "⚠️  Concat backward skipped: need >=2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }

            size_t total = 0;
            if (have_input_sizes && input_sizes.size() == input_names.size()) {
                for (size_t i = 0; i < input_sizes.size(); ++i) total += input_sizes[i];
            } else {
                for (const auto* inp : inputs) total += inp->size();
            }
            if (grad_out.size() != total) {
                std::cerr << "⚠️  Concat backward shape mismatch (" << layer.name << ")"
                          << " grad=" << grad_out.size() << " expected=" << total << std::endl;
                continue;
            }

            size_t off = 0;
            const size_t nin = have_input_sizes ? input_sizes.size() : input_names.size();
            for (size_t i = 0; i < nin; ++i) {
                const size_t n = (have_input_sizes && i < input_sizes.size()) ? input_sizes[i] : getTensor(input_names[i]).size();
                std::vector<float> grad_in(n, 0.0f);
                if (n > 0) {
                    std::copy_n(grad_out.data() + off, n, grad_in.data());
                }
                off += n;
                accumulate_grad(grad_store[input_names[i]], grad_in);
            }

        } else if (layer.type == "Split") {
            // IMPORTANT: Split forward produit des sorties nommées base_0, base_1, ...
            // Le tensor `layer.output` contient seulement base_0; le gradient utile est donc
            // dans les branches base_i. On les concatène pour former grad_input.
            if (layer.split_sizes.empty()) {
                std::cerr << "⚠️  Split backward skipped: split_sizes not configured (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::string base = layer.output.empty() ? "x" : layer.output;

            size_t total = 0;
            for (int sz : layer.split_sizes) total += static_cast<size_t>(std::max(0, sz));
            if (total != input0_size) {
                // Tolérance: on peut encore propager si la taille correspond au total.
                // Sinon, on évite de corrompre les grads.
                std::cerr << "⚠️  Split backward shape mismatch (" << layer.name << ")"
                          << " input=" << input0_size << " expected=" << total << std::endl;
                continue;
            }

            std::vector<float> grad_in(total, 0.0f);
            size_t off = 0;
            for (size_t i = 0; i < layer.split_sizes.size(); ++i) {
                const size_t n = static_cast<size_t>(std::max(0, layer.split_sizes[i]));
                const std::string out_i = base + "_" + std::to_string(i);
                auto it = grad_store.find(out_i);
                if (it != grad_store.end()) {
                    if (it->second.size() != n) {
                        std::cerr << "⚠️  Split backward: grad size mismatch for " << out_i
                                  << " got=" << it->second.size() << " expected=" << n << std::endl;
                        continue;
                    }
                    if (n > 0) {
                        std::copy_n(it->second.data(), n, grad_in.data() + off);
                    }
                }
                off += n;
            }

            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "Add") {
            if ((!have_input_sizes || input_sizes.size() < 2) && inputs.size() < 2) {
                std::cerr << "⚠️  Add backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }

            const size_t n0 = (have_input_sizes && input_sizes.size() >= 1) ? input_sizes[0] : inputs[0]->size();
            const size_t n1 = (have_input_sizes && input_sizes.size() >= 2) ? input_sizes[1] : inputs[1]->size();

            if (grad_out.size() == n0 && grad_out.size() == n1) {
                // Standard elementwise add
                accumulate_grad(grad_store[input_names[0]], grad_out);
                accumulate_grad(grad_store[input_names[1]], grad_out);
            } else {
                // Broadcast support (match LayerOps::add_forward)
                const size_t g = grad_out.size();

                const bool g_matches_0 = (g == n0);
                const bool g_matches_1 = (g == n1);
                if (!g_matches_0 && !g_matches_1) {
                    std::cerr << "⚠️  Add backward shape mismatch (" << layer.name << ")" << std::endl;
                    continue;
                }

                const bool in0_is_big = (g == n0);
                const std::string& big_name = in0_is_big ? input_names[0] : input_names[1];
                const std::string& small_name = in0_is_big ? input_names[1] : input_names[0];

                const size_t big_n = in0_is_big ? n0 : n1;
                const size_t small_n = in0_is_big ? n1 : n0;

                if (big_n != g) {
                    std::cerr << "⚠️  Add backward shape mismatch (" << layer.name << ")" << std::endl;
                    continue;
                }

                // Gradient wrt big input is direct
                accumulate_grad(grad_store[big_name], grad_out);

                // Gradient wrt small input is reduction
                if (small_n == 1) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < g; ++i) sum += grad_out[i];
                    std::vector<float> gs(1, sum);
                    accumulate_grad(grad_store[small_name], gs);
                } else if (small_n > 1 && (big_n % small_n) == 0) {
                    std::vector<float> gs(small_n, 0.0f);
                    const size_t tiles = big_n / small_n;
                    for (size_t t = 0; t < tiles; ++t) {
                        const size_t base = t * small_n;
                        for (size_t j = 0; j < small_n; ++j) {
                            gs[j] += grad_out[base + j];
                        }
                    }
                    accumulate_grad(grad_store[small_name], gs);
                } else {
                    std::cerr << "⚠️  Add backward broadcast unsupported (" << layer.name << ")"
                              << " big=" << big_n << " small=" << small_n << std::endl;
                    continue;
                }
            }

        } else if (layer.type == "TokenMeanPool") {
            const int seq_len = layer.seq_len > 0 ? layer.seq_len : 0;
            const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : 0;
            if (seq_len <= 0 || embed_dim <= 0) {
                std::cerr << "⚠️  TokenMeanPool backward skipped: seq_len/embed_dim missing (" << layer.name << ")" << std::endl;
                continue;
            }
            if (grad_out.size() != static_cast<size_t>(embed_dim)) {
                std::cerr << "⚠️  TokenMeanPool backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            if (input0_size != static_cast<size_t>(seq_len * embed_dim)) {
                std::cerr << "⚠️  TokenMeanPool backward input shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            std::vector<float> grad_in(input0_size, 0.0f);
            const float inv = 1.0f / static_cast<float>(seq_len);
            for (int t = 0; t < seq_len; ++t) {
                const int base = t * embed_dim;
                for (int d = 0; d < embed_dim; ++d) {
                    grad_in[static_cast<size_t>(base + d)] = grad_out[static_cast<size_t>(d)] * inv;
                }
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "Subtract") {
            if ((!have_input_sizes || input_sizes.size() < 2) && inputs.size() < 2) {
                std::cerr << "⚠️  Subtract backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }

            const size_t n0 = (have_input_sizes && input_sizes.size() >= 1) ? input_sizes[0] : inputs[0]->size();
            const size_t n1 = (have_input_sizes && input_sizes.size() >= 2) ? input_sizes[1] : inputs[1]->size();
            if (grad_out.size() != n0 || grad_out.size() != n1) {
                std::cerr << "⚠️  Subtract backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            accumulate_grad(grad_store[input_names[0]], grad_out);
            std::vector<float> g1(grad_out.size());
            #pragma omp simd
            for (size_t i = 0; i < grad_out.size(); ++i) g1[i] = -grad_out[i];
            accumulate_grad(grad_store[input_names[1]], g1);

        } else if (layer.type == "Multiply") {
            if (inputs.size() < 2) {
                std::cerr << "⚠️  Multiply backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& in0 = *inputs[0];
            const std::vector<float>& in1 = *inputs[1];
            if (grad_out.size() != in0.size() || grad_out.size() != in1.size()) {
                std::cerr << "⚠️  Multiply backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            std::vector<float> g0(grad_out.size()), g1(grad_out.size());
            #pragma omp simd
            for (size_t i = 0; i < grad_out.size(); ++i) {
                g0[i] = grad_out[i] * in1[i];
                g1[i] = grad_out[i] * in0[i];
            }
            accumulate_grad(grad_store[input_names[0]], g0);
            accumulate_grad(grad_store[input_names[1]], g1);

        } else if (layer.type == "Reparameterize") {
            if (inputs.size() < 2) {
                std::cerr << "⚠️  Reparameterize backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& mu = *inputs[0];
            const std::vector<float>& logvar = *inputs[1];
            if (grad_out.size() != mu.size() || grad_out.size() != logvar.size()) {
                std::cerr << "⚠️  Reparameterize backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            // If forward ran in inference mode (z=mu), grad_logvar is zero.
            // We detect this best-effort: if no snapshot of output, fallback.
            if (!layer_out_snap || layer_out_snap->size() != grad_out.size()) {
                accumulate_grad(grad_store[input_names[0]], grad_out);
                continue;
            }
            const std::vector<float>& z = *layer_out_snap;
            std::vector<float> g_mu(grad_out.size(), 0.0f);
            std::vector<float> g_logvar(grad_out.size(), 0.0f);
            for (size_t i = 0; i < grad_out.size(); ++i) {
                const float lv = std::clamp(logvar[i], -20.0f, 20.0f);
                const float stdv = std::exp(0.5f * lv);
                const float inv_std = (stdv > 1e-12f) ? (1.0f / stdv) : 0.0f;
                const float eps = (z[i] - mu[i]) * inv_std;
                g_mu[i] = grad_out[i];
                g_logvar[i] = grad_out[i] * eps * stdv * 0.5f;
            }
            accumulate_grad(grad_store[input_names[0]], g_mu);
            accumulate_grad(grad_store[input_names[1]], g_logvar);

        } else if (layer.type == "Divide") {
            if (inputs.size() < 2) {
                std::cerr << "⚠️  Divide backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& in0 = *inputs[0];
            const std::vector<float>& in1 = *inputs[1];
            if (grad_out.size() != in0.size() || grad_out.size() != in1.size()) {
                std::cerr << "⚠️  Divide backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            std::vector<float> g0(grad_out.size()), g1(grad_out.size());
            for (size_t i = 0; i < grad_out.size(); ++i) {
                const float denom = in1[i];
                const float inv = 1.0f / (denom + 1e-12f);
                g0[i] = grad_out[i] * inv;
                g1[i] = grad_out[i] * (-in0[i]) * inv * inv;
            }
            accumulate_grad(grad_store[input_names[0]], g0);
            accumulate_grad(grad_store[input_names[1]], g1);

        } else if (layer.type == "LayerNorm") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            if (grad_out.size() != layer_input0.size()) {
                std::cerr << "⚠️  LayerNorm backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            const int N = static_cast<int>(layer_input0.size());
            const int feat = (layer.in_features > 0) ? layer.in_features : N;
            const int seq_len = (layer.seq_len > 0) ? layer.seq_len : 0;
            const int tokens = (seq_len > 0 && N == seq_len * feat) ? seq_len : ((feat > 0 && (N % feat) == 0) ? (N / feat) : 1);
            if (feat <= 0 || tokens <= 0 || tokens * feat != N) {
                std::cerr << "⚠️  LayerNorm backward invalid shape (" << layer.name << ")"
                          << " N=" << N << " feat=" << feat << " tokens=" << tokens << std::endl;
                continue;
            }

            const float eps = layer.eps;

            const bool affine = layer.affine;
            const float* w = (affine ? layer.getWeights() : nullptr);
            const bool has_bias = (affine && layer.use_bias);

            if (affine) {
                const size_t expected_w = static_cast<size_t>(feat) * (has_bias ? 2ULL : 1ULL);
                if (layer.getWeights() == nullptr || layer.getWeightsSize() < expected_w) {
                    std::cerr << "⚠️  LayerNorm backward skipped: weights invalid (" << layer.name << ")" << std::endl;
                    continue;
                }
                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
            }

            // dgamma/dbeta accumulés sur tous les tokens
            std::vector<float> dgamma(affine ? static_cast<size_t>(feat) : 0ULL, 0.0f);
            std::vector<float> dbeta((affine && has_bias) ? static_cast<size_t>(feat) : 0ULL, 0.0f);

            for (int t = 0; t < tokens; ++t) {
                const float* x = &layer_input0[static_cast<size_t>(t) * static_cast<size_t>(feat)];
                const float* dy = &grad_out[static_cast<size_t>(t) * static_cast<size_t>(feat)];
                float* dx = &grad_input0[static_cast<size_t>(t) * static_cast<size_t>(feat)];

                float mean = 0.0f;
                #pragma omp simd reduction(+:mean)
                for (int i = 0; i < feat; ++i) mean += x[static_cast<size_t>(i)];
                mean /= std::max(1, feat);

                float var = 0.0f;
                #pragma omp simd reduction(+:var)
                for (int i = 0; i < feat; ++i) {
                    const float d = x[static_cast<size_t>(i)] - mean;
                    var += d * d;
                }
                var /= std::max(1, feat);
                const float inv_std = 1.0f / std::sqrt(var + eps);

                // xhat + dxhat
                float sum_dxhat = 0.0f;
                float sum_dxhat_xhat = 0.0f;

                for (int i = 0; i < feat; ++i) {
                    const float xhat = (x[static_cast<size_t>(i)] - mean) * inv_std;
                    const float gamma = (w ? w[static_cast<size_t>(i)] : 1.0f);
                    const float dxhat = dy[static_cast<size_t>(i)] * gamma;

                    if (affine) {
                        dgamma[static_cast<size_t>(i)] += dy[static_cast<size_t>(i)] * xhat;
                        if (has_bias) {
                            dbeta[static_cast<size_t>(i)] += dy[static_cast<size_t>(i)];
                        }
                    }

                    sum_dxhat += dxhat;
                    sum_dxhat_xhat += dxhat * xhat;
                }

                const float invF = 1.0f / std::max(1, feat);
                for (int i = 0; i < feat; ++i) {
                    const float xhat = (x[static_cast<size_t>(i)] - mean) * inv_std;
                    const float gamma = (w ? w[static_cast<size_t>(i)] : 1.0f);
                    const float dxhat = dy[static_cast<size_t>(i)] * gamma;
                    dx[static_cast<size_t>(i)] = inv_std * invF * (static_cast<float>(feat) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
                }
            }

            if (affine) {
                // Copier dgamma/dbeta vers le buffer grad_weights
                for (int i = 0; i < feat; ++i) {
                    layer.grad_weights[static_cast<size_t>(i)] += dgamma[static_cast<size_t>(i)];
                }
                if (has_bias) {
                    const size_t off = static_cast<size_t>(feat);
                    for (int i = 0; i < feat; ++i) {
                        layer.grad_weights[off + static_cast<size_t>(i)] += dbeta[static_cast<size_t>(i)];
                    }
                }
            }

            accumulate_grad(grad_store[input_names[0]], grad_input0);

        } else if (layer.type == "Dropout" || layer.type == "Dropout2d") {
            if (grad_out.size() != input0_size) {
                std::cerr << "⚠️  Dropout backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            // Préférence: masque sauvegardé (évite copie d'output).
            const std::vector<uint8_t>* maskp = nullptr;
            if (static_cast<size_t>(layer_idx) < forward_state.layer_output_masks.size()) {
                const auto& m = forward_state.layer_output_masks[static_cast<size_t>(layer_idx)];
                if (!m.empty() && m.size() == grad_out.size()) maskp = &m;
            }
            // Fallback legacy: sortie snapshotée
            if (!maskp && (!layer_out_snap || layer_out_snap->size() != grad_out.size())) {
                accumulate_grad(grad_store[input_names[0]], grad_out);
                continue;
            }
            const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
            const float scale = (p < 1.0f) ? (1.0f / (1.0f - p)) : 0.0f;
            std::vector<float> grad_in(grad_out.size(), 0.0f);
            for (size_t i = 0; i < grad_in.size(); ++i) {
                const bool kept = maskp ? ((*maskp)[i] != 0) : ((*layer_out_snap)[i] != 0.0f);
                grad_in[i] = kept ? (grad_out[i] * scale) : 0.0f;
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "AlphaDropout") {
            if (grad_out.size() != input0_size) {
                std::cerr << "⚠️  AlphaDropout backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            const std::vector<uint8_t>* maskp = nullptr;
            if (static_cast<size_t>(layer_idx) < forward_state.layer_output_masks.size()) {
                const auto& m = forward_state.layer_output_masks[static_cast<size_t>(layer_idx)];
                if (!m.empty() && m.size() == grad_out.size()) maskp = &m;
            }
            if (!maskp && (!layer_out_snap || layer_out_snap->size() != grad_out.size())) {
                accumulate_grad(grad_store[input_names[0]], grad_out);
                continue;
            }
            const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
            const float alpha = 1.6732632423543772848170429916717f;
            const float scale_selu = 1.0507009873554804934193349852946f;
            const float alpha_p = -alpha * scale_selu;
            const float a = 1.0f / std::sqrt((1.0f - p) * (1.0f + p * alpha_p * alpha_p));
            const float b = -a * alpha_p * p;
            const float dropped_out = a * alpha_p + b;
            std::vector<float> grad_in(grad_out.size(), 0.0f);
            for (size_t i = 0; i < grad_in.size(); ++i) {
                const float y = maskp ? ( (*maskp)[i] ? 0.0f : dropped_out ) : (*layer_out_snap)[i];
                const bool kept = std::fabs(y - dropped_out) > 1e-6f;
                grad_in[i] = kept ? (grad_out[i] * a) : 0.0f;
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "Reshape") {
            // Reshape: pas de changement de layout (uniquement une vue logique)
            if (grad_out.size() != input0_size) {
                std::cerr << "⚠️  Reshape backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            accumulate_grad(grad_store[input_names[0]], grad_out);

        } else if (layer.type == "Permute") {
            // Permute: nécessite l'inverse de la permutation
            // On supporte le cas générique N-D quand shape + permute_dims sont fournis.
            if (layer.shape.empty() || layer.permute_dims.empty() || layer.shape.size() != layer.permute_dims.size()) {
                std::cerr << "⚠️  Permute backward skipped: missing shape/permute_dims (" << layer.name << ")" << std::endl;
                continue;
            }
            const size_t ndim = layer.shape.size();
            size_t total = 1;
            for (size_t d = 0; d < ndim; ++d) {
                total *= static_cast<size_t>(std::max(1, layer.shape[d]));
            }
            if (grad_out.size() != total || input0_size != total) {
                std::cerr << "⚠️  Permute backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            std::vector<int> perm = layer.permute_dims;
            if (perm.size() != ndim) {
                std::cerr << "⚠️  Permute backward invalid permute_dims (" << layer.name << ")" << std::endl;
                continue;
            }
            std::vector<size_t> in_shape(ndim);
            for (size_t i = 0; i < ndim; ++i) in_shape[i] = static_cast<size_t>(std::max(1, layer.shape[i]));
            std::vector<size_t> out_shape(ndim);
            for (size_t i = 0; i < ndim; ++i) {
                const int src = perm[i];
                if (src < 0 || static_cast<size_t>(src) >= ndim) {
                    std::cerr << "⚠️  Permute backward invalid perm index (" << layer.name << ")" << std::endl;
                    out_shape.clear();
                    break;
                }
                out_shape[i] = in_shape[static_cast<size_t>(src)];
            }
            if (out_shape.empty()) continue;

            auto make_strides = [](const std::vector<size_t>& shape) {
                std::vector<size_t> strides(shape.size(), 1);
                for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
                    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
                }
                return strides;
            };

            const std::vector<size_t> in_strides = make_strides(in_shape);
            const std::vector<size_t> out_strides = make_strides(out_shape);

            std::vector<float> grad_in(total, 0.0f);
            std::vector<size_t> out_idx(ndim, 0);
            std::vector<size_t> in_idx(ndim, 0);
            for (size_t flat = 0; flat < total; ++flat) {
                // flat -> out multi-index
                size_t rem = flat;
                for (size_t d = 0; d < ndim; ++d) {
                    const size_t sd = out_strides[d];
                    out_idx[d] = (sd > 0) ? (rem / sd) : 0;
                    rem = (sd > 0) ? (rem % sd) : 0;
                }
                // out_idx -> in_idx via permute mapping
                for (size_t d = 0; d < ndim; ++d) in_idx[d] = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    const size_t src = static_cast<size_t>(perm[d]);
                    in_idx[src] = out_idx[d];
                }
                // in_idx -> in flat
                size_t in_flat = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    in_flat += in_idx[d] * in_strides[d];
                }
                grad_in[in_flat] += grad_out[flat];
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "Sigmoid") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            if (grad_out.size() != layer_input0.size()) {
                std::cerr << "⚠️  Sigmoid backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            std::vector<float> grad_in(layer_input0.size(), 0.0f);
            for (size_t i = 0; i < grad_in.size(); ++i) {
                const float x = layer_input0[i];
                const float s = 1.0f / (1.0f + std::exp(-x));
                grad_in[i] = grad_out[i] * (s * (1.0f - s));
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "Tanh") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            if (grad_out.size() != layer_input0.size()) {
                std::cerr << "⚠️  Tanh backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            std::vector<float> grad_in(layer_input0.size(), 0.0f);
            for (size_t i = 0; i < grad_in.size(); ++i) {
                const float t = std::tanh(layer_input0[i]);
                grad_in[i] = grad_out[i] * (1.0f - t * t);
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "SiLU") {
            if (inputs.empty()) {
                continue;
            }
            const size_t expected = (input0_size > 0) ? input0_size : inputs[0]->size();
            if (grad_out.size() != expected) {
                std::cerr << "⚠️  SiLU backward shape mismatch (" << layer.name << ")"
                          << " grad=" << grad_out.size() << " expected=" << expected
                          << " out='" << output_name << "' in='" << input_names[0] << "'"
                          << std::endl;
                continue;
            }

            // Les snapshots peuvent être désactivés/vides selon les chemins; on retombe sur le TensorStore.
            const std::vector<float>* xptr = inputs[0];
            if (!xptr || xptr->size() != expected) {
                try {
                    xptr = &getTensor(input_names[0]);
                } catch (...) {
                    xptr = nullptr;
                }
            }
            if (!xptr || xptr->size() != expected) {
                std::cerr << "⚠️  SiLU backward: cannot recover input tensor (" << layer.name << ")"
                          << " x_size=" << (xptr ? xptr->size() : 0ULL)
                          << " expected=" << expected
                          << " in='" << input_names[0] << "'"
                          << std::endl;
                continue;
            }

            std::vector<float> grad_in(expected, 0.0f);
            for (size_t i = 0; i < expected; ++i) {
                const float x = (*xptr)[i];
                const float s = 1.0f / (1.0f + std::exp(-x));
                const float ds = s * (1.0f - s);
                const float dy_dx = s + x * ds;
                grad_in[i] = grad_out[i] * dy_dx;
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "SelfAttention" || layer.type == "MultiHeadAttention") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            const int seq_len = (layer.seq_len > 0) ? layer.seq_len : 1;
            int embed_dim = (layer.embed_dim > 0) ? layer.embed_dim : 0;
            if (embed_dim <= 0) {
                if (seq_len > 0 && (layer_input0.size() % static_cast<size_t>(seq_len)) == 0) {
                    embed_dim = static_cast<int>(layer_input0.size() / static_cast<size_t>(seq_len));
                } else {
                    embed_dim = static_cast<int>(layer_input0.size());
                }
            }
            const int num_heads = std::max(1, layer.num_heads);

            const int qkv_size = embed_dim * embed_dim * 3;
            const int out_size = embed_dim * embed_dim;
            const size_t expected = static_cast<size_t>(qkv_size + out_size);
            if (layer.getWeights() == nullptr || layer.getWeightsSize() < expected) {
                std::cerr << "⚠️  Attention backward skipped: weights invalid (" << layer.name << ")" << std::endl;
                continue;
            }
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }

            const float* Wqkv = layer.getWeights();
            const float* Wout = layer.getWeights() + qkv_size;
            float* dWqkv = layer.grad_weights.data();
            float* dWout = layer.grad_weights.data() + qkv_size;

            std::vector<float> dx;
            if (!backward_self_attention(layer_input0, grad_out, seq_len, embed_dim, num_heads, layer.causal,
                                         Wqkv, Wout, dWqkv, dWout, dx)) {
                std::cerr << "⚠️  Attention backward failed (" << layer.name << ")" << std::endl;
                continue;
            }
            accumulate_grad(grad_store[input_names[0]], dx);

        } else if (layer.type == "CrossAttention") {
            if (inputs.size() < 2) {
                std::cerr << "⚠️  CrossAttention backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& q_in = *inputs[0];
            const std::vector<float>& kv_in = *inputs[1];

            const int num_heads = std::max(1, layer.num_heads);
            int embed_dim = layer.embed_dim;
            if (embed_dim <= 0 && layer.head_dim > 0) {
                embed_dim = layer.head_dim * num_heads;
            }
            if (embed_dim <= 0 || (embed_dim % num_heads) != 0) {
                std::cerr << "⚠️  CrossAttention backward invalid embed_dim (" << layer.name << ")" << std::endl;
                continue;
            }
            if ((q_in.size() % static_cast<size_t>(embed_dim)) != 0 || (kv_in.size() % static_cast<size_t>(embed_dim)) != 0) {
                std::cerr << "⚠️  CrossAttention backward input size mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            const int query_len = static_cast<int>(q_in.size() / static_cast<size_t>(embed_dim));
            const int kv_len = static_cast<int>(kv_in.size() / static_cast<size_t>(embed_dim));
            if (static_cast<int>(grad_out.size()) != query_len * embed_dim) {
                std::cerr << "⚠️  CrossAttention backward grad size mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            const int q_size = embed_dim * embed_dim;
            const int kv_size = embed_dim * (2 * embed_dim);
            const int out_size = embed_dim * embed_dim;
            const size_t expected = static_cast<size_t>(q_size + kv_size + out_size);
            if (layer.getWeights() == nullptr || layer.getWeightsSize() < expected) {
                std::cerr << "⚠️  CrossAttention backward skipped: weights invalid (" << layer.name << ")" << std::endl;
                continue;
            }
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }

            const float* Wq = layer.getWeights();
            const float* Wkv = layer.getWeights() + q_size;
            const float* Wout = layer.getWeights() + q_size + kv_size;

            float* dWq = layer.grad_weights.data();
            float* dWkv = layer.grad_weights.data() + q_size;
            float* dWout = layer.grad_weights.data() + q_size + kv_size;

            std::vector<float> dq;
            std::vector<float> dkv;
            if (!backward_cross_attention(q_in, kv_in, grad_out, query_len, kv_len, embed_dim, num_heads, layer.causal,
                                          Wq, Wkv, Wout, dWq, dWkv, dWout, dq, dkv)) {
                std::cerr << "⚠️  CrossAttention backward failed (" << layer.name << ")" << std::endl;
                continue;
            }
            accumulate_grad(grad_store[input_names[0]], dq);
            accumulate_grad(grad_store[input_names[1]], dkv);

        } else if (layer.type == "MatMul") {
            if (inputs.size() < 2) {
                std::cerr << "⚠️  MatMul backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (M <= 0 || K <= 0 || N <= 0) {
                std::cerr << "⚠️  MatMul backward skipped: dims not configured (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& Bm = *inputs[1];
            if (static_cast<int>(A.size()) != M * K || static_cast<int>(Bm.size()) != K * N || static_cast<int>(grad_out.size()) != M * N) {
                std::cerr << "⚠️  MatMul backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            std::vector<float> dA(static_cast<size_t>(M) * static_cast<size_t>(K), 0.0f);
            std::vector<float> dB(static_cast<size_t>(K) * static_cast<size_t>(N), 0.0f);

            // dA = dC @ B^T
            for (int i = 0; i < M; ++i) {
                for (int k = 0; k < K; ++k) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        sum += grad_out[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] *
                               Bm[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)];
                    }
                    dA[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
                }
            }
            // dB = A^T @ dC
            for (int k = 0; k < K; ++k) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int i = 0; i < M; ++i) {
                        sum += A[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
                               grad_out[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)];
                    }
                    dB[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)] = sum;
                }
            }
            accumulate_grad(grad_store[input_names[0]], dA);
            accumulate_grad(grad_store[input_names[1]], dB);

        } else if (layer.type == "BatchMatMul") {
            if (inputs.size() < 2) {
                std::cerr << "⚠️  BatchMatMul backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const int B = layer.seq_len;
            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (B <= 0 || M <= 0 || K <= 0 || N <= 0) {
                std::cerr << "⚠️  BatchMatMul backward skipped: dims not configured (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& Bm = *inputs[1];
            if (static_cast<int>(A.size()) != B * M * K || static_cast<int>(Bm.size()) != B * K * N || static_cast<int>(grad_out.size()) != B * M * N) {
                std::cerr << "⚠️  BatchMatMul backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            std::vector<float> dA(static_cast<size_t>(B) * static_cast<size_t>(M) * static_cast<size_t>(K), 0.0f);
            std::vector<float> dB(static_cast<size_t>(B) * static_cast<size_t>(K) * static_cast<size_t>(N), 0.0f);
            #pragma omp parallel for schedule(static) if(static_cast<long long>(B) * M * N * K > 262144)
            for (int b = 0; b < B; ++b) {
                const float* Ap = &A[static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(K)];
                const float* Bp = &Bm[static_cast<size_t>(b) * static_cast<size_t>(K) * static_cast<size_t>(N)];
                const float* dCp = &grad_out[static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(N)];
                float* dAp = &dA[static_cast<size_t>(b) * static_cast<size_t>(M) * static_cast<size_t>(K)];
                float* dBp = &dB[static_cast<size_t>(b) * static_cast<size_t>(K) * static_cast<size_t>(N)];

                // dA = dC @ B^T
                for (int i = 0; i < M; ++i) {
                    for (int k = 0; k < K; ++k) {
                        float sum = 0.0f;
                        for (int j = 0; j < N; ++j) {
                            sum += dCp[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] *
                                   Bp[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)];
                        }
                        dAp[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
                    }
                }
                // dB = A^T @ dC
                for (int k = 0; k < K; ++k) {
                    for (int j = 0; j < N; ++j) {
                        float sum = 0.0f;
                        for (int i = 0; i < M; ++i) {
                            sum += Ap[static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k)] *
                                   dCp[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)];
                        }
                        dBp[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(j)] = sum;
                    }
                }
            }
            accumulate_grad(grad_store[input_names[0]], dA);
            accumulate_grad(grad_store[input_names[1]], dB);

        } else if (layer.type == "Bilinear") {
            if (inputs.size() < 2) {
                std::cerr << "⚠️  Bilinear backward skipped: need 2 inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& x1 = *inputs[0];
            const std::vector<float>& x2 = *inputs[1];
            const int in1 = std::max(1, layer.in_features);
            const int in2 = std::max(1, layer.out_features);
            const int out_f = (layer.embed_dim > 0) ? layer.embed_dim : (layer.target_shape.empty() ? 0 : layer.target_shape[0]);
            if (out_f <= 0) {
                std::cerr << "⚠️  Bilinear backward skipped: output dim not configured (" << layer.name << ")" << std::endl;
                continue;
            }
            if ((static_cast<int>(x1.size()) % in1) != 0 || (static_cast<int>(x2.size()) % in2) != 0) {
                std::cerr << "⚠️  Bilinear backward size mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            const int B = static_cast<int>(x1.size()) / in1;
            if (static_cast<int>(x2.size()) / in2 != B || static_cast<int>(grad_out.size()) != B * out_f) {
                std::cerr << "⚠️  Bilinear backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }
            const bool use_bias = layer.use_bias;
            const size_t Wsz = static_cast<size_t>(out_f) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
            const size_t need = Wsz + (use_bias ? static_cast<size_t>(out_f) : 0ULL);
            if (layer.getWeights() == nullptr || layer.getWeightsSize() < need) {
                std::cerr << "⚠️  Bilinear backward skipped: weights invalid (" << layer.name << ")" << std::endl;
                continue;
            }
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }
            const float* W = layer.getWeights();
            float* dW = layer.grad_weights.data();
            float* db = use_bias ? (layer.grad_weights.data() + Wsz) : nullptr;

            std::vector<float> dx1(x1.size(), 0.0f);
            std::vector<float> dx2(x2.size(), 0.0f);

            for (int b = 0; b < B; ++b) {
                const float* arow = &x1[static_cast<size_t>(b) * static_cast<size_t>(in1)];
                const float* brow = &x2[static_cast<size_t>(b) * static_cast<size_t>(in2)];
                const float* go = &grad_out[static_cast<size_t>(b) * static_cast<size_t>(out_f)];
                float* ga = &dx1[static_cast<size_t>(b) * static_cast<size_t>(in1)];
                float* gb = &dx2[static_cast<size_t>(b) * static_cast<size_t>(in2)];
                for (int o = 0; o < out_f; ++o) {
                    const float g = go[static_cast<size_t>(o)];
                    const size_t base_o = static_cast<size_t>(o) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                    if (db) db[static_cast<size_t>(o)] += g;
                    for (int i = 0; i < in1; ++i) {
                        const float ai = arow[static_cast<size_t>(i)];
                        const size_t base_i = base_o + static_cast<size_t>(i) * static_cast<size_t>(in2);
                        for (int j = 0; j < in2; ++j) {
                            const float wv = W[base_i + static_cast<size_t>(j)];
                            dW[base_i + static_cast<size_t>(j)] += g * ai * brow[static_cast<size_t>(j)];
                            ga[static_cast<size_t>(i)] += g * wv * brow[static_cast<size_t>(j)];
                            gb[static_cast<size_t>(j)] += g * wv * ai;
                        }
                    }
                }
            }

            accumulate_grad(grad_store[input_names[0]], dx1);
            accumulate_grad(grad_store[input_names[1]], dx2);

        } else if (layer.type == "EmbeddingBag") {
            // Gradient uniquement sur la table d'embedding
            if (inputs.empty()) {
                std::cerr << "⚠️  EmbeddingBag backward skipped: missing inputs (" << layer.name << ")" << std::endl;
                continue;
            }
            const std::vector<float>& ids_f = *inputs[0];
            const std::vector<float>* offsets_f = (inputs.size() >= 2) ? inputs[1] : nullptr;
            const int vocab = std::max(1, layer.vocab_size);
            const int dim = std::max(1, layer.embed_dim);
            const int pad = layer.padding_idx;
            const size_t w_expected = static_cast<size_t>(vocab) * static_cast<size_t>(dim);
            if (layer.getWeights() == nullptr || layer.getWeightsSize() < w_expected) {
                std::cerr << "⚠️  EmbeddingBag backward skipped: weights invalid (" << layer.name << ")" << std::endl;
                continue;
            }
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }

            std::vector<int> offsets;
            int num_bags = 1;
            if (offsets_f && !offsets_f->empty()) {
                offsets.reserve(offsets_f->size());
                for (float v : *offsets_f) offsets.push_back(static_cast<int>(std::llround(static_cast<double>(v))));
                if (offsets.size() < 2) {
                    std::cerr << "⚠️  EmbeddingBag backward: offsets invalid (" << layer.name << ")" << std::endl;
                    continue;
                }
                num_bags = static_cast<int>(offsets.size()) - 1;
            } else {
                offsets = {0, static_cast<int>(ids_f.size())};
                num_bags = 1;
            }

            if (grad_out.size() != static_cast<size_t>(num_bags) * static_cast<size_t>(dim)) {
                std::cerr << "⚠️  EmbeddingBag backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            for (int b = 0; b < num_bags; ++b) {
                const int start = std::clamp(offsets[static_cast<size_t>(b)], 0, static_cast<int>(ids_f.size()));
                const int end = std::clamp(offsets[static_cast<size_t>(b + 1)], start, static_cast<int>(ids_f.size()));
                const float* g = &grad_out[static_cast<size_t>(b) * static_cast<size_t>(dim)];
                for (int t = start; t < end; ++t) {
                    const int id = static_cast<int>(std::llround(static_cast<double>(ids_f[static_cast<size_t>(t)])));
                    if (pad >= 0 && id == pad) continue;
                    if (id < 0 || id >= vocab) continue;
                    const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        layer.grad_weights[base_w + static_cast<size_t>(d)] += g[static_cast<size_t>(d)];
                    }
                }
            }

        } else if (layer.type == "Conv2d" || layer.type == "ConvTranspose2d") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];

            // Backward ReLU (uniquement si activation appliquée au forward)
            std::vector<float> grad_pre_relu = grad_out;
            if (layer.activation != ActivationType::NONE) {
                const std::vector<uint8_t>* maskp = nullptr;
                if (static_cast<size_t>(layer_idx) < forward_state.layer_output_masks.size()) {
                    const auto& m = forward_state.layer_output_masks[static_cast<size_t>(layer_idx)];
                    if (!m.empty() && m.size() == grad_pre_relu.size()) maskp = &m;
                }
                const std::vector<float>* actp = nullptr;
                if (!maskp) {
                    if (layer_out_snap && layer_out_snap->size() == grad_pre_relu.size()) {
                        actp = layer_out_snap;
                    } else {
                        try {
                            actp = &getTensor(output_name);
                        } catch (...) {
                            actp = nullptr;
                        }
                    }
                }

                #pragma omp simd
                for (size_t i = 0; i < grad_pre_relu.size(); ++i) {
                    const bool active = maskp ? ((*maskp)[i] != 0) : (actp && i < actp->size() && (*actp)[i] > 0.0f);
                    if (!active) grad_pre_relu[i] = 0.0f;
                }
            }
            
            // NOUVEAU: Récupérer les poids et gradients depuis le weight_block
            const float* layer_weights = layer.getWeights();
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.resize(layer.getWeightsSize(), 0.0f);
            }
            
            // Backward Conv : gradients via im2col+GEMM (tuilé) quand possible
            int kernel_size = layer.kernel_size > 0 ? layer.kernel_size : 3;
            int in_channels = layer.in_channels > 0 ? layer.in_channels : 64;
            int out_channels = layer.out_channels > 0 ? layer.out_channels : 64;
            int height = layer.input_height > 0 ? layer.input_height : 64;
            int width = layer.input_width > 0 ? layer.input_width : 64;
            int stride = layer.stride > 0 ? layer.stride : 1;
            int padding = layer.padding;

            // Robustesse: si H/W configurés ne matchent pas la taille réelle de l'input, on tente d'inférer.
            if (in_channels > 0) {
                const size_t ic = static_cast<size_t>(in_channels);
                if (ic > 0 && (layer_input0.size() % ic) == 0) {
                    const size_t hw = layer_input0.size() / ic;
                    const size_t cfg_hw = static_cast<size_t>(std::max(1, height)) * static_cast<size_t>(std::max(1, width));
                    if (cfg_hw != hw) {
                        const int cfg_h = layer.input_height;
                        const int cfg_w = layer.input_width;
                        bool fixed = false;
                        if (cfg_h > 0 && (hw % static_cast<size_t>(cfg_h)) == 0) {
                            height = cfg_h;
                            width = static_cast<int>(hw / static_cast<size_t>(cfg_h));
                            fixed = true;
                        } else if (cfg_w > 0 && (hw % static_cast<size_t>(cfg_w)) == 0) {
                            width = cfg_w;
                            height = static_cast<int>(hw / static_cast<size_t>(cfg_w));
                            fixed = true;
                        }
                        if (!fixed) {
                            const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                            if (s > 0 && s * s == hw) {
                                height = static_cast<int>(s);
                                width = static_cast<int>(s);
                            }
                        }
                    }
                }
            }

            // Dimensions output (cohérentes avec le forward Conv2d)
            const int out_h = (height + 2 * padding - kernel_size) / stride + 1;
            const int out_w = (width + 2 * padding - kernel_size) / stride + 1;
            const int out_spatial = std::max(0, out_h) * std::max(0, out_w);

            // NOTE: pour ConvTranspose2d, on conserve l'ancien chemin (complexe à rewriter ici).
            const bool can_fast_bwd = (layer.type == "Conv2d") && global_use_hardware && hasAVX2() && hasFMA();

            if (can_fast_bwd) {
                const int K = in_channels * kernel_size * kernel_size;
                if (out_h <= 0 || out_w <= 0 || out_spatial <= 0 || K <= 0) {
                    continue;
                }
                const size_t w_need = static_cast<size_t>(out_channels) * static_cast<size_t>(K);
                if (layer.getWeights() == nullptr || layer.getWeightsSize() < w_need) {
                    continue;
                }

                // Assurer buffers grads
                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.resize(layer.getWeightsSize(), 0.0f);
                }

                // dX
                std::vector<float> dX(static_cast<size_t>(in_channels) * static_cast<size_t>(height) * static_cast<size_t>(width), 0.0f);

                // Tuilage sur M=out_spatial
                const size_t target_bytes = 24ULL * 1024ULL * 1024ULL; // un peu plus petit en backward
                const size_t floats_budget = target_bytes / sizeof(float);
                int tile_m = static_cast<int>(std::max<size_t>(256, std::min<size_t>(4096, floats_budget / static_cast<size_t>(K))));
                if (tile_m > out_spatial) tile_m = out_spatial;

                auto xcol_buf = allocator.get_scratchpad(static_cast<size_t>(tile_m) * static_cast<size_t>(K) * sizeof(float), layer.name + "/bwd_im2col");
                auto dy_buf = allocator.get_scratchpad(static_cast<size_t>(tile_m) * static_cast<size_t>(out_channels) * sizeof(float), layer.name + "/bwd_dy");
                auto dyT_buf = allocator.get_scratchpad(static_cast<size_t>(out_channels) * static_cast<size_t>(tile_m) * sizeof(float), layer.name + "/bwd_dyT");
                auto dxcol_buf = allocator.get_scratchpad(static_cast<size_t>(tile_m) * static_cast<size_t>(K) * sizeof(float), layer.name + "/bwd_dxcol");
                auto dw_buf = allocator.get_scratchpad(static_cast<size_t>(out_channels) * static_cast<size_t>(K) * sizeof(float), layer.name + "/bwd_dw_tile");

                float* Xcol = xcol_buf.data();
                float* dY = dy_buf.data();
                float* dYT = dyT_buf.data();
                float* dXcol = dxcol_buf.data();
                float* dWtile = dw_buf.data();

                const float* W = layer_weights; // [out_c x K]

                for (int m0 = 0; m0 < out_spatial; m0 += tile_m) {
                    const int m1 = std::min(out_spatial, m0 + tile_m);
                    const int tm = m1 - m0;

                    // im2col + pack dY (row-major tm x out_c) + pack dY^T (out_c x tm)
                    for (int r = 0; r < tm; ++r) {
                        const int m = m0 + r;
                        const int oh = m / out_w;
                        const int ow = m - oh * out_w;

                        // dY row
                        float* dy_row = dY + static_cast<size_t>(r) * static_cast<size_t>(out_channels);
                        for (int oc = 0; oc < out_channels; ++oc) {
                            const float gv = grad_pre_relu[static_cast<size_t>(oc) * static_cast<size_t>(out_spatial) + static_cast<size_t>(m)];
                            dy_row[static_cast<size_t>(oc)] = gv;
                            // Pack dY^T en layout contigu [out_c x tm] (stride = tm)
                            dYT[static_cast<size_t>(oc) * static_cast<size_t>(tm) + static_cast<size_t>(r)] = gv;
                        }

                        // Xcol row
                        float* x_row = Xcol + static_cast<size_t>(r) * static_cast<size_t>(K);
                        int col = 0;
                        for (int ic = 0; ic < in_channels; ++ic) {
                            const int in_base_c = ic * (height * width);
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                const int ih = oh * stride + kh - padding;
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    const int iw = ow * stride + kw - padding;
                                    float v = 0.0f;
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        const int in_idx = in_base_c + ih * width + iw;
                                        if (in_idx >= 0 && static_cast<size_t>(in_idx) < layer_input0.size()) {
                                            v = layer_input0[static_cast<size_t>(in_idx)];
                                        }
                                    }
                                    x_row[col++] = v;
                                }
                            }
                        }
                    }

                    // dWtile[out_c x K] = dY^T[out_c x tm] @ Xcol[tm x K]
                    HardwareOpt::matmul_fma_saturated(dWtile, dYT, Xcol, static_cast<size_t>(out_channels), static_cast<size_t>(K), static_cast<size_t>(tm));
                    // Accumuler dans grad_weights (layout [out_c x K] contigu)
                    for (size_t i = 0; i < w_need; ++i) {
                        layer.grad_weights[i] += dWtile[i];
                    }

                    // dXcol[tm x K] = dY[tm x out_c] @ W[out_c x K]
                    HardwareOpt::matmul_fma_saturated(dXcol, dY, W, static_cast<size_t>(tm), static_cast<size_t>(K), static_cast<size_t>(out_channels));

                    // col2im accumulate vers dX
                    for (int r = 0; r < tm; ++r) {
                        const int m = m0 + r;
                        const int oh = m / out_w;
                        const int ow = m - oh * out_w;
                        const float* dx_row = dXcol + static_cast<size_t>(r) * static_cast<size_t>(K);

                        int col = 0;
                        for (int ic = 0; ic < in_channels; ++ic) {
                            const int in_base_c = ic * (height * width);
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                const int ih = oh * stride + kh - padding;
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    const int iw = ow * stride + kw - padding;
                                    const float v = dx_row[col++];
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        const int in_idx = in_base_c + ih * width + iw;
                                        if (in_idx >= 0 && static_cast<size_t>(in_idx) < dX.size()) {
                                            dX[static_cast<size_t>(in_idx)] += v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                allocator.return_scratchpad(std::move(dw_buf));
                allocator.return_scratchpad(std::move(dxcol_buf));
                allocator.return_scratchpad(std::move(dyT_buf));
                allocator.return_scratchpad(std::move(dy_buf));
                allocator.return_scratchpad(std::move(xcol_buf));

                accumulate_grad(grad_store[input_names[0]], dX);
                continue;
            }
            
            // Gradient des poids: dL/dW = grad_output ⊗ input (parallélisé sur oc)
            const size_t conv_weight_work = static_cast<size_t>(out_channels) * static_cast<size_t>(in_channels)
                * static_cast<size_t>(kernel_size) * static_cast<size_t>(kernel_size)
                * static_cast<size_t>(height) * static_cast<size_t>(width);
            const size_t conv_input_work = static_cast<size_t>(in_channels)
                * static_cast<size_t>(height) * static_cast<size_t>(width);
            const bool do_parallel = (conv_weight_work >= 262144) || (conv_input_work >= 262144);
            if (do_parallel) {
                #pragma omp parallel
                {
                    // Gradient des poids: dL/dW = grad_output ⊗ input
                    #pragma omp for schedule(static) collapse(2) nowait
                    for (int oc = 0; oc < out_channels; ++oc) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    float grad_weight = 0.0f;

                                    for (int oh = 0; oh < out_h; ++oh) {
                                        for (int ow = 0; ow < out_w; ++ow) {
                                            int ih = oh * stride + kh - padding;
                                            int iw = ow * stride + kw - padding;

                                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                                int out_idx = oc * out_spatial + oh * out_w + ow;
                                                int in_idx = ic * (height * width) + ih * width + iw;

                                                if (out_idx < static_cast<int>(grad_pre_relu.size()) &&
                                                    in_idx < static_cast<int>(layer_input0.size())) {
                                                    grad_weight += grad_pre_relu[out_idx] * layer_input0[static_cast<size_t>(in_idx)];
                                                }
                                            }
                                        }
                                    }

                                    int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                    if (w_idx < static_cast<int>(layer.grad_weights.size())) {
                                        layer.grad_weights[w_idx] += grad_weight;
                                    }
                                }
                            }
                        }
                    }

                    // Gradient de l'entrée: convolution transposée de grad avec poids flip
                    #pragma omp for schedule(static) collapse(3)
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ih = 0; ih < height; ++ih) {
                            for (int iw = 0; iw < width; ++iw) {
                                float grad_sum = 0.0f;

                                for (int oc = 0; oc < out_channels; ++oc) {
                                    for (int kh = 0; kh < kernel_size; ++kh) {
                                        for (int kw = 0; kw < kernel_size; ++kw) {
                                            int oh = ih - kh + padding;
                                            int ow = iw - kw + padding;

                                            if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w &&
                                                oh % stride == 0 && ow % stride == 0) {
                                                oh /= stride;
                                                ow /= stride;

                                                int out_idx = oc * out_spatial + oh * out_w + ow;
                                                int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                                                if (out_idx < static_cast<int>(grad_pre_relu.size()) &&
                                                    w_idx < static_cast<int>(layer.getWeightsSize())) {
                                                    float weight = layer_weights[w_idx];
                                                    grad_sum += grad_pre_relu[out_idx] * weight;
                                                }
                                            }
                                        }
                                    }
                                }

                                int in_idx = ic * (height * width) + ih * width + iw;
                                if (in_idx < static_cast<int>(grad_input0.size())) {
                                    grad_input0[static_cast<size_t>(in_idx)] = grad_sum;
                                }
                            }
                        }
                    }
                }
            } else {
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                float grad_weight = 0.0f;

                                for (int oh = 0; oh < out_h; ++oh) {
                                    for (int ow = 0; ow < out_w; ++ow) {
                                        int ih = oh * stride + kh - padding;
                                        int iw = ow * stride + kw - padding;

                                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                            int out_idx = oc * out_spatial + oh * out_w + ow;
                                            int in_idx = ic * (height * width) + ih * width + iw;

                                            if (out_idx < static_cast<int>(grad_pre_relu.size()) &&
                                                in_idx < static_cast<int>(layer_input0.size())) {
                                                grad_weight += grad_pre_relu[out_idx] * layer_input0[static_cast<size_t>(in_idx)];
                                            }
                                        }
                                    }
                                }

                                int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                if (w_idx < static_cast<int>(layer.grad_weights.size())) {
                                    layer.grad_weights[w_idx] += grad_weight;
                                }
                            }
                        }
                    }
                }

                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int ih = 0; ih < height; ++ih) {
                        for (int iw = 0; iw < width; ++iw) {
                            float grad_sum = 0.0f;

                            for (int oc = 0; oc < out_channels; ++oc) {
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int oh = ih - kh + padding;
                                        int ow = iw - kw + padding;

                                        if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w &&
                                            oh % stride == 0 && ow % stride == 0) {
                                            oh /= stride;
                                            ow /= stride;

                                            int out_idx = oc * out_spatial + oh * out_w + ow;
                                            int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                                            if (out_idx < static_cast<int>(grad_pre_relu.size()) &&
                                                w_idx < static_cast<int>(layer.getWeightsSize())) {
                                                float weight = layer_weights[w_idx];
                                                grad_sum += grad_pre_relu[out_idx] * weight;
                                            }
                                        }
                                    }
                                }
                            }

                            int in_idx = ic * (height * width) + ih * width + iw;
                            if (in_idx < static_cast<int>(grad_input0.size())) {
                                grad_input0[static_cast<size_t>(in_idx)] = grad_sum;
                            }
                        }
                    }
                }
            }

            accumulate_grad(grad_store[input_names[0]], grad_input0);
            
        } else if (layer.type == "BatchNorm2d") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            // NOUVEAU: Récupérer les poids et gradients depuis le weight_block
            const float* layer_weights = layer.getWeights();
            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.resize(layer.getWeightsSize(), 0.0f);
            }
            
            // Backward BatchNorm avec formule compacte standard (CORRIGÉ)
            int channels = 64; // À adapter
            int height = 64, width = 64;
            int spatial_size = height * width;
            const float eps = 1e-5f;

            for (int c = 0; c < channels; ++c) {
                // Calculer mean et variance pour ce canal (forward)
                float mean = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(layer_input0.size())) {
                            mean += layer_input0[static_cast<size_t>(idx)];
                        }
                    }
                }
                mean /= spatial_size;
                
                float var = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(layer_input0.size())) {
                            float diff = layer_input0[static_cast<size_t>(idx)] - mean;
                            var += diff * diff;
                        }
                    }
                }
                var /= spatial_size;
                float invstd = 1.0f / std::sqrt(var + eps);
                
                // Récupérer gamma (scale parameter) depuis weight_block
                float gamma = 1.0f;
                if (c * 2 < static_cast<int>(layer.getWeightsSize())) {
                    gamma = layer_weights[c * 2];
                }
                
                // Gradient gamma: sum(dY * x_normalized)
                float grad_gamma = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad_out.size()) && 
                            idx < static_cast<int>(layer_input0.size())) {
                            float x_normalized = (layer_input0[static_cast<size_t>(idx)] - mean) * invstd;
                            grad_gamma += grad_out[static_cast<size_t>(idx)] * x_normalized;
                        }
                    }
                }
                if (c * 2 < static_cast<int>(layer.grad_weights.size())) {
                    layer.grad_weights[c * 2] += grad_gamma;
                }
                
                // Gradient beta: sum(dY)
                float grad_beta = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad_out.size())) {
                            grad_beta += grad_out[static_cast<size_t>(idx)];
                        }
                    }
                }
                if (c * 2 + 1 < static_cast<int>(layer.grad_weights.size())) {
                    layer.grad_weights[c * 2 + 1] += grad_beta;
                }
                
                // Calculer sum(dY) et sum(dY * (x - mean))
                float sum_dy = grad_beta; // Déjà calculé
                float sum_dy_xmu = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad_out.size()) && 
                            idx < static_cast<int>(layer_input0.size())) {
                            sum_dy_xmu += grad_out[static_cast<size_t>(idx)] * (layer_input0[static_cast<size_t>(idx)] - mean);
                        }
                    }
                }
                
                // Gradient input avec formule compacte standard (CORRIGÉ)
                // dx = (1/N) * gamma * invstd * (N*dY - sum(dY) - (x-mean)*invstd^2*sum(dY*(x-mean)))
                float invstd2 = invstd * invstd;
                float inv_N = 1.0f / spatial_size;
                
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = c * spatial_size + h * width + w;
                        if (idx < static_cast<int>(grad_input0.size()) &&
                            idx < static_cast<int>(layer_input0.size())) {
                            float x_mu = layer_input0[static_cast<size_t>(idx)] - mean;
                            float dx = inv_N * gamma * invstd * (
                                spatial_size * grad_out[static_cast<size_t>(idx)] 
                                - sum_dy 
                                - x_mu * invstd2 * sum_dy_xmu
                            );
                            grad_input0[static_cast<size_t>(idx)] = dx;
                        }
                    }
                }
            }

            accumulate_grad(grad_store[input_names[0]], grad_input0);

        } else if (layer.type == "Linear") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            // Backward Linear: y = W x + b
            const int out_f = layer.out_features;
            const int seq_len = layer.seq_len > 0 ? layer.seq_len : 1;
            const bool has_seq = (layer.seq_len > 0);

            int in_f = layer.in_features;
            if (in_f <= 0) {
                if (has_seq && seq_len > 0 && static_cast<int>(layer_input0.size()) % seq_len == 0) {
                    in_f = static_cast<int>(layer_input0.size()) / seq_len;
                } else {
                    in_f = static_cast<int>(layer_input0.size());
                }
            }

            if (out_f <= 0) {
                std::cerr << "⚠️  Linear backward skipped: out_features not set (" << layer.name << ")" << std::endl;
                continue;
            }

            const bool seq_mode = has_seq &&
                                  (static_cast<int>(layer_input0.size()) == seq_len * in_f) &&
                                  (static_cast<int>(grad_out.size()) == seq_len * out_f);
            const bool vec_mode = (!has_seq) &&
                                  (static_cast<int>(layer_input0.size()) == in_f) &&
                                  (static_cast<int>(grad_out.size()) == out_f);

            if (!seq_mode && !vec_mode) {
                const int expected_in = has_seq ? (seq_len * in_f) : in_f;
                const int expected_out = has_seq ? (seq_len * out_f) : out_f;
                std::cerr << "⚠️  Linear backward shape mismatch (" << layer.name << ")"
                          << " in=" << layer_input0.size() << " expected_in=" << expected_in
                          << " grad=" << grad_out.size() << " expected_out=" << expected_out << std::endl;
                continue;
            }

            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }

            const float* W = layer.getWeights();
            const bool use_bias = layer.use_bias;
            const size_t w_count = static_cast<size_t>(in_f) * static_cast<size_t>(out_f);

            if (seq_mode) {
                // Mode séquence: [seq_len, in_f] -> [seq_len, out_f]
                std::vector<float> grad_input(static_cast<size_t>(seq_len) * static_cast<size_t>(in_f), 0.0f);

                for (int t = 0; t < seq_len; ++t) {
                    const float* x = &layer_input0[static_cast<size_t>(t) * static_cast<size_t>(in_f)];
                    const float* go = &grad_out[static_cast<size_t>(t) * static_cast<size_t>(out_f)];
                    float* gi = &grad_input[static_cast<size_t>(t) * static_cast<size_t>(in_f)];

                    // dW/db (accumule sur t)
                    for (int o = 0; o < out_f; ++o) {
                        const float g = go[static_cast<size_t>(o)];
                        const size_t row_off = static_cast<size_t>(o) * static_cast<size_t>(in_f);
                        for (int i = 0; i < in_f; ++i) {
                            layer.grad_weights[row_off + static_cast<size_t>(i)] += g * x[static_cast<size_t>(i)];
                        }
                        if (use_bias && layer.grad_weights.size() >= w_count + static_cast<size_t>(out_f)) {
                            layer.grad_weights[w_count + static_cast<size_t>(o)] += g;
                        }
                    }

                    // grad_input[t] = W^T * grad_out[t]
                    for (int o = 0; o < out_f; ++o) {
                        const float g = go[static_cast<size_t>(o)];
                        const float* wrow = W + static_cast<size_t>(o) * static_cast<size_t>(in_f);
                        for (int i = 0; i < in_f; ++i) {
                            gi[static_cast<size_t>(i)] += wrow[static_cast<size_t>(i)] * g;
                        }
                    }
                }

                accumulate_grad(grad_store[input_names[0]], grad_input);
            } else {
                // Fallback historique: vector -> vector

                // dW += grad_out ⊗ x
                for (int o = 0; o < out_f; ++o) {
                    const float go = grad_out[static_cast<size_t>(o)];
                    const size_t row_off = static_cast<size_t>(o) * static_cast<size_t>(in_f);
                    for (int i = 0; i < in_f; ++i) {
                        layer.grad_weights[row_off + static_cast<size_t>(i)] += go * layer_input0[static_cast<size_t>(i)];
                    }
                }

                // db += grad_out
                if (use_bias && layer.grad_weights.size() >= w_count + static_cast<size_t>(out_f)) {
                    for (int o = 0; o < out_f; ++o) {
                        layer.grad_weights[w_count + static_cast<size_t>(o)] += grad_out[static_cast<size_t>(o)];
                    }
                }

                // grad_input = W^T * grad_out
                std::vector<float> grad_input(static_cast<size_t>(in_f), 0.0f);
                for (int o = 0; o < out_f; ++o) {
                    const float go = grad_out[static_cast<size_t>(o)];
                    const float* wrow = W + static_cast<size_t>(o) * static_cast<size_t>(in_f);
                    for (int i = 0; i < in_f; ++i) {
                        grad_input[static_cast<size_t>(i)] += wrow[static_cast<size_t>(i)] * go;
                    }
                }

                accumulate_grad(grad_store[input_names[0]], grad_input);
            }

        } else if (layer.type == "PatchEmbed") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            const int d_model = layer.embed_dim > 0 ? layer.embed_dim : layer.out_features;
            const int seq_text = std::max(1, layer.seq_text);
            const int num_patches = std::max(1, layer.num_patches);
            const int patch_dim = std::max(1, layer.patch_dim);

            const int text_dim = (seq_text + 1) * d_model;
            const int in_dim = text_dim + num_patches * patch_dim;
            const int out_dim = (seq_text + 1 + num_patches) * d_model;

            if (static_cast<int>(layer_input0.size()) != in_dim || static_cast<int>(grad_out.size()) != out_dim) {
                std::cerr << "⚠️  PatchEmbed backward shape mismatch (" << layer.name << ")"
                          << " in=" << layer_input0.size() << " expected_in=" << in_dim
                          << " grad=" << grad_out.size() << " expected_out=" << out_dim << std::endl;
                continue;
            }
            if (layer.getWeights() == nullptr || static_cast<int>(layer.getWeightsSize()) != (patch_dim * d_model + d_model)) {
                std::cerr << "⚠️  PatchEmbed backward skipped: weights not initialized/invalid (" << layer.name << ")" << std::endl;
                continue;
            }

            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }

            const float* w = layer.getWeights();
            const float inv = 1.0f / std::sqrt(static_cast<float>(patch_dim));

            std::vector<float> grad_in(static_cast<size_t>(in_dim), 0.0f);

            // Texte + time: identité
            for (int i = 0; i < text_dim; ++i) {
                grad_in[static_cast<size_t>(i)] = grad_out[static_cast<size_t>(i)];
            }

            // Patches: projection learnable
            // Weights layout: W[patch_dim, d_model] puis b[d_model]
            float* gradW = layer.grad_weights.data();
            float* gradB = layer.grad_weights.data() + patch_dim * d_model;

            for (int p = 0; p < num_patches; ++p) {
                const int in_off = text_dim + p * patch_dim;
                const int out_off = (seq_text + 1 + p) * d_model;

                // db += grad_out_patch
                for (int d = 0; d < d_model; ++d) {
                    gradB[d] += grad_out[static_cast<size_t>(out_off + d)];
                }

                // dW += (x_patch*inv) outer grad_out_patch
                for (int k = 0; k < patch_dim; ++k) {
                    const float xk = layer_input0[static_cast<size_t>(in_off + k)] * inv;
                    const size_t row = static_cast<size_t>(k) * static_cast<size_t>(d_model);
                    for (int d = 0; d < d_model; ++d) {
                        gradW[row + static_cast<size_t>(d)] += xk * grad_out[static_cast<size_t>(out_off + d)];
                    }
                }

                // grad_in_patch = inv * W * grad_out_patch
                for (int k = 0; k < patch_dim; ++k) {
                    float acc = 0.0f;
                    const size_t row = static_cast<size_t>(k) * static_cast<size_t>(d_model);
                    for (int d = 0; d < d_model; ++d) {
                        acc += w[row + static_cast<size_t>(d)] * grad_out[static_cast<size_t>(out_off + d)];
                    }
                    grad_in[static_cast<size_t>(in_off + k)] += acc * inv;
                }
            }

            accumulate_grad(grad_store[input_names[0]], grad_in);

        } else if (layer.type == "GELU") {
            // Backward GELU (approx tanh), en utilisant layer_input comme pré-activation
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            if (grad_out.size() != layer_input0.size()) {
                std::cerr << "⚠️  GELU backward shape mismatch (" << layer.name << ")" << std::endl;
                continue;
            }

            std::vector<float> grad_input;
            grad_input.resize(layer_input0.size());
            for (size_t i = 0; i < grad_input.size(); ++i) {
                const float x = layer_input0[i];
                const float c = 0.044715f;
                const float s = std::sqrt(2.0f / 3.14159265359f);
                const float u = s * (x + c * x * x * x);
                const float t = std::tanh(u);
                const float sech2 = 1.0f - t * t;
                const float du_dx = s * (1.0f + 3.0f * c * x * x);
                const float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
                grad_input[i] = grad_out[i] * dgelu;
            }

            accumulate_grad(grad_store[input_names[0]], grad_input);
            
        } else if (layer.type == "MaxPool2d") {
            if (inputs.empty()) {
                continue;
            }
            const std::vector<float>& layer_input0 = *inputs[0];
            // Backward MaxPool : propager le gradient au max (parallélisé)
            std::vector<float> grad_input(layer_input0.size(), 0.0f);
            #pragma omp simd
            for (size_t i = 0; i < grad_out.size(); ++i) {
                size_t idx1 = i * 2;
                size_t idx2 = i * 2 + 1;
                
                if (idx1 < layer_input0.size() && idx2 < layer_input0.size()) {
                    if (layer_input0[idx1] >= layer_input0[idx2]) {
                        grad_input[idx1] = grad_out[i];
                    } else {
                        grad_input[idx2] = grad_out[i];
                    }
                }
            }

            accumulate_grad(grad_store[input_names[0]], grad_input);
        } else if (layer.type == "GlobalAvgPool2d" || layer.type == "AdaptiveAvgPool2d") {
            if (inputs.empty()) {
                continue;
            }

            const int in_channels = (layer.in_channels > 0) ? layer.in_channels : static_cast<int>(grad_out.size());
            int height = (layer.input_height > 0) ? layer.input_height : 0;
            int width = (layer.input_width > 0) ? layer.input_width : 0;

            if (in_channels <= 0) {
                std::cerr << "⚠️  GlobalAvgPool2d backward: missing channels (" << layer.name << ")" << std::endl;
                continue;
            }
            if (grad_out.size() != static_cast<size_t>(in_channels)) {
                std::cerr << "⚠️  GlobalAvgPool2d backward shape mismatch (" << layer.name << ")"
                          << " grad=" << grad_out.size() << " expected=" << in_channels
                          << std::endl;
                continue;
            }

            // Use real input size as hint to infer H/W if needed.
            size_t store_in_size = 0ULL;
            bool have_store_in = false;
            try {
                store_in_size = getTensor(input_names[0]).size();
                have_store_in = true;
            } catch (...) {
                have_store_in = false;
            }
            const size_t in_size_hint = have_store_in ? store_in_size : input0_size;

            // Infer H/W if missing or inconsistent.
            if (height <= 0 || width <= 0) {
                const size_t c = static_cast<size_t>(in_channels);
                if (c > 0 && in_size_hint > 0ULL && (in_size_hint % c) == 0ULL) {
                    const size_t hw = in_size_hint / c;
                    const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                    if (s > 0 && s * s == hw) {
                        height = static_cast<int>(s);
                        width = static_cast<int>(s);
                    }
                }
            }

            if (height <= 0 || width <= 0) {
                std::cerr << "⚠️  GlobalAvgPool2d backward: missing H/W (" << layer.name << ")" << std::endl;
                continue;
            }

            const size_t spatial = static_cast<size_t>(height) * static_cast<size_t>(width);
            const size_t expected_in = static_cast<size_t>(in_channels) * spatial;
            if (in_size_hint != 0ULL && in_size_hint != expected_in) {
                std::cerr << "⚠️  GlobalAvgPool2d backward input mismatch (" << layer.name << ")"
                          << " in=" << in_size_hint << " expected=" << expected_in
                          << " (C=" << in_channels << " H=" << height << " W=" << width << ")"
                          << std::endl;
                continue;
            }

            std::vector<float> grad_in = LayerOps::global_avgpool2d_backward(grad_out, layer, expected_in);
            if (grad_in.empty()) {
                std::cerr << "⚠️  GlobalAvgPool2d backward failed (" << layer.name << ")" << std::endl;
                continue;
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);
        } else if (layer.type == "UpsampleNearest") {
            // Backward UpsampleNearest: accumulation vers le pixel source (nearest)
            if (layer.out_h <= 0 || layer.out_w <= 0 || layer.in_channels <= 0) {
                std::cerr << "⚠️  UpsampleNearest backward: missing dims (" << layer.name << ")" << std::endl;
                continue;
            }

            const int in_h = layer.out_h;
            const int in_w = layer.out_w;
            const int channels = layer.in_channels;
            const int scale_h = (layer.scale_h > 0) ? layer.scale_h : 2;
            const int scale_w = (layer.scale_w > 0) ? layer.scale_w : 2;
            const int out_h = in_h * scale_h;
            const int out_w = in_w * scale_w;

            const size_t expected_in = static_cast<size_t>(channels) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
            const size_t expected_out = static_cast<size_t>(channels) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w);

            // Les tailles snapshotées peuvent être incohérentes si des métadonnées ont été inférées tard.
            // On retombe sur la taille réelle stockée dans le TensorStore.
            size_t store_in_size = 0ULL;
            bool have_store_in = false;
            try {
                store_in_size = getTensor(input_names[0]).size();
                have_store_in = true;
            } catch (...) {
                have_store_in = false;
            }

            const bool grad_ok = (grad_out.size() == expected_out);
            const bool in_ok_snapshot = (input0_size == expected_in);
            const bool in_ok_store = (have_store_in && store_in_size == expected_in);

            if (!grad_ok || (!in_ok_snapshot && !in_ok_store)) {
                std::cerr << "⚠️  UpsampleNearest backward shape mismatch (" << layer.name << ")"
                          << " in=" << input0_size << " expected_in=" << expected_in
                          << " store_in=" << (have_store_in ? store_in_size : 0ULL)
                          << " grad=" << grad_out.size() << " expected_out=" << expected_out
                          << " out='" << output_name << "' in='" << input_names[0] << "'"
                          << std::endl;
                continue;
            }

            std::vector<float> grad_in(expected_in, 0.0f);
            for (int c = 0; c < channels; ++c) {
                const size_t in_plane = static_cast<size_t>(c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
                const size_t out_plane = static_cast<size_t>(c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
                for (int oh = 0; oh < out_h; ++oh) {
                    const int ih = oh / scale_h;
                    for (int ow = 0; ow < out_w; ++ow) {
                        const int iw = ow / scale_w;
                        const size_t in_idx = in_plane + static_cast<size_t>(ih) * static_cast<size_t>(in_w) + static_cast<size_t>(iw);
                        const size_t out_idx = out_plane + static_cast<size_t>(oh) * static_cast<size_t>(out_w) + static_cast<size_t>(ow);
                        grad_in[in_idx] += grad_out[out_idx];
                    }
                }
            }
            accumulate_grad(grad_store[input_names[0]], grad_in);
        } else {
            // Fallback: si non géré, on essaie au moins de propager sur l'input principal
            if (grad_out.size() == input0_size) {
                accumulate_grad(grad_store[input_names[0]], grad_out);
            } else {
                std::cerr << "⚠️  Backward not implemented for layer type '" << layer.type
                          << "' (" << layer.name << ")" << std::endl;
            }
        }
    }

    // Exposer le gradient d'entrée si présent (architecture utilisant "__input__")
    has_last_input_gradient_ = false;
    last_input_gradient_.clear();
    auto it_in = grad_store.find("__input__");
    if (it_in != grad_store.end()) {
        last_input_gradient_ = it_in->second;
        has_last_input_gradient_ = true;
    }

    return grads;
}

float Model::computeLoss(const std::vector<float> &prediction, 
                        const std::vector<float> &target, 
                        const std::string &loss_type) {
    if (prediction.size() != target.size()) {
        std::cerr << "⚠️  Prediction and target size mismatch" << std::endl;
        return 0.0f;
    }
    
    float loss = 0.0f;
    
    if (loss_type == "mse") {
        // Mean Squared Error
        for (size_t i = 0; i < prediction.size(); ++i) {
            float diff = prediction[i] - target[i];
            loss += diff * diff;
        }
        loss /= prediction.size();
        
    } else if (loss_type == "mae") {
        // Mean Absolute Error
        for (size_t i = 0; i < prediction.size(); ++i) {
            loss += std::abs(prediction[i] - target[i]);
        }
        loss /= prediction.size();
        
    } else if (loss_type == "bce") {
        // Binary Cross Entropy
        for (size_t i = 0; i < prediction.size(); ++i) {
            float p = std::clamp(prediction[i], 1e-7f, 1.0f - 1e-7f);
            float t = target[i];
            loss += -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        }
        loss /= prediction.size();
    } else if (loss_type == "huber" || loss_type == "smoothl1") {
        // Huber / SmoothL1 (delta=1)
        const float delta = 1.0f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            const float diff = prediction[i] - target[i];
            const float ad = std::abs(diff);
            if (ad <= delta) {
                loss += 0.5f * diff * diff;
            } else {
                loss += delta * (ad - 0.5f * delta);
            }
        }
        loss /= prediction.size();
    } else if (loss_type == "charbonnier") {
        // Charbonnier (eps=1e-3)
        const float eps = 1e-3f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            const float diff = prediction[i] - target[i];
            loss += std::sqrt(diff * diff + eps * eps);
        }
        loss /= prediction.size();
    } else if (loss_type == "gaussian_nll" || loss_type == "nll_gaussian") {
        // Gaussian NLL with fixed sigma=1 (equivalent to scaled MSE + const)
        const float sigma = 1.0f;
        const float inv_var = 1.0f / (sigma * sigma);
        const float log_var = std::log(sigma * sigma);
        for (size_t i = 0; i < prediction.size(); ++i) {
            const float diff = prediction[i] - target[i];
            loss += 0.5f * (diff * diff * inv_var + log_var);
        }
        loss /= prediction.size();
    }
    
    return loss;
}

std::vector<float> Model::computeLossGradient(const std::vector<float> &prediction,
                                              const std::vector<float> &target,
                                              const std::string &loss_type) {
    std::vector<float> gradient;
    computeLossGradientInto(prediction, target, gradient, loss_type);
    return gradient;
}

void Model::computeLossGradientInto(const std::vector<float> &prediction,
                                    const std::vector<float> &target,
                                    std::vector<float> &gradient,
                                    const std::string &loss_type) {
    gradient.resize(prediction.size());
    if (prediction.size() != target.size()) {
        std::fill(gradient.begin(), gradient.end(), 0.0f);
        return;
    }

    if (loss_type == "mse") {
        // Gradient MSE: 2(pred - target) / n avec AVX2
        const size_t size = prediction.size();
        const float scale = (size > 0) ? (2.0f / static_cast<float>(size)) : 0.0f;
        size_t i = 0;

#ifdef __AVX2__
        __m256 scale_vec = _mm256_set1_ps(scale);
        for (; i + 8 <= size; i += 8) {
            __m256 pred = _mm256_loadu_ps(&prediction[i]);
            __m256 tgt = _mm256_loadu_ps(&target[i]);
            __m256 diff = _mm256_sub_ps(pred, tgt);
            __m256 grad = _mm256_mul_ps(diff, scale_vec);
            _mm256_storeu_ps(&gradient[i], grad);
        }
#endif

        for (; i < size; ++i) {
            gradient[i] = scale * (prediction[i] - target[i]);
        }
    } else if (loss_type == "mae") {
        // Gradient MAE: sign(pred - target) / n
        const float inv_n = prediction.empty() ? 0.0f : (1.0f / static_cast<float>(prediction.size()));
        for (size_t i = 0; i < prediction.size(); ++i) {
            const float diff = prediction[i] - target[i];
            if (diff > 0.0f) {
                gradient[i] = inv_n;
            } else if (diff < 0.0f) {
                gradient[i] = -inv_n;
            } else {
                gradient[i] = 0.0f;
            }
        }
    } else if (loss_type == "bce") {
        // Gradient BCE avec AVX2
        const size_t size = prediction.size();
        const float inv_size = (size > 0) ? (1.0f / static_cast<float>(size)) : 0.0f;
        size_t i = 0;

#ifdef __AVX2__
        __m256 eps = _mm256_set1_ps(1e-7f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 one_minus_eps = _mm256_set1_ps(1.0f - 1e-7f);
        __m256 inv_size_vec = _mm256_set1_ps(inv_size);

        for (; i + 8 <= size; i += 8) {
            __m256 p = _mm256_loadu_ps(&prediction[i]);
            __m256 t = _mm256_loadu_ps(&target[i]);

            // Clamp p to [1e-7, 1-1e-7]
            p = _mm256_max_ps(p, eps);
            p = _mm256_min_ps(p, one_minus_eps);

            // grad = (p - t) / (p * (1 - p)) / size
            __m256 diff = _mm256_sub_ps(p, t);
            __m256 one_minus_p = _mm256_sub_ps(one, p);
            __m256 denom = _mm256_mul_ps(p, one_minus_p);
            __m256 grad = _mm256_div_ps(diff, denom);
            grad = _mm256_mul_ps(grad, inv_size_vec);

            _mm256_storeu_ps(&gradient[i], grad);
        }
#endif

        for (; i < size; ++i) {
            const float p = std::clamp(prediction[i], 1e-7f, 1.0f - 1e-7f);
            const float t = target[i];
            gradient[i] = (p - t) / (p * (1.0f - p)) * inv_size;
        }
    } else if (loss_type == "huber" || loss_type == "smoothl1") {
        const float delta = 1.0f;
        const float inv_n = prediction.empty() ? 0.0f : (1.0f / static_cast<float>(prediction.size()));
        for (size_t i = 0; i < prediction.size(); ++i) {
            const float diff = prediction[i] - target[i];
            const float ad = std::abs(diff);
            if (ad <= delta) {
                gradient[i] = inv_n * diff;
            } else {
                const float s = (diff > 0.0f) ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                gradient[i] = inv_n * delta * s;
            }
        }
    } else if (loss_type == "charbonnier") {
        const float eps = 1e-3f;
        const float inv_n = prediction.empty() ? 0.0f : (1.0f / static_cast<float>(prediction.size()));
        for (size_t i = 0; i < prediction.size(); ++i) {
            const float diff = prediction[i] - target[i];
            const float denom = std::sqrt(diff * diff + eps * eps);
            gradient[i] = (denom > 0.0f) ? (inv_n * diff / denom) : 0.0f;
        }
    } else if (loss_type == "gaussian_nll" || loss_type == "nll_gaussian") {
        const float sigma = 1.0f;
        const float inv_var = 1.0f / (sigma * sigma);
        const float inv_n = prediction.empty() ? 0.0f : (1.0f / static_cast<float>(prediction.size()));
        for (size_t i = 0; i < prediction.size(); ++i) {
            gradient[i] = inv_n * (prediction[i] - target[i]) * inv_var;
        }
    } else {
        std::fill(gradient.begin(), gradient.end(), 0.0f);
    }
}

// === build & autoBuildFromDataset ===

void Model::build()
{
    // Construction générique du modèle
    // Peut être surchargée pour définir une architecture spécifique
    
    // Exemple: backbone U-Net simple
    buildBackboneUNet(4, 2, 3);  // 4 stages, 2 blocs par stage, 3 blocs bottleneck
    
    std::cout << "Model::build() - Architecture construite" << std::endl;
    std::cout << "  Couches: " << layers.size() << std::endl;
    std::cout << "  Paramètres totaux: " << totalParamCount() << std::endl;
    
    // Allocation automatique des paramètres
    size_t total = totalParamCount();
    if (total > 0) {
        std::cout << "  Allocation des paramètres..." << std::endl;
        allocateParams();
        std::cout << "  ✓ " << layer_weight_blocks.size() << " blocs de poids alloués" << std::endl;
        
        // Initialisation automatique des poids (méthode He par défaut)
        std::cout << "  Initialisation des poids (He)..." << std::endl;
        initializeWeights("he", 0);
        std::cout << "  ✓ Poids initialisés" << std::endl;
    }
}

void Model::autoBuildFromDataset(const std::string &dataset_dir)
{
    // Analyse automatique du dataset pour construire l'architecture appropriée
    
    std::cout << "Model::autoBuildFromDataset(" << dataset_dir << ")" << std::endl;
    
    // Charger le dataset avec cache et validation flexible (min 1 modalité)
    std::vector<DatasetItem> items;
    try {
        items = loadDatasetCached(dataset_dir, 64, 64, 1);  // min_modalities = 1
    } catch (const std::exception &e) {
        std::cerr << "Erreur chargement dataset: " << e.what() << std::endl;
        // Fallback: construction par défaut
        build();
        return;
    }
    
    if (items.empty()) {
        std::cerr << "Dataset vide, construction par défaut" << std::endl;
        build();
        return;
    }
    
    std::cout << "  Items trouvés: " << items.size() << std::endl;
    
    // Analyser les modalités présentes et les linkables
    bool has_text = false;
    bool has_image = false;
    bool has_audio = false;
    bool has_video = false;
    size_t linkable_count = 0;
    
    for (const auto &item : items) {
        if (!item.text_file.empty()) has_text = true;
        if (!item.image_file.empty()) has_image = true;
        if (!item.audio_file.empty()) has_audio = true;
        if (!item.video_file.empty()) has_video = true;
        if (item.is_linked && item.countModalities() >= 2) linkable_count++;
    }
    
    std::cout << "  Modalités détectées:" << std::endl;
    std::cout << "    - Texte:  " << (has_text ? "✓" : "✗") << std::endl;
    std::cout << "    - Image:  " << (has_image ? "✓" : "✗") << std::endl;
    std::cout << "    - Audio:  " << (has_audio ? "✓" : "✗") << std::endl;
    std::cout << "    - Vidéo:  " << (has_video ? "✓" : "✗") << std::endl;
    std::cout << "    - Linkables validés: " << linkable_count << std::endl;
    
    // Construire le backbone de base
    buildBackboneUNet(4, 2, 3);
    
    // Créer les magic tokens pour chaque modalité détectée
    std::vector<MagicToken> magic_tokens;
    
    if (has_text) {
        MagicToken tok;
        tok.modality_mask = 0x01;  // bit 0 = text
        tok.seed = 42;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.1f * (i + 1);
        magic_tokens.push_back(tok);
        buildTextBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche texte ajoutée" << std::endl;
    }
    
    if (has_image) {
        MagicToken tok;
        tok.modality_mask = 0x02;  // bit 1 = image
        tok.seed = 43;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.2f * (i + 1);
        magic_tokens.push_back(tok);
        buildImageBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche image ajoutée" << std::endl;
    }
    
    if (has_audio) {
        MagicToken tok;
        tok.modality_mask = 0x04;  // bit 2 = audio
        tok.seed = 44;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.3f * (i + 1);
        magic_tokens.push_back(tok);
        buildAudioBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche audio ajoutée" << std::endl;
    }
    
    if (has_video) {
        MagicToken tok;
        tok.modality_mask = 0x08;  // bit 3 = video
        tok.seed = 45;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.4f * (i + 1);
        magic_tokens.push_back(tok);
        buildVideoBranch(tok);
        injectMagicToken(tok);
        std::cout << "  → Branche vidéo ajoutée" << std::endl;
    }
    
    std::cout << "  Architecture auto-construite:" << std::endl;
    std::cout << "    - Couches: " << layers.size() << std::endl;
    std::cout << "    - Paramètres: " << totalParamCount() << std::endl;
    std::cout << "    - Magic tokens: " << magic_tokens.size() << std::endl;
}

// --- Fin ---

// --- Fin ---
