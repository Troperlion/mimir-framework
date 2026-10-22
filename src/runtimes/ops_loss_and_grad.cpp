#include "runtimes/losses/LossDispatch.hpp"
#include <cctype>
#include <array>
#include <stdexcept>
#include "runtimes/ops_loss_and_grad.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <mutex>
#include <unordered_map>

#if defined(_MSC_VER)
#define MIMIR_LOSS_RESTRICT __restrict
#else
#define MIMIR_LOSS_RESTRICT __restrict__
#endif

namespace RuntimeLossGrad {

float sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

std::string canonical_pixel_loss_name(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (name == "l2") return "mse";
    if (name == "l1") return "mae";
    if (name == "smooth_l1") return "smoothl1";
    if (name == "gaussian-nll") return "gaussian_nll";
    return name;
}

namespace Host {

LossWithGrad pixel_loss_and_grad(
    const float* pred,
    const float* target,
    size_t count,
    const std::string& loss_type,
    const PixelLossOptions& options
) {
    LossWithGrad out;
    out.grad.assign(count, 0.0f);
    if (count == 0 || pred == nullptr || target == nullptr) return out;

    const std::string type = canonical_pixel_loss_name(loss_type);
    const double inv_n = 1.0 / static_cast<double>(count);
    const double huber_delta = std::max(1e-6, static_cast<double>(options.huber_delta));
    const double charbonnier_eps = std::max(1e-12, static_cast<double>(options.charbonnier_eps));
    const double sigma = std::max(1e-6, static_cast<double>(options.gaussian_nll_sigma));
    const double inv_variance = 1.0 / (sigma * sigma);
    const double log_variance = std::log(sigma * sigma);
    constexpr double bce_eps = 1e-7;

    double sum = 0.0;
    for (size_t i = 0; !options.relation_only && i < count; ++i) {
        const double prediction = static_cast<double>(pred[i]);
        const double expected = static_cast<double>(target[i]);
        const double diff = prediction - expected;
        const double weight = options.weights && options.weights->size() >= count
            ? std::max(0.0, static_cast<double>((*options.weights)[i])) : 1.0;

        if (type == "mae") {
            sum += weight * std::abs(diff);
            out.grad[i] = static_cast<float>(weight * inv_n * (diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0)));
        } else if (type == "huber" || type == "smoothl1") {
            const double abs_diff = std::abs(diff);
            if (abs_diff <= huber_delta) {
                sum += weight * 0.5 * diff * diff;
                out.grad[i] = static_cast<float>(weight * inv_n * diff);
            } else {
                sum += weight * huber_delta * (abs_diff - 0.5 * huber_delta);
                out.grad[i] = static_cast<float>(weight * inv_n * huber_delta * (diff > 0.0 ? 1.0 : -1.0));
            }
        } else if (type == "charbonnier") {
            const double denominator = std::sqrt(diff * diff + charbonnier_eps * charbonnier_eps);
            sum += weight * denominator;
            out.grad[i] = static_cast<float>(weight * inv_n * diff / denominator);
        } else if (type == "gaussian_nll" || type == "nll_gaussian") {
            sum += weight * 0.5 * (diff * diff * inv_variance + log_variance);
            out.grad[i] = static_cast<float>(weight * inv_n * diff * inv_variance);
        } else if (type == "bce") {
            const double probability = std::clamp(prediction, bce_eps, 1.0 - bce_eps);
            sum += weight * -(expected * std::log(probability) +
                              (1.0 - expected) * std::log(1.0 - probability));
            if (prediction > bce_eps && prediction < 1.0 - bce_eps) {
                out.grad[i] = static_cast<float>(
                    weight * inv_n * (probability - expected) / (probability * (1.0 - probability)));
            }
        } else {
            sum += weight * diff * diff;
            out.grad[i] = static_cast<float>(2.0 * weight * inv_n * diff);
        }
    }

    if (options.relations && !options.relations->empty()) {
        for (const auto& relation : *options.relations) {
            if (relation.width<=0 || relation.height<=0 || relation.channels<=0 ||
                size_t(relation.width)*relation.height*relation.channels>count ||
                !std::isfinite(relation.weight) || relation.weight<0)
                throw std::invalid_argument("Invalid relation supervision grid/weight");
            if (relation.weight==0) continue;
            std::string predicate=relation.predicate;
            std::transform(predicate.begin(),predicate.end(),predicate.begin(),[](unsigned char c){return char(std::tolower(c));});
            const bool horizontal=predicate=="left_of" || predicate=="right_of" || predicate=="left" || predicate=="right";
            const bool vertical=predicate=="above" || predicate=="below" || predicate=="above_of" || predicate=="below_of";
            const std::array<bool,3> enabled={true,!vertical,!horizontal};
            const int features=horizontal || vertical ? 2 : 3;
            const double scale=relation.weight/(double(options.relations->size())*relation.channels*features);
            for(int channel=0;channel<relation.channels;++channel) {
                auto visit=[&](const std::array<double,4>& box,auto fn) {
                    const double x0=box[0]*relation.width,y0=box[1]*relation.height;
                    const double x1=box[2]*relation.width,y1=box[3]*relation.height;
                    if (!(x0>=0 && y0>=0 && x1<=relation.width && y1<=relation.height && x1>x0 && y1>y0))
                        throw std::invalid_argument("Invalid relation region");
                    for(int y=int(std::floor(y0));y<std::min(relation.height,int(std::ceil(y1)));++y)
                        for(int x=int(std::floor(x0));x<std::min(relation.width,int(std::ceil(x1)));++x) {
                            const size_t p=size_t(y)*relation.width+x;
                            const size_t i=relation.chw ? size_t(channel)*relation.width*relation.height+p : p*relation.channels+channel;
                            if(options.weights && options.weights->size()>i && (*options.weights)[i]<=0) continue;
                            const double xa=std::max(x0,double(x)),xb=std::min(x1,x+1.0);
                            const double ya=std::max(y0,double(y)),yb=std::min(y1,y+1.0);
                            const double area=(xb-xa)*(yb-ya);
                            fn(i,area,std::array<double,3>{1,((xa+xb)-(x0+x1))/(x1-x0),((ya+yb)-(y0+y1))/(y1-y0)});
                        }
                };
                std::array<double,3> subject{},object{};double sa=0,oa=0;
                auto pool=[&](const std::array<double,4>& box,std::array<double,3>& moments,double& area) {
                    visit(box,[&](size_t i,double a,const std::array<double,3>& f){
                        area+=a;for(int k=0;k<3;++k) moments[k]+=a*f[k]*(double(pred[i])-target[i]);
                    });
                    if(area>0) for(auto& m:moments) m/=area;
                };
                pool(relation.subject,subject,sa);pool(relation.object,object,oa);
                if(sa<=0 || oa<=0) continue;
                std::array<double,3> delta{};
                for(int k=0;k<3;++k) if(enabled[k]) {
                    delta[k]=subject[k]-object[k];
                    out.relation_loss+=scale*delta[k]*delta[k];
                }
                auto backward=[&](const std::array<double,4>& box,double area,double sign) {
                    visit(box,[&](size_t i,double a,const std::array<double,3>& f){
                        double gradient=0;for(int k=0;k<3;++k) gradient+=delta[k]*f[k];
                        out.grad[i]+=float(sign*2*scale*a/area*gradient);
                    });
                };
                backward(relation.subject,sa,1);backward(relation.object,oa,-1);
            }
        }
    }
    out.loss = sum * inv_n + out.relation_loss;
    return out;
}

LossWithGrad pixel_loss_and_grad(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    const std::string& loss_type,
    const PixelLossOptions& options
) {
    if (pred.size() != target.size()) return {};
    return Host::pixel_loss_and_grad(pred.data(), target.data(), pred.size(), loss_type, options);
}

GlobalSSIM ssim_global_hwc(
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

    double loss_sum = 0.0;

    const size_t HW = static_cast<size_t>(W) * static_cast<size_t>(H);
    const double inv_hw = 1.0 / static_cast<double>(std::max<size_t>(1, HW));
    for (int ch = 0; ch < C; ++ch) {
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

void avgpool2x2_hwc(
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

void avgpool2x2_back_hwc(
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

struct Dct1DCache {
    int n = 0;
    std::vector<float> cos_nk;
    std::vector<float> alpha_k;
};

static const Dct1DCache& get_dct1d_cache(int N) {
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

static void dct2d_ortho_hwc(
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

    const float* MIMIR_LOSS_RESTRICT in_ptr = in.data();
    float* MIMIR_LOSS_RESTRICT out_ptr = out.data();

    const auto& cw = get_dct1d_cache(W);
    const auto& ch = get_dct1d_cache(H);
    const float* MIMIR_LOSS_RESTRICT cw_cos = cw.cos_nk.data();
    const float* MIMIR_LOSS_RESTRICT cw_alpha = cw.alpha_k.data();
    const float* MIMIR_LOSS_RESTRICT ch_cos = ch.cos_nk.data();
    const float* MIMIR_LOSS_RESTRICT ch_alpha = ch.alpha_k.data();

    const size_t n2 = static_cast<size_t>(W) * static_cast<size_t>(H);
    static thread_local std::vector<float> tmp_row;
    static thread_local std::vector<float> tmp_col;
    tmp_row.resize(n2);
    tmp_col.resize(n2);
    float* MIMIR_LOSS_RESTRICT tmp_row_ptr = tmp_row.data();
    float* MIMIR_LOSS_RESTRICT tmp_col_ptr = tmp_col.data();

    const size_t stride_c = static_cast<size_t>(C);
    const size_t stride_wc = static_cast<size_t>(W) * stride_c;
    const size_t stride_w = static_cast<size_t>(W);
    const size_t stride_h = static_cast<size_t>(H);

    for (int cc = 0; cc < C; ++cc) {
        for (int yy = 0; yy < H; ++yy) {
            for (int kk = 0; kk < W; ++kk) {
                double sum = 0.0;
                const float alpha = cw_alpha[static_cast<size_t>(kk)];
                for (int xx = 0; xx < W; ++xx) {
                    const size_t iidx = static_cast<size_t>(yy) * stride_wc + static_cast<size_t>(xx) * stride_c + static_cast<size_t>(cc);
                    const float x = in_ptr[iidx];
                    const float co = cw_cos[static_cast<size_t>(xx) * stride_w + static_cast<size_t>(kk)];
                    sum += static_cast<double>(x) * static_cast<double>(co);
                }
                tmp_row_ptr[static_cast<size_t>(yy) * stride_w + static_cast<size_t>(kk)] = static_cast<float>(static_cast<double>(alpha) * sum);
            }
        }

        for (int kkx = 0; kkx < W; ++kkx) {
            for (int kky = 0; kky < H; ++kky) {
                double sum = 0.0;
                const float alpha = ch_alpha[static_cast<size_t>(kky)];
                for (int yy = 0; yy < H; ++yy) {
                    const float v = tmp_row_ptr[static_cast<size_t>(yy) * stride_w + static_cast<size_t>(kkx)];
                    const float co = ch_cos[static_cast<size_t>(yy) * stride_h + static_cast<size_t>(kky)];
                    sum += static_cast<double>(v) * static_cast<double>(co);
                }
                tmp_col_ptr[static_cast<size_t>(kky) * stride_w + static_cast<size_t>(kkx)] = static_cast<float>(static_cast<double>(alpha) * sum);
            }
        }

        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                const size_t oidx = static_cast<size_t>(yy) * stride_wc + static_cast<size_t>(xx) * stride_c + static_cast<size_t>(cc);
                out_ptr[oidx] = tmp_col_ptr[static_cast<size_t>(yy) * stride_w + static_cast<size_t>(xx)];
            }
        }
    }
}

static void idct2d_ortho_hwc(
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

    const float* MIMIR_LOSS_RESTRICT in_ptr = in.data();
    float* MIMIR_LOSS_RESTRICT out_ptr = out.data();

    const auto& cw = get_dct1d_cache(W);
    const auto& ch = get_dct1d_cache(H);
    const float* MIMIR_LOSS_RESTRICT cw_cos = cw.cos_nk.data();
    const float* MIMIR_LOSS_RESTRICT cw_alpha = cw.alpha_k.data();
    const float* MIMIR_LOSS_RESTRICT ch_cos = ch.cos_nk.data();
    const float* MIMIR_LOSS_RESTRICT ch_alpha = ch.alpha_k.data();

    const size_t n2 = static_cast<size_t>(W) * static_cast<size_t>(H);
    static thread_local std::vector<float> tmp_row;
    static thread_local std::vector<float> tmp_col;
    tmp_row.resize(n2);
    tmp_col.resize(n2);
    float* MIMIR_LOSS_RESTRICT tmp_row_ptr = tmp_row.data();
    float* MIMIR_LOSS_RESTRICT tmp_col_ptr = tmp_col.data();

    const size_t stride_c = static_cast<size_t>(C);
    const size_t stride_wc = static_cast<size_t>(W) * stride_c;
    const size_t stride_w = static_cast<size_t>(W);
    const size_t stride_h = static_cast<size_t>(H);

    for (int cc = 0; cc < C; ++cc) {
        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                const size_t iidx = static_cast<size_t>(yy) * stride_wc + static_cast<size_t>(xx) * stride_c + static_cast<size_t>(cc);
                tmp_col_ptr[static_cast<size_t>(yy) * stride_w + static_cast<size_t>(xx)] = in_ptr[iidx];
            }
        }

        for (int kkx = 0; kkx < W; ++kkx) {
            for (int yy = 0; yy < H; ++yy) {
                double sum = 0.0;
                for (int kky = 0; kky < H; ++kky) {
                    const float alpha = ch_alpha[static_cast<size_t>(kky)];
                    const float v = tmp_col_ptr[static_cast<size_t>(kky) * stride_w + static_cast<size_t>(kkx)];
                    const float co = ch_cos[static_cast<size_t>(yy) * stride_h + static_cast<size_t>(kky)];
                    sum += static_cast<double>(alpha) * static_cast<double>(v) * static_cast<double>(co);
                }
                tmp_row_ptr[static_cast<size_t>(yy) * stride_w + static_cast<size_t>(kkx)] = static_cast<float>(sum);
            }
        }

        for (int yy = 0; yy < H; ++yy) {
            for (int xx = 0; xx < W; ++xx) {
                double sum = 0.0;
                for (int kk = 0; kk < W; ++kk) {
                    const float alpha = cw_alpha[static_cast<size_t>(kk)];
                    const float v = tmp_row_ptr[static_cast<size_t>(yy) * stride_w + static_cast<size_t>(kk)];
                    const float co = cw_cos[static_cast<size_t>(xx) * stride_w + static_cast<size_t>(kk)];
                    sum += static_cast<double>(alpha) * static_cast<double>(v) * static_cast<double>(co);
                }
                const size_t oidx = static_cast<size_t>(yy) * stride_wc + static_cast<size_t>(xx) * stride_c + static_cast<size_t>(cc);
                out_ptr[oidx] = static_cast<float>(sum);
            }
        }
    }
}

LossWithGrad spectral_dct_l1_hwc(
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

    static thread_local std::vector<float> coeff_p;
    static thread_local std::vector<float> coeff_t;
    dct2d_ortho_hwc(pred, W, H, C, coeff_p);
    dct2d_ortho_hwc(target, W, H, C, coeff_t);

    static thread_local std::vector<float> grad_coeff;
    grad_coeff.assign(n, 0.0f);
    const float* MIMIR_LOSS_RESTRICT coeff_p_ptr = coeff_p.data();
    const float* MIMIR_LOSS_RESTRICT coeff_t_ptr = coeff_t.data();
    float* MIMIR_LOSS_RESTRICT grad_coeff_ptr = grad_coeff.data();
    const double inv_n = 1.0 / static_cast<double>(std::max<size_t>(1, n));
    double sum_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = coeff_p_ptr[i] - coeff_t_ptr[i];
        sum_abs += static_cast<double>(std::abs(d));
        const float s = (d > 0.0f) ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
        grad_coeff_ptr[i] = static_cast<float>(inv_n) * s;
    }
    out.loss = sum_abs * inv_n;

    idct2d_ortho_hwc(grad_coeff, W, H, C, out.grad);
    return out;
}

} // namespace Host
} // namespace RuntimeLossGrad

#undef MIMIR_LOSS_RESTRICT
