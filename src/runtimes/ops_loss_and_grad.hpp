#pragma once

#include <cstddef>
#include "training/SpatialRelation.hpp"
#include <string>
#include <vector>

namespace RuntimeLossGrad {

struct GlobalSSIM {
    double loss = 0.0;
    std::vector<float> grad;
};

struct LossWithGrad {
    double loss = 0.0;
    std::vector<float> grad;
    double relation_loss = 0.0;
};

struct PixelLossOptions {
    float huber_delta = 1.0f;
    float charbonnier_eps = 1e-3f;
    float gaussian_nll_sigma = 1.0f;
    const std::vector<float>* weights = nullptr;
    const std::vector<SpatialRelation>* relations = nullptr;
    bool relation_only = false;
};

float sigmoid_scalar(float x);

std::string canonical_pixel_loss_name(std::string name);

LossWithGrad pixel_loss_and_grad(
    const float* pred,
    const float* target,
    size_t count,
    const std::string& loss_type,
    const PixelLossOptions& options = {}
);

LossWithGrad pixel_loss_and_grad(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    const std::string& loss_type,
    const PixelLossOptions& options = {}
);

GlobalSSIM ssim_global_hwc(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    int w,
    int h,
    int c,
    float k1,
    float k2,
    float L
);

void avgpool2x2_hwc(
    const std::vector<float>& in,
    int w,
    int h,
    int c,
    std::vector<float>& out,
    int& out_w,
    int& out_h
);

void avgpool2x2_back_hwc(
    const std::vector<float>& grad_out,
    int in_w,
    int in_h,
    int c,
    std::vector<float>& grad_in
);

LossWithGrad spectral_dct_l1_hwc(
    const std::vector<float>& pred,
    const std::vector<float>& target,
    int w,
    int h,
    int c
);

} // namespace RuntimeLossGrad
