#include "OptimizerDispatch.hpp"
#include "Model.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <climits>
#include <stdexcept>
bool optimizerRequestValid(const RuntimeOptimizerRequest& r) {
    if(!r.optimizer || !r.weights || !r.gradients || !r.count)return false;
    const auto& o=*r.optimizer;
    const int kind=static_cast<int>(o.type);
    return kind>=0 && kind<=8 && o.step>0 && std::isfinite(r.learning_rate) &&
        (!o.usesFirstMoment() || r.first_moment) && (!o.usesSecondMoment() || r.second_moment);
}
void optimizerHostStep(const RuntimeOptimizerRequest& r) {
    auto& opt=*r.optimizer;auto* weights=r.weights;const auto n=r.count;
    const float effective_lr=r.learning_rate;
    struct Moments {float* m;float* v;} state{r.first_moment,r.second_moment};
    auto* moments=&state;
        switch (opt.type) {
            case OptimizerType::SGD: {
                // SGD simple
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    float grad = r.gradients[i];
                    weights[i] -= effective_lr * grad;
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
                    float grad = r.gradients[i];

                    float& mi = (*moments).m[i];
                    float& vi = (*moments).v[i];
                    mi = b1 * mi + (1.0f - b1) * grad;
                    vi = b2 * vi + (1.0f - b2) * grad * grad;

                    float m_hat = mi / bias_correction1;
                    float v_hat = vi / bias_correction2;
                    
                    float denom = std::sqrt(v_hat) + opt.eps;
                    weights[i] -= effective_lr * (m_hat / denom);
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
                    float grad = r.gradients[i];
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
                }
                break;
            }

            case OptimizerType::LION: {
                const float b1 = std::clamp(opt.beta1, 0.0f, 1.0f);
                const float b2 = std::clamp(opt.beta2, 0.0f, 1.0f);
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    const float grad = r.gradients[i];
                    float& mi = moments->m[i];
                    const float update = b1 * mi + (1.0f - b1) * grad;
                    const float direction = (update > 0.0f) - (update < 0.0f);
                    weights[i] -= effective_lr * (direction + opt.weight_decay * weights[i]);
                    mi = b2 * mi + (1.0f - b2) * grad;
                }
                break;
            }

            case OptimizerType::ADAFACTOR: {
                const float decay = std::clamp(
                    1.0f - std::pow(static_cast<float>(opt.step), opt.adafactor_decay_rate),
                    0.0f, 1.0f);
                double update_sq_sum = 0.0;
                double param_sq_sum = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    const float grad = r.gradients[i];
                    float& vi = moments->v[i];
                    vi = decay * vi + (1.0f - decay) * grad * grad;
                    const float update = grad / std::sqrt(vi + std::max(1e-30f, opt.eps));
                    update_sq_sum += static_cast<double>(update) * update;
                    param_sq_sum += static_cast<double>(weights[i]) * weights[i];
                }
                const float update_rms = n > 0 ? static_cast<float>(std::sqrt(update_sq_sum / n)) : 0.0f;
                const float clip = std::max(1e-6f, opt.adafactor_clip_threshold);
                const float clip_scale = 1.0f / std::max(1.0f, update_rms / clip);
                const float param_scale = opt.adafactor_scale_parameter && n > 0
                    ? std::max(opt.adafactor_eps2, static_cast<float>(std::sqrt(param_sq_sum / n)))
                    : 1.0f;
                const float relative_lr = opt.adafactor_relative_step
                    ? std::min(1e-2f, 1.0f / std::sqrt(static_cast<float>(opt.step)))
                    : effective_lr;
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    float update = clip_scale * r.gradients[i]
                        / std::sqrt(moments->v[i] + std::max(1e-30f, opt.eps));
                    if (opt.adafactor_beta1 > 0.0f) {
                        float& momentum = moments->m[i];
                        momentum = opt.adafactor_beta1 * momentum
                            + (1.0f - opt.adafactor_beta1) * update;
                        update = momentum;
                    }
                    weights[i] -= relative_lr * param_scale * update;
                }
                break;
            }

            case OptimizerType::RADAM: {
                const float b1 = std::clamp(opt.beta1, 0.0f, 1.0f);
                const float b2 = std::clamp(opt.beta2, 0.0f, 1.0f - 1e-7f);
                const float b1_power = std::pow(b1, static_cast<float>(opt.step));
                const float b2_power = std::pow(b2, static_cast<float>(opt.step));
                const float bias_correction1 = std::max(1e-8f, 1.0f - b1_power);
                const float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
                const float rho = rho_inf - 2.0f * static_cast<float>(opt.step) * b2_power
                    / std::max(1e-8f, 1.0f - b2_power);
                const float rectification = rho > 5.0f
                    ? std::sqrt(((rho - 4.0f) * (rho - 2.0f) * rho_inf)
                                / ((rho_inf - 4.0f) * (rho_inf - 2.0f) * rho))
                    : 1.0f;
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    const float grad = r.gradients[i];
                    float& mi = moments->m[i];
                    float& vi = moments->v[i];
                    mi = b1 * mi + (1.0f - b1) * grad;
                    vi = b2 * vi + (1.0f - b2) * grad * grad;
                    const float m_hat = mi / bias_correction1;
                    const float update = rho > 5.0f
                        ? rectification * m_hat / (std::sqrt(vi / std::max(1e-8f, 1.0f - b2_power)) + opt.eps)
                        : m_hat;
                    weights[i] -= effective_lr * update;
                }
                break;
            }

            case OptimizerType::NADAM: {
                const float b1 = std::clamp(opt.beta1, 0.0f, 1.0f);
                const float b2 = std::clamp(opt.beta2, 0.0f, 1.0f);
                const float bc1 = std::max(1e-8f, 1.0f - std::pow(b1, static_cast<float>(opt.step)));
                const float bc2 = std::max(1e-8f, 1.0f - std::pow(b2, static_cast<float>(opt.step)));
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    const float grad = r.gradients[i];
                    float& mi = moments->m[i];
                    float& vi = moments->v[i];
                    mi = b1 * mi + (1.0f - b1) * grad;
                    vi = b2 * vi + (1.0f - b2) * grad * grad;
                    const float nesterov = b1 * (mi / bc1) + (1.0f - b1) * grad / bc1;
                    weights[i] -= effective_lr * nesterov / (std::sqrt(vi / bc2) + opt.eps);
                }
                break;
            }

            case OptimizerType::RMSPROP: {
                const float alpha = std::clamp(opt.rmsprop_alpha, 0.0f, 1.0f);
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    const float grad = r.gradients[i];
                    float& vi = moments->v[i];
                    vi = alpha * vi + (1.0f - alpha) * grad * grad;
                    weights[i] -= effective_lr * grad / (std::sqrt(vi) + opt.eps);
                }
                break;
            }

            case OptimizerType::LAMB: {
                const float b1 = std::clamp(opt.beta1, 0.0f, 1.0f);
                const float b2 = std::clamp(opt.beta2, 0.0f, 1.0f);
                const float bc1 = std::max(1e-8f, 1.0f - std::pow(b1, static_cast<float>(opt.step)));
                const float bc2 = std::max(1e-8f, 1.0f - std::pow(b2, static_cast<float>(opt.step)));
                double weight_norm_sq = 0.0;
                double update_norm_sq = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    const float grad = r.gradients[i];
                    float& mi = moments->m[i];
                    float& vi = moments->v[i];
                    mi = b1 * mi + (1.0f - b1) * grad;
                    vi = b2 * vi + (1.0f - b2) * grad * grad;
                    const float update = (mi / bc1) / (std::sqrt(vi / bc2) + opt.eps)
                        + opt.weight_decay * weights[i];
                    weight_norm_sq += static_cast<double>(weights[i]) * weights[i];
                    update_norm_sq += static_cast<double>(update) * update;
                }
                const float weight_norm = static_cast<float>(std::sqrt(weight_norm_sq));
                const float update_norm = static_cast<float>(std::sqrt(update_norm_sq));
                const float trust_ratio = weight_norm > 0.0f && update_norm > 0.0f
                    ? weight_norm / update_norm : 1.0f;
                #pragma omp simd
                for (size_t i = 0; i < n; ++i) {
                    const float update = (moments->m[i] / bc1)
                        / (std::sqrt(moments->v[i] / bc2) + opt.eps)
                        + opt.weight_decay * weights[i];
                    weights[i] -= effective_lr * trust_ratio * update;
                }
                break;
            }
        }

}
OptimizerDeviceLayout optimizerDeviceLayout(const RuntimeOptimizerRequest& r) {
    OptimizerDeviceLayout layout;
    const auto& optimizer = *r.optimizer;
    size_t offset = r.count;
    if (optimizer.usesFirstMoment()) {
        layout.first_moment = offset;
        layout.moment_flags |= 1;
        offset += r.count;
    }
    if (optimizer.usesSecondMoment()) {
        layout.second_moment = offset;
        layout.moment_flags |= 2;
        offset += r.count;
    }
    layout.result_count = offset;
    layout.gradients = offset;
    offset += r.count;
    if (optimizer.type == OptimizerType::ADAFACTOR || optimizer.type == OptimizerType::LAMB) {
        layout.updates = offset;
        offset += r.count;
        layout.reductions = offset;
        offset += 2*((r.count+255)/256)+2;
    }
    layout.total_count = offset;
    return layout;
}
std::vector<int> optimizerParameters(const RuntimeOptimizerRequest& r, bool compact) {
    const auto& o=*r.optimizer;
    auto finish = [&](std::vector<int> parameters) {
        if (compact) {
            const auto layout = optimizerDeviceLayout(r);
            parameters.insert(parameters.end(), {
                int(layout.result_count), int(layout.first_moment), int(layout.second_moment),
                int(layout.gradients), int(layout.updates), int(layout.reductions), int(layout.moment_flags)});
        }
        return parameters;
    };
    if (o.type == OptimizerType::SGD) {
        // Preserve the shared parameter ABI; SGD only consumes F(0).
        std::vector<int> parameters;
        parameters.reserve(compact ? 26 : 19);
        parameters.resize(19, 0);
        parameters[1] = static_cast<int>(r.count);
        std::memcpy(&parameters[3], &r.learning_rate, sizeof(float));
        return finish(std::move(parameters));
    }
    auto clamp=[](float v,float lo,float hi){return std::clamp(v,lo,hi);};
    const float b1=(o.type==OptimizerType::ADAM || o.type==OptimizerType::ADAMW)?o.beta1:clamp(o.beta1,0.f,1.f);
    const float b2=(o.type==OptimizerType::RADAM)?clamp(o.beta2,0.f,1.f-1e-7f):
        (o.type==OptimizerType::ADAM || o.type==OptimizerType::ADAMW)?o.beta2:clamp(o.beta2,0.f,1.f);
    const bool bias_corrected = o.type == OptimizerType::ADAM || o.type == OptimizerType::ADAMW ||
        o.type == OptimizerType::RADAM || o.type == OptimizerType::NADAM || o.type == OptimizerType::LAMB;
    const float power1 = bias_corrected ? std::pow(b1,float(o.step)) : 0.f;
    const float power2 = bias_corrected ? std::pow(b2,float(o.step)) : 0.f;
    float bc1=1.f-power1,bc2=1.f-power2;
    if(o.type==OptimizerType::ADAM || o.type==OptimizerType::ADAMW){if(bc1<=0)bc1=1e-8f;if(bc2<=0)bc2=1e-8f;}
    else {bc1=std::max(1e-8f,bc1);bc2=std::max(1e-8f,bc2);}
    float rho=0,rect=1;
    if(o.type==OptimizerType::RADAM){const float power=power2;const float inf=2.f/(1.f-b2)-1.f;
        rho=inf-2.f*float(o.step)*power/std::max(1e-8f,1.f-power);
        if(rho>5.f)rect=std::sqrt(((rho-4)*(rho-2)*inf)/((inf-4)*(inf-2)*rho));}
    const float values[]={r.learning_rate,b1,b2,o.eps,o.weight_decay,bc1,bc2,
        (o.type==OptimizerType::ADAFACTOR ? clamp(1.f-std::pow(float(o.step),o.adafactor_decay_rate),0.f,1.f) : 0.f),o.adafactor_beta1,
        std::max(1e-6f,o.adafactor_clip_threshold),o.adafactor_eps2,
        o.type==OptimizerType::ADAFACTOR && o.adafactor_relative_step?std::min(1e-2f,1.f/std::sqrt(float(o.step))):r.learning_rate,
        o.adafactor_scale_parameter?1.f:0.f,rect,rho,clamp(o.rmsprop_alpha,0.f,1.f)};
    std::vector<int> p;
    p.reserve(compact ? 26 : 19);
    p.insert(p.end(), {0,int(r.count),int(o.type)});
    for(float value:values){int bits;std::memcpy(&bits,&value,4);p.push_back(bits);}return finish(std::move(p));
}
std::vector<float> optimizerPack(const RuntimeOptimizerRequest& r, bool compact) {
    if (compact) {
        const auto layout = optimizerDeviceLayout(r);
        std::vector<float> memory;
        memory.reserve(layout.total_count);
        memory.insert(memory.end(), r.weights, r.weights+r.count);
        if (layout.moment_flags & 1) memory.insert(memory.end(), r.first_moment, r.first_moment+r.count);
        if (layout.moment_flags & 2) memory.insert(memory.end(), r.second_moment, r.second_moment+r.count);
        memory.insert(memory.end(), r.gradients, r.gradients+r.count);
        memory.resize(layout.total_count, 0.f);
        return memory;
    }
    std::vector<float> memory(5*r.count+2*((r.count+255)/256)+2,0.f);
    std::copy_n(r.weights,r.count,memory.data());std::copy_n(r.gradients,r.count,memory.data()+r.count);
    if(r.first_moment)std::copy_n(r.first_moment,r.count,memory.data()+2*r.count);
    if(r.second_moment)std::copy_n(r.second_moment,r.count,memory.data()+3*r.count);
    return memory;
}
bool optimizerCommit(const RuntimeOptimizerRequest& r,const std::vector<float>& memory, bool compact) {
    if (compact) {
        const auto layout = optimizerDeviceLayout(r);
        if (memory.size() != layout.total_count) return false;
        std::copy_n(memory.data(), r.count, r.weights);
        if (layout.moment_flags & 1) std::copy_n(memory.data()+layout.first_moment, r.count, r.first_moment);
        if (layout.moment_flags & 2) std::copy_n(memory.data()+layout.second_moment, r.count, r.second_moment);
        return true;
    }
    if (r.optimizer->type == OptimizerType::SGD && memory.size() == 2*r.count) {
        std::copy_n(memory.data(), r.count, r.weights);
        return true;
    }
    if(memory.size()!=5*r.count+2*((r.count+255)/256)+2)return false;
    std::copy_n(memory.data(),r.count,r.weights);
    if(r.first_moment)std::copy_n(memory.data()+2*r.count,r.count,r.first_moment);
    if(r.second_moment)std::copy_n(memory.data()+3*r.count,r.count,r.second_moment);
    return true;
}
