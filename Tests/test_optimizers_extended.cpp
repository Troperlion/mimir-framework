#include "test_utils.hpp"

#include "Model.hpp"
#include "runtimes/ops_loss_and_grad.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct Fixture {
    Model model;
    Layer* layer = nullptr;
    float* weights = nullptr;

    Fixture() {
        model.push("parameter", "Constant", 3);
        layer = model.getLayerByName("parameter");
        layer->inputs = {};
        layer->output = "x";
        layer->trainable_parameter = true;
        model.allocateParams();
        weights = layer->getWeights();
        weights[0] = 1.0f;
        weights[1] = -2.0f;
        weights[2] = 0.5f;
        layer->grad_weights = {0.2f, -0.4f, 0.1f};
    }

    const Optimizer::MomentBlock& moments(const Optimizer& optimizer) const {
        return optimizer.mv_by_param_ptr.at(reinterpret_cast<std::uintptr_t>(weights));
    }
};

Optimizer optimizer(OptimizerType type) {
    Optimizer result;
    result.type = type;
    result.decay_strategy = LRDecayStrategy::NONE;
    result.weight_decay = 0.0f;
    result.beta1 = 0.9f;
    result.beta2 = 0.9f;
    result.eps = 1e-8f;
    return result;
}

}  // namespace

int main() {
    constexpr float learning_rate = 0.01f;

    // A zero gradient must not clip valid parameters outside [-3, 3].
    for (const auto type : {OptimizerType::SGD, OptimizerType::ADAM,
         OptimizerType::ADAMW, OptimizerType::LION, OptimizerType::ADAFACTOR,
         OptimizerType::RADAM, OptimizerType::NADAM, OptimizerType::RMSPROP,
         OptimizerType::LAMB}) {
        Fixture fixture;
        auto opt = optimizer(type);
        fixture.weights[0] = 5.0f;
        fixture.weights[1] = -6.0f;
        fixture.layer->grad_weights = {0.0f, 0.0f, 0.0f};
        fixture.model.optimizerStep(opt, learning_rate);
        TASSERT_NEAR(fixture.weights[0], 5.0f, 0.0f);
        TASSERT_NEAR(fixture.weights[1], -6.0f, 0.0f);
        TASSERT_NEAR(fixture.weights[2], 0.5f, 0.0f);
    }


    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::LION);
        fixture.model.optimizerStep(opt, learning_rate);
        TASSERT_NEAR(fixture.weights[0], 0.99f, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -1.99f, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).m[0], 0.02f, 1e-6f);
        TASSERT_TRUE(opt.step == 1);
    }

    // Lion + Huber: distinct moment coefficients, sign reversals, zero gradient,
    // and decoupled decay, compared with an independent double reference.
    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::LION);
        opt.beta1 = 0.9f;
        opt.beta2 = 0.99f;
        opt.weight_decay = 0.1f;
        RuntimeLossGrad::PixelLossOptions loss_options;
        loss_options.huber_delta = 0.02f;
        double expected_weights[] = {1.0, -2.0, 0.5};
        double momentum[] = {0.0, 0.0, 0.0};
        for (int step = 0; step < 8; ++step) {
            const float residual = step < 2 ? 0.5f : (step < 5 ? -0.01f : 0.0f);
            std::vector<float> prediction(fixture.weights, fixture.weights + 3);
            const std::vector<float> target = {
                prediction[0] - residual, prediction[1] + residual, prediction[2]};
            const auto loss = RuntimeLossGrad::pixel_loss_and_grad(
                prediction, target, "huber", loss_options);
            fixture.layer->grad_weights = loss.grad;
            for (int i = 0; i < 3; ++i) {
                const double residual_i = static_cast<double>(prediction[i]) - target[i];
                const double gradient = std::clamp(residual_i,
                    -static_cast<double>(loss_options.huber_delta),
                    static_cast<double>(loss_options.huber_delta)) / 3.0;
                TASSERT_NEAR(loss.grad[i], gradient, 1e-9f);
                const double update = opt.beta1 * momentum[i] + (1.0 - opt.beta1) * gradient;
                expected_weights[i] *= 1.0 - learning_rate * opt.weight_decay;
                expected_weights[i] -= learning_rate * ((update > 0) - (update < 0));
                momentum[i] = opt.beta2 * momentum[i] + (1.0 - opt.beta2) * gradient;
            }
            fixture.model.optimizerStep(opt, learning_rate);
            for (int i = 0; i < 3; ++i) {
                TASSERT_NEAR(fixture.weights[i], expected_weights[i], 1e-6f);
                TASSERT_NEAR(fixture.moments(opt).m[i], momentum[i], 1e-8f);
            }
        }
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::ADAFACTOR);
        opt.adafactor_scale_parameter = false;
        opt.adafactor_clip_threshold = 10.0f;
        fixture.model.optimizerStep(opt, learning_rate);
        const float update = 1.0f;
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).v[0], 0.04f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::RADAM);
        fixture.model.optimizerStep(opt, learning_rate);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * 0.2f, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * 0.4f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::NADAM);
        fixture.model.optimizerStep(opt, learning_rate);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * 1.9f, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * 1.9f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::RMSPROP);
        opt.rmsprop_alpha = 0.9f;
        fixture.model.optimizerStep(opt, learning_rate);
        const float update = 1.0f / std::sqrt(0.1f);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * update, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).v[1], 0.016f, 1e-6f);
    }

    {
        Fixture fixture;
        auto opt = optimizer(OptimizerType::LAMB);
        fixture.model.optimizerStep(opt, learning_rate);
        const float trust_ratio = std::sqrt(5.25f / 3.0f);
        TASSERT_NEAR(fixture.weights[0], 1.0f - learning_rate * trust_ratio, 1e-6f);
        TASSERT_NEAR(fixture.weights[1], -2.0f + learning_rate * trust_ratio, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).m[0], 0.02f, 1e-6f);
        TASSERT_NEAR(fixture.moments(opt).v[0], 0.004f, 1e-6f);
    }

    return 0;
}