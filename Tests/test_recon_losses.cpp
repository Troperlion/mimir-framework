#include "test_utils.hpp"

#include "runtimes/ops_loss_and_grad.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace {

bool gradient_matches(const std::vector<float>& prediction,
                      const std::vector<float>& target,
                      const std::string& loss_type,
                      const RuntimeLossGrad::PixelLossOptions& options = {}) {
    const auto result = RuntimeLossGrad::pixel_loss_and_grad(
        prediction, target, loss_type, options);
    if (result.grad.size() != prediction.size()) return false;

    constexpr float step = 1e-3f;
    constexpr float tolerance = 2e-3f;
    std::vector<float> perturbed = prediction;
    for (size_t i = 0; i < prediction.size(); ++i) {
        perturbed[i] = prediction[i] + step;
        const double plus = RuntimeLossGrad::pixel_loss_and_grad(
            perturbed, target, loss_type, options).loss;
        perturbed[i] = prediction[i] - step;
        const double minus = RuntimeLossGrad::pixel_loss_and_grad(
            perturbed, target, loss_type, options).loss;
        perturbed[i] = prediction[i];
        const double numerical = (plus - minus) / (2.0 * step);
        if (!std::isfinite(result.grad[i]) ||
            std::abs(static_cast<double>(result.grad[i]) - numerical) > tolerance) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const std::vector<float> prediction = {0.15f, 0.35f, 0.65f, 0.85f};
    const std::vector<float> target = {0.05f, 0.75f, 0.25f, 0.95f};

    RuntimeLossGrad::PixelLossOptions options;
    options.huber_delta = 0.25f;
    options.charbonnier_eps = 3e-5f;
    options.gaussian_nll_sigma = 0.7f;

    for (const std::string loss_type : {
             "mse", "l2", "mae", "l1", "bce", "huber", "smoothl1",
             "smooth_l1", "charbonnier", "gaussian_nll", "nll_gaussian",
             "gaussian-nll"}) {
        TASSERT_TRUE(gradient_matches(prediction, target, loss_type, options));
    }

    const std::vector<float> weights = {0.4f, 1.6f, 0.4f, 1.6f};
    options.weights = &weights;
    TASSERT_TRUE(gradient_matches(prediction, target, "mse", options));
    const auto weighted = RuntimeLossGrad::pixel_loss_and_grad(prediction, target, "mse", options);
    TASSERT_TRUE(std::abs(weighted.grad[1]) > std::abs(weighted.grad[0]));
    options.weights = nullptr;

    for (const auto& aliases : {
             std::pair<std::string, std::string>{"mse", "l2"},
             {"mae", "l1"},
             {"smoothl1", "smooth_l1"},
             {"gaussian_nll", "gaussian-nll"},
             {"gaussian_nll", "nll_gaussian"}}) {
        const double canonical = RuntimeLossGrad::pixel_loss_and_grad(
            prediction, target, aliases.first, options).loss;
        const double alias = RuntimeLossGrad::pixel_loss_and_grad(
            prediction, target, aliases.second, options).loss;
        TASSERT_NEAR(canonical, alias, 1e-12);
    }

    const auto empty = RuntimeLossGrad::pixel_loss_and_grad(
        std::vector<float>{}, std::vector<float>{}, "mse", options);
    TASSERT_NEAR(empty.loss, 0.0, 0.0);
    TASSERT_TRUE(empty.grad.empty());

    const auto mismatch = RuntimeLossGrad::pixel_loss_and_grad(
        std::vector<float>{1.0f}, std::vector<float>{}, "mse", options);
    TASSERT_NEAR(mismatch.loss, 0.0, 0.0);
    TASSERT_TRUE(mismatch.grad.empty());
    return 0;
}
