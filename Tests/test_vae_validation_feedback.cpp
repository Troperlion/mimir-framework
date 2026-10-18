#include "training/VaeValidationFeedback.hpp"
#include "test_utils.hpp"

int main() {
    VaeValidationFeedback::Config config;
    config.initial_kl_beta = 0.5f;
    config.kl_beta_min = 0.025f;
    config.kl_beta_max = 1.0f;
    VaeValidationFeedback feedback(config);

    auto decision = feedback.update(1.0f, 1.0f, 1);
    TASSERT_TRUE(decision.action == VaeValidationFeedback::Action::FirstPoint);
    TASSERT_NEAR(decision.lr_scale, 1.0f, 1e-6f);
    TASSERT_NEAR(decision.kl_beta, 0.5f, 1e-6f);

    decision = feedback.update(0.8f, 1.0f, 2);
    TASSERT_TRUE(decision.action == VaeValidationFeedback::Action::Reward);
    TASSERT_TRUE(decision.rewarded);
    TASSERT_TRUE(!decision.penalized);
    TASSERT_NEAR(decision.lr_scale, 1.05f, 1e-6f);
    TASSERT_NEAR(decision.kl_beta, 0.49f, 1e-6f);

    decision = feedback.update(0.9f, 1.0f, 3);
    TASSERT_TRUE(decision.action == VaeValidationFeedback::Action::Penalty);
    TASSERT_TRUE(decision.penalized);
    TASSERT_TRUE(!decision.rewarded);
    TASSERT_NEAR(decision.lr_scale, 0.735f, 1e-6f);
    TASSERT_NEAR(decision.kl_beta, 0.4165f, 1e-6f);

    const float beta_before_collapse = decision.kl_beta;
    decision = feedback.update(0.9f, 0.001f, 4);
    TASSERT_TRUE(decision.action == VaeValidationFeedback::Action::KlCollapse);
    TASSERT_TRUE(decision.kl_beta > beta_before_collapse);

    const float beta_before_excess = decision.kl_beta;
    decision = feedback.update(0.9f, 6.0f, 5);
    TASSERT_TRUE(decision.action == VaeValidationFeedback::Action::KlExcess);
    TASSERT_TRUE(decision.kl_beta < beta_before_excess);

    VaeValidationFeedback::Config disabled_kl_config;
    disabled_kl_config.initial_kl_beta = 0.0f;
    disabled_kl_config.kl_beta_min = 0.0f;
    disabled_kl_config.kl_beta_max = 1.0f;
    VaeValidationFeedback disabled_kl(disabled_kl_config);
    decision = disabled_kl.update(1.0f, 0.0f, 1);
    TASSERT_NEAR(decision.kl_beta, 0.0f, 1e-6f);

    VaeValidationFeedback viz_override(config);
    (void)viz_override.update(1.0f, 1.0f, 1);
    decision = viz_override.update(0.8f, 0.001f, 2, false);
    TASSERT_TRUE(decision.action == VaeValidationFeedback::Action::Reward);
    TASSERT_NEAR(decision.lr_scale, 1.05f, 1e-6f);
    TASSERT_NEAR(decision.kl_beta, 0.5f, 1e-6f);
    return 0;
}
