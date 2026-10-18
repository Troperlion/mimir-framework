#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

class VaeValidationFeedback {
public:
    enum class Action {
        Ignored,
        FirstPoint,
        Reward,
        Penalty,
        Plateau,
        KlCollapse,
        KlExcess
    };

    struct Config {
        bool enabled = true;
        float lr_reward_factor = 1.05f;
        float lr_penalty_factor = 0.70f;
        float lr_scale_min = 0.10f;
        float lr_scale_max = 1.50f;
        float improve_threshold = 0.001f;
        int min_steps = 0;

        bool kl_enabled = true;
        float initial_kl_beta = 0.0f;
        float kl_beta_min = 0.0f;
        float kl_beta_max = 0.0f;
        float kl_target_min = 0.01f;
        float kl_target_max = 5.0f;
        float kl_reward_factor = 0.98f;
        float kl_penalty_factor = 0.85f;
        float kl_plateau_factor = 0.95f;
        float kl_collapse_factor = 1.20f;
        float kl_excess_factor = 0.80f;
    };

    struct Decision {
        Action action = Action::Ignored;
        float relative_change = 0.0f;
        float lr_scale = 1.0f;
        float kl_beta = 0.0f;
        bool lr_changed = false;
        bool kl_changed = false;
        bool rewarded = false;
        bool penalized = false;
    };

    explicit VaeValidationFeedback(Config config)
        : config_(sanitize(config)), kl_beta_(config_.initial_kl_beta) {}

    Decision update(float reconstruction, float kl, int step, bool adjust_kl = true) {
        Decision decision;
        decision.lr_scale = lr_scale_;
        decision.kl_beta = kl_beta_;
        if (!config_.enabled || step < config_.min_steps ||
            !std::isfinite(reconstruction) || reconstruction < 0.0f ||
            !std::isfinite(kl) || kl < 0.0f) {
            return decision;
        }

        const bool first = !std::isfinite(best_reconstruction_);
        if (first) {
            best_reconstruction_ = reconstruction;
            decision.action = Action::FirstPoint;
        } else {
            const float denominator = std::max(1e-12f, std::abs(best_reconstruction_));
            decision.relative_change =
                (best_reconstruction_ - reconstruction) / denominator;
            if (decision.relative_change > config_.improve_threshold) {
                const float previous = lr_scale_;
                lr_scale_ = std::min(
                    config_.lr_scale_max,
                    lr_scale_ * config_.lr_reward_factor
                );
                decision.lr_changed = lr_scale_ != previous;
                decision.rewarded = true;
                best_reconstruction_ = reconstruction;
                decision.action = Action::Reward;
            } else if (decision.relative_change < -config_.improve_threshold) {
                const float previous = lr_scale_;
                lr_scale_ = std::max(
                    config_.lr_scale_min,
                    lr_scale_ * config_.lr_penalty_factor
                );
                decision.lr_changed = lr_scale_ != previous;
                decision.penalized = true;
                decision.action = Action::Penalty;
            } else {
                decision.action = Action::Plateau;
            }
        }

        if (adjust_kl && config_.kl_enabled && kl_beta_ > 0.0f) {
            float factor = 1.0f;
            if (kl < config_.kl_target_min) {
                factor = config_.kl_collapse_factor;
                decision.action = Action::KlCollapse;
            } else if (config_.kl_target_max > 0.0f && kl > config_.kl_target_max) {
                factor = config_.kl_excess_factor;
                decision.action = Action::KlExcess;
            } else if (decision.action == Action::Reward) {
                factor = config_.kl_reward_factor;
            } else if (decision.action == Action::Penalty) {
                factor = config_.kl_penalty_factor;
            } else if (decision.action == Action::Plateau) {
                factor = config_.kl_plateau_factor;
            }

            const float previous = kl_beta_;
            kl_beta_ = std::clamp(
                kl_beta_ * factor,
                config_.kl_beta_min,
                config_.kl_beta_max
            );
            decision.kl_changed = kl_beta_ != previous;
        }

        decision.lr_scale = lr_scale_;
        decision.kl_beta = kl_beta_;
        return decision;
    }

    float lrScale() const { return lr_scale_; }
    float klBeta() const { return kl_beta_; }

private:
    static Config sanitize(Config config) {
        config.lr_reward_factor = std::max(1.0f, config.lr_reward_factor);
        config.lr_penalty_factor = std::clamp(config.lr_penalty_factor, 0.0f, 1.0f);
        config.lr_scale_min = std::max(0.0f, config.lr_scale_min);
        config.lr_scale_max = std::max(config.lr_scale_min, config.lr_scale_max);
        config.improve_threshold = std::max(0.0f, config.improve_threshold);
        config.min_steps = std::max(0, config.min_steps);
        config.initial_kl_beta = std::max(0.0f, config.initial_kl_beta);
        config.kl_beta_min = std::max(0.0f, config.kl_beta_min);
        config.kl_beta_max = std::max(config.kl_beta_min, config.kl_beta_max);
        config.initial_kl_beta = std::clamp(
            config.initial_kl_beta,
            config.kl_beta_min,
            config.kl_beta_max
        );
        config.kl_target_min = std::max(0.0f, config.kl_target_min);
        config.kl_target_max = std::max(config.kl_target_min, config.kl_target_max);
        config.kl_reward_factor = std::clamp(config.kl_reward_factor, 0.0f, 1.0f);
        config.kl_penalty_factor = std::clamp(config.kl_penalty_factor, 0.0f, 1.0f);
        config.kl_plateau_factor = std::clamp(config.kl_plateau_factor, 0.0f, 1.0f);
        config.kl_collapse_factor = std::max(1.0f, config.kl_collapse_factor);
        config.kl_excess_factor = std::clamp(config.kl_excess_factor, 0.0f, 1.0f);
        return config;
    }

    Config config_;
    float lr_scale_ = 1.0f;
    float kl_beta_ = 0.0f;
    float best_reconstruction_ = std::numeric_limits<float>::quiet_NaN();
};
