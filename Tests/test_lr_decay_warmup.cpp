#include "test_utils.hpp"
#include "Model.hpp"

int main() {
    Optimizer opt;
    opt.initial_lr = 1.0f;
    opt.warmup_steps = 4;
    opt.decay_strategy = LRDecayStrategy::NONE;
    TASSERT_NEAR(opt.getCurrentLR(), 0.25f, 1e-6f);
    opt.step = 1; TASSERT_NEAR(opt.getCurrentLR(), 0.50f, 1e-6f);
    opt.step = 2; TASSERT_NEAR(opt.getCurrentLR(), 0.75f, 1e-6f);
    opt.step = 3; TASSERT_NEAR(opt.getCurrentLR(), 1.00f, 1e-6f);
    opt.step = 4; TASSERT_NEAR(opt.getCurrentLR(), 1.00f, 1e-6f);
    return 0;
}
