#include "test_utils.hpp"
#include "Model.hpp"

int main() {
    Optimizer opt;
    opt.initial_lr = 0.1f;
    opt.min_lr = 0.01f;
    opt.warmup_steps = 2;
    opt.total_steps = 6;

    opt.decay_strategy = LRDecayStrategy::LINEAR;
    opt.step = 2; TASSERT_NEAR(opt.getCurrentLR(), 0.1f, 1e-6f);
    opt.step = 6; TASSERT_NEAR(opt.getCurrentLR(), 0.01f, 1e-6f);

    opt.decay_strategy = LRDecayStrategy::STEP;
    opt.decay_steps = 0; // doit rester sûr et se comporter comme 1
    opt.decay_rate = 0.5f;
    opt.step = 3; TASSERT_NEAR(opt.getCurrentLR(), 0.05f, 1e-6f);
    return 0;
}
