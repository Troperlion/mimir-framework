#pragma once

#include "../../Model.hpp"

// Decoder-only language model implemented entirely with native Mimir layers.
// Inputs: token ids in "__input__". Output: [seq_len, vocab_size] logits.
class CausalLMModel : public Model {
public:
    struct Config {
        int vocab_size = 4096;
        int seq_len = 128;
        int d_model = 256;
        int num_layers = 6;
        int num_heads = 8;
        int num_kv_heads = 2;
        int mlp_hidden = 704;
        int padding_idx = 0;
        float norm_eps = 1e-5f;
        float rope_theta = 10000.0f;
    };

    CausalLMModel();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);
    const Config& getConfig() const { return cfg_; }

private:
    Config cfg_;
};
