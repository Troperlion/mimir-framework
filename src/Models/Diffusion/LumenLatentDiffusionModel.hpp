#pragma once

#include "../Model.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class LumenLatentDiffusionModel : public Model {
public:
    struct Config {
        int seed = 1337;
        int image_w = 512;
        int image_h = 512;
        int image_c = 3;
        int latent_w = 64;
        int latent_h = 64;
        int latent_c = 4;
        int vae_base_channels = 8;
        bool vae_latent_spatial_context = false;
        bool vae_latent_detail_path = false;
        int vae_latent_context_channels = 32;
        bool vae_stochastic_latent = true;
        bool vae_resnet = true;
        bool vae_attention = false;
        bool vae_use_skip_connections = false;
        bool vae_use_encoder_prior = false;
        std::string vae_enc_norm = "none";
        std::string vae_dec_norm = "groupnorm";
        std::string vae_decoder_upsample = "nearest_conv";
        int vae_enc_gn_groups = 8;
        int vae_dec_gn_groups = 32;
        int vae_attn_heads = 4;
        int vae_attn_max_tokens = 0;
        int vae_resnet_max_tokens = 4096;
        float vae_scale = 0.0f;
        float vae_shift = 0.0f;
        std::string vae_checkpoint;
        int patch_size = 4;
        int hidden_size = 384;
        int depth = 8;
        float mlp_ratio = 4.0f;

        int vocab_size = 32000;
        int text_seq_len = 77; // reference positional table, not an input limit in dynamic mode
        bool dynamic_text_seq_len = true;
        int text_layers = 2;
        int num_heads = 8;

        int diffusion_steps = 1000;
        float beta_start = 0.00085f;
        float beta_end = 0.012f;
        int preview_timestep = 50; // legacy checkpoint field
        float kl_beta = 0.0f;
        int kl_warmup_steps = 0;
    };

    struct GeneratedImage {
        std::vector<unsigned char> pixels;
        int w = 0;
        int h = 0;
        int channels = 0;
    };

    struct TrainStats {
        float loss = 0.0f;
        float mse = 0.0f;
        float mse_unweighted = 0.0f;
        size_t bbox_count = 0;
        size_t skeleton_region_count = 0;
        size_t relation_count = 0;
        float relation_loss = 0.0f;
        float bbox_loss_weight = 0.0f;
        float kl = 0.0f;
        float kl_beta_effective = 0.0f;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;
        float reconstruction_mae = 0.0f;
        float reconstruction_mse = 0.0f;
        float wasserstein = 0.0f;
        float entropy_diff = 0.0f;
        float moment_mismatch = 0.0f;
        float spatial_coherence = 0.0f;
        float temporal_consistency = 0.0f;
        int timestep = 0;
    };

    struct VaeCalibrationStats {
        size_t items = 0;
        size_t values = 0;
        float shift = 0.0f;
        float scale = 0.0f;
    };

    LumenLatentDiffusionModel();
    std::shared_ptr<SkipConnectionControl> skipConnectionControl() override;

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }
    std::optional<TrainStepResult> trainStep(const TrainStepRequest& request) override;

    const std::vector<float>& forwardPassNamedView(
        const std::unordered_map<std::string,std::vector<float>>& float_inputs,
        const std::unordered_map<std::string,std::vector<int>>& int_inputs,
        bool training = true) override;

    GeneratedImage generate(const std::string& prompt,
                            int seed,
                            int sample_steps = 30,
                            float guidance_scale = 5.0f,
                            const std::function<void(int, int)>& progress = {});
    TrainStats trainDiffusionStep(const std::vector<unsigned char>& rgb_image,
                                  const std::string& prompt,
                                  unsigned int seed,
                                  Optimizer& optimizer,
                                  float learning_rate,
                                  const json* annotation = nullptr,
                                  float bbox_loss_weight = 1.0f,
                                  bool drop_conditioning = false);
    TrainStats validateDiffusionStep(const std::vector<unsigned char>& image,
                                     const std::string& prompt,
                                     unsigned int seed,
                                     const json* annotation = nullptr,
                                     float bbox_loss_weight = 1.0f);
    void beginVaeCalibration();
    VaeCalibrationStats addVaeCalibrationImage(
        const std::vector<unsigned char>& rgb_image);
    VaeCalibrationStats finishVaeCalibration();

    bool InitVizTips() override;
    bool UpdateVizTips(const Layer& layer, VizFrame& frame) override;

    static void buildInto(Model& model, const Config& cfg);

protected:
    void addDiffusionVizTips(const std::vector<float>& generated_image,
                             const std::vector<unsigned char>& original_image);
    void addDiffusionComparisonVizTips(
        const std::vector<float>& oracle_image,
        const std::vector<float>& noisy_baseline_image,
        const std::vector<float>& predicted_image,
        int timestep);
    std::vector<float> encodeImage(const std::vector<float>& image);
    std::vector<float> decodeImage(const std::vector<float>& rgb_chw,
                                   bool use_encoder_skips = false);

private:
    struct DiffusionExample {
        std::vector<float> clean, noise, noisy;
        int timestep = 0;
    };
    DiffusionExample prepareExample(const std::vector<unsigned char>& pixels, unsigned int seed);
    std::vector<int> promptIds(const std::string& prompt) const;
    std::vector<int> conditionedPromptIds(const std::string& prompt, const json* annotation) const;
    std::vector<float> timestepEmbedding(int timestep, int dim) const;
    std::vector<float> encodeImageUncalibrated(const std::vector<float>& image);
    void appendReconstructionGraph();
    Config cfg_;
    std::unique_ptr<Model> vae_encoder_;
    std::unique_ptr<Model> vae_decoder_;
    std::string vae_encoder_output_;
    struct VaeSkipBinding {
        std::string decoder_input;
        std::string encoder_output;
        size_t values = 0;
    };
    std::vector<VaeSkipBinding> vae_skip_bindings_;
    std::unordered_map<std::string, std::vector<float>> vae_encoder_skips_;
    size_t vae_calibration_items_ = 0;
    size_t vae_calibration_values_ = 0;
    double vae_calibration_mean_ = 0.0;
    double vae_calibration_m2_ = 0.0;
};
