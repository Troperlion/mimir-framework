#pragma once

#include "../Model.hpp"

#include <string>

// VAEConvModel:
// - Convolutional VAE (downsample -> mu/logvar -> reparameterize -> upsample)
// - Output packs: [recon(image_dim), mu(latent_dim), logvar(latent_dim)]
// - Latent is spatial: latent_dim = latent_h * latent_w * latent_c (CHW)

class VAEConvModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;

        int latent_h = 16;
        int latent_w = 16;
        int latent_c = 256;

        int base_channels = 64;

        // Normalization (encoder only)
        // - "none" (default): comportement historique
        // - "layernorm" / "ln": LayerNorm par token spatial (H*W groupes, normalized=C)
        // - "groupnorm" / "gn": GroupNorm sur CHW
        std::string enc_norm = "none";
        int enc_gn_groups = 32;

        // NOTE compat: historiquement, `use_attention` pilotait des blocs d'attention.
        // Dans cette branche, il pilote les blocs ResNet 2D injectés dans le décodeur.
        // On conserve le nom pour éviter de casser les configs / checkpoints existants.
        bool use_attention = true;

        // Garde-fou perf: on n'insère les blocs ResNet que si H*W <= resnet_max_tokens.
        // NOTE compat: historiquement le même champ `attn_max_tokens` servait aussi à ce gating.
        int resnet_max_tokens = 256;

        // Attention (optionnelle) *en plus* des blocs ResNet.
        // Implémentation actuelle: SelfAttention sur tokens spatiaux (H*W) avec embed_dim=C.
        // Par défaut: activé.
        bool use_attn = true;

        // Nombre de heads pour l'attention. Sera automatiquement réduit si non divisible.
        int attn_heads = 4;

        // Garde-fou perf: on n'insère l'attention que si H*W <= attn_max_tokens.
        // Ex: latent 16x16=256 OK, 32x32=1024 potentiellement trop cher.
        int attn_max_tokens = 256;

        // Si false, le latent utilisé par le décodeur est déterministe (z = mu).
        // Si true, on active le bruit de réparameterisation (z = mu + std * eps).
        // Pour des pipelines type SD/SDXL, on veut généralement un encodeur déterministe.
        bool stochastic_latent = false;

        // Optional: texte (conditionnement + alignement). Si false, VAEConv reste image-only.
        bool text_cond = false;
        int vocab_size = 32000;
        int seq_len = 64;
        int text_d_model = 256;
        int proj_dim = 256;   // dim de l'espace commun image/texte pour alignement

        // Optional: perceptual loss (utilisée dans Model::trainStepVAE)
        // IMPORTANT: nécessite recon en HWC (image_w*image_h*image_c).
        float perceptual_weight = 0.0f;
        std::string perceptual_arch = "vgg16_feat";
        std::string perceptual_checkpoint; // vide = init aléatoire
        int perceptual_base_channels = 8;
    };

    VAEConvModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);
    
    // Decoder-only graph: input is z (latent CHW vector), output is recon (RGB vector).
    // Layer names match the decoder portion of buildInto() so checkpoints can be loaded
    // with strict_mode=false (encoder tensors will be ignored).
    static void buildDecoderInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
