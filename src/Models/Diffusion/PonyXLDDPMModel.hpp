#pragma once

#include "../Model.hpp"

#include <random>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

// PonyXLDDPMModel (PonyXL SDXL-like):
// - Entraînement diffusion en espace latent: image -> VAE(mu) = x0, puis
//   x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * eps
//   et on apprend eps_pred = f(x_t, text_ids, timestep) avec loss = MSE(eps_pred, eps).

class PonyXLDDPMModel : public Model {
public:
    struct Config {
        // RNG seed (pour la reproductibilité de l'échantillonnage t/eps et des dropouts)
        int seed = 1337;

        // Texte (SDXL-like)
        int d_model = 256;     // dimension embedding
        int max_vocab = 32000;  // taille max vocab tokenizer (français: recommandé >= 32k)

        // Texte/latents: entrée attendue
        // Format cible: CLIP-like avec ctx_len court (typiquement 77).
        // NOTE: ce modèle consomme des IDs (`text_ids`) et tronque/pad à `text_ctx_len`.
        int text_ctx_len = 77;

        // Si true: rend l'encodeur texte plus proche d'un CLIP text encoder:
        // - self-attention causale
        // - layernorm final
        bool text_clip_like = true;

        // Tokens de contexte global (appris) injectés après <BOS>.
        // Objectif: permettre au modèle d'encoder un prior “dataset global” indépendamment du prompt.
        // NOTE: réduit d'autant le budget de tokens disponibles pour le texte (car on garde <BOS>/<EOS>).
        int global_ctx_tokens = 4;

        // Compression du contexte texte pour mieux gérer les prompts très longs.
        // Si activé, on transforme la séquence (text_ctx_len) en un seul token contexte (mean-pool)
        // avant la cross-attention UNet. Ça réduit énormément le coût et force un “résumé” global.
        bool text_bottleneck_meanpool = false;

        // NOTE: Le VAE texte a été extrait en modèle séparé (`vae_text`).
        // Ce modèle de diffusion ne contient plus de bottleneck variational côté texte.
        // Par défaut on vise le VAE_conv 256x256: latent 64x64x64 => seq=4096, in_dim=64.
        int latent_seq_len = 4096;
        int latent_in_dim = 64;
        int num_heads = 8;

        // Conditionnement temporel (diffusion-like): si true, le modèle attend une entrée float
        // nommée "timestep" (scalaire) et l’injecte via un petit MLP dans les tokens latents.
        bool sdxl_time_cond = true;

        // Représentation du timestep injectée quand sdxl_time_cond=true.
        // - "log_snr" (défaut): utilise logSNR=log(alpha_bar/(1-alpha_bar)), compressé via tanh pour être ~[-1,1].
        //   C'est généralement mieux conditionné que t_norm linéaire.
        // - "t_norm": utilise t normalisé dans [0,1] (comportement historique).
        // - "t_norm_centered": 2*t_norm-1 dans [-1,1].
        std::string timestep_cond = "log_snr";
        // Par défaut: texte réduit, image renforcée.
        int unet_layers = 16;
        int text_layers = 4;
        int mlp_hidden = 1024;

        // UNet 2D (multi-échelle)
        int latent_h = 0;     // 0 = auto (inféré de latent_seq_len)
        int latent_w = 0;     // 0 = auto
        int unet_depth = 4;   // nombre de niveaux down/up

        // Nombre de blocs (conv/conv + cross-attn) par niveau UNet (down et up).
        // 1 = comportement historique.
        int unet_blocks_per_level = 2;
        // Nombre de blocs au bottleneck.
        int unet_bottleneck_blocks = 2;

        // Image (RGB) - dimensions attendues côté dataset loader
        int image_w = 512;
        int image_h = 512;
        int image_c = 3;       // 3 canaux (PNG)

        // Diffusion schedule (DDPM)
        int ddpm_steps = 1000;
        float ddpm_beta_start = 1e-4f;
        float ddpm_beta_end = 2e-2f;

        // Loss utilisée pour l'entraînement DDPM (comparaison eps_pred vs eps)
        // Valeurs supportées (voir Model::computeLoss): "mse", "mae", "huber"/"smoothl1", "charbonnier", "gaussian_nll".
        std::string recon_loss = "mse";

        // === KL regularization (distribution eps_pred) ===
        // Optionnel: pénalise la divergence de eps_pred vers N(0,1) en se basant
        // sur les moments (mean/var) du vecteur eps_pred courant.
        // Objectif: stabiliser l'amplitude/variance de la prédiction et éviter
        // les sorties trop "écrasées" ou trop explosives.
        // Même logique que VAEConv: warmup linéaire de beta sur kl_warmup_steps.
        float kl_beta = 0.0f;
        int kl_warmup_steps = 0;
        // Clip de log-variance (log(var)) utilisé dans ce KL (sécurité numérique)
        float logvar_clip_min = -10.0f;
        float logvar_clip_max = 10.0f;

        // Pondération de loss en fonction du timestep: loss = w(t) * recon_loss(...)
        // - "none" (défaut): w(t)=1
        // - "min_snr": w(t)=min(SNR(t), gamma) / SNR(t) (stabilise l'entraînement en évitant de sur-pondérer les timesteps à très haut SNR)
        // où SNR(t)=alpha_bar/(1-alpha_bar).
        std::string loss_weighting = "none";
        float min_snr_gamma = 5.0f;

        // Activation de sortie du prédicteur eps.
        // - "linear" (défaut): sortie non bornée (Linear + Identity)
        // - "tanh": borne la sortie dans [-1,1]
        // NOTE: en diffusion, une sortie non bornée est standard; tanh est un mode de stabilisation optionnel.
        std::string output_activation = "linear";

        // Nombre d'updates diffusion par image ("par étapes" au sens entraînement).
        // 1 = standard DDPM (un timestep aléatoire par image).
        int ddpm_steps_per_image = 1;

        // Bruit "Peltier" (ancien): historiquement on mixait un bruit corrélé (blur) au bruit blanc.
        // Désormais DÉPRÉCIÉ/IGNORÉ: PonyXL-DDPM se comporte comme un DDPM latent standard
        // avec eps ~ N(0, I). Les champs restent pour compatibilité des anciens checkpoints/config.
        bool peltier_noise = false;
        float peltier_mix = 0.65f;
        int peltier_blur_radius = 2;

        // === VAE (Stable Diffusion-like) ===
        // En mode sdxl_like, l'image du dataset est encodée via un VAE externe (pré-entraîné)
        // pour obtenir x0 en espace latent. Aucun placeholder.
        std::string vae_arch = "vae_conv";
        std::string vae_checkpoint = "";
        float vae_scale = 1.0f;
        // Optionnel: permet d'aligner l'architecture VAE (ex: vae_conv base_channels)
        // avec un checkpoint pré-entraîné.
        int vae_base_channels = 0;

        // CFG dropout: parfois on retire le prompt
        float cfg_dropout_prob = 0.10f;

        // Prompt preprocessing (optionnel)
        int max_text_chars = 262144;
        float dropout = 0.0f;

        // === Captions structurées (dataset texte) ===
        // Supporte des sections optionnelles sous forme:
        //   --- TAGS ---
        //   ...
        //   --- CONTEXTE ---
        //   ...
        //   --- MENTALITÉ ---
        //   ...
        //   --- TEXTE (langue...) ---
        //   --- DESCRIPTION --- (alias accepté pour compatibilité captioners)
        //   ...
        // Si aucune balise n'est détectée, le prompt est utilisé tel quel.
        bool caption_structured_enable = true;
        // Si true, on recompose une forme canonique "TAGS: ...\nCONTEXTE: ...".
        // Si false, on conserve le prompt original (mais on peut appliquer le dropout par section).
        bool caption_structured_canonicalize = true;

        // Dropout par section (appliqué uniquement pendant l'entraînement).
        float caption_tags_dropout_prob = 0.0f;
        float caption_contexte_dropout_prob = 0.0f;
        float caption_mentalite_dropout_prob = 0.0f;
        float caption_texte_dropout_prob = 0.0f;

        // === Captions key/value (format simple) ===
        // Supporte le format:
        //   caption: ...
        //   theme: ...
        //   keywords: ...
        //   tokens: ...
        // Utilisé si aucune balise "--- SECTION ---" n'est détectée.
        bool caption_kv_enable = true;

        // === Majoration des termes fréquents (conditioning) ===
        // Objectif: renforcer la corrélation texte-image en sur-représentant
        // les termes les plus fréquents du dataset pendant l'entraînement.
        // Implémentation: répétition contrôlée de certains tokens avant padding/troncature.
        bool term_freq_boost_enable = true;
        // Source: on préfère compter les termes explicites de `tokens:` / `keywords:` si présents.
        bool term_freq_boost_use_tokens = true;
        bool term_freq_boost_use_keywords = true;
        // Nombre de tokens "top" considérés comme fréquents.
        int term_freq_boost_top_k = 64;
        // Nombre de répétitions ajoutées (1 => token apparaît 2 fois).
        int term_freq_boost_repeat = 1;
        // Warmup en steps d'optimizer avant d'activer la majoration.
        int term_freq_boost_start_step = 100;
        // Recalcule du top-K toutes les N steps.
        int term_freq_boost_update_every_steps = 200;

        // === Viz: séquence DDPM (noising/denoising) ===
        // 0 = désactivé. Sinon: toutes les N steps d'optimizer, on génère quelques frames
        // montrant x_t (noisé) et x0_hat (reconstruit) pour plusieurs timesteps.
        int viz_ddpm_every_steps = 0;
        // Nombre de timesteps affichés (linéairement espacés de 0 à T-1).
        int viz_ddpm_num_steps = 5;

        // === Optional: image-space auxiliary loss (via VAE decoder) ===
        // Objectif: pousser le modèle de diffusion à produire des latents qui, une fois
        // décodés, matchent mieux l'image originale (malgré un décodeur VAE imparfait).
        // Coûteux: nécessite forward+backward dans le décodeur VAE. Désactivé par défaut.
        float img_loss_weight = 0.0f;
        int img_loss_every_steps = 0; // 0 = désactivé, sinon appliqué toutes les N steps d'optimizer (sur la dernière inner-step)
    };

    struct StepStats {
        float loss = 0.0f;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;

        // DDPM timestep (normalisé [0..1] quand utilisé, sinon 0)
        float timestep = 0.0f;

        // KL warmup: beta effectif (après warmup)
        float kl_beta_effective = 0.0f;

        float kl_divergence = 0.0f;
        float wasserstein = 0.0f;
        float entropy_diff = 0.0f;
        float moment_mismatch = 0.0f;
        float spatial_coherence = 0.0f;
        float temporal_consistency = 0.0f;
    };

    PonyXLDDPMModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    // Permet de modifier l'échelle VAE après création du modèle (ex: calibration auto).
    void setVaeScale(float s) { cfg_.vae_scale = s; }

    // Live tuning: permet de modifier à chaud les paramètres KL.
    // NOTE: ce modèle lit le KL depuis cfg_ (et non via Model::modelConfig).
    void setLiveKL(float kl_beta, int kl_warmup_steps);

    // Accumule les moments (sum, sumsq, n) sur le vecteur mu du VAE pour une image.
    // mu est extrait du packed output [image || mu || logvar] du VAE.
    void accumulateVaeMuMoments(const std::vector<uint8_t>& rgb,
                                int w,
                                int h,
                                double& sum,
                                double& sumsq,
                                size_t& n);

    // Entraînement diffusion latent SDXL-like (eps predictor)
    StepStats trainStepSdxlLatentDiffusion(const std::string& prompt,
                                           const std::vector<uint8_t>& rgb,
                                           int w,
                                           int h,
                                           Optimizer& opt,
                                           float learning_rate);

    // Validation (forward-only): calcule MSE(eps_pred, eps) et MSE(x0_hat, x0) et une marge
    // d'association texte->image (comparaison prompt correct vs prompt "mauvais").
    struct ValStats {
        double eps_mse = 0.0;
        double x0_mse = 0.0;
        // Reconstruction en espace image (après VAE decode de x0_hat), MSE sur [-1,1].
        double img_mse = 0.0;
        double eps_mse_wrong = 0.0;
        double assoc_margin = 0.0; // eps_mse_wrong - eps_mse
        float t_norm = 0.0f;
    };
    ValStats validateStepSdxlLatentDiffusion(const std::string& prompt,
                                             const std::string& wrong_prompt,
                                             const std::vector<uint8_t>& rgb,
                                             int w,
                                             int h,
                                             int seed = 12345,
                                             int ddpm_step = -1);

    // Recon preview (forward-only): produit une image RGB reconstruite via VAE decode
    // à partir de x0_hat (débruitage au timestep fixe utilisé en validation).
    struct ReconPreview {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 3;
    };
    ReconPreview reconstructPreviewSdxlLatentDiffusion(const std::string& prompt,
                                                       const std::vector<uint8_t>& rgb,
                                                       int w,
                                                       int h,
                                                       int max_side = 256,
                                                       int seed = 12345,
                                                       int ddpm_step = -1);

    static void buildInto(Model& model, const Config& cfg);

    static std::vector<float> imageBytesToFloatRGB(const std::vector<uint8_t>& rgb, int w, int h);

private:
    Config cfg_;

    // RNG propre au modèle (seedé depuis cfg_.seed) pour éviter un seed codé en dur.
    std::mt19937 rng_;

    // VAE chargé pour encoder les images en latents
    std::shared_ptr<Model> vae_;

    // VAE decoder (pour visualizer reconstructions en espace image, lazy-load)
    std::shared_ptr<Model> vae_decode_;

    // Stats de fréquence (incrémentales) pour la majoration pendant l'entraînement.
    // Clé: token id (dans le tokenizer courant)
    std::unordered_map<int, uint64_t> term_freq_counts_;
    std::unordered_set<int> term_freq_top_set_;
    std::vector<int> term_freq_top_ids_;
    int64_t term_freq_last_rebuild_step_ = -1;
};
