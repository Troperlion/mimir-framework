---@meta
---@version 2.4.0
---@author <bri45> for "Mímir Framework"
---@date 15 mars 2026 (dernière sync)
---@diagnostic disable: missing-return, unused-local, unused-vararg, duplicate-doc-field, redundant-parameter

--=============================================================================
-- Mímir Framework v2.4 — IDE Stub (EmmyLua)
--=============================================================================
-- Ce fichier est un "stub" destiné aux IDE (LuaLS / EmmyLua / IntelliJ, etc.).
-- Il documente l'API globale exposée par le binaire `mimir` (bindings C/C++).
--
-- ⚠️  IMPORTANT: Ce fichier est synchronisé avec src/LuaScripting.cpp
--    Toute modification de l'API C++ doit être reflétée ici.
--    Dernière synchronisation: 28 février 2026 - 14 modules, 120+ fonctions
--
-- Objectifs :
--  • Autocomplétion IDE, signatures, types, docstrings
--  • Stable : les scripts Lua sont l'API publique
--  • Documentation de référence pour les utilisateurs
--
-- 🆕 Nouveautés v2.4.0 (15 mars 2026) :
--  • Synchronisation doc/scripts/stub/template
--  • Ajout du fichier VERSION pour exposer la version runtime
--  • Ajustements packaging (gitignore) pour éviter les artefacts dans les commits
--
-- Nouveautés v2.3.0 (28 décembre 2025) :
--  • Multi-Input / Branch Support - TensorStore system
--  • model.set_layer_io() - Configuration des entrées/sorties des layers
--  • Operations multi-input complètes (Add, Multiply, Concat, MatMul, Split)
--  • Support residual connections et skip connections
--  • Documentation complète (MULTI_INPUT_SUPPORT.md)
--  • Mode Strict activé (0 pass-through, RUNTIME_CHECK/RUNTIME_ERROR_STRICT)
--
-- Nouveautés v2.1.0 :
--  • Scripts réorganisés en catégories (demos, examples, tests, etc.)
--  • Documentation corrigée (33 fixes)
--  • Synchronisation API validée (114 fonctions)
--
-- Historique v2.0.0 :
--  • MemoryGuard API (limite stricte 10 GB par défaut)
--  • Mimir.Allocator.configure() pour configuration mémoire (OBLIGATOIRE)
--  • Modes train()/eval() pour tous les modèles
--  • API Mimir.Htop et Mimir.Viz pour monitoring temps réel
--
-- Remarques :
--  • Les fonctions retournent souvent (ok:boolean, value) ou (ok:boolean, err:string)
--  • ⚠️  Toujours appeler Mimir.Allocator.configure() au début des scripts!
--  • Les tables de stats sont des structs (tables Lua avec champs nommés)
--  • Les "Mimir.Layers.*" sont des opérations bas niveau (préférer Mimir.Model.forward())
--  • Mimir.MemoryGuard: API moderne recommandée (Mimir.Guard: API legacy compatible)
--
-- CLI (qualité de vie / IDE) :
--  • `mimir --config <config.json> --override <path=value> [--override ...]`
--    Permet de surcharger une config sans éditer le JSON (pratique pour les launch configs).
--  • Côté scripts Lua, beaucoup de scripts utilisent `scripts/modules/args.lua`.
--    Ce parseur supporte aussi `--override key=value` (si le script appelle `Args.apply_overrides(cfg, opts)`).
--    Exemples:
--      mimir --config config.json --override max_vocab=64000
--      mimir --config config.json --override optimizer=\"adamw\" --override weight_decay=0.01
--=============================================================================

--=============================================================================
-- Namespace Mimir
--=============================================================================

---@class Mimir
Mimir = {}

---@class Mimir
---@field Model MimirModelAPI
---@field Architectures MimirArchitecturesAPI
---@field Layers MimirLayersAPI
---@field Checkpoint MimirCheckpointAPI
---@field Tokenizer MimirTokenizerAPI
---@field Dataset MimirDatasetAPI
---@field Memory MimirMemoryAPI
---@field Guard MimirGuardAPI
---@field MemoryGuard MimirMemoryGuardAPI
---@field Allocator MimirAllocatorAPI
---@field Htop MimirHtopAPI
---@field Viz MimirVizAPI
---@field Serialization MimirSerializationAPI
---@field NeuroPulse MimirNeuroPulseAPI

--=============================================================================
-- Aliases / Types de base
--=============================================================================

---@alias int integer
---@alias float number
---@alias bool boolean
---@alias bytes integer

---@alias TokenId integer
---@alias TokenIds TokenId[]     -- liste d'IDs token (indices Lua 1..N)
---@alias TokenText string

---@alias ModelType
---| "t2i_autoencoder"
---| "ponyxl_ddpm"
---| "ponyxl_sdxl_stub"
---| "ponyxl_sdxl_unet2d"
---| "ponyxl_sdxl"
---| "basic_mlp"
---| "transformer"
---| "vae_text"
---| "vae_text_decode"
---| "vit"
---| "vae"
---| "vae_conv"
---| "vae_conv_decode"
---| "resnet"
---| "unet"
---| "mobilenet"
---| "vgg16"
---| "vgg19"
---| "diffusion"
---| "cond_diffusion"
---| "sd3_5"
---| "neuropulse"
---| "gan_latent"

---@alias ArchitectureName
---| "basic_mlp"
---| "transformer"
---| "vae_text"
---| "vae_text_decode"
---| "vit"
---| "vae"
---| "vae_conv"
---| "vae_conv_decode"
---| "resnet"
---| "unet"
---| "mobilenet"
---| "vgg16"
---| "vgg19"
---| "ponyxl_ddpm"
---| "diffusion"
---| "gan_latent"
---| "cond_diffusion"
---| "sd3_5"

---Lire un JSON depuis le disque.
---@param path string
---@return table|nil json
function read_json(path) end

---Écrire un fichier (bytes string) sur le disque.
---@param path string
---@param content string
---@return boolean ok
---@return string? err
function write_file(path, content) end

---@alias ActivationType
---| "relu"
---| "gelu"
---| "silu"
---| "swish"
---| "tanh"
---| "sigmoid"
---| "softmax"

---@alias WeightInit
---| "xavier"
---| "he"
---| "normal"
---| "uniform"
---| "zeros"

---@alias HardwareAccelMode boolean

---@alias KeywordList string[]

--=============================================================================
-- Configs de modèles
--=============================================================================

---@class ModelConfig
---@field type? ModelType @Injecté côté C++ lors de `Model.create()` (metadata)
---@field dropout? float @Dropout générique (si supporté par l'architecture)
---@field optimizer? string @"sgd"|"adam"|"adamw" (utilisé par `Model.train()`)
---@field beta1? float
---@field beta2? float
---@field epsilon? float
---@field weight_decay? float
---@field min_lr? float
---@field decay_rate? float
---@field decay_steps? int
---@field warmup_steps? int
---@field decay_strategy? string @"none"|"cosine"|"step"|"exponential"|"linear"
---@field checkpoint_dir? string @Répertoire de checkpoints (utilisé par `Model.train()` et features de robustesse/validation)
---@field max_items? int @Limite d'items par epoch (0=pas de limite) - lu par `Model.train()`
---@field log_every? int @Intervalle de logs (steps) - lu par `Model.train()`
---@field seed? int @Seed générique - lu par `Model.train()` et certains helpers
---@field autosave_every_epochs? int @Autosave checkpoint toutes les N epochs (0=désactivé)
---@field autosave_every_epoch? int @Alias de autosave_every_epochs
---@field viz_taps_max_frames? int @Limite frames "viz taps" (si viz active)
---@field viz_taps_max_side? int @Limite taille preview "viz taps" (si viz active)
---@field validate_every_steps? int @Validation toutes les N étapes d'optimizer (0 = désactivé)
---@field validate_items? int @Nombre d'items évalués par validation
---@field validate_holdout? bool @Active un split holdout (si supporté par le script/archi)
---@field validate_holdout_frac? float @Fraction du dataset allouée au holdout
---@field validate_holdout_items? int @Nombre d'items holdout (optionnel, selon script)
---@field validate_save_debug? bool @Sauvegarde un checkpoint debug (ex: _val_debug.json) lors des validations
---@field triple_fault? bool @Active le mécanisme de rollback sur dernier checkpoint "bon"
---@field triple_fault_every_steps? int @Intervalle (en steps) de sauvegarde des checkpoints triple-fault

---@class BasicMLPConfig: ModelConfig
---@field input_dim? int
---@field hidden_dim? int
---@field output_dim? int
---@field hidden_layers? int

---@class TransformerConfig: ModelConfig
---@field seq_len? int
---@field d_model? int
---@field vocab_size? int
---@field padding_idx? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_hidden? int
---@field output_dim? int
---@field causal? bool

---@class VAETextConfig: ModelConfig
---@field vocab_size? int
---@field padding_idx? int
---@field seq_len? int
---@field d_model? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_hidden? int
---@field latent_tokens? int
---@field proj_dim? int
---@field stochastic_latent? bool
---@field dropout? float
-- Training helper defaults (Model::trainStepVAEText)
---@field image_dim? int
---@field output_dim? int
---@field target_tensor? string
---@field align_weight? float
---@field kl_beta? float
---@field kl_warmup_steps? int
---@field recon_loss? string @"ce"|"mse"
---@field logvar_clip_min? float
---@field logvar_clip_max? float

---@class ViTConfig: ModelConfig
---@field num_tokens? int @Nombre de tokens/patches (entrée = num_tokens*d_model)
---@field d_model? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_hidden? int
---@field output_dim? int
---@field causal? bool

---@class VAEConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field latent_dim? int
---@field hidden_dim? int

---@class VAEConvConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field latent_h? int
---@field latent_w? int
---@field latent_c? int
---@field latent_dim? int @latent_h*latent_w*latent_c (souvent rempli côté scripts)
---@field base_channels? int
---@field use_attention? bool @Compat: pilote les blocs ResNet injectés dans le décodeur (anciennement des blocs d'attention)
---@field resnet_max_tokens? int @Garde-fou perf ResNet: on n'insère des blocs que si H*W <= resnet_max_tokens
---@field use_attn? bool @SelfAttention optionnelle (en plus des blocs ResNet)
---@field attn_heads? int @Heads SelfAttention
---@field attn_max_tokens? int @Garde-fou perf SelfAttention: H*W <= attn_max_tokens
---@field stochastic_latent? bool
---@field text_cond? bool
---@field vocab_size? int
---@field seq_len? int
---@field text_d_model? int
---@field proj_dim? int
---@field align_weight? float
---@field kl_beta? float
---@field kl_warmup_steps? int
---@field recon_loss? string @"l1"|"l2"|"mse"|"huber"|"charbonnier"|"nll" (selon impl.)
---@field logvar_clip_min? float
---@field logvar_clip_max? float
---@field grad_clip_norm? float
---@field grad_accum_steps? int
---@field ssim_weight? float
---@field ssim_mode? string
---@field ssim_k1? float
---@field ssim_k2? float
---@field ssim_L? float
---@field spectral_weight? float
---@field spectral_scales? int
---@field perceptual_weight? float
---@field perceptual_arch? string
---@field perceptual_checkpoint? string
---@field perceptual_base_channels? int
---@field huber_delta? float
---@field charbonnier_eps? float
---@field nll_sigma? float
---@field marker_wass_scale? float
---@field marker_temp_scale? float
---@field marker_warmup_steps? int
---@field marker_scale_max? float
---@field tokenizer_frozen? bool

---@class ResNetConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int
---@field blocks1? int
---@field blocks2? int
---@field blocks3? int
---@field blocks4? int

---@class UNetConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field depth? int

---@class MobileNetConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int

---@class VGG16Config: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int
---@field fc_hidden? int

---@class VGG19Config: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int
---@field fc_hidden? int

---@class DiffusionConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field time_dim? int
---@field hidden_dim? int

---@class CondDiffusionConfig: ModelConfig
---@field prompt_dim? int
---@field latent_w? int
---@field latent_h? int
---@field latent_c? int
---@field time_dim? int
---@field hidden_dim? int

---@class GanLatentConfig: ModelConfig
---@field prompt_dim? int
---@field noise_dim? int
---@field latent_dim? int
---@field hidden_dim? int
---@field num_hidden_layers? int

---@class SD35Config: ModelConfig
---@field stub_only? bool
---@field q_len? int
---@field kv_len? int
---@field d_model? int
---@field num_heads? int
---@field num_layers? int
---@field mlp_hidden? int
---@field causal? bool

---@class PonyXLDDPMConfig: ModelConfig
---@field d_model? int @Dimension embedding texte
---@field max_vocab? int @Capacité max du tokenizer (important: aligner avec Tokenizer.get_max_vocab())
---@field text_ctx_len? int @Contexte texte max (tokens)
---@field text_bottleneck_meanpool? bool @Compression du prompt long en 1 token contexte (mean-pool)
---@field latent_seq_len? int @Séquence latente (ex: 64*64=4096)
---@field latent_in_dim? int @Canaux latents (ex: 64)
---@field num_heads? int
---@field sdxl_time_cond? bool @Injecte le timestep via petit MLP (SDXL-like)
---@field unet_layers? int
---@field text_layers? int
---@field mlp_hidden? int
---@field latent_h? int @0=auto (inféré depuis latent_seq_len)
---@field latent_w? int @0=auto
---@field unet_depth? int @Nombre de niveaux down/up (UNet 2D)
---@field image_w? int @Largeur image attendue côté dataset
---@field image_h? int @Hauteur image
---@field image_c? int @Canaux image (3=RGB)
---@field ddpm_steps? int @T timesteps
---@field ddpm_beta_start? float
---@field ddpm_beta_end? float
---@field ddpm_steps_per_image? int @Nb updates diffusion par image (1=standard)
---@field recon_loss? string @Loss diffusion eps_pred vs eps (ex: "mse","mae","huber"/"smoothl1","charbonnier","gaussian_nll")
---@field loss_weighting? string @Pondération par timestep ("none"|"min_snr")
---@field min_snr_gamma? float @Gamma utilisé si loss_weighting="min_snr"
---@field output_activation? string @Activation de sortie eps_pred ("linear"|"tanh")
---@field kl_beta? float @Regularisation KL (eps_pred -> N(0,1))
---@field kl_warmup_steps? int @Warmup linéaire de kl_beta (comme VAE_conv)
---@field logvar_clip_min? float @Clip log(var) utilisé dans le KL (sécurité numérique)
---@field logvar_clip_max? float @Clip log(var) utilisé dans le KL
---@field peltier_noise? bool @DÉPRÉCIÉ/IGNORÉ: ancien "bruit peltier" (corrélé). DDPM latent standard utilise eps~N(0,I).
---@field peltier_mix? float @DÉPRÉCIÉ/IGNORÉ
---@field peltier_blur_radius? int @DÉPRÉCIÉ/IGNORÉ
---@field vae_arch? string @Archi VAE externe (ex: "vae_conv")
---@field vae_checkpoint? string @Chemin checkpoint VAE
---@field vae_scale? float @Échelle latents VAE
---@field vae_base_channels? int @Optionnel: aligner l'archi VAE avec le checkpoint
---@field cfg_dropout_prob? float @Dropout prompt (CFG training)
---@field max_text_chars? int @Clamp taille texte en chars (pré-tokenization)
---@field caption_structured_enable? bool @Active le parsing des captions multi-sections (`--- TAGS ---`, etc.)
---@field caption_structured_canonicalize? bool @Recompose un prompt canonique `TAGS:/CONTEXTE:/MENTALITE:/TEXTE:`
---@field caption_tags_dropout_prob? float @Dropout de la section TAGS (train uniquement)
---@field caption_contexte_dropout_prob? float @Dropout de la section CONTEXTE (train uniquement)
---@field caption_mentalite_dropout_prob? float @Dropout de la section MENTALITÉ (train uniquement)
---@field caption_texte_dropout_prob? float @Dropout de la section TEXTE (train uniquement)
---@field viz_ddpm_every_steps? int @0=off, sinon: toutes les N steps on affiche des frames DDPM
---@field viz_ddpm_num_steps? int @Nb timesteps affichés

-- NOTE: les schémas ci-dessus reflètent le registre C++ (ModelArchitectures::defaultConfig).
-- Pour obtenir la config exacte à jour côté runtime:
--   local cfg = Mimir.Architectures.default_config("transformer")
--   local ok = Mimir.Model.create("transformer", cfg)

--=============================================================================
-- Stats / Structs
--=============================================================================

---@class HardwareCaps
---@field avx2 bool @Support AVX2 SIMD instructions
---@field fma bool @Support FMA (Fused Multiply-Add)
---@field f16c bool @Support Half-precision floats
---@field bmi2 bool @Support BMI2 instructions
---@field vulkan_compiled? bool @Backend Vulkan compilé dans le binaire
---@field opencl_compiled? bool @Backend OpenCL compilé dans le binaire

---@class MemoryStats
---@field current_mb float
---@field peak_mb float
---@field usage_percent float

---@class GuardStats
---@field current_mb float
---@field peak_mb float
---@field limit_mb float
---@field usage_percent float

---@class AllocatorStats
---@field tensor_count int @Nombre de tensors alloués
---@field loaded_count int @Nombre de tensors chargés en RAM

---@class HtopMetrics
---@field epoch int
---@field total_epochs int
---@field batch int
---@field total_batches int
---@field loss float
---@field avg_loss float
---@field lr float
---@field batch_time_ms int
---@field memory_mb int
---@field memory_freed int
---@field bps float
---@field params int
---@field timestep float
---@field kl float
---@field wass float
---@field ent float
---@field mom float
---@field spat float
---@field temp float
---@field mse float
---@field grad_norm? float
---@field grad_max? float
---@field opt_type? int
---@field opt_step? int
---@field opt_beta1? float
---@field opt_beta2? float
---@field opt_eps? float
---@field opt_weight_decay? float

---@class HtopCreateConfig
---@field enable_viz? boolean @Active la viz SFML (alias: viz)
---@field viz? boolean @Alias de enable_viz
---@field viz_config? table @Config passée au Visualizer (mêmes clés que config.json/visualization)
---@field csv? boolean @Active l'export CSV des métriques côté htop
---@field csv_enabled? boolean @Alias de csv
---@field csv_path? string @Chemin du CSV (ex: "checkpoint/run1/htop_metrics.csv")
---@field csv_file? string @Alias de csv_path

---@class VizMetrics
---@field epoch? int
---@field total_epochs? int
---@field loss? float
---@field avg_loss? float
---@field lr? float
---@field memory_mb? float
---@field tokens_per_sec? float
---@field custom? table<string, number>

--=============================================================================
-- Module: Mimir.Model
--=============================================================================

---@class MimirModelAPI
Mimir.Model = {}

---Créer un modèle via le registre d'architectures (C++).
---Le modèle est construit via le registre C++ immédiatement.
---Note: `Mimir.Model.create()` ne fait plus allocate/init automatiquement.
---Utilisez ensuite `Mimir.Model.allocate_params()` et `Mimir.Model.init_weights()` si nécessaire.
---@overload fun(model_type: "basic_mlp", config?: BasicMLPConfig): (boolean, string?)
---@overload fun(model_type: "transformer", config?: TransformerConfig): (boolean, string?)
---@overload fun(model_type: "vae_text", config?: VAETextConfig): (boolean, string?)
---@overload fun(model_type: "vae_text_decode", config?: VAETextConfig): (boolean, string?)
---@overload fun(model_type: "vit", config?: ViTConfig): (boolean, string?)
---@overload fun(model_type: "vae", config?: VAEConfig): (boolean, string?)
---@overload fun(model_type: "vae_conv", config?: VAEConvConfig): (boolean, string?)
---@overload fun(model_type: "vae_conv_decode", config?: VAEConvConfig): (boolean, string?)
---@overload fun(model_type: "resnet", config?: ResNetConfig): (boolean, string?)
---@overload fun(model_type: "unet", config?: UNetConfig): (boolean, string?)
---@overload fun(model_type: "mobilenet", config?: MobileNetConfig): (boolean, string?)
---@overload fun(model_type: "vgg16", config?: VGG16Config): (boolean, string?)
---@overload fun(model_type: "vgg19", config?: VGG19Config): (boolean, string?)
---@overload fun(model_type: "diffusion", config?: DiffusionConfig): (boolean, string?)
---@overload fun(model_type: "cond_diffusion", config?: CondDiffusionConfig): (boolean, string?)
---@overload fun(model_type: "gan_latent", config?: GanLatentConfig): (boolean, string?)
---@overload fun(model_type: "sd3_5", config?: SD35Config): (boolean, string?)
---@overload fun(model_type: "ponyxl_ddpm", config?: PonyXLDDPMConfig): (boolean, string?)
---@param model_type ModelType
---@param config? ModelConfig|BasicMLPConfig|TransformerConfig|ViTConfig|VAEConfig|ResNetConfig|UNetConfig|MobileNetConfig|VGG16Config|VGG19Config|DiffusionConfig|PonyXLDDPMConfig|table
---@return boolean ok
---@return string? err
function Mimir.Model.create(model_type, config) end

---[COMPAT] Reconstruit le modèle courant via le registre.
---Préférez `Mimir.Model.create(type, cfg)`.
---Retour: ok + nombre de paramètres (scalars).
---@return boolean ok
---@return integer? params
---@return string? err
function Mimir.Model.build() end

---Entraîner le modèle courant.
---Nécessite un dataset chargé et des séquences préparées via `Mimir.Dataset.prepare_sequences(max_seq_len)`.
---@param epochs integer @Nombre d'epochs
---@param learning_rate number @LR (ex: 3e-4)
---@return boolean ok
---@return string? err
function Mimir.Model.train(epochs, learning_rate) end

---Inférence sur un prompt (string) ou une séquence de tokens.
---Retour: string (texte) ou nil si échec.
---@param input string|TokenIds
---@return string|nil output
function Mimir.Model.infer(input) end

---[DÉPRÉCIÉ] Sauvegarder le modèle (ancienne API).
---⚠️  Utilisez Mimir.Serialization.save() pour la nouvelle API v2.4
---@param dir string @Ex: "checkpoints/run1/epoch_10"
---@return boolean ok
---@return string? err
---@deprecated Utilisez Mimir.Serialization.save()
function Mimir.Model.save(dir) end

---[DÉPRÉCIÉ] Charger un modèle depuis un répertoire (ancienne API).
---⚠️  Utilisez Mimir.Serialization.load() pour la nouvelle API v2.4
---@param dir string
---@return boolean ok
---@return string? err
---@deprecated Utilisez Mimir.Serialization.load()
function Mimir.Model.load(dir) end

---Allouer explicitement les paramètres (si supporté).
---@return boolean ok
---@return integer? params @Nombre total de paramètres alloués (si ok)
---@return string? err
function Mimir.Model.allocate_params() end

---Initialiser les poids du modèle (si supporté).
---@param init? WeightInit
---@param seed? integer
---@return boolean ok
---@return string? err
function Mimir.Model.init_weights(init, seed) end

---Nombre total de paramètres (si disponible).
---@return integer params
function Mimir.Model.total_params() end

---Pousser une couche (API bas niveau / description) dans le modèle.
---Note: cette fonction est surtout utilisée par les builders d'architectures.
---@param name string
---@param layer_type string
---@param params_count integer
---@return boolean ok
---@return string? err
function Mimir.Model.push_layer(name, layer_type, params_count) end

---Configure les entrées et sortie d'un layer pour le support multi-input/branch.
---Permet de créer des architectures avec skip connections, concat, split, etc.
---@param layer_name string @Nom du layer à configurer
---@param inputs table @Table des noms de tensors en entrée, ex: {"x", "skip"}
---@param output? string @Nom du tensor de sortie (optionnel, défaut: "x")
---@return boolean ok @true si succès, false si layer non trouvé
---@return string? err @Message d'erreur éventuel
---
---Exemples:
---```lua
--- -- Residual connection (skip connection)
--- Mimir.Model.set_layer_io("conv1", {"x"}, "skip")
--- Mimir.Model.set_layer_io("add", {"x", "skip"}, "x")
---
--- -- Concat multiple branches
--- Mimir.Model.set_layer_io("concat", {"branch1", "branch2", "branch3"}, "fused")
---
--- -- MatMul (A × B)
--- Mimir.Model.set_layer_io("matmul", {"A", "B"}, "result")
---
--- -- Split (1 input → N outputs nommés output_0, output_1, ...)
--- Mimir.Model.set_layer_io("split", {"x"}, "branches")
---```
function Mimir.Model.set_layer_io(layer_name, inputs, output) end

---Forward pass (si exposé par l'implémentation).
---Mode training activé par défaut pour permettre le backward pass.
---@param input TokenIds|float[] @Données d'entrée (table d'entiers -> chemin token ids, table de floats -> chemin float)
---@param training? bool @Mode training (défaut: true) pour calculer les gradients
---@return float[]|nil @Sortie du modèle
---@return string? err
function Mimir.Model.forward(input, training) end

---Forward pass utilitaire pour les modèles texte→image: fournit explicitement
---un prompt déjà encodé, une image (entrée conditionnelle), et une seed.
---La seed sert à rendre déterministes les opérations stochastiques du forward (si présentes).
---@param text_vec float[] @Vecteur texte (len = d_model)
---@param image_vec float[] @Image (len = image_w*image_h*image_c) normalisée [-1,1]
---@param seed integer
---@param training? bool @Défaut: false
---@return float[]|nil output
---@return string? err
function Mimir.Model.forward_prompt_image_seed(text_vec, image_vec, seed, training) end

---Encode un prompt texte en vecteur embedding (utile pour les modèles type PonyXL).
---Note: cette fonction dépend d'un modèle courant déjà créé via `Mimir.Model.create()`.
---Retour: table de floats (len = d_model) ou (nil, err).
---@param prompt string
---@return float[]|nil text_vec
---@return string? err
function Mimir.Model.encode_prompt(prompt) end

---Alias camelCase de `encode_prompt`.
---@param prompt string
---@return float[]|nil text_vec
---@return string? err
function Mimir.Model.encodePrompt(prompt) end

---Backward pass pour calculer les gradients.
---@param loss_gradient float[] @Gradient de la loss
---@return boolean ok
---@return string? err
function Mimir.Model.backward(loss_gradient) end

---Réinitialise tous les gradients à zéro.
---Important: à appeler avant chaque itération d'entraînement.
---@return boolean ok
---@return string? err
function Mimir.Model.zero_grads() end

---Récupère les gradients actuels de tous les paramètres.
---@return float[]|nil @Vecteur de tous les gradients
---@return string? err
function Mimir.Model.get_gradients() end

---Step optimiseur (si exposé). Le LR peut être transmis.
---@param learning_rate number
---@param opt_type? string @"sgd"|"adam"|"adamw" (défaut: "adamw")
---@return boolean ok
---@return string? err
function Mimir.Model.optimizer_step(learning_rate, opt_type) end

---Récupérer la configuration/état courant de l'optimizer.
---Best-effort: lit le serialized optimizer si présent, sinon reconstruit depuis la config.
---@return table opts @Champs: optimizer, type, beta1, beta2, epsilon, weight_decay, step, decay_strategy, initial_lr, min_lr, decay_rate, decay_steps, total_steps, warmup_steps
---@return string? err
function Mimir.Model.get_optimizer() end

---Mettre à jour l'optimizer depuis Lua.
---Applique les valeurs dans la config du modèle (persistant pour `Model.train()`)
---et, si un modèle courant existe, met à jour son optimizer sérialisé.
---
---Champs supportés (table opts):
---  optimizer = "sgd"|"adam"|"adamw" (ou type=0/1/2)
---  beta1, beta2, epsilon (ou eps), weight_decay
---  min_lr, decay_rate, decay_steps, total_steps, warmup_steps, decay_strategy
---  reset_state=true (optionnel) pour effacer moments + step
---@param opts table
---@return boolean ok
---@return string? err
function Mimir.Model.set_optimizer(opts) end

---Réinitialiser l'état interne de l'optimizer (moments + step).
---Conserve les hyperparamètres courants.
---@return boolean ok
---@return string? err
function Mimir.Model.reset_optimizer_state() end

---Active/désactive l'accélération matérielle côté modèle.
---Note: le binding actuel prend un booléen (il ne sélectionne pas un backend nommé).
---@param enable boolean @true pour activer l'accélération (si dispo), false pour forcer CPU
---@return boolean ok
function Mimir.Model.set_hardware(enable) end

---Retourne les capacités détectées (AVX2/FMA/F16C/BMI2).
---@return HardwareCaps caps
function Mimir.Model.hardware_caps() end

--=============================================================================
-- Module: Mimir.Architectures
--=============================================================================

---@class MimirArchitecturesAPI
Mimir.Architectures = {}

---Lister les architectures disponibles (côté C++ registry).
---@return ArchitectureName[]|nil names
---@return string? err
function Mimir.Architectures.available() end

---Retourner la config par défaut d'une architecture.
---@overload fun(name: "basic_mlp"): BasicMLPConfig
---@overload fun(name: "transformer"): TransformerConfig
---@overload fun(name: "vae_text"): VAETextConfig
---@overload fun(name: "vae_text_decode"): VAETextConfig
---@overload fun(name: "vit"): ViTConfig
---@overload fun(name: "vae"): VAEConfig
---@overload fun(name: "vae_conv"): VAEConvConfig
---@overload fun(name: "vae_conv_decode"): VAEConvConfig
---@overload fun(name: "resnet"): ResNetConfig
---@overload fun(name: "unet"): UNetConfig
---@overload fun(name: "mobilenet"): MobileNetConfig
---@overload fun(name: "vgg16"): VGG16Config
---@overload fun(name: "vgg19"): VGG19Config
---@overload fun(name: "diffusion"): DiffusionConfig
---@overload fun(name: "cond_diffusion"): CondDiffusionConfig
---@overload fun(name: "gan_latent"): GanLatentConfig
---@overload fun(name: "sd3_5"): SD35Config
---@overload fun(name: "ponyxl_ddpm"): PonyXLDDPMConfig
---@param name ArchitectureName|string
---@return table|nil config
---@return string? err
function Mimir.Architectures.default_config(name) end

--=============================================================================
-- Module: Mimir.Layers (placeholder / low-level)
--=============================================================================

---@class MimirLayersAPI
Mimir.Layers = {}

---Conv2D low-level (placeholder). Utiliser `Mimir.Model.forward()` à la place.
---@return boolean ok
---@return string err
function Mimir.Layers.conv2d(...) end

---Linear low-level (placeholder). Utiliser `Mimir.Model.forward()` à la place.
---@return boolean ok
---@return string err
function Mimir.Layers.linear(...) end

---MaxPool2D low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.maxpool2d(...) end

---AvgPool2D low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.avgpool2d(...) end

---Activation low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.activation(...) end

---BatchNorm low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.batchnorm(...) end

---LayerNorm low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.layernorm(...) end

---Attention low-level (placeholder).
---@return boolean ok
---@return string err
function Mimir.Layers.attention(...) end

--=============================================================================
-- Module: Mimir.Checkpoint (legacy, deprecated)
--=============================================================================

---@class MimirCheckpointAPI
Mimir.Checkpoint = {}

---Sauvegarder un checkpoint (API legacy; préférer `Mimir.Serialization.save`).
---@param path string
---@param format? string @Ex: "raw_folder"
---@param options? table
---@return boolean ok
---@return string? err
function Mimir.Checkpoint.save(path, format, options) end

---Charger un checkpoint (API legacy; préférer `Mimir.Serialization.load`).
---@param path string
---@param format? string @Ex: "raw_folder"
---@param options? table
---@return boolean ok
---@return string? err
function Mimir.Checkpoint.load(path, format, options) end

--=============================================================================
-- Module: Mimir.Tokenizer
--=============================================================================

---@class MimirTokenizerAPI
Mimir.Tokenizer = {}

---Créer un tokenizer avec vocab max.
---@param max_vocab integer
---@return boolean ok
function Mimir.Tokenizer.create(max_vocab) end

---Tokeniser un texte (word/BPE selon impl).
---@param text string
---@return TokenIds tokens
function Mimir.Tokenizer.tokenize(text) end

---Dé-tokeniser une séquence (ids -> texte).
---@param tokens TokenIds
---@return string text
function Mimir.Tokenizer.detokenize(tokens) end

---Retourne la taille actuelle du vocab.
---@return integer size
function Mimir.Tokenizer.vocab_size() end

---Retourne la capacité maximale du vocab (limite d'ajout de nouveaux tokens).
---Note: si `vocab_size()` atteint `max_vocab`, `add_token()` peut échouer / retomber sur UNK selon impl.
---@return integer max_vocab
function Mimir.Tokenizer.get_max_vocab() end

---Définir la capacité maximale du vocab.
---La valeur est clampée pour être >= `vocab_size()`.
---@param max_vocab integer
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.set_max_vocab(max_vocab) end

---Sauvegarder le tokenizer dans un fichier.
---@param path string @Ex: "checkpoints/run1/tokenizer.json"
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.save(path) end

---Charger le tokenizer depuis un fichier.
---@param path string
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.load(path) end

---Ajouter un token au vocab.
---@param token string
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.add_token(token) end

---Étendre le vocab à partir d'un texte (analyse corpus simple).
---@param text string
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.ensure_vocab_from_text(text) end

---Tokenize et étend le vocab si nécessaire.
---@param text string
---@return TokenIds tokens
function Mimir.Tokenizer.tokenize_ensure(text) end

---IDs spéciaux.
---@return integer id
function Mimir.Tokenizer.pad_id() end
---@return integer id
function Mimir.Tokenizer.unk_id() end
---@return integer id
function Mimir.Tokenizer.seq_id() end
---@return integer id
function Mimir.Tokenizer.mod_id() end
---@return integer id
function Mimir.Tokenizer.mag_id() end

---Récupérer un token string depuis un id.
---@param id integer
---@return string token
function Mimir.Tokenizer.get_token_by_id(id) end

---Apprendre un BPE depuis un corpus (chemin ou texte selon impl).
---@param corpus_path string
---@param vocab_target? integer
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.learn_bpe(corpus_path, vocab_target) end

---Tokeniser via BPE (si appris/chargé).
---@param text string
---@return TokenIds tokens
function Mimir.Tokenizer.tokenize_bpe(text) end

---Définir la longueur max des séquences côté tokenizer.
---@param max_length integer
---@return boolean ok
---@return string? err
function Mimir.Tokenizer.set_max_length(max_length) end

---Pad une séquence à une longueur cible.
---@param tokens TokenIds
---@param max_length integer
---@param pad_id? integer
---@return TokenIds padded
function Mimir.Tokenizer.pad_sequence(tokens, max_length, pad_id) end

---Tokeniser une liste de textes.
---@param texts string[]
---@return TokenIds[] batches
function Mimir.Tokenizer.batch_tokenize(texts) end

---Afficher des stats (stdout/log).
function Mimir.Tokenizer.print_stats() end

---Fréquences tokens (si supporté).
---@return table<string, integer> freqs
function Mimir.Tokenizer.get_frequencies() end

---Analyse texte (mots, chars, densité, etc. selon impl).
---Retourne une table avec des statistiques et analyses sémantiques.
---@param text string @Texte à analyser
---@return TextAnalysisResult results @Résultats de l'analyse
---
---**Exemple:**
---```lua
--- local analysis = Mimir.Tokenizer.analyze_text("A majestic dragon flying over mountains")
--- print("Complexité:", analysis.complexity)
--- print("Sujet principal:", analysis.main_subject)
--- if analysis.entities then
---   for _, entity in ipairs(analysis.entities) do
---     print("  Entité:", entity)
---   end
--- end
---```
function Mimir.Tokenizer.analyze_text(text) end

---Extraction keywords (heuristique).
---@param text string
---@param top_k? integer
---@return KeywordList keywords
function Mimir.Tokenizer.extract_keywords(text, top_k) end

--=============================================================================
-- Module: Mimir.Dataset
--=============================================================================

---@class DatasetItem
---@field text_file? string Chemin du fichier texte
---@field image_file? string Chemin du fichier image
---@field audio_file? string Chemin du fichier audio
---@field video_file? string Chemin du fichier vidéo
---@field text? string Contenu texte (si chargé)
---@field width? int Largeur de l'image
---@field height? int Hauteur de l'image
---@field size? int Taille du fichier en bytes

---@class TextAnalysisResult
---@field complexity? float Complexité du texte (métrique interne)
---@field main_subject? string Sujet principal détecté
---@field entities? string[] Entités nommées extraites
---@field actions? string[] Actions/verbes principaux
---@field modifiers? string[] Modificateurs/adjectifs importants
---@field context? string Contexte général du texte
---@field resolution? int Résolution estimée pour génération d'image

---@class MimirDatasetAPI
Mimir.Dataset = {}

--- Charge un dataset depuis un répertoire.
---
--- Signature:
---   `Mimir.Dataset.load(dir, target_w?, target_h?, min_modalities?, use_cache?, cache_path?, max_ram_mb?, lazy_loading?)`
---
--- Si `use_cache` vaut `true`, utilise le cache JSON via `loadDatasetCached()`.
---@param dir string
---@param target_w? integer
---@param target_h? integer
---@param min_modalities? integer
---@param use_cache? boolean
---@param cache_path? string
---@param max_ram_mb? integer
---@param lazy_loading? boolean
---@return boolean ok
---@return integer|string count_or_err @Nombre d'items si ok, sinon message d'erreur
function Mimir.Dataset.load(dir, target_w, target_h, min_modalities, use_cache, cache_path, max_ram_mb, lazy_loading) end

---Récupérer un item du dataset par son index (1-based).
---Retourne une table avec les chemins et métadonnées de l'item.
---@param index integer Index de l'item (commence à 1)
---@return DatasetItem|nil item Item du dataset
---@return string? err Message d'erreur si échec
function Mimir.Dataset.get(index) end

---Préparer les séquences (stockées dans le contexte interne).
---La séquence length est utilisée ensuite par Mimir.Model.train().
---@param max_length integer
---@return boolean ok
---@return string? err
function Mimir.Dataset.prepare_sequences(max_length) end

--=============================================================================
-- Module: Mimir.Database (loader builder avec cache)
--=============================================================================

---@class MimirDatabaseLoader
local _MimirDatabaseLoader = {}

---Charger un dataset via un loader avec cache.
---
---**Usage:**
---```lua
---local loader = Mimir.Database.load(DATASET_DIR, 64, 64, 1)
---local ok, n_or_err = loader.cache("dataset_cache.json", 10240, true)
---assert(ok, n_or_err)
---```
---
---La méthode `cache()` utilise le loader C++ `loadDatasetCached()`.
---Signature flexible:
---`cache([dir], [target_w], [target_h], [min_modalities], [cache_path], [max_ram_mb], [lazy_loading])`
---@return boolean ok
---@return integer|string? count_or_err
function _MimirDatabaseLoader.cache(...) end

---@class MimirDatabaseAPI
Mimir.Database = {}

---Créer un loader (builder) pour charger un dataset.
---Le chargement effectif se fait via `:cache(...)` ou `.cache(...)`.
---@param dir? string
---@param target_w? integer
---@param target_h? integer
---@param min_modalities? integer
---@return MimirDatabaseLoader loader
function Mimir.Database.load(dir, target_w, target_h, min_modalities) end

--=============================================================================
-- Module: Mimir.Memory (AdvancedRAMManager)
--=============================================================================

---@class MimirMemoryAPI
Mimir.Memory = {}

---Configurer le gestionnaire RAM avancé (implémentation dépendante).
---@param cfg table
---@return boolean ok
---@return string? err
function Mimir.Memory.config(cfg) end

---Récupérer stats RAM.
---@return MemoryStats stats
function Mimir.Memory.get_stats() end

---Alias camelCase de `get_stats`.
---@return MemoryStats stats
function Mimir.Memory.getStats() end

---Imprimer stats RAM.
function Mimir.Memory.print_stats() end

---Alias camelCase de `print_stats`.
function Mimir.Memory.printStats() end

---Purger / clear caches mémoire.
---@return boolean ok
---@return string? err
function Mimir.Memory.clear() end

---Usage actuel en MB (alias utilitaire).
---@return number mb
function Mimir.Memory.get_usage() end

---Alias camelCase de `get_usage`.
---@return number mb
function Mimir.Memory.getUsage() end

---Définir une limite RAM (en MB).
---@param limit_mb number
---@return boolean ok
---@return string? err
function Mimir.Memory.set_limit(limit_mb) end

---Alias camelCase de `set_limit`.
---@param limit_mb number
---@return boolean ok
---@return string? err
function Mimir.Memory.setLimit(limit_mb) end

--=============================================================================
-- Module: Mimir.Guard (MemoryGuard) - API Ancienne
--=============================================================================

---@class MimirGuardAPI
Mimir.Guard = {}

---Définir la limite de mémoire (MB).
---@param limit_mb number
---@return boolean ok
---@return string? err
function Mimir.Guard.set_limit(limit_mb) end

---Alias camelCase de `set_limit`.
---@param limit_mb number
---@return boolean ok
---@return string? err
function Mimir.Guard.setLimit(limit_mb) end

---Stats de la limite stricte.
---@return GuardStats stats
function Mimir.Guard.get_stats() end

---Alias camelCase de `get_stats`.
---@return GuardStats stats
function Mimir.Guard.getStats() end

---Imprimer stats guard.
function Mimir.Guard.print_stats() end

---Alias camelCase de `print_stats`.
function Mimir.Guard.printStats() end

---Reset stats (peak, compteurs).
function Mimir.Guard.reset() end

--=============================================================================
-- Module: Mimir.MemoryGuard (API Moderne - Recommandée)
--=============================================================================

---@class MemoryGuardStats
---@field current_mb float @Utilisation RAM courante en MB
---@field peak_mb float @Pic d'utilisation en MB
---@field limit_mb float @Limite configurée en MB
---@field usage_percent float @Pourcentage d'utilisation

---@class MimirMemoryGuardAPI
Mimir.MemoryGuard = {}

---Définir la limite de mémoire RAM stricte.
---Accepte des valeurs en bytes (grands nombres) ou en GB (si <= 1000).
---@param limit number @Limite en bytes ou en GB (si valeur <= 1000)
---@return boolean ok @true si succès
---
---**Exemples:**
---```lua
--- -- Définir limite à 10 Go
--- Mimir.MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)  -- en bytes
--- Mimir.MemoryGuard.setLimit(10)  -- en GB (auto-détecté car < 1000)
---```
function Mimir.MemoryGuard.setLimit(limit) end

---Obtenir la limite de mémoire configurée.
---@return integer bytes @Limite en bytes
---
---**Exemple:**
---```lua
--- local limit = Mimir.MemoryGuard.getLimit()
--- print(string.format("Limite: %.2f GB", limit / 1024 / 1024 / 1024))
---```
function Mimir.MemoryGuard.getLimit() end

---Obtenir l'utilisation RAM courante.
---@return integer bytes @RAM utilisée actuellement en bytes
---
---**Exemple:**
---```lua
--- local current = Mimir.MemoryGuard.getCurrentUsage()
--- local limit = Mimir.MemoryGuard.getLimit()
--- local percent = (current / limit) * 100
--- print(string.format("RAM: %.2f%%", percent))
---```
function Mimir.MemoryGuard.getCurrentUsage() end

---Obtenir le pic d'utilisation RAM.
---@return integer bytes @Pic d'utilisation en bytes depuis le démarrage
---
---**Exemple:**
---```lua
--- local peak = Mimir.MemoryGuard.getPeakUsage()
--- print(string.format("Pic RAM: %.2f GB", peak / 1024 / 1024 / 1024))
---```
function Mimir.MemoryGuard.getPeakUsage() end

---Récupérer toutes les statistiques en une seule fois.
---@return MemoryGuardStats stats @Structure contenant toutes les stats
---
---**Exemple:**
---```lua
--- local stats = Mimir.MemoryGuard.getStats()
--- print("RAM courante: " .. stats.current_mb .. " MB")
--- print("Pic: " .. stats.peak_mb .. " MB")
--- print("Limite: " .. stats.limit_mb .. " MB")
--- print("Utilisation: " .. stats.usage_percent .. "%")
---```
function Mimir.MemoryGuard.getStats() end

---Afficher les statistiques formatées dans la console.
---Affiche un tableau détaillé avec toutes les métriques.
---
---**Format de sortie:**
---```
---╔═══════════════════════════════════════════════════════╗
---║           MEMORY GUARD - STATISTIQUES                ║
---╠═══════════════════════════════════════════════════════╣
---║ Limite:          10240 MB                         ║
---║ Actuel:           2456 MB                         ║
---║ Pic:              3892 MB                         ║
---║ Utilisation:     38.0 %                          ║
---║ Allocations:      1523                            ║
---║ Libérations:       892                            ║
---╠═══════════════════════════════════════════════════════╣
---║ État:       🔓 ACTIF                             ║
---╚═══════════════════════════════════════════════════════╝
---```
---
---**Exemple:**
---```lua
--- Mimir.MemoryGuard.setLimit(10 * 1024 * 1024 * 1024)
--- -- ... opérations ...
--- Mimir.MemoryGuard.printStats()  -- Affiche le rapport complet
---```
function Mimir.MemoryGuard.printStats() end

---Réinitialiser les compteurs de statistiques.
---Remet à zéro le pic et les compteurs d'allocations/libérations.
---La limite configurée est préservée.
---
---**Exemple:**
---```lua
--- Mimir.MemoryGuard.reset()
--- print("✓ Statistiques réinitialisées")
---```
function Mimir.MemoryGuard.reset() end

--=============================================================================
-- Module: Mimir.Allocator (DynamicTensorAllocator)
--=============================================================================

---@class AllocatorConfig
---@field max_tensors? integer @Nombre max de tenseurs en mémoire
---@field offload_threshold_mb? float @Seuil de RAM pour offload
---@field swap_strategy? string @Stratégie de swap (lru, fifo, etc.)
---@field max_ram_gb? number @Limite RAM globale
---@field enable_compression? boolean @Activer compression LZ4

---@class MimirAllocatorAPI
Mimir.Allocator = {}

---Configurer l'allocator dynamique (tenseurs, offload, compression).
---@param cfg AllocatorConfig|table
---@return boolean ok
---@return string? err
function Mimir.Allocator.configure(cfg) end

---Imprimer stats allocator (stdout/log).
function Mimir.Allocator.print_stats() end

---Alias camelCase de `print_stats`.
function Mimir.Allocator.printStats() end

---Récupérer stats allocator.
---@return AllocatorStats stats
function Mimir.Allocator.get_stats() end

---Alias camelCase de `get_stats`.
---@return AllocatorStats stats
function Mimir.Allocator.getStats() end

--=============================================================================
-- Module: Mimir.Htop (AsyncMonitor / Terminal UI)
--=============================================================================

---@class MimirHtopAPI
Mimir.Htop = {} --avec configuration optionnelle.
---Créer/démarrer l'UI htop (AsyncMonitor).
---
---Compat:
---- `Mimir.Htop.create(true|false)` : active/désactive la viz
---- `Mimir.Htop.create({ ... })` : options avancées
---
---Notes:
---- Si `enable_viz=true`, le CSV htop est désactivé par défaut (sauf si `csv=true` ou `csv_path` fourni).
---
---@param config? boolean|HtopCreateConfig
---@return boolean ok
---@return string? err
function Mimir.Htop.create(config) end

---Activer/désactiver l'affichage htop.
---@param enabled boolean
---@return boolean ok
---@return string? err
function Mimir.Htop.enable(enabled) end

---Mettre à jour les métriques affichées dans htop.
---Les paramètres peuvent être passés via une table HtopMetrics ou individuellement.
---@param metrics? HtopMetrics|table @Structure de métriques (optionnel)
---@param ... any @Paramètres legacy (optionnels)
---@return boolean ok
---@return string? err
function Mimir.Htop.update(metrics, ...) end

---Forcer un render (si supporté).
---@return boolean ok
---@return string? err
function Mimir.Htop.render() end

---Clear écran / reset UI.
---@return boolean ok
---@return string? err
function Mimir.Htop.clear() end

--=============================================================================
-- Module: Mimir.Viz (SFML Visualizer)
--=============================================================================

---@class MimirVizAPI
Mimir.Viz = {}

---Créer la fenêtre visualiseur SFML avec titre et dimensions optionnels.
---Argument réel: table de configuration (passée à AsyncMonitor.start()).
---@param config? table @Ex: {visualization={enabled=true, window_title="..."}}
---@return boolean ok
---@return string? err
function Mimir.Viz.create(config) end

---Initialiser le visualiseur (ouvre la fenêtre SFML).
---@return boolean ok
---@return string? err
function Mimir.Viz.initialize() end

---Vérifier si la fenêtre est ouverte.
---@return boolean open
function Mimir.Viz.is_open() end

---Traiter les événements fenêtre (fermeture, clavier, souris).
function Mimir.Viz.process_events() end

---Mettre à jour et afficher le rendu de la fenêtre.
function Mimir.Viz.update() end

---Ajouter/afficher une image dans le visualiseur.
---Les pixels sont fournis sous forme de tableau plat (row-major).
---@param pixels number[] @Tableau de valeurs pixel [0-255], RGBA ou RGB
---@param width integer @Largeur de l'image
---@param height integer @Hauteur de l'image
---@param channels integer @Nombre de canaux (3=RGB, 4=RGBA)
---@return boolean ok
---@return string? err
function Mimir.Viz.add_image(pixels, width, height, channels) end

---Mettre à jour les métriques d'entraînement affichées.
---@param metrics VizMetrics|table @Métriques: epoch, loss, lr, memory_mb, etc.
---@return boolean ok
---@return string? err
function Mimir.Viz.update_metrics(metrics) end

---@class VizValidationState
---@field in_progress? boolean @Validation en cours (affiche "EN COURS" dans l'UI)
---@field step? int @Step globale (pour l'affichage)
---@field done? int @Progression (items traités)
---@field total? int @Total attendu
---@field has? boolean @Indique si les métriques val sont disponibles
---@field ok? boolean @Résultat final (si in_progress=false)
---@field recon? float @Métrique principale (ex: recon/img_mse)
---@field kl? float @Métrique secondaire (ex: kl/eps_mse)
---@field align? float @Métrique optionnelle (ex: align/margin)

---Mettre à jour l'état/progression de validation affichée (sans écraser les métriques train).
---@param state VizValidationState|table
---@return boolean ok
---@return string? err
function Mimir.Viz.set_validation(state) end

---Ajouter un point à l'historique de loss (pour graphe).
---@param loss number @Valeur de loss
function Mimir.Viz.add_loss_point(loss) end

---Clear viz.
---@return boolean ok
---@return string? err
function Mimir.Viz.clear() end

---Activer/désactiver la viz (NO-OP si non compilée).
---@param enabled boolean
---@return boolean ok
---@return string? err
function Mimir.Viz.set_enabled(enabled) end

---Sauvegarder l'historique de loss.
---@param path string
---@return boolean ok
---@return string? err
function Mimir.Viz.save_loss_history(path) end

--=============================================================================
-- Fonctions utilitaires globales (Image IO)
--=============================================================================

---Charger une image depuis le disque et retourner des pixels RGB u8 redimensionnés.
---Supporte les formats `png/jpg/jpeg/bmp/tiff/webp` via stb_image.
---
---Signature réelle: retourne soit `(pixels, w, h, c)` soit `(nil, err)`.
---@param path string
---@param target_w integer
---@param target_h integer
---@return number[]|nil pixels
---@return integer|string w_or_err
---@return integer? h
---@return integer? c
function read_image_rgb_u8(path, target_w, target_h) end

-- Alias FR (même API): Mimir.visualiser
---@type MimirVizAPI
Mimir.visualiser = Mimir.Viz

--=============================================================================
-- Mimir.Serialization API (v2.4.0)
--=============================================================================
-- Nouveau système de sérialisation avec 3 formats :
--  • SafeTensors : Format production compatible HuggingFace
--  • RawFolder   : Format debug avec checksums SHA256
--  • DebugJson   : Format inspection avec statistiques

---@class MimirSerializationAPI
Mimir.Serialization = {}

---@alias SerializationFormat
---| "safetensors"  # Format production (défaut)
---| "raw_folder"    # Format debug avec checksums
---| "debug_json"    # Format inspection avec stats

---@class SaveOptions
---@field format? SerializationFormat @Format de sauvegarde (défaut: SAFETENSORS)
---@field save_tokenizer? boolean @Sauvegarder le tokenizer (défaut: true)
---@field save_encoder? boolean @Sauvegarder l'encoder (défaut: true)
---@field save_optimizer? boolean @Sauvegarder l'état optimizer (défaut: false)
---@field debug_max_values? integer @Nombre max de valeurs debug (legacy, défaut: 100)
---@field include_git_info? boolean @Inclure info git (défaut: true)
---@field include_gradients? boolean @[DebugJson v1.1] Inclure gradients (défaut: false)
---@field include_optimizer_state? boolean @[DebugJson v1.1] Inclure optimizer state (défaut: false)
---@field max_values_per_tensor? integer @[DebugJson v1.1] Nb valeurs par tensor (défaut: 20)
---@field include_activations? boolean @[DebugJson v1.1] Inclure activations (défaut: false)
---@field include_checksums? boolean @[DebugJson v1.1] Inclure checksums (défaut: false)
---@field include_weight_deltas? boolean @[DebugJson v1.1] Inclure weight deltas (défaut: false)

---@class LoadOptions
---@field load_tokenizer? boolean @Charger le tokenizer (défaut: true)
---@field load_encoder? boolean @Charger l'encoder (défaut: true)
---@field load_optimizer? boolean @Charger l'état optimizer (défaut: false)
---@field verify_checksums? boolean @Vérifier checksums SHA256 (défaut: true, RawFolder uniquement)

---Sauvegarder un checkpoint avec le nouveau système de sérialisation v2.4.
---
---Formats supportés :
---  • SAFETENSORS : Format production compatible HuggingFace/PyTorch
---  • RAWFOLDER   : Format debug avec checksums SHA256
---  • DEBUGJSON   : Format inspection avec statistiques
---
---@param path string @Chemin du fichier/dossier (ex: "model.safetensors", "checkpoint/")
---@param format? SerializationFormat @Format (défaut: "SAFETENSORS")
---@param options? SaveOptions @Options de sauvegarde
---@return boolean ok @true si succès
---@return string? err @Message d'erreur si échec
---
---Exemples :
---```lua
--- -- SafeTensors (production)
--- Mimir.Serialization.save("model.safetensors")
--- 
--- -- RawFolder (debug)
--- Mimir.Serialization.save("checkpoint/", "raw_folder")
--- 
--- -- DebugJson (inspection)
--- Mimir.Serialization.save("debug.json", "debug_json", {debug_max_values = 20})
--- 
--- -- DebugJson Enhanced v1.1 (diagnostic entraînement)
--- Mimir.Serialization.save("debug.json", "debug_json", {
---     include_gradients = true,
---     include_optimizer_state = true,
---     include_weight_deltas = true,
---     include_checksums = true,
---     max_values_per_tensor = 20
--- })
--- 
--- -- Avec options
--- Mimir.Serialization.save("model.safetensors", "safetensors", {
---     save_optimizer = true,
---     include_git_info = true
--- })
---```
function Mimir.Serialization.save(path, format, options) end

---Charger un checkpoint avec détection automatique du format.
---
---La fonction détecte automatiquement le format :
---  • Fichier .safetensors → SafeTensors
---  • Dossier avec manifest.json → RawFolder
---
---@param path string @Chemin du fichier/dossier
---@param format? SerializationFormat @Format (optionnel, auto-détecté)
---@param options? LoadOptions @Options de chargement
---@return boolean ok @true si succès
---@return string? err @Message d'erreur si échec
---
---Exemples :
---```lua
--- -- Auto-détection du format
--- Mimir.Serialization.load("model.safetensors")
--- Mimir.Serialization.load("checkpoint/")
--- 
--- -- Format explicite
--- Mimir.Serialization.load("model.safetensors", "safetensors")
--- 
--- -- Avec vérification checksums
--- Mimir.Serialization.load("checkpoint/", "raw_folder", {
---     verify_checksums = true
--- })
---```
function Mimir.Serialization.load(path, format, options) end

---Détecter automatiquement le format d'un checkpoint.
---@param path string @Chemin du fichier/dossier
---@return SerializationFormat? format @Format détecté ou nil si inconnu
---@return string? err @Message d'erreur si échec
function Mimir.Serialization.detect_format(path) end

---Sauvegarder un checkpoint avec Enhanced Debug JSON v1.1.0.
---Fonction spécialisée pour le diagnostic d'entraînement avec toutes les options v1.1.
---@param path string @Chemin du fichier JSON
---@param options? SaveOptions @Options Enhanced Debug JSON v1.1
---@return boolean ok @true si succès
---@return string? err @Message d'erreur si échec
function Mimir.Serialization.save_enhanced_debug(path, options) end

--=============================================================================
-- Mimir.NeuroPulse API (texte -> audio/lumière)
--=============================================================================
-- Ce module génère des signaux (fichier WAV + enveloppe "lumière" en CSV) à partir
-- d'un texte. Il s'agit d'un générateur déterministe (pas une thérapie, pas un dispositif
-- médical). Par défaut, la "lumière" est une variation lente et lisse (safe_light=true)
-- pour éviter les effets stroboscopiques.

---@class MimirNeuroPulseAPI
Mimir.NeuroPulse = {}

---@class NeuroPulseOptions
---@field duration_s? number @Durée en secondes (défaut: 10)
---@field sample_rate? integer @Hz (défaut: 48000)
---@field carrier_hz? number @Fréquence porteuse audio (défaut: 220)
---@field audio_mod_depth? number @Profondeur modulation AM 0..1 (défaut: 0.8)
---@field binaural_hz? number @Décalage binaural (0=off) (défaut: 0)
---@field cognitive_band? string @"auto"|"delta"|"theta"|"alpha"|"beta"|"gamma" (défaut: auto)
---@field cognitive_hz? number @Override fréquence (0=auto)
---@field safe_light? boolean @Clamp + enveloppe lisse (défaut: true)
---@field light_fps? integer @Échantillonnage CSV lumière (défaut: 60)
---@field light_hz? number @Fréquence enveloppe lumière (défaut: 0.2, clamp <=2 si safe)
---@field light_depth? number @Profondeur 0..1 (défaut: 0.6)
---@field out_wav? string @Chemin sortie WAV (défaut: neuropulse.wav)
---@field out_light_csv? string @Chemin sortie CSV lumière (défaut: neuropulse_light.csv)
---@field organic_nn? boolean @Activer modulation "organique" via petit MLP interne (défaut: true)
---@field nn_control_fps? integer @Résolution temporelle du contrôle NN (défaut: 200)
---@field nn_hidden_dim? integer @Largeur hidden du MLP (défaut: 32)
---@field nn_hidden_layers? integer @Nombre de couches hidden (défaut: 2)
---@field nn_dropout? number @Dropout (0..0.95, surtout utile en training) (défaut: 0)
---@field nn_strength? number @Force de modulation (0..1) (défaut: 0.35)
---@field nn_smooth? number @Lissage IIR des sorties (0..0.999) (défaut: 0.85)
---@field nn_seed? integer @Seed (0=auto dérivé du sha256) (défaut: 0)

---@class NeuroPulseMeta
---@field sha256 string
---@field band string
---@field cognitive_hz number
---@field carrier_hz number
---@field binaural_hz number
---@field sample_rate integer
---@field duration_s number
---@field light_fps integer
---@field light_hz number
---@field out_wav string
---@field out_light_csv string
---@field organic_nn boolean
---@field nn_seed integer
---@field nn_control_fps integer
---@field warnings string[]

---Générer un WAV + un CSV "lumière" depuis un texte.
---@param text string
---@param opts? NeuroPulseOptions
---@return boolean ok
---@return NeuroPulseMeta|string meta_or_err
function Mimir.NeuroPulse.render(text, opts) end

---Calculer les paramètres (band/frequences) qui seraient utilisés.
---@param text string
---@param opts? NeuroPulseOptions
---@return NeuroPulseMeta meta
function Mimir.NeuroPulse.params(text, opts) end

--=============================================================================
-- Fonctions globales utilitaires
--=============================================================================

---Logger (côté C++). Equivalent console + buffer.
---@param msg string
function log(msg) end

---Lire un JSON depuis un fichier.
---@param path string
---@return table|nil obj
---@return string? err
function read_json(path) end

---Écrire un JSON dans un fichier.
---@param path string
---@param obj table
---@return boolean ok
---@return string? err
function write_json(path, obj) end

---Print (redirigé) — peut être surchargé par le runtime.
---@param ... any
function print(...) end

--=============================================================================
-- Pipeline API (optionnel)
--=============================================================================
-- Si vous chargez pipeline_api.lua dans votre script, l'IDE bénéficiera aussi
-- de ces signatures. Ces fonctions sont en Lua pur, mais on les déclare ici
-- pour l'autocomplétion (sans dépendre du require dans l'IDE).

---@class PipelineConfig
---@field dataset_dir string
---@field out_dir string
---@field model_type ModelType
---@field model_config table
---@field tokenizer_vocab int
---@field max_seq_len int
---@field epochs int
---@field lr float
---@field min_lr? float
---@field warmup_epochs? int
---@field save_every? int
---@field gen_every? int
---@field prompt? string
---@field max_ram_gb? float
---@field enable_compression? bool
---@field enable_htop? bool
---@field enable_viz? bool


--=============================================================================
-- Module: Pipeline
--=============================================================================

---@class Pipeline
Pipeline = {}

---@param cfg PipelineConfig|table
---@return boolean ok
---@return string? err
function Pipeline.setup(cfg) end

---Exécuter un pipeline training complet (si présent).
---@return boolean ok
---@return string? err
function Pipeline.run() end

---Sauvegarder un checkpoint pipeline (si présent).
---@param tag string
---@return boolean ok
---@return string? err
function Pipeline.save(tag) end

---Tenter une reprise depuis un checkpoint (si présent).
---@param dir string
---@return boolean ok
---@return string? err
function Pipeline.resume(dir) end

--=============================================================================
-- Exports globaux (pour l'IDE)
--=============================================================================
---@type MimirModelAPI
Mimir.Model = Mimir.Model
---@type MimirArchitecturesAPI
Mimir.Architectures = Mimir.Architectures
---@type MimirLayersAPI
Mimir.Layers = Mimir.Layers
---@type MimirCheckpointAPI
Mimir.Checkpoint = Mimir.Checkpoint
---@type MimirTokenizerAPI
Mimir.Tokenizer = Mimir.Tokenizer
---@type MimirDatasetAPI
Mimir.Dataset = Mimir.Dataset
---@type MimirMemoryAPI
Mimir.Memory = Mimir.Memory
---@type MimirGuardAPI
Mimir.Guard = Mimir.Guard
---@type MimirMemoryGuardAPI
Mimir.MemoryGuard = Mimir.MemoryGuard
---@type MimirAllocatorAPI
Mimir.Allocator = Mimir.Allocator
---@type MimirHtopAPI
Mimir.Htop = Mimir.Htop
---@type MimirVizAPI
Mimir.Viz = Mimir.Viz

---@type MimirVizAPI
Mimir.visualiser = Mimir.visualiser

---@type MimirSerializationAPI
Mimir.Serialization = Mimir.Serialization

---@type MimirNeuroPulseAPI
Mimir.NeuroPulse = Mimir.NeuroPulse
