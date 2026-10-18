---@meta
---@version 3.5.0
---@author <bri45> for "Mímir Framework"
---@date 1 octobre 2026 (dernière sync)
---@diagnostic disable: missing-return, unused-local, unused-vararg, duplicate-doc-field, redundant-parameter

--=============================================================================
-- Mímir Framework v3.5 — IDE Stub (EmmyLua)
--=============================================================================
-- Ce fichier est un "stub" destiné aux IDE (LuaLS / EmmyLua / IntelliJ, etc.).
-- Il documente l'API globale exposée par le binaire `mimir` (bindings C/C++).
--
-- ⚠️  IMPORTANT: Ce fichier est synchronisé avec
--    src/scriptings/Lua/luaScripting/LuaScripting.cpp
--    Toute modification de l'API C++ doit être reflétée ici.
--    Dernière synchronisation: 1 octobre 2026
--  • Alias `Mimir.model` (lowercase) + stub explicite `Mimir.model.dtype`
--  • Operations multi-input complètes (Add, Multiply, Concat, MatMul, Split)
--  • Configuration live du modèle dans Htop/Viz (`C`, édition avec `Entrée`)
-- Historique v2.0.0 :
-- Namespace Mimir
---@class Mimir
Mimir = {}

---@class Mimir
---@field Model MimirModelAPI
---@field model MimirModelAPI @Alias de `Mimir.Model`
---@field Benchmark MimirBenchmarkAPI
---@field Architectures MimirArchitecturesAPI
---@field Layers MimirLayersAPI
---@field Checkpoint MimirCheckpointAPI
---@field Tokenizer MimirTokenizerAPI
---@field Dataset MimirDatasetAPI
---@field Database MimirDatabaseAPI
---@field IO MimirIOAPI
---@field Memory MimirMemoryAPI
---@field Guard MimirGuardAPI
---@field MemoryGuard MimirMemoryGuardAPI
---@field Allocator MimirAllocatorAPI
---@field Htop MimirHtopAPI
---@field Viz MimirVizAPI
---@field Serialization MimirSerializationAPI

--=============================================================================
-- Aliases / Types de base
--=============================================================================

---@alias int integer
---@alias float number
---@alias bool boolean
---@alias bytes integer

---@alias TokenId integer
---@alias TokenIds TokenId[]
---@alias TokenText string

---@class MimirBenchmarkAPI
Mimir.Benchmark = {}

---Retourne le temps mural monotone en secondes avec une résolution haute précision.
---Cette valeur sert à mesurer des intervalles; son origine est volontairement non définie.
---@return number seconds
function Mimir.Benchmark.now() end

---@alias ArchitectureName
---| "basic_mlp"
---| "causal_lm"
---| "deeplab"
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
---| "patch_discriminator"
---| "r_cnn"
---| "ssd"
---| "yolo"
---| "vgg16"
---| "vgg16_feat"
---| "vgg19"
---| "diffusion"
---| "lumen_diffusion"
---| "gan_latent"
---| "Gan_Style"
---| "cond_diffusion"
---| "sd3_5"
---| "external_safetensors_base"
---| "hf_clip_text_encoder_1"
---| "hf_clip_text_encoder_2"
---| "hf_vae_decoder"
---| "hf_sdxl_transformer_block"

---@alias ModelType ArchitectureName

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

---@alias DTypeName
---| "float32"|"f32"|"float"
---| "float64"|"f64"|"double"
---| "float16"|"f16"|"fp16"
---| "bfloat16"|"bf16"
---| "int8"|"i8"
---| "uint8"|"u8"
---| "int16"|"i16"
---| "uint16"|"u16"
---| "int32"|"i32"
---| "uint32"|"u32"
---| "int64"|"i64"
---| "uint64"|"u64"
---| "bool"|"b1"

--=============================================================================
-- Configs de modèles
--=============================================================================

---@class ModelConfig
---@field type? ModelType @Injecté côté C++ lors de `Model.create()` (metadata)
---@field dtype? DTypeName @Préférence dtype du modèle (si supporté par la runtime)
---@field dropout? float @Dropout générique (si supporté par l'architecture)
---@field optimizer? string @"sgd"|"adam"|"adamw" (utilisé par `Model.train()`)
---@field beta1? float @Modifiable en direct si un optimiseur courant est publié; 0 <= valeur < 1
---@field beta2? float @Modifiable en direct si un optimiseur courant est publié; 0 <= valeur < 1
---@field epsilon? float @Modifiable en direct si un optimiseur courant est publié; valeur > 0
---@field weight_decay? float @Modifiable en direct si un optimiseur courant est publié; valeur >= 0
---@field rmsprop_alpha? float @Modifiable en direct si un optimiseur courant est publié; 0 <= valeur < 1
---@field grad_clip_norm? float @Clipping global modifiable en direct; 0 désactive le clipping
---@field clip_norm? float @Alias historique utilisé comme valeur initiale de grad_clip_norm
---@field use_kv_cache? boolean @Modifiable en direct lorsque la clé existe dans la configuration
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
---@field csv_file? string @Chemin CSV htop explicite (ex: "runs/myrun/metrics.csv") — appliqué par `Model.train()` à tous les types de modèles
---@field csv_path? string @Alias de csv_file
---@field csv_dir? string @Dossier CSV : génère `{csv_dir}/{name}_htop_metrics.csv`
---@field viz_taps_max_frames? int @Limite frames "viz taps" (si viz active)
---@field viz_taps_max_side? int @Limite taille preview "viz taps" (si viz active)
---@field validate_every_steps? int @Validation toutes les N étapes d'optimizer (0 = désactivé)
---@field validate_items? int @Nombre d'items évalués par validation
---@field validate_holdout? bool @Active un split holdout (si supporté par le script/archi)
---@field validate_holdout_frac? float @Fraction du dataset allouée au holdout
---@field validate_holdout_items? int @Nombre d'items holdout (optionnel, selon script)
---@field validate_save_debug? bool @Sauvegarde un checkpoint debug (ex: _val_debug.json) lors des validations
---@field kl_beta? float @Coefficient KL modifiable en direct pour les modèles VAE
---@field vae_kl_beta? float @Valeur initiale historique de kl_beta
---@field kl_warmup_steps? int @Warmup KL modifiable en direct pour les modèles VAE
---@field recon_loss? string @Loss VAE modifiable en direct: mse, mae/l1, huber, charbonnier, gaussian_nll ou bce
---@field ssim_weight? float @Poids SSIM modifiable en direct pour les VAE image
---@field spectral_weight? float @Poids spectral modifiable en direct pour les VAE image
---@field marker_wass_scale? float @Échelle Wasserstein modifiable en direct pour les VAE image
---@field marker_temp_scale? float @Échelle temporelle modifiable en direct pour les VAE image
---@field huber_delta? float @Delta Huber modifiable en direct; valeur > 0
---@field charbonnier_eps? float @Epsilon Charbonnier modifiable en direct; valeur > 0
---@field gaussian_nll_sigma? float @Sigma Gaussian NLL modifiable en direct; valeur > 0
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

---@class CausalLMConfig: ModelConfig
---@field vocab_size? int
---@field padding_idx? int
---@field seq_len? int
---@field d_model? int
---@field num_layers? int
---@field num_heads? int
---@field num_kv_heads? int @Nombre de têtes K/V; doit diviser num_heads (GQA)
---@field mlp_hidden? int @Largeur intermédiaire SwiGLU
---@field norm_eps? float
---@field rope_theta? float

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
---@field bbox_input_w? integer
---@field bbox_input_h? integer
---@field bbox_input_c? integer
---@field bbox_input_layout? string @HWC ou CHW
---@field bbox_input_prefix_tokens? integer @Tokens globaux préservés ; défaut 1 pour ViT
---@field bbox_output_w? integer
---@field bbox_output_h? integer
---@field bbox_output_c? integer
---@field bbox_output_layout? string @HWC ou CHW
---@field relation_primary_weight? float @Poids relationnel du sujet principal ; défaut 3
---@field relation_secondary_weight? float @Poids relationnel du sujet secondaire ; défaut 2
---@field relation_other_weight? float @Poids relationnel des autres rôles ; défaut 1
---@field bbox_loss_weight? float @Pondération des régions annotées ; défaut 1, 0 désactive
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
---@field base_channels? int
---@field stochastic_latent? boolean
---@field resnet? boolean
---@field attention? boolean
---@field enc_norm? string
---@field enc_gn_groups? int
---@field dec_norm? string
---@field dec_gn_groups? int
---@field decoder_upsample? string
---@field attn_heads? int
---@field resnet_max_tokens? int
---@field attn_max_tokens? int
---@field text_cond? boolean
---@field vocab_size? int
---@field seq_len? int
---@field text_d_model? int
---@field proj_dim? int

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

---@class MedicalDetectorConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field num_classes? int
---@field class_names? string[]
---@field object_loss_weight? number
---@field box_loss_weight? number
---@field class_loss_weight? number
---@field object_pos_weight? number
---@field validation_fraction? number
---@field validation_items? int
---@field validate_every_epochs? int
---@field score_threshold? number
---@field nms_iou_threshold? number
---@field match_iou_threshold? number
---@field max_detections? int

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

---@class VGG16FeatConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field base_channels? int
---@field enc_norm? 'groupnorm'|'lineargroup' Normalisation appliquée après les convolutions.
---@field enc_gn_groups? int Nombre maximal de groupes pour GroupNorm.

---@class DiffusionConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field time_dim? int
---@field hidden_dim? int

---@class LumenDiffusionConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field latent_w? int
---@field latent_h? int
---@field latent_c? int
---@field vae_checkpoint? string
---@field vae_scale? float
---@field vae_shift? float
---@field patch_size? int
---@field hidden_size? int
---@field depth? int
---@field mlp_ratio? float
---@field vocab_size? int
---@field text_seq_len? int
---@field text_layers? int
---@field num_heads? int
---@field diffusion_steps? int
---@field beta_start? float
---@field beta_end? float
---@field preview_timestep? int

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

---@class ExternalSafeTensorsBaseConfig: ModelConfig
---@field source_safetensors string @Chemin vers le fichier safetensors source à refléter
---@field include_prefixes? string[] @Limiter la création aux tenseurs commençant par un de ces préfixes
---@field exclude_prefixes? string[] @Exclure les tenseurs commençant par un de ces préfixes
---@field max_tensors? int @Limiter le nombre de tenseurs créés (0=tous)

---@class HFCLIPTextEncoder1Config: ModelConfig
---@field vocab_size? int
---@field padding_idx? int
---@field seq_len? int
---@field d_model? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_hidden? int
---@field causal? boolean

---@class HFCLIPTextEncoder2Config: ModelConfig
---@field vocab_size? int
---@field padding_idx? int
---@field seq_len? int
---@field d_model? int
---@field num_layers? int
---@field num_heads? int
---@field mlp_hidden? int
---@field proj_dim? int
---@field causal? boolean
---@field include_logit_scale? boolean

---@class HFVaeDecoderConfig: ModelConfig
---@field image_w? int
---@field image_h? int
---@field image_c? int
---@field latent_w? int
---@field latent_h? int
---@field latent_c? int
---@field num_heads? int
---@field norm_groups? int

---@class HFSDXLTransformerBlockConfig: ModelConfig
---@field q_len? int
---@field kv_len? int
---@field d_model? int
---@field context_dim? int
---@field num_heads? int
---@field ff_hidden? int
---@field self_attn_qkv_bias? boolean
---@field self_attn_out_bias? boolean
---@field cross_attn_out_bias? boolean

-- NOTE: les schémas ci-dessus reflètent le registre C++ (ModelArchitectures::defaultConfig).
-- Pour obtenir la config exacte à jour côté runtime:
--   local cfg = Mimir.Architectures.default_config("transformer")
--   local ok = Mimir.Model.create("transformer", cfg)

--=============================================================================
-- Stats / Structs
--=============================================================================

---@class FpgaCaps
---@field detected boolean
---@field connected boolean
---@field status string
---@field device? string
---@field protocol? string
---@field layer_acceleration boolean @false: FPGA used for I/O, not model layers
---@field dataset_buffer? boolean
---@field checkpoint_controller? boolean
---@field tensor_conversion? boolean @F32 <-> F16/BF16 on FPGA
---@field buffer_input_bytes? integer @256-byte blocks (BRAM, not SDRAM)

---@class HardwareCaps
---@field fpga? FpgaCaps
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
---@field recon_loss_type? string @Label effectif de la loss de reconstruction
---@field optimizer? string|table @Nom ou table {type,step,beta1,beta2,eps,weight_decay}

---@class HtopCreateConfig
---@field enable_htop? boolean @Active l'interface terminal; défaut true
---@field defer_htop? boolean @Diffère l'interface terminal jusqu'à la première métrique; défaut false
---@field enable_viz? boolean @Active le backend Viz compilé (alias: viz)
---@field viz? boolean @Alias de enable_viz
---@field viz_config? table @Config passée au Visualizer (mêmes clés que config.json/visualization)
---@field csv? boolean @Active l'export CSV des métriques côté htop
---@field csv_enabled? boolean @Alias de csv
---@field csv_path? string @Chemin du CSV (ex: "checkpoint/run1/htop_metrics.csv")
---@field csv_file? string @Alias de csv_path

---@class VizMetrics
---@field epoch? int
---@field total_epochs? int
---@field batch? int
---@field total_batches? int
---@field step? int @Alias utilisé comme batch si batch est absent
---@field loss? float
---@field avg_loss? float
---@field lr? float
---@field batch_time_ms? int
---@field memory_mb? float
---@field bps? float
---@field params? int
---@field mse? float
---@field kl? float
---@field wass? float
---@field ent? float
---@field mom? float
---@field spat? float
---@field temp? float
---@field timestep? float
---@field grad_norm? float
---@field grad_max? float
---@field kl_beta_effective? float
---@field recon_loss_type? string @Label effectif de la loss de reconstruction

---@class LumenVaeCalibrationStats
---@field items integer
---@field values integer
---@field shift float
---@field scale float

---@class LumenStepStats
---@field skeleton_region_count integer @Régions des points, segments et associations du squelette
---@field mse_unweighted float @MSE globale avant pondération spatiale
---@field bbox_count integer
---@field bbox_loss_weight float
---@field relation_count integer @Relations entre deux régions utilisées
---@field relation_loss float @Contribution de la perte relationnelle
---@field loss float
---@field mse float
---@field kl float
---@field kl_beta_effective float
---@field grad_norm float
---@field grad_max_abs float
---@field reconstruction_mae float
---@field reconstruction_mse float
---@field wasserstein float
---@field entropy_diff float
---@field moment_mismatch float
---@field spatial_coherence float
---@field temporal_consistency float
---@field timestep integer
---@field stop_requested boolean
---@field learning_rate? float @LR réellement appliqué après contrôle live Viz

---@class LumenStepMonitorOptions
---@field annotation? DatasetAnnotation
---@field bbox_loss_weight? number @Défaut 1 ; 0 désactive la pondération
---@field epoch? integer
---@field total_epochs? integer
---@field batch? integer
---@field total_batches? integer
---@field step? integer
---@field completed_steps? integer
---@field loss_sum_before? float
---@field lr_warmup_steps? integer

--=============================================================================
-- Module: Mimir.Model
--=============================================================================

---@class GanStyleConfig
---@field image_w? integer
---@field image_h? integer
---@field image_c? integer @3 (RGB CHW, valeurs [-1,1])
---@field base_channels? integer @Largeur du générateur (16)
---@field discriminator_channels? integer @Largeur du discriminateur (16)
---@field content_weight? number @Poids L1 (10)
---@field discriminator_lr? number @SGD discriminateur (0.001)

---@class MimirModelAPI
Mimir.Model = {}

---Créer un modèle via le registre d'architectures (C++).
---Le modèle est construit via le registre C++ immédiatement.
---Note: `Mimir.Model.create()` ne fait plus allocate/init automatiquement.
---Utilisez ensuite `Mimir.Model.allocate_params()` et `Mimir.Model.init_weights()` si nécessaire.
---Un chemin finissant par `.mpk` est décodé, vérifié puis créé via le registre C++.
---@overload fun(model_type: string, config?: nil): (boolean, string?)
---@overload fun(model_type: "basic_mlp", config?: BasicMLPConfig): (boolean, string?)
---@overload fun(model_type: "causal_lm", config?: CausalLMConfig): (boolean, string?)
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
---@overload fun(model_type: "vgg16_feat", config?: VGG16FeatConfig): (boolean, string?)
---@overload fun(model_type: "diffusion", config?: DiffusionConfig): (boolean, string?)
---@overload fun(model_type: "lumen_diffusion", config?: LumenDiffusionConfig): (boolean, string?)
---@overload fun(model_type: "cond_diffusion", config?: CondDiffusionConfig): (boolean, string?)
---@overload fun(model_type: "gan_latent", config?: GanLatentConfig): (boolean, string?)
---@overload fun(model_type: "Gan_Style", config?: GanStyleConfig): (boolean, string?)
---@overload fun(model_type: "sd3_5", config?: SD35Config): (boolean, string?)
---@overload fun(model_type: "external_safetensors_base", config: ExternalSafeTensorsBaseConfig): (boolean, string?)
---@overload fun(model_type: "hf_clip_text_encoder_1", config?: HFCLIPTextEncoder1Config): (boolean, string?)
---@overload fun(model_type: "hf_clip_text_encoder_2", config?: HFCLIPTextEncoder2Config): (boolean, string?)
---@overload fun(model_type: "hf_vae_decoder", config?: HFVaeDecoderConfig): (boolean, string?)
---@overload fun(model_type: "hf_sdxl_transformer_block", config?: HFSDXLTransformerBlockConfig): (boolean, string?)
---@param model_type ModelType
---@param config? ModelConfig|BasicMLPConfig|CausalLMConfig|TransformerConfig|ViTConfig|VAEConfig|ResNetConfig|UNetConfig|MobileNetConfig|VGG16Config|VGG19Config|DiffusionConfig|ExternalSafeTensorsBaseConfig|HFCLIPTextEncoder1Config|HFCLIPTextEncoder2Config|HFVaeDecoderConfig|HFSDXLTransformerBlockConfig|table
---@return boolean ok
---@return string? err
function Mimir.Model.create(model_type, config) end

---Retourner la configuration effective du modèle construit.
---Elle peut différer de la requête lorsque le modèle s'adapte à un checkpoint externe.
---@return ModelConfig|table|nil config
---@return string? err
function Mimir.Model.get_config() end

---Créer un modèle vide (hors registre) pour importer une architecture nodale custom.
---Utilisé pour les MPK standalone quand le type n'existe pas dans le registre.
---@param model_type string
---@param config? table
---@return boolean ok
---@return string? err
function Mimir.Model.create_empty(model_type, config) end

---Créer un modèle à partir d'une config "complète" (injection de conf externe).
---Retourne (ok, arch_or_err).
---@param cfg table
---@return boolean ok
---@return string? arch_or_err
function Mimir.Model.create_from_config(cfg) end

---[COMPAT] Reconstruit le modèle courant via le registre.
---Préférez `Mimir.Model.create(type, cfg)`.
---Retour: ok + nombre de paramètres (scalars).
---@return boolean ok
---@return integer|string params_or_err
function Mimir.Model.build() end

---Entraîner le modèle courant.
---Nécessite un dataset chargé et des séquences préparées via `Mimir.Dataset.prepare_sequences(max_seq_len)`.
---@param epochs integer @Nombre d'epochs
---@param learning_rate number @LR (ex: 3e-4)
---@return boolean ok
---@return integer|string? step_or_err @Step global final si ok, message sinon
function Mimir.Model.train(epochs, learning_rate) end

---@class VisionStepOptions
---@field annotation? DatasetAnnotation
---@field learning_rate? number @Défaut 0.0001
---@field optimizer? string @Défaut adamw
---@field loss_weights? number[] @Masque ou pondération supplémentaire
---@field text_ids? integer[] @Conditionnement VAE optionnel

---Entraîner un modèle de vision natif avec des tensors dans son format d'entrée/sortie.
---@param input number[] @Valeurs déjà normalisées ; disposition attendue par le modèle
---@param target number[] @Cible propre à la tâche du modèle
---@param options? VisionStepOptions
---@return table|nil stats
---@return string? error
function Mimir.Model.vision_train_step(input, target, options) end


---Inférence sur un prompt (string) ou une séquence de tokens.
---Retour: string (texte), nil pour une entrée invalide, ou `(false, err)` sans modèle.
---@param input string|TokenIds
---@return string|nil|false output
---@return string? err
function Mimir.Model.infer(input) end

---[Alias] `Mimir.model` (lowercase) pointe vers `Mimir.Model`.
---Beaucoup de scripts utilisent `Mimir.model.dtype(...)`.
---@type MimirModelAPI
Mimir.model = Mimir.Model

---[Alias] Stub explicite pour l'EmmyLua: `Mimir.model.dtype`.
---Voir `Mimir.Model.dtype` pour la doc complète.
---@overload fun(): DTypeName|false, string?
---@overload fun(dtype: DTypeName|string): (boolean, string)
function Mimir.model.dtype(dtype) end

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
---@return integer|string params_or_err @Nombre alloué si ok, message sinon
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

---Retourne la liste des layers du modèle courant.
---@return table layers
function Mimir.Model.get_layers() end

---Supprime tous les layers du modèle courant.
---Utile avant de ré-appliquer un graphe nodal importé.
---@return boolean ok
---@return integer|string? old_count_or_err
function Mimir.Model.clear_layers() end

---Pousser une couche (API bas niveau / description) dans le modèle.
---Note: cette fonction est surtout utilisée par les builders d'architectures.
---@param name string
---@param layer_type string
---@param params_count integer
---@param params? table @Configuration propre au layer, utilisée par les graphes MPK dynamiques
---@return boolean ok
---@return string? err
function Mimir.Model.push_layer(name, layer_type, params_count, params) end

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
---@param input TokenIds|float[]|table<string, number[]> @Liste simple ou tenseurs nommés; `text_ids` conserve le chemin entier
---@param training? bool @Mode training (défaut: true) pour calculer les gradients
---@return float[]|nil @Sortie du modèle
---@return string? err
function Mimir.Model.forward(input, training) end

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

---Commencer une calibration globale des latents du VAE associé à Lumen.
---@return boolean ok
---@return string? err
function Mimir.Model.lumen_begin_vae_calibration() end

---Ajouter une image RGB u8 à la calibration VAE courante.
---@param image integer[]
---@return LumenVaeCalibrationStats|nil stats
---@return string? err
function Mimir.Model.lumen_add_vae_calibration_image(image) end

---Finaliser et persister la calibration VAE courante.
---@return LumenVaeCalibrationStats|nil stats
---@return string? err
function Mimir.Model.lumen_finish_vae_calibration() end

---Effectuer un step de diffusion Lumen et publier ses métriques vers Htop/Viz.
---@param image integer[]
---@param prompt string
---@param seed integer
---@param learning_rate float
---@param optimizer? string @"sgd"|"adam"|"adamw"
---@param monitor? LumenStepMonitorOptions
---@return LumenStepStats|nil stats
---@return string? err
function Mimir.Model.lumen_train_step(image, prompt, seed, learning_rate, optimizer, monitor) end

---Générer une image avec le modèle Lumen chargé, en mode inférence.
---@param prompt string
---@param seed? integer
---@param steps? integer
---@param guidance? number
---@return integer[]|nil pixels @Octets entrelacés HWC ; nil en cas d'erreur.
---@return integer|string width_or_error
---@return integer? height
---@return integer? channels
function Mimir.Model.lumen_text2img(prompt, seed, steps, guidance) end

---Effectuer une validation Lumen et publier ses previews vers Viz.
---@param image integer[]
---@param prompt string
---@param seed integer
---@return LumenStepStats|nil stats
---@return string? err
---@param options? LumenStepMonitorOptions @Annotation et pondération spatiale
function Mimir.Model.lumen_validate_step(image, prompt, seed, options) end

---Step optimiseur (si exposé). Le LR peut être transmis.
---@param learning_rate number
---@param opt_type? string @"sgd"|"adam"|"adamw" (défaut: "adamw")
---@return boolean ok
---@return string? err
function Mimir.Model.optimizer_step(learning_rate, opt_type) end

---Active/désactive l'accélération matérielle côté modèle.
---Note: le binding actuel prend un booléen (il ne sélectionne pas un backend nommé).
---@param enable boolean @true pour activer l'accélération (si dispo), false pour forcer CPU
---@return boolean ok
function Mimir.Model.set_hardware(enable) end

---Retourne les capacités CPU et le statut de connexion/contrôle FPGA.
---@return HardwareCaps caps
function Mimir.Model.hardware_caps() end

---Construire le contexte natif : entités, bbox, squelettes, relations, attributs et description.
---Indépendant du poids de loss spatiale ; le modèle tokenise ensuite ce texte.
---@param caption string
---@param annotation? table @Annotation structurée du dataset
---@return string|nil prompt
---@return string? err
function Mimir.Model.conditioning_prompt(caption, annotation) end


---@alias RuntimeOperationCapability 'Unsupported'|'HostFallback'|'Native'|'NativeOptimized'

---@class RuntimeOperationInfo
---@field type string @Type de couche enregistré
---@field forward RuntimeOperationCapability
---@field backward RuntimeOperationCapability

---@class RuntimeOperationsInfo
---@field name 'CPU'|'CUDA'|'ROCM'|'VULKAN'|'OPENCL'
---@field compiled boolean @Backend inclus dans cette compilation
---@field scope string @Portée des déclarations de capacités
---@field ops RuntimeOperationInfo[]

---Lister les capacités déclarées par type de couche pour les cinq backends.
---Les périphériques, configurations et formes de tenseurs ne sont pas testés par cet appel.
---@return RuntimeOperationsInfo[] runtimes
function Mimir.Model.runtime_ops() end


---Lire ou définir le dtype par défaut du modèle courant.
---Getter: `Mimir.Model.dtype()` -> string
---Setter: `Mimir.Model.dtype("float16")` -> (ok, dtype|err)
---@overload fun(): DTypeName|false, string?
---@overload fun(dtype: DTypeName|string): (boolean, string)
function Mimir.Model.dtype(dtype) end

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
---@overload fun(name: "causal_lm"): CausalLMConfig
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
---@overload fun(name: "vgg16_feat"): VGG16FeatConfig
---@overload fun(name: "diffusion"): DiffusionConfig
---@overload fun(name: "cond_diffusion"): CondDiffusionConfig
---@overload fun(name: "gan_latent"): GanLatentConfig
---@overload fun(name: "Gan_Style"): GanStyleConfig
---@overload fun(name: "sd3_5"): SD35Config
---@overload fun(name: "external_safetensors_base"): ExternalSafeTensorsBaseConfig
---@overload fun(name: "hf_clip_text_encoder_1"): HFCLIPTextEncoder1Config
---@overload fun(name: "hf_clip_text_encoder_2"): HFCLIPTextEncoder2Config
---@overload fun(name: "hf_vae_decoder"): HFVaeDecoderConfig
---@overload fun(name: "hf_sdxl_transformer_block"): HFSDXLTransformerBlockConfig
---@param name ArchitectureName|string
---@return table|nil config
---@return string? err
function Mimir.Architectures.default_config(name) end

---@class ArchitectureInfo
---@field name string @Nom canonique de l'architecture (clé du registry).
---@field description string @Description courte (peut être vide).
---@field config table @Config par défaut complète (peut contenir un champ `dtype`).
---@field origin "native"|"mpk" @Origine explicite de l'entrée du registre.
---@field source_path? string @Chemin du package pour une architecture MPK.

---Lire toutes les infos du registry pour une (ou toutes les) architecture(s).
---Sans argument: renvoie la liste complète des entrées du registry.
---Avec un nom: renvoie l'entrée correspondante, ou `(nil, err)` si inconnue.
---@overload fun(): ArchitectureInfo[]
---@overload fun(name: ArchitectureName|string): ArchitectureInfo|nil, string?
---@param name? ArchitectureName|string
---@return ArchitectureInfo[]|ArchitectureInfo|nil info
---@return string? err
function Mimir.Architectures.info(name) end

---@class DTypeInfo
---@field name string @Nom canonique (ex: "float32", "bfloat16").
---@field aliases string @Alias acceptés, séparés par des virgules (ex: "f32, float32").
---@field bytes integer @Taille en octets d'un élément.
---@field kind "float"|"int"|"uint"|"bool" @Famille du dtype.

---Lister les dtypes pris en charge par le framework.
---@return DTypeInfo[] dtypes
function Mimir.Architectures.dtypes() end

--=============================================================================
-- Module: Mimir.Layers (inspection read-only du modèle courant)
--=============================================================================

---@class MimirLayersAPI
Mimir.Layers = {}

---Lister tous les types de layers supportés par le framework, basés sur `LayerType`.
---@return string[] layer_types
function Mimir.Layers.available() end

---Lister les layers du modèle courant pour un type canonique donné.
---Exemple: `Mimir.Layers.by_type("Conv2d")`.
---@param layer_type string
---@return MimirLayerInfo[] layers
function Mimir.Layers.by_type(layer_type) end

---@class MimirLayerInfo
---@field index integer
---@field name string
---@field type string
---@field param_count integer
---@field output string
---@field inputs string[]
---@field in_features integer
---@field out_features integer
---@field in_channels integer
---@field out_channels integer
---@field kernel_size integer
---@field stride integer
---@field padding integer
---@field seq_len integer
---@field embed_dim integer
---@field num_heads integer
---@field vocab_size integer
---@field input_height integer
---@field input_width integer
---@field kernel_h? integer
---@field kernel_w? integer
---@field stride_h? integer
---@field stride_w? integer
---@field pad_h? integer
---@field pad_w? integer
---@field dilation? integer
---@field groups? integer
---@field eps? number
---@field num_groups? integer
---@field dropout_p? number
---@field axis? integer
---@field concat_axis? integer
---@field split_axis? integer
---@field num_splits? integer
---@field scale_h? number
---@field scale_w? number
---@field out_h? integer
---@field out_w? integer
---@field head_dim? integer
---@field causal? boolean
---@field use_bias? boolean
---@field nms_iou_threshold? number
---@field nms_score_threshold? number
---@field nms_max_detections? integer
---@field nms_class_agnostic? boolean
---@field target_shape? integer[]
---@field permute_dims? integer[]
---@field split_sizes? integer[]

---Liste les layers de type convolution présents dans le modèle courant.
---API en lecture seule: ne modifie jamais le modèle.
---@return MimirLayerInfo[] layers
function Mimir.Layers.conv2d(...) end

---Liste les layers `Conv2d` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Conv2d() end

---Liste les layers `ConvTranspose2d` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.ConvTranspose2d() end

---Liste les layers `Conv1d` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Conv1d() end

---Liste les layers `DepthwiseConv2d` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.DepthwiseConv2d() end

---Liste les layers linéaires du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.linear(...) end

---Liste les layers `Linear` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Linear() end

---Liste les layers `Bilinear` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Bilinear() end

---Liste les layers `Embedding` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Embedding() end

---Liste les layers `EmbeddingBag` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.EmbeddingBag() end

---Liste les layers `BatchNorm2d` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.BatchNorm2d() end

---Liste les layers `BatchNorm1d` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.BatchNorm1d() end

---Liste les layers `LayerNorm` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.LayerNorm() end

---Liste les layers `GroupNorm` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.GroupNorm() end

---Liste les layers `InstanceNorm2d` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.InstanceNorm2d() end

---Liste les layers `RMSNorm` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.RMSNorm() end

---Liste les layers `ReLU` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.ReLU() end

---Liste les layers `LeakyReLU` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.LeakyReLU() end

---Liste les layers `GELU` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.GELU() end

---Liste les layers `GEGLU` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.GEGLU() end

---Liste les layers `SiLU` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.SiLU() end

---Liste les layers `Tanh` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Tanh() end

---Liste les layers `Sigmoid` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Sigmoid() end

---Liste les layers `Softmax` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Softmax() end

---Liste les layers `LogSoftmax` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.LogSoftmax() end

---Liste les layers `Softplus` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Softplus() end

---Liste les layers `Mish` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.Mish() end

---Liste les layers `HardSigmoid` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.HardSigmoid() end

---Liste les layers `HardSwish` du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.HardSwish() end

-- Chaque type canonique retourné par `Mimir.Layers.available()` est aussi
-- enregistré comme fonction d'inspection `Mimir.Layers.<Type>()`.
---@return MimirLayerInfo[] layers
function Mimir.Layers.MaxPool2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.AvgPool2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.AdaptiveAvgPool2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.GlobalAvgPool2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.MaxPool1d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.AvgPool1d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.TokenMeanPool() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Dropout() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Dropout2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.AlphaDropout() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Flatten() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Reshape() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Transpose() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Permute() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Squeeze() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Unsqueeze() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.View() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Add() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Subtract() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Multiply() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Divide() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Concat() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Split() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Chunk() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Stack() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.MatMul() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.BatchMatMul() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.NMS() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.SelfAttention() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.MultiHeadAttention() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.CrossAttention() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.UpsampleNearest() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.UpsampleBilinear() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.UpsampleBicubic() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.PixelShuffle() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.LSTM() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.GRU() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.RNN() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.ZeroPad2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.ReflectionPad2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.ReplicationPad2d() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Identity() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Constant() end
---@return MimirLayerInfo[] layers
function Mimir.Layers.Lambda() end

---Liste les layers de pooling max du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.maxpool2d(...) end

---Liste les layers de pooling moyen du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.avgpool2d(...) end

---Liste les layers d'activation du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.activation(...) end

---Liste les layers de normalisation de type batch/instance du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.batchnorm(...) end

---Liste les layers de normalisation de type layer/group/rms du modèle courant.
---@return MimirLayerInfo[] layers
function Mimir.Layers.layernorm(...) end

---Liste les layers d'attention du modèle courant.
---@return MimirLayerInfo[] layers
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

---@class DatasetAnnotationEntity
---@field class_id? integer @Classe du détecteur ; défaut 0
---@field id string
---@field phrase string
---@field role string
---@field bbox? number[] @Coordonnées [x_min, y_min, x_max, y_max] dans [0,1]

---@class DatasetAnnotation
---@field image string @Nom de fichier de l'image associée
---@field captions {primary: string, alternatives?: string[]}
---@field entities? DatasetAnnotationEntity[]
---@field attributes? {entity: string, property: string, value: string}[]
---@field relations? {subject: string, predicate: string, object: string}[]
---@field lighting? {description: string, source: string, target: string}
---@field composition? {description: string}
---@field quality? {caption_verified: boolean, relations_verified: boolean}

---@class DatasetItem
---@field text_file? string Chemin du fichier texte
---@field image_file? string Chemin du fichier image
---@field audio_file? string Chemin du fichier audio
---@field video_file? string Chemin du fichier vidéo
---@field text? string @Description captions.primary si annotation JSON, sinon texte historique
---@field annotation_file? string @Chemin du JSON structuré prioritaire
---@field annotation? DatasetAnnotation @Document complet chargé à la demande, champs supplémentaires conservés
---@field image? integer[] Pixels RGB u8 redimensionnés, si demandés et décodables
---@field width? int Largeur de l'image
---@field height? int Hauteur de l'image
---@field channels? int Nombre de canaux de l'image retournée
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
---Charge les pixels RGB par défaut; passer false pour ne lire que texte et métadonnées.
---Une annotation JSON structurée est prioritaire sur le .txt; son captions.primary devient item.text.
---@param index integer Index de l'item (commence à 1)
---@param load_image? boolean Charge et retourne le tableau image (défaut: true)
---@return DatasetItem|nil item Item du dataset
---@return string? err Message d'erreur si échec
function Mimir.Dataset.get(index, load_image) end

---Libérer les données lazy actuellement chargées pour un item.
---@param index integer Index de l'item (commence à 1)
---@return boolean ok
---@return string? err
function Mimir.Dataset.release(index) end

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
---Active automatiquement l'éviction LRU avec spill disque.
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
---- L'export CSV est unique: Viz le porte lorsqu'elle est active, Htop sinon.
---- `csv` active/désactive cet export commun; `csv_path`/`csv_file` choisit son chemin.
---- `C` ouvre le panneau de configuration du modèle courant; `Entrée` édite puis valide.
---- Les changements sont consommés par le thread d'entraînement à une frontière de calcul sûre.
---
---@param config? boolean|HtopCreateConfig
---@return boolean ok
---@return string? err
function Mimir.Htop.create(config) end

---Arrêter le monitor si `enabled=false`.
---Avec `enabled=true`, la fonction confirme seulement l'état; utilisez `create()` pour démarrer.
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

---Compatibilité: le rendu est automatique, cette fonction est un no-op qui retourne true.
---@return boolean ok
---@return string? err
function Mimir.Htop.render() end

---Clear écran / reset UI.
---@return boolean ok
---@return string? err
function Mimir.Htop.clear() end

--=============================================================================
-- Module: Mimir.Viz (SFML / QT / GTK / WEB)
--=============================================================================

---@class MimirVizAPI
Mimir.Viz = {}

---Créer la fenêtre visualiseur avec titre et dimensions optionnels.
---Argument réel: table de configuration (passée à AsyncMonitor.start()).
---`C` ouvre le panneau de configuration du modèle courant; `Entrée` édite puis valide.
---@param config? table @Ex: {visualization={enabled=true, window_title="..."}}
---@return boolean ok
---@return string? err
function Mimir.Viz.create(config) end

---Backend compilé, y compris avant create().
---@return string backend @SFML, QT, GTK, WEB ou NONE
function Mimir.Viz.backend() end

---Modifier la scène sur le thread Viz ; seuls les champs fournis changent.
---panels: {id=0..6,x?,y?,w?,h?,title?,visible?}[]
---controls: {id=string,label?,x?,y?,w?,h?}[] (boutons, liste remplacée)
---help: texte d'aide du script. events: activer clavier/texte/pointeur.
---image_size: taille des aperçus Generated en pixels (32..1024, défaut 200).
---@param scene table
---@return boolean ok
---@return string? err
function Mimir.Viz.configure(scene) end

---Retirer les événements de la file (256 entrées max, anciennes supprimées).
---Types: configured, click (id,x,y), close, resize (width,height).
---Avec events=true: key (code,control,shift,alt), text (unicode),
---pointer_move/down/up (x,y,button pour down/up). Coordonnées fenêtre en pixels.
---Les callbacks Lua sont à exécuter par le script, jamais sur le thread Viz.
---@param timeout_ms? integer @0..1000, défaut 0
---@return table[] events
function Mimir.Viz.poll_events(timeout_ms) end


---Vérifier que le visualiseur est créé et que sa fenêtre est ouverte.
---@return boolean ok
---@return string? err
function Mimir.Viz.initialize() end

---Vérifier si la fenêtre est ouverte.
---@return boolean open
function Mimir.Viz.is_open() end

---Compatibilité: les événements sont traités automatiquement par AsyncMonitor; no-op.
---@return boolean ok
function Mimir.Viz.process_events() end

---Compatibilité: le rendu est mis à jour automatiquement par AsyncMonitor; no-op.
---@return boolean ok
function Mimir.Viz.update() end

---Ajouter/afficher une image dans le visualiseur.
---Les pixels sont fournis sous forme de tableau plat (row-major).
---@param pixels number[] @Tableau de valeurs pixel [0-255], RGBA ou RGB
---@overload fun(pixels: number[], prompt?: string, width?: integer, height?: integer, channels?: integer): boolean, string?
---@param width integer @Largeur de l'image
---@param height integer @Hauteur de l'image
---@param channels? integer @Nombre de canaux
---@param prompt? string
---@return boolean ok
---@return string? err
function Mimir.Viz.add_image(pixels, width, height, channels, prompt) end

---@class VizDetection
---@field x1 number @Bord gauche normalisé [0,1]
---@field y1 number @Bord haut normalisé [0,1]
---@field x2 number @Bord droit normalisé [0,1]
---@field y2 number @Bord bas normalisé [0,1]
---@field score? number @Confiance, généralement [0,1]
---@field class_id? integer @Identifiant de classe utilisé pour la couleur
---@field label? string @Nom de classe affiché

---Publier les détections à superposer sur l'image dataset, la sortie, ou les deux.
---Un tableau vide efface les détections précédentes. Les coordonnées sont normalisées.
---@param detections VizDetection[]
---@param target? 'dataset'|'output'|'both' @Défaut: dataset
---@return boolean ok
---@return string? err
function Mimir.Viz.set_detections(detections, target) end

---Afficher ou masquer l'overlay de détection sans effacer ses données.
---@param enabled boolean
---@return boolean ok
---@return string? err
function Mimir.Viz.show_detections(enabled) end

---Définir le seuil de confiance des détections affichées dans VIZ.
---Une valeur non finie ou hors de [0,1] déclenche une erreur Lua.
---@param threshold number @Nombre fini entre 0 et 1
---@return boolean ok @false si le visualiseur n’est pas créé
---@return string? err
function Mimir.Viz.set_detection_threshold(threshold) end


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
---@field enabled? boolean @Modifie l'état de validation live avant de publier les métriques

---Mettre à jour l'état/progression de validation affichée (sans écraser les métriques train).
---@param state VizValidationState|table
---@return boolean ok
---@return string? err
function Mimir.Viz.set_validation(state) end

---@return boolean enabled
function Mimir.Viz.validation_enabled() end

---Compatibilité: l'historique est alimenté par update_metrics; no-op retournant true.
---@param loss number @Valeur de loss
---@return boolean ok
function Mimir.Viz.add_loss_point(loss) end

---Clear viz.
---@return boolean ok
---@return string? err
function Mimir.Viz.clear() end

---Arrêter le monitor si `enabled=false`.
---Avec `enabled=true`, la fonction confirme seulement l'état; utilisez `create()` pour démarrer.
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
-- Module: Mimir.IO
--=============================================================================

---@class MimirIOAPI
Mimir.IO = {}

---Charger et redimensionner une image en RGB u8.
---Supporte les formats `png/jpg/jpeg/bmp/tiff/webp` via stb_image.
---
---@param path string
---@param target_w? integer @Défaut: 256
---@param target_h? integer @Défaut: 256
---@return {image: integer[], width: integer, height: integer, channels: 3}|nil image
---@return string? err
function Mimir.IO.read_image_rgb_u8(path, target_w, target_h) end

---Alias camelCase de `read_image_rgb_u8`.
---@param path string
---@param target_w? integer
---@param target_h? integer
---@return {image: integer[], width: integer, height: integer, channels: 3}|nil image
---@return string? err
function Mimir.IO.readImageRGBU8(path, target_w, target_h) end

---Lire ou définir la suppression des logs framework vers stdout/stderr pendant l'exécution Lua.
---Getter: `Mimir.IO.suppress_stdout_logs()` -> boolean
---Setter: `Mimir.IO.suppress_stdout_logs(true)` -> (previous, current)
---@overload fun(): boolean
---@overload fun(enabled: boolean): (boolean, boolean)
function Mimir.IO.suppress_stdout_logs(enabled) end

---Alias camelCase de `suppress_stdout_logs`.
---@overload fun(): boolean
---@overload fun(enabled: boolean): (boolean, boolean)
function Mimir.IO.suppressStdoutLogs(enabled) end

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
---| "safetensors"|"st" # Format production (défaut)
---| "raw_folder"|"raw"|"folder" # Dossier brut avec checksums
---| "debug_json"|"debug"|"json" # JSON d'inspection

---@alias DetectedSerializationFormat "SAFETENSORS"|"RAWFOLDER"|"DEBUGJSON"

---@class SaveOptions
---@field save_tokenizer? boolean @Sauvegarder le tokenizer (défaut: true)
---@field save_encoder? boolean @Sauvegarder l'encoder (défaut: true)
---@field save_optimizer? boolean @Sauvegarder l'état optimizer (défaut: false)
---@field debug_max_values? integer @Nombre max de valeurs debug (legacy, défaut: 100)
---@field include_git_info? boolean @Inclure info git (défaut: true)
---@field include_gradients? boolean @[DebugJson v1.3] Inclure gradients (défaut: false)
---@field training_state? table @État de la boucle (curseur, RNG, feedback), restauré dans Model.get_config().training_state.
---@field include_optimizer_state? boolean @[DebugJson v1.3] Inclure optimizer state (défaut: false)
---@field max_values_per_tensor? integer @[DebugJson v1.3] Nb valeurs par tensor (défaut: 20)
---@field include_activations? boolean @[DebugJson v1.3] Inclure activations (défaut: false)
---@field include_checksums? boolean @[DebugJson v1.3] Inclure checksums (défaut: false)
---@field include_weight_deltas? boolean @[DebugJson v1.3] Inclure weight deltas (défaut: false)

---@class LoadOptions
---@field load_tokenizer? boolean @Charger le tokenizer (défaut: true)
---@field load_encoder? boolean @Charger l'encoder (défaut: true)
---@field load_optimizer? boolean @Charger l'état optimizer (défaut: false)
---@field strict_mode? boolean @Refuser les tensors inconnus ou incompatibles
---@field validate_checksums? boolean @Vérifier les checksums disponibles
---@field mapping_json? string @Chemin du mapping de noms de tensors
---@field tensor_mapping_json? string @Alias de mapping_json

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
--- -- DebugJson Enhanced v1.3 (diagnostic entraînement + snapshot framework_state)
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
---     validate_checksums = true
--- })
---```
function Mimir.Serialization.load(path, format, options) end

---Détecter automatiquement le format d'un checkpoint.
---@param path string @Chemin du fichier/dossier
---@return DetectedSerializationFormat? format @Format détecté ou nil si inconnu
---@return string? err @Message d'erreur si échec
function Mimir.Serialization.detect_format(path) end

---Sauvegarder un checkpoint avec Enhanced Debug JSON v1.3.0.
---Fonction spécialisée pour le diagnostic d'entraînement avec snapshot `framework_state`.
---@param path string @Chemin du fichier JSON
---@param options? SaveOptions @Options Enhanced Debug JSON v1.3
---@return boolean ok @true si succès
---@return string? err @Message d'erreur si échec
function Mimir.Serialization.save_enhanced_debug(path, options) end

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
-- Modules Lua (`dofile(ROOTWORK.."/scripts/modules/...")`)
--=============================================================================
-- Ces types décrivent des helpers écrits en Lua. Ils ne sont pas membres de
-- `Mimir` et leur chargement reste explicite.

---@class MimirScriptArgs
---@field override? string[]
---@field [string] string|boolean|string[]

---@class MimirArgsModule
---@field parse fun(argv?:string[]):MimirScriptArgs, string[]
---@field apply_overrides fun(cfg?:table, opts?:MimirScriptArgs):table
---@field apply_validation_config fun(cfg?:table, opts?:MimirScriptArgs, defaults?:table):table
---@field has fun(opts:MimirScriptArgs, key:string):boolean
---@field get_str fun(opts:MimirScriptArgs, key:string, default?:string):string?
---@field get_num fun(opts:MimirScriptArgs, key:string, default?:number):number?
---@field get_int fun(opts:MimirScriptArgs, key:string, default?:integer):integer?
---@field get_bool fun(opts:MimirScriptArgs, key:string, default?:boolean):boolean?
---@field opt_str fun(key:string, default?:string):string?
---@field opt_num fun(key:string, default?:number):number?
---@field opt_int fun(key:string, default?:integer):integer?
---@field opt_bool fun(key:string, default?:boolean):boolean?

---@class MimirFSModule
---@field is_windows fun():boolean
---@field sep fun():string
---@field quote fun(path:string):string
---@field normalize fun(path:string):string
---@field join fun(...:string):string
---@field dirname fun(path:string):string?
---@field file_exists fun(path:string):boolean
---@field is_dir fun(path:string):boolean
---@field exists fun(path:string):boolean
---@field mkdir_p fun(path:string):boolean
---@field list_dir fun(path:string):string[]

---@class MimirCheckpointResumeModule
---@field file_exists fun(path:string):boolean
---@field find_latest_epoch_dir fun(base:string):string?
---@field resolve_dir fun(base:string):string?

---@class MimirBaseTokenizerOptions
---@field path? string
---@field require? boolean
---@field max_vocab? integer

---@class MimirBaseTokenizerModule
---@field default_path fun():string
---@field load_base fun(opts?:MimirBaseTokenizerOptions):boolean, string?
---@field vocab_size fun():integer
---@field save_current_as_base fun(path?:string):boolean, string?

---@class MimirCausalTokenizerOptions
---@field corpus string
---@field vocab_size integer
---@field padding_idx? integer
---@field seq_len? integer
---@field path? string
---@field bpe_merges? integer

---@class MimirCausalTokenizerInfo
---@field generated boolean
---@field replaced_loaded_tokenizer? boolean
---@field reason? string
---@field path? string
---@field vocab_size integer
---@field max_vocab integer

---@class MimirCausalTokenizerModule
---@field ensure fun(options:MimirCausalTokenizerOptions):TokenIds, MimirCausalTokenizerInfo

---@class MimirHelpModule
---@field should_show_help fun(argv?:string[]):boolean
---@field find_script_from_stack fun(start_level?:integer, max_level?:integer):string?
---@field print_help fun(params?:table)
---@field auto_exit_help fun(params?:table)

---@class MimirMPKModule
---@field build fun(spec:table):table
---@field compile fun(source_path:string, output_path:string):boolean?, string?
---@field read fun(path:string):table?, string?
---@field write fun(path:string, pkg:table, opts?:table):boolean?, string?
---@field verify_checksum fun(pkg:table):boolean, string?
---@field decode_payload fun(pkg:table):table?, string?
---@field to_registry_full_config fun(pkg:table):table?, string?
---@field model_structure_template fun(kind?:string):table
---@field encode_pseudocode fun(value:table):string
---@field decode_pseudocode fun(source:string):table

---@class MimirMPKLayersModule
---@field available_layer_types fun():string[]
---@field infer_layer_type fun(layer_name:string):string
---@field canonical_layer_type fun(raw_type:string, params?:table):string?, string?
---@field predict_params_count fun(layer_type:string, params?:table, context?:table):integer, string
---@field normalize_graph_in_place fun(model_structure:table, opts?:table):boolean, string?

--=============================================================================
-- Pipeline API (optionnel)
--=============================================================================
-- Si vous chargez `scripts/modules/pipeline_api.lua` (ou `pipeline.lua`),
-- l'IDE bénéficiera de ces signatures pour l'autocomplétion.

---@class PipelineOptions
---@field name? string
---@field fallback_config? table
---@field allowed_keys? string[]
---@field legacy_mapper? fun(cfg:table, user:table)
---@field create_tokenizer? boolean

---@class PipelineAPI
---@field name string
---@field config table
---@field base_config table
---@field arch? string
---@field model? boolean
---@field tokenizer? boolean
---@field trained boolean
---@field steps table
local Pipeline = {}

---@param arch string
---@param patch? table
---@return boolean ok
---@return table|string cfg_or_err
function Pipeline:loadDefaultConfig(arch, patch) end

---@param patch table
---@return boolean ok
---@return table|string cfg_or_err
function Pipeline:patchConfig(patch) end

---@param cfg table
---@return boolean ok
---@return table|string cfg_or_err
function Pipeline:setConfig(cfg) end

---@return table
function Pipeline:getConfig() end

---@return table
function Pipeline:getBaseConfig() end

---@return boolean ok
---@return integer|string? params_or_err
function Pipeline:build() end

---@param dataset_path? string
---@param epochs? integer
---@param lr? number
---@return boolean ok
---@return string? err
function Pipeline:train(dataset_path, epochs, lr) end

---@param input any
---@return any
function Pipeline:infer(input) end

---@param path string
---@return boolean ok
---@return string? err
function Pipeline:save(path) end

---@class PipelineManagerAPI
---@field pipelines table<string, PipelineAPI>
local PipelineManager = {}

---@return PipelineManagerAPI
function PipelineManager:new() end

---@param name string
---@param pipeline PipelineAPI
function PipelineManager:add(name, pipeline) end

---@param name string
---@return PipelineAPI|nil
function PipelineManager:get(name) end

function PipelineManager:list() end

---@param base_path string
function PipelineManager:save_all(base_path) end

---@class PipelineModule
---@field Pipeline PipelineAPI
---@field PipelineManager PipelineManagerAPI
---@field FromRegistry fun(model_type:string, config?:table, options?:PipelineOptions):PipelineAPI|nil, string?
---@field Transformer fun(config?:table):PipelineAPI
---@field UNet fun(config?:table):PipelineAPI
---@field VAE fun(config?:table):PipelineAPI
---@field ViT fun(config?:table):PipelineAPI
---@field GAN fun(config?:table):PipelineAPI
---@field Diffusion fun(config?:table):PipelineAPI
---@field ResNet fun(config?:table):PipelineAPI
---@field MobileNet fun(config?:table):PipelineAPI

---@type PipelineModule
PipelineModule = {
	Pipeline = Pipeline,
	PipelineManager = PipelineManager,
	FromRegistry = function(model_type, config, options) end,
	Transformer = function(config) end,
	UNet = function(config) end,
	VAE = function(config) end,
	ViT = function(config) end,
	GAN = function(config) end,
	Diffusion = function(config) end,
	ResNet = function(config) end,
	MobileNet = function(config) end,
}

--=============================================================================
-- Exports globaux (pour l'IDE)
--=============================================================================
---@type MimirModelAPI
Mimir.Model = Mimir.Model
---@type MimirModelAPI
Mimir.model = Mimir.Model

---@type MimirModelAPI
model = Mimir.Model
---@type MimirArchitecturesAPI
architectures = Mimir.Architectures
---@type MimirTokenizerAPI
tokenizer = Mimir.Tokenizer
---@type MimirDatasetAPI
dataset = Mimir.Dataset
---@type MimirMemoryAPI
Memory = Mimir.Memory
---@type MimirMemoryGuardAPI
MemoryGuard = Mimir.MemoryGuard
---@type MimirAllocatorAPI
Allocator = Mimir.Allocator
---@type MimirHtopAPI
htop = Mimir.Htop
---@type MimirVizAPI
viz = Mimir.Viz
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
---@type MimirDatabaseAPI
Mimir.Database = Mimir.Database
---@type MimirIOAPI
Mimir.IO = Mimir.IO
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

---@type MimirSerializationAPI
Mimir.Serialization = Mimir.Serialization
