#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local BaseTok = dofile(ROOTWORK.."/scripts/modules/base_tokenizer.lua")
local Checkpoint = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
local opts = Args.parse(arg) or {}

local function logf(format, ...)
  log(string.format(format, ...))
end

local function die(message)
  error(tostring(message), 0)
end

local function check(ok, err, operation)
  if not ok then die(operation .. ": " .. tostring(err or "échec")) end
end

local function is_finite(value)
  return type(value) == "number" and value == value and
    value > -math.huge and value < math.huge
end

local config_path = Args.get_str(opts, "config", "configs/lumen_diffusion.json")
local document, config_err = read_json(config_path)
if type(document) ~= "table" then die("configuration: " .. tostring(config_err)) end

local cfg = document.lumen_diffusion or {}
local training = document.training or {}
local tokenizer_cfg = document.tokenizer or {}
local viz_enabled = Args.get_bool(opts, "viz", false)

if viz_enabled then
  local ok_viz, viz_err = Mimir.Viz.create(document)
  check(ok_viz, viz_err, "Viz.create")
  check(Mimir.Viz.set_validation({ enabled = true }), nil,
    "Viz.set_validation(enabled)")
  logf("interface VIZ activée")
end

local dataset_root = Args.get_str(opts, "dataset", "")
if dataset_root == "" then
  die("--dataset est requis (images avec annotations .json ou captions .txt de même nom)")
end

local tokenizer_path = Args.get_str(opts, "tokenizer", tokenizer_cfg.path or
  "checkpoint/base_tokenizer/tokenizer.json")
local require_tokenizer = Args.get_bool(opts, "require-tokenizer", false)
-- Capacity is derived from additions during hydration, never from a base/default cap.
local tokenizer_bpe_merges = math.max(0, Args.get_int(opts,"tokenizer-bpe-merges",1000))
if opts["max-vocab"] then logf("--max-vocab=%s ignoré : capacité issue de l'hydratation ou du checkpoint",tostring(opts["max-vocab"])) end
local output_dir = Args.get_str(opts, "out", "checkpoints/lumen_diffusion")
local resume_path = Args.get_str(opts, "resume", "")
local prepare_tokenizer_only = Args.get_bool(opts, "prepare-tokenizer-only", false)
if resume_path == "0" or resume_path == "false" or resume_path == "off" or
    resume_path == "none" then
  resume_path = ""
end
local checkpoint_vocab_size = nil
if resume_path ~= "" then
  resume_path = Checkpoint.resolve_dir(resume_path)
  if not resume_path then die("--resume ne contient aucun checkpoint Lumen exploitable") end
  local architecture, architecture_err=read_json(FS.join(resume_path,"model","architecture.json"))
  local saved=type(architecture)=="table" and architecture.model_config
  if type(saved)~="table" or saved.type~="lumen_diffusion" then
    die("Checkpoint Lumen invalide : "..tostring(architecture_err or "model_config absent ou incompatible"))
  end
  checkpoint_vocab_size=tonumber(saved.vocab_size)
  if not checkpoint_vocab_size or checkpoint_vocab_size<1 then die("vocab_size absent du checkpoint Lumen") end
end
local epochs = math.max(1, Args.get_int(opts, "epochs", training.epochs or 10))
local learning_rate = Args.get_num(opts, "lr", training.learning_rate or 1e-4)
local optimizer = Args.get_str(opts, "optimizer", training.optimizer or "adamw")
local weight_decay = Args.get_num(opts, "weight-decay", training.weight_decay or 0.01)
local cfg_dropout = Args.get_num(opts, "cfg-dropout", training.cfg_dropout or 0.1)
local warmup_steps = math.max(0, Args.get_int(opts, "warmup-steps",
  Args.get_int(opts, "lr-warmup_steps", training.warmup_steps or 100)))
local requested_kl_beta = math.max(0.0, Args.get_num(opts, "kl-beta",
  training.kl_beta or cfg.kl_beta or 0.0))
local requested_kl_warmup_steps = math.max(0, Args.get_int(opts, "kl-warmup-steps",
  Args.get_int(opts, "kl-warmup_steps",
    training.kl_warmup_steps or cfg.kl_warmup_steps or 0)))
local kl_beta = 0.0
local kl_warmup_steps = 0
if requested_kl_beta > 0.0 then
  logf("--kl-beta=%.8g ignoré: Lumen entraîne epsilon avec MSE pure",
    requested_kl_beta)
end
local log_every = math.max(1, Args.get_int(opts, "logf-every", training.log_every or 10))
local save_every = math.max(0, Args.get_int(opts, "save-every", training.save_every or 500))
local max_items = math.max(0, Args.get_int(opts, "max-items", training.max_items or 0))
local vae_calibration_items = math.max(1, Args.get_int(opts,
  "vae-calibration-items", training.vae_calibration_items or 100))
local seed = Args.get_int(opts, "seed", cfg.seed or 1337)
local validation_every_steps = math.max(0, Args.get_int(opts, "validation-every-steps",
  training.validation_every_steps or 100))
local validation_every_epochs = math.max(0, Args.get_int(opts, "validation-every-epochs",
  training.validation_every_epochs or 1))
local validation_items = math.max(0, Args.get_int(opts, "validation-items",
  training.validation_items or 4))
local validation_holdout = Args.get_bool(opts, "validation-holdout",
  training.validation_holdout ~= false)
local validation_feedback_enabled = Args.get_bool(opts, "validation-feedback",
  training.validation_feedback_enabled ~= false)
local validation_reward_factor = Args.get_num(opts, "validation-reward-factor",
  training.validation_reward_factor or 1.05)
local validation_penalty_factor = Args.get_num(opts, "validation-penalty-factor",
  training.validation_penalty_factor or 0.70)
local validation_lr_scale_min = Args.get_num(opts, "validation-lr-scale-min",
  training.validation_lr_scale_min or 0.10)
local validation_lr_scale_max = Args.get_num(opts, "validation-lr-scale-max",
  training.validation_lr_scale_max or 1.50)
local validation_improve_threshold = Args.get_num(opts, "validation-improve-threshold",
  training.validation_improve_threshold or 0.001)
local validation_feedback_min_steps = math.max(0, Args.get_int(opts,
  "validation-feedback-min-steps", training.validation_feedback_min_steps or 0))
local validation_epsilon_weight = math.max(0.0, Args.get_num(opts,
  "validation-epsilon-weight", training.validation_epsilon_weight or 0.5))
local validation_reconstruction_weight = math.max(0.0, Args.get_num(opts,
  "validation-reconstruction-weight", training.validation_reconstruction_weight or 0.5))
local validation_ema_alpha = Args.get_num(opts, "validation-ema-alpha",
  training.validation_ema_alpha or 0.25)
local validation_penalty_patience = math.max(1, Args.get_int(opts,
  "validation-penalty-patience", training.validation_penalty_patience or 2))
local validation_max_penalties = math.max(0, Args.get_int(opts,
  "validation-max-penalties", training.validation_max_penalties or 0))
local validation_save_best = Args.get_bool(opts, "validation-save-best",
  training.validation_save_best ~= false)

if type(Mimir.Viz.set_validation) == "function" then
  Mimir.Viz.set_validation({
    enabled = validation_items > 0 and
      (validation_every_steps > 0 or validation_every_epochs > 0)
  })
end

if learning_rate <= 0 then die("--lr doit être strictement positif") end
if cfg_dropout < 0 or cfg_dropout > 1 then die("--cfg-dropout doit être dans [0,1]") end
if validation_reward_factor < 1 then die("--validation-reward-factor doit être >= 1") end
if validation_penalty_factor <= 0 or validation_penalty_factor > 1 then
  die("--validation-penalty-factor doit être dans ]0,1]")
end
if validation_lr_scale_min <= 0 or validation_lr_scale_max < validation_lr_scale_min then
  die("bornes validation-lr-scale invalides")
end
if validation_improve_threshold < 0 then
  die("--validation-improve-threshold doit être positif")
end
if validation_epsilon_weight + validation_reconstruction_weight <= 0 then
  die("au moins un poids de validation doit être strictement positif")
end
if validation_ema_alpha <= 0 or validation_ema_alpha > 1 then
  die("--validation-ema-alpha doit être dans ]0,1]")
end

cfg.image_w = Args.get_int(opts, "image-w", cfg.image_w or 512)
cfg.image_h = Args.get_int(opts, "image-h", cfg.image_h or 512)
cfg.image_c = Args.get_int(opts, "image-c", cfg.image_c or 3)
cfg.latent_w = Args.get_int(opts, "latent-w", cfg.latent_w or 64)
cfg.latent_h = Args.get_int(opts, "latent-h", cfg.latent_h or 64)
cfg.latent_c = Args.get_int(opts, "latent-c", cfg.latent_c or 4)
cfg.vae_checkpoint = Args.get_str(opts, "vae-checkpoint", cfg.vae_checkpoint or "")
cfg.vae_base_channels = Args.get_int(opts, "vae-base-channels", cfg.vae_base_channels or 16)
cfg.vae_stochastic_latent = Args.get_bool(opts, "vae-stochastic-latent",
  cfg.vae_stochastic_latent ~= false)
cfg.vae_resnet = Args.get_bool(opts, "vae-resnet", cfg.vae_resnet ~= false)
cfg.vae_attention = Args.get_bool(opts, "vae-attn", cfg.vae_attention ~= false)
cfg.vae_use_skip_connections = Args.get_bool(opts, "vae-use-skip-connections",
  cfg.vae_use_skip_connections ~= false)
cfg.vae_use_encoder_prior = Args.get_bool(opts, "vae-encoder-prior",
  cfg.vae_use_encoder_prior ~= false)
cfg.vae_enc_norm = Args.get_str(opts, "vae-enc-norm", cfg.vae_enc_norm or "none")
cfg.vae_dec_norm = Args.get_str(opts, "vae-dec-norm", cfg.vae_dec_norm or "groupnorm")
cfg.vae_decoder_upsample = Args.get_str(opts, "vae-decoder-upsample",
  cfg.vae_decoder_upsample or "nearest_conv")
cfg.vae_enc_gn_groups = Args.get_int(opts, "vae-enc-gn-groups", cfg.vae_enc_gn_groups or 16)
cfg.vae_dec_gn_groups = Args.get_int(opts, "vae-dec-gn-groups", cfg.vae_dec_gn_groups or 32)
cfg.vae_attn_heads = Args.get_int(opts, "vae-attn-heads", cfg.vae_attn_heads or 4)
cfg.vae_attn_max_tokens = Args.get_int(opts, "vae-attn-max-tokens",
  cfg.vae_attn_max_tokens or 4096)
cfg.vae_resnet_max_tokens = Args.get_int(opts, "vae-resnet-max-tokens",
  cfg.vae_resnet_max_tokens or 4096)
cfg.vae_scale = Args.get_num(opts, "vae-scale", cfg.vae_scale or 0.0)
cfg.vae_shift = Args.get_num(opts, "vae-shift", cfg.vae_shift or 0.0)
cfg.preview_timestep = Args.get_int(opts, "preview-timestep", cfg.preview_timestep or 50)
cfg.patch_size = Args.get_int(opts, "patch-size", cfg.patch_size or 4)
cfg.hidden_size = Args.get_int(opts, "hidden-size", cfg.hidden_size or 384)
cfg.depth = Args.get_int(opts, "depth", cfg.depth or 8)
cfg.mlp_ratio = Args.get_num(opts, "mlp-ratio", cfg.mlp_ratio or 4.0)
-- This is the reference positional table size; active length follows each prompt.
local text_length = opts["text-seq-len"] or opts["txt-seq-len"] or "auto"
if tostring(text_length):lower()=="auto" or tonumber(text_length)==0 then
  cfg.text_seq_len = math.max(1,tonumber(cfg.text_seq_len) or 77)
else
  cfg.text_seq_len = tonumber(text_length)
  if not cfg.text_seq_len or cfg.text_seq_len%1~=0 then die("--text-seq-len doit être auto ou un entier positif") end
end
cfg.dynamic_text_seq_len = Args.get_bool(opts,"dynamic-text-seq-len",cfg.dynamic_text_seq_len ~= false)
cfg.text_layers = Args.get_int(opts, "text-layers", cfg.text_layers or 2)
cfg.num_heads = Args.get_int(opts, "num-heads", cfg.num_heads or 8)
cfg.diffusion_steps = Args.get_int(opts, "diffusion-steps", cfg.diffusion_steps or 1000)
cfg.beta_start = Args.get_num(opts, "beta-start", cfg.beta_start or 0.00085)
cfg.beta_end = Args.get_num(opts, "beta-end", cfg.beta_end or 0.012)
if cfg.vae_checkpoint == "" and not prepare_tokenizer_only then
  die("--vae-checkpoint est requis (dossier epoch_* du VAEConv entraîné)")
end
if not prepare_tokenizer_only then
  local resolved_vae_checkpoint = Checkpoint.resolve_dir(cfg.vae_checkpoint)
  if not resolved_vae_checkpoint then
    die("--vae-checkpoint ne contient aucun checkpoint VAEConv exploitable: " ..
      cfg.vae_checkpoint)
  end
  if resolved_vae_checkpoint ~= cfg.vae_checkpoint then
    logf("checkpoint VAE résolu: " .. resolved_vae_checkpoint)
    cfg.vae_checkpoint = resolved_vae_checkpoint
  end
end
if cfg.hidden_size < 1 or cfg.depth < 1 or cfg.mlp_ratio < 1 then
  die("--hidden-size, --depth et --mlp-ratio doivent être positifs")
end
if cfg.text_seq_len < 1 or cfg.text_layers < 1 or cfg.num_heads < 1 then
  die("--text-seq-len, --text-layers et --num-heads doivent être positifs")
end
if cfg.hidden_size % cfg.num_heads ~= 0 then
  die("--num-heads doit diviser exactement --hidden-size")
end
if cfg.diffusion_steps < 2 then die("--diffusion-steps doit être >= 2") end
if cfg.beta_start <= 0 or cfg.beta_start >= 1 or
    cfg.beta_end < cfg.beta_start or cfg.beta_end >= 1 then
  die("--beta-start et --beta-end doivent vérifier 0 < start <= end < 1")
end
local max_patch_tokens = math.max(1, Args.get_int(opts, "max-patch-tokens", 1024))

cfg.weight_decay = weight_decay
cfg.beta1 = training.beta1 or 0.9
cfg.beta2 = training.beta2 or 0.999
cfg.epsilon = training.epsilon or 1e-8
cfg.kl_beta = kl_beta
cfg.kl_warmup_steps = kl_warmup_steps
cfg.seed = seed
-- cfg.vocab_size is assigned after tokenizer hydration/loading.
local bbox_loss_weight = Args.get_num(opts, "bbox-loss-weight", training.bbox_loss_weight or cfg.bbox_loss_weight or 1.0)
if not is_finite(bbox_loss_weight) or bbox_loss_weight < 0 then die("--bbox-loss-weight doit être fini et >= 0") end
cfg.bbox_loss_weight = bbox_loss_weight
for role, default in pairs({primary=3.0, secondary=2.0, other=1.0}) do
  local key = "relation_" .. role .. "_weight"
  cfg[key] = Args.get_num(opts, "relation-" .. role .. "-weight", training[key] or cfg[key] or default)
  if not is_finite(cfg[key]) or cfg[key] < 0 then die("--relation-" .. role .. "-weight doit être fini et >= 0") end
end

local function convert_image_channels(rgb)
  if cfg.image_c == 3 then return rgb end
  local converted = {}
  for index = 1, #rgb, 3 do
    local red = tonumber(rgb[index]) or 0
    local green = tonumber(rgb[index + 1]) or 0
    local blue = tonumber(rgb[index + 2]) or 0
    if cfg.image_c == 1 then
      converted[#converted + 1] = math.floor(
        0.299 * red + 0.587 * green + 0.114 * blue + 0.5)
    else
      converted[#converted + 1] = red
      converted[#converted + 1] = green
      converted[#converted + 1] = blue
      converted[#converted + 1] = 255
    end
  end
  return converted
end

local skipped_images = 0

local function release_dataset_item(index)
  local ok, err = Mimir.Dataset.release(index)
  if not ok then
    die("Dataset.release(" .. index .. "): " .. tostring(err or "échec"))
  end
end

local function get_image_item(index, phase)
  local item, item_err = Mimir.Dataset.get(index)
  if type(item) == "table" and type(item.image) == "table" then
    return item
  end
  skipped_images = skipped_images + 1
  local path = type(item) == "table" and item.image_file or nil
  release_dataset_item(index)
  logf(
    "image ignorée (%s): index=%d path=%s erreur=%s",
    phase, index, tostring(path or "?"),
    tostring(item_err or "décodage ou allocation impossible"))
  return nil
end

if resume_path ~= "" then
  local resume_tokenizer_path=FS.join(resume_path,"tokenizer","tokenizer.json")
  if not FS.file_exists(resume_tokenizer_path) then die("Tokenizer du checkpoint Lumen introuvable : "..resume_tokenizer_path) end
  local ok_tokenizer,tokenizer_err=Mimir.Tokenizer.load(resume_tokenizer_path)
  check(ok_tokenizer,tokenizer_err,"Tokenizer.load(checkpoint Lumen)")
  logf("tokenizer de reprise conservé : %s vocab=%d capacité=%d",resume_tokenizer_path,Mimir.Tokenizer.vocab_size(),Mimir.Tokenizer.get_max_vocab())
else
  local ok_tokenizer,tokenizer_err=BaseTok.load_base({path=tokenizer_path,max_vocab=7,require=require_tokenizer})
  check(ok_tokenizer,tokenizer_err,"Tokenizer.load_or_create")
end

local cache_path = Args.get_str(opts, "cache", dataset_root .. "/dataset_cache.json")
local ok_dataset, dataset_count_or_error = Mimir.Dataset.load(
  dataset_root, cfg.image_w, cfg.image_h, 2, true, cache_path, 10240, true)
check(ok_dataset, dataset_count_or_error, "Dataset.load")
local dataset_total = tonumber(dataset_count_or_error) or 0
local dataset_count = dataset_total
if max_items > 0 then dataset_count = math.min(dataset_count, max_items) end
if dataset_count < 1 then
  die("dataset vide; format attendu: <nom>.png|jpg accompagné de <nom>.json (annotations) ou <nom>.txt")
end
local validation_capacity = validation_holdout and
  math.max(0, dataset_count - 1) or dataset_count
local validation_count = math.min(validation_items, validation_capacity)
local training_count = validation_holdout and
  (dataset_count - validation_count) or dataset_count
if validation_items > 0 and validation_count < validation_items then
  logf(
    "validation réduite à %d item(s) pour conserver un item d'entraînement",
    validation_count)
end
logf(
  "partition: train=%d validation=%d holdout=%s",
  training_count, validation_count, tostring(validation_holdout))

local function hydrate_tokenizer(reason)
  local ensure_vocab = Mimir.Tokenizer.ensure_vocab_from_text
  local tokenize_ensure = Mimir.Tokenizer.learn_bpe
  if type(ensure_vocab) ~= "function" or type(tokenize_ensure) ~= "function" then
    die("tokenizer: ensure_vocab_from_text et learn_bpe sont requis")
  end
  local vocab_before = math.floor(tonumber(Mimir.Tokenizer.vocab_size()) or 0)
  local captions = 0
  local corpus = {}
  local errors = 0
  for index = 1, dataset_total do
    local item = Mimir.Dataset.get(index, false)
    local text = type(item) == "table" and tostring(item.text or "") or ""
    -- Use the exact native conditioning shared with training and validation.
    if type(item) == "table" and type(item.annotation) == "table" then
      local context, context_err = Mimir.Model.conditioning_prompt(text, item.annotation)
      if not context then die("Contexte d’annotation invalide : " .. tostring(context_err)) end
      text = context
    end
    if text ~= "" then
      captions = captions + 1
      -- At most one new token per byte: enough room for this caption's additions,
      -- independently of the capacity serialized in the base tokenizer.
      check(Mimir.Tokenizer.set_max_vocab(Mimir.Tokenizer.vocab_size()+#text),nil,"Tokenizer.grow_for_caption")
      local ok_call, result = pcall(ensure_vocab, text)
      if not ok_call or result == false then
        die("Tokenizer.ensure_vocab_from_text a échoué : " .. tostring(result))
      else
        corpus[#corpus + 1] = text
      end
    end
    release_dataset_item(index)
  end
  -- Each BPE merge adds at most one token. Reserve only the learning budget.
  check(Mimir.Tokenizer.set_max_vocab(Mimir.Tokenizer.vocab_size()+tokenizer_bpe_merges),nil,"Tokenizer.grow_for_bpe")
  local ok_bpe, bpe_err = tokenize_ensure(corpus,tokenizer_bpe_merges)
  check(ok_bpe, bpe_err, "Tokenizer.learn_bpe")
  local longest_context = 0
  for _, context in ipairs(corpus) do
    local tokens, token_err = Mimir.Tokenizer.tokenize(context)
    if not tokens then die("Tokenisation du contexte : " .. tostring(token_err)) end
    longest_context = math.max(longest_context, #tokens)
  end
  if not prepare_tokenizer_only and not cfg.dynamic_text_seq_len and longest_context > cfg.text_seq_len then
    die(string.format("Contexte dataset trop long : %d tokens requis ; utilisez --text-seq-len %d (actuel : %d)", longest_context, longest_context, cfg.text_seq_len))
  end
  logf("contexte texte: mode=%s maximum_dataset=%d tokens reference_positions=%d",cfg.dynamic_text_seq_len and "dynamique par entrée" or "fixe",longest_context,cfg.text_seq_len)
  local vocab_after = math.floor(tonumber(Mimir.Tokenizer.vocab_size()) or 0)
  logf(
    "tokenizer hydraté (%s): captions=%d/%d vocab=%d->%d max=%d erreurs=%d",
    reason, captions, dataset_total, vocab_before, vocab_after,
    Mimir.Tokenizer.get_max_vocab(), errors)
end


if resume_path=="" then hydrate_tokenizer("dataset") end
local tokenizer_vocab_size = math.floor(tonumber(Mimir.Tokenizer.vocab_size()) or 0)
if tokenizer_vocab_size < 7 then
  die("tokenizer incomplet après hydratation: vocab=" .. tokenizer_vocab_size)
end
if resume_path=="" then
  check(Mimir.Tokenizer.set_max_vocab(tokenizer_vocab_size),nil,"Tokenizer.freeze_vocab")
  cfg.vocab_size=tokenizer_vocab_size
else
  if tokenizer_vocab_size>checkpoint_vocab_size then die("Tokenizer incompatible avec les embeddings du checkpoint Lumen") end
  cfg.vocab_size=checkpoint_vocab_size
end

if not cfg.dynamic_text_seq_len and type(Mimir.Dataset.prepare_sequences) == "function" then
  local ok_sequences, sequences_err =
    Mimir.Dataset.prepare_sequences(cfg.text_seq_len or 77)
  check(ok_sequences, sequences_err, "Dataset.prepare_sequences")
end

FS.mkdir_p(output_dir)
if resume_path=="" then
  local ok_source_save, source_save_err = Mimir.Tokenizer.save(tokenizer_path)
  check(ok_source_save, source_save_err, "Tokenizer.save(source)")
end
local output_tokenizer_path = FS.join(output_dir, "tokenizer.json")
if FS.normalize(output_tokenizer_path) ~= FS.normalize(tokenizer_path) then
  local ok_output_save, output_save_err = Mimir.Tokenizer.save(output_tokenizer_path)
  check(ok_output_save, output_save_err, "Tokenizer.save(output)")
end
logf(
  "tokenizer construit et sérialisé avant modèle: vocab=%d path=%s",
  tokenizer_vocab_size, output_tokenizer_path)
if prepare_tokenizer_only then
  logf("préparation tokenizer terminée; modèle non créé")
  return
end

local indexed_image_w,indexed_image_h=cfg.image_w,cfg.image_h
if not prepare_tokenizer_only then
  local ok_create, create_err = Mimir.Model.create("lumen_diffusion", cfg)
  check(ok_create, create_err, "Model.create")
  local effective_cfg, effective_cfg_err = Mimir.Model.get_config()
  if type(effective_cfg) ~= "table" then
    die("Model.get_config: " .. tostring(effective_cfg_err or "échec"))
  end
  for _, key in ipairs({
    "image_w", "image_h", "image_c", "latent_w", "latent_h", "latent_c",
    "patch_size", "vae_base_channels", "vae_stochastic_latent",
    "vae_latent_spatial_context", "vae_latent_detail_path", "vae_latent_context_channels",
    "vae_resnet", "vae_attention", "vae_use_skip_connections",
    "vae_use_encoder_prior", "vae_enc_norm", "vae_dec_norm",
    "vae_decoder_upsample", "vae_enc_gn_groups", "vae_dec_gn_groups",
    "vae_attn_heads", "vae_attn_max_tokens", "vae_resnet_max_tokens",
  }) do
    if effective_cfg[key] ~= nil then cfg[key] = effective_cfg[key] end
  end
  logf(
    "Lumen adapté au VAE: image=%dx%dx%d latent=%dx%dx%d patch=%d",
    cfg.image_w, cfg.image_h, cfg.image_c,
    cfg.latent_w, cfg.latent_h, cfg.latent_c, cfg.patch_size)
end

if cfg.image_w <= 0 or cfg.image_h <= 0 then
  die("dimensions image effectives invalides")
end
if cfg.latent_w <= 0 or cfg.latent_h <= 0 or cfg.latent_c <= 0 then
  die("dimensions latentes effectives invalides")
end
if cfg.patch_size < 1 or cfg.latent_w % cfg.patch_size ~= 0 or
    cfg.latent_h % cfg.patch_size ~= 0 then
  die("le patch-size effectif doit diviser exactement le latent du VAE")
end
local patch_tokens = (cfg.latent_w / cfg.patch_size) * (cfg.latent_h / cfg.patch_size)
if patch_tokens > max_patch_tokens then
  die(string.format(
    "grille DiT latente effective trop grande: %d tokens (%dx%d, patch=%d); " ..
    "augmentez --patch-size ou explicitement --max-patch-tokens.",
    patch_tokens, cfg.latent_w, cfg.latent_h, cfg.patch_size))
end
if cfg.image_c ~= 1 and cfg.image_c ~= 3 and cfg.image_c ~= 4 then
  die("le nombre de canaux image effectif doit valoir 1, 3 ou 4")
end


-- The native VAE checkpoint can change effective image dimensions. Re-index
-- with those dimensions before decoding images; vocabulary discovery used text only.
if cfg.image_w~=indexed_image_w or cfg.image_h~=indexed_image_h then
  local ok_reload,reloaded=Mimir.Dataset.load(dataset_root,cfg.image_w,cfg.image_h,2,true,cache_path,10240,true)
  check(ok_reload,reloaded,"Dataset.load(dimensions VAE)")
  if tonumber(reloaded)~=dataset_total then die("Dataset modifié pendant la préparation du tokenizer") end
end

local ok_allocate, allocate_err = Mimir.Model.allocate_params()
check(ok_allocate, allocate_err, "Model.allocate_params")

local resume_training_state = nil
if resume_path ~= "" then
  local ok_resume, resume_err = Mimir.Serialization.load(resume_path, "raw_folder", {
    load_tokenizer = true,
    load_encoder = false,
    load_optimizer = true,
    strict_mode = false,
    validate_checksums = true,
  })
  check(ok_resume, resume_err, "Serialization.load")
  local resumed_config = Mimir.Model.get_config()
  resume_training_state = type(resumed_config) == "table" and resumed_config.training_state or nil
  logf("reprise: " .. resume_path)
else
  local ok_init, init_err =
    Mimir.Model.init_weights(Args.get_str(opts, "init", "xavier"), seed)
  check(ok_init, init_err, "Model.init_weights")
end

local ok_build, build_err = Mimir.Model.build()
check(ok_build, build_err, "Model.build(tokenizer hydraté)")
logf(
  "loss: epsilon_mse pure (KL/Wasserstein diagnostiques), " ..
  "kl_beta=%.8g kl_warmup_steps=%d",
  cfg.kl_beta, cfg.kl_warmup_steps)

local calibration_count = math.min(vae_calibration_items, training_count)
local ok_calibration, calibration_err = Mimir.Model.lumen_begin_vae_calibration()
check(ok_calibration, calibration_err, "Model.lumen_begin_vae_calibration")
local calibrated_items = 0
local calibration_index = 1
while calibrated_items < calibration_count and calibration_index <= training_count do
  local item = get_image_item(calibration_index, "calibration")
  if item then
    local stats, add_err = Mimir.Model.lumen_add_vae_calibration_image(
      convert_image_channels(item.image))
    item = nil
    release_dataset_item(calibration_index)
    if type(stats) ~= "table" then
      die(add_err or "ajout calibration VAE impossible")
    end
    calibrated_items = calibrated_items + 1
  end
  calibration_index = calibration_index + 1
end
if calibrated_items < 1 then
  die("calibration VAE impossible: aucune image décodable dans la partition d'entraînement")
end
local calibration, finish_err = Mimir.Model.lumen_finish_vae_calibration()
if type(calibration) ~= "table" then
  die(finish_err or "finalisation calibration VAE impossible")
end
logf(
  "VAE auto-calibré avant entraînement: items=%d/%d values=%d " ..
  "shift=%.9g scale=%.9g",
  calibration.items or 0, calibration_count, calibration.values or 0,
  calibration.shift or 0.0, calibration.scale or 0.0)

check(Mimir.Serialization.save(output_dir .. "/starttrain.json", "debug_json", {
  save_tokenizer = true,
  save_encoder = false,
  include_git_info = true,
}), nil, "Serialization.save(starttrain)")

local saved_loop = type(resume_training_state) == "table" and resume_training_state or {}
if saved_loop.training_count and saved_loop.training_count ~= training_count then
  die("reprise incompatible: le nombre d'items d'entraînement a changé")
end
if saved_loop.seed and saved_loop.seed ~= seed then die("reprise incompatible: seed différent du checkpoint") end
local rng_state = tonumber(saved_loop.rng_state) or ((seed % 2147483646) + 1)
local function random_uniform()
  rng_state = (rng_state * 48271) % 2147483647
  return rng_state / 2147483647
end
local order = saved_loop.order or {}
if #order == 0 then for index = 1, training_count do order[index] = index end end
if #order ~= training_count then die("ordre du dataset invalide dans le checkpoint") end
local global_step = tonumber(saved_loop.global_step) or 0
local running_loss = tonumber(saved_loop.running_loss) or 0.0
local total_loss = tonumber(saved_loop.total_loss) or 0.0
local stop_requested = false
local validation_lr_scale = tonumber(saved_loop.validation_lr_scale) or 1.0
local best_validation_score = tonumber(saved_loop.best_validation_score) or math.huge
local validation_score_ema = tonumber(saved_loop.validation_score_ema)
local validation_bad_rounds = tonumber(saved_loop.validation_bad_rounds) or 0
local validation_penalties = tonumber(saved_loop.validation_penalties) or 0
local next_epoch = tonumber(saved_loop.next_epoch) or 1
local next_position = tonumber(saved_loop.next_position) or 1

local function save_checkpoint(label)
  local path = output_dir .. "/" .. label
  FS.mkdir_p(path)
  check(Mimir.Serialization.save(path, "raw_folder", {
    save_optimizer = true,
    training_state = {
      global_step = global_step, next_epoch = next_epoch, next_position = next_position,
      training_count = training_count, order = order, rng_state = rng_state, seed = seed,
      running_loss = running_loss, total_loss = total_loss,
      validation_lr_scale = validation_lr_scale,
      best_validation_score = best_validation_score < math.huge and best_validation_score or nil,
      validation_score_ema = validation_score_ema, validation_bad_rounds = validation_bad_rounds,
      validation_penalties = validation_penalties,
    },
    save_tokenizer = true,
    save_encoder = false,
    include_checksums = true,
    include_git_info = true,
    include_optimizer_state = true,
  }), nil, "Serialization.save(" .. label .. ")")
  logf("checkpoint: " .. path)
end

local function run_validation(reason)
  if validation_count < 1 then return nil end
  if type(Mimir.Viz.validation_enabled) == "function" and
      not Mimir.Viz.validation_enabled() then
    return nil
  end
  local target = validation_count
  local total = target
  local loss_sum = 0.0
  local reconstruction_mae_sum = 0.0
  local reconstruction_mse_sum = 0.0
  local kl_sum = 0.0
  local wasserstein_sum = 0.0
  local entropy_sum = 0.0
  local moment_sum = 0.0
  local spatial_sum = 0.0
  local temporal_sum = 0.0
  Mimir.Viz.set_validation({
    in_progress = true, step = global_step, done = 0, total = total,
    has = false, ok = true, recon = 0.0, kl = 0.0, align = 0.0,
  })
  local done = 0
  local dataset_index = dataset_count
  local validation_first_index = validation_holdout and (training_count + 1) or 1
  while done < target and dataset_index >= validation_first_index do
    if type(Mimir.Viz.validation_enabled) == "function" and
        not Mimir.Viz.validation_enabled() then
      Mimir.Viz.set_validation({
        in_progress = false, step = global_step, done = done, total = target,
        has = false, ok = false,
      })
      logf(
        "validation interrompue step=%d items=%d/%d",
        global_step, done, target)
      return nil
    end
    local item = get_image_item(dataset_index, "validation")
    if item then
      local stats, validation_err = Mimir.Model.lumen_validate_step(
        convert_image_channels(item.image), tostring(item.text or ""),
        math.tointeger(seed + dataset_index * 1000003), {
          annotation = item.annotation,
        })
      item = nil
      release_dataset_item(dataset_index)
      if type(stats) ~= "table" then
        die(validation_err or "validation diffusion impossible")
      end
      done = done + 1
      loss_sum = loss_sum + stats.loss
      reconstruction_mae_sum = reconstruction_mae_sum + stats.reconstruction_mae
      reconstruction_mse_sum = reconstruction_mse_sum + stats.reconstruction_mse
      kl_sum = kl_sum + stats.kl
      wasserstein_sum = wasserstein_sum + stats.wasserstein
      entropy_sum = entropy_sum + stats.entropy_diff
      moment_sum = moment_sum + stats.moment_mismatch
      spatial_sum = spatial_sum + stats.spatial_coherence
      temporal_sum = temporal_sum + stats.temporal_consistency
      Mimir.Viz.set_validation({
        in_progress = true, step = global_step, done = done, total = target,
        has = true, ok = true, recon = reconstruction_mae_sum / done,
        kl = reconstruction_mse_sum / done, align = loss_sum / done,
      })
    end
    dataset_index = dataset_index - 1
  end
  if done < 1 then
    Mimir.Viz.set_validation({
      in_progress = false, step = global_step, done = 0, total = target,
      has = false, ok = false,
    })
    logf("validation=%s ignorée: aucune image décodable", reason)
    return nil
  end
  total = done
  local epsilon_mse = loss_sum / total
  local reconstruction_mae = reconstruction_mae_sum / total
  local reconstruction_mse = reconstruction_mse_sum / total
  local kl = kl_sum / total
  local wasserstein = wasserstein_sum / total
  local entropy_diff = entropy_sum / total
  local moment_mismatch = moment_sum / total
  local spatial_coherence = spatial_sum / total
  local temporal_consistency = temporal_sum / total
  local validation_weight_sum = validation_epsilon_weight +
    validation_reconstruction_weight
  local validation_score =
    (validation_epsilon_weight * epsilon_mse +
     validation_reconstruction_weight * reconstruction_mae) /
    validation_weight_sum
  if not is_finite(validation_score) then
    validation_score_ema = validation_score
  elseif validation_score_ema == nil or not is_finite(validation_score_ema) then
    validation_score_ema = validation_score
  else
    validation_score_ema = validation_ema_alpha * validation_score +
      (1.0 - validation_ema_alpha) * validation_score_ema
  end
  local feedback = "NEUTRAL"
  local reward_penalty = 0.0
  if validation_feedback_enabled and global_step >= validation_feedback_min_steps then
    if not is_finite(validation_score_ema) then
      validation_bad_rounds = validation_penalty_patience
      feedback = "INVALID"
    elseif best_validation_score == math.huge then
      best_validation_score = validation_score_ema
      validation_bad_rounds = 0
      feedback = "BASELINE"
      if validation_save_best then save_checkpoint("validation_best") end
    else
      local denominator = math.max(math.abs(best_validation_score), 1e-12)
      reward_penalty = (best_validation_score - validation_score_ema) / denominator
      if reward_penalty > validation_improve_threshold then
        validation_lr_scale = math.min(validation_lr_scale_max,
          validation_lr_scale * validation_reward_factor)
        best_validation_score = validation_score_ema
        validation_bad_rounds = 0
        feedback = "REWARD"
        if validation_save_best then save_checkpoint("validation_best") end
      elseif reward_penalty < -validation_improve_threshold then
        validation_bad_rounds = validation_bad_rounds + 1
        feedback = "WATCH"
      else
        validation_bad_rounds = math.max(0, validation_bad_rounds - 1)
      end
    end
    if validation_bad_rounds >= validation_penalty_patience then
      validation_lr_scale = math.max(validation_lr_scale_min,
        validation_lr_scale * validation_penalty_factor)
      validation_bad_rounds = 0
      validation_penalties = validation_penalties + 1
      feedback = "PENALTY"
      if validation_max_penalties > 0 and
          validation_penalties >= validation_max_penalties then
        stop_requested = true
        feedback = "EARLY_STOP"
      end
    end
  end
  Mimir.Viz.set_validation({
    in_progress = false, step = global_step, done = total, total = total,
    has = true, ok = is_finite(validation_score),
    recon = reconstruction_mae, kl = reconstruction_mse, align = epsilon_mse,
    feedback = feedback == "REWARD" and "reward" or
      feedback == "PENALTY" and "penalty" or
      feedback == "WATCH" and "plateau" or
      feedback == "EARLY_STOP" and "penalty" or "none",
  })
  logf(
    "validation=%s step=%d items=%d epsilon_mse=%.7g recon_mae=%.7g " ..
    "recon_mse=%.7g kl=%.7g wa=%.7g entropy=%.7g moment_miss=%.7g spatial_co=%.7g " ..
    "temporal=%.7g score=%.7g ema=%.7g feedback=%s signal=%+.6g lr_scale=%.6g " ..
    "bad=%d penalties=%d",
    reason, global_step, total, epsilon_mse, reconstruction_mae,
    reconstruction_mse, kl, wasserstein, entropy_diff, moment_mismatch,
    spatial_coherence, temporal_consistency, validation_score,
    validation_score_ema, feedback, reward_penalty, validation_lr_scale,
    validation_bad_rounds, validation_penalties)
  return validation_score
end

for epoch = next_epoch, next_epoch + epochs - 1 do
  local first_position = next_position
  if first_position == 1 then
    for index = training_count, 2, -1 do
      local other = math.floor(random_uniform() * index) + 1
      order[index], order[other] = order[other], order[index]
    end
  end

  for position = first_position, training_count do
    next_epoch = position == training_count and epoch + 1 or epoch
    next_position = position == training_count and 1 or position + 1
    local dataset_index = order[position]
    local item = get_image_item(dataset_index, "entraînement")
    if item then

    global_step = global_step + 1
    local prompt = tostring(item.text or "")
    local drop_conditioning = random_uniform() < cfg_dropout
    local warmup = warmup_steps > 0 and math.min(1.0, global_step / warmup_steps) or 1.0
    local step_lr = learning_rate * warmup * validation_lr_scale
    local step_seed = seed + global_step * 1000003

    local stats, step_err = Mimir.Model.lumen_train_step(
      convert_image_channels(item.image), prompt, step_seed, step_lr, optimizer, {
        annotation = item.annotation,
        drop_conditioning = drop_conditioning,
        epoch = epoch,
        total_epochs = epochs,
        batch = position,
        total_batches = training_count,
        step = global_step,
        completed_steps = global_step - 1,
        loss_sum_before = total_loss,
        lr_warmup_steps = warmup_steps,
      })
    item = nil
    release_dataset_item(dataset_index)
    if type(stats) ~= "table" then die(step_err or "lumen_train_step impossible") end
    local applied_lr = tonumber(stats.learning_rate) or step_lr
    running_loss = running_loss + stats.loss
    total_loss = total_loss + stats.loss

    if stats.stop_requested then
      logf("arrêt sécurisé demandé au step=%d", global_step)
      stop_requested = true
      break
    end

    if global_step == 1 or global_step % log_every == 0 then
      local divisor = global_step == 1 and 1 or log_every
      logf(
        "epoch=%d/%d step=%d item=%d/%d loss=%.7g avg=%.7g grad=%.6g " ..
        "mse=%.6g kl=%.6g kl_beta_eff=%.6g wa=%.6g ent=%.6g mom=%.6g " ..
        "spat=%.6g temp=%.6g t=%d lr=%.6g",
        epoch, epochs, global_step, position, training_count, stats.loss,
        running_loss / divisor, stats.grad_norm, stats.mse, stats.kl,
        stats.kl_beta_effective, stats.wasserstein,
        stats.entropy_diff, stats.moment_mismatch, stats.spatial_coherence,
        stats.temporal_consistency, stats.timestep, applied_lr)
      running_loss = 0.0
    end

    if validation_every_steps > 0 and global_step % validation_every_steps == 0 then
      run_validation("step")
      if stop_requested then break end
    end
    if save_every > 0 and global_step % save_every == 0 then
      save_checkpoint(string.format("step_%08d", global_step))
    end
    end
  end
  if stop_requested then break end
  if validation_every_epochs > 0 and epoch % validation_every_epochs == 0 then
    run_validation("epoch")
    if stop_requested then break end
  end
end

save_checkpoint("final")
check(Mimir.Serialization.save(output_dir .. "/endtrain.json", "debug_json", {
  save_tokenizer = true,
  save_encoder = false,
  include_git_info = true,
  include_optimizer_state = true,
}), nil, "Serialization.save(endtrain)")

logf("terminé: epochs=%d steps=%d sortie=%s",
  epochs, global_step, output_dir)
if skipped_images > 0 then
  logf("images ignorées au total: %d", skipped_images)
end
