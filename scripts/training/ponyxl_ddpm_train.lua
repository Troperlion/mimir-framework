#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil
-- Entraînement PonyXL-DDPM (diffusion latente SDXL-like) via boucle Lua + Mimir.Model.ponyxl_ddpm_train_step()
--
-- Dataset attendu: un DOSSIER, indexé par Helper.hpp::loadDataset via Mimir.Dataset.load().
-- Chaque item doit contenir une paire image+texte associée par basename:
--   - image: .png (ou autre format supporté)
--   - texte: .txt (caption/prompt)
--
-- Notes:
-- - Ici, `recon_loss` désigne la loss de diffusion utilisée pour eps_pred vs eps (ex: mse/huber/charbonnier).
-- - Le modèle gère: VAE encode (mu), sampling t/eps, schedule DDPM, etc.
-- - NOTE: les anciennes options "peltier-*" sont désormais DÉPRÉCIÉES et ignorées
--   (le modèle C++ utilise uniquement eps ~ N(0,I), i.e. DDPM latent standard).
--
-- Usage (exemple):
--   ./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua -- \
--     --dataset ./dataset_pairs \
--     --epochs 5 --lr 1e-4 \
--     --train-timesteps 1000 --seed 1337 \
--     --tokenizer checkpoint/base_tokenizer/tokenizer.json \
--     --vae-checkpoint checkpoint/vae_conv_base_tok_latent-16-BS-320 \
--     --recon-loss huber \
--     --autosave-every 200
--
-- Recommandé pour dataset image+texte:
--   --dataset-w 512 --dataset-h 512 --dataset-min-modalities 2
--
-- Viz (optionnel):
--   --viz --viz-title "PonyXL" --viz-width 1600 --viz-height 900 --viz-fps 60
--   --viz-ddpm --viz-ddpm-every 200 --viz-ddpm-steps 5

local function die(msg)
  log("❌ " .. tostring(msg))
  os.exit(1)
end

local function logf(fmt, ...)
  log(string.format(fmt, ...))
end

-- ---------------------------------------------------------------------------
-- Args
-- ---------------------------------------------------------------------------

local Args = dofile("scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local function opt_has(k)
  return opts[k] ~= nil
end

local function opt_bool(k, d)
  local v = opts[k]
  if v == nil then return d end
  if v == true or v == false then return v end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local function opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

-- ---------------------------------------------------------------------------
-- Helpers: checkpoint resolution (no external deps)
-- ---------------------------------------------------------------------------

local function shell_quote(s)
  s = tostring(s or "")
  return "'" .. s:gsub("'", "'\"'\"'") .. "'"
end

local function file_exists(path)
  local f = io.open(path, "rb")
  if f then f:close(); return true end
  return false
end

local function list_dir_names(path)
  local out = {}
  local cmd = "ls -1 " .. shell_quote(path) .. " 2>/dev/null"
  local p = io.popen(cmd, "r")
  if not p then return out end
  for line in p:lines() do
    if line and #line > 0 then out[#out + 1] = line end
  end
  p:close()
  return out
end

local function resolve_checkpoint_dir(path_in)
  local p = tostring(path_in or "")
  if p == "" then return p end

  if file_exists(p .. "/model/architecture.json") or file_exists(p .. "/architecture.json") then
    return p
  end

  if p:match("/model/?$") then
    p = p:gsub("/model/?$", "")
    if file_exists(p .. "/model/architecture.json") then return p end
  end

  local best_name, best_epoch, best_stop = nil, -1, false
  for _, name in ipairs(list_dir_names(p)) do
    local digits, suffix = name:match("^epoch_(%d+)(.*)$")
    if digits then
      local e = tonumber(digits) or -1
      local sfx = tostring(suffix or "")
      local is_stop = (sfx:lower():find("stop", 1, true) ~= nil) or (sfx:lower():find("final", 1, true) ~= nil)
      if e > best_epoch then
        best_epoch, best_name, best_stop = e, name, is_stop
      elseif e == best_epoch and best_name ~= nil then
        if (not best_stop and is_stop) or (best_stop == is_stop and name > best_name) then
          best_name, best_stop = name, is_stop
        end
      end
    end
  end

  if best_name then
    return p .. "/" .. best_name
  end
  return p
end

-- ---------------------------------------------------------------------------
-- Diffusion schedule helpers
-- ---------------------------------------------------------------------------

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function ddpm_alpha_bar_end(T, beta0, beta1)
  T = math.max(2, math.floor(tonumber(T) or 2))
  beta0 = clamp(tonumber(beta0) or 0, 0.0, 0.999)
  beta1 = clamp(tonumber(beta1) or 0, 0.0, 0.999)
  local log_ab = 0.0
  for i = 0, T - 1 do
    local frac = (T > 1) and (i / (T - 1)) or 0.0
    local beta = clamp(beta0 + (beta1 - beta0) * frac, 0.0, 0.999)
    log_ab = log_ab + math.log(math.max(1e-12, 1.0 - beta))
  end
  return math.exp(log_ab)
end

local function solve_beta_end_for_target_ab(T, beta_start, target_ab_end)
  T = math.max(2, math.floor(tonumber(T) or 2))
  beta_start = clamp(tonumber(beta_start) or 1e-4, 0.0, 0.999)
  target_ab_end = clamp(tonumber(target_ab_end) or 4e-5, 1e-8, 0.999)

  local lo = beta_start
  local hi = 0.999
  for _ = 1, 40 do
    local mid = 0.5 * (lo + hi)
    local ab = ddpm_alpha_bar_end(T, beta_start, mid)
    if ab > target_ab_end then
      lo = mid
    else
      hi = mid
    end
  end
  return clamp(0.5 * (lo + hi), beta_start, 0.999)
end

-- ---------------------------------------------------------------------------
-- Runtime / mémoire / hardware
-- ---------------------------------------------------------------------------

local MEM_GB = opt_num("mem-gb", 15)
local ALLOC_GB = opt_num("alloc-gb", MEM_GB)
local ENABLE_COMPRESSION = opt_bool("compress", false)

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({ max_ram_gb = ALLOC_GB, enable_compression = ENABLE_COMPRESSION })
end
if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, MEM_GB)
end
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  -- Désactivé par défaut: activez via `--hw true` si souhaité.
  pcall(Mimir.Model.set_hardware, opt_bool("hw", true))
end

-- ---------------------------------------------------------------------------
-- Entrées principales
-- ---------------------------------------------------------------------------

local DATASET_DIR = Args.get_str(opts, "dataset", "dataset_2")
local EPOCHS = Args.get_int(opts, "epochs", 5)
local LR = Args.get_num(opts, "lr", 1e-5)

local TRAIN_SEED = Args.get_int(opts, "seed", 4242)
local INIT_SEED = Args.get_int(opts, "init-seed", TRAIN_SEED)

local MAX_ITEMS = Args.get_int(opts, "max-items", 0)

local DATASET_W = Args.get_int(opts, "dataset-w", 512)
local DATASET_H = Args.get_int(opts, "dataset-h", 512)
local DATASET_MIN_MODALITIES = Args.get_int(opts, "dataset-min-modalities", 2)

local TOKENIZER_PATH = Args.get_str(opts, "tokenizer", "checkpoint/base_tokenizer/tokenizer.json")
local DESIRED_MAX_VOCAB = Args.get_int(opts, "max-vocab", 100000)

local DEFAULT_VAE_CKPT = "checkpoint/vae_conv_cpu_512_latent-64-64-32_base-64-2/epoch_0011"
local VAE_CKPT_IN = Args.get_str(opts, "vae-checkpoint", DEFAULT_VAE_CKPT)
local VAE_CKPT = resolve_checkpoint_dir(VAE_CKPT_IN)

local OUT_DIR = Args.get_str(opts, "out-dir", "checkpoint/PonyXL_SDXL")

local TRAIN_TIMESTEPS = Args.get_int(opts, "train-timesteps", Args.get_int(opts, "ddpm-steps", 1000))
local BETA_START = Args.get_num(opts, "beta-start", 1e-4)
local BETA_END = Args.get_num(opts, "beta-end", 0.02)
local TARGET_AB_END = Args.get_num(opts, "ddpm-alpha-bar-end", Args.get_num(opts, "alpha-bar-end", 4e-6))
local AUTO_BETA_END = opt_bool("auto-beta-end", true)
if AUTO_BETA_END and (not opt_has("beta-end")) then
  BETA_END = solve_beta_end_for_target_ab(TRAIN_TIMESTEPS, BETA_START, TARGET_AB_END)
end

local DDPM_STEPS_PER_IMAGE = Args.get_int(opts, "ddpm-steps-per-image", 1)

local function default_steps_per_image(train_items)
  if train_items and train_items > 0 and train_items < 512 then return 2 end
  return 1
end

local DEPRECATED_PELTIER_USED = opt_has("peltier-noise") or opt_has("peltier-mix") or opt_has("peltier-blur") or opt_has("peltier-blur-radius")
if DEPRECATED_PELTIER_USED then
  log("[ponyxl_ddpm_train] ⚠️ options 'peltier-*' dépréciées et ignorées (DDPM latent standard: eps~N(0,I))")
end

local TEXT_CTX_LEN = Args.get_int(opts, "text-ctx-len", 75)
local TEXT_MEANPOOL = opt_bool("text-meanpool", false)

local RECON_LOSS = Args.get_str(opts, "recon-loss", "mse")

-- Diffusion conditioning / stabilization knobs (C++ config keys)
local TIMESTEP_COND = Args.get_str(opts, "timestep-cond", "log_snr")
local LOSS_WEIGHTING = Args.get_str(opts, "loss-weighting", "none")
local MIN_SNR_GAMMA = Args.get_num(opts, "min-snr-gamma", 5.0)
local OUTPUT_ACTIVATION = Args.get_str(opts, "output-activation", "linear")
local CFG_DROPOUT = Args.get_num(opts, "cfg-dropout", 0.10)

-- Optional: image-space auxiliary loss (via VAE decoder). Coûteux.
local IMG_LOSS_WEIGHT = Args.get_num(opts, "img-loss-weight", 0.0)
local IMG_LOSS_EVERY = Args.get_int(opts, "img-loss-every", Args.get_int(opts, "img-loss-every-steps", 0))

local KL_BETA = Args.get_num(opts, "kl-beta", 0.0)
local KL_WARMUP_STEPS = Args.get_int(opts, "kl-warmup-steps", 0)
local LOGVAR_CLIP_MIN = Args.get_num(opts, "logvar-clip-min", -10.0)
local LOGVAR_CLIP_MAX = Args.get_num(opts, "logvar-clip-max", 10.0)

local OPTIMIZER = Args.get_str(opts, "optimizer", "adamw")
local WARMUP_STEPS = Args.get_int(opts, "lr-warmup-steps", Args.get_int(opts, "warmup-steps", 800))

local VIZ_DDPM = opt_bool("viz-ddpm", true)
local VIZ_DDPM_EVERY = Args.get_int(opts, "viz-ddpm-every", 200)
local VIZ_DDPM_STEPS = Args.get_int(opts, "viz-ddpm-steps", 1)

local VALIDATE_EVERY = Args.get_int(opts, "validate-every", Args.get_int(opts, "validate-every-steps", 100))
local VALIDATE_ITEMS = math.max(1, Args.get_int(opts, "validate-items", 4))
local VALIDATE_HOLDOUT = opt_bool("validate-holdout", true)
local VALIDATE_HOLDOUT_FRAC = Args.get_num(opts, "validate-holdout-frac", 0.01)
local VALIDATE_HOLDOUT_ITEMS = Args.get_int(opts, "validate-holdout-items", 6)
local VALIDATE_SAVE_DEBUG = opt_bool("validate-save-debug", true)
local VALIDATE_SEED = Args.get_int(opts, "validate-seed", 4242)
local VALIDATE_T = Args.get_int(opts, "validate-t", 1000)

local AUTOSAVE_EVERY = Args.get_int(opts, "autosave-every", 800)
local AUTOSAVE_DIR = Args.get_str(opts, "autosave-dir", OUT_DIR .. "/autosave_latest")

-- Avec `Mimir.Model.train`, l'autosave est géré côté framework à la fin des epochs.
-- On mappe l'ancienne option (par steps) vers un autosave par epoch si activé.
local AUTOSAVE_EVERY_EPOCHS = Args.get_int(opts, "autosave-every-epochs", (AUTOSAVE_EVERY > 0) and 1 or 0)

local CALIBRATE_VAE = opt_bool("calibrate-vae", true)
local CALIBRATE_ITEMS = Args.get_int(opts, "calibrate-items", 30)

local INIT_WEIGHTS = opt_bool("init-weights", true)

-- ---------------------------------------------------------------------------
-- Tokenizer
-- ---------------------------------------------------------------------------

log("[ponyxl_ddpm_train] init tokenizer")
Mimir.Tokenizer.load(TOKENIZER_PATH)
if Mimir.Tokenizer.set_max_vocab then
  pcall(Mimir.Tokenizer.set_max_vocab, DESIRED_MAX_VOCAB)
end
Mimir.Tokenizer.ensure_vocab_from_text("pony horse snow forest portrait TAGS CONTEXTE MENTALITE TEXTE DESCRIPTION")

local TOKENIZER_MAX_VOCAB = (Mimir.Tokenizer.get_max_vocab and Mimir.Tokenizer.get_max_vocab()) or Mimir.Tokenizer.vocab_size()
logf("[ponyxl_ddpm_train] tokenizer loaded: path=%s vocab_size=%d max_vocab=%d",
  tostring(TOKENIZER_PATH),
  tonumber(Mimir.Tokenizer.vocab_size() or 0) or 0,
  tonumber(TOKENIZER_MAX_VOCAB or 0) or 0)

-- ---------------------------------------------------------------------------
-- Dataset
-- ---------------------------------------------------------------------------

log("[ponyxl_ddpm_train] load dataset")
local ok_ds, n_or_err = Mimir.Dataset.load(DATASET_DIR, DATASET_W, DATASET_H, DATASET_MIN_MODALITIES)
if not ok_ds then die(n_or_err or "Dataset.load a échoué") end
local DATASET_TOTAL = math.floor(tonumber(n_or_err) or 0)
if DATASET_TOTAL <= 0 then die("dataset vide") end
logf("[ponyxl_ddpm_train] dataset_total=%d (dir=%s) w=%d h=%d min_modalities=%d",
  DATASET_TOTAL, tostring(DATASET_DIR), DATASET_W, DATASET_H, DATASET_MIN_MODALITIES)

local function load_image_text_item(i)
  local item, err_item = Mimir.Dataset.get(i)
  if not item then return nil, err_item or "Dataset.get a échoué" end

  local prompt = item.text
  if prompt == nil then return nil, "item sans texte (attendu paire image+txt)" end
  if type(prompt) ~= "string" then prompt = tostring(prompt) end
  if #prompt == 0 then return nil, "item texte vide" end

  local img = item.image
  if type(img) ~= "table" then return nil, "item sans image" end

  local w = tonumber(item.width) or DATASET_W
  local h = tonumber(item.height) or DATASET_H
  local expected = math.floor(w * h * 3)
  if expected <= 0 then return nil, "dimensions invalides w/h" end
  if #img ~= expected then
    return nil, "buffer RGB invalide: got=" .. tostring(#img) .. " expected=" .. tostring(expected)
  end

  return { prompt = prompt, img = img, w = w, h = h }
end

local function shuffle_in_place(t, seed)
  if type(t) ~= "table" then return end
  local state = tonumber(seed) or 0
  if state < 0 then state = -state end
  state = math.floor(state) % 2147483647
  if state == 0 then state = 123456789 end
  local function rand_u32()
    state = (1103515245 * state + 12345) % 2147483647
    return state
  end
  local function rand_int(n)
    if n <= 1 then return 1 end
    return (rand_u32() % n) + 1
  end
  for i = #t, 2, -1 do
    local j = rand_int(i)
    t[i], t[j] = t[j], t[i]
  end
end

local all_indices = {}
for i = 1, DATASET_TOTAL do all_indices[i] = i end
shuffle_in_place(all_indices, TRAIN_SEED)

local holdout_n = 0
if VALIDATE_EVERY > 0 and VALIDATE_HOLDOUT then
  if VALIDATE_HOLDOUT_ITEMS > 0 then
    holdout_n = VALIDATE_HOLDOUT_ITEMS
  else
    holdout_n = math.floor((VALIDATE_HOLDOUT_FRAC or 0.0) * DATASET_TOTAL)
  end
  if holdout_n < VALIDATE_ITEMS then holdout_n = VALIDATE_ITEMS end
  if holdout_n > (DATASET_TOTAL - 1) then holdout_n = math.max(0, DATASET_TOTAL - 1) end
end

local holdout_indices, train_indices = {}, {}
if holdout_n > 0 then
  for k = 1, holdout_n do holdout_indices[#holdout_indices + 1] = all_indices[k] end
  for k = holdout_n + 1, #all_indices do train_indices[#train_indices + 1] = all_indices[k] end
  logf("[ponyxl_ddpm_train] holdout split: holdout=%d train=%d", #holdout_indices, #train_indices)
else
  train_indices = all_indices
end

local TRAIN_ITEMS_PER_EPOCH = #train_indices
if MAX_ITEMS > 0 and MAX_ITEMS < TRAIN_ITEMS_PER_EPOCH then
  TRAIN_ITEMS_PER_EPOCH = MAX_ITEMS
end

if DDPM_STEPS_PER_IMAGE <= 0 then
  DDPM_STEPS_PER_IMAGE = default_steps_per_image(TRAIN_ITEMS_PER_EPOCH)
end

-- ---------------------------------------------------------------------------
-- Modèle
-- ---------------------------------------------------------------------------

log("[ponyxl_ddpm_train] create model")
---@type any
local cfg = Mimir.Architectures.default_config("ponyxl_ddpm")

cfg.max_vocab = TOKENIZER_MAX_VOCAB or cfg.max_vocab
cfg.seed = TRAIN_SEED
cfg.recon_loss = RECON_LOSS

cfg.timestep_cond = TIMESTEP_COND
cfg.loss_weighting = LOSS_WEIGHTING
cfg.min_snr_gamma = MIN_SNR_GAMMA
cfg.output_activation = OUTPUT_ACTIVATION
cfg.cfg_dropout_prob = CFG_DROPOUT

cfg.text_ctx_len = TEXT_CTX_LEN
cfg.text_bottleneck_meanpool = TEXT_MEANPOOL

-- Texte: options SDXL-like
cfg.text_clip_like = opt_bool("text-clip-like", cfg.text_clip_like ~= false)
cfg.global_ctx_tokens = Args.get_int(opts, "global-ctx-tokens", tonumber(cfg.global_ctx_tokens or 0) or 0)
cfg.sdxl_time_cond = opt_bool("sdxl-time-cond", cfg.sdxl_time_cond ~= false)

cfg.vae_checkpoint = VAE_CKPT
cfg.checkpoint_dir = OUT_DIR

cfg.image_w = DATASET_W
cfg.image_h = DATASET_H
cfg.image_c = 3

cfg.ddpm_steps = TRAIN_TIMESTEPS
cfg.ddpm_beta_start = BETA_START
cfg.ddpm_beta_end = BETA_END
cfg.ddpm_steps_per_image = DDPM_STEPS_PER_IMAGE

cfg.optimizer = OPTIMIZER
cfg.beta1 = 0.9
cfg.beta2 = 0.999
cfg.weight_decay = Args.get_num(opts, "weight-decay", 0.0)
cfg.decay_strategy = Args.get_str(opts, "decay-strategy", "linear")
cfg.warmup_steps = Args.get_int(opts, "warmup-steps", WARMUP_STEPS)

cfg.log_every = 1
cfg.max_items = MAX_ITEMS
cfg.autosave_every_epochs = AUTOSAVE_EVERY_EPOCHS

-- KL prior regularization (optional)
cfg.kl_beta = KL_BETA
cfg.kl_warmup_steps = KL_WARMUP_STEPS
cfg.logvar_clip_min = LOGVAR_CLIP_MIN
cfg.logvar_clip_max = LOGVAR_CLIP_MAX

-- Validation (gérée par `Mimir.Model.train` côté framework)
cfg.validate_every_steps = VALIDATE_EVERY
cfg.validate_items = VALIDATE_ITEMS
cfg.validate_holdout = VALIDATE_HOLDOUT
cfg.validate_holdout_frac = VALIDATE_HOLDOUT_FRAC
cfg.validate_holdout_items = VALIDATE_HOLDOUT_ITEMS
cfg.validate_seed = VALIDATE_SEED
cfg.validate_t = VALIDATE_T

cfg.caption_structured_enable = opt_bool("caption-structured-enable", cfg.caption_structured_enable ~= false)
cfg.caption_structured_canonicalize = opt_bool("caption-structured-canonicalize", cfg.caption_structured_canonicalize ~= false)
cfg.caption_tags_dropout_prob = opt_num("caption-tags-dropout", tonumber(cfg.caption_tags_dropout_prob) or 0.0)
cfg.caption_contexte_dropout_prob = opt_num("caption-contexte-dropout", tonumber(cfg.caption_contexte_dropout_prob) or 0.0)
cfg.caption_mentalite_dropout_prob = opt_num("caption-mentalite-dropout", tonumber(cfg.caption_mentalite_dropout_prob) or 0.0)
cfg.caption_texte_dropout_prob = opt_num("caption-texte-dropout", tonumber(cfg.caption_texte_dropout_prob) or 0.0)

-- Captions key/value + term-frequency boost (optionnel)
if cfg.caption_kv_enable ~= nil then
  cfg.caption_kv_enable = opt_bool("caption-kv-enable", cfg.caption_kv_enable ~= false)
end
if cfg.term_freq_boost_enable ~= nil then
  cfg.term_freq_boost_enable = opt_bool("term-freq-boost-enable", cfg.term_freq_boost_enable ~= false)
  cfg.term_freq_boost_use_tokens = opt_bool("term-freq-boost-use-tokens", cfg.term_freq_boost_use_tokens ~= false)
  cfg.term_freq_boost_use_keywords = opt_bool("term-freq-boost-use-keywords", cfg.term_freq_boost_use_keywords ~= false)
  cfg.term_freq_boost_top_k = Args.get_int(opts, "term-freq-boost-top-k", tonumber(cfg.term_freq_boost_top_k or 0) or 0)
  cfg.term_freq_boost_repeat = Args.get_int(opts, "term-freq-boost-repeat", tonumber(cfg.term_freq_boost_repeat or 0) or 0)
  cfg.term_freq_boost_start_step = Args.get_int(opts, "term-freq-boost-start-step", tonumber(cfg.term_freq_boost_start_step or 0) or 0)
  cfg.term_freq_boost_update_every_steps = Args.get_int(opts, "term-freq-boost-update-every-steps", tonumber(cfg.term_freq_boost_update_every_steps or 0) or 0)
end

cfg.viz_taps_max_frames = Args.get_int(opts, "viz-taps", Args.get_int(opts, "viz-taps-max-frames", 64))
cfg.viz_taps_max_side = Args.get_int(opts, "viz-taps-size", Args.get_int(opts, "viz-taps-max-side", 720))
if VIZ_DDPM then
  cfg.viz_ddpm_every_steps = VIZ_DDPM_EVERY
  cfg.viz_ddpm_num_steps = VIZ_DDPM_STEPS
end

-- Image-space auxiliary loss (disabled by default)
cfg.img_loss_weight = IMG_LOSS_WEIGHT
cfg.img_loss_every_steps = IMG_LOSS_EVERY

if Args.apply_overrides and opts and opts.override ~= nil then
  local ok_ov, err_ov = pcall(Args.apply_overrides, cfg, opts)
  if not ok_ov then die(err_ov) end
  log("[ponyxl_ddpm_train] overrides applied")
end

logf("[ponyxl_ddpm_train] vae-checkpoint=%s", tostring(VAE_CKPT))
logf("[ponyxl_ddpm_train] ddpm: T=%d beta_start=%.6g beta_end=%.6g alpha_bar_end=%.6g steps_per_image=%d",
  cfg.ddpm_steps,
  cfg.ddpm_beta_start,
  cfg.ddpm_beta_end,
  ddpm_alpha_bar_end(cfg.ddpm_steps, cfg.ddpm_beta_start, cfg.ddpm_beta_end),
  cfg.ddpm_steps_per_image)
logf("[ponyxl_ddpm_train] loss: recon_loss=%s", tostring(cfg.recon_loss))
logf("[ponyxl_ddpm_train] timestep_cond=%s loss_weighting=%s min_snr_gamma=%.6g output_activation=%s cfg_dropout_prob=%.6g",
  tostring(cfg.timestep_cond),
  tostring(cfg.loss_weighting),
  tonumber(cfg.min_snr_gamma or 0.0) or 0.0,
  tostring(cfg.output_activation),
  tonumber(cfg.cfg_dropout_prob or 0.0) or 0.0)
logf("[ponyxl_ddpm_train] kl: kl_beta=%.6g kl_warmup_steps=%d logvar_clip=[%.6g,%.6g]",
  tonumber(cfg.kl_beta or 0.0) or 0.0,
  tonumber(cfg.kl_warmup_steps or 0) or 0,
  tonumber(cfg.logvar_clip_min or -10.0) or -10.0,
  tonumber(cfg.logvar_clip_max or 10.0) or 10.0)
logf("[ponyxl_ddpm_train] text: ctx_len=%d meanpool=%s", cfg.text_ctx_len, tostring(cfg.text_bottleneck_meanpool))
logf("[ponyxl_ddpm_train] optimizer: type=%s decay_strategy=%s warmup_steps=%d",
  tostring(cfg.optimizer), tostring(cfg.decay_strategy), tonumber(cfg.warmup_steps or 0) or 0)
logf("[ponyxl_ddpm_train] viz_taps: max_frames=%d max_side=%d (actif seulement si mimir est lancé avec --viz)",
  tonumber(cfg.viz_taps_max_frames or 0) or 0, tonumber(cfg.viz_taps_max_side or 0) or 0)

local ok_create, err_create = Mimir.Model.create("ponyxl_ddpm", cfg)
if not ok_create then die(err_create or "Model.create a échoué") end

log("[ponyxl_ddpm_train] allocate/init")
Mimir.Model.allocate_params()
if INIT_WEIGHTS then
  Mimir.Model.init_weights("xavier", INIT_SEED)
end

-- ---------------------------------------------------------------------------
-- Calibrage VAE scale (optionnel)
-- ---------------------------------------------------------------------------

if CALIBRATE_VAE then
  local n_items = math.max(0, math.floor(tonumber(CALIBRATE_ITEMS) or 0))
  if n_items > 0 then
    logf("[ponyxl_ddpm_train] calibrate vae_scale: items=%d", n_items)
    local sum, sumsq, n = 0.0, 0.0, 0
    local used = 0
    for j = 1, math.min(n_items, #train_indices) do
      local idx = train_indices[j]
      local s, err_s = load_image_text_item(idx)
      if not s then
        log("[ponyxl_ddpm_train] calibrate skip item " .. tostring(idx) .. ": " .. tostring(err_s))
      else
        local m, err_m = Mimir.Model.ponyxl_ddpm_vae_mu_moments(s.img, s.w, s.h)
        if not m then
          log("[ponyxl_ddpm_train] calibrate error: " .. tostring(err_m))
        else
          sum = sum + (tonumber(m.sum) or 0.0)
          sumsq = sumsq + (tonumber(m.sumsq) or 0.0)
          n = n + (tonumber(m.n) or 0)
          used = used + 1
        end
      end
    end

    if n > 0 then
      local mean = sum / n
      local var = (sumsq / n) - (mean * mean)
      if var < 1e-12 then var = 1e-12 end
      local std = math.sqrt(var)
      local scale = 1.0 / std
      local ok_set, err_set = Mimir.Model.ponyxl_ddpm_set_vae_scale(scale)
      if not ok_set then
        log("[ponyxl_ddpm_train] calibrate: set_vae_scale failed: " .. tostring(err_set))
      else
        local cur = Mimir.Model.ponyxl_ddpm_get_vae_scale()
        logf("[ponyxl_ddpm_train] calibrate done: used=%d n=%d mean=%.6g std=%.6g -> vae_scale=%.6g (cur=%.6g)",
          used, n, mean, std, scale, tonumber(cur) or scale)
      end
    else
      log("[ponyxl_ddpm_train] calibrate skipped: n==0")
    end
  end
end

-- ---------------------------------------------------------------------------
-- Entraînement
-- ---------------------------------------------------------------------------

log("[ponyxl_ddpm_train] start ....")
logf("[ponyxl_ddpm_train] epochs=%d lr=%.6g max_items=%d autosave_every_epochs=%d", EPOCHS, LR, MAX_ITEMS, AUTOSAVE_EVERY_EPOCHS)

local ok_train, steps_or_err = Mimir.Model.train(EPOCHS, LR)
if not ok_train then
  if steps_or_err == "STOP_REQUESTED" then
    log("[ponyxl_ddpm_train] ⛔ stop demandé via Viz")
  else
    die(steps_or_err or "Model.train a échoué")
  end
else
  log("✓ ponyxl_ddpm training complete (steps=" .. tostring(steps_or_err) .. ")")
end

-- ---------------------------------------------------------------------------
-- Sauvegarde finale
-- ---------------------------------------------------------------------------

os.execute("mkdir -p '" .. OUT_DIR:gsub("'", "'\\''") .. "' 2>/dev/null")
local ok_save, err_save = Mimir.Serialization.save(OUT_DIR, "raw_folder", {
  save_optimizer = true,
  save_tokenizer = true,
  save_encoder = true,
  include_checksums = true,
  include_git_info = true,
  include_gradients = true,
  include_activations = true,
  include_optimizer_state = true,
  include_weight_deltas = true,
})

if not ok_save then
  die("save a échoué: " .. tostring(err_save))
end

log("✓ saved to " .. tostring(OUT_DIR))
