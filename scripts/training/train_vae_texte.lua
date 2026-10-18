---@diagnostic disable: undefined-global, undefined-field, inject-field

-- Train VAEText (vae_text) sur un dataset de textes (text_file).
--
-- Usage:
--   ./bin/mimir --lua scripts/training/train_vae_texte.lua -- \
--     --dataset-root ../dataset_2 \
--     --out-dir checkpoint/vae_text_trained \
--     --epochs 5 --lr 1e-4 \
--     --seq-len 256 --d-model 256 --latent-tokens 32 \
--     --kl-beta 0.01 --recon-loss ce
--
-- Compat args alignée avec train_vae_conv.lua:
--   --arch/--model/--model-type, --seed/--init-seed, --resume,
--   --optimizer/beta1/beta2/epsilon/weight-decay,
--   --validate-*, --autosave-every-epochs, --dtype, --override, etc.

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")

local function opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

local function opt_int(k, d)
  return math.floor(opt_num(k, d))
end

local function opt_str(k, d)
  local v = opts[k]
  if v == nil or v == true then return d end
  return tostring(v)
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

local BaseTok = dofile(ROOTWORK.."/scripts/modules/base_tokenizer.lua")

local function assert_ok(ok, err, msg)
  if ok == false then
    error((msg or "Operation failed") .. ": " .. tostring(err))
  end
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" then return true end

  local dtype_fn = nil
  if type(Mimir.model) == "table" and type(Mimir.model.dtype) == "function" then
    dtype_fn = Mimir.model.dtype
  elseif type(Mimir.Model) == "table" and type(Mimir.Model.dtype) == "function" then
    dtype_fn = Mimir.Model.dtype
  end
  if dtype_fn == nil then return true end

  local ok, dt_or_err = dtype_fn(dtype)
  assert_ok(ok, dt_or_err, "Model.dtype(" .. tostring(dtype) .. ") failed")
  return true
end

local CPU_ONLY = opt_bool("cpu-only", opt_bool("cpu_only", opt_bool("cpu", false)))
local MEM_GB = opt_num("mem-gb", 15)
local ALLOC_GB = opt_num("alloc-gb", MEM_GB)
local ENABLE_COMPRESSION = opt_bool("compression", opt_bool("compress", true))

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({max_ram_gb = ALLOC_GB, enable_compression = ENABLE_COMPRESSION})
end
if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, MEM_GB)
end
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  local hw_enabled = opt_bool("hw", true)
  if CPU_ONLY then hw_enabled = false end
  pcall(Mimir.Model.set_hardware, hw_enabled)
end

local dataset_root = opt_str("dataset-root", "./dataset_2")
local arch = opt_str("arch", opt_str("model", opt_str("model-type", "vae_text")))
if arch ~= "vae_text" then
  error("Unknown --arch: " .. tostring(arch) .. " (expected: vae_text)")
end

local out_dir = opt_str("out-dir", "checkpoint/vae_text_trained")
local output_model_dir = opt_str("output-model", opt_str("output_model", ""))
local save_out_dir = ((output_model_dir ~= nil) and (output_model_dir ~= "")) and output_model_dir or out_dir
local RESUME = opt_bool("resume", false)
local epochs = opt_int("epochs", 10)
local lr = opt_num("lr", 1e-4)
local seed = opt_int("seed", opt_int("init-seed", 1337))

local cfg0, err = Mimir.Architectures.default_config(arch)
assert(type(cfg0) == "table", "default_config(" .. tostring(arch) .. ") failed: " .. tostring(err))
---@cast cfg0 table
local cfg = cfg0

cfg = Args.apply_overrides(cfg, opts)

cfg.seq_len = opt_int("seq-len", opt_int("seq_len", cfg.seq_len or 256))
cfg.d_model = opt_int("d-model", opt_int("d_model", cfg.d_model or 256))
cfg.num_layers = opt_int("layers", opt_int("num-layers", opt_int("num_layers", cfg.num_layers or 4)))
cfg.num_heads = opt_int("heads", opt_int("num-heads", opt_int("num_heads", cfg.num_heads or 8)))
cfg.mlp_hidden = opt_int("mlp-hidden", opt_int("mlp_hidden", cfg.mlp_hidden or 1024))
cfg.latent_tokens = opt_int("latent-tokens", opt_int("latent_tokens", cfg.latent_tokens or 32))
cfg.proj_dim = opt_int("proj-dim", opt_int("proj_dim", cfg.proj_dim or 256))
cfg.decoder_causal = opt_bool("decoder-causal", opt_bool("decoder_causal", cfg.decoder_causal ~= false))
cfg.enable_conditional_encoder = opt_bool("enable-conditional-encoder", opt_bool("enable_conditional_encoder", cfg.enable_conditional_encoder ~= false))
cfg.enable_context_heads = opt_bool("enable-context-heads", opt_bool("enable_context_heads", cfg.enable_context_heads ~= false))
cfg.context_semantic_dim = opt_int("context-semantic-dim", opt_int("context_semantic_dim", cfg.context_semantic_dim or 64))
cfg.context_thematic_dim = opt_int("context-thematic-dim", opt_int("context_thematic_dim", cfg.context_thematic_dim or 32))
cfg.context_dialog_dim = opt_int("context-dialog-dim", opt_int("context_dialog_dim", cfg.context_dialog_dim or 64))
cfg.stochastic_latent = opt_bool(
  "stochastic-latent",
  opt_bool(
    "stochastic_latent",
    opt_bool("vae-stochastic-latent", opt_bool("vae_stochastic_latent", cfg.stochastic_latent ~= false))
  )
)

-- Training knobs consumed by the Model::trainStep hook
cfg.align_weight = opt_num("align-weight", opt_num("align_weight", cfg.align_weight or 0.0))
cfg.kl_beta = opt_num("kl-beta", opt_num("kl_beta", cfg.kl_beta or 0.01))
cfg.kl_warmup_steps = opt_int("kl-warmup-steps", opt_int("kl-warmup", opt_int("kl_warmup", cfg.kl_warmup_steps or 0)))
cfg.recon_loss = opt_str("recon-loss", opt_str("recon_loss", cfg.recon_loss or "ce"))
cfg.logvar_clip_min = opt_num("logvar-clip-min", opt_num("logvar_clip_min", cfg.logvar_clip_min or -6.0))
cfg.logvar_clip_max = opt_num("logvar-clip-max", opt_num("logvar_clip_max", cfg.logvar_clip_max or 2.0))
cfg.context_semantic_weight = opt_num("context-semantic-weight", opt_num("context_semantic_weight", cfg.context_semantic_weight or 0.08))
cfg.context_thematic_weight = opt_num("context-thematic-weight", opt_num("context_thematic_weight", cfg.context_thematic_weight or 0.05))
cfg.context_dialog_weight = opt_num("context-dialog-weight", opt_num("context_dialog_weight", cfg.context_dialog_weight or 0.10))
cfg.grad_accum_steps = opt_int("grad-accum-steps", cfg.grad_accum_steps or 1)
cfg.grad_clip_norm = opt_num("grad-clip-norm", cfg.grad_clip_norm or 1.0)
cfg.max_items = opt_int("max-items", cfg.max_items or 0)
cfg.log_every = opt_int("log-every", cfg.log_every or 10)
cfg.warmup_steps = opt_int("lr-warmup-steps", opt_int("warmup-steps", cfg.warmup_steps or 0))
cfg.autosave_every_epochs = opt_int("autosave-every-epochs", opt_int("autosave_every_epochs", cfg.autosave_every_epochs or 1))

cfg.decay_strategy = opt_str("decay-strategy", opt_str("decay_strategy", cfg.decay_strategy or "linear"))
cfg.decay_rate = opt_num("decay-rate", opt_num("decay_rate", cfg.decay_rate or 0.9999))
cfg.decay_steps = opt_int("decay-steps", opt_int("decay_steps", cfg.decay_steps or 100))

Args.apply_validation_config(cfg, opts, {
  validate_every_steps = 0,
  validate_items = 8,
  validate_holdout_frac = 0.01,
})

-- IMPORTANT: utilisé côté C++ pour le shuffle/ordre dataset.
cfg.seed = seed

-- Viz taps (consommés côté C++ si viz active)
cfg.viz_taps_max_frames = opt_int("viz-taps-max-frames", cfg.viz_taps_max_frames or 12)
cfg.viz_taps_max_side = opt_int("viz-taps-max-side", cfg.viz_taps_max_side or 64)
cfg.viz_taps_force_inference = opt_bool("viz-taps-force-inference", opt_bool("viz_taps_force_inference", cfg.viz_taps_force_inference or false))

cfg.optimizer = opt_str("optimizer", cfg.optimizer or "adamw")
cfg.beta1 = opt_num("beta1", cfg.beta1 or 0.9)
cfg.beta2 = opt_num("beta2", cfg.beta2 or 0.999)
cfg.epsilon = opt_num("epsilon", cfg.epsilon or 1e-8)
cfg.weight_decay = opt_num("weight-decay", cfg.weight_decay or 1e-6)
cfg.dtype = opt_str("dtype", os.getenv("MIMIR_DTYPE") or cfg.dtype or "float32")

-- IMPORTANT: base tokenizer commun
local base_tok_path = opt_str("base-tokenizer", BaseTok.default_path())
do
  local ok_bt, err_bt = BaseTok.load_base({
    path = base_tok_path,
    max_vocab = opt_int("max-vocab", cfg.vocab_size or 32000),
    require = true,
  })
  assert(ok_bt == true, "Base tokenizer: " .. tostring(err_bt))
end
cfg.vocab_size = BaseTok.vocab_size()
cfg.tokenizer_integrated = true
cfg.tokenizer_frozen = true
cfg.padding_idx = BaseTok.pad_id and BaseTok.pad_id() or (cfg.padding_idx or 0)

-- Checkpoint dir (utilisé pour interrupt checkpoints côté C++)
cfg.checkpoint_dir = save_out_dir

log("VAEText train")
log(string.format("  arch=%s", arch))
log(string.format("  dataset_root=%s", dataset_root))
log(string.format("  out_dir(resume_from)=%s", out_dir))
log(string.format("  output_model(save_to)=%s", save_out_dir))
log(string.format("  seq_len=%d d_model=%d latent_tokens=%d proj_dim=%d", cfg.seq_len, cfg.d_model, cfg.latent_tokens, cfg.proj_dim))
log(string.format("  layers=%d heads=%d mlp_hidden=%d", cfg.num_layers, cfg.num_heads, cfg.mlp_hidden))
log(string.format("  decoder_causal=%s cond_encoder=%s context_heads=%s", tostring(cfg.decoder_causal), tostring(cfg.enable_conditional_encoder), tostring(cfg.enable_context_heads)))
log(string.format("  context_dims: semantic=%d thematic=%d dialog=%d", cfg.context_semantic_dim, cfg.context_thematic_dim, cfg.context_dialog_dim))
log(string.format("  epochs=%d lr=%g kl_beta=%g", epochs, lr, cfg.kl_beta))
log(string.format("  recon_loss=%s align_weight=%g", tostring(cfg.recon_loss), tonumber(cfg.align_weight or 0.0) or 0.0))
log(string.format("  context_weights: semantic=%g thematic=%g dialog=%g", cfg.context_semantic_weight, cfg.context_thematic_weight, cfg.context_dialog_weight))
log(string.format("  logvar_clip=[%g,%g] grad_accum_steps=%d grad_clip=%g", cfg.logvar_clip_min, cfg.logvar_clip_max, cfg.grad_accum_steps, cfg.grad_clip_norm))
log(string.format("  optimizer=%s beta1=%g beta2=%g wd=%g", tostring(cfg.optimizer), cfg.beta1, cfg.beta2, cfg.weight_decay))
log(string.format("  warmup_steps=%d decay=%s", cfg.warmup_steps or 0, tostring(cfg.decay_strategy)))
log(string.format("  base_tokenizer=%s vocab_size=%d", base_tok_path, cfg.vocab_size))
log(string.format("  validate_every_steps=%d validate_items=%d", cfg.validate_every_steps or 0, cfg.validate_items or 0))
log(string.format("  seed=%d", seed))

-- Dataset
local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
if (tonumber(n_or_err) or 0) == 0 then
  error(string.format("Dataset vide: 0 item chargé depuis '%s'.", dataset_root))
end
log("✓ Dataset chargé: " .. tostring(n_or_err))

-- Modèle
assert_ok(Mimir.Model.create(arch, cfg), nil, "Model.create(" .. tostring(arch) .. ") failed")
apply_dtype(cfg)
local params = Mimir.Model.total_params()
log("✓ Model créé (registry): params=" .. tostring(params))

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
assert_ok(ok_alloc, err_alloc, "Model.allocate_params failed")

local resumed_from = nil
if RESUME and Ckpt and Ckpt.resolve_dir then
  local resolver = Ckpt.resolve_dir_prefer_final or Ckpt.resolve_dir
  local resume_dir = resolver and resolver(out_dir) or nil
  if resume_dir then
    log("↩︎ Resume: chargement checkpoint: " .. tostring(resume_dir))
    local load_opts = {
      load_encoder = true,
      load_tokenizer = true,
      load_optimizer = true,
      strict_mode = false,
      validate_checksums = true
    }
    local ok_load, err_load = Mimir.Serialization.load(resume_dir, "raw_folder", load_opts)
    assert_ok(ok_load, err_load, "Serialization.load(resume) failed")
    resumed_from = resume_dir
  end
end

if not resumed_from then
  local init_method = opt_str("init", "xavier")
  local ok_init, err_init = Mimir.Model.init_weights(init_method, seed)
  assert_ok(ok_init, err_init, "Model.init_weights failed")
end

-- Snapshot debug JSON avant entraînement
do
  local starttrain_path = save_out_dir .. "/starttrain.json"
  FS.mkdir_p(save_out_dir)
  local ok_dbg, err_dbg = Mimir.Serialization.save(starttrain_path, "debug_json", {
    save_tokenizer = true,
    save_encoder = true,
    include_git_info = true,
  })
  assert_ok(ok_dbg, err_dbg, "Serialization.save(starttrain.json, debug_json) failed")
  log("✓ Debug JSON écrit: " .. starttrain_path)
end

-- Entraînement
local ok_train, err_train = Mimir.Model.train(epochs, lr)
if ok_train == false and tostring(err_train) == "STOP_REQUESTED" then
  log("⛔ Stop demandé via Viz: sauvegarde finale puis fin du programme.")

  local last_dir = nil
  if Ckpt and Ckpt.find_latest_epoch_dir then
    last_dir = Ckpt.find_latest_epoch_dir(save_out_dir)
  end
  if not last_dir and Ckpt and Ckpt.resolve_dir then
    last_dir = Ckpt.resolve_dir(save_out_dir)
  end
  if not last_dir then last_dir = save_out_dir end

  FS.mkdir_p(last_dir)
  local ok_save_stop, err_save_stop = Mimir.Serialization.save(last_dir, "raw_folder", {
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
  assert_ok(ok_save_stop, err_save_stop, "Serialization.save(stop) failed")
  log("✓ Checkpoint STOP écrit: " .. tostring(last_dir))
  return
end
assert_ok(ok_train, err_train, "Model.train failed")

-- Sauvegarde
FS.mkdir_p(save_out_dir)
local ok_save, err_save = Mimir.Serialization.save(save_out_dir, "raw_folder", {
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
assert_ok(ok_save, err_save, "Serialization.save failed")

log("✓ Checkpoint VAEText écrit: " .. save_out_dir)
