---@diagnostic disable: undefined-global, undefined-field, inject-field
local Args = dofile("scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local Ckpt = dofile("scripts/modules/checkpoint_resume.lua")

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

local function assert_ok(ok, err, msg)
  if ok == false then
    error((msg or "Operation failed") .. ": " .. tostring(err))
  end
end

-- Objectif: pré-entraîner `vgg16_feat` pour pouvoir l'utiliser en `--perceptual-ckpt`.
-- Note perf: par défaut on prétrain en 256x256 (rapide) puis on réutilise les poids
-- en 512x512 (les convs ont la même forme, les reshape/pool n'ont pas de poids).

local MEM_GB = opt_num("mem-gb", 10)
local ALLOC_GB = opt_num("alloc-gb", MEM_GB)
local ENABLE_COMPRESSION = opt_bool("compression", opt_bool("compress", true))

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({max_ram_gb = ALLOC_GB, enable_compression = ENABLE_COMPRESSION})
end
if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, MEM_GB)
end
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  pcall(Mimir.Model.set_hardware, opt_bool("hw", true))
end

local dataset_root = opt_str("dataset-root", "dataset_2")
local out_dir = opt_str("out-dir", "checkpoint/vgg16_feat_pretrain")
local RESUME = opt_bool("resume", false)
local epochs = opt_int("epochs", 10)
local lr = opt_num("lr", 1e-4)
local seed = opt_int("seed", opt_int("init-seed", 1337))

local image_w = opt_int("image-w", 256)
local image_h = opt_int("image-h", 256)
local image_c = opt_int("image-c", 3)

-- `base_channels` du vgg16_feat = `perceptual_base_channels` côté VAEConv.
-- NOTE: l'archi `vgg16_feat` force `base_channels >= 4` côté C++.
local base_channels = opt_int("base-channels", opt_int("perceptual-base-channels", 2))
if base_channels < 4 then base_channels = 4 end

local autosave_every_epochs = opt_int("autosave-every-epochs", opt_int("autosave", 1))
local max_items = opt_int("max-items", 0)
local log_every = opt_int("log-every", 10)
local pretrain_grid = opt_int("pretrain-grid", 8)

-- Viz/monitoring (consommés par LuaScripting.cpp)
-- NOTE: l'activation de la viz se fait via les flags `--viz` / `--htop` (Args.parse).
local viz_taps_every_steps = opt_int("viz-taps-every-steps", opt_int("viz-every-steps", 0))
local viz_taps_max_frames = opt_int("viz-taps-max-frames", opt_int("viz-max-frames", 0))
local viz_taps_max_side = opt_int("viz-taps-max-side", opt_int("viz-max-side", 0))

-- Optimizer knobs (consommés par LuaScripting::lua_trainModel)
local optimizer = opt_str("optimizer", "adamw")
local beta1 = opt_num("beta1", 0.9)
local beta2 = opt_num("beta2", 0.999)
local epsilon = opt_num("epsilon", 1e-8)
local weight_decay = opt_num("weight-decay", 1e-6)
local decay_strategy = opt_str("decay-strategy", "cosine")
local warmup_steps = opt_int("warmup-steps", opt_int("lr-warmup-steps", 0))

os.execute("mkdir -p '" .. out_dir:gsub("'", "'\\''") .. "' 2>/dev/null")

log("=== Pretrain vgg16_feat ===")
log(string.format("- dataset_root=%s", dataset_root))
log(string.format("- out_dir=%s", out_dir))
log(string.format("- image=%dx%dx%d", image_w, image_h, image_c))
log(string.format("- base_channels=%d (output_dim=%d)", base_channels, 23 * base_channels))
log(string.format("- epochs=%d lr=%.6g seed=%d", epochs, lr, seed))
log(string.format("- autosave_every_epochs=%d max_items=%d log_every=%d", autosave_every_epochs, max_items, log_every))
log(string.format("- pretrain_grid=%d", pretrain_grid))
log(string.format("- optimizer=%s wd=%.3g decay=%s warmup_steps=%d", optimizer, weight_decay, decay_strategy, warmup_steps))

-- Dataset
local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root, image_w, image_h, 1, true, 'dataset_cache.json', 10240, true)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
log("✓ Dataset chargé: " .. tostring(n_or_err))

-- Modèle
local cfg0, err = Mimir.Architectures.default_config("vgg16_feat")
assert(type(cfg0) == "table", "default_config(vgg16_feat) failed: " .. tostring(err))
local cfg = cfg0
cfg.image_w = image_w
cfg.image_h = image_h
cfg.image_c = image_c
cfg.base_channels = base_channels

-- Knobs consommés par la boucle C++ (lua_trainModel branche vgg16_feat)
cfg.checkpoint_dir = out_dir
cfg.autosave_every_epochs = autosave_every_epochs
cfg.max_items = max_items
cfg.log_every = log_every
cfg.seed = seed
cfg.pretrain_grid = pretrain_grid

if viz_taps_every_steps and viz_taps_every_steps > 0 then cfg.viz_taps_every_steps = 1 end
if viz_taps_max_frames and viz_taps_max_frames > 0 then cfg.viz_taps_max_frames = 60 end
if viz_taps_max_side and viz_taps_max_side > 0 then cfg.viz_taps_max_side = 512 end

cfg.optimizer = optimizer
cfg.beta1 = beta1
cfg.beta2 = beta2
cfg.epsilon = epsilon
cfg.weight_decay = weight_decay
cfg.decay_strategy = decay_strategy
cfg.warmup_steps = warmup_steps

assert_ok(Mimir.Model.create("vgg16_feat", cfg), nil, "Model.create(vgg16_feat) failed")
local params = Mimir.Model.total_params()
log("✓ Model créé (registry): params=" .. tostring(params))

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
assert_ok(ok_alloc, err_alloc, "Model.allocate_params failed")

local resumed_from = nil
if RESUME and Ckpt and Ckpt.resolve_dir then
  local resume_dir = Ckpt.resolve_dir(out_dir)
  if resume_dir then
    log("↩︎ Resume: chargement checkpoint: " .. tostring(resume_dir))
    local load_opts = {
      load_optimizer = true,
      strict_mode = false,
      validate_checksums = true,
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

-- Snapshot debug avant train
do
  local starttrain_path = out_dir .. "/starttrain.json"
  local ok_dbg, err_dbg = Mimir.Serialization.save(starttrain_path, "debug_json", {
    include_git_info = true,
  })
  assert_ok(ok_dbg, err_dbg, "Serialization.save(starttrain.json, debug_json) failed")
  log("✓ Debug JSON écrit: " .. starttrain_path)
end

-- Entraînement
local ok_train, err_train = Mimir.Model.train(epochs, lr)
if ok_train == false and tostring(err_train) == "STOP_REQUESTED" then
  log("⛔ Stop demandé via Viz. Sauvegarde finale puis fin du programme.")
  local last_dir = nil
  if Ckpt and Ckpt.find_latest_epoch_dir then
    last_dir = Ckpt.find_latest_epoch_dir(out_dir)
  end
  if not last_dir and Ckpt and Ckpt.resolve_dir then
    last_dir = Ckpt.resolve_dir(out_dir)
  end
  if not last_dir then
    last_dir = out_dir
  end

  os.execute("mkdir -p '" .. tostring(last_dir):gsub("'", "'\\''") .. "' 2>/dev/null")
  local ok_save_stop, err_save_stop = Mimir.Serialization.save(last_dir, "raw_folder", {
    save_optimizer = true,
    include_checksums = true,
    include_git_info = true,
    include_gradients = true,
    include_optimizer_state = true,
  })
  assert_ok(ok_save_stop, err_save_stop, "Serialization.save(stop) failed")
  log("✓ Checkpoint STOP écrit: " .. tostring(last_dir))
  return
end
assert_ok(ok_train, err_train, "Model.train failed")

-- Sauvegarde finale
do
  local final_dir = out_dir .. "/final"
  os.execute("mkdir -p '" .. final_dir:gsub("'", "'\\''") .. "' 2>/dev/null")
  local ok_save, err_save = Mimir.Serialization.save(final_dir, "raw_folder", {
    save_optimizer = true,
    include_checksums = true,
    include_git_info = true,
  })
  assert_ok(ok_save, err_save, "Serialization.save(final) failed")
  log("✓ Checkpoint final écrit: " .. final_dir)
end

log("✓ Pretrain terminé")
