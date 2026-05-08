-- Train VGG16/VGG19 en classification multi-label à partir de tags dot-séparés.
--
-- Dataset attendu: pour chaque image `a.png`, un texte `a.txt` contenant des tags/phrases courtes,
-- séparés par des points (.) ex: "tag1. tag2. une courte phrase."
--
-- IMPORTANT: les classes doivent être fixes -> fournir un vocabulaire de tags (une entrée par ligne)
-- via `--tags-vocab`.
--
-- Usage:
--   ./bin/mimir --lua scripts/training/train_vgg16_tags_multilabel.lua -- \
--     --arch vgg16 \
--     --dataset-root dataset_2 \
--     --tags-vocab checkpoint/tags_vocab.txt \
--     --out-dir checkpoint/vgg16_tags \
--     --image-w 256 --image-h 256 \
--     --base-channels 32 --fc-hidden 512 \
--     --epochs 10 --lr 1e-4 \
--     --log-every 10 --autosave-every-epochs 1
--
-- Viz (optionnel):
--   --viz --viz-taps-every-steps 50 --viz-taps-max-frames 12 --viz-taps-max-side 96
--
-- TUI (optionnel):
--   --htop

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

local function shell_quote(s)
  if s == nil then return "''" end
  s = tostring(s)
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function file_exists(path)
  if not path or #tostring(path) == 0 then return false end
  local cmd = "test -f " .. shell_quote(path) .. " >/dev/null 2>&1"
  local ok, why, code = os.execute(cmd)
  if type(ok) == "number" then return ok == 0 end
  if type(ok) == "boolean" then return ok end
  return (why == "exit" and code == 0)
end

local function read_lines(path)
  local f = io.open(path, "r")
  if not f then return nil, "cannot open file: " .. tostring(path) end
  local out = {}
  for line in f:lines() do
    local s = tostring(line)
    -- trim
    s = s:gsub("^%s+", ""):gsub("%s+$", "")
    if #s > 0 and s:sub(1, 1) ~= "#" then
      table.insert(out, s)
    end
  end
  f:close()
  return out, nil
end

-- Alloc/memory
local MEM_GB = opt_num("mem-gb", 12)
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

-- Config
local arch = opt_str("arch", "vgg16")
if arch ~= "vgg16" and arch ~= "vgg19" then
  error("--arch doit être vgg16 ou vgg19")
end

local dataset_root = opt_str("dataset-root", "dataset_2")
local tags_vocab_path = opt_str("tags-vocab", "")
local out_dir = opt_str("out-dir", "checkpoint/vgg16_tags")

local RESUME = opt_bool("resume", false)
local epochs = opt_int("epochs", 10)
local lr = opt_num("lr", 1e-4)
local seed = opt_int("seed", opt_int("init-seed", 1337))

local image_w = opt_int("image-w", 512)
local image_h = opt_int("image-h", 512)
local image_c = opt_int("image-c", 3)

local base_channels = opt_int("base-channels", 64)
local fc_hidden = opt_int("fc-hidden", 512)

local autosave_every_epochs = opt_int("autosave-every-epochs", opt_int("autosave", 1))
local max_items = opt_int("max-items", 0)
local log_every = opt_int("log-every", 1)

local lowercase_tags = opt_bool("lowercase-tags", true)

-- Validation (anti-collapse)
local validate_items = opt_int("validate-items", opt_int("val-items", 32))
local validate_every_steps = opt_int("validate-every-steps", opt_int("validate-every", 10))
local validate_every_epochs = opt_int("validate-every-epochs", opt_int("validate-epochs", 1))
local validate_threshold = opt_num("validate-threshold", opt_num("val-threshold", 0.5))
local validate_holdout = opt_bool("validate-holdout", opt_bool("val-holdout", true))
local validate_holdout_frac = opt_num("validate-holdout-frac", opt_num("val-holdout-frac", 0.0))
local validate_holdout_items = opt_int("validate-holdout-items", opt_int("val-holdout-items", 4))
local validate_seed = opt_int("validate-seed", opt_int("val-seed", 4242))

-- BCE pos weight (utile si classes très rares -> évite un collapse "tout à 0")
local bce_pos_weight = opt_num("pos-weight", opt_num("bce-pos-weight", 1.0))

-- Viz taps (consommés côté C++)
local viz_taps_every_steps = opt_int("viz-taps-every-steps", opt_int("viz-every-steps", 0))
local viz_taps_max_frames = opt_int("viz-taps-max-frames", 12)
local viz_taps_max_side = opt_int("viz-taps-max-side", 64)

os.execute("mkdir -p " .. shell_quote(out_dir) .. " 2>/dev/null")

-- Vocab de tags/classes
if tags_vocab_path == "" then
  error("--tags-vocab est requis (liste fixe des tags/classes, 1 par ligne)")
end
if not file_exists(tags_vocab_path) then
  error("tags vocab introuvable: " .. tostring(tags_vocab_path))
end

local tags_vocab, err_vocab = read_lines(tags_vocab_path)
if not tags_vocab then
  error("lecture tags vocab échouée: " .. tostring(err_vocab))
end
if #tags_vocab < 2 then
  error("tags vocab trop petit (>=2 recommandé): " .. tostring(tags_vocab_path))
end

log("=== Train " .. arch .. " multi-label tags ===")
log(string.format("- dataset_root=%s", dataset_root))
log(string.format("- tags_vocab=%s (classes=%d)", tags_vocab_path, #tags_vocab))
log(string.format("- out_dir=%s", out_dir))
log(string.format("- image=%dx%dx%d", image_w, image_h, image_c))
log(string.format("- base_channels=%d fc_hidden=%d", base_channels, fc_hidden))
log(string.format("- epochs=%d lr=%.6g seed=%d", epochs, lr, seed))
log(string.format("- autosave_every_epochs=%d max_items=%d log_every=%d", autosave_every_epochs, max_items, log_every))
log(string.format("- lowercase_tags=%s", tostring(lowercase_tags)))
log(string.format("- val: every_epochs=%d every_steps=%d items=%d threshold=%.3f holdout=%s (frac=%.3f items=%d) pos_weight=%.3f",
  validate_every_epochs, validate_every_steps, validate_items, validate_threshold,
  tostring(validate_holdout), validate_holdout_frac, validate_holdout_items, bce_pos_weight))

-- Dataset (doit charger les textes)
local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root, image_w, image_h, 1, true, 'dataset_cache.json', 10240, true)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
log("✓ Dataset chargé: " .. tostring(n_or_err))

-- Modèle
local cfg0, err = Mimir.Architectures.default_config(arch)
assert(type(cfg0) == "table", "default_config(" .. arch .. ") failed: " .. tostring(err))
local cfg = cfg0
cfg.image_w = image_w
cfg.image_h = image_h
cfg.image_c = image_c
cfg.base_channels = base_channels
cfg.fc_hidden = fc_hidden
cfg.num_classes = #tags_vocab

-- Training knobs consommés par LuaScripting::lua_trainModel (branche vgg16/vgg19)
cfg.checkpoint_dir = out_dir
cfg.autosave_every_epochs = autosave_every_epochs
cfg.max_items = max_items
cfg.log_every = log_every
cfg.seed = seed

cfg.lowercase_tags = lowercase_tags
cfg.tags_vocab = tags_vocab

-- Validation knobs (consommés côté C++)
cfg.validate_items = validate_items
cfg.validate_every_steps = validate_every_steps
cfg.validate_every_epochs = validate_every_epochs
cfg.validate_threshold = validate_threshold
cfg.validate_holdout = validate_holdout
cfg.validate_holdout_frac = validate_holdout_frac
cfg.validate_holdout_items = validate_holdout_items
cfg.validate_seed = validate_seed

cfg.bce_pos_weight = bce_pos_weight

if viz_taps_every_steps > 0 then cfg.viz_taps_every_steps = viz_taps_every_steps end
if viz_taps_max_frames and viz_taps_max_frames > 0 then cfg.viz_taps_max_frames = viz_taps_max_frames end
if viz_taps_max_side and viz_taps_max_side > 0 then cfg.viz_taps_max_side = viz_taps_max_side end

assert_ok(Mimir.Model.create(arch, cfg), nil, "Model.create(" .. arch .. ") failed")
log("✓ Model créé (registry): params=" .. tostring(Mimir.Model.total_params()))

assert_ok(Mimir.Model.allocate_params(), nil, "Model.allocate_params failed")

-- Resume (optionnel)
local resumed_from = nil
if RESUME and Ckpt and Ckpt.resolve_dir then
  local resume_dir = Ckpt.resolve_dir(out_dir)
  if resume_dir then
    log("↩︎ Resume: chargement checkpoint: " .. tostring(resume_dir))
    local ok_load, err_load = Mimir.Serialization.load(resume_dir, "raw_folder", {
      load_optimizer = true,
      strict_mode = false,
      validate_checksums = true,
    })
    assert_ok(ok_load, err_load, "Serialization.load(resume) failed")
    resumed_from = resume_dir
  end
end

if not resumed_from then
  local init_method = opt_str("init", "xavier")
  local ok_init, err_init = Mimir.Model.init_weights(init_method, seed)
  assert_ok(ok_init, err_init, "Model.init_weights failed")
end

-- Train
local ok_train, err_train = Mimir.Model.train(epochs, lr)
if ok_train == false and tostring(err_train) == "STOP_REQUESTED" then
  log("⛔ Stop demandé via Viz. Fin du programme.")
  return
end
assert_ok(ok_train, err_train, "Model.train failed")

-- Sauvegarde finale
local final_dir = out_dir .. "/final"
os.execute("mkdir -p " .. shell_quote(final_dir) .. " 2>/dev/null")
local ok_save, err_save = Mimir.Serialization.save(final_dir, "raw_folder", {
  save_optimizer = true,
  include_checksums = true,
  include_git_info = true,
})
assert_ok(ok_save, err_save, "Serialization.save(final) failed")
log("✓ Checkpoint final écrit: " .. final_dir)

log("✓ Train terminé")
