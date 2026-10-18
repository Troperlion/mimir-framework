-- Entraînement d'un modèle de reconnaissance faciale avec Mimir (vgg16/vgg19).
--
-- Objectif:
-- - apprendre une classification d'identité (1 personne = 1 classe)
-- - en utilisant l'entraînement natif Mimir (branche vgg16/vgg19 multi-label)
--
-- Dataset attendu:
-- - pour chaque image visage: un fichier .txt associé (même stem)
-- - le label d'identité est pris depuis la 1ère ligne non vide du .txt
--   (si la ligne contient des points, seul le segment avant le 1er point est gardé)
--
-- Exemple:
--   ./bin/mimir --lua scripts/training/train_face_recognition_mimir.lua -- \
--     --dataset-root dataset_faces \
--     --out-dir checkpoint/face_id_vgg16 \
--     --image-w 160 --image-h 160 \
--     --epochs 20 --lr 1e-4
--
-- Inférence (image visage recadrée):
--   ./bin/mimir --lua scripts/examples/classify_vgg16_feat.lua -- \
--     --ckpt-dir checkpoint/face_id_vgg16/final \
--     --labels-file checkpoint/face_id_vgg16/labels.txt \
--     --image /tmp/face_crop.png \
--     --topk 5

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

local function assert_ok(ok, err, msg)
  if ok == false then
    error((msg or "Operation failed") .. ": " .. tostring(err))
  end
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  assert_ok(ok, dt_or_err, "Mimir.model.dtype(" .. tostring(dtype) .. ") failed")
  return true
end

local function trim(s)
  s = tostring(s or "")
  s = s:gsub("^%s+", "")
  s = s:gsub("%s+$", "")
  return s
end

local function first_identity_label(txt)
  if txt == nil then return nil end
  local s = tostring(txt):gsub("\r", "\n")
  for line in s:gmatch("[^\n]+") do
    local t = trim(line)
    if #t > 0 then
      local a = t:match("^([^%.]+)")
      a = trim(a or t)
      if #a > 0 then
        return a
      end
      return nil
    end
  end
  return nil
end

local function read_first_non_empty_line(path)
  path = tostring(path or "")
  if path == "" then return nil end
  local f = io.open(path, "r")
  if not f then return nil end
  for line in f:lines() do
    local t = trim(line)
    if #t > 0 then
      f:close()
      return t
    end
  end
  f:close()
  return nil
end

local function write_labels_file(path, labels)
  local f, err = io.open(path, "w")
  if not f then
    return false, err
  end
  for i = 1, #labels do
    f:write(tostring(labels[i]), "\n")
  end
  f:close()
  return true, nil
end

local function collect_labels_from_dataset(n_items)
  local uniq = {}
  local labels = {}
  local labeled_items = 0

  for i = 1, n_items do
    local it, err = Mimir.Dataset.get(i)
    if it ~= nil and type(it) == "table" then
      local lbl = first_identity_label(it.text)
      if (lbl == nil or #lbl == 0) and type(it.text_file) == "string" and #it.text_file > 0 then
        local line = read_first_non_empty_line(it.text_file)
        lbl = first_identity_label(line)
      end
      if lbl ~= nil and #lbl > 0 then
        labeled_items = labeled_items + 1
        if not uniq[lbl] then
          uniq[lbl] = true
          labels[#labels + 1] = lbl
        end
      end
    end
  end

  table.sort(labels)
  return labels, labeled_items
end

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

local arch = opt_str("arch", "vgg16")
if arch ~= "vgg16" and arch ~= "vgg19" then
  error("--arch doit être vgg16 ou vgg19")
end

local dataset_root = opt_str("dataset-root", "dataset_faces")
local out_dir = opt_str("out-dir", "checkpoint/face_id_vgg16")
local labels_out = opt_str("labels-out", FS.join(out_dir, "labels.txt"))

local RESUME = opt_bool("resume", false)
local epochs = opt_int("epochs", 20)
local lr = opt_num("lr", 1e-4)
local seed = opt_int("seed", opt_int("init-seed", 1337))

local image_w = opt_int("image-w", 160)
local image_h = opt_int("image-h", 160)
local image_c = opt_int("image-c", 3)
local base_channels = opt_int("base-channels", 64)
local fc_hidden = opt_int("fc-hidden", 512)

local autosave_every_epochs = opt_int("autosave-every-epochs", opt_int("autosave", 1))
local max_items = opt_int("max-items", 0)
local log_every = opt_int("log-every", 10)

local validate_items = opt_int("validate-items", 64)
local validate_every_steps = opt_int("validate-every-steps", 50)
local validate_every_epochs = opt_int("validate-every-epochs", 1)
local validate_threshold = opt_num("validate-threshold", 0.5)
local validate_holdout = opt_bool("validate-holdout", true)
local validate_holdout_frac = opt_num("validate-holdout-frac", 0.1)
local validate_holdout_items = opt_int("validate-holdout-items", 64)
local validate_seed = opt_int("validate-seed", 4242)

FS.mkdir_p(out_dir)

log("=== Train Face Recognition (Mimir) ===")
log(string.format("- arch=%s", arch))
log(string.format("- dataset_root=%s", dataset_root))
log(string.format("- out_dir=%s", out_dir))
log(string.format("- labels_out=%s", labels_out))
log(string.format("- image=%dx%dx%d", image_w, image_h, image_c))
log(string.format("- epochs=%d lr=%.6g seed=%d", epochs, lr, seed))

local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root, image_w, image_h, 2, true, "dataset_cache.json", 10240, true)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
local n_items = tonumber(n_or_err) or 0
if n_items <= 0 then
  error("Dataset vide ou inaccessible. Attendu: image + txt par item.")
end
log(string.format("✓ Dataset chargé: %d items", n_items))

local labels, labeled_items = collect_labels_from_dataset(n_items)
if #labels < 2 then
  error("Nombre de classes insuffisant (<2). Vérifie les .txt (1ère ligne = identité).")
end
if labeled_items <= 0 then
  error("Aucun item labelisé détecté via les .txt.")
end
log(string.format("✓ Classes identifiées: %d (items labelisés=%d)", #labels, labeled_items))

local ok_lbl, err_lbl = write_labels_file(labels_out, labels)
assert_ok(ok_lbl, err_lbl, "Écriture labels_out échouée")
log("✓ Fichier labels écrit: " .. labels_out)

local cfg0, err = Mimir.Architectures.default_config(arch)
assert(type(cfg0) == "table", "default_config(" .. arch .. ") failed: " .. tostring(err))

local cfg = cfg0
cfg.image_w = image_w
cfg.image_h = image_h
cfg.image_c = image_c
cfg.base_channels = base_channels
cfg.fc_hidden = fc_hidden
cfg.num_classes = #labels

cfg.checkpoint_dir = out_dir
cfg.autosave_every_epochs = autosave_every_epochs
cfg.max_items = max_items
cfg.log_every = log_every
cfg.seed = seed

cfg.lowercase_tags = false
cfg.tags_vocab = labels

cfg.validate_items = validate_items
cfg.validate_every_steps = validate_every_steps
cfg.validate_every_epochs = validate_every_epochs
cfg.validate_threshold = validate_threshold
cfg.validate_holdout = validate_holdout
cfg.validate_holdout_frac = validate_holdout_frac
cfg.validate_holdout_items = validate_holdout_items
cfg.validate_seed = validate_seed

assert_ok(Mimir.Model.create(arch, cfg), nil, "Model.create(" .. arch .. ") failed")
apply_dtype(cfg)
log("✓ Model créé (registry): params=" .. tostring(Mimir.Model.total_params()))

assert_ok(Mimir.Model.allocate_params(), nil, "Model.allocate_params failed")

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

local ok_train, err_train = Mimir.Model.train(epochs, lr)
if ok_train == false and tostring(err_train) == "STOP_REQUESTED" then
  log("⛔ Stop demandé via Viz. Fin du programme.")
  return
end
assert_ok(ok_train, err_train, "Model.train failed")

local final_dir = FS.join(out_dir, "final")
FS.mkdir_p(final_dir)
local ok_save, err_save = Mimir.Serialization.save(final_dir, "raw_folder", {
  save_optimizer = true,
  include_checksums = true,
  include_git_info = true,
})
assert_ok(ok_save, err_save, "Serialization.save(final) failed")
log("✓ Checkpoint final écrit: " .. final_dir)

log("✓ Train Face Recognition terminé")
