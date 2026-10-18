#!/usr/bin/env lua
---@diagnostic disable: undefined-global, undefined-field, need-check-nil

local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help({
  script_path = "scripts/training/train_medical_detector.lua",
  description = "Entraîne le détecteur d'anomalies médicales natif Mímir.",
  options = {
    "--dataset-root PATH : dossier récursif d'images et sidecars JSON (requis)",
    "--out-dir PATH : checkpoint raw_folder (checkpoint/medical_detector)",
    "--epochs N, --lr N, --seed N : paramètres d'entraînement",
    "--image-w N, --image-h N : résolution d'entrée (256x256)",
    "--base-channels N : largeur du CNN (16)",
    "--class-names LISTE : classes séparées par des virgules (taxonomie mammaire à 7 classes)",
    "--validation-fraction N, --validation-items N : holdout de validation",
    "--score-threshold N, --nms-iou-threshold N, --match-iou-threshold N",
    "--resume : reprend le dernier checkpoint trouvé dans --out-dir",
    "--viz : active le moniteur Viz avec les cadres prédits",
  },
  examples = {
    "./bin/mimir --lua scripts/training/train_medical_detector.lua -- --dataset-root dataset/mammographies --epochs 30 --viz",
  },
})

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
local opts = Args.parse(arg) or {}

local function die(message) error("[medical_detector] "..tostring(message), 0) end
local function check(ok, err, operation)
  if not ok then die(operation..": "..tostring(err or "échec")) end
end
local function finite(value, name)
  value = tonumber(value)
  if not value or value ~= value or math.abs(value) == math.huge then die(name.." invalide") end
  return value
end
local function split_names(value)
  local names = {}
  for name in tostring(value):gmatch("[^,]+") do
    name = name:match("^%s*(.-)%s*$")
    if name ~= "" then names[#names + 1] = name end
  end
  return names
end
local DEFAULT_CLASS_NAMES = table.concat({
  "masse",
  "calcification",
  "asymétrie",
  "distorsion architecturale",
  "ganglion lymphatique",
  "épaississement cutané",
  "rétraction du mamelon",
}, ",")

local dataset_root = Args.get_str(opts, "dataset-root", nil)
if not dataset_root or dataset_root == "" then die("--dataset-root est requis") end
local out_dir = Args.get_str(opts, "out-dir", "checkpoint/medical_detector")
local epochs = Args.get_int(opts, "epochs", 30)
local lr = finite(Args.get_num(opts, "lr", 1e-4), "--lr")
local seed = Args.get_int(opts, "seed", 1337)
local resume = Args.get_bool(opts, "resume", false)
local class_names = split_names(Args.get_str(opts, "class-names", DEFAULT_CLASS_NAMES))
if #class_names == 0 then die("--class-names doit contenir au moins une classe") end
if epochs < 1 or lr <= 0 then die("--epochs et --lr doivent être positifs") end

local resume_dir = nil
if resume then
  resume_dir = Ckpt.resolve_dir_prefer_final and Ckpt.resolve_dir_prefer_final(out_dir)
    or Ckpt.resolve_dir(out_dir)
  if not resume_dir then die("aucun checkpoint trouvé dans "..out_dir) end
end

local cfg, cfg_err = Mimir.Architectures.default_config("medical_detector")
if type(cfg) ~= "table" then die("architecture medical_detector indisponible: "..tostring(cfg_err)) end
if resume_dir then
  local architecture = read_json(FS.join(resume_dir, "model", "architecture.json"))
  local saved = type(architecture) == "table" and
    (architecture.model_config or architecture.modelConfig) or nil
  if type(saved) ~= "table" or saved.type ~= "medical_detector" then
    die("checkpoint medical_detector invalide: "..resume_dir)
  end
  ---@cast saved table
  for key, value in pairs(saved) do cfg[key] = value end
  if opts["class-names"] == nil and type(saved.class_names) == "table" then
    class_names = saved.class_names
  end
end

cfg.image_w = Args.get_int(opts, "image-w", cfg.image_w or 256)
cfg.image_h = Args.get_int(opts, "image-h", cfg.image_h or 256)
cfg.image_c = 3
cfg.base_channels = Args.get_int(opts, "base-channels", cfg.base_channels or 16)
if not resume_dir then cfg.num_classes = #class_names end
if cfg.num_classes ~= #class_names then
  die(string.format("%d noms de classes fournis, mais le modèle en attend %d", #class_names, cfg.num_classes))
end
cfg.class_names = class_names
for role, default in pairs({primary=3.0, secondary=2.0, other=1.0}) do
  local key = "relation_" .. role .. "_weight"
  cfg[key] = finite(Args.get_num(opts, "relation-" .. role .. "-weight", cfg[key] or default), key)
  if cfg[key] < 0 then die(key .. " doit être >= 0") end
end
cfg.object_loss_weight = Args.get_num(opts, "object-loss-weight", cfg.object_loss_weight or 1.0)
cfg.box_loss_weight = Args.get_num(opts, "box-loss-weight", cfg.box_loss_weight or 5.0)
cfg.class_loss_weight = Args.get_num(opts, "class-loss-weight", cfg.class_loss_weight or 1.0)
cfg.object_pos_weight = Args.get_num(opts, "object-pos-weight", cfg.object_pos_weight or 8.0)
cfg.validation_fraction = Args.get_num(opts, "validation-fraction", 0.2)
cfg.validation_items = Args.get_int(opts, "validation-items", 32)
cfg.validate_every_epochs = Args.get_int(opts, "validate-every-epochs", 1)
cfg.score_threshold = Args.get_num(opts, "score-threshold", 0.25)
cfg.nms_iou_threshold = Args.get_num(opts, "nms-iou-threshold", 0.45)
cfg.match_iou_threshold = Args.get_num(opts, "match-iou-threshold", 0.30)
cfg.max_detections = Args.get_int(opts, "max-detections", 100)
cfg.optimizer = Args.get_str(opts, "optimizer", "adamw")
cfg.beta1 = Args.get_num(opts, "beta1", 0.9)
cfg.beta2 = Args.get_num(opts, "beta2", 0.999)
cfg.epsilon = Args.get_num(opts, "epsilon", 1e-8)
cfg.weight_decay = Args.get_num(opts, "weight-decay", 1e-6)
cfg.grad_clip_norm = Args.get_num(opts, "grad-clip-norm", 1.0)
cfg.log_every = Args.get_int(opts, "log-every", 10)
cfg.viz_every_steps = Args.get_int(opts, "viz-every-steps", cfg.log_every)
cfg.max_items = Args.get_int(opts, "max-items", 0)
cfg.autosave_every_epochs = Args.get_int(opts, "autosave-every-epochs", 1)
cfg.checkpoint_dir = out_dir
cfg.seed = seed
cfg.dtype = Args.get_str(opts, "dtype", cfg.dtype or "float32")
cfg = Args.apply_overrides(cfg, opts)
local dtype_aliases = {F32="float32", F16="float16", BF16="bfloat16", FP32="float32", FP16="float16"}
cfg.dtype = dtype_aliases[tostring(cfg.dtype):upper()] or cfg.dtype

for _, key in ipairs({"image_w", "image_h", "image_c", "base_channels", "num_classes"}) do
  if type(cfg[key]) ~= "number" or cfg[key] < 1 or cfg[key] ~= math.floor(cfg[key]) then
    die("configuration invalide: "..key)
  end
end

local mem_gb = Args.get_num(opts, "mem-gb", 8)
check(Mimir.Allocator.configure({max_ram_gb=mem_gb, enable_compression=true}), nil,
  "Allocator.configure")
pcall(Mimir.MemoryGuard.setLimit, mem_gb)
pcall(Mimir.Model.set_hardware, not Args.get_bool(opts, "cpu-only", false))

if Args.get_bool(opts, "viz", false) then
  local ok_viz, viz_err = Mimir.Htop.create({
    enable_viz=true, enable_htop=true,
    csv_path=FS.join(out_dir, "metrics.csv"),
  })
  check(ok_viz, viz_err, "Htop.create")
end

local ok_dataset, dataset_count = Mimir.Dataset.load(
  dataset_root, cfg.image_w, cfg.image_h, 1, false, "", math.floor(mem_gb * 1024), true)
check(ok_dataset, dataset_count, "Dataset.load")
if (tonumber(dataset_count) or 0) == 0 then die("dataset vide: "..dataset_root) end

local ok_create, create_err = Mimir.Model.create("medical_detector", cfg)
check(ok_create, create_err, "Model.create")
if type(Mimir.Model.dtype) == "function" then
  local ok_dtype, dtype_err = Mimir.Model.dtype(cfg.dtype)
  check(ok_dtype, dtype_err, "Model.dtype")
end
local ok_alloc, alloc_err = Mimir.Model.allocate_params()
check(ok_alloc, alloc_err, "Model.allocate_params")

if resume_dir then
  local ok_load, load_err = Mimir.Serialization.load(resume_dir, "raw_folder", {
    load_tokenizer=false, load_encoder=false, load_optimizer=true,
    strict_mode=true, validate_checksums=true,
  })
  check(ok_load, load_err, "Serialization.load")
  print("Reprise: "..resume_dir)
else
  local ok_init, init_err = Mimir.Model.init_weights(Args.get_str(opts, "init", "xavier"), seed)
  check(ok_init, init_err, "Model.init_weights")
end

print(string.format("medical_detector: images=%s train=%dx%d classes=%d epochs=%d lr=%g",
  tostring(dataset_count), cfg.image_w, cfg.image_h, cfg.num_classes, epochs, lr))
local ok_train, train_err = Mimir.Model.train(epochs, lr)
if ok_train == false and tostring(train_err) ~= "STOP_REQUESTED" then
  die("Model.train: "..tostring(train_err))
end

local final_dir = FS.join(out_dir, "final")
FS.mkdir_p(final_dir)
local ok_save, save_err = Mimir.Serialization.save(final_dir, "raw_folder", {
  save_optimizer=true, save_tokenizer=false, save_encoder=false,
  include_checksums=true, include_git_info=true, include_optimizer_state=true,
})
check(ok_save, save_err, "Serialization.save")
print("Checkpoint écrit: "..final_dir)
