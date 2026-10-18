#!/usr/bin/env lua
---@diagnostic disable: undefined-global, undefined-field, need-check-nil

local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help({
  script_path = "scripts/inferences/detect_medical_anomalies.lua",
  description = "Détecte et encadre les anomalies médicales avec un checkpoint Mímir natif.",
  options = {
    "--checkpoint PATH : checkpoint medical_detector raw_folder (requis)",
    "--input PATH : mammographie PNG/JPEG/BMP/PPM (requise)",
    "--output PATH : résultats JSON (outputs/medical_detections.json)",
    "--score-threshold N : confiance minimale (valeur du checkpoint ou 0.25)",
    "--nms-iou-threshold N : seuil NMS (valeur du checkpoint ou 0.45)",
    "--max-detections N : nombre maximal de cadres (100)",
    "--viz : affiche l'image et les cadres jusqu'à fermeture de la fenêtre",
  },
  examples = {
    "./bin/mimir --lua scripts/inferences/detect_medical_anomalies.lua -- --checkpoint checkpoint/medical_detector --input mammographie.png --viz",
  },
})

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
local opts = Args.parse(arg) or {}

local function die(message) error("[detect_medical_anomalies] "..tostring(message), 0) end
local function check(ok, err, operation)
  if not ok then die(operation..": "..tostring(err or "échec")) end
end
local function clamp(value, low, high)
  return math.max(low, math.min(high, value))
end
local function sigmoid(value)
  if value >= 0 then
    local z = math.exp(-value)
    return 1 / (1 + z)
  end
  local z = math.exp(value)
  return z / (1 + z)
end
local function iou(a, b)
  local x1, y1 = math.max(a.x1, b.x1), math.max(a.y1, b.y1)
  local x2, y2 = math.min(a.x2, b.x2), math.min(a.y2, b.y2)
  local intersection = math.max(0, x2-x1) * math.max(0, y2-y1)
  local area_a = math.max(0, a.x2-a.x1) * math.max(0, a.y2-a.y1)
  local area_b = math.max(0, b.x2-b.x1) * math.max(0, b.y2-b.y1)
  local union = area_a + area_b - intersection
  return union > 0 and intersection / union or 0
end
local function grid_size(size)
  for _ = 1, 5 do size = math.floor((size + 1) / 2) end
  return math.max(1, size)
end

local checkpoint_arg = Args.get_str(opts, "checkpoint", nil)
local input_path = Args.get_str(opts, "input", nil)
local output_path = Args.get_str(opts, "output", "outputs/medical_detections.json")
if not checkpoint_arg or checkpoint_arg == "" then die("--checkpoint est requis") end
if not input_path or input_path == "" then die("--input est requis") end
local checkpoint = Ckpt.resolve_dir(checkpoint_arg)
if not checkpoint then die("checkpoint raw_folder introuvable: "..checkpoint_arg) end

local architecture_path = FS.join(checkpoint, "model", "architecture.json")
local architecture = read_json(architecture_path)
local cfg = type(architecture) == "table" and
  (architecture.model_config or architecture.modelConfig) or nil
if type(cfg) ~= "table" or cfg.type ~= "medical_detector" then
  die("checkpoint medical_detector attendu: "..architecture_path)
end
local image_w, image_h, image_c = cfg.image_w, cfg.image_h, cfg.image_c
local num_classes = cfg.num_classes
local grid_w = cfg.grid_w or grid_size(image_w)
local grid_h = cfg.grid_h or grid_size(image_h)
if image_c ~= 3 then die("le script d'inférence attend image_c=3") end

local class_names = cfg.class_names or {}
for class_id = 0, num_classes - 1 do
  if type(class_names[class_id + 1]) ~= "string" then
    class_names[class_id + 1] = "anomaly_"..class_id
  end
end
local dtype_aliases = {F32="float32", F16="float16", BF16="bfloat16", FP32="float32", FP16="float16"}
if cfg.dtype then cfg.dtype = dtype_aliases[tostring(cfg.dtype):upper()] or cfg.dtype end
local score_threshold = clamp(Args.get_num(opts, "score-threshold", cfg.score_threshold or 0.25), 0, 1)
local nms_threshold = clamp(Args.get_num(opts, "nms-iou-threshold", cfg.nms_iou_threshold or 0.45), 0, 1)
local max_detections = math.max(1, Args.get_int(opts, "max-detections", cfg.max_detections or 100))
local mem_gb = Args.get_num(opts, "mem-gb", 8)

check(Mimir.Allocator.configure({max_ram_gb=mem_gb, enable_compression=true}), nil,
  "Allocator.configure")
local ok_create, create_err = Mimir.Model.create("medical_detector", cfg)
check(ok_create, create_err, "Model.create")
if cfg.dtype and type(Mimir.Model.dtype) == "function" then
  local ok_dtype, dtype_err = Mimir.Model.dtype(cfg.dtype)
  check(ok_dtype, dtype_err, "Model.dtype")
end
local ok_alloc, alloc_err = Mimir.Model.allocate_params()
check(ok_alloc, alloc_err, "Model.allocate_params")
local ok_load, load_err = Mimir.Serialization.load(checkpoint, "raw_folder", {
  load_tokenizer=false, load_encoder=false, load_optimizer=false,
  strict_mode=true, validate_checksums=true,
})
check(ok_load, load_err, "Serialization.load")

local loaded, image_err = Mimir.IO.read_image_rgb_u8(input_path, image_w, image_h)
if type(loaded) ~= "table" then die("lecture image: "..tostring(image_err)) end
local pixels = loaded.image
local input = {}
for y = 0, image_h - 1 do
  for x = 0, image_w - 1 do
    local pixel = (y * image_w + x) * image_c
    for channel = 0, image_c - 1 do
      local chw = channel * image_h * image_w + y * image_w + x
      input[chw + 1] = pixels[pixel + channel + 1] / 127.5 - 1
    end
  end
end

local output, forward_err = Mimir.Model.forward({__input__=input}, false)
if type(output) ~= "table" then die("Model.forward: "..tostring(forward_err)) end
local cells = grid_w * grid_h
local expected = (5 + num_classes) * cells
if #output ~= expected then
  die(string.format("sortie inattendue: %d valeurs, %d attendues", #output, expected))
end
local function channel_value(channel, cell)
  return output[channel * cells + cell + 1]
end

local candidates = {}
for grid_y = 0, grid_h - 1 do
  for grid_x = 0, grid_w - 1 do
    local cell = grid_y * grid_w + grid_x
    local objectness = sigmoid(channel_value(0, cell))
    local best_class, best_probability = 0, -1
    for class_id = 0, num_classes - 1 do
      local probability = sigmoid(channel_value(5 + class_id, cell))
      if probability > best_probability then
        best_class, best_probability = class_id, probability
      end
    end
    local score = objectness * best_probability
    local cx = (grid_x + sigmoid(channel_value(1, cell))) / grid_w
    local cy = (grid_y + sigmoid(channel_value(2, cell))) / grid_h
    local width = sigmoid(channel_value(3, cell))
    local height = sigmoid(channel_value(4, cell))
    local detection = {
      x1=clamp(cx-width/2, 0, 1), y1=clamp(cy-height/2, 0, 1),
      x2=clamp(cx+width/2, 0, 1), y2=clamp(cy+height/2, 0, 1),
      score=score, class_id=best_class, label=class_names[best_class+1],
    }
    if detection.x2 > detection.x1 and detection.y2 > detection.y1 then
      candidates[#candidates + 1] = detection
    end
  end
end
table.sort(candidates, function(a, b) return a.score > b.score end)
local viz_detections = {}
for _, candidate in ipairs(candidates) do
  local suppressed = false
  for _, accepted in ipairs(viz_detections) do
    if candidate.class_id == accepted.class_id and iou(candidate, accepted) > nms_threshold then
      suppressed = true
      break
    end
  end
  if not suppressed then viz_detections[#viz_detections + 1] = candidate end
  if #viz_detections >= max_detections then break end
end
local detections = {}
for _, detection in ipairs(viz_detections) do
  if detection.score >= score_threshold then
    detections[#detections + 1] = detection
  end
end

local serializable = {}
for index, detection in ipairs(detections) do
  serializable[index] = {
    x1=detection.x1, y1=detection.y1, x2=detection.x2, y2=detection.y2,
    pixel_x1=math.floor(detection.x1*image_w+0.5),
    pixel_y1=math.floor(detection.y1*image_h+0.5),
    pixel_x2=math.floor(detection.x2*image_w+0.5),
    pixel_y2=math.floor(detection.y2*image_h+0.5),
    score=detection.score, class_id=detection.class_id, label=detection.label,
  }
end
local directory = FS.dirname(output_path)
if directory and directory ~= "" then FS.mkdir_p(directory) end
if type(write_json) ~= "function" then die("API Mímir write_json indisponible") end
write_json(output_path, {
  image=input_path, resized_width=image_w, resized_height=image_h,
  score_threshold=score_threshold, nms_iou_threshold=nms_threshold,
  detections=serializable,
})
print(string.format("%d anomalie(s) détectée(s), résultats: %s", #detections, output_path))
for _, detection in ipairs(detections) do
  print(string.format("  %s score=%.4f box=[%.3f %.3f %.3f %.3f]",
    detection.label, detection.score, detection.x1, detection.y1, detection.x2, detection.y2))
end

if Args.get_bool(opts, "viz", false) then
  check(Mimir.Viz.create({visualization={enabled=true, window_title="Mímir - Détection médicale"}}),
    nil, "Viz.create")
  check(Mimir.Viz.add_image(pixels, image_w, image_h, image_c, input_path), nil, "Viz.add_image")
  check(Mimir.Viz.set_detections(viz_detections, "output"), nil, "Viz.set_detections")
  check(Mimir.Viz.set_detection_threshold(score_threshold), nil,
    "Viz.set_detection_threshold")
  check(Mimir.Viz.show_detections(true), nil, "Viz.show_detections")
  while Mimir.Viz.is_open() do Mimir.Viz.process_events() end
end
