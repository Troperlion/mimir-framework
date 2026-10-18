---@diagnostic disable: undefined-global

-- Generate the vision architecture MPK prototypes installed in _archi/.
-- Their registry execution delegates to an existing native backbone; the MPK
-- graph documents the intended detection/segmentation head for development.

dofile(ROOTWORK.."/scripts/modules/mpk_help.lua").show("add_vision_mpk_architectures")

local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local MPKLayers = dofile(ROOTWORK.."/scripts/modules/mpk_layers.lua")

local function die(msg)
  io.stderr:write("[add_vision_mpk_architectures] " .. tostring(msg) .. "\n")
  os.exit(1)
end

local function node(id, layer_type, inputs, output, params, x, y)
  return {
    id = id,
    name = id,
    type = layer_type,
    inputs = inputs,
    output = output,
    params = params or {},
    params_count = 0,
    position = { x = x or 0, y = y or 0 },
  }
end

local specs = {
  {
    name = "r_cnn",
    label = "R-CNN",
    base = "vgg16",
    description = "Prototype MPK R-CNN: exécution déléguée au backbone VGG16; propositions de régions, ROI et décodage restent externes, NMS est fourni par le runtime.",
    task = "object_detection",
    external = { "region_proposal", "roi_crop_or_align", "box_decode" },
    nodes = {
      node("region_proposals", "Identity", { "image" }, "regions", { role = "external_region_proposal" }, 40, 80),
      node("vgg16_roi_features", "Conv2d", { "regions" }, "roi_features", { role = "shared_cnn_backbone" }, 220, 80),
      node("roi_pool", "AdaptiveAvgPool2d", { "roi_features" }, "roi_vector", { output_h = 7, output_w = 7 }, 400, 80),
      node("class_head", "Linear", { "roi_vector" }, "class_logits", { role = "classification_head" }, 580, 40),
      node("bbox_head", "Linear", { "roi_vector" }, "bbox_deltas", { role = "box_regression_head" }, 580, 140),
      node("nms", "NMS", { "decoded_boxes", "class_scores", "class_ids" }, "kept_indices", { iou_threshold = 0.5, score_threshold = 0.05, class_agnostic = false }, 780, 90),
    },
  },
  {
    name = "yolo",
    label = "YOLO",
    base = "mobilenet",
    description = "Prototype MPK YOLO à backbone MobileNet: pyramide et têtes multi-échelles décrites; le décodage reste externe et NMS est fourni par le runtime.",
    task = "one_stage_object_detection",
    external = { "anchor_or_grid_decode", "confidence_filter" },
    nodes = {
      node("mobilenet_backbone", "DepthwiseConv2d", { "image" }, "features", { role = "backbone" }, 40, 100),
      node("feature_neck", "Conv2d", { "features" }, "pyramid", { role = "feature_pyramid" }, 220, 100),
      node("detect_small", "Conv2d", { "pyramid" }, "pred_small", { role = "yolo_head_small" }, 420, 20),
      node("detect_medium", "Conv2d", { "pyramid" }, "pred_medium", { role = "yolo_head_medium" }, 420, 100),
      node("detect_large", "Conv2d", { "pyramid" }, "pred_large", { role = "yolo_head_large" }, 420, 180),
      node("objectness", "Sigmoid", { "pred_small", "pred_medium", "pred_large" }, "detections_raw", { role = "objectness_and_classes" }, 620, 100),
      node("nms", "NMS", { "decoded_boxes", "class_scores", "class_ids" }, "kept_indices", { iou_threshold = 0.45, score_threshold = 0.25, class_agnostic = false }, 820, 100),
    },
  },
  {
    name = "ssd",
    label = "SSD",
    base = "vgg16",
    description = "Prototype MPK SSD à backbone VGG16: cartes multi-résolutions et têtes classe/boîte décrites; génération des priors et décodage restent externes, NMS est fourni par le runtime.",
    task = "single_shot_object_detection",
    external = { "default_box_generation", "box_decode" },
    nodes = {
      node("vgg16_backbone", "Conv2d", { "image" }, "base_features", { role = "backbone" }, 40, 100),
      node("extra_features_1", "Conv2d", { "base_features" }, "feature_1", { stride = 2 }, 220, 40),
      node("extra_features_2", "Conv2d", { "feature_1" }, "feature_2", { stride = 2 }, 220, 160),
      node("class_predictors", "Conv2d", { "base_features", "feature_1", "feature_2" }, "class_logits", { role = "multibox_classification" }, 440, 40),
      node("box_predictors", "Conv2d", { "base_features", "feature_1", "feature_2" }, "box_offsets", { role = "multibox_regression" }, 440, 160),
      node("ssd_outputs", "Concat", { "class_logits", "box_offsets" }, "detections_raw", { axis = 0 }, 640, 100),
      node("nms", "NMS", { "decoded_boxes", "class_scores", "class_ids" }, "kept_indices", { iou_threshold = 0.45, score_threshold = 0.05, class_agnostic = false }, 840, 100),
    },
  },
  {
    name = "deeplab",
    label = "DeepLab",
    base = "resnet",
    description = "Prototype MPK DeepLab à backbone ResNet: ASPP et décodeur de segmentation décrits; la convolution dilatée spécialisée reste une métadonnée.",
    task = "semantic_segmentation",
    external = { "specialized_atrous_kernel" },
    nodes = {
      node("resnet_backbone", "Conv2d", { "image" }, "encoder_features", { role = "backbone" }, 40, 100),
      node("aspp_rate_1", "Conv2d", { "encoder_features" }, "aspp_1", { dilation = 1 }, 240, 20),
      node("aspp_rate_6", "Conv2d", { "encoder_features" }, "aspp_6", { dilation = 6 }, 240, 80),
      node("aspp_rate_12", "Conv2d", { "encoder_features" }, "aspp_12", { dilation = 12 }, 240, 140),
      node("aspp_rate_18", "Conv2d", { "encoder_features" }, "aspp_18", { dilation = 18 }, 240, 200),
      node("aspp_merge", "Concat", { "aspp_1", "aspp_6", "aspp_12", "aspp_18" }, "aspp_features", { axis = 0 }, 460, 100),
      node("segmentation_logits", "Conv2d", { "aspp_features" }, "lowres_logits", { role = "class_logits" }, 640, 100),
      node("decoder_upsample", "UpsampleBilinear", { "lowres_logits" }, "segmentation_mask", { role = "restore_image_resolution" }, 820, 100),
    },
  },
}

FS.mkdir_p("_archi")

for _, spec in ipairs(specs) do
  local cfg, cfg_err = Mimir.Architectures.default_config(spec.base)
  if type(cfg) ~= "table" then
    die("default_config(" .. spec.base .. ") failed: " .. tostring(cfg_err))
  end

  cfg.mpk_architecture = spec.name
  cfg.mpk_display_name = spec.label
  cfg.mpk_task = spec.task
  cfg.mpk_prototype = true
  cfg.mpk_execution_delegate = spec.base

  local links = {}
  for _, n in ipairs(spec.nodes) do
    for _, input in ipairs(n.inputs or {}) do
      links[#links + 1] = { from = input, to = n.id, kind = "tensor" }
    end
  end

  local structure = {
    template = spec.name,
    architecture = spec.name,
    version = 1,
    status = "prototype",
    execution = {
      mode = "registry_delegate",
      base_architecture = spec.base,
      graph_is_documentary = true,
    },
    task = spec.task,
    external_components = spec.external,
    graph = {
      mode = "node",
      nodes = spec.nodes,
      links = links,
    },
    build = {
      dynamic_layer_assembly = false,
      generated_by = "scripts/tools/add_vision_mpk_architectures.lua",
    },
  }

  local ok_graph, graph_err =
    MPKLayers.normalize_graph_in_place(structure)
  if not ok_graph then die(spec.name .. ": " .. tostring(graph_err)) end

  local pkg, build_err = MPK.build({
    name = spec.name,
    type = spec.base,
    author = "Mimir",
    modifiable = true,
    viz_specified = true,
    base_config = cfg,
    model_structure = structure,
    description = spec.description,
  })
  if not pkg then die(spec.name .. ": " .. tostring(build_err)) end

  local output = "_archi/" .. spec.name .. ".mpk"
  local ok_write, write_err = MPK.write(output, pkg)
  if not ok_write then die(output .. ": " .. tostring(write_err)) end
  io.stdout:write("[add_vision_mpk_architectures] wrote " .. output .. "\n")
end
