---@diagnostic disable: undefined-global

-- Shared helpers for MPK node-graph layer type normalization/validation.

---@class MimirMPKLayersModule
local M = {}

local KNOWN_CANONICAL = {
  ["Conv2d"] = true,
  ["ConvTranspose2d"] = true,
  ["Conv1d"] = true,
  ["DepthwiseConv2d"] = true,
  ["Linear"] = true,
  ["Bilinear"] = true,
  ["Embedding"] = true,
  ["EmbeddingBag"] = true,
  ["BatchNorm2d"] = true,
  ["BatchNorm1d"] = true,
  ["LayerNorm"] = true,
  ["GroupNorm"] = true,
  ["InstanceNorm2d"] = true,
  ["RMSNorm"] = true,
  ["ReLU"] = true,
  ["LeakyReLU"] = true,
  ["GELU"] = true,
  ["GEGLU"] = true,
  ["SiLU"] = true,
  ["Tanh"] = true,
  ["Sigmoid"] = true,
  ["Softmax"] = true,
  ["LogSoftmax"] = true,
  ["Softplus"] = true,
  ["Mish"] = true,
  ["HardSigmoid"] = true,
  ["HardSwish"] = true,
  ["MaxPool2d"] = true,
  ["AvgPool2d"] = true,
  ["AdaptiveAvgPool2d"] = true,
  ["GlobalAvgPool2d"] = true,
  ["MaxPool1d"] = true,
  ["AvgPool1d"] = true,
  ["TokenMeanPool"] = true,
  ["Dropout"] = true,
  ["Dropout2d"] = true,
  ["AlphaDropout"] = true,
  ["Flatten"] = true,
  ["Reshape"] = true,
  ["Transpose"] = true,
  ["Permute"] = true,
  ["Squeeze"] = true,
  ["Unsqueeze"] = true,
  ["View"] = true,
  ["Add"] = true,
  ["Subtract"] = true,
  ["Multiply"] = true,
  ["Divide"] = true,
  ["Concat"] = true,
  ["Split"] = true,
  ["Chunk"] = true,
  ["Stack"] = true,
  ["MatMul"] = true,
  ["BatchMatMul"] = true,
  ["NMS"] = true,
  ["SelfAttention"] = true,
  ["MultiHeadAttention"] = true,
  ["CrossAttention"] = true,
  ["UpsampleNearest"] = true,
  ["UpsampleBilinear"] = true,
  ["UpsampleBicubic"] = true,
  ["PixelShuffle"] = true,
  ["LSTM"] = true,
  ["GRU"] = true,
  ["RNN"] = true,
  ["ZeroPad2d"] = true,
  ["ReflectionPad2d"] = true,
  ["ReplicationPad2d"] = true,
  ["Identity"] = true,
  ["Constant"] = true,
  ["Lambda"] = true,
}

local function to_lower_map(tbl)
  local out = {}
  for k in pairs(tbl) do out[k:lower()] = k end
  return out
end

local KNOWN_BY_LOWER = to_lower_map(KNOWN_CANONICAL)

function M.available_layer_types()
  local out = {}
  for name in pairs(KNOWN_CANONICAL) do out[#out + 1] = name end
  table.sort(out)
  return out
end

function M.infer_layer_type(layer_name)
  local raw = tostring(layer_name or "")
  if raw == "" then return "Linear" end

  local direct = KNOWN_BY_LOWER[raw:lower()]
  if direct ~= nil then return direct end

  local name = raw
    :gsub("([a-z])([A-Z])", "%1_%2")
    :lower()
    :gsub("[^%w]+", "_")
    :gsub("^_+", "")
    :gsub("_+$", "")
  local padded = "_" .. name .. "_"
  local function contains(...)
    for i = 1, select("#", ...) do
      local token = tostring(select(i, ...))
      if padded:find("_" .. token .. "_", 1, true) then return true end
    end
    return false
  end

  -- Les formes spécialisées doivent précéder leur famille générique.
  if name:find("conv_transpose", 1, true) or
      contains("deconv", "upconv") then return "ConvTranspose2d" end
  if name:find("depthwise_conv", 1, true) or
      contains("depthwise") then return "DepthwiseConv2d" end
  if name:find("conv1d", 1, true) then return "Conv1d" end
  if contains("conv", "conv2d") or name:find("conv2d", 1, true) then
    return "Conv2d"
  end

  if name:find("batch_norm", 1, true) or contains("batchnorm", "bn") then
    return name:find("1d", 1, true) and "BatchNorm1d" or "BatchNorm2d"
  end
  if name:find("instance_norm", 1, true) or contains("instancenorm") then
    return "InstanceNorm2d"
  end
  if name:find("layer_norm", 1, true) or contains("layernorm", "ln") then
    return "LayerNorm"
  end
  if name:find("group_norm", 1, true) or contains("groupnorm", "gn") then
    return "GroupNorm"
  end
  if name:find("rms_norm", 1, true) or contains("rmsnorm") then
    return "RMSNorm"
  end

  if name:find("cross_attention", 1, true) or
      contains("crossattn") then return "CrossAttention" end
  if name:find("multi_head_attention", 1, true) or
      contains("multiheadattention", "mha") then return "MultiHeadAttention" end
  if name:find("self_attention", 1, true) or
      contains("selfattn", "attention", "attn") then return "SelfAttention" end

  if name:find("adaptive_avg_pool", 1, true) then return "AdaptiveAvgPool2d" end
  if name:find("global_avg_pool", 1, true) or
      contains("gap") then return "GlobalAvgPool2d" end
  if name:find("max_pool1d", 1, true) then return "MaxPool1d" end
  if name:find("avg_pool1d", 1, true) then return "AvgPool1d" end
  if name:find("max_pool", 1, true) or contains("maxpool") then
    return "MaxPool2d"
  end
  if name:find("avg_pool", 1, true) or contains("avgpool") then
    return "AvgPool2d"
  end

  if name:find("upsample_bilinear", 1, true) then return "UpsampleBilinear" end
  if name:find("upsample_bicubic", 1, true) then return "UpsampleBicubic" end
  if contains("upsample") or name:find("upsample_nearest", 1, true) then
    return "UpsampleNearest"
  end
  if name:find("pixel_shuffle", 1, true) or contains("pixelshuffle") then
    return "PixelShuffle"
  end

  local named_types = {
    {"log_softmax", "LogSoftmax"}, {"leaky_relu", "LeakyReLU"},
    {"hard_sigmoid", "HardSigmoid"}, {"hard_swish", "HardSwish"},
    {"alpha_dropout", "AlphaDropout"}, {"dropout2d", "Dropout2d"},
    {"embedding_bag", "EmbeddingBag"}, {"batch_mat_mul", "BatchMatMul"},
    {"zero_pad", "ZeroPad2d"}, {"reflection_pad", "ReflectionPad2d"},
    {"replication_pad", "ReplicationPad2d"}, {"token_mean_pool", "TokenMeanPool"},
  }
  for _, mapping in ipairs(named_types) do
    if name:find(mapping[1], 1, true) then return mapping[2] end
  end

  local token_types = {
    nms = "NMS", relu = "ReLU", gelu = "GELU", geglu = "GEGLU",
    silu = "SiLU", swish = "SiLU", tanh = "Tanh", sigmoid = "Sigmoid",
    softmax = "Softmax", softplus = "Softplus", mish = "Mish",
    dropout = "Dropout", flatten = "Flatten", reshape = "Reshape",
    transpose = "Transpose", permute = "Permute", squeeze = "Squeeze",
    unsqueeze = "Unsqueeze", view = "View", add = "Add", sum = "Add",
    subtract = "Subtract", sub = "Subtract", multiply = "Multiply",
    mul = "Multiply", divide = "Divide", div = "Divide", concat = "Concat",
    split = "Split", chunk = "Chunk", stack = "Stack", matmul = "MatMul",
    embedding = "Embedding", embed = "Embedding", lstm = "LSTM", gru = "GRU",
    rnn = "RNN", identity = "Identity", constant = "Constant",
    lambda = "Lambda", bilinear = "Bilinear",
  }
  for token, layer_type in pairs(token_types) do
    if contains(token) then return layer_type end
  end

  if contains("linear", "dense", "fc", "head", "projection", "proj") then
    return "Linear"
  end
  return "Linear"
end

local function positive_number(...)
  for i = 1, select("#", ...) do
    local candidate = select(i, ...)
    local value = tonumber(candidate)
    if value and value > 0 then return value end
  end
  return nil
end

function M.predict_params_count(layer_type, params, context)
  params = type(params) == "table" and params or {}
  context = type(context) == "table" and context or {}
  local cfg = type(context.base_config) == "table" and context.base_config or {}
  local name = tostring(context.layer_name or ""):lower()
  local use_bias = params.use_bias
  if use_bias == nil then use_bias = params.bias end
  if use_bias == nil then use_bias = true end

  local in_features = positive_number(
    params.in_features, params.input_dim, context.in_features,
    context.current_features, cfg.input_dim, cfg.d_model, cfg.hidden_dim)
  local out_features = positive_number(params.out_features, params.output_dim)
  if not out_features then
    if name:find("head", 1, true) or name:find("output", 1, true) then
      out_features = positive_number(cfg.output_dim, cfg.num_classes, cfg.vocab_size)
    else
      out_features = positive_number(
        context.out_features, cfg.hidden_dim, cfg.d_model, cfg.output_dim)
    end
  end

  local in_channels = positive_number(
    params.in_channels, context.in_channels, context.current_channels,
    cfg.image_c, cfg.in_channels, cfg.base_channels)
  local out_channels = positive_number(params.out_channels)
  if not out_channels then
    if name:find("latent", 1, true) then
      out_channels = positive_number(cfg.latent_c, cfg.latent_channels)
    elseif name:find("output", 1, true) or name:find("recon", 1, true) then
      out_channels = positive_number(cfg.image_c, cfg.out_channels)
    else
      out_channels = positive_number(
        context.out_channels, cfg.base_channels, cfg.out_channels, in_channels)
    end
  end

  local canonical = M.canonical_layer_type(layer_type, params)
  canonical = canonical or tostring(layer_type or "")

  if canonical == "Linear" then
    if not in_features or not out_features then return 0, "dimensions insuffisantes" end
    return math.floor(in_features * out_features +
      (use_bias and out_features or 0)), "in_features*out_features + bias"
  end
  if canonical == "Bilinear" then
    local in1 = positive_number(params.in1_features, params.in_features, in_features)
    local in2 = positive_number(params.in2_features, params.out_features)
    local out = positive_number(params.output_features, params.embed_dim, cfg.output_dim)
    if not in1 or not in2 or not out then return 0, "dimensions insuffisantes" end
    return math.floor(out * in1 * in2 + (use_bias and out or 0)),
      "out*in1*in2 + bias"
  end
  if canonical == "Conv2d" or canonical == "ConvTranspose2d" or
      canonical == "DepthwiseConv2d" or canonical == "Conv1d" then
    if not in_channels or not out_channels then
      return 0, "canaux insuffisants"
    end
    local groups = positive_number(params.groups) or
      (canonical == "DepthwiseConv2d" and in_channels or 1)
    local kh = positive_number(params.kernel_h, params.kernel_size) or 3
    local kw = canonical == "Conv1d" and 1 or
      (positive_number(params.kernel_w, params.kernel_size) or kh)
    local weights = out_channels * (in_channels / groups) * kh * kw
    return math.floor(weights + (use_bias and out_channels or 0)),
      "out_channels*(in_channels/groups)*kernel + bias"
  end
  if canonical == "Embedding" or canonical == "EmbeddingBag" then
    local vocab = positive_number(params.vocab_size, cfg.vocab_size)
    local dim = positive_number(
      params.embed_dim, params.out_features, cfg.embed_dim, cfg.d_model)
    if not vocab or not dim then return 0, "vocabulaire/dimension insuffisants" end
    return math.floor(vocab * dim), "vocab_size*embed_dim"
  end
  if canonical == "BatchNorm2d" or canonical == "BatchNorm1d" or
      canonical == "GroupNorm" or canonical == "InstanceNorm2d" then
    local width = positive_number(
      params.num_features, params.in_channels, in_channels, out_channels)
    if not width then return 0, "dimension de normalisation insuffisante" end
    return math.floor(2 * width), "scale + bias"
  end
  if canonical == "LayerNorm" then
    local width = positive_number(
      params.normalized_size, params.out_features, out_features, in_features)
    if not width then return 0, "dimension de normalisation insuffisante" end
    return math.floor(2 * width), "scale + bias"
  end
  if canonical == "RMSNorm" then
    local width = positive_number(
      params.normalized_size, params.out_features, out_features, in_features)
    if not width then return 0, "dimension de normalisation insuffisante" end
    return math.floor(width), "scale"
  end
  if canonical == "SelfAttention" or canonical == "MultiHeadAttention" or
      canonical == "CrossAttention" then
    local dim = positive_number(
      params.embed_dim, params.d_model, out_features, in_features, cfg.d_model)
    if not dim then return 0, "dimension d'attention insuffisante" end
    return math.floor(4 * dim * dim + (use_bias and 4 * dim or 0)),
      "projections QKV + sortie"
  end
  if canonical == "LSTM" or canonical == "GRU" or canonical == "RNN" then
    local input_size = positive_number(params.input_size, in_features)
    local hidden_size = positive_number(
      params.hidden_size, params.out_features, cfg.hidden_dim)
    if not input_size or not hidden_size then
      return 0, "dimensions recurrentes insuffisantes"
    end
    local gates = canonical == "LSTM" and 4 or (canonical == "GRU" and 3 or 1)
    return math.floor(gates * hidden_size * (input_size + hidden_size) +
      (use_bias and 2 * gates * hidden_size or 0)),
      "poids entree/recurrent + biais"
  end

  return 0, "layer sans parametres entrainables"
end

function M.canonical_layer_type(raw_type, params)
  local t = tostring(raw_type or "")
  if t == "" then
    return nil, "layer type vide"
  end
  local low = t:lower()
  params = params or {}

  if low == "conv2d" then return "Conv2d" end
  if low == "convtranspose2d" or low == "conv_transpose2d" then return "ConvTranspose2d" end
  if low == "linear" then return "Linear" end
  if low == "batchnorm" or low == "batch_norm" or low == "batchnorm2d" then return "BatchNorm2d" end
  if low == "layernorm" or low == "layer_norm" then return "LayerNorm" end
  if low == "groupnorm" or low == "group_norm" then return "GroupNorm" end
  if low == "nms" or low == "nonmaxsuppression" or
      low == "non_max_suppression" then
    return "NMS"
  end
  if low == "identity" then return "Identity" end

  if low == "activation" then
    local kind = tostring(params.kind or params.activation or "relu"):lower()
    if kind == "relu" then return "ReLU" end
    if kind == "leakyrelu" or kind == "leaky_relu" then return "LeakyReLU" end
    if kind == "gelu" then return "GELU" end
    if kind == "silu" or kind == "swish" then return "SiLU" end
    if kind == "tanh" then return "Tanh" end
    if kind == "sigmoid" then return "Sigmoid" end
    if kind == "softmax" then return "Softmax" end
    if kind == "logsoftmax" or kind == "log_softmax" then return "LogSoftmax" end
    return nil, "activation kind non supporte: " .. tostring(kind)
  end

  local known = KNOWN_BY_LOWER[low]
  if known ~= nil then
    return known
  end

  return nil, "type de layer non mappable: " .. tostring(t)
end

function M.normalize_graph_in_place(model_structure, opts)
  opts = opts or {}
  local allow_unknown = not not opts.allow_unknown

  if type(model_structure) ~= "table" then
    return false, "model_structure manquant"
  end
  local graph = model_structure.graph
  if type(graph) ~= "table" then
    return true
  end
  local nodes = graph.nodes
  if type(nodes) ~= "table" then
    return false, "model_structure.graph.nodes manquant"
  end
  if #nodes == 0 then
    return false, "model_structure.graph.nodes vide"
  end

  local seen_names = {}
  for i = 1, #nodes do
    local n = nodes[i]
    if type(n) ~= "table" then
      return false, "node #" .. tostring(i) .. " invalide"
    end
    local nid = tostring(n.id or n.name or "")
    local nname = tostring(n.name or n.id or "")
    if nid == "" or nname == "" then
      return false, "node #" .. tostring(i) .. ": id/name manquant"
    end
    if seen_names[nid] or seen_names[nname] then
      return false, "node duplique: " .. nid
    end
    seen_names[nid] = true
    seen_names[nname] = true
    n.id = nid
    n.name = nname

    local params_count = tonumber(n.params_count or 0)
    if params_count == nil or params_count < 0 or
        params_count ~= math.floor(params_count) then
      return false, "node '" .. nid .. "': params_count invalide"
    end
    n.params_count = params_count
    if n.params == nil then n.params = {} end
    if type(n.params) ~= "table" then
      return false, "node '" .. nid .. "': params doit etre une map"
    end
    if n.inputs == nil then n.inputs = { "x" } end
    if type(n.inputs) ~= "table" then
      return false, "node '" .. nid .. "': inputs doit etre une liste"
    end
    for input_i, input_name in ipairs(n.inputs) do
      if type(input_name) ~= "string" or input_name == "" then
        return false, "node '" .. nid .. "': input #" ..
          tostring(input_i) .. " invalide"
      end
    end
    if n.output == nil then n.output = nid .. "_out" end
    if type(n.output) ~= "string" or n.output == "" then
      return false, "node '" .. nid .. "': output invalide"
    end

    local c, err = M.canonical_layer_type(n.type, n.params)
    if not c then
      if allow_unknown then
        n.unresolved_type = tostring(n.type or "")
        goto continue
      end
      return false, "node '" .. nid .. "': " .. tostring(err)
    end
    n.type = c
    ::continue::
  end

  return true
end

return M
