-- Shared architecture extraction for MPK tools. No checkpoint weights are loaded.
local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local M = {}
local function collect_layer_params(layer)
  local p = {}
  local scalar_keys = {
    "in_features", "out_features", "in_channels", "out_channels",
    "input_height", "input_width", "kernel_size", "kernel_h", "kernel_w",
    "stride", "stride_h", "stride_w", "padding", "pad_h", "pad_w",
    "dilation", "groups", "eps", "num_groups", "dropout_p", "vocab_size",
    "embed_dim", "axis", "concat_axis", "split_axis", "num_splits",
    "scale_h", "scale_w", "out_h", "out_w", "num_heads", "head_dim",
    "seq_len", "causal", "use_bias", "nms_iou_threshold",
    "nms_score_threshold", "nms_max_detections", "nms_class_agnostic",
  }
  for _, key in ipairs(scalar_keys) do
    local value = layer[key]
    if type(value) == "number" or type(value) == "boolean" then
      p[key] = value
    end
  end
  for _, key in ipairs({"target_shape", "permute_dims", "split_sizes"}) do
    if type(layer[key]) == "table" and #layer[key] > 0 then
      p[key] = layer[key]
    end
  end
  for k, v in pairs(layer.config or layer.params or {}) do p[k] = v end
  if p.has_bias ~= nil then p.use_bias = p.has_bias; p.has_bias = nil end
  return p
end

local function build_graph_from_layers(layers)
  local nodes, links = {}, {}
  for i = 1, #layers do
    local la = layers[i]
    local name = tostring(la.name or ("layer_" .. i))
    local output = tostring(la.output or (name .. "_out"))
    local inputs = type(la.inputs) == "table" and la.inputs or { "x" }

    local node = {
      id = name,
      name = name,
      type = tostring(la.type or "Identity"),
      params_count = tonumber(la.params_count or la.param_count) or 0,
      inputs = inputs,
      output = output,
      params = collect_layer_params(la),
      position = { x = 80 + ((i - 1) * 180), y = 80 },
    }
    nodes[#nodes + 1] = node

    for _, inp in ipairs(inputs) do
      links[#links + 1] = {
        from = tostring(inp),
        to = name,
        kind = "tensor",
      }
    end
  end
  return nodes, links
end


M.build_graph_from_layers = build_graph_from_layers

local function read_safetensors(path)
  local f, err = io.open(path, "rb")
  if not f then return nil, err end
  local function fail(msg) f:close(); return nil, msg end
  local size = f:seek("end")
  f:seek("set", 0)
  local prefix = f:read(8)
  if not prefix or #prefix ~= 8 then return fail("truncated SafeTensors header") end
  local n = 0
  for i = 8, 1, -1 do n = n * 256 + prefix:byte(i) end
  if n < 2 or n > 64 * 1024 * 1024 or n > size - 8 then
    return fail("invalid SafeTensors header length")
  end
  local header, parse_err = MPK.decode_json(f:read(n))
  if type(header) ~= "table" then return fail(parse_err or "invalid SafeTensors header") end
  local entry = header["model/architecture_json"]
  if type(entry) ~= "table" or entry.dtype ~= "U8" or type(entry.data_offsets) ~= "table" then
    return fail("SafeTensors has no Mimir model/architecture_json; tensor names alone cannot define a graph")
  end
  local first, last = entry.data_offsets[1], entry.data_offsets[2]
  if type(first) ~= "number" or type(last) ~= "number" or first < 0 or last <= first
      or first % 1 ~= 0 or last % 1 ~= 0 or last > size - 8 - n
      or last - first > 64 * 1024 * 1024 then
    return fail("invalid architecture tensor offsets")
  end
  f:seek("set", 8 + n + first)
  local raw = f:read(last - first)
  f:close()
  return MPK.decode_json(raw)
end

function M.checkpoint(path, format)
  local aliases = {rawfolder="raw_folder", safetensor="safetensors", debugjson="debug_json"}
  format = aliases[format] or format or "auto"
  local arch, err
  if format == "auto" then
    if path:lower():match("%.safetensors$") or path:lower():match("%.st$") then format = "safetensors"
    elseif path:lower():match("%.json$") then format = "debug_json"
    else format = "raw_folder" end
  end
  if format == "raw_folder" then
    local root = path:gsub("/+$", "")
    local candidate = root.."/model/architecture.json"
    local probe = io.open(candidate, "rb")
    if probe then probe:close() else candidate = root.."/architecture.json" end
    arch, err = MPK.read_json_file(candidate)
  elseif format == "debug_json" then arch, err = MPK.read_json_file(path)
  elseif format == "safetensors" then arch, err = read_safetensors(path)
  else return nil, "unknown checkpoint format: "..tostring(format) end
  if type(arch) ~= "table" then return nil, err end
  if type(arch.layers) ~= "table" or #arch.layers == 0 then return nil, "checkpoint has no serialized layers" end
  local cfg = arch.model_config or {}
  local info = type(arch.model) == "table" and arch.model or {}
  local kind = arch.model_type or cfg.type or arch.architecture or info.type or arch.model_name or "custom_graph"
  return {layers=arch.layers, config=cfg, type=kind, source=format}
end

-- Validate all destination names before writing either output.
function M.output_paths(opts, out)
  if out == "" or not out:lower():match("%.mpk$") then return nil, "--out must end with .mpk" end
  local compiled
  if opts.compile ~= nil and opts.compile ~= false then
    compiled = type(opts.compile) == "string" and opts.compile or (out..".bin")
    if not compiled:lower():match("%.mpk%.bin$") then return nil, "--compile output must end with .mpk.bin" end
  end
  return {source=out, compiled=compiled}
end
function M.write_outputs(pkg, paths)
  local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
  local parent = FS.dirname(paths.source)
  if parent and parent ~= "" then FS.mkdir_p(parent) end
  local ok, err = MPK.write(paths.source, pkg)
  if not ok then return nil, err end
  if paths.compiled then
    parent = FS.dirname(paths.compiled)
    if parent and parent ~= "" then FS.mkdir_p(parent) end
    return MPK.compile(paths.source, paths.compiled)
  end
  return true
end

return M
