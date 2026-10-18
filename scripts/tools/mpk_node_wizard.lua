---@diagnostic disable: undefined-global

-- Interactive MPK node wizard.
-- Builds a full model architecture package by asking questions and assembling
-- a node graph (layers + links) that can later be applied dynamically.

dofile(ROOTWORK.."/scripts/modules/mpk_help.lua").show("mpk_node_wizard")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local MPKLayers = dofile(ROOTWORK.."/scripts/modules/mpk_layers.lua")

local function log(...)
  local out = {}
  for i = 1, select("#", ...) do
    out[#out + 1] = tostring(select(i, ...))
  end
  io.stdout:write(table.concat(out, " ") .. "\n")
end

local function die(msg)
  io.stderr:write("[mpk_node_wizard] " .. tostring(msg) .. "\n")
  os.exit(1)
end

local function trim(s)
  s = tostring(s or "")
  s = s:gsub("^%s+", "")
  s = s:gsub("%s+$", "")
  return s
end

local function split_csv(s)
  local out = {}
  s = tostring(s or "")
  for part in s:gmatch("[^,]+") do
    part = trim(part)
    if part ~= "" then out[#out + 1] = part end
  end
  return out
end

local function parse_scalar(raw)
  raw = trim(raw)
  if raw == "" then return "" end
  local low = raw:lower()
  if low == "true" then return true end
  if low == "false" then return false end
  if low == "null" or low == "nil" then return nil end
  local n = tonumber(raw)
  if n ~= nil then return n end
  if raw:sub(1, 1) == '"' and raw:sub(-1) == '"' and #raw >= 2 then
    return raw:sub(2, -2)
  end
  return raw
end

local function prompt(label, default)
  if default ~= nil then
    io.stdout:write(label .. " [" .. tostring(default) .. "]: ")
  else
    io.stdout:write(label .. ": ")
  end
  local line = io.read("*l")
  if line == nil then return default end
  line = trim(line)
  if line == "" then return default end
  return line
end

local function prompt_bool(label, default)
  local def_s = default and "yes" or "no"
  local v = tostring(prompt(label .. " (yes/no)", def_s) or def_s):lower()
  if v == "y" or v == "yes" or v == "1" or v == "true" then return true end
  if v == "n" or v == "no" or v == "0" or v == "false" then return false end
  return default
end

local function prompt_int(label, default)
  local v = prompt(label, tostring(default or 0))
  local n = tonumber(v)
  if n == nil then return default or 0 end
  return math.floor(n)
end

local function gather_base_config(model_type)
  local cfg = {}

  if type(_G.Mimir) == "table" and type(Mimir.Architectures) == "table" then
    local use_default = prompt_bool("Charger la config par defaut du registre ?", true)
    if use_default then
      local dc, err = Mimir.Architectures.default_config(model_type)
      if type(dc) == "table" then
        for k, v in pairs(dc) do cfg[k] = v end
      else
        log("[warn] default_config indisponible: " .. tostring(err))
      end
    end
  end

  log("Ajout d'overrides config (laisser la cle vide pour terminer)")
  while true do
    local k = prompt("  config key", "")
    if not k or trim(k) == "" then break end
    local rv = prompt("  config value", "")
    cfg[k] = parse_scalar(rv)
  end
  return cfg
end

local function gather_nodes(base_config)
  local nodes = {}
  local links = {}
  local tensor_context = {
    x = {
      in_features = base_config.input_dim or base_config.d_model,
      in_channels = base_config.image_c or base_config.in_channels,
    },
  }

  log("\n--- Assembly nodal des layers ---")
  log("Pour terminer: repondre vide au nom du layer")

  while true do
    local layer_name = prompt("Layer name", "")
    if not layer_name or trim(layer_name) == "" then break end

    local detected_type = MPKLayers.infer_layer_type(layer_name)
    log("  Type detecte depuis le nom: " .. tostring(detected_type))
    local layer_type = prompt("Layer type (Entree pour accepter)", detected_type)
    local inputs_csv = prompt("Inputs (csv, ex: x,skip1)", "x")
    local output_name = prompt("Output tensor", layer_name .. "_out")
    local pos_x = prompt_int("Node X (viz)", (#nodes * 160) + 40)
    local pos_y = prompt_int("Node Y (viz)", 80)

    local node = {
      id = layer_name,
      name = layer_name,
      type = layer_type,
      inputs = split_csv(inputs_csv),
      output = output_name,
      params = {},
      position = { x = pos_x, y = pos_y },
    }

    if #node.inputs == 0 then
      node.inputs = { "x" }
    end

    log("  Ajout de params layer (key=value). Ligne vide pour terminer.")
    while true do
      local kv = prompt("    param", "")
      if not kv or trim(kv) == "" then break end
      local eq = kv:find("=", 1, true)
      if not eq or eq <= 1 then
        log("    format attendu: key=value")
      else
        local k = trim(kv:sub(1, eq - 1))
        local rv = trim(kv:sub(eq + 1))
        node.params[k] = parse_scalar(rv)
      end
    end

    local input_context = tensor_context[node.inputs[1]] or tensor_context.x or {}
    local prediction_context = {
      layer_name = layer_name,
      base_config = base_config,
      in_features = input_context.out_features or input_context.in_features,
      in_channels = input_context.out_channels or input_context.in_channels,
      current_features = input_context.out_features or input_context.in_features,
      current_channels = input_context.out_channels or input_context.in_channels,
    }
    local predicted_count, prediction_reason =
      MPKLayers.predict_params_count(layer_type, node.params, prediction_context)
    log(string.format(
      "  Params count predit: %d (%s)",
      predicted_count, tostring(prediction_reason)))
    node.params_count = prompt_int(
      "Params count (Entree pour accepter)", predicted_count)

    tensor_context[output_name] = {
      in_features = prediction_context.in_features,
      out_features = tonumber(node.params.out_features or
        node.params.output_dim or prediction_context.current_features),
      in_channels = prediction_context.in_channels,
      out_channels = tonumber(node.params.out_channels or
        prediction_context.current_channels),
    }

    for _, input_name in ipairs(node.inputs) do
      links[#links + 1] = {
        from = input_name,
        to = node.id,
        kind = "tensor",
      }
    end

    nodes[#nodes + 1] = node
  end

  return nodes, links
end


local opts = Args.parse(arg) or {}
if Args.has(opts, "list-layer-types") then
  log(table.concat(MPKLayers.available_layer_types(), "\n"))
  return
end
if Args.has(opts, "j" .. "son") or Args.has(opts, "bin" .. "ary") then
  die("--json/--binary supprimes : ecrire le pseudocode puis utiliser compile_mpk.lua")
end

log("=== MPK Interactive Node Wizard ===")

local name = prompt("Nom du package", "custom_model_pack")
local author = prompt("Auteur", os.getenv("USER") or "unknown")
local model_type = prompt("Type architecture registre (ex: basic_mlp, vae_conv, unet)", "basic_mlp")
local modifiable = prompt_bool("Package modifiable ?", true)
local viz_specified = prompt_bool("Section viz specifiee ?", true)
local description = prompt("Description", "Architecture nodale construite via wizard")

local base_config = gather_base_config(model_type)
local nodes, links = gather_nodes(base_config)

if #nodes == 0 then
  die("aucun layer saisi")
end

local structure = {
  template = "interactive_nodal",
  architecture = model_type,
  version = 1,
  graph = {
    mode = "node",
    nodes = nodes,
    links = links,
  },
  build = {
    dynamic_layer_assembly = true,
    generated_by = "scripts/tools/mpk_node_wizard.lua",
  },
}

local ok_graph, err_graph = MPKLayers.normalize_graph_in_place(structure)
if not ok_graph then
  die("graph validation failed: " .. tostring(err_graph))
end

local pkg, err_build = MPK.build({
  name = name,
  type = model_type,
  author = author,
  modifiable = modifiable,
  viz_specified = viz_specified,
  base_config = base_config,
  model_structure = structure,
  description = description,
})
if not pkg then
  die("build failed: " .. tostring(err_build))
end

local out_path = Args.get_str(opts, "out", "")
if out_path == "" then
  out_path = prompt("Sortie .mpk", "exports/" .. tostring(name) .. ".mpk")
end
if not tostring(out_path):lower():match("%.mpk$") then
  die("la sortie doit finir par .mpk")
end

local parent = FS.dirname(out_path)
if parent and parent ~= "" then FS.mkdir_p(parent) end

local ok_write, err_write = MPK.write(out_path, pkg)
if not ok_write then
  die("write failed: " .. tostring(err_write))
end

local compiled_path = nil
if opts.compile ~= nil and opts.compile ~= false then
  compiled_path = type(opts.compile) == "string" and opts.compile or
    (out_path .. ".bin")
  if not compiled_path:lower():match("%.mpk%.bin$") then
    die("--compile output must end with .mpk.bin")
  end
  local compiled_parent = FS.dirname(compiled_path)
  if compiled_parent and compiled_parent ~= "" then FS.mkdir_p(compiled_parent) end
  local ok_compile, err_compile = MPK.compile(out_path, compiled_path)
  if not ok_compile then die("compile failed: " .. tostring(err_compile)) end
end

log("\n[mpk_node_wizard] OK")
log("  file:      " .. tostring(out_path))
log("  type:      " .. tostring(model_type))
log("  nodes:     " .. tostring(#nodes))
log("  container: " .. tostring(pkg.container))
if compiled_path then log("  binary-v4:" .. tostring(compiled_path)) end
if type(pkg.header.checksum) == "table" then
  log("  checksum:  " .. tostring(pkg.header.checksum.algorithm) .. ":" .. tostring(pkg.header.checksum.value))
end
log("")
log("Chargement dynamique:")
log("  ./bin/mimir --lua scripts/tools/load_mpk.lua -- --in " .. tostring(out_path) .. " --create --apply-graph")
