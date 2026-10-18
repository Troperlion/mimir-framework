---@diagnostic disable: undefined-global

-- Load an MPK file and optionally instantiate the model through registry.
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/model.mpk --create

dofile(ROOTWORK.."/scripts/modules/mpk_help.lua").show("load_mpk")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
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
  io.stderr:write("[load_mpk] " .. tostring(msg) .. "\n")
  os.exit(1)
end


local function is_table(v)
  return type(v) == "table"
end

local function apply_graph(decoded, replace_layers)
  if not is_table(decoded) then
    return false, "decoded payload manquant"
  end
  local ms = decoded.model_structure
  if not is_table(ms) then
    return false, "model_structure manquant"
  end
  local graph = ms.graph
  if not is_table(graph) then
    return false, "model_structure.graph manquant"
  end
  local nodes = graph.nodes
  if type(nodes) ~= "table" or #nodes == 0 then
    return false, "graph.nodes vide"
  end

  if replace_layers then
    if type(Mimir.Model.clear_layers) ~= "function" then
      return false, "Mimir.Model.clear_layers indisponible"
    end
    local ok_clear, cleared_or_err = Mimir.Model.clear_layers()
    if ok_clear == false then
      return false, "clear_layers failed: " .. tostring(cleared_or_err)
    end
  end

  for i = 1, #nodes do
    local n = nodes[i]
    if type(n) == "table" then
      local name = tostring(n.name or n.id or ("layer_" .. i))
      local ltype, err_type = MPKLayers.canonical_layer_type(n.type or "linear", n.params)
      if not ltype then
        return false, "invalid layer type for " .. name .. ": " .. tostring(err_type)
      end
      local params_count = math.floor(tonumber(n.params_count) or 0)

      local params = type(n.params) == "table" and n.params or {}
      local ok_push, err_push =
        Mimir.Model.push_layer(name, ltype, params_count, params)
      if ok_push == false then
        return false, "push_layer failed for " .. name .. ": " .. tostring(err_push)
      end

      local inputs = n.inputs
      if type(inputs) ~= "table" then inputs = { "x" } end
      local output = tostring(n.output or (name .. "_out"))
      local ok_io, err_io = Mimir.Model.set_layer_io(name, inputs, output)
      if ok_io == false then
        return false, "set_layer_io failed for " .. name .. ": " .. tostring(err_io)
      end
    end
  end

  return true
end

local opts = Args.parse(arg) or {}

local in_path = Args.get_str(opts, "in", "")
if in_path == "" then
  die("missing --in <file.mpk>")
end

local do_create = Args.get_bool(opts, "create", true)
local show_config = Args.get_bool(opts, "show-config", false)
local verify_only = Args.get_bool(opts, "verify-only", false)
local apply_graph_flag = Args.get_bool(opts, "apply-graph", false)
local replace_layers = Args.get_bool(opts, "replace-layers", true)
local allow_non_registry = Args.get_bool(opts, "allow-non-registry", true)
local allocate_graph = Args.get_bool(opts, "allocate", true)
local init_method = Args.get_str(opts, "init", "xavier")
local init_seed = math.floor(tonumber(Args.get_str(opts, "seed", "0")) or 0)

local pkg, err_pkg = MPK.read(in_path)
if not pkg then
  die("read failed: " .. tostring(err_pkg))
end

local decoded, err_dec = MPK.decode_payload(pkg)
if not decoded then
  die("decode failed: " .. tostring(err_dec))
end

local graph_nodes = 0
if type(decoded.model_structure) == "table" and
    type(decoded.model_structure.graph) == "table" then
  local ok_graph, err_graph =
    MPKLayers.normalize_graph_in_place(decoded.model_structure)
  if not ok_graph then
    die("graph validation failed: " .. tostring(err_graph))
  end
  graph_nodes = #(decoded.model_structure.graph.nodes or {})
end

local ok_sig, err_sig = MPK.verify_checksum(pkg)
if not ok_sig then
  die("checksum invalid: " .. tostring(err_sig))
end

log("[load_mpk] OK")
log("  file:       " .. tostring(in_path))
log("  name:       " .. tostring(pkg.header.name))
log("  type:       " .. tostring(pkg.header.type))
log("  author:     " .. tostring(pkg.header.author))
log("  created_at: " .. tostring(pkg.header.created_at))
log("  modifiable: " .. tostring(pkg.header.modifiable))
log("  viz:        " .. tostring(pkg.header.viz_specified))
log("  container:  " .. tostring(pkg.container or "json"))
if type(pkg.header.checksum) == "table" then
  log("  checksum:   " .. tostring(pkg.header.checksum.algorithm) .. ":" .. tostring(pkg.header.checksum.value) .. " (verified)")
end
log("  size(bytes):" .. tostring(pkg.header.size))
log("  description:" .. tostring(decoded.description or ""))
log("  graph_nodes:" .. tostring(graph_nodes))

if show_config then
  log("  base_config:")
  log(MPK.encode_json(decoded.base_config))
end

if verify_only then
  return
end

if not do_create then
  return
end

if type(_G.Mimir) ~= "table" or type(Mimir.Model) ~= "table" then
  die("--create requires running through ./bin/mimir")
end

local full_cfg, err_cfg = MPK.to_registry_full_config(pkg)
if not full_cfg then
  die("registry config error: " .. tostring(err_cfg))
end

-- Chemin principal : laisser Mimir.Model.create charger le MPK, le convertir
-- en configuration complète puis déléguer sa création au registre C++.
local ok, arch_or_err = Mimir.Model.create(in_path)
local created_via_registry = (ok ~= false)

if created_via_registry then
  log("[load_mpk] MPK loaded directly via model registry")
  log("  architecture: " .. tostring(full_cfg.architecture))
else
  if not allow_non_registry then
    die("Mimir.Model.create(MPK) failed: " .. tostring(arch_or_err))
  end

  if type(Mimir.Model.create_empty) ~= "function" then
    die("fallback non-registry impossible: Mimir.Model.create_empty indisponible")
  end

  local fallback_type = tostring(full_cfg.architecture or (pkg.header and pkg.header.type) or "custom_graph")
  local ok_empty, err_empty = Mimir.Model.create_empty(fallback_type, full_cfg.model or {})
  if ok_empty == false then
    die("Model.create_empty failed: " .. tostring(err_empty))
  end

  log("[load_mpk] registry architecture not found, fallback to standalone graph mode")
  log("  requested_type: " .. fallback_type)

  if not apply_graph_flag then
    log("[load_mpk] note: graph non applique (ajouter --apply-graph pour reconstruction complete)")
  end
end

if apply_graph_flag then
  if graph_nodes == 0 then
    die("--apply-graph requested but package has no node graph")
  end
  local ok_graph, err_graph = apply_graph(decoded, replace_layers)
  if not ok_graph then
    die("graph apply failed: " .. tostring(err_graph))
  end
  log("[load_mpk] dynamic graph applied")
  if allocate_graph then
    local ok_alloc, alloc_or_err = Mimir.Model.allocate_params()
    if ok_alloc == false then
      die("graph allocation failed: " .. tostring(alloc_or_err))
    end
    log("[load_mpk] dynamic graph parameters allocated")
    if init_method ~= "" and init_method:lower() ~= "none" then
      local ok_init, init_err = Mimir.Model.init_weights(init_method, init_seed)
      if ok_init == false then
        die("graph initialization failed: " .. tostring(init_err))
      end
      log("[load_mpk] dynamic graph initialized: " ..
          init_method .. " seed=" .. tostring(init_seed))
    end
  end
end
