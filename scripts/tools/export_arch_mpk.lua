---@diagnostic disable: undefined-global

-- Export a complete model architecture to MPK, from registry or standalone/current model.

dofile(ROOTWORK.."/scripts/modules/mpk_help.lua").show("export_arch_mpk")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local MPKTools = dofile(ROOTWORK.."/scripts/modules/mpk_tools.lua")
local MPKLayers = dofile(ROOTWORK.."/scripts/modules/mpk_layers.lua")

local function log(...)
  local out = {}
  for i = 1, select("#", ...) do
    out[#out + 1] = tostring(select(i, ...))
  end
  io.stdout:write(table.concat(out, " ") .. "\n")
end

local function die(msg)
  io.stderr:write("[export_arch_mpk] " .. tostring(msg) .. "\n")
  os.exit(1)
end


local function read_json_or_die(path, label)
  local obj, err = MPK.read_json_file(path)
  if type(obj) ~= "table" then
    die((label or "json") .. " read failed: " .. tostring(err or path))
  end
  return obj
end

local opts = Args.parse(arg) or {}
if Args.has(opts, "j" .. "son") or Args.has(opts, "bin" .. "ary") then
  die("--json/--binary supprimés : exportez le pseudocode puis utilisez compile_mpk.lua")
end

if type(_G.Mimir) ~= "table" or type(Mimir.Model) ~= "table" then
  die("ce script doit etre lance via ./bin/mimir")
end

local out_path = Args.get_str(opts, "out", "")
if out_path == "" then die("missing --out <file.mpk>") end
if not out_path:lower():match("%.mpk$") then die("--out must end with .mpk") end

local paths, path_err = MPKTools.output_paths(opts, out_path)
if not paths then die(path_err) end

local package_name = Args.get_str(opts, "name", "exported_arch")
local author = Args.get_str(opts, "author", "unknown")
local description = Args.get_str(opts, "description", "Architecture exportee vers MPK")
local description_file = Args.get_str(opts, "description-file", "")
if description_file ~= "" then
  local err
  description, err = MPK.read_text_file(description_file)
  if not description then die(err) end
end
local viz_specified = Args.get_bool(opts, "viz", false)

local arch = Args.get_str(opts, "arch", Args.get_str(opts, "register", ""))
local from_current = Args.get_bool(opts, "from-current-model", false)
local type_override = Args.get_str(opts, "type", "")
local config_json_path = Args.get_str(opts, "config-json", "")

local checkpoint = Args.get_str(opts, "checkpoint", "")
local checkpoint_data
local base_config = {}

local function shallow_copy_table(src)
  local out = {}
  if type(src) ~= "table" then
    return out
  end
  for k, v in pairs(src) do
    out[k] = v
  end
  return out
end

if arch ~= "" and from_current then die("--arch conflicts with --from-current-model") end
if checkpoint ~= "" then
  if arch ~= "" or from_current or config_json_path ~= "" then die("--checkpoint conflicts with --arch, --from-current-model and --config-json") end
  local err
  checkpoint_data, err = MPKTools.checkpoint(checkpoint, Args.get_str(opts, "format", "auto"))
  if not checkpoint_data then die(err) end
  base_config = checkpoint_data.config
elseif arch ~= "" then
  local cfg = {}
  if config_json_path ~= "" then
    cfg = read_json_or_die(config_json_path, "config-json")
  else
    local dc, derr = Mimir.Architectures.default_config(arch)
    if type(dc) ~= "table" then
      die("default_config(" .. tostring(arch) .. ") failed: " .. tostring(derr))
    end
    cfg = shallow_copy_table(dc)
  end

  local ok_create, err_create = Mimir.Model.create(arch, cfg)
  if ok_create == false then
    die("Model.create(" .. tostring(arch) .. ") failed: " .. tostring(err_create))
  end
  base_config = shallow_copy_table(cfg)
elseif not from_current then
  die("specifier --arch <name> ou --from-current-model")
end

local get_layers_fn = Mimir.Model.get_layers
if type(get_layers_fn) ~= "function" then
  die("Mimir.Model.get_layers indisponible")
end

local layers = checkpoint_data and checkpoint_data.layers or get_layers_fn()
if type(layers) ~= "table" or #layers == 0 then
  die("aucun layer disponible à exporter")
end

if next(base_config) == nil then
  base_config = {}
end

local nodes, links = MPKTools.build_graph_from_layers(layers)

local model_type = type_override
if model_type == "" then
  model_type = checkpoint_data and checkpoint_data.type or (arch ~= "" and arch or "custom_graph")
end

local model_structure = {
  template = "exported_full_arch",
  version = 1,
  architecture = model_type,
  build = {
    dynamic_layer_assembly = true,
    generated_by = "scripts/tools/export_arch_mpk.lua",
    source = checkpoint_data and checkpoint_data.source or ((arch ~= "") and "registry" or "current_model"),
  },
  graph = {
    mode = "node",
    nodes = nodes,
    links = links,
  },
}

local ok_graph, err_graph = MPKLayers.normalize_graph_in_place(model_structure, { allow_unknown = true })
if not ok_graph then
  die("graph validation failed: " .. tostring(err_graph))
end

local pkg, err_build = MPK.build({
  name = package_name,
  type = model_type,
  author = author,
  modifiable = Args.get_bool(opts, "modifiable", true),
  created_at = Args.get_str(opts, "created-at", nil),
  viz_specified = viz_specified,
  base_config = base_config,
  model_structure = model_structure,
  description = description,
  container = "pseudocode",
})
if not pkg then
  die("build failed: " .. tostring(err_build))
end

local ok_write, err_write = MPKTools.write_outputs(pkg, paths)
if not ok_write then die("write/compile failed: "..tostring(err_write)) end
local compiled_path = paths.compiled

log("[export_arch_mpk] OK")
log("  out:        " .. out_path)
log("  type:       " .. tostring(model_type))
log("  source:     " .. (checkpoint_data and (checkpoint_data.source..":"..checkpoint) or ((arch ~= "") and ("registry:" .. arch) or "current_model")))
log("  nodes:      " .. tostring(#nodes))
log("  container:  " .. tostring(pkg.container))
if compiled_path then log("  binary-v4:  " .. compiled_path) end
if type(pkg.header.checksum) == "table" then
  log("  checksum:   " .. tostring(pkg.header.checksum.algorithm) .. ":" .. tostring(pkg.header.checksum.value))
end
