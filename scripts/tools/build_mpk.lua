---@diagnostic disable: undefined-global

-- Build an MPK file (Mimir Package Template).
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/build_mpk.lua -- \
--     --name my_vae_pack \
--     --type vae_conv \
--     --author bri45 \
--     --description "VAEConv baseline package" \
--     --out exports/my_vae_pack.mpk

dofile(ROOTWORK.."/scripts/modules/mpk_help.lua").show("build_mpk")

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
  io.stderr:write("[build_mpk] " .. tostring(msg) .. "\n")
  os.exit(1)
end


local function read_json_or_die(path, label)
  local obj, err = MPK.read_json_file(path)
  if type(obj) ~= "table" then
    die((label or "json") .. " read failed: " .. tostring(err or path))
  end
  return obj
end

local function read_text_or_die(path, label)
  local txt, err = MPK.read_text_file(path)
  if txt == nil then
    die((label or "text") .. " read failed: " .. tostring(err or path))
  end
  return txt
end

local opts = Args.parse(arg) or {}

-- Keep the legacy template command while routing real source exports through one path.
if opts.checkpoint or opts.arch or opts.register or Args.get_bool(opts, "from-registry", false) then
  if opts.register and not opts.arch then
    arg[#arg + 1] = "--arch"
    arg[#arg + 1] = type(opts.register) == "string" and opts.register or Args.get_str(opts, "type", "")
  elseif Args.get_bool(opts, "from-registry", false) and not opts.arch then
    arg[#arg + 1] = "--arch"; arg[#arg + 1] = Args.get_str(opts, "type", "")
  end
  if opts["structure-json"] or opts.template then die("source export conflicts with --structure-json/--template") end
  return dofile(ROOTWORK.."/scripts/tools/export_arch_mpk.lua")
end
if Args.has(opts, "j" .. "son") or Args.has(opts, "bin" .. "ary") then
  die("--json/--binary removed: write pseudocode, then use compile_mpk.lua for binary output")
end

local name = Args.get_str(opts, "name", "")
local model_type = Args.get_str(opts, "type", "")
local out_path = Args.get_str(opts, "out", "")

if name == "" then die("missing --name") end
if model_type == "" then die("missing --type") end
if out_path == "" then die("missing --out") end
if not out_path:lower():match("%.mpk$") then die("--out must end with .mpk") end

local paths, path_err = MPKTools.output_paths(opts, out_path)
if not paths then die(path_err) end

local author = Args.get_str(opts, "author", "unknown")
local created_at = Args.get_str(opts, "created-at", nil)
local modifiable = Args.get_bool(opts, "modifiable", true)
local viz_specified = Args.get_bool(opts, "viz", false)
local description = Args.get_str(opts, "description", "")
local description_file = Args.get_str(opts, "description-file", "")
if description_file ~= "" then
  description = read_text_or_die(description_file, "description-file")
end

local base_config = {}
local cfg_json = Args.get_str(opts, "config-json", "")
if cfg_json ~= "" then
  base_config = read_json_or_die(cfg_json, "config-json")
end

local model_structure = {
  architecture = model_type,
  generated_by = "scripts/tools/build_mpk.lua",
}

local structure_json = Args.get_str(opts, "structure-json", "")
local template_name = Args.get_str(opts, "template", "auto")
if structure_json ~= "" then
  model_structure = read_json_or_die(structure_json, "structure-json")
elseif template_name ~= "" then
  if template_name == "auto" then
    model_structure = MPK.model_structure_template(model_type)
  else
    model_structure = MPK.model_structure_template(template_name)
  end
  model_structure.architecture = model_type
  model_structure.generated_by = "scripts/tools/build_mpk.lua"
elseif type(_G.Mimir) == "table" and type(Mimir.Architectures) == "table" then
  local entry = Mimir.Architectures.info(model_type)
  if type(entry) == "table" then
    model_structure = {
      architecture = model_type,
      registry_name = entry.name,
      registry_description = entry.description,
      default_config = entry.config,
      generated_by = "scripts/tools/build_mpk.lua",
    }
  end
end

local ok_graph, err_graph = MPKLayers.normalize_graph_in_place(model_structure)
if not ok_graph then
  die("graph validation failed: " .. tostring(err_graph))
end

local pkg, err_build = MPK.build({
  name = name,
  type = model_type,
  author = author,
  created_at = created_at,
  modifiable = modifiable,
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

log("[build_mpk] OK")
log("  out:        " .. out_path)
log("  name:       " .. tostring(pkg.header.name))
log("  type:       " .. tostring(pkg.header.type))
log("  author:     " .. tostring(pkg.header.author))
log("  created_at: " .. tostring(pkg.header.created_at))
log("  modifiable: " .. tostring(pkg.header.modifiable))
log("  viz:        " .. tostring(pkg.header.viz_specified))
log("  container:  " .. tostring(pkg.container))
if type(pkg.header.checksum) == "table" then
  log("  checksum:   " .. tostring(pkg.header.checksum.algorithm) .. ":" .. tostring(pkg.header.checksum.value))
end
log("  size(bytes):" .. tostring(pkg.header.size))
if compiled_path then log("  binary-v4:  " .. compiled_path) end
log("")
log("Registry load example:")
log("  ./bin/mimir --lua scripts/tools/load_mpk.lua -- --in " .. out_path .. " --create")
