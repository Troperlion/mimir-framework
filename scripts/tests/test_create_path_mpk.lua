---@diagnostic disable: undefined-global

local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")

local tmp = os.tmpname() .. ".mpk"
local pkg = MPK.build({
  name = "path_plugin_smoke",
  type = "basic_mlp",
  author = "test",
  description = "smoke create(path.mpk)",
  base_config = { input_dim = 2, hidden_dim = 4, output_dim = 1, hidden_layers = 1, dropout = 0.0 },
  model_structure = MPK.model_structure_template("basic_mlp"),
  container = "pseudocode",
})
assert(pkg, "MPK.build failed")

local ok_write, err_write = MPK.write(tmp, pkg, { binary = false })
assert(ok_write, tostring(err_write))

local ok = Mimir.Model.create(tmp)
assert(ok == true, "Mimir.Model.create(path) failed")

print("create(path.mpk) OK: " .. tmp)
