#!/usr/bin/env mimir --lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()


-- Vérifie (via logs C++ MIMIR_ACCEL_VERBOSE=1) que Conv2d prend le fast path AVX2+FMA.
-- Usage:
--   MIMIR_ACCEL_VERBOSE=1 ./bin/mimir --lua scripts/debug/check_fma_conv2d.lua

math.randomseed(123)

local model = Mimir.Model

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  assert(ok ~= false, tostring(dt_or_err or "Model.dtype failed"))
  return true
end

if model.set_hardware then
  pcall(model.set_hardware, true)
end

-- Petit VGG16 pour déclencher Conv2d sans coût énorme.
local cfg = {
  image_w = 16,
  image_h = 16,
  image_c = 3,
  base_channels = 8,
  num_classes = 10,
  fc_hidden = 32,
}

local ok, err = model.create("vgg16", cfg)
assert(ok ~= false, tostring(err or "Model.create(vgg16) failed"))
apply_dtype(cfg)

assert(model.allocate_params())
assert(model.init_weights("xavier", 123))

local n = 16 * 16 * 3
local x = {}
for i = 1, n do
  x[i] = (math.random() * 2.0 - 1.0)
end

local y, yerr = model.forward(x, false)
assert(y ~= false and y ~= nil, tostring(yerr or "Model.forward failed"))

log("✓ forward ok | out_size=" .. tostring(#y))
