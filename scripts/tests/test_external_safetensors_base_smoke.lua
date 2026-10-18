#!/usr/bin/env lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

---@diagnostic disable: undefined-field, need-check-nil

local function die(msg)
  log("❌ " .. tostring(msg))
  os.exit(1)
end

local function ok_or_die(ok, err, ctx)
  if ok then return end
  die((ctx or "operation") .. ": " .. tostring(err or "unknown"))
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  ok_or_die(ok, dt_or_err, "Mimir.model.dtype(" .. tostring(dtype) .. ")")
end

local checkpoint_path = os.getenv("MIMIR_EXTERNAL_SAFETENSORS") or "../model.safetensors"
local include_prefix = os.getenv("MIMIR_EXTERNAL_SAFETENSORS_INCLUDE") or "conditioner.embedders.1."
local max_tensors = tonumber(os.getenv("MIMIR_MAX_TENSORS") or "16") or 16

log("🧪 external_safetensors_base smoke test")
log("  - checkpoint: " .. tostring(checkpoint_path))
log("  - include_prefix: " .. tostring(include_prefix))
log("  - max_tensors: " .. tostring(max_tensors))

ok_or_die(Mimir.MemoryGuard.setLimit(24), nil, "MemoryGuard.setLimit")
ok_or_die(Mimir.Allocator.configure({
  max_ram_gb = 24.0,
  enable_compression = true,
  swap_strategy = "lru",
}), nil, "Allocator.configure")

local cfg = {
  source_safetensors = checkpoint_path,
  include_prefixes = { include_prefix },
  max_tensors = max_tensors,
  dtype = os.getenv("MIMIR_DTYPE") or "float16",
}

local ok_create, err_create = Mimir.Model.create("external_safetensors_base", cfg)
ok_or_die(ok_create, err_create, "Model.create(external_safetensors_base)")
apply_dtype(cfg)

local ok_params, err_params = Mimir.Model.allocate_params()
ok_or_die(ok_params, err_params, "Model.allocate_params")

local ok_load, err_load = Mimir.Serialization.load(checkpoint_path, "safetensors", {
  load_tokenizer = false,
  load_encoder = false,
  load_optimizer = false,
})
ok_or_die(ok_load, err_load, "Serialization.load")

log("✅ OK: external_safetensors_base a chargé le checkpoint via noms exacts")