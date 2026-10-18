#!/usr/bin/env lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

---@diagnostic disable: undefined-field, need-check-nil

-- Smoke test sérialisation (SafeTensors) — Mímir Framework v3.5.0
-- Objectif: valider rapidement la surface API Mimir.Serialization.save/load.
-- Usage:
--   ./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

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
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  ok_or_die(ok, dt_or_err, "Mimir.model.dtype(" .. tostring(dtype) .. ")")
  return true
end

-- ---------------------------------------------------------------------------
-- 1) Sécurité mémoire (recommandé)
-- ---------------------------------------------------------------------------

ok_or_die(Mimir.MemoryGuard.setLimit(2), nil, "MemoryGuard.setLimit")

local ok_alloc, err_alloc = Mimir.Allocator.configure({
  max_ram_gb = 2.0,
  enable_compression = true,
  swap_strategy = "lru",
})
ok_or_die(ok_alloc, err_alloc, "Allocator.configure")

-- ---------------------------------------------------------------------------
-- 2) Construire un mini-modèle
-- ---------------------------------------------------------------------------

local model_type = "transformer"
local cfg = {
  -- valeurs petites pour que ça passe partout
  seq_len = 16,
  d_model = 64,
  num_layers = 2,
  num_heads = 4,
  mlp_hidden = 128,
  vocab_size = 256,
}

local ok_create, err_create = Mimir.Model.create(model_type, cfg)
ok_or_die(ok_create, err_create, "Model.create")
apply_dtype(cfg)

local ok_build, err_build = Mimir.Model.build()
ok_or_die(ok_build, err_build, "Model.build")

local ok_params, err_params = Mimir.Model.allocate_params()
ok_or_die(ok_params, err_params, "Model.allocate_params")

local ok_init, err_init = Mimir.Model.init_weights("xavier", 42)
ok_or_die(ok_init, err_init, "Model.init_weights")

-- ---------------------------------------------------------------------------
-- 3) Save/Load (SafeTensors)
-- ---------------------------------------------------------------------------

local out_path = "/tmp/mimir_serialization_smoke_v2.4.0.safetensors"
log("🧪 Sérialisation smoke test")
log("  - path: " .. out_path)

local ok_save, err_save = Mimir.Serialization.save(out_path, "safetensors", {
  save_tokenizer = false,
  save_encoder = false,
  save_optimizer = false,
  include_git_info = true,
})
ok_or_die(ok_save, err_save, "Serialization.save")

-- Re-créer le modèle pour valider que load ne dépend pas uniquement de l'état précédent.
local ok_create2, err_create2 = Mimir.Model.create(model_type, cfg)
ok_or_die(ok_create2, err_create2, "Model.create (reload)")
apply_dtype(cfg)

ok_or_die(Mimir.Model.build(), nil, "Model.build (reload)")
ok_or_die(Mimir.Model.allocate_params(), nil, "Model.allocate_params (reload)")

local ok_load, err_load = Mimir.Serialization.load(out_path, "safetensors", {
  load_tokenizer = false,
  load_encoder = false,
  load_optimizer = false,
})
ok_or_die(ok_load, err_load, "Serialization.load")

log("✅ OK: save/load SafeTensors")
