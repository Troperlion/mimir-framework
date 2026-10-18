#!/usr/bin/env lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

---@diagnostic disable: undefined-global

-- scripts/examples/example_conf_inference.lua
--
-- Exemple: charger une config via --conf et faire de l'inférence
--
-- Usage:
--   ./bin/mimir --conf configs/example_conf_driven.json
--
-- Notes:
-- - CONF et CONF_PATH sont injectés par le runtime (--conf mode)
-- - Ce script montre comment accéder à la config et l'utiliser pour l'inférence

local log = print

if not CONF then
    error("CONF not injected! Use --conf mode.")
end

log("")
log("═══════════════════════════════════════════════")
log("🔮 Exemple: Config-Driven Inference")
log("═══════════════════════════════════════════════")
log("")

-- Récupérer les params d'inférence depuis CONF
local inference_cfg = CONF.inference or {}
log("📖 Configuration d'inférence depuis CONF.inference:")
log("  • Temperature: " .. (inference_cfg.temperature or 0.8))
log("  • Top-K:       " .. (inference_cfg.top_k or 40))
log("  • Top-P:       " .. (inference_cfg.top_p or 0.95))

-- Récupérer le modèle (créé par template_conf_load_and_train.lua)
-- Note: dans un vrai workflow multi-script, le modèle resterait en mémoire
-- entre les appels Lua. Ici on le recrée pour l'exemple.

log("")
log("🔨 Création du modèle pour inférence...")

local model_cfg = CONF.model or {}
local arch = model_cfg.architecture or "transformer"

local cfg, cfg_err = Mimir.Architectures.default_config(arch)
if not cfg then
    error("Cannot get config for " .. arch .. ": " .. tostring(cfg_err))
end

for k, v in pairs(model_cfg) do
    if k ~= "architecture" then
        cfg[k] = v
    end
end

local model_ok, _ = Mimir.Model.create(arch, cfg)
if not model_ok then
    error("Model.create failed")
end

local alloc_ok, _ = Mimir.Model.allocate_params()
if not alloc_ok then
    error("Model.allocate_params failed")
end

local init_ok, _ = Mimir.Model.init_weights("xavier", 42)
if not init_ok then
    error("Model.init_weights failed")
end

log("  ✓ Modèle prêt")

-- Générer quelques tokens via forward pass
log("")
log("🚀 Inférence avec forward pass...")

local num_samples = 3
local seq_len = cfg.seq_len or 64

for sample_id = 1, num_samples do
    local input_ids = {}
    for i = 1, math.min(10, seq_len) do
        table.insert(input_ids, math.random(1, math.min(100, cfg.vocab_size or 1000)))
    end
    
    local forward_ok, output = Mimir.Model.forward({ __input__ = input_ids }, false)
    if forward_ok and output then
        log("  Sample " .. sample_id .. ": output length = " .. (#output or "?"))
    else
        log("  Sample " .. sample_id .. ": forward failed")
    end
end

log("")
log("✅ Inférence terminée!")
log("")
