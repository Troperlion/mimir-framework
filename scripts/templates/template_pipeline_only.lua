#!/usr/bin/env lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- ══════════════════════════════════════════════════════════════
--  TEMPLATE SCRIPT - Pipeline Only (Mímir)
--
--  Objectif:
--    - Exemple minimal d'utilisation de `scripts/modules/pipeline.lua`
--    - Sans parseur d'arguments (pas de `args.lua`)
--
--  Usage:
--    ./bin/mimir --lua scripts/templates/template_pipeline_only.lua
--
--  (Optionnel via env)
--    MIMIR_DATASET=path/to/dataset.bin   -> lance un entraînement
--    MIMIR_EPOCHS=1                      -> epochs (défaut: 1)
--    MIMIR_LR=0.0003                     -> learning rate (défaut: 0.0003)
--    MIMIR_SAVE=checkpoints/run.safetensors -> sauvegarde
-- ═════════════════════════════════════════════════════════════=

local RUN_TAG = os.getenv("MIMIR_RUN") or "pipeline_only"

-- Petit logger local: utilise `log()` si le runtime l'expose,
-- sinon retombe sur `print()`.
local function log(msg)
  (_G.log or print)(msg)
end

-- ---------------------------------------------------------------------------
-- Système: MemoryGuard + Allocator (recommandé avant build)
-- ---------------------------------------------------------------------------

-- Ce template suppose une exécution via ./bin/mimir.
-- On configure donc directement les limites mémoire du runtime.
Mimir.MemoryGuard.setLimit(10) -- 10 GB
Mimir.Allocator.configure({
  max_ram_gb = 10.0,
  enable_compression = true,
  swap_strategy = "lru",
})

-- ---------------------------------------------------------------------------
-- Pipeline
-- ---------------------------------------------------------------------------

local P = dofile(ROOTWORK.."/scripts/modules/pipeline.lua")

-- Config volontairement petite (rapide à builder)
-- L'objectif est de produire un exemple qui passe vite en build même sur une
-- machine modeste.
local cfg = {
  -- Taille du vocab si l'architecture manipule des tokens.
  vocab_size = 32000,
  -- Longueur de séquence maximale.
  seq_len = 64,
  -- Dimension cachée principale du modèle.
  d_model = 128,
  -- Nombre de blocs/couches.
  num_layers = 2,
  -- Nombre de têtes d'attention.
  num_heads = 4,
  -- Taille du MLP interne.
  mlp_hidden = 256,
  -- Dropout désactivé pour garder l'exemple simple.
  dropout = 0.0,
  -- Mode causal pour un Transformer autoregressif.
  causal = true,
}

-- Constructeur spécialisé le plus simple pour un template minimal.
local pipe = P.Transformer(cfg)

-- Options "non filtrées" par Pipeline API: on peut les injecter après coup.
-- Elles modifient le comportement du build, pas la structure de base du template.
pipe.config.init = os.getenv("MIMIR_INIT") or "xavier"
pipe.config.seed = tonumber(os.getenv("MIMIR_SEED") or "") or 1337
pipe.config.dtype = os.getenv("MIMIR_DTYPE") -- si votre runtime supporte le dtype via pipeline

-- Résumé minimal avant lancement du build.
log("\n== Pipeline (only) ==")
log("run=" .. tostring(RUN_TAG))

-- Build complet: create -> dtype -> build -> allocate -> init.
local ok_build, params_or_err = pipe:build()
if ok_build == false then
  error("build() a échoué: " .. tostring(params_or_err))
end
log("✓ build() OK | params=" .. tostring(params_or_err))

-- Entraînement optionnel (si dataset fourni)
-- Ici on passe uniquement par des variables d'environnement pour garder
-- le template sans parseur d'arguments.
local dataset_path = os.getenv("MIMIR_DATASET")
if dataset_path and dataset_path ~= "" then
  -- Hyperparamètres d'entraînement minimaux pour un run de démo.
  local epochs = tonumber(os.getenv("MIMIR_EPOCHS") or "") or 1
  local lr = tonumber(os.getenv("MIMIR_LR") or "") or 0.0003
  log("\n▶ train(dataset=" .. tostring(dataset_path) .. ", epochs=" .. tostring(epochs) .. ", lr=" .. tostring(lr) .. ")")

  -- Déclenche la boucle d'entraînement encapsulée dans le pipeline.
  local ok_train, err_train = pipe:train(dataset_path, epochs, lr)
  if ok_train == false then
    error("train() a échoué: " .. tostring(err_train))
  end
end

-- Sauvegarde optionnelle
-- Le format est déduit automatiquement depuis le chemin si besoin.
local save_path = os.getenv("MIMIR_SAVE")
if save_path and save_path ~= "" then
  log("\n💾 save(" .. tostring(save_path) .. ")")
  local ok_save, err_save = pipe:save(save_path)
  if ok_save == false then
    error("save() a échoué: " .. tostring(err_save))
  end
end

-- Fin normale du template.
log("\n✓ Terminé")
