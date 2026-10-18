-- #!./bin/mimir --lua
-- ══════════════════════════════════════════════════════════════
--  TEMPLATE SCRIPT - Args + Pipeline (Mímir)
--
--  Objectif:
--    - Utiliser `scripts/modules/args.lua` (flags + overrides)
--    - Piloter un pipeline via `scripts/modules/pipeline.lua`
--
--  Usage:
--    ./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
--      --dataset dataset_2/tokenizer.bin \
--      --epochs 1 --lr 0.0003 \
--      --save checkpoints/my_run.safetensors \
--      --d-model 256 --layers 4 --heads 8 --seq-len 128 \
--      --override mlp_hidden=1024 \
--      --viz
--
--  Exemple registry-first (nouveau):
--    ./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
--      --from-registry --arch transformer \
--      --d-model 256 --layers 4 --heads 8 --seq-len 128 \
--      --override mlp_hidden=1024
--
--  Notes:
--    - `--override a.b.c=value` supporte bool/number/null/string.
--    - `args.lua` supporte `--no-flag` (ex: `--no-train`).
-- ═════════════════════════════════════════════════════════════=

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local P = dofile(ROOTWORK.."/scripts/modules/pipeline.lua")

-- Parse tous les flags CLI dès le début.
-- `args.lua` gère aussi les booléens, alias et `--override`.
local opts = Args.parse(arg) or {}

-- Petit logger local: passe par `log()` si le runtime le fournit,
-- sinon retombe simplement sur `print()`.
local function log(msg)
  (_G.log or print)(msg)
end

-- ---------------------------------------------------------------------------
-- Système: MemoryGuard + Allocator (optionnels)
-- ---------------------------------------------------------------------------

do
  -- Limite mémoire globale pour le runtime.
  local mem_gb = Args.get_num(opts, "mem-gb", 10)
  -- Limite propre à l'allocateur dynamique.
  local alloc_gb = Args.get_num(opts, "alloc-gb", mem_gb)
  -- Compression des blocs mémoire si supportée/utile.
  local compression = Args.get_bool(opts, "compression", true)

  -- Ce template suppose qu'il tourne sous ./bin/mimir.
  -- On configure donc directement les garde-fous mémoire du framework.
  Mimir.MemoryGuard.setLimit(mem_gb)
  Mimir.Allocator.configure(
    {
      max_ram_gb = alloc_gb,
      enable_compression = compression,
      swap_strategy = "lru"
    }
  )
end

-- ---------------------------------------------------------------------------
-- Config pipeline (Transformer) + overrides
-- ---------------------------------------------------------------------------

-- Config utilisateur minimale. En mode registry-first, elle sera fusionnée
-- au-dessus de la config par défaut de l'architecture demandée.
local cfg = {
  -- Taille du vocab si l'architecture manipule du texte/token ids.
  vocab_size = Args.get_int(opts, "vocab", 32000),
  -- Longueur de séquence maximale.
  seq_len = Args.get_int(opts, "seq-len", 64),
  -- Dimension cachée principale du modèle.
  d_model = Args.get_int(opts, "d-model", 128),
  -- Profondeur du réseau / nombre de blocs.
  num_layers = Args.get_int(opts, "layers", 2),
  -- Nombre de têtes d'attention.
  num_heads = Args.get_int(opts, "heads", 4),
  -- Taille du MLP interne (FFN).
  mlp_hidden = Args.get_int(opts, "mlp", 256),
  -- Dropout générique si l'architecture le supporte.
  dropout = Args.get_num(opts, "dropout", 0.0),
  -- Pour les archis Transformer-like: causal=true pour mode autoregressif.
  causal = Args.get_bool(opts, "causal", true)
}

-- Applique les éventuels overrides CLI de haut niveau.
cfg = Args.apply_overrides(cfg, opts)

-- `--from-registry` force l'utilisation du registre d'architectures du framework.
local use_registry = Args.get_bool(opts, "from-registry", false)
-- `--arch` permet de choisir l'architecture à demander au registre.
local arch = Args.get_str(opts, "arch", "transformer")

local pipe
if use_registry then
  -- Mode générique: on part de la config par défaut du registre, puis on applique
  -- les flags/overrides fournis par l'utilisateur.
  local p, errp = P.FromRegistry(arch)
  if not p then
    error("FromRegistry(" .. tostring(arch) .. ") a échoué: " .. tostring(errp))
  end
  pipe = p

  -- Recharge explicitement la config de base du registre dans l'objet pipeline.
  local ok_base, base_or_err = pipe:loadDefaultConfig(arch)
  if ok_base == false then
    error("loadDefaultConfig(" .. tostring(arch) .. ") a échoué: " .. tostring(base_or_err))
  end

  -- Fusionne ensuite notre config locale au-dessus de cette base.
  local ok_patch, patch_or_err = pipe:patchConfig(cfg)
  if ok_patch == false then
    error("patchConfig() a échoué: " .. tostring(patch_or_err))
  end
else
  -- Mode spécialisé: constructeur dédié, pratique pour les exemples ciblés.
  -- Ici le template reste centré sur Transformer pour garder un exemple compact.
  pipe = P.Transformer(cfg)
end

-- Quelques options runtime restent pilotées ici, indépendamment de l'architecture.
-- Elles sont utiles pour le build mais ne font pas partie des champs "métier"
-- de la config utilisateur minimale ci-dessus.
pipe.config.init = Args.get_str(opts, "init", "xavier")
pipe.config.seed = Args.get_int(opts, "seed", 1337)
pipe.config.dtype = Args.get_str(opts, "dtype", nil)

-- Petit identifiant lisible pour les logs/exports éventuels.
local run_tag = Args.get_str(opts, "run", "pipeline_args")

-- Affiche le mode réellement choisi avant de lancer le build.
log("\n== Pipeline (args) ==")
log("run=" .. tostring(run_tag))
if use_registry then
  log("mode=registry-first arch=" .. tostring(arch))
else
  log("mode=specialized-constructor (Transformer)")
end

-- Build du pipeline: création du modèle, application du dtype,
-- construction des layers, allocation des paramètres et initialisation.
local ok_build, params_or_err = pipe:build()
if ok_build == false then
  error("build() a échoué: " .. tostring(params_or_err))
end
log("✓ build() OK | params=" .. tostring(params_or_err))

-- ---------------------------------------------------------------------------
-- Train (optionnel)
-- ---------------------------------------------------------------------------

local do_train = Args.get_bool(opts, "train", true)
if do_train then
  -- Le dataset est facultatif ici: ce template peut servir uniquement à tester
  -- la config et le build d'une architecture.
  local dataset_path = Args.get_str(opts, "dataset", nil)
  if dataset_path == nil or dataset_path == "" then
    log("ℹ️  Pas de --dataset => skip train() (utilisez --no-train pour enlever ce message)")
  else
    -- Hyperparamètres d'entraînement minimaux exposés par le template.
    local epochs = Args.get_int(opts, "epochs", 1)
    local lr = Args.get_num(opts, "lr", 0.0003)

    -- Le template garde volontairement un train() très simple: dataset + epochs + lr.
    log(
      "\n▶ train(dataset=" ..
        tostring(dataset_path) .. ", epochs=" .. tostring(epochs) .. ", lr=" .. tostring(lr) .. ")"
    )

    -- Déclenche la boucle d'entraînement fournie par le pipeline.
    local ok_train, err_train = pipe:train(dataset_path, epochs, lr)
    if ok_train == false then
      error("train() a échoué: " .. tostring(err_train))
    end
  end
end

-- ---------------------------------------------------------------------------
-- Save (optionnel)
-- ---------------------------------------------------------------------------

local save_path = Args.get_str(opts, "save", nil)
if save_path ~= nil and save_path ~= "" then
  -- Sauvegarde optionnelle du modèle construit/entraîné.
  -- Le pipeline choisit automatiquement le format selon le chemin de sortie.
  log("\n💾 save(" .. tostring(save_path) .. ")")
  local ok_save, err_save = pipe:save(save_path)
  if ok_save == false then
    error("save() a échoué: " .. tostring(err_save))
  end
end

-- Sortie normale du script.
log("\n✓ Terminé")
