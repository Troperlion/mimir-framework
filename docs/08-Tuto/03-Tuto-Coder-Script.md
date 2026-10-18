# Écrire un script Lua

Ce tutoriel construit un modèle `basic_mlp` avec la Pipeline API actuelle,
sans dataset. Il explique aussi la séparation entre les arguments du binaire
et ceux du script.

## Sources de vérité

- `scripts/modules/args.lua`
- `scripts/modules/pipeline.lua`
- `scripts/modules/pipeline_api.lua`
- `scripts/templates/template_pipeline_args.lua`
- `src/scriptings/Lua/luaScripting/`

`pipeline.lua` est un module de compatibilité qui retourne actuellement
`pipeline_api.lua`. La logique de la Pipeline API se trouve donc dans ce
dernier fichier.

## Étape 1 — Choisir le bon point de départ

- `template_new_model.lua` montre l'API `Mimir.*` de bas niveau ;
- `template_pipeline_only.lua` montre les constructeurs spécialisés ;
- `template_pipeline_args.lua` montre `args.lua` et le mode registry-first.

Pour une nouvelle architecture présente dans le registre, préférez
`P.FromRegistry`. Cela évite de modifier `pipeline_api.lua` uniquement pour
ajouter un constructeur spécialisé.

## Étape 2 — Comprendre les arguments

La forme de lancement est :

```bash
./bin/mimir --lua chemin/script.lua -- arguments_du_script
```

Le premier `--` termine les options de `mimir`. `args.lua` analyse ensuite la
table globale `arg`.

```lua
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local arch = Args.get_str(opts, "arch", "basic_mlp")
local seed = Args.get_int(opts, "seed", 1337)
local save = Args.get_str(opts, "save", nil)
```

`Args.get_bool` comprend les formes `--flag` et `--no-flag`.
`Args.apply_overrides` applique les occurrences
`--override chemin.valeur=contenu`.

## Étape 3 — Construire depuis le registre

Ce script utilise uniquement des méthodes présentes dans
`pipeline_api.lua` :

```lua
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local P = dofile(ROOTWORK.."/scripts/modules/pipeline.lua")

local opts = Args.parse(arg) or {}
local arch = Args.get_str(opts, "arch", "basic_mlp")

local pipe, err = P.FromRegistry(arch)
if not pipe then
    error("FromRegistry: " .. tostring(err))
end

local ok, cfg_or_err = pipe:loadDefaultConfig(arch)
if ok == false then
    error("loadDefaultConfig: " .. tostring(cfg_or_err))
end

local patch = {
    input_dim = Args.get_int(opts, "input-dim", 16),
    hidden_dim = Args.get_int(opts, "hidden-dim", 32),
    output_dim = Args.get_int(opts, "output-dim", 8),
    hidden_layers = Args.get_int(opts, "layers", 2),
}

ok, cfg_or_err = pipe:patchConfig(patch)
if ok == false then
    error("patchConfig: " .. tostring(cfg_or_err))
end

pipe.config.init = Args.get_str(opts, "init", "xavier")
pipe.config.seed = Args.get_int(opts, "seed", 1337)

local params_or_err
ok, params_or_err = pipe:build()
if ok == false then
    error("build: " .. tostring(params_or_err))
end

local save = Args.get_str(opts, "save", nil)
if save and save ~= "" then
    local save_err
    ok, save_err = pipe:save(save)
    if ok == false then
        error("save: " .. tostring(save_err))
    end
end

print("Modèle construit, paramètres:", params_or_err)
```

`pipe:build()` appelle actuellement :

1. `Mimir.Model.create` ;
2. la sélection du dtype si configuré ;
3. `Mimir.Model.build`, qui est un no-op moderne conservé pour compatibilité ;
4. `Mimir.Model.allocate_params` ;
5. `Mimir.Model.init_weights`.

## Étape 4 — Utiliser le template fourni

Le template existant expose un ensemble différent de flags, centré sur les
configurations Transformer :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- --help

./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry \
  --arch basic_mlp \
  --no-train \
  --save /tmp/mimir_script_tutorial.safetensors
```

Les champs Transformer ajoutés par le template sont fusionnés dans la
configuration. `basic_mlp` ignore ceux qui ne sont pas consommés par son
parser, mais un script spécialisé devrait éviter de les ajouter.

## Étape 5 — Entraînement et inférence

`pipe:train(dataset_path, epochs, lr)` charge réellement le chemin via
`Mimir.Dataset.load`. Ne l'appelez pas pour un simple test de construction.

La méthode générique `pipe:infer` de `FromRegistry` délègue à
`Mimir.Model.infer`, qui est un chemin historique prenant une entrée de type
chaîne. Elle ne remplace pas un `Mimir.Model.forward` numérique correctement
dimensionné. Pour un MLP ou une architecture multi-entrée, utilisez l'API
`forward` et respectez son contrat de tenseurs.

## Règles de robustesse

- Vérifiez les couples `(ok, erreur)` retournés par l'API.
- Ne supposez pas qu'une option est disponible : lancez `--help`.
- Utilisez `scripts/modules/fs.lua` plutôt que des commandes shell pour les
  opérations de fichiers portables.
- Sauvegardez seulement après un `build` ou un chargement réussi.
- N'initialisez jamais de nouveau les poids après le chargement d'un
  checkpoint.

## Étapes suivantes

- [Du registre au checkpoint](08-Tuto-Registre-Pipeline-Checkpoint.md)
- [Scripting Lua](../02-User-Guide/06-Lua-Scripting.md)
- [Référence de `Mimir.Model`](../03-API-Reference/10-Model.md)
