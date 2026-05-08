# Checkpoints / reprise d’entraînement

## Deux APIs existent

- `Mimir.Serialization.*` : API “moderne” (recommandée)
- `Mimir.Checkpoint.*` : legacy (dépréciée, alias)

## Sauvegarder

```lua
local ok, err = Mimir.Serialization.save("checkpoint/run1", "raw_folder", {
  save_optimizer = true,
  save_tokenizer = true,
  save_encoder = true,
  include_checksums = true,
  include_git_info = true,
})
assert(ok, err)
```

Formats disponibles :

- `safetensors`
- `raw_folder`
- `debug_json`

## Charger

Auto-détection :

```lua
local ok, err = Mimir.Serialization.load("checkpoint/run1")
assert(ok, err)
```

Options utiles :

- `strict_mode` : rend les mismatches de shapes/clefs plus stricts.
- `validate_checksums` : vérifie les SHA256 si présents.

## Reprise (resume)

Bon pattern :

- garder un dossier fixe `out_dir`
- écrire dans des sous-dossiers `_interrupt_*` en cas de Ctrl+C
- reprendre via `scripts/modules/checkpoint_resume.lua`

VAEText : voir `scripts/training/train_vae_texte.lua`.

## Analyser un checkpoint (outil CLI Lua)

Quand vous récupérez un artefact sur disque (checkpoint d’entraînement, SafeTensors export, dossier RawFolder…), il est souvent utile de **vérifier rapidement** :

- le **format** et les **métadonnées** (version, commit, date)
- la **structure** (couches) et les **IO** (inputs/outputs) si disponible
- les **plus gros tenseurs** (pour expliquer un poids énorme / un OOM)

Le script [scripts/tools/analyze_model.lua](../../scripts/tools/analyze_model.lua) fait ça **sans charger** le modèle dans un pipeline d’entraînement : il lit les headers/JSON/manifest et affiche un résumé + tables.

### Usage

```bash
# SafeTensors
./bin/mimir --lua scripts/tools/analyze_model.lua --in model.safetensors

# RawFolder (dossier avec manifest.json)
./bin/mimir --lua scripts/tools/analyze_model.lua --in checkpoint/run1/

# DebugJson (dump d’inspection)
./bin/mimir --lua scripts/tools/analyze_model.lua --in debug.json
```

### Options utiles

- `--max-layers N` : limite l’affichage (couches + graphe)
- `--top-tensors N` : combien de tenseurs afficher (triés par taille)
- `--graph-format mermaid` : sort un bloc Markdown ` ```mermaid``` ` (pratique pour coller dans une doc)

Pour la liste complète :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua --h
```
