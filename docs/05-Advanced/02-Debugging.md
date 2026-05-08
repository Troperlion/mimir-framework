# Debugging & stabilité numérique

Objectif : isoler rapidement si le problème vient de la **mémoire**, de la **stabilité numérique**, ou d’un **chemin API** (Lua → C++).

Point d’entrée conseillé : [04-Source-Code-Map.md](04-Source-Code-Map.md).

## 1) Symptômes fréquents

- loss = NaN / Inf
- poids/gradients non-finis
- OOM, ou process “kill” (OS)
- perf qui s’effondre après quelques steps (allocations qui explosent)

## 2) Checklist rapide (Lua)

1) **Activer un garde-fou mémoire** (strict) :
	 - `Mimir.MemoryGuard.setLimit(gb)` (alias moderne)
	 - ou `Mimir.Guard.setLimit(gb)` (alias)
	 - Note : si la valeur est <= 1000, c’est interprété en **GB**, sinon en **bytes** (implémenté dans `src/LuaScripting.cpp`).
2) **Configurer l’allocateur dynamique** : `Mimir.Allocator.configure({max_ram_gb = ..., enable_compression = true})`
3) Démarrer petit : réduire `seq_len`, `d_model`, `num_layers`, batch.
4) Sur NaN : réduire LR, activer un clipping global.

## 3) Comprendre la mémoire (3 “couches”)

Le repo a plusieurs mécanismes, avec des rôles distincts :

- `MemoryGuard` (strict) : compte/refuse certaines allocations (voir `src/MemoryGuard.hpp`).
	- API Lua : `Mimir.MemoryGuard.*` / `Mimir.Guard.*`.
- `DynamicTensorAllocator` : backend d’allocations pour les tenseurs “dynamiques” (voir `src/DynamicTensorAllocator.*`, utilisé depuis `src/tensors.cpp`).
	- API Lua : `Mimir.Allocator.*`.
- `AdvancedRAMManager` : gestionnaire best-effort type cache/compression/spill (voir `src/AdvancedRAMManager.hpp`).
	- API Lua : `Mimir.Memory.*`.

En pratique : si vous voulez un comportement **fail-fast** et reproductible, c’est `MemoryGuard` + alloc dynamique qui comptent.

## 4) NaN / Inf : ce que le runtime protège déjà

Dans l’optimizer, il y a un garde-fou explicite :

- Si `opt.eps` n’est pas fini ou <= 0, il est forcé à `1e-8` (voir `Model::optimizerStep` dans `src/Model.cpp`).

Et un clipping global optionnel, via config JSON :

- `grad_clip_norm` (ou alias `clip_norm`) : applique un clipping L2 global sur `grad_weights` et `grad_bias` avant la mise à jour (dans `src/Model.cpp`).

Si vous avez encore des NaN :

- baissez le LR (souvent la cause #1)
- vérifiez que vos entrées sont dans des plages raisonnables (normalisation)
- réduisez la taille du modèle (explosion de gradients plus fréquente)

## 5) OOM / allocation inattendue

À faire :

- Afficher les stats guard : `Mimir.MemoryGuard.printStats()` (ou `Mimir.Guard.printStats()`).
- Lire les stats sous forme de table : `Mimir.MemoryGuard.getStats()` (ou `Mimir.Guard.getStats()`).
- Lire usage/peak/limit directement : `Mimir.MemoryGuard.getCurrentUsage()`, `getPeakUsage()`, `getLimit()`.
- Afficher les stats allocateur : `Mimir.Allocator.printStats()`.
- Si vous utilisez `Mimir.Memory.*`, comparez avec `Mimir.Memory.printStats()` (ça ne couvre pas forcément les mêmes allocations).

Point important : certaines allocations “non-guardées” peuvent exister selon les chemins (ex: `std::vector`), donc un OOM OS peut survenir même si le guard paraît “OK”.

## 6) Debug JSON (état modèle, gradients)

Important : `debug_json` / `save_enhanced_debug` sont des **dumps de contrôle** destinés à l’inspection humaine (triage NaN/OOM, vérification des stats, snapshots avant/après).
Ce n’est **pas** un format de sérialisation “stable” ni un format **rechargeable** par le framework pour reprendre un entraînement.

Pour un artefact reloadable, utilisez les chemins de sérialisation/checkpoint (ex: SafeTensors / checkpoints) plutôt que le debug JSON.

Deux entrées utiles (implémentées côté C++ via `src/Serialization/DebugJsonDump.*` et exposées Lua) :

- `Mimir.Serialization.save("debug.json", "debug_json", { ... })`
- `Mimir.Serialization.save_enhanced_debug("debug.json", { include_gradients = true, ... })`

Options “enhanced” réellement supportées côté C++ (voir `src/Serialization/DebugJsonDump.hpp`) :

- `include_gradients` (bool)
- `include_optimizer_state` (bool)
- `max_values_per_tensor` (int, défaut 20) : limite de l’échantillonnage par tenseur
- `include_activations` (bool)
- `include_checksums` (bool)
- `include_weight_deltas` (bool)
- `include_git_info` (bool, défaut true)
- `save_tokenizer` (bool)
- `save_encoder` (bool)

Usage conseillé :

- faire un dump juste avant le step qui casse
- activer `include_gradients` uniquement si nécessaire (volume plus gros)
- comparer deux dumps successifs (avant/après) sur min/max/mean

## 7) Inspecter un artefact sur disque (analyze_model.lua)

But : quand vous avez un fichier/dossier “checkpoint” mais que vous ne voulez pas (ou ne pouvez pas) relancer un pipeline, le script [scripts/tools/analyze_model.lua](../../scripts/tools/analyze_model.lua) permet de **trier rapidement** :

- est-ce un SafeTensors, un RawFolder, ou un DebugJson ?
- quelle architecture/couches sont déclarées ?
- quels tenseurs dominent la taille disque ?
- comment visualiser le flux (IO) sous forme de graphe ?

### Fonctionnement (résumé)

- **SafeTensors** : lit le header, liste les tensors (`dtype`, `shape`, offsets), et tente de décoder `model/architecture_json` si présent.
- **RawFolder** : lit `manifest.json`, estime la taille des `.bin`, et peut enrichir les N plus gros via les sidecars JSON.
- **DebugJson** : lit le dump (v1.0 “mimir_debug_dump” ou v1.1 enhanced) et reconstruit un résumé (couches + tensors si présents).

### Exemples

```bash
# 1) SafeTensors
./bin/mimir --lua scripts/tools/analyze_model.lua --in model.safetensors

# 2) RawFolder
./bin/mimir --lua scripts/tools/analyze_model.lua --in checkpoint/run1/

# 3) DebugJson
./bin/mimir --lua scripts/tools/analyze_model.lua --in debug.json
```

### Graphe Mermaid (Markdown)

Pour coller un graphe directement dans un README/rapport, utilisez :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua --in model.safetensors --graph-format mermaid
```

Ça émet un bloc :

- ` ```mermaid` + `flowchart LR` + noeuds/arêtes

Note : les dumps DebugJson n’embarquent pas forcément `inputs`/`output` par couche ; dans ce cas le rendu Mermaid utilise un **fallback linéaire** (layer1 → layer2 → …) pour rester lisible.

### Help du script

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua --h
```
