# Documentation Mímir (réécrite)

Version framework : **2.4.0**  
Révision documentation : **2026-03-15**

Cette documentation remplace l’ancienne doc (archivée dans [docs_archive/2026-02-14/](../docs_archive/2026-02-14/)).

Si tu as l’impression que “la doc n’explique rien”, commence par les 3 pages ci-dessous :

- [docs/01-Getting-Started/01-Quick-Start.md](01-Getting-Started/01-Quick-Start.md) (faire tourner un script + forward)
- [docs/02-User-Guide/02-Model-Lifecycle.md](02-User-Guide/02-Model-Lifecycle.md) (ordre des appels et pourquoi)
- [docs/03-API-Reference/19-Globals.md](03-API-Reference/19-Globals.md) (ce que le runtime injecte, alias, pièges)

## Index par tâche (guide rapide)

- Vérifier que l’environnement marche
	- Lis : [docs/01-Getting-Started/01-Quick-Start.md](01-Getting-Started/01-Quick-Start.md)
	- Lance :

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

- Comprendre create/build/alloc/init
	- Lis : [docs/02-User-Guide/02-Model-Lifecycle.md](02-User-Guide/02-Model-Lifecycle.md)
	- Lance :

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

- Apprendre à passer des arguments
	- Lis : [docs/02-User-Guide/06-Lua-Scripting.md](02-User-Guide/06-Lua-Scripting.md)
	- Lance :

```bash
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua -- --help
```

- Éviter les OOM et stabiliser les runs
	- Lis : [docs/02-User-Guide/09-Memory.md](02-User-Guide/09-Memory.md)
	- Lance :

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1
```

- Sauver/charger proprement
	- Lis : [docs/03-API-Reference/02-Serialization.md](03-API-Reference/02-Serialization.md)
	- Lance :

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

## Cheat sheet (conventions + champs importants)

### Conventions d’IO (noms de tenseurs)

| Nom | Type typique | Sens |
| --- | ---: | --- |
| `__input__` | float *ou* ids int (selon archi) | entrée par défaut |
| `text_ids` | ids int | entrée texte dédiée (NLP) |
| `x` | float | sortie principale (convention) |

### Champs de config Transformer (v2.4)

| Champ | Sens |
| --- | --- |
| `seq_len` | longueur de séquence traitée |
| `vocab_size` | taille du vocab |
| `d_model` | dimension embedding/model |
| `mlp_hidden` | hidden du MLP (FFN) |
| `num_layers` | nombre de blocs |
| `num_heads` | nombre de têtes |

### Legacy → moderne (scripts)

| Ancien | Nouveau | Pourquoi |
| --- | --- | --- |
| `max_seq_len` | `seq_len` | nom canonique registre |
| `d_ff` | `mlp_hidden` | explicite (FFN/MLP) |
| `embed_dim` | `d_model` | cohérent Transformer |

## 1) Démarrer (10 minutes)

- Quick start : [docs/01-Getting-Started/01-Quick-Start.md](01-Getting-Started/01-Quick-Start.md)
- Installer / compiler : [docs/01-Getting-Started/02-Installation.md](01-Getting-Started/02-Installation.md)
- CLI (binaire `mimir`) : [docs/01-Getting-Started/03-CLI.md](01-Getting-Started/03-CLI.md)
- Organisation du repo : [docs/01-Getting-Started/04-Repo-Layout.md](01-Getting-Started/04-Repo-Layout.md)
- Smoketest (valider l’environnement rapidement) : [docs/01-Getting-Started/05-Smoketest.md](01-Getting-Started/05-Smoketest.md)

## 2) Utiliser le framework

- Concepts essentiels : [docs/02-User-Guide/01-Core-Concepts.md](02-User-Guide/01-Core-Concepts.md)
- Workflow modèle (create/build/allocate/forward/backward) : [docs/02-User-Guide/02-Model-Lifecycle.md](02-User-Guide/02-Model-Lifecycle.md)
- Données & datasets : [docs/02-User-Guide/03-Data.md](02-User-Guide/03-Data.md)
- Entraînement : [docs/02-User-Guide/04-Training.md](02-User-Guide/04-Training.md)
- Inférence : [docs/02-User-Guide/05-Inference.md](02-User-Guide/05-Inference.md)
- Scripting Lua (args, globals) : [docs/02-User-Guide/06-Lua-Scripting.md](02-User-Guide/06-Lua-Scripting.md)
- Tokenizer & Encoder : [docs/02-User-Guide/07-Tokenizer-Encoder.md](02-User-Guide/07-Tokenizer-Encoder.md)
- Checkpoints / reprise : [docs/02-User-Guide/08-Checkpoints.md](02-User-Guide/08-Checkpoints.md)
- Analyse d’un artefact modèle sur disque (SafeTensors / RawFolder / DebugJson) : [docs/02-User-Guide/08-Checkpoints.md](02-User-Guide/08-Checkpoints.md)
- Mémoire (Allocator, MemoryGuard) : [docs/02-User-Guide/09-Memory.md](02-User-Guide/09-Memory.md)
- Scripts d’exemples : [docs/02-User-Guide/10-Examples.md](02-User-Guide/10-Examples.md)
- Tutoriel VAEText : [docs/02-User-Guide/11-VAEText.md](02-User-Guide/11-VAEText.md)
- Tutoriel Transformer/GPT : [docs/02-User-Guide/12-Transformer-GPT.md](02-User-Guide/12-Transformer-GPT.md)
- Tutoriel diffusion (PonyXL/SD3.5) : [docs/02-User-Guide/13-Diffusion.md](02-User-Guide/13-Diffusion.md)

## 3) Référence API

- Vue d’ensemble API Lua : [docs/03-API-Reference/00-API-Overview.md](03-API-Reference/00-API-Overview.md)
- Layers (statut, paramètres, compat) : [docs/03-API-Reference/01-Layers.md](03-API-Reference/01-Layers.md)
- Sérialisation (save/load, formats, checksums) : [docs/03-API-Reference/02-Serialization.md](03-API-Reference/02-Serialization.md)
- `Mimir.Model` : [docs/03-API-Reference/10-Model.md](03-API-Reference/10-Model.md)
- `Mimir.Architectures` : [docs/03-API-Reference/11-Architectures.md](03-API-Reference/11-Architectures.md)
- `Mimir.Tokenizer` : [docs/03-API-Reference/12-Tokenizer.md](03-API-Reference/12-Tokenizer.md)
- `Mimir.Dataset` : [docs/03-API-Reference/13-Dataset.md](03-API-Reference/13-Dataset.md)
- Mémoire / allocator : [docs/03-API-Reference/14-Memory.md](03-API-Reference/14-Memory.md)
- Visualisation & monitoring : [docs/03-API-Reference/15-Viz-Htop.md](03-API-Reference/15-Viz-Htop.md)
- Sérialisation (détaillé) : [docs/03-API-Reference/16-Serialization.md](03-API-Reference/16-Serialization.md)
- `Mimir.NeuroPulse` : [docs/03-API-Reference/17-NeuroPulse.md](03-API-Reference/17-NeuroPulse.md)
- `Mimir.Layers` (ops) : [docs/03-API-Reference/18-Layers-Module.md](03-API-Reference/18-Layers-Module.md)
- Globals & aliases : [docs/03-API-Reference/19-Globals.md](03-API-Reference/19-Globals.md)
- Mapping Lua ↔ C++ (sommaire) : [docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md](03-API-Reference/20-Lua-API-Cpp-Mapping.md)

## 4) Internals (comment ça marche)

- Index internals (étendu) : [docs/04-Architecture-Internals/00-Internals-Index.md](04-Architecture-Internals/00-Internals-Index.md)
- Moteur d’exécution : [docs/04-Architecture-Internals/01-Engine-Overview.md](04-Architecture-Internals/01-Engine-Overview.md)
- Mémoire & allocateur : [docs/04-Architecture-Internals/02-Memory.md](04-Architecture-Internals/02-Memory.md)
- Backends hardware (CPU/Vulkan/OpenCL) : [docs/04-Architecture-Internals/03-Hardware-Backends.md](04-Architecture-Internals/03-Hardware-Backends.md)
- Monitoring (Htop/SFML/threads) : [docs/04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md](04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md)
- AdvancedRAMManager (cache/compression/spill) : [docs/04-Architecture-Internals/05-AdvancedRAMManager.md](04-Architecture-Internals/05-AdvancedRAMManager.md)
- Classe `Model` (C++) : [docs/04-Architecture-Internals/10-Model-Class.md](04-Architecture-Internals/10-Model-Class.md)
- Helpers C++ (`Helpers.hpp`) : [docs/04-Architecture-Internals/11-Helpers.md](04-Architecture-Internals/11-Helpers.md)
- Stockage `tensor` + alloc dynamique : [docs/04-Architecture-Internals/12-Tensor-Storage.md](04-Architecture-Internals/12-Tensor-Storage.md)
- Autograd / gradients / backward : [docs/04-Architecture-Internals/13-Autograd-Gradients.md](04-Architecture-Internals/13-Autograd-Gradients.md)
- Layers / `LayerOps` / layouts : [docs/04-Architecture-Internals/14-Layers-And-Ops.md](04-Architecture-Internals/14-Layers-And-Ops.md)
- Sérialisation (implémentation) : [docs/04-Architecture-Internals/15-Serialization-Internals.md](04-Architecture-Internals/15-Serialization-Internals.md)
- Tokenizer / Encoder (implémentation) : [docs/04-Architecture-Internals/16-Tokenizer-Encoder-Internals.md](04-Architecture-Internals/16-Tokenizer-Encoder-Internals.md)
- Bindings Lua (implémentation) : [docs/04-Architecture-Internals/17-Lua-Bindings-Internals.md](04-Architecture-Internals/17-Lua-Bindings-Internals.md)
- RuntimeAllocator / scratchpads : [docs/04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md](04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md)
- Registre modèles / builders : [docs/04-Architecture-Internals/19-Models-Registry-And-Builders.md](04-Architecture-Internals/19-Models-Registry-And-Builders.md)
- CLI / entry points : [docs/04-Architecture-Internals/20-CLI-EntryPoints.md](04-Architecture-Internals/20-CLI-EntryPoints.md)

## 5) Performance

- Performance & tuning CPU : [docs/05-Advanced/01-Performance.md](05-Advanced/01-Performance.md)
- Debug & stabilité numérique : [docs/05-Advanced/02-Debugging.md](05-Advanced/02-Debugging.md)
- Analyse modèle (outil `analyze_model.lua`, graphe Mermaid Markdown) : [docs/05-Advanced/02-Debugging.md](05-Advanced/02-Debugging.md)
- LLM (état / manque / roadmap) : [docs/05-Advanced/03-LLM-Readiness.md](05-Advanced/03-LLM-Readiness.md)
- Carte du code source (C/C++, fichier par fichier) : [docs/05-Advanced/04-Source-Code-Map.md](05-Advanced/04-Source-Code-Map.md)

## 6) Contribution

- Contribuer : [docs/06-Contributing/01-Contributing.md](06-Contributing/01-Contributing.md)

## Convention de noms

- `__input__` : entrée float par défaut (ou ids int selon l’archi)
- `text_ids` : entrée ids int pour les architectures NLP qui consomment un Embedding
- `x` : sortie float principale du modèle (convention dans les architectures)

## Où est la “vérité” ?

- API Lua exportée : [src/LuaScripting.cpp](../src/LuaScripting.cpp)
- Moteur et exécution des layers : [src/Model.cpp](../src/Model.cpp)
- Registre des architectures : [src/Models/Registry/ModelArchitectures.cpp](../src/Models/Registry/ModelArchitectures.cpp)
- Tokenizer/Encoder : [src/Tokenizer.cpp](../src/Tokenizer.cpp), [src/Encoder.cpp](../src/Encoder.cpp)
