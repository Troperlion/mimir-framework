# Carte du code source (C/C++)

Optimiser, diagnostiquer et stabiliser des runs complexes.

**Public concerné :** Utilisateur intermédiaire à avancé.

> **Prérequis**
>
> Avoir déjà exécuté au moins un pipeline complet.


Cette page sert de « table des matières technique » : comment le framework est structuré côté C/C++, fichier par fichier.

Pages Advanced associées (explications orientées usage) :

- [01-Performance.md](01-Performance.md)
- [02-Debugging.md](02-Debugging.md)
- [03-LLM-Readiness.md](03-LLM-Readiness.md)

Objectif : quand une fonctionnalité est mentionnée dans la doc (mémoire stricte, forward tokens, multi-input, sérialisation…), vous devez pouvoir retrouver rapidement **le fichier source qui fait foi**, et comprendre le rôle exact de chaque composant.

> Notes
> - La plupart des features sont exposées via l’API Lua (bindings dans `src/scriptings/Lua/luaScripting/LuaScripting.cpp`).
> - Les modèles « prêts à l’emploi » sont construits via le registre d’architectures (`src/Models/Registry/ModelArchitectures.*`).

## Sur cette page

- [1) Point d’entrée et exposition API](#1-point-dentrée-et-exposition-api)
- [2) Noyau du runtime : Model + exécution](#2-noyau-du-runtime-model-exécution)
- [3) Tenseurs : stockage et allocations](#3-tenseurs-stockage-et-allocations)
- [4) Tokenizer et encodage](#4-tokenizer-et-encodage)
- [5) Registry d’architectures et modèles built-in](#5-registry-darchitectures-et-modèles-built-in)
- [6) Sérialisation / debug](#6-sérialisation-debug)
- [7) Hardware, SIMD et backends compute](#7-hardware-simd-et-backends-compute)
- [8) Monitoring / visualisation](#8-monitoring-visualisation)
- [9) Utilitaires](#9-utilitaires)
- [Lecture guidée (par tâche)](#lecture-guidée-par-tâche)
- [Étapes suivantes](#étapes-suivantes)

## 1) Point d’entrée et exposition API

### `src/main.cpp`

- Rôle : point d’entrée du binaire (init, chargement, exécution Lua / modes CLI selon build).
- À lire quand : vous voulez comprendre comment le runtime démarre, et à quel moment les sous-systèmes (Lua, compute backends, etc.) sont initialisés.

### `src/scriptings/Lua/luaScripting/LuaScripting.hpp` / `src/scriptings/Lua/luaScripting/LuaScripting.cpp`

- Rôle : **API publique** (bindings Lua) sous la table globale `Mimir`.
- Contient :
  - `registerAPI()` : enregistre `Mimir.Model`, `Mimir.Architectures`, `Mimir.Tokenizer`, `Mimir.MemoryGuard`, `Mimir.Allocator`, `Mimir.Serialization`, `Mimir.Htop`, `Mimir.Viz`, etc.
  - Forward : `Mimir.Model.forward(input, training)` route vers :
    - `Model::forwardPass(input_ids)` si `input` est une table d’entiers
    - `Model::forwardPass(input)` si `input` est une table de floats
    - `Model::forwardPassNamed(float_inputs, int_inputs)` si `input` est une map `{name -> table}` (multi-input mixte)
  - Mémoire :
    - `Mimir.MemoryGuard.setLimit(...)`, `getCurrentUsage()`, `getPeakUsage()`, `getLimit()`, `printStats()`, etc.
    - `Mimir.Allocator.configure({max_ram_gb, enable_compression})` configure le `DynamicTensorAllocator`
  - Debug JSON : `Mimir.Serialization.save_enhanced_debug(path, options)` (v1.3 inclut `framework_state`) (voir aussi `src/Serialization/DebugJsonDump.*`).

À lire quand :

- vous cherchez “comment un appel Lua devient un appel C++”
- vous voulez des signatures exactes, erreurs retournées, comportements de conversion table->vector, etc.

## 2) Noyau du runtime : `Model` + exécution

### `src/Model.hpp` / `src/Model.cpp`

- Rôle : cœur du framework.
- Responsabilités :
  - stocker la liste des layers (`std::vector<Layer> layers`)
  - construire/exécuter le forward (float, tokens int, ou named inputs)
  - gérer l’entraînement (loss, backward, optimizer)
  - orchestrer la mémoire runtime via `MemoryGuard` + allocateurs
  - (optionnel) orchestrer des backends compute (Vulkan/OpenCL) pour certaines ops en inférence

Points à connaître :

- **Trois APIs forward** importantes :
  - `forwardPass(const std::vector<float>&, training)`
  - `forwardPass(const std::vector<int>&, training)` (chemin tokens : `Embedding` lit dans l’IntTensorStore)
  - `forwardPassNamed(float_inputs, int_inputs, training)` (multi-input)
- **Optimiseur** : `optimizerStep(Optimizer&, learning_rate)` gère SGD, Adam, AdamW, Lion, Adafactor, RAdam, NAdam, RMSprop et LAMB, avec protections (epsilon non-fini -> fallback), warmup/decay, et *grad clipping optionnel* via `modelConfig` (`grad_clip_norm` / `clip_norm`).
- **Accélération compute** (en inférence, opt-in) :
  - certaines ops (ex: `Linear`) peuvent être dispatchées vers Vulkan/OpenCL si activé via variables d’environnement (voir plus bas).

À lire quand : vous voulez comprendre l’ordre d’exécution des layers, les invariants de shapes, la logique training vs inference, ou les routes “float vs token ids”.

### `src/Layers.hpp`

- Rôle : structure “Layer” unifiée, avec un grand nombre de paramètres optionnels.
- Contient :
  - la struct `Layer` (nom, type string + enum, `inputs`/`output` pour le routing)
  - les champs “universels” (dimensions, conv, norm, dropout, attention, reshape, split/concat, etc.)
  - le lien poids unifié : `tensor* weight_block` (poids du layer dans un bloc unique)

À lire quand : vous documentez un type de layer, ses champs, ou pourquoi une op lit tel paramètre (ex: `seq_len`, `embed_dim`, `num_heads`, `split_sizes`, …).

### `src/LayerTypes.hpp`

- Rôle : enum central `LayerType` + normalisation d’alias (rétrocompat).
- À lire quand : vous voulez la **liste canonique** des layers supportés par le runtime, et les alias acceptés.

### `src/runtimes/cpu/LayerOps.hpp` / `src/runtimes/cpu/LayerOpsExt.hpp`

- Rôle : implémentations CPU des opérations (forward/backward) utilisées par `Model`.
- `LayerOpsExt` regroupe souvent des implémentations plus spécialisées/étendues.
- À lire quand : vous cherchez “où est l’implémentation réelle de Conv/Linear/Norm/Activation/Attention”.

### `src/Autograd.hpp`

- Rôle : structures/utilitaires pour gradients (type `Gradients`, helpers).
- À lire quand : vous voulez comprendre la représentation des gradients, et comment ils sont manipulés.

## 3) Tenseurs : stockage et allocations

### `src/tensors.hpp` / `src/tensors.cpp`

- Rôle : type `tensor` (bloc float), et intégration optionnelle avec l’allocation dynamique.
- Point clé : un `tensor` peut être créé en mode “dynamic” et utiliser `DynamicTensorAllocator`.
- À lire quand : vous cherchez pourquoi certaines allocations passent (ou non) par le guard/allocator.

### `src/MemoryGuard.hpp`

- Rôle : garde-fou strict (comptabilisation courant/pic + limite) pour éviter OOM silencieux.
- Utilisé par : runtime, allocateurs, wrappers (ex: allocations stb_image routées).

### `src/runtimes/cpu/DynamicTensorAllocator.hpp`

- Rôle : allocateur dynamique global (singleton) utilisé par `tensor(dynamic=true)` et certains chemins runtime.
- Fonctionnalités : réservations, stats, (optionnel) compression/éviction selon config.

### `src/runtimes/cpu/RuntimeAllocator.hpp`

- Rôle : allocateur runtime RAII (handles auto-release) + pool scratchpad.
- Utilisé dans des chemins `Model.cpp` pour gérer buffers temporaires (ex: conv im2col/tiles, scratch buffers).

### `src/AdvancedRAMManager.hpp`

- Rôle : gestion RAM avancée / cache (configuration, statistiques, clear).
- Exposé via `Mimir.Memory.*` dans Lua (stats, limite en GB, etc.).

### `src/MemorySafety.hpp`

- Rôle : helpers/guardrails “safety” (assertions, checks, comportements stricts) selon build.

### `src/stb_image_impl.cpp` + `src/stb_image.h`

- Rôle : chargement image via stb_image.
- Particularité : **certaines allocations** sont routées via `MemoryGuard` pour éviter des pics incontrôlés.

## 4) Tokenizer et encodage

### `src/Tokenizer.hpp` / `src/Tokenizer.cpp`

- Rôle : tokenizer (dont BPE), vocab, encode/decode.
- Exposé via `Mimir.Tokenizer.*`.
- À lire quand : vous voulez comprendre `tokenize_bpe`, `decode`, ou les IDs spéciaux.

### `src/Encoder.hpp` / `src/Encoder.cpp` (ConditioningEncoder)

- Rôle : embeddings / encodeur “mag/mod” et embeddings tokens.
- Important : le `Model` maintient la compatibilité `Tokenizer` <-> `ConditioningEncoder` (ex: ensure vocab size).

## 5) Registry d’architectures et modèles built-in

### `src/Models/Registry/ModelArchitectures.hpp` / `src/Models/Registry/ModelArchitectures.cpp`

- Rôle : registre d’architectures (noms -> (default_config, factory)).
- Comportement important :
  - canonicalisation d’alias (`canonicalArchName`)
  - merge récursif `default_config` + overrides
  - impose `cfg["type"] = <nom canonique>`
  - écrit `model->modelConfig = cfg`

À lire quand : vous voulez la liste des archis et leurs configs par défaut, ou comprendre pourquoi `cfg.type` est standardisé.

### `src/Models/**` (implémentations de modèles)

Chaque modèle “builder” sait pousser une séquence de layers cohérente (noms, routage, paramètres) :

- NLP :
  - `src/Models/NLP/CausalLMModel.hpp` / `.cpp` : decoder-only causal (RoPE/GQA/SwiGLU/RMSNorm)
  - `src/Models/NLP/TransformerModel.hpp` / `.cpp` : builder Transformer (peut être causal)
  - `src/Models/NLP/VAETextModel.hpp` / `.cpp` : VAE sur texte (encode/decode)
- Vision :
  - `src/Models/Vision/ViTModel.*`, `VAEModel.*`, `VAEConvModel.*`, `UNetModel.*`
  - CNN : `ResNetModel.*`, `MobileNetModel.*`, `VGG16Model.*`, `VGG19Model.*`
- Diffusion :
  - `src/Models/Diffusion/DiffusionModel.*`, `CondDiffusionModel.*`, `SD35Model.*`

À lire quand : vous voulez voir comment une architecture “réaliste” se traduit en layers, et quelles conventions de noms sont utilisées.

## 6) Sérialisation / debug

### `src/Serialization/Serialization.hpp` / `src/Serialization/Serialization.cpp`

- Rôle : façade de sérialisation (sauvegarde/chargement) utilisée par l’API Lua `Mimir.Serialization.*`.

### `src/Serialization/SafeTensorsReader.*` et `src/Serialization/SafeTensorsWriter.*`

- Rôle : IO `safetensors`.
- Important : peut inclure des payloads extra (ex: tokenizer JSON) selon options.

### `src/Serialization/RawCheckpointReader.*` / `src/Serialization/RawCheckpointWriter.*`

- Rôle : format legacy/raw (compatibilité / debug).

### `src/Serialization/DebugJsonDump.hpp` / `.cpp`

- Rôle : export debug JSON “amélioré” (statistiques, gradients, checksums, infos git, tokenizer/encoder… selon options).
- Depuis v1.3 : ajoute `framework_state` (runtime, mémoire, registre layers, état modèle) au moment du dump.
- Exposé via `Mimir.Serialization.save_enhanced_debug(path, options)`.

## 7) Hardware, SIMD et backends compute

### `src/runtimes/cpu/HardwareOpt.hpp`

- Rôle : détection capacités CPU et helpers (AVX2/FMA/F16C/BMI2, etc.).
- Exposé via `Mimir.Model.hardware_caps()`.

### `src/runtimes/cpu/SIMD_Ops.hpp`

- Rôle : primitives SIMD (CPU) utilisées par des ops.

### `src/runtimes/vulkan/VulkanCompute.hpp` et `src/runtimes/opencl/OpenCLCompute.hpp`

- Rôle : engines compute GPU pour Vulkan/OpenCL (si compilés).
- Build flags : `ENABLE_VULKAN`, `ENABLE_OPENCL`.
- Variables d'environnement (Vulkan/OpenCL) : `MIMIR_VULKAN_LINEAR_SPV`, `MIMIR_VULKAN_LINEAR`, `MIMIR_OPENCL_LINEAR`.

Portée actuelle:

- Vulkan : `Linear`, `MatMul`, `BatchMatMul`, `Add`, `Multiply`, `ReLU`, `SiLU`, `GELU`, `Sigmoid`, `Tanh`.
- OpenCL : `Linear`, `MatMul`, `BatchMatMul`.

### `src/runtimes/AbstractRuntime.hpp`

- Rôle : interface commune `AbstractRuntime` + structure `RuntimeConfig` (6 fast-paths configurables).
- `RuntimeConfig::fromEnv(backend_upper)` : parse les variables `MIMIR_CUDA_*` / `MIMIR_ROCM_*`.

### `src/runtimes/cuda/CudaRuntime.hpp` / `.cpp`

- Rôle : fast-paths cuBLAS pour Linear, Conv2d (im2col+SGEMM), LayerNorm/RMSNorm (hybride), SelfAttention, MHA, CrossAttention.
- Build flag : `ENABLE_CUDA`.
- Variables : `MIMIR_CUDA_LINEAR=1`, `MIMIR_CUDA_CONV=1`, `MIMIR_CUDA_NORM=1`, `MIMIR_CUDA_ATTENTION=1`.
- Internals : `src/runtimes/cuda/CudaRuntime.cpp` (Impl, DeviceBuf, fast-path par LayerType).

### `src/runtimes/rocm/RocmRuntime.hpp` / `.cpp`

- Rôle : miroir de CudaRuntime avec rocBLAS/HIP.
- Build flag : `ENABLE_ROCM`.
- Variables : `MIMIR_ROCM_*` (mêmes suffixes que CUDA).

### `src/runtimes/cpu/CpuRuntime.hpp` / `.cpp`

- Rôle : fallback CPU universel via `RuntimeLayerDispatch::cpu_forward_layer`.
- Délègue à `LayerOps.hpp` / `LayerOpsExt.hpp`.

Documentation détaillée : [docs/04-Architecture-Internals/03-Hardware-Backends.md](../04-Architecture-Internals/03-Hardware-Backends.md), [docs/04-Architecture-Internals/21-GPU-Runtimes.md](../04-Architecture-Internals/21-GPU-Runtimes.md), [docs/05-Advanced/05-GPU-Acceleration.md](05-GPU-Acceleration.md).

### `src/Planning/Planner.hpp`

- Rôle : analyse statique du graphe — durées de vie des tenseurs, fusions Conv2d+ReLU, pré-calcul des scratchpads.
- Au lancement : le plan est construit au premier forward (ou reconstruit si invalidé), puis réutilisé tant que le graphe et le mode training restent compatibles.
- Documentation : [docs/04-Architecture-Internals/22-Planning.md](../04-Architecture-Internals/22-Planning.md).

## 8) Monitoring / visualisation

### `src/AsyncMonitor.hpp`

- Rôle : collecte asynchrone métriques + frames (si activé).

### `src/HtopDisplay.hpp`

- Rôle : affichage/collecte style htop (CPU/RAM/loss/bps…).
- Exposé via `Mimir.Htop.*`.

### `src/Visualizer.hpp` / `src/Visualizer.cpp`

- Rôle : visualisation SFML (taps d’images par bloc/layer, métriques).
- Exposé via `Mimir.Viz.*`.

## 9) Utilitaires

### `src/Helpers.hpp`

- Rôle : types et helpers transverses (dataset items, IO, magic tokens, etc.).

### `src/RngContext.hpp`

- Rôle : PRNG/seed management (déterminisme best-effort selon ops).

### `src/Sha256.hpp` / `src/Sha256.cpp`

- Rôle : checksum (utilisé par debug/serialization selon options).

---

## Lecture guidée (par tâche)

- "Je veux activer l'accélération GPU" : lire [docs/05-Advanced/05-GPU-Acceleration.md](05-GPU-Acceleration.md), puis variables d'environnement `MIMIR_CUDA_*` / `MIMIR_ROCM_*`, puis internals dans `src/runtimes/cuda/CudaRuntime.cpp`.
- "Je veux comprendre l'exécution d'un forward multi-input" : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` (forward), puis `src/Model.cpp` (forwardPassNamed), puis `src/Layers.hpp` (inputs/output).
- "Je veux comprendre la mémoire stricte (OOM explicites)" : `src/MemoryGuard.hpp`, `src/runtimes/cpu/DynamicTensorAllocator.hpp`, `src/runtimes/cpu/RuntimeAllocator.hpp`, et les usages dans `src/Model.cpp` et `src/tensors.cpp`.
- "Je veux comprendre le Transformer causal" : `src/Models/NLP/TransformerModel.cpp` (builder) + ops dans `src/LayerOps*` + forward tokens dans `src/Model.cpp`.
- "Je veux un dump exploitable pour debug" : `src/Serialization/DebugJsonDump.*` + binding `Mimir.Serialization.save_enhanced_debug` dans `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.
- "Je veux ajouter un nouveau fast-path GPU" : `src/runtimes/AbstractRuntime.hpp` (RuntimeConfig) → `src/runtimes/cuda/CudaRuntime.cpp` (case + DeviceBuf) → `src/runtimes/rocm/RocmRuntime.cpp` (miroir).

## Étapes suivantes

- [Page précédente : LLM readiness (état réel)](03-LLM-Readiness.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Accélération GPU](05-GPU-Acceleration.md)
