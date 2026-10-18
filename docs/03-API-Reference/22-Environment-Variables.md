# Référence : variables d'environnement

Cette page répertorie les variables d'environnement effectivement lues par le
runtime Mímir. Utilisez-la pour configurer le dispatch, les fast-paths et les
limites propres à chaque backend sans recompiler le projet.

## Sur cette page

- [1) Runtime global et dispatch](#1-runtime-global-et-dispatch)
- [Configuration JSON `env`](#configuration-json-env)
- [2) Configuration par backend (CPU/CUDA/ROCM)](#2-configuration-par-backend-cpucudarocm)
- [3) Vulkan/OpenCL (offload Linear/MatMul + Conv2d/ConvTranspose2d)](#3-vulkanopencl-offload-linearmatmul-conv2dconvtranspose2d)
- [4) Planner et fusion](#4-planner-et-fusion)
- [4.1) Verbose runtime et cartographie planner](#41-verbose-runtime-et-cartographie-planner)
- [5) Bridge scripting (injectées par Mímir)](#5-bridge-scripting-injectées-par-mímir)
- [6) Variables système utilisées](#6-variables-système-utilisées)
- [7) Variables utilisées côté scripts Lua (scripts/)](#7-variables-utilisées-côté-scripts-lua-scripts)
- [8) Non-env (souvent confondu)](#8-non-env-souvent-confondu)
- [9) Références code](#9-références-code)
- [Étapes suivantes](#étapes-suivantes)

> **Attention**
> Une variable inconnue est généralement ignorée. Vérifiez son nom dans cette
> page et dans la source indiquée avant de conclure qu'un backend l'a appliquée.

Conventions :

- Pour les booléens, utilisez préférentiellement `0` ou `1`.
- Dans les parsers principaux (`RuntimeConfig::fromEnv`), les valeurs `0`, `false`, `no`, `off` désactivent; toute autre valeur non vide active.
- Les variables marquées "injectée" sont posées par Mímir pour les bridges, pas destinées à être fixées manuellement dans un usage standard.

## Configuration JSON `env`

Les variables de cette page peuvent être placées dans la section racine `env`
d'un fichier utilisé avec `--conf` ou `--config` :

```json
{
  "env": {
    "OMP_NUM_THREADS": 8,
    "MIMIR_ACCEL_VERBOSE": 1,
    "MIMIR_DISABLE_OPENCL": true
  }
}
```

L'ordre réel est : lecture JSON, application des `--override`, export de `env`,
puis construction/exécution du run. La config a priorité sur le shell. Les
valeurs acceptées sont chaîne, nombre ou booléen. `OMP_NUM_THREADS` déclenche en
plus `omp_set_num_threads()` dans les builds OpenMP.

> **Limitation**
> Une variable consommée avant le chargement de la config ne peut pas modifier
> rétroactivement cette phase. Par exemple, la verbosité du tee de logs est
> choisie au tout début du processus; placez `MIMIR_CONSOLE_VERBOSE=1` dans le
> shell si vous devez modifier ce démarrage précis.

## 1) Runtime global et dispatch

| Variable | Type | Défaut | Effet |
| --- | --- | --- | --- |
| `MIMIR_ACCEL_VERBOSE` | bool | `0` | Active des logs de décision d'accélération (CPU/GPU/offload). |
| `MIMIR_RUNTIME_TRACE` | bool | `0` | Active une trace d'exécution layer-par-layer (backend réellement utilisé, chemin d'appel, taille de sortie). |
| `MIMIR_ALLOCATOR_LOG` | bool | `0` | Émet un résumé des métriques `RuntimeAllocator` en fin de `forward` et `backward` (allocations, total alloué, usage courant/pic, pool scratchpad, fuite potentielle). |
| `MIMIR_ALLOCATOR_LOG_VERBOSE` | bool | `0` | Ajoute le détail des buffers du pool scratchpad (`tag`, taille) quand `MIMIR_ALLOCATOR_LOG=1`. |
| `MIMIR_DISABLE_CPU` | bool | `0` | Désactive explicitement le runtime CPU. |
| `MIMIR_DISABLE_CUDA` | bool | `0` | Désactive explicitement le runtime CUDA. |
| `MIMIR_DISABLE_ROCM` | bool | `0` | Désactive explicitement le runtime ROCm. |
| `MIMIR_DISABLE_VULKAN` | bool | `0` | Désactive le backend Vulkan Compute. |
| `MIMIR_DISABLE_OPENCL` | bool | `0` | Désactive le backend OpenCL Compute. |

Notes sur les colonnes allocator (quand `MIMIR_ALLOCATOR_LOG=1`) :

- Attribution mémoire par backend sur le forward : `backend_cpu_*`, `backend_vulkan_*`, `backend_cuda_*`, `backend_rocm_*`, `backend_other_*`.
- Les métriques sont disponibles en MB et en bytes (`*_mb`, `*_bytes`) pour éviter les faux zéros sur petits runs.
- Les champs `guard_*` et `dyn_*` reflètent la mémoire globale suivie par `MemoryGuard` et `DynamicTensorAllocator`.

## 2) Configuration par backend (CPU/CUDA/ROCM)

Ces variables sont lues via `RuntimeConfig::fromEnv("CPU"|"CUDA"|"ROCM")`.

### Suffixes supportés

| Suffixe | Type | Défaut | Effet |
| --- | --- | --- | --- |
| `_LINEAR` | bool | `1` | Active fast-path `Linear` sur le backend. |
| `_LINEAR_MIN_OPS` | int | `0` | Seuil minimal d'opérations pour offload `Linear`. |
| `_CONV` | bool | `1` (CUDA/ROCM) | Active fast-path `Conv2d`. |
| `_CONV_MIN_OPS` | int | `0` (CUDA/ROCM) | Seuil minimal d'opérations pour `Conv2d`. |
| `_NORM` | bool | `1` (CUDA/ROCM) | Active fast-path normalisations (`LayerNorm`/`RMSNorm`). |
| `_NORM_MIN_ELEMS` | int | `0` (CUDA/ROCM) | Seuil minimal en nombre d'éléments pour normalisations. |
| `_ATTENTION` | bool | `1` (CUDA/ROCM) | Active fast-path attention (`Self`/`MultiHead`/`Cross`). |
| `_ATTENTION_MIN_OPS` | int | `0` (CUDA/ROCM) | Seuil minimal d'opérations pour attention. |
| `_DEVICE` | int | `0` | Index de device à utiliser pour le backend. |

### Exemples concrets

- CUDA: `MIMIR_CUDA_LINEAR`, `MIMIR_CUDA_CONV`, `MIMIR_CUDA_DEVICE`.
- ROCm: `MIMIR_ROCM_LINEAR`, `MIMIR_ROCM_ATTENTION_MIN_OPS`.
- CPU: `MIMIR_CPU_LINEAR`, `MIMIR_CPU_LINEAR_MIN_OPS`.

Notes CPU:

- Le runtime CPU est le fallback de référence.
- Le code force un comportement spécifique CPU: `MIMIR_CPU_LINEAR` est actif par défaut et `MIMIR_CPU_LINEAR_MIN_OPS` vaut `0` si non défini.

Important:

- Il n'existe pas de variable globale `MIMIR_CUDA=1` ou `MIMIR_ROCM=1` dans le code actuel. L'activation se fait par les flags de fast-path (`MIMIR_CUDA_*`, `MIMIR_ROCM_*`) et les kill-switch `MIMIR_DISABLE_*`.
- Par défaut, les fast-paths sont auto-activés. Pour forcer un mode CPU-only, utilisez `MIMIR_DISABLE_CUDA=1`, `MIMIR_DISABLE_ROCM=1`, `MIMIR_DISABLE_VULKAN=1`, `MIMIR_DISABLE_OPENCL=1`.

## 3) Vulkan/OpenCL (offload Linear/MatMul + Conv2d/ConvTranspose2d)

| Variable | Type | Défaut | Effet |
| --- | --- | --- | --- |
| `MIMIR_VULKAN_LINEAR` | bool | `1` | Active offload `Linear` (et gating MatMul/BatchMatMul du runtime) vers Vulkan. |
| `MIMIR_VULKAN_LINEAR_MIN_OPS` | int | `0` | Seuil minimal d'opérations pour Vulkan `Linear`/MatMul. |
| `MIMIR_VULKAN_CONV` | bool | `1` | Active offload `Conv2d` et `ConvTranspose2d` vers Vulkan. |
| `MIMIR_VULKAN_CONV_MIN_OPS` | int | `0` | Seuil minimal d'opérations pour Vulkan `Conv2d`/`ConvTranspose2d`. |
| `MIMIR_VULKAN_LINEAR_SPV` | path | auto | Chemin explicite du shader SPIR-V `linear_forward.comp.spv` (Linear). |
| `MIMIR_OPENCL_LINEAR` | bool | `1` | Active offload `Linear` (et gating MatMul/BatchMatMul du runtime) vers OpenCL. |
| `MIMIR_OPENCL_LINEAR_MIN_OPS` | int | `0` | Seuil minimal d'opérations pour OpenCL `Linear`/MatMul. |

Notes Vulkan SPIR-V:

- Le build compile aussi `add_forward.comp.spv`, `mul_forward.comp.spv`, `relu_forward.comp.spv`, `conv2d_forward.comp.spv` et `conv_transpose2d_forward.comp.spv`.
- Ces shaders sont chargés automatiquement depuis `bin/shaders`/`build/shaders` (pas de variable d'environnement dédiée nécessaire pour ces trois kernels).

## 4) Planner et fusion

| Variable | Type | Défaut | Effet |
| --- | --- | --- | --- |
| `MIMIR_ENABLE_PLANNER` | bool | `1` | Active la planification statique d'exécution. |
| `MIMIR_ENABLE_FUSION` | bool | `1` | Active les chemins de fusion lorsqu'un plan est utilisé. |
| `MIMIR_ENABLE_FUSION_TRAIN` | bool | `0` | Autorise la fusion `Conv2d+ReLU` en mode `training=true` (opt-in). Les fusions génériques restent inférence-only. |
| `MIMIR_PLANNER_DUMP` | bool | `0` | Émet un dump de stats planner au premier forward. |

## 4.1) Verbose runtime et cartographie planner

Avec `MIMIR_ACCEL_VERBOSE=1`, le runtime émet au début du forward:

- le matériel/runtime sélectionné en priorité,
- un scan des types de layers demandés par le modèle,
- une cartographie planner (`planner_map`) indiquant, pour chaque layer, la fusion prévue et le chemin d'appel anticipé.

Avec `MIMIR_RUNTIME_TRACE=1`, le runtime ajoute une trace d'exécution réelle layer-par-layer:

- backend effectivement utilisé (`CUDA`, `ROCM`, `CPU_RUNTIME`, `CPU`, `VULKAN`, `OPENCL`, `FUSED_SKIP`),
- chemin d'appel effectif (`runtime_router.dispatchForwardLayer`, `linear_accel_chain`, `cpu_switch_kernel`, etc.),
- taille de sortie produite.

Exemple rapide:

```bash
export MIMIR_ACCEL_VERBOSE=1
export MIMIR_RUNTIME_TRACE=1
./bin/mimir --lua scripts/benchmarks/benchmark.lua
```

## 5) Bridge scripting (injectées par Mímir)

### Contrôle bridge + metadata

| Variable | Type | Portée | Description |
| --- | --- | --- | --- |
| `MIMIR_BRIDGE_CMD_FILE` | path | injectée | Fichier de commandes bridge (retour script -> hôte). |
| `MIMIR_BRIDGE_ARCH_INFO_JSON` | json | injectée | Metadata des architectures (nom, description, config). |
| `MIMIR_BRIDGE_DTYPES_JSON` | json | injectée | Liste des dtypes exposés au bridge. |
| `MIMIR_BRIDGE_ARCH_AVAILABLE_JSON` | json | injectée | Liste des architectures disponibles. |
| `MIMIR_BRIDGE_ARCH_CACHE_JSON` | json | injectée | Cache architecture/params côté bridge. |
| `MIMIR_BRIDGE_DTYPES_COUNT` | int | injectée | Nombre de dtypes exposés. |
| `MIMIR_BRIDGE_ARCH_AVAIL_COUNT` | int | injectée | Nombre d'architectures disponibles. |

### Contexte script

| Variable | Type | Portée | Description |
| --- | --- | --- | --- |
| `MIMIR_ARG_JSON` | json | injectée | Arguments script sérialisés (tableau JSON). |
| `MIMIR_CONF_JSON` | json | injectée | Configuration active sérialisée. |
| `MIMIR_CONF_PATH` | string | injectée | Chemin de conf courant (si applicable). |
| `MIMIR_CONF_DIR` | string | injectée | Dossier de conf courant (si applicable). |
| `MIMIR_GLOBAL_NAMESPACE` | string | injectée | Namespace racine exposé au script (ex: `Mimir`). |

### Aliases injectés

| Variable | Type | Portée | Description |
| --- | --- | --- | --- |
| `MIMIR_ALIAS_MODEL` | string | injectée | Alias objet modèle. |
| `MIMIR_ALIAS_ARCHITECTURES` | string | injectée | Alias registre architectures. |
| `MIMIR_ALIAS_TOKENIZER` | string | injectée | Alias tokenizer. |
| `MIMIR_ALIAS_DATASET` | string | injectée | Alias dataset. |
| `MIMIR_ALIAS_MEMORY` | string | injectée | Alias module mémoire. |
| `MIMIR_ALIAS_MEMORY_GUARD` | string | injectée | Alias guard mémoire. |
| `MIMIR_ALIAS_ALLOCATOR` | string | injectée | Alias allocator. |
| `MIMIR_ALIAS_HTOP` | string | injectée | Alias monitoring terminal. |
| `MIMIR_ALIAS_VIZ` | string | injectée | Alias visualisation. |

## 6) Variables système utilisées

| Variable | Type | Défaut | Utilisation |
| --- | --- | --- | --- |
| `TMPDIR` | path | `/tmp` | Emplacement des fichiers temporaires bridge (dont cache `mimir_bridge_arch_cache.json`). |

## 7) Variables utilisées côté scripts Lua (`scripts/`)

Ces variables ne pilotent pas le runtime C++ directement; elles servent de paramètres d'exécution pour les scripts Lua (training, benchmark, templates, tooling).

### Variables fréquentes (templates/training)

- `MIMIR_ARCH`
- `MIMIR_DATASET`
- `MIMIR_DTYPE`
- `MIMIR_EPOCHS`
- `MIMIR_LR`
- `MIMIR_SEED`
- `MIMIR_INIT`
- `MIMIR_SAVE`
- `MIMIR_RUN`

### Variables benchmark

- `MIMIR_BENCH_MODE`
- `MIMIR_BENCH_ITERS`
- `MIMIR_BENCH_SEQ`
- `MIMIR_BENCH_VOCAB`
- `MIMIR_BENCH_RAM_GB`
- `MIMIR_BENCH_COMPRESS`
- `MIMIR_NMS_BOXES`
- `MIMIR_NMS_CLASSES`
- `MIMIR_NMS_WARMUP`
- `MIMIR_NMS_ITERS`
- `MIMIR_NMS_IOU`
- `MIMIR_NMS_SCORE`
- `MIMIR_NMS_MAX_DETECTIONS`
- `MIMIR_NMS_CLASS_AGNOSTIC`

Les variables `MIMIR_NMS_*` pilotent
`scripts/benchmarks/benchmark_nms.lua`. Ce benchmark fabrique ses boîtes en
mémoire et ne lit aucun dataset.

### Variables tokenizer / SafeTensors externe

- `MIMIR_BASE_TOKENIZER`
- `MIMIR_BASE_TOKENIZER_MAX_VOCAB`
- `MIMIR_REQUIRE_BASE_TOKENIZER`
- `MIMIR_EXTERNAL_SAFETENSORS_INCLUDE`
- `MIMIR_EXTERNAL_SAFETENSORS`
- `MIMIR_MAX_TENSORS`

### Variables système lues par scripts

- `OMP_NUM_THREADS`
- `NO_COLOR`
- `BROWSER`

## 8) Non-env (souvent confondu)

- `MIMIR_STRICT_MODE` n'est pas une variable d'environnement: c'est un macro de compilation (préprocesseur) utilisé par `RuntimeAllocator`.

## 9) Références code

- `src/runtimes/AbstractRuntime.cpp`
- `src/Model.cpp`
- `src/runtimes/vulkan/VulkanCompute.hpp`
- `src/scriptings/ScriptingBridgeCommon.hpp`
- `src/scriptings/ScriptingBridgeCommon.cpp`
- `src/scriptings/JavaScript/jsScripting/JSScripting.cpp`
- `src/scriptings/CSharp/csharpScripting/CSharpScripting.cpp`
- `src/scriptings/Rust/rustScripting/RustScripting.cpp`

## Étapes suivantes

- [Page précédente : API : `Mimir.IO`](21-IO.md)
- [Index de la documentation](../00-INDEX.md)
- [Revenir à la documentation](../00-INDEX.md)
