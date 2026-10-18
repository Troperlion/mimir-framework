# Runtimes GPU : CUDA et ROCm

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

Cette page documente l'implémentation interne des backends GPU de Mímir : architecture, fast-paths, conventions de code.

Sources de vérité :

- `src/runtimes/AbstractRuntime.hpp` — interface + `RuntimeConfig`
- `src/runtimes/cuda/CudaRuntime.hpp` / `.cpp` — CUDA (cuBLAS)
- `src/runtimes/rocm/RocmRuntime.hpp` / `.cpp` — ROCm (rocBLAS)
- `src/runtimes/cpu/CpuRuntime.hpp` / `.cpp` — CPU référence

Guide utilisateur associé : [docs/05-Advanced/05-GPU-Acceleration.md](../05-Advanced/05-GPU-Acceleration.md).

---

## Sur cette page

- [1) Interface commune (AbstractRuntime)](#1-interface-commune-abstractruntime)
- [2) RuntimeConfig — paramètres et parsing](#2-runtimeconfig-paramètres-et-parsing)
- [3) CudaRuntime — implémentation](#3-cudaruntime-implémentation)
- [4) Fast-paths détaillés](#4-fast-paths-détaillés)
- [5) ROCm Runtime — différences avec CUDA](#5-rocm-runtime-différences-avec-cuda)
- [6) Vulkan Runtime — kernels compute réels (SPIR-V)](#6-vulkan-runtime-kernels-compute-réels-spir-v)
- [7) Invariants et gotchas](#7-invariants-et-gotchas)
- [8) Ajouter un nouveau fast-path GPU](#8-ajouter-un-nouveau-fast-path-gpu)
- [Étapes suivantes](#étapes-suivantes)

## 1) Interface commune (`AbstractRuntime`)

```cpp
class AbstractRuntime {
public:
    virtual const char* name() const = 0;
    virtual bool initialize(const RuntimeConfig& cfg) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;

    // Fast-path pour Linear (SGEMM)
    virtual bool linearForward(const float* input, const float* weights,
                               const float* bias_or_null, float* output,
                               int batch, int in_f, int out_f) = 0;

    // Dispatch générique par type de layer
    virtual bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training) = 0;
};
```

### Patron de dispatch (Model.cpp + RuntimeRouter)

```cpp
// Le forward route via RuntimeRouter.
// Le router essaie chaque backend initialisé dans l'ordre,
// et appelle forwardLayer(...).
// Chaque runtime sélectionne les LayerType supportés via
// switch (layer.type_enum) et retourne false sinon.
```

Le dispatch `forwardLayer(...)` est utilisé dans le forward principal pour Linear, Convolution, Normes, Attention, et autres ops supportées par les backends.

---

## 2) `RuntimeConfig` — paramètres et parsing

### Structure

```cpp
struct RuntimeConfig {
    std::string backend;   // "CUDA" ou "ROCM"
    bool disabled         = false;
    bool verbose          = false;

    bool linear_enabled   = false;
    int  linear_min_ops   = 1 << 20;

    bool conv_enabled     = false;
    int  conv_min_ops     = 1 << 18;

    bool norm_enabled         = false;
    int  norm_min_elements    = 1 << 12;

    bool attention_enabled  = false;
    int  attention_min_ops  = 1 << 18;

    int device_index = 0;
};
```

### Parsing depuis l'environnement

```cpp
RuntimeConfig RuntimeConfig::fromEnv(const char* backend_upper) {
    RuntimeConfig cfg;
    cfg.backend = backend_upper;
    // Préfixe : "MIMIR_CUDA_" ou "MIMIR_ROCM_"
    std::string pfx = std::string("MIMIR_") + backend_upper + "_";

    auto get_bool = [&](const char* key, bool def) -> bool { ... };
    auto get_int  = [&](const char* key, int  def) -> int  { ... };

    cfg.disabled    = !get_bool("", true);     // MIMIR_CUDA=0 → disabled
    cfg.verbose     = get_bool("VERBOSE", false);
    cfg.device_index = get_int("DEVICE", 0);

    cfg.linear_enabled    = get_bool("LINEAR",    false);
    cfg.linear_min_ops    = get_int ("LINEAR_MIN_OPS", 1 << 20);
    cfg.conv_enabled      = get_bool("CONV",      false);
    cfg.conv_min_ops      = get_int ("CONV_MIN_OPS",  1 << 18);
    cfg.norm_enabled      = get_bool("NORM",      false);
    cfg.norm_min_elements = get_int ("NORM_MIN_ELEMS", 1 << 12);
    cfg.attention_enabled = get_bool("ATTENTION", false);
    cfg.attention_min_ops = get_int ("ATTENTION_MIN_OPS", 1 << 18);
    return cfg;
}
```

---

## 3) CudaRuntime — implémentation

### Structure interne

```cpp
class CudaRuntime final : public AbstractRuntime {
    struct Impl {
        cublasHandle_t handle = nullptr;
        RuntimeConfig  config;

        // Buffers temporaires réutilisables (arène simple)
        struct DeviceBuf {
            void*  ptr   = nullptr;
            size_t bytes = 0;
            bool alloc(size_t n);
            bool copyFromHost(const void* src, size_t n);
            bool copyToHost(void* dst, size_t n) const;
            ~DeviceBuf();
        };

        DeviceBuf d_A, d_B, d_C, d_bias, d_col, d_scratch;
    };
    Impl* impl_ = nullptr;
};
```

### Initialisation

```cpp
bool CudaRuntime::initialize(const RuntimeConfig& cfg) {
    impl_ = new Impl{};
    impl_->config = cfg;
    if (cfg.disabled) return false;
    cudaError_t err = cudaSetDevice(cfg.device_index);
    if (err != cudaSuccess) { /* log, return false */ }
    cublasStatus_t st = cublasCreate(&impl_->handle);
    return st == CUBLAS_STATUS_SUCCESS;
}
```

---

## 4) Fast-paths détaillés

> **Important :** la section ci-dessous décrit l'implémentation de `CudaRuntime::forwardLayer` / `RocmRuntime::forwardLayer`, qui est effectivement sollicitée par le dispatch principal selon le type de layer et les flags runtime.

### 4.1 Pattern commun

Chaque fast-path dans `CudaRuntime::forwardLayer` suit ce squelette :

```cpp
case LayerType::Linear:
{
    if (!impl_->config.linear_enabled) break;
    int M = batch, K = in_f, N = out_f;
    long long ops = (long long)M * K * N * 2;
    if (ops < impl_->config.linear_min_ops) break;

    // 1. Allouer device
    if (!impl_->d_A.alloc(M*K*sizeof(float))) break;
    if (!impl_->d_B.alloc(K*N*sizeof(float))) break;
    if (!impl_->d_C.alloc(M*N*sizeof(float))) break;

    // 2. Upload
    if (!impl_->d_A.copyFromHost(input.data(), M*K*sizeof(float))) break;
    if (!impl_->d_B.copyFromHost(weights, K*N*sizeof(float))) break;

    // 3. SGEMM
    float alpha=1.f, beta=0.f;
    auto st = cublasSgemm(impl_->handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (float*)impl_->d_B.ptr, N,
        (float*)impl_->d_A.ptr, K,
        &beta,
        (float*)impl_->d_C.ptr, N);
    if (st != CUBLAS_STATUS_SUCCESS) break;

    // 4. Biais (optionnel)
    if (bias && out_f > 0) { /* cublasSaxpy par ligne */ }

    // 5. Sync + download
    if (cudaDeviceSynchronize() != cudaSuccess) break;
    if (!impl_->d_C.copyToHost(output.data(), M*N*sizeof(float))) break;

    return true; // ← succès
}
while(false); // break → fallback CPU
return false;
```

### 4.2 Convolutions et normalisations natives

`CudaNativeOps.cuh` contient les kernels Conv2d et les normalisations
LayerNorm/RMSNorm/GroupNorm/InstanceNorm. `CudaTrainingOps.cuh` couvre les
variantes BatchNorm et les opérations stochastiques. Les convolutions étendues
(Conv1d, DepthwiseConv2d, ConvTranspose2d) et leurs gradients utilisent également
des kernels device. Les calculs de réduction et de convolution des chemins
natifs restent sur GPU ; les tenseurs sont transférés pour le staging.

### 4.3 Attention, récurrence et NMS

`CudaAttentionOps.cuh` fournit les projections, scores, masque causal,
softmax et gradients d'attention sur le device. `CudaRecurrentOps.cuh` couvre
RNN, GRU et LSTM selon les formes admises par le contrat CPU.

`CudaDetectionOps.cuh` effectue la sélection NMS sur GPU et fournit le backward
stop-gradient : un gradient nul de la taille de chaque entrée. Le dispatcher
CPU utilise le même contrat. Le backward `Lambda` reste indisponible en CUDA.
La présence de ces kernels ne prouve pas leur exécution sur GPU NVIDIA :
les validations locales récentes portent sur CPU, OpenCL et Vulkan.

---

## 5) ROCm Runtime — différences avec CUDA

Les chemins historiques ROCm utilisent les équivalents HIP/rocBLAS des API
CUDA. Les extensions natives CUDA récentes ne sont pas automatiquement
portées dans `RocmRuntime` :

| CUDA | ROCm |
| --- | --- |
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cublasCreate` | `rocblas_create_handle` |
| `cublasSgemm` | `rocblas_sgemm` |
| `cublasSaxpy` | `rocblas_saxpy` |
| `cublasSger` | `rocblas_sger` |
| `cublasSdgmm` | `rocblas_sdgmm` |
| `CUBLAS_STATUS_SUCCESS` | `rocblas_status_success` |
| `CUBLAS_OP_N` | `rocblas_operation_none` |
| `CUBLAS_OP_T` | `rocblas_operation_transpose` |
| `CUBLAS_SIDE_LEFT` | `rocblas_side_left` |

Les flags et seuils sont propres à chaque backend. Consulter les capacités
déclarées et les traces pour distinguer calcul natif et repli CPU.

---

## 6) Vulkan Runtime — kernels compute réels (SPIR-V)

Vulkan est branché via `VulkanRuntime` + `VulkanCompute::ComputeEngine` et dispatché par `RuntimeRouter`.

Quelques chemins spécialisés historiques de `VulkanCompute::ComputeEngine` :

- `Linear`
- `MatMul`
- `BatchMatMul`
- `Add`
- `Multiply`
- `ReLU`

Shaders associés (compilés au build):

- `shaders/linear_forward.comp`
- `shaders/add_forward.comp`
- `shaders/mul_forward.comp`
- `shaders/relu_forward.comp`

Pipeline technique:

1. CMake compile les `.comp` en `.spv` via `glslangValidator`.
2. Les `.spv` sont copiés dans `bin/shaders`.
3. `VulkanCompute.hpp` charge les SPIR-V, crée pipeline/layout/descriptor pool.
4. `VulkanRuntime::forwardLayer` route les layers compatibles vers ces kernels.
5. Les autres opérations et le backward passent par `PortableRuntime`, décrit ci-dessous.
6. En cas d'indisponibilité shader/device, le runtime retourne `false` et le routeur propose l'opération au backend suivant.

---

## 7) Invariants et gotchas

### Aucun état persistent entre les couches

Les `DeviceBuf` sont alloués/libérés à chaque appel de fast-path. Il n'y a pas de pool de buffers GPU. C'est délibéré : simplicité d'implémentation. L'overhead d'allocation est limité car `cudaMalloc` est relativement rapide pour des buffers de même taille (le driver CUDA met en cache).

**`training=true` et dispatch réel**

Le dispatch runtime est disponible en entraînement comme en inférence pour les fast-paths activés (`Linear`, `MatMul/BatchMatMul`, `Conv2d`, normes, attentions) côté CUDA/ROCm.

Le backward n'est pas déporté vers ces runtimes dans ces fast-paths : l'accélération concerne la passe forward des layers.

### Pas de mixed precision

Tous les fast-paths travaillent en `float32`. Les types `F16`/`BF16` du modèle ne sont pas utilisés ici (la conversion est faite ailleurs, si applicable).

### Masque causal sur host

Les kernels d'attention de référence appliquent le masque causal sur le device.
Le coût du staging et des kernels séquentiels doit être mesuré pour les longues
séquences ; la couverture native ne constitue pas une garantie de performance.

### Ordre des builds

CUDA et ROCm sont deux options CMake indépendantes (`ENABLE_CUDA` et
`ENABLE_ROCM`). Les deux backends peuvent donc être présents dans le même
binaire si leurs bibliothèques sont détectées. Au runtime, le routeur essaie
ROCm avant CUDA, puis Vulkan, OpenCL et CPU ; une opération refusée par un
backend est proposée au suivant.

---

## 8) Ajouter un nouveau fast-path GPU

Les layers du modèle passent par `RuntimeRouter` en forward et backward. Ajouter le calcul dans le runtime, déclarer sa capacité réelle et tester le runtime sélectionné ; aucun nouveau switch mathématique ne doit être ajouté dans `Model.cpp`.

1. Ajouter les flags dans `RuntimeConfig` (`src/runtimes/AbstractRuntime.hpp`)
2. Ajouter le parsing `fromEnv` correspondant
3. Implémenter le `case` dans `CudaRuntime::forwardLayer`
4. Implémenter le `case` miroir dans `RocmRuntime::forwardLayer`
5. Mettre à jour le tableau de synthèse dans `docs/04-Architecture-Internals/03-Hardware-Backends.md`

## Étapes suivantes

- [Page précédente : Internals : CLI (binaire `mimir`) et points d’entrée](20-CLI-EntryPoints.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : Execution Planner (C++)](22-Planning.md)

## OpenCL et Vulkan : kernels de référence partagés

Les backends Vulkan et OpenCL conservent leurs chemins spécialisés et utilisent
`PortableRuntime` pour les autres opérations et les passes arrière. Leur
couverture par type correspond au dispatcher CPU exécutable : **72 types en
forward et 73 en backward**, dont NMS avec stop-gradient. `Lambda` est refusé
en forward par le dispatcher CPU lui-même ; son backward identité est supporté
sur CPU, Vulkan et OpenCL. CUDA possède les mêmes familles de kernels mais
n'implémente pas le backward `Lambda`.

Les familles couvertes comprennent convolutions, normalisations, attention,
projections/embeddings, pooling, transformations de tenseurs, récurrence,
upsampling, dropout et reparamétrisation. Les configurations, formes de tenseurs
et limites du device restent vérifiées ; cette couverture par type ne garantit
pas que toute forme soit acceptée.

Les nouveaux kernels de référence OpenCL/Vulkan exécutent leur calcul sur le
device avec une invocation qui traite le travail logique séquentiellement.
Ils privilégient la conformité numérique ; leur couverture ne garantit pas des
performances supérieures au CPU. Le staging des tenseurs et l'accumulation des
gradients de paramètres retournés restent sur l'hôte. Les calculs forward et
backward ne sont pas remplacés par un appel arithmétique CPU caché.

### Sources et génération

- `src/runtimes/portable/PortableRuntime.cpp` valide et orchestre les opérations.
- `PortableStaging.hpp` prépare les buffers et les arguments des kernels.
- `PortableDevice.cpp` soumet les calculs à OpenCL compute ou Vulkan compute.
- `reference.cl` et `reference.comp` contiennent les kernels générés.
- `PortableShaderData.hpp` embarque le source OpenCL et le SPIR-V Vulkan.

Le générateur réutilise les algorithmes GPU existants alignés sur le contrat
CPU. Pour régénérer les fichiers, il faut Python 3, `glslangValidator` et
`spirv-val` :

```bash
python3 scripts/dev/generate_portable_reference.py
```

Cette génération a été exécutée lors de l'implémentation. Les builds ordinaires
utilisent les sources/SPIR-V embarqués ; les kernels de référence ne dépendent
pas d'un toolkit CUDA.

### Validation

Recette CMake/CTest pour un build avec tests (adapter le dossier de build ;
les validations matérielles locales sont précisées ci-dessous) :

```bash
cmake -S . -B build-gpu-tests -DENABLE_OPENCL=ON -DENABLE_VULKAN=ON -DMIMIR_ENABLE_TESTS=ON -DBUILD_TESTING=ON
cmake --build build-gpu-tests --target PortableNativeOps RuntimeTest.NMS GpuBackwardParity -j2
ctest --test-dir build-gpu-tests --output-on-failure -R '^RuntimeTest\.(NMS|(OPENCL|VULKAN)(NativeOps|BackwardParity))$'
```

Le 11 octobre 2026, les exécutables de tests ont été compilés et lancés
directement contre la bibliothèque locale : 165 cas numériques et quatre
configurations NMS par backend passent avec OpenCL/Rusticl sur le GPU AMD
Radeon Graphics et Vulkan/RADV RENOIR sur ce même GPU. Les régressions de
gradients historiques et le test CPU NMS passent également. Vulkan/Lavapipe
a aussi passé les comparaisons numériques pendant l'implémentation.

Les tests NMS incluent les classes optionnelles, les entrées vides, les formes
invalides, le stop-gradient et la sélection du runtime. Le backward CPU est
la référence des comparaisons GPU. CUDA dispose du kernel NMS forward/backward
et des tests correspondants ; son exécution sur GPU NVIDIA reste à valider.
Ces résultats prouvent la conformité et le routage, sans mesure de performance.

Voir aussi [le contrat NMS](../03-API-Reference/01-Layers.md#contrat-de-nms)
et [les sources portables](../../src/runtimes/portable/README.md).


### Benchmarks Colab et validation du 18 octobre 2026

Les 165 cas de parité native par backend ont été revérifiés sur AMD Radeon
Graphics avec Vulkan/RADV et OpenCL/Rusticl. Le contrôle des budgets portables
passe aussi. Le profil Colab `portable` valide sept cas par runtime, avec
sélection explicite de `VULKAN` ou `OPENCL`, sans repli CPU attribué au GPU.
Cette validation locale ne prouve pas la disponibilité du pilote NVIDIA dans Colab.

Le microbenchmark peut mesurer plusieurs runtimes Mímir dans le même run, chacun
dans son processus et sous son identité. Le profil portable partage les mêmes
formes entre tous les concurrents ; les budgets de sécurité restent actifs.
Voir [Benchmarks Colab](../05-Advanced/07-Colab-Benchmarks.md) pour les profils,
la provenance, le RSS et le dépannage des anciennes cellules.
