# Stockage des tenseurs et allocation dynamique

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente la structure `tensor` (C++) utilisée comme **bloc de stockage** pour certains buffers (notamment `Layer::weight_block`) et le sous-système d’allocation dynamique qui l’entoure.

Source de vérité :

- Structure `tensor` : `src/tensors.hpp`
- Implémentation (dynamic alloc + OpenCL legacy) : `src/tensors.cpp`
- Allocateur dynamique : `src/runtimes/cpu/DynamicTensorAllocator.hpp`
- Garde-fou mémoire : `src/MemoryGuard.hpp`
- Utilitaires sécurité : `src/MemorySafety.hpp`
- Utilisation “poids unifiés” : `src/Layers.hpp` (champ `Layer::weight_block`) + `src/Model.cpp` (`allocateParams`, forwards)

## TL;DR

- Le type `tensor` est un conteneur **hybride** : il conserve des champs legacy (`Weight/Pos/Value/Length`) et a été étendu avec `std::vector<float> data` pour supporter le framework.
- Pour limiter la RAM et éviter les OOM, `tensor` peut déléguer son stockage à `DynamicTensorAllocator` (lazy + réserve via `MemoryGuard`).
- Attention : l’allocateur dynamique utilise `malloc/free` sous le capot (donc **MemoryGuard ne peut pas “intercepter”** la RAM au niveau OS) ; la protection se fait par **réservation comptable** (`requestAllocation`/`releaseAllocation`).

## 1) Pourquoi ce type existe

Historiquement, `tensor` représente une unité de données “poids+position+valeur” (les champs `Weight`, `Pos`, `Value`, `Length`). Dans la v2.x du framework, il est réutilisé comme **storage** pour:

- des blocs de poids par layer (`Layer::weight_block`),
- des buffers internes alloués via un gestionnaire plus strict.

Conséquence : ce n’est pas un “Tensor N-D complet” avec shape/stride. C’est un **buffer 1D** de floats (via `data` ou via le handle dynamique).

## 2) API de `tensor` (ce qui est réellement utilisé)

Définition principale : `struct tensor` dans `src/tensors.hpp`.

Champs importants (côté framework moderne) :

- `std::vector<float> data` : stockage classique.
- `void* dynamic_handle` + `bool use_dynamic_alloc` : stockage dynamique.

Méthodes d’accès :

- `float* getData()` / `const float* getData() const`
- `size_t getSize() const`

Règles de construction :

- `tensor(size_t size)` : alloue `data` en dur (non contrôlé par `MemoryGuard`).
- `tensor(size_t size, bool dynamic)` : si `dynamic==true`, crée un handle **sans allouer immédiatement** la RAM (lazy). L’allocation réelle arrive au premier `getData()`.

⚠️ Invariant important : **copie interdite**.

- Copy constructor et copy assignment sont `delete` pour éviter double-free/double-handle.
- Move constructor/assignment déplacent le handle ou le `data`.

## 3) Allocation dynamique : `DynamicTensorAllocator`

But : ne pas faire exploser la RAM lors de gros graphes / gros checkpoints.

### Composants

- `MemoryGuard` : limite stricte, blocage/freeze des allocations.
- `AdvancedRAMManager` : compression/éviction (LRU) et cache.
- `DynamicTensorAllocator` : façade qui fournit des `TensorHandle`.

### `TensorHandle`

Champs clés (`DynamicTensorAllocator::TensorHandle`) :

- `size` (en éléments float)
- `is_loaded`, `is_compressed`
- `cache_key` (identifiant)
- `data_ptr` (mémoire active)
- `reserved`, `reserved_bytes` (réservation comptable côté `MemoryGuard`)

### Lazy vs non-lazy

- **lazy_mode = true** :
  - `allocateTensor` crée seulement le handle.
  - `getTensorData` fait la réservation `MemoryGuard` + alloue via `malloc`.

- **lazy_mode = false** :
  - la réservation `MemoryGuard` est faite dès `allocateTensor`.

### Compression / éviction

- `compressTensor(handle)` : copie les floats vers un blob, demande au RAMManager de stocker compressé, libère `data_ptr`, puis **relâche la réservation** côté `MemoryGuard`.
- `evictLRU(bytes_needed)` (interne) : tente de libérer des tensors peu utilisés pour faire rentrer une nouvelle réservation.

## 4) Sécurité mémoire : comment `MemoryGuard` “protège” vraiment

`MemoryGuard` n’empêche pas l’OS d’allouer : c’est un **contrôle logique**.

- `requestAllocation(bytes)` : accepte/refuse, trace et maintient `current_bytes_`.
- `releaseAllocation(bytes)` : décrémente `current_bytes_`.

Modes spéciaux :

- `blockAllocations(true)` : refuse toute nouvelle allocation.
- `freezeAllocations(true)` : refuse les nouvelles allocations mais autorise les libérations.

**Piège fréquent** : une allocation `std::vector<float>` (ou `new`) ne passe pas par `MemoryGuard`.

- Pour les poids, le chemin “unified weight blocks” via `tensor + DynamicTensorAllocator` limite mieux les risques.

## 5) Interaction avec les layers : `Layer::weight_block`

Dans `src/Layers.hpp` :

- chaque `Layer` possède `tensor* weight_block`.
- `Layer::getWeights()` renvoie soit `weight_block->getData()` (si présent), soit `weights.data()` (fallback compat).
- `Layer::getWeightsSize()` privilégie `params_count` quand `weight_block` est présent.

Invariant : `params_count` doit être correct **avant** l’allocation des poids (voir `docs/04-Architecture-Internals/10-Model-Class.md`).

## 6) OpenCL “TensorSystem” (legacy)

`TensorSystem` (dans `src/tensors.hpp` et `src/tensors.cpp`) contient un kernel OpenCL `compute_weights` destiné à calculer un champ `Weight` à partir de `Pos/Value/Length`.

- Ce chemin est orthogonal aux “tensors du modèle”.
- S’il est compilé (`ENABLE_OPENCL`), il peut exister dans le binaire mais ne signifie pas que les poids du modèle sont sur GPU.

## 7) Checklist (lorsque vous ajoutez/modifies un codepath)

- Si vous créez des poids : préférer `weight_block` + `params_count` exact.
- Éviter d’allouer des gros `std::vector<float>` dans des boucles hot-path sans passer par les scratchpads / allocateurs prévus.
- Si vous modifiez à `DynamicTensorAllocator`, vérifier la cohérence `reserved/reserved_bytes` (sinon MemoryGuard “pense” qu’on fuit).

## Étapes suivantes

- [Page précédente : Internals : `Helpers.hpp` (helpers C++)](11-Helpers.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : Autograd + gradients + backward (C++)](13-Autograd-Gradients.md)
