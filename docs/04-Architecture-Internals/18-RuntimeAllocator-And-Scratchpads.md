# Internals : `RuntimeAllocator` et scratchpads

Cette page documente le gestionnaire mémoire runtime utilisé dans les hot-path (forward/backward) pour éviter les allocations sauvages et respecter les limites `MemoryGuard`.

Source de vérité :

- Runtime allocator : `src/RuntimeAllocator.hpp`
- Memory guard : `src/MemoryGuard.hpp`
- Dynamic allocator : `src/DynamicTensorAllocator.hpp` (autre stratégie)
- Utilisation dans Model : `src/Model.cpp` (création `RuntimeAllocator allocator(guard, cap_mb)`)

## 1) Objectif

- Toute allocation d’activations/buffers temporaires doit être **comptabilisée**.
- Réduire la fragmentation et les `new/delete` répétés via des scratchpads réutilisables.

## 2) Deux types de handles RAII

### A) `TensorHandle`

- encapsule un `std::vector<float>` + un `TensorDescriptor`.
- au destructeur : `MemoryGuard::releaseAllocation(size_bytes)`.

### B) `BufferHandle`

- encapsule un `float*` alloué avec `new[]`.
- au destructeur : `delete[]` + `releaseAllocation(size_bytes)`.

Invariant : ces handles sont non-copyable et movables.

## 3) Cap local vs limite globale

`RuntimeAllocator` prend :

- une référence vers `MemoryGuard` (limite globale),
- un `max_ram_mb` optionnel (cap local).

Le cap local **ne modifie pas** la limite globale : il sert à garder une marge de sécurité dans un codepath.

## 4) Scratchpad pool

- `get_scratchpad(min_bytes, tag)` :
  - réutilise un buffer existant (par `tag`) si sa taille suffit,
  - sinon alloue un nouveau buffer.

- `return_scratchpad(BufferHandle&&)` : remet un buffer dans le pool.

ATTENTION : le pool retient de la mémoire (comptabilisée) tant qu’il vit. Le caller doit décider quand nettoyer via `clear_scratchpad_pool()`.

## 5) `RUNTIME_CHECK` / mode strict

Le header définit :

- `MIMIR_STRICT_MODE` (par défaut `1`).
- `RUNTIME_ERROR_STRICT(msg)` : throw en strict, sinon log en permissif.
- `RUNTIME_CHECK(cond, msg)`.

Ce mécanisme est utilisé dans le runtime forward/backward pour éviter des corruptions silencieuses.