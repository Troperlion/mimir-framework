# Mémoire

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

## Diagramme d'explication

![Sous-système mémoire](../graphs/08_memory_subsystem.svg)


Cette page décrit le système mémoire côté runtime : comment la limite RAM est appliquée, quelles allocations sont comptabilisées, et comment les buffers temporaires sont gérés.

Sources principales :

- `src/MemoryGuard.hpp`
- `src/runtimes/cpu/RuntimeAllocator.hpp`
- `src/runtimes/cpu/DynamicTensorAllocator.hpp`
- (optionnel, implémentation header-only) `src/AdvancedRAMManager.hpp`

## Objectifs

- Éviter les OOM “silencieux” en rendant la consommation mémoire visible et contrôlée.
- Permettre une **limite stricte** et une comptabilisation (courant / pic).
- Centraliser les allocations runtime (activations, temporaires, intermédiaires) au lieu de disperser des `std::vector` et `new[]` un peu partout.

## Les 3 briques principales

### 1) `MemoryGuard` : comptabilisation et limites

`MemoryGuard` est le gardien : il conserve un état “courant” et “peak”, et impose une limite globale en octets.

Fonctions typiques :

- `requestAllocation(bytes, tag)` : demande d’allocation, retourne `false` si la limite serait dépassée.
- `releaseAllocation(bytes)` : libère la réservation correspondante.
- `getCurrentBytes()`, `getPeakBytes()`, `getLimit()` : statistiques.

Modes “stricts” utiles au debug :

- blocage des allocations (mode “block”) pour détecter des chemins qui allouent au mauvais endroit
- gel / freeze (selon l’implémentation) pour figer un budget et vérifier que le forward reste stable

Important : `MemoryGuard` ne peut pas intercepter *toutes* les allocations C++ par magie. Il ne comptabilise que ce qui passe volontairement par lui (ou par des wrappers qui l’utilisent).

### 2) `RuntimeAllocator` : allocateur runtime RAII

`RuntimeAllocator` est l’interface “user-friendly” utilisée par le runtime pour les activations et buffers temporaires.

Ce qu’il fait :

- applique `MemoryGuard::requestAllocation` avant d’allouer
- renvoie des handles RAII (`TensorHandle`, `BufferHandle`) qui relâchent automatiquement via `releaseAllocation` au destructeur
- propose un **pool de scratchpads** pour réutiliser des buffers et éviter `alloc/free` répétitifs

Points importants (observables dans `src/runtimes/cpu/RuntimeAllocator.hpp`) :

- `allocate_tensor(shape, dtype, name)` alloue un `std::vector<float>` et le comptabilise via `MemoryGuard`.
- `allocate_buffer(bytes, tag)` alloue via `new[]` et comptabilise via `MemoryGuard`.
- `get_scratchpad(min_bytes, tag)` / `return_scratchpad(...)` : un buffer peut être conservé dans un pool.

Conséquence :

- tant qu’un scratchpad est conservé dans le pool, la mémoire reste *réservée* (donc `getCurrentBytes()` reste non nul).
- il faut vider le pool (`clear_scratchpad_pool`) quand on veut relâcher réellement ce budget.

`RuntimeAllocator` implémente aussi un **cap local optionnel** (en plus de la limite du `MemoryGuard`) : c’est un “garde-fou” additionnel, sans modifier la limite globale.

### 3) `DynamicTensorAllocator` : gestion avancée (réservation, cache, compression)

`DynamicTensorAllocator` vise à gérer des tenseurs/buffers de manière plus flexible :

- réservations “lazy” vs “non-lazy” (côté `MemoryGuard`)
- intégration avec un backend de cache (AdvancedRAMManager)
- éviction / compression (ex: LZ4) selon configuration

Attention à un détail important : dans la version actuelle, certaines allocations de buffers se font via `malloc/free` tout en conservant une **comptabilisation par réservation**.

Autrement dit :

- `MemoryGuard` sert de **compteur/budget**
- mais l’allocation réelle mémoire peut être effectuée par le système (malloc)

Ce design fonctionne tant que :

- toutes les allocations runtime importantes suivent la discipline “requestAllocation/releaseAllocation”
- et qu’on évite de créer des buffers géants hors allocateur (sinon le compteur ne reflète plus la réalité)

## Comment diagnostiquer une explosion mémoire

1) Démarrer avec une limite faible et augmenter progressivement.
2) Surveiller `peak` : c’est souvent plus informatif que `current`.
3) En cas d’OOM, regarder quel tag a déclenché la demande (les APIs acceptent des tags / noms).
4) Si `current` ne redescend jamais :

- vérifier le pool de scratchpads
- vérifier que les handles RAII se destructent bien (pas de stockage global involontaire)

## Recommandations pratiques

- Pour les ops temporaires, utiliser `RuntimeAllocator::get_scratchpad` plutôt que des `std::vector` locaux dans des boucles.
- Éviter d’allouer de gros buffers directement dans les scripts Lua : préférer des primitives runtime/ops qui passent par `MemoryGuard`.
- Pour des workloads “streaming”, vider périodiquement le pool scratchpad si la mémoire doit être rendue au système.

## Étapes suivantes

- [Page précédente : Internals - moteur d’exécution](01-Engine-Overview.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Backends hardware : CPU / CUDA / ROCm / Vulkan / OpenCL](03-Hardware-Backends.md)
