# AdvancedRAMManager

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente `AdvancedRAMManager` et comment il se positionne par rapport aux autres mécanismes mémoire (`MemoryGuard`, `DynamicTensorAllocator`).

## Source de vérité (C++)

- `src/AdvancedRAMManager.hpp`

## Intention

`AdvancedRAMManager` est un gestionnaire “best-effort” pour:

- stocker des blobs mémoire associés à une clé (`std::string key`)
- limiter la RAM consommée (budget `max_ram_bytes`)
- évincer en LRU quand nécessaire
- compresser (optionnel LZ4)
- spill sur disque (optionnel) pour éviter des évictions destructives
- précharger en arrière-plan (optionnel)

Ce n’est pas l’allocator du runtime des tenseurs.

- Les tenseurs runtime passent principalement par `tensor` + `DynamicTensorAllocator`.
- `AdvancedRAMManager` ressemble plutôt à un cache pour assets/buffers annexes (dataset, artefacts, préloads), avec des options “système”.

## Gating compile-time: LZ4

- Si `ENABLE_LZ4` est activé, `LZ4Compression::{compress,decompress}` utilisent LZ4.
- Sinon, les fonctions sont des no-op (retourne l’input).

Le code choisit une compression seulement si elle est “bénéfique”:

- compress si `data.size() > 1024`
- accepte le blob compressé si ratio < 0.9 (≥10% gain)

## Configuration (`AdvancedRAMManager::Config`)

Champs importants:

- `max_ram_bytes`: budget RAM (par défaut 10GB).
- `enable_compression`: permet la compression.
- `enable_async_loading`: workers de préchargement.
- `enable_prediction`: heuristiques de prédiction d’usage.
- `enable_statistics`: collecte des stats d’accès.
- `enable_disk_spill`: autorise l’éviction non destructive (sur disque).
- `spill_dir`: dossier de spill (par défaut `.mimir-spill`).
- `worker_threads`: nombre de threads.

`configure(cfg)`:

- applique la config sous mutex
- prépare `spill_dir` (best-effort)
- démarre les workers si async activé

## API: blocage / freeze

Le gestionnaire expose 2 garde-fous:

- `blockAllocations(true)`: refuse toutes nouvelles allocations (compte les tentatives).
- `freezeAllocations(true)`: similaire, sémantique “gel” (compte les tentatives).

But:

- debugging/stabilité: empêcher des spikes mémoire lors d’une phase critique.

Attention:

- Ce mécanisme ne bloque que les allocations *gérées par* `AdvancedRAMManager`.
- Il ne remplace pas `MemoryGuard` (qui vise l’allocation des tenseurs runtime).

## Stockage interne et métadonnées

Le gestionnaire garde, par clé:

- le blob `data` (compressé ou non)
- `original_size`, `stored_bytes`
- flags: `is_compressed`, `on_disk`
- chemin disque si spill
- compteurs d’accès / timestamps

Il peut aussi conserver des stats (`AccessStats`) pour:

- estimer l’intervalle moyen entre accès
- prédire “sera utilisé bientôt” (`predictWillBeUsedSoon`)

## Allocation: `allocate(key, data, compress)`

Flux simplifié:

1. mutex global
2. refuser si blocked/frozen
3. si `key` existe déjà: succès immédiat
4. compression optionnelle (LZ4)
5. vérifier capacité (`canAllocate(required)`)
6. sinon: éviction LRU (`evictLRU(required, &key)`) et re-check
7. stocker le blob + init metadonnées

Points de design:

- L’éviction LRU est le mécanisme principal.
- Le spill disque (si activé) permet de déplacer un item hors RAM sans le “perdre”.

## Interactions avec le reste du runtime

### Avec `MemoryGuard`

- `MemoryGuard` vise l’allocation runtime (tensors / scratchpads), et fait de l’accounting.
- `AdvancedRAMManager` a son propre budget (bytes) et ses propres règles.

Si vous utilisez les deux:

- vous pouvez être “dans le budget MemoryGuard” mais “hors budget AdvancedRAMManager”, et inversement.

### Avec `DynamicTensorAllocator`

- `DynamicTensorAllocator` est pensé pour les `tensor` runtime (alloc lazy, éviction/compression, handles).
- `AdvancedRAMManager` est un cache générique key→blob.

En pratique:

- éviter d’utiliser `AdvancedRAMManager` pour stocker des poids/activations “au cœur” du forward/backward.

## Pièges & invariants

- “Best-effort”: le manager privilégie la robustesse et la simplicité à la performance maximale.
- Compression: un blob compressé garde un header de taille; si le contenu est déjà entropique, la compression peut être ignorée.
- Spill disque:
  - dépend fortement des IO disque
  - attention aux environnements read-only

## À documenter ensuite

- Où `AdvancedRAMManager` est appelé concrètement (sites d’appel) pour définir son rôle exact dans le framework.
- Les politiques précises d’éviction (LRU vs prédiction) dans les sections plus bas du fichier.

## Étapes suivantes

- [Page précédente : Internals: Monitoring (HtopDisplay / Visualizer / AsyncMonitor)](04-Monitoring-Htop-Visualizer.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : classe `Model` (C++)](10-Model-Class.md)
