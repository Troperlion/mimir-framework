# Mémoire (Allocator, MemoryGuard)

Éviter les OOM avec une configuration mémoire simple et sûre.

**Public concerné :** Débutant à intermédiaire.

> **Prérequis**
>
> Connaître la RAM disponible sur votre machine.


Mímir est “CPU-first” et peut allouer de gros buffers. La pratique recommandée est d’activer :

1) une limite stricte (MemoryGuard)
2) un allocateur dynamique avec compression (Allocator)

## Configurer

Exemple (Lua) :

```lua
-- Limite stricte (GB)
if Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, 10)
end

-- Allocateur dynamique (GB)
local ok, err = Mimir.Allocator.configure({
  max_ram_gb = 10.0,
  enable_compression = true,
  swap_strategy = "lru",
})
assert(ok ~= false, err)
```

## Pourquoi deux systèmes ?

- `MemoryGuard` : coupe court (évite les OOM silencieux / swap infini).
- `Allocator` : gère une “pression mémoire” (évictions/compression) pour les tenseurs dynamiques.

## Paramètres importants (Allocator)

| Clé | Type | Effet | Conseil |
| --- | --- | --- | --- |
| `max_ram_gb` | number | budget RAM pour tenseurs dynamiques | commence bas (8–12GB) et monte |
| `enable_compression` | bool | compresse des buffers évincés | `true` si vous visez stabilité |
| `swap_strategy` | string | stratégie d’éviction | `"lru"` est un bon défaut |

## Patterns recommandés

### Toujours définir une limite dure

Sans limite, un script peut sembler “bloqué” (swap) ou crasher tard.

```lua
pcall(Mimir.MemoryGuard.setLimit, 10)
```

### Toujours configurer l’Allocator dans les scripts “longs”

Bench/training = pression mémoire variable.

```lua
assert(Mimir.Allocator.configure({
  max_ram_gb = 10.0,
  enable_compression = true,
  swap_strategy = "lru",
}))
```

## Diagnostiquer

- `Mimir.Memory.get_stats()`, `print_stats()`
- `Mimir.MemoryGuard.getStats()`
- `Mimir.Allocator.get_stats()`

Conseils debug :

- Si ça dépasse la limite, baisse `seq_len`, `d_model`, ou `batch` (quand applicable) avant d’augmenter la RAM.
- Quand vous comparez des runs, fixe un `seed` et garde `max_ram_gb` constant.

## Étapes suivantes

- [Page précédente : Mode `--conf`: Config-Driven Scripting](08-Config-Driven-Scripting.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Scripts et outils Lua](10-Examples.md)
