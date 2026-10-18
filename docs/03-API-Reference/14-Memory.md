# Mémoire

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Source : `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.

## `Mimir.Memory`

- `config(table)`
- `get_stats()` / `print_stats()`
- `clear()`
- `get_usage()`
- `set_limit(gb)`

## `Mimir.Guard` (strict)

- `set_limit(gb)`
- `get_stats()` / `print_stats()`
- `reset()`

## `Mimir.MemoryGuard` (alias moderne)

- `setLimit(gb)`
- `getCurrentUsage()`, `getPeakUsage()`, `getLimit()`
- `getStats()`, `printStats()`, `reset()`

## `Mimir.Allocator`

- `configure({max_ram_gb, enable_compression, swap_strategy, ...})`
- `get_stats()` / `print_stats()`

Recommandation : activer `MemoryGuard` + `Allocator` au début des scripts.

## Étapes suivantes

- [Page précédente : API : `Mimir.Dataset`](13-Dataset.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : monitoring / visualisation](15-Viz-Htop.md)
