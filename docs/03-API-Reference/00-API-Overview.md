# Vue d’ensemble de l’API Lua

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.

La table publique est enregistrée dans
`src/scriptings/Lua/luaScripting/LuaScripting.cpp`. Les implémentations sont
réparties dans les autres `LuaScripting*.cpp` du même répertoire.

Le contrat système commun (noms globaux + aliases, partagé entre langages de scripting) est défini dans `src/scriptings/ScriptingContext.hpp`.

La specification officielle du contrat de scripting est ici: [docs/03-API-Reference/00-Scripting-Contract.md](00-Scripting-Contract.md).

## Modules principaux

- `Mimir.Model` : create/build/allocate/init/forward/backward + helpers d'entraînement + `dtype` + `create_from_config`
- `Mimir.Architectures` : liste et config par défaut des architectures (+ `info`, `dtypes`)
- `Mimir.Serialization` : save/load, formats, debug dumps
- `Mimir.Tokenizer` : tokenize/detokenize, BPE
- `Mimir.Dataset` : chargement et itération (selon scripts)
- `Mimir.Database` : builder de chargement avec cache
- `Mimir.IO` : lecture d'images (RGB u8)
- `Mimir.Layers` : inspection du graphe courant par type
- `Mimir.Memory`, `Guard`, `MemoryGuard`, `Allocator` : gestion mémoire
- `Mimir.Htop`, `Mimir.Viz` : monitoring
- Environnement : variables `MIMIR_*` runtime/bridge

Pages de référence associées :

- `Mimir.Model` : `docs/03-API-Reference/10-Model.md`
- `Mimir.Architectures` : `docs/03-API-Reference/11-Architectures.md`
- `Mimir.Tokenizer` : `docs/03-API-Reference/12-Tokenizer.md`
- `Mimir.Dataset` : `docs/03-API-Reference/13-Dataset.md`
- Mémoire : `docs/03-API-Reference/14-Memory.md`
- Viz/Htop : `docs/03-API-Reference/15-Viz-Htop.md`
- Sérialisation (détaillé) : `docs/03-API-Reference/16-Serialization.md`
- Ops layers : `docs/03-API-Reference/18-Layers-Module.md`
- Globals/aliases : `docs/03-API-Reference/19-Globals.md`
- Mapping Lua ↔ C++ (sommaire) : `docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md`
- `Mimir.IO` (I/O images) : `docs/03-API-Reference/21-IO.md`
- Variables d'environnement : `docs/03-API-Reference/22-Environment-Variables.md`

## Convention d’arguments

La plupart des API retournent soit :

- `true/false` + message d’erreur
- ou une table Lua + erreur

Les exemples de scripts dans `scripts/` restent les meilleurs “tests vivants” de l’API.

## Étapes suivantes

- [Revenir à la documentation](../00-INDEX.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Scripting Contract](00-Scripting-Contract.md)
