# Contrat du système de scripting

## Statut

Cette page est la specification officielle du contrat de scripting que tous les bridges doivent respecter.

Source de verite d'implementation:

- Lua: `src/scriptings/Lua/luaScripting/LuaScripting.cpp`
- Contrat commun des bridges: `src/scriptings/ScriptingContext.hpp`
- Commandes de bridge partagees: `src/scriptings/ScriptingBridgeCommon.cpp`

## Contrat obligatoire

Un bridge conforme doit exposer les elements suivants, dans le meme esprit que Lua.

### Variables globales

- `Mimir`
- `arg`
- `CONF`
- `CONF_PATH`
- `CONF_DIR`

### Namespaces

- `Model`
- `Dataset`
- `Tokenizer`
- `Memory`
- `Architectures`
- `Viz`

### Regles d'usage

- `Mimir` est le namespace racine public.
- `arg` doit contenir le chemin du script en position `0`, puis les arguments utilisateurs.
- `CONF`, `CONF_PATH` et `CONF_DIR` doivent etre disponibles pendant toute l'execution du script.
- Les namespaces exposes doivent rester stables et nommes de facon consistente entre les bridges.
- Un bridge peut offrir des raccourcis, mais il ne doit pas masquer le contrat principal.

## Cycle de vie

L'ordre attendu pour la creation et l'utilisation d'un modele est:

```text
Model.create()
↓
allocate_params()
↓
init_weights()
↓
forward()
↓
train()
```

### Regles de cycle de vie

- `Model.create()` doit preparer l'architecture et ses metadonnees.
- `allocate_params()` doit reserver les parametres avant toute initialisation.
- `init_weights()` doit initialiser les poids apres allocation.
- `forward()` ne doit pas supposer des parametres non alloues.
- `train()` doit fonctionner uniquement sur un modele deja initialise.

## Contrat des architectures

- Les architectures sont decouvrables via `Architectures`.
- Les definitions d'architecture doivent etre deterministes.
- Une architecture valide doit pouvoir etre reconstruite a partir de sa configuration.
- Le bridge ne doit pas inventer de comportement implicite hors contrat.

## Priorite d'implementation

1. Lua reste la reference fonctionnelle.
2. Les autres bridges doivent converger vers ce contrat.
3. Toute divergence doit etre consideree comme un ecart a documenter ou a corriger.

## Étapes suivantes

- [Page précédente : API Lua - vue d’ensemble](00-API-Overview.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Layers](01-Layers.md)
