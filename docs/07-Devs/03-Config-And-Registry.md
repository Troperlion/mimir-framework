# Configuration et registre d’architectures

Implémenter ou modifier des briques techniques sans casser le contrat global.

**Public concerné :** Développeur framework (C/C++/runtime/scripting).

> **Prérequis**
>
> Comprendre le registre d'architectures et les conventions I/O.


Ce chapitre explique comment composer une config, creer une architecture, et la renseigner au registre.

## Sur cette page

- [1. Pourquoi le registre existe](#1-pourquoi-le-registre-existe)
- [2. Contrat du registre](#2-contrat-du-registre)
- [3. Composer une config modele](#3-composer-une-config-modele)
- [4. Ajouter une nouvelle architecture](#4-ajouter-une-nouvelle-architecture)
- [5. Exemple conceptuel d'entree registre](#5-exemple-conceptuel-dentree-registre)
- [6. Validation rapide](#6-validation-rapide)
- [7. Anti-patterns a eviter](#7-anti-patterns-a-eviter)
- [8. Demo metier - cycle complet registre](#8-demo-metier---cycle-complet-registre)
- [9. Demo metier - overlays de config sans casser la compat](#9-demo-metier---overlays-de-config-sans-casser-la-compat)
- [10. Checklist metier de release d'une architecture](#10-checklist-metier-de-release-dune-architecture)
- [Étapes suivantes](#étapes-suivantes)

## 1. Pourquoi le registre existe

Le registre `ModelArchitectures` fournit :

- la liste des architectures supportees,
- la config par defaut de chaque architecture,
- la creation standardisee d'un modele a partir d'un nom + config.

C'est le point de convergence entre C++, CLI, scripts, et outils.

## 2. Contrat du registre

Elements principaux :

- `Entry` : `{ name, description, default_config, create }`
- `Registry::registerArchitecture(Entry)`
- `Registry::defaultConfig(name)`
- `Registry::create(name, config)`

Le `create` est une lambda/fonction qui instancie le bon modele et applique sa config.

## 3. Composer une config modele

Bonne pratique en 3 couches :

1. config par defaut (dans le registre),
2. overrides architecture-specifiques,
3. overrides runtime/CLI/script.

Regle de qualite : la config finale doit rester serialisable et explicite.

## 4. Ajouter une nouvelle architecture

Procedure conseillee :

1. Creer `MyNewModel.hpp/.cpp` dans `src/Models/...`.
2. Definir une `Config` interne et des defaults robustes.
3. Ajouter les helpers JSON -> Config et default JSON dans le registre.
4. Ajouter une `Entry` avec `name/description/default_config/create`.
5. Verifier avec `inspect_architectures.lua`.

## 5. Exemple conceptuel d'entree registre

```cpp
registry.registerArchitecture({
    "my_new_model",
    "Description courte",
    myNewDefaultConfigJson(),
    [](const json& cfg) {
        auto c = myNewCfgFromJson(cfg);
        auto m = std::make_shared<MyNewModel>();
        m->buildFromConfig(c);
        return m;
    }
});
```

## 6. Validation rapide

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l my_new_model -p --layers --stats
```

Attendu :

- architecture visible,
- config par defaut lisible,
- layers coherents,
- total params coherent.

## 7. Anti-patterns a eviter

- hardcoder des hyperparams dans les scripts au lieu du registre,
- oublier de mettre a jour `default_config`,
- creer des aliases implicites non documentes,
- changer silencieusement le sens d'un champ de config existant.

## 8. Demo metier - cycle complet registre

Objectif : ajouter une architecture et la rendre exploitable par tous les points d'entree.

1. Ajouter l'entree dans le registre C++ (`name`, `description`, `default_config`, `create`).
2. Recompiler.
3. Lister l'architecture dans l'outil d'inspection.
4. Charger la config par defaut en Lua.

Commandes de validation :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l my_new_model -p --layers --stats
```

Smoke Lua de verification :

```lua
local cfg, err = Mimir.Architectures.default_config("my_new_model")
if not cfg then error(err) end
print("default config chargee")
```

## 9. Demo metier - overlays de config sans casser la compat

Principe : ne jamais casser les anciennes cles, ajouter des cles optionnelles.

Exemple :

```json
{
    "arch": "my_new_model",
    "model": {
        "width": 256,
        "depth": 12,
        "dropout": 0.1
    }
}
```

Puis override ponctuel :

```bash
./bin/mimir --conf config.json --override model.depth=16
```

Attendu metier :

1. Le run reste compatible avec les anciennes configs.
2. La provenance de la valeur finale est explicable (default + override).
3. Le schema final reste serialisable pour audit.

## 10. Checklist metier de release d'une architecture

1. `default_config` present et documente.
2. Inspection outillage OK (`-a`, `-l`, `--layers`, `--stats`).
3. Un script training/inference de smoke passe.
4. Aucun alias scripting metier existant n'est casse.

## Étapes suivantes

- [Page précédente : Construire Un Modele Et Ses Layers](02-Building-Models-And-Layers.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Runtime - Modifier Et Ajouter Un Backend](04-Runtime-Development.md)
