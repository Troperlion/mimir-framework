# Fonctionnement du framework

Implémenter ou modifier des briques techniques sans casser le contrat global.

**Public concerné :** Développeur framework (C/C++/runtime/scripting).

> **Prérequis**
>
> Comprendre le registre d'architectures et les conventions I/O.

Ce chapitre decrit la boucle centrale du framework, du point de vue developpeur.

## Sur cette page

- [1. Vue d'ensemble](#1-vue-densemble)
- [Role explicite de la Viz dans le framework](#role-explicite-de-la-viz-dans-le-framework)
- [2. Cycle de vie standard d'un modele](#2-cycle-de-vie-standard-dun-modele)
- [3. Topologie et execution](#3-topologie-et-execution)
- [4. Pourquoi le registre est central](#4-pourquoi-le-registre-est-central)
- [5. Runtime et dispatch](#5-runtime-et-dispatch)
- [6. Contrat scripting et stabilite API](#6-contrat-scripting-et-stabilite-api)
- [7. Contraintes metier a respecter](#7-contraintes-metier-a-respecter)
- [8. Demo metier - run reproductible en mode conf](#8-demo-metier---run-reproductible-en-mode-conf)
- [9. Demo metier - boucle d'analyse architecture](#9-demo-metier---boucle-danalyse-architecture)
- [Étapes suivantes](#étapes-suivantes)

## 1. Vue d'ensemble

Le framework tourne autour de 5 blocs :

1. `Model` : topologie, forward/backward, et etat d'entrainement.
2. `Layer` : unite de calcul (Linear, Conv2d, Add, Norm, etc.).
3. Registre d'architectures : fabrique de modeles a partir d'un nom + config.
4. Runtime backend : execution CPU/GPU des operations.
5. Bridges scripting : exposition de l'API framework (Lua aujourd'hui, autres demain).

## Role explicite de la Viz dans le framework

La visualisation n'est pas un "widget UI" annexe. Dans Mimir, la Viz est une brique
d'observabilite du runtime et de l'entrainement.

Role fonctionnel:

1. Rendre visibles les etats internes (tensors intermediaires, reconstructions, latent, diff).
2. Permettre le diagnostic rapide des regressions de shape/layout/couleur.
3. Exposer les metriques live de training sans bloquer la boucle de calcul.
4. Servir de contrat operatoire entre modele, monitor asynchrone et interface SFML.

Role architectural:

- Le modele produit des "tips" (viz taps) pendant `forwardPass`.
- `AsyncMonitor` transporte ces donnees vers la couche UI.
- `Visualizer` transforme ces frames en textures, panels et controles interactifs.

Consequence dev:

- Toute modification de la Viz doit preserver la stabilite du chemin training/runtime.
- Filtrer une preview ne doit jamais interrompre l'execution du layer (pas de `continue`
  sur la boucle principale d'execution de `Model`).

## 2. Cycle de vie standard d'un modele

Cycle conseille en production :

1. `create` du modele (depuis le registre).
2. `allocateParams`.
3. `initializeWeights` ou chargement checkpoint.
4. `forwardPass` / `forwardPassNamed`.
5. training step (`trainStep*`) puis `optimizerStep` si entrainement.
6. sauvegarde (`Serialization`).

Le pipeline doit rester deterministic et verifiable sur ce cycle.

## 3. Topologie et execution

La topologie est une liste ordonnee de `layers` dans `Model`.

- Chaque layer a un `name`, un `type` et un `params_count`.
- Les dimensions et metadonnees (features/channels/stride/padding/etc.) sont stockees sur le layer.
- Les routes de donnees utilisent `inputs` et `output` (nommes).

Consequence :

- si les I/O sont mal nommes, le graphe devient incoherent,
- si les dimensions sont mal parametrees, l'execution leve une erreur ou produit des sorties invalides.

## 4. Pourquoi le registre est central

Le registre est la porte d'entree stable pour les architectures.

- Il mappe un nom (`vae_conv`, `transformer`, etc.) vers une fonction de creation.
- Il fournit une config par defaut coherente.
- Il harmonise CLI, scripts, et outillage (`inspect_architectures.lua`).

Sans registre, les scripts finissent en hardcoding fragile.

## 5. Runtime et dispatch

Le runtime concret execute les ops.

- `AbstractRuntime` definit le contrat minimum.
- Le backend peut refuser une op (retour `false`) et le framework peut fallback.
- Le tuning runtime est pilote par `RuntimeConfig::fromEnv(...)`.

Point cle : toujours privilegier la correction avant la performance.

## 6. Contrat scripting et stabilite API

Le contrat systeme (globals/aliases) est centralise dans `ScriptingContext`.

Exemples :

- namespace global : `Mimir`
- conf runtime : `CONF`, `CONF_PATH`, `CONF_DIR`
- args script : `arg`
- aliases : `model`, `tokenizer`, `dataset`, `MemoryGuard`, etc.

Objectif : un script conceptuellement identique doit fonctionner entre langages de scripting.

## 7. Contraintes metier a respecter

Le framework est oriente recherche/production reproductible. En pratique :

1. Pipeline config-driven prioritaire (`--conf`) pour rejouer un run a l'identique.
2. Pas de changement silencieux de schema de config.
3. Pas de rupture de contrat sur les aliases scripting systeme.
4. Tout ajout runtime doit garder un fallback correct.
5. Toute nouvelle architecture doit etre inspectable via l'outillage registry.

## 8. Demo metier - run reproductible en mode conf

Exemple de config minimale:

```json
{
  "script": "scripts/templates/template_conf_load_and_train.lua",
  "arch": "vae_conv",
  "seed": 42,
  "epochs": 1,
  "batch_size": 2,
  "dataset_path": "dataset_2"
}
```

Commande de demo :

```bash
./bin/mimir --conf config.json
```

Ce que le dev doit verifier :

1. Le script recoit bien `CONF`, `CONF_PATH`, `CONF_DIR`.
2. Le meme `seed` produit des courbes comparables entre runs.
3. Les logs permettent de retracer archi/config/runtime utilises.

## 9. Demo metier - boucle d'analyse architecture

Avant un changement de modele, lancer une photo de reference :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l vae_conv -p --layers --stats
```

But metier :

- valider que le schema des layers est coherent,
- valider que le nombre de parametres n'a pas derive sans justification,
- garder un point de comparaison avant/apres refactor.

## Étapes suivantes

- [Page précédente : Développement du framework](00-INDEX.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Construire Un Modele Et Ses Layers](02-Building-Models-And-Layers.md)
