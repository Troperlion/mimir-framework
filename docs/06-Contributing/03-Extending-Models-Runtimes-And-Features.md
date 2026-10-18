# Étendre le framework

Contribuer avec des changements cohérents et maintenables.

**Public concerné :** Contributeur du projet.

> **Prérequis**
>
> Connaître le workflow Git et les bases du projet.

## Sur cette page

- [Diagramme d'explication](#diagramme-dexplication)
- [Comment utiliser ce guide](#comment-utiliser-ce-guide)
- [1. Vue d'ensemble de l'extension](#1-vue-densemble-de-lextension)
- [2. Ajouter un nouveau modèle](#2-ajouter-un-nouveau-modèle)
- [3. Ajouter un runtime (backend d'exécution)](#3-ajouter-un-runtime-backend-dexécution)
- [4. Ajouter une fonctionnalité transverse](#4-ajouter-une-fonctionnalité-transverse)
- [5. Ajouter une nouvelle entrée de scripting (autre que Lua)](#5-ajouter-une-nouvelle-entrée-de-scripting-autre-que-lua)
- [6. Plan de validation complet (avant merge)](#6-plan-de-validation-complet-avant-merge)
- [7. Template de PR recommandé](#7-template-de-pr-recommandé)
- [Étapes suivantes](#étapes-suivantes)

## Diagramme d'explication

![Points d'extension du framework](../graphs/16_extension_points.svg)

Ce chapitre est le guide de référence pour ajouter des capacités au framework côté code source.

## Comment utiliser ce guide

Lis ce document comme une carte de navigation :

1. Vous modifiez un modèle : va à la section 2.
2. Vous modifiez un runtime : va à la section 3.
3. Vous ajoutez une feature transverse : va à la section 4.
4. Vous ajoutez une entrée de scripting : va à la section 5.

Objectif pédagogique : transformer une idée technique en changement testable avec un minimum de régressions.

Objectifs :

- ajouter un nouveau modèle (architecture),
- ajouter un runtime d'exécution,
- ajouter une fonctionnalité transverse (ops, sérialisation, CLI, outils),
- ouvrir une nouvelle entrée de scripting autre que Lua.

## 1. Vue d'ensemble de l'extension

Le framework se compose de 5 zones qui évoluent ensemble :

1. noyau modèle/layers,
2. registre d'architectures,
3. runtime backend,
4. binding scripting,
5. scripts et outils de validation.

Quand vous ajoutez une brique, validez toujours la chaîne complète :

create -> allocate -> init/load -> forward -> save/load -> script smoke test.

Règle pratique : si une étape de cette chaîne casse, la PR n'est pas prête.

## 2. Ajouter un nouveau modèle

### 2.1 Fichiers à créer

Créer le modèle dans un sous-dossier métier de `src/Models/`, par exemple :

- `src/Models/MyDomain/MyNewModel.hpp`
- `src/Models/MyDomain/MyNewModel.cpp`

### 2.2 Contrat minimal attendu

Le modèle doit :

- exposer une config avec valeurs par défaut stables,
- construire explicitement ses layers,
- supporter allocation des paramètres,
- supporter initialisation ou chargement des poids,
- exécuter un forward cohérent,
- rester sérialisable.

### 2.3 Enregistrement dans le registre

Déclarer l'architecture dans :

- `src/Models/Registry/ModelArchitectures.hpp`
- `src/Models/Registry/ModelArchitectures.cpp`

Checklist registre :

- include du nouveau modèle,
- helper JSON -> Config,
- helper config JSON par défaut,
- entrée `name/description/default_config/create`.

### 2.4 Validation rapide

Commandes utiles :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l my_new_model -p --layers --stats
```

Attendu :

1. Le modèle apparaît dans la liste.
2. La config par défaut est lisible.
3. Le graphe est inspectable sans erreur.

## 3. Ajouter un runtime (backend d'exécution)

Le framework inclut une base runtime dans `src/runtimes/`.

Fichiers actuels :

- `src/runtimes/AbstractRuntime.hpp`
- `src/runtimes/AbstractRuntime.cpp`
- `src/runtimes/cpu/`
- `src/runtimes/cuda/`
- `src/runtimes/rocm/`

### 3.1 Stratégie recommandée

1. Implémenter le nouveau backend à partir du contrat `AbstractRuntime`.
2. Définir clairement les capacités supportées (ops, dtype, layouts).
3. Gérer un fallback sûr vers CPU quand une op n'est pas disponible.
4. Ajouter des logs de capacité au démarrage (alignés avec la détection hardware existante).
5. Ajouter au minimum un test de non-régression sur une graph simple.

### 3.2 Points de vigilance

- cohérence des shapes (NCHW/NHWC selon les ops),
- cohérence dtype (float16/bfloat16/float32...),
- stabilité numérique,
- mémoire temporaire (scratchpads) et cycle de vie,
- comportement identique entre runtimes pour un même graphe.

Approche recommandée :

1. Commencer par une seule op stable.
2. Vérifier fallback et logs.
3. Élargir progressivement le périmètre d'ops.

## 4. Ajouter une fonctionnalité transverse

Exemples de fonctionnalités transverses :

- nouvelle op/layer,
- nouveau format ou champ de sérialisation,
- nouvelle option CLI,
- nouvel outil dans `scripts/tools/`.

### 4.1 Nouvelle op/layer

Zone principale :

- `src/Layers.hpp`
- `src/Model.cpp`

Checklist :

- type layer défini,
- forward implémenté,
- backward implémenté si entraînement,
- validation shape/dtype,
- tests ciblés.

Bon pattern de livraison :

1. Ajouter le type et le dispatch.
2. Ajouter au moins un test nominal.
3. Ajouter un test de forme invalide.
4. Mettre à jour la doc API correspondante.

### 4.2 Option CLI et comportement d'exécution

Zone principale :

- `src/main.cpp`

Checklist :

- parsing robuste,
- messages d'aide clairs,
- interaction correcte avec le mode config-driven (`--conf`),
- comportement déterministe si option absente.

### 4.3 Outil développeur

Zone principale :

- `scripts/tools/*.lua`

Checklist :

- mode texte lisible,
- option JSON pour automation CI,
- erreurs explicites,
- exemple d'utilisation dans la documentation.

## 5. Ajouter une nouvelle entrée de scripting (autre que Lua)

Principe : pour supporter un autre langage (python, ruby, js, perl, java, rust, etc.), il suffit de créer un système équivalent à Lua, en reproduisant le même contrat d'intégration que :

- `src/scriptings/ScriptingRuntime.hpp`
- `src/scriptings/ScriptingContext.hpp`
- `src/scriptings/Lua/luaScripting/LuaScripting.hpp`
- `src/scriptings/Lua/luaScripting/LuaScripting.cpp`

Note : `src/scriptings/Lua/luaScripting/LuaScripting.hpp` reste un include de compatibilité. Le point d'extension canonique est désormais `src/scriptings/...`.

### 5.1 Ce qu'il faut reproduire

Le nouveau bridge de scripting doit au minimum gérer :

1. initialisation/arrêt du runtime du langage,
2. exécution d'un script avec passage d'arguments,
3. exposition des APIs framework (Model, Architectures, Dataset, Tokenizer...),
4. propagation des erreurs script vers C++,
5. intégration avec le mode `--conf` et l'injection des variables standard (`CONF`, `CONF_PATH`, `CONF_DIR`, `arg`).

Ces noms globaux sont normalisés dans `ScriptingContext` (constantes `kGlobal*` et `kAlias*`) pour éviter les divergences entre langages.

### 5.2 Intégration CLI

Ajouter une nouvelle entrée d'exécution dans `src/main.cpp`.

Exemples de patterns :

- `--python path/to/script.py -- ...`
- `--ruby path/to/script.rb -- ...`
- `--js path/to/script.js -- ...`

Le séparateur `--` doit conserver le même comportement que l'entrée Lua pour garder une UX homogène.

### 5.3 Architecture technique possible

Deux approches principales :

1. embedding natif de l'interpréteur/VM dans le process C++,
2. process externe + protocole RPC local.

Approche 1 : meilleure perf et accès direct mémoire.
Approche 2 : isolation plus forte et crash containment.

### 5.4 Recommandation d'alignement API

Pour limiter la dette de maintenance, garder une surface API identique entre langages :

- mêmes noms de fonctions haut niveau,
- mêmes conventions d'entrée/sortie,
- mêmes codes d'erreur,
- mêmes comportements par défaut.

Objectif : un script de pipeline doit rester conceptuellement portable d'un langage à l'autre.

## 6. Plan de validation complet (avant merge)

1. Build clean en local (Debug puis Release si possible).
2. Inspection registre et config par défaut.
3. Exécution d'un run minimal train/inference.
4. Test save/load sur checkpoint.
5. Test smoke scripting (et multi-entry si ajout d'un nouveau langage).
6. Documentation mise à jour dans Contributing + API/Guide utilisateur si nécessaire.

Conseil : exécuter les commandes de validation dans l'ordre et coller les sorties clés dans la PR.

## 7. Template de PR recommandé

Inclure dans la PR :

- scope exact (model/runtime/feature/scripting),
- fichiers touchés,
- risques techniques,
- plan de rollback,
- commandes de validation exécutées,
- sortie de test principale.

Ce format accélère la revue et réduit les regressions silencieuses.

## Étapes suivantes

- [Page précédente : Développeurs : Ajouter une Architecture et Utiliser les Outils](02-New-Architecture-And-Tools.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tutoriel pas-à-pas : Ajouter une entrée scripting Python](04-Tutorial-Add-Python-Scripting-Entry.md)
