# Planificateur d’exécution

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

Cette page explique le système de planification statique du graphe d'exécution, avec une lecture orientée compréhension progressive.

Source de vérité : `src/Planning/Planner.hpp`.

Intégration runtime : `src/Model.cpp` (forward) et `Model::static_plan_` dans `src/Model.hpp`.

---

## Sur cette page

- [Lecture rapide (5 minutes)](#lecture-rapide-5-minutes)
- [Rôle](#rôle)
- [Vue mentale : ce qui se passe réellement](#vue-mentale-ce-qui-se-passe-réellement)
- [Ce que fait le planner au lancement](#ce-que-fait-le-planner-au-lancement)
- [1) Analyse des durées de vie (TensorLifetime)](#1-analyse-des-durées-de-vie-tensorlifetime)
- [2) Fusions de layers (FusionKind)](#2-fusions-de-layers-fusionkind)
- [3) Plan d'exécution (ExecutionPlan)](#3-plan-dexécution-executionplan)
- [4) Scratchpad pour Conv2d (Conv2dScratchPlan)](#4-scratchpad-pour-conv2d-conv2dscratchplan)
- [Exemple guidé](#exemple-guidé)
- [5) Intégration avec le runtime](#5-intégration-avec-le-runtime)
- [Déboguer efficacement le planner](#déboguer-efficacement-le-planner)
- [Étapes suivantes](#étapes-suivantes)

## Lecture rapide (5 minutes)

Si vous devez comprendre vite, retiens ceci :

1. Le planner construit un plan au premier `forward`.
2. Ce plan contient les durées de vie, les fusions et les règles de skip.
3. Le plan est mis en cache et réutilisé tant que le graphe et le mode (`training`) ne changent pas.
4. En `training`, les fusions génériques restent désactivées par sécurité.
5. Le planner améliore surtout la stabilité d'exécution et la mémoire, pas seulement la vitesse brute.

## Rôle

Le `Planner` analyse la liste des layers **avant** l'exécution pour :

1. Calculer les durées de vie des tenseurs (quand chaque tenseur est créé/consommé en dernière fois)
2. Identifier les fusions de layers possibles (Conv2d+ReLU et fusions génériques)
3. Pré-calculer les tailles de buffers scratch nécessaires pour Conv2d

Ces informations permettent de :

- Allouer des activations de façon optimale (réutilisation mémoire)
- Éviter des passes intermédiaires (fusions)

En pratique, le planner est le composant qui transforme la liste brute des layers en un plan exécutable stable et réutilisable entre forwards.

---

## Vue mentale : ce qui se passe réellement

Cycle simplifié d'un run :

1. Le modèle reçoit un `forward`.
2. Le runtime vérifie si un plan valide existe déjà.
3. Si non, il calcule un plan statique.
4. Il exécute la boucle de layers en appliquant les règles du plan.
5. Les passes suivantes réutilisent ce plan.

Le gain principal vient du fait que les décisions coûteuses sont déplacées vers une phase de préparation unique.

---

## Ce que fait le planner au lancement

Au démarrage du process, aucun plan n'est encore calculé. Le planner intervient au **premier forward** (ou au premier forward qui invalide le cache plan) :

1. Lecture des toggles runtime (`MIMIR_ENABLE_PLANNER`, `MIMIR_ENABLE_FUSION`, `MIMIR_ENABLE_FUSION_TRAIN`).
2. Vérification du cache `static_plan_` (présence + compatibilité avec le mode `training` + taille des layers inchangée).
3. Si invalide ou absent : construction du plan statique (lifetimes, fusions, skip map, scratch Conv2d).
4. Optionnel : dump de diagnostic si `MIMIR_PLANNER_DUMP=1`.
5. Exécution du forward avec application du plan (skip des layers fusionnés et réutilisation du plan sur les passes suivantes).

Cette phase remplace une partie du coût "décisionnel" à chaque layer par un coût de préparation unique, puis des exécutions plus régulières.

---

## 1) Analyse des durées de vie (`TensorLifetime`)

```cpp
struct TensorLifetime {
    int first_def;          // Index du layer qui crée le tenseur
    int last_use;           // Index du layer qui le consomme en dernier
    std::string dtype;      // Type de données
    std::vector<int> shape; // Forme du tenseur
};

std::unordered_map<std::string, TensorLifetime>
analyze_tensor_lifetimes(const std::vector<Layer>& layers);
```

Utilisation : déterminer quels buffers peuvent être aliasés/réutilisés.

Lecture pratique : plus le `last_use` d'un tenseur est tôt, plus tôt son buffer peut être recyclé.

---

## 2) Fusions de layers (`FusionKind`)

```cpp
enum class FusionKind : uint8_t {
    NONE = 0,
    CONV2D_RELU = 1,
    GENERIC_ACTIVATION = 2,
    GENERIC_SPLIT = 3,
    GENERIC_CHUNK = 4,
    GENERIC_ACTIVATION_SPLIT = 5,
    GENERIC_ACTIVATION_CHUNK = 6,
    GENERIC_UNARY_SHAPE = 7,
};
```

Fusions actuellement implémentées (inférence uniquement, pas en training) :

- Conv2d + ReLU
- Producer + activation standalone (ReLU/SiLU/GELU/...)
- Producer + unary shape (`Flatten`, `Reshape`, `View`, `Transpose`, `Permute`, `Squeeze`, `Unsqueeze`, `Identity`)
- Producer + `Split`/`Chunk`
- Producer + activation + `Split`/`Chunk`
- Chaînes répétées adjacentes automatiquement: `producer -> activation* -> unary_shape*` (et terminaison optionnelle par `Split`/`Chunk`)

Lecture pratique : la fusion en chaîne évite les matérialisations intermédiaires répétitives (exemple typique : `Add -> ReLU -> ReLU -> View`).

---

## 3) Plan d'exécution (`ExecutionPlan`)

```cpp
struct PlannedOp {
    int       layer_index;  // Index dans la liste des layers
    FusionKind fusion;      // Fusion applicable (ou NONE)
};

struct ExecutionPlan {
    std::vector<PlannedOp>  ops;
    std::vector<uint8_t>    skip_layer;
    std::vector<uint8_t>    fuse_relu_for_conv2d;
    std::vector<int>        fuse_activation_consumer;
    std::vector<int>        fuse_unary_consumer;
    std::vector<int>        fuse_split_consumer;
    std::vector<uint8_t>    fuse_split_kind; // 0=none, 1=Split, 2=Chunk
    std::vector<int>        fuse_chain_next;  // edges producer->consumer pour fusions en chaîne
};

ExecutionPlan build_execution_plan_static(
    const std::vector<Layer>& layers,
    bool training);
```

`build_execution_plan_static(...)` est conservateur : en mode `training=true`, les fusions sont désactivées.

`skip_layer` marque les consommateurs fusionnés qui ne doivent plus être exécutés comme ops indépendantes dans la boucle de forward.

`fuse_chain_next` relie les couches enchaînées qui sont absorbées dans une même exécution logique.

---

## 4) Scratchpad pour Conv2d (`Conv2dScratchPlan`)

```cpp
struct Conv2dScratchPlan {
    size_t wT_bytes;   // Taille du buffer transpose de poids W^T
    size_t xcol_bytes; // Taille du buffer im2col
    size_t c_bytes;    // Taille du buffer de sortie
};

Conv2dScratchPlan plan_conv2d_fastpath_scratch(
    const std::vector<Layer>& layers);
```

Retourne la taille maximale nécessaire sur l'ensemble des layers Conv2d du modèle. Permet de pré-allouer un scratchpad unique réutilisé sur tous les Conv2d, évitant les allocations dynamiques pendant le forward.

---

## Exemple guidé

Supposons ce sous-graphe :

1. `conv0` (`Conv2d`)
2. `relu0` (`ReLU`)
3. `relu1` (`ReLU`)
4. `view0` (`View`)
5. `split0` (`Split`)

Ce que le planner peut produire en inférence :

1. `conv0 + relu0` fusionnés (`CONV2D_RELU`)
2. `relu1`, `view0`, `split0` absorbés via la chaîne de fusion générique
3. `skip_layer=1` pour les consommateurs déjà fusionnés

Résultat attendu : moins de passages intermédiaires, moins de copies, boucle de forward plus régulière.

---

## 5) Intégration avec le runtime

Le plan est géré via un cache `static_plan_` dans `Model`.

Pendant le forward :

- `MIMIR_ENABLE_PLANNER` (défaut `1`) active la planification statique.
- `MIMIR_ENABLE_FUSION` (défaut `1`) active l'application des fusions issues du plan.
- `MIMIR_ENABLE_FUSION_TRAIN` (défaut `0`) autorise uniquement `Conv2d+ReLU` en mode `training=true` (opt-in). Les fusions génériques restent inférence-only.
- Le plan est (re)construit si nécessaire s'il n'a jamais été construit, si le mode `training` change, ou si le nombre d'ops planifiées ne correspond plus au nombre de layers.
- `MIMIR_PLANNER_DUMP=1` déclenche un dump des stats planner (durées de vie, fusions, scratch Conv2d).

Résumé opérationnel au lancement d'un run :

- lancement process : pas de plan calculé,
- premier forward : build du plan si planner activé,
- forwards suivants : réutilisation du plan tant que les invariants ne changent pas.

Exemple prêt à copier/coller :

```bash
export MIMIR_ENABLE_PLANNER=1
export MIMIR_ENABLE_FUSION=1
export MIMIR_ENABLE_FUSION_TRAIN=1
export MIMIR_PLANNER_DUMP=1
```

Le forward utilise ensuite le plan pour :

- sauter les layers marqués `skip_layer`,
- appliquer les fusions au moment du traitement de l'output,
- garder un chemin d'exécution stable entre passes tant que les conditions ci-dessus ne changent pas.

En pratique, la reconstruction n'est pas liée à un appel `build()` explicite : elle est pilotée par les invariants vérifiés dans le forward.

---

## Déboguer efficacement le planner

Checklist terrain :

1. Activer `MIMIR_PLANNER_DUMP=1`.
2. Vérifier que le nombre d'ops planifiées correspond au graphe.
3. Vérifier que les `skip_layer` correspondent bien aux consommateurs fusionnés.
4. Refaire un run en `MIMIR_ENABLE_FUSION=0` pour comparer le comportement.
5. En cas d'écart training/inférence, vérifier `MIMIR_ENABLE_FUSION_TRAIN`.

Erreurs fréquentes :

1. Oublier qu'un changement de mode `training` invalide le cache plan.
2. Interpréter un `skip_layer` comme un layer perdu alors qu'il est exécuté dans la fusion.
3. Ajouter une nouvelle fusion sans mettre à jour la logique de consommation d'output.

## Étapes suivantes

- [Page précédente : Internals : GPU Runtimes — CUDA & ROCm (C++)](21-GPU-Runtimes.md)
- [Index de la documentation](../00-INDEX.md)
- [Revenir à la documentation](../00-INDEX.md)
