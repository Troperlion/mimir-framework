# Construire un modèle et ses layers

Implémenter ou modifier des briques techniques sans casser le contrat global.

**Public concerné :** Développeur framework (C/C++/runtime/scripting).

> **Prérequis**
>
> Comprendre le registre d'architectures et les conventions I/O.

Ce chapitre explique concrètement comment assembler un modèle, configurer ses layers et fiabiliser les routes I/O.

## Sur cette page

- [Lecture guidée](#lecture-guidée)
- [1. API de base cote C++](#1-api-de-base-cote-c)
- [2. Exemple minimal C++](#2-exemple-minimal-c)
- [3. Parametrer les layers](#3-parametrer-les-layers)
- [4. Regle source de verite: modele declare en C/C++](#4-regle-source-de-verite-modele-declare-en-cc)
- [5. Cote Lua: creation/chargement, pas definition principale du graphe](#5-cote-lua-creationchargement-pas-definition-principale-du-graphe)
- [6. API legacy a ne plus utiliser: setlayerio](#6-api-legacy-a-ne-plus-utiliser-setlayerio)
- [7. Erreurs frequentes](#7-erreurs-frequentes)
- [8. Checklist avant commit](#8-checklist-avant-commit)
- [9. Demo correcte - meme intention cote C++ (source de verite)](#9-demo-correcte---meme-intention-cote-c-source-de-verite)
- [10. Demo script correcte (registre -> create)](#10-demo-script-correcte-registre---create)
- [11. Criteres metier de validation](#11-criteres-metier-de-validation)
- [12. Demos existantes a relire](#12-demos-existantes-a-relire)
- [Étapes suivantes](#étapes-suivantes)

## Lecture guidée

Parcours conseillé :

1. Comprendre le rôle de `model.push(...)`.
2. Construire une topologie minimale claire.
3. Renseigner explicitement `inputs` et `output`.
4. Valider par un smoke test.

Principe clé : la source de vérité de la topologie est côté C/C++, pas côté script.

## 1. API de base cote C++

La methode cle est :

```cpp
void Model::push(const std::string &name, const std::string &type, size_t params_count);
```

Effet : ajoute un `Layer` dans `Model::layers`.

Ce que fait `push` en pratique :

- normalise le type de layer,
- valide que le type est supporte,
- applique des champs de `modelConfig` sur le layer (in/out channels, kernel, stride, padding, etc.),
- calcule certaines dimensions de sortie (ex: Conv2d / ConvTranspose2d),
- ajoute le layer a la topologie.

Lecture pratique : `push` crée le noeud; le wiring des flux de tenseurs (`inputs/output`) donne la sémantique du graphe.

## 2. Exemple minimal C++

```cpp
model.push("enc/conv_in", "Conv2d", 64 * 3 * 3 * 3);
model.push("enc/act0", "SiLU", 0);
model.push("enc/res/add", "Add", 0);
```

Regle generale :

- `params_count > 0` pour les layers parametres,
- `params_count = 0` pour les layers purement operationnels (activation, add, reshape, etc.).

Bon réflexe : nommer les layers avec une hiérarchie stable (`enc/...`, `mid/...`, `dec/...`) pour faciliter debug et sérialisation.

## 3. Parametrer les layers

Le parametrage passe principalement par la config modele, puis est lu dans `push`.

Exemples de champs utilises par les convs :

- `in_channels`
- `out_channels`
- `height` / `width`
- `kernel`
- `stride`
- `padding`

Conseil :

- garder les noms de champs stables,
- documenter les defaults dans la config d'architecture,
- eviter les dependances implicites entre layers distants.

Conseil de pédagogie de code : déclarer les champs dimensionnels au plus près de la création du layer pour réduire les ambiguïtés.

## 4. Regle source de verite: modele declare en C/C++

Dans ce framework, la declaration de topologie et le parametrage des layers se font cote C/C++ (classes de modeles + registre), pas cote Lua.

Flux normal :

1. Le registre cree le modele (`ModelArchitectures::create(...)`).
2. La classe C++ (ex: `VAEConvModel::buildInto`) pousse les layers (`model.push(...)`).
3. La classe C++ renseigne les I/O (`Layer.inputs`, `Layer.output`) et metadonnees (channels, kernel, stride, etc.).

Exemple reel a lire: `src/Models/Vision/VAEConvModel.cpp`.

Pourquoi ce choix est important :

1. Le graphe reste traçable dans un seul endroit.
2. Les scripts Lua restent simples (configuration et exécution).
3. Le risque de divergence train/inférence baisse.

## 5. Cote Lua: creation/chargement, pas definition principale du graphe

Cote script, le chemin standard est :

```lua
local cfg = Mimir.Architectures.default_config("vae_conv")
cfg.image_w = 512
cfg.image_h = 512
cfg.base_channels = 16

local ok, err = Mimir.Model.create("vae_conv", cfg)
if not ok then error(err) end
```

Note : `Mimir.Model.build()` est conserve pour compatibilite, mais en mode moderne la construction est deja faite par `Model.create(...)` via le registre.

Résumé mental : Lua pilote le run, C++ décrit la structure.

## 6. API legacy a ne plus utiliser: set_layer_io

`Mimir.Model.set_layer_io(...)` est deprecie/obsolete et ne doit plus etre utilise dans les nouveaux scripts ni dans la documentation d'usage.

Regle projet :

- declaration des layers: C/C++,
- wiring I/O (`Layer.inputs`, `Layer.output`): C/C++,
- scripts Lua: creation via registre + execution/inspection uniquement.

Pour une architecture produit (ex: `vae_conv`, `diffusion`, `hf_vae_decoder`), modifier la topologie dans les fichiers C/C++ du modele et son enregistrement registre.

## 7. Erreurs frequentes

1. Type de layer invalide dans `push`.
2. `params_count` incoherent avec la structure reelle du layer.
3. Inputs non renseignes ou mal orthographies.
4. Sortie ecrasee accidentellement (meme nom `output` sur plusieurs branches).
5. Dimensions incompatibles entre layers relies.

Signaux d'alerte précoces :

1. Un `output` réutilisé sans intention explicite.
2. Des `inputs` vides sur un layer multi-entrée.
3. Un `params_count` calculé à la main sans formule documentée.

## 8. Checklist avant commit

- topologie lisible (`name` explicites),
- I/O explicites pour les layers multi-entrees,
- shape checks passes,
- allocate/init/forward passent,
- test smoke ajoute (script court).

Ajouter si possible :

1. un test qui couvre le chemin nominal,
2. un test qui couvre au moins un cas de forme invalide.

## 9. Demo correcte - meme intention cote C++ (source de verite)

```cpp
model.push("blk/conv_main", "Conv2d", 64 * 64 * 3 * 3);
model.push("blk/act_main", "SiLU", 0);
model.push("blk/conv_skip", "Conv2d", 64 * 64 * 1 * 1);
model.push("blk/add", "Add", 0);
```

Puis associer les I/O directement en C++ (exemple style `VAEConvModel.cpp`) :

```cpp
if (auto* l = model.getLayerByName("blk/conv_main")) {
    l->inputs = {"__input__"};
    l->output = "blk/main0";
    l->in_channels = 64;
    l->out_channels = 64;
    l->kernel_size = 3;
    l->stride = 1;
    l->padding = 1;
}
if (auto* l = model.getLayerByName("blk/act_main")) {
    l->inputs = {"blk/main0"};
    l->output = "blk/main1";
}
if (auto* l = model.getLayerByName("blk/conv_skip")) {
    l->inputs = {"__input__"};
    l->output = "blk/skip";
    l->in_channels = 64;
    l->out_channels = 64;
    l->kernel_size = 1;
    l->stride = 1;
    l->padding = 0;
}
if (auto* l = model.getLayerByName("blk/add")) {
    l->inputs = {"blk/main1", "blk/skip"};
    l->output = "x";
}
```

## 10. Demo script correcte (registre -> create)

```lua
local cfg = Mimir.Architectures.default_config("vae_conv")
cfg.image_w = 512
cfg.image_h = 512
cfg.base_channels = 16

assert(Mimir.Model.create("vae_conv", cfg))
```

Ce qui est important : la topologie vient de C/C++ (classe modele), le script fournit la config et declenche la creation.

## 11. Criteres metier de validation

Definition de done pour ce type d'assemblage :

1. Les noms d'IO restent stables entre entrainement et inference.
2. Le bloc est serialisable et rechargeable sans renommer les noeuds.
3. Les outils d'inspection affichent le graphe attendu.
4. Le bloc accepte une batch de smoke test sans NaN/shape mismatch.

Critère pratique complémentaire : le modèle doit être lisible par quelqu'un qui ne connaît pas votre contexte initial.

## 12. Demos existantes a relire

- `scripts/templates/template_new_model.lua`
- `scripts/tests/test_vae_conv_resnet_smoke.lua`
- `scripts/tools/inspect_architectures.lua`

## Étapes suivantes

- [Page précédente : Comment Fonctionne Le Framework](01-How-The-Framework-Works.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Config Et Registre D'Architectures](03-Config-And-Registry.md)
