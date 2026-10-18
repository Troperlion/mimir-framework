# MPK : packages d’architecture

Le format MPK (`Mimir Package Template`) transporte une configuration de
modèle, des métadonnées et, si nécessaire, la description complète d’un graphe.
Utilisez-le pour partager une variante d’architecture, prototyper un graphe ou
exporter une architecture sans confondre ce package avec un checkpoint de poids.

## Sur cette page

- [Pourquoi le format MPK existe](#pourquoi-le-format-mpk-existe)
- [Ce que contient un MPK](#ce-que-contient-un-mpk)
- [Le format moderne en pseudocode](#le-format-moderne-en-pseudocode)
- [Créer un MPK depuis le registre](#créer-un-mpk-depuis-le-registre)
- [Exporter l’architecture complète d’un modèle](#exporter-larchitecture-complète-dun-modèle)
- [Construire un MPK en Lua](#construire-un-mpk-en-lua)
- [Compiler le pseudocode en binaire](#compiler-le-pseudocode-en-binaire)
- [Prototyper un graphe nodal](#prototyper-un-graphe-nodal)
- [Vérifier et inspecter un MPK](#vérifier-et-inspecter-un-mpk)
- [Charger un MPK](#charger-un-mpk)
- [Découverte automatique avec `_archi`](#découverte-automatique-avec-_archi)
- [Choisir entre MPK, configuration et checkpoint](#choisir-entre-mpk-configuration-et-checkpoint)
- [Compatibilité et limites](#compatibilité-et-limites)

## Pourquoi le format MPK existe

Un nom du registre comme `vae_conv` désigne une fabrique C++ et sa configuration
par défaut. Il ne suffit pas toujours pour conserver une variante précise ou
échanger la structure observée d’un modèle. MPK ajoute une enveloppe portable
autour de ces informations.

Les usages principaux sont :

- **développement** : conserver une configuration et une description de graphe
  proches du code qui les a produites ;
- **prototypage** : assembler un graphe nodal, l’inspecter, puis tenter de
  l’appliquer à un modèle vide ;
- **export** : extraire une architecture du registre ou du modèle courant dans
  un artefact autonome ;
- **distribution locale** : déposer des variantes dans `_archi/` afin qu’elles
  apparaissent automatiquement dans le registre au prochain démarrage.

Un MPK ne contient pas les poids entraînés. Pour sauvegarder ou reprendre un
état numérique, utilisez un checkpoint RawFolder ou SafeTensors.

## Générer depuis un checkpoint ou le registre

Les deux commandes `build_mpk.lua` et `export_arch_mpk.lua` acceptent
`--checkpoint <chemin>`. Elles exportent les couches sérialisées, leurs connexions,
les paramètres de couches disponibles et `model_config`, sans charger les poids.

```bash
./bin/mimir --lua scripts/tools/build_mpk.lua -- --checkpoint checkpoints/run --out exports/run.mpk
./bin/mimir --lua scripts/tools/build_mpk.lua -- --checkpoint model.safetensors --out exports/model.mpk --compile
./bin/mimir --lua scripts/tools/build_mpk.lua -- --checkpoint debug.json --out exports/debug.mpk
./bin/mimir --lua scripts/tools/build_mpk.lua -- --register transformer --out exports/transformer.mpk
```

`--format auto` est implicite : `.safetensors`/`.st`, `.json`, sinon dossier.
Pour imposer le format, utilisez `rawfolder`, `safetensor` ou `debugjson` ;
`raw_folder`, `safetensors` et `debug_json` sont également acceptés.
RawFolder accepte `model/architecture.json` ou `architecture.json` à la racine.
SafeTensors exige le tenseur Mímir `model/architecture_json` : un fichier externe
contenant seulement des poids est refusé explicitement.

`--register <nom>` et `--arch <nom>` exportent le graphe réel du registre.
La commande historique `--type <nom> --from-registry` utilise le même export.
Sans source, `build_mpk.lua` conserve la création par template ou structure JSON.
Les options de description, auteur, date et modification restent utilisables.
Les formats MPK existants restent lisibles ; aucune migration des fichiers
présents n'est effectuée. `--out` désigne le fichier à écrire et remplace ce
fichier s'il existe déjà.

Un export ne peut conserver que les champs présents dans le checkpoint : les
anciens RawFolder/SafeTensors ne sérialisent pas toutes les options de couches.
Le MPK produit ne constitue donc pas une sauvegarde des poids ni une garantie
de reprise d'entraînement. Le DebugJSON est lu en mémoire, tandis que le lecteur
SafeTensors ne lit que l'en-tête et le tenseur d'architecture.

Validation ciblée :

```bash
./bin/mimir --lua scripts/tests/test_mpk_checkpoint_export.lua
```

## Ce que contient un MPK

Le package logique possède les sections suivantes :

| Section | Rôle |
| --- | --- |
| `format`, `format_short`, `version` | identification et version du package |
| `container` | source `pseudocode` ou représentation compilée `binary` |
| `header` | nom, type de base, auteur, date, options et checksum |
| `payload.base_config_b64` | configuration du modèle, JSON encodé en Base64 |
| `payload.description_b64` | description libre encodée en Base64 |
| `payload.model_structure` | template, blocs ou graphe nodal exporté |

Le checksum `fnv1a32` porte sur la représentation JSON canonique du `payload`.
`MPK.read` le vérifie avant de retourner le package. Le champ `header.size`
décrit la taille du conteneur écrit.

## Le format moderne en pseudocode

Le conteneur texte par défaut utilise une syntaxe inspirée du pseudocode Visu.
Voici un extrait schématique :

```text
# MPK - Mimir Package Template
# Syntaxe lisible inspirée du pseudocode Visu.

map mpk = []
mpk.set("container", "pseudocode")
mpk.set("format", "Mimir Package Template")
mpk.set("format_short", "MPK")

map mpk_header_1 = []
mpk_header_1.set("name", "vae_conv_experiment")
mpk_header_1.set("type", "vae_conv")
mpk.set("header", mpk_header_1)

map mpk_payload_2 = []
mpk_payload_2.set("base_config_b64", "...")
mpk_payload_2.set("description_b64", "...")
mpk.set("payload", mpk_payload_2)
```

Les constructions reconnues sont :

- `map nom = []` pour une table associative ;
- `list nom = []` ou `array nom = []` pour une séquence ;
- `nom.set("clé", valeur)` pour renseigner une map ;
- `nom.append(valeur)` pour ajouter un élément ;
- chaînes JSON, nombres et booléens comme valeurs scalaires ;
- lignes commençant par `#` comme commentaires.

Les variables intermédiaires générées, comme `mpk_header_1`, n’ont pas de
signification publique. Elles rendent seulement les structures imbriquées
référençables dans le pseudocode.

### Aide IDE VS Code

Le dépôt fournit une extension locale dans `ide/vscode-mpk`. Son stub
déclaratif `mpk-stub.json` décrit les mots-clés, méthodes, champs connus et
`LayerType`. L’extension apporte :

- l’association des sources `.mpk` au langage `mimir-mpk` ;
- la coloration syntaxique sans traiter le pseudocode comme du Lua ;
- la complétion et la documentation au survol ;
- des snippets, dont `mpk-package` et `mpk-node`.

Pour la tester, lancez
`code --extensionDevelopmentPath="$PWD/ide/vscode-mpk"`. Pour l’installer
durablement, générez un VSIX comme indiqué dans le README de l’extension.

Les fichiers `.mpk.bin` ne sont pas concernés : ce sont des conteneurs binaires
opaques.

> **Attention**
> Le format est lisible et modifiable, mais changer directement le payload rend
> son checksum invalide. Pour écrire ou modifier un MPK, passez normalement par
> `scripts/modules/mpk.lua`, qui recalcule le checksum et la taille.

## Créer un MPK depuis le registre

Pour créer un package léger à partir de la configuration par défaut d’une
architecture :

```bash
./bin/mimir --lua scripts/tools/build_mpk.lua -- \
  --name vae_conv_base \
  --type vae_conv \
  --author votre_nom \
  --from-registry \
  --template auto \
  --description "Configuration VAEConv de référence" \
  --compile \
  --out exports/vae_conv_base.mpk
```

La source est toujours écrite dans le pseudocode moderne. `--compile` produit
en plus `exports/vae_conv_base.mpk.bin` au format binaire v4 ; un chemin
explicite peut être passé après l’option.
`--config-json chemin.json` remplace `--from-registry` lorsque la configuration
provient déjà d’un fichier. `--structure-json` permet de fournir séparément une
structure de modèle.

Cet outil convient à une configuration partageable ou à un template. Pour
capturer les couches et routes effectivement construites, utilisez
`export_arch_mpk.lua`.

## Exporter l’architecture complète d’un modèle

### Depuis une architecture du registre

```bash
./bin/mimir --lua scripts/tools/export_arch_mpk.lua -- \
  --arch vae_conv \
  --name vae_conv_export \
  --description "Export complet de VAEConv" \
  --compile \
  --out exports/vae_conv_export.mpk
```

L’outil instancie l’architecture, récupère ses couches et produit
`model_structure.graph.nodes` ainsi que les liens de tensors.

Une configuration d’override peut être fournie :

```bash
./bin/mimir --lua scripts/tools/export_arch_mpk.lua -- \
  --arch vae_conv \
  --config-json configs/mon_vae.json \
  --name mon_vae \
  --out exports/mon_vae.mpk
```

### Depuis le modèle courant

L’option `--from-current-model` est destinée à un workflow Lua qui a déjà créé
un modèle dans le même processus, par exemple lorsqu’un script pilote
`export_arch_mpk.lua` avec `dofile`. Elle exporte alors les couches retournées
par `Mimir.Model.get_layers()`.

> **Attention**
> Lancer directement `export_arch_mpk.lua --from-current-model` dans un nouveau
> processus ne crée aucun modèle et échoue avec `aucun layer disponible à
> exporter`. Pour un lancement CLI autonome, préférez `--arch <nom>`.

## Construire un MPK en Lua

Pour générer un package dans un outil ou un script :

```lua
local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")

local pkg, build_err = MPK.build({
  name = "tiny_mlp",
  type = "basic_mlp",
  author = "dev",
  description = "Prototype MLP compact",
  modifiable = true,
  viz_specified = false,
  base_config = {
    input_dim = 8,
    hidden_dim = 16,
    output_dim = 2,
    hidden_layers = 1,
    dropout = 0.0,
  },
  model_structure = MPK.model_structure_template("basic_mlp"),
})
assert(pkg, build_err)

local ok, write_err = MPK.write("exports/tiny_mlp.mpk", pkg)
assert(ok, write_err)
```

`MPK.write` produit exclusivement le pseudocode moderne :

```lua
MPK.write(path, pkg)
```

Les options historiques `{ json = true }` et `{ binary = true }` sont refusées.
La production binaire est une étape distincte qui compile un fichier source
pseudocode déjà écrit et vérifiable.

Fonctions principales du module :

| Fonction | Contrat |
| --- | --- |
| `MPK.build(spec)` | construit le package logique et son checksum |
| `MPK.write(path, pkg, options)` | écrit un fichier `.mpk` |
| `MPK.read(path)` | détecte le conteneur, parse et vérifie l’intégrité |
| `MPK.verify_checksum(pkg)` | revérifie explicitement le payload |
| `MPK.decode_payload(pkg)` | retourne config, description et structure |
| `MPK.to_registry_full_config(pkg)` | produit la configuration complète attendue par le registre |
| `MPK.encode_pseudocode(pkg)` | sérialise une table MPK en pseudocode |
| `MPK.decode_pseudocode(text)` | parse le sous-langage MPK sans exécuter le texte comme Lua |
| `MPK.compile(source, output)` | compile une source pseudocode validée vers le binaire v4 |

## Compiler le pseudocode en binaire

Une source MPK moderne peut être compilée après sa création :

```bash
./bin/mimir --lua scripts/tools/compile_mpk.lua -- \
  --in exports/tiny_mlp.mpk \
  --out exports/tiny_mlp.mpk.bin
```

Le compilateur :

1. refuse une entrée JSON historique ou déjà binaire ;
2. parse le pseudocode sans l’exécuter comme du Lua ;
3. vérifie le checksum logique du payload ;
4. produit un header binaire fixe de 64 octets ;
5. sérialise directement les maps, listes, chaînes, nombres et booléens dans
   un payload typé `TYP4` ;
6. retire le Base64 interne de la configuration et de la description pendant
   le stockage afin de réduire la taille ;
7. relit le résultat pour vérifier qu’il reste chargeable.

Le binaire utilise la version de conteneur 4. Il peut être lu par `MPK.read`,
`load_mpk.lua`, `Mimir.Model.create(path)` et la découverte `_archi`, comme la
source textuelle :

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/tiny_mlp.mpk.bin \
  --verify-only
```

La compilation ne transforme pas l’architecture en code machine et n’ajoute
pas de poids. Elle produit une représentation binaire compacte et
auto-vérifiable du même package.

## Prototyper un graphe nodal

L’assistant interactif construit des nœuds, leurs entrées/sorties et leurs
paramètres :

```bash
./bin/mimir --lua scripts/tools/mpk_node_wizard.lua -- \
  --out exports/prototype_graph.mpk
```

Pour chaque couche, il demande notamment :

- le nom et le type ;
- les tensors d’entrée et de sortie ;
- les paramètres structurants ;
- une position destinée à la visualisation.

Le module `mpk_layers.lua` normalise et valide les types de couches avant
l’écriture. Il refuse aussi les identifiants dupliqués, les entrées invalides,
les paramètres qui ne sont pas une map et les `params_count` négatifs.

Les types disponibles peuvent être affichés sans lancer le questionnaire :

```bash
./bin/mimir --lua scripts/tools/mpk_node_wizard.lua -- --list-layer-types
```

Pour une architecture absente du registre, le chargeur peut créer un modèle
vide puis appliquer le graphe :

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/prototype_graph.mpk \
  --create \
  --apply-graph \
  --init xavier \
  --seed 42
```

Pendant la reconstruction, `load_mpk.lua` :

1. valide le graphe et normalise les alias de layers ;
2. crée le modèle via le registre ou utilise `create_empty` en repli ;
3. transmet à `push_layer` les paramètres propres à chaque nœud ;
4. configure les routes d’entrée/sortie ;
5. alloue et initialise les paramètres, sauf avec `--no-allocate`.

La reconstruction reste limitée aux types et paramètres exposés par
`Mimir.Model.push_layer`.

## Vérifier et inspecter un MPK

Vérifier le conteneur, sa structure et son checksum sans créer de modèle :

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/vae_conv_export.mpk \
  --verify-only
```

Afficher également la configuration décodée :

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/vae_conv_export.mpk \
  --show-config \
  --no-create
```

Une vérification réussie confirme l’intégrité du package. Elle ne prouve pas
que l’architecture de base existe dans le registre ni que chaque nœud d’un
graphe custom est exécutable.

## Charger un MPK

### Directement comme chemin

```lua
local ok, err = Mimir.Model.create("exports/vae_conv_export.mpk")
assert(ok, err)
```

Le chemin `.mpk` suit ce flux :

```text
MPK.read
  → vérification du checksum
  → MPK.to_registry_full_config
  → ModelArchitectures::createFromConfig
  → fabrique de l’architecture indiquée par header.type
```

La configuration du payload est fusionnée avec la configuration par défaut de
l’architecture native. La création n’alloue et n’initialise pas
automatiquement les paramètres :

```lua
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier"))
```

### Avec l’outil de chargement

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/vae_conv_export.mpk \
  --create
```

L’outil affiche les métadonnées, vérifie le checksum, puis appelle le chemin
direct `Mimir.Model.create(fichier_mpk)`.

## Découverte automatique avec `_archi`

Au démarrage, Mímir cherche un dossier `_archi` directement à la racine du
projet courant. Tous les fichiers réguliers finissant par `.mpk` ou
`.mpk.bin`, sans distinction de casse, sont examinés dans l’ordre alphabétique.

```text
racine-du-projet/
├── _archi/
│   ├── tiny_mlp.mpk
│   └── vae_conv_pseudocode.mpk
├── scripts/
└── bin/
```

Chaque package valide devient une entrée de
`Mimir.Architectures.available()`. Il peut ensuite être utilisé comme une
architecture native :

```lua
local cfg, cfg_err =
  Mimir.Architectures.default_config("vae_conv_pseudocode")
assert(cfg, cfg_err)

local ok, create_err =
  Mimir.Model.create("vae_conv_pseudocode", cfg)
assert(ok, create_err)
```

Règles de nommage :

1. `header.name` devient le nom d’entrée préféré ;
2. si ce nom existe déjà, le nom du fichier sans `.mpk` devient l’alias ;
3. si cet alias existe également, le MPK est ignoré ;
4. une architecture native n’est jamais remplacée silencieusement.

Le type déclaré dans `header.type` doit déjà exister dans le registre. Une
entrée autochargée réutilise la fabrique native de ce type avec la
`base_config` du MPK.

Le démarrage indique le nombre de packages chargés :

```text
[startup] mpk_architectures_loaded=2 from=/projet/_archi
[startup] registry_architectures=28
```

Un fichier invalide, un checksum incorrect ou un type de base inconnu produit
un avertissement `[startup] MPK ignoré: ...` sans interrompre le démarrage.

> **Note**
> La racine utilisée est le répertoire courant au lancement de `mimir`.
> Exécutez les commandes depuis la racine du dépôt pour que `_archi/` et
> `scripts/modules/mpk.lua` soient résolus correctement.

### Prototypes vision fournis

Le dépôt fournit quatre architectures MPK de prototypage :

| Nom du registre | Fichier | Délégation exécutable | Graphe documenté |
| --- | --- | --- | --- |
| `r_cnn` | `_archi/r_cnn.mpk` | `vgg16` | propositions, ROI, classification et régression de boîtes |
| `yolo` | `_archi/yolo.mpk` | `mobilenet` | neck et têtes de détection multi-échelles |
| `ssd` | `_archi/ssd.mpk` | `vgg16` | cartes multi-résolutions et têtes MultiBox |
| `deeplab` | `_archi/deeplab.mpk` | `resnet` | ASPP et décodeur de segmentation |

Ils peuvent être régénérés avec :

```bash
./bin/mimir --lua scripts/tools/add_vision_mpk_architectures.lua
```

Et validés avec :

```bash
./bin/mimir --lua scripts/tests/test_mpk_vision_architectures.lua
```

> **Limitation**
> Ces packages rendent les noms disponibles dans le registre et construisent
> un backbone natif exécutable. Leur graphe MPK décrit la cible architecturale,
> mais il n’est pas appliqué pendant la délégation au backbone. Les propositions
> de régions, ROI, ancres/grilles, décodage de boîtes et kernels atrous
> spécialisés restent externes. Leurs graphes utilisent désormais le layer
> runtime `NMS` après décodage. Il ne s’agit donc pas encore de pipelines de
> détection ou segmentation prêts pour la production.

## Choisir entre MPK, configuration et checkpoint

| Besoin | Artefact recommandé |
| --- | --- |
| Modifier des options de lancement | JSON `--conf` ou table Lua |
| Nommer et partager une variante d’architecture | MPK |
| Inspecter ou échanger une structure de graphe | MPK avec `model_structure` |
| Ajouter localement une variante au registre | MPK dans `_archi/` |
| Sauvegarder les poids entraînés | SafeTensors ou RawFolder |
| Reprendre optimiseur, époque et métriques | checkpoint du workflow d’entraînement |

MPK et checkpoint sont complémentaires : le premier décrit principalement
l’architecture et sa configuration, le second conserve l’état numérique.

## Compatibilité et limites

Le format moderne possède une source et une représentation compilée :

| Représentation | Production | Lecture | Usage |
| --- | --- | --- | --- |
| pseudocode `.mpk` | `MPK.write` et outils d’export | oui | développement, revue, export lisible |
| binaire v4 `.mpk.bin` | `MPK.compile` ou `compile_mpk.lua` | oui | distribution opaque et compacte |

La lecture accepte encore les anciens JSON MPK et binaires v1/v2/v3 afin de
permettre leur migration. Aucun outil moderne ne les écrit.

Limites actuelles :

- aucun poids n’est inclus ;
- le checksum FNV-1a détecte une corruption accidentelle, mais ne constitue pas
  une signature cryptographique ;
- le binaire v4 n’embarque ni le pseudocode, ni ses commentaires ou sa mise en
  forme ; il ne s’agit toutefois pas de code natif ni d’un format chiffré ;
- l’autochargement ne parcourt pas les sous-dossiers de `_archi/` ;
- un MPK autochargé doit référencer une architecture de base déjà enregistrée ;
- la présence d’un graphe exporté ne garantit pas sa reconstruction dynamique.

## Étapes suivantes

- [Registre des architectures](../03-API-Reference/11-Architectures.md)
- [Cycle de vie d’un modèle](02-Model-Lifecycle.md)
- [Checkpoints](08-Checkpoints.md)
- [Scripts et outils Lua](10-Examples.md)
- [VAEConv](14-VAEConv.md)
