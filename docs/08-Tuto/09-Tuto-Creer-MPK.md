# Créer un fichier MPK à la main ou avec les outils

Ce tutoriel construit un package d'architecture MPK, le vérifie, le compile et
le charge. Il présente deux méthodes : l'utilitaire `build_mpk.lua` et une
spécification Lua écrite à la main.

**Public concerné :** utilisateurs qui veulent partager une configuration,
prototyper un graphe ou ajouter une architecture locale dans `_archi/`.

> **Prérequis**
>
> Le binaire `./bin/mimir` doit être compilé. Aucun dataset n'est nécessaire.

## 1. Comprendre les deux fichiers

Un package peut avoir deux représentations :

| Fichier | Contenu | Usage |
| --- | --- | --- |
| `modele.mpk` | pseudocode lisible et révisable | développement et revue |
| `modele.mpk.bin` | représentation binaire v4 opaque | distribution et chargement compact |

Le MPK décrit une configuration et éventuellement un graphe. Il ne contient pas
les poids entraînés. Les poids restent dans un checkpoint RawFolder ou
SafeTensors.

Le package logique contient :

- `header` : nom, type de base, auteur, options et checksum ;
- `base_config` : valeurs qui configurent l'architecture ;
- `model_structure` : graphe statique ou programme de construction ;
- `description` : texte libre.

## 2. Méthode rapide avec `build_mpk.lua`

Le gabarit `configurable_stack` contient une variable, une boucle et une
condition. Avec ses valeurs par défaut, il crée deux couches linéaires suivies
d'une activation :

```bash
mkdir -p exports

./bin/mimir --lua scripts/tools/build_mpk.lua -- \
  --name configurable_stack \
  --type custom_graph \
  --author "$USER" \
  --template configurable_stack \
  --description "Empilement piloté par configuration" \
  --compile \
  --out exports/configurable_stack.mpk
```

La commande écrit :

```text
exports/configurable_stack.mpk
exports/configurable_stack.mpk.bin
```

Pour utiliser votre propre configuration, créez par exemple
`configs/my_stack.json` :

```json
{
  "hidden_dim": 128,
  "depth": 4,
  "use_output": false
}
```

Puis reconstruisez le package :

```bash
./bin/mimir --lua scripts/tools/build_mpk.lua -- \
  --name my_stack \
  --type custom_graph \
  --config-json configs/my_stack.json \
  --template configurable_stack \
  --compile \
  --out exports/my_stack.mpk
```

Ici, `depth=4` produit quatre nœuds linéaires. Comme `use_output=false`, le
nœud d'activation optionnel n'est pas ajouté.

### Autres sources proposées par l'utilitaire

- `--from-registry` récupère la configuration par défaut d'un modèle enregistré ;
- `--config-json` charge une configuration JSON ;
- `--structure-json` charge une structure préparée séparément ;
- `--template auto` choisit le gabarit associé au type ;
- `--compile` produit aussi le binaire v4.

Affichez toutes les options avec :

```bash
./bin/mimir --lua scripts/tools/build_mpk.lua -- --help
```

## 3. Méthode manuelle recommandée

Écrire un MPK à la main ne signifie pas calculer soi-même le Base64, la taille
et le checksum. Écrivez plutôt la spécification dans un petit script Lua, puis
laissez `MPK.build` et `MPK.write` produire le pseudocode valide.

Créez `scripts/local/create_my_stack_mpk.lua` avec le contenu suivant :

```lua
---@diagnostic disable: undefined-global

local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")

local config = {
  hidden_dim = 32,
  depth = 3,
  use_output = true,
}

local function linear_node(input_name)
  return {
    op = "node",
    value = {
      name = "block_${i}",
      type = "Linear",
      inputs = { input_name },
      output = "block_${i}_out",
      params_count = {
        op = "add",
        left = { op = "mul", left = "$width", right = "$width" },
        right = "$width",
      },
      params = {
        in_features = "$width",
        out_features = "$width",
      },
    },
  }
end

local structure = {
  template = "my_configurable_stack",
  architecture = "custom_graph",
  version = 1,
  graph = {
    mode = "node",
    nodes = {},
    links = {},
  },
  control = {
    variables = {
      width = "$config.hidden_dim",
    },
    steps = {
      {
        op = "for",
        variable = "i",
        from = 1,
        to = "$config.depth",
        body = {
          {
            op = "set",
            name = "previous",
            value = { op = "sub", left = "$i", right = 1 },
          },
          {
            op = "if",
            condition = { op = "eq", left = "$i", right = 1 },
            then_steps = { linear_node("x") },
            else_steps = { linear_node("block_${previous}_out") },
          },
        },
      },
      {
        op = "if",
        condition = {
          op = "eq",
          left = "$config.use_output",
          right = true,
        },
        then_steps = {
          {
            op = "node",
            value = {
              name = "output_activation",
              type = "ReLU",
              inputs = { "block_${config.depth}_out" },
              output = "x",
              params_count = 0,
              params = {},
            },
          },
        },
      },
    },
  },
  build = {
    dynamic_layer_assembly = true,
    generated_by = "scripts/local/create_my_stack_mpk.lua",
  },
}

local expanded, expand_err = MPK.expand_model_structure(structure, config)
assert(expanded, expand_err)
assert(#expanded.graph.nodes == 4)

local pkg, build_err = MPK.build({
  name = "my_configurable_stack",
  type = "custom_graph",
  author = os.getenv("USER") or "unknown",
  description = "Graphe linéaire écrit à la main",
  modifiable = true,
  base_config = config,
  model_structure = structure,
})
assert(pkg, build_err)

local ok, write_err = MPK.write("exports/my_configurable_stack.mpk", pkg)
assert(ok, write_err)

local compiled, compile_err = MPK.compile(
  "exports/my_configurable_stack.mpk",
  "exports/my_configurable_stack.mpk.bin"
)
assert(compiled, compile_err)

print("MPK source et binaire créés")
```

Exécutez le générateur :

```bash
mkdir -p scripts/local exports
./bin/mimir --lua scripts/local/create_my_stack_mpk.lua
```

Pour modifier l'architecture, changez uniquement `config` ou les étapes de
`structure.control`, puis relancez le script.

## 4. Variables, conditions et boucles

La section `model_structure.control` est interprétée lors du décodage du
payload. Elle travaille sur une copie de la structure, puis ajoute les éléments
générés dans `graph.nodes` et `graph.links`.

### Références et interpolation

| Forme | Signification |
| --- | --- |
| `$config.depth` | valeur exacte de `base_config.depth` |
| `$width` | valeur exacte de la variable `width` |
| `${i}` | interpolation dans une chaîne |
| `block_${i}_out` | nom calculé, par exemple `block_2_out` |

### Étapes de contrôle

| `op` | Rôle |
| --- | --- |
| `set` | crée ou remplace une variable |
| `if` | exécute `then_steps` ou `else_steps` |
| `for` | répète `body` entre `from` et `to`, avec `step` optionnel |
| `node` | ajoute `value` à `graph.nodes` |
| `link` | ajoute `value` à `graph.links` |

### Expressions

Les expressions sont des tables structurées, jamais du code Lua arbitraire :

```lua
{ op = "mul", left = "$config.hidden_dim", right = 4 }
{ op = "ge", left = "$config.depth", right = 2 }
{ op = "and", left = "$config.enabled", right = "$config.use_output" }
{ op = "not", value = "$config.disabled" }
```

Opérations disponibles :

```text
add sub mul div mod
eq ne gt ge lt le
and or not
```

L'expansion est limitée à 10 000 itérations et 100 000 nœuds ou liens. Ces
limites empêchent un fichier mal configuré de construire une structure sans
borne.

## 5. Lire le pseudocode généré

Le fichier `.mpk` contient uniquement les constructions suivantes :

```text
map nom = []
list nom = []
nom.set("cle", valeur)
nom.append(valeur)
```

Les variables telles que `mpk_payload_3_model_structure_4` sont des identifiants
générés pour représenter les tables imbriquées. La boucle et la condition sont
elles-mêmes sérialisées sous forme de maps avec un champ `op`. Elles ne sont pas
exécutées comme du Lua par le parseur MPK.

Le `base_config` et la description sont stockés en Base64 dans le payload. Le
checksum `fnv1a32` protège le payload complet.

> **Important**
>
> Modifier directement une ligne `.set()` ou `.append()` invalide le checksum.
> Pour une modification manuelle sûre, modifiez le script générateur puis
> régénérez le fichier avec `MPK.write`.

## 6. Vérifier, inspecter et compiler

Vérifiez la source sans créer de modèle :

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/my_configurable_stack.mpk \
  --verify-only
```

Affichez la configuration décodée :

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/my_configurable_stack.mpk \
  --show-config \
  --no-create
```

Compilez séparément si `--compile` n'a pas été utilisé :

```bash
./bin/mimir --lua scripts/tools/compile_mpk.lua -- \
  --in exports/my_configurable_stack.mpk \
  --out exports/my_configurable_stack.mpk.bin
```

Le fichier source conserve `model_structure.control`. Le binaire contient le
graphe déjà développé. Une lecture du binaire ne répète donc pas la boucle.

## 7. Créer et exécuter le graphe

Pour une architecture `custom_graph`, demandez au chargeur de créer un modèle
vide et d'appliquer les nœuds :

```bash
./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in exports/my_configurable_stack.mpk.bin \
  --create \
  --apply-graph \
  --replace-layers \
  --allocate \
  --init xavier \
  --seed 42
```

Le chargeur :

1. décode la configuration ;
2. développe `model_structure.control` pour une source textuelle ;
3. normalise les types de couches ;
4. crée un modèle vide si le type n'existe pas dans le registre ;
5. ajoute les couches avec `Mimir.Model.push_layer` ;
6. configure leurs entrées et sorties ;
7. alloue et initialise les paramètres.

Tous les types disponibles sont affichés avec :

```bash
./bin/mimir --lua scripts/tools/mpk_node_wizard.lua -- --list-layer-types
```

## 8. Utiliser les autres utilitaires

### Assistant interactif

Pour construire un graphe nœud par nœud sans écrire de table Lua :

```bash
./bin/mimir --lua scripts/tools/mpk_node_wizard.lua --
```

L'assistant demande le type, le nom, les paramètres et les routes de chaque
couche. Il convient aux graphes statiques. Pour les boucles et conditions,
utilisez un script générateur comme dans l'étape 3.

### Export depuis le registre

Pour capturer les couches réellement créées par une architecture existante :

```bash
./bin/mimir --lua scripts/tools/export_arch_mpk.lua -- \
  --arch lumen_diffusion \
  --name lumen_export \
  --compile \
  --out exports/lumen_export.mpk
```

L'export produit un graphe statique correspondant à la configuration fournie.
Il ne recrée pas automatiquement une boucle à partir de couches répétées.

## 9. Installer le package dans le registre local

Copiez la source ou le binaire validé dans `_archi/` :

```bash
cp exports/my_configurable_stack.mpk.bin _archi/
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

Au démarrage, Mímir inspecte les `.mpk` et `.mpk.bin` de `_archi/`. Si le nom du
package entre en collision avec une architecture existante, le nom du fichier
sert d'alias lorsque cela est possible.

## 10. Erreurs fréquentes

| Erreur | Cause probable | Correction |
| --- | --- | --- |
| `checksum mismatch` | pseudocode modifié directement | régénérer avec `MPK.write` |
| `graph.nodes vide` | aucune étape `node`, boucle à zéro itération | vérifier `depth`, `from` et `to` |
| `unsupported MPK control step` | valeur `op` inconnue | utiliser `set`, `if`, `for`, `node` ou `link` |
| `unsupported MPK control expression` | opérateur d'expression inconnu | utiliser la liste de l'étape 4 |
| `iteration limit exceeded` | boucle trop grande | réduire la borne configurée |
| `invalid layer type` | type non pris en charge par `push_layer` | consulter `--list-layer-types` |
| `unknown architecture: custom_graph` | type absent du registre | charger avec le repli non-registry et `--apply-graph` |

## Pour aller plus loin

- [Référence complète du format MPK](../02-User-Guide/15-MPK.md)
- [API des architectures](../03-API-Reference/11-Architectures.md)
- [Ajouter un modèle au framework](02-Tuto-Ajouter-Modele.md)
