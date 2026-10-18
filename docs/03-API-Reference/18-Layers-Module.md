# `Mimir.Layers` — inspection en lecture seule

`Mimir.Layers` retourne des listes des couches du modèle courant. Ces appels ne
calculent pas de tenseurs et n'allouent pas les poids du modèle.

```lua
Mimir.Model.create("basic_mlp")
for _, layer in ipairs(Mimir.Layers.linear()) do
  print(layer.index, layer.name, layer.type, layer.param_count)
end

-- Chaque nom canonique possède également son alias entièrement minuscule.
local convs = Mimir.Layers.Conv2d() -- identique à Mimir.Layers.conv2d()
local attention = Mimir.Layers.crossattention()
```

| Appel | Retour |
|---|---|
| `Mimir.Layers.available()` | Les 73 noms canoniques du registre |
| `Mimir.Layers.<Type>()` / `Mimir.Layers.<type_minuscule>()` | Toutes les couches du type exact, dans l'ordre du graphe |
| `Mimir.Layers.by_type("GroupNorm")` | Le même filtre, avec un nom passé en argument |
| `Mimir.Layers.activation()` | Le groupe des activations |
| `Mimir.Layers.batchnorm()` | Le groupe BatchNorm2d / InstanceNorm2d |
| `Mimir.Layers.attention()` | Le groupe des attentions |

Les listes sont des snapshots détachés du modèle. Les affectations ordinaires
aux listes, aux fiches et aux tableaux `inputs` lèvent une erreur. `#`, `ipairs`
et `pairs` permettent leur lecture. Sans modèle ou sans couche correspondante,
le résultat est une liste vide.

Chaque fiche contient `index` (base 1), `name`, `type`, `param_count`, `output`,
`inputs` et les dimensions de convolution, projection, embedding ou attention.
`param_count` décrit les paramètres du graphe même avant leur allocation.

## Ops disponibles par runtime

`Mimir.Model.runtime_ops()` retourne une liste de backends CPU, CUDA, ROCM,
VULKAN et OPENCL. Chaque entrée contient `name`, `compiled`, `scope` et `ops`.
Chaque op contient `type`, `forward` et `backward`.

Les niveaux sont `NativeOptimized`, `Native`, `HostFallback` (repli CPU) et
`Unsupported`. Un backend absent du build indique `compiled=false` et des
ops `Unsupported`. La requête décrit les capacités déclarées par type sans
initialiser les devices ; les formes de tenseurs, la configuration et le
matériel peuvent ensuite restreindre l'exécution effective.

```sh
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --ops --runtime
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --all
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --all --list vae_conv
```

Pour inspecter un fichier externe, `--extern-model` sélectionne
`external_safetensors_base` et transmet le chemin à sa configuration native :

```sh
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --extern-model model.safetensors
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --extern-model "chemin avec espaces/model.safetensors" --all
```

Sans autre sélection de rapport, les paramètres, layers et statistiques du
fichier sont affichés. `--layers`, `--stats`, `--ops`, `--json` et `--export`
restent utilisables. `--list` peut désigner `external_safetensors_base`, mais une
autre architecture est incompatible avec `--extern-model`.

L'inspection lit l'en-tête SafeTensors sans allouer les poids. Le modèle externe
représente les clés de tenseurs par des couches `Identity` ; il ne reconstruit pas
un graphe exécutable à partir des seuls poids. En cas d'export explicite, les
poids du fichier source sont chargés avant la sauvegarde.

`--all` active les rapports, y compris les vues détaillée et compacte des ops.
Sans `--list`, les paramètres, layers et statistiques de toutes les architectures
sont affichés. Aucun export n'est déclenché implicitement ; `--json` et
`--export <path>` restent explicites.

- [Index de la documentation](../00-INDEX.md)
- [Globals & aliases](19-Globals.md)
