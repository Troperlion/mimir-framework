# Du registre au checkpoint

Choisir une architecture, construire le modèle par la Pipeline API, sauvegarder ses poids puis inspecter le checkpoint.

**Public concerné :** Débutant avancé ou utilisateur intermédiaire qui veut comprendre un workflow complet sans écrire immédiatement une boucle d’entraînement.

> **Prérequis**
>
> Le binaire `./bin/mimir` doit être compilé. Aucun dataset n’est nécessaire.

## Étape 1 — Vérifier le registre

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list basic_mlp --params --layers
```

Le registre fournit :

- le nom canonique ;
- la description ;
- la configuration par défaut ;
- la fonction C++ qui construit le modèle.

## Étape 2 — Construire et sauvegarder

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry \
  --arch basic_mlp \
  --no-train \
  --seed 1234 \
  --save /tmp/mimir_mlp_demo.safetensors
```

Le template exécute conceptuellement :

```lua
local P = dofile(ROOTWORK.."/scripts/modules/pipeline.lua")
local pipe = assert(P.FromRegistry("basic_mlp"))

assert(pipe:loadDefaultConfig("basic_mlp"))
assert(pipe:patchConfig({
  init = "xavier",
  seed = 1234,
}))

assert(pipe:build())
assert(pipe:save("/tmp/mimir_mlp_demo.safetensors"))
```

`pipe:build()` regroupe :

```text
Model.create
  → sélection dtype
  → construction du graphe
  → allocate_params
  → init_weights
```

Lors d’une reprise, il faut charger un checkpoint à la place d’une nouvelle initialisation. Ne réinitialise pas les poids après le load.

## Étape 3 — Inspecter le résultat

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua -- \
  --in /tmp/mimir_mlp_demo.safetensors \
  --graph-format mermaid
```

Vérifie :

- le type du modèle ;
- la configuration sauvegardée ;
- les layers ;
- les shapes des tensors ;
- le dtype ;
- le nombre de paramètres.

## Étape 4 — Comprendre les responsabilités

| Composant | Responsabilité |
| --- | --- |
| `ModelArchitectures` | choisir et construire l’architecture |
| `Pipeline` | orchestrer le cycle de vie |
| `Model` | posséder layers, poids, forward et backward |
| `RuntimeRouter` | choisir un backend pour chaque layer |
| `Serialization` | écrire/lire poids et métadonnées |
| script Lua | choisir config, chemins et workflow |

Le script Lua ne réimplémente pas les convolutions ou l’autograd. Il orchestre les services C++ exposés par l’API.

## Étape 5 — Modifier proprement la configuration

Le mode registry-first part du défaut puis applique les valeurs fournies :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry \
  --arch transformer \
  --d-model 128 \
  --layers 2 \
  --heads 4 \
  --seq-len 32 \
  --override dropout=0.1 \
  --no-train
```

Après chaque modification structurelle, inspecte de nouveau les layers. Deux configs ayant le même nom d’architecture peuvent être incompatibles si elles n’ont pas la même topologie ou les mêmes shapes.

## Étape 6 — Vérification automatisée

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
ctest --test-dir build --output-on-failure \
  -R 'RegistryTest|SerializationTest'
```

Le smoke Lua valide l’intégration. Les tests C++ isolent le registre et les formats de sérialisation.

## Pour continuer

- entraîner : [Guide d’entraînement](../02-User-Guide/04-Training.md) ;
- reprendre : [Checkpoints](../02-User-Guide/08-Checkpoints.md) ;
- écrire un script : [Tutoriel script](03-Tuto-Coder-Script.md) ;
- ajouter un modèle : [Tutoriel modèle](02-Tuto-Ajouter-Modele.md) ;
- comprendre le dispatch : [Backends hardware](../04-Architecture-Internals/03-Hardware-Backends.md).

## Étapes suivantes

- [Page précédente : Tutoriel : valider VAEConv sans dataset](07-Tuto-VAEConv-Sans-Dataset.md)
- [Index de la documentation](../00-INDEX.md)
- [Revenir à la documentation](../00-INDEX.md)
