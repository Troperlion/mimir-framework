# Parcours par niveau de compétence

Proposer plusieurs parcours progressifs, avec des commandes vérifiables et un résultat concret à chaque étape.

**Public concerné :** Toute personne qui découvre Mímir ou qui cherche le prochain chapitre adapté à son niveau.

> **Prérequis**
>
> Être placé à la racine du dépôt. Pour les parcours utilisateur, le binaire `./bin/mimir` doit être compilé.

## Sur cette page

- [Comment choisir son niveau](#comment-choisir-son-niveau)
- [Parcours A — Découvrir sans dataset](#parcours-a-découvrir-sans-dataset)
- [Parcours B — Utiliser Lua et les outils](#parcours-b-utiliser-lua-et-les-outils)
- [Parcours C — Entraîner et reprendre](#parcours-c-entraîner-et-reprendre)
- [Parcours D — Ajouter ou modifier un modèle](#parcours-d-ajouter-ou-modifier-un-modèle)
- [Parcours E — Modifier un runtime](#parcours-e-modifier-un-runtime)
- [Où chercher la vérité](#où-chercher-la-vérité)
- [Étapes suivantes](#étapes-suivantes)

## Comment choisir son niveau

| Niveau | Vous savez déjà… | Commence par… |
| --- | --- | --- |
| Débutant | utiliser un terminal, mais pas Mímir | Parcours A |
| Utilisateur | modifier une config et lire un log | Parcours B |
| Utilisateur avancé | entraîner/reprendre un modèle | Parcours C |
| Développeur modèle | lire du C++ et raisonner sur les shapes | Parcours D |
| Développeur runtime | écrire des kernels et des backward | Parcours E |

## Parcours A — Découvrir sans dataset

### 1. Vérifier le binaire

```bash
./bin/mimir --help
```

Vous devez voir la version, les modes `--lua` et `--conf`, puis les capacités matérielles.

### 2. Lister le registre

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

Le registre est la liste canonique des architectures que `Mimir.Model.create()` peut construire.

### 3. Construire un petit modèle

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch basic_mlp \
  --no-train
```

Ce chemin effectue `create → allocate_params → init_weights`, sans données.

### 4. Inspecter un VAE convolutionnel

```bash
./bin/mimir --lua scripts/examples/inspect_vae_conv.lua
```

Vous pouvez passer au parcours B lorsque vous savez reconnaître :

- une architecture ;
- sa config ;
- ses layers ;
- son nombre de paramètres.

## Parcours B — Utiliser Lua et les outils

Lis d’abord :

1. [Scripting Lua](../02-User-Guide/06-Lua-Scripting.md) ;
2. [Cycle de vie d’un modèle](../02-User-Guide/02-Model-Lifecycle.md) ;
3. [Scripts et outils Lua](../02-User-Guide/10-Examples.md).

Puis exécute :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_conv --params --layers --stats

./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- \
  --safe --iters 1
```

Utilise `--help` après le séparateur `--` :

```bash
./bin/mimir --lua scripts/training/train_vae_conv.lua -- --help
```

Le premier `--` appartient au CLI Mímir ; les arguments suivants sont transmis au script Lua dans la table globale `arg`.

## Parcours C — Entraîner et reprendre

Lis :

1. [Données](../02-User-Guide/03-Data.md) ;
2. [Entraînement](../02-User-Guide/04-Training.md) ;
3. [Checkpoints](../02-User-Guide/08-Checkpoints.md) ;
4. le guide de l’architecture choisie, par exemple [VAEConv](../02-User-Guide/14-VAEConv.md).

Avant un long run :

```bash
# Afficher les options réelles du script
./bin/mimir --lua scripts/training/train_vae_conv.lua -- --help

# Valider les primitives sans dataset
ctest --test-dir build --output-on-failure \
  -R 'ModelTest.VAEConvContract|AutogradTest.Numerical|RuntimeTest.Math'
```

Règles pratiques :

- commence avec une petite résolution ;
- fixe un seed ;
- conserve config, tokenizer et optimizer avec le checkpoint ;
- ne change pas la topologie lors d’une reprise ;
- mesure avant d’activer un backend GPU ou une attention coûteuse.

## Parcours D — Ajouter ou modifier un modèle

Lis dans cet ordre :

1. [Construire des modèles et layers](../07-Devs/02-Building-Models-And-Layers.md) ;
2. [Config et registre](../07-Devs/03-Config-And-Registry.md) ;
3. [Layers et opérations](../04-Architecture-Internals/14-Layers-And-Ops.md) ;
4. [Autograd](../04-Architecture-Internals/13-Autograd-Gradients.md) ;
5. [Tutoriel ajouter un modèle](../08-Tuto/02-Tuto-Ajouter-Modele.md).

Checklist minimale :

```text
Config publique
  → parser JSON
  → default_config du registre
  → builder C++
  → noms d’entrées/sorties
  → allocation et initialisation
  → forward
  → backward
  → sérialisation
  → test ciblé
  → documentation
```

Ne déduis pas le support d’un layer depuis sa seule présence dans `LayerType`. Vérifie ses chemins forward et backward dans les runtimes.

## Parcours E — Modifier un runtime

Lis :

1. [Runtime Development](../07-Devs/04-Runtime-Development.md) ;
2. [Backends hardware](../04-Architecture-Internals/03-Hardware-Backends.md) ;
3. [Tutoriel runtime](../08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md) ;
4. [Tutoriel ajouter une op](../08-Tuto/05-Tuto-Ajouter-Op.md).

Le contrat essentiel est :

```text
RuntimeRouter
  → essaie les runtimes dans l’ordre configuré
  → un runtime retourne true seulement si le résultat est valide
  → false demande le fallback suivant
  → CPU constitue la référence générale
```

Après une modification :

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure \
  -R 'RuntimeTest|AutogradTest|Planner'
```

## Où chercher la vérité

| Question | Source principale |
| --- | --- |
| Quelles architectures existent ? | `src/Models/Registry/ModelArchitectures.cpp` |
| Quelle config construit un modèle ? | header du modèle + parser du registre |
| Quelle API Lua existe ? | `LuaScripting::registerAPI()` |
| Comment un script lit ses flags ? | `scripts/modules/args.lua` |
| Où part un layer ? | `src/runtimes/RuntimeRouter.cpp` |
| Quelle implémentation CPU est utilisée ? | `src/runtimes/cpu/RuntimeLayerDispatch.hpp` |
| Comment les gradients sont accumulés ? | `Model::backwardPass()` et les `backwardLayer()` |
| Comment un checkpoint est écrit ? | `src/Serialization/` |

## Étapes suivantes

- [Page précédente : Smoketest (5 minutes)](05-Smoketest.md)
- [Index de la documentation](../00-INDEX.md)
- [Revenir à la documentation](../00-INDEX.md)
