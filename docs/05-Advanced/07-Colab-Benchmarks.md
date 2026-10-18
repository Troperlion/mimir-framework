# Benchmarks Colab : CPU, CUDA, Vulkan, OpenCL et TPU

Le [notebook de comparaison](../../Colab-bench-multi-framework.ipynb) contient
un microbenchmark multi-framework et un benchmark MNIST complet séparé.
Le [notebook de validation](../../Colab-test.ipynb) compile et vérifie les
runtimes, sans produire le même classement. Les deux embarquent les sources
locales, y compris les nouveaux headers et tests, avec un manifeste SHA256.

## Démarrer une session cohérente

1. Charger la version actuelle du notebook dans Colab et exécuter les cellules
   depuis la configuration, l’extraction des sources et la compilation.
2. Choisir le matériel et les runtimes Mímir dans la configuration. Sur une VM
   GPU, `GPU_CUDA`, `GPU_VULKAN` et `GPU_OPENCL` sont indépendants ; les concurrents
   PyTorch, TensorFlow et JAX utilisent CUDA pour cette catégorie de matériel.
3. Exécuter la validation avant les mesures. Un SDK présent ne prouve pas qu’un
   pilote ou un device est utilisable. Un runtime indisponible est consigné ;
   `GPU_REQUIRE_ALL=True` rend son absence bloquante.
4. Conserver le dossier de résultats : manifeste, fixtures, métriques brutes,
   versions, erreurs, CSV, HTML et archive exportée.

`SOURCE_MODE="embedded"` extrait l’instantané dans un dossier dédié. Le mode
`checkout` est une option explicite ; il peut pointer sur une branche distante
plus ancienne que les sources embarquées. Le générateur de maintenance est
`scripts/colab/embed_sources.py`. Il synchronise les archives et le helper du
microbenchmark, sans régénérer les kernels du core.

## Choisir le profil du microbenchmark

| Profil | Cas | Usage |
| --- | ---: | --- |
| `portable` | 7 | Formes bornées pour les kernels de référence Vulkan/OpenCL ; choix initial si l’un de ces runtimes est demandé sur GPU. |
| `quick` | 7 | Premier contrôle des chemins CPU/CUDA/TPU et des nouvelles opérations. |
| `standard` | 11 | Matrices plus grandes et attention à 2 048 tokens ; les formes peuvent être refusées par un runtime. |

Les sept familles de base sont Linear, Conv2d, LayerNorm, Linear+SGD,
BatchMatMul forward, BatchMatMul backward et attention causale. Standard ajoute
deux anciens cas larges, un BatchMatMul large et une attention longue.
Les dimensions d’un profil sont identiques pour tous les concurrents : le
notebook ne réduit pas une forme uniquement pour améliorer le score d’un runtime.

Le backward matriciel mesure uniquement dA et dB à partir de A, B et du gradient
de sortie fourni. Tous les adaptateurs calculent ces deux produits, sans forward
ni mise à jour des poids dans ce cas. Linear+SGD suit un autre contrat, avec
forward, perte, backward et mise à jour vérifiés séparément.

## Identités des runtimes et classement

Sur GPU, chaque runtime Mímir activé et validé produit son concurrent :
`mimir-cuda`, `mimir-vulkan`, `mimir-opencl`. Les autres frameworks ne sont pas
dupliqués. Le résultat est accepté seulement si le runtime demandé correspond
au champ retourné et à la sélection native du dispatcher. Un repli CPU n’est
pas une mesure Vulkan/OpenCL valide.

Les kernels portables conservent leurs limites de taille et de travail.
Un gros cas refusé reste incomplet ; il n’est ni remplacé par un plus petit cas
ni compté comme une latence nulle. La note exige une couverture complète et au
moins deux concurrents complets. Elle utilise la moyenne géométrique des ratios
entre meilleure médiane et médiane du concurrent, avec le même poids par cas.

Le protocole microbenchmark est `mimir-colab-micro-v3`. Comparer les notes entre
runs du même protocole, profil, matériel, précision et frontière de mesure.
Le score mesure ces opérations ; il n’évalue pas la qualité générale d’un framework.

## Temps, mémoire et provenance

Les microbenchmarks utilisent FP32 et incluent les entrées/poids depuis l’hôte,
les résultats matérialisés sur l’hôte et leur synchronisation. Imports et
initialisation sont distingués du chemin chaud ; chauffe et essais sont explicites.
Médiane, p95, IQR, débit et erreurs numériques sont conservés.

Le RSS du microbenchmark est obtenu par `/usr/bin/time -f %M`, pour éviter un
pic hérité du processus notebook. Il inclut les imports et allocations hôte du
processus mesuré. Il ne mesure pas la VRAM/HBM et ne participe pas à la note.
La mesure remontée par le worker est aussi conservée pour diagnostic.

Le benchmark MNIST utilise les API de modèle complètes, des poids persistants,
une référence indépendante de la première étape, un jeu de test séparé et un
seuil de qualité. Ses limites de temps et RSS+swap concernent l’arbre des processus
et sont contrôlées par `scripts/benchmarks/colab_real_suite.py`. Les métriques,
le score et les profils MNIST restent séparés de ceux des microbenchmarks.

## Éviter les cellules et sources décalées

Une trace contenant `spec_from_file_location('mimir_colab_micro_suite', ...)`
puis un fichier absent sous `/content/mimir-source-...` indique une ancienne
cellule de chargement. La cellule actuelle embarque `MICRO_SUITE_SOURCE`, écrit
le helper dans le dossier de résultats et le charge sans dépendre d’un fichier
dans REPO. Elle fonctionne même avec un ancien checkout. Cela ne met pas à jour
le binaire : après changement des sources ou du pilote, reprendre l’extraction,
la compilation, puis la validation.

## Validation effectuée et limites

Le 18 octobre 2026, les onze cas standard ont passé la référence NumPy sur CPU
dans Mímir, PyTorch, TensorFlow et JAX : 44 comparaisons numériques. Les sept
cas portables ont passé la même validation sur Vulkan/RADV et OpenCL/Rusticl
avec le GPU AMD Radeon Graphics : 14 combinaisons natives. Les cellules de mesure,
classement et export ont également été exécutées pour les deux runtimes séparés.
Les tests de parité du core passent 165 cas par runtime ; les 19 tests Python
Colab passent, dont le chargement du helper sans checkout récent.

Ces essais courts valident le fonctionnement et la précision, pas un classement
publiable. Ils ne prouvent pas l’exécution sur le GPU NVIDIA d’une session Colab,
ni sur TPU. Les traces et tests matériels de la session cible restent nécessaires.

## Vérifier les helpers sans accélérateur

Depuis la racine, avec NumPy et pandas dans l’interpréteur choisi :

```bash
python3 -m unittest discover -s scripts/tests -p 'test_colab_*.py'
```

Cette suite a été exécutée localement avec ces dépendances. Sans elles, certains
tests numériques sont ignorés ; un résultat avec skips n’équivaut pas à cette
validation complète. Les tests Python n’exécutent pas les GPU Colab.

- [Performance et limites des gains](01-Performance.md)
- [Runtimes GPU](../04-Architecture-Internals/21-GPU-Runtimes.md)
- [Index](../00-INDEX.md)
