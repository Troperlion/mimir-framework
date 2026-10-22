# Performance : calcul, copies et mémoire

Cette page décrit les chemins implémentés dans le core. `Model.cpp` orchestre
le graphe, les versions de tenseurs et les gradients ; les calculs des couches
restent dans les runtimes et leur dispatcher. Les signatures C++/Lua et les
formats des modèles existants sont conservés.

## Chemins CPU optimisés

| Opération | Implémentation | Effet et limite |
| --- | --- | --- |
| Linear backward | `src/runtimes/cpu/RuntimeLayerDispatch.hpp` | Réductions par blocs de features contiguës, SIMD et OpenMP selon la charge ; gradients de paramètres accumulés dans le bloc packed. |
| MatMul/BatchMatMul backward | `src/runtimes/cpu/MatMulKernels.hpp` | Réduction SIMD pour dA et colonnes contiguës pour dB ; pas de tampon de transposition. |
| BatchMatMul forward | Même header | Blocs de 24 puis 8 colonnes et reste scalaire ; une région parallèle pour les lignes des lots. |
| Conv2d backward | `src/runtimes/cpu/ConvolutionKernels.hpp` | Réutilisation du tampon im2col pour dCol et accumulation directe de dW ; un scratchpad découpé en trois vues avec deux espacements de 64 octets pour limiter les conflits de cache. |
| Convolution 1×1 | Même header | Chemin CHW direct pour stride 1, padding 0 et certaines formes ; les formes larges gardent le packing. |
| Concat/Stack forward | `src/runtimes/cpu/FlatJoinKernels.hpp` | Entrées empruntées en lecture seule ; une allocation de sortie, sans copies propriétaires de toutes les entrées. |
| SelfAttention/MultiHeadAttention/CrossAttention forward | `src/runtimes/cpu/LayerOps.hpp` | Lecture des projections combinées sans copies Q/K/V complètes ; scores et contexte temporaire par blocs de 128 requêtes. |

Le bloc d’attention contient toutes les clés de chaque ligne : le softmax et
le masque causal conservent leur sens. Le tampon de scores occupe
`min(128, queries) × keys` flottants, contre `queries × keys` auparavant.
La complexité du calcul reste quadratique. Les projections, le contexte final
et le backward nécessitent encore leurs propres buffers.

Les kernels comportent des seuils OpenMP : augmenter le nombre de threads
n’accélère pas automatiquement les petites matrices. AVX2/FMA dépendent des
options de compilation et de la machine. Les tailles non multiples de huit
sont couvertes par les restes scalaires ; une réduction SIMD peut modifier
légèrement l’arrondi flottant.

## Orchestration et allocations

- Le backward déplace les gradients consommés au lieu de les recopier lorsque
  leur version ne doit plus être conservée. Les branches continuent de sommer
  leurs contributions.
- Pour les couches dont le backward ne lit que les tailles, `Model.cpp` peut
  emprunter un gradient immuable de même taille. Si aucun ne convient, il crée
  un tampon zéro par taille distincte. Il n’emprunte pas une version écrasée
  dans TensorStore.
- Le routeur réutilise les conteneurs de routes forward, mais réévalue type,
  capacité et initialisation. Le backward garde son instantané de routes sur
  la pile pour les six backends usuels.
- `src/runtimes/FiniteCheck.hpp` classe les NaN/Inf par groupes SIMD avec reste
  scalaire. Les contrôles de finitude et le repli après échec restent actifs.
- Les conversions F32 de `TypedTensor` évitent un remplissage zéro suivi d’une
  copie lorsque les données peuvent être construites directement.

Les scratchpads sont décrits dans
[RuntimeAllocator et scratchpads](../04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md).
Leurs compteurs ne représentent pas le RSS total du processus ni la VRAM.

## Vulkan/OpenCL et entraînement

Vulkan et OpenCL ont des chemins forward **et backward**, avec optimizers
routés selon les capacités disponibles. Ils ne sont pas limités à l’inférence.
Les kernels de référence privilégient la couverture et la conformité numérique ;
les transferts et les limites de travail peuvent rendre le CPU plus rapide.
Une capacité par type ne garantit pas l’acceptation de toute forme.

`RuntimeRouter` choisit une route native admissible ; dans un modèle qui autorise
le repli, un refus peut conduire au CPU. Le microbenchmark Colab vérifie le
runtime effectivement sélectionné et refuse d’attribuer une exécution CPU à
Vulkan/OpenCL. Les budgets de sécurité des kernels portables restent actifs.
Voir [Runtimes GPU](../04-Architecture-Internals/21-GPU-Runtimes.md).

## Sélection du runtime des optimizers

`RuntimeRouter::dispatchOptimizer` compare les runtimes disponibles et compatibles
pour le type d’optimizer, la taille exacte du bloc et les buffers de moments
présents. Il effectue une chauffe puis cinq mesures sur des copies de l’état,
et classe les candidats par médiane. La mesure inclut le coût de l’update réel :
copies de protection, transferts et synchronisation côté device. Un optimizer
CPU peut donc gagner même si les couches du modèle sont calculées sur GPU.

Le choix est amorti par un cache borné à 512 configurations par thread. Il est
réévalué après 1 024 updates réussis pour cette configuration, lors d’une reconfiguration des
runtimes, d’un changement de device/candidats ou après un échec. Les options
Adafactor et la phase RAdam qui changent le travail sont distinguées. Une route
seule ne déclenche pas une compétition inutile.

La calibration ne modifie ni les poids/moments du modèle ni le compteur de pas.
Un échec device ne valide pas son état temporaire ; le routeur essaie le candidat
suivant selon les médianes mesurées. Les audits d’optimizer comptent les updates
réels, pas les essais. Le log `[optimizer]` donne le backend retenu et les temps
médians des candidats sous `candidates_seconds`. Le choix désigne le plus rapide
mesuré, pas une garantie permanente face à des variations instantanées de charge.

## Diagnostics de gradients à la demande

`LiveModelConfig::gradient_diagnostics` est observé par le monitoring. Sans
observateur, le modèle ne demande pas de parcours supplémentaire pour ces
statistiques. Lorsqu’il est activé, une observation est enregistrée par mise à
jour d’optimizer, indépendamment de la fréquence de rafraîchissement de la Viz.

La fenêtre contient les 256 dernières observations de norme avant clipping ;
les couches gelées sont exclues. Le snapshot expose le nombre d’échantillons,
les valeurs non finies, la médiane, le p95, le maximum non tronqué et le taux de
clipping effectif. Le taux tient aussi compte des étapes où le clipping est
désactivé. Il s’agit de diagnostics de gradients, pas d’un indicateur de qualité
ou de convergence à lui seul. `MetricsTest.GradientDiagnostics` couvre la fenêtre
et `ModelTest.LiveConfiguration` couvre l’activation via le contrôle runtime.

## Mesurer et interpréter

Comparer sur la même machine et le même build : dtype, formes, threads,
affinité, nombre d’essais, chauffe et frontière hôte/device doivent être fixes.
Séparer forward, backward seul, étape d’entraînement et inférence complète.
Conserver les sorties, les tolérances et les erreurs numériques avec les temps.

Les mesures CPU locales du 18 octobre 2026, sur Ryzen 5 5500U, GCC 15,
Release et un thread OpenMP, illustrent les gains ciblés :

| Cas isolé | Avant → après | Interprétation |
| --- | --- | --- |
| BatchMatMul backward, M128 K256 N192, batch 2 | 22,27 → 1,91 ms | Comparaison avec l’état déjà optimisé de la passe précédente. |
| BatchMatMul forward, mêmes dimensions | 9,13 → 0,77 ms | Forward CPU isolé, pas une accélération globale du modèle. |
| Concat, huit entrées de 1 048 576 flottants | 42,00 → 21,03 ms ; RSS 125,13 → 93,15 Mio | Allocations pendant l’appel : 10 → 1. |
| SelfAttention, 2 048 tokens, dimension 128, quatre têtes | 323,21 → 263,04 ms ; RSS 32,48 → 14,23 Mio | Forward complet CPU ; scores seuls : 16 → 1 Mio. |

Ces comparaisons utilisent quatre essais alternés avant/après et les médianes
après chauffe. Le RSS est mesuré extérieurement par GNU time. Les étapes MLP
MNIST de contrôle restent pratiquement stables, avec les pertes et sondes de
paramètres comparées identiques ; les gains d’un kernel ne se transfèrent pas
à tous les modèles. Ces chiffres ne mesurent pas CUDA/ROCm ni un GPU Colab.

Pour une procédure réutilisable, voir les
[benchmarks Colab](07-Colab-Benchmarks.md), avec leurs profils, sources embarquées,
validations natives, métriques de mémoire et limites de comparaison.

## Étapes suivantes

- [Débogage et stabilité numérique](02-Debugging.md)
- [Carte du code source](04-Source-Code-Map.md)
- [Index de la documentation](../00-INDEX.md)
