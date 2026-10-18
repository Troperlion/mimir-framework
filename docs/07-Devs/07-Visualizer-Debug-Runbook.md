# Déboguer le visualizer

Ce runbook permet de diagnostiquer les problèmes du visualizer Mímir à partir
d'un symptôme observable. Chaque fiche associe vérification, cause probable et
correction.

**Public concerné :** développeurs C++, intégrateurs et personnes qui
maintiennent les outils de visualisation.

## Sur cette page

- [Principe](#principe)
- [Checklist express avant toute investigation](#checklist-express-avant-toute-investigation)
- [Cas 1: preview en bande horizontale](#cas-1-preview-en-bande-horizontale)
- [Cas 2: image repetee/decalee dans Outputs](#cas-2-image-repeteedecalee-dans-outputs)
- [Cas 3: perte de sortie training apres patch Viz](#cas-3-perte-de-sortie-training-apres-patch-viz)
- [Cas 4: palettes/toggles sans effet immediat](#cas-4-palettestoggles-sans-effet-immediat)
- [Cas 5: tips majoritairement noir/blanc](#cas-5-tips-majoritairement-noirblanc)
- [Cas 6: ordre recon/diff incoherent](#cas-6-ordre-recondiff-incoherent)
- [Cas 7: ProgressBar incoherente](#cas-7-progressbar-incoherente)
- [Procedure de debug standard](#procedure-de-debug-standard)
- [Gate de validation avant merge](#gate-de-validation-avant-merge)
- [Étapes suivantes](#étapes-suivantes)

## Principe

Format de chaque fiche:

1. Symptome observable
2. Causes probables
3. Verification rapide
4. Correctif recommande
5. Fichier(s) a inspecter

## Checklist express avant toute investigation

1. Verifier que le build est a jour.
2. Reproduire sur un run court et deterministic (seed fixe).
3. Confirmer que le souci est Viz (et non un bug modele/runtime).
4. Noter le label exact du tip/preview qui pose probleme.

Commande utile:

```bash
cmake --build build --target mimir --parallel
```

## Cas 1: preview en bande horizontale

Symptome:

- une vignette ressemble a une ligne horizontale/verticale au lieu d'une image.

Causes probables:

- mauvaise inference H/W/C (layout CHW/HWC),
- cas `Permute` interprete comme shape directe alors que c'est un reorder.

Verification rapide:

1. verifier le label du layer (ex: `recon_to_hwc`),
2. verifier les dimensions deduites dans la logique de viz tap.

Correctif recommande:

- traiter explicitement les permutations connues (ex: CHW -> HWC),
- prioriser une interpretation de shape coherente avec `permute_dims`.

Fichiers a inspecter:

- `src/Model.cpp` (inference shape pour viz taps)

## Cas 2: image repetee/decalee dans Outputs

Symptome:

- rendu mosaïque/duplique, image visiblement incoherente dans Outputs.

Causes probables:

- preview d'un tenseur de packing final (`out_concat`/`out_pack`) interprete comme image.

Verification rapide:

1. regarder le label du tip (`out_concat`, `out_pack`, `x`),
2. verifier si ce tip devrait etre une image ou un pack de vecteurs.

Correctif recommande:

- exclure ces labels du rendu image Viz,
- conserver l'execution normale du layer (ne jamais court-circuiter le forward pour la Viz).

Fichiers a inspecter:

- `src/Visualizer.cpp` (`setLayerBlockImages`)
- `src/Model.cpp` (generation viz taps, sans interrompre la boucle d'execution)

## Cas 3: perte de sortie training apres patch Viz

Symptome:

- erreur du type `invalid output/image_dim` pendant train,
- regression apparue juste apres modification Viz.

Causes probables:

- un `continue`/skip dans la boucle principale des layers a ete introduit pour filtrer une preview.

Verification rapide:

1. chercher les filtres Viz dans la boucle forward,
2. verifier que le chemin de stockage output (`storeTensor`) est toujours execute.

Correctif recommande:

- transformer le filtre en condition locale de generation de tips,
- ne jamais stopper le flux d'execution du layer pour un besoin UI.

Fichiers a inspecter:

- `src/Model.cpp`

## Cas 4: palettes/toggles sans effet immediat

Symptome:

- touche clavier prise en compte, mais rendu change seulement a la step suivante.

Causes probables:

- etat UI modifie sans recolorisation/rebuild immediate,
- dependance implicite a l'arrivee de nouveaux tips.

Verification rapide:

1. tester `M`/`K` en pause relative,
2. verifier si `rebuildLayerBlockTextures` est appele apres changement.

Correctif recommande:

- appliquer le changement sur les buffers actifs,
- lancer un rebuild texture immediat,
- persister l'etat UI si necessaire.

Fichiers a inspecter:

- `src/Visualizer.cpp` (`processEvents`, `setLayerBlockImages`, `rebuildLayerBlockTextures`)

## Cas 5: tips majoritairement noir/blanc

Symptome:

- la plupart des tips restent en niveaux de gris alors que le mode heatmap est actif.

Causes probables:

- conversion 1 canal -> RGB incomplète,
- canaux mal inferes (`channels` faux vs buffer reel),
- fallback vers mode reel involontaire.

Verification rapide:

1. verifier `channels` et la taille buffer,
2. verifier le chemin de colorisation en mode heatmap.

Correctif recommande:

- inferer les canaux depuis la taille buffer,
- convertir les tips mono-canal en RGB selon la palette active,
- garder une source grise de reference pour recolorisation.

Fichiers a inspecter:

- `src/Visualizer.cpp` (`infer_channels_from_buffer_size`, `setLayerBlockImages`)

## Cas 6: ordre recon/diff incoherent

Symptome:

- `diff/resdiff` apparait loin de `recon`, lecture comparative difficile.

Causes probables:

- classification dans des sections differentes,
- absence de regle de tri post-construction des entries.

Verification rapide:

1. verifier la section assignee (`Outputs` vs `Blocks`),
2. verifier la passe de reordonnancement.

Correctif recommande:

- classer `diff/resdiff` dans `Outputs`,
- imposer `recon` juste avant `diff` dans la meme section.

Fichiers a inspecter:

- `src/Visualizer.cpp` (`section_for`, construction/reordonnancement des entries)

## Cas 7: ProgressBar incoherente

Symptome:

- barre globale instable, sous-barre mal placee, progression non intuitive.

Causes probables:

- calcul epoch/batch non borne,
- rendu de sous-barre aligne sur la barre globale sans segment de contexte,
- absence de lissage visuel.

Verification rapide:

1. comparer `current_epoch/current_total_epochs` et `current_batch/current_total_batches`,
2. verifier la geometrie de rendu des deux barres,
3. verifier la presence d'un lissage d'affichage.

Correctif recommande:

- separer barre globale et barre batch,
- borner les ratios,
- lisser les valeurs affichees,
- ajouter fallback `running` si total epochs absent.

Fichiers a inspecter:

- `src/Visualizer.cpp` (`renderTrainingProgress`)
- `src/Visualizer.hpp` (etat de progression lissee)

## Procedure de debug standard

1. Reproduire le symptome avec config minimale.
2. Isoler le label tip impacte.
3. Verifier la generation dans `Model.cpp`.
4. Verifier la transformation dans `Visualizer.cpp`.
5. Appliquer un patch local limite.
6. Recompiler et verifier diagnostics.
7. Tester l'effet en runtime (immediat, sans step suivante si interaction UI).
8. Documenter le fix si le cas est recurrent.

## Gate de validation avant merge

1. `cmake --build build --target mimir --parallel` passe.
2. Aucun diagnostic sur les fichiers modifies.
3. Le training reste fonctionnel (pas de regression `out_dim/image_dim`).
4. Le comportement UI attendu est visible sans workaround.
5. La doc utilisateur/dev est alignee avec le comportement reel.

## Étapes suivantes

- [Page précédente : Visualizer: Tips Et Nouvelles Features](06-Visualizer-Tips-And-Features.md)
- [Index de la documentation](../00-INDEX.md)
- [Revenir à la documentation](../00-INDEX.md)
