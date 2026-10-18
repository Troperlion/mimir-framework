# Étendre le visualizer

Savoir brancher correctement les tips (viz taps), et ajouter/configurer/implementer
une nouvelle feature Viz sans casser le run.

**Public concerné :** Developpeur framework (C++/runtime/monitoring/UI SFML).

> **Prérequis**
>
> Comprendre les chapitres Dev 01, 04 et 05.

1. ajouter un nouveau tip visuel dans le flux training,
2. etendre l'UI Visualizer (rendu, raccourcis, options),
3. garder un comportement stable et backward-compatible.

## Sur cette page

- [1. Pipeline Viz dans Mimir](#1-pipeline-viz-dans-mimir)
- [2. Comment brancher un tip](#2-comment-brancher-un-tip)
- [3. Ajouter une feature Viz (rendu/comportement)](#3-ajouter-une-feature-viz-renducomportement)
- [4. Configurer une nouvelle feature](#4-configurer-une-nouvelle-feature)
- [5. Checklist de securite (anti-regressions)](#5-checklist-de-securite-anti-regressions)
- [6. Recette concrete: ajouter une feature de palette](#6-recette-concrete-ajouter-une-feature-de-palette)
- [7. Validation rapide avant commit](#7-validation-rapide-avant-commit)
- [Étapes suivantes](#étapes-suivantes)

## 1. Pipeline Viz dans Mimir

Chemin de donnees (vue simplifiee):

1. `Model` calcule le forward.
2. `Model` produit des `VizFrame` (tips) quand `viz_taps_enabled_` est actif.
3. `AsyncMonitor` recupere les frames et les passe au `Visualizer`.
4. `Visualizer::setLayerBlockImages(...)` convertit en `ImageData`.
5. `Visualizer` cree/rebuild les textures SFML puis rend les panels.

Fichiers de reference:

- `src/Model.cpp` (generation des viz taps)
- `src/AsyncMonitor.hpp` (transport async des metriques + frames)
- `src/Visualizer.hpp`
- `src/Visualizer.cpp`

## 2. Comment brancher un tip

Depuis la mise a jour recente, il existe deux chemins complementaires:

1. generation directe de frames (`VizFrame`) dans le flux modele,
2. personnalisation de labels via hooks optionnels dans la classe modele enfant.

Le chemin 2 est recommande pour standardiser les labels par architecture.

### 2.1 Produire la frame cote modele

Dans la section viz taps du forward (`src/Model.cpp`):

1. construire un `VizFrame` valide,
2. remplir `pixels`, `w`, `h`, `channels`, `label`,
3. optionnel: `pixels_real` (version naturelle pour mode non-heatmap),
4. pousser avec `addVizTapFrame(std::move(vf))`.

Champs minimaux:

- `w > 0`, `h > 0`
- `channels` dans `{1,3,4}`
- `pixels.size() == w*h*channels`
- `label` stable et explicite

### 2.2 Conventions de label (important)

Le label pilote le tri/couleur/section UI.
Utiliser une forme hiérarchique stable, par exemple:

`<model>/blocks/<path>/<type>`

Exemples:

- `vae_conv/blocks/dec/out/Conv2d`
- `vae_conv/latent/mu`
- `vae_conv/err/resdiff_abs`

Bonnes pratiques:

- garder des noms predictibles,
- eviter les labels ambigus,
- ne pas reutiliser un label pour un contenu semantiquement different.

### 2.3 Hooks optionnels par modele (recommande)

Le socle `Model` expose maintenant deux hooks virtuels:

- `InitVizTips()`
- `UpdateVizTips(const Layer& layer, VizFrame& frame)`

Comportement par defaut:

- dans `Model` base, les hooks sont inactifs (mode opt-in),
- aucun tip custom n'est applique si le modele enfant ne surcharge pas ces methodes.

Contrat pratique:

1. `InitVizTips()` est appelee une fois quand les viz taps sont actives,
2. elle peut remplir un registre `layer.name -> tip label` via `registerVizTip(...)`,
3. `UpdateVizTips(...)` peut appliquer ce registre et/ou des regles dynamiques.

Exemple minimal (modele enfant):

```cpp
bool MyModel::InitVizTips() {
	clearVizTipsRegistry();
	registerVizTip("my_model/enc/in", "Dataset/raw");
	registerVizTip("my_model/enc/mu", "Latent/mu");
	return true; // active le chemin custom
}

bool MyModel::UpdateVizTips(const Layer& layer, VizFrame& frame) {
	if (Model::UpdateVizTips(layer, frame)) return true; // applique le mapping exact

	if (layer.name.find("/attn") != std::string::npos) {
		frame.label = "Attention|" + frame.label;
		return true;
	}
	return false;
}
```

Points importants:

1. Toujours appeler `Model::UpdateVizTips(...)` en premier si vous voulez conserver le mapping exact.
2. Garder les prefixes de labels stables (ex: `Dataset/...`, `Encoder/...`, `Latent/...`, `Output/...`).
3. Ne pas faire de logique couteuse dans `UpdateVizTips(...)` (appelee tres souvent).

### 2.4 Exemples existants dans le repo

- `src/Models/Vision/VAEConvModel.cpp`

Ces deux modeles montrent:

1. un mapping statique de layers cle,
2. des regles dynamiques basees sur des fragments de nom de layer,
3. une composition du label sous la forme `prefix|label_existant`.

## 3. Ajouter une feature Viz (rendu/comportement)

### 3.1 Choisir le bon niveau

- Generation de donnees: `Model.cpp`
- Transport async: `AsyncMonitor.hpp`
- Rendu/interactions: `Visualizer.cpp/.hpp`

Regle cle:

- Une feature purement UI doit rester dans `Visualizer`.
- Ne pas deplacer inutilement de logique UI dans `Model`.

### 3.2 Ajouter un nouvel etat UI

1. ajouter l'etat dans `Visualizer.hpp` (bool, enum, valeur),
2. initialiser dans le constructeur `Visualizer::Visualizer(...)`,
3. brancher la persistance dans `serializeUILayout()` / `applyUILayout()`,
4. afficher l'etat (badge/texte) dans `drawPanelChrome(...)` si utile.

### 3.3 Ajouter un raccourci clavier

Dans `Visualizer::processEvents()`:

1. intercepter la touche,
2. modifier l'etat,
3. appeler le rebuild adequat (`rebuildLayerBlockTextures()` ou `rebuildAllTextures()`),
4. sauvegarder le layout (`saveUILayoutToLast()`) si l'etat est persistant,
5. documenter la touche dans `renderHelpOverlay()`.

### 3.4 Mettre a jour le rendu

Selon la feature:

- creation/traitement pixels: `setLayerBlockImages(...)`
- upload textures: `createLayerBlockTexture(...)`, `createImageTexture(...)`
- rendu panel: `renderLayerBlocks()`, `renderTrainingProgress()`, etc.

## 4. Configurer une nouvelle feature

### 4.1 Runtime config (JSON modelConfig)

Si la feature depend du modele/training:

- ajouter une cle de config dans le chemin de lecture approprie,
- conserver un default stable,
- ne pas casser les anciens configs.

### 4.2 UI config (`viz_ui_settings.json`)

Si la feature est pure UI:

- persister via `serializeUILayout()`,
- relire via `applyUILayout()`,
- valider les bornes (`clamp`) a la lecture.

## 5. Checklist de securite (anti-regressions)

1. Ne jamais interrompre le forward pour un besoin Viz.
2. Filtrer une preview dans le chemin Viz, pas dans l'execution du layer.
3. Verifier les dimensions/canaux avant upload texture.
4. Eviter les interpretations de layout implicites non verifiees (CHW/HWC).
5. Pour toute interaction clavier, verifier l'effet immediat sans attendre la step suivante.
6. Compiler et verifier les erreurs avant merge.
7. Lorsque vous ajoutez des hooks tips, verifier qu'un modele sans surcharge garde bien le comportement precedent.

## 6. Recette concrete: ajouter une feature de palette

Exemple de sequence complete:

1. Ajouter l'etat (enum palette) dans `Visualizer.hpp`.
2. Ajouter la conversion palette -> couleur dans `Visualizer.cpp`.
3. Appliquer la palette dans `setLayerBlockImages(...)` / rebuild.
4. Ajouter la touche (`K`) dans `processEvents()`.
5. Persister la palette dans le layout UI.
6. Ajouter un badge `PAL ...` dans le header du panel.
7. Documenter dans l'overlay aide et dans la doc utilisateur.

## 7. Validation rapide avant commit

Commandes conseillees:

```bash
cmake --build build --target mimir --parallel
```

Verification manuelle:

1. la Viz s'ouvre et reste fluide,
2. les tips cibles apparaissent dans la bonne section,
3. les toggles clavier ont un effet immediat,
4. la progression training reste coherent (global + batch),
5. pas d'artefact de texture evident (bande, repetition, couleurs incoherentes).

Verification hooks tips:

1. `InitVizTips()` est appelee une fois a l'activation des taps,
2. les labels mappes via `registerVizTip(...)` apparaissent en priorite,
3. les regles dynamiques de `UpdateVizTips(...)` ne dupliquent pas les prefixes,
4. la desactivation/re-activation des taps reinitialise proprement l'etat des tips.

## Étapes suivantes

- [Page précédente : Contrat API Scripting Inter-Langages](05-Scripting-System-Contract.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Runbook Debug Visualizer](07-Visualizer-Debug-Runbook.md)
