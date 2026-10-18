# Monitoring (HtopDisplay / Visualizer / AsyncMonitor)

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

Cette page documente la stack “monitoring” côté C++.

## Sur cette page

- [Source de vérité (C++)](#source-de-vérité-c)
- [Intention](#intention)
- [Architecture globale](#architecture-globale)
- [HtopDisplay (UI terminal)](#htopdisplay-ui-terminal)
- [Visualizer (SFML, optionnel)](#visualizer-sfml-optionnel)
- [AsyncMonitor](#asyncmonitor)
- [Pilotage depuis Lua](#pilotage-depuis-lua)
- [Étapes suivantes](#étapes-suivantes)

## Source de vérité (C++)

- `src/HtopDisplay.hpp`
- `src/Visualizer.hpp` (SFML optionnel)
- `src/AsyncMonitor.hpp`

## Intention

Le monitoring a 2 objectifs distincts, souvent activés ensemble pendant l’entraînement:

- **HtopDisplay**: UI terminal (style htop), très légère, export CSV.
- **Visualizer (SFML)**: UI fenêtrée, affichage d’images/frames + courbes loss + contexte dataset.

`AsyncMonitor` orchestre le tout de façon **asynchrone**, pour éviter de bloquer le thread principal.

## Architecture globale

### Qui tourne dans quel thread ?

- Le training loop (Lua/C++) tourne dans le **thread principal**.
- `AsyncMonitor` peut lancer:
  - un thread `htop_thread_` qui redessine l’UI terminal périodiquement.
  - un thread `viz_thread_` qui initialise SFML puis fait la boucle de rendu.

Pourquoi c’est important:

- SFML/OpenGL impose que la fenêtre soit créée et manipulée dans **le même thread**. `AsyncMonitor::start()` initialise donc `Visualizer` *dans* le thread viz (voir le commentaire “IMPORTANT (SFML)”).

### Synchronisation (best-effort)

- Les métriques sont copiées via `AsyncMonitor::updateMetrics()` (mutex + flag `has_update_`).
- Les frames/images destinées au visualizer sont stockées en “pending” (mutex dédié) puis consommées par la boucle viz.

Le design est volontairement simple:

- **Pas de lock-free**, pas de ring-buffer complexe.
- L’objectif est d’éviter les crashs et les blocages, pas de garantir un rendu 100% déterministe.

## `HtopDisplay` (UI terminal)

### Rôle

- Afficher:
  - progression (epoch/batch)
  - loss, métriques diverses (KL, Wasserstein, etc.)
  - mémoire et perf (MB, batches/s, ETA)
  - état optimizer (type/step/betas/eps/weight_decay)
- Exporter un CSV (par défaut `checkpoints/htop_metrics.csv`).

### I/O

- Ecrit dans `stdout` en utilisant des codes ANSI.
- Utilise `ioctl(TIOCGWINSZ)` pour récupérer la taille du terminal.

### Export CSV (HtopDisplay)

- La structure `CsvRecord` reflète un superset des métriques.
- `saveLossHistoryCsv()`:
  - crée le dossier parent en best-effort
  - écrit l’en-tête + toutes les lignes

### Pièges

- Les codes ANSI ne sont pas “portable UI”: si sortie redirigée vers un fichier, le rendu est illisible.
- Les écritures fréquentes peuvent coûter si le terminal est lent.

## `Visualizer` (SFML, optionnel)

### Gating compile-time

`src/Visualizer.hpp` expose 2 implémentations:

- si `ENABLE_SFML`: vraie classe SFML avec fenêtre et textures.
- sinon: version **headless** API-compatible (toutes les méthodes sont no-op / false).

Conséquence:

- Le code peut appeler `Visualizer` sans `#ifdef` partout.

### Données gérées

Le visualizer stocke typiquement:

- images générées (grayscale 64×64 dans `addGeneratedImage()`)
- image dataset courante (+ label)
- texte dataset (raw / tokenized / encoded)
- images de projection / “understanding”
- frames de blocs/layers (`setLayerBlockImages()`)
- métriques d’entraînement + historique de loss

### “Architecture awareness”

Le visualizer peut (optionnel) charger un fichier d’architecture (json) pour:

- masquer certains blocs (`hide_activation_blocks`)
- filtrer par noms de layers / outputs / sinks

La logique est best-effort (`maybeLoadArchitecture()`), pensée pour améliorer la lisibilité.

### Export CSV

`Visualizer::saveLossHistory()` exporte une histoire structurée (similaire aux colonnes htop).

Note importante:

- `AsyncMonitor::configureMetricsCsv()` maintient un export unique : Viz est
  prioritaire lorsqu'elle est active, et htop sert de repli lorsqu'il tourne
  seul. Le chemin et l'état d'activation sont communs aux deux interfaces.

## `AsyncMonitor`

### Rôle (Visualizer)

- Unifier htop + viz derrière une API thread-safe.
- Gérer le lifecycle:
  - `start(enable_htop, enable_viz, viz_config)`
  - `stop()` (join threads, restore cursor)

### Invariants

- `start()` est **idempotent** (peut être rappelé pour activer viz plus tard).
- La destruction (`~AsyncMonitor`) appelle `stop()`.
- Le thread viz:
  - initialise SFML
  - notifie `vizInitOk()` / `vizInitError()`
  - appelle `viz_->shutdown()` dans le même thread à la fin.

### API d’update

- `updateMetrics(Metrics)`
- `addImage(pixels, prompt)`
- `setDatasetImage(...)`, `setDatasetText(...)`, `setDatasetSample(...)`
- `setProjectionImage(...)`, `setUnderstandingImage(...)`

### Pièges (AsyncMonitor)

- Les images peuvent être volumineuses: garder des copies “pending” a un coût.
- Les métriques sont “dernier état connu”: pas d’historique côté `AsyncMonitor` (hormis ce que visualizer/htop accumulent).

## Pilotage depuis Lua

La mise en place/activation du monitoring est exposée côté Lua via `src/scriptings/Lua/luaScripting/LuaScripting.cpp` (voir la doc API: `docs/03-API-Reference/15-Viz-Htop.md`).

## Étapes suivantes

- [Page précédente : Backends hardware : CPU / CUDA / ROCm / Vulkan / OpenCL](03-Hardware-Backends.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals: AdvancedRAMManager (cache RAM / compression / spill disque)](05-AdvancedRAMManager.md)
