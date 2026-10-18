# Choisir l’interface Viz avant compilation

`Mimir.Viz` conserve son API Lua, ses panneaux, métriques, images, raccourcis,
réglages LIVE et captures PNG. Le choix de l’interface est exclusif et se fait
avec `MIMIR_VIZ_BACKEND` dans CMake, ou avec le menu de `./config.sh`.

| Valeur | Interface | Dépendances graphiques |
|---|---|---|
| `SFML` (défaut) | Fenêtre SFML historique | SFML 3 |
| `QT` | Widget Qt 6 | Qt 6 Widgets et Cairo |
| `GTK` | DrawingArea GTK 3 | GTK 3, Cairo et pkg-config |
| `WEB` | Canvas dans le navigateur local | Cairo, navigateur avec JavaScript |
| `NONE` | Sans Viz graphique | Aucune |

Les nouveaux hôtes QT/GTK/WEB sont implémentés pour Linux. SFML reste soumis aux
plateformes prises en charge par le projet. Qt et GTK tournent dans un processus
séparé pour que leur boucle GUI s’exécute sur le thread principal, tandis que
l’entraînement et le thread Viz restent dans Mímir.

## Compiler

Depuis la racine du dépôt, choisir **une** valeur :

```bash
cmake -S . -B build-qt -DMIMIR_VIZ_BACKEND=QT -DENABLE_SFML=OFF
cmake --build build-qt --target mimir -j2
```

Remplacer `QT` par `GTK`, `WEB` ou `SFML` et adapter le dossier de build.
Les dépendances de développement supplémentaires sur Debian/Ubuntu sont
`qt6-base-dev` pour Qt, `libgtk-3-dev` pour GTK et `libcairo2-dev` pour les trois
interfaces alternatives. Seul le backend SFML exige **SFML 3**. Une dépendance absente provoque une erreur CMake
explicite ; le backend demandé n’est pas remplacé silencieusement.

```bash
cmake -S . -B build-none -DMIMIR_VIZ_BACKEND=NONE
cmake --build build-none --target mimir -j2
```

La commande historique `-DENABLE_SFML=OFF` avec le backend par défaut reste
équivalente à `NONE`. Avec `MIMIR_VIZ_BACKEND=QT`, `GTK` ou `WEB`, cette option
reste désactivée et SFML n’est ni recherchée ni liée. Pour réactiver le backend
SFML dans un ancien cache, passer `-DENABLE_SFML=ON -DMIMIR_VIZ_BACKEND=SFML`.

`./config.sh` propose un menu dédié, puis installe les dépendances et configure
CMake. Sans interaction :

```bash
NON_INTERACTIVE=1 MIMIR_VIZ_BACKEND=WEB SKIP_SYSTEM_DEPS=1 ./config.sh
```

`SKIP_SYSTEM_DEPS=1` suppose que les dépendances sont déjà installées. Pour QT/GTK,
conserver `mimir_viz_host` à côté de `mimir` lors d’une copie manuelle des binaires.
`cmake --install` installe les deux. `MIMIR_VIZ_HOST=/chemin/mimir_viz_host` permet
un emplacement spécifique, notamment pour tester une application liée à la
bibliothèque statique `mimir_core`.

### Lancement depuis un éditeur installé avec Snap

Le sous-processus Qt/GTK reçoit une copie de l’environnement dont les chemins
Snap de bibliothèques et de modules GUI ont été retirés. Cela évite de mélanger
la glibc native avec les bibliothèques de l’éditeur (par exemple l’erreur
`libpthread.so.0: undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE`).
Les chemins natifs personnalisés et les variables de session graphique sont
conservés ; l’environnement du processus Mímir n’est pas modifié.

## Utiliser l’interface Web

Lancer le script Lua habituel qui appelle `Mimir.Viz.create(...)`. La console
indique une URL locale :

```text
[Viz WEB] http://127.0.0.1:PORT/JETON/
```

Ouvrir cette URL complète dans le navigateur. Le port est attribué automatiquement ;
`MIMIR_VIZ_WEB_PORT=8765` permet d’en imposer un. Le serveur écoute uniquement sur
`127.0.0.1`. Un jeton aléatoire propre à chaque instance fait partie de l’URL ; il
ne s’agit pas d’un service à publier sur Internet.

Cliquer dans le canvas pour lui donner le focus. Les touches, saisies de texte,
modificateurs, clics, glissements et molette sont transmis au gestionnaire
`Visualizer` existant. Les raccourcis réservés par le navigateur peuvent interférer
avec ceux de la Viz. Le bouton **Fermer la Viz** transmet la fermeture à Mímir ;
fermer simplement l’onglet laisse la Viz disponible pour une reconnexion. Plusieurs
onglets partagent le même état et les mêmes contrôles.

## Rendu et limites

SFML conserve sa `sf::RenderWindow` et son dessin historique. La scène commune
emploie désormais des primitives neutres. Pour Qt, GTK et Web, un moteur CPU
Cairo indépendant dessine les mêmes rectangles, textes, courbes, images, vues et
clips ; aucune classe, en-tête ou bibliothèque SFML n’entre dans ces builds. Qt
et GTK présentent ce framebuffer dans leur widget natif, et Web le transmet au
canvas. Le redimensionnement se fait par les contrôles/raccourcis de la Viz,
comme pour la fenêtre SFML historique. Les images Qt/GTK et le canvas conservent
leurs dimensions logiques ; le DPI du bureau et le zoom du navigateur peuvent
changer leur taille physique à l’écran.

Le backend WEB fonctionne sans serveur graphique, sans OpenGL et sans variable
`DISPLAY`. Qt et GTK nécessitent naturellement une session graphique ; Xvfb peut
être utilisé pour leurs tests automatisés. `NONE` reste le seul backend sans
moteur de dessin ni serveur d’interface.

La transmission implique une copie RGBA pour Qt/GTK et des images BMP pour le Web.
Cela consomme davantage de mémoire et de bande passante que la fenêtre SFML directe.
Les hôtes alternatifs limitent chaque framebuffer à 8192 pixels par côté et
33 554 432 pixels au total. Les captures composites PNG restent produites par le
code existant de `Visualizer`.

Les unités de compilation sont séparées par interface :

- `src/viz/backends/SfmlVizWindow.cpp` contient uniquement la fenêtre SFML ;
- `src/viz/backends/QtVizWindow.cpp` sélectionne l'hôte Qt ;
- `src/viz/backends/GtkVizWindow.cpp` sélectionne l'hôte GTK ;
- `src/viz/backends/WebVizWindow.cpp` sélectionne l'hôte Web ;
- `src/viz/SoftwareVizWindow.cpp` contient le transport framebuffer partagé par
  Qt, GTK et Web, sans code SFML.

`Visualizer.cpp` ne sélectionne plus lui-même une bibliothèque GUI. Il construit
l'interface abstraite `VizWindow` via `createVizWindow()` ; CMake compile une
seule fabrique de backend, celle demandée par `MIMIR_VIZ_BACKEND`.

## Vérification reproductible

Une cible graphique explicite évite d’ajouter une dépendance à un écran aux tests
CTest habituels :

```bash
cmake --build build-qt --target mimir_viz_smoke -j2
xvfb-run -a python3 Tests/viz_backend_smoke.py \
  --backend QT --executable "$PWD/build-qt/bin/mimir_viz_smoke" \
  --output /tmp/mimir-viz-qt-results
```

Adapter le backend et le chemin pour `SFML`, `GTK` ou `WEB`. Le test requiert
`python3-pil` et `xdotool` pour les fenêtres desktop, et `xvfb`/`xauth` pour
`xvfb-run`. Il vérifie des pixels connus, les interactions, le redimensionnement
et la fermeture ; pour WEB, il vérifie le protocole HTTP et les images BMP.
Il ne lance pas de navigateur automatiquement.

Le fonctionnement des panneaux et du monitoring est décrit dans
[Monitoring, htop et visualisation](../04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md).

## Pilotage par script et édition d'image

`Mimir.Viz.backend()` identifie le backend actif ; **H** ouvre l'aide avec cette
information et les raccourcis regroupés. Les quatre interfaces partagent les
panneaux et boutons configurables avec `Viz.configure` et la file
`Viz.poll_events`. Voir [l'API des scènes Lua et l'éditeur VAEConv](../03-API-Reference/15-Viz-Htop.md#scènes-et-événements-pilotés-en-lua).
