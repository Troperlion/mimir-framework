# Choisir l’interface Viz avant compilation

`Mimir.Viz` conserve son API Lua, ses panneaux, métriques, images, raccourcis,
réglages LIVE et captures PNG. Le choix de l’interface est exclusif et se fait
avec `MIMIR_VIZ_BACKEND` dans CMake, ou avec le menu de `./config.sh`.

| Valeur | Interface | Dépendances graphiques |
|---|---|---|
| `SFML` (défaut) | Fenêtre et primitives SFML | SFML 3 |
| `QT` | QMainWindow, QToolBar et QPainter | Qt 6 Widgets et Cairo |
| `GTK` | GtkWindow, GtkToolbar et Cairo | GTK 3, Cairo et pkg-config |
| `WEB` | HTML/CSS et Canvas2D dans le navigateur local | Cairo, navigateur avec JavaScript |
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
avec ceux de la Viz. Le bouton **Fermer** transmet la fermeture à Mímir ;
fermer simplement l’onglet laisse la Viz disponible pour une reconnexion. Plusieurs
onglets partagent le même état et les mêmes contrôles.

## Rendu et limites

Les panneaux partagent un thème d'éditeur sombre inspiré de Blender : surfaces
gris neutre, en-têtes compacts, séparateurs fins et accents de sélection. Les
contrôles du panneau des tenseurs se répartissent sur plusieurs lignes quand le
panneau devient étroit. Les aperçus s'adaptent à la surface disponible. Le
placement initial répartit contexte, sortie, tenseurs et validation à gauche,
entraînement, propriétés et courbes à droite. Les dispositions sauvegardées et
le déplacement/redimensionnement des panneaux restent disponibles.

Chaque backend dessine réellement les primitives avec sa technologie :

- SFML utilise `sf::RenderWindow`, ses formes, textes et textures ;
- Qt utilise un `QMainWindow`, une `QToolBar` et un widget qui dessine les
  rectangles, textes, courbes et images avec `QPainter` ;
- GTK utilise un `GtkWindow`, une `GtkToolbar` et un `GtkDrawingArea` dont le
  signal `draw` dessine la scène avec Cairo ;
- Web utilise une barre d'outils HTML/CSS et dessine la scène avec Canvas2D.

Qt/GTK/Web reçoivent les commandes de dessin et les textures PNG nécessaires,
sans framebuffer complet de l'interface. Le processus Mímir enregistre la
scène ; il ne rasterise cette interface que lors d'une capture explicite. Le protocole desktop reste interne à
une même compilation. Le Web expose la scène JSON sur `/JETON/frame` et son
module JavaScript sur `/JETON/renderer.js`. Les barres d'outils natives ouvrent
l'aide, les propriétés et le graphe, demandent une capture et actualisent les
textures via les mêmes actions que les raccourcis H/C/N/E/R.

Les builds Qt/GTK/Web n'utilisent pas SFML. Cairo reste utilisé dans le
processus Mímir pour les textures hors écran et les captures PNG, et pour le
rendu GTK. Qt/GTK nécessitent une session graphique ; Web fonctionne sans
`DISPLAY`, sans serveur graphique ni OpenGL. `NONE` reste le backend sans
moteur de dessin ni serveur d'interface.

Les hôtes alternatifs limitent la surface logique à 8192 pixels par côté et
33 554 432 pixels au total, et chaque scène transportée à 128 Mio. Les textures
sont partagées entre leurs occurrences dans une même scène. Les captures PNG
utilisent le chemin hors écran commun ; les métriques de police peuvent varier
légèrement par rapport à l'affichage natif Qt ou au navigateur. Le DPI du
bureau et le zoom du navigateur changent la taille physique à l'écran.

Les modules de rendu sont séparés :

- `src/viz/backends/SfmlVizWindow.cpp` : fenêtre et rendu SFML ;
- `src/viz/backends/QtDesktop.cpp` et `src/viz/renderers/QtScene.hpp` : Qt ;
- `src/viz/backends/GtkDesktop.cpp` et `src/viz/renderers/GtkScene.hpp` : GTK ;
- `src/viz/backends/WebPage.hpp`, `src/viz/renderers/WebScene.hpp` et
  `src/viz/WebHost.cpp` : interface, rendu et serveur Web ;
- `src/viz/SoftwareVizWindow.cpp` : enregistrement et transport des scènes
  pour Qt/GTK/Web, avec capture hors écran commune ;
- `src/viz/renderers/CairoScene.hpp` : primitives Cairo partagées par le
  signal GTK et les captures hors écran.

`Visualizer.cpp` ne sélectionne plus lui-même une bibliothèque GUI. Il construit
l'interface abstraite `VizWindow` via `createVizWindow()` ; CMake compile une
seule fabrique de backend, celle demandée par `MIMIR_VIZ_BACKEND`.

## Réactivité de Qt et GTK

GTK transmet les clics, mouvements, molettes, touches et événements de focus
au même flux Viz. Un clic rend le focus au canvas ; les boutons de la barre
d’outils émettent une pression puis un relâchement de touche, puis rendent le
focus au canvas. Qt conserve aussi le clavier sur le canvas, notamment pour
Tab, et traduit les actions de barre d’outils dans ce flux commun.

Les hôtes ne redessinent pas une scène inchangée à chaque tick. Les textures
stables réutilisent leur image encodée et les caches de décodage Qt/Cairo ; une
mise à jour invalide le contenu concerné, et les assets inutilisés sont évincés.
Les captures continuent de lire les pixels actualisés. Ces caches réduisent le
coût de rendu et de transport sans déplacer le calcul du modèle dans la GUI.

## Vérification reproductible

Une cible graphique explicite évite d’ajouter une dépendance à un écran aux tests
CTest habituels :

```bash
cmake --build build-qt --target mimir_viz_smoke -j2
xvfb-run -a python3 Tests/viz_backend_smoke.py \
  --backend QT --executable "$PWD/build-qt/bin/mimir_viz_smoke" \
  --output /tmp/mimir-viz-qt-results
```

Pour WEB, le test ci-dessus vérifie les captures locales, la scène JSON, le
protocole HTTP, les événements, le redimensionnement, les requêtes invalides et
la fermeture. Il ne lance pas de navigateur.

Pour vérifier les pixels effectivement rendus dans les fenêtres desktop, sans
Pillow ni xdotool (bibliothèques X11 et XTest requises), utiliser une session X11 avec XTest (ou Xvfb) :

```bash
python3 Tests/viz_native_window_smoke.py \
  --backend QT --executable /tmp/mimir-gui-qt/bin/mimir_viz_smoke
```

Adapter le backend et le chemin à la compilation locale. Ce test vérifie les
primitives, textes, textures, clips, clics, touches, barre d'outils, dimensions
après redimensionnement et fermeture. Le test historique desktop
`viz_backend_smoke.py` reste disponible avec Pillow et xdotool ; `xvfb-run`
requiert Xvfb et xauth.

Un contrôle du rendu Web dans un vrai navigateur utilise Node 22 et Chrome :

```bash
node Tests/viz_web_canvas_smoke.mjs /tmp/mimir-gui-web/bin/mimir_viz_smoke
```

`CHROME` permet de choisir un autre exécutable Chromium compatible. Le test
compare les pixels du Canvas2D et vérifie les textures, le clipping, les boutons
HTML, la souris, le clavier, le redimensionnement et la fermeture. Il produit
une capture `mimir-web-native.png` dans le dossier temporaire du système.

Le fonctionnement des panneaux et du monitoring est décrit dans
[Monitoring, htop et visualisation](../04-Architecture-Internals/04-Monitoring-Htop-Visualizer.md).

## Pilotage par script et édition d'image

`Mimir.Viz.backend()` identifie le backend actif ; **H** ouvre l'aide avec cette
information et les raccourcis regroupés. Les quatre interfaces partagent les
panneaux et boutons configurables avec `Viz.configure` et la file
`Viz.poll_events`. Voir [l'API des scènes Lua et l'éditeur VAEConv](../03-API-Reference/15-Viz-Htop.md#scènes-et-événements-pilotés-en-lua).
