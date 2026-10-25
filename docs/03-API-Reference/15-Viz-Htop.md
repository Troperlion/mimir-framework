# Monitoring et visualisation

Trouver rapidement le contrat API réel et les paramètres utilisables.

## Vue d'ensemble

Le monitoring Mímir repose sur 2 interfaces complémentaires :

- `HtopDisplay` : rendu terminal temps réel (métriques, progression, gradients, mémoire, logs).
- `Visualizer` (SFML, Qt, GTK ou Web au choix à la compilation) : rendu graphique des images/tensors intermédiaires, panels interactifs et réglages live.

En pratique, les deux passent par `AsyncMonitor`, qui met à jour l'UI sans bloquer l'entraînement.

Les options CLI communes sont indépendantes :

```bash
./bin/mimir --lua script.lua -- --viz          # Viz uniquement
./bin/mimir --lua script.lua -- --htop         # Htop uniquement
./bin/mimir --lua script.lua -- --viz --htop   # Viz et Htop
```

Avec le flag `--htop`, l'interface terminal est préparée immédiatement mais
n'entre en écran alternatif qu'à la première métrique du run. Les messages de
configuration, le chargement du modèle et les instructions de lancement restent
ainsi visibles jusqu'à la fin de l'initialisation. Viz démarre immédiatement.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.

## `Mimir.Htop`

- `create()`
- `update()`
- `render()`
- `clear()`
- `enable(bool)`

État réel des méthodes de cycle de vie : `create()` démarre `AsyncMonitor`,
`render()` est conservée pour compatibilité mais le rendu est automatique, et
`enable(false)` arrête le monitor. `enable(true)` ne redémarre pas un monitor
arrêté : il faut rappeler `create()`.

L'export CSV est unique et appartient au thread d’écriture d’`AsyncMonitor`,
indépendamment de htop et de la Viz. Fermer la fenêtre Viz laisse l’entraînement
et l’écriture CSV actifs ; htop peut la rouvrir. Le CSV existant est relu au
démarrage pour restaurer le graphe, puis les nouvelles métriques sont ajoutées
sans effacer l’historique. Les options `csv`, `csv_enabled`, `csv_path` et
`csv_file` de `Htop.create()` configurent cet export commun.

### Graphe du modèle dans HTOP

Pendant un run, `G` ouvre ou ferme l'arbre nodal du modèle et du planner actif.
La vue utilise le snapshot immuable publié par la configuration runtime ; elle
n'inspecte pas directement les layers en cours d'exécution. Les nœuds indiquent
leurs dépendances, runtime, capacité, fusion et tailles disponibles. Le panneau
signale explicitement quand le planner n'a fourni que la topologie, et les
placements ne représentent pas une trace GPU.

Dans cette vue, `↑`/`↓` ou `Tab` sélectionnent un layer, `Page Up`/`Page Down`
déplacent la sélection par page, `←`/`→` déplacent la vue horizontalement et
`R` réinitialise sélection et position. `G` ou `Esc` ferme le graphe et reprend
l'affichage HTOP.

### Usage recommandé

1. Créer/activer le monitor en début de run.
2. Pousser des `Metrics` régulièrement (`updateMetrics`).
3. Laisser le rendu asynchrone faire l'affichage.

### Ce que vous voyez dans le terminal

- progression epoch/batch,
- loss courante + moyenne,
- type de `recon_loss` réellement utilisé (ex: `mse`, `l1`, `bce_logits`),
- composantes (KL, wasserstein, etc.),
- gradients, mémoire, ETA, logs.

Notes :

- Le label de la métrique de reconstruction est dynamique et suit `recon_loss_type` quand fourni.
- Si `recon_loss_type` est absent, l'affichage retombe sur un label générique `RECON`.
- `C` ouvre ou ferme la section de configuration du modèle. Les flèches sélectionnent
  un paramètre ; `Entrée` commence l'édition puis valide la nouvelle valeur.
- Les lignes `DIRECT` sont appliquées au prochain point de synchronisation du calcul.
  Les lignes `FIXE` restent consultables mais nécessitent une reconstruction ou un redémarrage.
- `Tab` ou `↑`/`↓` sélectionne `RECON`, puis `←`/`→` ou `-`/`+` change la loss pendant l'entraînement.
- `R` restaure tous les paramètres live, dont la reconstruction loss, à leur valeur native.

## `Mimir.Viz`

Le choix de l’hôte graphique conserve cette API. Voir [Configurer le backend Viz](../02-User-Guide/17-Viz-Backends.md) pour les dépendances, commandes et limites.

- `create()`
- `initialize()`
- `is_open()`
- `process_events()`
- `update()`
- `add_image(...)`
- `set_detections(detections, target)`
- `show_detections(bool)`
- `update_metrics(...)`
- `add_loss_point(...)`
- `clear()`
- `set_enabled(bool)`
- `save_loss_history(path)`
- `backend()`
- `configure(scene)`
- `poll_events(timeout_ms)`
- `set_validation(state)`
- `validation_enabled()`

`initialize()` vérifie que la fenêtre existe et est ouverte. `process_events()`
et `update()` sont des fonctions de compatibilité : `AsyncMonitor` exécute déjà
ces opérations sur le thread Viz. De même, `add_loss_point()` est un no-op ;
`update_metrics()` alimente l'historique. `set_enabled(false)` arrête le monitor,
mais `set_enabled(true)` ne le redémarre pas : utiliser `create()`.

### Démarrage / activation

- En script/JSON : active la visualisation via la config du run (`visualization.enabled=true`).
- En C++ : instancie `Visualizer`, puis `initialize()`, ensuite boucle `process_events()` + `update()`.

### Contrôles clavier (UI)

- `H` : aide overlay.
- `C` : ouvre ou ferme la section de configuration du modèle.
- `M` : bascule mode de rendu Blocks/Layers (`HEATMAP` / `REEL`).
- `K` : change la palette heatmap (`CLASSIC` -> `TURBO` -> `INFERNO` -> `VIRIDIS`).
- `A` : active/désactive le lissage des previews de blocks.
- `B` : affiche/masque les encadrés de détection (`BOX ON/OFF`).
- `R` : resynchronise/rebuild textures + reload hints architecture.
- `N` : ouvre le graphe nodal du modèle ; `N` ou `Esc` le ferme.
- Dans le graphe : flèches ou molette pour naviguer, clic sur un nœud pour
  afficher ses détails, `R` pour revenir à l’origine et `X` pour exporter le
  graphe complet en PNG.
- `E` : ouvre la boîte de sélection du snapshot PNG.
- `Tab`, `F1..F5`, `←/→` : navigation focus/éléments.
- `Z`/`Entrée` : zoom, `Esc` : quitter zoom.
- Survol d’un tip zoomé : bulle des valeurs brutes et des indices du tenseur,
  jusqu’à 16 canaux échantillonnés. Les coordonnées suivent le redimensionnement
  de la vignette ; la valeur affichée précède la colorisation. Si le tip ne
  fournit pas de valeurs brutes, la bulle indique explicitement les pixels uint8.
- Bouton `PAUSE training` / `PLAY / Reprendre` dans Metrics : suspend/reprend
  le calcul entre étapes, tout en laissant les interfaces actives. Fermer la
  Viz pendant la pause libère l’entraînement ; `STOP` reste disponible.

Pour Lumen, les tokens et l’encodage affichés viennent de `text_ids` et de
`lumen/text/context` du dernier forward. Les vignettes de projection et de
compréhension suivent les couches texte capturées avec ce contexte.

### Encadrés de détection

`Mimir.Viz.set_detections()` reçoit des boîtes normalisées dans `[0,1]`, avec
un label, un score et un identifiant de classe optionnels. La cible vaut
`dataset` par défaut, ou `output`/`both`. Un tableau vide efface l'overlay.

```lua
Mimir.Viz.set_detections({
  {x1=0.12, y1=0.18, x2=0.64, y2=0.81, label="person", score=0.94, class_id=0}
}, "dataset")
```

Le badge `BOX ON/OFF` du panneau Context, le raccourci `B` et
`Mimir.Viz.show_detections(bool)` permettent de masquer ou réafficher les
encadrés sans perdre les prédictions publiées. Les architectures YOLO, SSD et
R-CNN décrivent encore un décodage externe : leur décodeur doit transmettre les
boîtes finales après NMS avec cette API.

Dans la section de configuration, les flèches ou la molette sélectionnent une
ligne. `Entrée` ou un clic commence l'édition, puis `Entrée` valide la valeur ;
`Esc` annule. Les paramètres `DIRECT` sont appliqués à la prochaine frontière de
calcul sûre, tandis que les paramètres `FIXE` indiquent qu'une reconstruction ou
un redémarrage est nécessaire.

Le panneau est alimenté automatiquement quand un modèle courant est utilisé par
`Model.train()`, `Model.forward()` ou `Model.optimizer_step()`. Il n'existe pas de
fonction Lua séparée pour écrire une ligne du panneau. Les paramètres `DIRECT`
réellement publiés dépendent du modèle et de l'optimiseur :

- entraînement : `learning_rate`, `lr_warmup_steps`, `validation_enabled` ;
- modèle : `grad_clip_norm`, skip connections, dropout, cache KV, latent
  stochastique et seuils NMS quand les couches correspondantes existent ;
- VAE : `kl_beta`, `kl_warmup_steps`, `recon_loss`, poids de loss et constantes
  Huber/Charbonnier/Gaussian NLL ;
- optimiseur courant : `beta1`, `beta2`, `epsilon`, `weight_decay` et
  `rmsprop_alpha`.

Toutes les autres clés de configuration sont affichées en `FIXE`. Modifier une
dimension, un nombre de couches ou une forme de tenseur exige de reconstruire le
modèle ; le panneau ne réalloue ni les poids ni l'état de l'optimiseur.

### Badges et état live

Le panneau `Blocks / Layers` affiche des badges en en-tête :

- mode de rendu courant (`HEATMAP` ou `REEL`),
- état du lissage (`LISSAGE ON/OFF`),
- palette active (`PAL CLASSIC/TURBO/INFERNO/VIRIDIS`).

Comportement de rendu important :

- Les sorties de packing (`out_concat`, `out_pack`, etc.) sont exclues des previews image.
- Dans la section `Outputs`, `recon` est priorisé juste avant `diff/resdiff` pour comparaison directe.
- En mode `HEATMAP`, la palette choisie recolorise les tips de `Blocks / Layers`,
  y compris les activations autres que `resdiff_abs`. Les layers `Identity`
  restent non recolorisés ; le mode `REEL` montre les couleurs naturelles.
- Les sections de tips sont regroupées par catégorie (par exemple `Encoder`,
  `Latent`, `Decoder`, `Outputs`). La section `Outputs` ne reçoit pas de tip
  artificiel `outputs` en tête : seuls les tips issus du modèle y apparaissent.

### Graphe nodal et export PNG

`N` affiche une vue arborescente du snapshot de graphe/planner associé au
modèle actif. Les nœuds sont placés par profondeur et par voie ; les arêtes
orthogonales entrent et sortent des cartes par des ports distincts. Les
embranchements et les connexions longues utilisent des rails séparés afin que
les trajets passent entre les cartes plutôt que sous elles. La sélection d'un
nœud affiche ses entrées/sortie, le runtime, les paramètres, les formes et les
informations de fusion. Les placements et octets sont des estimations du
planner, pas une trace garantie des kernels effectivement exécutés.

Dans cette vue, `X` exporte le graphe entier dans
`snapshots/model_graph_<horodatage>.png`. Le rendu est assemblé par tuiles pour
éviter la limite de dimension d'une texture de rendu unique ; l'image finale
conserve les dimensions complètes de la topologie.

`E` ouvre une boîte de dialogue permettant de choisir le contenu du snapshot :
fenêtre actuelle, tips des couches, projection et sortie. Les éléments absents
sont signalés comme indisponibles. Les tips sont ajoutés sous la capture de la
fenêtre, groupés par catégorie et accompagnés de leurs labels complets, renvoyés
à la ligne. L'export est écrit dans `snapshots/viz_snapshot_<horodatage>.png`.

### Annotations sur les sorties et correspondances de features

Quand l’image consommée possède les annotations JSON de l’éditeur, l’input et les taps spatiaux de sortie montrent les bbox, les relations entre leurs centres, les points et segments du squelette et ses attaches aux entités. Une même couleur identifie une entité d’un tap à l’autre ; les relations sont jaunes et les attaches aux entités roses. Les identifiants, noms et prédicats figurent dans les informations du tap. Les zones renforcées par la pondération spatiale reçoivent une teinte orange. Les vecteurs de packing et les matrices textuelles restent exclus.

Après `resdiff_abs`, dans la même section, deux taps sont ajoutés :

- `annotation_features` : chaque région bbox fournit un prototype moyen des features spatiales capturées. La couleur indique le prototype le plus similaire au vecteur local, et sa luminosité la similarité cosinus. Les informations indiquent la couche source, le nombre de canaux échantillonnés, les similarités entre régions liées et entre points de squelette et entités. Les valeurs inspectables sont ces similarités ; les valeurs brutes des taps source restent intactes.
- `annotation_impact` : poids régionaux de la loss rasterisés sur la grille du tap. `1` est neutre, orange signifie un renforcement, bleu une réduction. La géométrie utilise les mêmes coordonnées normalisées que l’input et les sorties. Le poids courant et les poids de rôles sont capturés avec l’annotation.

Ces vues montrent des correspondances de features et la pondération spatiale ; elles ne mesurent pas l’effet causal d’une annotation sur la prédiction. Sans features brutes alignées, `annotation_features` affiche uniquement la géométrie et indique explicitement que les features sont indisponibles. Les annotations de l’image et les taps sont publiés ensemble comme un snapshot ; aucun calcul de pooling ou de rasterisation ne conserve le verrou de publication. Les limites manuelles du nombre de taps continuent à s’appliquer.

### Progression Training

Le panneau `Training` affiche 2 barres distinctes :

- barre globale : avancement total du run (epochs + batch courant),
- barre batch : avancement du batch courant (sous la barre globale).

Détails :

- les deux barres utilisent un lissage visuel pour limiter les sauts,
- la barre batch est colorée selon `batch_time_ms` (rapide=vert, plus lent=orange/rouge),
- en absence de total epochs, la barre globale passe en mode fallback `running`.

### Sliders live (panneau Metrics)

Les sliders live supportent :

- drag souris sur le track/thumb,
- graduation visuelle (repères min/max),
- saisie directe via cellule de valeur (clic).

La ligne `Recon` est un sélecteur cyclique cliquable. Elle applique immédiatement l'une des losses supportées : `mse`, `mae`, `huber`, `charbonnier`, `gaussian_nll` ou `bce`.

Formats de saisie acceptés :

- décimal : `0.00025`, `0.5`, `1.0`
- scientifique : `1e-4`, `2.5e-3`, `5e4`

Validation édition :

- `Entrée` : applique,
- `Backspace` : efface,
- `Esc` : annule édition.

### Bonnes pratiques

- Garder `visualization.update_interval_ms` autour de `16..33` ms pour un bon compromis fluidité/coût.
- Utiliser la resync (`R`) seulement quand nécessaire (debug), pas en continu.
- Si vous envoyez des taps volumineux, réduisez la fréquence (`viz_taps_every_steps`) pour limiter la charge.

Notes :

- Le backend est choisi à la compilation : `SFML`, `QT`, `GTK`, `WEB` ou `NONE`.
- `Viz.backend()` retourne ce choix même avant `Viz.create()`.
- Le runtime peut publier des “viz taps” pendant `Model.forward()` si un monitor async est actif.

## Étapes suivantes

- [Page précédente : API : mémoire](14-Memory.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Serialization`](16-Serialization.md)

## Scènes et événements pilotés en Lua

Les backends **SFML, QT, GTK et WEB** partagent `Viz.configure`,
`Viz.poll_events` et `Viz.backend`. `backend()` retourne le backend compilé,
y compris avant la création ; `NONE` ne permet pas de créer une interface.
L'aide **H** affiche le backend, le moteur de rendu, les raccourcis groupés
et le texte d'aide fourni par le script.

```lua
assert(Mimir.Viz.create({visualization={enabled=true}}))
local UI = dofile(ROOTWORK.."/scripts/modules/viz_ui.lua")
local ui = UI.new({
  panels={{id=2, title="Aperçu", x=20,y=100,w=800,h=500,visible=true}},
  controls={{id="apply",label="Appliquer",x=20,y=20,w=160,h=40}},
  help="Appliquer : recalculer la prévisualisation.",
  events=true,
})
ui:on("apply", function(event) print("Appliquer",event.x,event.y) end)
while Mimir.Viz.is_open() do ui:dispatch(30) end
Mimir.Viz.set_enabled(false)
```

| ID panneau | Contenu |
| --- | --- |
| 0 | Context |
| 1 | Blocks / Layers |
| 2 | Generated |
| 3 | Training |
| 4 | Metrics |
| 5 | Graph |
| 6 | Output |

`configure(scene)` valide les données et retourne `true` ou `false, erreur`.
Les champs omis restent inchangés. `panels` modifie les panneaux indiqués ;
`controls` remplace la liste de boutons (table vide pour les retirer).
`image_size` règle la taille des aperçus Generated (32 à 1024 pixels, défaut 200).
Maximum : 64 entrées par liste. Les coordonnées sont en pixels dans la fenêtre ;
le script peut réagir à `resize` pour adapter son placement. Les contrôles sont
rendus au-dessus des panneaux, sous les overlays d'aide et de zoom. Un clic sur
un bouton est consommé, sans déclencher le panneau situé dessous.

`poll_events(timeout_ms)` retire les événements disponibles, avec une attente
optionnelle de 0 à 1000 ms. Types : `configured`, `click` (`id`, `x`, `y`),
`close`, `resize` (`width`, `height`). Avec `events=true`, s'ajoutent `key`
(`code` enum clavier Viz, `control`, `shift`, `alt`), `text` (`unicode`),
`pointer_move`, `pointer_down`, `pointer_up` (`x`, `y`, et `button` pour les deux
derniers). Les événements bruts observent les entrées sans désactiver les
raccourcis intégrés. La file est bornée à 256 entrées : les plus anciennes sont
supprimées en cas de saturation. Les clics ne rejouent pas automatiquement un
callback ; `viz_ui.lua` les distribue pendant `dispatch`, sur le thread Lua.
Les erreurs des callbacks remontent au script.

Les configurations sont transmises par une boîte aux lettres protégée ;
plusieurs mises à jour non encore appliquées fusionnent leurs champs de premier
niveau. Les modifications de panneaux fusionnent par ID ; la dernière liste
de boutons remplace la précédente. L'événement
`configured` confirme leur application par le thread de rendu. Les contrôles
restent des boutons rectangulaires ; cette API n'est pas un constructeur de
widgets Qt/GTK natifs.

### Éditeur d'image VAEConv

```bash
./bin/mimir --lua scripts/inferences/edit_vae_conv.lua -- \
  --checkpoint checkpoint/vae/epoch_0010 \
  --input photo.ppm --output outputs/edition.ppm
```

Le script accepte un checkpoint **raw_folder** `vae_conv` entraîné et une image
PNG/JPEG/BMP ou PPM P3/P6 RGB. La géométrie provient du checkpoint.
Les formats PNG/JPEG/BMP utilisent le chargement RGB natif Mímir avec
redimensionnement bicubique ; le PPM utilise le voisin le plus proche si nécessaire. Il encode la moyenne latente, charge
`vae_conv_decode`, puis permet de décaler un canal latent, mélanger le résultat
avec la source, réinitialiser et enregistrer en PPM. La source et le dernier
résultat sont présentés côte à côte ; `Z` agrandit l'image sélectionnée.
`L` bascule vers une galerie scrollable de tous les canaux du latent courant,
normalisés indépendamment pour rendre leur structure spatiale inspectable.
`Ctrl+U` permet de choisir une autre image puis recalcule son latent sans fermer
l'éditeur. Le script utilise un sélecteur système (`zenity`, `kdialog` ou `yad`)
s'il est disponible ; sinon, le chemin peut être saisi directement dans la Viz
et validé avec `Entrée`.

Ces modifications latentes ne sont pas des commandes sémantiques et leur effet
dépend de l'entraînement. Le décodeur autonome utilise des skips encodeur nuls
si le modèle comporte des connexions de ce type ; sa sortie à force zéro peut
donc différer d'une reconstruction par le VAE complet. `--mix 0` conserve
l'image source redimensionnée. L'interface ne sauvegarde pas à la fermeture :
utiliser **Enregistrer**. Les fichiers existants au chemin de sortie sont
remplacés à chaque enregistrement.

`--headless --strength 0.2 --channel 1 --mix 0.8` calcule et enregistre une fois,
sans interface. `--help` détaille les options sans charger de modèle. Le script
vérifie la présence et la taille des poids requis avant de charger le décodeur
partiel ; il ne modifie ni les poids ni le checkpoint.
