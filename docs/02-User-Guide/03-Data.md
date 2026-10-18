# Données et datasets

Préparer un dataset compatible et comprendre comment il est lu.

**Public concerné :** Débutant à intermédiaire.

> **Prérequis**
>
> Avoir un dossier de données local.


Cette page décrit le comportement **réel** du loader de dataset actuellement exposé à Lua via `Mimir.Dataset`, et ses limitations.

Sources principales :

- API Lua: `src/scriptings/Lua/luaScripting/LuaScripting.cpp` (`lua_loadDataset`, `lua_getDataset`, `lua_prepareSequences`)
- Indexation + lazy-loading: `src/Helpers.hpp` (`loadDataset`, `DatasetItem`, `DatasetMemoryManager`, `DatasetManager`)

## Sur cette page

- [Vue d’ensemble](#vue-densemble)
- [Format disque et règle de “linking”](#format-disque-et-règle-de-linking)
- [API Lua: Mimir.Dataset](#api-lua-mimirdataset)
- [Détails utiles (mémoire et lazy-loading)](#détails-utiles-mémoire-et-lazy-loading)
- [Exemples d’usage](#exemples-dusage)
- [Bonnes pratiques](#bonnes-pratiques)
- [Limites actuelles (et implications)](#limites-actuelles-et-implications)
- [Étapes suivantes](#étapes-suivantes)

## Vue d’ensemble

- `Mimir.Dataset.load(dir)` indexe récursivement un dossier et construit une liste d’items (métadonnées uniquement).
- `Mimir.Dataset.get(i)` charge le texte et les pixels RGB redimensionnés à la demande.
- `Mimir.Dataset.get(i, false)` retourne texte, chemins et dimensions sans décoder l'image.
- `Mimir.Dataset.release(i)` libère les buffers lazy de l'item après utilisation.
- `Mimir.Dataset.prepare_sequences(seq_len)` construit des séquences de tokens à partir des fichiers texte du dataset (tokenizer requis pour que des séquences soient effectivement produites).

## Format disque et règle de “linking”

L’indexation fonctionne par **chemin relatif sans extension** :

- Tous les fichiers sont parcourus récursivement.
- Ils sont groupés par leur chemin relatif au dataset, sans extension (`portraits/photo`, par exemple).
- Un item peut donc regrouper plusieurs modalités si elles partagent le même basename.

Exemple :

```text
dataset/
  0001.txt
  0001.png
  0002.txt
  0003.jpg
```

Ici, `0001` est “linké” (texte + image). `0002` est texte seul. `0003` est image seule.

Extensions reconnues (actuellement):

- Images: `.png .jpg .jpeg .bmp .tiff .webp`
- Texte: `.txt .md .json .csv`
- Audio: `.wav .flac .mp3 .ogg .m4a .aac`
- Vidéo: `.mp4 .mkv .avi .mov .webm .flv .ts`

### Points d’attention importants

- Deux fichiers `portraits/foo.txt` et `animals/foo.txt` produisent des items distincts. Les chemins lexicaux préservent les associations avec les images symboliques.
- Un seul fichier par modalité : plusieurs images de même chemin sans extension sont départagées par extension (PNG en premier), puis par chemin. L’éditeur d’annotation demande de renommer les collisions.
- Le seuil `min_modalities` est configurable dans `Mimir.Dataset.load` et vaut 1 par défaut.

## Annotations JSON de l’éditeur Lumen

Pour une image `portraits/photo.jpg`, le fichier `portraits/photo.json` peut contenir le document produit par [l’atelier d’annotation](../../annotator/README.md). Le loader reconnaît son objet `captions` et valide la description principale, le nom de l’image, les identifiants d’entités, les références et les boîtes normalisées.

```json
{
  "image": "photo.jpg",
  "captions": {
    "primary": "Une femme se tient près d’une fenêtre.",
    "alternatives": []
  },
  "entities": [
    {"id": "person_1", "phrase": "une femme", "role": "primary_subject", "bbox": [0.1, 0.2, 0.5, 0.9]}
  ],
  "attributes": [],
  "relations": [],
  "lighting": {"description": "", "source": "", "target": ""},
  "composition": {"description": ""},
  "quality": {"caption_verified": false, "relations_verified": false}
}
```

**Priorité :** lorsque cette annotation est présente, `DatasetItem::loadText()` et `Mimir.Dataset.get(i).text` fournissent exclusivement `captions.primary`, même si un ancien `photo.txt` existe. Les captions alternatives, les entités et les boîtes ne sont pas concaténées au texte de conditionnement. En l’absence d’annotation structurée, le loader conserve la sélection historique : `.txt`, puis `.md`, JSON générique, puis `.csv`. Les captions COCO conservent leur chemin historique, avec la même priorité pour un JSON structuré local.

Une annotation invalide provoque une erreur explicite au chargement, sans remplacement discret par le `.txt`. Un JSON générique sans objet `captions` garde son traitement historique. Les sections structurées sont facultatives pour les annotations ne contenant qu’une caption ; les champs supplémentaires sont conservés. Les cases `quality` ne filtrent pas automatiquement les exemples.

Le document complet est disponible sous `item.annotation`, son chemin sous `item.annotation_file`. Le contenu est chargé à la demande et libéré par `Mimir.Dataset.release(i)`. Les boîtes restent normalisées dans le document après redimensionnement des pixels.

```lua
local ok, n = Mimir.Dataset.load("/chemin/dataset", 64, 64, 2)
assert(ok, n)
for i = 1, n do
  local item, err = Mimir.Dataset.get(i, false)
  assert(item, err)
  print(item.text) -- captions.primary ou description historique
  if item.annotation then
    print(item.annotation.captions.primary)
    for _, entity in ipairs(item.annotation.entities or {}) do
      print(entity.id, entity.phrase, entity.bbox and entity.bbox[1])
    end
  end
  assert(Mimir.Dataset.release(i))
end
```

Les entraînements VAE avec conditionnement texte, VAEText et DiT/Lumen qui consomment déjà `loadText()` ou `item.text` reçoivent cette description sans conversion préalable en `.txt`. Les entraînements VAE image, avec ou sans texte, utilisent automatiquement les `entities[].bbox` pour pondérer la reconstruction. DiT/Lumen les applique à la prédiction du bruit sur sa grille latente, en entraînement et en validation. Chaque boîte contribue à parts égales, quelle que soit sa taille ; son recouvrement exact avec les cellules préserve aussi les petites boîtes. Les masques de segmentation existants sont combinés avec cette pondération.

Avec `bbox_loss_weight = 1` (défaut), les zones annotées contribuent à la loss en plus de l’image globale. En présence de relations, la pondération régionale inclut aussi le contexte entre les centres des boîtes liées. Chaque relation dont les deux entités possèdent une bbox ajoute une perte différentiable : elle compare les différences d’apparence moyenne et de moments spatiaux entre les régions prédites et les régions cibles. Ses gradients couplent les deux régions. `left_of/right_of` sélectionnent le moment horizontal, `above/below` le moment vertical ; les autres prédicats utilisent les deux axes. Cette perte s’applique aux sorties spatiales et au bruit latent de Lumen ; les classifieurs à sortie globale utilisent la pondération du contexte en entrée.

Pour les modèles vision conditionnés par du texte, le contexte natif contient les identifiants et descriptions des entités, leurs rôles, les coordonnées des bbox, les squelettes, leurs attaches et leurs relations, puis les attributs, l’éclairage, la composition et la description principale. Le texte brut du dataset et de VIZ reste intact. Ce mécanisme fournit une supervision visuelle et textuelle ; il ne garantit pas à lui seul une compréhension sémantique générale. `bbox_loss_weight = 0` désactive la supervision spatiale ; le contexte reste transmis au modèle. Sans annotation, le comportement historique est conservé.

Les relations sont pondérées par le rôle de leurs deux entités : poids moyen `(poids_sujet + poids_objet) / 2`. Les valeurs par défaut sont `relation_primary_weight = 3`, `relation_secondary_weight = 2`, `relation_other_weight = 1`. `primary_subject`/`secondary_subject` et les variantes `primary_subjet`/`secondary_subjet` sont reconnus ; les rôles absents ou autres (`object`, `background`, etc.) utilisent le troisième poids. Cette importance multiplie la perte relationnelle et ses gradients, et pondère le contexte entre régions. Les bbox individuelles gardent leur pondération existante. Une relation dont les deux rôles valent zéro ne contribue plus aux contraintes visuelles.

Les scripts VAEConv, Lumen et détecteur médical acceptent `--relation-primary-weight 3 --relation-secondary-weight 2 --relation-other-weight 1`. Les mêmes paramètres sont éditables à chaud dans le panneau `C` de VIZ et htop, puis appliqués par le thread d’entraînement au prochain pas. Les valeurs doivent être finies et positives ou nulles.

Pour appeler directement Lumen, transmettre `{annotation = item.annotation}` comme sixième argument de `lumen_train_step` ou quatrième argument de `lumen_validate_step`. Une option `bbox_loss_weight` explicite remplace la valeur courante. Les statistiques `bbox_count`, `bbox_loss_weight`, `relation_count` et `relation_loss` exposent la supervision utilisée ; `mse_unweighted` conserve la MSE globale.

Les squelettes personnalisés de l’éditeur sont reconnus dans `entities[].skeleton` : `name`, `points` (`name`, `position: [x,y]` ou `null`, `parent` optionnel, `entities` optionnel) et `edges` optionnel (`from`, `to`, `label`). Les noms de points sont uniques par squelette ; les liaisons et identifiants d’entités sont validés au chargement. Branches et cycles sont autorisés. Les positions sont normalisées sur l’image entière, et les points non placés ne contribuent pas à la pondération spatiale.

Lumen et les VAE conditionnés utilisent `Mimir.Model.conditioning_prompt(caption, annotation)`, construit par la même fonction C++ que leur conditionnement natif. Les points nommés et les liaisons libres sont associés au squelette propriétaire, identifié par l’entité ; les associations de points portent aussi les identifiants des entités cibles. Lumen et la boucle VAE conditionnée hydratent systématiquement le tokenizer avant les steps : `ensure_vocab_from_text`/`ensureVocabFromText` sur chaque contexte complet, puis `learn_bpe`/`learnBPEFromCorpus` une fois sur le corpus. Le tokenizer est ensuite transmis au modèle et sauvegardé avec le checkpoint. Le script Lumen ne reconstruit pas le format de contexte en Lua. Le contexte est tokenisé puis transformé en embeddings par l’encodeur textuel. Dans Lumen, ses embeddings alimentent la cross-attention du DiT. Le contexte annoté complet doit tenir dans `text_seq_len` : Lumen refuse une troncature et le script indique la valeur nécessaire de `--text-seq-len`. Les entraînements VAE conditionnés vérifient de même `seq_len`. La loss de diffusion met l’accent sur les points placés, les segments parent/enfant et libres, et les connexions vers les centres des boîtes des entités associées. Chaque segment est échantillonné avec un pas maximal de 0,025 en coordonnées normalisées ; chaque échantillon fournit une région de largeur 0,05, tronquée aux bords. Ces régions partagent la moyenne régionale avec les bbox et conservent une pondération globale de moyenne un. Ce mécanisme utilise la loss et ses gradients existants, sans ajouter de tête de pose ni garantir la précision des articulations générées. Il fonctionne aussi sans bbox. `skeleton_region_count` expose le nombre de régions de squelette rasterisées dans les statistiques Lumen ; `--bbox-loss-weight 0` désactive cette supervision, en conservant le conditionnement par les annotations. Les datasets sans annotation gardent leur comportement précédent.

Pour une inférence textuelle ou un appel avec des tokens explicites, construire le même contexte avant la tokenisation :

```lua
local context, err = Mimir.Model.conditioning_prompt(item.text, item.annotation)
assert(context, err)
-- Lumen reçoit automatiquement ce contexte lorsque annotation est passée
-- à lumen_train_step / lumen_validate_step. Pour la génération :
local pixels, width, height, channels = Mimir.Model.lumen_text2img(context, 123, 30, 4)
```

Le simulateur de l’éditeur n’alimente pas le modèle : seules les annotations sauvegardées fournissent ce contexte. Cette représentation rend les liens disponibles à l’apprentissage ; elle ne garantit pas à elle seule leur compréhension ou leur respect lors de la génération.

L’historique `.annotation_state.json`, son verrou et les fichiers temporaires de l’éditeur sont exclus de l’index et du hash du dataset. Le cache conserve le chemin du JSON, jamais son contenu lazy. Il est invalidé après modification des annotations, y compris une modification JSON de longueur identique, et tient compte des dimensions et du seuil de modalités demandés. Les anciens caches sont régénérés.

## API Lua: `Mimir.Dataset`

### `Mimir.Dataset.load(dataset_dir)`

Charge (indexe) le dataset.

- Entrée: `dataset_dir` (string)
- Sortie: `(ok, n_or_err)`
  - si `ok == true`, `n_or_err` est le nombre d’items indexés
  - si `ok == false`, `n_or_err` est un message d’erreur

Remarques:

- La fonction vérifie que le dossier existe.
- Elle appelle `loadDataset(dataset_dir)` côté C++.
- Les paramètres optionnels permettent de choisir les dimensions cibles, le seuil de modalités, le cache, la limite RAM et le chargement lazy.
- L’indexation est en mode “lazy”: les données (texte/image/audio/vidéo) ne sont pas chargées en RAM à ce stade.

### Vidéos comme dataset

Quand un item contient une vidéo, FFmpeg inspecte ses flux pendant l’indexation. Une piste audio intégrée est exposée comme source audio de l’item, sans extraction préalable sur disque.

Pour `vae_conv`, la vidéo est ouverte une fois par époque et décodée à la demande. Chaque frame RGB redimensionnée produit un pas d’entraînement, dans l’ordre temporel du média. Les frames d’une même vidéo restent contiguës même si l’ordre des items du dataset est mélangé entre les époques.

Ce chemin nécessite une compilation avec `ENABLE_FFMPEG=ON`. Si le dataset vise une vidéo alors que FFmpeg est absent, l’entraînement échoue avec un message explicite au lieu d’ignorer le média.

### `Mimir.Dataset.get(index, load_image?)`

Retourne un item du dataset (Lua est 1-indexed).

- Entrées: `index` dans `[1, num_items]`; `load_image` vaut `true` par défaut
- Sortie: `(item)` en cas de succès, ou `(nil, err)` en cas d’erreur

Champs possibles dans `item`:

- `text_file`, `image_file`, `audio_file`, `video_file`: chemins vers les fichiers (si présents)
- `width`, `height`, `channels`: dimensions cibles et canaux
- `text`: description naturelle principale si annotation JSON structurée, sinon contenu texte historique
- `annotation_file`, `annotation`: chemin et document structuré complet si présents
- `image`: pixels RGB u8 si `load_image` est vrai et le décodage réussit

Pour les grands datasets, appeler `Mimir.Dataset.release(index)` après avoir consommé
l'image. Cela évite que les buffers lazy s'accumulent jusqu'à la limite RAM configurée.

Si seul le texte est requis, utiliser `Mimir.Dataset.get(index, false)` pour éviter le
décodage et la construction de la table de pixels Lua.

### `Mimir.Dataset.prepare_sequences(seq_len)`

Prépare des séquences de tokens à partir des fichiers texte.

- Entrée: `seq_len` (integer)
- Sortie: `(ok, n_or_err)`
  - si `ok == true`, `n_or_err` est le nombre de séquences construites
  - si `ok == false`, `n_or_err` est un message d’erreur

Comportement:

- Nécessite qu’un dataset ait été chargé au préalable (via la config interne `dataset.dir`).
- Ré-indexe le dataset depuis `dataset.dir`.
- Pour chaque item: charge le texte à la demande (`DatasetItem::loadText()`), puis tokenise via le tokenizer courant (`ctx.currentTokenizer`).
- Padding/troncature:
  - si la séquence est plus courte que `seq_len`, elle est paddée avec `pad_id`
  - si elle est plus longue que `seq_len`, elle est tronquée

À savoir:

- Si aucun tokenizer n’est présent, la fonction ne génère pas de tokens (et peut retourner `ok=true` avec `0` séquences). Vérifie la valeur retournée.
- Les séquences sont stockées en interne (dans le contexte Lua/C++), pas retournées directement à Lua.

## Détails utiles (mémoire et lazy-loading)

Même si `Mimir.Dataset.load()` n’effectue que l’indexation, le C++ contient des loaders lazy (`DatasetItem::loadText/loadImage/loadAudio/loadVideo`) protégés par un petit gestionnaire RAM interne (`DatasetMemoryManager`).

Conséquences côté Lua:

- `prepare_sequences()` appelle `loadText()` en interne. Si un fichier texte est trop gros (ou si le gestionnaire RAM interne considère qu’il ne peut pas allouer), le texte ne sera pas chargé et l’item ne produira pas de séquence.
- Le gestionnaire RAM dataset a une limite par défaut de 10 GB dans `Helpers.hpp`, mais **elle n’est pas configurable via l’API Lua dataset actuelle**.

À retenir:

- Évite de mettre des fichiers texte énormes “bruts” dans le dataset; préfère des items plus petits (ou un pré-traitement hors runtime).
- Si vous observez `0` séquences préparées, vérifie d’abord: tokenizer présent, puis taille/qualité des fichiers texte.

## Exemples d’usage

### Dataset texte (préparation de séquences)

```lua
local ok_ds, n_or_err = Mimir.Dataset.load("datasets.old/text")
assert(ok_ds, n_or_err)

-- tokenizer doit exister (voir scripts/modules/base_tokenizer.lua, ou Mimir.Tokenizer)
local ok_seq, n_seq_or_err = Mimir.Dataset.prepare_sequences(512)
assert(ok_seq, n_seq_or_err)
print("sequences:", n_seq_or_err)
```

### Inspecter les items (chemins)

```lua
local ok_ds, n = Mimir.Dataset.load("checkpoints/llm_simple")
if ok_ds then

  local item, err = Mimir.Dataset.get(1)
  if item then
    print(item.text_file, item.image_file)
  else
    print("get failed:", err)
  end
end
```

## Bonnes pratiques

- Datasets multi-modaux: préfère des basenames uniques sur tout le dataset (évite les collisions entre sous-dossiers).
- Reproductibilité: loggue dans votre checkpoint les infos “pipeline” (chemin dataset, `seq_len`, vocab/tokenizer utilisé).
- Validation rapide: ajoute un script de smoke test qui fait `load()` puis `get(1)` et vérifie la présence des champs attendus.

## Limites actuelles (et implications)

- `Mimir.Dataset.get()` expose actuellement les pixels image RGB, mais pas les buffers audio ou vidéo.
- Le chargement image lazy reste explicite côté appelant: utiliser `release(index)` après consommation sur les grands datasets.

## Étapes suivantes

- [Page précédente : Workflow modèle (lifecycle)](02-Model-Lifecycle.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Entraînement](04-Training.md)


### Bbox pour les autres modèles de vision

Le contrat C++ `Model::TrainStepRequest.annotation` accepte le document structuré pour tous les modèles de vision natifs. UNet, les décodeurs VAE et le discriminateur par patches pondèrent la loss spatiale ; le VAE MLP conserve la pondération historique des statistiques latentes. ResNet, MobileNet, VGG16/VGG19 et VGG16Feat accentuent les régions de l’entrée : leurs classes ou caractéristiques globales ne représentent pas des pixels. Les boucles VGG existantes chargent aussi automatiquement les bbox du dataset, y compris en validation VGG.

Les détecteurs médicaux acceptent `entities[].bbox` comme vérités terrain normalisées et pondèrent leurs losses d’objectness, coordonnées et classes. `class_id` indique la classe entière ; son absence utilise la classe 0. Les identifiants et phrases d’entités ne sont pas convertis automatiquement en classes. Les anciens fichiers `boxes` restent compatibles.

```lua
local item = assert(Mimir.Dataset.get(index))
-- input et target : tensors normalisés et disposés selon le contrat du modèle.
local stats, err = Mimir.Model.vision_train_step(input, target, {
  annotation = item.annotation,
  learning_rate = 0.0001,
  optimizer = "adamw",
})
assert(stats, err)
print(stats.loss, stats.bbox_count, stats.bbox_loss_weight)
```

Cette API commune utilise la cible propre à la tâche ; elle ne fabrique pas de labels depuis une description. Les scripts d’entraînement personnalisés doivent lui transmettre `item.annotation`.

Pour ViT, définir `bbox_input_w` et `bbox_input_h` comme la grille de patches, `bbox_input_c` comme la dimension d’embedding et `bbox_input_layout = "HWC"`. Le token CLS initial reste inchangé ; `bbox_input_prefix_tokens = 0` convient aux entrées sans CLS. Pour un générateur latent (`gan_latent`), préciser la grille de sortie avec `bbox_output_w/h/c` et `bbox_output_layout`. Les autres dispositions personnalisées peuvent utiliser ces mêmes champs. Une géométrie absente ou incompatible provoque une erreur explicite lorsqu’une bbox est active. Sans bbox ou avec `bbox_loss_weight = 0`, le calcul historique est conservé.
