# Détection d'anomalies médicales

Mímir fournit l'architecture native `medical_detector` pour entraîner un détecteur spatial sur des images médicales annotées. Le CNN, la loss, l'optimiseur, la validation, le décodage et le NMS sont exécutés par Mímir.

> Ce modèle assiste le repérage. Il ne remplace pas l'interprétation d'un radiologue, une validation clinique indépendante ni les exigences réglementaires applicables.

## Composer le dataset

Pour un dataset organisé ainsi :

```text
donnees/mammo/
├── finding_annotations.csv
├── manifest.csv
├── images/
└── masks/
```

utilisez directement :

```bash
./bin/mimir --lua scripts/tools/compose_medical_detector_dataset.lua -- \
  --dataset-root donnees/mammo \
  --out-dir dataset/mammographies-multiclasse
```

La taxonomie par défaut est stable et partagée avec le script d'entraînement :

| `class_id` | Classe Mímir | Catégories VinDr associées |
|---:|---|---|
| 0 | `masse` | `Mass` |
| 1 | `calcification` | `Suspicious Calcification` |
| 2 | `asymétrie` | `Asymmetry`, `Focal Asymmetry`, `Global Asymmetry` |
| 3 | `distorsion architecturale` | `Architectural Distortion` |
| 4 | `ganglion lymphatique` | `Suspicious Lymph Node` |
| 5 | `épaississement cutané` | `Skin Thickening` |
| 6 | `rétraction du mamelon` | `Nipple Retraction` |

Les catégories VinDr qui ne figurent pas dans cette taxonomie sont exclues. Une image contenant uniquement une catégorie exclue n'est jamais convertie en examen négatif. Le bilan affiche le nombre de lignes `hors_taxonomie`, les images exclues et le nombre de boîtes par classe.

Le composeur joint les lignes par `image_id`, chemin, nom de fichier ou stem. Pour MVinDr, un fichier nommé `study_id_image_id.png` est aussi associé automatiquement à la colonne `image_id`. Les lignes du CSV global dont l'image n'appartient pas au sous-ensemble local sont comptées comme `hors_sous-ensemble` puis ignorées. Il reconnaît notamment `image_path` et `mask_path` dans `manifest.csv`, ainsi que `xmin`, `ymin`, `xmax`, `ymax`, `width`, `height` et `finding_categories` dans `finding_annotations.csv`. Les variantes usuelles comme `x_min`, `image_width` et `file_name` sont également acceptées.

Les coordonnées de `finding_annotations.csv` sont prioritaires. Lorsqu'elles sont absentes, le masque référencé par `manifest.csv` est chargé par Mímir et sa boîte englobante est calculée. Une ligne `No Finding`, `Normal`, une image manifestée sans boîte ou un masque vide produit un négatif `boxes: []`.

Utilisez `--mask-scan-size 1024` pour augmenter la précision du calcul depuis les masques. La valeur par défaut de 512 suffit généralement pour produire une boîte normalisée.

Le sous-ensemble « Mass Segmentation » ne contient que la classe `masse`; il ne suffit donc pas pour entraîner les sept classes. Utilisez les annotations de détection VinDr complètes pour obtenir les autres catégories. Pour limiter les faux positifs, placez les mammographies normales dans `normal_images/` à la racine du dataset. Le composeur détecte automatiquement ce dossier, copie ces images sous `normals/` et génère leurs annotations `boxes: []`. Les noms `negative_images/`, `normals/` et `negatives/` sont également reconnus. Pour un dossier externe, utilisez `--normal-images-root PATH`.

```text
donnees/mammo/
├── finding_annotations.csv
├── manifest.csv
├── images/
├── masks/
└── normal_images/
```

Seules des images dont l'absence de tumeur est confirmée doivent être placées dans ce dossier. Le composeur ne transforme jamais une mammographie tumorale complète en négatif et ne fabrique pas d'examen médical absent des sources.

Dataset_BUSI est un dataset d'échographies mammaires. Si sa racine est passée à `--normal-images-root`, le composeur sélectionne automatiquement `Dataset_BUSI_with_GT/normal` et exclut les fichiers `_mask`. Le mélange avec des mammographies est bloqué par défaut, car il introduit un biais de modalité important. L'option `--allow-cross-modality-negatives` permet de confirmer explicitement ce choix pour une expérimentation, mais elle n'est pas recommandée pour un modèle médical final.

### Manifeste JSON générique

Le composeur copie les images, génère les sidecars attendus et recharge le résultat avec Mímir pour le valider :

```bash
./bin/mimir --lua scripts/tools/compose_medical_detector_dataset.lua -- \
  --images-root donnees/images \
  --annotations donnees/annotations.json \
  --out-dir dataset/mammographies-multiclasse
```

Le manifeste regroupe toutes les annotations sources :

```json
{
  "classes": [
    "masse",
    "calcification",
    "asymétrie",
    "distorsion architecturale",
    "ganglion lymphatique",
    "épaississement cutané",
    "rétraction du mamelon"
  ],
  "images": [
    {
      "file": "patiente-001/examen.png",
      "normalized": true,
      "boxes": [
        {
          "x1": 0.20,
          "y1": 0.25,
          "x2": 0.70,
          "y2": 0.80,
          "class_id": 0,
          "label": "masse"
        }
      ]
    },
    {
      "file": "patiente-002/examen.png",
      "negative": true
    }
  ]
}
```

Sans `--annotations`, le composeur recherche les fichiers `.json` ou `.boxes.json` déjà placés à côté des images sources. Les images sans annotation sont ignorées. L'option `--unannotated-as-negative` permet de les inclure comme examens négatifs, mais elle ne doit être utilisée que si leur absence d'anomalie a été vérifiée.

Utilisez `--dry-run` pour contrôler le manifeste sans écrire de fichiers et `--overwrite` pour remplacer explicitement une sortie existante.

## Dataset

Chaque image doit avoir un sidecar portant le même nom, avec l'extension `.json` ou `.boxes.json`.

```text
dataset/mammographies/
├── examen_001.png
├── examen_001.json
├── examen_002.png
└── examen_002.json
```

Annotation en coordonnées pixels :

```json
{
  "width": 2048,
  "height": 2048,
  "normalized": false,
  "boxes": [
    {
      "x1": 420,
      "y1": 610,
      "x2": 770,
      "y2": 980,
      "class_id": 0,
      "label": "masse"
    }
  ]
}
```

Les formats `x, y, w, h` et les coordonnées normalisées entre 0 et 1 sont également acceptés. Un examen sans anomalie doit rester dans le dataset avec une liste vide :

```json
{
  "width": 2048,
  "height": 2048,
  "boxes": []
}
```

Une annotation absente n'est pas interprétée comme un négatif : l'image est ignorée et comptabilisée dans les logs.

## Entraînement

```bash
./bin/mimir --lua scripts/training/train_medical_detector.lua -- \
  --dataset-root dataset/mammographies-multiclasse \
  --out-dir checkpoint/medical_detector-multiclasse \
  --image-w 512 --image-h 512 \
  --base-channels 16 \
  --epochs 50 --lr 0.0001 \
  --validation-fraction 0.2 \
  --viz
```

Le holdout mesure précision, rappel, F1 et IoU moyen. Le checkpoint final est écrit dans `checkpoint/medical_detector-multiclasse/final`; les sauvegardes intermédiaires utilisent `epoch_NNNN`. Un ancien checkpoint mono-classe ne peut pas être converti en sept classes avec `--resume`, car la tête de détection n'a pas la même dimension : entraînez un nouveau checkpoint dans un autre dossier.

Dans Viz, le curseur `CONF` de la section Sortie filtre immédiatement les boîtes prédites selon leur probabilité, y compris pendant le zoom. Il démarre à 60 %. La piste et les boîtes suivent une rampe heatmap : bleu pour une faible confiance, puis cyan, vert, jaune et rouge pour les scores les plus élevés. Les boîtes cibles conservent leur couleur de classe et restent toujours affichées. Ce réglage est uniquement visuel : il ne modifie ni le seuil utilisé pour les métriques de validation, ni le modèle entraîné.

Le panneau Métriques affiche automatiquement les valeurs propres au détecteur : loss totale et moyenne, `object_loss`, `box_loss`, `class_loss`, cellules positives, normes de gradient, temps et débit des batches, mémoire, learning rate et état de l'optimiseur. Après chaque validation, il conserve précision, rappel, F1, IoU moyen ainsi que les comptes TP, FP et FN. Ces lignes n'apparaissent que pour `medical_detector`; les autres architectures conservent leur affichage habituel.

Pour reprendre :

```bash
./bin/mimir --lua scripts/training/train_medical_detector.lua -- \
  --dataset-root dataset/mammographies-multiclasse \
  --out-dir checkpoint/medical_detector-multiclasse \
  --resume --epochs 10
```

Lors d'une reprise, les noms de classes sont relus automatiquement depuis le checkpoint. Pour conserver volontairement l'ancien mode agrégé, passez explicitement `--class-names tumeur` lors de la composition et du premier entraînement.

## Inférence

```bash
./bin/mimir --lua scripts/inferences/detect_medical_anomalies.lua -- \
  --checkpoint checkpoint/medical_detector-multiclasse \
  --input examen.png \
  --output outputs/examen-detections.json \
  --score-threshold 0.25 \
  --viz
```

Le JSON de sortie contient les coordonnées normalisées et les coordonnées en pixels de l'image redimensionnée. Ajustez le seuil sur un jeu de validation séparé et conservez un jeu de test patient-indépendant pour l'évaluation finale.

Avec `--viz`, le curseur `CONF` démarre à la valeur de `--score-threshold`. Les candidats pré-seuil sont conservés dans Viz : déplacer le curseur vers la gauche peut donc révéler immédiatement des détections moins confiantes, sans relancer l'inférence. Le fichier JSON reste filtré par la valeur passée sur la ligne de commande.
