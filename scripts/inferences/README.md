# Inférence native Lumen et VAEConv

Exécuter depuis la racine du dépôt, après compilation de `mimir` :

```sh
cmake --build build -j2 --target mimir
```

Les chemins ci-dessous sont des exemples à remplacer par vos checkpoints.
Chaque script accepte `--help` ou `-h`, sans charger de modèle.

## Lumen : texte vers image

```sh
./bin/mimir --lua scripts/inferences/lumen_text2img.lua -- \
  --checkpoint checkpoints/lumen_diffusion \
  --vae-checkpoint checkpoint/mon_vae \
  --prompt "un paysage de montagne au lever du soleil" \
  --seed 1337 --steps 30 --guidance 5 --out outputs/lumen.ppm
```

Le checkpoint Lumen doit être au format `raw_folder`. Le script récupère son
`model_config` et son `tokenizer/tokenizer.json`. Le vocabulaire est utilisé tel
quel, sans apprentissage pendant l'inférence. `--tokenizer` permet de préciser
le fichier correspondant à l'entraînement si celui-ci a été déplacé.
Le VAEConv utilisé doit correspondre à celui de l'entraînement ; son chemin
sauvegardé est utilisé si `--vae-checkpoint` est omis. Les options qui changent
la configuration du modèle entraîné sont refusées.

Sans checkpoint Lumen, `--allow-untrained` autorise un débruiteur aléatoire
pour vérifier le parcours technique ; il faut aussi fournir une échelle latente
positive avec `--vae-scale`. Sa sortie ne valide pas la qualité du modèle.

## VAEConv : image vers latent, puis latent vers image

```sh
./bin/mimir --lua scripts/inferences/infer_vae_conv.lua -- \
  --encode --checkpoint checkpoint/mon_vae \
  --input image.ppm --output outputs/image.raw

./bin/mimir --lua scripts/inferences/infer_vae_conv.lua -- \
  --decode --checkpoint checkpoint/mon_vae \
  --input outputs/image.raw --output outputs/reconstruction.ppm
```

L'entrée image est un PPM RGB P3 ou P6, avec une profondeur maximale de 8 bits.
Elle est redimensionnée aux dimensions du checkpoint, sauf avec `--no-resize`.
L'encodage écrit la moyenne latente `mu` en float32 little-endian, ordre CHW,
sans en-tête. Le décodage exige exactement `latent_w × latent_h × latent_c × 4`
octets et produit un PPM RGB. Utiliser le même checkpoint pour les deux étapes.
Le décodeur autonome ne dispose pas des connexions de saut provenant de l'image
encodée ; elles sont remplacées par des zéros lorsque le modèle en utilise.

Avec `raw_folder`, les dimensions et la configuration proviennent du checkpoint,
et la normalisation du décodeur est déduite des couches sérialisées pour les
anciens checkpoints. Le chargement du décodeur est partiel : les poids de
l'encodeur présents dans le checkpoint complet ne sont pas utilisés.
Le mode SafeTensors existant exige une configuration modèle correspondante
fournie au binaire via `--config` ; il ne lit pas automatiquement la configuration
depuis l'en-tête SafeTensors.

Un dossier parent contenant des sous-dossiers `epoch_*` est accepté par les deux
scripts via le résolveur de checkpoints du projet.

## VAEConv : édition interactive d'image

```sh
./bin/mimir --lua scripts/inferences/edit_vae_conv.lua -- \
  --checkpoint checkpoint/mon_vae \
  --input photo.ppm --output outputs/edition.ppm
```

Compiler un backend Viz actif (`SFML`, `QT`, `GTK` ou `WEB`). L'éditeur affiche
la source et le résultat, avec des boutons pour décaler un canal latent,
mélanger avec la source, réinitialiser et enregistrer. **H** présente l'aide et
le backend utilisé. Le checkpoint doit être au format `raw_folder` ; les images
peuvent être en PNG/JPEG/BMP ou PPM RGB, avec export en PPM. Avec `--headless`, le script calcule et enregistre une fois :
`--strength 0.2 --channel 1 --mix 0.8` règle l'édition sans interface.

Le module `scripts/modules/viz_ui.lua` permet de réutiliser les callbacks dans
un autre script. Voir [l'API Viz](../../docs/03-API-Reference/15-Viz-Htop.md#scènes-et-événements-pilotés-en-lua)
pour les panneaux, boutons et événements disponibles.
