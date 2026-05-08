# Tutoriel : diffusion (PonyXL / SD3.5 / autoencoder)

Ce dépôt contient des architectures diffusion/autoencoder exposées via le registre.

Voir aussi:

- Exemples: `10-Examples.md`
- Datasets (linking par basename): `03-Data.md`
- Checkpoints: `08-Checkpoints.md`

## PonyXL (SDXL-like)

- Architecture canonique : `ponyxl_sdxl` (anciens alias: `ponyxl_ddpm`, `t2i_autoencoder`, etc.)
- Script training : `scripts/training/ponyxl_ddpm_train.lua`

### Entraîner (PonyXL DDPM)

Script: `scripts/training/ponyxl_ddpm_train.lua`

Pré-requis:

- dataset chargé via `MIMIR_DATASET_ROOT` (sinon fallback `../dataset`)
- un dataset multi-modal “image+texte” est généralement attendu

Important (format dataset):

- le loader indexe récursivement et associe les modalités par **basename**
- en pratique, pour un exemple image+caption, il faut des fichiers comme:

```text
dataset/
  0001.png
  0001.txt
  0002.jpg
  0002.txt
```

L’organisation en sous-dossiers est possible, mais évite les collisions de basename.

Lancement:

```bash
export MIMIR_DATASET_ROOT="/chemin/vers/dataset"
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua -- --help
./bin/mimir --lua scripts/training/ponyxl_ddpm_train.lua
```

Notes :

- Les anciennes options `peltier-*` sont **dépréciées** et ignorées (DDPM latent standard : $\epsilon \sim \mathcal{N}(0,I)$).
- Pour activer/désactiver certains chemins hardware, le script expose une option `--hw`.

Resume:

- le script supporte un mode auto-resume (variable `MIMIR_RESUME`)
- il charge un checkpoint existant dans `checkpoint_dir` si trouvé

Le script écrit généralement:

- un checkpoint `raw_folder` dans `cfg.checkpoint_dir`
- un `debug.json` (format debug, pour inspection humaine)

## SD3.5 (statut)

- Architecture : `sd3_5` (alias: `SD3.5`)
Cette partie est surtout un statut d’architecture côté registre : il n’y a pas forcément de “démos” prêtes à l’emploi dans `scripts/examples/`.

## Autoencoder image

- VAE conv : `vae_conv` (vision)
- Script training : `scripts/training/train_vae_conv.lua`

Astuce: commence par un VAE conv petit et valide le save/load avant d’attaquer une pipeline diffusion complète.

## Statut

Les scripts de diffusion sont des **démos / squelettes** tant qu’aucun checkpoint entraîné n’est fourni.
Pour générer des images cohérentes, il faut :

- un VAE image entraîné
- un U-Net/diffusion entraîné
- une config de scheduler cohérente
