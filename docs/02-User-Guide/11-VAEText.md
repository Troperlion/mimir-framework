# VAEText

Entraîner et évaluer VAEText avec les bonnes options.

**Public concerné :** Intermédiaire guidé.

> **Prérequis**
>
> Dataset texte disponible.


VAEText est un VAE “texte” qui reconstruit des tokens via logits, entraîné avec reconstruction (Cross-Entropy) + KL.

Version actuelle: VAEText peut aussi exposer un encodeur conditionnel de corrélation textuelle,
et des têtes de contexte internes (sémantique, thématique, dialogue) pour améliorer la régénération de dialogue.

## Entraîner

Script : `scripts/training/train_vae_texte.lua`

Exemple :

```bash
./bin/mimir --lua scripts/training/train_vae_texte.lua -- \
  --dataset-root ../dataset_2 \
  --out-dir checkpoint/vae_text_trained \
  --epochs 5 --lr 1e-4 \
  --seq-len 256 --d-model 256 --latent-tokens 32 \
  --kl-beta 0.01 \
  --decoder-causal true \
  --enable-conditional-encoder true \
  --enable-context-heads true \
  --context-semantic-dim 64 \
  --context-thematic-dim 32 \
  --context-dialog-dim 64 \
  --context-semantic-weight 0.08 \
  --context-thematic-weight 0.05 \
  --context-dialog-weight 0.10
```

Points importants :

- Utilise un “base tokenizer” commun (`scripts/modules/base_tokenizer.lua`) et `tokenizer_frozen=true`.
- Met `cfg.checkpoint_dir = out_dir` pour permettre des checkpoints d’interruption Ctrl+C.
- `decoder_causal=true` est recommandé pour la génération de dialogue auto-régressive.
- Les contextes internes sont appris en auto-supervision à partir des tokens d’entrée.

## Décodage et sampling

Le dépôt courant ne fournit pas de script `vae_text_sample.lua` prêt à l’emploi.
Le registre fournit toutefois `vae_text_decode`, qui transforme un latent en logits
de taille `seq_len × vocab_size`. Un sampler doit ensuite appliquer argmax,
température ou top-k sur ces logits, puis convertir les ids avec le même tokenizer
que celui du checkpoint.

Avant d’écrire ce workflow, inspecte les deux graphes :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_text --params --layers

./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_text_decode --params --layers
```

## Compat checkpoint

Tout sampler externe doit lire la configuration du checkpoint et refuser les
overrides incompatibles de `seq_len`, `vocab_size`, `d_model`, `latent_tokens` ou
tokenizer.
C’est volontaire : changer `seq_len` après entraînement casse souvent les shapes.

## Étapes suivantes

- [Page précédente : Scripts et outils Lua](10-Examples.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tutoriel : Transformer causal (GPT-style)](12-Transformer-GPT.md)
