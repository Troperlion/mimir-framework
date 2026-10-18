# Modèle causal decoder-only

Mímir fournit deux architectures texte distinctes :

- `transformer`, un encodeur Transformer recevant des embeddings flottants ;
- `causal_lm`, un modèle de langage decoder-only recevant des identifiants de
  tokens et produisant `seq_len * vocab_size` logits.

Pour entraîner un LLM causal, utilisez `causal_lm`.

## Architecture `causal_lm`

Le builder natif se trouve dans `src/Models/NLP/CausalLMModel.cpp`. Il assemble :

- embeddings de tokens ;
- RMSNorm pré-attention et pré-MLP ;
- attention causale avec RoPE et GQA (`num_kv_heads`) ;
- bloc SwiGLU construit avec deux projections, SiLU et Multiply ;
- connexions résiduelles ;
- RMSNorm final ;
- tête LM partageant les poids de l'embedding.

Configuration principale :

| Clé | Rôle |
| --- | --- |
| `vocab_size` | Taille maximale du vocabulaire |
| `seq_len` | Longueur du contexte d'entraînement |
| `d_model` | Dimension cachée |
| `num_layers` | Nombre de blocs decoder |
| `num_heads` | Nombre de têtes de requête |
| `num_kv_heads` | Nombre de têtes clé/valeur ; doit diviser `num_heads` |
| `mlp_hidden` | Largeur intermédiaire SwiGLU |
| `padding_idx` | Identifiant PAD |
| `norm_eps` | Epsilon RMSNorm |
| `rope_theta` | Base des fréquences RoPE |
| `dtype` | Dtype du modèle |

La configuration fournie est `configs/causal_lm.json` et le script
d'entraînement est `scripts/training/train_causal_lm.lua`.

## Lancer l'entraînement

Le corpus est un fichier texte fourni par l'utilisateur. Le script ne
télécharge et ne lit aucun dataset implicite.

```bash
./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
  --corpus /chemin/vers/corpus.txt \
  --tokenizer checkpoints/causal_lm/tokenizer.json \
  --steps 1000 \
  --lr 3e-4 \
  --no-amp \
  --no-ddp
```

Les options sont analysées par `scripts/modules/args.lua`. Les raccourcis
principaux sont `--vocab-size`, `--seq-len`, `--d-model`, `--layers`,
`--heads`, `--kv-heads`, `--mlp-hidden`, `--dtype`, `--steps`, `--lr`,
`--optimizer`, `--save-every` et `--checkpoint-dir`.

Les overrides structurés ont la priorité finale :

```bash
./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
  --corpus /chemin/vers/corpus.txt \
  --no-amp --no-ddp \
  --override model.num_layers=8 \
  --override model.num_kv_heads=2
```

La configuration livrée demande AMP. Le script refuse explicitement AMP et
DDP lorsque les backends autocast/collectifs ne sont pas disponibles ; utilisez
`--no-amp --no-ddp` dans ce cas.

## Tokenizer

Au démarrage, le script :

1. charge le tokenizer indiqué s'il existe ;
2. vérifie sa taille réelle et maximale, ses tokens spéciaux, son PAD et les
   identifiants produits sur le corpus ;
3. le reconstruit depuis le corpus s'il est absent ou incompatible ;
4. sauvegarde le tokenizer régénéré au chemin demandé.

Le corpus doit produire plus de `seq_len` tokens.

## Boucle Lua bas niveau

```lua
local cfg, cfg_err = Mimir.Architectures.default_config("causal_lm")
assert(cfg, cfg_err)
cfg.vocab_size = 256
cfg.seq_len = 16
cfg.d_model = 32
cfg.num_layers = 1
cfg.num_heads = 4
cfg.num_kv_heads = 2
cfg.mlp_hidden = 64

assert(Mimir.Model.create("causal_lm", cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

local ids = {}
for i = 1, cfg.seq_len do ids[i] = (i - 1) % cfg.vocab_size end
local logits, err = Mimir.Model.forward({ __input__ = ids }, true)
assert(logits, err)
assert(#logits == cfg.seq_len * cfg.vocab_size)
```

Pour un step manuel complet, consultez
`scripts/training/train_causal_lm.lua` : il calcule la cross-entropy, appelle
`zero_grads()`, `backward(gradient)` puis
`optimizer_step(learning_rate, optimizer)`.

## Limites actuelles

- Le script fourni entraîne par fenêtres échantillonnées, sans batching.
- L'API Lua n'expose pas encore une primitive de génération token par token.
- `Mimir.Model.infer()` reste un chemin historique et n'est pas le frontend de
  génération de `causal_lm`.

## Étapes suivantes

- [Tokenizer et encodeur](07-Tokenizer-Encoder.md)
- [Entraînement](04-Training.md)
- [API `Mimir.Model`](../03-API-Reference/10-Model.md)
