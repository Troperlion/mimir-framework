# Tokenizer et encodeur de conditionnement

Comprendre le rôle du tokenizer et de l'encoder dans les workflows texte.

**Public concerné :** Débutant NLP à intermédiaire.

> **Prérequis**
>
> Savoir ce qu'est un vocabulaire de tokens.

## Rôle des composants

- `Tokenizer` : texte ↔ ids.
- `ConditioningEncoder` : embeddings (vecteurs) associés au tokenizer/vocab.

## Synchronisation modèle

Le runtime maintient un invariant : **tout modèle doit avoir un conditioning encoder**.

Lors de `Mimir.Model.create(...)`, si aucun conditioning encoder n’est disponible, un conditioning encoder par défaut est créé à partir de la config (`embed_dim` ou `d_model`) et du vocab du tokenizer.

## Tokenizer gelé vs composé

Des scripts utilisent :

- `tokenizer_frozen = true` : on n’agrandit pas le vocab en lisant le dataset.
- `tokenizer_frozen = false` : le vocab peut être “composé” depuis le dataset (`tokenizeEnsure`).

Recommandation :

- Entraîner des briques réutilisables (VAEText, etc.) avec un **base tokenizer stable** et `tokenizer_frozen=true`.

## Fonctions pratiques

Voir la référence API :

- `Mimir.Tokenizer.tokenize`, `tokenize_ensure`
- `pad_sequence`, `vocab_size`, `save/load`
- BPE: `learn_bpe`, `tokenize_bpe`

## Étapes suivantes

- [Page précédente : Scripting Lua](06-Lua-Scripting.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Checkpoints / reprise d’entraînement](08-Checkpoints.md)
