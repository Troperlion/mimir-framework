# `Mimir.Tokenizer`

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Source : `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.

## Création

- `create(max_vocab: int) -> bool`

## Tokenisation

- `tokenize(text: string) -> table<int> | (nil, err)`
- `detokenize(ids: table<int>) -> string | (nil, err)`

Note : `tokenize()` utilise actuellement la tokenisation BPE (`tokenizeBPE`) côté C++.

## Vocab

- `vocab_size() -> int`
- `add_token(token: string) -> ...` (selon implémentation)
- `ensure_vocab_from_text(text: string)`
- `tokenize_ensure(text: string)`

## Tokens spéciaux

- `pad_id()`, `unk_id()`
- (selon build/tokenizer) `bos_id()`, `eos_id()` peuvent exister côté scripts.

## BPE

- `learn_bpe(corpus_path_or_table, ...)`
- `tokenize_bpe(text: string)`

## Séquences

- `set_max_length(n: int)`
- `pad_sequence(ids: table<int>, seq_len: int) -> table<int>`
- `batch_tokenize(texts: table<string>) -> table<table<int>>`

## I/O

- `save(path: string)`
- `load(path: string)`

## Étapes suivantes

- [Page précédente : API : `Mimir.Architectures`](11-Architectures.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Dataset`](13-Dataset.md)
