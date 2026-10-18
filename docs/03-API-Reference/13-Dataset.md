# `Mimir.Dataset`

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Source : `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.

## `load(dir: string) -> (bool, n|err)`

Charge un dataset depuis un dossier.

Les items peuvent contenir :

- `text_file`, `image_file`, `audio_file`, `video_file`
- `width`, `height`
- `text` (si chargé)

## `get(index: int) -> table | (nil, err)`

Retourne les métadonnées et éventuellement le texte.

## `prepare_sequences(seq_len: int) -> (bool, n|err)`

Prépare des séquences tokenisées/paddées à longueur fixe.

Notes :

- Nécessite un tokenizer courant.
- Ce chemin est utile pour certains scripts legacy.

## Étapes suivantes

- [Page précédente : API : `Mimir.Tokenizer`](12-Tokenizer.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : mémoire](14-Memory.md)
