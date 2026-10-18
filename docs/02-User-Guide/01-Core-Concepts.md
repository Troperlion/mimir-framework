# Concepts essentiels

Comprendre le fonctionnement global du framework avec un vocabulaire simple.

**Public concerné :** Débutant non ingénieur jusqu'à intermédiaire.

> **Prérequis**
>
> Connaître les notions entrée/sortie et configuration.


Cette page donne le “mental model” du framework : si vous comprenez ces idées, le reste de l’API devient lisible.

## 1) Un modèle Mímir = graph de tenseurs nommés

Un modèle est une suite (ou un petit graphe) d’opérations (“layers”) qui :

- lisent des tenseurs **par nom** (ex: `__input__`, `text_ids`)
- écrivent des tenseurs **par nom** (ex: `x`)

Ça explique pourquoi `forward()` accepte une table “map” : vous passez des entrées nommées et le graphe connecte le reste.

### Conventions de noms (pragmatiques)

| Nom | Type | Signification |
| --- | --- | --- |
| `__input__` | float ou ids int | entrée “par défaut” |
| `text_ids` | ids int | entrée texte dédiée (NLP) |
| `x` | float | sortie principale |

Conseil : quand vous écrivez une nouvelle architecture, garde ces conventions : les scripts deviennent interchangeables.

## 2) Registre d’architectures (le point d’entrée)

Vous ne construisez pas les layers “à la main” dans un script : vous demandez au registre de créer l’architecture.

- config par défaut : `Mimir.Architectures.default_config(name)`
- création : `Mimir.Model.create(name, cfg)`

Pourquoi c’est important : le registre centralise les configs canoniques (`seq_len`, `mlp_hidden`, etc.) et la compat save/load.

## 3) Lifecycle : create → build → allocate → init/load

Les appels ont un ordre logique (voir `docs/02-User-Guide/02-Model-Lifecycle.md`).

| Étape | Rôle | Erreur typique si oubli |
| --- | --- | --- |
| `create` | choisit type + config | type inconnu, config vide |
| `build` | construit la structure | forward échoue (graph absent) |
| `allocate_params` | alloue les poids | init/load échoue (pas de params) |
| `init_weights` | init aléatoire | sorties NaN/0 “bizarres” |
| `Serialization.load` | restaure checkpoint | mismatch shapes si cfg incompatible |

## 4) `forward` accepte 2 formes d’input

| Forme | Exemple | Quand l’utiliser |
| --- | --- | --- |
| liste | `{1,2,3}` ou `{0.1, 0.2}` | mono-input rapide |
| map | `{ __input__ = {...}, text_ids = {...} }` | multi-input, types mixtes |

Conseil : pour rester stable entre architectures, préfère le format map.

## 5) Mémoire : MemoryGuard + Allocator

Mímir peut allouer de gros buffers. Les scripts “sérieux” devraient configurer :

- `Mimir.MemoryGuard.setLimit(gb)` : limite dure (sécurité)
- `Mimir.Allocator.configure(...)` : pression mémoire (compression/éviction)

Voir `docs/02-User-Guide/09-Memory.md`.

## 6) Sérialisation : `Mimir.Serialization.*`

Pour sauvegarder/charger : utiliser `Mimir.Serialization.save/load`.

Voir `docs/03-API-Reference/02-Serialization.md`.

## Étapes suivantes

- [Revenir à la documentation](../00-INDEX.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Workflow modèle (lifecycle)](02-Model-Lifecycle.md)
