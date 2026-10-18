# `Mimir.Serialization`

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


Source : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` et `src/Serialization/*`.

## Formats

- `safetensors` (alias: `st`)
- `raw_folder` (alias: `raw`, `folder`)
- `debug_json` (alias: `debug`, `json`)

Les packages `.mpk` sont gérés séparément par `scripts/modules/mpk.lua` et le
chargeur de modèles. Ils décrivent une architecture/configuration mais ne
contiennent pas les poids. Voir
[MPK : packages d’architecture](../02-User-Guide/15-MPK.md).

## `save(path: string, format?: string="safetensors", opts?: table) -> bool | (false, err)`

Options (principales) :

- `save_tokenizer` (bool)
- `save_encoder` (bool)
- `save_optimizer` (bool)
- `include_git_info` (bool)
- `include_checksums` (bool)

Options DebugJson “enhanced” (v1.4) :

- `include_gradients` (bool)
- `include_optimizer_state` (bool)
- `include_activations` (bool)
- `include_weight_deltas` (bool)
- `max_values_per_tensor` (int)

## DType (stockage sur disque)

Le `dtype` du modèle participe au choix du dtype des tenseurs lors de
`save()`. Le dtype effectivement utilisé en calcul dépend ensuite du runtime et
de l'opération ; consultez `Architectures.dtypes()` pour les types reconnus.

- Source : `Model.default_dtype` (exposé via `Mimir.Model.dtype()`).
- Valeurs usuelles : `"float16"`, `"bfloat16"`, `"float32"`, `"float64"`.

Règles :

- `save()` (SafeTensors/RawFolder) convertit les tenseurs float du modèle en fonction de `Model.default_dtype`.
- `load()` convertit les tenseurs vers la représentation attendue par le
  modèle et **réapplique** `model_config.dtype` s'il est présent dans les
  métadonnées.

### Conformité SafeTensors

Les fichiers produits suivent le format SafeTensors officiel :

- longueur du header encodée sur 8 octets little-endian ;
- header UTF-8 commençant par `{`, complété uniquement avec des espaces ;
- `__metadata__` est strictement une map chaîne-vers-chaîne. L’option
  `custom_metadata` est validée puis stockée comme chaîne JSON canonique ;
- dtypes canoniques, formes et offsets vérifiés sans overflow ;
- buffer entièrement indexé, contigu, sans trou, chevauchement ou octet final
  non référencé ;
- tenseurs scalaires (`shape=[]`) et vides acceptés.

Le lecteur rejette les anciens fichiers Mímir dont `__metadata__` contient des
nombres ou objets JSON non conformes. Ils doivent être réexportés avec une
version corrigée du writer.

Bon pattern :

- mettre `cfg.dtype` dans la config passée à `Mimir.Model.create(...)`.
- ou appeler `Mimir.Model.dtype("float16")` après `create()` si vous voulez forcer un override.

## `load(path: string, format?: string|"auto", opts?: table) -> bool | (false, err)`

- Si `format` absent : auto-détection.

Options :

- `load_tokenizer`, `load_encoder`, `load_optimizer`
- `strict_mode`
- `validate_checksums`

## `detect_format(path: string) -> string | (nil, err)`

Retourne : `SAFETENSORS`, `RAWFOLDER`, `DEBUGJSON`.

## `save_enhanced_debug(path: string, opts?: table) -> bool | (false, err)`

Écrit un JSON d’inspection (stats + options avancées).

Depuis v1.3, le dump contient aussi `framework_state` avec un snapshot au moment du dump.
La v1.4 ajoute des sections aux périmètres explicites :

- `metadata` : version du format/framework, date de création, commit Git disponible ;
- `model` : identité, dtype, dimensions et nombre logique de paramètres ;
- `export_metrics` : éléments référencés et uniques, poids partagés, octets runtime
  et octets selon le dtype sérialisé. Ces compteurs excluent explicitement
  tokenizer, encodeur et optimiseur ;
- les statistiques de tenseurs comptent séparément les valeurs finies, zéros,
  `NaN`, `+Inf` et `-Inf`. Les échantillons non finis sont représentés par
  `null` afin que le fichier reste un JSON valide.

Le snapshot `framework_state` expose :

- runtime/backends (`cpu`, `cuda`, `rocm`, `opencl`, `vulkan`) + flags de config,
- capacités CPU (`avx2`, `fma`, `f16c`, `bmi2`),
- état mémoire (`MemoryGuard`, `DynamicTensorAllocator`, `AdvancedRAMManager`),
- registre de layers supportés,
- état modèle (dtype par défaut, params, layers, model_config, etc.).

## Étapes suivantes

- [Page précédente : API : monitoring / visualisation](15-Viz-Htop.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Layers`](18-Layers-Module.md)
