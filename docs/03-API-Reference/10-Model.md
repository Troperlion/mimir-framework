# `Mimir.Model`

Trouver rapidement le contrat API réel et les paramètres utilisables.

**Public concerné :** Développeur et utilisateur intermédiaire/avancé.

> **Prérequis**
>
> Connaître les commandes de base de Mímir.


`Mimir.Model` est le point d'entrée principal du framework. Il regroupe toutes les opérations sur le modèle courant : création, construction du graphe de layers, allocation des poids, exécution du forward/backward, et entraînement haut-niveau.

> **Note :** Mímir ne gère qu'un seul modèle actif à la fois par contexte. Toutes les fonctions de ce module opèrent sur ce modèle global.

Source C++ : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` — liaisons Lua → C++.

---

## Sur cette page

- [Cycle de vie d'un modèle](#cycle-de-vie-dun-modèle)
- [Création et construction](#création-et-construction)
- [Paramètres (poids)](#paramètres-poids)
- [Exécution (forward / backward)](#exécution-forward-backward)
- [Entraînement haut-niveau](#entraînement-haut-niveau)
- [Recon-Loss supportées (état actuel)](#recon-loss-supportées-état-actuel)
- [DType (précision des poids)](#dtype-précision-des-poids)
- [Matériel](#matériel)
- [Exemple complet (nouveau modèle)](#exemple-complet-nouveau-modèle)
- [API legacy](#api-legacy)
- [Calibration par feedback de validation](#calibration-par-feedback-de-validation)
- [Voir aussi](#voir-aussi)
- [Étapes suivantes](#étapes-suivantes)

## Cycle de vie d'un modèle

Avant tout appel à `forward()` ou `train()`, le modèle doit passer par ces étapes dans l'ordre :

| Étape | Appel | Rôle |
| ---: | --- | --- |
| 1 | `Mimir.Model.create(type, cfg)` | Enregistre le type et fusionne la config ; **construit aussi le réseau automatiquement (v3.0+)** |
| 2 | `Mimir.Model.allocate_params()` | Alloue les blocs de poids en mémoire |
| 3 | `Mimir.Model.init_weights(method, seed)` OU `Mimir.Serialization.load(path)` | Initialise/charge les poids |
| 4 | `Mimir.Model.forward(input, training)` | Exécute le forward pass |

> **Avertissement :** appeler `forward()` avant `allocate_params()` produit un comportement indéfini. Les étapes 1 à 3 sont obligatoires.

Détails du cycle de vie : [docs/02-User-Guide/02-Model-Lifecycle.md](../02-User-Guide/02-Model-Lifecycle.md).

---

## Création et construction

### `Mimir.Model.create(name, cfg?)`

```
Mimir.Model.create(name: string, cfg?: table) -> true | (false, string)
```

Enregistre l'architecture `name` comme modèle courant et fusionne `cfg` avec la configuration par défaut de cette architecture. **Construit automatiquement le graphe en v3.0+** (l'appel à `build()` est legacy/no-op).

**Paramètres :**

- `name` — nom canonique de l'architecture (ex: `"causal_lm"`, `"transformer"`, `"vae_conv"`, `"diffusion"`). Voir [la liste complète](./11-Architectures.md).
- `cfg` *(optionnel)* — table Lua de surcharge de config. Les clés non spécifiées conservent leurs valeurs par défaut.

**Retour :** `true` en cas de succès, ou `(false, message_erreur)`.

**Exemple :**

```lua
-- Créer un Transformer avec des dimensions réduites
local ok, err = Mimir.Model.create("transformer", {
    seq_len    = 128,
    d_model    = 256,
    num_layers = 4,
    vocab_size = 8192,
})
assert(ok, err)
```

Un chemin terminé par `.mpk` est également accepté : le package est décodé,
vérifié puis créé via le registre.

---

### `Mimir.Model.build()` (legacy — no-op en v3.0+)

```
Mimir.Model.build() -> (true, nb_params: int) | (false, string)
```

**Depuis v3.0:** cette fonction est un **no-op** — le modèle est déjà construit lors de `create()`. Cette fonction existe uniquement pour compatibilité avec les scripts legacy.

Historiquement, elle instanciait le graphe de layers à partir du type et de la config enregistrés par `create()`. Maintenant, cette construction se fait **automatiquement dans `create()`**.

**Retour :** `(true, nombre_total_de_paramètres)` ou `(false, message_erreur)` (invariant; nombre toujours disponible après `create()`).

```lua
-- Mode ancien (ne pas utiliser pour nouveaux scripts)
local ok, nb = Mimir.Model.build()
assert(ok, nb)

-- Mode moderne (recommandé)
-- Le réseau est déjà prêt après create() ci-dessus
```

---

## Paramètres (poids)

### `Mimir.Model.allocate_params()`

```
Mimir.Model.allocate_params() -> (true, total: int) | (false, string)
```

Alloue les blocs de mémoire pour tous les poids du modèle. Après cet appel, les poids existent en mémoire mais contiennent des valeurs non initialisées.

```lua
local ok, total = Mimir.Model.allocate_params()
assert(ok, total)
-- Vient ensuite init_weights() ou Serialization.load()
```

---

### `Mimir.Model.init_weights(method?, seed?)`

```
Mimir.Model.init_weights(method?: string = "he", seed?: int = 0) -> true | (false, string)
```

Initialise les poids selon la méthode spécifiée. À appeler uniquement pour un **nouveau** modèle — pour reprendre un entraînement, utilisez `Mimir.Serialization.load()` à la place.

**Méthodes disponibles :**

| Méthode | Description |
|---|---|
| `"he"` | He (Kaiming) — recommandé pour ReLU et variantes |
| `"xavier"` | Xavier/Glorot — recommandé pour tanh/sigmoïd |

```lua
assert(Mimir.Model.init_weights("he", 42))  -- seed 42 pour reproductibilité
```

---

### `Mimir.Model.total_params()`

```
Mimir.Model.total_params() -> int
```

Retourne le nombre total de paramètres scalaires du modèle. Pour estimer la
taille brute, multipliez par `bytes` du dtype correspondant, disponible via
`Mimir.Architectures.dtypes()`.

---

## Exécution (forward / backward)

### `Mimir.Model.forward(input, training?)`

```
Mimir.Model.forward(input: table, training?: bool = true)
    -> table<float> | (nil, string)
```

Exécute le forward pass du modèle. `input` peut prendre deux formes :

**Forme liste** — pour un seul tenseur d'entrée :

```lua
local out = Mimir.Model.forward({0.1, 0.2, 0.3, ...}, false)
```

**Forme map** — pour les entrées nommées (multi-input) :

```lua
local out = Mimir.Model.forward({
    __input__ = float_tensor,
    text_ids  = int_ids,
}, false)
```

Toutes les clés chaîne associées à un tableau sont transmises comme tenseurs
nommés en `float`, à l'exception de `text_ids` qui conserve le chemin entier.
Cela permet notamment d'appeler un graphe NMS avec `boxes`, `scores` et
`classes`.

> **Conseil :** privilégiez toujours la forme **map**, même pour un seul tenseur d'entrée. Elle rend vos scripts compatibles avec les architectures multi-input sans modification.

**Le flag `training`** contrôle plusieurs comportements :
- `true` — active le dropout, accumule les statistiques de normalisation, autorise le backward
- `false` — désactive le dropout, utilise les statistiques figées (BatchNorm), et peut activer certains fast-paths GPU supplémentaires (ex: Conv2d)

```lua
-- Inférence
local output, err = Mimir.Model.forward({ __input__ = ids }, false)
assert(output, err)

-- Training
local output = Mimir.Model.forward({ __input__ = ids }, true)
```

---

### `Mimir.Model.backward(grad_out)`

```
Mimir.Model.backward(grad_out: table<float>) -> true | (false, string)
```

Exécute la passe backward (rétropropagation). `grad_out` est le gradient de la loss par rapport à la sortie du modèle — il doit avoir la même taille que la sortie de `forward()`.

> **Note :** dans les scripts haut-niveau (`Model.train()`), `backward()` est appelé automatiquement. Utilisez cette fonction uniquement si vous écrivez une boucle d'entraînement manuelle.

---

### `Mimir.Model.optimizer_step(learning_rate, optimizer?)`

```
Mimir.Model.optimizer_step(learning_rate: number, optimizer?: string = "adamw")
    -> true | (false, string)
```

Applique une étape SGD, Adam, AdamW, Lion, Adafactor, RAdam, NAdam, RMSprop
ou LAMB. L'état de l'optimiseur est conservé entre les appels et peut être
inclus dans un checkpoint.

---

### `Mimir.Model.zero_grads()`

```
Mimir.Model.zero_grads() -> true | (false, string)
```

Remet à zéro tous les gradients accumulés. À appeler au début de chaque step d'entraînement manuel, avant `forward()`.

---

## Entraînement haut-niveau

### `Mimir.Model.train(epochs, lr)`

```
Mimir.Model.train(epochs: int, lr: number)
    -> (true, step_global: int) | (false, string)
```

Lance la boucle d'entraînement complète. Le comportement exact dépend de l'architecture : certains modèles, notamment VAEConv, ont des chemins dédiés dans `LuaScriptingModelAndRegistry.cpp`. Les fonctions mathématiques d’un step VAE vivent dans `Model.cpp`.

**Prérequis :** le dataset doit être chargé avant cet appel (`Mimir.Dataset.load()`).

```lua
Mimir.Dataset.load("dataset_2/")
assert(Mimir.Model.train(50, 1e-4))
```

> **Note :** si le processus reçoit `Ctrl+C` pendant l'entraînement, Mímir effectue un checkpoint d'interruption propre avant de terminer (si `cfg.checkpoint_dir` est défini).

## Recon-Loss supportées (état actuel)

Le paramètre `modelConfig.recon_loss` est interprété selon le chemin d'entraînement (VAE, diffusion, VAEText, etc.).

### Noms supportés (généraux)

| Valeur `recon_loss` | Alias | Disponibilité | Notes |
| --- | --- | --- | --- |
| `mse` | `l2` | générale | défaut dans la plupart des chemins |
| `mae` | `l1` | générale | erreur absolue moyenne |
| `bce` | - | générale | binary cross-entropy |
| `huber` | - | générale | `delta` configurable |
| `smoothl1` | `smooth_l1` | générale | équivalent Huber |
| `charbonnier` | - | générale | robuste, `eps` configurable |
| `gaussian_nll` | `nll_gaussian`, `gaussian-nll` | générale | NLL gaussienne |

### Noms supportés (spécifiques)

| Valeur `recon_loss` | Disponibilité | Notes |
| --- | --- | --- |
| `ce` | VAEText | cross-entropy token-level |
| `cross_entropy` | VAEText | alias de `ce` |
| `xent` | VAEText | alias court |

### Hyperparamètres associés

Selon le type choisi, ces clés de config peuvent s'appliquer :

- `huber_delta` (ou `smoothl1_delta`, `smoothl1_beta`)
- `charbonnier_eps`
- `nll_sigma` (ou `gaussian_nll_sigma`)

### Important

- Certains chemins internes publient `recon_loss_type` avec des valeurs comme `bce_logits` pour l'affichage/monitoring; ce label peut être émis même si la clé `recon_loss` n'est pas utilisée telle quelle dans la config.
- Si une valeur n'est pas reconnue par un modèle, un fallback explicite vers `mse` peut être appliqué.

---

## DType (précision des poids)

### `Mimir.Model.dtype(dtype?)`

```
Mimir.Model.dtype()             -> string
Mimir.Model.dtype(dtype: string) -> true | (false, string)
```

Lit ou fixe le **dtype par défaut** du modèle. Ce dtype est utilisé notamment lors de la sérialisation pour déterminer le format de stockage des poids.

Dtypes et alias acceptés :

- flottants : `float32`/`float`/`f32`, `float16`/`f16`/`fp16`,
  `bfloat16`/`bf16`, `float64`/`double`/`f64` ;
- signés : `int8`, `int16`, `int32`, `int64` et alias `i8` à `i64` ;
- non signés : `uint8`, `uint16`, `uint32`, `uint64` et alias `u8` à `u64` ;
- booléen : `bool`/`b1`.

La liste autoritative est accessible avec `Mimir.Architectures.dtypes()`.

Accessible aussi via l'alias **`Mimir.model.dtype`** (lowercase) :

```lua
-- Lire le dtype courant
local dt = Mimir.model.dtype()
print("dtype:", dt)  -- ex: "float32"

-- Changer le dtype (impact sur la sérialisation)
local ok, err = Mimir.model.dtype("float16")
assert(ok, err)
```

> **Note :** vous pouvez aussi définir `cfg.dtype = "float16"` lors de `Model.create()`. Dans ce cas, le dtype est propagé automatiquement sans appel explicite à cette fonction.

---

## Matériel

### `Mimir.Model.hardware_caps()`

```
Mimir.Model.hardware_caps() -> table
```

Retourne une table avec les capacités matérielles détectées sur le CPU courant :

```lua
local caps = Mimir.Model.hardware_caps()
print(caps.avx2, caps.fma, caps.f16c, caps.bmi2)
```

---

### `Mimir.Model.set_hardware(enabled)`

```
Mimir.Model.set_hardware(enabled: bool)
```

Active ou désactive les optimisations matérielles CPU (SIMD, etc.). Désactiver peut être utile pour déboguer des résultats numériques.

---

## Exemple complet (nouveau modèle)

```lua
-- 1. Configuration
local cfg = {
    seq_len    = 128,
    d_model    = 256,
    num_layers = 4,
    num_heads  = 4,
    vocab_size = 8192,
    causal     = true,
}

-- 2. Cycle de vie
assert(Mimir.Model.create("transformer", cfg))
local nb_params = Mimir.Model.total_params()
print(string.format("%.2f M paramètres", nb_params / 1e6))

assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("he", 0))

-- 3. Forward (inférence)
local ids = {}
for i = 1, cfg.seq_len do ids[i] = math.random(0, cfg.vocab_size - 1) end

local output, err = Mimir.Model.forward({ __input__ = ids }, false)
assert(output, err)
print(string.format("Sortie : %d flottants", #output))
```

---

## API legacy

### `Mimir.Model.infer(prompt)`

```
Mimir.Model.infer(prompt: string) -> string | nil
```

Chemin d'inférence historique (texte → texte). Pour les nouveaux scripts,
utilisez `forward()` avec le contrat d'entrée propre à l'architecture.

---

## Calibration par feedback de validation

Quand la validation est activée, Mímir peut automatiquement ajuster le learning rate effectif en fonction de l'évolution de la métrique de validation. Ce mécanisme de **récompense/punition** est décrit en détail dans [Entraînement — Calibration](../02-User-Guide/04-Training.md#calibration-par-feedback-de-validation).

### Paramètres dans `modelConfig`

| Paramètre | Type | Défaut | Description |
| --- | --- | --- | --- |
| `val_feedback_enabled` | bool | `false` | Active le mécanisme |
| `val_reward_factor` | float | `1.05` | Multiplicateur LR si amélioration |
| `val_penalty_factor` | float | `0.70` | Multiplicateur LR si dégradation |
| `val_lr_scale_min` | float | `0.10` | Borne inférieure du multiplicateur cumulé |
| `val_lr_scale_max` | float | `1.50` | Borne supérieure du multiplicateur cumulé |
| `val_improve_thresh` | float | `0.001` | Seuil d'amélioration relative minimale pour déclencher la récompense |
| `val_feedback_min_steps` | int | `0` | Steps d'entraînement avant activation |

**Comment fonctionne le multiplicateur :**

Un facteur `val_lr_scale` (initialement `1.0`) est appliqué en permanence au learning rate :

```
lr_effectif = lr_base × val_lr_scale
```

Après chaque validation :
- Si `métrique_actuelle < meilleure_métrique × (1 - val_improve_thresh)` → **récompense** : `val_lr_scale × val_reward_factor`
- Sinon → **pénalité** : `val_lr_scale × val_penalty_factor`
- Dans les deux cas, le résultat est clampé dans `[val_lr_scale_min, val_lr_scale_max]`

```lua
-- Exemple : feedback agressif
local cfg = {
    val_feedback_enabled   = true,
    val_reward_factor      = 1.10,   -- +10 % LR si amélioration
    val_penalty_factor     = 0.65,   -- -35 % LR si dégradation
    val_lr_scale_min       = 0.05,   -- LR ne descend jamais en dessous de 5 % du LR de base
    val_lr_scale_max       = 2.00,
    val_improve_thresh     = 0.005,  -- l'amélioration doit être > 0.5 %
    val_feedback_min_steps = 200,    -- attendre 200 steps avant d'activer
}
```

---

## Voir aussi

- [Cycle de vie d'un modèle](../02-User-Guide/02-Model-Lifecycle.md)
- [Entraînement](../02-User-Guide/04-Training.md)
- [Architectures disponibles](./11-Architectures.md)
- [Sérialisation](./16-Serialization.md)

## Étapes suivantes

- [Page précédente : Sérialisation (save/load) — résumé](02-Serialization.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : API : `Mimir.Architectures`](11-Architectures.md)
