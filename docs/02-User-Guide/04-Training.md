# Entraînement

Lancer un entraînement fiable avec checkpoints et mémoire maîtrisée.

**Public concerné :** Intermédiaire (mais guidé pas à pas).

> **Prérequis**
>
> Comprendre la différence entre entraînement from scratch et reprise.

## Sur cette page

- [Diagramme d'explication](#diagramme-dexplication)
- [Les deux styles d'entraînement](#les-deux-styles-dentraînement)
- [Workflow recommandé étape par étape](#workflow-recommandé-étape-par-étape)
- [Calibration par feedback de validation](#calibration-par-feedback-de-validation)
- [Références](#références)
- [Étapes suivantes](#étapes-suivantes)

## Diagramme d'explication

![Pipeline d'entraînement](../graphs/11_training_pipeline.svg)


Ce guide couvre le workflow complet pour entraîner un modèle Mímir : de la configuration mémoire jusqu'au checkpoint final. Il présente les deux styles disponibles — **haut-niveau** (recommandé pour la plupart des cas) et **boucle manuelle** (pour les entraînements personnalisés).

> **Conseil :** si vous démarrez un nouvel entraînement, commencez par copier le script le plus proche de votre cas dans `scripts/training/` et adaptez les paramètres. C'est la façon la plus fiable de ne pas manquer une étape subtile.

---

## Les deux styles d'entraînement

### Style haut-niveau : `Mimir.Model.train(epochs, lr)`

Le plus simple. Mímir gère lui-même la boucle, le dataset, la validation et les checkpoints. La plupart des scripts actuels utilisent ce chemin.

```lua
Mimir.Dataset.load("dataset_2/")
assert(Mimir.Model.train(100, 1e-4))
```

L'implémentation exacte de la boucle dépend de l'architecture. Par exemple, VAEConv calcule reconstruction + KL. La fonction `lua_trainModel` construit un `TrainStepRequest` générique dans `src/scriptings/Lua/luaScripting/LuaScriptingModelAndRegistry.cpp`. Les overrides VAE délèguent leurs calculs spécialisés à `VAETraining`, dans `src/Models/VAETraining.cpp`.

### Style boucle manuelle

Pour les cas où vous devez contrôler précisément chaque étape :

```lua
for epoch = 1, epochs do
    for _, batch in ipairs(dataset) do
        Mimir.Model.zero_grads()
        local output = Mimir.Model.forward(batch.input, true)
        local grad = compute_grad(output, batch.target)  -- votre fonction
        Mimir.Model.backward(grad)
        Mimir.Model.optimizer_step(learning_rate, "adamw")
    end
end
```

> **Note :** la boucle manuelle nécessite que vous calculiez vous-même les gradients de la loss. Pour les architectures complexes (diffusion, VAE), la boucle haut-niveau est beaucoup plus simple car elle intègre les spécificités numériques de chaque modèle.

`optimizer_step` exige le learning rate en premier argument. Le second argument
optionnel vaut `"adamw"` par défaut. Les valeurs acceptées sont `"sgd"`,
`"adam"`, `"adamw"`, `"lion"`, `"adafactor"`, `"radam"`, `"nadam"`,
`"rmsprop"` et `"lamb"`.

Le nom Lua exact est `zero_grads()`. Le nom C++ correspondant est `Model::zeroGradients()`.

---

## Workflow recommandé étape par étape

### 1 — Configurer la mémoire

Cette étape est fortement recommandée avant toute autre. Elle évite les crashs OOM silencieux et active la compression mémoire pour les grands datasets.

```lua
-- Limiter l'utilisation RAM à 10 Go (adapté à votre machine)
pcall(Mimir.MemoryGuard.setLimit, 10)

-- Configurer l'allocateur dynamique
pcall(Mimir.Allocator.configure, {
    max_ram_gb         = 10.0,
    enable_compression = true,
    swap_strategy      = "lru",    -- éviction des tenseurs les moins récents en premier
})

-- Activer les optimisations matérielles CPU
pcall(Mimir.Model.set_hardware, true)
```

> **Note :** les appels sont enveloppés dans `pcall()` car ces fonctionnalités sont optionnelles (elles peuvent ne pas être compilées dans certains builds). `pcall` garantit que le script continue même si elles ne sont pas disponibles.

---

### 2 — Charger le tokenizer (si nécessaire)

Certaines architectures texte, comme Transformer, nécessitent un tokenizer. Si vous partez de zéro, créez-en un simple :

```lua
-- Tokenizer minimal pour les tests
Mimir.Tokenizer.create(8192)  -- vocab_size
```

Pour un entraînement sérieux, chargez le tokenizer de base préentraîné :

```lua
-- Recommandé pour les modèles texte
local tok = require("scripts/modules/base_tokenizer")
tok.load("checkpoint/base_tokenizer/")
```

> **Avertissement :** si vous chargez un checkpoint existant à l'étape 5, le tokenizer sera restauré automatiquement depuis le checkpoint (si `save_tokenizer = true` était activé lors de la sauvegarde). Dans ce cas, ne chargez pas un tokenizer séparé pour éviter les conflits de vocab.

---

### 3 — Charger le dataset

```lua
Mimir.Dataset.load("dataset_2/")
```

Le loader indexe récursivement le dossier, détecte les paires image+texte par basename et les met en cache dans `dataset_cache.json`.

Pour les modèles texte-seul (legacy) :

```lua
Mimir.Dataset.prepare_sequences(cfg.seq_len)
```

Voir [docs/02-User-Guide/03-Data.md](03-Data.md) pour les détails et limitations.

---

### 4 — Créer, construire et allouer le modèle

```lua
local cfg = {
    image_w      = 64,
    image_h      = 64,
    image_c      = 3,
    latent_h     = 8,
    latent_w     = 8,
    latent_c     = 4,
    base_channels = 16,
}

assert(Mimir.Model.create("vae_conv", cfg))
-- Model.build() n'est plus nécessaire (v3.0+: network construit automatiquement)

assert(Mimir.Model.allocate_params())
```

---

### 5 — Initialiser ou reprendre

**Nouveau modèle :**

```lua
assert(Mimir.Model.init_weights("he", 0))
```

**Reprise d'un entraînement :**

```lua
local ok, err = Mimir.Serialization.load("checkpoint/vae_conv_run1/model.safetensors")
assert(ok, err)
print("Checkpoint chargé")
```

---

### 6 — Lancer l'entraînement

```lua
assert(Mimir.Model.train(100, 1e-4))
```

---

### 7 — Sauvegarder

```lua
local ok, err = Mimir.Serialization.save("checkpoint/vae_conv_run1/", {
    format         = "safetensors",
    save_optimizer = true,
    save_tokenizer = true,
})
assert(ok, err)
```

---

## Calibration par feedback de validation

Lorsque la validation est activée, Mímir peut automatiquement ajuster le learning rate effectif en fonction de l'évolution de la métrique de validation. C'est un mécanisme de **récompense/punition** qui permet au modèle de s'autoréguler sans intervention manuelle.

### Le principe

Un multiplicateur `val_lr_scale` (initialement `1.0`) est appliqué en permanence au learning rate :

```
lr_effectif = lr_base × val_lr_scale
```

Après chaque validation, Mímir compare la métrique courante (lower is better) au meilleur résultat historique :

- **Amélioration suffisante** (`métrique < meilleur × (1 - val_improve_thresh)`) → **récompense** : `val_lr_scale × val_reward_factor`
- **Pas d'amélioration** (ou validation échouée / NaN) → **pénalité** : `val_lr_scale × val_penalty_factor`

Le multiplicateur est ensuite clampé dans `[val_lr_scale_min, val_lr_scale_max]` pour éviter les valeurs extrêmes.

**Illustration :**

```
Epoch  1 : val_loss = 0.45  → amélioration (+inf %)  → récompense → val_lr_scale = 1.05
Epoch  2 : val_loss = 0.42  → amélioration (6.7 %)   → récompense → val_lr_scale = 1.10
Epoch  3 : val_loss = 0.43  → dégradation            → pénalité   → val_lr_scale = 0.77
Epoch  4 : val_loss = 0.41  → amélioration (4.6 %)   → récompense → val_lr_scale = 0.81
```

### Paramètres

| Paramètre | Défaut | Description |
| --- | --- | --- |
| `val_feedback_enabled` | `false` | Active le mécanisme — **doit être `true`** |
| `val_reward_factor` | `1.05` | Facteur de récompense (+5 % LR) |
| `val_penalty_factor` | `0.70` | Facteur de pénalité (−30 % LR) |
| `val_lr_scale_min` | `0.10` | Plancher du multiplicateur (LR ne descend jamais en dessous de 10 %) |
| `val_lr_scale_max` | `1.50` | Plafond du multiplicateur |
| `val_improve_thresh` | `0.001` | L'amélioration doit dépasser 0.1 % pour déclencher la récompense |
| `val_feedback_min_steps` | `0` | Nombre de steps avant activation (0 = immédiat) |

### Exemple de configuration

```lua
local cfg = {
    -- Validation
    validate_every_epochs = 5,

    -- Feedback de calibration
    val_feedback_enabled   = true,
    val_reward_factor      = 1.08,   -- +8 % LR si la validation s'améliore
    val_penalty_factor     = 0.75,   -- -25 % LR si elle se dégrade
    val_lr_scale_min       = 0.05,   -- ne jamais descendre en dessous de 5 % du LR de base
    val_lr_scale_max       = 2.00,
    val_improve_thresh     = 0.005,  -- récompense uniquement si amélioration > 0.5 %
    val_feedback_min_steps = 100,    -- laisser le modèle chauffer 100 steps avant d'activer
}
```

> **Conseil :** commencez avec les valeurs par défaut (`val_reward_factor = 1.05`, `val_penalty_factor = 0.70`). Elles sont conservatives. Si le training oscille, augmentez `val_feedback_min_steps` ou réduisez `val_reward_factor`.

> **Avertissement :** un `val_penalty_factor` trop faible (ex: `0.3`) peut faire chuter le LR effectif très rapidement sur quelques epochs de validation consécutives mauvaises, ce qui peut bloquer l'entraînement. Préférez des valeurs entre `0.60` et `0.85`.

---

## Références

- API complète : [docs/03-API-Reference/10-Model.md](../03-API-Reference/10-Model.md)
- Format des datasets : [docs/02-User-Guide/03-Data.md](03-Data.md)
- Checkpoints : [docs/02-User-Guide/08-Checkpoints.md](08-Checkpoints.md)
- Scripts d'exemple : `scripts/training/`, `scripts/examples/`
- VAEConv, latent et prior appris : [docs/02-User-Guide/14-VAEConv.md](14-VAEConv.md)

## Étapes suivantes

- [Page précédente : Données / datasets](03-Data.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Inférence](05-Inference.md)
