# Ajouter une architecture et ses outils

Contribuer avec des changements cohérents et maintenables.

**Public concerné :** Contributeur du projet.

> **Prérequis**
>
> Connaître le workflow Git et les bases du projet.


Cette page explique, de manière opérationnelle, comment :

- ajouter une nouvelle architecture de modèle dans le framework,
- l'enregistrer dans le registre C++,
- écrire un script Lua de training/inférence pour l'utiliser,
- utiliser les scripts de `scripts/tools/`.

## Sur cette page

- [1. Ajouter une nouvelle architecture C++](#1-ajouter-une-nouvelle-architecture-c)
- [2. Enregistrer l'architecture dans le registre](#2-enregistrer-larchitecture-dans-le-registre)
- [3. Écrire un script Lua pour utiliser l'architecture](#3-écrire-un-script-lua-pour-utiliser-larchitecture)
- [4. Checklist d'intégration](#4-checklist-dintégration)
- [5. Scripts outils (scripts/tools/.lua)](#5-scripts-outils-scriptstoolslua)
- [6. Commandes rapides pour un dev](#6-commandes-rapides-pour-un-dev)
- [Étapes suivantes](#étapes-suivantes)

## 1. Ajouter une nouvelle architecture C++

### 1.1 Créer la classe modèle

Créer un modèle dans `src/Models/...` (par exemple `src/Models/MyDomain/MyNewModel.hpp` et `src/Models/MyDomain/MyNewModel.cpp`).

Bonnes pratiques minimales :

- définir une `Config` interne avec des valeurs par défaut raisonnables,
- construire explicitement la topologie dans `build()` (ou équivalent),
- garantir que le modèle est compatible avec :
  - allocation des paramètres,
  - initialisation des poids,
  - forward,
  - sérialisation.

Exemple de squelette :

```cpp
class MyNewModel : public Model {
public:
    struct Config {
        int d_model = 256;
        int num_layers = 4;
        int output_dim = 1024;
    };

    explicit MyNewModel(const Config& cfg);
    void build();

private:
    Config cfg_;
};
```

## 2. Enregistrer l'architecture dans le registre

Le registre central est :

- `src/Models/Registry/ModelArchitectures.hpp`
- `src/Models/Registry/ModelArchitectures.cpp`

### 2.1 Inclure la nouvelle classe

Dans `src/Models/Registry/ModelArchitectures.cpp`, ajouter l'include de la nouvelle classe.

### 2.2 Mapper JSON -> Config

Ajouter une fonction helper du style `myNewCfgFromJson(const json& cfg)` qui lit les champs de config avec fallback.

### 2.3 Définir la config par défaut

Ajouter une fonction `myNewDefaultConfigJson()` retournant les champs JSON attendus par le modèle.

### 2.4 Ajouter l'entrée du registre

Dans l'enregistrement des builtins, ajouter une entrée `Entry` avec :

- `name` : nom CLI/Lua de l'architecture,
- `description` : description courte,
- `default_config` : config JSON par défaut,
- `create` : lambda/fonction qui instancie le modèle.

Exemple conceptuel :

```cpp
registry.registerArchitecture({
    "my_new_model",
    "Mon nouveau modèle",
    myNewDefaultConfigJson(),
    [](const json& cfg) {
        auto c = myNewCfgFromJson(cfg);
        return std::make_shared<MyNewModel>(c);
    }
});
```

### 2.5 Vérifier la visibilité API

Valider que l'architecture est visible côté runtime :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

## 3. Écrire un script Lua pour utiliser l'architecture

Partir de :

- `scripts/templates/template_new_model.lua` (API directe bas niveau)
- `scripts/templates/template_pipeline_only.lua` (pipeline minimal)
- `scripts/templates/template_pipeline_args.lua` (pipeline + args + registry-first)

### 3.1 Mode pipeline + registry-first (recommandé)

C'est le flux moderne via `scripts/modules/pipeline.lua`.

Pattern simple :

```lua
local P = dofile(ROOTWORK.."/scripts/modules/pipeline.lua")

-- Créer un pipeline générique basé sur le registre.
local pipe, err = P.FromRegistry("my_new_model")
if not pipe then error(err) end

-- Charger la config par défaut du registre.
local ok, cfg = pipe:loadDefaultConfig("my_new_model")
if not ok then error(cfg) end

-- Appliquer des patches/overrides locaux.
local ok, _ = pipe:patchConfig({ d_model = 256, num_layers = 4 })
if not ok then error(_) end

-- Build, train, save.
pipe:build()
pipe:train("dataset.bin", 10, 0.0003)
pipe:save("checkpoint/my_new_model.safetensors")
```

Avantages :

- No hardcoding d'architecture dans le script.
- Réutilisable pour n'importe quelle archi du registre.
- Config chargée depuis le C++ = garantie de cohérence.

### 3.2 Avec parseur d'arguments (template complet)

Voir [scripts/templates/template_pipeline_args.lua](../../scripts/templates/template_pipeline_args.lua) pour un exemple complet avec `--from-registry` et `--arch`.

Usage :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch transformer \
  --d-model 256 --layers 4 --heads 8 --seq-len 128 \
  --override mlp_hidden=1024 \
  --dataset dataset.bin --epochs 10 --lr 0.0003 \
  --save checkpoint/run.safetensors
```

## 4. Checklist d'intégration

Avant de merger une nouvelle architecture :

- vérifier l'entrée dans `inspect_architectures.lua`,
- vérifier le cycle create -> allocate -> init -> forward,
- vérifier un run de training court,
- vérifier save/load du checkpoint,
- documenter les paramètres principaux dans la doc API,
- ajouter au moins un script smoke test.

## 5. Scripts outils (`scripts/tools/*.lua`)

Ces scripts sont pensés pour le debug, la conversion d'artefacts et l'inspection rapide.

### 5.1 inspect_architectures.lua

Rôle : lister les architectures disponibles, dtypes, paramètres de config, couches.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l vae_conv -p --layers
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --json
```

Options clés : `-a`, `-l`, `-p`, `--layers`, `--stats`, `-d`, `--json`.

Exemple de sortie réelle (commande complète) :

```text
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a -l vae_conv -p --layers --stats
╔════════════════════════════════════════╗
║       Mímir Framework v3.5.0           ║
║     Deep Learning Architectures        ║
╚════════════════════════════════════════╝

🛡️  Vérification de la sécurité mémoire...
✅ Structure legacy désactivée (configuration optimale)

🧪 TEST D'INTÉGRITÉ MÉMOIRE
═══════════════════════════
Test 1: MemoryGuard accessible... ✓ (Limite: 10 GB)
Test 2: DynamicTensorAllocator accessible... ✓
Test 3: Cycle allocation/libération... ✓
Test 4: Structure legacy désactivée... ✅ Structure legacy désactivée (configuration optimale)

✅ TOUS LES TESTS PASSÉS
═══════════════════════════


🔧 OpenMP: 12 threads disponibles
🚀 Optimisations hardware:
  • AVX2: ✓
  • FMA: ✓
  • F16C: ✓
  • BMI2: ✓
  • CUDA: ✗
  • ROCM: ✗

📜 Exécution du script Lua: scripts/tools/inspect_architectures.lua
═══════════════════════════════════════════════


* Architectures supportées par le Framework :
+---------------------------+--------------+--------------------------------------------------------------------------+
| Architecture              | dtype défaut | Description                                                              |
+===========================+==============+==========================================================================+
| basic_mlp                 | float32      | Basic MLP (régression: input->MLP->output)                               |
| cond_diffusion            | float32      | Conditional diffusion epsilon predictor (baseline MLP: input=prompt||... |
| diffusion                 | float32      | Diffusion epsilon predictor (baseline MLP: input=t_embed||x_t)           |
| external_safetensors_base | float32      | Base non-executable qui reflète exactement les clés d'un checkpoint s... |
| gan_latent                | float32      | GAN-like latent generator (baseline MLP: input=prompt||noise -> latent)  |
| hf_clip_text_encoder_1    | float32      | Encodeur texte CLIP/SDXL exécutable pour checkpoints HuggingFace/PyTo... |
| hf_clip_text_encoder_2    | float32      | Encodeur texte OpenCLIP/SDXL exécutable pour checkpoints HuggingFace/... |
| hf_sdxl_transformer_block | float32      | Bloc transformer SDXL/HuggingFace exécutable avec SelfAttention, Cros... |
| hf_vae_decoder            | float32      | Décodeur VAE SDXL/HuggingFace exécutable pour le composant first_stag... |
| mobilenet                 | float32      | MobileNetV1-style (DepthwiseConv2d + pointwise Conv2d)                   |
| patch_discriminator       | float32      | PatchGAN-like discriminator (image -> patch logits)                      |
| resnet                    | float32      | ResNet (simplified ResNet18-like)                                        |
| sd3_5                     | float32      | Stable Diffusion 3.5 (placeholder: registry + skeleton model)            |
| transformer               | float32      | Transformer encoder (float-only: input=seq_len*d_model)                  |
| unet                      | float32      | UNet (simplified encoder-decoder with skip concatenations)               |
| vae                       | float32      | VAE-style autoencoder (output packs recon||mu||logvar)                   |
| vae_conv                  | float32      | Convolutional VAE (output packs recon||mu||logvar; spatial latent)       |
| vae_conv_decode           | float32      | Convolutional VAE decoder-only (input=z latent, output=recon RGB)        |
| vae_text                  | float32      | VAEText (text_ids -> recon||mu||logvar||img_proj||text_proj)             |
| vae_text_decode           | float32      | VAEText decoder-only (input=z latent, output=logits seq_len*vocab)       |
| vgg16                     | float32      | VGG16 (simplified, downsample via stride-2 conv)                         |
| vgg16_feat                | float32      | VGG16 feature extractor (GAP features for perceptual loss)               |
| vgg19                     | float32      | VGG19 (simplified, downsample via stride-2 conv)                         |
| vit                       | float32      | ViT (float-only: input=patch embeddings num_tokens*d_model)              |
+---------------------------+--------------+--------------------------------------------------------------------------+

* dtypes pris en charge par le Framework :
+----------+---------+--------+----------------------+
| dtype    | Famille | Octets | Alias acceptés       |
+==========+=========+========+======================+
| float32  | float   |      4 | float, f32, float32  |
| float16  | float   |      2 | f16, float16, fp16   |
| bfloat16 | float   |      2 | bf16, bfloat16       |
| float64  | float   |      8 | double, f64, float64 |
| int8     | int     |      1 | i8, int8             |
| int16    | int     |      2 | i16, int16           |
| int32    | int     |      4 | i32, int32           |
| int64    | int     |      8 | i64, int64           |
| uint8    | uint    |      1 | u8, uint8            |
| uint16   | uint    |      2 | u16, uint16          |
| uint32   | uint    |      4 | u32, uint32          |
| uint64   | uint    |      8 | u64, uint64          |
| bool     | bool    |      1 | bool, b1             |
+----------+---------+--------+----------------------+

* Architecture 'vae_conv' :
  description: Convolutional VAE (output packs recon||mu||logvar; spatial latent)
  dtype par défaut: float32
  paramètres (config par défaut) :
+----------------------+------------------+---------+
| Paramètre            | Valeur           | Type    |
+======================+==================+=========+
| attn_heads           | 4                | number  |
| attn_max_tokens      | 0                | number  |
| base_channels        | 64               | number  |
| d_model              | 0                | number  |
| dec_gn_groups        | 32               | number  |
| dec_norm             | "groupnorm"      | string  |
| decoder_upsample     | "conv_transpose" | string  |
| enc_gn_groups        | 32               | number  |
| enc_norm             | "groupnorm"      | string  |
| image_c              | 3                | number  |
| image_h              | 64               | number  |
| image_w              | 64               | number  |
| latent_c             | 256              | number  |
| latent_h             | 16               | number  |
| latent_w             | 16               | number  |
| proj_dim             | 256              | number  |
| resnet_max_tokens    | 0                | number  |
| seq_len              | 64               | number  |
| stochastic_latent    | false            | boolean |
| text_cond            | false            | boolean |
| text_d_model         | 256              | number  |
| resnet               | true             | boolean |
| attention            | false            | boolean |
| use_encoder_prior    | false            | boolean |
| use_skip_connections | false            | boolean |
| vocab_size           | 32000            | number  |
+----------------------+------------------+---------+

* Layers – vae_conv

> Capture historique : les nombres de layers et de paramètres ci-dessous proviennent
> d’un graphe construit sans normalisation émise. Avec les valeurs courantes
> `enc_norm=groupnorm` et `dec_norm=groupnorm`, le builder ajoute des couches
> `GroupNorm` et le total change. Pour une valeur à jour, exécuter
> `scripts/examples/inspect_vae_conv.lua` ou interroger `Mimir.Model.get_layers()`.

✓ Vulkan Compute initialized
[vae_conv] Modèle créé via registre: vae_conv
📦 Allocation de 52 blocs de poids (941440 paramètres au total)...
  Layer 3 (vae_conv/enc/conv_in): 1728 paramètres dans 1 tensor
  Layer 5 (vae_conv/enc/res0/conv1): 36864 paramètres dans 1 tensor
  Layer 7 (vae_conv/enc/res0/conv2): 36864 paramètres dans 1 tensor
  Layer 9 (vae_conv/enc/down1/conv): 36864 paramètres dans 1 tensor
  Layer 11 (vae_conv/enc/down1/res/conv1): 36864 paramètres dans 1 tensor
  Layer 13 (vae_conv/enc/down1/res/conv2): 36864 paramètres dans 1 tensor
  Layer 15 (vae_conv/enc/down2/conv): 36864 paramètres dans 1 tensor
  Layer 17 (vae_conv/enc/down2/res/conv1): 36864 paramètres dans 1 tensor
  Layer 19 (vae_conv/enc/down2/res/conv2): 36864 paramètres dans 1 tensor
  Layer 21 (vae_conv/enc/proj): 36864 paramètres dans 1 tensor
  Layer 23 (vae_conv/enc/bot_res/conv1): 36864 paramètres dans 1 tensor
  Layer 25 (vae_conv/enc/bot_res/conv2): 36864 paramètres dans 1 tensor
  Layer 27 (vae_conv/enc/mu): 16384 paramètres dans 1 tensor
  Layer 28 (vae_conv/enc/logvar): 16384 paramètres dans 1 tensor
  Layer 30 (vae_conv/dec/conv_in): 147456 paramètres dans 1 tensor
  Layer 32 (vae_conv/dec/bot_res/conv1): 36864 paramètres dans 1 tensor
  Layer 34 (vae_conv/dec/bot_res/conv2): 36864 paramètres dans 1 tensor
  Layer 36 (vae_conv/dec/up2/up): 65536 paramètres dans 1 tensor
  Layer 38 (vae_conv/dec/up2/res/conv1): 36864 paramètres dans 1 tensor
  Layer 40 (vae_conv/dec/up2/res/conv2): 36864 paramètres dans 1 tensor
  Layer 42 (vae_conv/dec/up1/up): 65536 paramètres dans 1 tensor
  Layer 44 (vae_conv/dec/up1/res/conv1): 36864 paramètres dans 1 tensor
  Layer 46 (vae_conv/dec/up1/res/conv2): 36864 paramètres dans 1 tensor
  Layer 48 (vae_conv/dec/out): 1728 paramètres dans 1 tensor
✓ 52 blocs de poids créés (1 tensor par layer)
[vae_conv] Paramètres alloués: 941440
[vae_conv] +----+------------------------------+-----------------+--------------------+--------+----------------------------+
| #  | Nom                          | Type            | Shape              | Params | Output                     |
+====+==============================+=================+====================+========+============================+
|  1 | vae_conv/raw_in              | Identity        |                    |      - | vae_conv/in_vec            |
|  2 | vae_conv/in_reshape          | Reshape         | →reshape           |      - | vae_conv/in_hwc            |
|  3 | vae_conv/in_to_chw           | Permute         | →permute           |      - | vae_conv/in_chw            |
|  4 | vae_conv/enc/conv_in         | Conv2d          | [3→64, k3 s1 p1]   |   1728 | vae_conv/enc/c0            |
|  5 | vae_conv/enc/conv_in/act     | SiLU            |                    |      - | vae_conv/enc/c0_act        |
|  6 | vae_conv/enc/res0/conv1      | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/res0/c1       |
|  7 | vae_conv/enc/res0/act1       | SiLU            |                    |      - | vae_conv/enc/res0/c1a      |
|  8 | vae_conv/enc/res0/conv2      | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/res0/c2       |
|  9 | vae_conv/enc/res0/add        | Add             |                    |      - | vae_conv/enc/res0/out      |
| 10 | vae_conv/enc/down1/conv      | Conv2d          | [64→64, k3 s2 p1]  |  36864 | vae_conv/enc/down1/y       |
| 11 | vae_conv/enc/down1/conv/act  | SiLU            |                    |      - | vae_conv/enc/down1/y_act   |
| 12 | vae_conv/enc/down1/res/conv1 | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/down1/res/c1  |
| 13 | vae_conv/enc/down1/res/act1  | SiLU            |                    |      - | vae_conv/enc/down1/res/c1a |
| 14 | vae_conv/enc/down1/res/conv2 | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/down1/res/c2  |
| 15 | vae_conv/enc/down1/res/add   | Add             |                    |      - | vae_conv/enc/down1/res/out |
| 16 | vae_conv/enc/down2/conv      | Conv2d          | [64→64, k3 s2 p1]  |  36864 | vae_conv/enc/down2/y       |
| 17 | vae_conv/enc/down2/conv/act  | SiLU            |                    |      - | vae_conv/enc/down2/y_act   |
| 18 | vae_conv/enc/down2/res/conv1 | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/down2/res/c1  |
| 19 | vae_conv/enc/down2/res/act1  | SiLU            |                    |      - | vae_conv/enc/down2/res/c1a |
| 20 | vae_conv/enc/down2/res/conv2 | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/down2/res/c2  |
| 21 | vae_conv/enc/down2/res/add   | Add             |                    |      - | vae_conv/enc/down2/res/out |
| 22 | vae_conv/enc/proj            | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/h             |
| 23 | vae_conv/enc/proj/act        | SiLU            |                    |      - | vae_conv/enc/h_act         |
| 24 | vae_conv/enc/bot_res/conv1   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/bot_res/c1    |
| 25 | vae_conv/enc/bot_res/act1    | SiLU            |                    |      - | vae_conv/enc/bot_res/c1a   |
| 26 | vae_conv/enc/bot_res/conv2   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/enc/bot_res/c2    |
| 27 | vae_conv/enc/bot_res/add     | Add             |                    |      - | vae_conv/enc/bot_res/out   |
| 28 | vae_conv/enc/mu              | Conv2d          | [64→256, k1 s1 p0] |  16384 | vae_conv/mu                |
| 29 | vae_conv/enc/logvar          | Conv2d          | [64→256, k1 s1 p0] |  16384 | vae_conv/logvar            |
| 30 | vae_conv/reparam             | Reparameterize  |                    |      - | vae_conv/z                 |
| 31 | vae_conv/dec/conv_in         | Conv2d          | [256→64, k3 s1 p1] | 147456 | vae_conv/dec/c0            |
| 32 | vae_conv/dec/conv_in/act     | SiLU            |                    |      - | vae_conv/dec/c0_act        |
| 33 | vae_conv/dec/bot_res/conv1   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/dec/bot_res/c1    |
| 34 | vae_conv/dec/bot_res/act1    | SiLU            |                    |      - | vae_conv/dec/bot_res/c1a   |
| 35 | vae_conv/dec/bot_res/conv2   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/dec/bot_res/c2    |
| 36 | vae_conv/dec/bot_res/add     | Add             |                    |      - | vae_conv/dec/bot_res/out   |
| 37 | vae_conv/dec/up2/up          | ConvTranspose2d | [64→64, k4 s2 p1]  |  65536 | vae_conv/dec/up2/up_y      |
| 38 | vae_conv/dec/up2/up/act      | SiLU            |                    |      - | vae_conv/dec/up2/up_y_act  |
| 39 | vae_conv/dec/up2/res/conv1   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/dec/up2/res/c1    |
| 40 | vae_conv/dec/up2/res/act1    | SiLU            |                    |      - | vae_conv/dec/up2/res/c1a   |
| 41 | vae_conv/dec/up2/res/conv2   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/dec/up2/res/c2    |
| 42 | vae_conv/dec/up2/res/add     | Add             |                    |      - | vae_conv/dec/up2/res/out   |
| 43 | vae_conv/dec/up1/up          | ConvTranspose2d | [64→64, k4 s2 p1]  |  65536 | vae_conv/dec/up1/up_y      |
| 44 | vae_conv/dec/up1/up/act      | SiLU            |                    |      - | vae_conv/dec/up1/up_y_act  |
| 45 | vae_conv/dec/up1/res/conv1   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/dec/up1/res/c1    |
| 46 | vae_conv/dec/up1/res/act1    | SiLU            |                    |      - | vae_conv/dec/up1/res/c1a   |
| 47 | vae_conv/dec/up1/res/conv2   | Conv2d          | [64→64, k3 s1 p1]  |  36864 | vae_conv/dec/up1/res/c2    |
| 48 | vae_conv/dec/up1/res/add     | Add             |                    |      - | vae_conv/dec/up1/res/out   |
| 49 | vae_conv/dec/out             | Conv2d          | [64→3, k3 s1 p1]   |   1728 | vae_conv/dec/out_pre       |
| 50 | vae_conv/dec/tanh            | Tanh            |                    |      - | vae_conv/recon_chw         |
| 51 | vae_conv/recon_to_hwc        | Permute         | →permute           |      - | vae_conv/recon             |
| 52 | vae_conv/out_concat          | Concat          | →concat            |      - | x                          |
+----+------------------------------+-----------------+--------------------+--------+----------------------------+
[vae_conv]
* Stats – vae_conv
[vae_conv] Modèle créé via registre: vae_conv
📦 Allocation de 52 blocs de poids (941440 paramètres au total)...
  Layer 3 (vae_conv/enc/conv_in): 1728 paramètres dans 1 tensor
  Layer 5 (vae_conv/enc/res0/conv1): 36864 paramètres dans 1 tensor
  Layer 7 (vae_conv/enc/res0/conv2): 36864 paramètres dans 1 tensor
  Layer 9 (vae_conv/enc/down1/conv): 36864 paramètres dans 1 tensor
  Layer 11 (vae_conv/enc/down1/res/conv1): 36864 paramètres dans 1 tensor
  Layer 13 (vae_conv/enc/down1/res/conv2): 36864 paramètres dans 1 tensor
  Layer 15 (vae_conv/enc/down2/conv): 36864 paramètres dans 1 tensor
  Layer 17 (vae_conv/enc/down2/res/conv1): 36864 paramètres dans 1 tensor
  Layer 19 (vae_conv/enc/down2/res/conv2): 36864 paramètres dans 1 tensor
  Layer 21 (vae_conv/enc/proj): 36864 paramètres dans 1 tensor
  Layer 23 (vae_conv/enc/bot_res/conv1): 36864 paramètres dans 1 tensor
  Layer 25 (vae_conv/enc/bot_res/conv2): 36864 paramètres dans 1 tensor
  Layer 27 (vae_conv/enc/mu): 16384 paramètres dans 1 tensor
  Layer 28 (vae_conv/enc/logvar): 16384 paramètres dans 1 tensor
  Layer 30 (vae_conv/dec/conv_in): 147456 paramètres dans 1 tensor
  Layer 32 (vae_conv/dec/bot_res/conv1): 36864 paramètres dans 1 tensor
  Layer 34 (vae_conv/dec/bot_res/conv2): 36864 paramètres dans 1 tensor
  Layer 36 (vae_conv/dec/up2/up): 65536 paramètres dans 1 tensor
  Layer 38 (vae_conv/dec/up2/res/conv1): 36864 paramètres dans 1 tensor
  Layer 40 (vae_conv/dec/up2/res/conv2): 36864 paramètres dans 1 tensor
  Layer 42 (vae_conv/dec/up1/up): 65536 paramètres dans 1 tensor
  Layer 44 (vae_conv/dec/up1/res/conv1): 36864 paramètres dans 1 tensor
  Layer 46 (vae_conv/dec/up1/res/conv2): 36864 paramètres dans 1 tensor
  Layer 48 (vae_conv/dec/out): 1728 paramètres dans 1 tensor
✓ 52 blocs de poids créés (1 tensor par layer)
[vae_conv] Paramètres alloués: 941440
[vae_conv]   Total paramètres: 941440  (0.94 M)
[vae_conv]   Total layers: 52
[vae_conv]
[vae_conv]   Répartition par type :
[vae_conv] +-----------------+----+------------+-------+
| Type            | Nb | Paramètres | %     |
+=================+====+============+=======+
| Conv2d          | 22 |     810368 | 86.1% |
| ConvTranspose2d |  2 |     131072 | 13.9% |
| Tanh            |  1 |          0 |  0.0% |
| Concat          |  1 |          0 |  0.0% |
| SiLU            | 14 |          0 |  0.0% |
| Identity        |  1 |          0 |  0.0% |
| Reparameterize  |  1 |          0 |  0.0% |
| Reshape         |  1 |          0 |  0.0% |
| Add             |  7 |          0 |  0.0% |
| Permute         |  2 |          0 |  0.0% |
+-----------------+----+------------+-------+
[vae_conv]
  Top-20 layers (params desc) :
[vae_conv] +----+------------------------------+-----------------+------------+-------+
| #  | Layer                        | Type            | Paramètres | %     |
+====+==============================+=================+============+=======+
|  1 | vae_conv/dec/conv_in         | Conv2d          |     147456 | 15.7% |
|  2 | vae_conv/dec/up2/up          | ConvTranspose2d |      65536 |  7.0% |
|  3 | vae_conv/dec/up1/up          | ConvTranspose2d |      65536 |  7.0% |
|  4 | vae_conv/enc/bot_res/conv2   | Conv2d          |      36864 |  3.9% |
|  5 | vae_conv/dec/up1/res/conv2   | Conv2d          |      36864 |  3.9% |
|  6 | vae_conv/dec/bot_res/conv1   | Conv2d          |      36864 |  3.9% |
|  7 | vae_conv/dec/bot_res/conv2   | Conv2d          |      36864 |  3.9% |
|  8 | vae_conv/dec/up2/res/conv2   | Conv2d          |      36864 |  3.9% |
|  9 | vae_conv/dec/up2/res/conv1   | Conv2d          |      36864 |  3.9% |
| 10 | vae_conv/enc/proj            | Conv2d          |      36864 |  3.9% |
| 11 | vae_conv/enc/bot_res/conv1   | Conv2d          |      36864 |  3.9% |
| 12 | vae_conv/enc/down2/res/conv1 | Conv2d          |      36864 |  3.9% |
| 13 | vae_conv/enc/down1/conv      | Conv2d          |      36864 |  3.9% |
| 14 | vae_conv/enc/res0/conv2      | Conv2d          |      36864 |  3.9% |
| 15 | vae_conv/enc/res0/conv1      | Conv2d          |      36864 |  3.9% |
| 16 | vae_conv/enc/down2/res/conv2 | Conv2d          |      36864 |  3.9% |
| 17 | vae_conv/enc/down1/res/conv1 | Conv2d          |      36864 |  3.9% |
| 18 | vae_conv/enc/down2/conv      | Conv2d          |      36864 |  3.9% |
| 19 | vae_conv/enc/down1/res/conv2 | Conv2d          |      36864 |  3.9% |
| 20 | vae_conv/dec/up1/res/conv1   | Conv2d          |      36864 |  3.9% |
+----+------------------------------+-----------------+------------+-------+

✅ Script Lua exécuté avec succès
```

### 5.2 analyze_model.lua

Rôle : analyser un checkpoint/model (RawFolder, SafeTensors, Debug JSON), afficher synthèse utile (composants, tensors, tailles, etc.).

Usage typique :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/vae_conv_run
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/model.safetensors
```

Exemple concret (sortie versionnee dans la doc) :

```bash
lua scripts/tools/analyze_model.lua --in checkpoint/vae_conv-generique/epoch_0020-final --all true --graph-format mermaid
```

Exemple de sortie generee : [docs/graphs/analyze-model-report.md](../graphs/analyze-model-report.md)

Options cles : `--in`, `--all`, `--graph-format`.

### 5.3 build_tags_vocab.lua

Rôle : construire un vocabulaire de tags à partir des fichiers `.txt` d'un dataset.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/build_tags_vocab.lua -- \
  --dataset-root dataset_2 \
  --out checkpoint/tags_vocab.txt \
  --min-freq 2 --top-k 5000
```

Options utiles : `--lowercase`, `--min-freq`, `--top-k`, `--max-files`.

### 5.4 convert_checkpoint2safetensor.lua

Rôle : convertir un checkpoint RawFolder vers SafeTensors.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/convert_checkpoint2safetensor.lua -- \
  --checkpoint checkpoint/vae_conv_run \
  --out checkpoint/vae_conv_run.safetensors
```

Note : le script reconstruit un modèle depuis l'architecture du checkpoint avant de charger puis de sauvegarder.

### 5.5 convert_safetensors2raw_folder.lua

Rôle : convertir un checkpoint SafeTensors vers RawFolder.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/convert_safetensors2raw_folder.lua -- \
  --in checkpoint/model.safetensors \
  --out checkpoint/model_raw
```

### 5.6 show-graph.lua

Rôle : générer un rapport HTML interactif (Chart.js) à partir des CSV d'entraînement.

Usage typique :

```bash
./bin/mimir --lua scripts/tools/show-graph.lua -- checkpoints/loss_history.csv
./bin/mimir --lua scripts/tools/show-graph.lua -- --csv-dir checkpoints --watch --out graph_report.html
```

Exemple concret (sortie versionnee dans la doc) :

```bash
./bin/mimir --lua scripts/tools/show-graph.lua -- --csv-file checkpoint/vae_conv-generique/loss_history.csv --checkpoint checkpoint/vae_conv-generique/ --validate-every-steps 100 --out docs/graphs/graph-report.html
```

Exemple de sortie generee : [docs/graphs/graph-report.html](../graphs/graph-report.html)

Options clés : `--csv`, `--csv-file`, `--csv-dir`, `--model`, `--algo`, `--checkpoint`, `--checkpoint-dir`, `--validate-every-steps`, `--out`, `--watch`, `--watch-interval`.

## 6. Commandes rapides pour un dev

Lister les archis + vérifier la nouvelle :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

Tester conversion checkpoint :

```bash
./bin/mimir --lua scripts/tools/convert_checkpoint2safetensor.lua -- --checkpoint checkpoint/my_run --out checkpoint/my_run.safetensors
./bin/mimir --lua scripts/tools/convert_safetensors2raw_folder.lua -- --in checkpoint/my_run.safetensors --out checkpoint/my_run_raw
```

Analyser un artefact :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/my_run
```

## Étapes suivantes

- [Page précédente : Contribuer](01-Contributing.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Chapitre développeur complet : Étendre le framework](03-Extending-Models-Runtimes-And-Features.md)
