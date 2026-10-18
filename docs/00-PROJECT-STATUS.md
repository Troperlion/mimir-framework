# État réel du projet

Cette page est la carte de maturité du checkout Mímir 3.5.0 au 1 octobre 2026.
Elle distingue ce qui est compilé et exploitable de ce qui est partiel, en test,
prototype ou simple placeholder. Une entrée dans le registre, un fichier source
ou un bouton d'interface ne constitue pas à lui seul une garantie de production.

## Sur cette page

- [Comment lire les statuts](#comment-lire-les-statuts)
- [Chemin recommandé](#chemin-recommandé)
- [Moteur et sous-systèmes](#moteur-et-sous-systèmes)
- [Architectures](#architectures)
- [Runtimes matériels](#runtimes-matériels)
- [Scripting et configuration](#scripting-et-configuration)
- [Données, entraînement et checkpoints](#données-entraînement-et-checkpoints)
- [Fonctionnalités en construction](#fonctionnalités-en-construction)
- [Niveaux de preuve](#niveaux-de-preuve)
- [Reproduire l'inventaire](#reproduire-linventaire)

## Comment lire les statuts

| Statut | Signification vérifiable |
| --- | --- |
| Actif | Compilé dans la configuration concernée et relié à un chemin d'exécution. |
| Exploitable | Possède une API ou un workflow utilisable, avec au moins un test ou un exemple ciblé. |
| Partiel | Fonctionne dans un sous-ensemble documenté; fallback ou opérations manquantes possibles. |
| En test | Couvert par un test récent, mais sans validation représentative sur toutes les plateformes ou toutes les charges. |
| Expérimental | API et comportement susceptibles de changer; usage de recherche conseillé. |
| Placeholder | Nom/configuration présents, mais pas de pipeline complet exécutable. |
| En construction | Sources présentes dans le workspace, mais pas encore intégrées au build ou au registre principal. |

Ces statuts sont cumulatifs. Par exemple, CUDA peut être « actif » dans un build
CUDA et néanmoins « partiel » parce que certaines opérations utilisent le CPU.

## Chemin recommandé

Le chemin le plus reproductible aujourd'hui est :

1. construire en C++17 avec Lua 5.3, OpenMP et LZ4 ;
2. garder le runtime CPU activé ;
3. créer un modèle natif via `Mimir.Model.create` ;
4. piloter le run en Lua ou avec `--conf` ;
5. sauvegarder en RawFolder ou SafeTensors ;
6. valider avec les tests CTest ciblés.

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMIMIR_ENABLE_TESTS=ON \
  -DENABLE_VULKAN=OFF \
  -DENABLE_OPENCL=OFF \
  -DENABLE_SFML=OFF
cmake --build build -j"$(nproc)"
ctest --test-dir build -R 'RegistryTest|LuaTest|SerializationTest' \
  --output-on-failure
```

Ce profil CPU minimise les dépendances optionnelles. Il ne prouve pas la
disponibilité d'un GPU, de SFML, de FFmpeg ou d'un bridge externe.

## Moteur et sous-systèmes

| Système | Statut | Ce qui est réellement utilisable | Limites principales |
| --- | --- | --- | --- |
| Graphe `Model` et layers | Actif, exploitable | Construction C++, forward/backward, paramètres nommés, routage nommé, entraînement ciblé. | La couverture backward dépend du type de layer et du runtime. |
| Registry | Actif, exploitable | Création par nom, config par défaut, alias `sd3.5`, auto-enregistrement MPK. | « enregistré » ne veut pas dire « modèle complet ». |
| Planner | Expérimental, en test | Plans legacy/static/cost, réutilisation de buffers, fusion et inspection JSON selon variables. | Équivalence et coût sont testés sur des cas ciblés, pas sur tous les graphes. |
| RuntimeRouter | Actif, en test | Classe les capacités en `Unsupported`, `HostFallback`, `Native`, `NativeOptimized`; planifie forward et backward. | Le fallback hôte peut masquer une absence de kernel GPU natif. |
| CPU | Actif, exploitable | Référence générale, OpenMP, SIMD selon compilation, forward/backward étendus. | Certains chemins restent float32 internes ou coûtent beaucoup de mémoire. |
| Mémoire | Actif, exploitable | `MemoryGuard`, allocator dynamique, scratchpads, métriques et limites explicites. | Les politiques strictes doivent être dimensionnées pour la charge réelle. |
| Sérialisation | Actif, exploitable | RawFolder, SafeTensors, DebugJson, vérification de formes/types/checksums selon format. | Un checkpoint externe doit correspondre à la topologie attendue. |
| Tokenizer/Encoder | Actif, exploitable | Tokenizer natif, BPE, séquences, encodeur et assets de checkpoint. | La compatibilité d'un tokenizer externe doit être vérifiée, pas supposée. |
| Dataset multimodal | Actif, partiel par format | Texte, image, audio et vidéo sont représentés; FFmpeg décode les médias si compilé. | Les octets compressés ne sont pas des tenseurs décodés; FFmpeg est optionnel. |
| Viz SFML/Htop | Optionnel, exploitable | Monitoring, panneaux, métriques, contrôles et snapshots quand SFML est présent. | Headless et CI construisent généralement avec SFML désactivé. |
| WebSocket API | Expérimental | Serveur Lua dans `scripts/modules/api_ws_server.lua`. | Ce n'est pas le bridge REST natif CMake. |

## Architectures

### Architectures natives enregistrées

Le registre C++ contient 26 entrées natives :

| Famille | Entrées | Statut utile |
| --- | --- | --- |
| Général | `basic_mlp` | Exploitable et adapté aux smoke tests. |
| NLP | `transformer`, `causal_lm`, `vae_text`, `vae_text_decode` | Exploitables sur leurs contrats d'entrée respectifs; `causal_lm` est le chemin decoder-only natif. |
| Vision | `vit`, `vae`, `vae_conv`, `vae_conv_decode`, `resnet`, `unet`, `mobilenet`, `vgg16`, `vgg19`, `vgg16_feat`, `patch_discriminator` | Graphes exécutables; entraînement et géométrie varient selon l'architecture. |
| Interopérabilité HF/SDXL | `hf_clip_text_encoder_1`, `hf_clip_text_encoder_2`, `hf_vae_decoder`, `hf_sdxl_transformer_block`, `external_safetensors_base` | Les quatre premiers construisent des composants exécutables; `external_safetensors_base` reflète des clés et n'est pas un modèle autonome. |
| Génératif | `diffusion`, `cond_diffusion`, `gan_latent`, `lumen_diffusion` | Baselines natives exploitables; `lumen_diffusion` fournit le pipeline latent texte-vers-image avec VAE gelé. |
| Placeholder | `sd3_5` | Registre et squelette seulement; ne pas annoncer un pipeline SD3.5 complet. |

### Architectures MPK livrées

`_archi/` fournit `deeplab`, `r_cnn`, `ssd` et `yolo`. Elles démontrent le
format MPK et la composition avec des backbones natifs. Elles restent des
prototypes : les étapes spécialisées comme propositions ROI, priors, décodage
de détection ou convolution dilatée complète ne sont pas toutes exécutées par
le graphe MPK. La NMS native peut compléter certains pipelines, mais ne remplace
pas leur pré/post-traitement.

Pour afficher l'inventaire réellement chargé :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --show-archs
```

## Runtimes matériels

| Backend | Activation | Statut réel |
| --- | --- | --- |
| CPU | `ENABLE_OPENMP`, `ENABLE_SIMD` | Référence et fallback recommandé. Les capacités sont déclarées nativement par type de layer. |
| CUDA | `-DENABLE_CUDA=ON` | Kernels natifs pour les familles CPU exécutables et NMS forward/backward ; backward Lambda indisponible. Nécessite CUDA/cuBLAS et du matériel. |
| ROCm | `-DENABLE_ROCM=ON` | Partiel, symétrique à plusieurs chemins CUDA; nécessite HIP/rocBLAS et du matériel compatible. |
| Vulkan | `-DENABLE_VULKAN=ON` | 72 types forward / 73 backward natifs, avec kernels de référence séquentiels pour les nouvelles opérations. Configuration et formes vérifiées. |
| OpenCL | `-DENABLE_OPENCL=ON` | 72 types forward / 73 backward natifs, avec kernels de référence séquentiels pour les nouvelles opérations. Configuration et formes vérifiées. |
| FPGA iCESugar Pro | sources présentes, non intégrées au routeur global | En construction : protocole, runtime C++ et HDL existent; les composants isolés ont des cibles CTest, mais les sources FPGA ne sont pas encore dans `MIMIR_SOURCES` et `FpgaRuntime::supports*LayerType` ne publie pas encore de couverture de graphe générale. |

NMS possède un forward de sélection d'indices et un backward stop-gradient
sur CPU, CUDA, Vulkan et OpenCL. Les gradients de boîtes, scores et classes
sont nuls ; les losses d'entraînement doivent utiliser les prédictions avant
NMS. Les tests CPU et les comparaisons sur GPU AMD OpenCL/Vulkan passent ;
l'exécution NMS CUDA sur GPU NVIDIA reste à valider.

La page [Accélération GPU](05-Advanced/05-GPU-Acceleration.md) détaille les
familles d'opérations. Pour toute mesure, distinguez toujours : compilé,
initialisé, capacité native, fallback hôte, et backend effectivement exécuté.

## Scripting et configuration

| Surface | Statut | Contrat |
| --- | --- | --- |
| Lua | Actif, principal | Binding natif, table `Mimir`, global standard `arg`, modules sous `scripts/modules/`. |
| JSON `--conf` | Actif | Injecte `CONF`, `CONF_PATH`, `CONF_DIR` et les arguments; chaque script obtient une nouvelle VM Lua. |
| Tâches `--run` | Actif | Sélectionne `tasks.<nom>.lua` ou `tasks.<nom>.run.lua`. |
| `env` dans JSON | Actif | Appliqué au processus après les overrides et avant le run; remplace le shell. `OMP_NUM_THREADS` met aussi à jour OpenMP. |
| JavaScript | Optionnel, expérimental | Compilé par défaut si l'option reste active; nécessite Node.js au runtime. |
| Rust | Optionnel, expérimental | Nécessite `rust-script`; passe par le contrat de bridge. |
| C# | Optionnel, expérimental | Nécessite `dotnet-script` ou un outil compatible. |
| REST natif | Non supporté | `ENABLE_SCRIPTING_REST=ON` est explicitement forcé à `OFF` par CMake. |
| `Mimir.Layers.*` standalone | Placeholder | Les huit fonctions existent pour compatibilité mais retournent une erreur « non implémenté »; utilisez un modèle et `forward()`. |

Il n'existe pas de `Mimir.Args`. Utilisez :

```lua
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg)
```

## Données, entraînement et checkpoints

- Les scripts d'entraînement sont des programmes spécialisés, pas une boucle
  universelle magique. Lisez leurs options avec `--help` avant un long run.
- `train_vae_conv.lua` gère reconstruction, latent stochastique, KL/warmup,
  validation holdout, visualisation et reprise selon ses options actuelles.
- Lumen possède des sources et scripts de travail, mais son modèle C++ n'est pas
  encore enregistré ni ajouté au build principal : statut « en construction ».
- Un dataset n'est jamais téléchargé automatiquement. Les configs publiques
  utilisent des chemins relatifs sous `datasets/`, dossier ignoré par Git.
- Les checkpoints, caches, CSV, snapshots et rapports sont des sorties locales
  ignorées. Ils ne doivent pas être ajoutés à une pull request.
- `bf16`/`fp16` sont des formats supportés par le système de dtype et la
  sérialisation; cela ne garantit pas que chaque kernel calcule nativement dans
  ce dtype. Vérifiez le runtime et le chemin d'opération.

## Fonctionnalités en construction

Les éléments suivants sont utiles pour contribuer mais ne doivent pas être
présentés comme stables :

- modèle et scripts Lumen latent diffusion, non encore intégrés au registre/build ;
- runtime FPGA iCESugar Pro, protocole et validation, non encore routés comme
  backend de graphe général ;
- planner cost/static, fusion et résidence device, encore sous tests ciblés ;
- prototypes MPK de détection/segmentation ;
- bridge WebSocket Lua et bridges de langages externes ;
- `sd3_5` et `Mimir.Layers` standalone, qui sont des placeholders explicites.

## Niveaux de preuve

Une affirmation de support devrait citer au moins l'un des niveaux suivants :

1. **Source** : symbole présent et relié dans CMake/registre.
2. **Construction** : cible compilée dans une configuration précise.
3. **Test** : test CTest ou smoke réellement exécuté.
4. **Matériel** : exécution observée sur le backend concerné.
5. **Workflow** : entraînement/inférence complet avec données et checkpoint.

Un test CPU ne prouve pas CUDA, ROCm, Vulkan, OpenCL ou FPGA. Une compilation
SFML ne prouve pas que toutes les cartes soient visibles dans une fenêtre
réelle. Un chargement partiel de checkpoint ne prouve pas une inférence fidèle.

## Reproduire l'inventaire

Depuis la racine du dépôt :

```bash
# Registre et dtypes chargés
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- --show-archs

# Tests réellement déclarés dans le build courant
ctest --test-dir build -N

# API Lua C++ contre le stub EmmyLua
./scripts/tools/verify_api_sync.sh
./scripts/tools/verify_docs.py

# Références locales accidentelles avant publication
rg -n '/home/|/run/media/|Mimir\.Args|src/LuaScripting\.cpp' \
  README.md docs configs scripts

# Cohérence Git et JSON
git diff --check
jq empty configs/*.json
```

Les résultats sont propres au checkout et aux options CMake courantes. Mettez
cette page à jour dès qu'un système change de niveau de maturité.
