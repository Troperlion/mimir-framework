# Correspondance API Lua vers C++

Cette page reflète `LuaScripting::registerAPI()` dans
`src/scriptings/Lua/luaScripting/LuaScripting.cpp`. Les implémentations sont
réparties dans `LuaScriptingModelAndRegistry.cpp`,
`LuaScriptingRuntimeAndViz.cpp` et `LuaScriptingTokenizerDataset.cpp`.

Le stub IDE synchronisé est `mimir-api.lua`.

## Modules enregistrés

| Namespace Lua | Surface | Bindings principaux |
| --- | --- | --- |
| `Mimir.Model` | cycle de vie, graphe, forward/backward, optimiseur, dtype | `lua_createModel`, `lua_allocateParams`, `lua_forwardPass`, `lua_backwardPass`, `lua_optimizerStep`, `lua_modelDType` |
| `Mimir.Architectures` | registre, configs, informations, dtypes | `lua_archAvailable`, `lua_archDefaultConfig`, `lua_archInfo`, `lua_archDtypes` |
| `Mimir.Layers` | inspection du graphe courant | `lua_layersAvailable`, `lua_layersByType`, filtres `lua_compute*` |
| `Mimir.Serialization` | save/load/détection/DebugJSON | `lua_saveCheckpoint`, `lua_loadCheckpoint`, `lua_detectFormat`, `lua_saveEnhancedDebugJson` |
| `Mimir.Checkpoint` | alias legacy save/load | `lua_saveCheckpoint`, `lua_loadCheckpoint` |
| `Mimir.Tokenizer` | vocabulaire, BPE, tokens spéciaux, analyse | bindings `lua_*Tokenizer`, `lua_*Token`, `lua_*BPE` |
| `Mimir.Dataset` | chargement, accès, séquences | `lua_loadDataset`, `lua_getDataset`, `lua_prepareSequences` |
| `Mimir.Database` | builder de chargement avec cache | `lua_databaseLoad` |
| `Mimir.IO` | image RGB et contrôle des logs | `lua_readImageRGBU8`, `lua_setStdoutLogSuppressed` |
| `Mimir.Memory` | AdvancedRAMManager | `lua_memoryConfig`, `lua_memoryGetStats`, `lua_memoryClear` |
| `Mimir.Guard`, `Mimir.MemoryGuard` | limites et statistiques mémoire | bindings `lua_guard*`, `lua_memoryguard*` |
| `Mimir.Allocator` | allocateur dynamique | `lua_allocatorConfigure`, `lua_allocatorGetStats` |
| `Mimir.Htop` | monitoring asynchrone | bindings `lua_htop*` |
| `Mimir.Viz` | visualisation | bindings `lua_viz*` |

`Mimir.NeuroPulse`, `Mimir.visualiser` et le global `readImageRGBU8` ne sont pas
enregistrés dans l'API actuelle.

## `Mimir.Model`

| Appel Lua | Binding C++ | Contrat synthétique |
| --- | --- | --- |
| `create(name, cfg?)` | `lua_createModel` | construit via le registre |
| `create_empty(name, cfg?)` | `lua_createEmptyModel` | modèle vide pour graphe importé |
| `create_from_config(cfg)` | `lua_createModelFromConfig` | choisit l'architecture depuis une config complète |
| `build()` | `lua_buildModel` | no-op moderne, retourne le nombre de paramètres |
| `allocate_params()` | `lua_allocateParams` | `(true, count)` ou `(false, err)` |
| `init_weights(method?, seed?)` | `lua_initWeights` | initialise les poids |
| `get_layers()`, `clear_layers()` | `lua_getModelLayers`, `lua_clearModelLayers` | inspection/modification du graphe |
| `push_layer(...)`, `set_layer_io(...)` | `lua_pushLayer`, `lua_setLayerIO` | construction nodale bas niveau |
| `forward(input, training?)` | `lua_forwardPass` | liste int/float ou tenseurs nommés |
| `backward(gradient)` | `lua_backwardPass` | rétropropagation |
| `zero_grads()`, `get_gradients()` | `lua_zeroGradients`, `lua_getGradients` | gestion des gradients |
| `optimizer_step(lr, type?)` | `lua_optimizerStep` | SGD, Adam, AdamW, Lion, Adafactor, RAdam, NAdam, RMSprop ou LAMB |
| `dtype()`, `dtype(name)` | `lua_modelDType` | lit/fixe le dtype du modèle |
| `hardware_caps()`, `set_hardware(bool)` | `lua_getHardwareCaps`, `lua_setHardwareAccel` | capacités et activation matérielle |

Les helpers `forward_prompt_image_seed`, `encode_prompt`, `get_optimizer`,
`set_optimizer` et `reset_optimizer_state` ne sont pas enregistrés.

## `Mimir.Layers`

`Mimir.Layers.available()` retourne les types canoniques de
`LayerRegistry::get_all_supported_types()`. Pour chaque nom retourné, le
runtime crée une fonction d'inspection :

```lua
local types = Mimir.Layers.available()
local convs = Mimir.Layers.Conv2d()
local norms = Mimir.Layers.RMSNorm()
local same = Mimir.Layers.by_type("RMSNorm")
```

Les fonctions lowercase sont des filtres de compatibilité, pas des kernels
appelables depuis Lua :

- `conv2d`, `linear`, `maxpool2d`, `avgpool2d` ;
- `activation`, `batchnorm`, `layernorm`, `attention`.

Elles retournent les layers correspondants du modèle courant.

## `Mimir.IO`

```lua
local decoded, err = Mimir.IO.read_image_rgb_u8("image.png", 256, 256)
assert(decoded, err)
-- decoded = { image = {...}, width = 256, height = 256, channels = 3 }
```

`readImageRGBU8` est l'alias camelCase dans `Mimir.IO`. Les fonctions
`suppress_stdout_logs` et `suppressStdoutLogs` partagent le même binding.

## Sérialisation

| Appel | Formats/retour |
| --- | --- |
| `Serialization.save(path, format?, options?)` | `safetensors`, `raw_folder`, `debug_json` et alias |
| `Serialization.load(path, format?, options?)` | format explicite ou auto-détection |
| `Serialization.detect_format(path)` | `SAFETENSORS`, `RAWFOLDER`, `DEBUGJSON` ou `(nil, err)` |
| `Serialization.save_enhanced_debug(path, options?)` | DebugJSON enrichi |

Les options de chargement effectivement lues sont `load_tokenizer`,
`load_encoder`, `load_optimizer`, `strict_mode`, `validate_checksums`,
`mapping_json` et `tensor_mapping_json`.

## Fonctions et alias globaux

Fonctions : `log`, `read_json`, `write_json`.

Alias de tables : `model`, `architectures`, `tokenizer`, `dataset`, `Memory`,
`MemoryGuard`, `Allocator`, `htop`, `viz`. `Mimir.model` est aussi un alias de
`Mimir.Model`.

Il n'existe pas de global `Mimir.Args` : utilisez
`dofile(ROOTWORK.."/scripts/modules/args.lua")` puis `Args.parse(arg)`.

## Étapes suivantes

- [Vue d'ensemble de l'API](00-API-Overview.md)
- [Stub EmmyLua](../../mimir-api.lua)
- [`Mimir.IO`](21-IO.md)
