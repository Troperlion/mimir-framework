# Scripting Lua

Ce chapitre explique **comment écrire des scripts Lua “robustes”** pour Mímir : démarrage, arguments, conventions, patterns, et pièges fréquents.

## 1) Lancer un script (et passer des args)

Un script se lance via :

```bash
./bin/mimir --lua path/to/script.lua -- [args...]
```

Le `--` est important : il sépare les flags du binaire `mimir` des arguments passés au script Lua.

### Ce que le runtime fournit

| Nom | Contenu | À quoi ça sert |
| --- | --- | --- |
| `arg` | `arg[0]=script`, `arg[1..n]=args` | compat Lua classique |
| `Mimir.Args` | copie de `arg` | éviter ambiguïtés/collisions |
| `Mimir` | table API (`Mimir.Model`, `Mimir.Serialization`, …) | accès “moderne” |
| globals utilitaires | `log`, `read_json`, `write_json` | scripts rapides |
| aliases globaux | `model`, `architectures`, … | rétrocompat + confort |

Référence complète des globals/aliases : `docs/03-API-Reference/19-Globals.md`.

## 2) Template de script recommandé

Ce template couvre 80% des scripts (benchmark/test/training) :

```lua
local args = dofile("scripts/modules/args.lua")

local opts = args.parse(Mimir.Args, {

  {"--seed", "0"},
  {"--ram", "10"},
})

pcall(Mimir.MemoryGuard.setLimit, tonumber(opts["--ram"]))
local ok, err = Mimir.Allocator.configure({
  max_ram_gb = tonumber(opts["--ram"]),
  enable_compression = true,
  swap_strategy = "lru",
})
assert(ok ~= false, err)

local cfg, cfg_err = Mimir.Architectures.default_config("transformer")
assert(cfg, cfg_err)
cfg.seq_len = 64
cfg.vocab_size = 2000

assert(Mimir.Model.create("transformer", cfg))
assert(Mimir.Model.build())
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", tonumber(opts["--seed"]) or 0))

local ids = {}
for i = 1, cfg.seq_len do ids[i] = 1 end
local out = Mimir.Model.forward({ __input__ = ids }, false)
assert(out, "forward failed")
log("out_len=" .. tostring(#out))
```

## 3) Patterns utiles (pratiques en vrai)

### Charger une config JSON

```lua
local cfg, err = read_json("config.json")
assert(cfg, err)
```

Conseil : garde une `config.json` “humaine” et génère des configs plus détaillées via `Mimir.Architectures.default_config(...)` + overrides.

### Mode `--conf` : conf injectée + scripts pilotés par JSON

Le binaire supporte un mode `--conf <config.json>` qui exécute une liste de scripts définie dans `lua.scripts` (ou `run.lua.scripts`).

Dans ce mode, le runtime injecte :

- `CONF` : table Lua correspondant au JSON complet (la conf “full”)
- `CONF_PATH` : chemin absolu du JSON chargé
- `CONF_DIR` : dossier parent de la conf

Cas d’usage typique : un script peut créer un modèle directement depuis `CONF` :

```lua
if CONF then
  local ok, arch_or_err = Mimir.Model.create_from_config(CONF)
  assert(ok, arch_or_err)
  log("arch=" .. tostring(arch_or_err))
end
```

Note : en mode `--conf`, les arguments `arg` du script sont ceux définis par la conf (champ `args`).

### Modules `scripts/modules/*`

Ces modules ne sont pas “magiques” : ils sont juste là pour uniformiser les scripts.

- `scripts/modules/args.lua` : parse d’arguments style `--key value`
- `scripts/modules/checkpoint_resume.lua` : reprise automatique
- `scripts/modules/base_tokenizer.lua` : base vocab stable

## 4) Conseils

- Utilise `Mimir.*` pour un projet long terme (lisible, explicite). Les aliases globaux (`model`, `architectures`, …) sont OK pour des scripts rapides.
- Centralise le setup mémoire au début (MemoryGuard + Allocator). Ça rend les erreurs reproductibles et évite les OOM “bizarres”.
