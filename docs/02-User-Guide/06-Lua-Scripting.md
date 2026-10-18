# Scripting Lua

Écrire des scripts Lua robustes et lisibles.

**Public concerné :** Débutant script et contributeur.

> **Prérequis**
>
> Connaître les bases de Lua (variables, tables, fonctions).


Ce chapitre explique **comment écrire des scripts Lua “robustes”** pour Mímir : démarrage, arguments, conventions, patterns, et pièges fréquents.

## Etat des autres bindings

- Lua est le chemin supporte complet pour le scripting metier.
- Les bridges JS, C# et Rust existent, mais ils sont encore incomplets.
- Si vous lisez ce guide pour migrer un script, considere Lua comme la reference; les autres langages doivent etre valides au cas par cas.

## 1) Lancer un script (et passer des args)

Un script se lance via :

```bash
./bin/mimir --lua path/to/script.lua -- [args...]
```

Le `--` est important : il sépare les flags du binaire `mimir` des arguments passés au script Lua.

Pour afficher l'aide d'un script Lua:

```bash
./bin/mimir --lua path/to/script.lua -- --help
```

Le support `--help` est unifie sur les scripts executables du repo. La sortie contient la description du script et ses options disponibles.

### Ce que le runtime fournit

| Nom | Contenu | À quoi ça sert |
| --- | --- | --- |
| `arg` | `arg[0]=script`, `arg[1..n]=args` | compat Lua classique |
| `Mimir` | table API (`Mimir.Model`, `Mimir.Serialization`, …) | accès “moderne” |
| globals utilitaires | `log`, `read_json`, `write_json` | scripts rapides |
| aliases globaux | `model`, `architectures`, … | rétrocompat + confort |

Référence complète des globals/aliases : `docs/03-API-Reference/19-Globals.md`.

## 2) Template de script recommandé

Ce template couvre 80% des scripts (benchmark/test/training) :

```lua
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local seed = Args.get_int(opts, "seed", 0)

local cfg, cfg_err = Mimir.Architectures.default_config("transformer")
assert(cfg, cfg_err)
cfg.seq_len = 64
cfg.vocab_size = 2000

assert(Mimir.Model.create("transformer", cfg))
-- Model.build() n'est plus nécessaire (v3.0+: network construit automatiquement)
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", seed))

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

### Modules `scripts/modules/*`

Ces modules ne sont pas “magiques” : ils sont juste là pour uniformiser les scripts.

- `scripts/modules/args.lua` : parse d’arguments style `--key value`
- `scripts/modules/help_cli.lua` : aide CLI commune (`--help`) pour scripts executables
- `scripts/modules/checkpoint_resume.lua` : reprise automatique
- `scripts/modules/base_tokenizer.lua` : base vocab stable
- `scripts/modules/fs.lua` : helpers filesystem cross-plateforme

### Filesystem cross-plateforme (obligatoire)

Pour garantir la portabilité Linux/Windows, les scripts Lua doivent passer par `FS`:

```lua
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
FS.mkdir_p("checkpoint/my_run")
local arch = FS.join("checkpoint", "my_run", "model", "architecture.json")
if FS.file_exists(arch) then
  -- ...
end
```

À éviter dans les scripts métier:

- `os.execute("mkdir -p ...")`
- `io.popen("ls ...")`
- détection via `test -d`, `stat`, `find` shell

Ces patterns sont non portables et doivent être remplacés par `FS.*`.

## 4) Conseils

- Utilise `Mimir.*` pour un projet long terme (lisible, explicite). Les aliases globaux (`model`, `architectures`, …) sont OK pour des scripts rapides.
- Centralise le setup mémoire au début (MemoryGuard + Allocator). Ça rend les erreurs reproductibles et évite les OOM “bizarres”.

## Étapes suivantes

- [Page précédente : Inférence](05-Inference.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tokenizer & ConditioningEncoder](07-Tokenizer-Encoder.md)
