-- Petit parseur de flags pour scripts Lua lancés via `./bin/mimir --lua ...`.
-- Suppose que le binaire remplit la table globale `arg`.
--
-- Support:
--   --k v
--   --k=v
--   --flag (bool=true)
--   --no-flag (bool=false)
--   -- (fin des options)

---@class MimirArgsModule
local M = {}
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
local caller_script_path = Help.find_script_from_stack(3, 24)

Help.auto_exit_help({
  script_path = caller_script_path,
  common_flags = {
    "--help, -h : affiche cette aide",
    "--htop/--tux : active uniquement le moniteur TUI",
    "--viz : active uniquement la visualisation",
    "--no-htop, --no-viz : desactive explicitement un mode",
    "--csv, --csv-path, --csv-enabled : options export CSV",
    "--override key=value : override de configuration (repete)",
  },
})

local function maybe_start_htop(opts)
  if not opts then return end
  if rawget(_G, "__MIMIR_MONITOR_STARTED") == true then
    return
  end
  local requested_htop = (opts.htop == true) or (opts.tux == true)
  local requested_viz = (opts.viz == true) or (opts.visualiser == true) or (opts.visualizer == true) or (opts["enable-viz"] == true)

  local enable_viz = requested_viz
  local enable_htop = requested_htop

  if not (enable_htop or enable_viz) then return end

  local Mimir = rawget(_G, "Mimir")
  if not (Mimir and Mimir.Htop and Mimir.Htop.create) then
    -- Sous `lua` pur (hors binaire mimir), Mimir.* est absent.
    -- On ne doit pas casser le parse si l'utilisateur a juste passé `--viz`.
    if requested_htop then
      error("Option --htop/--tux demandée, mais Mimir.Htop.create est indisponible")
    end
    return
  end

  local cfg = {
    enable_viz = requested_viz,
    enable_htop = enable_htop,
    defer_htop = enable_htop,
    viz_config = {
      visualization = {
        enabled = true,
        window_title = "Mimir | Entraînement en cours ....",
      },
    },
  }

  -- Overrides (optionnels)
  -- Ex:
  --   --viz-title "Mon run" --viz-width 1600 --viz-height 900 --viz-fps 60
  --   --viz-hide-activation-blocks false
  local function as_int(v)
    local n = tonumber(v)
    if n == nil then return nil end
    return math.floor(n)
  end
  local function as_bool(v)
    if v == nil then return nil end
    if v == true or v == false then return v end
    local s = tostring(v):lower()
    if s == "1" or s == "true" or s == "on" or s == "yes" then return true end
    if s == "0" or s == "false" or s == "off" or s == "no" then return false end
    return nil
  end

  local title = opts["viz-title"] or opts["viz-window-title"] or opts["viz-titlebar"]
  if title ~= nil and title ~= true then
    cfg.viz_config.visualization.window_title = tostring(title)
  end

  local w = as_int(opts["viz-width"] or opts["viz-w"])
  local h = as_int(opts["viz-height"] or opts["viz-h"])
  if w and w > 0 then cfg.viz_config.visualization.window_width = w end
  if h and h > 0 then cfg.viz_config.visualization.window_height = h end

  local fps = as_int(opts["viz-fps"])
  if fps and fps > 0 then cfg.viz_config.visualization.fps_limit = fps end

  local hide = as_bool(opts["viz-hide-activation-blocks"])
  if hide ~= nil then cfg.viz_config.visualization.hide_activation_blocks = hide end

  local hide_norm = as_bool(opts["viz-hide-normalisation-blocks"])
  if hide_norm ~= nil then cfg.viz_config.visualization.hide_normalisation_blocks = hide_norm end

  if opts.csv ~= nil then cfg.csv = opts.csv end
  if opts["csv-enabled"] ~= nil then cfg.csv_enabled = opts["csv-enabled"] end
  if opts["csv-path"] ~= nil then cfg.csv_path = tostring(opts["csv-path"]) end
  if opts["csv-file"] ~= nil and cfg.csv_path == nil then cfg.csv_file = tostring(opts["csv-file"]) end

  do
    local l = rawget(_G, "log")
    local msg = "[args] monitor configured: htop=" .. tostring(enable_htop) .. " viz=" .. tostring(requested_viz)
    if type(l) == "function" then
      l(msg)
    else
      print(msg)
    end
  end

  local ok, err = Mimir.Htop.create(cfg)
  if ok == false then
    error("Mimir.Htop.create a échoué: " .. tostring(err))
  end
  rawset(_G, "__MIMIR_MONITOR_STARTED", true)
  -- `Htop.create` peut retourner (true, "Viz init failed: ...") sans casser le run.
  -- Avant, ce message était silencieux => impression que la viz/htop ne démarre pas.
  if err ~= nil and tostring(err) ~= "" then
    local msg = tostring(err)
    local l = rawget(_G, "log")
    if type(l) == "function" then
      l("⚠️  " .. msg)
    else
      print("[mimir][warn] " .. msg)
    end
  end
end


local function normalize_key(k)
  k = tostring(k or "")
  k = k:gsub("^%-%-+", "")
  -- Normalise les underscores en tirets (--csv_path == --csv-path, etc.)
  k = k:gsub("_", "-")
  return k
end

local function push_opt(opts, key, value)
  if key == "override" then
    if type(opts.override) ~= "table" then
      opts.override = {}
    end
    opts.override[#opts.override + 1] = value
    return
  end
  opts[key] = value
end

local function parse_override_value(raw)
  if raw == nil then return nil end
  if raw == "true" then return true end
  if raw == "false" then return false end
  if raw == "null" then return nil end
  local n = tonumber(raw)
  if n ~= nil then return n end
  local s = tostring(raw)
  -- Support minimal des strings JSON ("...") sans eval arbitraire.
  if #s >= 2 and s:sub(1, 1) == '"' and s:sub(-1) == '"' then
    s = s:sub(2, -2)
    s = s:gsub('\\"', '"')
    s = s:gsub('\\n', '\n')
    s = s:gsub('\\t', '\t')
    s = s:gsub('\\r', '\r')
  end
  return s
end

local function apply_override(cfg, expr)
  if type(cfg) ~= "table" then cfg = {} end
  expr = tostring(expr or "")
  local eq = expr:find("=", 1, true)
  if not eq or eq == 1 or eq == #expr then
    error("override invalide (attendu: path.to.key=value): " .. expr)
  end
  local path = expr:sub(1, eq - 1)
  local raw_value = expr:sub(eq + 1)
  local value = parse_override_value(raw_value)
  local keys = {}
  for part in string.gmatch(path, "[^%.]+") do
    keys[#keys + 1] = part
  end
  if #keys == 0 then
    error("override invalide (path vide): " .. expr)
  end
  local cur = cfg
  for i = 1, #keys - 1 do
    local k = keys[i]
    if k == "" then error("override invalide (segment vide): " .. expr) end
    if type(cur[k]) ~= "table" then
      cur[k] = {}
    end
    cur = cur[k]
  end
  local leaf = keys[#keys]
  if leaf == "" then error("override invalide (clé finale vide): " .. expr) end
  cur[leaf] = value
  return cfg
end

---@param argv? string[]
---@return MimirScriptArgs opts
---@return string[] positional
function M.parse(argv)
  local opts = {}
  local pos = {}
  local raw_args = argv or {}
  local a = {}

  -- Certains copier-coller (éditeur riche, navigateur, messagerie) remplacent
  -- l'espace ASCII entre une option et sa valeur par NBSP (U+00A0) ou NNBSP
  -- (U+202F). Le shell ne les considère pas comme des séparateurs et transmet
  -- alors un seul argument, par exemple "--image-h<U+202F>1024". On répare
  -- uniquement ce cas précis pour ne pas modifier les valeurs ordinaires.
  for _, raw in ipairs(raw_args) do
    local v = raw
    if type(v) == "string" and v:sub(1, 2) == "--" then
      local key, value = v:match("^(%-%-[^\194\226]+)\194\160(.+)$") -- U+00A0
      if key == nil then
        key, value = v:match("^(%-%-[^\194\226]+)\226\128\175(.+)$") -- U+202F
      end
      if key ~= nil then
        a[#a + 1] = key
        a[#a + 1] = value
      else
        a[#a + 1] = v
      end
    else
      a[#a + 1] = v
    end
  end

  local i = 1
  while i <= #a do
    local v = a[i]
    if v == "--" then
      -- Séparateur courant: `mimir --lua script.lua -- --flag ...`
      -- Ici on l'ignore et on continue à parser les flags.
      i = i + 1
      goto continue
    end

    if type(v) == "string" and v:sub(1, 2) == "--" then
      local k, eqv = v:match("^%-%-([^=]+)=(.*)$")
      if k then
        push_opt(opts, normalize_key(k), eqv)
      else
        local nk = normalize_key(v)
        if nk:sub(1, 3) == "no-" then
          push_opt(opts, nk:sub(4), false)
        else
          local nextv = a[i + 1]
          if nextv ~= nil and not (type(nextv) == "string" and nextv:sub(1, 2) == "--") then
            push_opt(opts, nk, nextv)
            i = i + 1
          else
            push_opt(opts, nk, true)
          end
        end
      end
    else
      pos[#pos + 1] = v
    end

    i = i + 1
    ::continue::
  end

  -- Option commune: démarrer l'interface TUI (HtopDisplay) si demandée.
  -- Ex:
  --   --htop (ou --tux)
  --   --viz (active uniquement la viz; combiner avec --htop pour les deux)
  --   --csv / --no-csv, --csv-path=..., --csv-enabled
  local ok, err = pcall(maybe_start_htop, opts)
  if not ok then
    error(err)
  end
  return opts, pos
end

---Applique des overrides (si présents) sur une config table.
---Supporte `--override key=value` répété, avec clés imbriquées `a.b.c=...`.
---Types supportés: bool (`true/false`), number, nil (`null`), string.
---@param cfg table|nil @Table config (sera mutée)
---@param opts table|nil @Options parse() (doit contenir opts.override)
---@return table cfg
function M.apply_overrides(cfg, opts)
  cfg = cfg or {}
  if not opts then return cfg end

  local o = opts.override
  if o == nil then return cfg end

  -- Compat: si override n'est pas une liste (vieux parseur), on le force.
  local list = o
  if type(o) ~= "table" then
    list = { o }
  end

  for _, expr in ipairs(list) do
    cfg = apply_override(cfg, expr)
  end
  return cfg
end

---@param opts MimirScriptArgs
---@param key string
---@return boolean
function M.has(opts, key)
  return opts and opts[key] == true
end

function M.get_str(opts, key, default)
  local v = opts and opts[key]
  if v == nil then return default end
  if v == true then return default end
  return tostring(v)
end

function M.get_num(opts, key, default)
  local v = opts and opts[key]
  if v == nil then return default end
  if v == true then return default end
  local n = tonumber(v)
  if n == nil then return default end
  return n
end

function M.get_int(opts, key, default)
  local n = M.get_num(opts, key, default)
  if n == nil then return default end
  return math.floor(n)
end

function M.get_bool(opts, key, default)
  local v = opts and opts[key]
  if v == nil then return default end
  if v == true then return true end
  if v == false then return false end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "on" or v == "yes" then return true end
  if v == "0" or v == "false" or v == "off" or v == "no" then return false end
  return default
end

-- Flags génériques: validation (consommés côté C++ via cfg.validate_*)
--
-- Supporte:
--   --validate-every / --validate-every-steps
--   --validate-items
--   --validate-holdout-frac
--   --validate-holdout-items
--   --validate-holdout / --no-validate-holdout
--   --validate-save-debug / --no-validate-save-debug
--
-- Utilisation:
--   Args.apply_validation_config(cfg, opts)
--   Args.apply_validation_config(cfg, opts, { validate_items = 4 })
---@param cfg table|nil @Table config modèle (sera mutée)
---@param opts table|nil @Options parse() (table de flags)
---@param defaults table|nil @Defaults optionnels (ex: { validate_items = 8 })
---@return table cfg
function M.apply_validation_config(cfg, opts, defaults)
  cfg = cfg or {}
  defaults = defaults or {}

  local function def(k, fallback)
    if defaults[k] ~= nil then return defaults[k] end
    return fallback
  end

  local every = M.get_int(opts, "validate-every", nil)
  if every == nil then
    every = M.get_int(opts, "validate-every-steps", def("validate_every_steps", 0))
  end
  cfg.validate_every_steps = every

  cfg.validate_items = M.get_int(opts, "validate-items", def("validate_items", 8))
  cfg.validate_holdout_frac = M.get_num(opts, "validate-holdout-frac", def("validate_holdout_frac", 0.01))
  cfg.validate_holdout_items = M.get_int(opts, "validate-holdout-items", def("validate_holdout_items", 0))
  cfg.validate_holdout = M.get_bool(opts, "validate-holdout", def("validate_holdout", true))
  cfg.validate_save_debug = M.get_bool(opts, "validate-save-debug", def("validate_save_debug", true))
  return cfg
end

local opts = M.parse(arg)


function M.opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

function M.opt_int(k, d)
  return math.floor(M.opt_num(k, d))
end

function M.opt_str(k, d)
  local v = opts[k]
  if v == nil or v == true then return d end
  return tostring(v)
end

function M.opt_bool(k, d)
  local v = opts[k]
  if v == nil then return d end
  if v == true or v == false then return v end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

return M
