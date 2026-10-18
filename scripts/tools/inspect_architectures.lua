---@diagnostic disable: undefined-global, undefined-field

-- Outil CLI d'inspection des architectures du framework.
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/inspect_architectures.lua -- [options]
--
-- Options:
--   -a, --show-archs        Liste les architectures disponibles (+ dtypes)
--   -l, --list <arch>       Sélectionne une architecture par son nom
--   --extern-model <path>   Inspecte un fichier SafeTensors externe
--   -e, --export <path>     Exporte l'architecture sélectionnée vers un checkpoint
--   -p, --params            Affiche les paramètres de l'archi sélectionnée
--   --layers                Affiche les layers de l'archi sélectionnée
--   --stats                 Affiche les statistiques théoriques (params/flops)
--   --ops                   Inventorie les layer types / ops observés dans les graphes du framework
--   --runtime               Affiche les capacités runtime exposées au Lua API
--   -d, --dtypes            Liste les dtypes pris en charge par le framework
--   --optimizers            Liste les algorithmes d'optimisation disponibles
--   --recon-losses          Liste les fonctions de reconstruction disponibles
--   --all                   Tous les rapports (toutes architectures sans --list), sans export
--   --json                  Export JSON complet du registre (toutes archs + params)
--   -h, --help              Affiche cette aide

-- Les rapports doivent rester propres et sans préfixes runtime.
-- On écrit donc directement sur stdout au lieu d'utiliser le logger Mimir.
local ToolHelp = dofile(ROOTWORK.."/scripts/modules/tools_help.lua")
ToolHelp.show("inspect_architectures")

local function log(...)
  local out = {}
  for i = 1, select("#", ...) do
    out[#out + 1] = tostring(select(i, ...))
  end
  io.stdout:write(table.concat(out, " ") .. "\n")
end

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

-- ---------------------------------------------------------------------------
-- Couleurs ANSI (désactivées si NO_COLOR ou sortie non-TTY)
-- ---------------------------------------------------------------------------
local COLOR_ENABLED = true
do
  if os.getenv("NO_COLOR") ~= nil then
    COLOR_ENABLED = false
  end
end

local C = {
  reset = "\27[0m",
  bold = "\27[1m",
  dim = "\27[2m",
  red = "\27[31m",
  green = "\27[32m",
  yellow = "\27[33m",
  blue = "\27[34m",
  magenta = "\27[35m",
  cyan = "\27[36m",
  gray = "\27[90m",
}

local function colorize(s, ...)
  s = tostring(s)
  if not COLOR_ENABLED then return s end
  local codes = { ... }
  if #codes == 0 then return s end
  return table.concat(codes) .. s .. C.reset
end

-- ---------------------------------------------------------------------------
-- Helpers d'alignement (UTF-8) + tableau ASCII (style analyze_model.lua)
-- ---------------------------------------------------------------------------
local function utf8_len_safe(s)
  s = tostring(s or "")
  if type(_G.utf8) == "table" and type(_G.utf8.len) == "function" then
    local ok, n = pcall(_G.utf8.len, s)
    if ok and type(n) == "number" then return n end
  end
  local i, n, bytes = 1, 0, #s
  while i <= bytes do
    local c = s:byte(i)
    if not c then break end
    if c < 0x80 then i = i + 1
    elseif c < 0xE0 then i = i + 2
    elseif c < 0xF0 then i = i + 3
    else i = i + 4 end
    n = n + 1
  end
  return n
end

local function utf8_sub_chars(s, n_chars)
  s = tostring(s or "")
  n_chars = math.floor(tonumber(n_chars) or 0)
  if n_chars <= 0 then return "" end
  if type(_G.utf8) == "table" and type(_G.utf8.offset) == "function" then
    local ok, off = pcall(_G.utf8.offset, s, n_chars + 1)
    if ok and type(off) == "number" then return s:sub(1, off - 1) end
  end
  local i, bytes, count = 1, #s, 0
  while i <= bytes and count < n_chars do
    local c = s:byte(i)
    if not c then break end
    if c < 0x80 then i = i + 1
    elseif c < 0xE0 then i = i + 2
    elseif c < 0xF0 then i = i + 3
    else i = i + 4 end
    count = count + 1
  end
  return s:sub(1, i - 1)
end

local function trunc(s, max_len)
  s = tostring(s or "")
  max_len = math.floor(tonumber(max_len) or 80)
  if utf8_len_safe(s) <= max_len then return s end
  if max_len < 6 then return utf8_sub_chars(s, max_len) end
  return utf8_sub_chars(s, max_len - 3) .. "..."
end

local function pad_right(s, w)
  s = tostring(s or "")
  w = math.floor(tonumber(w) or utf8_len_safe(s))
  local len = utf8_len_safe(s)
  if len >= w then return s end
  return s .. string.rep(" ", w - len)
end

-- Construit un tableau ASCII coloré.
--   columns: { {key=, title=, align="left|right", max=, color=<ansi>}, ... }
--   rows:    { { [key]=value, ... }, ... }
local function make_table(columns, rows)
  local widths = {}
  for ci = 1, #columns do
    local col = columns[ci]
    widths[ci] = utf8_len_safe(col.title or col.key or ("col" .. ci))
  end
  for ri = 1, #rows do
    for ci = 1, #columns do
      local col = columns[ci]
      local s = trunc(rows[ri][col.key], col.max or 120)
      local w = utf8_len_safe(s)
      if w > widths[ci] then widths[ci] = w end
    end
  end

  local function sep(ch)
    local parts = { "+" }
    for ci = 1, #columns do
      parts[#parts + 1] = string.rep(ch, widths[ci] + 2)
      parts[#parts + 1] = "+"
    end
    return colorize(table.concat(parts), C.gray)
  end

  local bar = colorize("|", C.gray)
  local out = {}
  out[#out + 1] = sep("-")

  do
    local parts = { bar }
    for ci = 1, #columns do
      local col = columns[ci]
      local title = pad_right(col.title or col.key or "", widths[ci])
      parts[#parts + 1] = " " .. colorize(title, C.bold, C.cyan) .. " "
      parts[#parts + 1] = bar
    end
    out[#out + 1] = table.concat(parts)
  end

  out[#out + 1] = sep("=")

  for ri = 1, #rows do
    local row = rows[ri]
    local parts = { bar }
    for ci = 1, #columns do
      local col = columns[ci]
      local s = trunc(row[col.key], col.max or 120)
      local cell
      if (col.align or "left") == "right" then
        cell = string.rep(" ", math.max(0, widths[ci] - utf8_len_safe(s))) .. s
      else
        cell = pad_right(s, widths[ci])
      end
      local cell_color = col.color
      if type(cell_color) == "function" then
        cell_color = cell_color(row, s)
      end
      if cell_color then cell = colorize(cell, cell_color) end
      parts[#parts + 1] = " " .. cell .. " "
      parts[#parts + 1] = bar
    end
    out[#out + 1] = table.concat(parts)
  end

  out[#out + 1] = sep("-")
  return table.concat(out, "\n")
end

-- Le module args.lua gère les flags longs (`--show-archs`, `--list`, `--params`).
-- Les flags courts (`-a`, `-l`, `-p`, `-h`) ne commencent pas par `--`, ils sont
-- donc collectés dans la liste des positionnels : on les analyse ici.
local function parse_flags(argv)
  local opts_long, pos = Args.parse(argv or {})

  local opts = {
    all        = Args.has(opts_long, "all"),
    show_archs = Args.has(opts_long, "show-archs"),
    params     = Args.has(opts_long, "params"),
    dtypes     = Args.has(opts_long, "dtypes"),
    help       = Args.has(opts_long, "help"),
    layers     = Args.has(opts_long, "layers"),
    stats      = Args.has(opts_long, "stats"),
    ops        = Args.has(opts_long, "ops"),
    ops_compact= Args.has(opts_long, "ops-compact") or Args.has(opts_long, "compact"),
    runtime    = Args.has(opts_long, "runtime"),
    optimizers = Args.has(opts_long, "optimizers"),
    recon_losses = Args.has(opts_long, "recon-losses"),
    json_out   = Args.has(opts_long, "json"),
    arch       = Args.get_str(opts_long, "list", nil),
    export_path= Args.get_str(opts_long, "export", nil),
    extern_model = Args.get_str(opts_long, "extern-model", nil),
    extern_model_requested = opts_long["extern-model"] ~= nil,
  }

  -- Scan des formes courtes depuis les positionnels.
  for i = 1, #pos do
    local v = pos[i]
    if v == "-a" then
      opts.show_archs = true
    elseif v == "-p" then
      opts.params = true
    elseif v == "-d" then
      opts.dtypes = true
    elseif v == "-h" then
      opts.help = true
    elseif v == "-l" then
      opts.arch = pos[i + 1]
    elseif v == "-e" then
      opts.export_path = pos[i + 1]
    end
  end

  if opts.all then
    opts.show_archs, opts.params, opts.layers, opts.stats = true, true, true, true
    opts.ops, opts.runtime, opts.dtypes = true, true, true
    opts.optimizers, opts.recon_losses = true, true
  end
  return opts
end

local opts

local function print_usage()
  ToolHelp.print("inspect_architectures")
end

local function bool_text(v)
  return v and "oui" or "non"
end

local function sorted_keys(map)
  local keys = {}
  for k in pairs(map or {}) do
    keys[#keys + 1] = k
  end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  return keys
end

local function join_sorted_set(map)
  local keys = sorted_keys(map)
  if #keys == 0 then return "-" end
  return table.concat(keys, ", ")
end

local function effective_arch_config(arch, config)
  if arch ~= "external_safetensors_base" or not opts.extern_model then return config end
  local result = {}
  for key, value in pairs(config or Mimir.Architectures.default_config(arch) or {}) do
    result[key] = value
  end
  result.source_safetensors = opts.extern_model
  return result
end

local function create_arch(arch, allocate_params)
  local ok, err = Mimir.Model.create(arch, effective_arch_config(arch))
  if not ok then
    log(colorize("[ERROR] ", C.bold, C.red) .. "Impossible d'instancier '" .. arch .. "': " .. tostring(err))
    return false
  end
  if allocate_params then
    Mimir.Model.allocate_params()
    if arch == "external_safetensors_base" and opts.extern_model then
      local loaded, load_err = Mimir.Serialization.load(opts.extern_model, "safetensors", {
        load_tokenizer = false, load_encoder = false, load_optimizer = false, strict_mode = true,
      })
      if not loaded then
        log(colorize("[ERROR] ", C.bold, C.red) .. "Chargement des poids externes : " .. tostring(load_err))
        return false
      end
    end
  end
  return true
end

local function framework_log_suppression_available()
  return type(Mimir) == "table"
    and type(Mimir.IO) == "table"
    and type(Mimir.IO.suppress_stdout_logs) == "function"
end

local function with_suppressed_framework_logs(fn)
  if type(fn) ~= "function" then return nil, "fn must be a function" end
  if not framework_log_suppression_available() then
    return pcall(fn)
  end

  local previous = false
  local ok_get, current = pcall(Mimir.IO.suppress_stdout_logs)
  if ok_get then previous = current and true or false end

  pcall(Mimir.IO.suppress_stdout_logs, true)
  local ok, a, b, c, d = pcall(fn)
  pcall(Mimir.IO.suppress_stdout_logs, previous)
  return ok, a, b, c, d
end

-- Aplatit récursivement une config en lignes { key=<chemin>, value=<scalaire> }.
local function flatten_config(cfg, prefix, rows)
  rows = rows or {}
  prefix = prefix or ""
  local keys = {}
  for k in pairs(cfg) do
    keys[#keys + 1] = k
  end
  table.sort(keys, function(x, y) return tostring(x) < tostring(y) end)
  for _, k in ipairs(keys) do
    local v = cfg[k]
    local path = (prefix == "") and tostring(k) or (prefix .. "." .. tostring(k))
    if type(v) == "table" then
      flatten_config(v, path, rows)
    else
      local val
      if type(v) == "string" then
        val = '"' .. v .. '"'
      else
        val = tostring(v)
      end
      rows[#rows + 1] = { key = path, value = val, vtype = type(v) }
    end
  end
  return rows
end

-- Affiche une config sous forme de tableau coloré (clé / valeur / type).
local function print_config_table(cfg)
  local rows = flatten_config(cfg, "", {})
  if #rows == 0 then
    log(colorize("\t(aucun paramètre)", C.dim))
    return
  end
  local columns = {
    { key = "key", title = "Paramètre", align = "left", color = C.yellow, max = 40 },
    { key = "value", title = "Valeur", align = "left", color = C.green, max = 60 },
    { key = "vtype", title = "Type", align = "left", color = C.magenta, max = 12 },
  }
  log(make_table(columns, rows))
end

-- dtype effectif d'une architecture: champ `config.dtype` s'il est présent,
-- sinon le défaut global du framework ("float32").
local function arch_default_dtype(entry)
  if type(entry) == "table" and type(entry.config) == "table" then
    local dt = entry.config.dtype
    if type(dt) == "string" and dt ~= "" then
      return dt
    end
  end
  return "float32"
end

local function list_archs()
  -- Lecture complète du registry : nom + description + config par défaut.
  local entries, err = Mimir.Architectures.info()
  if type(entries) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(err))
    return nil
  end
  log("\n" .. colorize("* Architectures supportées par le Framework :", C.bold, C.blue))
  local rows = {}
  for _, entry in ipairs(entries) do
    rows[#rows + 1] = {
      name = tostring(entry.name),
      dtype = arch_default_dtype(entry),
      description = (type(entry.description) == "string") and entry.description or "",
      origin = tostring(entry.origin or "native"),
    }
  end
  local columns = {
    {
      key = "name",
      title = "Architecture",
      align = "left",
      color = function(row)
        return row.origin == "mpk" and C.magenta or C.cyan
      end,
      max = 32,
    },
    { key = "dtype", title = "dtype défaut", align = "left", color = C.green, max = 12 },
    { key = "description", title = "Description", align = "left", color = nil, max = 72 },
  }
  log(make_table(columns, rows))
  log("  " .. colorize("native", C.cyan) .. " · " ..
      colorize("MPK", C.magenta))
  return entries
end

-- Affiche le tableau des dtypes disponibles dans le framework.
local function list_dtypes()
  local dtypes, err = Mimir.Architectures.dtypes()
  if type(dtypes) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(err))
    return nil
  end
  log("\n" .. colorize("* dtypes pris en charge par le Framework :", C.bold, C.blue))
  local rows = {}
  for _, dt in ipairs(dtypes) do
    rows[#rows + 1] = {
      name = tostring(dt.name),
      kind = tostring(dt.kind),
      bytes = tostring(dt.bytes),
      aliases = tostring(dt.aliases),
    }
  end
  local columns = {
    { key = "name", title = "dtype", align = "left", color = C.cyan, max = 12 },
    { key = "kind", title = "Famille", align = "left", color = C.magenta, max = 8 },
    { key = "bytes", title = "Octets", align = "right", color = C.green, max = 6 },
    { key = "aliases", title = "Alias acceptés", align = "left", color = C.yellow, max = 40 },
  }
  log(make_table(columns, rows))
  return dtypes
end

local OPTIMIZERS = {
  { name = "sgd", description = "Stochastic Gradient Descent" },
  { name = "adam", description = "Adaptive Moment Estimation" },
  { name = "adamw", description = "Adam avec décroissance des poids découplée" },
  { name = "lion", description = "Momentum signé avec décroissance découplée" },
  { name = "adafactor", description = "Second moment adaptatif, clipping RMS et pas relatif" },
  { name = "radam", description = "Adam avec variance rectifiée" },
  { name = "nadam", description = "Adam avec momentum de Nesterov" },
  { name = "rmsprop", description = "Moyenne mobile du carré des gradients" },
  { name = "lamb", description = "Adam avec trust ratio par couche" },
}

local RECON_LOSSES = {
  { name = "mse", aliases = {"l2"}, parameter = "-", description = "Erreur quadratique moyenne" },
  { name = "mae", aliases = {"l1"}, parameter = "-", description = "Erreur absolue moyenne" },
  { name = "huber", aliases = {"smooth_l1", "smoothl1"}, parameter = "huber_delta", description = "Perte quadratique puis linéaire" },
  { name = "charbonnier", aliases = {}, parameter = "charbonnier_eps", description = "Approximation différentiable de L1" },
  { name = "gaussian_nll", aliases = {"nll_gaussian", "gaussian-nll"}, parameter = "gaussian_nll_sigma", description = "Log-vraisemblance négative gaussienne" },
  { name = "bce", aliases = {}, parameter = "-", description = "Entropie croisée binaire" },
}

local function list_optimizers()
  log("\n" .. colorize("* Algorithmes d'optimisation disponibles :", C.bold, C.blue))
  log(make_table({
    { key = "name", title = "Optimiseur", align = "left", color = C.cyan, max = 16 },
    { key = "description", title = "Description", align = "left", max = 64 },
  }, OPTIMIZERS))
  return OPTIMIZERS
end

local function list_recon_losses()
  log("\n" .. colorize("* Recon-loss disponibles :", C.bold, C.blue))
  local rows = {}
  for _, loss in ipairs(RECON_LOSSES) do
    rows[#rows + 1] = {
      name = loss.name,
      aliases = #loss.aliases > 0 and table.concat(loss.aliases, ", ") or "-",
      parameter = loss.parameter,
      description = loss.description,
    }
  end
  log(make_table({
    { key = "name", title = "Recon-loss", align = "left", color = C.cyan, max = 20 },
    { key = "aliases", title = "Alias acceptés", align = "left", color = C.yellow, max = 32 },
    { key = "parameter", title = "Paramètre", align = "left", color = C.green, max = 24 },
    { key = "description", title = "Description", align = "left", max = 64 },
  }, rows))
  return RECON_LOSSES
end

local function show_params(arch)
  -- Lecture complète de l'entrée du registry pour cette architecture.
  local entry, err = Mimir.Architectures.info(arch)
  if type(entry) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. "info(" .. tostring(arch) .. ") a échoué: " .. tostring(err))
    return false
  end
  log("\n" .. colorize("* Architecture ", C.bold, C.blue) .. colorize("'" .. tostring(entry.name) .. "'", C.bold, C.cyan) .. colorize(" :", C.bold, C.blue))
  if type(entry.description) == "string" and entry.description ~= "" then
    log("  " .. colorize("description: ", C.bold) .. entry.description)
  end
  log("  " .. colorize("dtype par défaut: ", C.bold) .. colorize(arch_default_dtype(entry), C.green))
  local config = effective_arch_config(arch, entry.config)
  log("  " .. colorize(opts.extern_model and arch == "external_safetensors_base"
    and "paramètres (modèle externe) :" or "paramètres (config par défaut) :", C.bold))
  if type(config) == "table" then
    print_config_table(config)
  else
    log(colorize("\t(aucune config)", C.dim))
  end
  return true
end

-- Instancie l'architecture dans le contexte courant et retourne true/false.
local function instantiate_arch(arch, allocate_params)
  local ok, result = with_suppressed_framework_logs(function()
    return create_arch(arch, allocate_params == true)
  end)
  if not ok then
    log(colorize("[ERROR] ", C.bold, C.red) .. "Échec interne pendant l'analyse de '" .. arch .. "': " .. tostring(result))
    return false
  end
  return result and true or false
end

-- --layers: affiche le tableau des layers du modèle courant.
local function show_layers(arch)
  log("\n" .. colorize("* Layers – ", C.bold, C.blue) .. colorize(arch, C.bold, C.cyan))
  if not instantiate_arch(arch) then return false end
  local layers = Mimir.Model.get_layers()
  if type(layers) ~= "table" or #layers == 0 then
    log(colorize("\t(aucun layer)", C.dim))
    return true
  end

  -- Construire la colonne "shape" à partir des champs disponibles.
  local function layer_shape(la)
    local t = la.type or ""
    if t == "Linear" then
      if la.seq_len > 0 then
        return string.format("[%d, %d→%d]", la.seq_len, la.in_features, la.out_features)
      end
      return string.format("[%d→%d]", la.in_features, la.out_features)
    elseif t == "Embedding" then
      return string.format("[%d×%d]", la.vocab_size, la.in_features > 0 and la.in_features or la.embed_dim)
    elseif t == "Conv2d" or t == "ConvTranspose2d" then
      return string.format("[%d→%d, k%d s%d p%d]", la.in_channels, la.out_channels, la.kernel_size, la.stride, la.padding)
    elseif t == "GroupNorm" or t == "LayerNorm" then
      return string.format("[c%d]", la.in_channels > 0 and la.in_channels or la.in_features)
    elseif t == "MultiHeadAttention" or t == "SelfAttention" or t == "CrossAttention" then
      return string.format("[seq%d, d%d, h%d]", la.seq_len, la.embed_dim, la.num_heads)
    elseif t == "Add" or t == "Multiply" or t == "Subtract" then
      return string.format("[%d entrées]", #(la.inputs or {}))
    elseif t == "MatMul" then
      return string.format("[%d entrées→matmul]", #(la.inputs or {}))
    elseif t == "Split" then
      return string.format("[%d entrée→split]", #(la.inputs or {}))
    elseif t == "Reshape" then
      return "→reshape"
    elseif t == "Permute" then
      return "→permute"
    elseif t == "Concat" then
      return string.format("[%d entrées→concat]", #(la.inputs or {}))
    elseif t == "Chunk" then
      return string.format("[%d entrée→chunk]", #(la.inputs or {}))
    elseif t == "Stack" then
      return string.format("[%d entrées→stack]", #(la.inputs or {}))
    elseif t == "UpsampleNearest" or t == "UpsampleBilinear" or t == "Upsample" then
      return "→upsample"
    elseif t == "Softmax" or t == "LogSoftmax" then
      return "→softmax"
    elseif t == "Identity" then
      return "→identity"
    end
    return ""
  end

  local rows = {}
  for _, la in ipairs(layers) do
    local inputs_str = table.concat(la.inputs or {}, ", ")
    rows[#rows + 1] = {
      idx    = tostring(la.index),
      name   = tostring(la.name),
      ltype  = tostring(la.type),
      shape  = layer_shape(la),
      params = la.param_count > 0 and tostring(la.param_count) or "-",
      output = tostring(la.output),
      inputs = inputs_str,
    }
  end
  local columns = {
    { key = "idx",    title = "#",       align = "right",  color = C.gray,    max = 5  },
    { key = "name",   title = "Nom",     align = "left",   color = C.yellow,  max = 52 },
    { key = "ltype",  title = "Type",    align = "left",   color = C.cyan,    max = 22 },
    { key = "shape",  title = "Shape",   align = "left",   color = C.magenta, max = 28 },
    { key = "params", title = "Params",  align = "right",  color = C.green,   max = 12 },
    { key = "output", title = "Output",  align = "left",   color = C.dim,     max = 36 },
  }
  log(make_table(columns, rows))
  return true
end

local function collect_runtime_info()
  local caps = {}
  local ok_caps, caps_or_err = pcall(Mimir.Model.hardware_caps)
  if ok_caps and type(caps_or_err) == "table" then
    caps = caps_or_err
  end

  local guard = {}
  local ok_guard, guard_or_err = pcall(Mimir.MemoryGuard.getStats)
  if ok_guard and type(guard_or_err) == "table" then
    guard = guard_or_err
  end

  local allocator = {}
  local ok_allocator, allocator_or_err = pcall(Mimir.Allocator.getStats)
  if ok_allocator and type(allocator_or_err) == "table" then
    allocator = allocator_or_err
  end

  return {
    ops = type(Mimir.Model.runtime_ops) == "function" and Mimir.Model.runtime_ops() or {},
    hardware_caps = caps,
    memory_guard = guard,
    allocator = allocator,
    apis = {
      allocator = type(Mimir.Allocator) == "table",
      hardware_caps = type(Mimir.Model.hardware_caps) == "function",
      memory_guard = type(Mimir.MemoryGuard) == "table",
      model_dtype = type(Mimir.Model.dtype) == "function",
      serialization_detect_format = type(Mimir.Serialization) == "table" and type(Mimir.Serialization.detect_format) == "function",
      serialization_load = type(Mimir.Serialization) == "table" and type(Mimir.Serialization.load) == "function",
      serialization_save = type(Mimir.Serialization) == "table" and type(Mimir.Serialization.save) == "function",
    },
  }
end

local function show_runtime_ops()
  local backends = Mimir.Model.runtime_ops()
  local rows, columns, by_type = {}, {
    { key = "type", title = "Op / Layer", max = 24, color = C.cyan },
  }, {}
  for _, backend in ipairs(backends) do
    columns[#columns + 1] = {
      key = backend.name, title = backend.name .. (backend.compiled and "" or " (non compilé)"), max = 35,
    }
    for _, op in ipairs(backend.ops) do
      local row = by_type[op.type]
      if not row then row = { type = op.type }; by_type[op.type] = row end
      row[backend.name] = op.forward .. " / " .. op.backward
    end
  end
  for _, name in ipairs(sorted_keys(by_type)) do rows[#rows + 1] = by_type[name] end
  log("\n" .. colorize("* Ops par runtime — Forward / Backward", C.bold, C.blue))
  log(make_table(columns, rows))
  log("  NativeOptimized=natif optimisé ; Native=natif ; HostFallback=repli CPU ; Unsupported=indisponible.")
  log("  Capacités déclarées par type : matériel, configuration et formes de tenseurs non testés.")
  return backends
end

local function show_runtime()
  local info = collect_runtime_info()
  log("\n" .. colorize("* Runtime – capacités exposées au Lua API", C.bold, C.blue))

  log(colorize("  Hardware caps:", C.bold))
  log(make_table({
    { key = "feature", title = "Feature", align = "left", color = C.cyan, max = 12 },
    { key = "enabled", title = "Disponible", align = "left", color = C.green, max = 10 },
  }, {
    { feature = "AVX2", enabled = bool_text(info.hardware_caps.avx2) },
    { feature = "FMA", enabled = bool_text(info.hardware_caps.fma) },
    { feature = "F16C", enabled = bool_text(info.hardware_caps.f16c) },
    { feature = "BMI2", enabled = bool_text(info.hardware_caps.bmi2) },
  }))

  log(colorize("  MemoryGuard:", C.bold))
  log(make_table({
    { key = "metric", title = "Métrique", align = "left", color = C.cyan, max = 18 },
    { key = "value", title = "Valeur", align = "right", color = C.green, max = 14 },
  }, {
    { metric = "current_mb", value = string.format("%.2f", tonumber(info.memory_guard.current_mb) or 0) },
    { metric = "peak_mb", value = string.format("%.2f", tonumber(info.memory_guard.peak_mb) or 0) },
    { metric = "limit_mb", value = string.format("%.2f", tonumber(info.memory_guard.limit_mb) or 0) },
    { metric = "usage_percent", value = string.format("%.2f", tonumber(info.memory_guard.usage_percent) or 0) },
  }))

  log(colorize("  Allocator:", C.bold))
  log(make_table({
    { key = "metric", title = "Métrique", align = "left", color = C.cyan, max = 18 },
    { key = "value", title = "Valeur", align = "right", color = C.green, max = 14 },
  }, {
    { metric = "tensor_count", value = tostring(info.allocator.tensor_count or 0) },
    { metric = "loaded_count", value = tostring(info.allocator.loaded_count or 0) },
  }))

  local api_rows = {}
  for _, key in ipairs(sorted_keys(info.apis)) do
    api_rows[#api_rows + 1] = { api = key, present = bool_text(info.apis[key]) }
  end
  log(colorize("  APIs runtime disponibles:", C.bold))
  log(make_table({
    { key = "api", title = "API", align = "left", color = C.yellow, max = 30 },
    { key = "present", title = "Présente", align = "left", color = C.green, max = 10 },
  }, api_rows))

  return info
end

local function collect_layer_bindings()
  local ops = {}
  for _, name in ipairs(Mimir.Layers.available()) do
    ops[#ops + 1] = { name = name:lower(), label = name }
  end
  local rows = {}
  for _, op in ipairs(ops) do
    local fn = type(Mimir.Layers) == "table" and Mimir.Layers[op.name] or nil
    local status = "absente"
    local detail = ""
    if type(fn) == "function" then
      local ok, a, b = pcall(fn)
      if not ok then
        status = "erreur"
        detail = tostring(a)
      elseif a == false then
        status = "placeholder"
        detail = tostring(b or "")
      else
        status = "active"
        detail = tostring(b or "")
      end
    end
    rows[#rows + 1] = {
      op = op.label,
      binding = "Mimir.Layers." .. op.name,
      status = status,
      detail = detail,
    }
  end
  return rows
end

local function collect_graph_ops_summary()
  local entries, err = Mimir.Architectures.info()
  if type(entries) ~= "table" then
    return nil, tostring(err)
  end

  local graph_ops = {}
  local failures = {}
  local ok_collect, collect_err = with_suppressed_framework_logs(function()
    for _, entry in ipairs(entries) do
      local arch = tostring(entry.name)
      local cfg = effective_arch_config(arch, type(entry.config) == "table" and entry.config or {})
      if arch == "external_safetensors_base"
        and (type(cfg.source_safetensors) ~= "string" or cfg.source_safetensors == "") then
        failures[#failures + 1] = arch
      elseif create_arch(arch, false) then
        local layers = Mimir.Model.get_layers()
        if type(layers) == "table" then
          for _, la in ipairs(layers) do
            local ltype = tostring(la.type or "?")
            local info = graph_ops[ltype]
            if not info then
              info = {
                type = ltype,
                count = 0,
                archs = {},
                sample = tostring(la.name or ""),
                lua_binding = "-",
              }
              if type(Mimir.Layers) == "table" then
                local key = string.lower(ltype)
                if type(Mimir.Layers[key]) == "function" then
                  info.lua_binding = "Mimir.Layers." .. key
                end
              end
              graph_ops[ltype] = info
            end
            info.count = info.count + 1
            info.archs[arch] = true
          end
        end
      else
        failures[#failures + 1] = arch
      end
    end
  end)
  if not ok_collect then
    return nil, tostring(collect_err)
  end

  local rows = {}
  for _, ltype in ipairs(sorted_keys(graph_ops)) do
    local info = graph_ops[ltype]
    rows[#rows + 1] = {
      type = ltype,
      count = tostring(info.count),
      arch_count = tostring(#sorted_keys(info.archs)),
      archs = join_sorted_set(info.archs),
      binding = info.lua_binding,
      sample = info.sample,
    }
  end
  table.sort(rows, function(a, b)
    local ca = tonumber(a.arch_count) or 0
    local cb = tonumber(b.arch_count) or 0
    if ca ~= cb then return ca > cb end
    return a.type < b.type
  end)

  return {
    rows = rows,
    failures = failures,
    bindings = collect_layer_bindings(),
  }
end

local function classify_op_family(ltype)
  local t = tostring(ltype or "")
  if t == "Conv2d" or t == "ConvTranspose2d" or t == "DepthwiseConv2d"
    or t == "GlobalAvgPool2d" or t == "MaxPool2d" or t == "AvgPool2d"
    or t == "UpsampleNearest" or t == "UpsampleBilinear" or t == "Upsample" then
    return "vision_spatial"
  end
  if t == "LayerNorm" or t == "GroupNorm" or t == "BatchNorm2d"
    or t == "RMSNorm" or t == "InstanceNorm2d" then
    return "normalization"
  end
  if t == "MultiHeadAttention" or t == "SelfAttention" or t == "CrossAttention" then
    return "attention"
  end
  if t == "Linear" or t == "MatMul" or t == "BatchMatMul" then
    return "linear_algebra"
  end
  if t == "Embedding" or t == "TokenMeanPool" then
    return "token_embedding"
  end
  if t == "Add" or t == "Multiply" or t == "Subtract" or t == "Concat"
    or t == "Split" or t == "Chunk" or t == "Stack" or t == "Reshape"
    or t == "Permute" or t == "Transpose" or t == "Flatten" or t == "Identity"
    or t == "Constant" or t == "View" then
    return "routing_shape"
  end
  if t == "ReLU" or t == "GELU" or t == "GEGLU" or t == "SiLU"
    or t == "LeakyReLU" or t == "Tanh" or t == "Softmax" or t == "LogSoftmax"
    or t == "Sigmoid" or t == "Swish" then
    return "activations"
  end
  if t == "Reparameterize" then
    return "latent_sampling"
  end
  return "other"
end

local function family_label(family)
  local labels = {
    activations = "Activations",
    attention = "Attention",
    latent_sampling = "Latent / sampling",
    linear_algebra = "Linear / algebra",
    normalization = "Normalisation",
    other = "Autres",
    routing_shape = "Routing / shape",
    token_embedding = "Tokens / embeddings",
    vision_spatial = "Vision / spatial",
  }
  return labels[family] or family
end

local function summarize_op_families(summary)
  local families = {}
  for _, row in ipairs(summary.rows or {}) do
    local family = classify_op_family(row.type)
    local item = families[family]
    if not item then
      item = {
        family = family,
        types = 0,
        occ = 0,
        archs = {},
        bound = 0,
        samples = {},
      }
      families[family] = item
    end
    item.types = item.types + 1
    item.occ = item.occ + (tonumber(row.count) or 0)
    if row.binding and row.binding ~= "-" then
      item.bound = item.bound + 1
    end
    if row.sample and row.sample ~= "" and #item.samples < 3 then
      item.samples[#item.samples + 1] = row.sample
    end
    local archs = tostring(row.archs or "")
    if archs ~= "" and archs ~= "-" then
      for arch in archs:gmatch("[^,]+") do
        item.archs[(arch:gsub("^%s+", ""):gsub("%s+$", ""))] = true
      end
    end
  end

  local rows = {}
  for _, family in ipairs(sorted_keys(families)) do
    local item = families[family]
    rows[#rows + 1] = {
      family = family_label(family),
      types = tostring(item.types),
      occ = tostring(item.occ),
      arch_count = tostring(#sorted_keys(item.archs)),
      bound = string.format("%d/%d", item.bound, item.types),
      samples = (#item.samples > 0) and table.concat(item.samples, ", ") or "-",
    }
  end

  table.sort(rows, function(a, b)
    local oa = tonumber(a.occ) or 0
    local ob = tonumber(b.occ) or 0
    if oa ~= ob then return oa > ob end
    return a.family < b.family
  end)
  return rows
end

local function show_ops(opts)
  log("\n" .. colorize("* Ops / Layers – inventaire du framework", C.bold, C.blue))
  local summary, err = collect_graph_ops_summary()
  if not summary then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(err))
    return false
  end

  if opts and opts.ops_compact and not opts.all then
    local family_rows = summarize_op_families(summary)
    log(colorize("  Vue compacte par famille:", C.bold))
    log(make_table({
      { key = "family", title = "Famille", align = "left", color = C.cyan, max = 22 },
      { key = "types", title = "Types", align = "right", color = C.yellow, max = 7 },
      { key = "occ", title = "Occ.", align = "right", color = C.green, max = 8 },
      { key = "arch_count", title = "Archs", align = "right", color = C.magenta, max = 7 },
      { key = "bound", title = "Bindings", align = "right", color = C.blue, max = 10 },
      { key = "samples", title = "Exemples", align = "left", color = nil, max = 46 },
    }, family_rows))
  else
    log(colorize("  Layer types observés dans les graphes d'architectures:", C.bold))
    log(make_table({
      { key = "type", title = "Type", align = "left", color = C.cyan, max = 24 },
      { key = "count", title = "Occ.", align = "right", color = C.green, max = 8 },
      { key = "arch_count", title = "Archs", align = "right", color = C.yellow, max = 8 },
      { key = "binding", title = "Binding Lua", align = "left", color = C.magenta, max = 40 },
      { key = "sample", title = "Exemple", align = "left", color = nil, max = 46 },
    }, summary.rows))
  end

  if opts.all then
    log(colorize("  Résumé par famille:", C.bold))
    log(make_table({
      {key="family", title="Famille"}, {key="types", title="Types"},
      {key="occ", title="Occ."}, {key="arch_count", title="Archs"},
      {key="bound", title="Bindings"}, {key="samples", title="Exemples", max=46},
    }, summarize_op_families(summary)))
  end
  log(colorize("  Filtres en lecture seule Mimir.Layers:", C.bold))
  log(make_table({
    { key = "op", title = "Op", align = "left", color = C.cyan, max = 24 },
    { key = "binding", title = "Binding", align = "left", color = C.yellow, max = 40 },
    { key = "status", title = "État", align = "left", color = C.green, max = 14 },
    { key = "detail", title = "Détail", align = "left", color = nil, max = 60 },
  }, summary.bindings))

  if #summary.failures > 0 then
    log(colorize("  Architectures non instanciées pour l'inventaire: ", C.bold, C.yellow)
      .. table.concat(summary.failures, ", "))
  end
  return true
end

-- --stats: statistiques théoriques (params par layer + totaux par type).
local function show_stats(arch)
  log("\n" .. colorize("* Stats – ", C.bold, C.blue) .. colorize(arch, C.bold, C.cyan))
  if not instantiate_arch(arch) then return false end

  local layers = Mimir.Model.get_layers()
  local total_params = Mimir.Model.total_params()
  if type(layers) ~= "table" then layers = {} end

  -- Agrégation par type.
  local by_type = {}       -- type → {count, params}
  local top_layers = {}    -- top-20 par params

  for _, la in ipairs(layers) do
    local t = la.type or "?"
    if not by_type[t] then by_type[t] = { count = 0, params = 0 } end
    by_type[t].count  = by_type[t].count  + 1
    by_type[t].params = by_type[t].params + la.param_count
    if la.param_count > 0 then
      top_layers[#top_layers + 1] = la
    end
  end

  -- Top-20 layers par params desc.
  table.sort(top_layers, function(a, b) return a.param_count > b.param_count end)
  local top_n = math.min(20, #top_layers)

  log(colorize("  Total paramètres: ", C.bold) .. colorize(string.format("%d", total_params), C.green)
    .. colorize(string.format("  (%.2f M)", total_params / 1e6), C.gray))
  log(colorize("  Total layers: ", C.bold) .. colorize(tostring(#layers), C.yellow))
  log("")

  -- Tableau par type.
  local type_rows = {}
  for t, v in pairs(by_type) do
    type_rows[#type_rows + 1] = {
      ltype  = t,
      count  = tostring(v.count),
      params = string.format("%d", v.params),
      pct    = total_params > 0
               and string.format("%.1f%%", 100.0 * v.params / total_params)
               or  "0%",
    }
  end
  table.sort(type_rows, function(a, b)
    return tonumber(a.params) > tonumber(b.params)
  end)
  log(colorize("  Répartition par type :", C.bold))
  log(make_table(
    {
      { key = "ltype",  title = "Type",      align = "left",  color = C.cyan,    max = 24 },
      { key = "count",  title = "Nb",        align = "right", color = C.yellow,  max = 6  },
      { key = "params", title = "Paramètres",align = "right", color = C.green,   max = 14 },
      { key = "pct",    title = "%",         align = "right", color = C.magenta, max = 7  },
    },
    type_rows
  ))

  -- Top N layers.
  if top_n > 0 then
    log(colorize(string.format("\n  Top-%d layers (params desc) :", top_n), C.bold))
    local top_rows = {}
    for i = 1, top_n do
      local la = top_layers[i]
      top_rows[#top_rows + 1] = {
        rank   = tostring(i),
        name   = tostring(la.name),
        ltype  = tostring(la.type),
        params = string.format("%d", la.param_count),
        pct    = total_params > 0
                 and string.format("%.1f%%", 100.0 * la.param_count / total_params)
                 or  "0%",
      }
    end
    log(make_table(
      {
        { key = "rank",   title = "#",          align = "right", color = C.gray,    max = 4  },
        { key = "name",   title = "Layer",       align = "left",  color = C.yellow,  max = 52 },
        { key = "ltype",  title = "Type",        align = "left",  color = C.cyan,    max = 22 },
        { key = "params", title = "Paramètres",  align = "right", color = C.green,   max = 14 },
        { key = "pct",    title = "%",           align = "right", color = C.magenta, max = 7  },
      },
      top_rows
    ))
  end
  return true
end

local function infer_export_format(path)
  path = tostring(path or "")
  if path == "" then return nil, "chemin vide" end

  if path:sub(-1) == "/" then
    return "raw_folder", "rawFolder"
  end

  local lower = path:lower()
  if lower:match("%.json$") then
    return "debug_json", "debugJSON"
  end
  if lower:match("%.safetensors$") or lower:match("%.st$") then
    return "safetensors", "safetensors"
  end

  return nil, "extension non supportée (utiliser .json, .safetensors ou un chemin finissant par /)"
end

local function ensure_parent_dir(path, fmt)
  path = tostring(path or "")
  if fmt == "raw_folder" and path:sub(-1) == "/" then
    FS.mkdir_p(path)
    return
  end

  local dir = FS.dirname(path)
  if dir and dir ~= "" then
    FS.mkdir_p(dir)
  end
end

local function export_architecture_checkpoint(arch, export_path)
  local fmt_internal, fmt_label_or_err = infer_export_format(export_path)
  if not fmt_internal then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(fmt_label_or_err))
    return false
  end

  local ok_inst = instantiate_arch(arch, true)
  if not ok_inst then return false end

  ensure_parent_dir(export_path, fmt_internal)
  local ok_save, err_save = Mimir.Serialization.save(export_path, fmt_internal, {
    include_git_info = true,
    include_checksums = true,
    save_optimizer = true,
    save_tokenizer = true,
    save_encoder = true,
  })
  if not ok_save then
    log(colorize("[ERROR] ", C.bold, C.red) .. "Échec export " .. tostring(fmt_label_or_err)
      .. " vers '" .. tostring(export_path) .. "': " .. tostring(err_save))
    return false
  end

  log("\n" .. colorize("✓ Export réussi", C.bold, C.green)
    .. "  arch=" .. colorize(arch, C.cyan)
    .. "  format=" .. colorize(fmt_label_or_err, C.yellow)
    .. "  path=" .. colorize(export_path, C.magenta))
  return true
end

-- --json: exporte tout le registre en JSON (stdout).
local function export_json()
  local entries, err = Mimir.Architectures.info()
  if type(entries) ~= "table" then
    log(colorize("[ERROR] ", C.bold, C.red) .. tostring(err))
    return false
  end

  -- Sérialisation JSON minimale (pas de dépendance externe).
  local function to_json(v, indent)
    indent = indent or 0
    local sp = string.rep("  ", indent)
    local sp2 = string.rep("  ", indent + 1)
    local t = type(v)
    if t == "nil" then
      return "null"
    elseif t == "boolean" then
      return tostring(v)
    elseif t == "number" then
      if v == math.floor(v) and math.abs(v) < 2^53 then
        return string.format("%d", v)
      end
      return string.format("%.10g", v)
    elseif t == "string" then
      -- Échapper les caractères JSON.
      return '"' .. v:gsub('\\', '\\\\'):gsub('"', '\\"')
                     :gsub('\n', '\\n'):gsub('\r', '\\r')
                     :gsub('\t', '\\t') .. '"'
    elseif t == "table" then
      -- Détecter si c'est un tableau.
      local is_array = (#v > 0)
      if is_array then
        local parts = {}
        for i = 1, #v do
          parts[#parts + 1] = sp2 .. to_json(v[i], indent + 1)
        end
        return "[\n" .. table.concat(parts, ",\n") .. "\n" .. sp .. "]"
      else
        local keys = {}
        for k in pairs(v) do keys[#keys + 1] = k end
        table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
        local parts = {}
        for _, k in ipairs(keys) do
          parts[#parts + 1] = sp2 .. '"' .. tostring(k) .. '": ' .. to_json(v[k], indent + 1)
        end
        return "{\n" .. table.concat(parts, ",\n") .. "\n" .. sp .. "}"
      end
    end
    return '"[unsupported:' .. t .. ']"'
  end

  -- Construire la structure complète.
  local registry = {}
  for _, entry in ipairs(entries) do
    -- Optionnel: instancier pour obtenir le total_params réel.
    local total = 0
    local layer_list = {}
    local ok_inst = with_suppressed_framework_logs(function()
      assert(create_arch(entry.name, false), "architecture non instanciée")
      total = Mimir.Model.total_params()
      local layers = Mimir.Model.get_layers()
      if type(layers) == "table" then
        for _, la in ipairs(layers) do
          layer_list[#layer_list + 1] = {
            index       = la.index,
            name        = la.name,
            type        = la.type,
            param_count = la.param_count,
            output      = la.output,
            inputs      = la.inputs,
          }
        end
      end
    end)

    registry[#registry + 1] = {
      name         = entry.name,
      description  = entry.description,
      origin       = entry.origin or "native",
      source_path  = entry.source_path,
      dtype        = arch_default_dtype(entry),
      config       = effective_arch_config(entry.name, entry.config),
      total_params = ok_inst and total or nil,
      layers       = ok_inst and layer_list or nil,
    }
  end

  local ops_summary = collect_graph_ops_summary()
  local ops_compact = nil
  if type(ops_summary) == "table" then
    ops_compact = summarize_op_families(ops_summary)
  end
  io.write(to_json({
    registry = registry,
    optimizers = OPTIMIZERS,
    recon_losses = RECON_LOSSES,
    runtime = collect_runtime_info(),
    ops = ops_summary,
    ops_compact = ops_compact,
  }, 0))
  io.write("\n")
  io.flush()
  return true
end

opts = parse_flags(arg)

if opts.extern_model_requested then
  if not opts.extern_model or opts.extern_model == "" then
    error("--extern-model requiert un chemin vers un fichier .safetensors")
  end
  if not FS.file_exists(opts.extern_model) then
    error("Fichier SafeTensors introuvable ou illisible : " .. opts.extern_model)
  end
  if opts.arch and opts.arch ~= "external_safetensors_base" then
    error("--extern-model utilise external_safetensors_base ; --list " .. opts.arch .. " est incompatible")
  end
  opts.arch = "external_safetensors_base"
  if not opts.params and not opts.layers and not opts.stats and not opts.export_path and not opts.json_out then
    opts.params, opts.layers, opts.stats = true, true, true
  end
  if not instantiate_arch(opts.arch) then
    error("Impossible d'inspecter le fichier SafeTensors externe : " .. opts.extern_model)
  end
end

if opts.help then
  print_usage()
  return
end

-- --json : export complet, indépendant des autres flags.
if opts.json_out then
  export_json()
  return
end

-- Aucun flag utile : on affiche l'aide.
if not opts.show_archs and not opts.arch and not opts.dtypes and not opts.optimizers
  and not opts.recon_losses and not opts.ops and not opts.runtime and not opts.export_path then
  print_usage()
  return
end

if opts.show_archs then
  list_archs()
  list_dtypes()
  list_optimizers()
  list_recon_losses()
elseif opts.dtypes then
  list_dtypes()
end

if opts.optimizers and not opts.show_archs then
  list_optimizers()
end

if opts.recon_losses and not opts.show_archs then
  list_recon_losses()
end

if opts.runtime then
  show_runtime()
end

if opts.ops then
  show_ops(opts)
end
if opts.ops or opts.runtime then show_runtime_ops() end

if opts.arch then
  if not opts.params and not opts.layers and not opts.stats and not opts.export_path then
    -- -l sans aucune sous-option : confirmer l'existence + hint.
    local entry = Mimir.Architectures.info(opts.arch)
    if type(entry) == "table" then
      log("\n" .. colorize("Architecture '" .. opts.arch .. "' disponible.", C.green)
        .. " Options: " .. colorize("-p", C.bold) .. " params · "
        .. colorize("--layers", C.bold) .. " layers · "
        .. colorize("--stats", C.bold) .. " stats · "
        .. colorize("-e", C.bold) .. " export")
    else
      log("\n" .. colorize("[ERROR] ", C.bold, C.red) .. "Architecture inconnue: " .. tostring(opts.arch))
    end
    return
  end

  if opts.params  then show_params(opts.arch)  end
  if opts.layers  then show_layers(opts.arch)  end
  if opts.stats   then show_stats(opts.arch)   end
  if opts.export_path then export_architecture_checkpoint(opts.arch, opts.export_path) end
elseif opts.export_path then
  log(colorize("[ERROR] ", C.bold, C.red) .. "-e/--export requiert -l/--list <arch>")
end

-- --all without a selection reports every registered architecture.
if opts.all and not opts.arch then
  local entries = Mimir.Architectures.info()
  for _, entry in ipairs(entries or {}) do
    show_params(entry.name)
    local cfg = entry.config or {}
    if entry.name == "external_safetensors_base"
      and (type(cfg.source_safetensors) ~= "string" or cfg.source_safetensors == "") then
      log("  Layers / Stats indisponibles : source_safetensors non configuré.")
    else
      show_layers(entry.name)
      show_stats(entry.name)
    end
  end
end
