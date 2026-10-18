---@diagnostic disable: undefined-global, undefined-field

--[[
  show-graph.lua — Visualisation métriques d'entraînement
  Équivalent Lua de tools/show-graph.py
  Génère un rapport HTML interactif (Chart.js).

  Usage (via Mimir):
    ./bin/mimir --lua scripts/tools/show-graph.lua -- [CSV] [OPTIONS]
  Usage (Lua ≥ 5.3 standalone):
    lua scripts/tools/show-graph.lua [CSV] [OPTIONS]

  OPTIONS:
    [CSV]                    CSV à analyser (défaut: checkpoints/loss_history.csv)
    --csv PATH               Chemin CSV unique ou pattern glob (*part0.csv, ...)
    --csv-dir DIR            Dossier contenant des CSV *part[0-9].csv à fusionner
    --model NAME             Nom du modèle (auto-détection si absent)
    --algo NAME              Algorithme d'optimisation (alias de --optimizer)
    --optimizer NAME         Algorithme d'optimisation (adamw, adam, sgd, ...)
    --recon-loss NAME        Fonction de reconstruction (mse, l1, charbonnier, ...)
    --checkpoint PATH        Dossier checkpoint ou fichier DEBUGJSON
    --checkpoint-dir PATH    Alias rétrocompatible de --checkpoint
    -n, --no-interactive     Pas de prompts stdin
    --out PATH               HTML de sortie (défaut: ./graph_report.html)
    --out-text PATH          Rapport .md/.txt; accepte rapport.{md,txt}
    --watch                  Mode surveillance : régénère le rapport dès que le CSV change
                             et ouvre le navigateur automatiquement.
    --watch-interval N       Intervalle de polling en secondes (défaut: 2)
    -h, --help               Aide
]]

-- ══════════════════════════════════════════════════════════════
-- HELPERS FICHIERS / SYSTÈME
-- ══════════════════════════════════════════════════════════════

local ToolHelp = dofile(ROOTWORK.."/scripts/modules/tools_help.lua")
ToolHelp.show("show-graph")

local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local function file_exists(p)
  return FS.file_exists(p)
end

-- ══════════════════════════════════════════════════════════════
-- GLOB / DÉTECTION *part[0-9].csv
-- ══════════════════════════════════════════════════════════════

-- Retourne la liste triée des fichiers *part[0-9]+.csv dans un dossier,
-- ou nil si aucun trouvé.
local function find_part_csvs(dir)
  local names = FS.list_dir(dir)
  local found = {}
  for _, name in ipairs(names) do
    if name:match("part%d+%.csv$") then
      found[#found+1] = FS.join(dir, name)
    end
  end
  if #found == 0 then return nil end
  table.sort(found)   -- tri lexicographique = tri numérique sur part0..part9
  return found
end

-- Idem depuis un chemin CSV : si le chemin contient "part[0-9]" on cherche
-- les autres parties dans le même dossier.
local function expand_part_csvs(csv_path)
  if not csv_path:match("part%d+%.csv$") then return { csv_path } end
  local dir = csv_path:match("^(.*[/\\])") or "./"
  -- Retire le slash final pour ls
  dir = dir:gsub("[/\\]$", "")
  if dir == "" then dir = "." end
  local parts = find_part_csvs(dir)
  return parts or { csv_path }
end

-- Fusionne plusieurs tables CSV (même structure d'en-têtes, concatène les lignes).
-- Réassigne `step` de façon continue si la colonne existe.
local function merge_csvs(parts_data)
  if #parts_data == 1 then return parts_data[1] end
  -- Vérifier que tous ont les mêmes colonnes (on prend les headers du 1er)
  local merged = {
    headers = parts_data[1].headers,
    df      = {},
    n       = 0,
    parts   = #parts_data,
  }
  for _, h in ipairs(merged.headers) do merged.df[h] = {} end

  local step_offset = 0
  for pi, part in ipairs(parts_data) do
    local step_max_here = 0
    for i = 1, part.n do
      for _, h in ipairs(merged.headers) do
        local v = (part.df[h] or {})[i]
        if h == "step" and type(v) == "number" then
          v = v + step_offset
          if v - step_offset > step_max_here then step_max_here = v - step_offset end
        end
        if v ~= nil then
          local col = merged.df[h]
          col[#col+1] = v
        end
      end
      merged.n = merged.n + 1
    end
    -- Offset pour la prochaine partie : max step de cette partie + 1
    if part.df.step and #part.df.step > 0 then
      local last = part.df.step[#part.df.step]
      if type(last) == "number" then
        step_offset = step_offset + last + 1
      end
    end
  end
  return merged
end

local function read_file(p)
  local f = io.open(p, "r"); if not f then return nil end
  local s = f:read("*a"); f:close(); return s
end

local function write_file(p, s)
  local f = io.open(p, "w"); if not f then return false end
  f:write(s); f:close(); return true
end

local function is_dir(p)
  return FS.is_dir(p)
end

local function ls(dir)
  return FS.list_dir(dir)
end

local function file_mtime(p)
  local f = io.open(p, "rb")
  if not f then return 0 end
  local r = f:seek("end")
  f:close()
  return tonumber(r) or 0
end

-- mtime combinée d'une liste de fichiers (max)
local function mtimes_combined(paths)
  local mx = 0
  for _, p in ipairs(paths) do
    local m = file_mtime(p); if m > mx then mx = m end
  end
  return mx
end

-- Définie après repo_root() (forward-declaration Lua)
local open_browser

-- sleep portable (Lua n'a pas sleep natif sans LuaSocket)
local function sleep(sec)
  os.execute("sleep " .. tostring(sec))
end

local function file_size(p)
  local f = io.open(p, "rb"); if not f then return 0 end
  local s = f:seek("end"); f:close(); return s or 0
end

local function human_bytes(n)
  if     n < 1024       then return n .. " B"
  elseif n < 1048576    then return string.format("%.1f KiB", n / 1024)
  elseif n < 1073741824 then return string.format("%.1f MiB", n / 1048576)
  else                       return string.format("%.2f GiB", n / 1073741824)
  end
end

-- Forme décimale la plus courte qui relit exactement le même flottant, sans
-- imposer une précision d'affichage réduite aux métriques du rapport.
local function metric_number(v)
  return string.format("%.17g", v)
end

local function script_dir()
  local ok, info = pcall(debug.getinfo, 1, "S")
  if ok and info and info.source and info.source:sub(1,1) == "@" then
    return info.source:sub(2):match("^(.*[/\\])") or "./"
  end
  return "./"
end

local function repo_root()
  local sd = script_dir()
  return sd:match("^(.*[/\\])scripts[/\\]tools[/\\]?") or "./"
end

-- ══════════════════════════════════════════════════════════════
-- PARAMÈTRES UI (viz_ui_settings.json)
-- ══════════════════════════════════════════════════════════════

local function ui_settings_path()
  return repo_root() .. "viz_ui_settings.json"
end

-- Lit viz_ui_settings.json.
-- Extrait uniquement les clés connues de show-graph :
--   t.browser    (string top-level)
--   t.showgraph  (sous-objet string pairs)
-- Le reste du fichier n'est jamais désérialisé ni réécrit en entier.
local function load_ui_settings()
  local p = ui_settings_path()
  local s = read_file(p)
  if not s then return {} end
  local t = {}
  local br = s:match('"browser"%s*:%s*"([^"]*)"')
  if br then t.browser = br end
  -- Extraction inline du sous-objet "showgraph" (json_obj défini plus loin)
  local sg_start = s:find('"showgraph"%s*:%s*{')
  if sg_start then
    local depth, i = 0, sg_start
    while i <= #s do
      local c = s:sub(i, i)
      if c == '{' then depth = depth + 1
      elseif c == '}' then
        depth = depth - 1
        if depth == 0 then
          local sg_raw = s:sub(sg_start, i)
          local sg = {}
          for k, v in sg_raw:gmatch('"([^"]+)"%s*:%s*"([^"]*)"') do sg[k] = v end
          t.showgraph = sg
          break
        end
      end
      i = i + 1
    end
  end
  return t
end

-- Applique des patches dans viz_ui_settings.json SANS toucher aux autres clés.
--   patches.key = "string"  → remplace/insère la paire top-level "key": "value"
--   patches.key = { ... }   → remplace/insère le sous-objet top-level "key": { ... }
local function patch_ui_settings(patches)
  local p = ui_settings_path()
  local raw = read_file(p)
  if not raw or raw:match("^%s*$") then raw = "{}\n" end

  -- Position du dernier '}' dans la chaîne
  local function last_rbrace(s)
    for i = #s, 1, -1 do if s:sub(i,i) == '}' then return i end end
  end

  for k, v in pairs(patches) do
    if type(v) == "string" then
      local ek  = k:gsub('([^%w_])', '%%%1')
      local new = string.format('"%s": "%s"', k, v:gsub('"', '\\"'))
      local n
      raw, n = raw:gsub('"' .. ek .. '"%s*:%s*"[^"]*"', new, 1)
      if n == 0 then
        local pos = last_rbrace(raw)
        if pos then
          local before = raw:sub(1, pos - 1)
          local sep = before:match('[^%s,{]%s*$') and ',\n  ' or '  '
          raw = before .. sep .. new .. '\n' .. raw:sub(pos)
        end
      end
    elseif type(v) == "table" then
      local inner = {}
      for sk, sv in pairs(v) do
        inner[#inner+1] = string.format('    "%s": "%s"', sk, tostring(sv):gsub('"', '\\"'))
      end
      table.sort(inner)
      local block = '  "' .. k .. '": {\n'
                    .. (#inner > 0 and table.concat(inner, ',\n') .. '\n' or '')
                    .. '  }'

      local ek    = k:gsub('([^%w_])', '%%%1')
      local s_pos = raw:find('"' .. ek .. '"%s*:%s*{')
      if s_pos then
        -- Remplacer le bloc existant (comptage d'accolades)
        local depth, i = 0, s_pos
        while i <= #raw do
          local c = raw:sub(i, i)
          if c == '{' then depth = depth + 1
          elseif c == '}' then
            depth = depth - 1
            if depth == 0 then
              -- Début de ligne (indentation incluse)
              local ls = s_pos
              while ls > 1 and raw:sub(ls-1, ls-1) ~= '\n' do ls = ls - 1 end
              raw = raw:sub(1, ls - 1) .. block .. raw:sub(i + 1)
              break
            end
          end
          i = i + 1
        end
      else
        -- Insérer avant le dernier '}'
        local pos = last_rbrace(raw)
        if pos then
          local before = raw:sub(1, pos - 1)
          local sep = before:match('[^%s,{]%s*$') and ',\n' or '\n'
          raw = before .. sep .. block .. '\n' .. raw:sub(pos)
        end
      end
    end
  end

  local tmp = p .. '.tmp'
  if write_file(tmp, raw) then os.rename(tmp, p) end
end

-- Alias utilisé par choose_and_save_browser (compat descendante).
local function save_ui_settings(t)
  patch_ui_settings(t)
end

-- Retourne la liste des navigateurs installés sur le système.
local function detect_browsers()
  local candidates = {
    { cmd = os.getenv("BROWSER"),   label = "$BROWSER" },
    { cmd = "sensible-browser",     label = "sensible-browser" },
    { cmd = "x-www-browser",        label = "x-www-browser" },
    { cmd = "firefox",              label = "Firefox" },
    { cmd = "firefox-esr",          label = "Firefox ESR" },
    { cmd = "chromium-browser",     label = "Chromium" },
    { cmd = "chromium",             label = "Chromium" },
    { cmd = "google-chrome",        label = "Google Chrome" },
    { cmd = "google-chrome-stable", label = "Google Chrome (stable)" },
    { cmd = "brave-browser",        label = "Brave" },
  }
  local found, seen = {}, {}
  for _, c in ipairs(candidates) do
    if c.cmd and c.cmd ~= "" and not seen[c.cmd] then
      local ck = io.popen('command -v "' .. c.cmd .. '" 2>/dev/null')
      local bin = ck and ck:read("*l"); if ck then ck:close() end
      if bin and bin ~= "" then
        seen[c.cmd] = true
        found[#found+1] = { cmd = c.cmd, label = c.label, bin = bin }
      end
    end
  end
  return found
end

-- Propose interactivement la liste des navigateurs installés,
-- sauvegarde le choix dans viz_ui_settings.json et retourne la commande.
local function choose_and_save_browser()
  local browsers = detect_browsers()
  io.stderr:write("\n🌐 Premier lancement — quel navigateur souhaitez-vous utiliser ?\n")
  for i, b in ipairs(browsers) do
    io.stderr:write(string.format("  [%d] %-30s  %s\n", i, b.label, b.bin))
  end
  local xdg_idx = #browsers + 1
  io.stderr:write(string.format("  [%d] xdg-open (système par défaut)\n", xdg_idx))
  io.stderr:write(string.format("Votre choix [1-%d] (défaut: 1) : ", xdg_idx))
  io.stderr:flush()
  local line = io.read("*l")
  local choice = math.max(1, math.min(xdg_idx, tonumber(line) or 1))
  local cmd, label
  if choice <= #browsers then
    cmd   = browsers[choice].cmd
    label = browsers[choice].label
  else
    cmd   = "xdg-open"
    label = "xdg-open"
  end
  io.stderr:write("✓ Navigateur choisi : " .. label .. "\n")
  patch_ui_settings({ browser = cmd })
  io.stderr:write("💾 Préférence sauvegardée dans viz_ui_settings.json\n\n")
  return cmd
end

-- Ouvre le rapport HTML dans le navigateur préféré.
-- Au premier appel (aucune préférence enregistrée) demande à l'utilisateur.
open_browser = function(path)
  -- Si le chemin est déjà une URL http(s), l'utiliser telle quelle.
  local url
  if path:match("^https?://") then
    url = path
  else
    local abs = path:sub(1,1) == "/" and path
                or ((os.getenv("PWD") or ".") .. "/" .. path)
    url = "file://" .. abs
  end
  local settings = load_ui_settings()
  local browser  = settings.browser
  if not browser or browser == "" then
    browser = choose_and_save_browser()
  end
  os.execute('"' .. browser .. '" "' .. url .. '" >/dev/null 2>&1 &')
end

-- ══════════════════════════════════════════════════════════════
-- JSON MINIMAL (extraction de champs scalaires depuis architecture.json)
-- ══════════════════════════════════════════════════════════════

local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function json_str(text, key)
  return text:match('"' .. key .. '"%s*:%s*"([^"]*)"')
end

local function json_num(text, key)
  return tonumber(text:match('"' .. key .. '"%s*:%s*(%-?%d+%.?%d*[eE]?[+-]?%d*)'))
end

-- Extraire un sous-objet JSON { ... } à la clé donnée
local function json_obj(text, key)
  local s = text:find('"' .. key .. '"%s*:%s*{')
  if not s then return nil end
  local d, i = 0, s
  while i <= #text do
    local c = text:sub(i, i)
    if c == '{' then d = d + 1
    elseif c == '}' then
      d = d - 1; if d == 0 then return text:sub(s, i) end
    end
    i = i + 1
  end
  return nil
end

local function json_array(text, key)
  local s = text:find('"' .. key .. '"%s*:%s*%[')
  if not s then return nil end
  local start = text:find("%[", s)
  local depth, in_string, escaped = 0, false, false
  for i = start, #text do
    local c = text:sub(i, i)
    if in_string then
      if escaped then escaped = false
      elseif c == "\\" then escaped = true
      elseif c == '"' then in_string = false end
    elseif c == '"' then in_string = true
    elseif c == "[" then depth = depth + 1
    elseif c == "]" then
      depth = depth - 1
      if depth == 0 then return text:sub(start, i) end
    end
  end
  return nil
end

local function json_scalar_pairs(obj)
  local values = {}
  if not obj then return values end
  for key, raw in obj:gmatch('"([^"]+)"%s*:%s*([^,%}\n]+)') do
    raw = trim(raw)
    local value
    if raw:sub(1, 1) == '"' and raw:sub(-1) == '"' then
      value = raw:sub(2, -2):gsub('\\"', '"'):gsub('\\\\', '\\')
    elseif raw == "true" then value = true
    elseif raw == "false" then value = false
    elseif raw == "null" then value = "null"
    elseif not raw:match("^[%{%[]") then value = tonumber(raw) or raw end
    if value ~= nil then values[key] = value end
  end
  return values
end

local function json_object_list(array)
  local objects, depth, start = {}, 0, nil
  local in_string, escaped = false, false
  for i = 1, #(array or "") do
    local c = array:sub(i, i)
    if in_string then
      if escaped then escaped = false
      elseif c == "\\" then escaped = true
      elseif c == '"' then in_string = false end
    elseif c == '"' then in_string = true
    elseif c == "{" then
      if depth == 0 then start = i end
      depth = depth + 1
    elseif c == "}" then
      depth = depth - 1
      if depth == 0 and start then
        objects[#objects + 1] = array:sub(start, i)
        start = nil
      end
    end
  end
  return objects
end

local function analyze_model_metadata(text)
  local info = {
    config = {}, layers = {}, layer_types = {}, num_layers = 0,
    trainable_layers = 0, parameterless_layers = 0, trainable_params = 0,
  }
  if not text then return info end
  info.config = json_scalar_pairs(json_obj(text, "model_config"))
  info.model_name = json_str(text, "model_name") or info.config.type
  info.total_params = json_num(text, "total_params")
  info.image_width = json_num(text, "image_width")
  info.image_height = json_num(text, "image_height")
  for _, raw in ipairs(json_object_list(json_array(text, "layers"))) do
    local layer = json_scalar_pairs(raw)
    local inputs = json_array(raw, "inputs")
    if inputs then
      local names = {}
      for name in inputs:gmatch('"([^"]+)"') do names[#names + 1] = name end
      layer.inputs = table.concat(names, ", ")
    end
    local params_count = tonumber(layer.params_count) or 0
    -- Dans Mímir, trainable_parameter est réservé aux tenseurs-paramètres sans
    -- entrée. Une couche classique est entraînable lorsqu'elle porte des poids.
    layer.is_trainable = params_count > 0 or layer.trainable_parameter == true
    if layer.is_trainable then
      info.trainable_layers = info.trainable_layers + 1
      info.trainable_params = info.trainable_params + params_count
    else
      info.parameterless_layers = info.parameterless_layers + 1
    end
    info.layers[#info.layers + 1] = layer
    local kind = tostring(layer.type or "inconnu")
    info.layer_types[kind] = (info.layer_types[kind] or 0) + 1
  end
  info.num_layers = #info.layers
  if not info.total_params then
    local total = 0
    for _, layer in ipairs(info.layers) do total = total + (tonumber(layer.params_count) or 0) end
    if total > 0 then info.total_params = total end
  end
  return info
end

local function detect_metric_warmups(config, headers)
  local available = {}
  for _, name in ipairs(headers or {}) do available[name] = true end
  local warmups = {}
  for key, raw_steps in pairs(config or {}) do
    local steps = tonumber(raw_steps)
    if steps and steps > 0 and (key == "warmup_steps" or key:match("_warmup_steps$")) then
      local metric
      if key == "warmup_steps" then
        metric = "learning_rate"
      elseif key == "kl_warmup_steps" then
        metric = "kl_beta_effective"
      else
        local prefix = key:gsub("_warmup_steps$", "")
        metric = available[prefix] and prefix or prefix .. " (groupe de métriques)"
      end
      warmups[#warmups + 1] = { key = key, metric = metric, steps = steps }
    end
  end
  table.sort(warmups, function(a, b)
    if a.steps == b.steps then return a.key < b.key end
    return a.steps < b.steps
  end)
  return warmups
end

-- ══════════════════════════════════════════════════════════════
-- CSV PARSER
-- ══════════════════════════════════════════════════════════════

local function parse_csv(path)
  local f = io.open(path, "rb")
  if not f then return nil, "Fichier introuvable: " .. path end
  local headers = nil
  local df = nil
  local n = 0

  local row, field = {}, {}
  local in_quotes = false
  local data_started = false

  local function push_field()
    row[#row + 1] = table.concat(field)
    field = {}
  end

  local function row_has_content()
    for i = 1, #row do
      if row[i]:match("[^%s]") then return true end
    end
    return false
  end

  local function ensure_headers()
    headers = {}
    for i = 1, #row do
      local h = trim(row[i])
      if i == 1 then
        h = h:gsub("^\239\187\191", "")
      end
      headers[#headers + 1] = h
    end
    df = {}
    for _, h in ipairs(headers) do df[h] = {} end
    row = {}
  end

  local function commit_row()
    if not row_has_content() then
      row = {}
      return
    end
    if not headers then
      ensure_headers()
      return
    end

    n = n + 1
    for ci, h in ipairs(headers) do
      local v = row[ci]
      if v ~= nil then
        v = trim(v)
        if v ~= "" then
          df[h][n] = tonumber(v) or v
        end
      end
    end
    row = {}
    data_started = true
  end

  local chunk_size = 64 * 1024
  while true do
    local chunk = f:read(chunk_size)
    if not chunk then break end
    local i = 1
    while i <= #chunk do
      local c = chunk:sub(i, i)
      if in_quotes then
        if c == '"' then
          local next_c = chunk:sub(i + 1, i + 1)
          if next_c == '"' then
            field[#field + 1] = '"'
            i = i + 1
          else
            in_quotes = false
          end
        else
          field[#field + 1] = c
        end
      else
        if c == '"' then
          in_quotes = true
        elseif c == ',' then
          push_field()
        elseif c == '\n' then
          push_field()
          commit_row()
        elseif c ~= '\r' then
          field[#field + 1] = c
        end
      end
      i = i + 1
    end
  end

  if in_quotes or #field > 0 or #row > 0 then
    push_field()
    commit_row()
  end

  f:close()
  if not headers then return nil, "CSV vide" end
  return { headers = headers, df = df, n = n }
end

-- ══════════════════════════════════════════════════════════════
-- STATISTIQUES
-- ══════════════════════════════════════════════════════════════

local function col_stats(t)
  if not t or #t == 0 then return { min=0, max=0, mean=0, std=0, n=0 } end
  local n, s, mn, mx = #t, 0, t[1], t[1]
  for i = 1, n do
    local v = t[i]
    if type(v) == "number" then
      s = s + v
      if v < mn then mn = v end
      if v > mx then mx = v end
    end
  end
  local mu = s / n
  local var = 0
  for i = 1, n do
    local v = t[i]; if type(v) == "number" then var = var + (v - mu)^2 end
  end
  return { min=mn, max=mx, mean=mu, std=math.sqrt(var / n), n=n }
end

local function histogram(t, bins)
  bins = bins or 40
  if not t or #t == 0 then return {}, {} end
  local s = col_stats(t)
  local range = s.max - s.min
  if range == 0 then return { tostring(s.min) }, { #t } end
  local bw = range / bins
  local counts, labels = {}, {}
  for i = 1, bins do
    counts[i] = 0
    labels[i] = metric_number(s.min + (i - 0.5) * bw)
  end
  for _, v in ipairs(t) do
    if type(v) == "number" then
      local b = math.max(1, math.min(bins, math.floor((v - s.min) / bw) + 1))
      counts[b] = counts[b] + 1
    end
  end
  return labels, counts
end

local function pearson(a, b)
  local n = math.min(#a, #b); if n < 2 then return 0 end
  local sa, sb = 0, 0
  for i = 1, n do
    if type(a[i]) == "number" and type(b[i]) == "number" then sa = sa + a[i]; sb = sb + b[i] end
  end
  local ma, mb = sa / n, sb / n
  local num, da2, db2 = 0, 0, 0
  for i = 1, n do
    if type(a[i]) == "number" and type(b[i]) == "number" then
      local da, db = a[i] - ma, b[i] - mb
      num = num + da * db; da2 = da2 + da * da; db2 = db2 + db * db
    end
  end
  local d = math.sqrt(da2 * db2); return d > 0 and num / d or 0
end

-- Extrait les points de validation (val_loss numérique) avec le train_loss associé.
-- Retourne un tableau trié par step de { step, val_loss, val_mse, train_loss, gap }.
local function extract_val_points(df, n)
  local pts = {}
  if not df or not df.val_loss then return pts end

  -- Tableau trié (step, loss) pour recherche du train_loss le plus proche (bin search).
  local ts = {}
  local step_arr = df.step or {}
  local loss_arr = df.loss or {}
  for i = 1, n do
    local s, l = step_arr[i], loss_arr[i]
    if type(s) == "number" and type(l) == "number" then
      ts[#ts+1] = { s = s, l = l }
    end
  end
  table.sort(ts, function(a, b) return a.s < b.s end)

  local function nearest_loss(target)
    if #ts == 0 or type(target) ~= "number" then return nil end
    local lo, hi = 1, #ts
    while lo < hi do
      local mid = math.floor((lo + hi) / 2)
      if ts[mid].s < target then lo = mid + 1 else hi = mid end
    end
    local best = ts[lo]
    if lo > 1 and math.abs(ts[lo-1].s - target) < math.abs(best.s - target) then
      best = ts[lo-1]
    end
    return best.l
  end

  local vst_arr = df.val_step or {}
  local vmse_arr = df.val_mse  or {}

  for i = 1, n do
    local vl = df.val_loss[i]
    if type(vl) == "number" then
      local cs  = step_arr[i]
      local vs  = vst_arr[i]
      local vm  = vmse_arr[i]
      local actual = (type(vs) == "number" and vs >= 0) and vs
                     or (type(cs) == "number" and cs or i)
      local tl = nearest_loss(actual)
      pts[#pts+1] = {
        step       = actual,
        val_loss   = vl,
        val_mse    = type(vm) == "number" and vm or nil,
        train_loss = tl,
        gap        = tl and (vl - tl) or nil,
      }
    end
  end
  table.sort(pts, function(a, b) return a.step < b.step end)
  return pts
end

-- Retire toutes les lignes appartenant à un cycle de validation :
-- pendant la validation, l'optimiseur ne fait pas de mise à jour, donc opt_step reste
-- constant. Toutes les lignes qui partagent le même opt_step qu'une ligne de résultat
-- val_loss (y compris les items intermédiaires qui n'ont pas encore val_loss rempli)
-- sont exclues des métriques d'entraînement, sauf la PREMIÈRE occurrence (la vraie
-- étape d'entraînement qui a déclenché la validation).
-- Retourne (train_df, train_n, n_val_excluded).
local function filter_train_rows(df, headers, n)
  if not df.val_loss then return df, n, 0 end

  -- Clé pivot : opt_step est le compteur d'optimizer steps (ne bouge pas pendant val).
  -- Fallback sur step si opt_step absent du CSV.
  local step_key = (df.opt_step and #df.opt_step > 0) and "opt_step" or "step"

  -- Collecter les valeurs pivot présentes dans les lignes de résultat val_loss
  local val_pivot = {}
  for i = 1, n do
    if type(df.val_loss[i]) == "number" then
      local sv = df[step_key] and df[step_key][i]
      if type(sv) == "number" then val_pivot[sv] = true end
    end
  end
  if not next(val_pivot) then return df, n, 0 end

  -- Conserver seulement la première ligne pour chaque pivot (= étape d'entraînement)
  local first_seen = {}
  local new_df = {}
  for _, h in ipairs(headers) do new_df[h] = {} end
  local new_n = 0

  for i = 1, n do
    local sv = df[step_key] and df[step_key][i]
    if type(sv) == "number" and val_pivot[sv] then
      if not first_seen[sv] then
        first_seen[sv] = true
        new_n = new_n + 1
        for _, h in ipairs(headers) do new_df[h][new_n] = df[h][i] end
      end
      -- lignes suivantes avec le même pivot = items + résumé de validation → ignorées
    else
      new_n = new_n + 1
      for _, h in ipairs(headers) do new_df[h][new_n] = df[h][i] end
    end
  end
  return new_df, new_n, n - new_n
end

-- Auto-détection des paramètres de calibration depuis le CSV :
--   validate_every_steps : fréquence de validation (en opt_steps)
--   validate_items       : nombre d'images évaluées par validation
--   n_dataset            : nombre d'items dans le dataset (par epoch)
local function detect_validation_params(df, n)
  local p = {
    validate_every_steps = nil, validate_items = nil, n_dataset = nil,
    validate_holdout = nil, validate_holdout_frac = nil,
    validate_holdout_items = nil,
  }

  -- n_dataset : colonne total_batches (constante dans le CSV)
  if df.total_batches and #df.total_batches > 0 then
    p.n_dataset = df.total_batches[1]
  end

  if not df.val_loss then return p end

  local step_key = (df.opt_step and #df.opt_step > 0) and "opt_step" or "step"

  -- Collecter les opt_steps de validation dans l'ordre
  local seen, sorted = {}, {}
  for i = 1, n do
    if type(df.val_loss[i]) == "number" then
      local sv = df[step_key] and df[step_key][i]
      if type(sv) == "number" and not seen[sv] then
        seen[sv] = true; sorted[#sorted+1] = sv
      end
    end
  end
  table.sort(sorted)

  -- validate_every_steps : mode des espacements (valeur la plus fréquente)
  if #sorted >= 2 then
    local freq = {}
    for k = 2, math.min(#sorted, 12) do
      local d = sorted[k] - sorted[k-1]
      freq[d] = (freq[d] or 0) + 1
    end
    local best_d, best_f = nil, 0
    for d, f in pairs(freq) do
      if f > best_f then best_d = d; best_f = f end
    end
    p.validate_every_steps = best_d
  end

  -- validate_items : lignes avec le même pivot que le 1er résultat val, moins 1 (training row)
  if #sorted > 0 then
    p.validate_holdout = true
    local sample = sorted[1]
    local cnt = 0
    for i = 1, n do
      local sv = df[step_key] and df[step_key][i]
      if sv == sample then cnt = cnt + 1 end
    end
    -- cnt = 1 training + N items intermédiaires + 1 résumé val_loss
    -- items évalués = cnt - 1 (on exclut la ligne training)
    p.validate_items = math.max(0, cnt - 1)
    p.validate_holdout_items = p.validate_items
    if p.n_dataset and p.n_dataset > 0 and p.validate_holdout_items > 0 then
      p.validate_holdout_frac = p.validate_holdout_items / p.n_dataset
    end
  end

  return p
end

-- ══════════════════════════════════════════════════════════════
-- SÉRIALISATION JS
-- ══════════════════════════════════════════════════════════════

-- Tableau JS à partir d'une table Lua (nombres ou chaînes)
local function js_arr(t, stride)
  stride = stride or 1
  local p = {}
  for i = 1, #t, stride do
    local v = t[i]
    if type(v) == "number" then
      p[#p+1] = (v ~= v) and "null" or metric_number(v)
    elseif type(v) == "string" then
      p[#p+1] = '"' .. v:gsub('"', '\\"') .. '"'
    else
      p[#p+1] = "null"
    end
  end
  return "[" .. table.concat(p, ",") .. "]"
end

-- Tableau de points {x,y} pour Chart.js scatter/line
local function js_xy(xs, ys, stride)
  stride = stride or 1
  local p = {}
  local n = math.min(#xs, #ys)
  for i = 1, n, stride do
    if type(xs[i]) == "number" and type(ys[i]) == "number" then
      p[#p+1] = "{x:" .. metric_number(xs[i]) .. ",y:" .. metric_number(ys[i]) .. "}"
    end
  end
  return "[" .. table.concat(p, ",") .. "]"
end

-- Chaîne JS (avec guillemets, échappée)
local function js_s(s)
  if s == nil then return "null" end
  return '"' .. tostring(s):gsub('"', '\\"'):gsub('\n', '\\n') .. '"'
end

-- Couleur CSS pour la matrice de corrélation : -1=rouge, 0=blanc, +1=bleu
local function corr_bg(r)
  r = math.max(-1, math.min(1, r or 0))
  if r >= 0 then
    local g = math.floor(255 * (1 - r)); return string.format("rgb(%d,%d,255)", g, g)
  else
    local gb = math.floor(255 * (1 + r)); return string.format("rgb(255,%d,%d)", gb, gb)
  end
end

-- Sérialise les colonnes CSV (strided) en JSON pour les mises à jour DOM partielles.
-- Appelée depuis generate() après chaque régénération du HTML.
local function gen_data_json(df, headers, n, val_pts)
  local stride = math.max(1, math.ceil(n / 2000))
  local buf = {}
  local function e(s) buf[#buf+1] = s end
  e(string.format('{"stride":%d,"n":%d', stride, n))
  for _, c in ipairs(headers or {}) do
    local has_numeric = false
    for _, value in pairs(df[c] or {}) do
      if type(value) == "number" then has_numeric = true; break end
    end
    if has_numeric then
      e(',"' .. c .. '":' .. js_arr(df[c], stride))
    end
  end
  if val_pts and #val_pts > 0 then
    local vp = {}
    for _, p in ipairs(val_pts) do
      vp[#vp+1] = string.format(
        '{"step":%s,"val_loss":%s,"val_mse":%s,"train_loss":%s,"gap":%s}',
        metric_number(p.step), metric_number(p.val_loss),
        p.val_mse    and metric_number(p.val_mse)    or "null",
        p.train_loss and metric_number(p.train_loss) or "null",
        p.gap        and metric_number(p.gap)        or "null"
      )
    end
    e(',"val_pts":[' .. table.concat(vp, ",") .. "]")
  else
    e(',"val_pts":[]')
  end
  e("}")
  return table.concat(buf)
end

-- ══════════════════════════════════════════════════════════════
-- AUTO-DÉTECTION CHECKPOINT
-- ══════════════════════════════════════════════════════════════

local function find_latest_run()
  local ckpt = repo_root() .. "checkpoint"
  if not is_dir(ckpt) then return nil end
  local best, bmt = nil, 0
  for _, name in ipairs(ls(ckpt)) do
    if name ~= "base_tokenizer" and not name:match("^%.") then
      local p = ckpt .. "/" .. name
      if is_dir(p) then
        local mt = file_mtime(p); if mt > bmt then bmt = mt; best = p end
      end
    end
  end
  return best
end

local function find_latest_epoch(run_dir)
  local best, bmt = nil, 0
  for _, name in ipairs(ls(run_dir)) do
    if name:match("^epoch_") then
      local p = run_dir .. "/" .. name
      if is_dir(p) and file_exists(p .. "/model/architecture.json") then
        local mt = file_mtime(p); if mt > bmt then bmt = mt; best = p end
      end
    end
  end
  return best
end

local function set_training_meta(m, info)
  local config = info and info.config or {}
  m.model_info = info
  m.model = config.type or (info and info.model_name) or m.model
  m.algo = config.optimizer or config.algorithm or config.algo or m.algo
  m.recon_loss = config.recon_loss or m.recon_loss
  m.checkpoint_dir = config.checkpoint_dir or m.checkpoint_dir

  if m.recon_loss == "charbonnier" and tonumber(config.charbonnier_eps) then
    m.recon_loss_details = "eps=" .. metric_number(tonumber(config.charbonnier_eps))
  elseif m.recon_loss == "huber" and tonumber(config.huber_delta) then
    m.recon_loss_details = "delta=" .. metric_number(tonumber(config.huber_delta))
  elseif m.recon_loss and m.recon_loss:match("nll") and tonumber(config.nll_sigma) then
    m.recon_loss_details = "sigma=" .. metric_number(tonumber(config.nll_sigma))
  end
end

local function load_debugjson_meta(path)
  local text = read_file(path)
  if not text then return { source = path, error = "DEBUGJSON illisible: " .. path } end
  if json_str(text, "format") ~= "mimir_debug_dump" then
    return { source = path, error = "JSON non reconnu comme DEBUGJSON Mímir: " .. path }
  end

  local info = analyze_model_metadata(text)
  local m = { source = path, checkpoint_file = path }
  set_training_meta(m, info)
  m.model = m.model or json_str(text, "model_type")
  return m
end

local function load_meta(checkpoint_path)
  if file_exists(checkpoint_path) and not is_dir(checkpoint_path) then
    return load_debugjson_meta(checkpoint_path)
  end

  local run_dir = checkpoint_path
  local m = { source = run_dir }
  local raw_arch = run_dir .. "/model/architecture.json"
  if file_exists(run_dir .. "/manifest.json") and file_exists(raw_arch) then
    set_training_meta(m, analyze_model_metadata(read_file(raw_arch)))
    m.checkpoint_dir = m.checkpoint_dir or run_dir
    return m
  end

  local ep = find_latest_epoch(run_dir)
  if not ep then
    local start_text = read_file(run_dir .. "/starttrain.json")
    if not start_text then return m end
    m.source = (run_dir:match("[^/]+$") or run_dir) .. "/starttrain.json"
    set_training_meta(m, analyze_model_metadata(start_text))
    m.checkpoint_dir = m.checkpoint_dir or run_dir
    return m
  end
  m.source = (run_dir:match("[^/]+$") or run_dir) .. "/" .. (ep:match("[^/]+$") or ep)
  local text = read_file(ep .. "/model/architecture.json")
  if not text then return m end
  set_training_meta(m, analyze_model_metadata(text))

  -- Les warmups effectifs sont capturés dans starttrain.json après application
  -- des options CLI. Un architecture.json repris peut encore porter les valeurs
  -- historiques du checkpoint : ne l'utiliser qu'en repli pour ces champs.
  local start_text = read_file(run_dir .. "/starttrain.json")
  local start_config = start_text and json_scalar_pairs(json_obj(start_text, "model_config")) or {}
  for key, value in pairs(start_config) do
    if key == "optimizer" or key == "algorithm" or key == "algo"
        or key == "recon_loss" or key == "charbonnier_eps"
        or key == "huber_delta" or key == "nll_sigma"
        or key == "warmup_steps" or key:match("_warmup_steps$") or key == "decay_strategy" then
      m.model_info.config[key] = value
    end
  end
  set_training_meta(m, m.model_info)

  local mc = json_obj(text, "model_config")
  if mc then
    m.model = json_str(mc, "type")
    m.checkpoint_dir = json_str(mc, "checkpoint_dir") or run_dir
  else
    m.model = json_str(text, "model_name")
  end
  return m
end

-- ══════════════════════════════════════════════════════════════
-- GÉNÉRATION HTML
-- ══════════════════════════════════════════════════════════════

local PALETTE = {
  loss     = "#2E86AB",
  recon    = "#A23B72",
  lr       = "#F18F01",
  kl       = "#E63946",
  kl_beta  = "#D2A8FF",
  wass     = "#F4A261",
  entropy  = "#2A9D8F",
  moment   = "#E76F51",
  spatial  = "#52B788",
  temporal = "#8338EC",
  timestep = "#457B9D",
  opt_eps  = "#6A4C93",
  memory   = "#58A6FF",
  allocator= "#7EE787",
  val_loss = "#FF9F1C",
  val_recon= "#E040FB",
}

local EPOCH_PAL = {
  "#2E86AB","#A23B72","#F18F01","#E63946","#2A9D8F",
  "#8338EC","#F4A261","#52B788","#E76F51","#457B9D",
  "#E9C46A","#06D6A0","#EF476F","#118AB2","#FFD166",
}

local function gen_html(ctx)
  local df      = ctx.df
  local n       = ctx.n
  local rc      = ctx.recon_col
  local rl      = ctx.recon_label
  local val_rl  = "Validation " .. rl
  local epochs  = ctx.epochs

  local stride = math.max(1, math.ceil(n / 2000))

  -- Helpers locaux
  local function jd(c)
    return js_arr(df[c] or {}, stride)
  end
  local function jdxy(xc, yc)
    return js_xy(df[xc] or {}, df[yc] or {}, stride)
  end
  local function has(c)
    return df[c] ~= nil and #df[c] > 0
  end
  local steps = jd("step")

  -- Histogrammes
  local hlbl_loss, hcnt_loss
  if has("loss") then hlbl_loss, hcnt_loss = histogram(df.loss, 40) end
  local hlbl_rc, hcnt_rc
  if rc and has(rc) then hlbl_rc, hcnt_rc = histogram(df[rc], 40) end

  -- Colonnes pour matrice de corrélation. Le CSV définit les métriques : aucun
  -- catalogue lié à une architecture particulière n'est imposé ici.
  local corr_cols = {}
  local corr_excluded = {
    step=true, opt_step=true, batch=true, total_batches=true,
    epoch=true, total_epochs=true, val_step=true,
  }
  for _, c in ipairs(ctx.metric_columns or {}) do
    if not corr_excluded[c] then corr_cols[#corr_cols+1] = c end
  end

  -- Datasets par epoch (scatter+showLine)
  local ep_ds = {}
  if has("epoch") and has("loss") and has("step") then
    local ep_map = {}
    for i = 1, n do
      local e = df.epoch[i]
      if type(e) == "number" then
        if not ep_map[e] then ep_map[e] = { sx = {}, sy = {} } end
        local d = ep_map[e]
        d.sx[#d.sx+1] = df.step[i]
        d.sy[#d.sy+1] = df.loss[i]
      end
    end
    for idx, e in ipairs(epochs) do
      local ed = ep_map[e]
      if ed then
        local est = math.max(1, math.ceil(#ed.sx / 500))
        local col = EPOCH_PAL[((idx - 1) % #EPOCH_PAL) + 1]
        ep_ds[#ep_ds+1] = string.format(
          "{label:'Epoch %d',data:%s,borderColor:'%s',backgroundColor:'transparent',"
          .. "borderWidth:1.5,showLine:true,pointRadius:0,tension:0.2}",
          e, js_xy(ed.sx, ed.sy, est), col)
      end
    end
  end

  -- Tableau HTML de corrélation
  local corr_tbl = ""
  if #corr_cols > 0 then
    corr_tbl = corr_tbl .. "<tr><th></th>"
    for _, c in ipairs(corr_cols) do
      corr_tbl = corr_tbl .. "<th>" .. (c == rc and rl or c) .. "</th>"
    end
    corr_tbl = corr_tbl .. "</tr>\n"
    for _, ca in ipairs(corr_cols) do
      corr_tbl = corr_tbl .. "<tr><th>" .. (ca == rc and rl or ca) .. "</th>"
      for _, cb in ipairs(corr_cols) do
        local r = pearson(df[ca], df[cb])
        corr_tbl = corr_tbl .. string.format(
          '<td style="background:%s;color:%s">%s</td>',
          corr_bg(r), math.abs(r) > 0.5 and "#fff" or "#222", metric_number(r))
      end
      corr_tbl = corr_tbl .. "</tr>\n"
    end
  end

  -- Registre des instances Chart.js pour les mises à jour DOM partielles.
  -- Chaque builder y stocke {type, ...} ; sérialisé dans window._sg_chart_meta.
  local chart_meta = {}

  -- ── Builders de cartes Chart.js ────────────────────────────

  -- Clé x commune à tous les graphes : opt_step si disponible (pas de trous), sinon step.
  local x_key = has("opt_step") and "opt_step" or "step"

  -- Sérialise les frontières d'epoch en annotations Chart.js (valeurs réelles opt_step).
  local function chart_annotations_js(metric_name)
    local ep_bnd = ctx.epoch_boundaries or {}
    local parts = {}
    for _, b in ipairs(ep_bnd) do
      parts[#parts+1] = string.format(
        'ep%d:{type:\'line\',scaleID:\'x\',value:%s,'
        .. 'borderColor:\'rgba(255,255,255,0.18)\',borderWidth:1,borderDash:[4,4],'
        .. 'label:{display:true,content:\'E%d\',position:\'start\','
        .. 'color:\'#8b949e\',font:{size:8},padding:{y:2}}}',
        b.epoch, metric_number(b.step), b.epoch)
    end
    for index, warmup in ipairs(ctx.warmups or {}) do
      if warmup.metric == metric_name then
        parts[#parts + 1] = string.format(
          'warmup%d:{type:\'line\',scaleID:\'x\',value:%s,'
          .. 'borderColor:\'#FFD166\',borderWidth:2,borderDash:[7,4],'
          .. 'label:{display:true,content:\'Fin warmup (%s)\',position:\'end\','
          .. 'color:\'#FFD166\',backgroundColor:\'rgba(13,17,23,.8)\',font:{size:9}}}',
          index, metric_number(warmup.steps), metric_number(warmup.steps))
      end
    end
    return '{' .. table.concat(parts, ',') .. '}'
  end

  -- Graphe en ligne : scatter+showLine avec x_key comme abscisse réelle (axe linéaire).
  -- Tous les graphes partagent le même espace x → annotations d'epoch alignées.
  local function card_line(id, title, col_name, color, log_y)
    if not has(col_name) then return "" end
    local yscale = log_y and "logarithmic" or "linear"
    local lbl = col_name == rc and rl
      or (col_name == "kl_beta_effective" and "KL Beta")
      or col_name
    chart_meta[id] = string.format('{type:"line",xcol:"%s",ycol:"%s"}', x_key, col_name)
    return string.format(
      '<div class="card"><h3>%s</h3><canvas id="%s"></canvas></div>\n'
      .. "<script>window._sg_charts[%s]=new Chart(document.getElementById(%s),{type:'scatter',"
      .. "data:{datasets:[{label:%s,data:%s,"
      .. "borderColor:'%s',backgroundColor:'%s22',fill:true,"
      .. "borderWidth:1.5,tension:0.2,pointRadius:0,showLine:true}]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:false},annotation:{annotations:%s}},"
      .. "scales:{x:{type:'linear',ticks:{maxTicksLimit:8}},y:{type:'%s'}}}});</script>\n",
      title, id, js_s(id), js_s(id), js_s(lbl), jdxy(x_key, col_name),
      color, color, chart_annotations_js(col_name), yscale)
  end

  -- Histogramme (barres)
  local function card_bar(id, title, labels, counts, color)
    if not labels or #labels == 0 then return "" end
    return string.format(
      '<div class="card"><h3>%s</h3><canvas id="%s"></canvas></div>\n'
      .. "<script>new Chart(document.getElementById(%s),{type:'bar',"
      .. "data:{labels:%s,datasets:[{label:'',data:%s,"
      .. "backgroundColor:'%s88',borderColor:'%s',borderWidth:1}]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:false}},"
      .. "scales:{x:{ticks:{maxTicksLimit:10}},y:{ticks:{maxTicksLimit:6}}}}});</script>\n",
      title, id, js_s(id), js_arr(labels), js_arr(counts), color, color)
  end

  -- Nuage de points
  local function card_scatter(id, title, xcol, ycol, color, xl, yl)
    if not has(xcol) or not has(ycol) then return "" end
    chart_meta[id] = string.format('{type:"scatter",xcol:"%s",ycol:"%s"}', xcol, ycol)
    return string.format(
      '<div class="card"><h3>%s</h3><canvas id="%s"></canvas></div>\n'
      .. "<script>window._sg_charts[%s]=new Chart(document.getElementById(%s),{type:'scatter',"
      .. "data:{datasets:[{label:'',data:%s,"
      .. "backgroundColor:'%s55',borderColor:'transparent',"
      .. "pointRadius:2,pointHoverRadius:3}]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:false}},"
      .. "scales:{x:{title:{display:true,text:%s}},"
      .. "y:{title:{display:true,text:%s}}}}});</script>\n",
      title, id, js_s(id), js_s(id), jdxy(xcol, ycol), color, js_s(xl), js_s(yl))
  end

  -- ── Assemblage HTML ────────────────────────────────────────

  local out = {}
  local function emit(s) out[#out+1] = s end
  local function html_escape(v)
    if v == nil or v == "" then return "-" end
    local s
    if type(v) == "boolean" then s = v and "true" or "false"
    elseif type(v) == "number" then s = metric_number(v)
    else s = tostring(v) end
    return s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
      :gsub('"', "&quot;"):gsub("'", "&#39;")
  end
  local function html_sorted_keys(t)
    local keys = {}; for key in pairs(t or {}) do keys[#keys + 1] = key end
    table.sort(keys); return keys
  end
  local function html_table(headers, rows, class_name)
    emit("<div class='scroll'><table class='" .. (class_name or "stats") .. "'><thead><tr>")
    for _, header in ipairs(headers) do emit("<th>" .. html_escape(header) .. "</th>") end
    emit("</tr></thead><tbody>\n")
    for _, row in ipairs(rows) do
      emit("<tr>")
      for index = 1, #headers do emit("<td>" .. html_escape(row[index]) .. "</td>") end
      emit("</tr>\n")
    end
    emit("</tbody></table></div>\n")
  end

  -- DOCTYPE + head
  emit("<!DOCTYPE html>\n<html lang='fr'>\n<head>\n")
  emit("<meta charset='utf-8'>\n")
  emit("<meta name='viewport' content='width=device-width,initial-scale=1'>\n")
  emit("<title>Training Metrics — " .. (ctx.model or "Mimir") .. "</title>\n")
  emit("<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js'></script>\n")
  emit("<script src='https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js'></script>\n")
  emit("<style>\n")
  emit(":root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--accent:#58a6ff}\n")
  emit("*{box-sizing:border-box;margin:0;padding:0}\n")
  -- Fondu entrant à chaque chargement de page
  emit("@keyframes _sg_in{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}\n")
  emit("body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;")
  emit("background:var(--bg);color:var(--text);padding:24px;")
  emit("animation:_sg_in 0.35s ease both}\n")
  emit("header{background:var(--card);border:1px solid var(--border);border-radius:10px;")
  emit("padding:20px 24px;margin-bottom:24px}\n")
  emit("header h1{font-size:1.4rem;color:var(--accent);margin-bottom:12px}\n")
  emit(".meta{display:flex;flex-wrap:wrap;gap:10px;font-size:.82rem;color:var(--muted)}\n")
  emit(".meta span{background:#21262d;padding:4px 10px;border-radius:6px;white-space:nowrap}\n")
  emit(".meta b{color:var(--text)}\n")
  emit("section{margin-bottom:28px}\n")
  emit("section>h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);")
  emit("margin-bottom:12px;border-bottom:1px solid var(--border);padding-bottom:6px}\n")
  emit(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}\n")
  emit(".card{background:var(--card);border:1px solid var(--border);border-radius:8px;")
  emit("padding:14px;min-height:270px}\n")
  emit(".card h3{font-size:.74rem;font-weight:600;color:var(--muted);margin-bottom:10px;")
  emit("text-transform:uppercase;letter-spacing:.05em}\n")
  emit(".card canvas{height:210px!important;width:100%!important}\n")
  emit(".card.full{grid-column:1/-1;min-height:auto}\n")
  emit(".card.full canvas{height:280px!important}\n")
  emit(".scroll{overflow-x:auto;margin-top:8px}\n")
  emit("table.corr{border-collapse:collapse;font-size:.76rem}\n")
  emit("table.corr th,table.corr td{padding:5px 9px;border:1px solid var(--border);")
  emit("text-align:center;min-width:54px}\n")
  emit("table.corr th{background:#21262d;color:var(--muted);font-weight:600}\n")
  emit("table.stats{width:100%;border-collapse:collapse;font-size:.81rem}\n")
  emit("table.stats th{background:#21262d;color:var(--muted);padding:6px 10px;")
  emit("text-align:left;font-weight:600;border-bottom:1px solid var(--border)}\n")
  emit("table.stats td{padding:5px 10px;border-bottom:1px solid #21262d;font-family:monospace}\n")
  emit("table.stats tr:hover td{background:#21262d44}\n")
  emit("table.layers{width:max-content;min-width:100%;border-collapse:collapse;font-size:.76rem}\n")
  emit("table.layers th{position:sticky;top:0;background:#21262d;color:var(--muted);padding:6px 9px;text-align:left;border-bottom:1px solid var(--border)}\n")
  emit("table.layers td{padding:5px 9px;border-bottom:1px solid #21262d;font-family:monospace;white-space:nowrap}\n")
  emit("table.layers tr:hover td{background:#21262d44}\n")
  emit("footer{text-align:center;font-size:.72rem;color:var(--muted);margin-top:24px;")
  emit("padding-top:14px;border-top:1px solid var(--border)}\n")
  -- Bouton de sauvegarde (visible uniquement en mode watch)
  emit(".sg-save-btn{position:fixed;top:18px;right:20px;z-index:999;")
  emit("background:#238636;color:#fff;border:1px solid #2ea043;border-radius:6px;")
  emit("padding:7px 14px;font-size:.8rem;font-weight:600;cursor:pointer;")
  emit("transition:background 0.2s,transform 0.1s}\n")
  emit(".sg-save-btn:hover{background:#2ea043}\n")
  emit(".sg-save-btn:active{transform:scale(0.96)}\n")
  emit(".sg-toast{position:fixed;bottom:24px;right:20px;z-index:999;")
  emit("background:#161b22;border:1px solid #30363d;border-radius:8px;")
  emit("padding:10px 16px;font-size:.8rem;color:#c9d1d9;")
  emit("opacity:0;transform:translateY(8px);")
  emit("transition:opacity 0.25s,transform 0.25s;pointer-events:none}\n")
  emit(".sg-toast.sg-show{opacity:1;transform:none}\n")
  emit("</style>\n")
  -- Script d'initialisation : registre des charts + fonctions de mise à jour DOM.
  -- _sg_apply(d)       : met à jour les datasets Chart.js sans rechargement de page.
  -- _sg_fetch_update() : récupère /graph_data.json puis appelle _sg_apply.
  emit("<script>\n")
  emit("window._sg_charts={};\n")
  emit("function _sg_apply(d){\n")
  emit("  var M=window._sg_chart_meta||{},C=window._sg_charts||{};\n")
  emit("  function xy(xc,yc){\n")
  emit("    var xs=d[xc]||[],ys=d[yc]||[],p=[];\n")
  emit("    for(var i=0;i<Math.min(xs.length,ys.length);i++)\n")
  emit("      if(typeof xs[i]==='number'&&typeof ys[i]==='number')\n")
  emit("        p.push({x:xs[i],y:ys[i]});\n")
  emit("    return p;\n")
  emit("  }\n")
  emit("  for(var id in M){\n")
  emit("    var m=M[id],ch=C[id];if(!ch)continue;\n")
  emit("    if(m.type==='line'){\n")
  emit("      ch.data.datasets[0].data=xy(m.xcol,m.ycol);\n")
  emit("      ch.update('none');\n")
  emit("    }else if(m.type==='scatter'){\n")
  emit("      ch.data.datasets[0].data=xy(m.xcol,m.ycol);\n")
  emit("      ch.update('none');\n")
  emit("    }else if(m.type==='val_overlay'){\n")
  emit("      ch.data.datasets[0].data=xy(d.opt_step?'opt_step':'step','loss');\n")
  emit("      ch.data.datasets[1].data=(d.val_pts||[]).map(function(p){return{x:p.step,y:p.val_loss};});\n")
  emit("      ch.update('none');\n")
  emit("    }else if(m.type==='val_gap'){\n")
  emit("      var vp=(d.val_pts||[]).filter(function(p){return p.gap!=null;});\n")
  emit("      ch.data.labels=vp.map(function(p){return String(Math.floor(p.step));});\n")
  emit("      var gv=vp.map(function(p){return p.gap;});\n")
  emit("      ch.data.datasets[0].data=gv;\n")
  emit("      ch.data.datasets[0].backgroundColor=gv.map(function(v){return v>0?'#E6394666':'#2A9D8F66';});\n")
  emit("      ch.data.datasets[0].borderColor=gv.map(function(v){return v>0?'#E63946':'#2A9D8F';});\n")
  emit("      ch.update('none');\n")
  emit("    }\n")
  emit("  }\n")
  emit("  var b=document.body;\n")
  emit("  b.style.transition='opacity 0.12s ease';\n")
  emit("  b.style.opacity='0.65';\n")
  emit("  setTimeout(function(){b.style.opacity='1';},160);\n")
  emit("}\n")
  emit("function _sg_fetch_update(){\n")
  emit("  fetch('/graph_data.json')\n")
  emit("    .then(function(r){return r.json();})\n")
  emit("    .then(_sg_apply)\n")
  emit("    .catch(function(){});\n")
  emit("}\n")
  emit("</script>\n")
  emit("</head>\n<body>\n")

  -- Header
  emit("<header>\n<h1>📈 Training Metrics Dashboard</h1>\n<div class='meta'>\n")
  if ctx.model then
    emit("  <span>🧠 <b>Modèle :</b> " .. ctx.model .. "</span>\n")
  end
  if ctx.algo then
    emit("  <span>⚙️ <b>Algorithme d'optimisation :</b> " .. ctx.algo .. "</span>\n")
  end
  if ctx.recon_loss then
    local loss = ctx.recon_loss .. (ctx.recon_loss_details and (" [" .. ctx.recon_loss_details .. "]") or "")
    emit("  <span>📐 <b>Recon loss :</b> " .. loss .. "</span>\n")
  end
  if ctx.meta and ctx.meta.source then
    emit("  <span>📦 <b>Source :</b> " .. ctx.meta.source .. "</span>\n")
  end
  emit(string.format("  <span>📊 <b>Steps :</b> %d &nbsp;|&nbsp; <b>Epochs :</b> %d</span>\n", n, #epochs))
  if ctx.n_dataset and ctx.n_dataset > 0 then
    emit(string.format("  <span>🗂️ <b>Dataset :</b> %d items/epoch</span>\n", ctx.n_dataset))
  end
  if ctx.validate_every_steps and ctx.validate_every_steps > 0 then
    local frac = (ctx.n_dataset and ctx.n_dataset > 0)
      and (" = 1/" .. metric_number(ctx.n_dataset / ctx.validate_every_steps) .. " epoch") or ""
    emit(string.format("  <span>✅ <b>Val :</b> tous les %d opt_steps%s</span>\n",
      ctx.validate_every_steps, frac))
  end
  if ctx.validate_items and ctx.validate_items > 0 then
    emit(string.format("  <span>🔬 <b>Items/val :</b> %d</span>\n", ctx.validate_items))
  end
  if ctx.validate_holdout ~= nil then
    emit("  <span>🧪 <b>Holdout :</b> " .. (ctx.validate_holdout and "activé" or "désactivé") .. "</span>\n")
  end
  if ctx.validate_holdout_frac then
    emit("  <span>📐 <b>Holdout frac :</b> " .. metric_number(ctx.validate_holdout_frac) .. "</span>\n")
  end
  if ctx.validate_holdout_items then
    emit(string.format("  <span>🔢 <b>Holdout items :</b> %d</span>\n", ctx.validate_holdout_items))
  end
  emit("  <span>📄 <b>CSV :</b> " .. ctx.csv_path .. " (" .. human_bytes(ctx.csv_size) .. ")</span>\n")
  emit("</div>\n</header>\n")

  -- Modèle: même contenu exhaustif que les rapports Markdown et TXT.
  local model_info = ctx.meta and ctx.meta.model_info
  emit("\n<section><h2>Modèle et graphe interne</h2><div class='grid'>\n")
  if model_info then
    local cfg = model_info.config or {}
    local image_shape = (cfg.image_w and cfg.image_h and cfg.image_c)
      and string.format("%gx%gx%g (W×H×C)", cfg.image_w, cfg.image_h, cfg.image_c) or nil
    local latent_shape = (cfg.latent_w and cfg.latent_h and cfg.latent_c)
      and string.format("%gx%gx%g (W×H×C)", cfg.latent_w, cfg.latent_h, cfg.latent_c) or nil
    emit("<div class='card full'><h3>Synthèse du modèle</h3>")
    html_table({"Propriété", "Valeur"}, {
      {"Classe sérialisée", model_info.model_name or ctx.model},
      {"Dimensions image", image_shape},
      {"Dimensions latentes", latent_shape},
      {"Dimension latente totale", cfg.latent_dim},
      {"Couches du graphe", model_info.num_layers},
      {"Couches avec paramètres entraînables", model_info.trainable_layers},
      {"Opérations sans paramètres", model_info.parameterless_layers},
      {"Nombre de paramètres", model_info.total_params},
      {"Paramètres entraînables", model_info.trainable_params},
    })
    emit("</div>\n")

    local config_rows = {}
    for _, key in ipairs(html_sorted_keys(model_info.config)) do
      config_rows[#config_rows + 1] = {key, model_info.config[key]}
    end
    if #config_rows > 0 then
      emit("<div class='card full'><h3>Configuration complète du modèle</h3>")
      html_table({"Option", "Valeur"}, config_rows); emit("</div>\n")
    end

    local type_rows = {}
    for _, key in ipairs(html_sorted_keys(model_info.layer_types)) do
      type_rows[#type_rows + 1] = {key, model_info.layer_types[key]}
    end
    if #type_rows > 0 then
      emit("<div class='card full'><h3>Composition du graphe</h3>")
      html_table({"Type de couche", "Occurrences"}, type_rows); emit("</div>\n")
    end

    if #model_info.layers > 0 then
      local layer_rows = {}
      for index, layer in ipairs(model_info.layers) do
        layer_rows[#layer_rows + 1] = {
          index, layer.name, layer.type, layer.inputs, layer.output,
          layer.in_channels ~= 0 and layer.in_channels or layer.in_features,
          layer.out_channels ~= 0 and layer.out_channels or layer.out_features,
          layer.kernel_size, layer.stride, layer.padding, layer.params_count,
          layer.is_trainable,
        }
      end
      emit("<div class='card full'><h3>Détail de toutes les couches</h3>")
      html_table({"#", "Nom", "Type", "Entrées", "Sortie", "In", "Out", "Kernel", "Stride", "Padding", "Paramètres", "Entraînable"}, layer_rows, "layers")
      emit("</div>\n")
    end
  else
    emit("<div class='card full'><p>Aucune architecture sérialisée trouvée dans le checkpoint.</p></div>\n")
  end
  emit("</div></section>\n")
  if #(ctx.warmups or {}) > 0 then
    local rows = {}
    for _, warmup in ipairs(ctx.warmups) do
      rows[#rows + 1] = {warmup.metric, warmup.key, warmup.steps, "actif"}
    end
    emit("\n<section><h2>Warmups actifs</h2><div class='grid'>\n")
    emit("<div class='card full'><h3>Phases de warmup détectées</h3>")
    html_table({"Métrique", "Paramètre", "Fin au step", "État"}, rows)
    emit("</div></div></section>\n")
  end
  -- Bouton de sauvegarde (mode --watch uniquement)
  if ctx.sse_port and ctx.sse_port > 0 then
    emit(string.format(
      '<button class="sg-save-btn" onclick="_sg_save()">💾 Sauvegarder</button>\n'
      .. '<div class="sg-toast" id="sg_toast"></div>\n'
      .. '<script>\n'
      .. 'function _sg_save(){\n'
      .. '  fetch("http://127.0.0.1:%d/snapshot",{method:"POST"})\n'
      .. '    .then(function(r){return r.json();})\n'
      .. '    .then(function(j){\n'
      .. '      var name=j.saved||(j.file||"snapshot sauvegardé");\n'
      .. '      _sg_toast("✓ " + name);\n'
      .. '    })\n'
      .. '    .catch(function(e){_sg_toast("Erreur : "+e,true);});\n'
      .. '}\n'
      .. 'function _sg_toast(msg,err){\n'
      .. '  var t=document.getElementById("sg_toast");\n'
      .. '  t.textContent=msg;\n'
      .. '  t.style.borderColor=err?"#f85149":"#2ea043";\n'
      .. '  t.classList.add("sg-show");\n'
      .. '  setTimeout(function(){t.classList.remove("sg-show");},3500);\n'
      .. '}\n'
      .. '</script>\n', ctx.sse_port))
  end

  -- Section: métriques globales
  emit("\n<section><h2>Métriques globales</h2><div class='grid'>\n")
  emit(card_line("c_loss",  "Loss",               "loss",              PALETTE.loss,     false))
  if rc then
    emit(card_line("c_rc",  rl,                   rc,                  PALETTE.recon,    false))
  end
  emit(card_line("c_lr",    "Learning Rate",       "learning_rate",    PALETTE.lr,       true))
  emit(card_line("c_kl",    "KL Divergence",       "kl_divergence",    PALETTE.kl,       false))
  emit(card_line("c_kl_beta", "KL Beta",           "kl_beta_effective",PALETTE.kl_beta,  false))
  emit(card_line("c_wass",  "Wasserstein",         "wasserstein",      PALETTE.wass,     false))
  emit(card_line("c_ent",   "Entropy Δ",           "entropy_diff",     PALETTE.entropy,  false))
  emit(card_line("c_mom",   "Moment Mismatch",     "moment_mismatch",  PALETTE.moment,   false))
  emit(card_line("c_spat",  "Spatial Coherence",   "spatial_coherence",PALETTE.spatial,  false))
  emit(card_line("c_temp",  "Temporal Consistency","temporal_consistency",PALETTE.temporal,false))
  emit(card_line("c_ts",    "Timestep",            "timestep",         PALETTE.timestep, false))
  emit(card_line("c_eps",   "opt_eps",             "opt_eps",          PALETTE.opt_eps,  false))
  emit(card_line("c_mem",   "Mémoire RSS (MB)",     "memory_mb",        PALETTE.memory,   false))
  emit(card_line("c_alloc", "Allocateur (MB)",      "allocator_memory_mb", PALETTE.allocator, false))
  emit("</div></section>\n")

  local charted_metrics = {
    loss=true, learning_rate=true, kl_divergence=true, kl_beta_effective=true,
    wasserstein=true,
    entropy_diff=true, moment_mismatch=true, spatial_coherence=true,
    temporal_consistency=true, timestep=true, opt_eps=true,
    memory_mb=true, allocator_memory_mb=true,
  }
  if rc then charted_metrics[rc] = true end
  local structural_columns = {
    step=true, epoch=true, total_epochs=true, batch=true, total_batches=true,
    params=true, opt_type=true, opt_step=true, val_step=true, val_feedback=true,
    val_rewarded=true, val_penalized=true, val_loss=true, val_mse=true,
  }
  local extra_metrics = {}
  for _, name in ipairs(ctx.headers or {}) do
    if not structural_columns[name] and not charted_metrics[name] and has(name) then
      local numeric = false
      for _, value in ipairs(df[name]) do if type(value) == "number" then numeric = true; break end end
      if numeric then extra_metrics[#extra_metrics + 1] = name end
    end
  end
  if #extra_metrics > 0 then
    local extra_palette = {"#58A6FF", "#D2A8FF", "#7EE787", "#FFA657", "#FF7B72", "#79C0FF"}
    emit("\n<section><h2>Métriques additionnelles</h2><div class='grid'>\n")
    for index, name in ipairs(extra_metrics) do
      local id = "c_extra_" .. name:gsub("[^%w_]", "_")
      emit(card_line(id, name, name, extra_palette[((index - 1) % #extra_palette) + 1], false))
    end
    emit("</div></section>\n")
  end

  -- Section: validation + écart objectif/erreurs
  local val_pts  = ctx.val_pts or {}
  local val_cols = {}
  for _, c in ipairs(ctx.headers or {}) do
    if c:match("^val_") and has(c) then val_cols[#val_cols+1] = c end
  end

  if #val_pts > 0 or #val_cols > 0 then
    emit("\n<section><h2>Validation &amp; Écart objectif</h2><div class='grid'>\n")

    if #val_pts > 0 then
      -- Sérialisation des points de validation
      local vsc, gsv, gss = {}, {}, {}
      for _, p in ipairs(val_pts) do
        vsc[#vsc+1] = "{x:" .. metric_number(p.step) .. ",y:" .. metric_number(p.val_loss) .. "}"
        if p.gap then
          gsv[#gsv+1] = metric_number(p.gap)
          gss[#gss+1] = string.format("%d", math.floor(p.step))
        end
      end
      local val_js = "[" .. table.concat(vsc, ",") .. "]"
      local gap_v  = "[" .. table.concat(gsv, ",") .. "]"
      local gap_s  = '["' .. table.concat(gss, '","') .. '"]'

      -- Overlay : courbe train (ligne) + points val (scatter alignés sur step)
      emit('<div class="card full">\n')
      emit('<h3>Train vs Validation Loss')
      emit(' <small style="font-weight:normal;color:var(--muted)">')
      emit('&mdash; points = étapes de validation</small></h3>\n')
      emit('<canvas id="c_val_ov"></canvas></div>\n')
      chart_meta["c_val_ov"] = '{type:"val_overlay"}'
      emit('<script>\n')
      emit('window._sg_charts["c_val_ov"]=new Chart(document.getElementById("c_val_ov"),{\n')
      emit('  type:"scatter",\n')
      emit('  data:{datasets:[\n')
      emit('    {label:"Train loss",type:"line",\n')
      emit('     data:' .. jdxy(x_key, "loss") .. ',\n')
      emit('     borderColor:"' .. PALETTE.loss .. '",\n')
      emit('     backgroundColor:"' .. PALETTE.loss .. '22",\n')
      emit('     borderWidth:1.5,pointRadius:0,fill:true,showLine:true,tension:0.2,order:2},\n')
      emit('    {label:"Val loss",\n')
      emit('     data:' .. val_js .. ',\n')
      emit('     borderColor:"' .. PALETTE.val_loss .. '",\n')
      emit('     backgroundColor:"' .. PALETTE.val_loss .. 'CC",\n')
      emit('     pointRadius:6,pointHoverRadius:9,showLine:false,order:1}\n')
      emit('  ]},\n')
      emit('  options:{responsive:true,maintainAspectRatio:false,animation:false,\n')
      emit('    plugins:{legend:{display:true,labels:{boxWidth:10,font:{size:10}}},\n')
      emit('             annotation:{annotations:' .. chart_annotations_js("loss") .. '}},\n')
      emit('    scales:{x:{title:{display:true,text:"Step"},ticks:{maxTicksLimit:10}},\n')
      emit('            y:{title:{display:true,text:"Loss"}}}}\n')
      emit('});\n</script>\n')

      -- Histogramme des écarts val - train (rouge=surapprentissage, vert=sous)
      if #gsv > 0 then
        emit('<div class="card full">\n')
        emit('<h3>Écart Val &minus; Train')
        emit(' <small style="font-weight:normal;color:var(--muted)">')
        emit('&nbsp;<span style="color:#E63946">&#9632;</span> sur-apprentissage (val&gt;train)')
        emit('&nbsp;&nbsp;<span style="color:#2A9D8F">&#9632;</span> sous-apprentissage')
        emit('</small></h3>\n')
        emit('<canvas id="c_gap"></canvas></div>\n')
        chart_meta["c_gap"] = '{type:"val_gap"}'
        emit('<script>(function(){\n')
        emit('  var gv=' .. gap_v .. ', gs=' .. gap_s .. ';\n')
        emit('  var bg=gv.map(function(v){return v>0?"#E6394666":"#2A9D8F66";});\n')
        emit('  var brd=gv.map(function(v){return v>0?"#E63946":"#2A9D8F";});\n')
        emit('  window._sg_charts["c_gap"]=new Chart(document.getElementById("c_gap"),{\n')
        emit('    type:"bar",\n')
        emit('    data:{labels:gs,datasets:[{\n')
        emit('      label:"val-train",data:gv,backgroundColor:bg,borderColor:brd,borderWidth:1\n')
        emit('    }]},\n')
        emit('    options:{responsive:true,maintainAspectRatio:false,animation:false,\n')
        emit('      plugins:{legend:{display:false},\n')
        emit('               tooltip:{callbacks:{label:function(c){\n')
        emit('                 return(c.raw>0?"Surapprentissage":"Sous-appr.")+": "+c.raw.toFixed(5);\n')
        emit('               }}}},\n')
        emit('      scales:{\n')
        emit('        x:{title:{display:true,text:"Step (validation)"},ticks:{maxTicksLimit:15}},\n')
        emit('        y:{title:{display:true,text:"Écart (val-train)"},\n')
        emit('           grid:{color:function(c){\n')
        emit('             return c.tick&&c.tick.value===0?"#8b949e":"#30363d";\n')
        emit('           }}}\n')
        emit('      }}\n')
        emit('  });\n')
        emit('})();</script>\n')
      end
    end

    -- Val MSE (métrique secondaire) : scatter depuis val_pts si disponible.
    -- val_mse = eps-space MSE pour DDPM, KL pour VAE.
    -- val_step n'est pas une métrique à tracer (index de step uniquement).
    do
      local has_vm = false
      for _, p in ipairs(val_pts) do
        if p.val_mse then has_vm = true; break end
      end
      if has_vm then
        local vmsc = {}
        for _, p in ipairs(val_pts) do
          if p.val_mse then
            vmsc[#vmsc+1] = "{x:" .. metric_number(p.step) .. ",y:" .. metric_number(p.val_mse) .. "}"
          end
        end
        local vm_js = "[" .. table.concat(vmsc, ",") .. "]"
        chart_meta["c_val_mse"] = '{type:"val_mse_scatter"}'
        emit('<div class="card full">\n')
        emit('<h3>' .. html_escape(val_rl))
        emit(' <small style="font-weight:normal;color:var(--muted)">')
        emit('algorithme de reconstruction utilisé pendant l’entraînement')
        emit('</small></h3>\n')
        emit('<canvas id="c_val_mse"></canvas></div>\n')
        emit('<script>\n')
        emit('window._sg_charts["c_val_mse"]=new Chart(document.getElementById("c_val_mse"),{\n')
        emit('  type:"scatter",\n')
        emit('  data:{datasets:[{label:' .. js_s(val_rl) .. ',data:' .. vm_js .. ',\n')
        emit('    borderColor:"' .. PALETTE.val_recon .. '",\n')
        emit('    backgroundColor:"' .. PALETTE.val_recon .. 'CC",\n')
        emit('    pointRadius:6,pointHoverRadius:9,showLine:false}]},\n')
        emit('  options:{responsive:true,maintainAspectRatio:false,animation:false,\n')
        emit('    plugins:{legend:{display:false}},\n')
        emit('    scales:{x:{title:{display:true,text:"Step"},ticks:{maxTicksLimit:10}},\n')
        emit('            y:{title:{display:true,text:' .. js_s(val_rl) .. '}}}}\n')
        emit('});\n</script>\n')
      end
    end

    -- Autres colonnes val_* : exclure val_loss (overlay), val_mse (scatter ci-dessus),
    -- val_step (index de step, pas une métrique significative).
    for _, vc in ipairs(val_cols) do
      if vc ~= "val_loss" and vc ~= "val_mse" and vc ~= "val_step" then
        emit(card_line("c_" .. vc:gsub("[^%w]","_"),
          vc:gsub("^val_", "Val "), vc, PALETTE.val_loss, false))
      end
    end

    emit("</div></section>\n")
  end

  -- Section: loss par epoch
  if #ep_ds > 0 then
    emit("\n<section><h2>Loss par epoch</h2><div class='grid'>\n")
    emit("<div class='card full'><h3>Loss par epoch</h3>")
    emit("<canvas id='c_ep'></canvas></div>\n")
    emit(string.format(
      "<script>new Chart(document.getElementById('c_ep'),{type:'scatter',"
      .. "data:{datasets:[%s]},"
      .. "options:{responsive:true,maintainAspectRatio:false,animation:false,"
      .. "plugins:{legend:{display:%s,labels:{boxWidth:10,font:{size:8}}}},"
      .. "scales:{x:{title:{display:true,text:'Step'},ticks:{maxTicksLimit:10}},"
      .. "y:{title:{display:true,text:'Loss'}}},"
      .. "elements:{point:{radius:0}}}});</script>\n",
      table.concat(ep_ds, ","),
      (#epochs <= 20) and "true" or "false"))
    emit("</div></section>\n")
  end

  -- Section: distributions
  local has_dist = (hlbl_loss and #hlbl_loss > 0) or (hlbl_rc and #hlbl_rc > 0)
  if has_dist then
    emit("\n<section><h2>Distributions</h2><div class='grid'>\n")
    emit(card_bar("c_hl",  "Distribution Loss",       hlbl_loss, hcnt_loss, PALETTE.loss))
    if rc then
      emit(card_bar("c_hrc", "Distribution " .. rl,   hlbl_rc,  hcnt_rc,   PALETTE.recon))
    end
    emit("</div></section>\n")
  end

  -- Section: scatter / relations
  local has_scat = has("timestep") or (rc and has(rc)) or has("learning_rate")
  if has_scat then
    emit("\n<section><h2>Relations</h2><div class='grid'>\n")
    emit(card_scatter("c_sc1", "Loss vs Timestep",
      "timestep", "loss", PALETTE.timestep, "Timestep", "Loss"))
    if rc then
      emit(card_scatter("c_sc2", "Loss vs " .. rl,
        rc, "loss", PALETTE.recon, rl, "Loss"))
    end
    emit(card_scatter("c_sc3", "Loss vs LR",
      "learning_rate", "loss", PALETTE.lr, "Learning Rate", "Loss"))
    if rc and has("timestep") then
      emit(card_scatter("c_sc4", rl .. " vs Timestep",
        "timestep", rc, PALETTE.temporal, "Timestep", rl))
    end
    if has("kl_divergence") and has("timestep") then
      emit(card_scatter("c_sc5", "KL vs Timestep",
        "timestep", "kl_divergence", PALETTE.kl, "Timestep", "KL"))
    end
    emit("</div></section>\n")
  end

  -- Section: corrélation
  if corr_tbl ~= "" then
    emit("\n<section><h2>Corrélation (Pearson)</h2><div class='grid'>\n")
    emit("<div class='card full'><div class='scroll'><table class='corr'>\n")
    emit(corr_tbl)
    emit("</table></div></div>\n</div></section>\n")
  end

  -- Section: statistiques tabulaires
  emit("\n<section><h2>Statistiques</h2><div class='grid'>\n")
  emit("<div class='card full'><table class='stats'>\n")
  emit("<tr><th>Métrique</th><th>Min</th><th>Max</th><th>μ (mean)</th>")
  emit("<th>σ (std)</th><th>N</th></tr>\n")
  local emitted_stats = {}
  local function stat_row(lbl, cname)
    if not has(cname) then return end
    emitted_stats[cname] = true
    local s = col_stats(df[cname])
    emit(string.format(
      "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%d</td></tr>\n",
      lbl, metric_number(s.min), metric_number(s.max), metric_number(s.mean),
      metric_number(s.std), s.n))
  end
  stat_row("Loss",                "loss")
  if rc then stat_row(rl,         rc)               end
  stat_row("Learning Rate",       "learning_rate")
  stat_row("KL Divergence",       "kl_divergence")
  stat_row("KL Beta",             "kl_beta_effective")
  stat_row("Wasserstein",         "wasserstein")
  stat_row("Entropy Δ",           "entropy_diff")
  stat_row("Moment Mismatch",     "moment_mismatch")
  stat_row("Spatial Coherence",   "spatial_coherence")
  stat_row("Temporal Consistency","temporal_consistency")
  stat_row("Timestep",            "timestep")
  stat_row("opt_eps",             "opt_eps")
  stat_row("Mémoire RSS (MB)",    "memory_mb")
  stat_row("Allocateur (MB)",     "allocator_memory_mb")
  for _, name in ipairs(ctx.headers or {}) do
    if not structural_columns[name] and not emitted_stats[name] and has(name) then
      local numeric = false
      for _, value in ipairs(df[name]) do if type(value) == "number" then numeric = true; break end end
      if numeric then stat_row(name, name) end
    end
  end
  emit("</table></div>\n</div></section>\n")

  emit("\n<footer>Généré par <b>show-graph.lua</b> — Mímir Framework</footer>\n")

  -- Émettre le registre de méta-données de charts pour _sg_apply() (côté JS).
  do
    local parts = {}
    for id, ms in pairs(chart_meta) do
      parts[#parts+1] = js_s(id) .. ":" .. ms
    end
    emit("<script>window._sg_chart_meta={" .. table.concat(parts, ",") .. "};</script>\n")
  end

  -- Listener SSE : mise à jour DOM partielle (pas de location.reload()).
  -- Fallback timer : reload complet si pas de serveur SSE.
  if ctx.sse_port and ctx.sse_port > 0 then
    -- _sg_fetch_update() définie dans le <script> d'init du <head>.
    emit('<script>\n')
    emit(string.format(
      'var _es=new EventSource("http://127.0.0.1:%d/events");\n', ctx.sse_port))
    emit('_es.onmessage=function(e){if(e.data==="update") _sg_fetch_update();};\n')
    emit('</script>\n')
  elseif ctx.watch_interval and ctx.watch_interval > 0 then
    -- Fallback timer : reload complet (sans serveur SSE)
    emit('<script>(function(){\n')
    emit('function _sg_reload(){\n')
    emit('  document.body.style.transition="opacity 0.3s ease,transform 0.3s ease";\n')
    emit('  document.body.style.opacity="0";\n')
    emit('  document.body.style.transform="translateY(-4px)";\n')
    emit('  setTimeout(function(){location.reload();},600);\n')
    emit('}\n')
    emit(string.format('setTimeout(_sg_reload,%d*1000);\n', ctx.watch_interval))
    emit('})();</script>\n')
  end

  emit("</body>\n</html>\n")

  return table.concat(out)
end

local function sorted_keys(t)
  local keys = {}
  for k in pairs(t or {}) do keys[#keys + 1] = k end
  table.sort(keys)
  return keys
end

local function report_value(v)
  if type(v) == "boolean" then return v and "true" or "false" end
  if type(v) == "number" then return metric_number(v) end
  if v == nil or v == "" then return "-" end
  return tostring(v):gsub("[\r\n]+", " ")
end

local function numeric_values(col)
  local values, indices = {}, {}
  for index in pairs(col or {}) do
    if type(index) == "number" then indices[#indices + 1] = index end
  end
  table.sort(indices)
  for _, index in ipairs(indices) do
    local v = col[index]
    if type(v) == "number" and v == v and v ~= math.huge and v ~= -math.huge then
      values[#values + 1] = v
    end
  end
  return values
end

local function ascii_graph(col, width, height)
  local values = numeric_values(col)
  if #values == 0 then return nil end
  width, height = math.min(width or 72, #values), height or 10
  local sampled = {}
  for x = 1, width do
    local idx = math.floor((x - 1) * (#values - 1) / math.max(1, width - 1)) + 1
    sampled[x] = values[idx]
  end
  local stats = col_stats(sampled)
  local grid = {}
  for y = 1, height do grid[y] = {} end
  for x, v in ipairs(sampled) do
    local y = stats.max == stats.min and math.ceil(height / 2)
      or height - math.floor((v - stats.min) / (stats.max - stats.min) * (height - 1))
    grid[y][x] = "*"
  end
  local lines = {}
  for y = 1, height do
    local row = {}
    for x = 1, width do row[x] = grid[y][x] or " " end
    local level = stats.max - (y - 1) * (stats.max - stats.min) / math.max(1, height - 1)
    lines[#lines + 1] = string.format("%12s |%s", metric_number(level), table.concat(row))
  end
  lines[#lines + 1] = string.rep(" ", 13) .. "+" .. string.rep("-", width)
  return table.concat(lines, "\n")
end

local svg_graph_serial = 0
local function svg_graph(col, title, x_col, warmup_step, curve_strategy)
  curve_strategy = tostring(curve_strategy or "linear"):lower()
  if curve_strategy == "expo" then curve_strategy = "exponential" end
  local samples = {}
  for index, value in pairs(col or {}) do
    if type(value) == "number" then
      local x = x_col and x_col[index] or index
      if type(x) == "number" then samples[#samples + 1] = {x = x, y = value} end
    end
  end
  if #samples == 0 then return nil end
  table.sort(samples, function(a, b) return a.x < b.x end)

  local y_values, x_values = {}, {}
  for _, point in ipairs(samples) do
    x_values[#x_values + 1] = point.x
    y_values[#y_values + 1] = point.y
  end
  local ys, xs = col_stats(y_values), col_stats(x_values)
  local y_labels = {}
  local longest_y = 0
  for grid = 0, 4 do
    local value = ys.max - grid * (ys.max - ys.min) / 4
    y_labels[grid + 1] = metric_number(value)
    longest_y = math.max(longest_y, #y_labels[grid + 1])
  end

  local width, height = 1100, 360
  local left = math.max(96, math.min(260, 24 + longest_y * 7))
  local right, top, bottom = 34, 62, 62
  local plot_w, plot_h = width - left - right, height - top - bottom
  local function px(value)
    if xs.max == xs.min then return left + plot_w / 2 end
    return left + (value - xs.min) * plot_w / (xs.max - xs.min)
  end
  local function py(value)
    if ys.max == ys.min then return top + plot_h / 2 end
    return top + (ys.max - value) * plot_h / (ys.max - ys.min)
  end
  local function esc(value)
    return tostring(value):gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
      :gsub('"', "&quot;")
  end

  local selected = {}
  local point_limit = 600
  local selected_count = math.min(point_limit, #samples)
  for index = 1, selected_count do
    local source = math.floor((index - 1) * (#samples - 1) /
      math.max(1, selected_count - 1)) + 1
    selected[#selected + 1] = samples[source]
  end
  local screen = {}
  for _, point in ipairs(selected) do
    screen[#screen + 1] = {x = px(point.x), y = py(point.y)}
  end
  local path = {string.format("M%.2f %.2f", screen[1].x, screen[1].y)}
  if curve_strategy == "step" then
    for index = 2, #screen do
      path[#path + 1] = string.format("H%.2f V%.2f", screen[index].x, screen[index].y)
    end
  elseif curve_strategy == "cosine" or curve_strategy == "exponential" then
    -- Catmull-Rom converti en Bézier cubique : la courbe passe par chaque
    -- mesure réelle, sans remplacer les données par une formule théorique.
    for index = 1, #screen - 1 do
      local p0 = screen[math.max(1, index - 1)]
      local p1 = screen[index]
      local p2 = screen[index + 1]
      local p3 = screen[math.min(#screen, index + 2)]
      local c1x = p1.x + (p2.x - p0.x) / 6
      local c1y = math.max(top, math.min(top + plot_h, p1.y + (p2.y - p0.y) / 6))
      local c2x = p2.x - (p3.x - p1.x) / 6
      local c2y = math.max(top, math.min(top + plot_h, p2.y - (p3.y - p1.y) / 6))
      path[#path + 1] = string.format("C%.2f %.2f %.2f %.2f %.2f %.2f",
        c1x, c1y, c2x, c2y, p2.x, p2.y)
    end
  else
    for index = 2, #screen do
      path[#path + 1] = string.format("L%.2f %.2f", screen[index].x, screen[index].y)
    end
  end
  local first, last = selected[1], selected[#selected]
  local area_path = table.concat(path, " ") .. string.format(" L%.2f %d L%.2f %d Z",
    px(last.x), top + plot_h, px(first.x), top + plot_h)

  svg_graph_serial = svg_graph_serial + 1
  local gradient_id = "sg_fill_" .. svg_graph_serial
  local clip_id = "sg_clip_" .. svg_graph_serial
  local out = {
    string.format('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" role="img" aria-label="%s" style="display:block;max-width:100%%;height:auto;background:#0d1117;border:1px solid #30363d;border-radius:10px">', width, height, esc(title)),
    '<rect width="100%" height="100%" fill="#0d1117" rx="10"/>',
    string.format('<defs><linearGradient id="%s" x1="0" y1="0" x2="0" y2="1"><stop offset="0%%" stop-color="#58a6ff" stop-opacity="0.30"/><stop offset="100%%" stop-color="#58a6ff" stop-opacity="0.02"/></linearGradient><clipPath id="%s"><rect x="%d" y="%d" width="%d" height="%d" rx="4"/></clipPath></defs>', gradient_id, clip_id, left, top, plot_w, plot_h),
    string.format('<text x="%d" y="28" fill="#f0f6fc" font-family="sans-serif" font-size="17" font-weight="600">%s</text>', left, esc(title)),
    string.format('<text x="%d" y="47" fill="#8b949e" font-family="sans-serif" font-size="11">%d points · stratégie %s · min %s · max %s</text>', left, #samples, esc(curve_strategy), esc(metric_number(ys.min)), esc(metric_number(ys.max))),
    string.format('<rect x="%d" y="%d" width="%d" height="%d" rx="4" fill="#161b22" stroke="#30363d"/>', left, top, plot_w, plot_h),
  }

  for grid = 0, 4 do
    local y = top + grid * plot_h / 4
    out[#out + 1] = string.format('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#30363d" stroke-width="1"/>', left, y, width - right, y)
    out[#out + 1] = string.format('<text x="%d" y="%.2f" fill="#8b949e" font-family="monospace" font-size="10" text-anchor="end">%s</text>', left - 10, y + 4, esc(y_labels[grid + 1]))
  end
  for grid = 0, 5 do
    local x = left + grid * plot_w / 5
    local value = xs.min + grid * (xs.max - xs.min) / 5
    out[#out + 1] = string.format('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#21262d" stroke-width="1"/>', x, top, x, top + plot_h)
    out[#out + 1] = string.format('<text x="%.2f" y="%d" fill="#8b949e" font-family="monospace" font-size="10" text-anchor="middle">%s</text>', x, top + plot_h + 22, esc(metric_number(value)))
  end

  out[#out + 1] = string.format('<g clip-path="url(#%s)"><path d="%s" fill="url(#%s)"/><path d="%s" fill="none" stroke="#58a6ff" stroke-width="2.25" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>', clip_id, area_path, gradient_id, table.concat(path, " "))
  out[#out + 1] = string.format('<circle cx="%.2f" cy="%.2f" r="3.5" fill="#79c0ff" stroke="#0d1117" stroke-width="1.5"/>', px(last.x), py(last.y))
  out[#out + 1] = string.format('<text x="%.2f" y="%d" fill="#8b949e" font-family="sans-serif" font-size="11" text-anchor="middle">step</text>', left + plot_w / 2, height - 14)

  if warmup_step and xs.max > xs.min and warmup_step >= xs.min and warmup_step <= xs.max then
    local marker_x = px(warmup_step)
    local label = "fin warmup " .. metric_number(warmup_step)
    local label_w = 18 + #label * 6.5
    local label_x = marker_x + 8
    if label_x + label_w > width - right then label_x = marker_x - label_w - 8 end
    out[#out + 1] = string.format('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#FFD166" stroke-width="2" stroke-dasharray="7 4"/>', marker_x, top, marker_x, top + plot_h)
    out[#out + 1] = string.format('<rect x="%.2f" y="%d" width="%.2f" height="22" rx="4" fill="#2d2a1f" stroke="#FFD166" stroke-opacity="0.65"/>', label_x, top + 8, label_w, 22)
    out[#out + 1] = string.format('<text x="%.2f" y="%d" fill="#FFD166" font-family="sans-serif" font-size="11">%s</text>', label_x + 8, top + 23, esc(label))
  end
  out[#out + 1] = '</svg>'
  return table.concat(out, "\n")
end

local function report_metric_columns(ctx)
  local excluded = {
    step=true, epoch=true, total_epochs=true, batch=true, total_batches=true,
    params=true, opt_type=true, opt_step=true, val_step=true, val_feedback=true,
    val_rewarded=true, val_penalized=true,
  }
  local metrics = {}
  for _, name in ipairs(ctx.headers or {}) do
    if not excluded[name] and #numeric_values(ctx.df[name]) > 0 then
      metrics[#metrics + 1] = name
    end
  end
  return metrics
end

local function report_metric_label(ctx, name)
  if name == "mse" then return ctx.recon_label or "Reconstruction" end
  if name == "val_mse" then
    return "Validation " .. (ctx.recon_label or "Reconstruction")
  end
  if name == "kl_beta_effective" then return "KL Beta" end
  return name
end

local function report_curve_strategy(ctx, name)
  local config = ctx.meta and ctx.meta.model_info and ctx.meta.model_info.config or {}
  if name == "learning_rate" then
    return config.decay_strategy or "linear"
  end
  if name == "kl_beta_effective" then
    -- Le KL beta suit son coefficient réellement enregistré. Le warmup natif
    -- est linéaire, sauf stratégie KL explicitement sérialisée par un modèle.
    return config.kl_decay_strategy or config.kl_beta_strategy or "linear"
  end
  return "linear"
end

local function gen_text_report(ctx, markdown)
  local out = {}
  local function emit(s) out[#out + 1] = s end
  local function heading(level, title)
    if markdown then emit(string.rep("#", level) .. " " .. title .. "\n\n")
    else emit(title .. "\n" .. string.rep(level == 1 and "=" or "-", #title) .. "\n\n") end
  end
  local function md_cell(v)
    return report_value(v):gsub("|", "\\|")
  end
  local function table_rows(headers, rows)
    if markdown then
      emit("| " .. table.concat(headers, " | ") .. " |\n")
      local sep = {}; for i = 1, #headers do sep[i] = "---" end
      emit("| " .. table.concat(sep, " | ") .. " |\n")
      for _, row in ipairs(rows) do
        local cells = {}; for i = 1, #headers do cells[i] = md_cell(row[i]) end
        emit("| " .. table.concat(cells, " | ") .. " |\n")
      end
      emit("\n")
    else
      local widths = {}
      for i, h in ipairs(headers) do widths[i] = #h end
      for _, row in ipairs(rows) do
        for i = 1, #headers do widths[i] = math.min(48, math.max(widths[i], #report_value(row[i]))) end
      end
      local function line(row)
        local cells = {}
        for i = 1, #headers do
          local value = report_value(row[i])
          if #value > widths[i] then value = value:sub(1, widths[i] - 1) .. "…" end
          cells[i] = value .. string.rep(" ", widths[i] - #value)
        end
        return "| " .. table.concat(cells, " | ") .. " |\n"
      end
      emit(line(headers))
      local bars = {}; for i = 1, #headers do bars[i] = string.rep("-", widths[i]) end
      emit("+-" .. table.concat(bars, "-+-") .. "-+\n")
      for _, row in ipairs(rows) do emit(line(row)) end
      emit("\n")
    end
  end

  heading(1, "Rapport d'entraînement Mímir — " .. (ctx.model or "modèle inconnu"))
  local summary = {
    {"Modèle", ctx.model}, {"Algorithme d'optimisation", ctx.algo},
    {"Recon loss", ctx.recon_loss}, {"Détails recon loss", ctx.recon_loss_details},
    {"Checkpoint", ctx.meta and ctx.meta.source}, {"CSV", ctx.csv_path},
    {"Taille CSV", human_bytes(ctx.csv_size)}, {"Steps d'entraînement", ctx.n},
    {"Epochs", #ctx.epochs}, {"Items par epoch", ctx.n_dataset},
    {"Validation tous les opt_steps", ctx.validate_every_steps},
    {"Items par validation", ctx.validate_items},
    {"Validation holdout", ctx.validate_holdout},
    {"Fraction holdout", ctx.validate_holdout_frac},
    {"Items holdout", ctx.validate_holdout_items},
  }
  table_rows({"Propriété", "Valeur"}, summary)

  local info = ctx.meta and ctx.meta.model_info
  heading(2, "Modèle et graphe interne")
  if info then
    local cfg = info.config or {}
    local image_shape = (cfg.image_w and cfg.image_h and cfg.image_c)
      and string.format("%gx%gx%g (W×H×C)", cfg.image_w, cfg.image_h, cfg.image_c) or nil
    local latent_shape = (cfg.latent_w and cfg.latent_h and cfg.latent_c)
      and string.format("%gx%gx%g (W×H×C)", cfg.latent_w, cfg.latent_h, cfg.latent_c) or nil
    local overview = {
      {"Classe sérialisée", info.model_name or ctx.model},
      {"Dimensions image", image_shape},
      {"Dimensions latentes", latent_shape},
      {"Dimension latente totale", cfg.latent_dim},
      {"Couches du graphe", info.num_layers},
      {"Couches avec paramètres entraînables", info.trainable_layers},
      {"Opérations sans paramètres", info.parameterless_layers},
      {"Nombre de paramètres", info.total_params},
      {"Paramètres entraînables", info.trainable_params},
    }
    table_rows({"Propriété", "Valeur"}, overview)

    local config_rows = {}
    for _, key in ipairs(sorted_keys(info.config)) do config_rows[#config_rows + 1] = {key, info.config[key]} end
    if #config_rows > 0 then heading(3, "Configuration complète du modèle"); table_rows({"Option", "Valeur"}, config_rows) end

    local type_rows = {}
    for _, key in ipairs(sorted_keys(info.layer_types)) do type_rows[#type_rows + 1] = {key, info.layer_types[key]} end
    if #type_rows > 0 then heading(3, "Composition du graphe"); table_rows({"Type de couche", "Occurrences"}, type_rows) end

    if #info.layers > 0 then
      heading(3, "Détail de toutes les couches")
      local layer_rows = {}
      for index, layer in ipairs(info.layers) do
        layer_rows[#layer_rows + 1] = {
          index, layer.name, layer.type, layer.inputs, layer.output,
          layer.in_channels ~= 0 and layer.in_channels or layer.in_features,
          layer.out_channels ~= 0 and layer.out_channels or layer.out_features,
          layer.kernel_size, layer.stride, layer.padding, layer.params_count,
          layer.is_trainable,
        }
      end
      table_rows({"#", "Nom", "Type", "Entrées", "Sortie", "In", "Out", "Kernel", "Stride", "Padding", "Paramètres", "Entraînable"}, layer_rows)
    end
  else
    emit("Aucune architecture sérialisée trouvée dans le checkpoint.\n\n")
  end

  if #(ctx.warmups or {}) > 0 then
    heading(2, "Warmups actifs")
    local rows = {}
    for _, warmup in ipairs(ctx.warmups) do
      rows[#rows + 1] = {warmup.metric, warmup.key, warmup.steps, "actif"}
    end
    table_rows({"Métrique", "Paramètre", "Fin au step", "État"}, rows)
  end

  local metric_names = report_metric_columns(ctx)
  heading(2, "Métriques")
  local stat_rows = {}
  for _, name in ipairs(metric_names) do
    local stats = col_stats(numeric_values(ctx.df[name]))
    stat_rows[#stat_rows + 1] = {
      report_metric_label(ctx, name), stats.min, stats.max, stats.mean, stats.std, stats.n
    }
  end
  table_rows({"Métrique", "Minimum", "Maximum", "Moyenne", "Écart-type", "N"}, stat_rows)

  heading(2, "Graphes des métriques")
  local warmup_by_metric = {}
  for _, warmup in ipairs(ctx.warmups or {}) do warmup_by_metric[warmup.metric] = warmup.steps end
  local graph_x = ctx.df.opt_step or ctx.df.step
  for _, name in ipairs(metric_names) do
    local display_name = report_metric_label(ctx, name)
    local curve_strategy = report_curve_strategy(ctx, name)
    local graph = markdown
      and svg_graph(ctx.df[name], display_name, graph_x, warmup_by_metric[name], curve_strategy)
      or ascii_graph(ctx.df[name])
    if graph then
      local suffix = warmup_by_metric[name]
        and (" — warmup actif jusqu'au step " .. metric_number(warmup_by_metric[name])) or ""
      heading(3, display_name .. suffix)
      emit(graph .. "\n\n")
    end
  end

  if #ctx.val_pts > 0 then
    heading(2, "Validation")
    local rows = {}
    local val_steps, val_loss_values, val_mse_values = {}, {}, {}
    for _, point in ipairs(ctx.val_pts) do
      rows[#rows + 1] = {point.step, point.val_loss, point.train_loss, point.gap, point.val_mse}
      if type(point.step) == "number" then val_steps[#val_steps + 1] = point.step end
      if type(point.val_loss) == "number" then val_loss_values[#val_loss_values + 1] = point.val_loss end
      if type(point.val_mse) == "number" then val_mse_values[#val_mse_values + 1] = point.val_mse end
    end
    table_rows({"Step", "Val loss", "Train loss", "Écart",
      "Validation " .. (ctx.recon_label or "Reconstruction")}, rows)
    for _, item in ipairs({{"val_loss", val_loss_values}, {"val_mse", val_mse_values}}) do
      local display_name = report_metric_label(ctx, item[1])
      local graph = markdown and svg_graph(item[2], display_name, val_steps, nil, "linear") or ascii_graph(item[2])
      if graph then
        heading(3, "Graphe " .. display_name)
        emit(graph .. "\n\n")
      end
    end
  end
  emit("Généré par scripts/tools/show-graph.lua — Mímir Framework\n")
  return table.concat(out)
end

-- ══════════════════════════════════════════════════════════════
-- ARGUMENT PARSING
-- ══════════════════════════════════════════════════════════════

local function parse_args()
  local raw = rawget(_G, "arg") or {}
  local opts = {
    csv                   = nil,
    csv_dir               = nil,
    model                 = nil,
    algo                  = nil,
    recon_loss            = nil,
    checkpoint_dir        = nil,
    no_interactive        = false,
    out                   = "./graph_report.html",
    out_text              = nil,
    out_text_extra        = {},
    watch                 = false,
    watch_interval        = 2,
    validate_every_steps  = nil,
    validate_items        = nil,
    n_dataset             = nil,
  }
  local pos = {}
  local i = 1
  while i <= #raw do
    local a = raw[i]
    if a == "--csv"            then opts.csv            = raw[i+1]; i = i+1
    elseif a:match("^--csv=")     then opts.csv            = a:sub(7)
    elseif a == "--csv-dir"        then opts.csv_dir        = raw[i+1]; i = i+1
    elseif a:match("^--csv%-dir=") then opts.csv_dir        = a:match("^--csv%-dir=(.+)")
    elseif a == "--model"          then opts.model          = raw[i+1]; i = i+1
    elseif a:match("^--model=")   then opts.model          = a:sub(9)
    elseif a == "--algo"           then opts.algo           = raw[i+1]; i = i+1
    elseif a:match("^--algo=")    then opts.algo           = a:sub(8)
    elseif a == "--optimizer"      then opts.algo           = raw[i+1]; i = i+1
    elseif a:match("^--optimizer=") then opts.algo          = a:sub(13)
    elseif a == "--recon-loss"     then opts.recon_loss     = raw[i+1]; i = i+1
    elseif a:match("^--recon%-loss=") then opts.recon_loss  = a:match("^--recon%-loss=(.+)")
    elseif a == "--checkpoint" then opts.checkpoint_dir = raw[i+1]; i = i+1
    elseif a:match("^%-%-checkpoint=") then
      opts.checkpoint_dir = a:match("^%-%-checkpoint=(.+)")
    elseif a == "--checkpoint-dir" then opts.checkpoint_dir = raw[i+1]; i = i+1
    elseif a:match("^--checkpoint%-dir=") then
      opts.checkpoint_dir = a:match("^--checkpoint%-dir=(.+)")
    elseif a == "--out"            then opts.out            = raw[i+1]; i = i+1
    elseif a:match("^--out=")     then opts.out            = a:sub(7)
    elseif a == "--out-text"       then opts.out_text       = raw[i+1]; i = i+1
    elseif a:match("^--out%-text=") then opts.out_text      = a:match("^--out%-text=(.+)")
    elseif a == "--watch"          then opts.watch          = true
    elseif a == "--watch-interval" then
      opts.watch_interval = tonumber(raw[i+1]) or 2; i = i+1
    elseif a:match("^--watch%-interval=") then
      opts.watch_interval = tonumber(a:match("^--watch%-interval=(.+)")) or 2
    elseif a == "-n" or a == "--no-interactive" then
      opts.no_interactive = true
    elseif a == "--validate-every-steps" then
      opts.validate_every_steps = tonumber(raw[i+1]); i = i+1
    elseif a:match("^--validate%-every%-steps=") then
      opts.validate_every_steps = tonumber(a:match("^--validate%-every%-steps=(.+)"))
    elseif a == "--validate-items" then
      opts.validate_items = tonumber(raw[i+1]); i = i+1
    elseif a:match("^--validate%-items=") then
      opts.validate_items = tonumber(a:match("^--validate%-items=(.+)"))
    elseif a == "--n-dataset" then
      opts.n_dataset = tonumber(raw[i+1]); i = i+1
    elseif a:match("^--n%-dataset=") then
      opts.n_dataset = tonumber(a:match("^--n%-dataset=(.+)"))
    elseif not a:match("^%-") then
      -- zsh/bash développent rapport.{md,txt} avant Lua : le second chemin
      -- arrive comme argument autonome après la valeur consommée par --out-text.
      if opts.out_text and (a:lower():match("%.md$") or a:lower():match("%.txt$")) then
        opts.out_text_extra[#opts.out_text_extra + 1] = a
      else
        pos[#pos+1] = a
      end
    else
      io.stderr:write("❌ Option inconnue : " .. tostring(a) .. "\n")
      os.exit(2)
    end
    i = i+1
  end
  if not opts.csv and pos[1] then opts.csv = pos[1] end
  opts.out_text_paths = {}
  if opts.out_text then
    local prefix, variants, suffix = opts.out_text:match("^(.-){([^{}]+)}(.*)$")
    if prefix then
      for variant in variants:gmatch("[^,]+") do
        variant = trim(variant)
        local path = prefix .. variant .. suffix
        if not path:lower():match("%.md$") and not path:lower():match("%.txt$") then
          io.stderr:write("❌ --out-text accepte uniquement les extensions .md et .txt : " .. path .. "\n")
          os.exit(2)
        end
        opts.out_text_paths[#opts.out_text_paths + 1] = path
      end
    elseif opts.out_text:lower():match("%.md$") or opts.out_text:lower():match("%.txt$") then
      opts.out_text_paths[1] = opts.out_text
    else
      io.stderr:write("❌ --out-text attend un chemin .md/.txt ou un motif comme rapport.{md,txt}\n")
      os.exit(2)
    end
    for _, path in ipairs(opts.out_text_extra) do
      opts.out_text_paths[#opts.out_text_paths + 1] = path
    end
    if #opts.out_text_paths == 0 then
      io.stderr:write("❌ --out-text ne contient aucune sortie\n")
      os.exit(2)
    end
  end
  return opts
end

-- ══════════════════════════════════════════════════════════════
-- SERVEUR SSE LOCAL (mode --watch)
-- ══════════════════════════════════════════════════════════════

-- Trouve un port TCP libre en lisant /proc/net/tcp (Linux).
local function find_free_port(start)
  start = start or 7700
  for p = start, start + 200 do
    local hex = string.format("%04X", p):upper()
    local free = true
    local f = io.open("/proc/net/tcp", "r")
    if f then
      for line in f:lines() do
        if line:find(hex, 1, true) then free = false; break end
      end
      f:close()
    end
    if free then return p end
  end
  return start
end

-- Lance un serveur HTTP+SSE Python en arrière-plan.
-- Sert `serve_dir` statiquement et expose GET /events (SSE)
-- qui envoie "reload" quand le fichier signal change.
-- Retourne le chemin du fichier signal.
local function start_sse_server(serve_dir, port)
  local sig_file = string.format("/tmp/sg_sig_%d", port)
  write_file(sig_file, "init")

  local py_src = string.format(
[[import http.server, time, socketserver

PORT = %d
DIRECTORY = %q
SIGNAL = %q

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=DIRECTORY, **k)
    def do_GET(self):
        if self.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            last = None
            while True:
                try:
                    with open(SIGNAL) as f:
                        v = f.read().strip()
                    if v != last:
                        last = v
                        if last != 'init':
                            self.wfile.write(b'data: update\n\n')
                            self.wfile.flush()
                    time.sleep(0.25)
                except Exception:
                    break
            return
        elif self.path == '/graph_data.json':
            import os as _os
            fp = _os.path.join(DIRECTORY, 'graph_data.json')
            try:
                with open(fp, 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_response(404)
                self.end_headers()
            return
        return super().do_GET()
    def do_POST(self):
        if self.path == '/snapshot':
            import os as _os, shutil, datetime, json as _json
            ts = datetime.datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S')
            src = _os.path.join(DIRECTORY, 'graph_report.html')
            dst = _os.path.join(DIRECTORY, 'graph_report_' + ts + '.html')
            try:
                shutil.copy2(src, dst)
                msg = _json.dumps({'saved': dst}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except Exception as ex:
                err = _json.dumps({'error': str(ex)}).encode()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            return
        self.send_response(405)
        self.end_headers()
    def log_message(self, *a): pass

class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

TS(('127.0.0.1', PORT), H).serve_forever()
]], port, serve_dir, sig_file)

  local py_path = string.format("/tmp/sg_sse_%d.py", port)
  write_file(py_path, py_src)
  os.execute(string.format("python3 %q >/dev/null 2>&1 &", py_path))
  sleep(1)  -- laisser le serveur démarrer
  return sig_file
end

-- ══════════════════════════════════════════════════════════════
-- MAIN
-- ══════════════════════════════════════════════════════════════

local function main()
  local opts = parse_args()

  -- ── Résolution des chemins CSV ──────────────────────────────────────────
  -- Priorité : --csv-dir > --csv > auto-détection
  local csv_paths = nil   -- liste de chemins à charger (et fusionner si > 1)

  if opts.csv_dir then
    -- Mode dossier : cherche tous les *part[0-9].csv dedans
    local parts = find_part_csvs(opts.csv_dir)
    if parts and #parts > 0 then
      csv_paths = parts
      io.stderr:write(string.format("📂 CSV dir : %s (%d part(s) détectées)\n",
        opts.csv_dir, #parts))
    else
      -- Fallback : tous les .csv du dossier
      csv_paths = {}
      for _, name in ipairs(FS.list_dir(opts.csv_dir)) do
        if name:match("%.csv$") then
          csv_paths[#csv_paths + 1] = FS.join(opts.csv_dir, name)
        end
      end
      if #csv_paths == 0 then
        io.stderr:write("❌ Aucun CSV trouvé dans : " .. opts.csv_dir .. "\n")
        os.exit(1)
      end
      table.sort(csv_paths)
    end
  elseif opts.csv then
    -- Un chemin donné : peut être une *part.csv → expansion auto
    csv_paths = expand_part_csvs(opts.csv)
    if #csv_paths > 1 then
      io.stderr:write(string.format("🗂️  Pattern part CSV : %d fichier(s) détectés\n", #csv_paths))
    end
  else
    -- Auto-détection
    for _, c in ipairs({
      repo_root() .. "checkpoints/loss_history.csv",
      repo_root() .. "checkpoint/loss_history.csv",
      "./loss_history.csv",
    }) do
      if file_exists(c) then csv_paths = { c }; break end
    end
    if not csv_paths then
      csv_paths = { repo_root() .. "checkpoints/loss_history.csv" }
    end
  end

  -- ── Auto-détection checkpoint ──────────────────────────────────────────
  local meta = {}
  local ckpt_dir = opts.checkpoint_dir
  if not ckpt_dir then
    ckpt_dir = csv_paths[1]:match("^(.*checkpoint/[^/]+)") or find_latest_run()
  end
  if ckpt_dir then
    meta = load_meta(ckpt_dir)
    if meta.error then
      io.stderr:write("❌ " .. meta.error .. "\n")
      os.exit(1)
    end
  end

  local model        = opts.model or meta.model
  local algo         = opts.algo  or meta.algo
  local recon_loss   = opts.recon_loss or meta.recon_loss
  local recon_loss_details = (not opts.recon_loss) and meta.recon_loss_details or nil

  -- ── Auto-détection des paramètres de calibration (pré-parse du 1er CSV) ────────
  local cal = {
    validate_every_steps = opts.validate_every_steps,
    validate_items       = opts.validate_items,
    n_dataset            = opts.n_dataset,
    validate_holdout      = nil,
    validate_holdout_frac = nil,
    validate_holdout_items = nil,
  }
  do
    -- Le checkpoint est la source autoritative lorsqu'il sérialise ces champs.
    local cfg = meta.model_info and meta.model_info.config or {}
    if cal.validate_every_steps == nil and cfg.validate_every_steps ~= nil then
      cal.validate_every_steps = tonumber(cfg.validate_every_steps)
    end
    if cfg.validate_holdout ~= nil then cal.validate_holdout = cfg.validate_holdout == true end
    if cfg.validate_holdout_frac ~= nil then
      cal.validate_holdout_frac = tonumber(cfg.validate_holdout_frac)
    end
    if cfg.validate_holdout_items ~= nil then
      cal.validate_holdout_items = tonumber(cfg.validate_holdout_items)
      if cal.validate_items == nil then cal.validate_items = cal.validate_holdout_items end
    end

    -- Repli champ par champ sur le CSV si le checkpoint ne fournit pas la valeur.
    local first_csv = csv_paths[1]
    if file_exists(first_csv) then
      local d_pre = parse_csv(first_csv)
      if d_pre then
        if not opts.algo and d_pre.df.opt_type then
          local optimizer_names = {
            [0] = "sgd", [1] = "adam", [2] = "adamw", [3] = "lion",
            [4] = "adafactor", [5] = "radam", [6] = "nadam",
            [7] = "rmsprop", [8] = "lamb",
          }
          for _, value in ipairs(d_pre.df.opt_type) do
            local detected = optimizer_names[tonumber(value)]
            if not detected and type(value) == "string" then
              local normalized = value:lower()
                if normalized == "sgd" or normalized == "adam" or normalized == "adamw"
                  or normalized == "lion" or normalized == "adafactor"
                  or normalized == "radam" or normalized == "nadam"
                  or normalized == "rmsprop" or normalized == "lamb" then
                detected = normalized
              end
            end
            if detected then
              algo = detected
              io.stderr:write("🔍 Auto-détecté optimizer=" .. detected .. " depuis opt_type\n")
              break
            end
          end
        end
        local det = detect_validation_params(d_pre.df, d_pre.n)
        if cal.validate_every_steps == nil and det.validate_every_steps then
          cal.validate_every_steps = det.validate_every_steps
          io.stderr:write(string.format("🔍 Auto-détecté validate_every_steps=%d\n", cal.validate_every_steps))
        end
        if cal.validate_holdout == nil and det.validate_holdout ~= nil then
          cal.validate_holdout = det.validate_holdout
        end
        if cal.validate_holdout_frac == nil and det.validate_holdout_frac then
          cal.validate_holdout_frac = det.validate_holdout_frac
        end
        if cal.validate_holdout_items == nil and det.validate_holdout_items then
          cal.validate_holdout_items = det.validate_holdout_items
        end
        if cal.validate_items == nil and det.validate_items then
          cal.validate_items = det.validate_items
          io.stderr:write(string.format("🔍 Auto-détecté validate_items=%d\n", cal.validate_items))
        end
        if cal.n_dataset == nil and det.n_dataset then
          cal.n_dataset = det.n_dataset
          io.stderr:write(string.format("🔍 Auto-détecté n_dataset=%d\n", cal.n_dataset))
        end
      end
    end

    -- Compatibilité avec les préférences historiques, seulement après checkpoint/CSV.
    local settings = load_ui_settings()
    local sg = type(settings.showgraph) == "table" and settings.showgraph or {}
    if cal.validate_every_steps == nil then cal.validate_every_steps = tonumber(sg.validate_every_steps) end
    if cal.validate_items == nil then cal.validate_items = tonumber(sg.validate_items) end
    if cal.n_dataset == nil then cal.n_dataset = tonumber(sg.n_dataset) end
  end

  -- ── Prompts interactifs ────────────────────────────────────────────────
  if not opts.no_interactive and not opts.watch then
    if not model then
      io.stderr:write("Modèle entraîné (laisser vide si inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); if v and v ~= "" then model = v end
    end
    if not algo then
      io.stderr:write("Algorithme d'optimisation (adamw, adam, sgd, ... ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); if v and v ~= "" then algo = v end
    end
    if not recon_loss then
      io.stderr:write("Recon loss (mse, l1, charbonnier, ... ; vide = inconnue) : ")
      io.stderr:flush()
      local v = io.read("*l"); if v and v ~= "" then recon_loss = v end
    end
    -- Calibration : demander si non trouvé automatiquement
    if not cal.validate_every_steps then
      io.stderr:write("Fréquence de validation (en opt_steps, ex: 50 ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); cal.validate_every_steps = tonumber(v) or nil
    end
    if not cal.validate_items then
      io.stderr:write("Nombre d'items par validation (ex: 6 ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); cal.validate_items = tonumber(v) or nil
    end
    if not cal.n_dataset then
      io.stderr:write("Taille dataset / epoch (items, ex: 4634 ; vide = inconnu) : ")
      io.stderr:flush()
      local v = io.read("*l"); cal.n_dataset = tonumber(v) or nil
    end
    -- Persister dans viz_ui_settings.json (sous-objet "showgraph", autres clés préservées)
    do
      local settings = load_ui_settings()
      local sg = type(settings.showgraph) == "table" and settings.showgraph or {}
      local changed = false
      if cal.validate_every_steps and tostring(sg.validate_every_steps) ~= tostring(cal.validate_every_steps) then
        sg.validate_every_steps = tostring(cal.validate_every_steps); changed = true
      end
      if cal.validate_items and tostring(sg.validate_items) ~= tostring(cal.validate_items) then
        sg.validate_items = tostring(cal.validate_items); changed = true
      end
      if cal.n_dataset and tostring(sg.n_dataset) ~= tostring(cal.n_dataset) then
        sg.n_dataset = tostring(cal.n_dataset); changed = true
      end
      if changed then
        patch_ui_settings({ showgraph = sg })
        io.stderr:write("💾 Paramètres de calibration sauvegardés dans viz_ui_settings.json\n")
      end
    end
  end

  -- ── Vérification existence CSV ─────────────────────────────────────────
  for _, p in ipairs(csv_paths) do
    if not file_exists(p) then
      io.stderr:write("❌ Fichier introuvable: " .. p .. "\n")
      io.stderr:write("   Vérifiez que le modèle a été entraîné.\n")
      os.exit(1)
    end
  end

  -- Upvalues partagées entre generate() et le bloc watch
  local sse_port   = nil   -- port du serveur SSE (nil = pas de SSE)
  local sig_file   = nil   -- fichier signal → déclenche reload côté browser
  local http_url   = nil   -- URL HTTP à ouvrir (remplace file://)
  local html_written = false  -- true après le premier rendu HTML complet

  -- Démarrage du serveur SSE AVANT le premier rendu :
  -- le HTML généré contiendra directement le bon port SSE.
  if opts.watch then
    local html_dir = opts.out:match("^(.*[/\\])") or "."
    html_dir = html_dir:gsub("[/\\]+$", "")
    if html_dir == "" then html_dir = "." end
    local port = find_free_port(7700)
    sig_file = start_sse_server(html_dir, port)
    sse_port = port
    local html_name = opts.out:match("[^/\\]+$") or "graph_report.html"
    http_url = string.format("http://127.0.0.1:%d/%s", port, html_name)
    io.stderr:write(string.format("🔌 Serveur SSE : http://127.0.0.1:%d\n", port))
  end

  -- ── Fonction de (re)génération du rapport ─────────────────────────────
  local function generate()
    -- Charger + fusionner
    local parts_data = {}
    for _, p in ipairs(csv_paths) do
      local d, err = parse_csv(p)
      if not d then
        io.stderr:write("❌ Erreur CSV (" .. p .. "): " .. (err or "?") .. "\n")
        return false
      end
      parts_data[#parts_data+1] = d
    end
    local data = merge_csvs(parts_data)
    local df = data.df

    -- Points de validation (val_loss sparse) pour le graphe d'écart
    -- (calculés sur le df complet avant filtrage, afin d'avoir le train_loss le plus proche)
    local val_pts = extract_val_points(df, data.n)
    -- Retirer les étapes de validation des métriques d'entraînement
    local train_df, train_n, _val_excl = filter_train_rows(df, data.headers, data.n)

    if data.parts and data.parts > 1 then
      io.stderr:write(string.format("🗂️  Fusion de %d parties → %d steps total\n",
        data.parts, data.n))
    end
    local val_skipped = data.n - train_n
    io.stderr:write(string.format("📊 %d steps d'entraînement%s | colonnes: %s\n", train_n,
      val_skipped > 0 and string.format(" (%d lignes val exclues)", val_skipped) or "",
      table.concat(data.headers, ", ")))

    -- Frontières d'epoch dans les données d'entraînement (opt_step comme abscisse réelle)
    local epoch_boundaries = {}  -- { { step=N, epoch=E }, ... }
    local ep_x_col = (train_df.opt_step and #train_df.opt_step > 0) and "opt_step" or "step"
    if train_df.epoch and train_df[ep_x_col] then
      local prev_e = nil
      for i = 1, train_n do
        local e = train_df.epoch[i]
        local s = train_df[ep_x_col][i]
        if type(e) == "number" and e ~= prev_e then
          if prev_e ~= nil then
            epoch_boundaries[#epoch_boundaries+1] = { step = s, epoch = e }
          end
          prev_e = e
        end
      end
    end

    -- Colonne de reconstruction résolue depuis la recon-loss déclarée dans les
    -- métadonnées. Sans métadonnée, seul un en-tête explicite est interprété.
    local recon_loss_norm = (recon_loss or ""):lower()
    local recon_aliases = {
      mse={"mse", "l2"}, l2={"l2", "mse"},
      mae={"mae", "l1"}, l1={"l1", "mae"},
      huber={"huber", "smooth_l1", "smoothl1"},
      smooth_l1={"smooth_l1", "smoothl1", "huber"},
      smoothl1={"smoothl1", "smooth_l1", "huber"},
      charbonnier={"charbonnier"},
      gaussian_nll={"gaussian_nll", "nll_gaussian", "gaussian-nll"},
      nll_gaussian={"nll_gaussian", "gaussian_nll", "gaussian-nll"},
      bce={"bce"},
    }
    local recon_candidates = recon_aliases[recon_loss_norm] or {}
    if recon_loss_norm ~= "" then table.insert(recon_candidates, 1, recon_loss_norm) end
    recon_candidates[#recon_candidates + 1] = "recon_loss"
    recon_candidates[#recon_candidates + 1] = "reconstruction_loss"
    local recon_col = nil
    for _, candidate in ipairs(recon_candidates) do
      if train_df[candidate] and #train_df[candidate] > 0 then
        recon_col = candidate
        break
      end
    end
    local recon_label = recon_loss_norm ~= "" and ("Reconstruction (" .. recon_loss_norm .. ")")
      or (recon_col and recon_col or "Reconstruction")

    local structural_columns = {
      step=true, epoch=true, total_epochs=true, batch=true, total_batches=true,
      params=true, opt_type=true, opt_step=true, val_step=true, val_feedback=true,
      val_rewarded=true, val_penalized=true,
    }
    local metric_columns = {}
    for _, name in ipairs(data.headers) do
      if not structural_columns[name] then
        for _, value in pairs(train_df[name] or {}) do
          if type(value) == "number" then
            metric_columns[#metric_columns + 1] = name
            break
          end
        end
      end
    end

    -- Liste des epochs
    local ep_set, ep_list = {}, {}
    if train_df.epoch then
      for _, e in ipairs(train_df.epoch) do
        if type(e) == "number" and not ep_set[e] then
          ep_set[e] = true; ep_list[#ep_list+1] = e
        end
      end
      table.sort(ep_list)
    end

    local step_min = (train_df.step and #train_df.step > 0) and train_df.step[1]        or 0
    local step_max = (train_df.step and #train_df.step > 0) and train_df.step[#train_df.step] or 0
    io.stderr:write(string.format("   Epochs: %d | Steps: %d → %d\n",
      #ep_list, step_min, step_max))

    -- Répertoire de sortie (partagé par HTML et graph_data.json)
    local json_dir = opts.out:match("^(.*[/\\])") or ""
    if json_dir == "" then json_dir = "./" end
    local json_path = json_dir .. "graph_data.json"

    -- ── Écriture graph_data.json (mise à jour DOM partielle) ──────────────
    local function flush_json()
      local jdata = gen_data_json(train_df, data.headers, train_n, val_pts)
      local jtmp = json_path .. ".tmp"
      if write_file(jtmp, jdata) then os.rename(jtmp, json_path) end
    end

    local total_bytes = 0
    for _, p in ipairs(csv_paths) do total_bytes = total_bytes + file_size(p) end
    local report_ctx = {
      csv_path       = #csv_paths == 1 and csv_paths[1]
                       or (csv_paths[1]:match("^(.*[/\\])") or "./") .. "["
                       .. #csv_paths .. " parts]",
      csv_size       = total_bytes,
      model          = model,
      algo           = algo,
      recon_loss     = recon_loss,
      recon_loss_details = recon_loss_details,
      meta           = meta,
      warmups        = detect_metric_warmups(
                         meta.model_info and meta.model_info.config or {}, data.headers),
      df             = train_df,
      headers        = data.headers,
      metric_columns = metric_columns,
      n              = train_n,
      epoch_boundaries = epoch_boundaries,
      validate_every_steps = cal.validate_every_steps,
      validate_items = cal.validate_items,
      validate_holdout = cal.validate_holdout,
      validate_holdout_frac = cal.validate_holdout_frac,
      validate_holdout_items = cal.validate_holdout_items,
      n_dataset      = cal.n_dataset,
      epochs         = ep_list,
      recon_col      = recon_col,
      recon_label    = recon_label,
      val_pts        = val_pts,
      sse_port       = sse_port,
      watch_interval = (not sse_port and opts.watch) and opts.watch_interval or nil,
    }

    local function flush_text_reports()
      for _, path in ipairs(opts.out_text_paths or {}) do
        local markdown = path:lower():match("%.md$") ~= nil
        local content = gen_text_report(report_ctx, markdown)
        local tmp = path .. ".tmp"
        if not write_file(tmp, content) or not os.rename(tmp, path) then
          io.stderr:write("❌ Impossible d'écrire : " .. path .. "\n")
          return false
        end
        io.stderr:write(string.format("💾 %s (%s)\n", path, human_bytes(#content)))
      end
      return true
    end

    -- ── En mode watch après le premier rendu : JSON seul ──────────────────
    if html_written then
      flush_json()
      if not flush_text_reports() then return false end
      -- Notifier le navigateur → _sg_fetch_update() met à jour les charts
      if sig_file then write_file(sig_file, tostring(os.time())) end
      io.stderr:write(string.format("🔄 %s mis à jour\n", json_path))
      return true
    end

    -- ── Premier rendu (ou mode one-shot) : HTML complet ───────────────────
    io.stderr:write("🖌️  Génération du rapport HTML...\n")

    local html = gen_html(report_ctx)

    local tmp = opts.out .. ".tmp"
    if write_file(tmp, html) then
      os.rename(tmp, opts.out)
      html_written = true
      flush_json()
      if not flush_text_reports() then return false end
      io.stderr:write(string.format("💾 %s (%s)\n", opts.out, human_bytes(#html)))
      return true
    else
      io.stderr:write("❌ Impossible d'écrire : " .. opts.out .. "\n")
      return false
    end
  end

  -- ── Banner ─────────────────────────────────────────────────────────────
  local SEP = string.rep("═", 59)
  io.stderr:write(SEP .. "\n")
  io.stderr:write("        SHOW GRAPH — Mímir training metrics\n")
  io.stderr:write(SEP .. "\n")
  for _, p in ipairs(csv_paths) do
    io.stderr:write(string.format("📄 CSV     : %s (%s)\n", p, human_bytes(file_size(p))))
  end
  io.stderr:write(string.format("🧠 Modèle  : %s\n", model or "(inconnu)"))
  if algo then
    io.stderr:write("⚙️  Algo    : " .. algo .. "\n")
  end
  if recon_loss then
    local loss = recon_loss .. (recon_loss_details and (" [" .. recon_loss_details .. "]") or "")
    io.stderr:write("📐 Recon   : " .. loss .. "\n")
  end
  if meta.source then
    io.stderr:write("📦 Source  : " .. meta.source .. "\n")
  end
  if cal.validate_every_steps or cal.validate_holdout ~= nil then
    io.stderr:write(string.format("✅ Val     : every=%s | holdout=%s | frac=%s | items=%s\n",
      tostring(cal.validate_every_steps or "?"),
      cal.validate_holdout == nil and "?" or tostring(cal.validate_holdout),
      tostring(cal.validate_holdout_frac or "?"),
      tostring(cal.validate_holdout_items or cal.validate_items or "?")))
  end
  local detected_warmups = detect_metric_warmups(
    meta.model_info and meta.model_info.config or {}, nil)
  for _, warmup in ipairs(detected_warmups) do
    io.stderr:write(string.format("🔥 Warmup  : %s → step %s (%s)\n",
      warmup.metric, metric_number(warmup.steps), warmup.key))
  end
  if opts.watch then
    io.stderr:write(string.format("👁️  Watch   : intervalle %ds\n", opts.watch_interval))
  end
  io.stderr:write(SEP .. "\n\n")

  -- ── Premier rendu ──────────────────────────────────────────────────────
  io.stderr:write("⏳ Chargement du CSV...\n")
  local ok = generate()
  if not ok then os.exit(1) end

  if opts.watch then
    -- Ouvre le navigateur (http:// via serveur SSE, ou file:// en fallback)
    open_browser(http_url or opts.out)
    io.stderr:write("👁️  Surveillance SSE active — Ctrl+C pour arrêter\n")
    if http_url then
      io.stderr:write(string.format("   URL : %s\n", http_url))
    end

    local last_mtime = mtimes_combined(csv_paths)
    while true do
      sleep(opts.watch_interval)
      local cur = mtimes_combined(csv_paths)
      if cur ~= last_mtime then
        last_mtime = cur
        io.stderr:write("🔄 Changement détecté, régénération...\n")
        generate()  -- met à jour HTML + sig_file → SSE → browser
      end
    end
  else
    io.stderr:write(string.format("🌐 Ouvrir : xdg-open %s\n", opts.out))
    io.stderr:write("\n✓ Terminé.\n")
  end
end

main()
