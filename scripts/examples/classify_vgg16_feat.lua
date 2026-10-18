---@diagnostic disable: undefined-global, undefined-field, inject-field

-- Exemple: classification simple à partir d'un checkpoint.
--
-- Ce script supporte 2 formats de `--labels-file`:
--
--  A) Labels (1 par ligne):
--     - --labels-file <txt>  (chaque ligne = un label, dans l'ordre des sorties)
--     - --image <path>
--     -> utilise un modèle `vgg16` (logits multi-label, probas via sigmoid)
--
--  B) Prototypes (label + images de référence):
--     - --labels-file <txt>  (chaque ligne: label<TAB>image_path ; label répété = plusieurs refs)
--     - --image <path>
--     -> utilise `vgg16_feat` et classe via centroïdes + cosine/L2
--
-- Auto-détection:
-- - on infère l'architecture du checkpoint (vgg16 vs vgg16_feat) via
--   `manifest.json` (on évite volontairement `model/architecture.json` pour ne pas
--   lire de vocab/tag interne au checkpoint).
-- - en mode (A), `num_classes` vient du fichier de labels, et on essaie quelques
--   configurations candidates (image/base/fc) en strict jusqu'à validation.
--
-- Similarité (mode B): --metric cosine (défaut) ou --metric l2
--
-- Hypothèses dataset:
-- - items avec image + texte
-- - le texte contient un label (par défaut: premier tag avant '.')
--
-- Usage (exemple):
--   ./run_mimir.sh --lua scripts/examples/classify_vgg16_feat.lua -- \
--     --dataset-root ./dataset_2 \
--     --ckpt-dir ./checkpoint/vgg16_feat_pretrain \
--     --image-w 256 --image-h 256 \
--     --max-items 500 \
--     --ref-per-class 3 \
--     --metric cosine \
--     --topk 5

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")

local function opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

local function opt_int(k, d)
  return math.floor(opt_num(k, d))
end

local function opt_str(k, d)
  local v = opts[k]
  if v == nil or v == true then return d end
  return tostring(v)
end

local function opt_bool(k, d)
  local v = opts[k]
  if v == nil then return d end
  if v == true or v == false then return v end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local function assert_ok(ok, err, msg)
  if ok == false then
    error((msg or "Operation failed") .. ": " .. tostring(err))
  end
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  assert_ok(ok, dt_or_err, "Mimir.model.dtype(" .. tostring(dtype) .. ") failed")
  return true
end

local function file_read_all(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  return s
end

local function normalize_u8_to_f32_minus1_1(u8)
  local n = #u8
  local x = {}
  x[n] = 0.0
  for i = 1, n do
    -- u8 in [0,255] => [-1,1]
    x[i] = (tonumber(u8[i]) or 0) / 127.5 - 1.0
  end
  return x
end

local function split_first_tag(text)
  if text == nil then return nil end
  local s = tostring(text)
  s = s:gsub("\r", "\n")
  -- prend la 1ère ligne non vide
  for line in s:gmatch("[^\n]+") do
    local t = line:match("^%s*(.-)%s*$")
    if t and #t > 0 then
      -- prend avant '.'
      local a = t:match("^([^%.]+)")
      a = a and a:match("^%s*(.-)%s*$") or nil
      if a and #a > 0 then return a end
      return t
    end
  end
  return nil
end

local function dot(a, b)
  local s = 0.0
  local n = math.min(#a, #b)
  for i = 1, n do
    s = s + a[i] * b[i]
  end
  return s
end

local function l2_norm(a)
  local s = 0.0
  for i = 1, #a do
    local v = a[i]
    s = s + v * v
  end
  return math.sqrt(s)
end

local function cosine_sim(a, b)
  local na = l2_norm(a)
  local nb = l2_norm(b)
  if na <= 0 or nb <= 0 then return 0.0 end
  return dot(a, b) / (na * nb)
end

local function l2_dist2(a, b)
  local s = 0.0
  local n = math.min(#a, #b)
  for i = 1, n do
    local d = a[i] - b[i]
    s = s + d * d
  end
  return s
end

local function emb_stats(e)
  local n = #e
  local norm = l2_norm(e)
  local maxabs = 0.0
  for i = 1, n do
    local v = math.abs(e[i])
    if v > maxabs then maxabs = v end
  end
  return n, norm, maxabs
end

local function topk(scores, k, higher_is_better)
  local idx = {}
  for i = 1, #scores do idx[i] = i end
  table.sort(idx, function(i, j)
    if higher_is_better then
      return scores[i] > scores[j]
    end
    return scores[i] < scores[j]
  end)
  local out = {}
  local m = math.min(k, #idx)
  for i = 1, m do out[i] = idx[i] end
  return out
end

local function sigmoid(z)
  z = tonumber(z) or 0.0
  if z >= 0.0 then
    local ez = math.exp(-z)
    return 1.0 / (1.0 + ez)
  end
  local ez = math.exp(z)
  return ez / (1.0 + ez)
end

-- -------------------------
-- Params
-- -------------------------
local dataset_root = opt_str("dataset-root", opt_str("dataset", "./dataset_2"))
local ckpt_root = opt_str("ckpt-dir", opt_str("ckpt", "./checkpoint/vgg16_feat_pretrain"))

local labels_file = opt_str("labels-file", "")
local query_image = opt_str("image", "")

local image_w = opt_int("image-w", 256)
local image_h = opt_int("image-h", 256)
local image_c = opt_int("image-c", 3)

local base_channels = opt_int("base-channels", 4)
if base_channels < 4 then base_channels = 4 end
local fc_hidden_cli = opt_int("fc-hidden", opt_int("fc_hidden", 0))
if fc_hidden_cli < 0 then fc_hidden_cli = 0 end

local function has_opt(k)
  return opts[k] ~= nil
end

local function has_any_opt(keys)
  for i = 1, #keys do
    if has_opt(keys[i]) then return true end
  end
  return false
end

local max_items = opt_int("max-items", 0)
local min_modalities = opt_int("min-modalities", 2) -- 2: image + text

local ref_per_class = opt_int("ref-per-class", opt_int("refs-per-class", 3))
local metric = opt_str("metric", "cosine"):lower() -- cosine | l2
local top_k = opt_int("topk", opt_int("top-k", 8))
local seed = opt_int("seed", 1337)

local higher_is_better = (metric ~= "l2" and metric ~= "l2sq")

local dry_run = opt_bool("dry-run", false)

local cpu_only = opt_bool("cpu-only", false)

-- perf/compat
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  local hw = opt_bool("hw", true)
  if cpu_only then hw = false end
  pcall(Mimir.Model.set_hardware, hw)
end

local is_file_mode = (labels_file ~= "" and query_image ~= "")

log("=== classify (vgg16 / vgg16_feat) ===")
log(string.format("- ckpt_root=%s", ckpt_root))
if labels_file ~= "" then log(string.format("- labels_file=%s", labels_file)) end
if query_image ~= "" then log(string.format("- image=%s", query_image)) end
log(string.format("- image=%dx%dx%d", image_w, image_h, image_c))

if not is_file_mode then
  log(string.format("- dataset_root=%s", dataset_root))
  log(string.format("- max_items=%d min_modalities=%d", max_items, min_modalities))
  log(string.format("- ref_per_class=%d metric=%s topk=%d", ref_per_class, metric, top_k))
else
  log(string.format("- topk=%d", top_k))
end

if cpu_only then
  log("Note: cpu-only=true coupe l'accélération des ops. Pour désactiver totalement Vulkan: export MIMIR_DISABLE_VULKAN=1")
end

local function resolve_ckpt_dir(root)
  if Ckpt then
    local resolver = Ckpt.resolve_dir_prefer_final or Ckpt.resolve_dir
    if resolver then
      local r = resolver(root)
      if r and tostring(r) ~= "" then return r end
    end
  end
  return root
end

local function infer_ckpt_arch(ckpt_dir)
  -- IMPORTANT: ne pas lire model/architecture.json ici (il peut contenir tags_vocab).
  -- On se limite à manifest.json (noms de tenseurs/layers) pour éviter de lire le vocab.
  local manifest = file_read_all(tostring(ckpt_dir) .. "/manifest.json")
  if manifest then
    local head = manifest:sub(1, 8192)
    if head:find('"name"%s*:%s*"vgg16_feat/', 1) then return "vgg16_feat" end
    if head:find('"name"%s*:%s*"vgg16/', 1) then return "vgg16" end
  end
  return nil
end

local function read_ckpt_model_config(ckpt_dir)
  local out = {}
  local arch_path = tostring(ckpt_dir) .. "/model/architecture.json"
  local s = file_read_all(arch_path)
  if not s then return out end
  -- Focus on the model_config object to avoid matching layer fields.
  local mc_start = s:find('"model_config"', 1, true)
  if not mc_start then return out end
  local mc = s:sub(mc_start)
  local function get_num(key)
    local v = mc:match('"' .. key .. '"%s*:%s*([%-%d%.]+)')
    return tonumber(v)
  end
  out.image_w = get_num("image_w")
  out.image_h = get_num("image_h")
  out.image_c = get_num("image_c")
  out.base_channels = get_num("base_channels")
  out.fc_hidden = get_num("fc_hidden")
  out.num_classes = get_num("num_classes")
  return out
end

local function create_and_load_model(model_type, cfg, ckpt_root0, strict_mode)
  assert_ok(Mimir.Model.create(model_type, cfg), nil, "Model.create(" .. tostring(model_type) .. ") failed")
  apply_dtype(cfg)
  assert_ok(Mimir.Model.allocate_params(), nil, "Model.allocate_params failed")

  local ckpt_dir0 = resolve_ckpt_dir(ckpt_root0)
  local load_opts = { load_optimizer = false, strict_mode = (strict_mode == true), validate_checksums = true }
  local ok_load, err_load = Mimir.Serialization.load(ckpt_dir0, "raw_folder", load_opts)
  if ok_load ~= true then
    local alt = ckpt_root0 .. "/final"
    ok_load, err_load = Mimir.Serialization.load(alt, "raw_folder", load_opts)
    assert_ok(ok_load, err_load, "Serialization.load failed")
    ckpt_dir0 = alt
  end
  log("✓ Checkpoint chargé: " .. tostring(ckpt_dir0))
  return ckpt_dir0
end

local function try_load_vgg16_with_candidates(ckpt_root0, labels_count)
  local cfg0, err = Mimir.Architectures.default_config("vgg16")
  assert(type(cfg0) == "table", "default_config(vgg16) failed: " .. tostring(err))

  local function clone(t)
    local o = {}
    for k, v in pairs(t) do o[k] = v end
    return o
  end

  -- Tailles d'images: VGG16 + GAP supporte en général plusieurs tailles.
  -- Pour une meilleure cohérence, on essaie d'abord la taille demandée (par défaut 256).
  local sizes = {}
  sizes[#sizes + 1] = { image_w, image_h }
  if not has_any_opt({"image-w", "image-h"}) then
    sizes[#sizes + 1] = { 256, 256 }
    sizes[#sizes + 1] = { 128, 128 }
    sizes[#sizes + 1] = { 64, 64 }
  end

  local bases = {}
  if has_any_opt({"base-channels", "base_channels"}) then
    bases[1] = base_channels
  else
    bases[1] = 64
    bases[2] = 32
    bases[3] = 16
  end

  local fcs = {}
  if fc_hidden_cli and fc_hidden_cli > 0 then
    fcs[1] = fc_hidden_cli
  else
    fcs[1] = 512
    fcs[2] = 256
  end

  local last_err = nil
  for si = 1, #sizes do
    for bi = 1, #bases do
      for fi = 1, #fcs do
        local w, h = sizes[si][1], sizes[si][2]
        local base = bases[bi]
        local fc = fcs[fi]

        local cfg = clone(cfg0)
        cfg.image_w = w
        cfg.image_h = h
        cfg.image_c = image_c
        cfg.base_channels = base
        cfg.fc_hidden = fc
        cfg.num_classes = labels_count

        log(string.format("Essai load vgg16: image=%dx%dx%d base=%d fc=%d classes=%d", w, h, image_c, base, fc, labels_count))
        local ok, err_load = pcall(create_and_load_model, "vgg16", cfg, ckpt_root0, true)
        if ok then
          image_w, image_h = w, h
          base_channels = base
          return cfg
        end
        last_err = err_load
      end
    end
  end

  error("Impossible de charger le checkpoint vgg16 en strict (sans lire la config/vocab). Dernière erreur: " .. tostring(last_err))
end

if dry_run then
  log("dry-run=true: fin (pas d'inférence).")
  return
end

local function path_dirname(p)
  local s = tostring(p)
  local d = s:match("^(.*)/[^/]+$")
  return d or "."
end

local function path_is_abs(p)
  return tostring(p):sub(1, 1) == "/"
end

local function path_join(a, b)
  if a == "" or a == "." then return b end
  if a:sub(-1) == "/" then return a .. b end
  return a .. "/" .. b
end

local function read_image_rgb_u8(path)
  if not (Mimir and Mimir.IO and Mimir.IO.read_image_rgb_u8) then
    error("Mimir.IO.read_image_rgb_u8 indisponible: rebuild requis")
  end
  local t, e = Mimir.IO.read_image_rgb_u8(path, image_w, image_h)
  if t == nil then
    return nil, e
  end
  return t
end

local function trim_punct(s)
  s = tostring(s or "")
  s = s:gsub("\r", "")
  s = s:match("^%s*(.-)%s*$") or ""
  -- retire guillemets/ponctuation autour
  local junk = "[%s\"'()%[%]{}<>:,;%-]+"
  while true do
    local t = s:gsub("^" .. junk, "")
    t = t:gsub(junk .. "$", "")
    t = t:match("^%s*(.-)%s*$") or ""
    if t == s then break end
    s = t
  end
  return s
end

local function parse_labels_file(p)
  local refs = {}
  local labels = {}
  local base = path_dirname(p)
  local n_lines = 0
  local n_refs = 0
  local n_labels = 0
  for line in io.lines(p) do
    n_lines = n_lines + 1
    line = tostring(line):gsub("\r", "")
    line = line:match("^%s*(.-)%s*$")
    if line ~= "" and not line:match("^#") then
      -- refs: déclenché uniquement via séparateurs explicites (TAB, ',' ou ';')
      local a, b = line:match("^([^\t]+)\t+(.+)$")
      if not (a and b) then
        local aa, bb = line:match("^([^,;]+)[,;]%s*(.+)$")
        if aa and bb then
          -- N'accepte ','/';' comme séparateur que si la 2e colonne ressemble à un path d'image.
          local cand = tostring(bb):match("^%s*(.-)%s*$")
          local is_img = false
          if cand then
            local lc = cand:lower()
            if lc:match("%.png$") or lc:match("%.jpg$") or lc:match("%.jpeg$") or lc:match("%.webp$") or lc:match("%.bmp$") then
              is_img = true
            elseif cand:find("/", 1, true) then
              is_img = true
            end
          end
          if is_img then
            a, b = aa, bb
          end
        end
      end
      if a and b then
        local label = trim_punct(a)
        local img = b:match("^%s*(.-)%s*$")
        if label ~= "" and img ~= "" then
          if not path_is_abs(img) then
            img = path_join(base, img)
          end
          if refs[label] == nil then refs[label] = {} end
          refs[label][#refs[label] + 1] = img
          n_refs = n_refs + 1
        end
      else
        local t = trim_punct(line)
        if t ~= "" then
          labels[#labels + 1] = t
          n_labels = n_labels + 1
        end
      end
    end
  end
  if n_refs > 0 then
    return { mode = "refs", refs = refs, n_lines = n_lines, n_refs = n_refs }
  end
  return { mode = "labels", labels = labels, n_lines = n_lines, n_labels = n_labels }
end

local function image_u8_hwc_to_chw_f32_minus1_1(u8, w, h, c)
  local n = w * h * c
  if #u8 ~= n then
    return nil, string.format("image size mismatch: got=%d expected=%d", #u8, n)
  end
  local x = {}
  x[n] = 0.0
  for cc = 0, c - 1 do
    for yy = 0, h - 1 do
      for xx = 0, w - 1 do
        local pix = (yy * w + xx) * c + cc
        local idx = cc * (w * h) + (yy * w + xx)
        x[idx + 1] = (tonumber(u8[pix + 1]) or 0) / 127.5 - 1.0
      end
    end
  end
  return x
end

local function embed_image_path(p)
  local im, e = read_image_rgb_u8(p)
  if im == nil then
    return nil, e
  end
  local x = normalize_u8_to_f32_minus1_1(im.image)
  -- NOTE: certains graphes avec branches (résiduels) ont besoin de training=true
  -- pour exécuter toutes les branches correctement.
  local emb, fe = Mimir.Model.forward(x, true)
  if emb == nil then
    return nil, fe
  end
  if type(emb) ~= "table" or #emb <= 0 then
    return nil, "embedding vide (vgg16_feat): vérifie image_w/image_h/base_channels/checkpoint"
  end
  return emb
end

local function forward_vgg16_logits(image_path)
  local im, e = read_image_rgb_u8(image_path)
  if im == nil then
    return nil, e
  end
  local x, xe = image_u8_hwc_to_chw_f32_minus1_1(im.image, image_w, image_h, image_c)
  if x == nil then
    return nil, xe
  end
  local logits, fe = Mimir.Model.forward(x, true)
  if logits == nil then
    return nil, fe
  end
  if type(logits) ~= "table" or #logits <= 0 then
    return nil, "logits vides (vgg16): vérifie image_w/image_h/num_classes/checkpoint"
  end
  return logits
end

local function classify_query_with_centroids(centroids, classes, emb)
  local scores = {}
  for ci = 1, #classes do
    local c = classes[ci]
    local proto = centroids[c]
    if not proto then
      scores[ci] = higher_is_better and -1e30 or 1e30
    else
      if metric == "l2" or metric == "l2sq" then
        scores[ci] = l2_dist2(emb, proto)
      else
        scores[ci] = cosine_sim(emb, proto)
      end
    end
  end
  local rank = topk(scores, math.max(1, top_k), higher_is_better)
  return rank, scores
end

-- =========================
-- Mode fichier: labels-file + image
-- =========================
if labels_file ~= "" or query_image ~= "" then
  if labels_file == "" or query_image == "" then
    error("Mode fichier: nécessite --labels-file et --image")
  end

  -- Détection best-effort via manifest.json uniquement.
  local ckpt_dir0 = resolve_ckpt_dir(ckpt_root)
  local ckpt_arch = infer_ckpt_arch(ckpt_dir0)
  if not ckpt_arch then
    ckpt_dir0 = ckpt_root .. "/final"
    ckpt_arch = infer_ckpt_arch(ckpt_dir0)
  end

  local parsed = parse_labels_file(labels_file)

  -- ---------
  -- Mode A: labels (vgg16)
  -- ---------
  if parsed.mode == "labels" then
    local labels = parsed.labels or {}
    if #labels <= 0 then
      error("labels-file: vide (attendu: 1 label par ligne)")
    end

    if ckpt_arch and ckpt_arch ~= "vgg16" then
      error("Checkpoint incompatible: labels-file=1-colonne => vgg16 requis, mais checkpoint semble être: " .. tostring(ckpt_arch))
    end

    local cfg = try_load_vgg16_with_candidates(ckpt_root, #labels)
    cfg.tags_vocab = labels

    -- Force top-8 (demande utilisateur), sauf si l'utilisateur a explicitement fourni topk.
    if not (has_opt("topk") or has_opt("top-k")) then
      top_k = 8
    end

    log(string.format("Mode labels (vgg16): image=%dx%dx%d num_classes=%d", image_w, image_h, image_c, #labels))
    if cfg.base_channels then log("- base_channels=" .. tostring(cfg.base_channels)) end
    if cfg.fc_hidden then log("- fc_hidden=" .. tostring(cfg.fc_hidden)) end

    local logits, le = forward_vgg16_logits(query_image)
    if logits == nil then
      error("Impossible de faire forward vgg16: " .. tostring(le))
    end

    local probs = {}
    probs[#logits] = 0.0
    for i = 1, #logits do probs[i] = sigmoid(logits[i]) end
    local rank = topk(probs, math.max(1, top_k), true)

    log("=== Résultat (top labels) ===")
    for t = 1, math.min(#rank, top_k) do
      local i = rank[t]
      log(string.format("%02d. %s\t%.6f", t, tostring(labels[i]), probs[i]))
    end
    return
  end

  -- ---------
  -- Mode B: prototypes (vgg16_feat)
  -- ---------
  local refs = parsed.refs or {}
  local classes = {}
  for c, _ in pairs(refs) do classes[#classes + 1] = c end
  table.sort(classes)
  if #classes <= 0 then
    error("labels-file: aucune classe trouvée (format attendu: label<TAB>image_path)")
  end

  if ckpt_arch and ckpt_arch ~= "vgg16_feat" then
    error("Checkpoint incompatible: labels-file=2-colonnes => vgg16_feat requis, mais checkpoint semble être: " .. tostring(ckpt_arch))
  end

  if not has_opt("image-w") and ckpt_mc.image_w then image_w = math.floor(ckpt_mc.image_w) end
  if not has_opt("image-h") and ckpt_mc.image_h then image_h = math.floor(ckpt_mc.image_h) end
  if not has_opt("image-c") and ckpt_mc.image_c then image_c = math.floor(ckpt_mc.image_c) end
  if not has_opt("base-channels") and ckpt_mc.base_channels then base_channels = math.floor(ckpt_mc.base_channels) end

  local cfg0, err = Mimir.Architectures.default_config("vgg16_feat")
  assert(type(cfg0) == "table", "default_config(vgg16_feat) failed: " .. tostring(err))
  local cfg = cfg0
  cfg.image_w = image_w
  cfg.image_h = image_h
  cfg.image_c = image_c
  cfg.base_channels = base_channels
  log(string.format("Mode prototypes (vgg16_feat): image=%dx%dx%d base_channels=%d (embed_dim=%d)", image_w, image_h, image_c, base_channels, 23 * base_channels))

  create_and_load_model("vgg16_feat", cfg, ckpt_root, true)

  local centroids = {}
  local refs_total = 0
  for _, c in ipairs(classes) do
    local paths = refs[c] or {}
    local used = 0
    local mu = nil
    for i = 1, #paths do
      if used >= ref_per_class then break end
      local emb, e = embed_image_path(paths[i])
      if emb ~= nil then
        if refs_total == 0 then
          local n, nm, mx = emb_stats(emb)
          log(string.format("emb(ref) dim=%d norm=%.4g maxabs=%.4g", n, nm, mx))
        end
        used = used + 1
        refs_total = refs_total + 1
        if mu == nil then
          mu = {}
          mu[#emb] = 0.0
          for d = 1, #emb do mu[d] = 0.0 end
        end
        for d = 1, #emb do
          mu[d] = mu[d] + emb[d]
        end
      else
        log(string.format("⚠ ref ignorée (%s): %s", tostring(c), tostring(e)))
      end
    end

    if mu ~= nil and used > 0 then
      local inv = 1.0 / used
      for d = 1, #mu do
        mu[d] = mu[d] * inv
      end
      centroids[c] = mu
    else
      log(string.format("⚠ aucune ref valide pour la classe '%s'", tostring(c)))
    end
  end

  local q_emb, q_err = embed_image_path(query_image)
  if q_emb == nil then
    error("Impossible de charger/classer l'image: " .. tostring(q_err))
  end

  do
    local n, nm, mx = emb_stats(q_emb)
    log(string.format("emb(query) dim=%d norm=%.4g maxabs=%.4g", n, nm, mx))
    if nm <= 0 then
      log("⚠ embedding norme nulle: les scores cosine seront peu informatifs")
    end
  end

  local rank, scores = classify_query_with_centroids(centroids, classes, q_emb)
  local pred = classes[rank[1]]

  log("=== Résultat ===")
  log(string.format("- predicted=%s", tostring(pred)))
  local out = {}
  for t = 1, math.min(#rank, top_k) do
    local ci = rank[t]
    local c = classes[ci]
    out[#out + 1] = string.format("%s(%.4g)", c, scores[ci])
  end
  log(string.format("- top%d=%s", top_k, table.concat(out, ", ")))
  log(string.format("- classes=%d refs_total=%d", #classes, refs_total))
  return
end

error("Mode dataset désactivé: ce script ne doit pas lire/analyser de dataset ni de vocab. Utilise --labels-file + --image.")

-- =========================
-- Mode dataset (fallback): nécessite un vrai checkpoint vgg16_feat
-- =========================

do
  local ckpt_dir0 = resolve_ckpt_dir(ckpt_root)
  local ckpt_arch = infer_ckpt_arch(ckpt_dir0)
  if not ckpt_arch then
    ckpt_dir0 = ckpt_root .. "/final"
    ckpt_arch = infer_ckpt_arch(ckpt_dir0)
  end
  if ckpt_arch and ckpt_arch ~= "vgg16_feat" then
    error("Mode dataset requiert un checkpoint vgg16_feat, mais le checkpoint semble être: " .. tostring(ckpt_arch))
  end

  local ckpt_mc = read_ckpt_model_config(ckpt_dir0)
  if not has_opt("image-w") and ckpt_mc.image_w then image_w = math.floor(ckpt_mc.image_w) end
  if not has_opt("image-h") and ckpt_mc.image_h then image_h = math.floor(ckpt_mc.image_h) end
  if not has_opt("image-c") and ckpt_mc.image_c then image_c = math.floor(ckpt_mc.image_c) end
  if not has_opt("base-channels") and ckpt_mc.base_channels then base_channels = math.floor(ckpt_mc.base_channels) end

  local cfg0, err = Mimir.Architectures.default_config("vgg16_feat")
  assert(type(cfg0) == "table", "default_config(vgg16_feat) failed: " .. tostring(err))
  local cfg = cfg0
  cfg.image_w = image_w
  cfg.image_h = image_h
  cfg.image_c = image_c
  cfg.base_channels = base_channels
  log(string.format("Mode dataset (vgg16_feat): image=%dx%dx%d base_channels=%d (embed_dim=%d)", image_w, image_h, image_c, base_channels, 23 * base_channels))

  create_and_load_model("vgg16_feat", cfg, ckpt_root, true)
end

-- -------------------------
-- Load dataset
-- -------------------------
local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root, image_w, image_h, min_modalities, true, "dataset_cache.json", 10240, true)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
local n_items = tonumber(n_or_err) or 0
if n_items <= 0 then
  error("Dataset vide ou inaccessible")
end

if max_items > 0 and n_items > max_items then
  n_items = max_items
end

-- Shuffle indices
math.randomseed(seed)
local indices = {}
for i = 1, n_items do indices[i] = i end
for i = n_items, 2, -1 do
  local j = math.random(1, i)
  indices[i], indices[j] = indices[j], indices[i]
end

-- -------------------------
-- Build embeddings + labels
-- -------------------------
local items = {}
items[n_items] = false

local n_labeled = 0
for ii = 1, n_items do
  local i = indices[ii]
  local it, e = Mimir.Dataset.get(i)
  if it ~= nil and type(it) == "table" and it.image ~= nil then
    local label = split_first_tag(it.text)
    if label ~= nil then
      local x = normalize_u8_to_f32_minus1_1(it.image)
      local emb = Mimir.Model.forward(x, true)
      items[ii] = { label = label, emb = emb }
      n_labeled = n_labeled + 1
    end
  end
end

if n_labeled <= 0 then
  error("Aucun item labelisé trouvé: besoin de text+image. (Par défaut label = premier tag avant '.')")
end

-- -------------------------
-- Choose refs per class
-- -------------------------
local class_refs = {}
local class_counts = {}

for ii = 1, n_items do
  local r = items[ii]
  if r then
    local c = r.label
    class_counts[c] = (class_counts[c] or 0) + 1
  end
end

for c, cnt in pairs(class_counts) do
  class_refs[c] = {}
end

local ref_used = {}
local ref_total = 0
for ii = 1, n_items do
  local r = items[ii]
  if r then
    local c = r.label
    local cur = class_refs[c]
    if #cur < ref_per_class then
      cur[#cur + 1] = r.emb
      ref_used[ii] = true
      ref_total = ref_total + 1
    end
  end
end

local classes = {}
for c, _ in pairs(class_refs) do classes[#classes + 1] = c end
table.sort(classes)

-- Compute centroids
local centroids = {}
for _, c in ipairs(classes) do
  local refs = class_refs[c]
  if #refs > 0 then
    local dim = #refs[1]
    local mu = {}
    mu[dim] = 0.0
    for d = 1, dim do mu[d] = 0.0 end
    for _, e in ipairs(refs) do
      for d = 1, dim do
        mu[d] = mu[d] + e[d]
      end
    end
    local inv = 1.0 / math.max(1, #refs)
    for d = 1, dim do
      mu[d] = mu[d] * inv
    end
    centroids[c] = mu
  end
end

-- -------------------------
-- Evaluate
-- -------------------------
local function score_to_class(emb, c)
  local proto = centroids[c]
  if not proto then return nil end
  if metric == "l2" or metric == "l2sq" then
    return l2_dist2(emb, proto)
  end
  return cosine_sim(emb, proto)
end

local n_eval = 0
local n_ok = 0
local per_class_ok = {}
local per_class_n = {}

for ii = 1, n_items do
  local r = items[ii]
  if r and not ref_used[ii] then
    n_eval = n_eval + 1

    local scores = {}
    for ci = 1, #classes do
      local c = classes[ci]
      scores[ci] = score_to_class(r.emb, c) or (higher_is_better and -1e30 or 1e30)
    end

    local rank = topk(scores, math.max(1, top_k), higher_is_better)
    local pred = classes[rank[1]]

    per_class_n[r.label] = (per_class_n[r.label] or 0) + 1
    if pred == r.label then
      n_ok = n_ok + 1
      per_class_ok[r.label] = (per_class_ok[r.label] or 0) + 1
    end

    if n_eval <= 20 then
      -- Print a short trace for first items
      local top = {}
      for t = 1, math.min(#rank, top_k) do
        local ci = rank[t]
        local c = classes[ci]
        local s = scores[ci]
        top[#top + 1] = string.format("%s(%.4g)", c, s)
      end
      log(string.format("[%d] gt=%s pred=%s top=%s", n_eval, r.label, pred, table.concat(top, ", ")))
    end
  end
end

log("=== Résumé ===")
log(string.format("- classes=%d", #classes))
log(string.format("- refs_total=%d (ref_per_class=%d)", ref_total, ref_per_class))
log(string.format("- eval_items=%d", n_eval))
if n_eval > 0 then
  log(string.format("- accuracy=%.2f%%", 100.0 * (n_ok / n_eval)))
else
  log("- accuracy=N/A (pas d'items d'éval; augmente max_items ou baisse ref_per_class)")
end

for _, c in ipairs(classes) do
  local ok = per_class_ok[c] or 0
  local nn = per_class_n[c] or 0
  if nn > 0 then
    log(string.format("- %s: %.1f%% (%d/%d)", c, 100.0 * (ok / nn), ok, nn))
  end
end
