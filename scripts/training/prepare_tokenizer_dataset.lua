---@diagnostic disable: undefined-global, undefined-field, inject-field
-- Prépare un tokenizer (load ou create), le compose avec un dataset (prepare_sequences)
-- et sauvegarde le tokenizer + le cache dataset.
--
-- Exemple:
--   ./bin/mimir --lua scripts/training/prepare_tokenizer_dataset.lua -- \
--     --dataset-root ./dataset_2 --seq-len 64 --max-vocab 50000 \
--     --base-tokenizer checkpoint/base_tokenizer/tokenizer.json
--
-- Options utiles:
--   --max-items N          (0 = tout le dataset)
--   --min-modalities 2     (2 = image+texte, 1 = image only)
--   --cache-path path      (défaut: dataset_cache.json)
--   --max-ram-mb 10240
--   --lazy-loading true
--   --freeze true          (défaut: true) -> set_max_vocab(vocab_size)
--   --require-base true    (si true: fail si tokenizer absent)

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local BaseTok = dofile(ROOTWORK.."/scripts/modules/base_tokenizer.lua")

local opts = Args.parse(arg) or {}

local function log_msg(s)
  local l = rawget(_G, "log")
  if type(l) == "function" then
    l(tostring(s))
  else
    print(tostring(s))
  end
end

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

local function read_text_file(path, max_bytes)
  if not path or path == "" then return nil, "empty path" end
  local f = io.open(path, "rb")
  if not f then return nil, "cannot open: " .. tostring(path) end
  local content
  if max_bytes and max_bytes > 0 then
    content = f:read(max_bytes)
  else
    content = f:read("*a")
  end
  f:close()
  if not content then return nil, "read failed: " .. tostring(path) end
  return content
end

local function file_exists(path)
  if not path or path == "" then return false end
  local f = io.open(tostring(path), "rb")
  if f then f:close(); return true end
  return false
end

local function path_join(a, b)
  if not a or a == "" then return b end
  if not b or b == "" then return a end
  if a:sub(-1) == "/" then return a .. b end
  return a .. "/" .. b
end

local function resolve_text_path(dataset_root, p)
  if not p then return nil end
  p = tostring(p)
  if p:match("^/") then return p end
  -- Essai direct
  local f = io.open(p, "rb")
  if f then f:close(); return p end
  -- Essai relatif au dataset_root
  local joined = path_join(dataset_root, p)
  local f2 = io.open(joined, "rb")
  if f2 then f2:close(); return joined end
  return p
end

-- --------------------------------------------------------------------------
-- Entrée
-- --------------------------------------------------------------------------

if not (Mimir and Mimir.Tokenizer and Mimir.Dataset) then
  error("Ce script doit être lancé via le binaire mimir (Mimir.* indisponible)")
end

local dataset_root = opt_str("dataset-root", opt_str("dataset_root", "./dataset_2"))
local cache_path = opt_str("cache-path", opt_str("dataset-cache", opt_str("dataset_cache", "dataset_cache.json")))

local base_tok_path = opt_str("base-tokenizer", opt_str("base_tokenizer", BaseTok.default_path()))
local max_vocab = opt_int("max-vocab", opt_int("max_vocab", 50000))
local seq_len = opt_int("seq-len", opt_int("seq_len", 64))

local max_items = opt_int("max-items", opt_int("max_items", 0))
local min_modalities = opt_int("min-modalities", opt_int("min_modalities", 2))

local image_w = opt_int("image-w", opt_int("image_w", 0))
local image_h = opt_int("image-h", opt_int("image_h", 0))
if image_w <= 0 then image_w = nil end
if image_h <= 0 then image_h = nil end

local max_ram_mb = opt_int("max-ram-mb", opt_int("max_ram_mb", opt_int("max-ram", opt_int("max_ram", 10240))))
local lazy_loading = opt_bool("lazy-loading", opt_bool("lazy_loading", opt_bool("lazy", true)))

local require_base = opt_bool("require-base", opt_bool("require_base", false))
local freeze_vocab = opt_bool("freeze", opt_bool("freeze-vocab", opt_bool("freeze_vocab", true)))
local prepare_sequences = opt_bool("prepare-sequences", opt_bool("prepare_sequences", true))
local build_cache = opt_bool("build-cache", opt_bool("build_cache", true))
local ignore_cache_errors = opt_bool("ignore-cache-errors", opt_bool("ignore_cache_errors", false))
local use_cache_opt = opt_bool("use-cache", opt_bool("use_cache", true))

local max_text_bytes = opt_int("max-text-bytes", opt_int("max_text_bytes", 1024 * 1024))

log_msg("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log_msg("  Préparation tokenizer + dataset")
log_msg("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log_msg(string.format("dataset_root=%s", dataset_root))
log_msg(string.format("cache_path=%s", cache_path))
log_msg(string.format("base_tokenizer=%s", base_tok_path))
log_msg(string.format("max_vocab=%d seq_len=%d", max_vocab, seq_len))
log_msg(string.format("min_modalities=%d max_items=%d", min_modalities, max_items))
log_msg(string.format("build_cache=%s ignore_cache_errors=%s", tostring(build_cache), tostring(ignore_cache_errors)))

-- 1) Tokenizer: load ou create
log_msg("\n[1/4] Tokenizer: load/create")
local ok_bt, err_bt = BaseTok.load_base({
  path = base_tok_path,
  max_vocab = max_vocab,
  require = require_base,
})
assert_ok(ok_bt, err_bt, "BaseTok.load_base")

if type(Mimir.Tokenizer.set_max_length) == "function" and seq_len and seq_len > 0 then
  pcall(Mimir.Tokenizer.set_max_length, seq_len)
end

if type(Mimir.Tokenizer.set_max_vocab) == "function" then
  local cur_max = nil
  if type(Mimir.Tokenizer.get_max_vocab) == "function" then
    cur_max = tonumber(Mimir.Tokenizer.get_max_vocab())
  end
  if cur_max == nil or max_vocab > cur_max then
    local ok_set, err_set = Mimir.Tokenizer.set_max_vocab(max_vocab)
    if ok_set == false then
      log_msg("⚠️  Tokenizer.set_max_vocab a échoué: " .. tostring(err_set))
    end
  end
end

-- 2) Dataset cache + load
log_msg("\n[2/4] Dataset: cache + load")
local cache_built_ok = false
if type(Mimir.Database) == "table" and type(Mimir.Database.load) == "function" then
  local loader = Mimir.Database.load(dataset_root, image_w, image_h, min_modalities)
  if build_cache and loader and type(loader.cache) == "function" then
    -- IMPORTANT: selon la build, `cache(a,b,c)` peut être interprété comme (dir, w, h)
    -- au lieu de (cache_path, max_ram_mb, lazy_loading). On passe donc tous les args.
    local function try_cache_call(call_with_self)
      if call_with_self then
        return pcall(loader.cache, loader, dataset_root, image_w, image_h, min_modalities, cache_path, max_ram_mb, lazy_loading)
      end
      return pcall(loader.cache, dataset_root, image_w, image_h, min_modalities, cache_path, max_ram_mb, lazy_loading)
    end

    local ok_pcall, ok_cache, n_or_err = try_cache_call(false)
    if not ok_pcall then
      ok_pcall, ok_cache, n_or_err = try_cache_call(true)
    end

    if not ok_pcall then
      if ignore_cache_errors then
        log_msg("⚠️  Cache dataset: exception ignorée: " .. tostring(ok_cache))
      else
        error("DatabaseLoader.cache: " .. tostring(ok_cache))
      end
    else
      if ok_cache == false then
        if ignore_cache_errors then
          log_msg("⚠️  Cache dataset: échec ignoré: " .. tostring(n_or_err))
        else
          assert_ok(ok_cache, n_or_err, "DatabaseLoader.cache")
        end
      else
        cache_built_ok = true
        log_msg("✓ Cache dataset écrit/à jour: " .. tostring(cache_path) .. " (items=" .. tostring(n_or_err) .. ")")
      end
    end
  end
end

local cache_exists = file_exists(cache_path)
local use_cache = (use_cache_opt == true) and (cache_built_ok or cache_exists)

local function dataset_load(use_cache_flag)
  return Mimir.Dataset.load(
    dataset_root,
    image_w,
    image_h,
    min_modalities,
    use_cache_flag,
    use_cache_flag and cache_path or nil,
    max_ram_mb,
    lazy_loading
  )
end

local ok_ds, n_or_err = dataset_load(use_cache)
if ok_ds == false and ignore_cache_errors and use_cache == true then
  log_msg("⚠️  Dataset.load (cached) a échoué, retry sans cache: " .. tostring(n_or_err))
  ok_ds, n_or_err = dataset_load(false)
end

assert_ok(ok_ds, n_or_err, "Dataset.load")
local n_items = tonumber(n_or_err) or 0
log_msg("✓ Dataset chargé: " .. tostring(n_items))

-- 3) Composer: construire vocab à partir des textes
log_msg("\n[3/4] Tokenizer: build vocab depuis dataset")
local scan_n = n_items
if max_items and max_items > 0 then
  scan_n = math.min(scan_n, max_items)
end

local had_text = 0
local err_text = 0
local err_tok = 0

local ensure_vocab = (type(Mimir.Tokenizer.ensure_vocab_from_text) == "function") and Mimir.Tokenizer.ensure_vocab_from_text or nil
local tok_ensure = (type(Mimir.Tokenizer.tokenize_ensure) == "function") and Mimir.Tokenizer.tokenize_ensure or nil
local tok_plain = (type(Mimir.Tokenizer.tokenize) == "function") and Mimir.Tokenizer.tokenize or nil

for i = 1, scan_n do
  local item, err_item = Mimir.Dataset.get(i)
  if not item then
    err_text = err_text + 1
    goto continue
  end

  local text = item.text
  if (text == nil or text == "") and item.text_file ~= nil then
    local tp = resolve_text_path(dataset_root, item.text_file)
    local t, terr = read_text_file(tp, max_text_bytes)
    if t then
      text = t
    else
      err_text = err_text + 1
      goto continue
    end
  end

  if text ~= nil and text ~= "" then
    had_text = had_text + 1
    if ensure_vocab then
      local ok_ev, err_ev = ensure_vocab(text)
      if ok_ev == false then
        err_tok = err_tok + 1
      end
    elseif tok_ensure then
      -- tokenize_ensure ne retourne pas ok/err, donc pcall.
      local ok_call = pcall(tok_ensure, text)
      if not ok_call then err_tok = err_tok + 1 end
    elseif tok_plain then
      local ok_call = pcall(tok_plain, text)
      if not ok_call then err_tok = err_tok + 1 end
    end
  end

  ::continue::
end

local vocab_size = (type(Mimir.Tokenizer.vocab_size) == "function") and tonumber(Mimir.Tokenizer.vocab_size()) or 0
log_msg(string.format("✓ Scan terminé: items=%d textes=%d err_text=%d err_tok=%d vocab=%d", scan_n, had_text, err_text, err_tok, vocab_size))

if type(Mimir.Tokenizer.print_stats) == "function" then
  pcall(Mimir.Tokenizer.print_stats)
end

if freeze_vocab and type(Mimir.Tokenizer.set_max_vocab) == "function" and type(Mimir.Tokenizer.vocab_size) == "function" then
  local vs = tonumber(Mimir.Tokenizer.vocab_size()) or 0
  if vs > 0 then
    local ok_fr, err_fr = Mimir.Tokenizer.set_max_vocab(vs)
    if ok_fr == false then
      log_msg("⚠️  freeze: Tokenizer.set_max_vocab(vocab_size) a échoué: " .. tostring(err_fr))
    else
      log_msg("✓ Vocab gelé: max_vocab=" .. tostring(vs))
    end
  end
end

-- 4) Compose: prepare sequences + save tokenizer
log_msg("\n[4/4] Compose: prepare_sequences + save")
if prepare_sequences and min_modalities >= 2 and type(Mimir.Dataset.prepare_sequences) == "function" and seq_len and seq_len > 0 then
  local ok_prep, err_prep = Mimir.Dataset.prepare_sequences(seq_len)
  assert_ok(ok_prep, err_prep, "Dataset.prepare_sequences")
end

local ok_save, err_save = BaseTok.save_current_as_base(base_tok_path)
assert_ok(ok_save, err_save, "Tokenizer.save")
log_msg("✓ Tokenizer sauvegardé: " .. tostring(base_tok_path))

log_msg("\n✅ Préparation terminée")
