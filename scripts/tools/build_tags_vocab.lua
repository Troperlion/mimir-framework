-- Build a tags vocabulary file from a dataset of (image + text) pairs.
--
-- It supports four input families:
--   1) text sidecars (`.txt`) under `--dataset-root`
--   2) COCO captions JSON (`captions_train2017.json` / `captions_val2017.json`)
--   3) VinDr-Mammo annotations (`breast-level_annotations.csv` and
--      `finding_annotations.csv`), optionally preparing image + `.txt` pairs.
--   4) MVinDr mass segmentation (`manifest.csv`, `images/`, `masks/`).
--
-- It normalizes, counts frequencies, and writes a vocab file (one tag/token per line).
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/build_tags_vocab.lua -- \
--     --dataset-root dataset_2 \
--     --out checkpoint/tags_vocab.txt \
--     --dataset-format auto \
--     --split-mode auto \
--     --lowercase true \
--     --min-freq 2 \
--     --top-k 5000
--
-- MVinDr mass segmentation -> dataset directly consumable by vgg16_tags_multilabel:
--   ./bin/mimir --lua scripts/tools/build_tags_vocab.lua -- \
--     --dataset-root "/path/to/MVinDr - Mammo Mass Segmentation Dataset" \
--     --vindr-prepared-root dataset/mvindr_tags \
--     --out checkpoint/mvindr_tags_vocab.txt
--
-- Notes:
-- - In txt/COCO mode this tool does not open images. VinDr preparation links,
--   copies, or converts images but never decodes them inside Lua.
-- - Output is sorted by (freq desc, tag asc) for determinism.

local ToolHelp = dofile(ROOTWORK.."/scripts/modules/tools_help.lua")
ToolHelp.show("build_tags_vocab")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

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

local function ensure_parent_dir(filepath)
  local dir = FS.dirname(filepath)
  if dir and #dir > 0 then
    FS.mkdir_p(dir)
  end
end

local function strip_extension(path)
  local s = tostring(path or "")
  local base = s:match("^(.*)%.([^.]+)$")
  return base or s
end

local function default_composition_out(vocab_path)
  return strip_extension(vocab_path) .. ".composition.json"
end

local function json_escape(s)
  s = tostring(s or "")
  s = s:gsub("\\", "\\\\")
  s = s:gsub('"', '\\"')
  s = s:gsub("\n", "\\n")
  s = s:gsub("\r", "\\r")
  s = s:gsub("\t", "\\t")
  return s
end

local function collect_txt_files(root, out)
  local entries = FS.list_dir(root)
  table.sort(entries)
  for _, name in ipairs(entries) do
    local full = FS.join(root, name)
    if FS.is_dir(full) then
      collect_txt_files(full, out)
    elseif name:match("%.txt$") then
      out[#out + 1] = full
    end
  end
end

local function trim(s)
  s = tostring(s or "")
  s = s:gsub("^%s+", "")
  s = s:gsub("%s+$", "")
  return s
end

local function normalize_spaces(s)
  -- Collapse whitespace to single spaces.
  s = s:gsub("[%s\t\r\n]+", " ")
  return trim(s)
end

local function decode_json_table(raw)
  local json_mod = rawget(_G, "json")
  if type(json_mod) == "table" and type(json_mod.decode) == "function" then
    local ok, v = pcall(json_mod.decode, raw)
    if ok and type(v) == "table" then return v end
  end

  local cjson_mod = rawget(_G, "cjson")
  if type(cjson_mod) == "table" and type(cjson_mod.decode) == "function" then
    local ok, v = pcall(cjson_mod.decode, raw)
    if ok and type(v) == "table" then return v end
  end

  return nil
end

local function read_all(path)
  local f = io.open(path, "r")
  if not f then return nil, "cannot open: " .. tostring(path) end
  local s = f:read("*a") or ""
  f:close()
  return s, nil
end

local function find_existing(paths)
  for _, p in ipairs(paths) do
    if FS.file_exists(p) then return p end
  end
  return nil
end

local function command_exists(name)
  if tostring(name or ""):match("[^%w_.%-]") then return false end
  local redirect = FS.is_windows() and " >NUL 2>NUL" or " >/dev/null 2>&1"
  local cmd = FS.is_windows() and ("where " .. name .. redirect) or ("command -v " .. name .. redirect)
  local ok, why, code = os.execute(cmd)
  if type(ok) == "number" then return ok == 0 end
  if type(ok) == "boolean" then return ok end
  return why == "exit" and code == 0
end

local function run_command(cmd)
  local ok, why, code = os.execute(cmd)
  if type(ok) == "number" then return ok == 0 end
  if type(ok) == "boolean" then return ok end
  return why == "exit" and code == 0
end

local function is_symlink(path)
  if FS.is_windows() then return false end
  return run_command("test -L " .. FS.quote(path) .. " >/dev/null 2>&1")
end

local function unlink_symlink(path)
  if not is_symlink(path) then return true end
  return run_command("unlink -- " .. FS.quote(path))
end

local function parent_dir(path)
  return FS.dirname(path)
end

local function basename(path)
  local p = tostring(path or "")
  local i = p:match("^.*()/")
  if not i then return p end
  return p:sub(i + 1)
end

local function guess_coco_annotations(dataset_root, explicit_path)
  if explicit_path and explicit_path ~= "" then
    return explicit_path
  end

  local root = tostring(dataset_root or "")
  local leaf = basename(root)
  local parent = parent_dir(root) or "."

  local candidates = {}
  if leaf == "train2017" then
    candidates[#candidates + 1] = FS.join(parent, "annotations/captions_train2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_train2017.json")
  elseif leaf == "val2017" then
    candidates[#candidates + 1] = FS.join(parent, "annotations/captions_val2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_val2017.json")
  else
    candidates[#candidates + 1] = FS.join(root, "annotations/captions_train2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_train2017.json")
    candidates[#candidates + 1] = FS.join(root, "annotations/captions_val2017.json")
    candidates[#candidates + 1] = FS.join(root, "captions_val2017.json")
  end

  return find_existing(candidates)
end

local function split_words(s)
  local out = {}
  local cur = {}
  local function flush()
    if #cur == 0 then return end
    out[#out + 1] = table.concat(cur)
    cur = {}
  end

  for i = 1, #s do
    local ch = s:sub(i, i)
    if ch:match("[%w_%-']") then
      cur[#cur + 1] = ch
    else
      flush()
    end
  end
  flush()
  return out
end

-- RFC 4180-compatible row parser (including commas and doubled quotes in fields).
local function parse_csv_row(line)
  local row, field, quoted = {}, {}, false
  local i = 1
  while i <= #line do
    local ch = line:sub(i, i)
    if quoted then
      if ch == '"' then
        if line:sub(i + 1, i + 1) == '"' then
          field[#field + 1] = '"'
          i = i + 1
        else
          quoted = false
        end
      else
        field[#field + 1] = ch
      end
    elseif ch == '"' and #field == 0 then
      quoted = true
    elseif ch == "," then
      row[#row + 1] = table.concat(field)
      field = {}
    else
      field[#field + 1] = ch
    end
    i = i + 1
  end
  row[#row + 1] = table.concat(field)
  return row
end

local function read_csv(path)
  local f = io.open(path, "r")
  if not f then return nil, "cannot open: " .. tostring(path) end
  local header_line = f:read("*l")
  if not header_line then f:close(); return nil, "empty CSV: " .. tostring(path) end
  header_line = header_line:gsub("^\239\187\191", "")
  local headers = parse_csv_row(header_line)
  local rows = {}
  for line in f:lines() do
    if trim(line) ~= "" then
      local values = parse_csv_row(line)
      local row = {}
      for i, name in ipairs(headers) do row[trim(name)] = values[i] or "" end
      rows[#rows + 1] = row
    end
  end
  f:close()
  return rows, nil
end

local function normalize_vindr_value(value)
  local s = normalize_spaces(value)
  if s == "" or s:lower() == "nan" or s:lower() == "none" or s:lower() == "null" then return nil end
  return s
end

local function parse_vindr_categories(value)
  local s = normalize_vindr_value(value)
  if not s then return {} end
  local out, seen = {}, {}
  local function push(v)
    v = normalize_spaces(v):gsub("^['\"]+", ""):gsub("['\"]+$", "")
    if v ~= "" and not seen[v] then seen[v] = true; out[#out + 1] = v end
  end
  local matched = false
  for v in s:gmatch("['\"](.-)['\"]") do push(v); matched = true end
  if not matched then
    s = s:gsub("^%s*%[", ""):gsub("%]%s*$", "")
    for v in s:gmatch("[^|;,]+") do push(v) end
  end
  return out
end

local function collect_image_files(root, out)
  if not FS.is_dir(root) then return end
  local entries = FS.list_dir(root)
  table.sort(entries)
  for _, name in ipairs(entries) do
    local full = FS.join(root, name)
    if FS.is_dir(full) then
      collect_image_files(full, out)
    elseif name:lower():match("%.(png)$") or name:lower():match("%.(jpe?g)$")
        or name:lower():match("%.(bmp)$") or name:lower():match("%.(tiff?)$")
        or name:lower():match("%.(webp)$") or name:lower():match("%.(dcm)$")
        or name:lower():match("%.(dicom)$") then
      local stem = name:gsub("%.[^.]+$", "")
      if not out[stem] then out[stem] = full end
    end
  end
end

local dataset_root = opt_str("dataset-root", "dataset_2")
local out_path = opt_str("out", "checkpoint/tags_vocab.txt")
local lowercase = opt_bool("lowercase", true)
local min_freq = opt_int("min-freq", 1)
local top_k = opt_int("top-k", 0)
local max_files = opt_int("max-files", 0)
local dataset_format = opt_str("dataset-format", "auto") -- auto|txt|coco|vindr-mammo
local split_mode = opt_str("split-mode", "auto") -- auto|phrases|tokens|both
local coco_annotations = opt_str("coco-annotations", "")
local composition_out = opt_str("composition-out", default_composition_out(out_path))
local vindr_breast_annotations = opt_str("vindr-breast-annotations", "")
local vindr_finding_annotations = opt_str("vindr-finding-annotations", "")
local vindr_images_root = opt_str("vindr-images-root", "")
local vindr_prepared_root = opt_str("vindr-prepared-root", "")
local vindr_split = opt_str("vindr-split", "training") -- training|test|all
local vindr_labels = opt_str("vindr-labels", "all") -- findings|diagnostic|all
local vindr_image_mode = opt_str("vindr-image-mode", "auto") -- auto|copy|hardlink|symlink|convert
local vindr_dicom_converter = opt_str("vindr-dicom-converter", "")
local mvindr_manifest = opt_str("mvindr-manifest", "")
local mvindr_mask_tags = opt_bool("mvindr-mask-tags", true)
local mvindr_small_max = opt_num("mvindr-small-max", 0.005)
local mvindr_large_min = opt_num("mvindr-large-min", 0.02)

dataset_format = tostring(dataset_format):lower()
split_mode = tostring(split_mode):lower()

if dataset_format == "vindr" then dataset_format = "vindr-mammo" end
if dataset_format == "mvindr" then dataset_format = "mvindr-mass" end
if dataset_format ~= "auto" and dataset_format ~= "txt" and dataset_format ~= "coco"
    and dataset_format ~= "vindr-mammo" and dataset_format ~= "mvindr-mass" then
  error("dataset-format invalide (auto|txt|coco|vindr-mammo|mvindr-mass): " .. tostring(dataset_format))
end
if split_mode ~= "auto" and split_mode ~= "phrases" and split_mode ~= "tokens" and split_mode ~= "both" then
  error("split-mode invalide (auto|phrases|tokens|both): " .. tostring(split_mode))
end
vindr_split = tostring(vindr_split):lower()
vindr_labels = tostring(vindr_labels):lower()
vindr_image_mode = tostring(vindr_image_mode):lower()
if vindr_split ~= "training" and vindr_split ~= "test" and vindr_split ~= "all" then
  error("vindr-split invalide (training|test|all): " .. tostring(vindr_split))
end
if vindr_labels ~= "findings" and vindr_labels ~= "diagnostic" and vindr_labels ~= "all" then
  error("vindr-labels invalide (findings|diagnostic|all): " .. tostring(vindr_labels))
end
if vindr_image_mode ~= "auto" and vindr_image_mode ~= "copy" and vindr_image_mode ~= "hardlink"
    and vindr_image_mode ~= "symlink" and vindr_image_mode ~= "convert" then
  error("vindr-image-mode invalide (auto|copy|hardlink|symlink|convert): " .. tostring(vindr_image_mode))
end
if mvindr_small_max <= 0 or mvindr_large_min <= mvindr_small_max then
  error("seuils MVinDr invalides: 0 < --mvindr-small-max < --mvindr-large-min requis")
end

if min_freq < 1 then min_freq = 1 end
if top_k < 0 then top_k = 0 end
if max_files < 0 then max_files = 0 end

log("=== build_tags_vocab ===")
log("- dataset_root=" .. tostring(dataset_root))
log("- out=" .. tostring(out_path))
log("- dataset_format=" .. tostring(dataset_format) .. " split_mode=" .. tostring(split_mode))
log("- lowercase=" .. tostring(lowercase))
log("- min_freq=" .. tostring(min_freq) .. " top_k=" .. tostring(top_k) .. " max_files=" .. tostring(max_files))
log("- composition_out=" .. tostring(composition_out))

local files = {}
collect_txt_files(dataset_root, files)

local detected_format = dataset_format
if detected_format == "auto" then
  if FS.file_exists(FS.join(dataset_root, "manifest.csv"))
      and FS.is_dir(FS.join(dataset_root, "images"))
      and FS.is_dir(FS.join(dataset_root, "masks")) then
    detected_format = "mvindr-mass"
  elseif FS.file_exists(FS.join(dataset_root, "breast-level_annotations.csv")) then
    detected_format = "vindr-mammo"
  elseif #files > 0 then
    detected_format = "txt"
  else
    local coco_json = guess_coco_annotations(dataset_root, coco_annotations)
    if coco_json then
      detected_format = "coco"
      coco_annotations = coco_json
    else
      error("Impossible de détecter le format dataset (ni .txt, ni captions COCO JSON)")
    end
  end
end

if detected_format == "txt" then
  if max_files > 0 and #files > max_files then
    local sliced = {}
    for i = 1, max_files do sliced[i] = files[i] end
    files = sliced
  end
  if #files == 0 then
    error("Aucun .txt trouvé sous dataset-root=" .. tostring(dataset_root))
  end
  log("- txt_files=" .. tostring(#files))
elseif detected_format == "coco" then
  if coco_annotations == "" then
    local guessed = guess_coco_annotations(dataset_root, nil)
    if guessed then coco_annotations = guessed end
  end
  if coco_annotations == "" or not FS.file_exists(coco_annotations) then
    error("Fichier COCO annotations introuvable. Utilise --coco-annotations <captions_*.json>")
  end
  log("- coco_annotations=" .. tostring(coco_annotations))
elseif detected_format == "vindr-mammo" then
  if vindr_breast_annotations == "" then
    vindr_breast_annotations = FS.join(dataset_root, "breast-level_annotations.csv")
  end
  if vindr_finding_annotations == "" then
    vindr_finding_annotations = FS.join(dataset_root, "finding_annotations.csv")
  end
  if vindr_images_root == "" then
    local image_root_candidates = {
      FS.join(dataset_root, "images_png"), FS.join(dataset_root, "png"), FS.join(dataset_root, "images")
    }
    for _, candidate in ipairs(image_root_candidates) do
      if FS.is_dir(candidate) then vindr_images_root = candidate; break end
    end
    if vindr_images_root == "" then vindr_images_root = FS.join(dataset_root, "images") end
  end
  if not FS.file_exists(vindr_breast_annotations) then
    error("VinDr-Mammo: breast-level_annotations.csv introuvable: " .. tostring(vindr_breast_annotations))
  end
  if not FS.file_exists(vindr_finding_annotations) then
    error("VinDr-Mammo: finding_annotations.csv introuvable: " .. tostring(vindr_finding_annotations))
  end
  if vindr_prepared_root == "" then
    error("VinDr-Mammo: --vindr-prepared-root est requis pour créer les paires image + .txt destinées à Mímir")
  end
  log("- vindr_breast_annotations=" .. tostring(vindr_breast_annotations))
  log("- vindr_finding_annotations=" .. tostring(vindr_finding_annotations))
  log("- vindr_images_root=" .. tostring(vindr_images_root))
  log("- vindr_prepared_root=" .. tostring(vindr_prepared_root))
  log("- vindr_split=" .. tostring(vindr_split) .. " vindr_labels=" .. tostring(vindr_labels))
else
  if mvindr_manifest == "" then mvindr_manifest = FS.join(dataset_root, "manifest.csv") end
  if not FS.file_exists(mvindr_manifest) then
    error("MVinDr: manifest.csv introuvable: " .. tostring(mvindr_manifest))
  end
  if vindr_prepared_root == "" then
    error("MVinDr: --vindr-prepared-root est requis pour créer les paires image + .txt destinées à Mímir")
  end
  log("- mvindr_manifest=" .. tostring(mvindr_manifest))
  log("- vindr_prepared_root=" .. tostring(vindr_prepared_root))
  log("- mvindr_mask_tags=" .. tostring(mvindr_mask_tags) ..
      " small_max=" .. tostring(mvindr_small_max) .. " large_min=" .. tostring(mvindr_large_min))
end

if split_mode == "auto" then
  if detected_format == "coco" then
    split_mode = "tokens"
  else
    split_mode = "phrases"
  end
end
log("- split_mode_effective=" .. tostring(split_mode))

local freq = {}
local item_freq = {}
local total_tags = 0
local total_samples = 0

local function add_tag(tag, seen)
  tag = normalize_spaces(tag)
  if tag == "" then return end
  if lowercase then tag = tag:lower() end
  freq[tag] = (freq[tag] or 0) + 1
  if seen and not seen[tag] then
    seen[tag] = true
    item_freq[tag] = (item_freq[tag] or 0) + 1
  end
  total_tags = total_tags + 1
end

local function process_text(txt)
  -- split on '.', puis optionnellement ajoute des tokens mot-à-mot.
  total_samples = total_samples + 1
  local seen = {}
  local cur = ""
  local function emit_piece(s)
    s = normalize_spaces(s)
    if s == "" then return end

    if split_mode == "phrases" or split_mode == "both" then
      add_tag(s, seen)
    end
    if split_mode == "tokens" or split_mode == "both" then
      local words = split_words(s)
      for _, w in ipairs(words) do
        add_tag(w, seen)
      end
    end
  end

  for i = 1, #txt do
    local ch = txt:sub(i, i)
    if ch == "." then
      emit_piece(cur)
      cur = ""
    else
      cur = cur .. ch
    end
  end
  emit_piece(cur)
end

local function unescape_json_string(s)
  s = s:gsub('\\"', '"')
  s = s:gsub("\\n", "\n")
  s = s:gsub("\\r", "\r")
  s = s:gsub("\\t", "\t")
  s = s:gsub("\\/", "/")
  s = s:gsub("\\\\", "\\")
  return s
end

local function process_coco_annotations(path)
  local raw, err = read_all(path)
  if not raw then error("lecture coco annotations échouée: " .. tostring(err)) end

  local decoded = decode_json_table(raw)
  local count = 0
  if type(decoded) == "table" and type(decoded.annotations) == "table" then
    for _, ann in ipairs(decoded.annotations) do
      if type(ann) == "table" and type(ann.caption) == "string" then
        process_text(ann.caption)
        count = count + 1
      end
    end
    return count
  end

  -- Fallback robuste si module JSON indisponible: extraction directe des champs "caption".
  for cap in raw:gmatch('"caption"%s*:%s*"(.-)"') do
    process_text(unescape_json_string(cap))
    count = count + 1
  end
  return count
end

local function process_vindr_mammo()
  local breast_rows, breast_err = read_csv(vindr_breast_annotations)
  if not breast_rows then error("VinDr-Mammo: " .. tostring(breast_err)) end
  local finding_rows, finding_err = read_csv(vindr_finding_annotations)
  if not finding_rows then error("VinDr-Mammo: " .. tostring(finding_err)) end

  local findings_by_image = {}
  for _, row in ipairs(finding_rows) do
    local image_id = normalize_vindr_value(row.image_id)
    if image_id then
      local bucket = findings_by_image[image_id] or {}
      local seen = {}; for _, v in ipairs(bucket) do seen[v] = true end
      for _, category in ipairs(parse_vindr_categories(row.finding_categories)) do
        if not seen[category] then bucket[#bucket + 1] = category; seen[category] = true end
      end
      findings_by_image[image_id] = bucket
    end
  end

  local images_by_id = {}
  collect_image_files(vindr_images_root, images_by_id)
  if next(images_by_id) == nil then
    error("VinDr-Mammo: aucune image PNG/JPEG/BMP/TIFF/WebP/DICOM trouvée sous " .. tostring(vindr_images_root))
  end
  FS.mkdir_p(vindr_prepared_root)

  local used, missing, failed, skipped = 0, 0, 0, 0
  local selected_seen = {}
  for _, row in ipairs(breast_rows) do
    local image_id = normalize_vindr_value(row.image_id)
    local study_id = normalize_vindr_value(row.study_id) or "unknown_study"
    local row_split = (normalize_vindr_value(row.split) or ""):lower()
    if not image_id or selected_seen[image_id] or (vindr_split ~= "all" and row_split ~= vindr_split) then
      skipped = skipped + 1
    else
      selected_seen[image_id] = true
      local source = images_by_id[image_id]
      if not source then
        missing = missing + 1
      else
        local labels, label_seen = {}, {}
        local function add_label(v)
          v = normalize_vindr_value(v)
          if v and not label_seen[v] then label_seen[v] = true; labels[#labels + 1] = v end
        end
        if vindr_labels == "findings" or vindr_labels == "all" then
          local image_findings = findings_by_image[image_id] or {}
          if #image_findings == 0 then add_label("No Finding") else
            for _, category in ipairs(image_findings) do add_label(category) end
          end
        end
        if vindr_labels == "diagnostic" or vindr_labels == "all" then
          local birads = normalize_vindr_value(row.breast_birads)
          local density = normalize_vindr_value(row.breast_density)
          if birads then add_label("breast_birads_" .. birads:gsub("%s+", "_")) end
          if density then add_label("breast_density_" .. density:gsub("%s+", "_")) end
        end

        if #labels == 0 then
          skipped = skipped + 1
        else
          table.sort(labels)
          local dest_dir = FS.join(vindr_prepared_root, study_id)
          FS.mkdir_p(dest_dir)
          local source_ext = source:match("(%.[^.]+)$") or ""
          local is_dicom = source_ext:lower() == ".dcm" or source_ext:lower() == ".dicom"
          local mode = vindr_image_mode
          if mode == "auto" then mode = is_dicom and "convert" or "symlink" end
          local dest_ext = (mode == "convert") and ".png" or source_ext
          local dest_image = FS.join(dest_dir, image_id .. dest_ext)
          local image_ok = FS.file_exists(dest_image)
          if not image_ok then
            if mode == "symlink" then
              if FS.is_windows() then
                image_ok = run_command("copy /Y " .. FS.quote(source) .. " " .. FS.quote(dest_image) .. " >NUL")
              else
                image_ok = run_command("ln -s " .. FS.quote(source) .. " " .. FS.quote(dest_image) .. " >/dev/null 2>&1")
              end
            elseif mode == "copy" then
              local cmd = FS.is_windows() and ("copy /Y " .. FS.quote(source) .. " " .. FS.quote(dest_image) .. " >NUL")
                or ("cp " .. FS.quote(source) .. " " .. FS.quote(dest_image))
              image_ok = run_command(cmd)
            else
              local cmd = vindr_dicom_converter
              if cmd ~= "" then
                cmd = cmd:gsub("{input}", FS.quote(source)):gsub("{output}", FS.quote(dest_image))
              elseif command_exists("magick") then
                cmd = "magick " .. FS.quote(source) .. " -auto-level " .. FS.quote(dest_image)
              elseif command_exists("dcmj2pnm") then
                cmd = "dcmj2pnm +on " .. FS.quote(source) .. " " .. FS.quote(dest_image)
              else
                error("VinDr-Mammo: images DICOM détectées, mais aucun convertisseur disponible. " ..
                  "Installe ImageMagick/DCMTK ou fournis --vindr-dicom-converter 'commande {input} {output}'")
              end
              image_ok = run_command(cmd) and FS.file_exists(dest_image)
            end
          end

          if image_ok then
            local text_path = FS.join(dest_dir, image_id .. ".txt")
            local text_file = io.open(text_path, "w")
            if not text_file then error("VinDr-Mammo: impossible d'écrire " .. tostring(text_path)) end
            text_file:write(table.concat(labels, ". "), ".\n")
            text_file:close()
            process_text(table.concat(labels, ". ") .. ".")
            used = used + 1
            if max_files > 0 and used >= max_files then break end
          else
            failed = failed + 1
          end
        end
      end
    end
  end
  log("- vindr_prepared=" .. tostring(used) .. " missing_images=" .. tostring(missing) ..
      " failed_images=" .. tostring(failed) .. " skipped_rows=" .. tostring(skipped))
  if used == 0 then error("VinDr-Mammo: aucun échantillon exploitable n'a été préparé") end
  if missing > 0 then log("⚠️  VinDr-Mammo: " .. tostring(missing) .. " annotations sans image correspondante") end
  if failed > 0 then error("VinDr-Mammo: échec de préparation pour " .. tostring(failed) .. " image(s)") end
  return used
end

local function resolve_manifest_path(root, value)
  value = normalize_vindr_value(value)
  if not value then return nil end
  if value:match("^/") or value:match("^%a:[/\\]") then return value end
  return FS.join(root, value)
end

local function analyze_mvindr_mask(mask_path)
  if not command_exists("magick") then
    return nil, "ImageMagick (`magick`) est requis pour analyser les masques MVinDr"
  end
  local cmd = "magick " .. FS.quote(mask_path) ..
    " -threshold 0 -format '%w %h %@ %[fx:mean]' info: 2>/dev/null"
  local p = io.popen(cmd)
  if not p then return nil, "impossible de lancer ImageMagick" end
  local raw = p:read("*a") or ""
  local ok = p:close()
  if ok == nil or raw == "" then return nil, "masque vide ou illisible: " .. tostring(mask_path) end
  local w, h, bw, bh, bx, by, ratio = raw:match("^(%d+)%s+(%d+)%s+(%d+)x(%d+)%+(%-?%d+)%+(%-?%d+)%s+([%d.eE+%-]+)")
  w, h, bw, bh, bx, by, ratio = tonumber(w), tonumber(h), tonumber(bw), tonumber(bh), tonumber(bx), tonumber(by), tonumber(ratio)
  if not w or not h or not bw or not bh or not bx or not by or not ratio or w <= 0 or h <= 0 then
    return nil, "géométrie de masque invalide: " .. tostring(raw)
  end
  local cx = (bx + bw * 0.5) / w
  local cy = (by + bh * 0.5) / h
  local tags = {}
  if ratio < mvindr_small_max then tags[#tags + 1] = "mass_size_small"
  elseif ratio >= mvindr_large_min then tags[#tags + 1] = "mass_size_large"
  else tags[#tags + 1] = "mass_size_medium" end
  if cx < 1 / 3 then tags[#tags + 1] = "mass_zone_image_left"
  elseif cx >= 2 / 3 then tags[#tags + 1] = "mass_zone_image_right"
  else tags[#tags + 1] = "mass_zone_image_center" end
  if cy < 1 / 3 then tags[#tags + 1] = "mass_zone_upper"
  elseif cy >= 2 / 3 then tags[#tags + 1] = "mass_zone_lower"
  else tags[#tags + 1] = "mass_zone_middle" end
  return tags, nil
end

local function process_mvindr_mass()
  local rows, err = read_csv(mvindr_manifest)
  if not rows then error("MVinDr: " .. tostring(err)) end
  local manifest_root = FS.dirname(mvindr_manifest) or dataset_root
  FS.mkdir_p(vindr_prepared_root)
  local used, missing, failed = 0, 0, 0
  for _, row in ipairs(rows) do
    local source = resolve_manifest_path(manifest_root, row.image_path)
    local mask = resolve_manifest_path(manifest_root, row.mask_path)
    if not source or not mask or not FS.file_exists(source) or not FS.file_exists(mask) then
      missing = missing + 1
    else
      local source_name = basename(source)
      local stem = source_name:gsub("%.[^.]+$", "")
      local source_ext = source_name:match("(%.[^.]+)$") or ".png"
      if not mvindr_mask_tags then
        error("MVinDr: --mvindr-mask-tags=false ne fournit qu'une classe constante `mass`, " ..
          "incompatible avec un entraînement multi-label utile. Laisse l'option activée.")
      end
      local tags, analyze_err = analyze_mvindr_mask(mask)
      if not tags then error("MVinDr: " .. tostring(analyze_err)) end
      table.sort(tags)

      local dest_image = FS.join(vindr_prepared_root, stem .. source_ext)
      local mode = vindr_image_mode == "auto" and "symlink" or vindr_image_mode
      -- Mímir canonicalise les symlinks pendant fs::relative(); une image liée
      -- hors du dataset ne partage alors plus la clé de son sidecar `.txt`.
      -- Réparer les sorties créées par les anciennes versions du préparateur.
      if mode ~= "symlink" and is_symlink(dest_image) and not unlink_symlink(dest_image) then
        error("MVinDr: impossible de remplacer l'ancien lien symbolique " .. tostring(dest_image))
      end
      local image_ok = FS.file_exists(dest_image) and not is_symlink(dest_image)
      if not image_ok then
        if mode == "convert" then
          image_ok = run_command("magick " .. FS.quote(source) .. " " .. FS.quote(dest_image))
        elseif mode == "hardlink" and not FS.is_windows() then
          image_ok = run_command("ln " .. FS.quote(source) .. " " .. FS.quote(dest_image) .. " >/dev/null 2>&1")
          if not image_ok then
            -- Les hardlinks ne traversent pas les systèmes de fichiers. Une
            -- copie est le seul format portable que l'indexeur Mímir apparie.
            image_ok = run_command("cp " .. FS.quote(source) .. " " .. FS.quote(dest_image))
          end
        elseif mode == "copy" or FS.is_windows() then
          local cmd = FS.is_windows() and ("copy /Y " .. FS.quote(source) .. " " .. FS.quote(dest_image) .. " >NUL")
            or ("cp " .. FS.quote(source) .. " " .. FS.quote(dest_image))
          image_ok = run_command(cmd)
        else
          image_ok = run_command("ln -s " .. FS.quote(source) .. " " .. FS.quote(dest_image) .. " >/dev/null 2>&1")
        end
      end
      if image_ok and FS.file_exists(dest_image) then
        local text_path = FS.join(vindr_prepared_root, stem .. ".txt")
        local f = io.open(text_path, "w")
        if not f then error("MVinDr: impossible d'écrire " .. tostring(text_path)) end
        f:write(table.concat(tags, ". "), ".\n")
        f:close()
        process_text(table.concat(tags, ". ") .. ".")
        used = used + 1
        if max_files > 0 and used >= max_files then break end
      else
        failed = failed + 1
      end
    end
  end
  log("- mvindr_prepared=" .. tostring(used) .. " missing_pairs=" .. tostring(missing) ..
      " failed_images=" .. tostring(failed))
  if used == 0 then error("MVinDr: aucun échantillon exploitable n'a été préparé") end
  if missing > 0 or failed > 0 then
    error("MVinDr: préparation incomplète (paires manquantes=" .. tostring(missing) ..
      ", images échouées=" .. tostring(failed) .. ")")
  end
  return used
end

local read_ok = 0
local read_fail = 0

if detected_format == "txt" then
  for _, path in ipairs(files) do
    local f = io.open(path, "r")
    if f then
      local content = f:read("*a") or ""
      f:close()
      process_text(content)
      read_ok = read_ok + 1
    else
      read_fail = read_fail + 1
    end
  end
elseif detected_format == "coco" then
  local n = process_coco_annotations(coco_annotations)
  read_ok = n
  read_fail = 0
elseif detected_format == "vindr-mammo" then
  read_ok = process_vindr_mammo()
  read_fail = 0
else
  read_ok = process_mvindr_mass()
  read_fail = 0
end

log("- read_ok=" .. tostring(read_ok) .. " read_fail=" .. tostring(read_fail))
log("- total_tags_seen=" .. tostring(total_tags))

local items = {}
for tag, n in pairs(freq) do
  if n >= min_freq then
    table.insert(items, {
      tag = tag,
      n = n,
      item_n = item_freq[tag] or 0,
    })
  end
end

table.sort(items, function(a, b)
  if a.n ~= b.n then return a.n > b.n end
  return a.tag < b.tag
end)

if top_k > 0 and #items > top_k then
  while #items > top_k do table.remove(items) end
end

ensure_parent_dir(out_path)
local out = io.open(out_path, "w")
if not out then
  error("Impossible d'écrire: " .. tostring(out_path))
end

for _, it in ipairs(items) do
  out:write(it.tag)
  out:write("\n")
end
out:close()

if composition_out ~= "" and composition_out ~= "false" and composition_out ~= "none" then
  ensure_parent_dir(composition_out)
  local comp = io.open(composition_out, "w")
  if not comp then
    error("Impossible d'écrire la composition de classes: " .. tostring(composition_out))
  end

  local kept_total_tags = 0
  for _, it in ipairs(items) do
    kept_total_tags = kept_total_tags + (it.n or 0)
  end

  comp:write("{\n")
  comp:write("  \"dataset_root\": \"" .. json_escape(dataset_root) .. "\",\n")
  comp:write("  \"vocab_path\": \"" .. json_escape(out_path) .. "\",\n")
  comp:write("  \"dataset_format\": \"" .. json_escape(detected_format) .. "\",\n")
  if detected_format == "vindr-mammo" then
    comp:write("  \"prepared_dataset_root\": \"" .. json_escape(vindr_prepared_root) .. "\",\n")
    comp:write("  \"vindr_split\": \"" .. json_escape(vindr_split) .. "\",\n")
    comp:write("  \"vindr_labels\": \"" .. json_escape(vindr_labels) .. "\",\n")
  elseif detected_format == "mvindr-mass" then
    comp:write("  \"prepared_dataset_root\": \"" .. json_escape(vindr_prepared_root) .. "\",\n")
    comp:write("  \"mvindr_manifest\": \"" .. json_escape(mvindr_manifest) .. "\",\n")
    comp:write("  \"mvindr_mask_tags\": " .. tostring(mvindr_mask_tags) .. ",\n")
    comp:write("  \"mvindr_small_max\": " .. string.format("%.12g", mvindr_small_max) .. ",\n")
    comp:write("  \"mvindr_large_min\": " .. string.format("%.12g", mvindr_large_min) .. ",\n")
  end
  comp:write("  \"split_mode\": \"" .. json_escape(split_mode) .. "\",\n")
  comp:write("  \"lowercase\": " .. tostring(lowercase) .. ",\n")
  comp:write("  \"total_samples\": " .. tostring(total_samples) .. ",\n")
  comp:write("  \"total_tags_seen\": " .. tostring(total_tags) .. ",\n")
  comp:write("  \"total_tags_kept\": " .. tostring(kept_total_tags) .. ",\n")
  comp:write("  \"num_classes\": " .. tostring(#items) .. ",\n")
  comp:write("  \"classes\": [\n")
  for i, it in ipairs(items) do
    local pos_items = math.max(0, tonumber(it.item_n) or 0)
    local neg_items = math.max(0, total_samples - pos_items)
    local raw_pos_weight = 1.0
    if pos_items > 0 then
      raw_pos_weight = neg_items / pos_items
    end
    if raw_pos_weight < 1e-6 then raw_pos_weight = 1e-6 end
    local tag_freq = 0.0
    if kept_total_tags > 0 then
      tag_freq = (it.n or 0) / kept_total_tags
    end
    local item_ratio = 0.0
    if total_samples > 0 then
      item_ratio = pos_items / total_samples
    end

    comp:write("    {\n")
    comp:write("      \"tag\": \"" .. json_escape(it.tag) .. "\",\n")
    comp:write("      \"count\": " .. tostring(it.n or 0) .. ",\n")
    comp:write("      \"sample_count\": " .. tostring(pos_items) .. ",\n")
    comp:write("      \"tag_frequency\": " .. string.format("%.12g", tag_freq) .. ",\n")
    comp:write("      \"sample_frequency\": " .. string.format("%.12g", item_ratio) .. ",\n")
    comp:write("      \"recommended_pos_weight\": " .. string.format("%.12g", raw_pos_weight) .. "\n")
    if i < #items then
      comp:write("    },\n")
    else
      comp:write("    }\n")
    end
  end
  comp:write("  ]\n")
  comp:write("}\n")
  comp:close()
  log("✓ composition de classes écrite: " .. tostring(composition_out))
end

log("✓ tags_vocab écrit: " .. tostring(out_path) .. " (classes=" .. tostring(#items) .. ")")
