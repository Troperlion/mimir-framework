#!/usr/bin/env lua
---@diagnostic disable: undefined-global, undefined-field, need-check-nil

local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help({
  script_path = "scripts/tools/compose_medical_detector_dataset.lua",
  description = "Compose et valide un dataset image + sidecar JSON pour medical_detector.",
  options = {
    "--dataset-root PATH : racine avec images/, masks/ et les deux CSV",
    "--images-root PATH : dossier contenant les images sources (alias: --image-root)",
    "--annotations PATH : manifeste JSON; sinon utilise les sidecars sources",
    "--findings-csv PATH : finding_annotations.csv (auto avec --dataset-root)",
    "--manifest-csv PATH : manifest.csv (auto avec --dataset-root)",
    "--masks-root PATH : dossier masks/ utilisé si les coordonnées CSV manquent",
    "--normal-images-root PATH : images normales ajoutées automatiquement avec boxes=[]",
    "--allow-cross-modality-negatives : autorise explicitement des négatifs d'une autre modalité",
    "--mask-scan-size N : résolution de lecture des masques (512)",
    "--out-dir PATH : dossier du dataset composé (requis)",
    "--class-names LISTE : classes séparées par des virgules (taxonomie mammaire à 7 classes)",
    "--unannotated-as-negative : crée boxes=[] pour les images sans annotation",
    "--overwrite : autorise le remplacement des fichiers de sortie existants",
    "--dry-run : valide et affiche le bilan sans écrire de fichier",
  },
  examples = {
    "./bin/mimir --lua scripts/tools/compose_medical_detector_dataset.lua -- --dataset-root donnees/mvindr --out-dir dataset/mammographies",
    "./bin/mimir --lua scripts/tools/compose_medical_detector_dataset.lua -- --images-root donnees/images --annotations donnees/annotations.json --out-dir dataset/mammographies",
    "./bin/mimir --lua scripts/tools/compose_medical_detector_dataset.lua -- --images-root donnees/deja_annotees --out-dir dataset/mammographies",
  },
})

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local opts = Args.parse(arg) or {}

local function die(message) error("[compose_medical_dataset] "..tostring(message), 0) end
local function normalize(path)
  return FS.normalize(tostring(path or "")):gsub("\\", "/"):gsub("^%./", ""):gsub("/+", "/")
end
local function split_names(value)
  local names = {}
  for name in tostring(value):gmatch("[^,]+") do
    name = name:match("^%s*(.-)%s*$")
    if name ~= "" then names[#names + 1] = name end
  end
  return names
end
local function trim(value)
  return tostring(value or ""):gsub("^%s+", ""):gsub("%s+$", "")
end
local DEFAULT_CLASS_NAMES = {
  "masse",
  "calcification",
  "asymétrie",
  "distorsion architecturale",
  "ganglion lymphatique",
  "épaississement cutané",
  "rétraction du mamelon",
}
local DEFAULT_CLASS_NAMES_ARG = table.concat(DEFAULT_CLASS_NAMES, ",")
local FINDING_ALIASES = {
  ["mass"] = "masse",
  ["masse"] = "masse",
  ["suspicious calcification"] = "calcification",
  ["calcification"] = "calcification",
  ["asymmetry"] = "asymétrie",
  ["asymétrie"] = "asymétrie",
  ["focal asymmetry"] = "asymétrie",
  ["global asymmetry"] = "asymétrie",
  ["architectural distortion"] = "distorsion architecturale",
  ["distorsion architecturale"] = "distorsion architecturale",
  ["suspicious lymph node"] = "ganglion lymphatique",
  ["lymph node"] = "ganglion lymphatique",
  ["ganglion lymphatique"] = "ganglion lymphatique",
  ["skin thickening"] = "épaississement cutané",
  ["épaississement cutané"] = "épaississement cutané",
  ["nipple retraction"] = "rétraction du mamelon",
  ["rétraction du mamelon"] = "rétraction du mamelon",
}
local function normalize_finding(value)
  return trim(value):lower():gsub("_", " "):gsub("[%[%]'\"]", ""):gsub("%s+", " ")
end
local function finding_class_id(category, class_names)
  if #class_names == 1 then return 0 end
  local normalized = normalize_finding(category)
  local canonical = FINDING_ALIASES[normalized]
  if not canonical then
    for alias, mapped in pairs(FINDING_ALIASES) do
      if normalized:find(alias, 1, true) then
        canonical = mapped
        break
      end
    end
  end
  if not canonical then return nil end
  for index, name in ipairs(class_names) do
    local normalized_name = normalize_finding(name)
    if name == canonical or FINDING_ALIASES[normalized_name] == canonical then
      return index - 1
    end
  end
  return nil
end
local function parse_csv_row(line)
  local row, field, quoted = {}, {}, false
  local index = 1
  while index <= #line do
    local character = line:sub(index, index)
    if quoted then
      if character == '"' then
        if line:sub(index + 1, index + 1) == '"' then
          field[#field + 1] = '"'
          index = index + 1
        else
          quoted = false
        end
      else
        field[#field + 1] = character
      end
    elseif character == '"' and #field == 0 then
      quoted = true
    elseif character == "," then
      row[#row + 1] = table.concat(field)
      field = {}
    else
      field[#field + 1] = character
    end
    index = index + 1
  end
  if quoted then return nil, "guillemet CSV non fermé" end
  row[#row + 1] = table.concat(field)
  return row
end
local function read_csv(path)
  local file, open_err = io.open(path, "r")
  if not file then return nil, open_err end
  local header_line = file:read("*l")
  if not header_line then file:close(); return nil, "CSV vide" end
  header_line = header_line:gsub("^\239\187\191", "")
  local headers, header_err = parse_csv_row(header_line)
  if not headers then file:close(); return nil, header_err end
  local rows = {}
  for line in file:lines() do
    if trim(line) ~= "" then
      local values, row_err = parse_csv_row(line)
      if not values then file:close(); return nil, row_err end
      local row = {}
      for index, header in ipairs(headers) do
        row[trim(header):lower()] = trim(values[index])
      end
      rows[#rows + 1] = row
    end
  end
  file:close()
  return rows
end
local function first_value(row, names)
  for _, name in ipairs(names) do
    local value = trim(row[name])
    if value ~= "" and value:lower() ~= "nan" and value:lower() ~= "null" then return value end
  end
  return nil
end
local function image_extension(path)
  return path:lower():match("%.([^./]+)$")
end
local supported_extensions = {png=true, jpg=true, jpeg=true, bmp=true, ppm=true, pgm=true}
local function is_image(path)
  return supported_extensions[image_extension(path) or ""] == true
end
local function stem_path(path)
  return (path:gsub("%.[^./]+$", ""))
end
local function copy_file(source, destination)
  local input, input_err = io.open(source, "rb")
  if not input then return false, input_err end
  local output, output_err = io.open(destination, "wb")
  if not output then input:close(); return false, output_err end
  while true do
    local chunk = input:read(1024 * 1024)
    if not chunk then break end
    local ok, write_err = output:write(chunk)
    if not ok then
      input:close(); output:close()
      return false, write_err
    end
  end
  input:close()
  output:close()
  return true
end
local function collect_images(root, relative, output)
  local directory = relative == "" and root or FS.join(root, relative)
  for _, name in ipairs(FS.list_dir(directory)) do
    if name ~= "." and name ~= ".." and name:sub(1, 1) ~= "." then
      local child_relative = relative == "" and name or FS.join(relative, name)
      local child = FS.join(root, child_relative)
      if FS.is_dir(child) then
        collect_images(root, child_relative, output)
      elseif is_image(name) then
        output[#output + 1] = normalize(child_relative)
      end
    end
  end
end
local function validate_box(box, normalized, class_count, image_name, box_index)
  if type(box) ~= "table" then
    die(string.format("%s: boîte %d invalide", image_name, box_index))
  end
  local x1, y1, x2, y2
  if box.x1 ~= nil and box.y1 ~= nil and box.x2 ~= nil and box.y2 ~= nil then
    x1, y1, x2, y2 = tonumber(box.x1), tonumber(box.y1), tonumber(box.x2), tonumber(box.y2)
  else
    local x, y, width, height = tonumber(box.x), tonumber(box.y), tonumber(box.w), tonumber(box.h)
    if x and y and width and height then x1, y1, x2, y2 = x, y, x + width, y + height end
  end
  if not x1 or not y1 or not x2 or not y2 or x2 <= x1 or y2 <= y1 then
    die(string.format("%s: coordonnées invalides pour la boîte %d", image_name, box_index))
  end
  if normalized and (x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1) then
    die(string.format("%s: boîte normalisée %d hors de [0,1]", image_name, box_index))
  end
  local class_id = tonumber(box.class_id ~= nil and box.class_id or box.class) or 0
  if class_id ~= math.floor(class_id) or class_id < 0 or class_id >= class_count then
    die(string.format("%s: class_id invalide pour la boîte %d", image_name, box_index))
  end
  return {
    x1=x1, y1=y1, x2=x2, y2=y2,
    class_id=class_id,
    label=type(box.label) == "string" and box.label or nil,
  }
end
local function validate_annotation(entry, class_names, image_name)
  if type(entry) ~= "table" then die(image_name..": annotation invalide") end
  local boxes = entry.boxes
  if entry.negative == true then
    if type(boxes) == "table" and #boxes > 0 then
      die(image_name..": negative=true est incompatible avec des boîtes")
    end
    boxes = {}
  end
  if type(boxes) ~= "table" then die(image_name..": champ boxes requis (ou negative=true)") end
  local normalized = entry.normalized == true
  local width, height = tonumber(entry.width), tonumber(entry.height)
  if not normalized and #boxes > 0 and
      (not width or not height or width <= 0 or height <= 0) then
    die(image_name..": width et height sont requis pour des coordonnées pixels")
  end
  local validated = {}
  for index, box in ipairs(boxes) do
    local value = validate_box(box, normalized, #class_names, image_name, index)
    value.label = value.label or class_names[value.class_id + 1]
    validated[#validated + 1] = value
  end
  return {
    width=width,
    height=height,
    normalized=normalized,
    boxes=validated,
  }
end
local function source_sidecar(images_root, relative_image)
  local base = FS.join(images_root, stem_path(relative_image))
  for _, suffix in ipairs({".json", ".boxes.json"}) do
    local candidate = base..suffix
    if FS.file_exists(candidate) then return candidate end
  end
  return nil
end

local dataset_root = Args.get_str(opts, "dataset-root", nil)
local images_root = Args.get_str(opts, "images-root", Args.get_str(opts, "image-root",
  dataset_root and FS.join(dataset_root, "images") or nil))
local annotations_path = Args.get_str(opts, "annotations", nil)
local findings_csv = Args.get_str(opts, "findings-csv",
  dataset_root and FS.join(dataset_root, "finding_annotations.csv") or nil)
local manifest_csv = Args.get_str(opts, "manifest-csv",
  dataset_root and FS.join(dataset_root, "manifest.csv") or nil)
local masks_root = Args.get_str(opts, "masks-root",
  dataset_root and FS.join(dataset_root, "masks") or nil)
local normal_images_root = Args.get_str(opts, "normal-images-root",
  Args.get_str(opts, "negative-images-root", nil))
if not normal_images_root and dataset_root then
  for _, directory in ipairs({"normal_images", "negative_images", "normals", "negatives"}) do
    local candidate = FS.join(dataset_root, directory)
    if FS.is_dir(candidate) then
      normal_images_root = candidate
      break
    end
  end
end
if normal_images_root then
  for _, relative in ipairs({FS.join("Dataset_BUSI_with_GT", "normal"), "normal"}) do
    local candidate = FS.join(normal_images_root, relative)
    if FS.is_dir(candidate) then
      normal_images_root = candidate
      break
    end
  end
end
local mask_scan_size = Args.get_int(opts, "mask-scan-size", 512)
local out_dir = Args.get_str(opts, "out-dir", nil)
local class_names = split_names(Args.get_str(opts, "class-names", DEFAULT_CLASS_NAMES_ARG))
local unannotated_as_negative = Args.get_bool(opts, "unannotated-as-negative", false)
local allow_cross_modality_negatives = Args.get_bool(opts, "allow-cross-modality-negatives", false)
local overwrite = Args.get_bool(opts, "overwrite", false)
local dry_run = Args.get_bool(opts, "dry-run", false)
if not images_root or images_root == "" then die("--images-root est requis") end
if not out_dir or out_dir == "" then die("--out-dir est requis") end
if not FS.is_dir(images_root) then die("dossier source introuvable: "..images_root) end
if #class_names == 0 then die("--class-names doit contenir au moins une classe") end
if mask_scan_size < 16 then die("--mask-scan-size doit être >= 16") end
if normalize(images_root) == normalize(out_dir) then die("--out-dir doit être distinct de --images-root") end
if normal_images_root and not FS.is_dir(normal_images_root) then
  die("dossier d'images normales introuvable: "..normal_images_root)
end
if normal_images_root and normalize(normal_images_root) == normalize(images_root) then
  die("--normal-images-root doit être distinct de --images-root")
end
if normal_images_root and normalize(normal_images_root):lower():find("dataset_busi", 1, true) and
    not normalize(images_root):lower():find("dataset_busi", 1, true) and
    not allow_cross_modality_negatives then
  die("Dataset_BUSI contient des échographies, pas des mammographies; utilisez "..
    "--allow-cross-modality-negatives pour confirmer explicitement ce mélange")
end
local csv_mode = (findings_csv and FS.file_exists(findings_csv)) or
  (manifest_csv and FS.file_exists(manifest_csv))
if annotations_path and annotations_path ~= "" and csv_mode then
  die("--annotations JSON ne peut pas être combiné aux CSV")
end
if dataset_root and not csv_mode then
  die("aucun finding_annotations.csv ou manifest.csv trouvé sous "..dataset_root)
end
if masks_root and masks_root ~= "" and not FS.is_dir(masks_root) then
  io.stderr:write("Avertissement: dossier masks absent: "..masks_root.."\n")
end

local manifest_by_image = {}
if annotations_path and annotations_path ~= "" then
  local manifest, manifest_err = read_json(annotations_path)
  if type(manifest) ~= "table" then die("lecture du manifeste: "..tostring(manifest_err)) end
  if type(manifest.classes) == "table" then
    class_names = manifest.classes
    if #class_names == 0 then die("manifest.classes est vide") end
  end
  if type(manifest.images) ~= "table" then die("le manifeste doit contenir images: []") end
  for index, entry in ipairs(manifest.images) do
    if type(entry) ~= "table" or type(entry.file) ~= "string" or entry.file == "" then
      die(string.format("manifest.images[%d].file est requis", index))
    end
    local key = normalize(entry.file)
    if key:match("^/") or key:match("^%.%./") or key:find("/../", 1, true) then
      die("chemin d'image hors de --images-root: "..key)
    end
    if manifest_by_image[key] then die("annotation dupliquée: "..key) end
    manifest_by_image[key] = entry
  end
end

local images = {}
collect_images(images_root, "", images)
table.sort(images)
if #images == 0 then die("aucune image supportée dans "..images_root) end

if csv_mode then
  local function basename(path)
    return normalize(path):match("([^/]+)$") or normalize(path)
  end
  local function stem(value)
    return basename(value):gsub("%.[^.]+$", "")
  end
  local image_lookup = {}
  local normalized_images_root = normalize(images_root):gsub("/$", "")
  local function add_lookup(key, relative)
    key = normalize(key):lower()
    if key ~= "" and image_lookup[key] == nil then image_lookup[key] = relative end
  end
  for _, relative in ipairs(images) do
    add_lookup(relative, relative)
    add_lookup(basename(relative), relative)
    local image_stem = stem(relative)
    add_lookup(image_stem, relative)
    add_lookup(image_stem:match("_([^_]+)$"), relative)
  end
  local function resolve_image(value)
    if not value then return nil end
    local candidate = normalize(value)
    if candidate:sub(1, #normalized_images_root + 1) == normalized_images_root.."/" then
      candidate = candidate:sub(#normalized_images_root + 2)
    end
    candidate = candidate:gsub("^images/", "")
    return image_lookup[candidate:lower()] or
      image_lookup[basename(candidate):lower()] or
      image_lookup[stem(candidate):lower()]
  end
  local function row_number(row, names)
    return tonumber(first_value(row, names))
  end
  local function row_dimensions(row)
    return row_number(row, {"width", "image_width", "img_width", "original_width"}),
      row_number(row, {"height", "image_height", "img_height", "original_height"})
  end
  local image_by_id = {}
  local masks_by_image = {}
  local manifest_root = manifest_csv and FS.dirname(manifest_csv) or dataset_root or images_root
  local function resolve_mask(value)
    if not value then return nil end
    local path = normalize(value)
    if path:match("^/") and FS.file_exists(path) then return path end
    local relative = FS.join(manifest_root, path)
    if FS.file_exists(relative) then return relative end
    if masks_root and masks_root ~= "" then
      local under_masks = FS.join(masks_root, basename(path))
      if FS.file_exists(under_masks) then return under_masks end
    end
    return nil
  end
  local unresolved_manifest = 0
  if manifest_csv and FS.file_exists(manifest_csv) then
    local rows, csv_err = read_csv(manifest_csv)
    if not rows then die("lecture manifest.csv: "..tostring(csv_err)) end
    ---@cast rows table[]
    for _, row in ipairs(rows) do
      local path = first_value(row,
        {"image_path", "file_path", "filepath", "file_name", "filename", "image", "path"})
      local image_id = first_value(row, {"image_id", "imageid", "id"})
      local relative = resolve_image(path or image_id)
      if relative then
        if image_id then image_by_id[image_id:lower()] = relative end
        local path_stem = stem(path or relative)
        image_by_id[path_stem:lower()] = relative
        local suffix_id = path_stem:match("_([^_]+)$")
        if suffix_id then image_by_id[suffix_id:lower()] = relative end
        local width, height = row_dimensions(row)
        local annotation = manifest_by_image[relative] or {
          width=width, height=height, normalized=false, boxes={}, negative=true,
        }
        annotation.width = annotation.width or width
        annotation.height = annotation.height or height
        manifest_by_image[relative] = annotation
        local mask_value = first_value(row,
          {"mask_path", "mask_file", "mask", "segmentation_path", "annotation_path"})
        local mask_path = resolve_mask(mask_value)
        if mask_value and not mask_path then
          die("masque référencé mais introuvable: "..mask_value)
        end
        if mask_path then
          local bucket = masks_by_image[relative] or {}
          bucket[#bucket + 1] = mask_path
          masks_by_image[relative] = bucket
        end
      else
        unresolved_manifest = unresolved_manifest + 1
      end
    end
  end
  if unresolved_manifest > 0 then
    die(string.format("manifest.csv: %d image(s) introuvable(s) sous images/", unresolved_manifest))
  end

  local finding_rows = 0
  local coordinate_rows = 0
  local no_finding_rows = 0
  local unsupported_finding_rows = 0
  local unresolved_findings = 0
  local missing_dimensions = 0
  if findings_csv and FS.file_exists(findings_csv) then
    local rows, csv_err = read_csv(findings_csv)
    if not rows then die("lecture finding_annotations.csv: "..tostring(csv_err)) end
    ---@cast rows table[]
    for _, row in ipairs(rows) do
      finding_rows = finding_rows + 1
      local image_id = first_value(row, {"image_id", "imageid", "id"})
      local path = first_value(row,
        {"image_path", "file_path", "filepath", "file_name", "filename", "image", "path"})
      local relative = image_id and image_by_id[image_id:lower()] or nil
      relative = relative or resolve_image(path or image_id)
      if not relative then
        unresolved_findings = unresolved_findings + 1
      else
        local width, height = row_dimensions(row)
        local annotation = manifest_by_image[relative] or {
          width=width, height=height, normalized=false, boxes={}, negative=true,
        }
        annotation.width = annotation.width or width
        annotation.height = annotation.height or height
        local category = first_value(row,
          {"finding_categories", "finding_category", "category", "finding", "label"})
        local category_lower = category and category:lower() or ""
        local x1 = row_number(row, {"xmin", "x_min", "x1", "left"})
        local y1 = row_number(row, {"ymin", "y_min", "y1", "top"})
        local x2 = row_number(row, {"xmax", "x_max", "x2", "right"})
        local y2 = row_number(row, {"ymax", "y_max", "y2", "bottom"})
        local explicitly_negative = category_lower == "no finding" or
          category_lower == "no_finding" or category_lower == "normal"
        if x1 and y1 and x2 and y2 and x2 > x1 and y2 > y1 and not explicitly_negative then
          local class_id = finding_class_id(category, class_names)
          if class_id == nil then
            annotation.has_unsupported_finding = true
            unsupported_finding_rows = unsupported_finding_rows + 1
          elseif not annotation.width or not annotation.height then
            missing_dimensions = missing_dimensions + 1
          else
            annotation.negative = false
            annotation.boxes[#annotation.boxes + 1] = {
              x1=x1, y1=y1, x2=x2, y2=y2,
              class_id=class_id, label=class_names[class_id + 1],
            }
            coordinate_rows = coordinate_rows + 1
          end
        else
          no_finding_rows = no_finding_rows + 1
        end
        manifest_by_image[relative] = annotation
      end
    end
  end
  if missing_dimensions > 0 then
    die(string.format("finding_annotations.csv: %d boîte(s) sans width/height", missing_dimensions))
  end
  local mask_boxes = 0
  local empty_masks = 0
  for relative, mask_paths in pairs(masks_by_image) do
    local annotation = manifest_by_image[relative]
    if annotation and #annotation.boxes == 0 then
      for _, mask_path in ipairs(mask_paths) do
        local loaded, load_err = Mimir.IO.read_image_rgb_u8(
          mask_path, mask_scan_size, mask_scan_size)
        if type(loaded) ~= "table" or type(loaded.image) ~= "table" then
          die("lecture du masque "..mask_path..": "..tostring(load_err))
        end
        local min_x, min_y = mask_scan_size, mask_scan_size
        local max_x, max_y = -1, -1
        for y = 0, mask_scan_size - 1 do
          for x = 0, mask_scan_size - 1 do
            local offset = (y * mask_scan_size + x) * 3
            if (loaded.image[offset + 1] or 0) > 0 or
                (loaded.image[offset + 2] or 0) > 0 or
                (loaded.image[offset + 3] or 0) > 0 then
              min_x, min_y = math.min(min_x, x), math.min(min_y, y)
              max_x, max_y = math.max(max_x, x), math.max(max_y, y)
            end
          end
        end
        if max_x >= min_x and max_y >= min_y then
          annotation.normalized = true
          annotation.negative = false
          annotation.boxes[#annotation.boxes + 1] = {
            x1=min_x / mask_scan_size,
            y1=min_y / mask_scan_size,
            x2=(max_x + 1) / mask_scan_size,
            y2=(max_y + 1) / mask_scan_size,
            class_id=0,
            label=class_names[1],
          }
          mask_boxes = mask_boxes + 1
        else
          empty_masks = empty_masks + 1
        end
      end
    end
  end
  local excluded_unsupported_images = 0
  for relative, annotation in pairs(manifest_by_image) do
    if annotation.has_unsupported_finding and #annotation.boxes == 0 then
      manifest_by_image[relative] = nil
      excluded_unsupported_images = excluded_unsupported_images + 1
    end
  end
  print(string.format(
    "CSV: lignes=%d boîtes_csv=%d boîtes_masques=%d masques_vides=%d sans_boîte=%d hors_taxonomie=%d images_exclues=%d hors_sous-ensemble=%d images_manifestées=%d",
    finding_rows, coordinate_rows, mask_boxes, empty_masks, no_finding_rows,
    unsupported_finding_rows, excluded_unsupported_images, unresolved_findings,
    (function() local count=0; for _ in pairs(manifest_by_image) do count=count+1 end; return count end)()))
end

local records = {}
local output_sidecars = {}
local skipped, positives, negatives, box_count = 0, 0, 0, 0
local class_counts = {}
for index = 1, #class_names do class_counts[index] = 0 end
for _, relative_image in ipairs(images) do
  local annotation = manifest_by_image[relative_image]
  if annotation then manifest_by_image[relative_image] = nil end
  if not annotation and not annotations_path and not csv_mode then
    local sidecar = source_sidecar(images_root, relative_image)
    if sidecar then
      local sidecar_err
      annotation, sidecar_err = read_json(sidecar)
      if type(annotation) ~= "table" then die(sidecar..": "..tostring(sidecar_err)) end
    end
  end
  if not annotation and unannotated_as_negative then annotation = {negative=true, boxes={}} end
  if not annotation then
    skipped = skipped + 1
  else
    local validated = validate_annotation(annotation, class_names, relative_image)
    local sidecar_relative = stem_path(relative_image)..".json"
    if output_sidecars[sidecar_relative] then
      die("collision de sidecars (extensions d'images différentes): "..sidecar_relative)
    end
    output_sidecars[sidecar_relative] = true
    records[#records + 1] = {
      image=relative_image,
      source=FS.join(images_root, relative_image),
      sidecar=sidecar_relative,
      annotation=validated,
    }
    box_count = box_count + #validated.boxes
    for _, box in ipairs(validated.boxes) do
      class_counts[box.class_id + 1] = class_counts[box.class_id + 1] + 1
    end
    if #validated.boxes > 0 then positives = positives + 1 else negatives = negatives + 1 end
  end
end

local imported_normals = 0
local excluded_normal_masks = 0
if normal_images_root then
  local normal_images = {}
  collect_images(normal_images_root, "", normal_images)
  table.sort(normal_images)
  for _, relative_image in ipairs(normal_images) do
    local relative_stem = stem_path(relative_image):lower()
    if relative_stem:match("_mask$") or relative_stem:match("_mask_%d+$") then
      excluded_normal_masks = excluded_normal_masks + 1
    else
      local target_image = normalize(FS.join("normals", relative_image))
      local sidecar_relative = stem_path(target_image)..".json"
      if output_sidecars[sidecar_relative] then
        die("collision de sidecars d'images normales: "..sidecar_relative)
      end
      output_sidecars[sidecar_relative] = true
      records[#records + 1] = {
        image=target_image,
        source=FS.join(normal_images_root, relative_image),
        sidecar=sidecar_relative,
        annotation={normalized=false, boxes={}},
      }
      negatives = negatives + 1
      imported_normals = imported_normals + 1
    end
  end
  print(string.format("Images normales importées automatiquement=%d masques_exclus=%d depuis %s",
    imported_normals, excluded_normal_masks, normal_images_root))
end

local orphan_count = 0
for image_name in pairs(manifest_by_image) do
  orphan_count = orphan_count + 1
  io.stderr:write("Annotation sans image: "..image_name.."\n")
end
if orphan_count > 0 then die(string.format("%d annotation(s) sans image correspondante", orphan_count)) end
if #records == 0 then die("aucune paire image + annotation à composer") end

print(string.format("Images trouvées=%d composées=%d positives=%d négatives=%d ignorées=%d boîtes=%d",
  #images, #records, positives, negatives, skipped, box_count))
for index, name in ipairs(class_names) do
  print(string.format("  classe[%d] %s: %d boîte(s)", index - 1, name, class_counts[index]))
end
if positives == 0 then die("aucune image positive avec boîte valide") end
if negatives == 0 then
  io.stderr:write("Avertissement: aucun examen négatif. Ajoutez des images normales avec boxes: [] "..
    "pour apprendre l'absence d'anomalie et limiter les faux positifs.\n")
end
if dry_run then
  print("Dry-run terminé: aucun fichier écrit")
  return
end

FS.mkdir_p(out_dir)
for _, record in ipairs(records) do
  local source_image = record.source
  local target_image = FS.join(out_dir, record.image)
  local target_sidecar = FS.join(out_dir, record.sidecar)
  if not overwrite and (FS.file_exists(target_image) or FS.file_exists(target_sidecar)) then
    die("sortie existante; utiliser --overwrite: "..record.image)
  end
  local parent = FS.dirname(target_image)
  if parent and parent ~= "" then FS.mkdir_p(parent) end
  local ok_copy, copy_err = copy_file(source_image, target_image)
  if not ok_copy then die("copie de "..record.image..": "..tostring(copy_err)) end
  local ok_json, json_err = write_json(target_sidecar, record.annotation)
  if not ok_json then die("écriture de "..record.sidecar..": "..tostring(json_err)) end
end

local summary_path = FS.join(out_dir, "composition.mimir-meta")
local ok_summary, summary_err = write_json(summary_path, {
  type="medical_detector_dataset",
  source=images_root,
  normal_source=normal_images_root,
  class_names=class_names,
  class_counts=class_counts,
  images=#records,
  positives=positives,
  negatives=negatives,
  imported_normals=imported_normals,
  excluded_normal_masks=excluded_normal_masks,
  boxes=box_count,
  skipped_unannotated=skipped,
})
if not ok_summary then die("écriture du bilan: "..tostring(summary_err)) end

local ok_dataset, count_or_err = Mimir.Dataset.load(out_dir, 64, 64, 1, false, "", 256, true)
if not ok_dataset then die("validation Mimir.Dataset.load: "..tostring(count_or_err)) end
local loaded_images = 0
for index = 1, tonumber(count_or_err) or 0 do
  local item = Mimir.Dataset.get(index)
  if type(item) == "table" and type(item.image_file) == "string" and item.image_file ~= "" then
    loaded_images = loaded_images + 1
  end
end
if loaded_images ~= #records then
  die(string.format("validation dataset: Mímir charge %d images, %d attendues", loaded_images, #records))
end
print(string.format("Dataset composé et validé: %s (%d images)", out_dir, #records))
