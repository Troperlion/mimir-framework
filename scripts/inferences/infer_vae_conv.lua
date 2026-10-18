#!/usr/bin/env lua
-- Encodage/décodage d'images avec un checkpoint VAEConv natif Mímir.
--
-- PPM -> latent float32 brut :
--   ./bin/mimir --lua scripts/inferences/infer_vae_conv.lua -- \
--     --encode --checkpoint CHECKPOINT --input image.ppm --output image.raw
--
-- latent float32 brut -> PPM :
--   ./bin/mimir --lua scripts/inferences/infer_vae_conv.lua -- \
--     --decode --checkpoint CHECKPOINT --input image.raw --output image_decoded.ppm

---@diagnostic disable: undefined-global, undefined-field

local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help({
  script_path = "scripts/inferences/infer_vae_conv.lua",
  description = "VAEConv natif : image PPM vers moyenne latente, ou latent RAW vers image PPM.",
  options = {
    "--encode / --decode : mode requis, mutuellement exclusifs",
    "--checkpoint, --ckpt PATH : checkpoint ou dossier contenant epoch_*",
    "--input, --in PATH : image PPM P6/P3 en encodage ; RAW en décodage",
    "--output, --out PATH : RAW en encodage ; PPM en décodage",
    "--format FORMAT : raw_folder ou safetensors (détecté par extension)",
    "--no-resize : refuse une image dont les dimensions diffèrent du checkpoint",
    "--mem-gb N : budget allocateur (8 Go)",
    "--dtype TYPE : float32, float16 ou bfloat16",
    "RAW : float32 little-endian sans en-tête, ordre latent CHW du modèle.",
    "SafeTensors : fournir la configuration modèle correspondante via --config.",
  },
  examples = {
    "./bin/mimir --lua scripts/inferences/infer_vae_conv.lua -- --encode --checkpoint checkpoint/vae --input image.ppm --output outputs/image.raw",
    "./bin/mimir --lua scripts/inferences/infer_vae_conv.lua -- --decode --checkpoint checkpoint/vae --input outputs/image.raw --output outputs/decoded.ppm",
  },
})
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
local opts = Args.parse(arg) or {}

local function logx(message)
  if type(log) == "function" then log(message) else print(message) end
end
local function die(message) error("[infer_vae_conv] " .. tostring(message or "erreur inconnue")) end
local function opt_str(name, default)
  local value = opts[name]
  if value == nil or value == true then return default end
  return tostring(value)
end
local function opt_num(name, default) return tonumber(opts[name]) or default end
local function opt_int(name, default) return math.floor(opt_num(name, default)) end
local function clamp(value, low, high)
  if value < low then return low end
  if value > high then return high end
  return value
end

local mode = nil
if opts.encode == true then mode = "encode" end
if opts.decode == true then
  if mode ~= nil then die("--encode et --decode sont mutuellement exclusifs") end
  mode = "decode"
end
if mode == nil then die("mode requis: --encode ou --decode") end

local conf_model = type(CONF) == "table" and type(CONF.model) == "table" and CONF.model or {}
local conf_training = type(CONF) == "table" and type(CONF.training) == "table" and CONF.training or {}
local conf_inference = type(CONF) == "table" and type(CONF.inference) == "table" and CONF.inference or {}
local input_path = opt_str("input", opt_str("in", nil))
local output_path = opt_str("output", opt_str("out", nil))
if input_path == nil or input_path == "" then die("--input est requis") end
if output_path == nil or output_path == "" then die("--output est requis") end
if mode == "encode" and not output_path:lower():match("%.raw$") then
  die("la sortie d'encodage doit porter l'extension .raw")
end
if mode == "decode" and not input_path:lower():match("%.raw$") then
  die("l'entrée de décodage doit porter l'extension .raw")
end

local checkpoint = opt_str("checkpoint", opt_str("ckpt",
  conf_inference.checkpoint or conf_training.out_dir or "checkpoint/vae_conv-generique.v8"))
local format = opt_str("format", "")
if format == "" then format = checkpoint:lower():match("%.safetensors$") and "safetensors" or "raw_folder" end
if format == "raw_folder" then
  checkpoint = Ckpt.resolve_dir(checkpoint)
  if checkpoint == nil then die("aucun checkpoint raw_folder chargeable") end
elseif format ~= "safetensors" then
  die("format non supporté: " .. tostring(format))
end

local architecture_name = mode == "encode" and "vae_conv" or "vae_conv_decode"
local cfg, cfg_err = Mimir.Architectures.default_config(architecture_name)
if type(cfg) ~= "table" then die("default_config(" .. architecture_name .. "): " .. tostring(cfg_err)) end

for key, value in pairs(conf_model) do if key ~= "architecture" then cfg[key] = value end end
local checkpoint_cfg = nil
-- Le checkpoint est la source de vérité de la topologie.
if format == "raw_folder" then
  local architecture_path = FS.join(checkpoint, "model", "architecture.json")
  local architecture = nil
  if type(read_json) == "function" then
    local ok_json, value = pcall(read_json, architecture_path)
    if ok_json and type(value) == "table" then architecture = value end
  end
  checkpoint_cfg = type(architecture) == "table"
      and (architecture.model_config or architecture.modelConfig) or nil
  if type(checkpoint_cfg) ~= "table" then die("model_config absent ou illisible dans " .. architecture_path) end
  if checkpoint_cfg.type ~= "vae_conv" then die("checkpoint attendu: vae_conv") end
  for key, value in pairs(checkpoint_cfg) do cfg[key] = value end
  -- A new launch config must not enable latent branches absent from a legacy checkpoint.
  cfg.latent_spatial_context = checkpoint_cfg.latent_spatial_context == true
  cfg.latent_detail_path = checkpoint_cfg.latent_detail_path == true
  cfg.latent_context_channels = checkpoint_cfg.latent_context_channels or 32
  if type(architecture.layers) == "table" then
    cfg.dec_norm = "none"
    for _, layer in ipairs(architecture.layers) do
      if tostring(layer.name):match("^vae_conv/dec/") then
        local kind = tostring(layer.type):lower()
        if kind == "groupnorm" then cfg.dec_norm = "groupnorm"; break end
        if kind == "layernorm" then cfg.dec_norm = "layernorm" end
      end
    end
  end
end

local integer_overrides = {
  { "image-w", "image_w" }, { "image-h", "image_h" }, { "image-c", "image_c" },
  { "latent-w", "latent_w" }, { "latent-h", "latent_h" }, { "latent-c", "latent_c" },
  { "base-channels", "base_channels" },
}
for _, names in ipairs(integer_overrides) do
  if opts[names[1]] ~= nil then
    local value = opt_int(names[1], cfg[names[2]])
    if checkpoint_cfg and value ~= cfg[names[2]] then
      die("configuration incompatible avec le checkpoint: --" .. names[1])
    end
    cfg[names[2]] = value
  end
end
for _, key in ipairs({"image_w", "image_h", "image_c", "latent_w", "latent_h", "latent_c"}) do
  local value = tonumber(cfg[key])
  if not value or value <= 0 or value ~= math.floor(value) then die("dimension invalide: " .. key) end
end
cfg.latent_dim = cfg.latent_w * cfg.latent_h * cfg.latent_c
cfg.stochastic_latent = false -- encodage déterministe : z = mu en inférence
if tonumber(cfg.image_c) ~= 3 then die("le format PPM exige image_c=3") end

local dtype = opt_str("dtype", cfg.dtype or os.getenv("MIMIR_DTYPE"))
local dtype_aliases = { F32 = "float32", F16 = "float16", BF16 = "bfloat16", FP32 = "float32", FP16 = "float16" }
if dtype ~= nil then dtype = dtype_aliases[tostring(dtype):upper()] or dtype end
cfg.dtype = dtype

local mem_gb = opt_num("mem-gb", opt_num("alloc-gb", 8))
if Mimir.Allocator and Mimir.Allocator.configure then
  local ok_mem, mem_err = Mimir.Allocator.configure({ max_ram_gb = mem_gb, enable_compression = true })
  if ok_mem == false then die("Allocator.configure: " .. tostring(mem_err)) end
end

local function create_and_load()
  local ok_create, create_err = Mimir.Model.create(architecture_name, cfg)
  if ok_create == false then die("Model.create: " .. tostring(create_err)) end
  if dtype and Mimir.Model and type(Mimir.Model.dtype) == "function" then
    local ok_dtype, dtype_err = Mimir.Model.dtype(dtype)
    if ok_dtype == false then die("Model.dtype: " .. tostring(dtype_err)) end
  end
  local ok_alloc, alloc_err = Mimir.Model.allocate_params()
  if ok_alloc == false then die("Model.allocate_params: " .. tostring(alloc_err)) end
  local ok_load, load_err = Mimir.Serialization.load(checkpoint, format, {
    load_encoder = mode == "encode", load_tokenizer = false, load_optimizer = false,
    strict_mode = false, validate_checksums = true,
  })
  if ok_load == false then die("Serialization.load: " .. tostring(load_err)) end
end

local function read_ppm(path)
  local file, open_err = io.open(path, "rb")
  if not file then return nil, open_err end
  local function skip_space_and_comments()
    while true do
      local position = file:seek()
      local char = file:read(1)
      if char == nil then return end
      if char == "#" then file:read("*l")
      elseif char:match("%s") == nil then file:seek("set", position); return end
    end
  end
  local function token()
    skip_space_and_comments()
    local chars = {}
    while true do
      local position = file:seek()
      local char = file:read(1)
      if char == nil then break end
      if char == "#" or char:match("%s") then file:seek("set", position); break end
      chars[#chars + 1] = char
    end
    return #chars > 0 and table.concat(chars) or nil
  end
  local magic = token()
  local width, height, max_value = tonumber(token()), tonumber(token()), tonumber(token())
  if (magic ~= "P6" and magic ~= "P3") or not width or not height or not max_value then
    file:close(); return nil, "PPM invalide (P6/P3 attendu)"
  end
  if width < 1 or height < 1 or width ~= math.floor(width) or height ~= math.floor(height) then
    file:close(); return nil, "dimensions PPM invalides"
  end
  if max_value < 1 or max_value > 255 then file:close(); return nil, "maxval PPM non supporté" end
  local count = width * height * 3
  local pixels = {}; pixels[count] = 0
  if magic == "P6" then
    -- Un seul séparateur termine l'en-tête : les pixels peuvent être 10, 32 ou 35.
    local separator = file:read(1)
    if not separator or not separator:match("%s") then file:close(); return nil, "séparateur P6 absent" end
    if separator == "\r" then
      local position = file:seek()
      if file:read(1) ~= "\n" then file:seek("set", position) end
    end
    local bytes = file:read(count); file:close()
    if bytes == nil or #bytes ~= count then return nil, "pixels P6 tronqués" end
    for index = 1, count do pixels[index] = math.floor((bytes:byte(index) or 0) * 255 / max_value + 0.5) end
  else
    for index = 1, count do
      local value = tonumber(token())
      if value == nil then file:close(); return nil, "pixels P3 tronqués" end
      pixels[index] = math.floor(clamp(value, 0, max_value) * 255 / max_value + 0.5)
    end
    file:close()
  end
  return { width = width, height = height, pixels = pixels }
end

local function resize_rgb_nearest(source, source_w, source_h, target_w, target_h)
  local output = {}; output[target_w * target_h * 3] = 0
  for y = 0, target_h - 1 do
    local sy = math.min(source_h - 1, math.floor((y + 0.5) * source_h / target_h))
    for x = 0, target_w - 1 do
      local sx = math.min(source_w - 1, math.floor((x + 0.5) * source_w / target_w))
      local si, di = (sy * source_w + sx) * 3, (y * target_w + x) * 3
      for channel = 1, 3 do output[di + channel] = source[si + channel] end
    end
  end
  return output
end

local function write_raw(path, values)
  local file, open_err = io.open(path, "wb")
  if not file then return false, open_err end
  local ok, write_err = pcall(function()
    for index = 1, #values do file:write(string.pack("<f", tonumber(values[index]) or 0.0)) end
  end)
  file:close(); return ok, write_err
end

local function read_raw(path, expected_count)
  local file, open_err = io.open(path, "rb")
  if not file then return nil, open_err end
  local bytes = file:read("*a"); file:close()
  local expected_bytes = expected_count * 4
  if #bytes ~= expected_bytes then
    return nil, string.format("taille RAW invalide: %d octets, attendu %d", #bytes, expected_bytes)
  end
  local values = {}; values[expected_count] = 0.0
  for index = 0, expected_count - 1 do values[index + 1] = string.unpack("<f", bytes, index * 4 + 1) end
  return values
end

local function write_ppm(path, pixels)
  local expected = cfg.image_w * cfg.image_h * 3
  if type(pixels) ~= "table" or #pixels ~= expected then return false, "buffer image de taille invalide" end
  local bytes = {}; bytes[expected] = ""
  for index = 1, expected do
    local value = tonumber(pixels[index]) or 0.0
    bytes[index] = string.char(math.floor(clamp(0.5 + 0.5 * value, 0, 1) * 255 + 0.5))
  end
  local file, open_err = io.open(path, "wb")
  if not file then return false, open_err end
  local ok, write_err = pcall(function()
    file:write(string.format("P6\n%d %d\n255\n", cfg.image_w, cfg.image_h)); file:write(table.concat(bytes))
  end)
  file:close(); return ok, write_err
end

local output_dir = FS.dirname(output_path)
if output_dir and output_dir ~= "" then FS.mkdir_p(output_dir) end
create_and_load()

local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local latent_dim = cfg.latent_dim
logx(string.format("[infer_vae_conv] mode=%s checkpoint=%s image=%dx%dx%d latent=%dx%dx%d",
  mode, checkpoint, cfg.image_w, cfg.image_h, cfg.image_c, cfg.latent_w, cfg.latent_h, cfg.latent_c))

if mode == "encode" then
  local ppm, ppm_err = read_ppm(input_path)
  if not ppm then die("lecture PPM: " .. tostring(ppm_err)) end
  local pixels = ppm.pixels
  if ppm.width ~= cfg.image_w or ppm.height ~= cfg.image_h then
    if not Args.get_bool(opts, "resize", true) then
      die(string.format("dimensions PPM %dx%d, attendu %dx%d", ppm.width, ppm.height, cfg.image_w, cfg.image_h))
    end
    pixels = resize_rgb_nearest(pixels, ppm.width, ppm.height, cfg.image_w, cfg.image_h)
    logx(string.format("[infer_vae_conv] redimensionnement %dx%d -> %dx%d", ppm.width, ppm.height, cfg.image_w, cfg.image_h))
  end
  local input = {}; input[image_dim] = 0.0
  for index = 1, image_dim do input[index] = (pixels[index] / 255.0) * 2.0 - 1.0 end
  local packed, forward_err = Mimir.Model.forward(input, false)
  if type(packed) ~= "table" then die("Model.forward(encode): " .. tostring(forward_err)) end
  if #packed < image_dim + 2 * latent_dim then die("taille de sortie encoder inattendue: " .. tostring(#packed)) end
  local latent = {}; latent[latent_dim] = 0.0
  for index = 1, latent_dim do latent[index] = packed[image_dim + index] end
  local ok_raw, raw_err = write_raw(output_path, latent)
  if not ok_raw then die("écriture RAW: " .. tostring(raw_err)) end
  logx(string.format("[infer_vae_conv] RAW écrit: %s (%d float32, %d octets)", output_path, latent_dim, latent_dim * 4))
else
  local latent, raw_err = read_raw(input_path, latent_dim)
  if not latent then die("lecture RAW: " .. tostring(raw_err)) end
  local pixels, forward_err = Mimir.Model.forward(latent, false)
  if type(pixels) ~= "table" then die("Model.forward(decode): " .. tostring(forward_err)) end
  local ok_ppm, ppm_err = write_ppm(output_path, pixels)
  if not ok_ppm then die("écriture PPM: " .. tostring(ppm_err)) end
  logx("[infer_vae_conv] PPM écrit: " .. output_path)
end
