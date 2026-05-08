-- Test reconstruction d'image via VAEConv (encoder+decoder).
--
-- Usage:
--   ./bin/mimir --lua scripts/tests/test_vae_conv_generate.lua -- \
--     --in 01.ppm \
--     --checkpoint checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop \
--     --out scripts/tests/out_vae_conv_recon.ppm
--
-- Par défaut, si l'image d'entrée n'a pas la même taille que le checkpoint,
-- elle est redimensionnée (nearest-neighbor) vers la taille attendue.
-- Désactiver via: --no-resize
--
-- Sortie: image PPM (P6) RGB, pixels en [0,255].

local Args = dofile("scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local function logx(msg)
  local l = rawget(_G, "log")
  if type(l) == "function" then l(msg) else print(msg) end
end

local function die(msg)
  error(tostring(msg or "error"))
end

local function opt_str(k, d)
  local v = opts[k]
  if v == nil or v == true then return d end
  return tostring(v)
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

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function mkdir_p(dir)
  if dir == nil or dir == "" then return end
  -- best-effort
  os.execute("mkdir -p '" .. tostring(dir):gsub("'", "'\\''") .. "' 2>/dev/null")
end

local function dirname(path)
  path = tostring(path or "")
  local p = path:match("^(.*)/[^/]+$")
  if p == nil or p == "" then return "." end
  return p
end

-- --------------------------------------------------------------------------
-- Lecture PPM (P6/P3)
-- --------------------------------------------------------------------------
local function read_ppm(path)
  local f, err = io.open(path, "rb")
  if not f then return nil, err end

  local function read_byte()
    local b = f:read(1)
    if not b then return nil end
    return string.byte(b)
  end

  local function skip_ws_and_comments()
    while true do
      local pos = f:seek()
      local b = read_byte()
      if not b then return end
      local c = string.char(b)
      if c == "#" then
        -- skip comment line
        f:read("*l")
      elseif c:match("%s") then
        -- keep skipping whitespace
      else
        -- unread 1 byte
        f:seek("set", pos)
        return
      end
    end
  end

  local function read_token()
    skip_ws_and_comments()
    local tok = {}
    while true do
      local pos = f:seek()
      local b = read_byte()
      if not b then break end
      local c = string.char(b)
      if c:match("%s") or c == "#" then
        -- token ended; if comment, rewind so skip_ws can consume it
        f:seek("set", pos)
        break
      end
      tok[#tok + 1] = c
    end
    if #tok == 0 then return nil end
    return table.concat(tok)
  end

  local magic = read_token()
  if magic ~= "P6" and magic ~= "P3" then
    f:close()
    return nil, "unsupported PPM (expected P6/P3), got=" .. tostring(magic)
  end

  local w = tonumber(read_token() or "")
  local h = tonumber(read_token() or "")
  local maxval = tonumber(read_token() or "")
  if not w or not h or not maxval then
    f:close()
    return nil, "invalid PPM header (w/h/maxval)"
  end
  w = math.floor(w)
  h = math.floor(h)
  maxval = math.floor(maxval)
  if w <= 0 or h <= 0 then
    f:close()
    return nil, "invalid PPM dims"
  end
  if maxval <= 0 or maxval > 255 then
    f:close()
    return nil, "unsupported PPM maxval (expected 1..255), got=" .. tostring(maxval)
  end

  if magic == "P6" then
    -- After maxval there is one whitespace char before binary payload.
    skip_ws_and_comments()
    local expected = w * h * 3
    local data = f:read(expected)
    f:close()
    if not data or #data ~= expected then
      return nil, string.format("truncated P6 payload: got=%d expected=%d", data and #data or 0, expected)
    end
    return { fmt = "P6", w = w, h = h, maxval = maxval, data = data }, nil
  end

  -- P3 (ASCII)
  local expected = w * h * 3
  local pixels = {}
  pixels[expected] = 0
  for i = 1, expected do
    local t = read_token()
    if t == nil then
      f:close()
      return nil, "truncated P3 payload"
    end
    local v = tonumber(t) or 0
    v = math.floor(v)
    if v < 0 then v = 0 end
    if v > maxval then v = maxval end
    -- normalize to [0,255]
    pixels[i] = math.floor((v / maxval) * 255.0 + 0.5)
  end
  f:close()
  return { fmt = "P3", w = w, h = h, maxval = maxval, pixels = pixels }, nil
end

local function resize_rgb_u8_nearest(src, src_w, src_h, dst_w, dst_h)
  local out = {}
  out[dst_w * dst_h * 3] = 0
  for y = 0, dst_h - 1 do
    local sy = math.floor((y + 0.5) * src_h / dst_h)
    if sy < 0 then sy = 0 end
    if sy >= src_h then sy = src_h - 1 end
    for x = 0, dst_w - 1 do
      local sx = math.floor((x + 0.5) * src_w / dst_w)
      if sx < 0 then sx = 0 end
      if sx >= src_w then sx = src_w - 1 end

      local si = (sy * src_w + sx) * 3
      local di = (y * dst_w + x) * 3
      out[di + 1] = src[si + 1]
      out[di + 2] = src[si + 2]
      out[di + 3] = src[si + 3]
    end
  end
  return out
end

local function to_rgb_u8_table(ppm)
  if ppm.fmt == "P3" then
    return ppm.pixels
  end
  -- P6 bytes string -> u8 table (only used for small-ish images)
  local expected = ppm.w * ppm.h * 3
  local t = {}
  t[expected] = 0
  for i = 1, expected do
    t[i] = string.byte(ppm.data, i)
  end
  return t
end

local function sample_resize_p6_bytes_to_u8(data, src_w, src_h, dst_w, dst_h, maxval)
  local out = {}
  out[dst_w * dst_h * 3] = 0
  local scale = 255.0 / (maxval > 0 and maxval or 255)
  for y = 0, dst_h - 1 do
    local sy = math.floor((y + 0.5) * src_h / dst_h)
    if sy < 0 then sy = 0 end
    if sy >= src_h then sy = src_h - 1 end
    for x = 0, dst_w - 1 do
      local sx = math.floor((x + 0.5) * src_w / dst_w)
      if sx < 0 then sx = 0 end
      if sx >= src_w then sx = src_w - 1 end
      local si = (sy * src_w + sx) * 3 + 1
      local r = string.byte(data, si) or 0
      local g = string.byte(data, si + 1) or 0
      local b = string.byte(data, si + 2) or 0
      if maxval ~= 255 then
        r = math.floor(r * scale + 0.5)
        g = math.floor(g * scale + 0.5)
        b = math.floor(b * scale + 0.5)
      end
      local di = (y * dst_w + x) * 3
      out[di + 1] = r
      out[di + 2] = g
      out[di + 3] = b
    end
  end
  return out
end

local function rgb_u8_to_f32_minus1_1(pixels_u8)
  local out = {}
  out[#pixels_u8] = 0.0
  for i = 1, #pixels_u8 do
    local u = tonumber(pixels_u8[i]) or 0
    local t = clamp(u / 255.0, 0.0, 1.0)
    out[i] = t * 2.0 - 1.0
  end
  return out
end

-- --------------------------------------------------------------------------
-- RNG déterministe (LCG) + Box-Muller pour N(0,1)
-- --------------------------------------------------------------------------
local function make_rng(seed)
  local state = tonumber(seed) or 0
  if state < 0 then state = -state end
  state = math.floor(state) % 2147483647
  if state == 0 then state = 123456789 end

  local function rand_u32()
    state = (1103515245 * state + 12345) % 2147483647
    return state
  end

  local function rand01()
    return rand_u32() / 2147483647
  end

  local have_spare = false
  local spare = 0.0
  local function randn()
    if have_spare then
      have_spare = false
      return spare
    end
    local u1 = rand01()
    local u2 = rand01()
    if u1 < 1e-12 then u1 = 1e-12 end
    local r = math.sqrt(-2.0 * math.log(u1))
    local theta = 2.0 * math.pi * u2
    spare = r * math.sin(theta)
    have_spare = true
    return r * math.cos(theta)
  end

  return {
    rand01 = rand01,
    randn = randn,
  }
end

-- --------------------------------------------------------------------------
-- Inférer une config VAEConv depuis un checkpoint RawFolder (architecture.json)
-- --------------------------------------------------------------------------
local function infer_cfg_from_checkpoint(ckpt_dir)
  local arch_path = tostring(ckpt_dir) .. "/model/architecture.json"
  local arch = read_json(arch_path)
  if type(arch) ~= "table" then
    return nil, "read_json failed: " .. tostring(arch_path)
  end

  local image_w = tonumber(arch.image_width) or tonumber(arch.image_w) or 0
  local image_h = tonumber(arch.image_height) or tonumber(arch.image_h) or 0
  if image_w <= 0 or image_h <= 0 then
    return nil, "invalid image dimensions in architecture.json"
  end

  local layers = arch.layers
  if type(layers) ~= "table" then
    return nil, "architecture.json missing layers"
  end

  local function find_layer(name)
    for _, L in ipairs(layers) do
      if type(L) == "table" and L.name == name then
        return L
      end
    end
    return nil
  end

  local enc_conv_in = find_layer("vae_conv/enc/conv_in")
  local dec_conv_in = find_layer("vae_conv/dec/conv_in")
  if type(dec_conv_in) ~= "table" then
    return nil, "cannot infer: missing layer vae_conv/dec/conv_in"
  end

  local image_c = 3
  if type(enc_conv_in) == "table" and tonumber(enc_conv_in.in_channels) then
    image_c = math.max(1, math.floor(tonumber(enc_conv_in.in_channels)))
  end

  local base_channels = tonumber(dec_conv_in.out_channels) or 0
  local latent_c = tonumber(dec_conv_in.in_channels) or 0
  if base_channels <= 0 or latent_c <= 0 then
    return nil, "cannot infer base_channels/latent_c from vae_conv/dec/conv_in"
  end

  local downsamples = 0
  for _, L in ipairs(layers) do
    if type(L) == "table" and type(L.name) == "string" then
      if L.name:match("^vae_conv/enc/down%d+/conv$") and tonumber(L.stride) == 2 then
        downsamples = downsamples + 1
      end
    end
  end

  local div = 2 ^ downsamples
  if div <= 0 then div = 1 end
  if (image_h % div) ~= 0 or (image_w % div) ~= 0 then
    return nil, string.format("cannot infer latent_h/w: image not divisible by 2^downsamples (image=%dx%d downsamples=%d)", image_w, image_h, downsamples)
  end

  local latent_h = math.floor(image_h / div)
  local latent_w = math.floor(image_w / div)

  -- Options d'architecture (si présentes dans arch.model_config)
  local use_attention = false
  local attn_heads = nil
  local attn_max_tokens = nil
  if type(arch.model_config) == "table" then
    if arch.model_config.use_attention == true then use_attention = true end
    attn_heads = tonumber(arch.model_config.attn_heads)
    attn_max_tokens = tonumber(arch.model_config.attn_max_tokens)
  end

  return {
    image_w = image_w,
    image_h = image_h,
    image_c = image_c,
    latent_h = latent_h,
    latent_w = latent_w,
    latent_c = math.floor(latent_c),
    base_channels = math.floor(base_channels),
    downsamples = downsamples,
    use_attention = use_attention,
    attn_heads = attn_heads,
    attn_max_tokens = attn_max_tokens,
  }, nil
end

-- --------------------------------------------------------------------------
-- Écriture PPM (P6) depuis un buffer float HWC en [-1,1]
-- --------------------------------------------------------------------------
local function write_ppm_rgb_f32_hwc(path, pixels, w, h)
  if type(pixels) ~= "table" then return false, "pixels must be table" end
  w = math.floor(tonumber(w) or 0)
  h = math.floor(tonumber(h) or 0)
  if w <= 0 or h <= 0 then return false, "invalid w/h" end
  local expected = w * h * 3
  if #pixels ~= expected then
    return false, string.format("invalid pixel buffer: got=%d expected=%d", #pixels, expected)
  end

  local header = string.format("P6\n%d %d\n255\n", w, h)
  local out = {}
  out[#out + 1] = header

  -- Convert HWC floats [-1,1] -> bytes
  local bytes = {}
  bytes[expected] = "" -- pre-allocate size
  for i = 1, expected do
    local x = tonumber(pixels[i]) or 0.0
    local t = 0.5 + 0.5 * x
    local p = math.floor(clamp(t, 0.0, 1.0) * 255.0 + 0.5)
    bytes[i] = string.char(clamp(p, 0, 255))
  end

  out[#out + 1] = table.concat(bytes)
  local content = table.concat(out)
  local f, ferr = io.open(path, "wb")
  if not f then return false, ferr end
  local ok_write, werr = pcall(function()
    f:write(content)
  end)
  f:close()
  if not ok_write then return false, werr end
  return true, nil
end

-- --------------------------------------------------------------------------
-- Main
-- --------------------------------------------------------------------------
local DEFAULT_CKPT = "checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop"
local checkpoint_dir = opt_str("checkpoint", opt_str("ckpt", DEFAULT_CKPT))
local in_path = opt_str("in", opt_str("input", "01.ppm"))
local out_path = opt_str("out", "scripts/tests/out_vae_conv_recon.ppm")
local out_in_path = opt_str("out-in", opt_str("out_in", ""))
local out_diff_path = opt_str("out-diff", opt_str("out_diff", ""))
local RESIZE_INPUT = true
if opts.resize == false then RESIZE_INPUT = false end
if opts["no-resize"] == true then RESIZE_INPUT = false end

-- Mémoire: keep simple (pas d'auto tuning)
if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  local mem_gb = opt_num("alloc-gb", opt_num("mem-gb", 8))
  local compression = opts.compression
  if compression == nil then compression = opts.compress end
  if compression == nil then compression = true end
  Mimir.Allocator.configure({ max_ram_gb = mem_gb, enable_compression = compression })
end

logx("[test_vae_conv_generate] checkpoint=" .. tostring(checkpoint_dir))
logx("[test_vae_conv_generate] in=" .. tostring(in_path))

local inferred, err_inf = infer_cfg_from_checkpoint(checkpoint_dir)
if not inferred then
  die("infer_cfg_from_checkpoint failed: " .. tostring(err_inf))
end

local cfg = Mimir.Architectures.default_config("vae_conv")
if type(cfg) ~= "table" then die("default_config(vae_conv) failed") end

cfg.image_w = inferred.image_w
cfg.image_h = inferred.image_h
cfg.image_c = inferred.image_c
cfg.latent_h = inferred.latent_h
cfg.latent_w = inferred.latent_w
cfg.latent_c = inferred.latent_c
cfg.base_channels = inferred.base_channels
cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c
cfg.text_cond = false
cfg.stochastic_latent = false

-- Attention (si présente dans le checkpoint)
if inferred.use_attention ~= nil then
  cfg.use_attention = inferred.use_attention
end
if inferred.attn_heads ~= nil then
  cfg.attn_heads = inferred.attn_heads
end
if inferred.attn_max_tokens ~= nil then
  cfg.attn_max_tokens = inferred.attn_max_tokens
end

-- Allow manual overrides (optionnels)
if opts["image-w"] then cfg.image_w = opt_int("image-w", cfg.image_w) end
if opts["image-h"] then cfg.image_h = opt_int("image-h", cfg.image_h) end
if opts["image-c"] then cfg.image_c = opt_int("image-c", cfg.image_c) end
if opts["latent-h"] then cfg.latent_h = opt_int("latent-h", cfg.latent_h) end
if opts["latent-w"] then cfg.latent_w = opt_int("latent-w", cfg.latent_w) end
if opts["latent-c"] then cfg.latent_c = opt_int("latent-c", cfg.latent_c) end
if opts["base-channels"] then cfg.base_channels = opt_int("base-channels", cfg.base_channels) end

-- Optional manual overrides for attention
if opts.attn ~= nil or opts["use-attn"] ~= nil then
  local v = opts.attn
  if v == nil then v = opts["use-attn"] end
  if v == true or v == false then cfg.use_attention = v end
  if type(v) == "string" then
    local s = v:lower()
    if s == "1" or s == "true" or s == "yes" or s == "on" then cfg.use_attention = true end
    if s == "0" or s == "false" or s == "no" or s == "off" then cfg.use_attention = false end
  end
end
if opts["attn-heads"] then cfg.attn_heads = opt_int("attn-heads", cfg.attn_heads) end
if opts["attn-max-tokens"] then cfg.attn_max_tokens = opt_int("attn-max-tokens", cfg.attn_max_tokens) end

cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c

logx(string.format("[test_vae_conv_generate] cfg image=%dx%dx%d latent=%dx%dx%d base=%d", cfg.image_w, cfg.image_h, cfg.image_c, cfg.latent_h, cfg.latent_w, cfg.latent_c, cfg.base_channels))

-- Lire l'image PPM d'entrée et l'adapter à la taille attendue
local ppm, err_ppm = read_ppm(in_path)
if not ppm then die("read_ppm failed: " .. tostring(err_ppm)) end

local input_u8 = nil
if ppm.fmt == "P6" then
  if ppm.w == cfg.image_w and ppm.h == cfg.image_h then
    input_u8 = sample_resize_p6_bytes_to_u8(ppm.data, ppm.w, ppm.h, cfg.image_w, cfg.image_h, ppm.maxval)
  else
    if not RESIZE_INPUT then
      die(string.format("input size mismatch: got=%dx%d expected=%dx%d (use --resize or provide matching PPM)", ppm.w, ppm.h, cfg.image_w, cfg.image_h))
    end
    input_u8 = sample_resize_p6_bytes_to_u8(ppm.data, ppm.w, ppm.h, cfg.image_w, cfg.image_h, ppm.maxval)
  end
else
  local src_u8 = to_rgb_u8_table(ppm)
  if ppm.w == cfg.image_w and ppm.h == cfg.image_h then
    input_u8 = src_u8
  else
    if not RESIZE_INPUT then
      die(string.format("input size mismatch: got=%dx%d expected=%dx%d (use --resize or provide matching PPM)", ppm.w, ppm.h, cfg.image_w, cfg.image_h))
    end
    input_u8 = resize_rgb_u8_nearest(src_u8, ppm.w, ppm.h, cfg.image_w, cfg.image_h)
  end
end

local input_f32 = rgb_u8_to_f32_minus1_1(input_u8)

local ok_create, err_create = Mimir.Model.create("vae_conv", cfg)
if not ok_create then die("Model.create(vae_conv) failed: " .. tostring(err_create)) end

local ok_alloc, nparams_or_err = Mimir.Model.allocate_params()
if ok_alloc == false then die("Model.allocate_params failed: " .. tostring(nparams_or_err)) end

local ok_load, err_load = Mimir.Serialization.load(checkpoint_dir, "raw_folder", {
  load_encoder = false,
  load_tokenizer = false,
  load_optimizer = false,
  strict_mode = false,
  validate_checksums = true,
})
if ok_load == false then die("Serialization.load failed: " .. tostring(err_load)) end

mkdir_p(dirname(out_path))

local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local latent_dim = cfg.latent_dim

-- Forward VAEConv complet: output pack = recon || mu || logvar
local packed, err_fwd = Mimir.Model.forward(input_f32, false)
if packed == nil then die("Model.forward failed: " .. tostring(err_fwd)) end
local expected = image_dim + 2 * latent_dim
if #packed ~= expected then
  die(string.format("unexpected output size: got=%d expected=%d (image_dim=%d latent_dim=%d)", #packed, expected, image_dim, latent_dim))
end

local recon = {}
recon[image_dim] = 0.0
for i = 1, image_dim do
  recon[i] = packed[i]
end

local ok_w, err_w = write_ppm_rgb_f32_hwc(out_path, recon, cfg.image_w, cfg.image_h)
if ok_w == false then die("write_ppm failed: " .. tostring(err_w)) end
logx("[test_vae_conv_generate] wrote recon " .. tostring(out_path))

if out_in_path ~= nil and out_in_path ~= "" then
  mkdir_p(dirname(out_in_path))
  local ok_in, err_in = write_ppm_rgb_f32_hwc(out_in_path, input_f32, cfg.image_w, cfg.image_h)
  if ok_in == false then die("write_ppm(input) failed: " .. tostring(err_in)) end
  logx("[test_vae_conv_generate] wrote input " .. tostring(out_in_path))
end

if out_diff_path ~= nil and out_diff_path ~= "" then
  mkdir_p(dirname(out_diff_path))
  local diff = {}
  diff[image_dim] = 0.0
  for i = 1, image_dim do
    diff[i] = math.abs((recon[i] or 0.0) - (input_f32[i] or 0.0))
  end
  local ok_d, err_d = write_ppm_rgb_f32_hwc(out_diff_path, diff, cfg.image_w, cfg.image_h)
  if ok_d == false then die("write_ppm(diff) failed: " .. tostring(err_d)) end
  logx("[test_vae_conv_generate] wrote diff " .. tostring(out_diff_path))
end

logx("[test_vae_conv_generate] done")
