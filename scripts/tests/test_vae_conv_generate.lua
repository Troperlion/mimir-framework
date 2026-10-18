-- Test reconstruction/encodage/décodage d'image via VAEConv.
--
-- Mode par défaut — reconstruction complète (encoder + décodeur) :
--   ./bin/mimir --lua scripts/tests/test_vae_conv_generate.lua -- \
--     --in 01.ppm \
--     --checkpoint checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop \
--     --out scripts/tests/out_vae_conv_recon.ppm
--
-- Mode encodage (--encode) — compresse l'image vers un fichier latent binaire :
--   ./bin/mimir --lua scripts/tests/test_vae_conv_generate.lua -- \
--     --encode \
--     --in 01.ppm \
--     --checkpoint checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop \
--     --out scripts/tests/out_vae_conv.lat
--
-- Mode décodage (--decode) — reconstruit une image depuis un fichier latent :
--   ./bin/mimir --lua scripts/tests/test_vae_conv_generate.lua -- \
--     --decode \
--     --in scripts/tests/out_vae_conv.lat \
--     --checkpoint checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop \
--     --out scripts/tests/out_vae_conv_decoded.ppm
--
-- Par défaut, si l'image d'entrée n'a pas la même taille que le checkpoint,
-- elle est redimensionnée (nearest-neighbor) vers la taille attendue.
-- Désactiver via: --no-resize
--
-- Sortie reconstruction/décodage : image PPM (P6) RGB, pixels en [0,255].
-- Sortie encodage : fichier binaire .lat (header texte + float32 LE).

---@diagnostic disable: need-check-nil, inject-field

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local function logx(msg)
  local l = rawget(_G, "log")
  if type(l) == "function" then l(msg) else print(msg) end
end

local function die(msg)
  error(tostring(msg or "error"))
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  if ok == false then
    die("dtype invalide: " .. tostring(dt_or_err))
  end
  return true
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

local function first_defined(values, keys)
  for _, key in ipairs(keys) do
    if values[key] ~= nil then return values[key] end
  end
  return nil
end

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function mkdir_p(dir)
  FS.mkdir_p(dir)
end

local function dirname(path)
  local p = FS.dirname(path)
  if p == nil or p == "" then return "." end
  return p
end

local function endswith(s, suffix)
  if type(s) ~= "string" or type(suffix) ~= "string" then return false end
  if #suffix == 0 then return true end
  if #s < #suffix then return false end
  return s:sub(-#suffix) == suffix
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
        local _ = f:read("*l")
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

  -- Chemin rapide : model_config est stocké directement dans architecture.json
  -- depuis le patch VAEConv 2026-06. Contient TOUS les flags architecturaux
  -- (resnet, attention, decoder_upsample, use_skip_connections, use_encoder_prior,
  -- resnet_max_tokens, etc.) en plus des dimensions image/latent.
  local mc = arch.model_config or arch.modelConfig
  if type(mc) == "table" and (tonumber(mc.image_w) or 0) > 0 then
    local function mci(k)
      local n = tonumber(mc[k] or 0) or 0
      return math.floor(n)
    end
    return {
      image_w              = mci("image_w"),
      image_h              = mci("image_h"),
      image_c              = math.max(1, mci("image_c")),
      latent_h             = mci("latent_h"),
      latent_w             = mci("latent_w"),
      latent_c             = mci("latent_c"),
      base_channels        = mci("base_channels"),
      resnet               = first_defined(mc, {"resnet", "use_resnet", "use_attention"}),
      attention            = first_defined(mc, {"attention", "use_attn"}),
      resnet_max_tokens    = mc.resnet_max_tokens,
      use_skip_connections = mc.use_skip_connections,
      use_encoder_prior    = mc.use_encoder_prior,
      decoder_upsample     = mc.decoder_upsample,
      stochastic_latent    = mc.stochastic_latent,
    }, nil
  end

  -- Repli legacy : parser les layers manuellement (anciens checkpoints sans model_config)
  local layers = arch.layers
  if type(layers) ~= "table" then
    return nil, "architecture.json missing layers and model_config"
  end

  local image_w = tonumber(arch.image_width) or tonumber(arch.image_w) or 0
  local image_h = tonumber(arch.image_height) or tonumber(arch.image_h) or 0
  if image_w <= 0 or image_h <= 0 then
    return nil, "invalid image dimensions in architecture.json"
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
  if type(enc_conv_in) == "table" then
    local in_channels = tonumber(enc_conv_in.in_channels)
    if in_channels then
      image_c = math.max(1, math.floor(in_channels))
    end
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

  return {
    image_w       = image_w,
    image_h       = image_h,
    image_c       = image_c,
    latent_h      = latent_h,
    latent_w      = latent_w,
    latent_c      = math.floor(latent_c),
    base_channels = math.floor(base_channels),
    downsamples   = downsamples,
  }, nil
end

local function infer_cfg_from_debug_json(debug_json_path)
  local j = read_json(debug_json_path)
  if type(j) ~= "table" then
    return nil, "read_json failed: " .. tostring(debug_json_path)
  end

  local mc = j.model_config
  if type(mc) ~= "table" then
    local m = j.model
    if type(m) == "table" and type(m.model_config) == "table" then
      mc = m.model_config
    end
  end
  if type(mc) ~= "table" then
    return nil, "missing model_config in " .. tostring(debug_json_path)
  end

  local function mci(k)
    local n = tonumber(mc[k] or 0) or 0
    return math.floor(n)
  end

  local image_w = mci("image_w")
  local image_h = mci("image_h")
  local latent_h = mci("latent_h")
  local latent_w = mci("latent_w")
  local latent_c = mci("latent_c")
  local base_channels = mci("base_channels")
  if image_w <= 0 or image_h <= 0 or latent_h <= 0 or latent_w <= 0 or latent_c <= 0 or base_channels <= 0 then
    return nil, "invalid dims in model_config from " .. tostring(debug_json_path)
  end

  return {
    image_w = image_w,
    image_h = image_h,
    image_c = math.max(1, mci("image_c")),
    latent_h = latent_h,
    latent_w = latent_w,
    latent_c = latent_c,
    base_channels = base_channels,
    resnet = first_defined(mc, {"resnet", "use_resnet", "use_attention"}),
    attention = first_defined(mc, {"attention", "use_attn"}),
    resnet_max_tokens = mc.resnet_max_tokens,
    use_skip_connections = mc.use_skip_connections,
    use_encoder_prior = mc.use_encoder_prior,
    decoder_upsample = mc.decoder_upsample,
    stochastic_latent = mc.stochastic_latent,
  }, nil
end

-- --------------------------------------------------------------------------
-- SafeTensors minimal reader: extract model/architecture_json
-- --------------------------------------------------------------------------
local function u64_le(bytes)
  local b = { bytes:byte(1, 8) }
  local n = 0
  local mul = 1
  for i = 1, 8 do
    n = n + (b[i] or 0) * mul
    mul = mul * 256
  end
  return n
end

local function read_exact(f, n)
  local s = f:read(n)
  if not s or #s ~= n then
    return nil, "short read"
  end
  return s
end

local function safe_json_decode(s)
  local json_mod = rawget(_G, "json")
  if type(json_mod) == "table" and type(json_mod.decode) == "function" then
    local ok, v = pcall(json_mod.decode, s)
    if ok and type(v) == "table" then return v end
  end
  local cjson_mod = rawget(_G, "cjson")
  if type(cjson_mod) == "table" and type(cjson_mod.decode) == "function" then
    local ok, v = pcall(cjson_mod.decode, s)
    if ok and type(v) == "table" then return v end
  end
  return nil
end

local function read_safetensors_header(path)
  local f = io.open(path, "rb")
  if not f then return nil, "cannot open" end
  local len_bytes, err = read_exact(f, 8)
  if not len_bytes then f:close(); return nil, err end
  local header_len = u64_le(len_bytes)
  if header_len <= 0 or header_len > 64 * 1024 * 1024 then
    f:close()
    return nil, "invalid header_len=" .. tostring(header_len)
  end
  local header_str, err2 = read_exact(f, header_len)
  if not header_str then f:close(); return nil, err2 end
  local header = safe_json_decode(header_str)
  if type(header) ~= "table" then
    f:close()
    return nil, "header JSON decode failed"
  end
  return { f = f, header_len = header_len, header = header }, nil
end

local function extract_tensor_bytes(ctx, tensor_name)
  local entry = ctx.header[tensor_name]
  if type(entry) ~= "table" then
    return nil, "tensor not found in header: " .. tostring(tensor_name)
  end
  local offsets = entry.data_offsets
  if type(offsets) ~= "table" or #offsets < 2 then
    return nil, "missing data_offsets"
  end
  local begin = tonumber(offsets[1])
  local end_ = tonumber(offsets[2])
  if not begin or not end_ or end_ < begin then
    return nil, "invalid data_offsets"
  end
  local size = end_ - begin
  if size <= 0 or size > 128 * 1024 * 1024 then
    return nil, "invalid tensor size=" .. tostring(size)
  end
  local data_base = 8 + ctx.header_len
  ctx.f:seek("set", data_base + begin)
  local bytes, err = read_exact(ctx.f, size)
  if not bytes then
    return nil, "read tensor bytes failed: " .. tostring(err)
  end
  return bytes, nil
end

local function infer_cfg_from_safetensors(st_path)
  local ctx, err = read_safetensors_header(st_path)
  if not ctx then return nil, err end

  local bytes, errb = extract_tensor_bytes(ctx, "model/architecture_json")
  ctx.f:close()
  if not bytes then return nil, errb end

  local arch = safe_json_decode(bytes)
  if type(arch) ~= "table" then
    return nil, "architecture_json decode failed"
  end

  local mc = arch.model_config or arch.modelConfig
  if type(mc) ~= "table" then
    return nil, "missing model_config in model/architecture_json"
  end

  local function mci(k)
    local n = tonumber(mc[k] or 0) or 0
    return math.floor(n)
  end

  local image_w = mci("image_w")
  local image_h = mci("image_h")
  local latent_h = mci("latent_h")
  local latent_w = mci("latent_w")
  local latent_c = mci("latent_c")
  local base_channels = mci("base_channels")
  if image_w <= 0 or image_h <= 0 or latent_h <= 0 or latent_w <= 0 or latent_c <= 0 or base_channels <= 0 then
    return nil, "invalid model_config dimensions in safetensors"
  end

  return {
    image_w = image_w,
    image_h = image_h,
    image_c = math.max(1, mci("image_c")),
    latent_h = latent_h,
    latent_w = latent_w,
    latent_c = latent_c,
    base_channels = base_channels,
    resnet = first_defined(mc, {"resnet", "use_resnet", "use_attention"}),
    attention = first_defined(mc, {"attention", "use_attn"}),
    resnet_max_tokens = mc.resnet_max_tokens,
    use_skip_connections = mc.use_skip_connections,
    use_encoder_prior = mc.use_encoder_prior,
    decoder_upsample = mc.decoder_upsample,
    stochastic_latent = mc.stochastic_latent,
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
-- Écriture / lecture fichier latent binaire (.lat)
--
-- Format :
--   Ligne 1 : "MIMIR_LATENT\n"
--   Ligne 2 : "{latent_h} {latent_w} {latent_c}\n"
--   Suite   : latent_h * latent_w * latent_c floats 32 bits little-endian
--
-- Le vecteur stocké est z_biased (= sortie de la couche Reparameterize + prior_bias),
-- i.e. ce que dec/conv_in reçoit dans le modèle complet. vae_conv_decode peut
-- donc l'utiliser directement sans connaissance de l'architecture.
-- --------------------------------------------------------------------------
local function write_latent_bin(path, latent, lh, lw, lc)
  if type(latent) ~= "table" then return false, "latent must be table" end
  local expected = lh * lw * lc
  if #latent ~= expected then
    return false, string.format("latent size mismatch: got=%d expected=%d", #latent, expected)
  end

  local f, ferr = io.open(path, "wb")
  if not f then return false, ferr end

  local ok_w, werr = pcall(function()
    f:write(string.format("MIMIR_LATENT\n%d %d %d\n", lh, lw, lc))
    for i = 1, expected do
      f:write(string.pack("<f", tonumber(latent[i]) or 0.0))
    end
  end)
  f:close()
  if not ok_w then return false, werr end
  return true, nil
end

local function read_latent_bin(path)
  local f, ferr = io.open(path, "rb")
  if not f then return nil, ferr end

  local magic = f:read("*l")
  if magic ~= "MIMIR_LATENT" then
    f:close()
    return nil, "invalid latent file (bad magic): " .. tostring(magic)
  end

  local dims_line = f:read("*l")
  if not dims_line then
    f:close()
    return nil, "invalid latent file (missing dims)"
  end
  local lh, lw, lc = dims_line:match("^(%d+)%s+(%d+)%s+(%d+)$")
  lh = tonumber(lh)
  lw = tonumber(lw)
  lc = tonumber(lc)
  if not lh or not lw or not lc or lh <= 0 or lw <= 0 or lc <= 0 then
    f:close()
    return nil, "invalid latent file (bad dims): " .. tostring(dims_line)
  end

  local n = lh * lw * lc
  local float_bytes = n * 4
  local data = f:read(math.floor(float_bytes))
  f:close()

  if not data or #data ~= float_bytes then
    return nil, string.format("truncated latent payload: got=%d expected=%d", data and #data or 0, float_bytes)
  end

  local latent = {}
  latent[n] = 0.0
  for i = 0, n - 1 do
    local v = string.unpack("<f", data, i * 4 + 1)
    latent[i + 1] = v
  end

  return { latent = latent, latent_h = lh, latent_w = lw, latent_c = lc }, nil
end

-- --------------------------------------------------------------------------
-- Main
-- --------------------------------------------------------------------------
local DEFAULT_CKPT = "checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop"
local checkpoint_ref = opt_str("checkpoint", opt_str("ckpt", DEFAULT_CKPT))
local checkpoint_format = opt_str("format", "")

if checkpoint_format == "" then
  if endswith(checkpoint_ref:lower(), ".safetensors") then
    checkpoint_format = "safetensors"
  else
    checkpoint_format = "raw_folder"
  end
end

local checkpoint_meta_dir = checkpoint_ref
if checkpoint_format == "safetensors" then
  checkpoint_meta_dir = dirname(checkpoint_ref)
end
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

logx("[test_vae_conv_generate] checkpoint=" .. tostring(checkpoint_ref))
logx("[test_vae_conv_generate] format=" .. tostring(checkpoint_format))
logx("[test_vae_conv_generate] in=" .. tostring(in_path))

local inferred, err_inf = infer_cfg_from_checkpoint(checkpoint_meta_dir)
if checkpoint_format == "safetensors" then
  local inf_st, err_st = infer_cfg_from_safetensors(checkpoint_ref)
  if inf_st then
    inferred = inf_st
    err_inf = nil
    logx("[test_vae_conv_generate] inferred config from model/architecture_json in safetensors")
  else
    err_inf = err_st
  end
end
if not inferred then
  local sidecars = {
    checkpoint_meta_dir .. "/endtrain.json",
    checkpoint_meta_dir .. "/starttrain.json",
  }
  for _, p in ipairs(sidecars) do
    local inf2, err2 = infer_cfg_from_debug_json(p)
    if inf2 then
      inferred = inf2
      err_inf = nil
      logx("[test_vae_conv_generate] inferred config from debug_json: " .. tostring(p))
      break
    else
      err_inf = err2
    end
  end
end

local cfg = Mimir.Architectures.default_config("vae_conv")
if type(cfg) ~= "table" then die("default_config(vae_conv) failed") end

if inferred then
  -- Appliquer tous les champs inférés depuis le checkpoint (dimensions + flags architecturaux)
  local cfg_fields = {
    "image_w", "image_h", "image_c",
    "latent_h", "latent_w", "latent_c", "base_channels",
    -- flags architecturaux (depuis model_config, patch 2026-06)
    "resnet", "attention", "resnet_max_tokens",
    "use_skip_connections", "use_encoder_prior", "decoder_upsample",
  }
  for _, k in ipairs(cfg_fields) do
    if inferred[k] ~= nil then cfg[k] = inferred[k] end
  end
else
  logx("[test_vae_conv_generate] WARNING: infer_cfg_from_checkpoint failed, using default_config + CLI overrides: " .. tostring(err_inf))
end
cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c
cfg.text_cond = false
cfg.stochastic_latent = false  -- encodeur déterministe (mu uniquement)

-- Allow manual overrides (optionnels)
if opts["image-w"] then cfg.image_w = opt_int("image-w", cfg.image_w) end
if opts["image-h"] then cfg.image_h = opt_int("image-h", cfg.image_h) end
if opts["image-c"] then cfg.image_c = opt_int("image-c", cfg.image_c) end
if opts["latent-h"] then cfg.latent_h = opt_int("latent-h", cfg.latent_h) end
if opts["latent-w"] then cfg.latent_w = opt_int("latent-w", cfg.latent_w) end
if opts["latent-c"] then cfg.latent_c = opt_int("latent-c", cfg.latent_c) end
if opts["base-channels"] then cfg.base_channels = opt_int("base-channels", cfg.base_channels) end
cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c

logx(string.format("[test_vae_conv_generate] cfg image=%dx%dx%d latent=%dx%dx%d base=%d attention=%s skip=%s enc_prior=%s upsample=%s",
  cfg.image_w, cfg.image_h, cfg.image_c,
  cfg.latent_h, cfg.latent_w, cfg.latent_c,
  cfg.base_channels,
  tostring(cfg.attention),
  tostring(cfg["use_skip_connections"]),
  tostring(cfg["use_encoder_prior"]),
  tostring(cfg.decoder_upsample)))

-- Détection du mode
local MODE = "recon"
if opts["encode"] == true then MODE = "encode" end
if opts["decode"] == true then MODE = "decode" end
logx("[test_vae_conv_generate] mode=" .. MODE)

-- ==========================================================================
-- MODE ENCODE : image PPM → fichier latent (.lat)
-- Utilise le modèle vae_conv complet et extrait z_biased depuis le pack de sortie.
-- Le pack C++ est recon || z_biased || logvar (z_biased = reparam + prior_bias).
-- ==========================================================================
if MODE == "encode" then

  local ppm, err_ppm = read_ppm(in_path)
  if not ppm then die("read_ppm failed: " .. tostring(err_ppm)) end

  local input_u8 = nil
  if ppm.fmt == "P6" then
    if ppm.w ~= cfg.image_w or ppm.h ~= cfg.image_h then
      if not RESIZE_INPUT then
        die(string.format("input size mismatch: got=%dx%d expected=%dx%d", ppm.w, ppm.h, cfg.image_w, cfg.image_h))
      end
    end
    input_u8 = sample_resize_p6_bytes_to_u8(ppm.data, ppm.w, ppm.h, cfg.image_w, cfg.image_h, ppm.maxval)
  else
    local src_u8 = to_rgb_u8_table(ppm)
    if ppm.w == cfg.image_w and ppm.h == cfg.image_h then
      input_u8 = src_u8
    else
      if not RESIZE_INPUT then
        die(string.format("input size mismatch: got=%dx%d expected=%dx%d", ppm.w, ppm.h, cfg.image_w, cfg.image_h))
      end
      input_u8 = resize_rgb_u8_nearest(src_u8, ppm.w, ppm.h, cfg.image_w, cfg.image_h)
    end
  end

  local input_f32 = rgb_u8_to_f32_minus1_1(input_u8)

  local ok_create, err_create = Mimir.Model.create("vae_conv", cfg)
  if not ok_create then die("Model.create(vae_conv) failed: " .. tostring(err_create)) end

  apply_dtype(cfg)

  local ok_alloc, nparams_or_err = Mimir.Model.allocate_params()
  if ok_alloc == false then die("Model.allocate_params failed: " .. tostring(nparams_or_err)) end

  local ok_load, err_load = Mimir.Serialization.load(checkpoint_ref, checkpoint_format, {
    load_encoder = true,
    load_tokenizer = false,
    load_optimizer = false,
    strict_mode = false,
    validate_checksums = true,
  })
  if ok_load == false then die("Serialization.load failed: " .. tostring(err_load)) end

  local image_dim  = cfg.image_w * cfg.image_h * cfg.image_c
  local latent_dim = cfg.latent_dim

  -- Forward complet : pack = recon || z_biased || logvar
  -- z_biased = reparam(mu, logvar) + prior_bias  (ou juste z si pas de prior)
  local packed, err_fwd = Mimir.Model.forward(input_f32, false)
  if packed == nil then die("Model.forward failed: " .. tostring(err_fwd)) end
  local expected_sz = image_dim + 2 * latent_dim
  if #packed ~= expected_sz then
    die(string.format("unexpected output size: got=%d expected=%d (image_dim=%d latent_dim=%d)",
      #packed, expected_sz, image_dim, latent_dim))
  end

  -- Extraire z_biased directement depuis le pack (indices image_dim+1 .. image_dim+latent_dim).
  -- Le modèle C++ expose maintenant z_biased (= sortie reparams + prior_bias) à cet offset,
  -- aucun calcul supplémentaire n'est nécessaire côté Lua.
  local z_biased = {}
  z_biased[latent_dim] = 0.0
  for i = 1, latent_dim do
    z_biased[i] = packed[image_dim + i]
  end

  local out_lat = out_path ~= "" and out_path or "scripts/tests/out_vae_conv.lat"
  mkdir_p(dirname(out_lat))
  local ok_lat, err_lat = write_latent_bin(out_lat, z_biased, cfg.latent_h, cfg.latent_w, cfg.latent_c)
  if not ok_lat then die("write_latent_bin failed: " .. tostring(err_lat)) end
  logx(string.format("[test_vae_conv_generate] wrote latent %s (lh=%d lw=%d lc=%d dim=%d)",
    out_lat, cfg.latent_h, cfg.latent_w, cfg.latent_c, latent_dim))

-- ==========================================================================
-- MODE DECODE : fichier latent (.lat) → image PPM
-- Utilise uniquement le décodeur (vae_conv_decode).
-- ==========================================================================
elseif MODE == "decode" then

  local lat_data, err_lat = read_latent_bin(in_path)
  if not lat_data then die("read_latent_bin failed: " .. tostring(err_lat)) end

  -- Vérifier compatibilité des dimensions avec le checkpoint
  if lat_data.latent_h ~= cfg.latent_h or lat_data.latent_w ~= cfg.latent_w or lat_data.latent_c ~= cfg.latent_c then
    die(string.format(
      "latent dims mismatch: file=%dx%dx%d checkpoint=%dx%dx%d",
      lat_data.latent_h, lat_data.latent_w, lat_data.latent_c,
      cfg.latent_h, cfg.latent_w, cfg.latent_c))
  end

  local ok_create, err_create = Mimir.Model.create("vae_conv_decode", cfg)
  if not ok_create then die("Model.create(vae_conv_decode) failed: " .. tostring(err_create)) end

  apply_dtype(cfg)

  local ok_alloc, nparams_or_err = Mimir.Model.allocate_params()
  if ok_alloc == false then die("Model.allocate_params failed: " .. tostring(nparams_or_err)) end

  local ok_load, err_load = Mimir.Serialization.load(checkpoint_ref, checkpoint_format, {
    load_encoder = false,
    load_tokenizer = false,
    load_optimizer = false,
    strict_mode = false,
    validate_checksums = true,
  })
  if ok_load == false then die("Serialization.load failed: " .. tostring(err_load)) end

  local image_dim = cfg.image_w * cfg.image_h * cfg.image_c

  local pixels, err_fwd = Mimir.Model.forward(lat_data.latent, false)
  if pixels == nil then die("Model.forward(decode) failed: " .. tostring(err_fwd)) end
  if #pixels ~= image_dim then
    die(string.format("unexpected decode output size: got=%d expected=%d", #pixels, image_dim))
  end

  local out_ppm = out_path ~= "" and out_path or "scripts/tests/out_vae_conv_decoded.ppm"
  mkdir_p(dirname(out_ppm))
  local ok_w, err_w = write_ppm_rgb_f32_hwc(out_ppm, pixels, cfg.image_w, cfg.image_h)
  if ok_w == false then die("write_ppm(decode) failed: " .. tostring(err_w)) end
  logx("[test_vae_conv_generate] wrote decoded image " .. tostring(out_ppm))

-- ==========================================================================
-- MODE RECON (défaut) : image PPM → encoder+décodeur → image PPM reconstruite
-- ==========================================================================
else

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

  apply_dtype(cfg)

  local ok_alloc, nparams_or_err = Mimir.Model.allocate_params()
  if ok_alloc == false then die("Model.allocate_params failed: " .. tostring(nparams_or_err)) end

  local ok_load, err_load = Mimir.Serialization.load(checkpoint_ref, checkpoint_format, {
    load_encoder = true,
    load_tokenizer = false,
    load_optimizer = false,
    strict_mode = false,
    validate_checksums = true,
  })
  if ok_load == false then die("Serialization.load failed: " .. tostring(err_load)) end

  mkdir_p(dirname(out_path))

  local image_dim  = cfg.image_w * cfg.image_h * cfg.image_c
  local latent_dim = cfg.latent_dim

  -- Forward VAEConv complet: output pack = recon || mu || logvar
  local packed, err_fwd = Mimir.Model.forward(input_f32, false)
  if packed == nil then die("Model.forward failed: " .. tostring(err_fwd)) end
  local expected_sz = image_dim + 2 * latent_dim
  if #packed ~= expected_sz then
    die(string.format("unexpected output size: got=%d expected=%d (image_dim=%d latent_dim=%d)", #packed, expected_sz, image_dim, latent_dim))
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

end

logx("[test_vae_conv_generate] done")
