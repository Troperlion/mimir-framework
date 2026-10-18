---@diagnostic disable: need-check-nil, inject-field, undefined-global

-- Test bidirectionnel VAEConv text_cond UNIQUEMENT:
--   - prompt -> image  : génération itérative (__input__ + text_ids) avec steps
--   - image  -> texte  : tokens dérivés directement de la sortie forward (pas de liste candidates)

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local function logf(fmt, ...)
  local msg = string.format(fmt, ...)
  if type(log) == "function" then log(msg) else print(msg) end
end

local function die(msg)
  error("[test_text_conditional_bidir] " .. tostring(msg or "error"))
end

local function trim(s)
  s = tostring(s or "")
  s = s:gsub("^%s+", "")
  s = s:gsub("%s+$", "")
  return s
end

local function dirname(path)
  local d = FS.dirname(path)
  if d == nil or d == "" then return "." end
  return d
end

local function mkdir_p(path)
  if path and path ~= "" then FS.mkdir_p(path) end
end

local function clamp(x, a, b)
  if x < a then return a end
  if x > b then return b end
  return x
end

local function first_defined(values, keys)
  for _, key in ipairs(keys) do
    if values[key] ~= nil then return values[key] end
  end
  return nil
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, err = Mimir.model.dtype(dtype)
  if ok == false then die("dtype invalide: " .. tostring(err)) end
  return true
end

local function split_candidates(raw)
  local out = {}
  raw = tostring(raw or "")
  if raw == "" then return out end
  local norm = raw:gsub(";", "|")
  for piece in string.gmatch(norm, "([^|]+)") do
    local p = trim(piece)
    if p ~= "" then out[#out + 1] = p end
  end
  return out
end

local function tokenize_and_decode(text)
  local tok = {}
  local decoded = tostring(text or "")
  if type(Mimir) ~= "table" or type(Mimir.Tokenizer) ~= "table" then
    return tok, decoded
  end

  local tfn = Mimir.Tokenizer.tokenize
  if type(tfn) == "function" then
    local ok_t, ids = pcall(tfn, tostring(text or ""))
    if ok_t and type(ids) == "table" then tok = ids end
  end

  local dfn = Mimir.Tokenizer.detokenize
  if type(dfn) == "function" and type(tok) == "table" and #tok > 0 then
    local ok_d, txt = pcall(dfn, tok)
    if ok_d and txt ~= nil then decoded = tostring(txt) end
  end

  return tok, decoded
end

local function unique_push(dst, seen, v)
  if seen[v] then return end
  seen[v] = true
  dst[#dst + 1] = v
end

local function infer_tokens_from_forward(packed, image_dim, latent_dim, proj_dim, vocab_size, token_count, pad_id)
  local out = {}
  local seen = {}
  local vmin = 0
  local vmax = math.max(1, math.floor(vocab_size - 1))

  local function push_from_value(v, salt)
    local x = tonumber(v) or 0.0
    local u = clamp((x + 1.0) * 0.5, 0.0, 1.0)
    local id = math.floor(u * vmax + 0.5)
    if salt and salt ~= 0 then
      id = (id + math.floor(math.abs(salt))) % (vmax + 1)
    end
    if id == pad_id then return end
    if id < vmin then id = vmin end
    if id > vmax then id = vmax end
    unique_push(out, seen, id)
  end

  local img_off = image_dim + 2 * latent_dim
  if proj_dim > 0 and #packed >= (img_off + proj_dim) then
    for i = 1, proj_dim do
      local v = packed[img_off + i]
      push_from_value(v, i * 131)
      if #out >= token_count then break end
    end
  end

  if #out < token_count then
    local stride = math.max(1, math.floor(image_dim / math.max(1, token_count * 3)))
    local i = 1
    while i <= image_dim and #out < token_count do
      local a = packed[i] or 0.0
      local b = packed[math.min(image_dim, i + stride)] or 0.0
      push_from_value(a - b, i * 17)
      i = i + stride
    end
  end

  return out
end

local function build_zone_vectors(img_hwc_f32, w, h, c, grid)
  local zones = {}
  if c < 3 then return zones end
  grid = math.max(1, math.floor(grid or 8))
  local zw = math.max(1, math.floor(w / grid))
  local zh = math.max(1, math.floor(h / grid))

  for gy = 0, grid - 1 do
    for gx = 0, grid - 1 do
      local x0 = gx * zw + 1
      local y0 = gy * zh + 1
      local x1 = (gx == grid - 1) and w or ((gx + 1) * zw)
      local y1 = (gy == grid - 1) and h or ((gy + 1) * zh)

      local sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
      local sq_r, sq_g, sq_b = 0.0, 0.0, 0.0
      local n = 0

      for y = y0, y1 do
        for x = x0, x1 do
          local idx = ((y - 1) * w + (x - 1)) * c
          local r = tonumber(img_hwc_f32[idx + 1]) or 0.0
          local g = tonumber(img_hwc_f32[idx + 2]) or 0.0
          local b = tonumber(img_hwc_f32[idx + 3]) or 0.0
          sum_r = sum_r + r
          sum_g = sum_g + g
          sum_b = sum_b + b
          sq_r = sq_r + r * r
          sq_g = sq_g + g * g
          sq_b = sq_b + b * b
          n = n + 1
        end
      end

      local inv_n = (n > 0) and (1.0 / n) or 0.0
      local mr = sum_r * inv_n
      local mg = sum_g * inv_n
      local mb = sum_b * inv_n
      local vr = math.max(0.0, sq_r * inv_n - mr * mr)
      local vg = math.max(0.0, sq_g * inv_n - mg * mg)
      local vb = math.max(0.0, sq_b * inv_n - mb * mb)

      zones[#zones + 1] = {
        gx = gx,
        gy = gy,
        vec = { mr, mg, mb, math.sqrt(vr), math.sqrt(vg), math.sqrt(vb) }
      }
    end
  end

  return zones
end

local function zone_corr_sum_near(zones_a, zones_b, grid, near_radius)
  if #zones_a == 0 or #zones_b == 0 then return -1e9 end
  grid = math.max(1, math.floor(grid or 8))
  near_radius = math.max(0, math.floor(near_radius or 1))

  local by_key = {}
  for i = 1, #zones_b do
    local z = zones_b[i]
    by_key[z.gy * grid + z.gx] = z
  end

  local function cos_vec(a, b)
    local dotp, na, nb = 0.0, 0.0, 0.0
    local n = math.min(#a, #b)
    for i = 1, n do
      local va = a[i]
      local vb = b[i]
      dotp = dotp + va * vb
      na = na + va * va
      nb = nb + vb * vb
    end
    na = math.sqrt(math.max(na, 0.0))
    nb = math.sqrt(math.max(nb, 0.0))
    if na < 1e-12 or nb < 1e-12 then return -1.0 end
    return dotp / (na * nb)
  end

  local s = 0.0
  local cnt = 0
  for i = 1, #zones_a do
    local za = zones_a[i]
    local best = -1e9
    for dy = -near_radius, near_radius do
      for dx = -near_radius, near_radius do
        local nx = za.gx + dx
        local ny = za.gy + dy
        if nx >= 0 and nx < grid and ny >= 0 and ny < grid then
          local zb = by_key[ny * grid + nx]
          if zb and zb.vec then
            local c = cos_vec(za.vec, zb.vec)
            if c > best then best = c end
          end
        end
      end
    end
    if best > -1e8 then
      s = s + best
      cnt = cnt + 1
    end
  end
  if cnt == 0 then return -1e9 end
  return s / cnt
end

local function build_token_seed_pool_from_forward(packed, image_dim, latent_dim, proj_dim, vocab_size, max_ids)
  local ids = {}
  local seen = {}
  local vmax = math.max(1, math.floor(vocab_size - 1))
  local function add_id(v, salt)
    local x = tonumber(v) or 0.0
    local u = clamp((x + 1.0) * 0.5, 0.0, 1.0)
    local id = math.floor(u * vmax + 0.5)
    if salt and salt ~= 0 then
      id = (id + math.floor(math.abs(salt))) % (vmax + 1)
    end
    if id < 0 then id = 0 end
    if id > vmax then id = vmax end
    if not seen[id] then
      seen[id] = true
      ids[#ids + 1] = id
    end
  end

  local img_off = image_dim + 2 * latent_dim
  if proj_dim > 0 and #packed >= (img_off + proj_dim) then
    for i = 1, proj_dim do
      add_id(packed[img_off + i], i * 97)
      if #ids >= max_ids then break end
    end
  end

  if #ids < max_ids then
    local stride = math.max(1, math.floor(image_dim / math.max(1, max_ids * 2)))
    local i = 1
    while i <= image_dim and #ids < max_ids do
      local a = packed[i] or 0.0
      local b = packed[math.min(image_dim, i + stride)] or 0.0
      add_id(a - b, i * 31)
      i = i + stride
    end
  end

  return ids
end

local function load_candidates_file(path)
  local out = {}
  local f, err = io.open(path, "r")
  if not f then return nil, err end
  ---@cast f file*
  for line in f:lines() do
    local s = trim(line)
    if s ~= "" and s:sub(1, 1) ~= "#" then out[#out + 1] = s end
  end
  f:close()
  return out, nil
end

local function write_text_file(path, content)
  mkdir_p(dirname(path))
  local f, err = io.open(path, "w")
  if not f then return false, err end
  ---@cast f file*
  f:write(content)
  f:close()
  return true
end

local function write_ppm_rgb_u8(path, pixels, w, h)
  if type(pixels) ~= "table" then die("pixels invalide") end
  w = math.floor(tonumber(w) or 0)
  h = math.floor(tonumber(h) or 0)
  if w <= 0 or h <= 0 then die("w/h invalides") end

  local expected = w * h * 3
  if #pixels ~= expected then
    die("taille pixels invalide: got=" .. tostring(#pixels) .. " expected=" .. tostring(expected))
  end

  mkdir_p(dirname(path))
  local f, err = io.open(path, "wb")
  if not f then die("open(" .. tostring(path) .. ") a echoue: " .. tostring(err)) end
  ---@cast f file*
  f:write(string.format("P6\n%d %d\n255\n", w, h))

  local chunk, n = {}, 0
  local CHUNK = 8192
  for i = 1, #pixels do
    local v = math.floor(tonumber(pixels[i]) or 0)
    if v < 0 then v = 0 end
    if v > 255 then v = 255 end
    n = n + 1
    chunk[n] = string.char(v)
    if n >= CHUNK then
      f:write(table.concat(chunk))
      n = 0
    end
  end
  if n > 0 then f:write(table.concat(chunk, "", 1, n)) end
  f:close()
end

local function read_ppm(path)
  local f, err = io.open(path, "rb")
  if not f then return nil, err end
  ---@cast f file*

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
        local _ = f:read("*l")
      elseif c:match("%s") then
      else
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
    return nil, "format image non supporte (PPM P6/P3 attendu)"
  end

  local w = tonumber(read_token() or "")
  local h = tonumber(read_token() or "")
  local maxval = tonumber(read_token() or "")
  if not w or not h or not maxval then
    f:close()
    return nil, "header PPM invalide"
  end

  w = math.floor(w)
  h = math.floor(h)
  maxval = math.floor(maxval)
  if w <= 0 or h <= 0 then
    f:close()
    return nil, "dimensions invalides"
  end
  if maxval <= 0 or maxval > 255 then
    f:close()
    return nil, "maxval invalide"
  end

  local expected = w * h * 3
  local pixels = {}
  pixels[expected] = 0

  if magic == "P6" then
    skip_ws_and_comments()
    local data = f:read(expected)
    f:close()
    if not data or #data ~= expected then
      return nil, "payload P6 tronque"
    end
    local scale = 255.0 / maxval
    for i = 1, expected do
      local v = string.byte(data, i) or 0
      if maxval ~= 255 then v = math.floor(v * scale + 0.5) end
      pixels[i] = v
    end
    return { w = w, h = h, pixels = pixels }, nil
  end

  for i = 1, expected do
    local t = read_token()
    if t == nil then
      f:close()
      return nil, "payload P3 tronque"
    end
    local v = math.floor(tonumber(t) or 0)
    if v < 0 then v = 0 end
    if v > maxval then v = maxval end
    pixels[i] = math.floor((v / maxval) * 255.0 + 0.5)
  end
  f:close()
  return { w = w, h = h, pixels = pixels }, nil
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

local function f32_minus1_1_to_u8_rgb(src, image_dim)
  local out = {}
  out[image_dim] = 0
  for i = 1, image_dim do
    local v = tonumber(src[i]) or 0.0
    if v < -1.0 then v = -1.0 end
    if v > 1.0 then v = 1.0 end
    out[i] = math.floor((v + 1.0) * 127.5 + 0.5)
  end
  return out
end

local function dot(a, b)
  local n = math.min(#a, #b)
  local s = 0.0
  for i = 1, n do s = s + (a[i] * b[i]) end
  return s
end

local function norm2(a)
  local s = 0.0
  for i = 1, #a do s = s + (a[i] * a[i]) end
  return math.sqrt(math.max(0.0, s))
end

local function cosine(a, b)
  local na = norm2(a)
  local nb = norm2(b)
  if na < 1e-12 or nb < 1e-12 then return -1.0 end
  return dot(a, b) / (na * nb)
end

local function rand_uniform_signed(scale)
  local r = math.random() * 2.0 - 1.0
  return r * scale
end

local function infer_cfg_from_checkpoint(ckpt_dir)
  local arch_path = tostring(ckpt_dir) .. "/model/architecture.json"
  local arch = read_json(arch_path)
  if type(arch) ~= "table" then
    return nil, "read_json failed: " .. tostring(arch_path)
  end

  local mc = arch.model_config or arch.modelConfig
  if type(mc) ~= "table" or (tonumber(mc.image_w) or 0) <= 0 then
    return nil, "model_config manquant dans architecture.json"
  end

  local function mci(k)
    local n = tonumber(mc[k] or 0) or 0
    return math.floor(n)
  end

  return {
    image_w = mci("image_w"),
    image_h = mci("image_h"),
    image_c = math.max(1, mci("image_c")),
    latent_h = mci("latent_h"),
    latent_w = mci("latent_w"),
    latent_c = mci("latent_c"),
    base_channels = mci("base_channels"),
    vocab_size = mci("vocab_size"),
    proj_dim = mci("proj_dim"),
    seq_len = mci("seq_len"),
    text_d_model = mci("text_d_model"),
    resnet = first_defined(mc, {"resnet", "use_resnet", "use_attention"}),
    attention = first_defined(mc, {"attention", "use_attn"}),
    use_skip_connections = mc.use_skip_connections,
    use_encoder_prior = mc.use_encoder_prior,
    decoder_upsample = mc.decoder_upsample,
    stochastic_latent = mc.stochastic_latent,
  }, nil
end

local function pad_or_trim(ids, seq_len, pad_id)
  local out = {}
  local n = math.min(#ids, seq_len)
  for i = 1, n do out[i] = ids[i] end
  for i = n + 1, seq_len do out[i] = pad_id end
  return out
end

local function main()
  local opts = Args.parse(arg)

  local mode = trim(Args.get_str(opts, "mode", "auto"))
  local checkpoint = trim(Args.get_str(opts, "checkpoint", ""))
  local tokenizer_path = trim(Args.get_str(opts, "tokenizer", ""))

  local prompt = Args.get_str(opts, "prompt", "")
  local in_image = Args.get_str(opts, "in-image", "")
  local out_image = Args.get_str(opts, "out-image", "scripts/tests/out_bidir_vaeconv.ppm")
  local out_text = Args.get_str(opts, "out-text", "scripts/tests/out_bidir_vaeconv.txt")
  local topk = math.max(1, Args.get_int(opts, "topk", 5))
  local img2txt_token_count = math.max(1, Args.get_int(opts, "img2txt-token-count", topk))
  local img2txt_zone_grid = math.max(2, Args.get_int(opts, "img2txt-zone-grid", 8))
  local img2txt_near_radius = math.max(0, Args.get_int(opts, "img2txt-near-radius", 1))
  local img2txt_seed_pool = math.max(img2txt_token_count, Args.get_int(opts, "img2txt-seed-pool", 32))
  local txt2img_samples = math.max(1, Args.get_int(opts, "txt2img-samples", 24))
  local txt2img_steps = math.max(1, Args.get_int(opts, "txt2img-steps", 8))
  local txt2img_noise = math.max(0.0, Args.get_num(opts, "txt2img-noise", 1.0))
  local txt2img_step_blend = clamp(Args.get_num(opts, "txt2img-step-blend", 0.75), 0.0, 1.0)
  local txt2img_seed = Args.get_int(opts, "txt2img-seed", os.time())

  if checkpoint == "" then die("--checkpoint requis (vae_conv text_cond)") end

  local do_txt2img, do_img2txt = false, false
  if mode == "auto" then
    do_txt2img = (trim(prompt) ~= "")
    do_img2txt = (trim(in_image) ~= "")
  elseif mode == "txt2img" then
    do_txt2img = true
  elseif mode == "img2txt" then
    do_img2txt = true
  elseif mode == "both" then
    do_txt2img = true
    do_img2txt = true
  else
    die("--mode invalide: auto|txt2img|img2txt|both")
  end

  if do_txt2img and trim(prompt) == "" then die("mode txt2img: --prompt requis") end
  if do_img2txt and trim(in_image) == "" then die("mode img2txt: --in-image requis") end
  if not do_txt2img and not do_img2txt then die("rien a faire") end

  if tokenizer_path == "" then
    tokenizer_path = checkpoint .. "/tokenizer/tokenizer.json"
  end

  local inferred, err_inf = infer_cfg_from_checkpoint(checkpoint)
  if type(inferred) ~= "table" then die("infer_cfg_from_checkpoint failed: " .. tostring(err_inf)) end

  local cfg, err_cfg = Mimir.Architectures.default_config("vae_conv")
  if type(cfg) ~= "table" then die("default_config(vae_conv) failed: " .. tostring(err_cfg)) end

  cfg.image_w = inferred.image_w
  cfg.image_h = inferred.image_h
  cfg.image_c = inferred.image_c
  cfg.latent_h = inferred.latent_h
  cfg.latent_w = inferred.latent_w
  cfg.latent_c = inferred.latent_c
  cfg.base_channels = inferred.base_channels
  cfg.text_cond = true
  cfg.seq_len = (inferred.seq_len and inferred.seq_len > 0) and inferred.seq_len or (cfg.seq_len or 64)
  cfg.proj_dim = (inferred.proj_dim and inferred.proj_dim > 0) and inferred.proj_dim or (cfg.proj_dim or 64)
  cfg.text_d_model = (inferred.text_d_model and inferred.text_d_model > 0) and inferred.text_d_model or (cfg.text_d_model or 64)
  if (inferred.vocab_size or 0) > 0 then
    cfg.vocab_size = inferred.vocab_size
  else
    cfg.vocab_size = math.floor(tonumber(cfg.vocab_size or 65536) or 65536)
  end
  cfg.resnet = inferred.resnet
  cfg.attention = inferred.attention
  cfg.use_skip_connections = inferred.use_skip_connections
  cfg.use_encoder_prior = inferred.use_encoder_prior
  cfg.decoder_upsample = inferred.decoder_upsample
  cfg.stochastic_latent = inferred.stochastic_latent
  local latent_tokens = math.max(1, cfg.latent_h * cfg.latent_w)
  cfg.attn_max_tokens = latent_tokens
  cfg.resnet_max_tokens = latent_tokens

  local ok_create, err_create = Mimir.Model.create("vae_conv", cfg)
  if ok_create == false then die("Model.create: " .. tostring(err_create)) end
  apply_dtype(cfg)

  local ok_alloc, err_alloc = Mimir.Model.allocate_params()
  if ok_alloc == false then die("allocate_params: " .. tostring(err_alloc)) end

  local ok_load, err_load = Mimir.Serialization.load(checkpoint, "raw_folder", {
    load_tokenizer = true,
    load_encoder = true,
    load_optimizer = false,
    strict_mode = false,
    validate_checksums = true,
  })
  if ok_load == false then die("Serialization.load: " .. tostring(err_load)) end

  -- Optionnel: surcharger le tokenizer par un fichier explicite APRES le load checkpoint.
  -- Si le vocab externe dépasse la capacité embeddings du checkpoint, on le borne.
  if tokenizer_path ~= "" then
    local ok_tok, err_tok = Mimir.Tokenizer.load(tokenizer_path)
    if ok_tok == false then die("Tokenizer.load: " .. tostring(err_tok)) end
    if Mimir.Tokenizer.set_max_vocab and cfg.vocab_size and cfg.vocab_size > 0 then
      pcall(Mimir.Tokenizer.set_max_vocab, cfg.vocab_size)
    end
  end

  local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
  local latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c
  local seq_len = math.max(1, math.floor(cfg.seq_len or 64))
  local pad_id = 0
  local seq_id = 2
  local unk_id = 1
  local tok_tbl = (type(Mimir) == "table") and Mimir.Tokenizer or nil
  local tok_get_pad = (type(tok_tbl) == "table") and rawget(tok_tbl, "pad_id") or nil
  if type(tok_get_pad) == "function" then
    local ok_pad, v_pad = pcall(tok_get_pad)
    if ok_pad then pad_id = tonumber(v_pad) or 0 end
  end
  local tok_get_seq = (type(tok_tbl) == "table") and rawget(tok_tbl, "seq_id") or nil
  if type(tok_get_seq) == "function" then
    local ok_seq, v_seq = pcall(tok_get_seq)
    if ok_seq then seq_id = tonumber(v_seq) or seq_id end
  end
  local tok_get_unk = (type(tok_tbl) == "table") and rawget(tok_tbl, "unk_id") or nil
  if type(tok_get_unk) == "function" then
    local ok_unk, v_unk = pcall(tok_get_unk)
    if ok_unk then unk_id = tonumber(v_unk) or unk_id end
  end
  local vocab_size = math.max(2, math.floor(tonumber((Mimir.Tokenizer and Mimir.Tokenizer.vocab_size and Mimir.Tokenizer.vocab_size()) or cfg.vocab_size or 32000) or 32000))

  local function run_forward_with_prompt(x_f32, p)
    local ids = Mimir.Tokenizer.tokenize(p)
    if type(ids) ~= "table" then ids = {} end
    local tids = pad_or_trim(ids, seq_len, pad_id)
    return Mimir.Model.forward({ __input__ = x_f32, text_ids = tids }, false)
  end

  local function run_forward_with_ids(x_f32, tids)
    return Mimir.Model.forward({ __input__ = x_f32, text_ids = tids }, false)
  end

  if do_txt2img then
    math.randomseed(txt2img_seed)
    local best_recon = nil
    local best_score = -1e9
    local had_success = false

    for s = 1, txt2img_samples do
      local x_cur = {}
      x_cur[image_dim] = 0.0
      for i = 1, image_dim do x_cur[i] = rand_uniform_signed(txt2img_noise) end

      local packed = nil
      local err_fwd = nil
      local sample_ok = false
      for t = 1, txt2img_steps do
        packed, err_fwd = run_forward_with_prompt(x_cur, prompt)
        if not packed or #packed < image_dim then
          break
        end
        sample_ok = true

        if t < txt2img_steps then
          local frac = (txt2img_steps > 1) and ((t - 1) / (txt2img_steps - 1)) or 1.0
          local anneal = (1.0 - frac) * txt2img_noise * 0.15
          local keep = 1.0 - txt2img_step_blend
          for i = 1, image_dim do
            local recon_i = tonumber(packed[i]) or 0.0
            local prev_i = tonumber(x_cur[i]) or 0.0
            local mixed = keep * prev_i + txt2img_step_blend * recon_i + rand_uniform_signed(anneal)
            x_cur[i] = clamp(mixed, -1.0, 1.0)
          end
        end
      end

      if sample_ok and packed and #packed >= image_dim then
        had_success = true
        local score = -1e6
        local proj_dim = math.floor((#packed - image_dim - 2 * latent_dim) / 2)
        if proj_dim > 0 and (#packed >= image_dim + 2 * latent_dim + 2 * proj_dim) then
          local img_off = image_dim + 2 * latent_dim
          local txt_off = img_off + proj_dim
          local img_proj, txt_proj = {}, {}
          for d = 1, proj_dim do
            img_proj[d] = tonumber(packed[img_off + d]) or 0.0
            txt_proj[d] = tonumber(packed[txt_off + d]) or 0.0
          end
          score = cosine(img_proj, txt_proj)
        else
          -- Fallback: éviter un résultat trop plat si le tail proj n'est pas présent.
          local mean = 0.0
          for i = 1, image_dim do mean = mean + (tonumber(packed[i]) or 0.0) end
          mean = mean / image_dim
          local var = 0.0
          for i = 1, image_dim do
            local dv = (tonumber(packed[i]) or 0.0) - mean
            var = var + dv * dv
          end
          score = var / image_dim
        end

        if score > best_score then
          best_score = score
          best_recon = {}
          for i = 1, image_dim do best_recon[i] = tonumber(packed[i]) or 0.0 end
        end
      elseif err_fwd and s == 1 then
        logf("[bidir-vae_conv] txt2img sample#%d error: %s", s, tostring(err_fwd))
      end
    end

    if not had_success or type(best_recon) ~= "table" then
      die("txt2img failed: aucun échantillon valide")
    end

    local recon_u8 = f32_minus1_1_to_u8_rgb(best_recon, image_dim)
    write_ppm_rgb_u8(out_image, recon_u8, cfg.image_w, cfg.image_h)
    logf("[bidir-vae_conv] txt2img samples=%d steps=%d blend=%.3f noise=%.3f seed=%d best_score=%.6f", txt2img_samples, txt2img_steps, txt2img_step_blend, txt2img_noise, txt2img_seed, best_score)
    logf("[bidir-vae_conv] prompt->image OK: %s", out_image)
  end

  if do_img2txt then
    local img, err_img = read_ppm(in_image)
    if not img then die("img2txt read image failed: " .. tostring(err_img)) end

    local in_u8 = img.pixels
    if img.w ~= cfg.image_w or img.h ~= cfg.image_h then
      in_u8 = resize_rgb_u8_nearest(in_u8, img.w, img.h, cfg.image_w, cfg.image_h)
    end
    local x = rgb_u8_to_f32_minus1_1(in_u8)
    local input_zones = build_zone_vectors(x, cfg.image_w, cfg.image_h, cfg.image_c, img2txt_zone_grid)

    local neutral_ids = {}
    neutral_ids[1] = seq_id
    for i = 2, seq_len do neutral_ids[i] = pad_id end

    local packed, err_fwd = run_forward_with_ids(x, neutral_ids)
    if not packed then die("img2txt forward failed: " .. tostring(err_fwd)) end

    local proj_dim = math.floor((#packed - image_dim - 2 * latent_dim) / 2)
    if proj_dim < 0 then proj_dim = 0 end

    local seed_ids = build_token_seed_pool_from_forward(
      packed,
      image_dim,
      latent_dim,
      proj_dim,
      vocab_size,
      img2txt_seed_pool
    )

    local scored = {}
    for i = 1, #seed_ids do
      local tid = seed_ids[i]
      local tids = {}
      tids[1] = seq_id
      tids[2] = tid
      for k = 3, seq_len do tids[k] = pad_id end

      local pred, err_pred = run_forward_with_ids(x, tids)
      if pred and #pred >= image_dim then
        local recon = {}
        recon[image_dim] = 0.0
        for j = 1, image_dim do recon[j] = tonumber(pred[j]) or 0.0 end
        local recon_zones = build_zone_vectors(recon, cfg.image_w, cfg.image_h, cfg.image_c, img2txt_zone_grid)
        local zcorr = zone_corr_sum_near(input_zones, recon_zones, img2txt_zone_grid, img2txt_near_radius)

        local pproj = math.floor((#pred - image_dim - 2 * latent_dim) / 2)
        local gcos = -1.0
        if pproj > 0 and (#pred >= image_dim + 2 * latent_dim + 2 * pproj) then
          local img_off = image_dim + 2 * latent_dim
          local txt_off = img_off + pproj
          local img_proj, txt_proj = {}, {}
          for d = 1, pproj do
            img_proj[d] = tonumber(pred[img_off + d]) or 0.0
            txt_proj[d] = tonumber(pred[txt_off + d]) or 0.0
          end
          gcos = cosine(img_proj, txt_proj)
        end

        local score = 0.75 * zcorr + 0.25 * gcos
        scored[#scored + 1] = { token = tid, score = score, zcorr = zcorr, gcos = gcos }
      elseif err_pred and i == 1 then
        logf("[bidir-vae_conv] img2txt token-eval error: %s", tostring(err_pred))
      end
    end

    table.sort(scored, function(a, b) return (a.score or -1e9) > (b.score or -1e9) end)

    local token_ids = {}
    local seen_tok = {}
    for i = 1, #scored do
      local tid = scored[i].token
      if tid ~= pad_id and not seen_tok[tid] then
        seen_tok[tid] = true
        token_ids[#token_ids + 1] = tid
      end
      if #token_ids >= img2txt_token_count then break end
    end
    if #token_ids == 0 then
      token_ids = infer_tokens_from_forward(
        packed,
        image_dim,
        latent_dim,
        proj_dim,
        vocab_size,
        img2txt_token_count,
        pad_id
      )
    end
    if #token_ids == 0 then token_ids = { unk_id } end

    local decoded = ""
    local ok_dec, dec_or_err = pcall(Mimir.Tokenizer.detokenize, token_ids)
    if ok_dec and dec_or_err ~= nil then decoded = tostring(dec_or_err) else decoded = "[decode failed]" end

    local lines = {}
    lines[#lines + 1] = "source=forward_output"
    lines[#lines + 1] = "decoded=" .. decoded
    lines[#lines + 1] = "tokens=" .. table.concat(token_ids, " ")
    lines[#lines + 1] = "token_count=" .. tostring(#token_ids)
    lines[#lines + 1] = "vocab_size=" .. tostring(vocab_size)
    lines[#lines + 1] = "zone_grid=" .. tostring(img2txt_zone_grid)
    lines[#lines + 1] = "near_radius=" .. tostring(img2txt_near_radius)
    lines[#lines + 1] = ""
    lines[#lines + 1] = "note=correspondance texte-image par zones proches + somme de corrélations vectorielles"
    if #scored > 0 then
      lines[#lines + 1] = ""
      lines[#lines + 1] = "top_token_scores:"
      local ks = math.min(#scored, img2txt_token_count)
      for i = 1, ks do
        local r = scored[i]
        lines[#lines + 1] = string.format("%d. token=%d | score=%.6f | zcorr=%.6f | gcos=%.6f", i, r.token, r.score, r.zcorr, r.gcos)
      end
    end

    local ok_txt, err_txt = write_text_file(out_text, table.concat(lines, "\n") .. "\n")
    if not ok_txt then die("write out-text failed: " .. tostring(err_txt)) end

    logf("[bidir-vae_conv] image->texte OK: %s", out_text)
    logf("[bidir-vae_conv] decoded: %s", tostring(decoded))
  end

  logf("[bidir-vae_conv] done")
end

main()
