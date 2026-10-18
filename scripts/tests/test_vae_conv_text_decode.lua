-- Génération d'image VAEConv à partir d'un texte (injection latent).
--
-- Le VAEConv n'ayant pas de conditionnement texte, le texte est converti
-- en seed déterministe (FNV-1a) puis utilisé pour échantillonner un vecteur
-- latent gaussien N(0, sigma) qui est injecté directement dans le décodeur
-- (architecture vae_conv_decode, pas besoin d'encodeur).
--
-- Usage:
--   ./bin/mimir --lua scripts/tests/test_vae_conv_text_decode.lua -- \
--     --text "a cat sitting on a red sofa" \
--     --checkpoint checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop \
--     --out scripts/tests/out_text_decode.ppm
--
-- Options:
--   --text <str>              Texte / prompt (requis)
--   --checkpoint <dir>        Checkpoint VAEConv RawFolder (requis)
--   --out <path>              Chemin PPM de sortie (défaut: scripts/tests/out_text_decode.ppm)
--   --seed <n>                Seed additif ajouté au hash du texte (défaut: 0)
--   --sigma <f>               Écart-type du bruit latent (défaut: 1.0)
--   --interp-text <str>       Second texte pour interpolation sphérique (optionnel)
--   --interp-alpha <f>        Coefficient d'interpolation [0..1] (défaut: 0.5)
--   --alloc-gb <n>            RAM max pour l'allocateur (défaut: 8)

---@diagnostic disable: need-check-nil, inject-field, undefined-global

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

-- ---------------------------------------------------------------------------
-- Logging / erreur
-- ---------------------------------------------------------------------------
local function logx(msg)
    local l = rawget(_G, "log")
    if type(l) == "function" then l(msg) else print(msg) end
end

local function logf(fmt, ...)
    logx(string.format(fmt, ...))
end

local function die(msg)
    error("[test_vae_conv_text_decode] " .. tostring(msg or "error"))
end

-- ---------------------------------------------------------------------------
-- Helpers d'options
-- ---------------------------------------------------------------------------
local function opt_str(k, d)
    local v = opts[k]
    if v == nil or v == true then return d end
    return tostring(v)
end

local function opt_num(k, d)
    local v = opts[k]
    if v == nil then return d end
    local n = tonumber(v)
    return n ~= nil and n or d
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

-- ---------------------------------------------------------------------------
-- Hash FNV-1a 32 bits (déterministe, sans dépendances)
-- Produit un entier dans [1 .. 2^31-1] pour initialiser le LCG.
-- ---------------------------------------------------------------------------
local function fnv1a32(s)
    s = tostring(s or "")
    local h = 2166136261  -- FNV offset basis
    for i = 1, #s do
        local b = string.byte(s, i)
        h = bit32.bxor(h, b)
        -- multiplication modulo 2^32 via accumulation (Lua 5.3+ integer)
        -- FNV prime = 16777619
        h = (h * 16777619) & 0xFFFFFFFF
    end
    -- Ramener dans [1..2^31-1] pour le LCG
    h = h & 0x7FFFFFFF
    if h == 0 then h = 1 end
    return h
end

-- Fallback si bit32 absent (Lua 5.4 ne l'a plus): version purement arithmétique.
local _has_bit32 = (type(bit32) == "table" and type(bit32.bxor) == "function")
if not _has_bit32 then
    fnv1a32 = function(s)
        s = tostring(s or "")
        local h = 2166136261
        for i = 1, #s do
            local b = string.byte(s, i)
            -- XOR sur 32 bits
            h = h ~ b
            -- multiplication mod 2^32
            local lo = h & 0xFFFF
            local hi = (h >> 16) & 0xFFFF
            local prime = 16777619
            local new_lo = (lo * prime) & 0xFFFF
            local new_hi = ((hi * prime) + math.floor(lo * prime / 65536)) & 0xFFFF
            h = (new_hi << 16) | new_lo
        end
        h = h & 0x7FFFFFFF
        if h == 0 then h = 1 end
        return h
    end
end

-- ---------------------------------------------------------------------------
-- RNG déterministe LCG + Box-Muller N(0,1)
-- ---------------------------------------------------------------------------
local function make_rng(seed)
    local state = tonumber(seed) or 1
    state = math.floor(state) % 2147483647
    if state <= 0 then state = 123456789 end

    local function rand_u32()
        state = (1103515245 * state + 12345) % 2147483647
        return state
    end

    local function rand01()
        return rand_u32() / 2147483647.0
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

    return { rand01 = rand01, randn = randn }
end

-- ---------------------------------------------------------------------------
-- Générer un vecteur latent gaussien à partir d'un texte
-- ---------------------------------------------------------------------------
local function text_to_latent(text, latent_dim, sigma, extra_seed)
    local hash = fnv1a32(tostring(text or ""))
    local seed = hash + math.floor(tonumber(extra_seed) or 0)
    local rng = make_rng(seed)
    local z = {}
    z[latent_dim] = 0.0
    local s = tonumber(sigma) or 1.0
    for i = 1, latent_dim do
        z[i] = rng.randn() * s
    end
    return z, seed
end

-- ---------------------------------------------------------------------------
-- Interpolation sphérique (slerp) entre deux vecteurs latents.
-- Retombe sur lerp si les vecteurs sont quasi-colinéaires.
-- ---------------------------------------------------------------------------
local function slerp(z0, z1, alpha)
    local n = math.min(#z0, #z1)

    -- Normes
    local n0, n1 = 0.0, 0.0
    for i = 1, n do
        n0 = n0 + z0[i] * z0[i]
        n1 = n1 + z1[i] * z1[i]
    end
    n0 = math.sqrt(n0)
    n1 = math.sqrt(n1)
    if n0 < 1e-12 or n1 < 1e-12 then
        -- lerp de secours
        local out = {}
        for i = 1, n do out[i] = z0[i] * (1 - alpha) + z1[i] * alpha end
        return out
    end

    -- Produit scalaire normalisé → cosinus
    local dot = 0.0
    for i = 1, n do
        dot = dot + (z0[i] / n0) * (z1[i] / n1)
    end
    dot = clamp(dot, -1.0, 1.0)

    local theta = math.acos(dot)
    if theta < 1e-6 then
        -- lerp
        local out = {}
        for i = 1, n do out[i] = z0[i] * (1 - alpha) + z1[i] * alpha end
        return out
    end

    local sin_theta = math.sin(theta)
    local s0 = math.sin((1.0 - alpha) * theta) / sin_theta
    local s1 = math.sin(alpha * theta) / sin_theta

    local out = {}
    for i = 1, n do
        out[i] = z0[i] * s0 + z1[i] * s1
    end
    return out
end

-- ---------------------------------------------------------------------------
-- Inférer la config VAEConv depuis un checkpoint RawFolder
-- (copié depuis test_vae_conv_generate.lua)
-- ---------------------------------------------------------------------------
local function infer_cfg_from_checkpoint(ckpt_dir)
    local arch_path = tostring(ckpt_dir) .. "/model/architecture.json"
    local arch = read_json(arch_path)
    if type(arch) ~= "table" then
        return nil, "read_json failed: " .. tostring(arch_path)
    end

    local mc = arch.model_config or arch.modelConfig
    if type(mc) == "table" and (tonumber(mc.image_w) or 0) > 0 then
        local function mci(k)
            local v = tonumber(mc[k])
            return math.floor(v or 0)
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

    -- Repli legacy
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
            if type(L) == "table" and L.name == name then return L end
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
        local in_c = tonumber(enc_conv_in.in_channels)
        if in_c then image_c = math.max(1, math.floor(in_c)) end
    end

    local base_channels = tonumber(dec_conv_in.out_channels) or 0
    local latent_c      = tonumber(dec_conv_in.in_channels)  or 0
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
    if (image_h % div) ~= 0 or (image_w % div) ~= 0 then
        return nil, string.format(
            "cannot infer latent_h/w: image not divisible by 2^%d (%dx%d)", downsamples, image_w, image_h)
    end

    return {
        image_w       = image_w,
        image_h       = image_h,
        image_c       = image_c,
        latent_h      = math.floor(image_h / div),
        latent_w      = math.floor(image_w / div),
        latent_c      = math.floor(latent_c),
        base_channels = math.floor(base_channels),
        downsamples   = downsamples,
    }, nil
end

-- ---------------------------------------------------------------------------
-- Écriture PPM P6 depuis un buffer float HWC en [-1,1]
-- ---------------------------------------------------------------------------
local function write_ppm_rgb_f32_hwc(path, pixels, w, h)
    if type(pixels) ~= "table" then return false, "pixels must be table" end
    w = math.floor(tonumber(w) or 0)
    h = math.floor(tonumber(h) or 0)
    if w <= 0 or h <= 0 then return false, "invalid w/h" end
    local expected = w * h * 3
    if #pixels ~= expected then
        return false, string.format("invalid pixel buffer: got=%d expected=%d", #pixels, expected)
    end

    local f, ferr = io.open(path, "wb")
    if not f then return false, ferr end

    f:write(string.format("P6\n%d %d\n255\n", w, h))

    local CHUNK = 8192
    local buf = {}
    local n = 0
    for i = 1, expected do
        local x = tonumber(pixels[i]) or 0.0
        local t = 0.5 + 0.5 * x
        local p = math.floor(clamp(t, 0.0, 1.0) * 255.0 + 0.5)
        n = n + 1
        buf[n] = string.char(clamp(p, 0, 255))
        if n >= CHUNK then
            f:write(table.concat(buf))
            buf = {}
            n = 0
        end
    end
    if n > 0 then f:write(table.concat(buf)) end
    f:close()
    return true, nil
end

-- ---------------------------------------------------------------------------
-- Appliquer un dtype optionnel (env MIMIR_DTYPE ou cfg.dtype)
-- ---------------------------------------------------------------------------
local function apply_dtype(cfg)
    local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
    if dtype == nil then return true end
    if type(Mimir) ~= "table" or type(Mimir.model) ~= "table"
            or type(Mimir.model.dtype) ~= "function" then
        return true
    end
    local ok, dt_or_err = Mimir.model.dtype(dtype)
    if ok == false then die("dtype invalide: " .. tostring(dt_or_err)) end
    return true
end

-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------
local DEFAULT_CKPT = "checkpoint/vae_conv_base_tok_latent-128-2/epoch_0024_stop"

local text1        = opt_str("text",         opt_str("prompt", ""))
local text2        = opt_str("interp-text",  opt_str("interp_text", ""))
local interp_alpha = opt_num("interp-alpha", opt_num("interp_alpha", 0.5))
local checkpoint   = opt_str("checkpoint",   opt_str("ckpt", DEFAULT_CKPT))
local out_path     = opt_str("out", "scripts/tests/out_text_decode.ppm")
local extra_seed   = opt_int("seed", 0)
local sigma        = opt_num("sigma", 1.0)

if text1 == "" then
    die("--text requis (ex: --text \"a cat on a sofa\")")
end

-- Mémoire
if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
    local mem_gb     = opt_num("alloc-gb", opt_num("mem-gb", 8))
    local compression = opts.compression
    if compression == nil then compression = opts.compress end
    if compression == nil then compression = true end
    Mimir.Allocator.configure({ max_ram_gb = mem_gb, enable_compression = compression })
end

logf('[test_vae_conv_text_decode] text="%s"', text1)
if text2 ~= "" then
    logf('[test_vae_conv_text_decode] interp-text="%s" alpha=%.3f', text2, interp_alpha)
end
logf("[test_vae_conv_text_decode] checkpoint=%s", checkpoint)
logf("[test_vae_conv_text_decode] sigma=%.4f  extra-seed=%d", sigma, extra_seed)

-- Inférer la config depuis le checkpoint
local inferred, err_inf = infer_cfg_from_checkpoint(checkpoint)
if not inferred then die("infer_cfg_from_checkpoint: " .. tostring(err_inf)) end

-- Construire la config pour le décodeur seul
local cfg = Mimir.Architectures.default_config("vae_conv")
if type(cfg) ~= "table" then die("default_config(vae_conv) échoué") end

local cfg_fields = {
    "image_w", "image_h", "image_c",
    "latent_h", "latent_w", "latent_c", "base_channels",
    "resnet", "attention", "resnet_max_tokens",
    "use_skip_connections", "use_encoder_prior", "decoder_upsample",
}
for _, k in ipairs(cfg_fields) do
    if inferred[k] ~= nil then cfg[k] = inferred[k] end
end
cfg.text_cond       = false
cfg.stochastic_latent = false
cfg.latent_dim      = cfg.latent_h * cfg.latent_w * cfg.latent_c

-- Surcharges manuelles optionnelles
if opts["image-w"]      then cfg.image_w      = opt_int("image-w",      cfg.image_w) end
if opts["image-h"]      then cfg.image_h      = opt_int("image-h",      cfg.image_h) end
if opts["latent-h"]     then cfg.latent_h     = opt_int("latent-h",     cfg.latent_h) end
if opts["latent-w"]     then cfg.latent_w     = opt_int("latent-w",     cfg.latent_w) end
if opts["latent-c"]     then cfg.latent_c     = opt_int("latent-c",     cfg.latent_c) end
if opts["base-channels"] then cfg.base_channels = opt_int("base-channels", cfg.base_channels) end
cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c

logf("[test_vae_conv_text_decode] cfg image=%dx%dx%d latent=%dx%dx%d (dim=%d) base=%d",
    cfg.image_w, cfg.image_h, cfg.image_c,
    cfg.latent_h, cfg.latent_w, cfg.latent_c, cfg.latent_dim,
    cfg.base_channels)

-- Générer le vecteur latent à partir du texte
local z, used_seed = text_to_latent(text1, cfg.latent_dim, sigma, extra_seed)
logf("[test_vae_conv_text_decode] latent généré: hash+seed=%d  dim=%d  sigma=%.4f", used_seed, #z, sigma)

-- Interpolation sphérique avec un second texte (optionnel)
if text2 ~= "" then
    local z2, seed2 = text_to_latent(text2, cfg.latent_dim, sigma, extra_seed)
    logf('[test_vae_conv_text_decode] second latent (text2): hash+seed=%d', seed2)
    z = slerp(z, z2, interp_alpha)
    logf("[test_vae_conv_text_decode] slerp alpha=%.3f appliqué", interp_alpha)
end

-- Créer, charger et exécuter le décodeur seul (vae_conv_decode)
local ok_create, err_create = Mimir.Model.create("vae_conv_decode", cfg)
if not ok_create then die("Model.create(vae_conv_decode) échoué: " .. tostring(err_create)) end

apply_dtype(cfg)

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
if ok_alloc == false then die("Model.allocate_params échoué: " .. tostring(err_alloc)) end

local ok_load, err_load = Mimir.Serialization.load(checkpoint, "raw_folder", {
    load_encoder    = false,
    load_tokenizer  = false,
    load_optimizer  = false,
    strict_mode     = false,
    validate_checksums = true,
})
if ok_load == false then die("Serialization.load échoué: " .. tostring(err_load)) end

-- Inférence : le décodeur prend z (latent_dim) et retourne recon (image_dim)
local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local pixels, err_fwd = Mimir.Model.forward(z, false)
if pixels == nil then die("Model.forward échoué: " .. tostring(err_fwd)) end
if #pixels ~= image_dim then
    die(string.format("taille de sortie inattendue: got=%d expected=%d", #pixels, image_dim))
end

-- Écriture PPM
mkdir_p(dirname(out_path))
local ok_w, err_w = write_ppm_rgb_f32_hwc(out_path, pixels, cfg.image_w, cfg.image_h)
if ok_w == false then die("write_ppm échoué: " .. tostring(err_w)) end

logf('[test_vae_conv_text_decode] image écrite → %s (%dx%d)', out_path, cfg.image_w, cfg.image_h)
logx("[test_vae_conv_text_decode] done")
