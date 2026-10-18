-- Convertir un checkpoint RawFolder (dossier) en SafeTensors (.safetensors)
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/convert_checkpoint2safetensor.lua \
--       --checkpoint /path/to/checkpoint_dir --out /path/to/model.safetensors
--
-- Notes:
--   - Le format RawFolder requiert un modèle déjà créé/allocaté avant `Serialization.load`.
--   - Ce script lit `architecture.json` dans le dossier, infère le type de modèle et
--     reconstruit une config minimale (spécialement pour `vae_conv`).

---@diagnostic disable: undefined-field, need-check-nil, param-type-mismatch

local ToolHelp = dofile(ROOTWORK.."/scripts/modules/tools_help.lua")
ToolHelp.show("convert_checkpoint2safetensor")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local function log(...)
    local out = {}
    for i = 1, select("#", ...) do
        out[#out + 1] = tostring(select(i, ...))
    end
    io.stdout:write(table.concat(out, " ") .. "\n")
end

local function die(msg)
    io.stderr:write("[convert] " .. tostring(msg) .. "\n")
    os.exit(1)
end

local function ok_or_die(ok, err, ctx)
    if ok == false then
        die((ctx and (ctx .. ": ") or "") .. tostring(err or "unknown"))
    end
end

if type(_G.Mimir) ~= "table" then
    die("Mimir indisponible: lancez via ./bin/mimir --lua ...")
end

-- ---------------------------------------------------------------------------
-- Args
-- ---------------------------------------------------------------------------

local opts = Args.parse(arg) or {}

local CHECKPOINT_DIR = Args.get_str(opts, "checkpoint", Args.get_str(opts, "in", ""))
local OUT = Args.get_str(opts, "out", "")

local MEM_GB = Args.get_num(opts, "mem-gb", 8)
local ALLOC_GB = Args.get_num(opts, "alloc-gb", 8)
local ENABLE_COMPRESSION = Args.get_bool(opts, "compression", true)
local ENABLE_HW = Args.get_bool(opts, "hw", false)

if CHECKPOINT_DIR == "" then
    die("missing arg: --checkpoint <dir> (ou --in <dir>)")
end
if OUT == "" then
    -- default: /path/to/dir.safetensors
    OUT = CHECKPOINT_DIR
    OUT = OUT:gsub("/*$", "")
    OUT = OUT .. ".safetensors"
end

local function file_exists(path)
    return FS.file_exists(path)
end

local function merge_into(dst, src)
    if type(dst) ~= "table" or type(src) ~= "table" then return end
    for k, v in pairs(src) do
        dst[k] = v
    end
end

local function canonicalize_dtype_name(dtype)
    if type(dtype) ~= "string" or dtype == "" then return dtype end
    local map = {
        F16 = "float16",
        F32 = "float32",
        F64 = "float64",
        BF16 = "bfloat16",
        I8 = "int8",
        U8 = "uint8",
        I16 = "int16",
        U16 = "uint16",
        I32 = "int32",
        U32 = "uint32",
        I64 = "int64",
        U64 = "uint64",
        BOOL = "bool",
    }
    return map[dtype] or dtype
end

local function normalize_config_dtype(cfg)
    if type(cfg) ~= "table" then return end
    if type(cfg.dtype) == "string" then
        cfg.dtype = canonicalize_dtype_name(cfg.dtype)
    end
end

-- ---------------------------------------------------------------------------
-- JSON utils (fallback)
-- ---------------------------------------------------------------------------

local function json_decode_fallback(s)
    -- Very small JSON decoder for objects/arrays/strings/numbers/bools/null.
    -- Enough for architecture.json in this repo.
    local i = 1
    local function skip()
        while true do
            local c = s:sub(i, i)
            if c == "" then return end
            if c ~= " " and c ~= "\t" and c ~= "\n" and c ~= "\r" then return end
            i = i + 1
        end
    end

    local function parse_string()
        local quote = s:sub(i, i)
        if quote ~= '"' then return nil, "expected string" end
        i = i + 1
        local out = {}
        while true do
            local c = s:sub(i, i)
            if c == "" then return nil, "unterminated string" end
            if c == '"' then
                i = i + 1
                return table.concat(out)
            end
            if c == "\\" then
                local n = s:sub(i + 1, i + 1)
                if n == "" then return nil, "unterminated escape" end
                if n == '"' or n == "\\" or n == "/" then
                    out[#out + 1] = n
                    i = i + 2
                elseif n == "b" then out[#out + 1] = "\b"; i = i + 2
                elseif n == "f" then out[#out + 1] = "\f"; i = i + 2
                elseif n == "n" then out[#out + 1] = "\n"; i = i + 2
                elseif n == "r" then out[#out + 1] = "\r"; i = i + 2
                elseif n == "t" then out[#out + 1] = "\t"; i = i + 2
                elseif n == "u" then
                    local hex = s:sub(i + 2, i + 5)
                    if #hex < 4 then return nil, "bad unicode escape" end
                    local code = tonumber(hex, 16)
                    if not code then return nil, "bad unicode escape" end
                    if code < 0x80 then
                        out[#out + 1] = string.char(code)
                    elseif code < 0x800 then
                        out[#out + 1] = string.char(0xC0 + math.floor(code / 0x40), 0x80 + (code % 0x40))
                    else
                        out[#out + 1] = string.char(0xE0 + math.floor(code / 0x1000), 0x80 + (math.floor(code / 0x40) % 0x40), 0x80 + (code % 0x40))
                    end
                    i = i + 6
                else
                    return nil, "unsupported escape"
                end
            else
                out[#out + 1] = c
                i = i + 1
            end
        end
    end

    local function parse_number()
        local start = i
        local c = s:sub(i, i)
        if c == "-" then i = i + 1 end
        while s:sub(i, i):match("%d") do i = i + 1 end
        if s:sub(i, i) == "." then
            i = i + 1
            while s:sub(i, i):match("%d") do i = i + 1 end
        end
        local e = s:sub(i, i)
        if e == "e" or e == "E" then
            i = i + 1
            local sgn = s:sub(i, i)
            if sgn == "+" or sgn == "-" then i = i + 1 end
            while s:sub(i, i):match("%d") do i = i + 1 end
        end
        local num = tonumber(s:sub(start, i - 1))
        if num == nil then return nil, "bad number" end
        return num
    end

    local parse_value

    local function parse_array()
        if s:sub(i, i) ~= "[" then return nil, "expected [" end
        i = i + 1
        skip()
        local arr = {}
        if s:sub(i, i) == "]" then i = i + 1; return arr end
        while true do
            skip()
            local v, err = parse_value()
            if err then return nil, err end
            arr[#arr + 1] = v
            skip()
            local c = s:sub(i, i)
            if c == "," then i = i + 1
            elseif c == "]" then i = i + 1; return arr
            else return nil, "expected , or ]" end
        end
    end

    local function parse_object()
        if s:sub(i, i) ~= "{" then return nil, "expected {" end
        i = i + 1
        skip()
        local obj = {}
        if s:sub(i, i) == "}" then i = i + 1; return obj end
        while true do
            skip()
            local k, errk = parse_string()
            if errk then return nil, errk end
            skip()
            if s:sub(i, i) ~= ":" then return nil, "expected :" end
            i = i + 1
            skip()
            local v, errv = parse_value()
            if errv then return nil, errv end
            obj[k] = v
            skip()
            local c = s:sub(i, i)
            if c == "," then i = i + 1
            elseif c == "}" then i = i + 1; return obj
            else return nil, "expected , or }" end
        end
    end

    function parse_value()
        skip()
        local c = s:sub(i, i)
        if c == '"' then return parse_string() end
        if c == "{" then return parse_object() end
        if c == "[" then return parse_array() end
        if c == "-" or c:match("%d") then return parse_number() end
        if s:sub(i, i + 3) == "true" then i = i + 4; return true end
        if s:sub(i, i + 4) == "false" then i = i + 5; return false end
        if s:sub(i, i + 3) == "null" then i = i + 4; return nil end
        return nil, "unexpected token" .. tostring(c)
    end

    local v, err = parse_value()
    if err then return nil, err end
    skip()
    if i <= #s then
        return nil, "trailing data"
    end
    return v
end

local function safe_read_json(path)
    local f = io.open(path, "rb")
    if not f then return nil, "cannot open" end
    local s = f:read("*a")
    f:close()

    -- Prefer json module if present
    if type(_G.json) == "table" and type(_G.json.decode) == "function" then
        local ok, v = pcall(_G.json.decode, s)
        if ok then return v end
    end
    if type(_G.cjson) == "table" and type(_G.cjson.decode) == "function" then
        local ok, v = pcall(_G.cjson.decode, s)
        if ok then return v end
    end

    return json_decode_fallback(s)
end

-- ---------------------------------------------------------------------------
-- Architecture inference
-- ---------------------------------------------------------------------------

local function infer_model_type_from_arch(arch)
    if type(arch) ~= "table" then return nil end
    if type(arch.architecture) == "string" and arch.architecture ~= "" then
        return arch.architecture
    end
    if type(arch.type) == "string" and arch.type ~= "" then
        return arch.type
    end
    if type(arch.model_name) == "string" and arch.model_name ~= "" then
        return arch.model_name
    end
    if type(arch.model) == "string" and arch.model ~= "" then
        return arch.model
    end
    if type(arch.model) == "table" then
        if type(arch.model.architecture) == "string" and arch.model.architecture ~= "" then
            return arch.model.architecture
        end
        if type(arch.model.type) == "string" and arch.model.type ~= "" then
            return arch.model.type
        end
    end
    if type(arch.layers) == "table" then
        for _, layer in ipairs(arch.layers) do
            local name = type(layer) == "table" and layer.name or nil
            if type(name) == "string" then
                if name:match("^vae_conv/") or name:match("^vae_conv") then
                    return "vae_conv"
                end
                if name:match("^transformer/") or name:match("^transformer") then
                    return "transformer"
                end
            end
        end
    end
    return nil
end

local function normalize_name_key(s)
    s = tostring(s or "")
    s = s:lower()
    -- Retire les séparateurs pour comparer des variantes proches.
    s = s:gsub("[^%w]", "")
    return s
end

local function resolve_model_type(candidate)
    candidate = tostring(candidate or "")
    if candidate == "" then return "" end

    -- Alias explicites fréquents rencontrés dans les architecture.json legacy.
    local alias = {
        VAEConvModel = "vae_conv",
        vaeconvmodel = "vae_conv",
        VAETexteModel = "vae_text",
        vaetextemodel = "vae_text",
        VAETextModel = "vae_text",
        vaetextmodel = "vae_text",
        TransformerModel = "transformer",
        transformermodel = "transformer",
        UNetModel = "unet",
        unetmodel = "unet",
        DiffusionModel = "diffusion",
        diffusionmodel = "diffusion",
    }

    if alias[candidate] then
        return alias[candidate]
    end
    if alias[candidate:lower()] then
        return alias[candidate:lower()]
    end

    -- Si déjà résoluble tel quel, on conserve.
    if Mimir and Mimir.Architectures and Mimir.Architectures.default_config then
        local cfg = Mimir.Architectures.default_config(candidate)
        if type(cfg) == "table" then return candidate end
    end

    -- Essayer une résolution fuzzy contre la liste des archis disponibles.
    local available = nil
    if Mimir and Mimir.Architectures and Mimir.Architectures.available then
        available = Mimir.Architectures.available()
    end
    if type(available) ~= "table" then
        return candidate
    end

    local want = normalize_name_key(candidate)
    if want == "" then return candidate end

    -- Variantes tolérées (ex: suffixe "Model").
    local want_nomodel = want:gsub("model$", "")
    for _, name in ipairs(available) do
        local n = tostring(name or "")
        local key = normalize_name_key(n)
        if key == want or key == want_nomodel then
            return n
        end
    end

    return candidate
end

local function infer_cfg_vae_conv_from_arch(arch)
    if type(arch) ~= "table" or type(arch.layers) ~= "table" then
        return nil, "arch.layers missing"
    end

    local image_w, image_h, image_c
    local latent_c
    local base_channels
    local downsamples = 0

    -- Heuristics:
    -- - Look for first conv in encoder as image_c and base_channels
    -- - Look for last conv out_channels as latent_c
    for _, layer in ipairs(arch.layers) do
        if type(layer) == "table" then
            local name = layer.name
            local t = layer.type
            if type(name) == "string" then
                local in_ch = tonumber(layer.in_channels)
                local out_ch = tonumber(layer.out_channels)

                if not base_channels and out_ch and name:match("encoder") and (t == "conv2d" or t == "Conv2D" or t == "conv") then
                    base_channels = out_ch
                    if in_ch then image_c = in_ch end
                end

                if out_ch and name:match("to_latent") then
                    latent_c = out_ch
                end

                if name:match("down") and (t == "conv2d" or t == "Conv2D" or t == "conv") then
                    local stride = tonumber(layer.stride) or 1
                    if stride == 2 then downsamples = downsamples + 1 end
                end
            end
        end
    end

    local cfg = arch.model_config
    local resnet        = false
    local attention     = false
    local enc_norm      = nil
    local enc_gn_groups = nil
    local attn_heads    = nil
    local resnet_max_tokens = nil
    local attn_max_tokens   = nil
    if type(cfg) == "table" then
        image_w = tonumber(cfg.image_w) or tonumber(cfg.image_width) or image_w
        image_h = tonumber(cfg.image_h) or tonumber(cfg.image_height) or image_h
        image_c = tonumber(cfg.image_c) or image_c
        latent_c = tonumber(cfg.latent_c) or latent_c
        base_channels = tonumber(cfg.base_channels) or base_channels
        downsamples = tonumber(cfg.downsamples) or downsamples

        local configured_resnet = cfg.resnet
        if configured_resnet == nil then configured_resnet = cfg.use_resnet end
        if configured_resnet == nil then configured_resnet = cfg.use_attention end
        local configured_attention = cfg.attention
        if configured_attention == nil then configured_attention = cfg.use_attn end
        resnet = configured_resnet == true
        attention = configured_attention == true
        enc_norm        = (type(cfg.enc_norm) == "string" and cfg.enc_norm ~= "") and cfg.enc_norm or nil
        enc_gn_groups   = tonumber(cfg.enc_gn_groups)
        attn_heads      = tonumber(cfg.attn_heads)
        resnet_max_tokens = tonumber(cfg.resnet_max_tokens)
        attn_max_tokens   = tonumber(cfg.attn_max_tokens)
    end

    if not image_w or not image_h then
        -- Fallback defaults; user can override by putting a correct model_config in the checkpoint.
        image_w = image_w or 512
        image_h = image_h or 512
    end

    image_c = image_c or 3
    latent_c = latent_c or 128
    base_channels = base_channels or 64

    local div = 2 ^ downsamples
    if div <= 0 then div = 1 end
    if (image_h % div) ~= 0 or (image_w % div) ~= 0 then
        return nil, string.format("cannot infer latent_h/w: image not divisible by 2^downsamples (image=%dx%d downsamples=%d)", image_w, image_h, downsamples)
    end

    local latent_h = math.floor(image_h / div)
    local latent_w = math.floor(image_w / div)

    return {
        image_w = image_w,
        image_h = image_h,
        image_c = image_c,
        latent_h = latent_h,
        latent_w = latent_w,
        latent_c = math.floor(latent_c),
        base_channels = math.floor(base_channels),
        downsamples = downsamples,
        resnet            = resnet,
        attention         = attention,
        enc_norm          = enc_norm,
        enc_gn_groups     = enc_gn_groups,
        attn_heads        = attn_heads,
        resnet_max_tokens = resnet_max_tokens,
        attn_max_tokens   = attn_max_tokens,
    }, nil
end

local function build_model_config_from_arch(model_type, arch)
    if not (Mimir and Mimir.Architectures and Mimir.Architectures.default_config) then
        return nil, "Mimir.Architectures.default_config indisponible"
    end

    local cfg = Mimir.Architectures.default_config(model_type)
    if type(cfg) ~= "table" then
        return nil, "default_config(" .. tostring(model_type) .. ") a échoué"
    end

    local merged_any = false
    if type(arch) == "table" then
        if type(arch.model_config) == "table" then
            merge_into(cfg, arch.model_config)
            merged_any = true
        end
        if type(arch.model) == "table" then
            merge_into(cfg, arch.model)
            merged_any = true
        end
        if type(arch[model_type]) == "table" then
            merge_into(cfg, arch[model_type])
            merged_any = true
        end
    end

    normalize_config_dtype(cfg)

    if model_type == "vae_conv" and not merged_any then
        local inferred, inf_err = infer_cfg_vae_conv_from_arch(arch)
        if not inferred then
            return nil, "infer_cfg_vae_conv_from_arch failed: " .. tostring(inf_err)
        end
        cfg.image_w = math.floor(inferred.image_w)
        cfg.image_h = math.floor(inferred.image_h)
        cfg.image_c = math.floor(inferred.image_c)
        cfg.latent_h = math.floor(inferred.latent_h)
        cfg.latent_w = math.floor(inferred.latent_w)
        cfg.latent_c = math.floor(inferred.latent_c)
        cfg.base_channels = math.floor(inferred.base_channels)
        if inferred.resnet            ~= nil then cfg.resnet            = inferred.resnet            end
        if inferred.attention         ~= nil then cfg.attention         = inferred.attention         end
        if inferred.enc_norm          ~= nil then cfg.enc_norm          = inferred.enc_norm          end
        if inferred.enc_gn_groups     ~= nil then cfg.enc_gn_groups     = math.floor(inferred.enc_gn_groups)     end
        if inferred.attn_heads        ~= nil then cfg.attn_heads        = math.floor(inferred.attn_heads)        end
        if inferred.resnet_max_tokens ~= nil then cfg.resnet_max_tokens = math.floor(inferred.resnet_max_tokens) end
        if inferred.attn_max_tokens   ~= nil then cfg.attn_max_tokens   = math.floor(inferred.attn_max_tokens)   end
    end

    return cfg
end

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  1. Configuration Système")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- ⚠️ CRITIQUE: Toujours configurer l'allocateur et MemoryGuard en premier!
if not (Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit) then
    die("Mimir.MemoryGuard.setLimit indisponible (script à lancer via ./bin/mimir)")
end
ok_or_die(Mimir.MemoryGuard.setLimit(MEM_GB), nil, "MemoryGuard.setLimit")
log(string.format("✓ MemoryGuard configuré (limite: %.3g GB)", MEM_GB))

if not (Mimir and Mimir.Allocator and Mimir.Allocator.configure) then
    die("Mimir.Allocator.configure indisponible (script à lancer via ./bin/mimir)")
end
local ok_alloc, err_alloc = Mimir.Allocator.configure({
    max_ram_gb = ALLOC_GB,
    enable_compression = ENABLE_COMPRESSION,
    swap_strategy = "lru",
})
ok_or_die(ok_alloc, err_alloc, "Allocator.configure")
log("✓ Allocateur configuré (compression=" .. tostring(ENABLE_COMPRESSION) .. ", max_ram_gb=" .. tostring(ALLOC_GB) .. ")")

-- Vérifier et activer l'accélération hardware (optionnel)
if Mimir and Mimir.Model and Mimir.Model.hardware_caps and Mimir.Model.set_hardware then
    local ok_caps, hw = pcall(Mimir.Model.hardware_caps)
    if ok_caps and type(hw) == "table" then
        log("\n🔧 Capacités Hardware:")
        log(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
        log(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
        log(string.format("  • F16C:  %s", hw.f16c and "✓" or "✗"))
        log(string.format("  • BMI2:  %s", hw.bmi2 and "✓" or "✗"))

        if ENABLE_HW and (hw.avx2 or hw.fma) then
            local ok_hw, err_hw = Mimir.Model.set_hardware(true)
            if ok_hw then
                log("\n✓ Accélération hardware activée\n")
            else
                log("\n⚠️  Accélération hardware non activée: " .. tostring(err_hw or "unknown"))
            end
        end
    end
end

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  2. Conversion (Serialization API v2.4)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if not (Mimir and Mimir.Serialization and Mimir.Serialization.load and Mimir.Serialization.save) then
    die("Mimir.Serialization.load/save indisponible (script à lancer via ./bin/mimir)")
end

local model_type = Args.get_str(opts, "model-type", Args.get_str(opts, "model", ""))

local arch_path = CHECKPOINT_DIR .. "/model/architecture.json"
if not file_exists(arch_path) then
    arch_path = CHECKPOINT_DIR .. "/architecture.json"
end
if not file_exists(arch_path) then
    die("architecture.json introuvable dans le checkpoint: " .. tostring(CHECKPOINT_DIR))
end

local arch, arch_err = safe_read_json(arch_path)
if type(arch) ~= "table" then
    die("lecture JSON échouée: " .. tostring(arch_path) .. ": " .. tostring(arch_err or "unknown"))
end

if model_type == "" then
    model_type = infer_model_type_from_arch(arch) or ""
end
if model_type == "" then
    die("impossible d'inférer le type de modèle. Passe `--model-type <...>` (ex: vae_conv)")
end

local resolved_model_type = resolve_model_type(model_type)
if resolved_model_type ~= model_type then
    log("[convert] model_type résolu: " .. tostring(model_type) .. " -> " .. tostring(resolved_model_type))
    model_type = resolved_model_type
end

local cfg, cfg_err = build_model_config_from_arch(model_type, arch)
if type(cfg) ~= "table" then
    die(tostring(cfg_err or "build_model_config_from_arch failed"))
end

log("[convert] create model: type=" .. tostring(model_type))
local ok_create, err_create = Mimir.Model.create(model_type, cfg)
ok_or_die(ok_create, err_create, "Model.create")

local ok_build, err_build = Mimir.Model.build()
ok_or_die(ok_build, err_build, "Model.build")

local ok_params, err_params = Mimir.Model.allocate_params()
ok_or_die(ok_params, err_params, "Model.allocate_params")

log("[convert] load: " .. tostring(CHECKPOINT_DIR) .. " (raw_folder)")
local ok_load, err_load = Mimir.Serialization.load(CHECKPOINT_DIR, "raw_folder", {
    load_optimizer = true,
    load_tokenizer = true,
    load_encoder = true,
    strict_mode = false,
    validate_checksums = true,
})
ok_or_die(ok_load, err_load, "Serialization.load")

log("[convert] save: " .. tostring(OUT) .. " (safetensors)")
local ok_save, err_save = Mimir.Serialization.save(OUT, "safetensors", {
    save_optimizer = true,
    save_tokenizer = true,
    save_encoder = true,
    include_git_info = true,
})
ok_or_die(ok_save, err_save, "Serialization.save")

log("✓ OK: écrit " .. tostring(OUT))
