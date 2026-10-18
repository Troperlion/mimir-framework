-- Convertir un checkpoint SafeTensors (.safetensors) en RawFolder (dossier)
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/convert_safetensors2raw_folder.lua \
--       --in model.safetensors --out checkpoint_dir/
--
-- Notes:
--   - Charger un safetensors requiert aussi un modèle déjà créé/allocaté.
--   - Ce script lit le tensor `model/architecture_json` directement depuis le fichier
--     `.safetensors`, puis reconstruit la config depuis `model_config`.

---@diagnostic disable: undefined-field, need-check-nil, param-type-mismatch

local ToolHelp = dofile(ROOTWORK.."/scripts/modules/tools_help.lua")
ToolHelp.show("convert_safetensors2raw_folder")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

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

local opts = Args.parse(arg) or {}

local IN = Args.get_str(opts, "in", Args.get_str(opts, "checkpoint", ""))
local OUT_DIR = Args.get_str(opts, "out", Args.get_str(opts, "out-dir", ""))

local MEM_GB = Args.get_num(opts, "mem-gb", 8)
local ALLOC_GB = Args.get_num(opts, "alloc-gb", 8)
local ENABLE_COMPRESSION = Args.get_bool(opts, "compression", true)
local ENABLE_HW = Args.get_bool(opts, "hw", false)

if IN == "" then
    die("missing arg: --in <model.safetensors>")
end
if OUT_DIR == "" then
    die("missing arg: --out <checkpoint_dir/>")
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

local function infer_model_type_from_arch(arch)
    if type(arch) ~= "table" then return nil end
    if type(arch.model_name) == "string" and arch.model_name ~= "" then
        return arch.model_name
    end
    if type(arch.architecture) == "string" and arch.architecture ~= "" then
        return arch.architecture
    end
    if type(arch.type) == "string" and arch.type ~= "" then
        return arch.type
    end
    if type(arch.model) == "table" then
        if type(arch.model.architecture) == "string" and arch.model.architecture ~= "" then
            return arch.model.architecture
        end
        if type(arch.model.type) == "string" and arch.model.type ~= "" then
            return arch.model.type
        end
    elseif type(arch.model) == "string" and arch.model ~= "" then
        return arch.model
    end
    return nil
end

-- ---------------------------------------------------------------------------
-- JSON fallback decoder (identique au convertisseur aller)
-- ---------------------------------------------------------------------------

local function json_decode_fallback(s)
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

local function safe_json_decode(s)
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
-- SafeTensors minimal reader: extract model/architecture_json
-- ---------------------------------------------------------------------------

local function u64_le(bytes)
    -- bytes: string len>=8
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
    local header, errj = safe_json_decode(header_str)
    if type(header) ~= "table" then
        f:close()
        return nil, "header JSON decode failed: " .. tostring(errj or "unknown")
    end
    return { f = f, header_len = header_len, header = header }, nil
end

local function extract_tensor_bytes(ctx, tensor_name)
    local header = ctx.header
    local entry = header[tensor_name]
    if type(entry) ~= "table" then
        return nil, "tensor not found in header: " .. tostring(tensor_name)
    end

    local offsets = entry.data_offsets
    if type(offsets) ~= "table" or #offsets < 2 then
        return nil, "missing data_offsets for " .. tostring(tensor_name)
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

local function load_arch_from_safetensors(path)
    local ctx, err = read_safetensors_header(path)
    if not ctx then return nil, err end

    local bytes, errb = extract_tensor_bytes(ctx, "model/architecture_json")
    ctx.f:close()
    if not bytes then return nil, errb end

    local arch, errj = safe_json_decode(bytes)
    if type(arch) ~= "table" then
        return nil, "architecture_json decode failed: " .. tostring(errj or "unknown")
    end
    return arch, nil
end

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  1. Configuration Système")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

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

if Mimir and Mimir.Model and Mimir.Model.hardware_caps and Mimir.Model.set_hardware then
    local ok_caps, hw = pcall(Mimir.Model.hardware_caps)
    if ok_caps and type(hw) == "table" and ENABLE_HW and (hw.avx2 or hw.fma) then
        pcall(Mimir.Model.set_hardware, true)
    end
end

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  2. Conversion (Serialization API v2.4)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if not (Mimir and Mimir.Serialization and Mimir.Serialization.load and Mimir.Serialization.save) then
    die("Mimir.Serialization.load/save indisponible (script à lancer via ./bin/mimir)")
end

local arch, arch_err = load_arch_from_safetensors(IN)
if not arch then
    die("impossible de lire model/architecture_json depuis safetensors: " .. tostring(arch_err))
end

local model_type = infer_model_type_from_arch(arch) or ""
if model_type == "" then
    die("type de modèle absent dans model/architecture_json")
end

if not (Mimir and Mimir.Architectures and Mimir.Architectures.default_config) then
    die("Mimir.Architectures.default_config indisponible")
end

local cfg = Mimir.Architectures.default_config(model_type)
if type(cfg) ~= "table" then
    die("default_config(" .. tostring(model_type) .. ") a échoué")
end

if type(arch.model_config) == "table" then
    merge_into(cfg, arch.model_config)
end
if type(arch.model) == "table" then
    merge_into(cfg, arch.model)
end
if type(arch[model_type]) == "table" then
    merge_into(cfg, arch[model_type])
end
normalize_config_dtype(cfg)

log("[convert] create model: type=" .. tostring(model_type))
local ok_create, err_create = Mimir.Model.create(model_type, cfg)
ok_or_die(ok_create, err_create, "Model.create")

local ok_build, err_build = Mimir.Model.build()
ok_or_die(ok_build, err_build, "Model.build")

local ok_params, err_params = Mimir.Model.allocate_params()
ok_or_die(ok_params, err_params, "Model.allocate_params")

log("[convert] load: " .. tostring(IN) .. " (safetensors)")
local ok_load, err_load = Mimir.Serialization.load(IN, "safetensors", {
    load_optimizer = true,
    load_tokenizer = true,
    load_encoder = true,
})
ok_or_die(ok_load, err_load, "Serialization.load")

-- Assurer que le dossier existe
FS.mkdir_p(OUT_DIR)

log("[convert] save: " .. tostring(OUT_DIR) .. " (raw_folder)")
local ok_save, err_save = Mimir.Serialization.save(OUT_DIR, "raw_folder", {
    save_optimizer = true,
    save_tokenizer = true,
    save_encoder = true,
    include_checksums = true,
    include_git_info = true,
})
ok_or_die(ok_save, err_save, "Serialization.save")

log("✓ OK: écrit " .. tostring(OUT_DIR))
