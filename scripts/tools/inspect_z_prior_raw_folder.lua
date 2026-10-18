-- Inspecteur des features z_prior dans un checkpoint Raw_Folder.
--
-- Usage minimal:
--   ./bin/mimir --lua scripts/tools/inspect_z_prior_raw_folder.lua \
--       --checkpoint checkpoint/vae_conv-generique.v5/epoch_0001
--
-- Options:
--   --tensor <name>        Nom exact du tensor (sinon auto-detection sur z_prior)
--   --max-values <n>       Nombre de valeurs affichees en console (defaut: 128)
--   --topk <k>             Top-K des valeurs absolues (defaut: 16)
--   --out <path.csv>       Exporte toutes les valeurs en CSV index,value
--   --image-out <path.ppm> Exporte une image des features (defaut auto)
--   --image-width <n>      Largeur forcee de la feature map (optionnel)
--   --hist-out <path.ppm>  Exporte une image histogramme (defaut auto)
--   --hist-bins <n>        Nombre de bins histogramme (defaut: 128)
--   --hist-width <n>       Largeur image histogramme (defaut: 960)
--   --hist-height <n>      Hauteur image histogramme (defaut: 420)
--   --quiet                Affichage reduit (garde resume + topk)

---@diagnostic disable: undefined-field, need-check-nil, param-type-mismatch

local ToolHelp = dofile(ROOTWORK.."/scripts/modules/tools_help.lua")
ToolHelp.show("inspect_z_prior_raw_folder")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")

local function die(msg)
    io.stderr:write("[inspect_z_prior] " .. tostring(msg) .. "\n")
    os.exit(1)
end

local function log(...)
    local out = {}
    for i = 1, select("#", ...) do
        out[#out + 1] = tostring(select(i, ...))
    end
    io.stdout:write(table.concat(out, " ") .. "\n")
end

local function read_all(path)
    local f = io.open(path, "rb")
    if not f then return nil, "open failed" end
    local s = f:read("*a")
    f:close()
    return s
end

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

local function json_decode_fallback(s)
    local i = 1

    local function skip_ws()
        while true do
            local c = s:sub(i, i)
            if c == "" then return end
            if c ~= " " and c ~= "\t" and c ~= "\n" and c ~= "\r" then return end
            i = i + 1
        end
    end

    local parse_value

    local function parse_string()
        if s:sub(i, i) ~= '"' then return nil, "expected string" end
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
                elseif n == "b" then
                    out[#out + 1] = "\b"
                    i = i + 2
                elseif n == "f" then
                    out[#out + 1] = "\f"
                    i = i + 2
                elseif n == "n" then
                    out[#out + 1] = "\n"
                    i = i + 2
                elseif n == "r" then
                    out[#out + 1] = "\r"
                    i = i + 2
                elseif n == "t" then
                    out[#out + 1] = "\t"
                    i = i + 2
                elseif n == "u" then
                    local hex = s:sub(i + 2, i + 5)
                    if #hex < 4 then return nil, "bad unicode escape" end
                    local code = tonumber(hex, 16)
                    if not code then return nil, "bad unicode escape" end
                    if code < 0x80 then
                        out[#out + 1] = string.char(code)
                    elseif code < 0x800 then
                        out[#out + 1] = string.char(
                            0xC0 + math.floor(code / 0x40),
                            0x80 + (code % 0x40)
                        )
                    else
                        out[#out + 1] = string.char(
                            0xE0 + math.floor(code / 0x1000),
                            0x80 + (math.floor(code / 0x40) % 0x40),
                            0x80 + (code % 0x40)
                        )
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
            local sign = s:sub(i, i)
            if sign == "+" or sign == "-" then i = i + 1 end
            while s:sub(i, i):match("%d") do i = i + 1 end
        end
        local num = tonumber(s:sub(start, i - 1))
        if num == nil then return nil, "bad number" end
        return num
    end

    local function parse_array()
        if s:sub(i, i) ~= "[" then return nil, "expected [" end
        i = i + 1
        skip_ws()
        local arr = {}
        if s:sub(i, i) == "]" then i = i + 1; return arr end
        while true do
            skip_ws()
            local v, err = parse_value()
            if err then return nil, err end
            arr[#arr + 1] = v
            skip_ws()
            local c = s:sub(i, i)
            if c == "," then
                i = i + 1
            elseif c == "]" then
                i = i + 1
                return arr
            else
                return nil, "expected , or ]"
            end
        end
    end

    local function parse_object()
        if s:sub(i, i) ~= "{" then return nil, "expected {" end
        i = i + 1
        skip_ws()
        local obj = {}
        if s:sub(i, i) == "}" then i = i + 1; return obj end
        while true do
            skip_ws()
            local k, errk = parse_string()
            if errk then return nil, errk end
            skip_ws()
            if s:sub(i, i) ~= ":" then return nil, "expected :" end
            i = i + 1
            skip_ws()
            local v, errv = parse_value()
            if errv then return nil, errv end
            obj[k] = v
            skip_ws()
            local c = s:sub(i, i)
            if c == "," then
                i = i + 1
            elseif c == "}" then
                i = i + 1
                return obj
            else
                return nil, "expected , or }"
            end
        end
    end

    function parse_value()
        skip_ws()
        local c = s:sub(i, i)
        if c == '"' then return parse_string() end
        if c == "{" then return parse_object() end
        if c == "[" then return parse_array() end
        if c == "-" or c:match("%d") then return parse_number() end
        if s:sub(i, i + 3) == "true" then i = i + 4; return true end
        if s:sub(i, i + 4) == "false" then i = i + 5; return false end
        if s:sub(i, i + 3) == "null" then i = i + 4; return nil end
        return nil, "unexpected token"
    end

    local v, err = parse_value()
    if err then return nil, err end
    skip_ws()
    if i <= #s then return nil, "trailing data" end
    return v
end

local function safe_json_decode(s)
    if type(_G.json) == "table" and type(_G.json.decode) == "function" then
        local ok, v = pcall(_G.json.decode, s)
        if ok then return v end
    end
    return json_decode_fallback(s)
end

local function safe_read_json(path)
    local raw, err = read_all(path)
    if not raw then return nil, err end
    return safe_json_decode(raw)
end

local function canonical_dtype(dtype)
    local d = tostring(dtype or "")
    if d == "F16" then return "float16" end
    if d == "F32" then return "float32" end
    if d == "F64" then return "float64" end
    if d == "BF16" then return "bfloat16" end
    if d == "U8" then return "uint8" end
    if d == "I16" then return "int16" end
    if d == "U16" then return "uint16" end
    if d == "I32" then return "int32" end
    return d
end

local function half_to_float(h)
    local sign = ((h >> 15) & 0x1) == 1 and -1.0 or 1.0
    local exp = (h >> 10) & 0x1F
    local frac = h & 0x03FF

    if exp == 0 then
        if frac == 0 then return sign * 0.0 end
        return sign * (2.0 ^ -14) * (frac / 1024.0)
    end
    if exp == 31 then
        if frac == 0 then
            return sign * math.huge
        end
        return 0.0 / 0.0
    end
    return sign * (2.0 ^ (exp - 15)) * (1.0 + frac / 1024.0)
end

local function bfloat16_to_float(b)
    local sign = ((b >> 15) & 0x1) == 1 and -1.0 or 1.0
    local exp = (b >> 7) & 0xFF
    local frac = b & 0x7F

    if exp == 0 then
        if frac == 0 then return sign * 0.0 end
        return sign * (2.0 ^ -126) * (frac / 128.0)
    end
    if exp == 255 then
        if frac == 0 then
            return sign * math.huge
        end
        return 0.0 / 0.0
    end
    return sign * (2.0 ^ (exp - 127)) * (1.0 + frac / 128.0)
end

local function prod_shape(shape)
    if type(shape) ~= "table" then return nil end
    local p = 1
    for i = 1, #shape do
        local v = tonumber(shape[i])
        if not v then return nil end
        p = p * v
    end
    return p
end

local function decode_tensor_values(bytes, dtype, expected_elems)
    local d = canonical_dtype(dtype)
    local out = {}

    if d == "float32" then
        local pos = 1
        for _ = 1, expected_elems do
            local v
            v, pos = string.unpack("<f", bytes, pos)
            out[#out + 1] = v
        end
        return out
    elseif d == "float64" then
        local pos = 1
        for _ = 1, expected_elems do
            local v
            v, pos = string.unpack("<d", bytes, pos)
            out[#out + 1] = v
        end
        return out
    elseif d == "uint8" then
        for i = 1, expected_elems do
            out[#out + 1] = bytes:byte(i) or 0
        end
        return out
    elseif d == "int16" then
        local pos = 1
        for _ = 1, expected_elems do
            local v
            v, pos = string.unpack("<i2", bytes, pos)
            out[#out + 1] = v
        end
        return out
    elseif d == "uint16" then
        local pos = 1
        for _ = 1, expected_elems do
            local v
            v, pos = string.unpack("<I2", bytes, pos)
            out[#out + 1] = v
        end
        return out
    elseif d == "int32" then
        local pos = 1
        for _ = 1, expected_elems do
            local v
            v, pos = string.unpack("<i4", bytes, pos)
            out[#out + 1] = v
        end
        return out
    elseif d == "float16" or d == "bfloat16" then
        local pos = 1
        for _ = 1, expected_elems do
            local u16
            u16, pos = string.unpack("<I2", bytes, pos)
            if d == "float16" then
                out[#out + 1] = half_to_float(u16)
            else
                out[#out + 1] = bfloat16_to_float(u16)
            end
        end
        return out
    else
        return nil, "dtype non supporte: " .. tostring(dtype)
    end
end

local function summarize(values)
    local n = #values
    if n == 0 then
        return {
            n = 0,
            min = 0,
            max = 0,
            mean = 0,
            std = 0,
            l1 = 0,
            l2 = 0,
            zeros = 0,
            non_finite = 0,
        }
    end

    local mn = math.huge
    local mx = -math.huge
    local sum = 0.0
    local sum_sq = 0.0
    local l1 = 0.0
    local zeros = 0
    local non_finite = 0

    for i = 1, n do
        local v = values[i]
        if v ~= v or v == math.huge or v == -math.huge then
            non_finite = non_finite + 1
        else
            if v < mn then mn = v end
            if v > mx then mx = v end
            sum = sum + v
            sum_sq = sum_sq + v * v
            l1 = l1 + math.abs(v)
            if v == 0 then zeros = zeros + 1 end
        end
    end

    local valid = n - non_finite
    if valid <= 0 then
        return {
            n = n,
            min = 0,
            max = 0,
            mean = 0,
            std = 0,
            l1 = 0,
            l2 = 0,
            zeros = zeros,
            non_finite = non_finite,
        }
    end

    local mean = sum / valid
    local var = (sum_sq / valid) - (mean * mean)
    if var < 0 then var = 0 end

    return {
        n = n,
        min = mn,
        max = mx,
        mean = mean,
        std = math.sqrt(var),
        l1 = l1,
        l2 = math.sqrt(sum_sq),
        zeros = zeros,
        non_finite = non_finite,
    }
end

local function topk_abs(values, k)
    local items = {}
    for i = 1, #values do
        items[#items + 1] = { index = i, value = values[i], abs = math.abs(values[i]) }
    end
    table.sort(items, function(a, b) return a.abs > b.abs end)
    local out = {}
    for i = 1, math.min(k, #items) do
        out[#out + 1] = items[i]
    end
    return out
end

local function find_tensor_entry(manifest, explicit_name)
    local idx = (type(manifest) == "table" and type(manifest.tensor_index) == "table") and manifest.tensor_index or {}

    if explicit_name and explicit_name ~= "" then
        for i = 1, #idx do
            local e = idx[i]
            if type(e) == "table" and tostring(e.name or "") == explicit_name then
                return e, nil
            end
        end
        return nil, "tensor introuvable: " .. explicit_name
    end

    local candidates = {}
    for i = 1, #idx do
        local e = idx[i]
        if type(e) == "table" then
            local n = tostring(e.name or "")
            if n:find("z_prior", 1, true) then
                candidates[#candidates + 1] = e
            end
        end
    end

    if #candidates == 0 then
        return nil, "aucun tensor contenant 'z_prior'"
    end

    table.sort(candidates, function(a, b)
        local an = tostring(a.name or "")
        local bn = tostring(b.name or "")
        local as = 0
        local bs = 0
        if an:find("z_prior_bias", 1, true) then as = as + 5 end
        if bn:find("z_prior_bias", 1, true) then bs = bs + 5 end
        if an:find("weights", 1, true) then as = as + 2 end
        if bn:find("weights", 1, true) then bs = bs + 2 end
        if as ~= bs then return as > bs end
        return an < bn
    end)

    return candidates[1], nil
end

local function write_csv(path, values)
    local f, err = io.open(path, "wb")
    if not f then return false, err or "open failed" end
    f:write("index,value\n")
    for i = 1, #values do
        f:write(tostring(i), ",", string.format("%.9g", values[i]), "\n")
    end
    f:close()
    return true
end

local function parent_dir(path)
    local p = tostring(path or "")
    local idx = p:match("^.*()/")
    if idx then
        if idx <= 1 then return "/" end
        return p:sub(1, idx - 1)
    end
    return "."
end

local function basename_no_ext(path)
    local p = tostring(path or "")
    p = p:gsub("^.*[/]", "")
    p = p:gsub("%.[^%.]+$", "")
    return p
end

local function ensure_dir(path)
    local dir = parent_dir(path)
    if dir == "." or dir == "" then return true end
    local ok = os.execute(string.format('mkdir -p "%s"', dir))
    return ok == true or ok == 0
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function infer_feature_dims(shape, n, forced_w)
    if forced_w and forced_w > 0 then
        local w = forced_w
        local h = math.ceil(n / w)
        return w, h
    end

    if type(shape) == "table" then
        if #shape == 2 then
            local h = tonumber(shape[1]) or 1
            local w = tonumber(shape[2]) or n
            if h * w == n then return w, h end
        elseif #shape == 3 then
            local c = tonumber(shape[1]) or 1
            local h = tonumber(shape[2]) or 1
            local w = tonumber(shape[3]) or 1
            if c * h * w == n then
                return w, h * c
            end
        elseif #shape == 1 then
            local side = math.ceil(math.sqrt(n))
            return side, side
        end
    end

    local side = math.ceil(math.sqrt(n))
    return side, side
end

local function value_to_rgb(v, max_abs)
    if max_abs <= 0 then
        return 128, 128, 128
    end
    local t = clamp(v / max_abs, -1.0, 1.0)
    local r, g, b
    if t >= 0 then
        local k = t
        r = 255
        g = math.floor(255 * (1.0 - k) + 0.5)
        b = math.floor(255 * (1.0 - k) + 0.5)
    else
        local k = -t
        r = math.floor(255 * (1.0 - k) + 0.5)
        g = math.floor(255 * (1.0 - k) + 0.5)
        b = 255
    end
    return clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)
end

local function write_ppm(path, w, h, rgb_bytes)
    if not ensure_dir(path) then return false, "mkdir failed" end
    local f, err = io.open(path, "wb")
    if not f then return false, err or "open failed" end
    f:write(string.format("P6\n%d %d\n255\n", w, h))
    f:write(rgb_bytes)
    f:close()
    return true
end

local function build_feature_image(values, shape, forced_w)
    local n = #values
    local w, h = infer_feature_dims(shape, n, forced_w)
    local max_abs = 0.0
    for i = 1, n do
        local a = math.abs(values[i])
        if a > max_abs then max_abs = a end
    end

    local bytes = {}
    local idx = 1
    for _y = 1, h do
        for _x = 1, w do
            local v = values[idx] or 0.0
            local r, g, b = value_to_rgb(v, max_abs)
            bytes[#bytes + 1] = string.char(r, g, b)
            idx = idx + 1
        end
    end
    return w, h, table.concat(bytes)
end

local function histogram_counts(values, bins)
    local mn = math.huge
    local mx = -math.huge
    for i = 1, #values do
        local v = values[i]
        if v < mn then mn = v end
        if v > mx then mx = v end
    end

    if mn == math.huge or mx == -math.huge then
        mn = -1
        mx = 1
    end
    if mn == mx then
        mn = mn - 1e-6
        mx = mx + 1e-6
    end

    local counts = {}
    for i = 1, bins do counts[i] = 0 end

    local range = mx - mn
    for i = 1, #values do
        local t = (values[i] - mn) / range
        local b = math.floor(t * bins) + 1
        b = clamp(b, 1, bins)
        counts[b] = counts[b] + 1
    end

    return counts, mn, mx
end

local function build_hist_image(values, bins, out_w, out_h)
    local counts, mn, mx = histogram_counts(values, bins)
    local maxc = 1
    for i = 1, #counts do
        if counts[i] > maxc then maxc = counts[i] end
    end

    local w = out_w
    local h = out_h
    local pixels = {}
    for _ = 1, w * h do
        pixels[#pixels + 1] = { 245, 245, 245 }
    end

    local function set_px(x, y, r, g, b)
        if x < 1 or x > w or y < 1 or y > h then return end
        local idx = (y - 1) * w + x
        pixels[idx][1] = r
        pixels[idx][2] = g
        pixels[idx][3] = b
    end

    local margin_l, margin_r, margin_t, margin_b = 56, 18, 18, 34
    local plot_w = math.max(1, w - margin_l - margin_r)
    local plot_h = math.max(1, h - margin_t - margin_b)
    local y_base = margin_t + plot_h

    for x = margin_l, margin_l + plot_w do
        set_px(x, y_base, 70, 70, 70)
    end
    for y = margin_t, y_base do
        set_px(margin_l, y, 70, 70, 70)
    end

    for i = 1, bins do
        local x0 = margin_l + math.floor((i - 1) * plot_w / bins)
        local x1 = margin_l + math.floor(i * plot_w / bins) - 1
        if x1 < x0 then x1 = x0 end
        local bh = math.floor((counts[i] / maxc) * (plot_h - 1))
        for x = x0, x1 do
            for y = y_base - bh, y_base - 1 do
                set_px(x, y, 55, 120, 210)
            end
        end
    end

    local zero_t = (0 - mn) / (mx - mn)
    if zero_t >= 0 and zero_t <= 1 then
        local zx = margin_l + math.floor(zero_t * plot_w)
        for y = margin_t, y_base do
            set_px(zx, y, 210, 60, 60)
        end
    end

    local raw = {}
    for i = 1, #pixels do
        local p = pixels[i]
        raw[#raw + 1] = string.char(p[1], p[2], p[3])
    end

    return w, h, table.concat(raw), mn, mx, maxc
end

local opts = Args.parse(arg) or {}

local checkpoint = Args.get_str(opts, "checkpoint", Args.get_str(opts, "in", ""))
local explicit_tensor = Args.get_str(opts, "tensor", "")
local max_values = Args.get_num(opts, "max-values", 128)
local topk = Args.get_num(opts, "topk", 16)
local out_csv = Args.get_str(opts, "out", "")
local image_out = Args.get_str(opts, "image-out", "")
local hist_out = Args.get_str(opts, "hist-out", "")
local image_width = Args.get_num(opts, "image-width", 0)
local hist_bins = Args.get_num(opts, "hist-bins", 128)
local hist_width = Args.get_num(opts, "hist-width", 960)
local hist_height = Args.get_num(opts, "hist-height", 420)
local quiet = Args.get_bool(opts, "quiet", false)

max_values = math.max(1, math.floor(tonumber(max_values) or 128))
topk = math.max(1, math.floor(tonumber(topk) or 16))
image_width = math.max(0, math.floor(tonumber(image_width) or 0))
hist_bins = math.max(8, math.floor(tonumber(hist_bins) or 128))
hist_width = math.max(320, math.floor(tonumber(hist_width) or 960))
hist_height = math.max(220, math.floor(tonumber(hist_height) or 420))

if checkpoint == "" then
    die("missing arg: --checkpoint <raw_folder>")
end

local root = checkpoint:gsub("/*$", "")
local manifest_path = root .. "/manifest.json"
if not file_exists(manifest_path) then
    die("manifest.json introuvable: " .. manifest_path)
end

local manifest, jerr = safe_read_json(manifest_path)
if not manifest then
    die("manifest.json invalide: " .. tostring(jerr))
end

local entry, ferr = find_tensor_entry(manifest, explicit_tensor)
if not entry then
    die(ferr)
end

local tensor_name = tostring(entry.name or "")
local json_rel = tostring(entry.json_file or "")
if json_rel == "" then
    die("json_file manquant pour tensor: " .. tensor_name)
end

local meta_path = root .. "/" .. json_rel
if not file_exists(meta_path) then
    die("metadata tensor introuvable: " .. meta_path)
end

local meta, merr = safe_read_json(meta_path)
if not meta then
    die("metadata tensor invalide: " .. tostring(merr))
end

local dtype = tostring(meta.dtype or "")
local shape = meta.shape
local expected_elems = prod_shape(shape)
if not expected_elems or expected_elems <= 0 then
    die("shape invalide pour tensor: " .. tensor_name)
end

local bin_rel = tostring(entry.bin_file or "")
if bin_rel == "" and type(meta.data_file) == "string" and meta.data_file ~= "" then
    bin_rel = "tensors/" .. meta.data_file
end
if bin_rel == "" then
    die("bin_file manquant pour tensor: " .. tensor_name)
end

local bin_path = root .. "/" .. bin_rel
if not file_exists(bin_path) then
    die("fichier binaire introuvable: " .. bin_path)
end

local bytes, berr = read_all(bin_path)
if not bytes then
    die("lecture binaire impossible: " .. tostring(berr))
end

local values, derr = decode_tensor_values(bytes, dtype, expected_elems)
if not values then
    die("decode impossible: " .. tostring(derr))
end

local stats = summarize(values)
local peaks = topk_abs(values, topk)

log("[inspect_z_prior] checkpoint:", root)
log("[inspect_z_prior] tensor:", tensor_name)
log("[inspect_z_prior] dtype:", tostring(dtype), "| shape:", "[" .. table.concat(shape, "x") .. "]")
log("[inspect_z_prior] values:", tostring(stats.n), "| non_finite:", tostring(stats.non_finite))
log(string.format("[inspect_z_prior] min=%.9g max=%.9g mean=%.9g std=%.9g", stats.min, stats.max, stats.mean, stats.std))
log(string.format("[inspect_z_prior] l1=%.9g l2=%.9g zeros=%d", stats.l1, stats.l2, stats.zeros))

if image_out == "" then
    image_out = root .. "/z_prior_feature_map.ppm"
end
if hist_out == "" then
    hist_out = root .. "/z_prior_histogram.ppm"
end

local fw, fh, fraw = build_feature_image(values, shape, image_width)
local ok_img, err_img = write_ppm(image_out, fw, fh, fraw)
if not ok_img then
    die("ecriture image feature impossible: " .. tostring(err_img))
end
log(string.format("[inspect_z_prior] feature_image: %s (%dx%d)", image_out, fw, fh))

local hw, hh, hraw, hmin, hmax, hpeak = build_hist_image(values, hist_bins, hist_width, hist_height)
local ok_hist, err_hist = write_ppm(hist_out, hw, hh, hraw)
if not ok_hist then
    die("ecriture image histogramme impossible: " .. tostring(err_hist))
end
log(string.format("[inspect_z_prior] histogram_image: %s (%dx%d)", hist_out, hw, hh))
log(string.format("[inspect_z_prior] histogram_range=[%.9g, %.9g] peak_count=%d bins=%d", hmin, hmax, hpeak, hist_bins))

if not quiet then
    local show_n = math.min(max_values, #values)
    log(string.format("[inspect_z_prior] first_%d_values:", show_n))
    for i = 1, show_n do
        log(string.format("  [%d] %.9g", i, values[i]))
    end
end

log(string.format("[inspect_z_prior] top_%d_abs:", #peaks))
for i = 1, #peaks do
    local p = peaks[i]
    log(string.format("  #%d idx=%d value=%.9g |abs|=%.9g", i, p.index, p.value, p.abs))
end

if out_csv ~= "" then
    local ok, werr = write_csv(out_csv, values)
    if not ok then
        die("ecriture CSV impossible: " .. tostring(werr))
    end
    log("[inspect_z_prior] csv_export:", out_csv)
end
