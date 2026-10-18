-- Analyseur de modèles/checkpoints Mimir (RawFolder / SafeTensors)
--
-- Usage:
--   ./bin/mimir --lua scripts/tools/analyze_model.lua --in model.safetensors
--   ./bin/mimir --lua scripts/tools/analyze_model.lua --in checkpoint_dir/
--
-- Objectif:
--   Afficher en console (tableaux) les infos utiles: nom, date de création,
--   versions, couches, paramètres, et un résumé des tensors.

---@diagnostic disable: undefined-field, need-check-nil

local ToolHelp = dofile(ROOTWORK.."/scripts/modules/tools_help.lua")
ToolHelp.show("analyze_model")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")

-- ---------------------------------------------------------------------------
-- Couleurs ANSI (désactivées si NO_COLOR)
-- ---------------------------------------------------------------------------

local COLOR_ENABLED = true
do
    if os.getenv("NO_COLOR") ~= nil then
        COLOR_ENABLED = false
    end
end

local C = {
    reset = "\27[0m",
    bold = "\27[1m",
    dim = "\27[2m",
    red = "\27[31m",
    green = "\27[32m",
    yellow = "\27[33m",
    blue = "\27[34m",
    magenta = "\27[35m",
    cyan = "\27[36m",
    gray = "\27[90m",
}

local function colorize(s, ...)
    s = tostring(s)
    if not COLOR_ENABLED then return s end
    local codes = { ... }
    if #codes == 0 then return s end
    return table.concat(codes) .. s .. C.reset
end

-- ---------------------------------------------------------------------------
-- Utils
-- ---------------------------------------------------------------------------

local function log(...)
    local out = {}
    for i = 1, select("#", ...) do
        out[#out + 1] = tostring(select(i, ...))
    end
    io.stdout:write(table.concat(out, " ") .. "\n")
end

local function die(msg)
    io.stderr:write("[analyze] " .. tostring(msg) .. "\n")
    os.exit(1)
end

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then f:close(); return true end
    return false
end

local function read_all(path)
    local f = io.open(path, "rb")
    if not f then return nil, "cannot open" end
    local s = f:read("*a")
    f:close()
    return s
end

local function read_exact(f, n)
    local s = f:read(n)
    if not s or #s ~= n then
        return nil, "short read"
    end
    return s
end

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

local function clamp(n, lo, hi)
    if n < lo then return lo end
    if n > hi then return hi end
    return n
end

local function format_int(n)
    n = tonumber(n)
    if not n then return "?" end
    local s = tostring(math.floor(n))
    local neg = false
    if s:sub(1, 1) == "-" then neg = true; s = s:sub(2) end
    local r = s:reverse():gsub("(%d%d%d)", "%1_")
    r = r:reverse():gsub("^_", "")
    if neg then r = "-" .. r end
    return r
end

local function format_bytes(bytes)
    bytes = tonumber(bytes)
    if not bytes then return "?" end
    if bytes < 1024 then return string.format("%d B", bytes) end
    local units = { "KB", "MB", "GB", "TB" }
    local v = bytes
    local u = 0
    while v >= 1024 and u < #units do
        v = v / 1024
        u = u + 1
    end
    return string.format("%.3g %s", v, units[u])
end

local function format_epoch(epoch)
    epoch = tonumber(epoch)
    if not epoch or epoch <= 0 then return "?" end
    epoch = math.floor(epoch)
    -- Format local time; suffisant pour l'analyse.
    return os.date("%Y-%m-%d %H:%M:%S", epoch)
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

local function dtype_bytes(dtype)
    dtype = tostring(dtype or "")
    if dtype == "F32" or dtype == "float32" then return 4 end
    if dtype == "F16" or dtype == "float16" then return 2 end
    if dtype == "I32" or dtype == "int32" then return 4 end
    if dtype == "I16" or dtype == "int16" then return 2 end
    if dtype == "U16" or dtype == "uint16" then return 2 end
    if dtype == "U8" or dtype == "uint8" then return 1 end
    return nil
end

-- ---------------------------------------------------------------------------
-- UTF-8 helpers (pour alignement console)
-- ---------------------------------------------------------------------------

local function utf8_len_safe(s)
    s = tostring(s or "")
    if type(_G.utf8) == "table" and type(_G.utf8.len) == "function" then
        local ok, n = pcall(_G.utf8.len, s)
        if ok and type(n) == "number" then
            return n
        end
    end

    -- Fallback: compter les points de code UTF-8 (approx largeur monospace)
    local i = 1
    local n = 0
    local bytes = #s
    while i <= bytes do
        local c = s:byte(i)
        if not c then break end
        if c < 0x80 then
            i = i + 1
        elseif c < 0xE0 then
            i = i + 2
        elseif c < 0xF0 then
            i = i + 3
        else
            i = i + 4
        end
        n = n + 1
    end
    return n
end

local function utf8_sub_chars(s, n_chars)
    s = tostring(s or "")
    n_chars = tonumber(n_chars) or 0
    n_chars = math.floor(n_chars)
    if n_chars <= 0 then return "" end

    if type(_G.utf8) == "table" and type(_G.utf8.offset) == "function" then
        local ok, off = pcall(_G.utf8.offset, s, n_chars + 1)
        if ok and type(off) == "number" then
            return s:sub(1, off - 1)
        end
        -- Si offset échoue, on tombe sur fallback ci-dessous.
    end

    -- Fallback: itération basique UTF-8
    local i = 1
    local bytes = #s
    local count = 0
    while i <= bytes and count < n_chars do
        local c = s:byte(i)
        if not c then break end
        if c < 0x80 then
            i = i + 1
        elseif c < 0xE0 then
            i = i + 2
        elseif c < 0xF0 then
            i = i + 3
        else
            i = i + 4
        end
        count = count + 1
    end
    return s:sub(1, i - 1)
end

local function trunc(s, max_len)
    s = tostring(s or "")
    max_len = tonumber(max_len) or 80
    max_len = math.floor(max_len)
    if utf8_len_safe(s) <= max_len then return s end
    if max_len < 6 then return utf8_sub_chars(s, max_len) end
    return utf8_sub_chars(s, max_len - 3) .. "..."
end

-- Découpe une chaîne en lignes de max_w caractères UTF-8 (sans troncature).
local function split_lines_wrap(s, max_w)
    s = tostring(s or "")
    max_w = math.floor(tonumber(max_w) or 80)
    if max_w <= 0 then return { s } end
    if utf8_len_safe(s) <= max_w then return { s } end
    local result = {}
    local safety = 0
    while utf8_len_safe(s) > max_w and safety < 10000 do
        safety = safety + 1
        local chunk = utf8_sub_chars(s, max_w)
        if #chunk == 0 then break end
        result[#result + 1] = chunk
        s = s:sub(#chunk + 1)
    end
    if s ~= "" then result[#result + 1] = s end
    return result
end

local function pad_right(s, w)
    s = tostring(s or "")
    w = tonumber(w) or utf8_len_safe(s)
    w = math.floor(w)
    local len = utf8_len_safe(s)
    if len >= w then return s end
    return s .. string.rep(" ", w - len)
end

local function pad_left(s, w)
    s = tostring(s or "")
    w = tonumber(w) or utf8_len_safe(s)
    w = math.floor(w)
    local len = utf8_len_safe(s)
    if len >= w then return s end
    return string.rep(" ", w - len) .. s
end

local function make_table(columns, rows)
    -- columns: { {key="name", title="Name", align="left|right", max=...}, ... }
    local widths = {}
    for ci = 1, #columns do
        local col = columns[ci]
        local title = tostring(col.title or col.key or ("col" .. ci))
        widths[ci] = utf8_len_safe(title)
    end

    for ri = 1, #rows do
        local row = rows[ri]
        for ci = 1, #columns do
            local col = columns[ci]
            if col.wrap then
                local cap = math.floor(tonumber(col.max) or 100)
                if cap > widths[ci] then widths[ci] = cap end
            else
                local v = row[col.key]
                local s = trunc(v, col.max or 120)
                local w = utf8_len_safe(s)
                if w > widths[ci] then widths[ci] = w end
            end
        end
    end

    local function sep(ch)
        local parts = { "+" }
        for ci = 1, #columns do
            parts[#parts + 1] = string.rep(ch, widths[ci] + 2)
            parts[#parts + 1] = "+"
        end
        return colorize(table.concat(parts), C.gray)
    end

    local bar = colorize("|", C.gray)
    local out = {}
    out[#out + 1] = sep("-")

    do
        local parts = { bar }
        for ci = 1, #columns do
            local col = columns[ci]
            local title = tostring(col.title or col.key or "")
            parts[#parts + 1] = " " .. colorize(pad_right(title, widths[ci]), C.bold, C.cyan) .. " "
            parts[#parts + 1] = bar
        end
        out[#out + 1] = table.concat(parts)
    end

    out[#out + 1] = sep("=")

    for ri = 1, #rows do
        local row = rows[ri]
        local col_lines = {}
        local max_lines = 1
        for ci = 1, #columns do
            local col = columns[ci]
            local v = tostring(row[col.key] or "")
            local lines_ci
            if col.wrap then
                lines_ci = split_lines_wrap(v, widths[ci])
            else
                lines_ci = { trunc(v, col.max or 120) }
            end
            col_lines[ci] = lines_ci
            if #lines_ci > max_lines then max_lines = #lines_ci end
        end
        for li = 1, max_lines do
            local parts = { bar }
            for ci = 1, #columns do
                local col = columns[ci]
                local s = col_lines[ci][li] or ""
                local cell
                if (col.align or "left") == "right" then
                    cell = pad_left(s, widths[ci])
                else
                    cell = pad_right(s, widths[ci])
                end
                if col.color then
                    cell = colorize(cell, col.color)
                end
                parts[#parts + 1] = " " .. cell .. " "
                parts[#parts + 1] = bar
            end
            out[#out + 1] = table.concat(parts)
        end
    end

    out[#out + 1] = sep("-")
    return table.concat(out, "\n")
end

-- ---------------------------------------------------------------------------
-- JSON utils (fallback)
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
        if s:sub(i, i) ~= '"' then return nil, "expected string" end
        i = i + 1
        local out = {}
        while true do
            local c = s:sub(i, i)
            if c == "" then return nil, "unterminated string" end
            if c == '"' then i = i + 1; return table.concat(out) end
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
    if i <= #s then return nil, "trailing data" end
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

local function safe_read_json(path)
    local s, err = read_all(path)
    if not s then return nil, err end
    local t, jerr = safe_json_decode(s)
    if type(t) ~= "table" then return nil, jerr end
    return t
end

local function read_prefix(path, n)
    n = math.tointeger(tonumber(n) or 65536) or 65536
    local f = io.open(path, "rb")
    if not f then return nil, "cannot open" end
    local s = f:read(n)
    f:close()
    return s or ""
end

local function write_all(path, content)
    local f = io.open(path, "wb")
    if not f then return nil, "cannot open for write" end
    f:write(content or "")
    f:close()
    return true
end

local function shell_quote(s)
    s = tostring(s or "")
    return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function run_cmd(cmd)
    local ok, why, code = os.execute(cmd)
    if type(ok) == "number" then
        return ok == 0, ok
    end
    if type(ok) == "boolean" then
        if ok then return true, 0 end
        return false, tonumber(code) or -1
    end
    return false, -1
end

local function cmd_exists(name)
    local ok = run_cmd("command -v " .. shell_quote(name) .. " >/dev/null 2>&1")
    return ok
end

local function extract_mermaid_block(markdown)
    markdown = tostring(markdown or "")
    local block = markdown:match("```mermaid%s*\n(.-)\n```")
    if type(block) == "string" and block ~= "" then
        return block
    end
    return nil
end

local function export_mermaid_graph(markdown, out_path)
    out_path = tostring(out_path or "")
    if out_path == "" then
        return nil, "chemin de sortie vide"
    end

    local ext = out_path:match("%.([^.]+)$")
    ext = tostring(ext or ""):lower()
    if ext == "jpeg" then ext = "jpg" end
    if ext ~= "svg" and ext ~= "png" and ext ~= "jpg" then
        return nil, "extension non supportée (utiliser .svg, .png ou .jpg)"
    end

    local mermaid = extract_mermaid_block(markdown)
    if not mermaid then
        return nil, "bloc mermaid introuvable dans la sortie générée"
    end

    local stamp = tostring(os.time()) .. "_" .. tostring(math.random(100000, 999999))
    local tmp_mmd = "/tmp/mimir_mlp_graph_" .. stamp .. ".mmd"
    local ok_w, err_w = write_all(tmp_mmd, mermaid .. "\n")
    if not ok_w then
        return nil, "écriture temporaire impossible: " .. tostring(err_w)
    end

    local mmdc_cmd = nil
    if cmd_exists("mmdc") then
        mmdc_cmd = "mmdc"
    elseif cmd_exists("npx") then
        mmdc_cmd = "npx -y @mermaid-js/mermaid-cli"
    else
        run_cmd("rm -f " .. shell_quote(tmp_mmd))
        return nil, "mermaid-cli introuvable (installer mmdc, ou Node+npx)"
    end

    local final_out = out_path
    local render_out = out_path
    local tmp_png = nil

    if ext == "jpg" then
        tmp_png = "/tmp/mimir_mlp_graph_" .. stamp .. ".png"
        render_out = tmp_png
    end

    local render_cmd = mmdc_cmd
        .. " -i " .. shell_quote(tmp_mmd)
        .. " -o " .. shell_quote(render_out)
        .. " >/dev/null 2>&1"

    local ok_render = run_cmd(render_cmd)
    if not ok_render then
        run_cmd("rm -f " .. shell_quote(tmp_mmd))
        if tmp_png then run_cmd("rm -f " .. shell_quote(tmp_png)) end
        return nil, "rendu Mermaid échoué (commande: " .. mmdc_cmd .. ")"
    end

    if ext == "jpg" then
        local convert_cmd
        if cmd_exists("magick") then
            convert_cmd = "magick " .. shell_quote(tmp_png) .. " " .. shell_quote(final_out)
        elseif cmd_exists("convert") then
            convert_cmd = "convert " .. shell_quote(tmp_png) .. " " .. shell_quote(final_out)
        else
            run_cmd("rm -f " .. shell_quote(tmp_mmd) .. " " .. shell_quote(tmp_png))
            return nil, "conversion JPG impossible (installer ImageMagick: magick/convert)"
        end

        local ok_convert = run_cmd(convert_cmd .. " >/dev/null 2>&1")
        if not ok_convert then
            run_cmd("rm -f " .. shell_quote(tmp_mmd) .. " " .. shell_quote(tmp_png))
            return nil, "conversion PNG -> JPG échouée"
        end
    end

    run_cmd("rm -f " .. shell_quote(tmp_mmd))
    if tmp_png then run_cmd("rm -f " .. shell_quote(tmp_png)) end
    return final_out
end

-- ---------------------------------------------------------------------------
-- SafeTensors reader (header + extraction)
-- ---------------------------------------------------------------------------

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

local function extract_tensor_bytes(ctx, tensor_name, max_bytes)
    max_bytes = tonumber(max_bytes) or (32 * 1024 * 1024)

    local entry = ctx.header[tensor_name]
    if type(entry) ~= "table" then
        return nil, "tensor not found: " .. tostring(tensor_name)
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
    if size <= 0 then return nil, "invalid tensor size" end
    if size > max_bytes then
        return nil, "tensor trop gros pour extraction (" .. format_bytes(size) .. ")"
    end

    local data_base = 8 + ctx.header_len
    ctx.f:seek("set", data_base + begin)

    local bytes, err = read_exact(ctx.f, size)
    if not bytes then
        return nil, "read tensor bytes failed: " .. tostring(err)
    end

    return bytes, nil
end

-- ---------------------------------------------------------------------------
-- Layer formatting
-- ---------------------------------------------------------------------------

local function layer_dims(layer)
    if type(layer) ~= "table" then return "" end

    -- DebugJson enhanced (v1.3+) met souvent les infos de dimensions sous `layer.config`.
    local cfg = (type(layer.config) == "table") and layer.config or layer
    local ltype = tostring(layer.type or cfg.type or "")

    local function n(x)
        x = tonumber(x)
        if not x or x <= 0 then return nil end
        return math.floor(x)
    end

    local function vec_to_string(v)
        if type(v) ~= "table" or #v == 0 then return nil end
        local out = {}
        for i = 1, #v do
            out[#out + 1] = tostring(v[i])
        end
        return table.concat(out, "x")
    end

    local function first_present(...)
        for i = 1, select("#", ...) do
            local v = select(i, ...)
            if v ~= nil then return v end
        end
        return nil
    end

    local in_f = n(cfg.in_features)
    local out_f = n(cfg.out_features)
    if in_f and out_f then
        return string.format("%d→%d", in_f, out_f)
    end

    local in_c = n(cfg.in_channels)
    local out_c = n(cfg.out_channels)
    if in_c and out_c then
        local extra = {}

        -- Convs (v1.1.0): kernel_h/kernel_w, stride_h/stride_w, pad_h/pad_w
        local kh = n(cfg.kernel_h)
        local kw = n(cfg.kernel_w)
        if kh and kw then extra[#extra + 1] = string.format("k=%dx%d", kh, kw)
        else
            local k = n(cfg.kernel_size)
            if k then extra[#extra + 1] = "k=" .. k end
        end

        local sh = n(cfg.stride_h)
        local sw = n(cfg.stride_w)
        if sh and sw then extra[#extra + 1] = string.format("s=%dx%d", sh, sw)
        else
            local s = n(cfg.stride)
            if s then extra[#extra + 1] = "s=" .. s end
        end

        local ph = n(cfg.pad_h)
        local pw = n(cfg.pad_w)
        if ph and pw then extra[#extra + 1] = string.format("p=%dx%d", ph, pw)
        else
            local p = n(cfg.padding)
            if p then extra[#extra + 1] = "p=" .. p end
        end

        -- (extra est déjà rempli ci-dessus)
        local tail = (#extra > 0) and (" (" .. table.concat(extra, " ") .. ")") or ""
        return string.format("%d→%d%s", in_c, out_c, tail)
    end

    -- GroupNorm / normes spatiales (in_channels sans out_channels distincts)
    local num_groups = n(cfg.num_groups)
    if in_c and num_groups then
        return string.format("%d ch g=%d", in_c, num_groups)
    end
    if in_c and not out_c then
        return string.format("%d ch", in_c)
    end

    local embed = n(cfg.embed_dim)
    local heads = n(cfg.num_heads)
    if embed and heads then
        local seq = n(cfg.seq_len)
        if seq then
            return string.format("embed=%d heads=%d seq=%d", embed, heads, seq)
        end
        return string.format("embed=%d heads=%d", embed, heads)
    end

    local vocab = n(cfg.vocab_size)
    if ltype == "Embedding" and vocab and embed then
        return string.format("%d×%d", vocab, embed)
    end

    if ltype == "Concat" then
        local axis = first_present(n(cfg.concat_axis), n(cfg.axis))
        local n_inputs = (type(layer.inputs) == "table") and #layer.inputs or nil
        local parts = {}
        if axis ~= nil then parts[#parts + 1] = "axis=" .. axis end
        if n_inputs and n_inputs > 0 then parts[#parts + 1] = "in=" .. n_inputs end
        return (#parts > 0) and table.concat(parts, " ") or "concat"
    end

    if ltype == "Split" then
        local axis = first_present(n(cfg.split_axis), n(cfg.axis))
        local sizes = vec_to_string(cfg.split_sizes)
        local num_splits = n(cfg.num_splits)
        local parts = {}
        if axis ~= nil then parts[#parts + 1] = "axis=" .. axis end
        if sizes then parts[#parts + 1] = "sizes=" .. sizes
        elseif num_splits then parts[#parts + 1] = "n=" .. num_splits end
        return (#parts > 0) and table.concat(parts, " ") or "split"
    end

    if ltype == "Add" or ltype == "Multiply" or ltype == "Subtract" then
        local n_inputs = (type(layer.inputs) == "table") and #layer.inputs or nil
        if n_inputs and n_inputs > 0 then
            return string.format("inputs=%d", n_inputs)
        end
        return string.lower(ltype)
    end

    if ltype == "MatMul" or ltype == "BatchMatMul" then
        if in_f and out_f then
            return string.format("%d×%d", in_f, out_f)
        end
        local seq = n(cfg.seq_len)
        if embed and seq then
            return string.format("seq=%d d=%d", seq, embed)
        end
        return string.lower(ltype)
    end

    if ltype == "Reshape" or ltype == "View" then
        local shape = vec_to_string(cfg.target_shape) or vec_to_string(cfg.shape)
        if shape then return "[" .. shape .. "]" end
        return string.lower(ltype)
    end

    if ltype == "Permute" or ltype == "Transpose" then
        local dims = vec_to_string(cfg.permute_dims) or vec_to_string(cfg.shape)
        if dims then return dims end
        return string.lower(ltype)
    end

    if ltype == "Upsample" or ltype == "UpsampleNearest" or ltype == "UpsampleBilinear" then
        local out_h = n(cfg.out_h)
        local out_w = n(cfg.out_w)
        if out_h and out_w then
            return string.format("out=%dx%d", out_h, out_w)
        end
        local scale_h = tonumber(cfg.scale_h)
        local scale_w = tonumber(cfg.scale_w)
        if scale_h or scale_w then
            return string.format("scale=%sx%s", tostring(scale_h or "?"), tostring(scale_w or "?"))
        end
        return "upsample"
    end

    if ltype == "Chunk" then
        local axis = first_present(n(cfg.axis), n(cfg.split_axis))
        local num_chunks = n(cfg.num_chunks)
        if axis ~= nil and num_chunks then
            return string.format("axis=%d n=%d", axis, num_chunks)
        elseif num_chunks then
            return string.format("n=%d", num_chunks)
        end
        return "chunk"
    end

    if ltype == "Stack" then
        local axis = first_present(n(cfg.stack_axis), n(cfg.axis))
        if axis ~= nil then return "axis=" .. axis end
        return "stack"
    end

    if ltype == "Softmax" or ltype == "LogSoftmax" then
        local axis = n(cfg.axis)
        if axis ~= nil then return "axis=" .. axis end
        return string.lower(ltype)
    end

    if ltype == "LayerNorm" or ltype == "RMSNorm" then
        if embed then return "d=" .. embed end
        if in_f then return tostring(in_f) end
    end

    if ltype == "Identity" then
        return "identity"
    end

    return ""
end

-- ---------------------------------------------------------------------------
-- Model analyzers
-- ---------------------------------------------------------------------------

-- Infer dominant float dtype from a tensors list (by total bytes)
local function infer_float_dtype(tensors)
    local acc = {}
    for _, t in ipairs(tensors) do
        local d = t.dtype
        if d == "F32" or d == "F16" or d == "BF16" or d == "F64"
                or d == "float32" or d == "float16" or d == "bfloat16" or d == "float64" then
            acc[d] = (acc[d] or 0) + (t.bytes_raw or 0)
        end
    end
    local best, best_b = nil, 0
    for d, b in pairs(acc) do
        if b > best_b then best_b = b; best = d end
    end
    return best
end

local function analyze_safetensors(path, opts)
    local ctx, err = read_safetensors_header(path)
    if not ctx then return nil, err end

    local header = ctx.header
    local meta = (type(header["__metadata__"]) == "table") and header["__metadata__"] or {}

    local tensors = {}
    local total_bytes = 0
    local total_elems = 0

    local has_tokenizer = false
    local has_encoder = false
    local has_optimizer = false

    for name, entry in pairs(header) do
        if name ~= "__metadata__" and type(entry) == "table" then
            local dtype = entry.dtype
            local shape = entry.shape
            local elems = prod_shape(shape)
            local offsets = entry.data_offsets
            local byte_size
            if type(offsets) == "table" and #offsets >= 2 then
                local b = tonumber(offsets[1])
                local e = tonumber(offsets[2])
                if b and e and e >= b then byte_size = e - b end
            end

            local row = {
                name = name,
                dtype = tostring(dtype or ""),
                shape = (type(shape) == "table") and ("[" .. table.concat(shape, "x") .. "]") or "",
                elems = elems and format_int(elems) or "?",
                bytes = byte_size and format_bytes(byte_size) or "?",
                bytes_raw = byte_size or 0,
            }
            tensors[#tensors + 1] = row

            if byte_size then total_bytes = total_bytes + byte_size end
            if elems then total_elems = total_elems + elems end

            if name:match("^tokenizer/") then has_tokenizer = true end
            if name:match("^encoder/") then has_encoder = true end
            if name:match("^optimizer/") then has_optimizer = true end
        end
    end

    -- Try extract model/architecture_json
    local arch
    do
        local bytes, errb = extract_tensor_bytes(ctx, "model/architecture_json", 32 * 1024 * 1024)
        if bytes then
            local a, ajerr = safe_json_decode(bytes)
            if type(a) == "table" then arch = a end
        else
            -- OK if missing
            if opts.debug then
                log("[analyze][debug] architecture_json: " .. tostring(errb))
            end
        end
    end

    ctx.f:close()

    return {
        format = "safetensors",
        path = path,
        metadata = meta,
        arch = arch,
        tensors = tensors,
        inferred_float_dtype = infer_float_dtype(tensors),
        totals = {
            tensors = #tensors,
            bytes = total_bytes,
            elems = total_elems,
        },
        components = {
            tokenizer = has_tokenizer,
            encoder = has_encoder,
            optimizer = has_optimizer,
        },
    }, nil
end

local function analyze_raw_folder(path, opts)
    local manifest_path = path:gsub("/*$", "") .. "/manifest.json"
    if not file_exists(manifest_path) then
        return nil, "manifest.json introuvable: " .. manifest_path
    end

    local manifest, err = safe_read_json(manifest_path)
    if not manifest then return nil, "manifest.json invalide: " .. tostring(err) end

    local arch_path = path:gsub("/*$", "") .. "/model/architecture.json"
    if not file_exists(arch_path) then
        arch_path = path:gsub("/*$", "") .. "/architecture.json" -- compat
    end

    local arch
    if file_exists(arch_path) then
        arch = select(1, safe_read_json(arch_path))
    end

    local tensor_index = (type(manifest.tensor_index) == "table") and manifest.tensor_index or {}

    -- Collect top tensors by file size (without parsing all JSON)
    local tensors = {}
    local total_bytes = 0
    for i = 1, #tensor_index do
        local entry = tensor_index[i]
        if type(entry) == "table" and type(entry.bin_file) == "string" and type(entry.name) == "string" then
            local bin_path = path:gsub("/*$", "") .. "/" .. entry.bin_file
            local f = io.open(bin_path, "rb")
            local size = 0
            if f then
                local cur = f:seek()
                size = f:seek("end") or 0
                if cur then f:seek("set", cur) end
                f:close()
            end
            total_bytes = total_bytes + (size or 0)
            tensors[#tensors + 1] = {
                name = entry.name,
                bin_file = entry.bin_file,
                json_file = entry.json_file,
                bytes_raw = size or 0,
                bytes = format_bytes(size or 0),
            }
        end
    end

    table.sort(tensors, function(a, b) return (a.bytes_raw or 0) > (b.bytes_raw or 0) end)

    -- Enrich top-N with dtype/shape from JSON sidecar
    local enrich_n = tonumber(opts.enrich_tensors) or 25
    enrich_n = clamp(enrich_n, 0, 200)

    for i = 1, math.min(enrich_n, #tensors) do
        local t = tensors[i]
        if t and type(t.json_file) == "string" then
            local jp = path:gsub("/*$", "") .. "/" .. t.json_file
            if file_exists(jp) then
                local tj = select(1, safe_read_json(jp))
                if type(tj) == "table" then
                    t.dtype = tostring(tj.dtype or "")
                    t.shape = (type(tj.shape) == "table") and ("[" .. table.concat(tj.shape, "x") .. "]") or ""
                    local elems = prod_shape(tj.shape)
                    t.elems = elems and format_int(elems) or ""
                end
            end
        end
    end

    return {
        format = "raw_folder",
        path = path,
        manifest = manifest,
        arch = arch,
        tensors = tensors,
        inferred_float_dtype = infer_float_dtype(tensors),
        totals = {
            tensors = #tensors,
            bytes = total_bytes,
        },
    }, nil
end

local function analyze_debug_json(path, opts)
    local root, err = safe_read_json(path)
    if not root then return nil, "debug_json invalide: " .. tostring(err) end

    -- Détection de version/forme
    local format = root.format
    local format_version = root.format_version

    local created_at = root.created_at or root.timestamp
    local mimir_version = root.mimir_version
    local git_commit = root.git_commit

    local model_name
    local model_type
    local total_params
    local num_layers

    if type(root.model) == "table" then
        model_name = root.model.name
        total_params = root.model.total_params
        num_layers = root.model.num_layers
        if type(root.model.type) == "string" and root.model.type ~= "" then
            model_type = root.model.type
        end
    end

    if model_name == nil then model_name = root.model_name end
    model_type = model_type or root.model_type or root.model_type_name or root.model_kind
    if model_type == nil and type(root.model_config) == "table" and type(root.model_config.type) == "string" then
        model_type = root.model_config.type
    end
    if total_params == nil then total_params = root.total_params end
    if num_layers == nil then num_layers = root.num_layers end

    local layers = {}
    if type(root.layers) == "table" then
        -- v1.0: {name,type,params_count}
        -- v1.1: {index,name,type,params_count,config,tensors}
        for i = 1, #root.layers do
            local l = root.layers[i]
            if type(l) == "table" then
                -- Poids par couche: somme des elems/bytes des tensors de la couche.
                local weights_elems = 0
                local weights_bytes = 0
                if type(l.tensors) == "table" then
                    for j = 1, #l.tensors do
                        local t = l.tensors[j]
                        if type(t) == "table" then
                            local elems = tonumber(t.total_elements) or prod_shape(t.shape)
                            local bpe = dtype_bytes(t.dtype)
                            if elems and bpe then
                                weights_elems = weights_elems + elems
                                weights_bytes = weights_bytes + (elems * bpe)
                            end
                        end
                    end
                end

                layers[#layers + 1] = {
                    name = l.name,
                    type = l.type,
                    params_count = l.params_count,
                    config = (type(l.config) == "table") and l.config or nil,
                    weights_size = (weights_elems > 0) and weights_elems or nil,
                    weights_bytes = (weights_bytes > 0) and weights_bytes or nil,
                    -- Certains DebugJson récents exposent l'IO; les anciens
                    -- continueront d'utiliser le fallback linéaire des renderers.
                    inputs = l.inputs,
                    output = l.output,
                }
            end
        end
    end

    local tensors = {}
    local total_bytes = 0
    local total_elems = 0

    -- v1.0: root.tensors[]
    if type(root.tensors) == "table" then
        for i = 1, #root.tensors do
            local t = root.tensors[i]
            if type(t) == "table" then
                local dtype = tostring(t.dtype or "")
                local shape = t.shape
                local elems = tonumber(t.total_elements) or prod_shape(shape)
                local bpe = dtype_bytes(dtype)
                local bytes_raw = (elems and bpe) and (elems * bpe) or 0

                tensors[#tensors + 1] = {
                    name = tostring(t.name or ""),
                    dtype = dtype,
                    shape = (type(shape) == "table") and ("[" .. table.concat(shape, "x") .. "]") or "",
                    elems = elems and format_int(elems) or "?",
                    bytes = (bytes_raw > 0) and format_bytes(bytes_raw) or "?",
                    bytes_raw = bytes_raw,
                }
                if elems then total_elems = total_elems + elems end
                if bytes_raw then total_bytes = total_bytes + bytes_raw end
            end
        end
    end

    -- v1.1: root.layers[].tensors[] (agrégation)
    if #tensors == 0 and type(root.layers) == "table" then
        for i = 1, #root.layers do
            local l = root.layers[i]
            if type(l) == "table" and type(l.tensors) == "table" then
                for j = 1, #l.tensors do
                    local t = l.tensors[j]
                    if type(t) == "table" then
                        local dtype = tostring(t.dtype or "")
                        local shape = t.shape
                        local elems = tonumber(t.total_elements) or prod_shape(shape)
                        local bpe = dtype_bytes(dtype)
                        local bytes_raw = (elems and bpe) and (elems * bpe) or 0

                        tensors[#tensors + 1] = {
                            name = tostring(t.name or ""),
                            dtype = dtype,
                            shape = (type(shape) == "table") and ("[" .. table.concat(shape, "x") .. "]") or "",
                            elems = elems and format_int(elems) or "?",
                            bytes = (bytes_raw > 0) and format_bytes(bytes_raw) or "?",
                            bytes_raw = bytes_raw,
                        }
                        if elems then total_elems = total_elems + elems end
                        if bytes_raw then total_bytes = total_bytes + bytes_raw end
                    end
                end
            end
        end
    end

    -- Heuristique simple: si pas de couche mais num_layers donné, laisser vide (pas de crash)
    local arch = {
        model_name = (type(model_name) == "string") and model_name or "?",
        model_type = (type(model_type) == "string") and model_type or nil,
        total_params = total_params,
        num_layers = num_layers or ((type(layers) == "table") and #layers or nil),
        layers = layers,
    }

    local meta = {
        created_at = created_at,
        mimir_version = mimir_version,
        format_version = format_version,
        git_commit = git_commit,
        format = format,
        warning = root.warning,
        debug_only = root.debug_only,
    }

    local framework_state = nil
    if type(root.framework_state) == "table" then
        framework_state = root.framework_state
    end

    return {
        format = "debug_json",
        path = path,
        metadata = meta,
        arch = arch,
        framework_state = framework_state,
        tensors = tensors,
        totals = {
            tensors = #tensors,
            bytes = total_bytes,
            elems = total_elems,
        },
    }, nil
end

-- ---------------------------------------------------------------------------
-- Rendering
-- ---------------------------------------------------------------------------

local function render_summary(info, opts)
    opts = opts or {}
    local meta = info.metadata or info.manifest or {}
    local arch = info.arch or {}

    local model_name = (type(arch.model_name) == "string" and arch.model_name ~= "") and arch.model_name or "?"
    local created_at = meta.created_at

    local mimir_version = meta.mimir_version
    local format_version = meta.format_version
    local git_commit = meta.git_commit

    local total_params = arch.total_params
    local num_layers = arch.num_layers
    if (not num_layers) and type(arch.layers) == "table" then num_layers = #arch.layers end
    local model_type_display = (type(arch.model_type) == "string" and arch.model_type ~= "") and arch.model_type
        or ((type(arch.model) == "table" and type(arch.model.type) == "string" and arch.model.type ~= "") and arch.model.type
        or ((type(arch.model_config) == "table" and type(arch.model_config.type) == "string" and arch.model_config.type ~= "") and arch.model_config.type
        or ""))

    local rows = {
        { k = "Chemin", v = info.path or "" },
        { k = "Format", v = info.format or "" },
        { k = "Modèle", v = model_name },
        { k = "Type modèle", v = model_type_display },
        { k = "Créé le", v = format_epoch(created_at) },
        { k = "Mímir version", v = mimir_version or "?" },
        { k = "Format version", v = format_version or "?" },
        { k = "Git commit", v = git_commit or "?" },
        { k = "Nb couches", v = num_layers and format_int(num_layers) or "?" },
        { k = "Params (arch)", v = total_params and format_int(total_params) or "?" },
        { k = "Nb tensors", v = (info.totals and info.totals.tensors) and format_int(info.totals.tensors) or "?" },
        { k = "Taille tensors", v = (info.totals and info.totals.bytes) and format_bytes(info.totals.bytes) or "?" },
    }

    if meta.debug_only ~= nil then
        rows[#rows + 1] = { k = "Debug only", v = tostring(meta.debug_only) }
    end

    if type(info.framework_state) == "table" then
        local fs = info.framework_state
        local snap_ver = tostring(fs.snapshot_version or "?")
        local snap_ts = fs.snapshot_timestamp
        local snap_txt = "v" .. snap_ver
        if snap_ts ~= nil then
            snap_txt = snap_txt .. " @ " .. format_epoch(snap_ts)
        end
        rows[#rows + 1] = { k = "Framework snapshot", v = snap_txt }
    end

    if type(meta.warning) == "string" and meta.warning ~= "" then
        rows[#rows + 1] = { k = "Alerte", v = meta.warning }
    end

    if info.format == "safetensors" and type(info.components) == "table" then
        rows[#rows + 1] = { k = "Composants", v = string.format("tokenizer=%s encoder=%s optimizer=%s", tostring(info.components.tokenizer), tostring(info.components.encoder), tostring(info.components.optimizer)) }
    end

    if info.format == "raw_folder" and type(meta.components) == "table" then
        local c = meta.components
        rows[#rows + 1] = { k = "Composants", v = string.format("tokenizer=%s encoder=%s optimizer=%s", tostring(c.tokenizer), tostring(c.encoder), tostring(c.optimizer)) }
    end

    -- Ligne DType (depuis model_config.dtype ou inféré depuis les tensors)
    do
        local float_dtype = nil
        local mc0 = arch.model_config
        if type(mc0) == "table" and type(mc0.dtype) == "string" and mc0.dtype ~= "" then
            float_dtype = mc0.dtype
        elseif type(info.inferred_float_dtype) == "string" and info.inferred_float_dtype ~= "" then
            float_dtype = info.inferred_float_dtype .. " (inféré)"
        end
        if float_dtype then
            rows[#rows + 1] = { k = "DType", v = float_dtype }
        end
    end

    -- Afficher les champs clés de model_config si disponibles
    local mc = arch.model_config
    if type(mc) == "table" then
        local interesting = {
            "task", "image_w", "image_h", "image_c",
            "latent_h", "latent_w", "latent_c", "base_channels", "downsamples",
            "resnet", "attention", "enc_norm", "enc_gn_groups",
            "attn_heads", "resnet_max_tokens", "attn_max_tokens",
            "stochastic_latent", "text_cond",
            "d_model", "num_heads", "num_layers", "mlp_hidden",
            "latent_dim", "hidden_dim",
        }
        local parts = {}
        for _, k in ipairs(interesting) do
            local v = mc[k]
            if v ~= nil and tostring(v) ~= "" then
                parts[#parts + 1] = k .. "=" .. tostring(v)
            end
        end
        if #parts > 0 then
            rows[#rows + 1] = { k = "model_config", v = table.concat(parts, "  ") }
        end
    end

    local wrap_all = opts.all == true
    local columns = {
        { key = "k", title = "Clé", align = "left", max = 28, color = C.yellow },
        { key = "v", title = "Valeur", align = "left", max = 100, wrap = wrap_all, color = C.green },
    }

    return make_table(columns, rows)
end

local function render_framework_state(info, opts)
    local fs = info.framework_state
    if type(fs) ~= "table" then return nil end

    local chunks = {}

    do
        local runtime = (type(fs.runtime) == "table") and fs.runtime or {}
        local rows = {
            { k = "snapshot_version", v = tostring(fs.snapshot_version or "?") },
            { k = "snapshot_time", v = format_epoch(fs.snapshot_timestamp) },
            { k = "framework_logs_suppressed", v = tostring(runtime.framework_logs_suppressed) },
        }

        if type(runtime.cpu_features) == "table" then
            local cf = runtime.cpu_features
            rows[#rows + 1] = {
                k = "cpu_features",
                v = string.format("avx2=%s fma=%s f16c=%s bmi2=%s", tostring(cf.avx2), tostring(cf.fma), tostring(cf.f16c), tostring(cf.bmi2))
            }
        end

        local columns = {
            { key = "k", title = "Runtime", align = "left", max = 28, color = C.yellow },
            { key = "v", title = "Valeur", align = "left", max = 100, wrap = true, color = C.green },
        }
        chunks[#chunks + 1] = make_table(columns, rows)
    end

    do
        local runtime = (type(fs.runtime) == "table") and fs.runtime or {}
        local backends = (type(runtime.backends) == "table") and runtime.backends or {}
        local order = { "cpu", "cuda", "rocm", "opencl", "vulkan" }
        local rows = {}
        for _, k in ipairs(order) do
            local b = backends[k]
            if type(b) == "table" then
                rows[#rows + 1] = {
                    backend = k,
                    compiled = tostring(b.compiled),
                    available = tostring(b.available),
                    disabled = (type(b.config) == "table") and tostring(b.config.disabled) or "",
                    verbose = (type(b.config) == "table") and tostring(b.config.verbose) or "",
                    device = (type(b.config) == "table" and b.config.device_index ~= nil) and tostring(b.config.device_index) or "",
                }
            end
        end
        if #rows > 0 then
            local columns = {
                { key = "backend", title = "Backend", align = "left", max = 10, color = C.cyan },
                { key = "compiled", title = "Built", align = "left", max = 8, color = C.magenta },
                { key = "available", title = "Avail", align = "left", max = 8, color = C.green },
                { key = "disabled", title = "Disabled", align = "left", max = 9, color = C.yellow },
                { key = "verbose", title = "Verbose", align = "left", max = 8, color = C.blue },
                { key = "device", title = "Device", align = "right", max = 7, color = C.gray },
            }
            chunks[#chunks + 1] = make_table(columns, rows)
        end
    end

    do
        local mem = (type(fs.memory) == "table") and fs.memory or {}
        local mg = (type(mem.memory_guard) == "table") and mem.memory_guard or {}
        local da = (type(mem.dynamic_tensor_allocator) == "table") and mem.dynamic_tensor_allocator or {}
        local ram = (type(mem.advanced_ram_manager) == "table") and mem.advanced_ram_manager or {}

        local rows = {
            { k = "memory_guard", v = string.format("current=%s peak=%s limit=%s usage=%.2f%% blocked=%s freeze=%s",
                format_bytes(tonumber(mg.current_bytes) or 0),
                format_bytes(tonumber(mg.peak_bytes) or 0),
                format_bytes(tonumber(mg.limit_bytes) or 0),
                tonumber(mg.usage_percent) or 0,
                tostring(mg.allocations_blocked),
                tostring(mg.freeze_mode)
            ) },
            { k = "allocator", v = string.format("tensors=%s loaded=%s",
                format_int(tonumber(da.tensor_count) or 0),
                format_int(tonumber(da.loaded_count) or 0)
            ) },
            { k = "advanced_ram", v = string.format("current=%s peak=%s usage=%.2f%% blocked=%s freeze=%s",
                format_bytes(tonumber(ram.current_bytes) or 0),
                format_bytes(tonumber(ram.peak_bytes) or 0),
                tonumber(ram.usage_percent) or 0,
                tostring(ram.allocations_blocked),
                tostring(ram.freeze_mode)
            ) },
        }

        local columns = {
            { key = "k", title = "Mémoire", align = "left", max = 18, color = C.yellow },
            { key = "v", title = "État", align = "left", max = 110, wrap = true, color = C.green },
        }
        chunks[#chunks + 1] = make_table(columns, rows)
    end

    do
        local reg = (type(fs.registry) == "table") and fs.registry or {}
        local mt = (type(fs.model) == "table") and fs.model or {}
        local env = (type(fs.runtime) == "table" and type(fs.runtime.environment_overrides) == "table") and fs.runtime.environment_overrides or {}

        local env_parts = {}
        for k, v in pairs(env) do
            env_parts[#env_parts + 1] = tostring(k) .. "=" .. tostring(v)
        end
        table.sort(env_parts)

        local rows = {
            { k = "supported_layers", v = format_int(tonumber(reg.supported_layer_count) or 0) },
            { k = "model_state", v = string.format("dtype=%s params=%s layers=%s frozen=%s",
                tostring(mt.default_dtype or "?"),
                format_int(tonumber(mt.total_params) or 0),
                format_int(tonumber(mt.num_layers) or 0),
                tostring(mt.parameters_frozen)
            ) },
            { k = "env_overrides", v = (#env_parts > 0) and table.concat(env_parts, "  ") or "(none)" },
        }

        local columns = {
            { key = "k", title = "Registry/Model", align = "left", max = 18, color = C.yellow },
            { key = "v", title = "Valeur", align = "left", max = 110, wrap = true, color = C.green },
        }
        chunks[#chunks + 1] = make_table(columns, rows)
    end

    return table.concat(chunks, "\n\n")
end

local function render_layers(info, opts)
    local arch = info.arch
    if type(arch) ~= "table" or type(arch.layers) ~= "table" then
        return nil
    end

    local max_layers = tonumber(opts.max_layers)
    if max_layers == nil then max_layers = #arch.layers end
    max_layers = clamp(max_layers, 0, #arch.layers)

    local rows = {}
    for i = 1, max_layers do
        local l = arch.layers[i]
        if type(l) == "table" then
            rows[#rows + 1] = {
                idx = tostring(i),
                name = tostring(l.name or ""),
                type = tostring(l.type or ""),
                params = (l.params_count ~= nil) and format_int(l.params_count) or "",
                weights = (l.weights_size ~= nil) and format_int(l.weights_size) or "",
                dims = layer_dims(l),
            }
        end
    end

    local columns = {
        { key = "idx", title = "#", align = "right", max = 6, color = C.gray },
        { key = "name", title = "Layer", align = "left", max = 4096, color = C.yellow },
        { key = "type", title = "Type", align = "left", max = 20, color = C.cyan },
        { key = "params", title = "Params", align = "right", max = 14, color = C.green },
        { key = "weights", title = "Weights", align = "right", max = 14, color = C.magenta },
        { key = "dims", title = "Dims", align = "left", max = 46, color = C.blue },
    }

    local table_str = make_table(columns, rows)

    if max_layers < #arch.layers then
        table_str = table_str .. "\n" .. string.format("(affiche %d/%d couches; utilisez --max-layers pour ajuster)", max_layers, #arch.layers)
    end

    return table_str
end

local function render_graph(info, opts)
    local arch = info.arch
    if type(arch) ~= "table" or type(arch.layers) ~= "table" then
        return nil
    end

    local function norm_inputs(v, i)
        if type(v) == "table" and #v > 0 then
            local out = {}
            for k = 1, #v do
                if v[k] ~= nil and tostring(v[k]) ~= "" then out[#out + 1] = tostring(v[k]) end
            end
            if #out > 0 then return out end
        elseif type(v) == "string" and v ~= "" then
            return { v }
        end

        -- DebugJson n'a pas forcément d'IO: fallback linéaire
        if i and i > 1 then
            return { "t" .. tostring(i - 1) }
        end
        return { "x" }
    end

    local function norm_output(v, i)
        if type(v) == "string" and v ~= "" then return v end
        -- DebugJson n'a pas forcément d'IO: fallback linéaire
        if i then return "t" .. tostring(i) end
        return "x"
    end

    local max_layers = tonumber(opts.max_layers)
    if max_layers == nil then max_layers = #arch.layers end
    max_layers = clamp(max_layers, 0, #arch.layers)

    local produced = {}
    local consumed = {}
    local edges = 0
    local named_tensors = {}

    local rows = {}
    for i = 1, max_layers do
        local l = arch.layers[i]
        if type(l) == "table" then
            local inputs = norm_inputs(l.inputs, i)
            local out = norm_output(l.output, i)
            produced[out] = true
            named_tensors[out] = true
            for _, inp in ipairs(inputs) do
                consumed[inp] = true
                named_tensors[inp] = true
                edges = edges + 1
            end

            rows[#rows + 1] = {
                idx = tostring(i),
                name = tostring(l.name or ""),
                type = tostring(l.type or ""),
                ins = table.concat(inputs, ","),
                out = out,
            }
        end
    end

    local sources = {}
    for tname, _ in pairs(consumed) do
        if not produced[tname] then sources[#sources + 1] = tname end
    end
    table.sort(sources)

    local sinks = {}
    for tname, _ in pairs(produced) do
        if not consumed[tname] then sinks[#sinks + 1] = tname end
    end
    table.sort(sinks)

    local named_count = 0
    for _, _ in pairs(named_tensors) do named_count = named_count + 1 end

    local header_lines = {
        string.format("Noeuds: %s | Arêtes: %s | Tensors nommés: %s", format_int(max_layers), format_int(edges), format_int(named_count)),
        string.format("Entrées (sources): %s", (#sources > 0) and table.concat(sources, ", ") or "?"),
        string.format("Sorties (sinks): %s", (#sinks > 0) and table.concat(sinks, ", ") or "?"),
    }

    local columns = {
        { key = "idx", title = "#", align = "right", max = 6, color = C.gray },
        { key = "name", title = "Layer", align = "left", max = 4096, color = C.yellow },
        { key = "type", title = "Type", align = "left", max = 20, color = C.cyan },
        { key = "ins", title = "Inputs", align = "left", max = 4096, color = C.magenta },
        { key = "out", title = "Output", align = "left", max = 4096, color = C.green },
    }

    local table_str = make_table(columns, rows)
    if max_layers < #arch.layers then
        table_str = table_str .. "\n" .. string.format("(graphe limité à %d/%d couches; utilisez --max-layers)", max_layers, #arch.layers)
    end

    return table.concat(header_lines, "\n") .. "\n\n" .. table_str
end

local function render_graph_blocks(info, opts)
    local arch = info.arch
    if type(arch) ~= "table" or type(arch.layers) ~= "table" then
        return nil
    end

    local function norm_inputs(v, i)
        if type(v) == "table" and #v > 0 then
            local out = {}
            for k = 1, #v do
                if v[k] ~= nil and tostring(v[k]) ~= "" then out[#out + 1] = tostring(v[k]) end
            end
            if #out > 0 then return out end
        elseif type(v) == "string" and v ~= "" then
            return { v }
        end

        -- DebugJson n'a pas forcément d'IO: fallback linéaire
        if i and i > 1 then
            return { "t" .. tostring(i - 1) }
        end
        return { "x" }
    end

    local function norm_output(v, i)
        if type(v) == "string" and v ~= "" then return v end
        -- DebugJson n'a pas forcément d'IO: fallback linéaire
        if i then return "t" .. tostring(i) end
        return "x"
    end

    local max_layers = tonumber(opts.max_layers)
    if max_layers == nil then max_layers = #arch.layers end
    max_layers = clamp(max_layers, 0, #arch.layers)

    -- Largeurs max des blocs (inner), avec caps (évite des lignes énormes)
    local cap_in = tonumber(opts.graph_in_width) or 4096
    local cap_out = tonumber(opts.graph_out_width) or 18
    cap_in = clamp(math.floor(cap_in), 8, 4096)
    cap_out = tonumber(opts.graph_out_width) or 4096
    cap_out = clamp(math.floor(cap_out), 6, 4096)

    local produced = {}
    local consumed = {}
    local edges = 0
    local named_tensors = {}

    local rows = {}
    local in_w, layer_w, out_w = 0, 0, 0
    local idx_w = utf8_len_safe(tostring(max_layers))

    for i = 1, max_layers do
        local l = arch.layers[i]
        if type(l) == "table" then
            local inputs = norm_inputs(l.inputs, i)
            local out = norm_output(l.output, i)
            produced[out] = true
            named_tensors[out] = true
            for _, inp in ipairs(inputs) do
                consumed[inp] = true
                named_tensors[inp] = true
                edges = edges + 1
            end

            local ins = table.concat(inputs, ",")
            local layer = tostring(l.name or "")
            local ltype = tostring(l.type or "")
            if ltype ~= "" then layer = layer .. " | " .. ltype end

            local ins_t = trunc(ins, cap_in)
            local layer_t = layer
            local out_t = trunc(out, cap_out)

            local iw = utf8_len_safe(ins_t)
            local lw = utf8_len_safe(layer_t)
            local ow = utf8_len_safe(out_t)
            if iw > in_w then in_w = iw end
            if lw > layer_w then layer_w = lw end
            if ow > out_w then out_w = ow end

            rows[#rows + 1] = {
                idx = tostring(i),
                ins = ins_t,
                layer = layer_t,
                out = out_t,
            }
        end
    end

    local sources = {}
    for tname, _ in pairs(consumed) do
        if not produced[tname] then sources[#sources + 1] = tname end
    end
    table.sort(sources)

    local sinks = {}
    for tname, _ in pairs(produced) do
        if not consumed[tname] then sinks[#sinks + 1] = tname end
    end
    table.sort(sinks)

    local named_count = 0
    for _, _ in pairs(named_tensors) do named_count = named_count + 1 end

    local header_lines = {
        string.format("Noeuds: %s | Arêtes: %s | Tensors nommés: %s", format_int(max_layers), format_int(edges), format_int(named_count)),
        string.format("Entrées (sources): %s", (#sources > 0) and table.concat(sources, ", ") or "?"),
        string.format("Sorties (sinks): %s", (#sinks > 0) and table.concat(sinks, ", ") or "?"),
        "",
        string.format("%s  [%s] -> [%s] -> [%s]", pad_left("#", idx_w), pad_right("inputs", in_w), pad_right("layer | type", layer_w), pad_right("out", out_w)),
    }

    local lines = {}
    for i = 1, #rows do
        local r = rows[i]
        local line = string.format(
            "%s  [%s] -> [%s] -> [%s]",
            pad_left(r.idx, idx_w),
            pad_right(r.ins, in_w),
            pad_right(r.layer, layer_w),
            pad_right(r.out, out_w)
        )
        lines[#lines + 1] = line
    end

    local out = table.concat(header_lines, "\n") .. "\n" .. table.concat(lines, "\n")
    if max_layers < #arch.layers then
        out = out .. "\n" .. string.format("(graphe limité à %d/%d couches; utilisez --max-layers)", max_layers, #arch.layers)
    end
    return out
end

local function render_graph_mermaid_markdown(info, opts)
    local arch = info.arch
    if type(arch) ~= "table" or type(arch.layers) ~= "table" then
        return nil
    end

    local max_layers = tonumber(opts.max_layers)
    if max_layers == nil then max_layers = #arch.layers end
    max_layers = clamp(max_layers, 0, #arch.layers)

    local function esc_label(s)
        s = tostring(s or "")
        s = s:gsub("\\r", " "):gsub("\\n", " ")
        -- mermaid labels in ["..."] => éviter les guillemets
        s = s:gsub('"', "'")
        return s
    end

    local function norm_inputs(v, i)
        if type(v) == "table" and #v > 0 then
            local out = {}
            for k = 1, #v do
                if v[k] ~= nil and tostring(v[k]) ~= "" then out[#out + 1] = tostring(v[k]) end
            end
            if #out > 0 then return out end
        elseif type(v) == "string" and v ~= "" then
            return { v }
        end

        -- DebugJson n'a pas d'IO: fallback linéaire
        if i and i > 1 then
            return { "t" .. tostring(i - 1) }
        end
        return { "x" }
    end

    local function norm_output(v, i)
        if type(v) == "string" and v ~= "" then return v end
        -- DebugJson n'a pas d'IO: fallback linéaire
        if i then return "t" .. tostring(i) end
        return "x"
    end

    local producers = {}  -- tensor -> layer idx
    local consumers = {}  -- tensor -> true
    local edges = {}
    local source_nodes = {} -- tensor -> nodeId

    local function layer_id(i)
        return "L" .. tostring(i)
    end

    local function source_id(tensor)
        if source_nodes[tensor] then return source_nodes[tensor] end
        -- id mermaid: alnum/_ uniquement
        local clean = tostring(tensor):gsub("[^%w_]", "_")
        clean = clean:gsub("_+", "_")
        if clean == "" then clean = "x" end
        local id = "S_" .. clean
        -- collision possible => stabiliser via suffix
        local base = id
        local n = 1
        while source_nodes[id] do
            n = n + 1
            id = base .. "_" .. tostring(n)
        end
        source_nodes[tensor] = id
        return id
    end

    -- Pass 1: compute producers
    for i = 1, max_layers do
        local l = arch.layers[i]
        if type(l) == "table" then
            local out = norm_output(l.output, i)
            producers[out] = i
        end
    end

    -- Pass 2: create edges between producing layers and consuming layers
    for i = 1, max_layers do
        local l = arch.layers[i]
        if type(l) == "table" then
            local inputs = norm_inputs(l.inputs, i)
            for _, inp in ipairs(inputs) do
                consumers[inp] = true
                local p = producers[inp]
                if p and p ~= i then
                    edges[#edges + 1] = { from = layer_id(p), to = layer_id(i), label = esc_label(inp) }
                else
                    local sid = source_id(inp)
                    edges[#edges + 1] = { from = sid, to = layer_id(i), label = esc_label(inp) }
                end
            end
        end
    end

    -- Build Mermaid
    local lines = {}
    lines[#lines + 1] = "```mermaid"
    lines[#lines + 1] = "flowchart LR"

    -- Layer nodes
    for i = 1, max_layers do
        local l = arch.layers[i]
        if type(l) == "table" then
            local name = esc_label(l.name or "")
            local typ = esc_label(l.type or "")
            local label = tostring(i)
            if name ~= "" then label = label .. ": " .. name end
            if typ ~= "" then label = label .. " | " .. typ end
            lines[#lines + 1] = string.format("  %s[\"%s\"]", layer_id(i), label)
        end
    end

    -- Source nodes
    for tensor, sid in pairs(source_nodes) do
        lines[#lines + 1] = string.format("  %s((\"%s\"))", sid, esc_label(tensor))
    end

    -- Edges
    for i = 1, #edges do
        local e = edges[i]
        if e.label and e.label ~= "" then
            lines[#lines + 1] = string.format("  %s -->|%s| %s", e.from, e.label, e.to)
        else
            lines[#lines + 1] = string.format("  %s --> %s", e.from, e.to)
        end
    end

    lines[#lines + 1] = "```"

    if max_layers < #arch.layers then
        lines[#lines + 1] = string.format("(graphe limité à %d/%d couches; utilisez --max-layers)", max_layers, #arch.layers)
    end

    return table.concat(lines, "\n")
end

local function render_graph_visu_bloc(info, opts)
    local arch = info.arch
    if type(arch) ~= "table" or type(arch.layers) ~= "table" then
        return nil
    end

    local max_layers = tonumber(opts.max_layers)
    if max_layers == nil then max_layers = #arch.layers end
    max_layers = clamp(max_layers, 0, #arch.layers)

    -- Chaîne compatible avec les littéraux du pseudocode Visu.
    local function visu_string(s)
        s = tostring(s or "")
        s = s:gsub("\\", "\\\\")
        s = s:gsub('"', '\\"')
        s = s:gsub("\r", "\\r")
        s = s:gsub("\n", "\\n")
        s = s:gsub("\t", "\\t")
        return '"' .. s .. '"'
    end

    local function norm_inputs(v, i)
        if type(v) == "table" and #v > 0 then
            local out = {}
            for k = 1, #v do
                local value = tostring(v[k] or "")
                if value ~= "" then out[#out + 1] = value end
            end
            if #out > 0 then return out end
        elseif type(v) == "string" and v ~= "" then
            return { v }
        end

        -- DebugJson n'a pas forcément d'IO: fallback linéaire.
        if i and i > 1 then return { "t" .. tostring(i - 1) } end
        return { "x" }
    end

    local function norm_output(v, i)
        if type(v) == "string" and v ~= "" then return v end
        if i then return "t" .. tostring(i) end
        return "x"
    end

    local function visu_list(values)
        local encoded = {}
        for i = 1, #values do
            local value = values[i]
            if type(value) == "number" then
                encoded[#encoded + 1] = tostring(value)
            else
                encoded[#encoded + 1] = visu_string(value)
            end
        end
        return "[" .. table.concat(encoded, ", ") .. "]"
    end

    local function convolution_kind(layer)
        local typ = tostring((type(layer) == "table" and layer.type) or "")
        local lower = typ:lower()
        if lower:find("transpose", 1, true) or lower:find("deconv", 1, true) then
            return "up-convolution"
        elseif lower:find("conv", 1, true) then
            return "convolution"
        elseif lower:find("pool", 1, true) then
            return "pooling"
        elseif lower:find("norm", 1, true) then
            return "normalization"
        elseif lower:find("relu", 1, true) or lower:find("gelu", 1, true)
            or lower:find("silu", 1, true) or lower:find("activation", 1, true) then
            return "activation"
        elseif lower:find("concat", 1, true) or lower == "add"
            or lower:find("residual", 1, true) then
            return "merge"
        elseif lower:find("reshape", 1, true) or lower:find("permute", 1, true)
            or lower:find("flatten", 1, true) then
            return "reshape"
        elseif lower:find("linear", 1, true) or lower:find("dense", 1, true) then
            return "fully-connected"
        end
        return "operation"
    end

    local function convolution_layer_payload(layer, index)
        local values = {
            convolution_kind(layer),
            index,
            tostring(layer.name or ("layer_" .. tostring(index))),
            tostring(layer.type or "unknown"),
        }
        local dims = layer_dims(layer)
        if dims ~= "" then values[#values + 1] = dims end
        local params = tonumber(layer.params_count)
        if params then
            values[#values + 1] = "params"
            values[#values + 1] = math.floor(params)
        end
        return values
    end

    local model = (type(arch.model) == "table") and arch.model or {}
    local model_name = arch.model_name or arch.name or model.name or info.path or "model"
    local model_type = arch.model_type or arch.model_type_name or arch.model_kind
        or model.type or "unknown"
    local total_params = tonumber(arch.total_params or model.total_params)

    local tensors = {}
    local tensor_order = {}
    local produced = {}
    local consumed = {}
    local layer_inputs = {}
    local layer_outputs = {}

    local function tensor_id(name)
        name = tostring(name)
        if not tensors[name] then
            tensors[name] = "T" .. tostring(#tensor_order + 1)
            tensor_order[#tensor_order + 1] = name
        end
        return tensors[name]
    end

    for i = 1, max_layers do
        local layer = arch.layers[i]
        if type(layer) == "table" then
            local inputs = norm_inputs(layer.inputs, i)
            local output = norm_output(layer.output, i)
            layer_inputs[i] = inputs
            layer_outputs[i] = output
            for _, input in ipairs(inputs) do
                consumed[input] = true
                tensor_id(input)
            end
            produced[output] = true
            tensor_id(output)
        end
    end

    local layer_ids = {}
    for i = 1, max_layers do
        if type(arch.layers[i]) == "table" then
            layer_ids[#layer_ids + 1] = visu_string("L" .. tostring(i))
        end
    end

    local lines = {
        "# Modèle IA généré par analyze_model.lua",
        "# Les structures sont séparées en métadonnées, pipeline CNN et cartes d'activation.",
        "",
        "map model_info = []",
        string.format("model_info.set(%s, %s)", visu_string("name"), visu_string(model_name)),
        string.format("model_info.set(%s, %s)", visu_string("type"), visu_string(model_type)),
        string.format("model_info.set(%s, %d)", visu_string("layers"), max_layers),
    }
    if total_params then
        lines[#lines + 1] = string.format(
            "model_info.set(%s, %d)",
            visu_string("parameters"),
            math.floor(total_params)
        )
    end

    lines[#lines + 1] = ""
    lines[#lines + 1] = "# Ordre séquentiel déclaré par le modèle"
    lines[#lines + 1] = "array layer_order = [" .. table.concat(layer_ids, ", ") .. "]"
    lines[#lines + 1] = ""
    lines[#lines + 1] = "# Pipeline CNN linéaire; une liste évite le layout circulaire des graphes Visu"
    lines[#lines + 1] = "list cnn_blocks = []"
    for i = 1, max_layers do
        local layer = arch.layers[i]
        if type(layer) == "table" then
            lines[#lines + 1] = string.format(
                "cnn_blocks.append(%s)",
                visu_list(convolution_layer_payload(layer, i))
            )
        end
    end

    lines[#lines + 1] = ""
    lines[#lines + 1] = "# Cartes d'activation et routes explicites (branches / skip connections)"
    lines[#lines + 1] = "map feature_maps = []"
    for _, tensor in ipairs(tensor_order) do
        local role = "activation map"
        if not produced[tensor] then
            role = "input image"
        elseif not consumed[tensor] then
            role = "output prediction"
        end
        lines[#lines + 1] = string.format(
            "feature_maps.set(%s, %s)",
            visu_string(tensor),
            visu_string(role)
        )
    end

    lines[#lines + 1] = "map routes = []"
    for i = 1, max_layers do
        if type(arch.layers[i]) == "table" then
            local encoded_inputs = {}
            for _, input in ipairs(layer_inputs[i] or {}) do
                encoded_inputs[#encoded_inputs + 1] = visu_string(input)
            end
            local output = layer_outputs[i]
            lines[#lines + 1] = string.format(
                "routes.set(%s, %s)",
                visu_string("L" .. tostring(i) .. " -> " .. tostring(output)),
                "[" .. table.concat(encoded_inputs, ", ") .. "]"
            )
        end
    end

    if max_layers < #arch.layers then
        lines[#lines + 1] = ""
        lines[#lines + 1] = string.format(
            "# Graphe limité à %d/%d couches; utilisez --max-layers",
            max_layers,
            #arch.layers
        )
    end

    return table.concat(lines, "\n")
end

local function render_graph_visu_tree(info, opts)
    local arch = info.arch
    if type(arch) ~= "table" or type(arch.layers) ~= "table" then
        return nil
    end

    local max_layers = tonumber(opts.max_layers)
    if max_layers == nil then max_layers = #arch.layers end
    max_layers = clamp(max_layers, 0, #arch.layers)

    local max_units = tonumber(opts.graph_units) or 8
    max_units = clamp(math.floor(max_units), 1, 32)

    local function visu_string(s)
        s = tostring(s or "")
        s = s:gsub("\\", "\\\\"):gsub('"', '\\"')
        s = s:gsub("\r", "\\r"):gsub("\n", "\\n"):gsub("\t", "\\t")
        return '"' .. s .. '"'
    end

    local function visu_list(values)
        local encoded = {}
        for i = 1, #values do
            if type(values[i]) == "number" then
                encoded[#encoded + 1] = tostring(values[i])
            else
                encoded[#encoded + 1] = visu_string(values[i])
            end
        end
        return "[" .. table.concat(encoded, ", ") .. "]"
    end

    local function norm_inputs(v, i)
        if type(v) == "table" and #v > 0 then
            local out = {}
            for k = 1, #v do
                local value = tostring(v[k] or "")
                if value ~= "" then out[#out + 1] = value end
            end
            if #out > 0 then return out end
        elseif type(v) == "string" and v ~= "" then
            return { v }
        end
        if i and i > 1 then return { "t" .. tostring(i - 1) } end
        return { "x" }
    end

    local function norm_output(v, i)
        if type(v) == "string" and v ~= "" then return v end
        return "t" .. tostring(i or 0)
    end

    local function positive_int(v)
        v = tonumber(v)
        if not v or v <= 0 then return nil end
        return math.floor(v)
    end

    local function unit_counts(layer, fallback)
        local cfg = (type(layer.config) == "table") and layer.config or layer
        local inputs = positive_int(cfg.in_channels)
            or positive_int(cfg.in_features)
            or fallback
            or 1
        local outputs = positive_int(cfg.out_channels)
            or positive_int(cfg.out_features)
            or inputs
        return inputs, outputs
    end

    local function is_fully_connected(layer)
        local typ = tostring(layer.type or ""):lower()
        return typ:find("conv", 1, true) ~= nil
            or typ:find("linear", 1, true) ~= nil
            or typ:find("dense", 1, true) ~= nil
            or typ:find("concat", 1, true) ~= nil
            or typ == "add"
    end

    local consumed = {}
    for i = 1, max_layers do
        local layer = arch.layers[i]
        if type(layer) == "table" then
            for _, input in ipairs(norm_inputs(layer.inputs, i)) do
                consumed[input] = true
            end
        end
    end

    local lines = {
        "# Vue neuronale générée par analyze_model.lua",
        "# Chaque tableau représente une couche; aucun layout circulaire n'est utilisé.",
        "map layer_info = []",
        "map connections = []",
    }
    local tensor_units = {}
    local input_index = 0

    local function add_group(prefix, count, role, detail)
        local ids = {}
        local encoded_ids = {}
        local visible = math.min(count, max_units)
        for unit = 1, visible do
            local id = prefix .. "_" .. tostring(unit)
            ids[#ids + 1] = id
            encoded_ids[#encoded_ids + 1] = visu_string(id)
        end
        lines[#lines + 1] = string.format(
            "array %s = [%s]",
            prefix,
            table.concat(encoded_ids, ", ")
        )
        lines[#lines + 1] = string.format(
            "layer_info.set(%s, %s)",
            visu_string(prefix),
            visu_list({ role, detail, count })
        )
        return ids
    end

    local function connect(from_ids, to_ids, dense)
        if dense then
            local encoded_from = {}
            for _, from_id in ipairs(from_ids) do
                encoded_from[#encoded_from + 1] = visu_string(from_id)
            end
            for _, to_id in ipairs(to_ids) do
                lines[#lines + 1] = string.format(
                    "connections.set(%s, [%s])",
                    visu_string(to_id),
                    table.concat(encoded_from, ", ")
                )
            end
        elseif #from_ids > 0 then
            for index, to_id in ipairs(to_ids) do
                local from_id = from_ids[((index - 1) % #from_ids) + 1]
                lines[#lines + 1] = string.format(
                    "connections.set(%s, [%s])",
                    visu_string(to_id),
                    visu_string(from_id)
                )
            end
        end
    end

    for i = 1, max_layers do
        local layer = arch.layers[i]
        if type(layer) == "table" then
            local incoming = {}
            local inputs = norm_inputs(layer.inputs, i)
            for _, input in ipairs(inputs) do
                local group = tensor_units[input]
                if not group then
                    local input_count = select(1, unit_counts(layer, nil))
                    input_index = input_index + 1
                    group = add_group("X" .. tostring(input_index), input_count, "input", input)
                    tensor_units[input] = group
                end
                for _, id in ipairs(group) do incoming[#incoming + 1] = id end
            end

            local _, output_count = unit_counts(layer, #incoming)
            local output = norm_output(layer.output, i)
            local role = consumed[output] and ("hidden layer " .. tostring(i)) or "output layer"
            local detail = tostring(layer.name or ("layer_" .. tostring(i)))
                .. " | " .. tostring(layer.type or "unknown")
            local dims = layer_dims(layer)
            if dims ~= "" then detail = detail .. " | " .. dims end
            local outgoing = add_group("H" .. tostring(i), output_count, role, detail)
            connect(incoming, outgoing, is_fully_connected(layer))
            tensor_units[output] = outgoing
        end
    end

    lines[#lines + 1] = ""
    lines[#lines + 1] = string.format(
        "# Limite visuelle: %d neurones/canaux par couche (--graph-units)",
        max_units
    )
    if max_layers < #arch.layers then
        lines[#lines + 1] = string.format(
            "# Graphe limité à %d/%d couches; utilisez --max-layers",
            max_layers,
            #arch.layers
        )
    end
    return table.concat(lines, "\n")
end

local function render_graph_mlp_markdown(info, opts)
    local arch = info.arch
    if type(arch) ~= "table" then
        return nil
    end

    local function n(x)
        x = tonumber(x)
        if not x or x <= 0 then return nil end
        return math.floor(x)
    end

    local max_layers = tonumber(opts.max_layers)
    local layers = (type(arch.layers) == "table") and arch.layers or {}
    if max_layers == nil then max_layers = #layers end
    max_layers = clamp(max_layers, 0, #layers)

    local function esc_label(s)
        s = tostring(s or "")
        s = s:gsub("\\r", " "):gsub("\\n", " ")
        s = s:gsub('"', "'")
        return s
    end

    local function short_name(s)
        s = tostring(s or "")
        local last = s:match("([^/]+)$")
        if last and last ~= "" then return last end
        return s
    end

    local function norm_inputs(v, i)
        if type(v) == "table" and #v > 0 then
            local out = {}
            for k = 1, #v do
                local x = tostring(v[k] or "")
                if x ~= "" then out[#out + 1] = x end
            end
            if #out > 0 then return out end
        elseif type(v) == "string" and v ~= "" then
            return { v }
        end
        if i and i > 1 then return { "t" .. tostring(i - 1) } end
        return { "x" }
    end

    local function norm_output(v, i)
        if type(v) == "string" and v ~= "" then return v end
        if i then return "t" .. tostring(i) end
        return "x"
    end

    local producers = {}
    local indeg = {}
    local outdeg = {}
    local source_nodes = {}
    local layer_edges = {}
    local source_edges = {}

    local function layer_id(i)
        return "L" .. tostring(i)
    end

    local function source_id(tensor)
        if source_nodes[tensor] then return source_nodes[tensor] end
        local clean = tostring(tensor):gsub("[^%w_]", "_"):gsub("_+", "_")
        if clean == "" then clean = "x" end
        local id = "S_" .. clean
        local base = id
        local m = 1
        while source_nodes[id] do
            m = m + 1
            id = base .. "_" .. tostring(m)
        end
        source_nodes[tensor] = id
        return id
    end

    for i = 1, max_layers do
        local l = layers[i]
        if type(l) == "table" then
            local out = norm_output(l.output, i)
            producers[out] = i
            indeg[i] = indeg[i] or 0
            outdeg[i] = outdeg[i] or 0
        end
    end

    for i = 1, max_layers do
        local l = layers[i]
        if type(l) == "table" then
            local inputs = norm_inputs(l.inputs, i)
            for _, inp in ipairs(inputs) do
                local p = producers[inp]
                if p and p ~= i then
                    layer_edges[#layer_edges + 1] = { from = p, to = i, label = inp }
                    indeg[i] = (indeg[i] or 0) + 1
                    outdeg[p] = (outdeg[p] or 0) + 1
                else
                    source_edges[#source_edges + 1] = { from = inp, to = i, label = inp }
                    indeg[i] = (indeg[i] or 0) + 1
                end
            end
        end
    end

    local has_real_edges = (#layer_edges + #source_edges) > 0

    -- Si on n'a pas un graphe exploitable, fallback sur un mini schéma MLP.
    if not has_real_edges then
        local sizes = {}
        for i = 1, max_layers do
            local l = layers[i]
            if type(l) == "table" then
                local cfg = (type(l.config) == "table") and l.config or l
                local in_f = n(cfg.in_features) or n(cfg.input_dim) or n(cfg.input_size) or n(cfg.in_dim)
                local out_f = n(cfg.out_features) or n(cfg.output_dim) or n(cfg.output_size) or n(cfg.out_dim)
                if in_f and out_f then
                    if #sizes == 0 then sizes[1] = in_f end
                    sizes[#sizes + 1] = out_f
                end
            end
        end
        if #sizes < 2 then sizes = { 2, 4, 3 } end

        local lines_fb = {}
        lines_fb[#lines_fb + 1] = "```mermaid"
        lines_fb[#lines_fb + 1] = "flowchart LR"
        lines_fb[#lines_fb + 1] = "  classDef input fill:#ef7f1a,stroke:#b95e17,color:#111,stroke-width:1px"
        lines_fb[#lines_fb + 1] = "  classDef hidden fill:#b6b2b2,stroke:#777,color:#111,stroke-width:1px"
        lines_fb[#lines_fb + 1] = "  classDef output fill:#04b454,stroke:#03813c,color:#111,stroke-width:1px"
        for li = 1, #sizes do
            local kind = (li == 1) and "input" or ((li == #sizes) and "output" or "hidden")
            local id = "F" .. tostring(li)
            lines_fb[#lines_fb + 1] = string.format("  %s[\"Couche %d\\n(%s neurones)\"]", id, li, format_int(sizes[li]))
            lines_fb[#lines_fb + 1] = string.format("  class %s %s", id, kind)
            if li > 1 then
                lines_fb[#lines_fb + 1] = string.format("  F%d --> %s", li - 1, id)
            end
        end
        lines_fb[#lines_fb + 1] = "```"
        lines_fb[#lines_fb + 1] = ""
        lines_fb[#lines_fb + 1] = "Mode fallback: topologie reelle indisponible, schema simplifie."
        return table.concat(lines_fb, "\n")
    end

    local lines = {}
    lines[#lines + 1] = "```mermaid"
    lines[#lines + 1] = "flowchart LR"
    lines[#lines + 1] = "  classDef input fill:#ef7f1a,stroke:#b95e17,color:#111,stroke-width:1px"
    lines[#lines + 1] = "  classDef hidden fill:#b6b2b2,stroke:#777,color:#111,stroke-width:1px"
    lines[#lines + 1] = "  classDef output fill:#04b454,stroke:#03813c,color:#111,stroke-width:1px"
    lines[#lines + 1] = "  classDef source fill:#ffffff,stroke:#999,color:#333,stroke-dasharray: 3 3"

    for i = 1, max_layers do
        local l = layers[i]
        if type(l) == "table" then
            local typ = esc_label(l.type or "?")
            local nm = esc_label(short_name(l.name or ("layer_" .. tostring(i))))
            local dims = esc_label(layer_dims(l) or "")
            local label = string.format("%d: %s | %s", i, nm, typ)
            if dims ~= "" then
                label = label .. "\\n" .. dims
            end
            lines[#lines + 1] = string.format("  %s[\"%s\"]", layer_id(i), label)
        end
    end

    for tensor, sid in pairs(source_nodes) do
        lines[#lines + 1] = string.format("  %s((\"%s\"))", sid, esc_label(short_name(tensor)))
        lines[#lines + 1] = string.format("  class %s source", sid)
    end

    for i = 1, #layer_edges do
        local e = layer_edges[i]
        lines[#lines + 1] = string.format("  %s --> %s", layer_id(e.from), layer_id(e.to))
    end
    for i = 1, #source_edges do
        local e = source_edges[i]
        lines[#lines + 1] = string.format("  %s --> %s", source_id(e.from), layer_id(e.to))
    end

    local inputs = {}
    local hiddens = {}
    local outputs = {}
    for i = 1, max_layers do
        local id = layer_id(i)
        local di = indeg[i] or 0
        local do_ = outdeg[i] or 0
        if di == 0 then
            inputs[#inputs + 1] = id
        elseif do_ == 0 then
            outputs[#outputs + 1] = id
        else
            hiddens[#hiddens + 1] = id
        end
    end
    if #inputs > 0 then lines[#lines + 1] = "  class " .. table.concat(inputs, ",") .. " input" end
    if #hiddens > 0 then lines[#lines + 1] = "  class " .. table.concat(hiddens, ",") .. " hidden" end
    if #outputs > 0 then lines[#lines + 1] = "  class " .. table.concat(outputs, ",") .. " output" end

    lines[#lines + 1] = "```"
    lines[#lines + 1] = ""
    lines[#lines + 1] = string.format("Diagramme fonctionnel: %s couches, %s connexions inter-couches, %s entrées externes.",
        format_int(max_layers), format_int(#layer_edges), format_int(#source_edges))

    if max_layers < #layers then
        lines[#lines + 1] = string.format("(graphe limité à %d/%d couches; utilisez --max-layers)", max_layers, #layers)
    end

    return table.concat(lines, "\n")
end

local function render_top_tensors(info, opts)
    local tensors = info.tensors
    if type(tensors) ~= "table" then return nil end

    local top = tonumber(opts.top_tensors) or 20
    top = clamp(top, 0, 200)

    -- Sort by bytes
    local list = {}
    for i = 1, #tensors do list[i] = tensors[i] end

    table.sort(list, function(a, b)
        return (a.bytes_raw or 0) > (b.bytes_raw or 0)
    end)

    local rows = {}
    for i = 1, math.min(top, #list) do
        local t = list[i]
        rows[#rows + 1] = {
            idx = tostring(i),
            name = t.name or "",
            dtype = t.dtype or "",
            shape = t.shape or "",
            elems = t.elems or "",
            bytes = t.bytes or format_bytes(t.bytes_raw or 0),
        }
    end

    local columns = {
        { key = "idx", title = "#", align = "right", max = 6, color = C.gray },
        { key = "name", title = "Tensor", align = "left", max = 78, color = C.yellow },
        { key = "dtype", title = "DType", align = "left", max = 8, color = C.cyan },
        { key = "shape", title = "Shape", align = "left", max = 36, color = C.blue },
        { key = "elems", title = "Elems", align = "right", max = 16, color = C.magenta },
        { key = "bytes", title = "Taille", align = "right", max = 12, color = C.green },
    }

    return make_table(columns, rows)
end



-- ---------------------------------------------------------------------------
-- Main
-- ---------------------------------------------------------------------------

local opts = Args.parse(arg) or {}

local IN = Args.get_str(opts, "in", Args.get_str(opts, "path", Args.get_str(opts, "checkpoint", "")))

-- Small optional knobs (reste simple)
opts.max_layers = Args.get_num(opts, "max-layers", Args.get_num(opts, "max_layers", nil))
opts.top_tensors = Args.get_num(opts, "top-tensors", Args.get_num(opts, "top_tensors", 20))
opts.enrich_tensors = Args.get_num(opts, "enrich-tensors", Args.get_num(opts, "enrich_tensors", 25))
opts.graph_blocks = Args.get_bool(opts, "graph-blocks", Args.get_bool(opts, "graph_blocks", false))
opts.graph_in_width = Args.get_num(opts, "graph-in-width", Args.get_num(opts, "graph_in_width", nil))
opts.graph_layer_width = Args.get_num(opts, "graph-layer-width", Args.get_num(opts, "graph_layer_width", nil))
opts.graph_out_width = Args.get_num(opts, "graph-out-width", Args.get_num(opts, "graph_out_width", nil))
opts.graph_units = Args.get_num(opts, "graph-units", Args.get_num(opts, "graph_units", 8))
opts.graph_format = Args.get_str(opts, "graph-format", Args.get_str(opts, "graph_format", "table"))
opts.graph_out = Args.get_str(opts, "graph-out", Args.get_str(opts, "graph_out", ""))
opts.debug = Args.get_bool(opts, "debug", false)
opts.all = Args.get_bool(opts, "all", false)

opts.script_help = Args.get_bool(opts, "script-help", Args.get_bool(opts, "script_help",
    Args.get_bool(opts, "help-script", Args.get_bool(opts, "help_script",
        Args.get_bool(opts, "h", false)
    ))
))

if IN == "" or opts.script_help then
    ToolHelp.print("analyze_model")
    if IN == "" and not opts.script_help then
        os.exit(1)
    else
        os.exit(0)
    end
end

local is_safetensors = IN:match("%.safetensors$") or IN:match("%.st$")
local is_raw = file_exists(IN:gsub("/*$", "") .. "/manifest.json")

local is_debug_json = false
if (not is_raw) and file_exists(IN) and IN:match("%.json$") then
    -- heuristique rapide (évite de parser des JSON non-model)
    local pfx = read_prefix(IN, 256 * 1024) or ""
    if pfx:find('"mimir_debug_dump"', 1, true) or pfx:find('"model_name"', 1, true) or pfx:find('"layers"', 1, true) then
        local t = select(1, safe_read_json(IN))
        if type(t) == "table" then
            if t.format == "mimir_debug_dump" then
                is_debug_json = true
            elseif t.model_name ~= nil and t.layers ~= nil and t.format_version ~= nil then
                is_debug_json = true
            elseif type(t.model) == "table" and t.layers ~= nil and t.format_version ~= nil then
                is_debug_json = true
            end
        end
    end
end

local info, err
if is_raw then
    info, err = analyze_raw_folder(IN, opts)
elseif is_debug_json then
    info, err = analyze_debug_json(IN, opts)
elseif is_safetensors or file_exists(IN) then
    info, err = analyze_safetensors(IN, opts)
else
    die("chemin introuvable ou format non reconnu: " .. tostring(IN))
end

if not info then
    die(err or "analyse échouée")
end

log(colorize("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C.blue, C.bold))
log(colorize("  Analyse modèle", C.bold, C.cyan))
log(colorize("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C.blue, C.bold))
log("")

log(render_summary(info, opts))

local fw_state = render_framework_state(info, opts)
if fw_state then
    log("")
    log(colorize("État framework (snapshot)", C.bold, C.blue))
    log(fw_state)
end

do
    local gf = tostring(opts.graph_format or "table"):lower()
    if gf == "mermaid" or gf == "mermaind" then
        local md = render_graph_mermaid_markdown(info, opts)
        if md then
            log("")
            log(colorize("Graphe (Mermaid/Markdown)", C.bold, C.blue))
            log(md)
        end
    elseif gf == "visu-tree" or gf == "visu_tree" then
        local visu = render_graph_visu_tree(info, opts)
        if visu then
            log("")
            log(colorize("Graphe neuronal (pseudocode Visu Tree)", C.bold, C.blue))
            log(visu)
        end
    elseif gf == "visu" or gf == "visu-bloc" or gf == "visu_bloc" then
        local visu = render_graph_visu_bloc(info, opts)
        if visu then
            log("")
            log(colorize("Architecture CNN (pseudocode Visu Bloc)", C.bold, C.blue))
            log(visu)
        end
    elseif gf == "mlp_graph" then
        local mlp = render_graph_mlp_markdown(info, opts)
        if mlp then
            log("")
            log(colorize("Graphe (MLP image/Mermaid)", C.bold, C.blue))
            log(mlp)

            if opts.graph_out ~= "" then
                local out_path, ex_err = export_mermaid_graph(mlp, opts.graph_out)
                if not out_path then
                    die("export image échoué: " .. tostring(ex_err))
                end
                log("")
                log(colorize("Image exportée", C.bold, C.green) .. " " .. tostring(out_path))
            end
        end
    elseif gf == "blocks" then
        local blocks = render_graph_blocks(info, opts)
        if blocks then
            log("")
            log(colorize("Graphe (blocks)", C.bold, C.blue))
            log(blocks)
        end
    else
        local graph = render_graph(info, opts)
        if graph then
            log("")
            log(colorize("Résumé (graphe)", C.bold, C.blue))
            log(graph)
        end

        if opts.graph_blocks then
            local blocks = render_graph_blocks(info, opts)
            if blocks then
                log("")
                log(colorize("Graphe (blocks)", C.bold, C.blue))
                log(blocks)
            end
        end
    end
end

local layers = render_layers(info, opts)
if layers then
    log("")
    log(colorize("Couches", C.bold, C.blue))
    log(layers)
end

local top_t = render_top_tensors(info, opts)
if top_t then
    log("")
    log(colorize("Top tensors (par taille)", C.bold, C.blue))
    log(top_t)
end

log("")
log(colorize("✓ Analyse terminée", C.bold, C.green))
