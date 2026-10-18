---@diagnostic disable: undefined-global

-- MPK (Mimir Package Template) helpers.
-- Modern MPK has one source format and one compiled representation:
--   1) pseudocode : readable Visu-like map/list source
--   2) binary-v4  : opaque typed representation (no embedded source)
-- Legacy JSON and binary-v1/v2/v3 remain readable, but are never written.

---@class MimirMPKModule
local M = {}

local MPK_BINARY_MAGIC = "MPKB"
local MPK_BINARY_HEADER_SIZE = 64
local MPK_BINARY_VERSION = 4

local function is_table(v)
  return type(v) == "table"
end

local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t or {}) do out[k] = v end
  return out
end

local function utc_iso8601()
  return os.date("!%Y-%m-%dT%H:%M:%SZ")
end

local function utc_epoch_u32()
  local n = tonumber(os.time()) or 0
  if n < 0 then n = 0 end
  if n > 0xFFFFFFFF then n = 0xFFFFFFFF end
  return math.floor(n)
end

local function is_array(t)
  if type(t) ~= "table" then return false end
  local n = #t
  for k in pairs(t) do
    if type(k) ~= "number" or k < 1 or k > n or k ~= math.floor(k) then
      return false
    end
  end
  return true
end

local function json_escape(s)
  s = tostring(s or "")
  s = s:gsub("\\", "\\\\")
  s = s:gsub('"', '\\"')
  s = s:gsub("\n", "\\n")
  s = s:gsub("\r", "\\r")
  s = s:gsub("\t", "\\t")
  return s
end

local function json_encode(v, indent)
  indent = indent or 0
  local sp = string.rep("  ", indent)
  local sp2 = string.rep("  ", indent + 1)
  local t = type(v)

  if t == "nil" then return "null" end
  if t == "boolean" then return tostring(v) end
  if t == "number" then
    if v == math.floor(v) and math.abs(v) < 2^53 then
      return string.format("%d", v)
    end
    return string.format("%.10g", v)
  end
  if t == "string" then
    return '"' .. json_escape(v) .. '"'
  end

  if t ~= "table" then
    return '"[unsupported:' .. tostring(t) .. ']"'
  end

  if is_array(v) then
    if #v == 0 then return "[]" end
    local parts = {}
    for i = 1, #v do
      parts[#parts + 1] = sp2 .. json_encode(v[i], indent + 1)
    end
    return "[\n" .. table.concat(parts, ",\n") .. "\n" .. sp .. "]"
  end

  local keys = {}
  for k in pairs(v) do keys[#keys + 1] = k end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  if #keys == 0 then return "{}" end

  local parts = {}
  for _, k in ipairs(keys) do
    parts[#parts + 1] = sp2 .. '"' .. json_escape(k) .. '": ' .. json_encode(v[k], indent + 1)
  end
  return "{\n" .. table.concat(parts, ",\n") .. "\n" .. sp .. "}"
end

local function parse_json_with_fallback(s)
  s = tostring(s or "")
  -- Sanitize common transport artifacts.
  if s:sub(1, 3) == "\239\187\191" then
    s = s:sub(4) -- UTF-8 BOM
  end
  s = s:gsub("^%z+", ""):gsub("%z+$", "")

  local json_mod = rawget(_G, "json")
  if type(json_mod) == "table" and type(json_mod.decode) == "function" then
    local ok, v = pcall(json_mod.decode, s)
    if ok then return v end
  end

  local cjson_mod = rawget(_G, "cjson")
  if type(cjson_mod) == "table" and type(cjson_mod.decode) == "function" then
    local ok, v = pcall(cjson_mod.decode, s)
    if ok then return v end
  end

  -- Fallback parser without external dependency.
  local i = 1
  local n = #s

  local function skip_ws()
    while i <= n do
      local c = s:sub(i, i)
      if c == " " or c == "\t" or c == "\n" or c == "\r" or c == "\0" then
        i = i + 1
      else
        break
      end
    end
  end

  local parse_value

  local function parse_string()
    if s:sub(i, i) ~= '"' then return nil, "expected string" end
    i = i + 1
    local out = {}
    while i <= n do
      local c = s:sub(i, i)
      if c == '"' then
        i = i + 1
        return table.concat(out)
      end
      if c == "\\" then
        local e = s:sub(i + 1, i + 1)
        if e == "" then return nil, "unterminated escape" end
        if e == '"' or e == "\\" or e == "/" then
          out[#out + 1] = e
          i = i + 2
        elseif e == "b" then
          out[#out + 1] = "\b"
          i = i + 2
        elseif e == "f" then
          out[#out + 1] = "\f"
          i = i + 2
        elseif e == "n" then
          out[#out + 1] = "\n"
          i = i + 2
        elseif e == "r" then
          out[#out + 1] = "\r"
          i = i + 2
        elseif e == "t" then
          out[#out + 1] = "\t"
          i = i + 2
        elseif e == "u" then
          local hex = s:sub(i + 2, i + 5)
          if #hex < 4 then return nil, "bad unicode escape" end
          local code = tonumber(hex, 16)
          if not code then return nil, "bad unicode escape" end
          if code < 0x80 then
            out[#out + 1] = string.char(code)
          elseif code < 0x800 then
            out[#out + 1] = string.char(0xC0 + math.floor(code / 0x40), 0x80 + (code % 0x40))
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
    return nil, "unterminated string"
  end

  local function parse_number()
    local start = i
    local c = s:sub(i, i)
    if c == "-" then i = i + 1 end
    while i <= n and s:sub(i, i):match("%d") do i = i + 1 end
    if s:sub(i, i) == "." then
      i = i + 1
      while i <= n and s:sub(i, i):match("%d") do i = i + 1 end
    end
    local e = s:sub(i, i)
    if e == "e" or e == "E" then
      i = i + 1
      local sign = s:sub(i, i)
      if sign == "+" or sign == "-" then i = i + 1 end
      while i <= n and s:sub(i, i):match("%d") do i = i + 1 end
    end
    local v = tonumber(s:sub(start, i - 1))
    if v == nil then return nil, "bad number" end
    return v
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
      if k == nil then return nil, "invalid object key" end
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
    return nil, "invalid value"
  end

  local value, perr = parse_value()
  if perr then
    return nil, "json fallback parser: " .. tostring(perr)
  end
  skip_ws()
  if i <= n and s:sub(i):match("%S") then
    return nil, "json fallback parser: trailing data"
  end
  return value
end

local function pseudocode_identifier(s)
  s = tostring(s or ""):gsub("[^%w_]", "_")
  if s == "" then s = "value" end
  if s:match("^%d") then s = "_" .. s end
  return s
end

local function encode_pseudocode(root)
  local lines = {
    "# MPK - Mimir Package Template",
    "# Syntaxe lisible inspirée du pseudocode Visu.",
    "",
  }
  local next_id = 0

  local function unique_name(parent, key)
    next_id = next_id + 1
    return pseudocode_identifier(parent .. "_" .. tostring(key)) .. "_" .. tostring(next_id)
  end

  local emit_table
  emit_table = function(name, value)
    local array = is_array(value)
    lines[#lines + 1] = (array and "list " or "map ") .. name .. " = []"

    if array then
      for i = 1, #value do
        local item = value[i]
        if type(item) == "table" then
          local child = unique_name(name, i)
          emit_table(child, item)
          lines[#lines + 1] = name .. ".append(" .. child .. ")"
        else
          lines[#lines + 1] = name .. ".append(" .. json_encode(item, 0) .. ")"
        end
      end
    else
      local keys = {}
      for key in pairs(value) do keys[#keys + 1] = key end
      table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
      for _, key in ipairs(keys) do
        local item = value[key]
        if type(item) == "table" then
          local child = unique_name(name, key)
          emit_table(child, item)
          lines[#lines + 1] = name .. ".set(" .. json_encode(tostring(key), 0) .. ", " .. child .. ")"
        else
          lines[#lines + 1] = name .. ".set(" .. json_encode(tostring(key), 0) .. ", " .. json_encode(item, 0) .. ")"
        end
      end
    end
  end

  emit_table("mpk", root)
  lines[#lines + 1] = ""
  return table.concat(lines, "\n")
end

local function parse_pseudocode(source)
  local values = {}
  local kinds = {}

  local function parse_expression(expr, line_no)
    expr = tostring(expr or ""):match("^%s*(.-)%s*$")
    if values[expr] ~= nil then return values[expr] end
    if expr == "null" then return nil end
    local value, err = parse_json_with_fallback(expr)
    if err ~= nil then
      return nil, "line " .. tostring(line_no) .. ": invalid value " .. expr
    end
    return value
  end

  local function split_set_arguments(args, line_no)
    local quoted, escaped = false, false
    for i = 1, #args do
      local c = args:sub(i, i)
      if quoted then
        if escaped then
          escaped = false
        elseif c == "\\" then
          escaped = true
        elseif c == '"' then
          quoted = false
        end
      elseif c == '"' then
        quoted = true
      elseif c == "," then
        return args:sub(1, i - 1), args:sub(i + 1)
      end
    end
    return nil, "line " .. tostring(line_no) .. ": .set expects two arguments"
  end

  local line_no = 0
  for raw_line in (tostring(source or "") .. "\n"):gmatch("(.-)\r?\n") do
    line_no = line_no + 1
    local line = raw_line:match("^%s*(.-)%s*$")
    if line ~= "" and line:sub(1, 1) ~= "#" then
      local kind, name = line:match("^(%a+)%s+([%a_][%w_]*)%s*=%s*%[%s*%]$")
      if kind == "map" or kind == "list" or kind == "array" then
        if values[name] ~= nil then
          return nil, "line " .. tostring(line_no) .. ": duplicate declaration " .. name
        end
        values[name] = {}
        kinds[name] = kind
      else
        local target, args = line:match("^([%a_][%w_]*)%.set%((.*)%)$")
        if target then
          if kinds[target] ~= "map" then
            return nil, "line " .. tostring(line_no) .. ": .set target is not a map"
          end
          local key_expr, value_expr = split_set_arguments(args, line_no)
          if key_expr == nil then return nil, value_expr end
          local key, key_err = parse_expression(key_expr, line_no)
          if key_err then return nil, key_err end
          if type(key) ~= "string" then
            return nil, "line " .. tostring(line_no) .. ": map key must be a string"
          end
          local value, value_err = parse_expression(value_expr, line_no)
          if value_err then return nil, value_err end
          values[target][key] = value
        else
          target, args = line:match("^([%a_][%w_]*)%.append%((.*)%)$")
          if target then
            if kinds[target] ~= "list" and kinds[target] ~= "array" then
              return nil, "line " .. tostring(line_no) .. ": .append target is not a list"
            end
            local value, value_err = parse_expression(args, line_no)
            if value_err then return nil, value_err end
            values[target][#values[target] + 1] = value
          else
            return nil, "line " .. tostring(line_no) .. ": unsupported MPK pseudocode statement"
          end
        end
      end
    end
  end

  if kinds.mpk ~= "map" then
    return nil, "missing root declaration: map mpk = []"
  end
  return values.mpk
end

local function read_file_text(path)
  local f, err = io.open(path, "rb")
  if not f then return nil, err end
  local c = f:read("*a")
  f:close()
  return c
end

local function write_file_text(path, content)
  local f, err = io.open(path, "wb")
  if not f then return nil, err end
  f:write(content)
  f:close()
  return true
end

local function base64_encode(data)
  data = tostring(data or "")
  local alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  return ((data:gsub('.', function(x)
    local r, b = '', x:byte()
    for i = 8, 1, -1 do
      r = r .. (b % 2^i - b % 2^(i - 1) > 0 and '1' or '0')
    end
    return r
  end) .. '0000'):gsub('%d%d%d?%d?%d?%d?', function(x)
    if #x < 6 then return '' end
    local c = 0
    for i = 1, 6 do
      c = c + (x:sub(i, i) == '1' and 2^(6 - i) or 0)
    end
    return alphabet:sub(c + 1, c + 1)
  end) .. ({ '', '==', '=' })[#data % 3 + 1])
end

local function base64_decode(data)
  data = tostring(data or "")
  local alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  -- Accept URL-safe base64 variants.
  data = data:gsub("-", "+"):gsub("_", "/")
  data = data:gsub("[^" .. alphabet .. "=]", "")
  local rem = #data % 4
  if rem ~= 0 then
    data = data .. string.rep("=", 4 - rem)
  end

  local bitstream = data:gsub('.', function(x)
    if x == '=' then return '' end
    local idx = alphabet:find(x, 1, true)
    if not idx then return '' end
    idx = idx - 1
    local r = ''
    for i = 6, 1, -1 do
      r = r .. (idx % 2^i - idx % 2^(i - 1) > 0 and '1' or '0')
    end
    return r
  end)

  -- Drop incomplete trailing bits (padding residue).
  local usable = #bitstream - (#bitstream % 8)
  if usable < #bitstream then
    bitstream = bitstream:sub(1, usable)
  end

  return (bitstream:gsub('%d%d%d%d%d%d%d%d', function(x)
    local c = 0
    for i = 1, 8 do
      c = c + (x:sub(i, i) == '1' and 2^(8 - i) or 0)
    end
    return string.char(c)
  end))
end

local function normalize_structure(v)
  if type(v) == "table" then return shallow_copy(v) end
  return {}
end

local function fnv1a32(data)
  local hash = 2166136261
  local prime = 16777619
  data = tostring(data or "")
  for i = 1, #data do
    hash = (hash ~ data:byte(i)) & 0xFFFFFFFF
    hash = (hash * prime) & 0xFFFFFFFF
  end
  return hash
end

local function hex_u32(n)
  n = tonumber(n) or 0
  if n < 0 then n = 0 end
  return string.format("%08x", math.floor(n) & 0xFFFFFFFF)
end

local function pack_u32le(n)
  n = (tonumber(n) or 0) & 0xFFFFFFFF
  local b1 = n & 0xFF
  local b2 = (n >> 8) & 0xFF
  local b3 = (n >> 16) & 0xFF
  local b4 = (n >> 24) & 0xFF
  return string.char(b1, b2, b3, b4)
end

local function unpack_u32le(s, offset)
  local i = offset or 1
  local b1, b2, b3, b4 = s:byte(i, i + 3)
  if not b4 then return nil end
  return ((b4 << 24) | (b3 << 16) | (b2 << 8) | b1) & 0xFFFFFFFF
end

local function read_u32le(s, offset)
  local v = unpack_u32le(s, offset)
  if v == nil then return nil, offset end
  return v, offset + 4
end

local function pack_u16le(n)
  n = (tonumber(n) or 0) & 0xFFFF
  local b1 = n & 0xFF
  local b2 = (n >> 8) & 0xFF
  return string.char(b1, b2)
end

local function read_u16le(s, offset)
  local b1, b2 = s:byte(offset, offset + 1)
  if not b2 then return nil, offset end
  return ((b2 << 8) | b1) & 0xFFFF, offset + 2
end

local function pack_u8(n)
  n = (tonumber(n) or 0) & 0xFF
  return string.char(n)
end

local function read_u8(s, offset)
  local b = s:byte(offset)
  if b == nil then return nil, offset end
  return b, offset + 1
end

local function pack_varuint(n)
  n = math.floor(tonumber(n) or 0)
  if n < 0 then error("varuint cannot encode a negative value") end
  local out = {}
  repeat
    local byte = n % 128
    n = math.floor(n / 128)
    if n > 0 then byte = byte + 128 end
    out[#out + 1] = string.char(byte)
  until n == 0
  return table.concat(out)
end

local function read_varuint(s, offset)
  local value, factor = 0, 1
  for _ = 1, 10 do
    local byte = s:byte(offset)
    if byte == nil then return nil, offset, "truncated varuint" end
    offset = offset + 1
    value = value + (byte % 128) * factor
    if byte < 128 then return value, offset end
    factor = factor * 128
    if factor > 2^53 then return nil, offset, "varuint overflow" end
  end
  return nil, offset, "varuint too long"
end

local function encode_typed_value(value, depth)
  depth = depth or 0
  if depth > 128 then error("binary MPK nesting is too deep") end
  local kind = type(value)
  if kind == "nil" then return string.char(0) end
  if kind == "boolean" then return string.char(value and 2 or 1) end
  if kind == "number" then
    if math.type and math.type(value) == "integer" then
      local zigzag = value >= 0 and value * 2 or (-value * 2 - 1)
      return string.char(3) .. pack_varuint(zigzag)
    end
    return string.char(4) .. string.pack("<d", value)
  end
  if kind == "string" then
    return string.char(5) .. pack_varuint(#value) .. value
  end
  if kind ~= "table" then
    error("unsupported binary MPK value: " .. kind)
  end

  local out = {}
  if is_array(value) then
    out[#out + 1] = string.char(6)
    out[#out + 1] = pack_varuint(#value)
    for i = 1, #value do
      out[#out + 1] = encode_typed_value(value[i], depth + 1)
    end
    return table.concat(out)
  end

  local keys = {}
  for key in pairs(value) do
    if type(key) ~= "string" then
      error("binary MPK maps only support string keys")
    end
    keys[#keys + 1] = key
  end
  table.sort(keys)
  out[#out + 1] = string.char(7)
  out[#out + 1] = pack_varuint(#keys)
  for _, key in ipairs(keys) do
    out[#out + 1] = pack_varuint(#key)
    out[#out + 1] = key
    out[#out + 1] = encode_typed_value(value[key], depth + 1)
  end
  return table.concat(out)
end

local function decode_typed_value(blob, offset, state, depth)
  depth = depth or 0
  if depth > 128 then return nil, offset, "binary MPK nesting is too deep" end
  state.nodes = state.nodes + 1
  if state.nodes > 1000000 then return nil, offset, "binary MPK has too many values" end

  local tag = blob:byte(offset)
  if tag == nil then return nil, offset, "truncated typed value" end
  offset = offset + 1
  if tag == 0 then return nil, offset end
  if tag == 1 then return false, offset end
  if tag == 2 then return true, offset end
  if tag == 3 then
    local zigzag, next_offset, err = read_varuint(blob, offset)
    if zigzag == nil then return nil, next_offset, err end
    local value
    if zigzag % 2 == 0 then value = zigzag / 2
    else value = -(zigzag + 1) / 2 end
    return math.tointeger(value) or value, next_offset
  end
  if tag == 4 then
    if offset + 7 > #blob then return nil, offset, "truncated float64" end
    local value, next_offset = string.unpack("<d", blob, offset)
    return value, next_offset
  end
  if tag == 5 then
    local len, next_offset, err = read_varuint(blob, offset)
    if len == nil then return nil, next_offset, err end
    if len > #blob - next_offset + 1 then return nil, next_offset, "truncated string" end
    return blob:sub(next_offset, next_offset + len - 1), next_offset + len
  end
  if tag == 6 then
    local count, next_offset, err = read_varuint(blob, offset)
    if count == nil then return nil, next_offset, err end
    local result = {}
    offset = next_offset
    for i = 1, count do
      local value
      value, offset, err = decode_typed_value(blob, offset, state, depth + 1)
      if err then return nil, offset, err end
      result[i] = value
    end
    return result, offset
  end
  if tag == 7 then
    local count, next_offset, err = read_varuint(blob, offset)
    if count == nil then return nil, next_offset, err end
    local result = {}
    offset = next_offset
    for _ = 1, count do
      local len
      len, offset, err = read_varuint(blob, offset)
      if len == nil then return nil, offset, err end
      if len > #blob - offset + 1 then return nil, offset, "truncated map key" end
      local key = blob:sub(offset, offset + len - 1)
      offset = offset + len
      local value
      value, offset, err = decode_typed_value(blob, offset, state, depth + 1)
      if err then return nil, offset, err end
      result[key] = value
    end
    return result, offset
  end
  return nil, offset, "unknown typed value tag: " .. tostring(tag)
end

local function make_model_structure_template(kind)
  kind = tostring(kind or ""):lower()
  if kind == "vae_conv" then
    return {
      template = "vae_conv",
      blocks = {
        { name = "encoder", notes = "conv downsampling stack" },
        { name = "latent", notes = "mu/logvar projection" },
        { name = "decoder", notes = "upsample reconstruction stack" },
      },
      io = {
        input = "image_rgb",
        latent = "z",
        output = "reconstruction_rgb",
      },
    }
  end
  if kind == "unet" then
    return {
      template = "unet",
      blocks = {
        { name = "down_path", notes = "encoder pyramid" },
        { name = "bottleneck", notes = "middle residual/attention" },
        { name = "up_path", notes = "decoder pyramid with skip connections" },
      },
      io = {
        input = "feature_map_or_latent",
        conditioning = "optional",
        output = "denoised_feature_map",
      },
    }
  end
  return {
    template = kind ~= "" and kind or "generic",
    blocks = {},
    io = {},
  }
end

local function compute_payload_checksum(pkg)
  if type(pkg) ~= "table" or type(pkg.payload) ~= "table" then
    return nil
  end
  local payload_json = json_encode(pkg.payload, 0)
  return fnv1a32(payload_json), payload_json
end

local function update_checksum(pkg)
  local checksum, _ = compute_payload_checksum(pkg)
  pkg.header = pkg.header or {}
  pkg.header.checksum = {
    algorithm = "fnv1a32",
    scope = "payload_json",
    value = hex_u32(checksum or 0),
  }
  return checksum or 0
end

function M.base64_encode(s)
  return base64_encode(s)
end

function M.base64_decode(s)
  return base64_decode(s)
end

function M.encode_json(v)
  return json_encode(v, 0)
end

function M.encode_pseudocode(v)
  if type(v) ~= "table" then return nil, "pseudocode root must be a table" end
  return encode_pseudocode(v)
end

function M.decode_pseudocode(s)
  return parse_pseudocode(s)
end

function M.decode_json(text)
  return parse_json_with_fallback(text)
end

function M.read_json_file(path)
  if type(read_json) == "function" then
    local v = read_json(path)
    if type(v) == "table" then return v end
  end

  local txt, err = read_file_text(path)
  if not txt then return nil, err or "cannot read file" end
  return parse_json_with_fallback(txt)
end

function M.read_text_file(path)
  return read_file_text(path)
end

function M.write_text_file(path, content)
  if type(write_file) == "function" then
    local ok, err = write_file(path, content)
    if ok == false then return nil, err end
    return true
  end
  return write_file_text(path, content)
end

function M.model_structure_template(kind)
  return make_model_structure_template(kind)
end

function M.build(spec)
  spec = spec or {}

  local name = tostring(spec.name or "")
  if name == "" then
    return nil, "missing package name"
  end

  local model_type = tostring(spec.type or spec.architecture or "")
  if model_type == "" then
    return nil, "missing model type"
  end

  local author = tostring(spec.author or "unknown")
  local created_at = tostring(spec.created_at or utc_iso8601())
  local modifiable = not not spec.modifiable
  local viz_specified = not not spec.viz_specified
  local description = tostring(spec.description or "")
  local base_config = is_table(spec.base_config) and spec.base_config or {}
  local model_structure = normalize_structure(spec.model_structure)

  local base_config_json = json_encode(base_config, 0)
  local description_b64 = base64_encode(description)
  local base_config_b64 = base64_encode(base_config_json)

  local pkg = {
    format = "Mimir Package Template",
    format_short = "MPK",
    extension = ".mpk",
    version = 2,
    container = tostring(spec.container or "pseudocode"),
    header = {
      author = author,
      created_at = created_at,
      modifiable = modifiable,
      size = 0,
      name = name,
      type = model_type,
      viz_specified = viz_specified,
      signature = "checksum",
    },
    payload = {
      base_config_b64 = base_config_b64,
      model_structure = model_structure,
      description_b64 = description_b64,
    },
  }

  update_checksum(pkg)

  -- Stabilize the size field until fixed-point.
  for _ = 1, 8 do
    local encoded = json_encode(pkg, 0)
    local n = #encoded
    if pkg.header.size == n then break end
    pkg.header.size = n
  end

  return pkg
end

function M.verify_checksum(pkg)
  if type(pkg) ~= "table" or type(pkg.header) ~= "table" or type(pkg.payload) ~= "table" then
    return false, "invalid MPK object"
  end
  local checksum_meta = pkg.header.checksum
  if type(checksum_meta) ~= "table" then
    return false, "missing checksum metadata"
  end
  if tostring(checksum_meta.algorithm or "") ~= "fnv1a32" then
    return false, "unsupported checksum algorithm"
  end
  local expected = tostring(checksum_meta.value or ""):lower()
  local got_num, _ = compute_payload_checksum(pkg)
  local got = hex_u32(got_num):lower()
  if expected ~= got then
    return false, "checksum mismatch (expected=" .. expected .. ", got=" .. got .. ")"
  end
  return true
end

function M.decode_payload(pkg)
  if type(pkg) ~= "table" or type(pkg.payload) ~= "table" then
    return nil, "invalid MPK object"
  end

  local raw_b64 = tostring(pkg.payload.base_config_b64 or "")
  local raw_cfg_json = base64_decode(raw_b64)
  local cfg, err_cfg = parse_json_with_fallback(raw_cfg_json)
  if type(cfg) ~= "table" then
    -- Backward compatibility: some early files may store raw JSON directly.
    local cfg_plain, err_plain = parse_json_with_fallback(raw_b64)
    if type(cfg_plain) == "table" then
      cfg = cfg_plain
      err_cfg = nil
    else
      err_cfg = tostring(err_cfg or "invalid JSON") .. " | plain-json fallback: " .. tostring(err_plain or "invalid JSON")
    end
  end
  if type(cfg) ~= "table" then
    return nil, "cannot decode base_config_b64: " .. tostring(err_cfg or "invalid JSON")
  end

  local desc = base64_decode(pkg.payload.description_b64 or "")
  return {
    base_config = cfg,
    description = desc,
    model_structure = pkg.payload.model_structure,
  }
end

local function write_binary(path, pkg)
  -- Binary-v4 stores semantic typed values rather than the source text.
  -- Base config and description are decoded before packing to avoid Base64
  -- expansion. The source remains the editable artifact; compiled output is
  -- deliberately not reversible to the original formatting/comments.
  pkg.container = "pseudocode"
  update_checksum(pkg)

  local decoded, decode_err = M.decode_payload(pkg)
  if type(decoded) ~= "table" then
    return nil, "cannot compile MPK payload: " .. tostring(decode_err)
  end

  local top = {}
  for key, value in pairs(pkg) do
    if key ~= "header" and key ~= "payload" and key ~= "container" then
      top[key] = value
    end
  end
  local packed_header = shallow_copy(pkg.header)
  packed_header.size = nil
  packed_header.checksum = nil
  local packed_payload = shallow_copy(pkg.payload)
  packed_payload.base_config_b64 = nil
  packed_payload.description_b64 = nil

  local semantic = {
    top = top,
    header = packed_header,
    payload = packed_payload,
    base_config = decoded.base_config,
    description = decoded.description,
  }
  local ok_encode, typed_or_err = pcall(encode_typed_value, semantic)
  if not ok_encode then
    return nil, "binary-v4 encoding failed: " .. tostring(typed_or_err)
  end
  local binary_payload = "TYP4" .. typed_or_err
  local payload_size = #binary_payload
  if payload_size > 0xFFFFFFFF then
    return nil, "binary MPK payload too large"
  end

  local checksum = fnv1a32(binary_payload)
  local flags = 0
  if pkg.header and pkg.header.modifiable then flags = flags | 0x01 end
  if pkg.header and pkg.header.viz_specified then flags = flags | 0x02 end

  local header = {}
  header[#header + 1] = MPK_BINARY_MAGIC
  header[#header + 1] = string.char(MPK_BINARY_VERSION)
  header[#header + 1] = string.char(1) -- mode: binary
  header[#header + 1] = string.char(flags)
  header[#header + 1] = string.char(0) -- reserved
  header[#header + 1] = pack_u32le(payload_size)
  header[#header + 1] = pack_u32le(checksum)
  header[#header + 1] = pack_u32le(utc_epoch_u32())

  local written = 4 + 1 + 1 + 1 + 1 + 4 + 4 + 4
  if written > MPK_BINARY_HEADER_SIZE then
    return nil, "invalid binary header size"
  end
  header[#header + 1] = string.rep("\0", MPK_BINARY_HEADER_SIZE - written)

  local blob = table.concat(header) .. binary_payload
  pkg.container = "binary"
  return M.write_text_file(path, blob)
end

local function read_binary(blob)
  if type(blob) ~= "string" or #blob < MPK_BINARY_HEADER_SIZE then
    return nil, "invalid binary MPK data"
  end
  if blob:sub(1, 4) ~= MPK_BINARY_MAGIC then
    return nil, "invalid binary magic"
  end

  local version = blob:byte(5)

  local payload_size = unpack_u32le(blob, 9)
  local expected_checksum = unpack_u32le(blob, 13)
  if payload_size == nil or expected_checksum == nil then
    return nil, "corrupted binary header"
  end

  local payload_start = MPK_BINARY_HEADER_SIZE + 1
  local payload_end = payload_start + payload_size - 1
  if #blob < payload_end then
    return nil, "truncated binary MPK payload"
  end

  local payload_data = blob:sub(payload_start, payload_end)
  local got_checksum = fnv1a32(payload_data)
  if got_checksum ~= expected_checksum then
    return nil, "binary payload checksum mismatch"
  end

  -- Legacy binary-v1: payload is full package JSON.
  if version == 1 then
    local obj, err = parse_json_with_fallback(payload_data)
    if type(obj) ~= "table" then
      return nil, "binary payload JSON decode failed: " .. tostring(err)
    end
    obj.container = "binary"
    return obj
  end

  -- Binary-v3: compiled, validated pseudocode source.
  if version == 3 then
    if payload_data:sub(1, 4) ~= "PSC3" then
      return nil, "invalid binary-v3 payload signature"
    end
    local source_size = unpack_u32le(payload_data, 5)
    if source_size == nil then
      return nil, "missing binary-v3 source size"
    end
    local source = payload_data:sub(9, 8 + source_size)
    if #source ~= source_size then
      return nil, "truncated binary-v3 pseudocode"
    end
    local obj, err = parse_pseudocode(source)
    if type(obj) ~= "table" then
      return nil, "binary-v3 pseudocode decode failed: " .. tostring(err)
    end
    obj.container = "binary"
    if type(obj.header) == "table" then obj.header.size = #blob end
    return obj
  end

  -- Binary-v4: opaque deterministic typed representation.
  if version == 4 then
    if payload_data:sub(1, 4) ~= "TYP4" then
      return nil, "invalid binary-v4 payload signature"
    end
    local semantic, next_offset, decode_err =
      decode_typed_value(payload_data, 5, { nodes = 0 }, 0)
    if type(semantic) ~= "table" then
      return nil, "binary-v4 decode failed: " .. tostring(decode_err)
    end
    if decode_err then
      return nil, "binary-v4 decode failed: " .. tostring(decode_err)
    end
    if next_offset ~= #payload_data + 1 then
      return nil, "binary-v4 payload has trailing data"
    end
    if type(semantic.header) ~= "table" or
        type(semantic.payload) ~= "table" or
        type(semantic.base_config) ~= "table" then
      return nil, "binary-v4 semantic structure is incomplete"
    end

    local obj = type(semantic.top) == "table" and semantic.top or {}
    obj.container = "binary"
    obj.header = semantic.header
    obj.header.size = #blob
    obj.payload = semantic.payload
    obj.payload.base_config_b64 =
      base64_encode(json_encode(semantic.base_config, 0))
    obj.payload.description_b64 =
      base64_encode(tostring(semantic.description or ""))
    update_checksum(obj)
    return obj
  end

  -- Binary-v2: payload is a packed binary structure.
  if version ~= 2 then
    return nil, "unsupported binary MPK version: " .. tostring(version)
  end

  local p = payload_data
  local off = 1

  local sig = p:sub(off, off + 3)
  off = off + 4
  if sig ~= "BIN2" then
    return nil, "invalid binary-v2 payload signature"
  end

  local format_v
  format_v, off = read_u32le(p, off)
  if format_v ~= 2 then
    return nil, "unsupported binary-v2 payload format"
  end

  local function read_len_string(label)
    local len
    len, off = read_u32le(p, off)
    if len == nil then return nil, "invalid length for " .. label end
    local s = p:sub(off, off + len - 1)
    if #s ~= len then return nil, "truncated string for " .. label end
    off = off + len
    return s
  end

  local name, err_name = read_len_string("name")
  if not name then return nil, err_name end
  local model_type, err_type = read_len_string("type")
  if not model_type then return nil, err_type end
  local author, err_author = read_len_string("author")
  if not author then return nil, err_author end
  local created_at, err_created = read_len_string("created_at")
  if not created_at then return nil, err_created end

  local modifiable
  modifiable, off = read_u8(p, off)
  if modifiable == nil then return nil, "missing modifiable flag" end
  local viz_specified
  viz_specified, off = read_u8(p, off)
  if viz_specified == nil then return nil, "missing viz_specified flag" end
  local _reserved
  _reserved, off = read_u16le(p, off)
  if _reserved == nil then return nil, "missing reserved field" end

  local base_cfg_raw, err_cfg = read_len_string("base_config")
  if not base_cfg_raw then return nil, err_cfg end
  local model_structure_raw, err_ms = read_len_string("model_structure")
  if not model_structure_raw then return nil, err_ms end
  local desc_raw, err_desc = read_len_string("description")
  if not desc_raw then return nil, err_desc end

  local model_structure, err_ms_json = parse_json_with_fallback(model_structure_raw)
  if type(model_structure) ~= "table" then
    return nil, "binary-v2 model_structure decode failed: " .. tostring(err_ms_json)
  end

  local pkg = {
    container = "binary",
    extension = ".mpk",
    format = "Mimir Package Template",
    format_short = "MPK",
    version = 2,
    header = {
      author = author,
      created_at = created_at,
      modifiable = (modifiable ~= 0),
      size = #blob,
      name = name,
      type = model_type,
      viz_specified = (viz_specified ~= 0),
      signature = "checksum",
    },
    payload = {
      base_config_b64 = base64_encode(base_cfg_raw),
      model_structure = model_structure,
      description_b64 = base64_encode(desc_raw),
    },
  }

  update_checksum(pkg)
  return pkg
end

function M.write(path, pkg, opts)
  path = tostring(path or "")
  if path == "" then return nil, "empty path" end
  opts = opts or {}

  if opts.json then
    return nil, "JSON MPK output was removed; MPK.write only writes pseudocode"
  end
  if opts.binary then
    return nil, "binary output moved to MPK.compile(source.mpk, output.mpk.bin)"
  end
  if not path:lower():match("%.mpk$") then
    return nil, "pseudocode source path must end with .mpk"
  end

  pkg.container = "pseudocode"
  update_checksum(pkg)

  local encoded = encode_pseudocode(pkg)
  for _ = 1, 8 do
    local n = #encoded
    if pkg.header.size == n then break end
    pkg.header.size = n
    encoded = encode_pseudocode(pkg)
  end
  if encoded:sub(-1) ~= "\n" then encoded = encoded .. "\n" end
  return M.write_text_file(path, encoded)
end

function M.compile(source_path, output_path)
  source_path = tostring(source_path or "")
  output_path = tostring(output_path or "")
  if not source_path:lower():match("%.mpk$") then
    return nil, "source path must end with .mpk"
  end
  if not output_path:lower():match("%.mpk%.bin$") then
    return nil, "compiled output path must end with .mpk.bin"
  end
  if source_path == output_path then
    return nil, "compiled output must differ from pseudocode source"
  end

  local source, read_err = M.read_text_file(source_path)
  if source == nil then return nil, tostring(read_err or "cannot read pseudocode source") end
  if source:sub(1, 4) == MPK_BINARY_MAGIC then
    return nil, "source is already a compiled binary MPK"
  end

  local pkg, parse_err = parse_pseudocode(source)
  if type(pkg) ~= "table" then
    return nil, "source is not modern MPK pseudocode: " .. tostring(parse_err)
  end
  local ok_checksum, checksum_err = M.verify_checksum(pkg)
  if not ok_checksum then
    return nil, "source integrity check failed: " .. tostring(checksum_err)
  end
  return write_binary(output_path, pkg)
end

function M.read(path)
  local raw, err = M.read_text_file(path)
  if raw == nil then
    return nil, tostring(err or "cannot read file")
  end

  local obj
  if #raw >= 4 and raw:sub(1, 4) == MPK_BINARY_MAGIC then
    obj, err = read_binary(raw)
  elseif raw:match("^%s*#%s*MPK") or raw:match("^%s*map%s+mpk%s*=") then
    obj, err = parse_pseudocode(raw)
    if type(obj) == "table" then
      obj.container = tostring(obj.container or "pseudocode")
    end
  else
    obj, err = M.read_json_file(path)
    if type(obj) == "table" then
      obj.container = tostring(obj.container or "json")
    end
  end

  if type(obj) ~= "table" then
    return nil, "cannot read or decode MPK: " .. tostring(err or "unknown")
  end
  if tostring(obj.format_short or "") ~= "MPK" then
    return nil, "invalid format_short (expected MPK)"
  end
  if type(obj.header) ~= "table" or type(obj.payload) ~= "table" then
    return nil, "invalid MPK structure"
  end

  local ok_sig, err_sig = M.verify_checksum(obj)
  if not ok_sig then
    return nil, "integrity check failed: " .. tostring(err_sig)
  end

  return obj
end

function M.to_registry_full_config(pkg)
  local decoded, err = M.decode_payload(pkg)
  if not decoded then return nil, err end
  local arch = tostring((pkg.header and pkg.header.type) or "")
  if arch == "" then
    return nil, "missing header.type"
  end

  return {
    architecture = arch,
    model = decoded.base_config,
    mpk = {
      name = pkg.header.name,
      author = pkg.header.author,
      created_at = pkg.header.created_at,
      modifiable = pkg.header.modifiable,
      viz_specified = pkg.header.viz_specified,
      description = decoded.description,
      model_structure = decoded.model_structure,
      checksum = pkg.header.checksum,
      container = pkg.container,
    },
  }
end

return M
