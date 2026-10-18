-- HTTP/WS pilotage: ce module est l'entrée réseau, sans modification des bindings.
ROOTWORK = rawget(_G, "ROOTWORK") or os.getenv("ROOTWORK")
  or ((arg and arg[0] or ""):match("^(.*)/scripts/modules/[^/]+$")) or "."
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- scripts/modules/api_ws_server.lua
-- Serveur HTTP + WebSocket minimal pour piloter Mímir via REST (GET/POST/PUT/PATCH/DELETE)
-- Dépendances: LuaSocket (luasocket)
-- Usage (Lua système):   lua scripts/modules/api_ws_server.lua
-- Usage (avec Mímir):    ./bin/mimir --lua scripts/modules/api_ws_server.lua -- --host 127.0.0.1 --port 8088
-- Alias flags: --api-host/--api-port
-- Auth: --auth off|token|api_key|either, --auth-token / --auth-token-file, --api-key / --api-key-file
-- TLS: --tls --tls-cert chemin --tls-key chemin (LuaSec requis).
-- --max-body-bytes 16777216 limite les bodies HTTP et les messages WS assemblés.
-- Découverte: GET /help et /capabilities. RPC: POST /rpc {"api":"Model.forward","args":[[0.1,0.2],false]}.
-- Les retours multiples RPC sont préservés dans results (nil -> null).
-- Serveur mono-client: une session WS monopolise la boucle; la fermer avant les requêtes HTTP.

local opts = {}
do
  local ok_args, mod_or_err = pcall(dofile, ROOTWORK.."/scripts/modules/args.lua")
  if ok_args and type(mod_or_err) == "table" and type(mod_or_err.parse) == "function" then
    opts = mod_or_err.parse(arg) or {}
  end
end

local function opt_str(key, default)
  if type(opts) ~= "table" then return default end
  local v = opts[key]
  if v == nil and key:find("-", 1, true) then
    v = opts[key:gsub("-", "_")]
  elseif v == nil then
    v = opts[key:gsub("_", "-")]
  end
  if v == nil or v == true then return default end
  return tostring(v)
end

local function opt_int(key, default)
  local v = opt_str(key, nil)
  if v == nil then return default end
  local n = tonumber(v)
  if n == nil then return default end
  return math.floor(n)
end

local function opt_bool(key, default)
  if type(opts) ~= "table" then return default end
  local v = opts[key]
  if v == nil and key:find("-", 1, true) then
    v = opts[key:gsub("-", "_")]
  elseif v == nil then
    v = opts[key:gsub("_", "-")]
  end
  if v == nil then return default end
  if type(v) == "boolean" then return v end
  local s = tostring(v):lower()
  if s == "1" or s == "true" or s == "yes" or s == "on" then return true end
  if s == "0" or s == "false" or s == "no" or s == "off" then return false end
  return default
end

local HOST = opt_str("host", opt_str("api-host", "127.0.0.1"))
local PORT = opt_int("port", opt_int("api-port", 8088))
local MAX_BODY_BYTES = opt_int("max-body-bytes", 16 * 1024 * 1024)
assert(PORT >= 1 and PORT <= 65535, "port doit être entre 1 et 65535")
assert(MAX_BODY_BYTES > 0, "max-body-bytes doit être positif")

local SERVER_NAME = opt_str("server-name", "mimir-lua-api")

local function safe_read_file(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local s = f:read("*a")
  f:close()
  if not s then return nil end
  return (s:gsub("%s+$", ""))
end

local SERVER_VERSION = safe_read_file(ROOTWORK.."/VERSION") or "unknown"

-- =========================
-- Auth / TLS options
-- =========================
local function opt_file_str(key)
  local path = opt_str(key, nil)
  if not path or path == "" then return nil end
  return safe_read_file(path)
end

-- Auth modes:
-- - off: no auth
-- - token: Authorization: Bearer <token> (or query ?token=... for WS/browser)
-- - api_key: X-API-Key: <key> (or query ?api_key=...)
-- - either: accept either token or api key
local AUTH_MODE = (opt_str("auth", "off") or "off"):lower()
local AUTH_TOKEN = opt_str("auth-token", nil) or opt_file_str("auth-token-file")
local API_KEY = opt_str("api-key", nil) or opt_file_str("api-key-file")
local AUTH_SKIP_PATHS = opt_str("auth-skip-paths", "/health,/help")

local TLS_ENABLED = opt_bool("tls", false) or opt_bool("https", false)
local TLS_CERT_FILE = opt_str("tls-cert", "")
local TLS_KEY_FILE = opt_str("tls-key", "")

-- =========================
-- CORS (pour UI web)
-- =========================
-- Par défaut, on active CORS pour simplifier l'usage depuis une page web.
-- Flags:
--   --cors true|false
--   --cors-origin "*" | "reflect" | "http://localhost:5173" | "null"
local CORS_ENABLED = opt_bool("cors", true)
local CORS_ORIGIN = opt_str("cors-origin", "*")
local CORS_ALLOW_METHODS = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
local CORS_ALLOW_HEADERS = "Content-Type, Authorization, X-API-Key, Accept"
local CORS_EXPOSE_HEADERS = "X-Response-Time-Ms"
local CORS_MAX_AGE = "600"

-- =========================
-- Dépendances (LuaSocket)
-- =========================
local ok_socket, socket = pcall(require, "socket")
if not ok_socket then
  error(
    "LuaSocket manquant: installez-le (ex: `luarocks install luasocket`) " ..
    "ou lancez ce script avec un Lua qui l'a. Détail: " .. tostring(socket)
  )
end

local ssl
do
  local ok_ssl, mod_or_err = pcall(require, "ssl")
  if ok_ssl then
    ssl = mod_or_err
  end
end

local function now_seconds()
  if type(socket) == "table" and type(socket.gettime) == "function" then
    return socket.gettime()
  end
  return os.clock()
end

local function iso_utc(ts)
  -- os.date("!...") ne gère que la seconde, donc on garde aussi la latence ms séparée.
  return os.date("!%Y-%m-%dT%H:%M:%SZ", ts or os.time())
end

-- =========================
-- JSON minimal (encode/decode)
-- =========================
local JSON_NULL = setmetatable({}, { __json = "null" })
local JSON_ARRAY = { __json = "array" }
local JSON_OBJECT = { __json = "object" }
local function json_array(t) return setmetatable(t or {}, JSON_ARRAY) end

local function is_array(t)
  if type(t) ~= "table" then return false end
  local mt = getmetatable(t)
  if mt == JSON_ARRAY then return true end
  if mt == JSON_OBJECT or t == JSON_NULL then return false end
  if next(t) == nil then return false end
  local n = 0
  for k, _ in pairs(t) do
    if type(k) ~= "number" then return false end
    n = math.max(n, k)
  end
  for i = 1, n do
    if rawget(t, i) == nil then return false end
  end
  return true
end

local function json_escape(s)
  return (s:gsub("[\\\"%z\1-\31]", function(c)
    local byte = string.byte(c)
    if c == "\\" then return "\\\\" end
    if c == "\"" then return "\\\"" end
    if c == "\b" then return "\\b" end
    if c == "\f" then return "\\f" end
    if c == "\n" then return "\\n" end
    if c == "\r" then return "\\r" end
    if c == "\t" then return "\\t" end
    return string.format("\\u%04x", byte)
  end))
end

local function json_encode(v)
  local tv = type(v)
  if v == nil or v == JSON_NULL then return "null" end
  if tv == "boolean" then return v and "true" or "false" end
  if tv == "number" then
    if v ~= v or v == math.huge or v == -math.huge then return "null" end
    return tostring(v)
  end
  if tv == "string" then return '"' .. json_escape(v) .. '"' end
  if tv == "table" then
    if is_array(v) then
      local out = {}
      for i = 1, #v do out[#out+1] = json_encode(v[i]) end
      return "[" .. table.concat(out, ",") .. "]"
    else
      local out = {}
      for k, val in pairs(v) do
        if type(k) == "string" then
          out[#out+1] = '"' .. json_escape(k) .. '":' .. json_encode(val)
        end
      end
      return "{" .. table.concat(out, ",") .. "}"
    end
  end
  error("Valeur non sérialisable en JSON: " .. tv)
end

local function json_decode(str)
  local i = 1
  local function skip_ws()
    while true do
      local c = str:sub(i, i)
      if c == "" then return end
      if c == " " or c == "\n" or c == "\r" or c == "\t" then
        i = i + 1
      else
        return
      end
    end
  end

  local function parse_string()
    local out = {}
    i = i + 1 -- skip opening quote
    while true do
      local c = str:sub(i, i)
      if c == "" then error("JSON: string non terminée") end
      if c == '"' then
        i = i + 1
        return table.concat(out)
      end
      if c == "\\" then
        local n = str:sub(i+1, i+1)
        if n == "\"" or n == "\\" or n == "/" then out[#out+1] = n; i = i + 2
        elseif n == "b" then out[#out+1] = "\b"; i = i + 2
        elseif n == "f" then out[#out+1] = "\f"; i = i + 2
        elseif n == "n" then out[#out+1] = "\n"; i = i + 2
        elseif n == "r" then out[#out+1] = "\r"; i = i + 2
        elseif n == "t" then out[#out+1] = "\t"; i = i + 2
        elseif n == "u" then
          local hex = str:sub(i+2, i+5)
          if not hex:match("^[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]$") then
            error("JSON: escape \\u invalide")
          end
          local code = tonumber(hex, 16)
          i = i + 6
          if code >= 0xD800 and code <= 0xDBFF then
            local low_hex = str:sub(i + 2, i + 5)
            if str:sub(i, i + 1) ~= "\\u" or not low_hex:match("^%x%x%x%x$") then
              error("JSON: paire de surrogates incomplète")
            end
            local low = tonumber(low_hex, 16)
            if low < 0xDC00 or low > 0xDFFF then error("JSON: surrogate bas invalide") end
            code = 0x10000 + (code - 0xD800) * 0x400 + low - 0xDC00
            i = i + 6
          elseif code >= 0xDC00 and code <= 0xDFFF then
            error("JSON: surrogate isolé")
          end
          out[#out+1] = utf8.char(code)
        else
          error("JSON: escape invalide")
        end
      else
        if c:byte() < 32 then error("JSON: caractère de contrôle non échappé") end
        out[#out+1] = c
        i = i + 1
      end
    end
  end

  local function parse_number()
    local start = i
    local c = str:sub(i, i)
    if c == "-" then i = i + 1 end
    if str:sub(i, i) == "0" then
      i = i + 1
      if str:sub(i, i):match("%d") then error("JSON: zéro initial interdit") end
    else
      if not str:sub(i, i):match("[1-9]") then error("JSON: nombre invalide") end
      while str:sub(i, i):match("%d") do i = i + 1 end
    end
    if str:sub(i, i) == "." then
      i = i + 1
      if not str:sub(i, i):match("%d") then error("JSON: fraction invalide") end
      while str:sub(i, i):match("%d") do i = i + 1 end
    end
    local e = str:sub(i, i)
    if e == "e" or e == "E" then
      i = i + 1
      local sign = str:sub(i, i)
      if sign == "+" or sign == "-" then i = i + 1 end
      if not str:sub(i, i):match("%d") then error("JSON: exposant invalide") end
      while str:sub(i, i):match("%d") do i = i + 1 end
    end
    local num = tonumber(str:sub(start, i-1))
    if num == nil or num == math.huge or num == -math.huge then error("JSON: nombre invalide") end
    return num
  end

  local parse_value
  local function parse_array()
    i = i + 1
    skip_ws()
    local arr = json_array()
    if str:sub(i, i) == "]" then i = i + 1; return arr end
    while true do
      arr[#arr+1] = parse_value()
      skip_ws()
      local c = str:sub(i, i)
      if c == "," then i = i + 1; skip_ws()
      elseif c == "]" then i = i + 1; return arr
      else error("JSON: tableau invalide") end
    end
  end

  local function parse_object()
    i = i + 1
    skip_ws()
    local obj = setmetatable({}, JSON_OBJECT)
    if str:sub(i, i) == "}" then i = i + 1; return obj end
    while true do
      if str:sub(i, i) ~= '"' then error("JSON: clé attendue") end
      local k = parse_string()
      skip_ws()
      if str:sub(i, i) ~= ":" then error("JSON: ':' attendu") end
      i = i + 1
      skip_ws()
      obj[k] = parse_value()
      skip_ws()
      local c = str:sub(i, i)
      if c == "," then i = i + 1; skip_ws()
      elseif c == "}" then i = i + 1; return obj
      else error("JSON: objet invalide") end
    end
  end

  function parse_value()
    skip_ws()
    local c = str:sub(i, i)
    if c == '"' then return parse_string() end
    if c == "{" then return parse_object() end
    if c == "[" then return parse_array() end
    if c == "-" or c:match("%d") then return parse_number() end
    if str:sub(i, i+3) == "true" then i = i + 4; return true end
    if str:sub(i, i+4) == "false" then i = i + 5; return false end
    if str:sub(i, i+3) == "null" then i = i + 4; return JSON_NULL end
    error("JSON: valeur invalide")
  end

  local val = parse_value()
  skip_ws()
  if i <= #str then
    local rest = str:sub(i)
    if rest:match("%S") then error("JSON: trailing data") end
  end
  return val
end

-- =========================
-- Base64 + SHA1 (WebSocket)
-- =========================
local function base64_encode(data)
  local b = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  return ((data:gsub('.', function(x)
    local r, byte = '', x:byte()
    for i2 = 8, 1, -1 do r = r .. (byte % 2^i2 - byte % 2^(i2-1) > 0 and '1' or '0') end
    return r
  end) .. '0000'):gsub('%d%d%d?%d?%d?%d?', function(x)
    if #x < 6 then return '' end
    local c = 0
    for i2 = 1, 6 do c = c + (x:sub(i2,i2) == '1' and 2^(6-i2) or 0) end
    return b:sub(c+1, c+1)
  end) .. ({ '', '==', '=' })[#data % 3 + 1])
end

local function sha1(msg)
  -- SHA1 pur Lua (suffisant pour handshake WebSocket)
  local function lrot(x, n)
    return ((x << n) | (x >> (32 - n))) & 0xffffffff
  end

  local bytes = { msg:byte(1, #msg) }
  local bit_len = #bytes * 8

  bytes[#bytes+1] = 0x80
  while (#bytes % 64) ~= 56 do
    bytes[#bytes+1] = 0
  end

  for shift = 56, 0, -8 do
    bytes[#bytes+1] = (bit_len >> shift) & 0xff
  end

  local h0, h1, h2, h3, h4 = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0

  local w = {}
  for chunk = 1, #bytes, 64 do
    for j = 0, 15 do
      local i0 = chunk + j * 4
      w[j] = ((bytes[i0] << 24) | (bytes[i0+1] << 16) | (bytes[i0+2] << 8) | bytes[i0+3]) & 0xffffffff
    end
    for j = 16, 79 do
      w[j] = lrot((w[j-3] ~ w[j-8] ~ w[j-14] ~ w[j-16]) & 0xffffffff, 1)
    end

    local a, b, c, d, e = h0, h1, h2, h3, h4
    for j = 0, 79 do
      local f, k
      if j <= 19 then
        f = (b & c) | ((~b) & d)
        k = 0x5A827999
      elseif j <= 39 then
        f = b ~ c ~ d
        k = 0x6ED9EBA1
      elseif j <= 59 then
        f = (b & c) | (b & d) | (c & d)
        k = 0x8F1BBCDC
      else
        f = b ~ c ~ d
        k = 0xCA62C1D6
      end
      local temp = (lrot(a, 5) + f + e + k + w[j]) & 0xffffffff
      e, d, c, b, a = d, c, lrot(b, 30), a, temp
    end

    h0 = (h0 + a) & 0xffffffff
    h1 = (h1 + b) & 0xffffffff
    h2 = (h2 + c) & 0xffffffff
    h3 = (h3 + d) & 0xffffffff
    h4 = (h4 + e) & 0xffffffff
  end

  local function word_to_bytes(x)
    return string.char((x >> 24) & 0xff, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff)
  end

  return word_to_bytes(h0) .. word_to_bytes(h1) .. word_to_bytes(h2) .. word_to_bytes(h3) .. word_to_bytes(h4)
end

local function ws_accept_key(sec_key)
  local GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
  return base64_encode(sha1(sec_key .. GUID))
end

-- =========================
-- HTTP utils
-- =========================
local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function parse_query(path)
  local p, q = path:match("^([^?]+)%??(.*)$")
  local query = {}
  if q and q ~= "" then
    for pair in q:gmatch("[^&]+") do
      local k, v = pair:match("^([^=]+)=?(.*)$")
      if k then
        k = k:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end)
        v = (v or ""):gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end)
        query[k] = v
      end
    end
  end
  return p or path, query
end

local function read_http_request(client)
  client:settimeout(10)
  local line, err = client:receive("*l")
  if not line then return nil, err end
  local method, path, httpver = line:match("^(%S+)%s+(%S+)%s+(HTTP/%d%.%d)$")
  if not method then return nil, "bad request line" end

  local headers = {}
  while true do
    local hline, herr = client:receive("*l")
    if not hline then return nil, herr end
    if hline == "" then break end
    local k, v = hline:match("^([^:]+):%s*(.*)$")
    if k then
      headers[k:lower()] = trim(v)
    end
  end

  local body = ""
  if headers["transfer-encoding"] then return nil, "Transfer-Encoding non supporté" end
  local length = headers["content-length"] or "0"
  if not length:match("^%d+$") then return nil, "Content-Length invalide" end
  local cl = tonumber(length)
  if not cl or cl > MAX_BODY_BYTES then return nil, "payload_too_large" end
  if cl and cl > 0 then
    local chunk, berr = client:receive(cl)
    if not chunk then return nil, berr end
    body = chunk
  end

  return { method = method, path = path, httpver = httpver, headers = headers, body = body }
end

local function http_response(status, headers, body)
  local reason = {
    [200] = "OK",
    [201] = "Created",
    [204] = "No Content",
    [401] = "Unauthorized",
    [403] = "Forbidden",
    [400] = "Bad Request",
    [404] = "Not Found",
    [405] = "Method Not Allowed",
    [413] = "Payload Too Large",
    [422] = "Unprocessable Entity",
    [426] = "Upgrade Required",
    [500] = "Internal Server Error",
    [503] = "Service Unavailable",
  }

  local lines = {}
  lines[#lines+1] = string.format("HTTP/1.1 %d %s", status, reason[status] or "")
  headers = headers or {}
  headers["Content-Length"] = tostring(#(body or ""))
  headers["Connection"] = headers["Connection"] or "close"
  for k, v in pairs(headers) do
    lines[#lines+1] = string.format("%s: %s", k, v)
  end
  lines[#lines+1] = ""
  lines[#lines+1] = body or ""
  return table.concat(lines, "\r\n")
end

-- =========================
-- Auth helpers
-- =========================
local function secure_equals(a, b)
  if type(a) ~= "string" or type(b) ~= "string" then return false end
  if #a ~= #b then return false end
  local diff = 0
  for i2 = 1, #a do
    diff = diff | (a:byte(i2) ~ b:byte(i2))
  end
  return diff == 0
end

local function csv_to_set(csv)
  local set = {}
  if type(csv) ~= "string" or csv == "" then return set end
  for item in csv:gmatch("[^,]+") do
    local v = trim(item)
    if v ~= "" then set[v] = true end
  end
  return set
end

local AUTH_SKIP_SET = csv_to_set(AUTH_SKIP_PATHS)

local function extract_bearer(headers)
  if type(headers) ~= "table" then return nil end
  local a = headers["authorization"]
  if type(a) ~= "string" or a == "" then return nil end
  local scheme, token = a:match("^(%S+)%s+(.+)$")
  if not scheme or not token then return nil end
  scheme = scheme:lower()
  if scheme ~= "bearer" and scheme ~= "token" then return nil end
  return trim(token)
end

local function extract_api_key(headers)
  if type(headers) ~= "table" then return nil end
  local k = headers["x-api-key"]
  if type(k) ~= "string" or k == "" then return nil end
  return trim(k)
end

local function auth_enabled()
  return AUTH_MODE ~= "off"
end

local function auth_mode_valid()
  return AUTH_MODE == "off" or AUTH_MODE == "token" or AUTH_MODE == "api_key" or AUTH_MODE == "either"
end

local function auth_check(method, clean_path, headers, query)
  if not auth_mode_valid() then
    return false, 500, { ["Content-Type"] = "application/json" }, json_encode({ ok = false, error = "auth mode invalide", detail = AUTH_MODE })
  end

  if not auth_enabled() then
    return true
  end

  if AUTH_SKIP_SET[clean_path] then
    return true
  end

  local want_token = AUTH_MODE == "token" or AUTH_MODE == "either"
  local want_key = AUTH_MODE == "api_key" or AUTH_MODE == "either"

  local provided_token = extract_bearer(headers) or (type(query) == "table" and (query.token or query.bearer) or nil)
  local provided_key = extract_api_key(headers) or (type(query) == "table" and (query.api_key or query.apikey or query.key) or nil)

  local token_ok = (want_token and type(AUTH_TOKEN) == "string" and AUTH_TOKEN ~= "" and type(provided_token) == "string" and secure_equals(provided_token, AUTH_TOKEN))
  local key_ok = (want_key and type(API_KEY) == "string" and API_KEY ~= "" and type(provided_key) == "string" and secure_equals(provided_key, API_KEY))

  if token_ok or key_ok then
    return true
  end

  local hdrs = {
    ["Content-Type"] = "application/json",
  }
  if want_token then
    hdrs["WWW-Authenticate"] = 'Bearer realm="mimir"'
  end
  return false, 401, hdrs, json_encode({ ok = false, error = "unauthorized" })
end

-- =========================
-- Framework dispatcher
-- =========================
local function mimir_available()
  return type(_G.Mimir) == "table" and type(_G.Mimir.Model) == "table"
end

local function framework_info()
  local available = mimir_available()
  local caps = {}
  if available then
    caps.model = type(Mimir.Model) == "table"
    caps.model_create = type(Mimir.Model.create) == "function"
    caps.model_build = type(Mimir.Model.build) == "function"
    caps.model_allocate_params = type(Mimir.Model.allocate_params) == "function"
    caps.model_infer = type(Mimir.Model.infer) == "function"
    caps.model_forward = type(Mimir.Model.forward) == "function"
    caps.model_init_weights = type(Mimir.Model.init_weights) == "function"
    caps.model_train = type(Mimir.Model.train) == "function"
    caps.serialization = type(Mimir.Serialization) == "table"
    caps.serialization_save = type(Mimir.Serialization) == "table" and type(Mimir.Serialization.save) == "function"
    caps.serialization_load = type(Mimir.Serialization) == "table" and type(Mimir.Serialization.load) == "function"
    caps.allocator = type(Mimir.Allocator) == "table"
    caps.allocator_configure = type(Mimir.Allocator) == "table" and type(Mimir.Allocator.configure) == "function"
    caps.allocator_stats = type(Mimir.Allocator) == "table" and type(Mimir.Allocator.get_stats) == "function"

    caps.architectures = type(Mimir.Architectures) == "table"
    caps.arch_available = type(Mimir.Architectures) == "table" and type(Mimir.Architectures.available) == "function"
    caps.arch_default_config = type(Mimir.Architectures) == "table" and type(Mimir.Architectures.default_config) == "function"
  end
  return {
    mimir_available = available,
    server_version = SERVER_VERSION,
    capabilities = caps,
  }
end

local request_counter = 0
local function next_request_id()
  request_counter = request_counter + 1
  local ms = math.floor(now_seconds() * 1000)
  return string.format("%x-%x", ms, request_counter)
end

local function enrich_json_body(body_json, req_meta)
  local ok, decoded = pcall(json_decode, body_json or "")
  local payload
  if ok and type(decoded) == "table" then
    payload = decoded
  else
    payload = { ok = false, error = "Réponse interne non-JSON", raw = tostring(body_json) }
  end

  payload._meta = payload._meta or {}
  for k, v in pairs(req_meta or {}) do
    payload._meta[k] = v
  end

  return json_encode(payload)
end

local ROUTES = {
  { method = "GET", path = "/health", desc = "/health"
  },
  { method = "GET", path = "/help", desc = "/help"
  },
  { method = "GET", path = "/capabilities", desc = "/capabilities"
  },
  { method = "POST", path = "/rpc",
    body = {api = "Mimir.Benchmark.now",
    args = json_array()}, desc = "Appel d’un binding public avec arguments positionnels et retours multiples"
  },
  { method = "GET", path = "/architectures",
    api = "Architectures.available", result = "architectures", desc = "Architectures.available"
  },
  { method = "GET", path = "/architectures/default_config",
    api = "Architectures.default_config",
    args = {"model_type"}, result = "config", desc = "Architectures.default_config",
    required = {"model_type"},
    query = {model_type = "basic_mlp"}
  },
  { method = "GET", path = "/architectures/info",
    api = "Architectures.info",
    args = {"model_type"}, result = "info", desc = "Architectures.info",
    query = {model_type = "basic_mlp"}
  },
  { method = "GET", path = "/architectures/dtypes",
    api = "Architectures.dtypes", result = "dtypes", desc = "Architectures.dtypes"
  },
  { method = "GET", path = "/hardware",
    api = "Model.hardware_caps", result = "caps", desc = "Model.hardware_caps"
  },
  { method = "PUT", path = "/hardware",
    api = "Model.set_hardware",
    args = {"enable"}, result = "enabled",
    body = {enable = false}, desc = "Model.set_hardware"
  },
  { method = "GET", path = "/allocator",
    api = "Allocator.get_stats", result = "stats", desc = "Allocator.get_stats"
  },
  { method = "PATCH", path = "/allocator",
    api = "Allocator.configure",
    args = {"$body"},
    body = {max_ram_gb = 10.0, enable_compression = true}, desc = "Allocator.configure"
  },
  { method = "DELETE", path = "/allocator", desc = "Ack + GC Lua ; aucun reset natif exposé"
  },
  { method = "POST", path = "/model/create",
    api = "Model.create",
    args = {"model_type", "config"},
    body = {model_type = "basic_mlp", config = {input_dim = 4, hidden_dim = 8, output_dim = 2}}, desc = "Model.create",
    required = {"model_type"}
  },
  { method = "POST", path = "/model/create_from_config",
    api = "Model.create_from_config",
    args = {"$body"}, result = "architecture",
    body = {architecture = "basic_mlp", basic_mlp = {input_dim = 4, hidden_dim = 8, output_dim = 2}}, desc = "Model.create_from_config"
  },
  { method = "POST", path = "/model/build",
    api = "Model.build", result = "params", desc = "Model.build"
  },
  { method = "POST", path = "/model/allocate_params",
    api = "Model.allocate_params", result = "params", desc = "Model.allocate_params"
  },
  { method = "POST", path = "/model/init_weights",
    api = "Model.init_weights",
    args = {"init", "seed"},
    body = {init = "he", seed = 42}, desc = "Model.init_weights",
    defaults = {init = "he", seed = 0}
  },
  { method = "GET", path = "/model/params",
    api = "Model.total_params", result = "params", desc = "Model.total_params"
  },
  { method = "GET", path = "/model/config",
    api = "Model.get_config", result = "config", desc = "Model.get_config"
  },
  { method = "GET", path = "/model/layers",
    api = "Model.get_layers", result = "layers", desc = "Model.get_layers"
  },
  { method = "GET", path = "/model/dtype",
    api = "Model.dtype", result = "dtype", desc = "Model.dtype"
  },
  { method = "PUT", path = "/model/dtype",
    api = "Model.dtype",
    args = {"dtype"}, result = "dtype",
    body = {dtype = "float32"}, desc = "Model.dtype",
    required = {"dtype"}
  },
  { method = "POST", path = "/model/forward",
    api = "Model.forward",
    args = {"input", "training"}, result = "output",
    body = {input = {0.1, 0.2, 0.3, 0.4}, training = false}, desc = "Model.forward",
    required = {"input"},
    defaults = {training = false}
  },
  { method = "POST", path = "/model/infer",
    api = "Model.infer",
    args = {"input"}, result = "output",
    body = {input = {0.1, 0.2, 0.3, 0.4}}, desc = "Model.infer",
    required = {"input"}
  },
  { method = "POST", path = "/model/backward",
    api = "Model.backward",
    args = {"gradient"},
    body = {gradient = {0.1, -0.2}}, desc = "Model.backward",
    required = {"gradient"}
  },
  { method = "POST", path = "/model/zero_grads",
    api = "Model.zero_grads", desc = "Model.zero_grads"
  },
  { method = "POST", path = "/model/optimizer_step",
    api = "Model.optimizer_step",
    args = {"learning_rate", "optimizer"},
    body = {learning_rate = 0.001, optimizer = "adamw"}, desc = "Model.optimizer_step",
    required = {"learning_rate"},
    defaults = {optimizer = "adamw"}
  },
  { method = "GET", path = "/model/gradients",
    api = "Model.get_gradients", result = "gradients", desc = "Model.get_gradients"
  },
  { method = "POST", path = "/model/train",
    api = "Model.train",
    args = {"epochs", "learning_rate"},
    body = {epochs = 1, learning_rate = 0.001}, desc = "Model.train",
    required = {"epochs", "learning_rate"}
  },
  { method = "DELETE", path = "/model", desc = "Ack + GC Lua ; aucun reset natif du modèle exposé"
  },
  { method = "POST", path = "/tokenizer/create",
    api = "Tokenizer.create",
    args = {"max_vocab"},
    body = {max_vocab = 256}, desc = "Tokenizer.create",
    required = {"max_vocab"}
  },
  { method = "POST", path = "/tokenizer/tokenize",
    api = "Tokenizer.tokenize",
    args = {"text"}, result = "tokens",
    body = {text = "bonjour monde"}, desc = "Tokenizer.tokenize",
    required = {"text"}
  },
  { method = "POST", path = "/tokenizer/tokenize_ensure",
    api = "Tokenizer.tokenize_ensure",
    args = {"text"}, result = "tokens",
    body = {text = "bonjour monde"}, desc = "Tokenizer.tokenize_ensure",
    required = {"text"}
  },
  { method = "POST", path = "/tokenizer/detokenize",
    api = "Tokenizer.detokenize",
    args = {"tokens"}, result = "text",
    body = {tokens = {1, 2}}, desc = "Tokenizer.detokenize",
    required = {"tokens"}
  },
  { method = "POST", path = "/tokenizer/learn_bpe",
    api = "Tokenizer.learn_bpe",
    args = {"corpus", "num_merges"},
    body = {corpus = {"bonjour monde", "bonjour Mimir"}, num_merges = 32}, desc = "Tokenizer.learn_bpe",
    required = {"corpus"},
    defaults = {num_merges = 32}
  },
  { method = "GET", path = "/tokenizer/vocab_size",
    api = "Tokenizer.vocab_size", result = "vocab_size", desc = "Tokenizer.vocab_size"
  },
  { method = "POST", path = "/tokenizer/save",
    api = "Tokenizer.save",
    args = {"path"},
    body = {path = "checkpoint/tokenizer.json"}, desc = "Tokenizer.save",
    required = {"path"}
  },
  { method = "POST", path = "/tokenizer/load",
    api = "Tokenizer.load",
    args = {"path"},
    body = {path = "checkpoint/tokenizer.json"}, desc = "Tokenizer.load",
    required = {"path"}
  },
  { method = "POST", path = "/dataset/load",
    api = "Dataset.load",
    args = {"dir", "width", "height", "min_modalities", "use_cache", "cache_path", "max_ram_mb", "lazy_loading"}, result = "count",
    body = {dir = "dataset/example", width = 64, height = 64, min_modalities = 1, use_cache = false, cache_path = "checkpoint/dataset-index.json", max_ram_mb = 128, lazy_loading = true}, desc = "Dataset.load",
    required = {"dir"},
    defaults = {width = 64, height = 64, min_modalities = 1, use_cache = false, max_ram_mb = 128, lazy_loading = true}
  },
  { method = "GET", path = "/dataset/item",
    api = "Dataset.get",
    args = {"index", "load_image"}, result = "item", desc = "Dataset.get",
    required = {"index"},
    defaults = {load_image = true},
    query = {index = 1, load_image = false}
  },
  { method = "POST", path = "/dataset/release",
    api = "Dataset.release",
    args = {"index"},
    body = {index = 1}, desc = "Dataset.release",
    required = {"index"}
  },
  { method = "POST", path = "/dataset/prepare_sequences",
    api = "Dataset.prepare_sequences",
    args = {"max_length"}, result = "count",
    body = {max_length = 128}, desc = "Dataset.prepare_sequences",
    defaults = {max_length = 128}
  },
  { method = "GET", path = "/memory",
    api = "Memory.get_stats", result = "stats", desc = "Memory.get_stats"
  },
  { method = "PATCH", path = "/memory",
    api = "Memory.config",
    args = {"$body"},
    body = {max_ram_gb = 2.0}, desc = "Memory.config"
  },
  { method = "GET", path = "/memory/guard",
    api = "MemoryGuard.getStats", result = "stats", desc = "MemoryGuard.getStats"
  },
  { method = "GET", path = "/layers",
    api = "Layers.available", result = "layers", desc = "Layers.available"
  },
  { method = "GET", path = "/layers/by_type",
    api = "Layers.by_type",
    args = {"type"}, result = "layers", desc = "Layers.by_type",
    required = {"type"},
    query = {type = "Linear"}
  },
  { method = "POST", path = "/serialization/load",
    api = "Serialization.load",
    args = {"path", "format", "options"},
    body = {path = "checkpoint/api-smoke", format = "raw_folder", options = {verify_checksums = true}}, desc = "Serialization.load",
    required = {"path"}
  },
  { method = "POST", path = "/serialization/save",
    api = "Serialization.save",
    args = {"path", "format", "options"},
    body = {path = "checkpoint/api-smoke", format = "raw_folder", options = {include_git_info = true}}, desc = "Serialization.save",
    required = {"path"}
  },
  { method = "POST", path = "/serialization/detect_format",
    api = "Serialization.detect_format",
    args = {"path"}, result = "format",
    body = {path = "checkpoint/api-smoke"}, desc = "Serialization.detect_format",
    required = {"path"}
  },
  { method = "GET", path = "/viz/backend",
    api = "Viz.backend", result = "backend", desc = "Viz.backend"
  },
  { method = "GET", path = "/viz/events",
    api = "Viz.poll_events",
    args = {"timeout_ms"}, result = "events", desc = "Viz.poll_events",
    defaults = {timeout_ms = 0},
    query = {timeout_ms = 0}
  },
  { method = "POST", path = "/viz/create",
    api = "Viz.create",
    args = {"$body"},
    body = {visualization = {enabled = true, window_width = 1000, window_height = 700}}, desc = "Viz.create"
  },
  { method = "PATCH", path = "/viz",
    api = "Viz.configure",
    args = {"$body"},
    body = {panels = {{id = 1, title = "Preview", x = 20, y = 100, w = 600, h = 400}}, controls = {{id = "apply", label = "Apply", x = 20, y = 20, w = 150, h = 40}}, events = true}, desc = "Viz.configure"
  },
  { method = "PUT", path = "/viz/validation",
    api = "Viz.set_validation",
    args = {"$body"},
    body = {enabled = true, in_progress = false, step = 0}, desc = "Viz.set_validation"
  },
  { method = "GET", path = "/viz/validation",
    api = "Viz.validation_enabled", result = "enabled", desc = "Viz.validation_enabled"
  },
  { method = "PUT", path = "/viz/enabled",
    api = "Viz.set_enabled",
    args = {"enabled"},
    body = {enabled = false}, desc = "Viz.set_enabled",
    required = {"enabled"}
  },
}

local function public_api(name)
  if type(name) ~= "string" then return nil end
  name = name:gsub("^Mimir%.", "")
  local namespace, method = name:match("^([%a_][%w_]*)%.([%a_][%w_]*)$")
  local root = rawget(_G, "Mimir")
  local ns = root and namespace and root[namespace]
  return type(ns) == "table" and type(ns[method]) == "function" and ns[method] or nil
end

local function api_surface()
  local surface = setmetatable({}, JSON_OBJECT)
  for namespace, ns in pairs(rawget(_G, "Mimir") or {}) do
    if type(ns) == "table" then
      local methods = json_array()
      for name, fn in pairs(ns) do
        if type(fn) == "function" then methods[#methods + 1] = name end
      end
      table.sort(methods)
      surface[namespace] = methods
    end
  end
  return surface
end

local function native_value(value)
  if value == JSON_NULL then return nil end
  if type(value) ~= "table" then return value end
  local out = {}
  for key, item in pairs(value) do out[key] = native_value(item) end
  return out
end

local function invoke_api(fn, args)
  local result = table.pack(pcall(fn, table.unpack(args, 1, args.n or #args)))
  if not result[1] then return nil, tostring(result[2]), 400 end
  if (result[2] == nil or result[2] == false) and type(result[3]) == "string" then
    return nil, result[3], 422
  end
  local values = json_array()
  for i = 2, result.n do values[i - 1] = result[i] == nil and JSON_NULL or result[i] end
  return values
end

local function api_handle_impl(method, path, headers, body)
  local clean_path, query = parse_query(path)
  local function response(payload, status, extra_headers)
    local hdrs = extra_headers or {}
    hdrs["Content-Type"] = "application/json"
    return status or 200, hdrs, json_encode(payload)
  end
  if method == "OPTIONS" then return response({ ok = true }) end
  do
    local authorized, status, hdrs, resp = auth_check(method, clean_path, headers, query)
    if not authorized then return status, hdrs, resp end
  end

  local payload
  if body and body ~= "" then
    if not (headers["content-type"] or ""):lower():find("application/json", 1, true) then
      return response({ ok = false, error = "Content-Type application/json requis" }, 400)
    end
    local success, decoded = pcall(json_decode, body)
    if not success or type(decoded) ~= "table" or decoded == JSON_NULL or getmetatable(decoded) ~= JSON_OBJECT then
      return response({ ok = false, error = "Objet JSON requis", detail = success and nil or tostring(decoded) }, 400)
    end
    payload = decoded
  end

  if method == "GET" and clean_path == "/health" then
    return response({ ok = true, mimir = mimir_available(), time = iso_utc() })
  end
  if method == "GET" and clean_path == "/capabilities" then
    return response({ ok = true, framework = framework_info(), api = api_surface() })
  end
  if method == "GET" and clean_path == "/help" then
    local descriptions = json_array()
    for _, spec in ipairs(ROUTES) do
      descriptions[#descriptions + 1] = { method = spec.method, path = spec.path, api = spec.api,
        desc = spec.desc, body = spec.body, query = spec.query,
        available = spec.api == nil or public_api(spec.api) ~= nil }
    end
    return response({ ok = true,
      server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
      routes = descriptions, api = api_surface(), max_body_bytes = MAX_BODY_BYTES,
      security = { tls = TLS_ENABLED, auth = { mode = AUTH_MODE, enabled = auth_enabled(),
        skip_paths = AUTH_SKIP_PATHS, accepted = { bearer = AUTH_MODE == "token" or AUTH_MODE == "either",
        api_key = AUTH_MODE == "api_key" or AUTH_MODE == "either", query_for_ws = true } } },
      websocket = { path = "/ws", example_message = { id = "health-1", method = "GET", path = "/health" },
        desc = "Messages {id, method, path, body}; réponses {id, ok, status, response, _meta}. Auth de session vérifiée au handshake. Serveur mono-client: fermer WS avant les requêtes HTTP." },
    })
  end

  if method == "POST" and clean_path == "/rpc" then
    if not payload or type(payload.api) ~= "string" or not is_array(payload.args) then
      return response({ ok = false, error = "body requis: {api='Model.forward', args=[...]}" }, 400)
    end
    local fn = public_api(payload.api)
    if not fn then return response({ ok = false, error = "API publique indisponible", api = payload.api }, 404) end
    if payload.api:gsub("^Mimir%.", "") == "Database.load" then
      return response({ ok = false, error = "Database.load retourne un builder ; utilisez /dataset/load avec use_cache=true" }, 400)
    end
    local args = { n = #payload.args }
    for i = 1, args.n do args[i] = native_value(payload.args[i]) end
    local values, err, status = invoke_api(fn, args)
    if not values then return response({ ok = false, error = err, api = payload.api }, status) end
    return response({ ok = true, results = values, api = payload.api })
  end

  local route, allowed = nil, {}
  for _, spec in ipairs(ROUTES) do
    if spec.path == clean_path then
      allowed[#allowed + 1] = spec.method
      if spec.method == method then route = spec end
    end
  end
  if not route then
    if #allowed > 0 then
      return response({ ok = false, error = "Méthode non autorisée" }, 405, { Allow = table.concat(allowed, ", ") })
    end
    return response({ ok = false, error = "route inconnue", method = method, path = clean_path }, 404)
  end
  if method == "DELETE" then
    collectgarbage("collect")
    return response({ ok = true, note = "GC Lua exécuté ; aucun reset natif exposé pour cette route." })
  end
  local fn = public_api(route.api)
  if not fn then return response({ ok = false, error = route.api .. " indisponible" }, 503) end
  local input = payload or query
  if method ~= "GET" and #(route.args or {}) > 0 and not payload and clean_path ~= "/hardware" then
    return response({ ok = false, error = "body JSON requis" }, 400)
  end

  -- Compatibility: backend=cpu/auto remains an acceleration toggle, not a GPU selector.
  if clean_path == "/hardware" and method == "PUT" then
    local enable = input.enable
    if type(enable) == "string" then
      local value = enable:lower()
      if value == "true" or value == "1" or value == "yes" or value == "on" then enable = true
      elseif value == "false" or value == "0" or value == "no" or value == "off" then enable = false end
    elseif type(enable) == "number" then enable = enable ~= 0 end
    if enable == nil and input.backend ~= nil then
      if input.backend == "cpu" then enable = false
      elseif input.backend == "auto" then enable = true end
    end
    if type(enable) ~= "boolean" then
      return response({ ok = false, error = "enable doit être un booléen (ou backend='cpu'/'auto')" }, 400)
    end
    input.enable = enable
  end
  for _, name in ipairs(route.required or {}) do
    if input[name] == nil or input[name] == JSON_NULL or input[name] == "" then
      return response({ ok = false, error = name .. " requis" }, 400)
    end
  end
  if clean_path == "/model/create" and input.config == nil then
    local default_config = public_api("Architectures.default_config")
    if default_config then
      local configs, err, status = invoke_api(default_config, {input.model_type})
      if not configs then return response({ ok = false, error = err }, status) end
      input.config = configs[1]
    end
  end
  local args = { n = #(route.args or {}) }
  for i, name in ipairs(route.args or {}) do
    local value = name == "$body" and payload or input[name]
    if value == nil and route.defaults then value = route.defaults[name] end
    if not payload and type(value) == "string" then
      if value == "true" then value = true elseif value == "false" then value = false
      elseif name == "index" or name == "timeout_ms" then value = tonumber(value) end
    end
    args[i] = native_value(value)
  end
  local values, err, status = invoke_api(fn, args)
  if not values then return response({ ok = false, error = err }, status) end
  local out = { ok = true }
  if route.result then
    local first = values[1]
    if first == true and #values > 1 then first = values[2] end
    -- JSON's empty arrays need explicit tagging because Lua native tables are ambiguous.
    if type(first) == "table" and first ~= JSON_NULL and
        (route.result == "architectures" or route.result == "dtypes" or route.result == "layers" or
         route.result == "tokens" or route.result == "output" or route.result == "events") and next(first) == nil then
      first = json_array(first)
    end
    out[route.result] = first
    if clean_path == "/hardware" and method == "PUT" then out.enabled = input.enable end
  end
  return response(out)
end

local function api_handle(...)
  local result = table.pack(pcall(api_handle_impl, ...))
  if result[1] then return table.unpack(result, 2, result.n) end
  return 500, { ["Content-Type"] = "application/json" },
    json_encode({ ok = false, error = "Erreur interne API", detail = tostring(result[2]) })
end

-- =========================
-- WebSocket framing
-- =========================
local function recv_exact(sock, n)
  local chunks = {}
  local got = 0
  while got < n do
    local part, err, partial = sock:receive(n - got)
    part = part or partial
    if part and #part > 0 then
      chunks[#chunks+1] = part
      got = got + #part
    else
      return nil, err or "receive failed"
    end
  end
  return table.concat(chunks)
end

local function send_all(sock, data)
  local index = 1
  while index <= #data do
    local last, err, partial = sock:send(data, index)
    if not last then return nil, err, partial end
    if last < index then return nil, "send failed" end
    index = last + 1
  end
  return true
end

local function ws_send(sock, opcode, payload)
  payload = payload or ""
  local len = #payload
  if len > MAX_BODY_BYTES then return nil, "Payload WS trop gros" end
  local header = string.char(0x80 | opcode)
  if len < 126 then header = header .. string.char(len)
  elseif len < 65536 then header = header .. string.char(126) .. string.pack(">I2", len)
  else header = header .. string.char(127) .. string.pack(">I8", len) end
  return send_all(sock, header .. payload)
end

local function ws_send_text(sock, text) return ws_send(sock, 0x1, text) end

local function ws_read_frame(sock)
  local hdr, err = recv_exact(sock, 2)
  if not hdr then return nil, err end
  local b1, b2 = hdr:byte(1,2)
  if (b1 & 0x70) ~= 0 then return nil, "Extensions WS non supportées" end
  local fin = (b1 & 0x80) ~= 0
  local opcode = b1 & 0x0f
  local masked = (b2 & 0x80) ~= 0
  if not masked then return nil, "Frame client WS non masquée" end
  local len = b2 & 0x7f
  if len == 126 then
    local ext, err2 = recv_exact(sock, 2)
    if not ext then return nil, err2 end
    len = (ext:byte(1) << 8) | ext:byte(2)
    if len < 126 then return nil, "Longueur WS non canonique" end
  elseif len == 127 then
    local ext, err2 = recv_exact(sock, 8)
    if not ext then return nil, err2 end
    if ext:byte(1) >= 128 then return nil, "Longueur WS invalide" end
    len = string.unpack(">I8", ext)
    if len < 65536 then return nil, "Longueur WS non canonique" end
  end

  if len > MAX_BODY_BYTES then return nil, "Payload WS trop gros" end
  if opcode >= 8 and (not fin or len > 125) then return nil, "Frame de contrôle WS invalide" end
  if opcode == 8 and len == 1 then return nil, "Payload close WS invalide" end
  local mask
  if masked then
    local m, err2 = recv_exact(sock, 4)
    if not m then return nil, err2 end
    mask = m
  end

  local payload = ""
  if len > 0 then
    local p, err2 = recv_exact(sock, len)
    if not p then return nil, err2 end
    payload = p
  end

  if masked and payload and mask then
    local mask_bytes, index = {mask:byte(1, 4)}, 0
    payload = payload:gsub(".", function(byte)
      local masked_byte = byte:byte() ~ mask_bytes[(index % 4) + 1]
      index = index + 1
      return string.char(masked_byte)
    end)
  end

  return { fin = fin, opcode = opcode, payload = payload }
end

-- =========================
-- Serveur
-- =========================
local function handle_websocket(client, req)
  local key = req.headers["sec-websocket-key"]
  if req.method ~= "GET" or req.headers["sec-websocket-version"] ~= "13" or
      not key or not key:match("^[%w+/]+==$" ) or #key ~= 24 then
    send_all(client, http_response(400, { ["Content-Type"] = "text/plain" }, "WebSocket GET, version 13 et clé 16 bytes requis"))
    client:close()
    return
  end

  local clean_path, query = parse_query(req.path or "/ws")
  do
    local ok_auth, status, hdrs, resp = auth_check("GET", clean_path, req.headers or {}, query)
    if not ok_auth then
      send_all(client, http_response(status, hdrs, resp))
      client:close()
      return
    end
  end

  local accept = ws_accept_key(key)
  local resp = table.concat({
    "HTTP/1.1 101 Switching Protocols",
    "Upgrade: websocket",
    "Connection: Upgrade",
    "Sec-WebSocket-Accept: " .. accept,
    "\r\n",
  }, "\r\n")

  send_all(client, resp)
  client:settimeout(10)
  local session_headers = { ["content-type"] = "application/json",
    authorization = req.headers.authorization, ["x-api-key"] = req.headers["x-api-key"] }
  local session_token = query.token or query.bearer
  local session_key = query.api_key or query.apikey or query.key
  if not session_headers.authorization and session_token then session_headers.authorization = "Bearer " .. session_token end
  if not session_headers["x-api-key"] and session_key then session_headers["x-api-key"] = session_key end
  local fragments, fragment_bytes = nil, 0
  while true do
    -- Wait before reading; a partial TCP frame is then completed without losing bytes.
    local buffered = type(client.dirty) == "function" and client:dirty()
    local ready = buffered and {client} or socket.select({client}, nil, 0.1)
    if #ready > 0 then
      local frame, frame_error = ws_read_frame(client)
      if not frame then
        ws_send(client, 0x8, string.pack(">I2", 1002))
        break
      end
      if frame.opcode == 0x8 then ws_send(client, 0x8, frame.payload); break
      elseif frame.opcode == 0x9 then
        if not ws_send(client, 0xA, frame.payload) then break end
      elseif frame.opcode == 0xA then -- pong
      elseif frame.opcode == 0x1 or frame.opcode == 0x0 then
        if (frame.opcode == 0x0 and not fragments) or (frame.opcode == 0x1 and fragments) then
          ws_send(client, 0x8, string.pack(">I2", 1002)); break
        end
        fragments = fragments or {}
        fragments[#fragments + 1] = frame.payload
        fragment_bytes = fragment_bytes + #frame.payload
        if fragment_bytes > MAX_BODY_BYTES then ws_send(client, 0x8, string.pack(">I2", 1009)); break end
        if frame.fin then
          local payload = table.concat(fragments)
          fragments, fragment_bytes = nil, 0
          if not utf8.len(payload) then ws_send(client, 0x8, string.pack(">I2", 1007)); break end
          local decoded_ok, msg = pcall(json_decode, payload)
          local response
          if not decoded_ok or type(msg) ~= "table" or getmetatable(msg) ~= JSON_OBJECT then
            response = { ok = false, status = 400, error = "message JSON invalide" }
          elseif (msg.method ~= nil and type(msg.method) ~= "string") or
                 (msg.path ~= nil and type(msg.path) ~= "string") then
            response = { id = msg.id, ok = false, status = 400, error = "method/path doivent être des chaînes" }
          else
            local t0 = now_seconds()
            local m, path = (msg.method or "GET"):upper(), msg.path or "/health"
            local body = msg.body ~= nil and json_encode(msg.body) or ""
            local status, _, data = api_handle(m, path, session_headers, body)
            local data_ok, result = pcall(json_decode, data)
            if not data_ok then result = { ok = false, error = "Réponse interne JSON invalide" }; status = 500 end
            response = { id = msg.id, ok = status < 400 and result.ok ~= false, status = status, response = result,
              _meta = { request_id = next_request_id(), method = m, path = path, received_at_utc = iso_utc(),
                latency_ms = math.floor((now_seconds() - t0) * 1000 + 0.5),
                server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
                framework = framework_info() } }
          end
          if not ws_send_text(client, json_encode(response)) then break end
        end
      else
        ws_send(client, 0x8, string.pack(">I2", 1003)); break
      end
    end
  end
  client:close()
end

local function serve_forever()
  if TLS_ENABLED then
    if not ssl then
      error("TLS demandé (--tls/--https) mais LuaSec n'est pas disponible (module 'ssl').")
    end
    if TLS_CERT_FILE == "" or TLS_KEY_FILE == "" then
      error("TLS activé mais --tls-cert et --tls-key sont requis.")
    end
  end

  local server = assert(socket.bind(HOST, PORT))
  server:settimeout(0)
  local scheme = TLS_ENABLED and "https" or "http"
  print(string.format("[api] listening on %s://%s:%d (ws: /ws) name=%s version=%s lua=%s", scheme, HOST, PORT, SERVER_NAME, SERVER_VERSION, _VERSION))

  while true do
    local client = server:accept()
    if client then
      local t0 = now_seconds()
      local request_id = next_request_id()

      if TLS_ENABLED then
        client:settimeout(10)
        local wrapped, werr = ssl.wrap(client, {
          mode = "server",
          protocol = "tlsv1_2",
          key = TLS_KEY_FILE,
          certificate = TLS_CERT_FILE,
          options = { "all" },
        })
        if not wrapped then
          client:close()
          goto continue
        end
        wrapped:settimeout(10)
        local ok_hs, herr = wrapped:dohandshake()
        if not ok_hs then
          wrapped:close()
          goto continue
        end
        client = wrapped
      end

      local req, err = read_http_request(client)
      if not req then
        local t1 = now_seconds()
        local latency_ms = math.floor((t1 - t0) * 1000 + 0.5)
        local body = json_encode({
          ok = false,
          error = "Bad request",
          detail = tostring(err),
          _meta = {
            request_id = request_id,
            received_at_utc = iso_utc(),
            latency_ms = latency_ms,
            server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
            framework = framework_info(),
          }
        })
        send_all(client, http_response(err == "payload_too_large" and 413 or 400,
          { ["Content-Type"] = "application/json", ["X-Response-Time-Ms"] = tostring(latency_ms) }, body))
        client:close()
      else
        local upgrade = (req.headers["upgrade"] or ""):lower()
        local connection = (req.headers["connection"] or ""):lower()
        if parse_query(req.path) == "/ws" and upgrade == "websocket" and connection:find("upgrade", 1, true) then
          handle_websocket(client, req)
        else
          local status, hdrs, resp_body = api_handle(req.method, req.path, req.headers, req.body)
          local t1 = now_seconds()
          local latency_ms = math.floor((t1 - t0) * 1000 + 0.5)
          hdrs = hdrs or {}
          hdrs["X-Response-Time-Ms"] = tostring(latency_ms)

          -- En-têtes de durcissement côté API
          hdrs["X-Content-Type-Options"] = hdrs["X-Content-Type-Options"] or "nosniff"
          hdrs["X-Frame-Options"] = hdrs["X-Frame-Options"] or "DENY"
          hdrs["Referrer-Policy"] = hdrs["Referrer-Policy"] or "no-referrer"
          hdrs["Cache-Control"] = hdrs["Cache-Control"] or "no-store"
          if TLS_ENABLED then
            hdrs["Strict-Transport-Security"] = hdrs["Strict-Transport-Security"] or "max-age=31536000"
          end

          -- CORS (utile quand la page web est sur un autre origin, y compris file://)
          if CORS_ENABLED then
            local origin = req.headers["origin"]
            local allow_origin = CORS_ORIGIN
            if allow_origin == "reflect" then
              if type(origin) == "string" and origin ~= "" then
                allow_origin = origin
                -- Indique au cache/proxy que la réponse dépend de Origin
                hdrs["Vary"] = hdrs["Vary"] or "Origin"
              else
                allow_origin = "*"
              end
            end

            hdrs["Access-Control-Allow-Origin"] = allow_origin
            hdrs["Access-Control-Allow-Methods"] = CORS_ALLOW_METHODS
            hdrs["Access-Control-Allow-Headers"] = CORS_ALLOW_HEADERS
            hdrs["Access-Control-Expose-Headers"] = CORS_EXPOSE_HEADERS
            hdrs["Access-Control-Max-Age"] = CORS_MAX_AGE
          end

          local meta = {
            request_id = request_id,
            method = req.method,
            path = req.path,
            received_at_utc = iso_utc(),
            latency_ms = latency_ms,
            server = { name = SERVER_NAME, version = SERVER_VERSION, host = HOST, port = PORT, lua = _VERSION },
            framework = framework_info(),
            client = {
              user_agent = req.headers["user-agent"],
              content_type = req.headers["content-type"],
            },
          }

          local enriched = enrich_json_body(resp_body, meta)
          send_all(client, http_response(status, hdrs, enriched))
          client:close()
        end
      end
    else
      socket.sleep(0.01)
    end

    ::continue::
  end
end

serve_forever()
