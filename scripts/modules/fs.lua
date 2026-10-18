-- Cross-platform filesystem helpers for Lua scripts (Linux/macOS/Windows).

---@class MimirFSModule
local FS = {}

local path_sep = package.config:sub(1, 1)
local is_windows = path_sep == "\\"

local function shell_quote(s)
  s = tostring(s or "")
  if is_windows then
    -- Use double quotes for cmd.exe and escape internal quotes.
    return '"' .. s:gsub('"', '\\"') .. '"'
  end
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function run_cmd_success(cmd)
  local ok, why, code = os.execute(cmd)
  if type(ok) == "number" then return ok == 0 end
  if type(ok) == "boolean" then return ok end
  return (why == "exit" and code == 0)
end

local function normalize_sep(path)
  path = tostring(path or "")
  if path == "" then return "" end
  if is_windows then
    return path:gsub("/", "\\")
  end
  return path:gsub("\\", "/")
end

function FS.is_windows()
  return is_windows
end

function FS.sep()
  return path_sep
end

function FS.quote(path)
  return shell_quote(path)
end

function FS.normalize(path)
  return normalize_sep(path)
end

function FS.join(...)
  local parts = {...}
  local out = ""
  for i = 1, #parts do
    local p = normalize_sep(parts[i])
    if p ~= "" then
      if out == "" then
        out = p
      else
        if out:sub(-1) ~= path_sep then
          out = out .. path_sep
        end
        while p:sub(1, 1) == "/" or p:sub(1, 1) == "\\" do
          p = p:sub(2)
        end
        out = out .. p
      end
    end
  end
  return out
end

function FS.dirname(path)
  path = normalize_sep(path)
  if path == "" then return nil end
  local pat = "^(.*)" .. (is_windows and "\\" or "/") .. "[^\\/]*$"
  return path:match(pat)
end

function FS.file_exists(path)
  path = tostring(path or "")
  if path == "" then return false end
  local f = io.open(path, "rb")
  if f then
    f:close()
    return true
  end
  return false
end

function FS.is_dir(path)
  path = tostring(path or "")
  if path == "" then return false end
  if is_windows then
    return run_cmd_success("if exist " .. shell_quote(FS.join(path, "NUL")) .. " (exit /b 0) else (exit /b 1)")
  end
  return run_cmd_success("test -d " .. shell_quote(path) .. " >/dev/null 2>&1")
end

function FS.exists(path)
  return FS.file_exists(path) or FS.is_dir(path)
end

function FS.mkdir_p(path)
  path = tostring(path or "")
  if path == "" then return true end
  if is_windows then
    return run_cmd_success("mkdir " .. shell_quote(normalize_sep(path)) .. " >NUL 2>NUL")
  end
  return run_cmd_success("mkdir -p " .. shell_quote(path) .. " >/dev/null 2>&1")
end

function FS.list_dir(path)
  path = tostring(path or "")
  if path == "" then return {} end

  local cmd
  if is_windows then
    cmd = "dir /b /a " .. shell_quote(normalize_sep(path)) .. " 2>NUL"
  else
    cmd = "ls -1 " .. shell_quote(path) .. " 2>/dev/null"
  end

  local p = io.popen(cmd)
  if not p then return {} end
  local out = {}
  for line in p:lines() do
    if line and line ~= "" then
      out[#out + 1] = line
    end
  end
  p:close()
  table.sort(out)
  return out
end

return FS
