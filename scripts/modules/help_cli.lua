---@class MimirHelpModule
local M = {}

local function read_file(path)
  if type(path) ~= "string" or path == "" then return nil end
  local f = io.open(path, "r")
  if not f then return nil end
  local ok, data = pcall(function()
    return f:read("*a")
  end)
  f:close()
  if not ok then return nil end
  return data
end

local function trim(s)
  return (tostring(s or ""):gsub("^%s+", ""):gsub("%s+$", ""))
end

local function split_lines(s)
  local out = {}
  for line in tostring(s or ""):gmatch("([^\n]*)\n?") do
    if line == "" and #out > 0 and out[#out] == "" then
      break
    end
    out[#out + 1] = line
  end
  return out
end

function M.should_show_help(argv)
  local a = argv or _G.arg or {}
  for i = 1, #a do
    local v = a[i]
    if v == "--help" or v == "-h" or v == "--h" then
      return true
    end
  end
  return false
end

function M.find_script_from_stack(start_level, max_level)
  local first = start_level or 2
  local last = max_level or 12
  local this_src = debug.getinfo(1, "S").source
  for lvl = first, last do
    local info = debug.getinfo(lvl, "S")
    if info and type(info.source) == "string" then
      local src = info.source
      if src:sub(1, 1) == "@" then
        local path = src:sub(2)
        if path:sub(-4) == ".lua" and src ~= this_src and not path:match("help_cli%.lua$") then
          return path
        end
      end
    end
  end
  return nil
end

local function infer_description(path)
  local data = read_file(path)
  if not data then return nil end
  local lines = split_lines(data)
  local comments = {}
  for i = 1, math.min(#lines, 80) do
    local line = lines[i]
    if line:match("^%s*%-%-") then
      local txt = trim((line:gsub("^%s*%-%-%s?", "")))
      local is_usage = txt:match("^[Uu]sage") or txt:match("^[Oo]ptions")
      local is_directive = txt:match("^@") or txt:match("^%-@")
      local is_separator = txt:match("^[=%-%*_#%.%s]+$") ~= nil
      if txt ~= "" and not is_usage and not is_directive and not is_separator then
        comments[#comments + 1] = txt
      end
    elseif line:match("^%s*$") then
      -- keep scanning through initial blank lines
    else
      break
    end
  end
  return comments[1]
end

local function add_option(set, key)
  if not key then return end
  local k = trim(key)
  if k == "" then return end
  k = k:gsub("_", "-")
  set[k] = true
end

local function infer_options(path)
  local data = read_file(path)
  if not data then return {} end
  local set = {}

  for key in data:gmatch('opt_%w+%s*%(%s*"([%w%-%_]+)"') do
    add_option(set, key)
  end
  for key in data:gmatch('Args%.get_%w+%s*%([^\n]-"([%w%-%_]+)"') do
    add_option(set, key)
  end
  -- Convention des scripts qui centralisent les getters dans un helper :
  -- apply_cli(section, "field", "cli-name", Args.get_int)
  for key in data:gmatch('apply_cli%s*%([^\n]-"[%w%-%_]+"%s*,%s*"([%w%-%_]+)"') do
    add_option(set, key)
  end
  for key in data:gmatch('opts%s*%[%s*"([%w%-%_]+)"%s*%]') do
    add_option(set, key)
  end

  local out = {}
  for k, _ in pairs(set) do
    out[#out + 1] = "--" .. k
  end
  table.sort(out)
  return out
end

-- Keep terminal styling local to the renderer; no shell probe or runtime startup.
local function color_enabled(p)
  if os.getenv("NO_COLOR") ~= nil or os.getenv("TERM") == "dumb" then return false end
  if p.color ~= nil then return p.color end
  local force = os.getenv("FORCE_COLOR")
  if force ~= nil then return force ~= "0" end
  local term = os.getenv("TERM")
  return term ~= nil and term ~= ""
end

local function display_length(text)
  if utf8 and utf8.len then return utf8.len(text) or #text end
  return #text
end

local function wrap(text, width)
  local lines, line = {}, ""
  for word in tostring(text):gmatch("%S+") do
    if line ~= "" and display_length(line) + 1 + display_length(word) > width then
      lines[#lines + 1], line = line, ""
    end
    line = line == "" and word or (line .. " " .. word)
  end
  if line ~= "" then lines[#lines + 1] = line end
  return lines
end

-- Accept legacy string entries and explicit {flags=..., description=...} entries.
local function option_parts(entry)
  if type(entry) == "table" then return entry.flags, entry.description or "" end
  local text = trim(entry)
  if not text:match("^%-") and not text:match("^%[") then return nil, text end
  local flags, rest = {}, text
  while rest ~= "" do
    local token = rest:match("^(%-%S+)") or rest:match("^(<[^>]+>)")
      or rest:match("^(%[[^%]]+%])") or rest:match("^([/,])")
    if not token then break end
    flags[#flags + 1] = token
    rest = trim(rest:sub(#token + 1))
  end
  return table.concat(flags, " "), (rest:gsub("^:%s*", ""))
end

function M.print_help(params)
  local p = params or {}
  -- Mimir overrides print and strips ANSI; write directly to preserve styling.
  local print = p.write_line or function(line) io.stdout:write(line .. "\n") end
  local script_path = p.script_path or M.find_script_from_stack(3, 18) or "<script.lua>"
  local description = p.description or infer_description(script_path) or "Script Lua du projet Mimir."
  local options = type(p.options) == "table" and p.options or infer_options(script_path)
  local common = p.common_flags or {"--help, -h : affiche cette aide"}
  local width = math.max(60, math.min(120, tonumber(p.width or os.getenv("COLUMNS")) or 100))
  local column = math.min(38, math.floor(width * 0.4))
  local colored = color_enabled(p)
  local function style(code, text)
    return colored and ("\27[" .. code .. "m" .. text .. "\27[0m") or text
  end
  local function section(title)
    print("")
    print(style("1;36", title .. ":"))
  end
  local function paragraph(text, indent)
    indent = indent or "  "
    for _, line in ipairs(wrap(text, width - #indent)) do print(indent .. line) end
  end
  local notes = {}
  local function rows(entries)
    for _, entry in ipairs(entries) do
      local flags, detail = option_parts(entry)
      if not flags then
        notes[#notes + 1] = detail
      else
        local lines = wrap(detail, width - column - 2)
        local label = "  " .. style("32", flags)
        if display_length(flags) > column - 4 then
          print(label)
          for _, line in ipairs(lines) do print(string.rep(" ", column) .. line) end
        else
          print(label .. string.rep(" ", column - 2 - display_length(flags)) .. (lines[1] or ""))
          for i = 2, #lines do print(string.rep(" ", column) .. lines[i]) end
        end
      end
    end
  end

  print("")
  print(style("1;36", "MÍMIR  /  AIDE SCRIPT"))
  print(style("1", script_path:match("[^/]+$") or script_path))
  print(style("2", string.rep("─", width)))
  paragraph(description)
  section("Usage")
  print("  " .. style("33", "./bin/mimir --lua " .. script_path .. " -- [options]"))
  section(p.options and "Options" or "Options detectees")
  if #options == 0 then paragraph("Aucune option supplémentaire.") else rows(options) end
  section("Aide et options communes")
  rows(common)
  if #notes > 0 then
    section("Notes")
    for _, note in ipairs(notes) do paragraph(note) end
  end
  if type(p.examples) == "table" and #p.examples > 0 then
    section("Exemples")
    for i, example in ipairs(p.examples) do
      print("  " .. style("2", tostring(i) .. ".") .. " " .. style("33", example))
    end
  end
  print("")
  print(style("2", "  Couleurs : NO_COLOR=1 pour désactiver ; FORCE_COLOR=1 pour activer."))
  print("")
end

function M.auto_exit_help(params)
  if M.should_show_help(_G.arg) then
    M.print_help(params)
    os.exit(0)
  end
end

return M
