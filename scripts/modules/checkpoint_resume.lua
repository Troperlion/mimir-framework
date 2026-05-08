-- Simple checkpoint resume helper (raw_folder + epoch_* layout)
-- Usage:
--   local Ckpt = dofile("scripts/modules/checkpoint_resume.lua")
--   local dir = Ckpt.resolve_dir("checkpoint/MyModel")

local M = {}

local function file_exists(path)
  local f = io.open(path, "rb")
  if f then f:close(); return true end
  return false
end

local function shell_quote(s)
  s = tostring(s or "")
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function find_latest_epoch_dir(base)
  base = tostring(base or "")
  if #base == 0 then return nil end
  local q = shell_quote(base)
  local p = io.popen("ls -1d " .. q .. "/epoch_* 2>/dev/null | sort | tail -n 1")
  if not p then return nil end
  local line = p:read("*l")
  p:close()
  if line and #line > 0 then return line end
  return nil
end

local function looks_like_raw_folder(dir)
  dir = tostring(dir or "")
  if #dir == 0 then return false end
  return file_exists(dir .. "/model/architecture.json")
      or file_exists(dir .. "/model/model.safetensors")
      or file_exists(dir .. "/model.safetensors")
end

local function find_final_dir(base)
  base = tostring(base or "")
  if #base == 0 then return nil end
  local cand = base .. "/final"
  if looks_like_raw_folder(cand) then
    return cand
  end
  return nil
end

function M.file_exists(path)
  return file_exists(path)
end

function M.find_latest_epoch_dir(base)
  return find_latest_epoch_dir(base)
end

function M.find_final_dir(base)
  return find_final_dir(base)
end

-- Returns the best candidate directory to load, or nil.
function M.resolve_dir(base)
  base = tostring(base or "")
  if #base == 0 then return nil end

  -- Prefer the most recent epoch dir if present.
  local latest = find_latest_epoch_dir(base)
  if latest and looks_like_raw_folder(latest) then
    return latest
  end

  if looks_like_raw_folder(base) then
    return base
  end

  -- Fallback: some trainings save only base/final.
  local final = find_final_dir(base)
  if final and looks_like_raw_folder(final) then
    return final
  end

  return nil
end

-- Same as resolve_dir(), but prefers base/final over epoch_*.
function M.resolve_dir_prefer_final(base)
  base = tostring(base or "")
  if #base == 0 then return nil end

  local final = find_final_dir(base)
  if final and looks_like_raw_folder(final) then
    return final
  end

  local latest = find_latest_epoch_dir(base)
  if latest and looks_like_raw_folder(latest) then
    return latest
  end

  if looks_like_raw_folder(base) then
    return base
  end

  return nil
end

return M
