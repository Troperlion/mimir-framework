-- Simple checkpoint resume helper (raw_folder + epoch_* layout)
-- Usage:
--   local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
--   local dir = Ckpt.resolve_dir("checkpoint/MyModel")

---@class MimirCheckpointResumeModule
local M = {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local function file_exists(path)
  return FS.file_exists(path)
end

local function find_latest_epoch_dir(base)
  base = tostring(base or "")
  if #base == 0 then return nil end

  local entries = FS.list_dir(base)
  local epochs = {}
  for _, name in ipairs(entries) do
    if tostring(name):match("^epoch_%d+") then
      local full = FS.join(base, name)
      if FS.is_dir(full) then
        epochs[#epochs + 1] = full
      end
    end
  end

  table.sort(epochs)
  if #epochs > 0 then
    return epochs[#epochs]
  end
  return nil
end

local function looks_like_raw_folder(dir)
  dir = tostring(dir or "")
  if #dir == 0 then return false end
    return file_exists(FS.join(dir, "model", "architecture.json"))
      or file_exists(FS.join(dir, "model", "model.safetensors"))
      or file_exists(FS.join(dir, "model.safetensors"))
end

function M.file_exists(path)
  return file_exists(path)
end

function M.find_latest_epoch_dir(base)
  return find_latest_epoch_dir(base)
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

  return nil
end

return M
