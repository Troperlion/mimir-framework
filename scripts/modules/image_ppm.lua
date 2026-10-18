-- PPM RGB helpers. Model values use interleaved RGB in [-1, 1].
local function clamp(v, lo, hi) return math.max(lo, math.min(hi, v)) end
local function read_ppm(path)
  local file, open_err = io.open(path, "rb")
  if not file then return nil, open_err end
  local function skip_space_and_comments()
    while true do
      local position = file:seek()
      local char = file:read(1)
      if char == nil then return end
      if char == "#" then file:read("*l")
      elseif char:match("%s") == nil then file:seek("set", position); return end
    end
  end
  local function token()
    skip_space_and_comments()
    local chars = {}
    while true do
      local position = file:seek()
      local char = file:read(1)
      if char == nil then break end
      if char == "#" or char:match("%s") then file:seek("set", position); break end
      chars[#chars + 1] = char
    end
    return #chars > 0 and table.concat(chars) or nil
  end
  local magic = token()
  local width, height, max_value = tonumber(token()), tonumber(token()), tonumber(token())
  if (magic ~= "P6" and magic ~= "P3") or not width or not height or not max_value then
    file:close(); return nil, "PPM invalide (P6/P3 attendu)"
  end
  if width < 1 or height < 1 or width ~= math.floor(width) or height ~= math.floor(height) then
    file:close(); return nil, "dimensions PPM invalides"
  end
  if max_value < 1 or max_value > 255 then file:close(); return nil, "maxval PPM non supporté" end
  local count = width * height * 3
  local pixels = {}; pixels[count] = 0
  if magic == "P6" then
    -- Un seul séparateur termine l'en-tête : les pixels peuvent être 10, 32 ou 35.
    local separator = file:read(1)
    if not separator or not separator:match("%s") then file:close(); return nil, "séparateur P6 absent" end
    if separator == "\r" then
      local position = file:seek()
      if file:read(1) ~= "\n" then file:seek("set", position) end
    end
    local bytes = file:read(count); file:close()
    if bytes == nil or #bytes ~= count then return nil, "pixels P6 tronqués" end
    for index = 1, count do pixels[index] = math.floor((bytes:byte(index) or 0) * 255 / max_value + 0.5) end
  else
    for index = 1, count do
      local value = tonumber(token())
      if value == nil then file:close(); return nil, "pixels P3 tronqués" end
      pixels[index] = math.floor(clamp(value, 0, max_value) * 255 / max_value + 0.5)
    end
    file:close()
  end
  return { width = width, height = height, pixels = pixels }
end

local function resize_rgb_nearest(source, source_w, source_h, target_w, target_h)
  local output = {}; output[target_w * target_h * 3] = 0
  for y = 0, target_h - 1 do
    local sy = math.min(source_h - 1, math.floor((y + 0.5) * source_h / target_h))
    for x = 0, target_w - 1 do
      local sx = math.min(source_w - 1, math.floor((x + 0.5) * source_w / target_w))
      local si, di = (sy * source_w + sx) * 3, (y * target_w + x) * 3
      for channel = 1, 3 do output[di + channel] = source[si + channel] end
    end
  end
  return output
end

local function write_ppm(path, pixels, width, height)
  local expected = width * height * 3
  if type(pixels) ~= "table" or #pixels ~= expected then return false, "buffer image de taille invalide" end
  local bytes = {}; bytes[expected] = ""
  for index = 1, expected do
    local value = tonumber(pixels[index]) or 0.0
    bytes[index] = string.char(math.floor(clamp(0.5 + 0.5 * value, 0, 1) * 255 + 0.5))
  end
  local file, open_err = io.open(path, "wb")
  if not file then return false, open_err end
  local ok, write_err = pcall(function()
    assert(file:write(string.format("P6\n%d %d\n255\n", width, height))); assert(file:write(table.concat(bytes)))
  end)
  local closed, close_err = file:close()
  return ok and closed, write_err or close_err
end

return {read=read_ppm, resize=resize_rgb_nearest, write=write_ppm}
