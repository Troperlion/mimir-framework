---@diagnostic disable: undefined-global

local expected = {
  r_cnn = "vgg16",
  yolo = "mobilenet",
  ssd = "vgg16",
  deeplab = "resnet",
}

local available = {}
for _, name in ipairs(Mimir.Architectures.available() or {}) do
  available[name] = true
end

for name, delegate in pairs(expected) do
  assert(available[name], "MPK architecture missing from registry: " .. name)

  local info, info_err = Mimir.Architectures.info(name)
  assert(info, info_err)
  assert(info.config.mpk_prototype == true, name .. ": prototype marker missing")
  assert(
    info.config.mpk_execution_delegate == delegate,
    name .. ": wrong execution delegate"
  )

  local ok, create_err = Mimir.Model.create(name, info.config)
  assert(ok, name .. ": registry creation failed: " .. tostring(create_err))
end

io.stdout:write("[test_mpk_vision_architectures] OK\n")
