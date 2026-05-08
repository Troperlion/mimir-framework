---@diagnostic disable: undefined-global, undefined-field

Mimir.Allocator.configure({
  enable_compression = true,
  swap_strategy = "lru",
  max_ram_gb = 10,
  offload_threshold_mb = 1000
})

local name, err = Mimir.Architectures.available()

if not name then
  log("[ERROR] " .. tostring(err))
  return
end


log("\n\n* Liste des architectures supporté par le Framework actuellement : ")


for _, arch in pairs(name) do
  local conf_arch = Mimir.Architectures.default_config(arch)
  if conf_arch then
    log("\t → " .. arch)
  else
    log("\t → " .. arch .. " [default_config ERROR]")
  end
end

-- Test: création depuis une conf complète injectée par --conf
if CONF then
  local ok, arch = Mimir.Model.create_from_config(CONF)
  if ok then
    log("\ncreate_from_config(CONF): OK (arch=" .. tostring(arch) .. ")")
  else
    log("\ncreate_from_config(CONF): ERROR (" .. tostring(arch) .. ")")
  end
end

local cfg = Mimir.Architectures.default_config("vae_conv")


local m_ok, m_err = Mimir.Model.create("vae_conv", cfg)

if not m_ok then

  log("[ERROR] " .. tostring(m_err))

end