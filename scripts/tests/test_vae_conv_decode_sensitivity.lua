local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- Test: le décodeur VAEConv doit dépendre du latent.
--
-- Usage:
--   ./bin/mimir --lua scripts/tests/test_vae_conv_decode_sensitivity.lua

local function logx(msg)
  local l = rawget(_G, "log")
  if type(l) == "function" then l(msg) else print(msg) end
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  assert(ok ~= false, "dtype invalide: " .. tostring(dt_or_err))
  return true
end

local cfg, err = Mimir.Architectures.default_config("vae_conv")
assert(type(cfg) == "table", "default_config(vae_conv) failed: " .. tostring(err))
---@cast cfg table<string, any>

-- Petit pour être rapide
cfg.image_w = 32
cfg.image_h = 32
cfg.image_c = 3
cfg.latent_h = 8
cfg.latent_w = 8
cfg.latent_c = 64
cfg.base_channels = 32

cfg.resnet = true
cfg.attention = false
cfg.resnet_max_tokens = cfg.latent_h * cfg.latent_w
cfg.attn_max_tokens = 0
cfg.attn_heads = 2

cfg.text_cond = false
cfg.stochastic_latent = false
cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c

local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local latent_dim = cfg.latent_dim

logx(string.format("[test_vae_conv_decode_sensitivity] image_dim=%d latent_dim=%d", image_dim, latent_dim))

assert(Mimir.Model.create("vae_conv_decode", cfg))
apply_dtype(cfg)
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("he", 123))

local function make_vec(n, v)
  local t = {}
  t[n] = 0.0
  for i = 1, n do t[i] = v end
  return t
end

local function make_latent_impulse(n, idx, amp)
  local t = make_vec(n, 0.0)
  idx = math.max(1, math.min(n, idx))
  t[idx] = amp
  return t
end

local z0 = make_vec(latent_dim, 0.0)
local z1 = make_vec(latent_dim, 0.1)
local z2 = make_latent_impulse(latent_dim, math.floor(latent_dim / 2), 1.0)

local y0, e0 = Mimir.Model.forward(z0, false)
assert(y0 ~= nil, "forward(z0) failed: " .. tostring(e0))
assert(#y0 == image_dim, string.format("unexpected output size: got=%d expected=%d", #y0, image_dim))

-- Compat: accepte aussi {__input__=...}
local y0c, e0c = Mimir.Model.forward({ __input__ = z0 }, false)
assert(y0c ~= nil, "forward({__input__=z0}) failed: " .. tostring(e0c))
assert(#y0c == image_dim)

local y1, e1 = Mimir.Model.forward(z1, false)
assert(y1 ~= nil, "forward(z1) failed: " .. tostring(e1))
assert(#y1 == image_dim)

local y2, e2 = Mimir.Model.forward(z2, false)
assert(y2 ~= nil, "forward(z2) failed: " .. tostring(e2))
assert(#y2 == image_dim)

local function l1diff(a, b)
  local n = math.min(#a, #b)
  local s = 0.0
  for i = 1, n do
    local d = a[i] - b[i]
    if d < 0 then d = -d end
    s = s + d
  end
  return s
end

local d01 = l1diff(y0, y1)
local d02 = l1diff(y0, y2)
local d0c = l1diff(y0, y0c)
logx(string.format("[test_vae_conv_decode_sensitivity] L1(y0,y1)=%.6g L1(y0,y2)=%.6g", d01, d02))
logx(string.format("[test_vae_conv_decode_sensitivity] L1(y0, compat)=%.6g", d0c))

-- Seuil volontairement très bas: on veut juste détecter un décodeur qui ignore z.
assert(d01 > 1e-3, "decoder output seems insensitive to latent (z0 vs z1)")
assert(d02 > 1e-3, "decoder output seems insensitive to latent (z0 vs z2)")
assert(d0c == 0, "forward compat {__input__=} differs from array input")

logx("[test_vae_conv_decode_sensitivity] OK")
