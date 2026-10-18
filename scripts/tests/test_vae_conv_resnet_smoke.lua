local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- Smoke test: VAEConv avec blocs ResNet à la place de l'attention.
--
-- Usage:
--   ./bin/mimir --lua scripts/tests/test_vae_conv_resnet_smoke.lua

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

-- Config minimale, stable et rapide
cfg.image_w = cfg.image_w or 64
cfg.image_h = cfg.image_h or 64
cfg.image_c = cfg.image_c or 3
cfg.latent_h = cfg.latent_h or math.max(1, math.floor(cfg.image_h / 4))
cfg.latent_w = cfg.latent_w or math.max(1, math.floor(cfg.image_w / 4))
cfg.latent_c = cfg.latent_c or 256
cfg.base_channels = cfg.base_channels or 32

cfg.resnet = true
-- Désactive SelfAttention: on veut tester ResNet seul.
cfg.attention = false
-- Gate assez haut pour injecter un bloc au latent (16x16=256)
-- + un bloc sur la première upsample (32x32=1024) et skip au-delà.
cfg.resnet_max_tokens = 1024
cfg.attn_max_tokens = 0
cfg.attn_heads = 2

cfg.text_cond = false
cfg.stochastic_latent = false

cfg.latent_dim = (cfg.latent_h or 1) * (cfg.latent_w or 1) * (cfg.latent_c or 1)

local image_dim = (cfg.image_w or 1) * (cfg.image_h or 1) * (cfg.image_c or 1)
local latent_dim = tonumber(cfg.latent_dim or 0) or 0
assert(latent_dim > 0, "latent_dim must be > 0")

logx(string.format("[test_vae_conv_resnet_smoke] image=%dx%dx%d latent=%dx%dx%d base=%d", cfg.image_w, cfg.image_h, cfg.image_c, cfg.latent_h, cfg.latent_w, cfg.latent_c, cfg.base_channels))
logx(string.format("[test_vae_conv_resnet_smoke] resnet=%s attn=%s heads=%d max_tokens=%d",
  tostring(cfg.resnet),
  tostring(cfg.attention),
  tonumber(cfg.attn_heads or 0) or 0,
  tonumber(cfg.resnet_max_tokens or 0) or 0))

assert(Mimir.Model.create("vae_conv", cfg))
apply_dtype(cfg)
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("he", 123))

-- Entrée factice dans [-1,1]
local x = {}
x[image_dim] = 0.0
for i = 1, image_dim do
  x[i] = 0.0
end

local out, fwd_err = Mimir.Model.forward(x, false)
assert(out ~= nil, "Model.forward failed: " .. tostring(fwd_err))

local expected = image_dim + 2 * latent_dim
assert(#out == expected,
  string.format("unexpected output size: got=%d expected=%d (image_dim=%d latent_dim=%d)", #out, expected, image_dim, latent_dim))

logx("[test_vae_conv_resnet_smoke] OK")

-- Decoder-only: input=z, output=recon
assert(Mimir.Model.create("vae_conv_decode", cfg))
apply_dtype(cfg)
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("he", 123))

local z = {}
z[latent_dim] = 0.0
for i = 1, latent_dim do
  z[i] = 0.0
end

local recon, dec_err = Mimir.Model.forward(z, false)
assert(recon ~= nil, "Model.forward (decode) failed: " .. tostring(dec_err))
assert(#recon == image_dim,
  string.format("unexpected decode output size: got=%d expected=%d", #recon, image_dim))

logx("[test_vae_conv_resnet_smoke] OK (decode)")
