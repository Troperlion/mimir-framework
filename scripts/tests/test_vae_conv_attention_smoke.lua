local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

---@diagnostic disable: undefined-global, undefined-field, inject-field

-- Smoke-test: VAEConv avec attention (sans mémoire encodeur).
-- Objectif: valider que Model.create + allocate_params + forward passent
-- quand attention=true.
--
-- Usage:
--   ./bin/mimir --lua scripts/tests/test_vae_conv_attention_smoke.lua

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({
    enable_compression = true,
    swap_strategy = "lru",
    max_ram_gb = 6,
    offload_threshold_mb = 1000,
  })
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

local cfg0, err = Mimir.Architectures.default_config("vae_conv")
assert(type(cfg0) == "table", "default_config(vae_conv) failed: " .. tostring(err))
---@cast cfg0 VAEConvConfig
local cfg = cfg0

-- Petite config fixe et rapide
cfg.image_w = 64
cfg.image_h = 64
cfg.image_c = 3
cfg.latent_h = 16
cfg.latent_w = 16
cfg.latent_c = 256
cfg.base_channels = 64
cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c
cfg.text_cond = false
cfg.stochastic_latent = false

-- Activer attention uniquement au bottleneck (16*16=256 tokens)
cfg.resnet = false
cfg.resnet_max_tokens = 0
cfg.attention = true
cfg.attn_heads = 4
cfg.attn_max_tokens = 256

local ok_create, err_create = Mimir.Model.create("vae_conv", cfg)
assert(ok_create == true, "Model.create failed: " .. tostring(err_create))
apply_dtype(cfg)

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
assert(ok_alloc == true, "Model.allocate_params failed: " .. tostring(err_alloc))

local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local x = {}
x[image_dim] = 0.0
for i = 1, image_dim do x[i] = 0.0 end

local packed, err_fwd = Mimir.Model.forward(x, false)
assert(type(packed) == "table", "Model.forward failed: " .. tostring(err_fwd))

local expected = image_dim + 2 * cfg.latent_dim
assert(#packed == expected, string.format("unexpected output size: got=%d expected=%d", #packed, expected))

log(string.format("[vae_conv_attention_smoke] OK output=%d (expected=%d)", #packed, expected))
