---@diagnostic disable: undefined-global, undefined-field

-- Inspection structurelle de VAEConv sans dataset.
-- Usage:
--   ./bin/mimir --lua scripts/examples/inspect_vae_conv.lua

local cfg, err = Mimir.Architectures.default_config("vae_conv")
assert(type(cfg) == "table", err)

-- Petite configuration pour garder l'exemple rapide.
cfg.image_w = 8
cfg.image_h = 8
cfg.image_c = 1
cfg.latent_w = 2
cfg.latent_h = 2
cfg.latent_c = 4
cfg.base_channels = 8
cfg.stochastic_latent = true
cfg.resnet = true
cfg.resnet_max_tokens = 4
cfg.attention = false
cfg.enc_norm = "groupnorm"
cfg.dec_norm = "groupnorm"
cfg.enc_gn_groups = 4
cfg.dec_gn_groups = 4
cfg.use_encoder_prior = true

local ok_create, create_err = Mimir.Model.create("vae_conv", cfg)
assert(ok_create, create_err)

local ok_alloc, params_or_err = Mimir.Model.allocate_params()
assert(ok_alloc, params_or_err)

local ok_init, init_err = Mimir.Model.init_weights("xavier", 1234)
assert(ok_init, init_err)

local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local latent_dim = cfg.latent_w * cfg.latent_h * cfg.latent_c
local expected_output_dim = image_dim + 2 * latent_dim

print(string.format(
  "VAEConv: image_dim=%d latent_dim=%d output_dim=%d params=%s",
  image_dim,
  latent_dim,
  expected_output_dim,
  tostring(Mimir.Model.total_params())
))

local found_prior = false
for _, layer in ipairs(Mimir.Model.get_layers()) do
  print(string.format(
    "%3d  %-42s %-20s params=%d",
    tonumber(layer.index) or -1,
    tostring(layer.name),
    tostring(layer.type),
    tonumber(layer.param_count) or 0
  ))
  if layer.name == "vae_conv/z_prior_bias" then
    found_prior = true
  end
end

assert(found_prior, "vae_conv/z_prior_bias absent malgré use_encoder_prior=true")
print("Inspection VAEConv réussie (aucun dataset chargé).")
