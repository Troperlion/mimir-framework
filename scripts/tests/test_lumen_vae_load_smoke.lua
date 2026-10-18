#!/usr/bin/env lua
---@diagnostic disable: undefined-field

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local checkpoint = Args.get_str(opts, "vae-checkpoint", "")
assert(checkpoint ~= "", "--vae-checkpoint est requis")
local output = Args.get_str(opts, "out", "")

local cfg, err = Mimir.Architectures.default_config("lumen_diffusion")
assert(type(cfg) == "table", tostring(err or "lumen_diffusion indisponible"))

cfg.image_w = 512
cfg.image_h = 512
cfg.image_c = 3
cfg.latent_w = Args.get_int(opts, "latent-w", cfg.latent_w)
cfg.latent_h = Args.get_int(opts, "latent-h", cfg.latent_h)
cfg.latent_c = Args.get_int(opts, "latent-c", cfg.latent_c)
cfg.vae_checkpoint = checkpoint
cfg.vae_decoder_upsample = Args.get_str(opts, "vae-decoder-upsample",
	cfg.vae_decoder_upsample)

local ok, create_err = Mimir.Model.create("lumen_diffusion", cfg)
assert(ok, tostring(create_err or "Model.create a échoué"))
if output ~= "" then
	local saved, save_err = Mimir.Serialization.save(output, "debug_json", {
		save_tokenizer = false,
		save_encoder = false,
	})
	assert(saved, tostring(save_err or "Serialization.save a échoué"))
end
print("[test_lumen_vae_load_smoke] OK: " .. checkpoint)