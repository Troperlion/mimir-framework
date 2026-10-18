#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")

local function logf(fmt, ...)
  local msg = string.format(fmt, ...)
  if type(log) == "function" then log(msg) else print(msg) end
end

local function die(msg)
  local text = "[hf_vae_decoder_infer] ❌ " .. tostring(msg)
  if type(log) == "function" then log(text) else print(text) end
  os.exit(1)
end

local function ok_or_die(ok, err, ctx)
  if ok then return end
  die((ctx or "operation") .. ": " .. tostring(err or "unknown"))
end

local opts = Args.parse(arg)
local checkpoint = Args.get_str(opts, "checkpoint", Args.get_str(opts, "ckpt", "../model.safetensors"))
local mapping_json = Args.get_str(opts, "mapping-json", "tools/vae_decoder_mapping.json")
local image_w = Args.get_int(opts, "image-w", 512)
local image_h = Args.get_int(opts, "image-h", 512)
local latent_w = Args.get_int(opts, "latent-w", math.floor(image_w / 8))
local latent_h = Args.get_int(opts, "latent-h", math.floor(image_h / 8))
local memory_gb = Args.get_num(opts, "memory-gb", 12.0)

local cfg, err_cfg = Mimir.Architectures.default_config("hf_vae_decoder")
if type(cfg) ~= "table" then die("default_config(hf_vae_decoder) a échoué: " .. tostring(err_cfg)) end
cfg.image_w = image_w
cfg.image_h = image_h
cfg.latent_w = latent_w
cfg.latent_h = latent_h
cfg.dtype = os.getenv("MIMIR_DTYPE") or "float16"

ok_or_die(Mimir.MemoryGuard.setLimit(memory_gb), nil, "MemoryGuard.setLimit")
ok_or_die(Mimir.Allocator.configure({ max_ram_gb = memory_gb, enable_compression = true, swap_strategy = "lru" }), nil, "Allocator.configure")
ok_or_die(Mimir.Model.create("hf_vae_decoder", cfg), nil, "Model.create(hf_vae_decoder)")
ok_or_die(Mimir.Model.allocate_params(), nil, "Model.allocate_params")
ok_or_die(Mimir.Serialization.load(checkpoint, "safetensors", {
  load_tokenizer = false,
  load_encoder = false,
  load_optimizer = false,
  strict_mode = true,
  mapping_json = mapping_json,
}), nil, "Serialization.load")

local latent = {}
for i = 1, latent_w * latent_h * (cfg.latent_c or 4) do latent[i] = 0.0 end
local output, fwd_err = Mimir.Model.forward(latent, false)
if type(output) ~= "table" then die(fwd_err or "Model.forward a échoué") end

local head = {}
for i = 1, math.min(#output, 8) do head[#head + 1] = string.format("%.6f", tonumber(output[i]) or 0.0) end
logf("[hf_vae_decoder_infer] latent=%d output=%d", #latent, #output)
logf("[hf_vae_decoder_infer] out_head=[%s]", table.concat(head, ", "))