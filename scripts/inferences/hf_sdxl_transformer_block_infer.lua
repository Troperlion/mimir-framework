#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")

local function logf(fmt, ...)
  local msg = string.format(fmt, ...)
  if type(log) == "function" then log(msg) else print(msg) end
end

local function die(msg)
  local text = "[hf_sdxl_transformer_block_infer] ❌ " .. tostring(msg)
  if type(log) == "function" then log(text) else print(text) end
  os.exit(1)
end

local function ok_or_die(ok, err, ctx)
  if ok then return end
  die((ctx or "operation") .. ": " .. tostring(err or "unknown"))
end

local function zeros(n)
  local out = {}
  for i = 1, n do out[i] = 0.0 end
  return out
end

local function dump_head(vec, n)
  local out = {}
  for i = 1, math.min(#vec, n) do
    out[#out + 1] = string.format("%.6f", tonumber(vec[i]) or 0.0)
  end
  return table.concat(out, ", ")
end

local opts = Args.parse(arg)
local checkpoint = Args.get_str(opts, "checkpoint", "../model.safetensors")
local mapping_json = Args.get_str(opts, "mapping-json", "tools/unet_transformer_block_mapping.json")
local q_len = Args.get_int(opts, "q-len", 64)
local kv_len = Args.get_int(opts, "kv-len", 77)
local d_model = Args.get_int(opts, "d-model", 640)
local context_dim = Args.get_int(opts, "context-dim", 2048)
local num_heads = Args.get_int(opts, "num-heads", 10)
local ff_hidden = Args.get_int(opts, "ff-hidden", 2560)

local cfg, err_cfg = Mimir.Architectures.default_config("hf_sdxl_transformer_block")
if type(cfg) ~= "table" then die("default_config(hf_sdxl_transformer_block) a échoué: " .. tostring(err_cfg)) end
cfg.q_len = q_len
cfg.kv_len = kv_len
cfg.d_model = d_model
cfg.context_dim = context_dim
cfg.num_heads = num_heads
cfg.ff_hidden = ff_hidden
cfg.dtype = os.getenv("MIMIR_DTYPE") or "float16"

ok_or_die(Mimir.Model.create("hf_sdxl_transformer_block", cfg), nil, "Model.create(hf_sdxl_transformer_block)")
ok_or_die(Mimir.Model.allocate_params(), nil, "Model.allocate_params")
ok_or_die(Mimir.Serialization.load(checkpoint, "safetensors", {
  load_tokenizer = false,
  load_encoder = false,
  load_optimizer = false,
  strict_mode = true,
  mapping_json = mapping_json,
}), nil, "Serialization.load")

local input = zeros(q_len * d_model + kv_len * context_dim)
local output, fwd_err = Mimir.Model.forward(input, false)
if type(output) ~= "table" then die(fwd_err or "Model.forward a échoué") end

logf("[hf_sdxl_transformer_block_infer] input=%d output=%d", #input, #output)
logf("[hf_sdxl_transformer_block_infer] out_head=[%s]", dump_head(output, 8))