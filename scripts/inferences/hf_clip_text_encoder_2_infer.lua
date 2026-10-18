#!/usr/bin/env lua
---@diagnostic disable: undefined-field, need-check-nil

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")

local function logf(fmt, ...)
  local msg = string.format(fmt, ...)
  if type(log) == "function" then log(msg) else print(msg) end
end

local function die(msg)
  local text = "[hf_clip_text_encoder_2_infer] ❌ " .. tostring(msg)
  if type(log) == "function" then log(text) else print(text) end
  os.exit(1)
end

local function ok_or_die(ok, err, ctx)
  if ok then return end
  die((ctx or "operation") .. ": " .. tostring(err or "unknown"))
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if not dtype then return end
  if Mimir and Mimir.model and type(Mimir.model.dtype) == "function" then
    local ok, err = Mimir.model.dtype(dtype)
    ok_or_die(ok, err, "Mimir.model.dtype(" .. tostring(dtype) .. ")")
  end
end

local function pad_or_trim(ids, seq_len, pad_id)
  local out = {}
  for i = 1, math.min(#ids, seq_len) do out[i] = ids[i] end
  for i = #out + 1, seq_len do out[i] = pad_id end
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
local checkpoint = Args.get_str(opts, "checkpoint", Args.get_str(opts, "ckpt", "../model.safetensors"))
local tokenizer_path = Args.get_str(opts, "tokenizer", "checkpoint/base_tokenizer/tokenizer.json")
local prompt = Args.get_str(opts, "prompt", "a pony in snowy forest")
local mapping_json = Args.get_str(opts, "mapping-json", "tools/text_encoder_2_mapping.json")
local seq_len = Args.get_int(opts, "seq-len", 77)
local memory_gb = Args.get_num(opts, "memory-gb", 12.0)

ok_or_die(Mimir.Tokenizer.load(tokenizer_path), nil, "Tokenizer.load")
local ids, tok_err = Mimir.Tokenizer.tokenize(prompt)
if type(ids) ~= "table" then die(tok_err or "Tokenizer.tokenize a échoué") end
local pad_id = Mimir.Tokenizer.pad_id() or 0
ids = pad_or_trim(ids, seq_len, pad_id)

local cfg, err_cfg = Mimir.Architectures.default_config("hf_clip_text_encoder_2")
if type(cfg) ~= "table" then die("default_config(hf_clip_text_encoder_2) a échoué: " .. tostring(err_cfg)) end
cfg.seq_len = seq_len
cfg.dtype = os.getenv("MIMIR_DTYPE") or "float16"

ok_or_die(Mimir.MemoryGuard.setLimit(memory_gb), nil, "MemoryGuard.setLimit")
ok_or_die(Mimir.Allocator.configure({ max_ram_gb = memory_gb, enable_compression = true, swap_strategy = "lru" }), nil, "Allocator.configure")
ok_or_die(Mimir.Model.create("hf_clip_text_encoder_2", cfg), nil, "Model.create(hf_clip_text_encoder_2)")
apply_dtype(cfg)
ok_or_die(Mimir.Model.allocate_params(), nil, "Model.allocate_params")
ok_or_die(Mimir.Serialization.load(checkpoint, "safetensors", {
  load_tokenizer = false,
  load_encoder = false,
  load_optimizer = false,
  strict_mode = true,
  mapping_json = mapping_json,
}), nil, "Serialization.load")

local output, fwd_err = Mimir.Model.forward(ids, false)
if type(output) ~= "table" then die(fwd_err or "Model.forward a échoué") end

local sequence_dim = tonumber(cfg.seq_len) * tonumber(cfg.d_model)
local pooled_dim = tonumber(cfg.proj_dim)
local logit_scale_dim = cfg.include_logit_scale and 1 or 0

logf("[hf_clip_text_encoder_2_infer] prompt=%s", tostring(prompt))
logf("[hf_clip_text_encoder_2_infer] ids=%d output=%d", #ids, #output)
logf("[hf_clip_text_encoder_2_infer] sequence_dim=%d pooled_dim=%d logit_scale_dim=%d", sequence_dim, pooled_dim, logit_scale_dim)
logf("[hf_clip_text_encoder_2_infer] seq_head=[%s]", dump_head(output, 8))

local pooled = {}
for i = sequence_dim + 1, math.min(sequence_dim + pooled_dim, sequence_dim + 8) do
  pooled[#pooled + 1] = output[i]
end
logf("[hf_clip_text_encoder_2_infer] pooled_head=[%s]", dump_head(pooled, 8))
if logit_scale_dim > 0 and #output >= sequence_dim + pooled_dim + 1 then
  logf("[hf_clip_text_encoder_2_infer] logit_scale=%.6f", tonumber(output[sequence_dim + pooled_dim + 1]) or 0.0)
end