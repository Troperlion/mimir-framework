#!/usr/bin/env lua
-- Native Mimir next-token training for the "causal_lm" architecture.
-- Run with the configuration driver:
--   ./run_mimir.sh --conf configs/causal_lm.json
-- Or directly (loads configs/causal_lm.json):
--   ./run_mimir.sh --lua scripts/training/train_causal_lm.lua -- \
--     --corpus data/corpus.txt --tokenizer checkpoints/causal_lm/tokenizer.json
-- All CLI parsing and --override handling goes through scripts/modules/args.lua.

local DEFAULT_CONFIG_PATH = "configs/causal_lm.json"
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local cli_config_path =
    Args.get_str(opts, "conf", Args.get_str(opts, "config", DEFAULT_CONFIG_PATH))

if type(CONF) ~= "table"
    or type(CONF.model) ~= "table"
    or type(CONF.training) ~= "table"
    or type(CONF.dataset) ~= "table" then
  local loaded, load_err = read_json(cli_config_path)
  assert(type(loaded) == "table",
         "cannot load causal LM config '" .. cli_config_path
         .. "': " .. tostring(load_err))
  CONF = loaded
  print("[causal_lm] configuration: " .. cli_config_path)
end

assert(type(CONF.model) == "table"
       and type(CONF.training) == "table"
       and type(CONF.dataset) == "table",
       "causal LM config requires model, training and dataset sections")

CONF.tokenizer = type(CONF.tokenizer) == "table" and CONF.tokenizer or {}

local function apply_cli(tbl, field, arg_name, getter)
  if opts[arg_name] ~= nil then
    tbl[field] = getter(opts, arg_name, tbl[field])
  end
end

apply_cli(CONF.dataset, "text_path", "corpus", Args.get_str)
apply_cli(CONF.dataset, "text_path", "text-path", Args.get_str)
apply_cli(CONF.tokenizer, "path", "tokenizer", Args.get_str)
apply_cli(CONF.tokenizer, "bpe_merges", "bpe-merges", Args.get_int)

apply_cli(CONF.model, "vocab_size", "vocab-size", Args.get_int)
apply_cli(CONF.model, "seq_len", "seq-len", Args.get_int)
apply_cli(CONF.model, "d_model", "d-model", Args.get_int)
apply_cli(CONF.model, "num_layers", "layers", Args.get_int)
apply_cli(CONF.model, "num_heads", "heads", Args.get_int)
apply_cli(CONF.model, "num_kv_heads", "kv-heads", Args.get_int)
apply_cli(CONF.model, "mlp_hidden", "mlp-hidden", Args.get_int)
apply_cli(CONF.model, "dtype", "dtype", Args.get_str)

apply_cli(CONF.training, "steps", "steps", Args.get_int)
apply_cli(CONF.training, "learning_rate", "learning-rate", Args.get_num)
apply_cli(CONF.training, "learning_rate", "lr", Args.get_num)
apply_cli(CONF.training, "optimizer", "optimizer", Args.get_str)
apply_cli(CONF.training, "seed", "seed", Args.get_int)
apply_cli(CONF.training, "weight_decay", "weight-decay", Args.get_num)
apply_cli(CONF.training, "grad_clip_norm", "grad-clip-norm", Args.get_num)
apply_cli(CONF.training, "log_every", "log-every", Args.get_int)
apply_cli(CONF.training, "save_every", "save-every", Args.get_int)
apply_cli(CONF.training, "checkpoint_dir", "checkpoint-dir", Args.get_str)
apply_cli(CONF.training, "amp", "amp", Args.get_bool)
apply_cli(CONF.training, "ddp", "ddp", Args.get_bool)

-- Les overrides structurés ont la priorité sur les raccourcis CLI.
CONF = Args.apply_overrides(CONF, opts)

local function require_readable_file(path, label)
  assert(type(path) == "string" and path ~= "",
         label .. " path is missing from the causal LM configuration")
  local file, open_err = io.open(path, "rb")
  assert(file,
         string.format("%s not found or unreadable: %s (%s). "
                       .. "Update '%s' or pass another --conf file.",
                       label, path, tostring(open_err), cli_config_path))
  return file
end

local model_cfg = CONF.model
local train_cfg = CONF.training
local tokenizer_cfg = CONF.tokenizer or {}
assert(train_cfg.amp ~= true,
       "AMP requested but Mimir has no mixed-precision compute/autocast backend yet")
assert(train_cfg.ddp ~= true,
       "DDP requested but Mimir has no collective communication backend yet")
local seq_len = assert(tonumber(model_cfg.seq_len))
local vocab_size = assert(tonumber(model_cfg.vocab_size))
local architecture = model_cfg.architecture or "causal_lm"

math.randomseed(tonumber(train_cfg.seed) or 1337)

local corpus_file = require_readable_file(CONF.dataset.text_path, "training corpus")
local corpus = corpus_file:read("*a")
corpus_file:close()

local tokenizer_path = tokenizer_cfg.path
if type(tokenizer_path) ~= "string" or tokenizer_path == "" then
  tokenizer_path = (train_cfg.checkpoint_dir or "checkpoints/causal_lm") .. "/tokenizer.json"
  tokenizer_cfg.path = tokenizer_path
  CONF.tokenizer = tokenizer_cfg
end

local CausalTokenizer = dofile(ROOTWORK.."/scripts/modules/causal_lm_tokenizer.lua")
local tokens, tokenizer_status = CausalTokenizer.ensure({
  corpus = corpus,
  path = tokenizer_path,
  vocab_size = vocab_size,
  padding_idx = tonumber(model_cfg.padding_idx) or 0,
  seq_len = seq_len,
  bpe_merges = tokenizer_cfg.bpe_merges,
})
if tokenizer_status.generated then
  print(string.format(
    "[causal_lm] tokenizer généré: path=%s vocab=%d/%d reason=%s",
    tostring(tokenizer_status.path),
    tonumber(tokenizer_status.vocab_size) or 0,
    vocab_size,
    tostring(tokenizer_status.reason)))
else
  print(string.format(
    "[causal_lm] tokenizer chargé: path=%s vocab=%d/%d",
    tostring(tokenizer_status.path),
    tonumber(tokenizer_status.vocab_size) or 0,
    vocab_size))
end
assert(#tokens > seq_len, "corpus must contain more than seq_len tokens")

local cfg, cfg_err = Mimir.Architectures.default_config(architecture)
assert(cfg, tostring(cfg_err))
for key, value in pairs(model_cfg) do
  if key ~= "architecture" then cfg[key] = value end
end
for _, key in ipairs({"beta1", "beta2", "epsilon", "weight_decay", "grad_clip_norm"}) do
  if train_cfg[key] ~= nil then cfg[key] = train_cfg[key] end
end
assert(Mimir.Model.create(architecture, cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", tonumber(train_cfg.seed) or 1337))

local function sample()
  local first = math.random(1, #tokens - seq_len)
  local input, target = {}, {}
  for i = 1, seq_len do
    input[i] = tokens[first + i - 1]
    target[i] = tokens[first + i]
  end
  return input, target
end

-- Mean cross-entropy and its exact gradient with respect to flattened logits.
local function cross_entropy(logits, target)
  assert(#logits == seq_len * vocab_size,
         string.format("expected %d logits, got %d", seq_len * vocab_size, #logits))
  local grad, loss = {}, 0.0
  for position = 1, seq_len do
    local offset = (position - 1) * vocab_size
    local maximum = -math.huge
    for token = 1, vocab_size do
      maximum = math.max(maximum, logits[offset + token])
    end
    local denominator = 0.0
    for token = 1, vocab_size do
      denominator = denominator + math.exp(logits[offset + token] - maximum)
    end
    local gold = target[position] + 1 -- Mimir token ids are zero-based.
    assert(gold >= 1 and gold <= vocab_size, "target token outside vocabulary")
    loss = loss + (math.log(denominator) + maximum - logits[offset + gold])
    for token = 1, vocab_size do
      local probability = math.exp(logits[offset + token] - maximum) / denominator
      grad[offset + token] = (probability - (token == gold and 1.0 or 0.0)) / seq_len
    end
  end
  return loss / seq_len, grad
end

local steps = tonumber(train_cfg.steps) or 1000
local learning_rate = tonumber(train_cfg.learning_rate) or 3e-4
local optimizer = train_cfg.optimizer or "adamw"
local log_every = tonumber(train_cfg.log_every) or 10
local save_every = tonumber(train_cfg.save_every) or 500
local checkpoint_dir = train_cfg.checkpoint_dir or "checkpoints/causal_lm"

for step = 1, steps do
  local input, target = sample()
  assert(Mimir.Model.zero_grads())
  local logits, forward_err =
      Mimir.Model.forward({__input__ = input}, true)
  assert(logits, "forward failed: " .. tostring(forward_err))
  local loss, gradient = cross_entropy(logits, target)
  local backward_ok, backward_err = Mimir.Model.backward(gradient)
  assert(backward_ok, "backward failed: " .. tostring(backward_err))
  local step_ok, step_err = Mimir.Model.optimizer_step(learning_rate, optimizer)
  assert(step_ok, "optimizer step failed: " .. tostring(step_err))

  if step == 1 or step % log_every == 0 then
    print(string.format("step=%d loss=%.6f perplexity=%.3f",
                        step, loss, math.exp(math.min(loss, 20))))
  end
  if step % save_every == 0 then
    local path = string.format("%s/step-%07d", checkpoint_dir, step)
    local save_ok, save_err = Mimir.Serialization.save(path, "raw_folder", {
      save_optimizer = true,
      save_tokenizer = true
    })
    assert(save_ok, "checkpoint save failed: " .. tostring(save_err))
  end
end
