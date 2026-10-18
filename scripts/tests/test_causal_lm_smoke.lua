local cfg = {
  vocab_size = 16,
  seq_len = 4,
  d_model = 8,
  num_layers = 1,
  num_heads = 2,
  num_kv_heads = 1,
  mlp_hidden = 16,
  padding_idx = 0,
}

local ok, err = Mimir.Model.create("causal_lm", cfg)
assert(ok, tostring(err))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

local logits, forward_err = Mimir.Model.forward({
  __input__ = {1, 2, 3, 4},
}, true)
assert(logits, tostring(forward_err))
assert(#logits == cfg.seq_len * cfg.vocab_size,
       string.format("unexpected logits size: %d", #logits))

local gradient = {}
for i = 1, #logits do gradient[i] = 0 end
gradient[1] = 1
local backward_ok, backward_err = Mimir.Model.backward(gradient)
assert(backward_ok, tostring(backward_err))
local step_ok, step_err = Mimir.Model.optimizer_step(1e-4, "adamw")
assert(step_ok, tostring(step_err))

print("PASS causal_lm native forward/backward/optimizer")
