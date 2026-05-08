#!/usr/bin/env mimir --lua
-- Benchmark minimal: mesure le coût du forward d'un modèle avec attention.
-- Usage:
--   MIMIR_ACCEL_VERBOSE=1 ./bin/mimir --lua scripts/benchmarks/benchmark_attention.lua

math.randomseed(7)

local model = Mimir.Model

if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, 6.0)
end
if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  pcall(Mimir.Allocator.configure, { max_ram_gb = 6.0, enable_compression = true, swap_strategy = "lru" })
end

if model and model.set_hardware then
  pcall(model.set_hardware, true)
end

local function rand_floats(n)
  local x = {}
  for i = 1, n do x[i] = (math.random() * 2.0 - 1.0) end
  return x
end

local function rand_ids(n, vocab_size)
  local ids = {}
  local vmax = math.max(1, (vocab_size or 1)) - 1
  for i = 1, n do
    -- Forcer un entier Lua (le binding choisit la voie "tokens int" si tout est integer)
    ids[i] = math.random(0, vmax)
  end
  return ids
end

local function bench(name, cfg, input, iters)
  iters = iters or 30
  log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
  log("BENCH: " .. name)
  log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

  local ok, err = model.create(name, cfg)
  assert(ok ~= false, tostring(err or "Model.create failed"))
  assert(model.allocate_params())
  assert(model.init_weights("xavier", 7))

  local x = input

  -- warmup
  model.forward(x, false)

  local t0 = os.clock()
  for i = 1, iters do
    model.forward(x, false)
  end
  local t1 = os.clock()
  local dt = (t1 - t0)
  log(string.format("iters=%d | total=%.3fs | per_iter=%.3fms", iters, dt, (dt * 1000.0) / iters))
end

-- 1) VAEConv attention 2D: tokens = H*W, embed_dim = C
-- Choix petit pour rester rapide.
bench("vae_conv", {
  image_w = 32,
  image_h = 32,
  image_c = 3,
  base_channels = 32,
  latent_channels = 32,
  use_attn = true,
  attn_heads = 4,
  attn_max_tokens = 256,
}, rand_floats(32 * 32 * 3), 20)

-- 2) Transformer NLP-like (si dispo): tokens=seq_len, d_model=embed_dim
-- Garde config petite pour éviter O(seq^2).
bench("transformer", {
  vocab_size = 2048,
  seq_len = 64,
  d_model = 128,
  num_heads = 4,
  num_layers = 2,
  ff_mult = 4,
  dropout = 0.0,
  causal = true,
}, rand_ids(64, 2048), 20)

log("\n✓ done")
