#!/usr/bin/env mimir --lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- Benchmark minimal: mesure le coût du forward d'un modèle avec attention.
-- Usage:
--   MIMIR_ACCEL_VERBOSE=1 ./bin/mimir --lua scripts/benchmarks/benchmark_attention.lua

local B = dofile(ROOTWORK.."/scripts/benchmarks/common.lua")
B.setup()
math.randomseed(7)

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

local function bench(name, cfg, input, expected)
  B.create(name, cfg)
  B.bench_forward(input, expected)
end

-- 1) VAEConv attention 2D: tokens = H*W, embed_dim = C
-- Choix petit pour rester rapide.
-- En inférence, forward expose la reconstruction seule ; mu/logvar restent
-- des tenseurs internes et ne sont empaquetés qu'en entraînement.
bench("vae_conv", {
  image_w = 32,
  image_h = 32,
  image_c = 3,
  base_channels = 32,
  latent_w = 8,
  latent_h = 8,
  latent_c = 4,
  stochastic_latent = false,
  text_cond = false,
  attention = true,
  attn_heads = 4,
  attn_max_tokens = 256,
}, rand_floats(32 * 32 * 3), 32*32*3)

-- 2) Transformer sur tokens entiers: tokens=seq_len, d_model=embed_dim
-- Garde config petite pour éviter O(seq^2).
bench("transformer", {
  vocab_size = 2048,
  seq_len = 64,
  d_model = 128,
  num_heads = 4,
  num_layers = 2,
  mlp_hidden = 512,
  output_dim = 128,
  dropout = 0.0,
  causal = true,
}, {__input__=rand_ids(64, 2048), mag=B.floats(64*128), mod=B.floats(64*128)}, 128)

log("\n✓ done")
B.report('attention')
