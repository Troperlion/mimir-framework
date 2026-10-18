#!/usr/bin/env mimir --lua
-- Construction, forward causal/non causal, tokenizer et roundtrip natifs.
-- Options: --quick, --full, --iters N, --warmup N, --ram GiB, --dtype TYPE.
local Help = dofile(ROOTWORK..'/scripts/modules/help_cli.lua')
Help.auto_exit_help()
local B = dofile(ROOTWORK..'/scripts/benchmarks/common.lua')
B.setup()
local mode = os.getenv('MIMIR_BENCH_MODE') or 'standard'
if B.has('--quick') then mode='quick' end
if B.has('--full') then mode='full' end
assert(mode=='standard' or mode=='quick' or mode=='full', 'mode inconnu: '..mode)
local quick = mode=='quick'
local seq, vocab = quick and 8 or 32, quick and 128 or 2048
B.measure('Tokenizer.create', function() assert(Mimir.Tokenizer.create(vocab)) end)
local dims = quick and {32,64} or (mode=='full' and {64,128,256} or {64,128})
for _, dim in ipairs(dims) do
  for _, causal in ipairs({false,true}) do
    log('Transformer causal=' .. tostring(causal))
    B.create('transformer', B.transformer(dim, 2, seq, vocab, causal))
    B.bench_forward(B.transformer_input(seq,vocab,dim), dim)
    B.snapshot('transformer')
  end
end
B.roundtrip({input_dim=16,hidden_dim=32,output_dim=8,hidden_layers=1}, B.floats(16))
log('PASS benchmark général')
B.report('general')
