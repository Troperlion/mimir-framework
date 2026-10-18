#!/usr/bin/env mimir --lua
-- Mesure create (graphe inclus), allocate et init; aucun second build().
-- Options: --safe (défaut), --quick, --full, --extreme, --iters N,
-- --seq N, --vocab N, --ram GiB, --dtype TYPE, --no-compress, --no-accel.
local Help = dofile(ROOTWORK..'/scripts/modules/help_cli.lua')
Help.auto_exit_help()
local B = dofile(ROOTWORK..'/scripts/benchmarks/common.lua')
B.setup()
local mode = B.value('--mode','MIMIR_BENCH_MODE','safe')
for _, m in ipairs({'safe','quick','full','extreme'}) do if B.has('--'..m) then mode=m end end
local levels = {quick={32,64}, safe={64,128}, full={128,256,384}, extreme={256,512,768}}
assert(levels[mode], 'mode inconnu: '..mode)
local seq = B.number('--seq','MIMIR_BENCH_SEQ',mode=='quick' and 8 or 32,1,true)
local vocab = B.number('--vocab','MIMIR_BENCH_VOCAB',mode=='quick' and 128 or 2048,2,true)
local iters = B.number('--iters','MIMIR_BENCH_ITERS',1,1,true)
for _, dim in ipairs(levels[mode]) do
  for i=1,iters do
    log(string.format('mode=%s dim=%d seq=%d vocab=%d iteration=%d',mode,dim,seq,vocab,i))
    B.create('transformer', B.transformer(dim,mode=='extreme' and 8 or 2,seq,vocab))
    B.snapshot('après init (pic cumulatif du processus)')
  end
end
log('PASS benchmark construction')
B.report('official')
