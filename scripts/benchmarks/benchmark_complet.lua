#!/usr/bin/env mimir --lua
-- Construction et inférence multi-architectures, routage nommé et sérialisation.
-- Options: --quick, --iters N, --warmup N, --ram GiB, --dtype TYPE, --no-accel.
local Help = dofile(ROOTWORK..'/scripts/modules/help_cli.lua')
Help.auto_exit_help()
local B = dofile(ROOTWORK..'/scripts/benchmarks/common.lua')
B.setup()
local side = B.has('--quick') and 8 or 16
local n = side*side*3
local seq, dim, vocab = 8, 32, 128
local cases = {
  {'transformer', B.transformer(dim,2,seq,vocab), B.transformer_input(seq,vocab,dim), dim},
  {'vit', {num_tokens=seq,d_model=dim,num_layers=2,num_heads=4,mlp_hidden=128,output_dim=8}, B.floats(seq*dim), 8},
  {'unet', {image_w=side,image_h=side,image_c=3,base_channels=8,depth=2}, B.floats(n), n},
  {'vae', {image_w=side,image_h=side,image_c=3,latent_dim=8,hidden_dim=32}, B.floats(n), n+16},
  {'resnet', {image_w=side,image_h=side,image_c=3,base_channels=4,num_classes=8,blocks1=1,blocks2=1,blocks3=1,blocks4=1}, B.floats(n), 8},
  {'diffusion', {image_w=side,image_h=side,image_c=3,time_dim=16,hidden_dim=32,dropout=0.0}, B.floats(n+16), n},
}
for _, c in ipairs(cases) do
  B.create(c[1], c[2])
  B.bench_forward(c[3], c[4])
  B.snapshot(c[1])
end
-- Graphe entièrement construit avant allocate/init; entrées nommées publiques.
assert(Mimir.Model.create_empty('benchmark_routing', {}))
assert(Mimir.Model.push_layer('route/add', 'Add', 0))
assert(Mimir.Model.set_layer_io('route/add', {'a','b'}, 'x'))
B.init()
local x = B.floats(32)
local y = B.bench_forward({a=x,b=x}, #x)
for i,v in ipairs(x) do assert(math.abs(y[i]-2*v)<1e-6, 'Add incorrect') end
B.roundtrip({input_dim=16,hidden_dim=32,output_dim=8,hidden_layers=1}, B.floats(16))
-- Observation des compteurs; ne constitue pas une preuve d'absence de fuite RSS.
B.snapshot('avant répétitions')
B.bench_forward(B.floats(16),8)
B.snapshot('après répétitions')
log('PASS benchmark complet')
B.report('complet')
