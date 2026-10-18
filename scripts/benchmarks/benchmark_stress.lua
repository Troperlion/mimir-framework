#!/usr/bin/env mimir --lua
-- Entraînement MLP croissant puis créations répétées de modèles.
-- Options: --quick, --steps N (>=2), --churn N, --ram GiB, --dtype TYPE, --no-accel.
local Help = dofile(ROOTWORK..'/scripts/modules/help_cli.lua')
Help.auto_exit_help()
local B = dofile(ROOTWORK..'/scripts/benchmarks/common.lua')
B.setup()
local levels = {{256,1,200,0.01},{512,2,250,0.008},{1024,2,200,0.006},{2048,3,150,0.004}}
if B.has('--quick') then levels={{32,1,20,0.01}} end
local function loss_grad(y)
  local loss = 0
  for _, v in ipairs(y) do loss=loss+0.5*v*v end
  assert(B.finite(loss), 'loss non finie')
  return loss, y -- L=0.5*sum(y²), dL/dy=y
end
for _, level in ipairs(levels) do
  local dim, depth, defaults, lr = table.unpack(level)
  local steps = B.number('--steps', nil, defaults, 2, true)
  B.create('basic_mlp', {input_dim=dim,hidden_dim=dim,output_dim=dim,hidden_layers=depth,dropout=0.0})
  local x = B.floats(dim)
  local first = loss_grad(B.forward(x, false, dim))
  local fwd, bwd, opt = 0, 0, 0
  local fwd_samples, bwd_samples, opt_samples = {}, {}, {}
  for step=1,steps do
    assert(Mimir.Model.zero_grads())
    local start = B.now()
    local y = B.forward(x, true, dim)
    local elapsed = B.now()-start
    fwd=fwd+elapsed
    fwd_samples[#fwd_samples+1]=elapsed*1000
    local loss, grad = loss_grad(y)
    start = B.now()
    assert(Mimir.Model.backward(grad))
    elapsed=B.now()-start
    bwd=bwd+elapsed
    bwd_samples[#bwd_samples+1]=elapsed*1000
    B.vector(assert(Mimir.Model.get_gradients()))
    start = B.now()
    assert(Mimir.Model.optimizer_step(lr, 'adamw'))
    elapsed=B.now()-start
    opt=opt+elapsed
    opt_samples[#opt_samples+1]=elapsed*1000
    if step==1 or step==steps then log(string.format('step=%d/%d loss=%.8f',step,steps,loss)) end
  end
  local final = loss_grad(B.forward(x, false, dim))
  assert(final < first*0.99, 'MLP: diminution de loss <1%')
  log(string.format('PASS MLP dim=%d steps=%d loss=%.8f -> %.8f wall_ms/step forward=%.3f backward=%.3f optimizer=%.3f',
    dim,steps,first,final,fwd*1000/steps,bwd*1000/steps,opt*1000/steps))
  B.record('forward MLP dim='..dim, fwd_samples, {iterations=steps})
  B.record('backward MLP dim='..dim, bwd_samples, {iterations=steps})
  B.record('optimizer MLP dim='..dim, opt_samples, {iterations=steps})
  B.snapshot('entraînement')
end
local churn = B.number('--churn',nil,B.has('--quick') and 3 or 120,1,true)
B.measure('créations répétées: '..churn, function()
  for i=1,churn do
    assert(Mimir.Model.create('basic_mlp',{input_dim=256,hidden_dim=256,output_dim=256,hidden_layers=1}))
    B.dtype()
    assert(Mimir.Model.allocate_params())
    assert(Mimir.Model.init_weights('xavier',i))
    B.forward(B.floats(256), false, 256)
  end
end)
B.snapshot('fin')
log('PASS benchmark stress')
B.report('stress')
