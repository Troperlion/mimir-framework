#!/usr/bin/env mimir --lua
-- Entraînement Conv2d natif: --quick, --steps N, --ram GiB, --dtype TYPE, --no-accel.
local Help = dofile(ROOTWORK..'/scripts/modules/help_cli.lua')
Help.auto_exit_help()
local B = dofile(ROOTWORK..'/scripts/benchmarks/common.lua')
B.setup()
local side = B.has('--quick') and 8 or 32
local steps = B.number('--steps', nil, B.has('--quick') and 20 or 100, 1, true)
assert(Mimir.Model.create_empty('benchmark_conv_train', {image_w=side,image_h=side,image_c=2}))
assert(Mimir.Model.push_layer('bench/conv', 'Conv2d', 2*2*3*3+2, {
  in_channels=2, out_channels=2, input_height=side, input_width=side,
  kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, pad_h=1, pad_w=1, use_bias=true,
}))
assert(Mimir.Model.set_layer_io('bench/conv', {'__input__'}, 'x'))
B.init()
local x = B.floats(side*side*2)
local function loss_grad(y)
  local loss, grad = 0, {}
  for i, v in ipairs(y) do loss=loss+v*v; grad[i]=2*v/#y end
  return loss/#y, grad
end
local first = loss_grad(B.forward(x, false, #x))
B.measure('Conv2d forward/backward/adamw', function()
  for _ = 1, steps do
    assert(Mimir.Model.zero_grads())
    local loss, grad = loss_grad(B.forward(x, true, #x))
    assert(B.finite(loss))
    assert(Mimir.Model.backward(grad))
    local gradients = B.vector(assert(Mimir.Model.get_gradients()))
    local norm = 0
    for _, v in ipairs(gradients) do norm=norm+v*v end
    assert(norm > 0, 'gradients nuls')
    assert(Mimir.Model.optimizer_step(0.01, 'adamw'))
  end
end)
local final = loss_grad(B.forward(x, false, #x))
assert(final < first, 'Conv2d: la loss ne diminue pas')
log(string.format('PASS Conv2d steps=%d MSE=%.8f -> %.8f', steps, first, final))
B.snapshot('fin')
B.report('conv_train')
