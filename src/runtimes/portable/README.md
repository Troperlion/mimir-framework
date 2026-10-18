# Native OpenCL / Vulkan reference operations

The OpenCL and Vulkan adapters retain their existing fast paths and use
`PortableRuntime` for the remaining operations and backward passes. Numerical
kernels execute through OpenCL compute or Vulkan compute shaders. Host memory
stages tensors and accumulates returned parameter gradients; it does not call
CPU forward/backward arithmetic as a hidden fallback.

Coverage matches the executable CPU dispatcher: 72 forward types and 73 backward
types. `Lambda` forward is rejected by the CPU dispatcher itself and therefore
also rejected here. Its identity backward is supported. Hard NMS returns discrete indices; its backward returns zero gradients for
boxes, scores and optional classes. Selection boundaries are nondifferentiable;
this is an explicit stop-gradient contract, not a differentiable NMS surrogate.
The CPU dispatcher uses the same NMS stop-gradient contract. Configuration flags and shape checks still apply.

These are correctness reference kernels: a single GPU invocation processes the
logical work serially. Coverage parity does not imply performance parity.
They are limited to 65,536 staged floats and a conservative serial work budget
of 4,194,304 iterations per dispatch (including GEMM reduction work). Larger
inputs/weights are rejected by operation capability queries; staging also checks
intermediate buffers and dispatch dimensions before submission. This prevents
large image training from sending unbounded serial kernels to the GPU watchdog.
The router applies its ordinary fallback policy; parallel fast paths remain available.

## Generation

From the repository root:

```sh
python3 scripts/dev/generate_portable_reference.py
```

Generation requires Python 3, `glslangValidator` and `spirv-val`. Generated host
wrappers and OpenCL/GLSL kernels reuse the existing GPU algorithms aligned with
the CPU contract. Embedded sources/SPIR-V allow ordinary builds without shader
compiler or CUDA toolkit dependencies. CPU sources are read-only inputs to
capability checks and numerical tests.

## Validation

With `MIMIR_ENABLE_TESTS=ON` and the relevant backend enabled, CTest registers
`RuntimeTest.OPENCLNativeOps` and `RuntimeTest.VULKANNativeOps`. Tests compare
forward values, input/parameter gradients and repeated accumulation against
the CPU dispatcher, validate stochastic saved context, and assert the router
selects the requested GPU backend. A missing device returns skip code 77.
