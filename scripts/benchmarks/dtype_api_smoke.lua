local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- Smoke test: dtype API bridging

log("dtype_api_smoke: start")

-- Create a tiny model (any arch that exists). Default config is fine.
local ok, err = Mimir.Model.create("basic_mlp", { input_dim = 16, hidden_dim = 16, output_dim = 16 })
if not ok then
  error("create failed: " .. tostring(err))
end

-- Setter (via requested API path)
local ok2, dt = Mimir.Model.dtype("float16")
if not ok2 then
  error("dtype setter failed: " .. tostring(dt))
end

-- Getter
local cur = Mimir.Model.dtype()
assert(cur == "float16", "dtype getter mismatch")
assert(Mimir.Model.dtype("float32"))
assert(Mimir.Model.dtype() == "float32")
local accepted = Mimir.Model.dtype("invalid_dtype")
assert(accepted == false, "invalid dtype accepted")
log("dtype_api_smoke: dtype now = " .. tostring(cur))

log("dtype_api_smoke: ok")
