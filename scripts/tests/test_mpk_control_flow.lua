---@diagnostic disable: undefined-global

local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")

local function build(depth, use_output)
  local pkg, err = MPK.build({
    name = "control_flow_test",
    type = "custom_graph",
    author = "test",
    base_config = { hidden_dim = 4, depth = depth, use_output = use_output },
    model_structure = MPK.model_structure_template("configurable_stack"),
    description = "MPK control flow test",
  })
  assert(pkg, err)
  return pkg
end

local expanded, expanded_err = MPK.decode_payload(build(3, true))
assert(expanded, expanded_err)
assert(expanded.model_structure.control == nil, "control section must be consumed")
assert(#expanded.model_structure.graph.nodes == 4, "loop + condition expansion failed")
assert(expanded.model_structure.graph.nodes[1].name == "layer_1")
assert(expanded.model_structure.graph.nodes[2].inputs[1] == "layer_1_out")
assert(expanded.model_structure.graph.nodes[3].params_count == 20)
assert(expanded.model_structure.graph.nodes[4].name == "output_activation")

local reduced, reduced_err = MPK.decode_payload(build(1, false))
assert(reduced, reduced_err)
assert(#reduced.model_structure.graph.nodes == 1, "base_config did not alter graph structure")

local temp_dir = os.getenv("TMPDIR") or "/tmp"
local unique = tostring(os.time()) .. "_" .. tostring(math.floor(os.clock() * 1000000))
local source = temp_dir .. "/mimir_mpk_control_flow_" .. unique .. ".mpk"
local binary = source .. ".bin"
assert(MPK.write(source, build(3, true)))
assert(MPK.compile(source, binary))
local compiled, compiled_err = MPK.read(binary)
assert(compiled, compiled_err)
local compiled_payload, payload_err = MPK.decode_payload(compiled)
assert(compiled_payload, payload_err)
assert(#compiled_payload.model_structure.graph.nodes == 4, "compiled graph expanded twice or not at all")

pcall(os.remove, source)
pcall(os.remove, binary)
io.stdout:write("[test_mpk_control_flow] OK\n")