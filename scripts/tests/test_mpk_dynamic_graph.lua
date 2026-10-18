---@diagnostic disable: undefined-global

local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local MPKLayers = dofile(ROOTWORK.."/scripts/modules/mpk_layers.lua")

local structure = {
  template = "dynamic_nms_test",
  architecture = "custom_graph",
  version = 2,
  build = {
    dynamic_layer_assembly = true,
    generated_by = "scripts/tests/test_mpk_dynamic_graph.lua",
  },
  graph = {
    mode = "node",
    nodes = {
      {
        id = "nms",
        name = "nms",
        type = "NonMaxSuppression",
        params_count = 0,
        inputs = {"boxes", "scores", "classes"},
        output = "x",
        params = {
          iou_threshold = 0.5,
          score_threshold = 0.1,
          max_detections = 1,
          class_agnostic = true,
        },
      },
    },
    links = {},
  },
}

local ok_graph, graph_err = MPKLayers.normalize_graph_in_place(structure)
assert(ok_graph, graph_err)
assert(structure.graph.nodes[1].type == "NMS")

local pkg, build_err = MPK.build({
  name = "dynamic_nms_test",
  type = "custom_graph",
  author = "test",
  base_config = {},
  model_structure = structure,
  description = "Dynamic MPK graph test",
})
assert(pkg, build_err)

local source = os.tmpname() .. ".mpk"
local binary = source .. ".bin"
assert(MPK.write(source, pkg))
assert(MPK.compile(source, binary))

local saved_arg = arg
arg = {
  "--in", binary,
  "--create",
  "--apply-graph",
  "--replace-layers",
  "--allocate",
  "--init", "zeros",
  "--seed", "7",
}
local ok_load, load_err = pcall(dofile, ROOTWORK.."/scripts/tools/load_mpk.lua")
arg = saved_arg
assert(ok_load, load_err)

local output, forward_err = Mimir.Model.forward({
  boxes = {
    0, 0, 10, 10,
    1, 1, 9, 9,
    20, 20, 30, 30,
  },
  scores = {0.9, 0.8, 0.7},
  classes = {0, 0, 1},
}, false)
assert(output, forward_err)
assert(#output == 1, "max_detections from MPK params was not applied")
assert(output[1] == 0, "unexpected retained index")

pcall(os.remove, source)
pcall(os.remove, binary)
io.stdout:write("[test_mpk_dynamic_graph] OK\n")
