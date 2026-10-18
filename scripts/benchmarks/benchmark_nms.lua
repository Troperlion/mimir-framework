#!/usr/bin/env mimir --lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()
local B = dofile(ROOTWORK.."/scripts/benchmarks/common.lua")

-- Benchmark dataset-free du layer runtime NMS.
--
-- Usage:
--   ./bin/mimir --lua scripts/benchmarks/benchmark_nms.lua -- --quick
--   ./bin/mimir --lua scripts/benchmarks/benchmark_nms.lua -- \
--     --boxes 4096 --classes 80 --warmup 5 --iters 50
--
-- Variables équivalentes:
--   MIMIR_NMS_BOXES, MIMIR_NMS_CLASSES, MIMIR_NMS_WARMUP,
--   MIMIR_NMS_ITERS, MIMIR_NMS_IOU, MIMIR_NMS_SCORE,
--   MIMIR_NMS_MAX_DETECTIONS, MIMIR_NMS_CLASS_AGNOSTIC.

local function arg_value(name)
  for i = 1, #(arg or {}) do
    if arg[i] == name then return arg[i + 1] end
  end
  return nil
end

local function has(name)
  for _, value in ipairs(arg or {}) do
    if value == name then return true end
  end
  return false
end

local function number_option(flag, env, default)
  local value = tonumber(arg_value(flag) or os.getenv(env) or default)
  assert(value and value == value and math.abs(value) < math.huge, "option invalide: " .. flag)
  return value
end

local function bool_env(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  value = value:lower()
  return value == "1" or value == "true" or value == "yes" or value == "on"
end

local quick = has("--quick")
local box_count = math.max(1, math.floor(number_option(
  "--boxes", "MIMIR_NMS_BOXES", quick and 256 or 2048)))
local class_count = math.max(1, math.floor(number_option(
  "--classes", "MIMIR_NMS_CLASSES", 20)))
local warmup = math.max(0, math.floor(number_option(
  "--warmup", "MIMIR_NMS_WARMUP", quick and 1 or 3)))
local iterations = math.max(1, math.floor(number_option(
  "--iters", "MIMIR_NMS_ITERS", quick and 3 or 20)))
local iou_threshold = number_option("--iou", "MIMIR_NMS_IOU", 0.5)
local score_threshold = number_option("--score", "MIMIR_NMS_SCORE", 0.05)
local max_detections = math.max(0, math.floor(number_option(
  "--max-detections", "MIMIR_NMS_MAX_DETECTIONS", 300)))
local class_agnostic =
  has("--class-agnostic") or bool_env("MIMIR_NMS_CLASS_AGNOSTIC", false)

local boxes, scores, classes = {}, {}, {}
for i = 1, box_count do
  -- Groupes de boîtes volontairement proches pour exercer la suppression IoU.
  local cluster = math.floor((i - 1) / 4)
  local offset = ((i - 1) % 4) * 0.75
  local x1 = (cluster % 64) * 12 + offset
  local y1 = (math.floor(cluster / 64) % 64) * 12 + offset
  local base = (i - 1) * 4
  boxes[base + 1] = x1
  boxes[base + 2] = y1
  boxes[base + 3] = x1 + 10
  boxes[base + 4] = y1 + 10
  scores[i] = 1.0 - ((i - 1) % 1000) / 1000
  classes[i] = cluster % class_count
end

local ok, err = Mimir.Model.create_empty("benchmark_nms", {
  nms_iou_threshold = iou_threshold,
  nms_score_threshold = score_threshold,
  nms_max_detections = max_detections,
  nms_class_agnostic = class_agnostic,
})
assert(ok, err)
assert(Mimir.Model.push_layer("benchmark/nms", "NMS", 0))
assert(Mimir.Model.set_layer_io(
  "benchmark/nms", {"boxes", "scores", "classes"}, "x"))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("zeros", 0))

local inputs = {boxes = boxes, scores = scores, classes = classes}
local function forward()
  local output, forward_err = Mimir.Model.forward(inputs, false)
  assert(output ~= nil, forward_err)
  return output
end

for _ = 1, warmup do forward() end

local first = forward()
assert(#first > 0, "NMS n'a conservé aucune boîte")
assert(#first <= box_count, "NMS a produit trop d'indices")
if max_detections > 0 then
  assert(#first <= max_detections, "nms_max_detections non respecté")
end

for _, index in ipairs(first) do
  assert(index % 1 == 0 and index >= 0 and index < box_count, "indice NMS invalide")
end
local samples, elapsed = {}, 0
for i = 1, iterations do
  local started = B.now()
  forward()
  samples[i] = (B.now() - started) * 1000
  elapsed = elapsed + samples[i] / 1000
end
local average_ms = elapsed * 1000 / iterations
local boxes_per_second =
  elapsed > 0 and (box_count * iterations / elapsed) or math.huge

log("NMS runtime benchmark")
log(string.format(
  "  boxes=%d classes=%d kept=%d iou=%.3f score=%.3f max=%d class_agnostic=%s",
  box_count, class_count, #first, iou_threshold, score_threshold,
  max_detections, tostring(class_agnostic)))
log(string.format(
  "  warmup=%d iterations=%d wall_total=%.3f ms wall_average=%.3f ms wall_throughput=%.0f boxes/s",
  warmup, iterations, elapsed * 1000, average_ms, boxes_per_second))
B.record('NMS forward', samples, {
  warmup=warmup, iterations=iterations, boxes=box_count, kept=#first,
  boxes_per_second=boxes_per_second,
})
B.report('nms')
