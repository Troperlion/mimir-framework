local MPKLayers = dofile(ROOTWORK.."/scripts/modules/mpk_layers.lua")

local cases = {
  ["encoder/conv_1"] = "Conv2d",
  ["decoder/upconv_2"] = "ConvTranspose2d",
  ["decoder/convTransposeBlock"] = "ConvTranspose2d",
  ["backbone/depthwise_conv_3"] = "DepthwiseConv2d",
  ["block_1/bn"] = "BatchNorm2d",
  ["transformer/block_1/ln"] = "LayerNorm",
  ["transformer/cross_attention"] = "CrossAttention",
  ["transformer/mha"] = "MultiHeadAttention",
  ["transformer/attn"] = "SelfAttention",
  ["head/nms"] = "NMS",
  ["decoder/upsample_bilinear"] = "UpsampleBilinear",
  ["classifier/fc"] = "Linear",
  ["activation/relu"] = "ReLU",
  ["merge/concat"] = "Concat",
  ["custom_layer_without_hint"] = "Linear",
}

for name, expected in pairs(cases) do
  local actual = MPKLayers.infer_layer_type(name)
  assert(actual == expected, string.format(
    "%s: attendu=%s obtenu=%s", name, expected, tostring(actual)))
end

local function expect_count(layer_type, params, context, expected)
  local actual = MPKLayers.predict_params_count(layer_type, params, context)
  assert(actual == expected, string.format(
    "%s: params attendus=%d obtenus=%s",
    layer_type, expected, tostring(actual)))
end

expect_count("Linear", {
  in_features = 16, out_features = 32, use_bias = true,
}, {}, 544)
expect_count("Conv2d", {
  in_channels = 3, out_channels = 8, kernel_size = 3, use_bias = true,
}, {}, 224)
expect_count("DepthwiseConv2d", {
  in_channels = 8, out_channels = 8, kernel_size = 3, use_bias = false,
}, {}, 72)
expect_count("LayerNorm", {
  normalized_size = 128,
}, {}, 256)
expect_count("Embedding", {
  vocab_size = 1000, embed_dim = 64,
}, {}, 64000)
expect_count("SelfAttention", {
  embed_dim = 64, use_bias = true,
}, {}, 16640)
expect_count("Linear", {}, {
  layer_name = "classifier/output",
  base_config = {input_dim = 16, output_dim = 4},
}, 68)

log("test_mpk_layer_inference: OK")
