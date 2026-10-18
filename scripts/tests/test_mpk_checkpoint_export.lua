-- Native checkpoint -> MPK regression; all artifacts stay in a temporary directory.
local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
local Tools = dofile(ROOTWORK.."/scripts/modules/mpk_tools.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local tmp = os.tmpname()
os.remove(tmp)
FS.mkdir_p(tmp)
assert(Mimir.Model.create("transformer", {seq_len=4, d_model=8, num_layers=1,
  num_heads=2, mlp_hidden=16, vocab_size=16}))
assert(Mimir.Model.build())
assert(Mimir.Model.allocate_params())
local count = #Mimir.Model.get_layers()
for _, case in ipairs({{"raw_folder", "raw"}, {"safetensors", "model.safetensors"}, {"debug_json", "model.json"}}) do
  local path = tmp.."/"..case[2]
  assert(Mimir.Serialization.save(path, case[1], {save_optimizer=false, save_tokenizer=false, save_encoder=false}))
  local source, err = Tools.checkpoint(path, "auto")
  assert(source, err)
  assert(#source.layers == count)
  local out = tmp.."/"..case[1]..".mpk"
  arg = {"--checkpoint", path, "--out", out, "--compile", "--name", "roundtrip"}
  dofile(ROOTWORK.."/scripts/tools/build_mpk.lua")
  for _, file in ipairs({out, out..".bin"}) do
    local pkg = assert(MPK.read(file))
    assert(MPK.verify_checksum(pkg))
    local decoded = assert(MPK.decode_payload(pkg))
    assert(decoded.base_config.d_model == source.config.d_model)
    local nodes = decoded.model_structure.graph.nodes
    assert(#nodes == count)
    for i, layer in ipairs(source.layers) do
      assert(nodes[i].name == layer.name)
      assert(nodes[i].params_count == layer.params_count)
      assert(nodes[i].output == layer.output)
      assert(MPK.encode_json(nodes[i].inputs) == MPK.encode_json(layer.inputs or {"x"}))
    end
  end
end
arg = {"--register", "transformer", "--out", tmp.."/registry.mpk"}
dofile(ROOTWORK.."/scripts/tools/build_mpk.lua")
assert(MPK.read(tmp.."/registry.mpk"))
assert(not Tools.output_paths({compile="bad.bin"}, tmp.."/bad.mpk"))
local bad = tmp.."/bad.safetensors"
assert(MPK.write_text_file(bad, "short"))
assert(not Tools.checkpoint(bad))
local header = MPK.encode_json({weight={dtype="F32", shape={1}, data_offsets={0,4}}})
assert(MPK.write_text_file(bad, string.pack("<I8", #header)..header..string.rep("\0", 4)))
local missing, missing_err = Tools.checkpoint(bad)
assert(not missing and missing_err:match("no Mimir"))
header = MPK.encode_json({["model/architecture_json"]={dtype="U8", shape={10}, data_offsets={0,10}}})
assert(MPK.write_text_file(bad, string.pack("<I8", #header)..header))
assert(not Tools.checkpoint(bad))
assert(not Tools.checkpoint(tmp, "unknown"))
print("test_mpk_checkpoint_export: OK ("..tmp..")")
