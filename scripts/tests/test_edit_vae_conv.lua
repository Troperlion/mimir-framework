-- Native end-to-end test: synthetic checkpoint only; no user dataset.
local directory = assert(os.getenv("MIMIR_EDITOR_TEST_DIR"), "Set MIMIR_EDITOR_TEST_DIR to a temporary directory")
local PPM = dofile(ROOTWORK.."/scripts/modules/image_ppm.lua")
local cfg = assert(Mimir.Architectures.default_config("vae_conv"))
cfg.image_w, cfg.image_h, cfg.image_c = 16, 16, 3
cfg.latent_w, cfg.latent_h, cfg.latent_c = 4, 4, 2
cfg.latent_dim, cfg.base_channels = 32, 8
cfg.resnet, cfg.attention, cfg.text_cond, cfg.stochastic_latent = false, false, false, false
cfg.dec_norm, cfg.enc_norm, cfg.dtype = "none", "none", "float32"
assert(Mimir.Model.create("vae_conv", cfg))
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("he", 123))
local checkpoint = directory.."/checkpoint"
assert(Mimir.Serialization.save(checkpoint, "raw_folder", {
  save_optimizer=false,save_encoder=false,save_tokenizer=false,
}))
local pixels = {}
for i=1,16*16*3 do pixels[i] = ((i*7)%255)/127.5-1 end
local input = directory.."/input.ppm"
assert(PPM.write(input,pixels,16,16))
local function run(name,strength,mix)
  arg = {"--checkpoint",checkpoint,"--input",input,"--output",directory.."/"..name..".ppm",
    "--headless","--strength",tostring(strength),"--mix",tostring(mix)}
  dofile(ROOTWORK.."/scripts/inferences/edit_vae_conv.lua")
  return assert(PPM.read(directory.."/"..name..".ppm"))
end
local baseline, edited, restored = run("baseline",0,1), run("edited",1,1), run("source",1,0)
local source = assert(PPM.read(input))
assert(baseline.width==16 and baseline.height==16)
local changed = false
for i=1,#source.pixels do
  assert(source.pixels[i] == restored.pixels[i], "mix=0 must preserve source")
  if baseline.pixels[i] ~= edited.pixels[i] then changed=true end
end
assert(changed, "latent channel edit must change decoded pixels")
-- An incomplete checkpoint must fail before producing a plausible black image.
local manifest_path = checkpoint.."/manifest.json"
local manifest = read_json(manifest_path)
local saved_index = manifest.tensor_index
manifest.tensor_index = {}
write_json(manifest_path, manifest)
local ok, err = pcall(run, "missing", 0, 1)
manifest.tensor_index = saved_index
write_json(manifest_path, manifest)
assert(not ok and tostring(err):find("Poids manquants"), "missing weights must fail explicitly")
print("EDITOR_NATIVE_OK")
