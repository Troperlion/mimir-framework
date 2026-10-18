-- Éditeur latent VAEConv natif Mímir, avec UI commune Qt / GTK / SFML / Web.
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help({
  script_path = "scripts/inferences/edit_vae_conv.lua",
  description = "Édition d'image par modification du latent VAEConv. Export PPM RGB. Backend Viz : "..Mimir.Viz.backend(),
  options = {
    "--checkpoint PATH : checkpoint raw_folder VAEConv entraîné (requis)",
    "--input PATH : image PNG/JPEG/BMP ou PPM P6/P3 (requise)",
    "--output PATH : image PPM à enregistrer (défaut outputs/edited.ppm)",
    "--strength N : décalage initial du canal latent (défaut 0)",
    "--channel N : canal latent, de 1 à latent_c (défaut 1)",
    "--mix N : mélange source/résultat VAE, de 0 à 1 (défaut 1)",
    "--headless : calculer et enregistrer une fois, sans interface",
    "--mem-gb N : budget mémoire (défaut 8 Go)",
    "Boutons : force ±, canal suivant, mélange ±, réinitialiser, enregistrer.",
    "L : afficher/masquer la galerie de tous les canaux latents.",
    "Ctrl+U : choisir une autre image et recalculer son latent.",
    "H : aide et backend actif. Le backend est sélectionné à la compilation.",
    "Les canaux latents ne sont pas des commandes sémantiques (couleur, objet...).",
  },
  examples = {"./bin/mimir --lua scripts/inferences/edit_vae_conv.lua -- --checkpoint checkpoint/vae/epoch_0010 --input photo.ppm --output outputs/edition.ppm"},
})
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")
local PPM = dofile(ROOTWORK.."/scripts/modules/image_ppm.lua")
local opts = Args.parse(arg) or {}
local allowed = {checkpoint=true,input=true,output=true,strength=true,channel=true,mix=true,headless=true,["mem-gb"]=true}
for key in pairs(opts) do assert(allowed[key], "Option inconnue : "..tostring(key)) end
local function number(key, default)
  local value = opts[key] == nil and default or tonumber(opts[key])
  assert(value and value == value and math.abs(value) < math.huge, "Nombre invalide : "..key)
  return value
end
assert(type(opts.checkpoint) == "string", "--checkpoint requis")
assert(type(opts.input) == "string", "--input requis")
local output = opts.output or "outputs/edited.ppm"
assert(type(output) == "string" and output:lower():match("%.ppm$"), "--output doit être un fichier .ppm")
assert(output ~= opts.input, "Choisir une sortie distincte de l'image source")
local checkpoint = assert(Ckpt.resolve_dir(opts.checkpoint), "Checkpoint raw_folder introuvable")
local architecture = read_json(FS.join(checkpoint, "model", "architecture.json"))
local cfg = assert(architecture.model_config or architecture.modelConfig, "model_config absent")
assert(cfg.type == "vae_conv", "Checkpoint vae_conv attendu")
for _, key in ipairs({"image_w","image_h","image_c","latent_w","latent_h","latent_c"}) do
  assert(type(cfg[key]) == "number" and cfg[key] > 0 and cfg[key] == math.floor(cfg[key]), "Dimension invalide : "..key)
end
assert(cfg.image_c == 3, "PPM exige image_c=3")
cfg.latent_dim = cfg.latent_w * cfg.latent_h * cfg.latent_c
cfg.stochastic_latent = false
cfg.dtype = ({F32="float32",F16="float16",BF16="bfloat16"})[tostring(cfg.dtype):upper()] or cfg.dtype
-- La topologie sérialisée prévaut sur les anciens champs de configuration.
cfg.dec_norm = "none"
for _, layer in ipairs(architecture.layers or {}) do
  if tostring(layer.name):match("^vae_conv/dec/") then
    local kind = tostring(layer.type):lower()
    if kind == "groupnorm" then cfg.dec_norm = "groupnorm"; break end
    if kind == "layernorm" then cfg.dec_norm = "layernorm" end
  end
end
local strength, channel, mix = number("strength", 0), number("channel", 1), number("mix", 1)
assert(math.abs(strength) <= 10, "--strength doit être entre -10 et 10")
assert(channel >= 1 and channel <= cfg.latent_c and channel == math.floor(channel), "--channel hors limites")
assert(mix >= 0 and mix <= 1, "--mix doit être entre 0 et 1")
local headless = Args.get_bool(opts, "headless", false)
assert(headless or Mimir.Viz.backend() ~= "NONE",
  "Viz désactivée : compiler avec -DMIMIR_VIZ_BACKEND=WEB (ou QT/GTK/SFML), ou utiliser --headless")
local mem = number("mem-gb", 8)
assert(mem > 0, "--mem-gb doit être positif")
assert(Mimir.Allocator.configure({max_ram_gb=mem, enable_compression=true}))
local function load_model(name)
  local model_cfg = {}
  for key, value in pairs(cfg) do model_cfg[key] = value end
  model_cfg.inference_output_tensor = name == "vae_conv_decode" and "x" or "vae_conv/recon"
  assert(Mimir.Model.create(name, model_cfg))
  assert(Mimir.Model.allocate_params())
  -- Le décodeur charge un sous-ensemble du checkpoint complet. Vérifier
  -- explicitement chaque poids attendu avant le chargement partiel.
  local manifest = read_json(FS.join(checkpoint, "manifest.json"))
  local tensors = {}
  for _, entry in ipairs(assert(manifest.tensor_index, "tensor_index absent")) do tensors[entry.name] = entry end
  for _, layer in ipairs(Mimir.Model.get_layers()) do
    if layer.param_count > 0 and not layer.name:match("/zero_enc_skip$") then
      local entry = assert(tensors[layer.name.."_weights"], "Poids manquants : "..layer.name)
      local metadata = read_json(FS.join(checkpoint, entry.json_file))
      local count = 1
      for _, size in ipairs(assert(metadata.shape, "shape absent")) do count = count * size end
      assert(count == layer.param_count, "Topologie incompatible : "..layer.name)
      assert(({F32=true,F16=true,BF16=true,F64=true,float32=true,float16=true,bfloat16=true,float64=true})[metadata.dtype], "dtype non supporté")
      assert(type(metadata.data_file)=="string", "data_file absent")
    end
  end
  assert(Mimir.Serialization.load(checkpoint, "raw_folder", {
    load_encoder=false, load_tokenizer=false, load_optimizer=false,
    strict_mode=name=="vae_conv", validate_checksums=true,
  }))
end
local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local latent_dim = cfg.latent_dim
local input_path = opts.input
local source, input, original

local function read_source(path)
  assert(type(path) == "string" and path ~= "", "Chemin d'image vide")
  local image
  if path:lower():match("%.ppm$") then
    image = assert(PPM.read(path))
  else
    local loaded = assert(Mimir.IO.read_image_rgb_u8(path,cfg.image_w,cfg.image_h))
    image = {pixels=loaded.image,width=loaded.width,height=loaded.height}
  end
  local pixels = image.pixels
  if image.width ~= cfg.image_w or image.height ~= cfg.image_h then
    pixels = PPM.resize(pixels, image.width, image.height, cfg.image_w, cfg.image_h)
  end
  local normalized = {}
  for i, value in ipairs(pixels) do normalized[i] = value / 127.5 - 1 end
  assert(#normalized == image_dim, "Dimensions d'image inattendues")
  return pixels, normalized
end

local function encode_source(path)
  local next_source, next_input = read_source(path)
  load_model("vae_conv")
  local packed = assert(Mimir.Model.forward(next_input, true))
  assert(#packed == image_dim + 2 * latent_dim,
    string.format("Sortie VAEConv inattendue : %d valeurs, %d attendues",#packed,image_dim+2*latent_dim))
  local next_original = {}
  for i = 1, latent_dim do next_original[i] = packed[image_dim+i] end
  -- Seul le décodeur reste chargé durant les interactions.
  load_model("vae_conv_decode")
  source, input, original, input_path = next_source, next_input, next_original, path
end

encode_source(input_path)
local result, preview, edited_latent
local function compute()
  local latent, plane = {}, cfg.latent_w * cfg.latent_h
  for i = 1, latent_dim do
    latent[i] = original[i] + (math.floor((i-1)/plane)+1 == channel and strength or 0)
  end
  local decoded = assert(Mimir.Model.forward(latent, true))
  assert(#decoded == image_dim, "Sortie décodeur inattendue")
  result, preview = {}, {}
  for i = 1, image_dim do
    local value = input[i] * (1-mix) + decoded[i] * mix
    assert(value == value and math.abs(value) < math.huge, "Sortie non finie")
    result[i] = math.max(-1, math.min(1, value))
    preview[i] = math.floor((result[i]+1)*127.5+0.5)
  end
  edited_latent = latent
end
local function save()
  local directory = FS.dirname(output)
  if directory and directory ~= "" then FS.mkdir_p(directory) end
  assert(PPM.write(output, result, cfg.image_w, cfg.image_h))
  print("Image enregistrée : "..output)
end
compute()
if headless then save(); return end
local UI = dofile(ROOTWORK.."/scripts/modules/viz_ui.lua")
local latent_grid_cols = math.max(1, math.min(8, cfg.latent_c))
assert(Mimir.Viz.create({visualization={enabled=true,window_width=1200,window_height=800,
  window_title="Mímir • Éditeur VAEConv",fps_limit=30,image_grid_cols=latent_grid_cols,
  image_grid_rows=math.max(1,math.ceil(cfg.latent_c/latent_grid_cols))}}))
local view_w, view_h = 1200, 800
local show_latents = false
local path_input_mode = false
local path_buffer = ""
local status_message = ""
local function scene(initial)
  local cols = math.max(1, math.floor((view_w-24)/167))
  local entries = path_input_mode
    and {{"load_path","Charger"},{"cancel_path","Annuler"}}
    or {{"less","Force −"},{"more","Force +"},{"channel","Canal suivant"},
      {"mix_less","Mélange −"},{"mix_more","Mélange +"},{"reset","Réinitialiser"},{"save","Enregistrer"}}
  local top = 32 + math.ceil(#entries/cols)*50
  local panels = {}
  for id = 0, 6 do panels[#panels+1] = {id=id,visible=id==2} end
  local title
  if path_input_mode then
    title = "Nouvelle image : "..(path_buffer ~= "" and path_buffer or "saisir un chemin")
  elseif show_latents then
    title = string.format("Latent • %d canaux • Canal édité %d • Force %.2f",cfg.latent_c,channel,strength)
  else
    title = string.format("Source / édition • Canal %d/%d • Force %.2f • Mélange %.0f%%",channel,cfg.latent_c,strength,mix*100)
  end
  if status_message ~= "" then title = status_message.." • "..title end
  panels[3] = {id=2,visible=true,x=12,y=top,w=math.max(220,view_w-24),h=math.max(140,view_h-top-20),
    title=title}
  local controls = {}
  for i, entry in ipairs(entries) do
    controls[#controls+1] = {id=entry[1],label=entry[2],x=12+((i-1)%cols)*167,y=20+math.floor((i-1)/cols)*50,w=155,h=38}
  end
  if not initial then panels = {{id=2,title=panels[3].title}} end
  local image_size = show_latents and math.max(64,math.min(160,math.floor((view_w-80)/latent_grid_cols)))
    or math.max(64,math.min(512,math.floor((view_w-80)/2)))
  return {panels=panels,controls=controls,image_size=image_size,events=true,capture_keys=path_input_mode,
    help="L : inspecter tous les canaux latents • Ctrl+U : autre image • Force : ±0.1 • Enregistrer : sortie PPM."}
end
local ui = UI.new(scene(true))
local function latent_plane_pixels(latent, latent_channel)
  local plane = cfg.latent_w * cfg.latent_h
  local offset = (latent_channel-1) * plane
  local min_value, max_value = math.huge, -math.huge
  for i = 1, plane do
    local value = latent[offset+i]
    min_value, max_value = math.min(min_value,value), math.max(max_value,value)
  end
  local pixels, range = {}, max_value-min_value
  for i = 1, plane do
    pixels[i] = range > 1e-12 and math.floor((latent[offset+i]-min_value)*255/range+0.5) or 127
  end
  return pixels, min_value, max_value
end
local function publish()
  assert(Mimir.Viz.clear())
  if show_latents then
    for latent_channel = 1, cfg.latent_c do
      local pixels, min_value, max_value = latent_plane_pixels(edited_latent, latent_channel)
      assert(Mimir.Viz.add_image(pixels,cfg.latent_w,cfg.latent_h,1,
        string.format("Canal %d/%d • %.3g à %.3g",latent_channel,cfg.latent_c,min_value,max_value)))
    end
  else
    assert(Mimir.Viz.add_image(source,cfg.image_w,cfg.image_h,3,"Source • "..input_path))
    assert(Mimir.Viz.add_image(preview,cfg.image_w,cfg.image_h,3,
      string.format("Canal %d • Force %.2f • Mélange %.0f%%",channel,strength,mix*100)))
  end
  ui:configure(scene())
end
local function edit(fn)
  return function() fn(); compute(); publish() end
end
ui:on("less",edit(function() strength=math.max(-10,strength-0.1) end))
ui:on("more",edit(function() strength=math.min(10,strength+0.1) end))
ui:on("channel",edit(function() channel=channel%cfg.latent_c+1 end))
ui:on("mix_less",edit(function() mix=math.max(0,mix-0.1) end))
ui:on("mix_more",edit(function() mix=math.min(1,mix+0.1) end))
ui:on("reset",edit(function() strength=0; channel=1; mix=1 end))
ui:on("save",function()
  save()
  ui:configure({panels={{id=2,title="Enregistré : "..output}}})
end)
local function finish_path_input()
  local path = path_buffer
  if path == "" then return end
  path_input_mode = false
  path_buffer = ""
  status_message = "Chargement..."
  ui:configure(scene(true))
  local ok, err = xpcall(function()
    encode_source(path)
    channel = math.min(channel,cfg.latent_c)
    compute()
  end, debug.traceback)
  if not ok then
    pcall(load_model,"vae_conv_decode")
    status_message = "Échec du chargement"
    print(err)
  else
    status_message = "Image chargée"
  end
  publish()
end
local function begin_path_input()
  path_input_mode = true
  path_buffer = ""
  status_message = ""
  ui:configure(scene(true))
end
local function capture_command(command)
  local pipe = io.popen(command.." 2>/dev/null")
  if not pipe then return nil end
  local selected = pipe:read("*l")
  pipe:close()
  return selected and selected:gsub("[\r\n]+$","") or nil
end
local function choose_image()
  local commands = {
    "zenity --file-selection --title='Choisir une image' --file-filter='Images | *.png *.jpg *.jpeg *.bmp *.ppm'",
    "kdialog --getopenfilename . 'Images (*.png *.jpg *.jpeg *.bmp *.ppm)'",
    "yad --file-selection --title='Choisir une image'",
  }
  for _, command in ipairs(commands) do
    local selected = capture_command(command)
    if selected and selected ~= "" then
      path_buffer = selected
      finish_path_input()
      return
    end
  end
  begin_path_input()
end
ui:on("load_path",finish_path_input)
ui:on("cancel_path",function()
  path_input_mode, path_buffer, status_message = false, "", ""
  ui:configure(scene(true))
end)
ui:on("key",function(event)
  if path_input_mode then
    if event.code == 59 then path_buffer = path_buffer:sub(1,-2); ui:configure(scene())
    elseif event.code == 58 then finish_path_input()
    elseif event.code == 36 then path_input_mode, path_buffer = false, ""; ui:configure(scene(true)) end
    return
  end
  if event.code == 11 and not event.control then
    show_latents = not show_latents
    status_message = ""
    publish()
  elseif event.code == 20 and event.control then
    choose_image()
  end
end)
ui:on("text",function(event)
  if not path_input_mode or not event.unicode or event.unicode < 32 then return end
  path_buffer = path_buffer..utf8.char(event.unicode)
  ui:configure(scene())
end)
local function adapt_window(event)
  if event.width and (view_w ~= event.width or view_h ~= event.height) then
    view_w, view_h = event.width, event.height
    ui:configure(scene(true))
  end
end
ui:on("resize", adapt_window)
ui:on("configured", adapt_window)
publish()
print("Viz : "..Mimir.Viz.backend().." • L : latent • Ctrl+U : autre image • H : aide")
local ok, err = xpcall(function()
  while Mimir.Viz.is_open() do ui:dispatch(30) end
end, debug.traceback)
Mimir.Viz.set_enabled(false)
if not ok then error(err) end
