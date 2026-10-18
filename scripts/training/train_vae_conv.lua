---@diagnostic disable: undefined-global, undefined-field, inject-field
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local Ckpt = dofile(ROOTWORK.."/scripts/modules/checkpoint_resume.lua")

local function opt_num(k, d)
  local v = opts[k]
  if v == nil then return d end
  local n = tonumber(v)
  if n == nil then return d end
  return n
end

local function opt_int(k, d)
  return math.floor(opt_num(k, d))
end

local function opt_str(k, d)
  local v = opts[k]
  if v == nil or v == true then return d end
  return tostring(v)
end

local function opt_bool(k, d)
  local v = opts[k]
  if v == nil then v = d end
  if v == nil then return nil end
  if v == true or v == false then return v end
  if type(v) == "number" then return v ~= 0 end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local BaseTok = dofile(ROOTWORK.."/scripts/modules/base_tokenizer.lua")

local function assert_ok(ok, err, msg)
  if ok == false then
    error((msg or "Operation failed") .. ": " .. tostring(err))
  end
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" then return true end

  local dtype_fn = nil
  if type(Mimir.model) == "table" and type(Mimir.model.dtype) == "function" then
    dtype_fn = Mimir.model.dtype
  elseif type(Mimir.Model) == "table" and type(Mimir.Model.dtype) == "function" then
    dtype_fn = Mimir.Model.dtype
  end
  if dtype_fn == nil then return true end

  local ok, dt_or_err = dtype_fn(dtype)
  assert_ok(ok, dt_or_err, "Model.dtype(" .. tostring(dtype) .. ") failed")
  return true
end

-- Mode CPU-only: ajuste quelques défauts pour éviter des options trop coûteuses
-- (l'utilisateur peut toujours forcer via les flags).
local CPU_ONLY = opt_bool(
  "cpu-only",
  opt_bool("cpu_only", opt_bool("cpu", false))
)

local function detect_mem_total_gb()
  -- Linux: /proc/meminfo (kB). Retourne un entier (Go) ou nil.
  local f = io.open("/proc/meminfo", "r")
  if not f then return nil end
  local line = f:read("*l")
  while line do
    local k, v = line:match("^(%S+):%s*(%d+)")
    if k == "MemTotal" and v then
      f:close()
      local kb = tonumber(v)
      if not kb then return nil end
      return math.max(1, math.floor((kb / 1024 / 1024) + 0.5))
    end
    line = f:read("*l")
  end
  f:close()
  return nil
end

local function detect_cpu_threads()
  -- Linux: /proc/cpuinfo. Retourne un entier (threads logiques) ou nil.
  local f = io.open("/proc/cpuinfo", "r")
  if not f then return nil end
  local n = 0
  for line in f:lines() do
    if line:match("^processor%s*:%s*%d+") then
      n = n + 1
    end
  end
  f:close()
  if n <= 0 then return nil end
  return n
end

local SYS_MEM_GB = detect_mem_total_gb()
local SYS_THREADS = detect_cpu_threads()

local function default_mem_limit_gb()
  -- Heuristique: réserver un peu de RAM pour l'OS + éviter d'aspirer toute la mémoire.
  -- Sur laptop 32GB: ~22-24GB est un bon compromis.
  if not SYS_MEM_GB then return 10 end
  local reserve_gb = 6
  local max_by_reserve = math.max(8, SYS_MEM_GB - reserve_gb)
  local max_by_ratio = math.max(8, math.floor(SYS_MEM_GB * 0.75))
  return math.max(8, math.min(max_by_reserve, max_by_ratio))
end

-- Defaults calibrés automatiquement (surchargés si l'utilisateur passe --mem-gb/--alloc-gb).
local MEM_GB = opt_num("mem-gb", default_mem_limit_gb())
local ALLOC_GB = opt_num("alloc-gb", MEM_GB)

-- Compression: utile quand la limite RAM est basse, coûte un peu de CPU.
-- On auto-désactive si on a une marge mémoire confortable (mais l'utilisateur peut forcer).
local compression_default = (MEM_GB <= 12)
local ENABLE_COMPRESSION = opt_bool("compression", opt_bool("compress", compression_default))

do
  if type(log) == "function" then
    log(string.format("Hardware détecté: mem_total=%sGB cpu_threads=%s", tostring(SYS_MEM_GB), tostring(SYS_THREADS)))
    if CPU_ONLY then
      log("Mode: CPU-only (override via --cpu-only false)")
    end
    if os.getenv("OMP_NUM_THREADS") == nil and SYS_THREADS then
      log(string.format("Tip: pour Ryzen 6c/12t, essaye OMP_NUM_THREADS=%d (ou %d si tu veux garder de la marge).",
        SYS_THREADS, math.max(1, math.floor(SYS_THREADS * 0.75))))
    end
    if opts["mem-gb"] == nil and SYS_MEM_GB then
      log(string.format("Auto mem-gb=%d (override via --mem-gb).", MEM_GB))
    end
  end
end

if Mimir and Mimir.Allocator and Mimir.Allocator.configure then
  Mimir.Allocator.configure({max_ram_gb = ALLOC_GB, enable_compression = ENABLE_COMPRESSION})
end
if Mimir and Mimir.MemoryGuard and Mimir.MemoryGuard.setLimit then
  pcall(Mimir.MemoryGuard.setLimit, MEM_GB)
end
if Mimir and Mimir.Model and Mimir.Model.set_hardware then
  pcall(Mimir.Model.set_hardware, opt_bool("hw", true))
end

local conf_model = (type(CONF) == "table" and type(CONF.model) == "table") and CONF.model or {}
local conf_training = (type(CONF) == "table" and type(CONF.training) == "table") and CONF.training or {}
local conf_dataset = (type(CONF) == "table" and type(CONF.dataset) == "table") and CONF.dataset or {}

local dataset_root = opt_str("dataset-root", conf_dataset.root or "./dataset_2")
local arch = opt_str("arch", opt_str("model", opt_str("model-type", conf_model.architecture or "vae_conv")))
if arch ~= "vae_conv" then
  error("Unknown --arch: " .. tostring(arch) .. " (expected: vae_conv)")
end

local default_out_dir = "checkpoint/vae_conv_512_latent-64-64-32_base-64"
local out_dir = opt_str("out-dir", conf_training.out_dir or default_out_dir)
local output_model_dir = opt_str("output-model", opt_str("output_model", ""))
local save_out_dir = ((output_model_dir ~= nil) and (output_model_dir ~= "")) and output_model_dir or out_dir
local RESUME = opt_bool("resume", conf_training.resume or false)
local epochs = opt_int("epochs", conf_training.num_epochs or 100)
local lr = opt_num("lr", conf_training.learning_rate or 3e-5)
local seed = opt_int("seed", opt_int("init-seed", conf_training.seed or 4242))

-- Hint: si on resume réellement (checkpoint trouvé), on évite de changer
-- la topo du modèle par défaut (sauf si l'utilisateur force explicitement enc_norm).
local resume_dir_hint = nil
if RESUME and Ckpt and Ckpt.resolve_dir then
  resume_dir_hint = Ckpt.resolve_dir(out_dir)
end

local resume_architecture = nil
if resume_dir_hint and type(read_json) == "function" then
  local architecture_path = FS.join(resume_dir_hint, "model", "architecture.json")
  local ok_arch, architecture = pcall(read_json, architecture_path)
  if ok_arch and type(architecture) == "table" then
    resume_architecture = architecture
  end
end

local cfg0, err = Mimir.Architectures.default_config(arch)
assert(type(cfg0) == "table", "default_config(" .. tostring(arch) .. ") failed: " .. tostring(err))
---@cast cfg0 table
local cfg = cfg0
-- En reprise, reconstruire d'abord la configuration enregistrée. Les options
-- du fichier de lancement puis la CLI restent prioritaires ci-dessous.
if resume_architecture and type(resume_architecture.model_config) == "table" then
  for key, value in pairs(resume_architecture.model_config) do
    cfg[key] = value
  end
end
for key, value in pairs(conf_model) do
  if key ~= "architecture" then cfg[key] = value end
end
for key, value in pairs(conf_training) do
  cfg[key] = value
end
cfg.resume = RESUME

-- Un checkpoint legacy peut annoncer `dec_norm=groupnorm` dans model_config
-- alors que son graphe sérialisé ne contient aucune couche de normalisation.
-- Reconstruire le modèle depuis la seule config ajouterait alors des GroupNorm
-- neuves (gamma/beta absents du checkpoint et laissés à zéro), ce qui annule
-- toutes les activations du décodeur après une reprise. La liste des layers est
-- la source de vérité pour la topologie réellement entraînée.
local resume_decoder_norm = nil
if resume_architecture and type(resume_architecture.layers) == "table" then
    resume_decoder_norm = "none"
    for _, layer in ipairs(resume_architecture.layers) do
      local name = tostring(layer.name or "")
      local layer_type = tostring(layer.type or ""):lower()
      if name:match("^vae_conv/dec/") then
        if layer_type == "groupnorm" or name:match("/gn$") then
          resume_decoder_norm = "groupnorm"
          break
        elseif layer_type == "layernorm" or name:match("/ln$") then
          resume_decoder_norm = "layernorm"
        end
      end
    end
    cfg.dec_norm = resume_decoder_norm
end

cfg.image_w = opt_int("image-w", cfg.image_w or 512)
cfg.image_h = opt_int("image-h", cfg.image_h or 512)
cfg.image_c = opt_int("image-c", cfg.image_c or 3)

-- Le snapshot de référence conserve les dimensions du fichier de config. Si
-- seule une dimension d'image est surchargée en CLI, conserver toutefois le
-- facteur x8 au lieu de laisser un latent historique incompatible.
local latent_h_explicit = opts["latent-h"] ~= nil or opts["latent_h"] ~= nil
local latent_w_explicit = opts["latent-w"] ~= nil or opts["latent_w"] ~= nil
local image_h_explicit = opts["image-h"] ~= nil or opts["image_h"] ~= nil
local image_w_explicit = opts["image-w"] ~= nil or opts["image_w"] ~= nil
cfg.latent_h = latent_h_explicit
    and opt_int("latent-h", opt_int("latent_h", cfg.latent_h or 64))
    or (image_h_explicit and math.floor(cfg.image_h / 8) or (cfg.latent_h or 64))
cfg.latent_w = latent_w_explicit
    and opt_int("latent-w", opt_int("latent_w", cfg.latent_w or 64))
    or (image_w_explicit and math.floor(cfg.image_w / 8) or (cfg.latent_w or 64))
cfg.latent_c = opt_int("latent-c", cfg.latent_c or 32)

cfg.base_channels = opt_int("base-channels", cfg.base_channels or 64)

-- Latent topology belongs to the checkpoint on resume. New configuration
-- defaults must never inject untrained branches into a legacy graph.
do
  local saved = resume_architecture and resume_architecture.model_config
  for _, key in ipairs({"latent_spatial_context", "latent_detail_path", "latent_context_channels"}) do
    local flag = key:gsub("_", "-")
    local is_width = key == "latent_context_channels"
    local fallback = is_width and 32 or false
    local checkpoint_value = saved and saved[key]
    if checkpoint_value == nil then checkpoint_value = fallback end
    local default = cfg[key]
    if saved then default = checkpoint_value end
    local value
    if is_width then value = opt_int(flag, opt_int(key, default or 32))
    else value = opt_bool(flag, opt_bool(key, default)) end
    if saved and (opts[flag] ~= nil or opts[key] ~= nil) and value ~= checkpoint_value then
      error("Topologie latente incompatible avec le checkpoint: --" .. flag .. "; démarrer un nouvel entraînement pour activer cette branche.")
    end
    cfg[key] = value
  end
end


-- Upsampling du décodeur. Un nouveau run conserve le défaut historique du
-- script, tandis qu'une reprise réutilise la topologie du checkpoint sauf
-- surcharge explicite dans CONF ou en CLI.
local decoder_upsample_default = "nearest_conv"
if resume_architecture and type(resume_architecture.model_config) == "table" then
  decoder_upsample_default = resume_architecture.model_config.decoder_upsample
      or decoder_upsample_default
end
decoder_upsample_default = conf_model.decoder_upsample
    or conf_training.decoder_upsample
    or decoder_upsample_default
cfg.decoder_upsample = opt_str(
  "decoder-upsample",
  opt_str("decoder_upsample", decoder_upsample_default)
)

do
  local valid_decoder_upsample = {
    conv_transpose = true,
    nearest_conv = true,
    bilinear_conv = true,
    pixel_shuffle = true,
  }
  if not valid_decoder_upsample[cfg.decoder_upsample] then
    error(
      "Upsampling décodeur invalide: " .. tostring(cfg.decoder_upsample) ..
      " (attendu: nearest_conv, bilinear_conv, pixel_shuffle ou conv_transpose)"
    )
  end
end

do
  local function downsample_steps(image_size, latent_size, axis)
    if image_size <= 0 or latent_size <= 0 or image_size % latent_size ~= 0 then
      error(string.format(
        "Géométrie VAEConv invalide sur %s: image=%d, latent=%d; le latent doit diviser l'image.",
        axis, image_size, latent_size
      ))
    end
    local ratio = image_size / latent_size
    local steps = 0
    while ratio > 1 and ratio % 2 == 0 do
      ratio = ratio / 2
      steps = steps + 1
    end
    if ratio ~= 1 then
      error(string.format(
        "Géométrie VAEConv invalide sur %s: image/latent doit être une puissance de 2.", axis
      ))
    end
    return steps
  end

  local width_steps = downsample_steps(cfg.image_w, cfg.latent_w, "la largeur")
  local height_steps = downsample_steps(cfg.image_h, cfg.latent_h, "la hauteur")
  if width_steps ~= height_steps then
    error(string.format(
      "Géométrie VAEConv asymétrique: image=%dx%d, latent=%dx%d donne %d downsamplings en largeur et %d en hauteur; les deux axes doivent avoir le même ratio. " ..
      "Vérifiez notamment que les espaces après --image-w/--image-h sont des espaces ASCII.",
      cfg.image_w, cfg.image_h, cfg.latent_w, cfg.latent_h, width_steps, height_steps
    ))
  end
end

-- Normalisation encodeur (nouveaux layers côté C++)
-- Valeurs: none | layernorm (ln) | groupnorm (gn)
do
  local enc_norm_opt_absent =
      opts["enc-norm"] == nil and
      opts["encoder-norm"] == nil and
      opts["enc_norm"] == nil

  local enc_gn_groups_opt_absent =
      opts["enc-gn-groups"] == nil and
      opts["enc_gn_groups"] == nil

  -- Si l'utilisateur fournit un flag, on le respecte.
  if not enc_norm_opt_absent then
    cfg.enc_norm = opt_str(
      "enc-norm",
      opt_str("encoder-norm", opt_str("enc_norm", cfg.enc_norm or "none"))
    )
  else
    -- Auto-default: activer la normalisation d'encodeur pour les nouveaux runs.
    -- En resume, on laisse le comportement historique pour ne pas changer la topo.
    if not resume_dir_hint then
      cfg.enc_norm = "groupnorm"
    else
      cfg.enc_norm = cfg.enc_norm or "none"
    end
  end

  if not enc_gn_groups_opt_absent then
    cfg.enc_gn_groups = opt_int(
      "enc-gn-groups",
      opt_int("enc_gn_groups", cfg.enc_gn_groups or 32)
    )
  else
    -- Auto: 32 groupes max, mais borné par base_channels (le C++ choisit ensuite
    -- le meilleur diviseur <= enc_gn_groups pour chaque layer).
    local bc = tonumber(cfg.base_channels or 32) or 32
    cfg.enc_gn_groups = math.max(1, math.min(32, bc))
  end
end

-- Normalisation décodeur indépendante de l'encodeur.
-- `--dec-norm none` et `--no-dec-norm` doivent réellement supprimer toutes
-- les couches GroupNorm/LayerNorm du décodeur. Une option absente conserve la
-- valeur du fichier de configuration/checkpoint.
do
  local raw_dec_norm = opts["dec-norm"]
  if raw_dec_norm == nil then raw_dec_norm = opts["decoder-norm"] end
  if raw_dec_norm == nil then raw_dec_norm = opts["dec_norm"] end

  if raw_dec_norm ~= nil then
    if raw_dec_norm == false then
      if resume_decoder_norm ~= nil and resume_decoder_norm ~= "none" then
        error(string.format(
          "Topologie decoder incompatible avec le checkpoint repris: --dec-norm=none, graphe checkpoint=%s. " ..
          "Une reprise stricte doit conserver les couches réellement sérialisées.",
          resume_decoder_norm
        ))
      end
      cfg.dec_norm = "none"
    else
      local value = tostring(raw_dec_norm):lower()
      if value == "off" or value == "false" or value == "disabled" then value = "none" end
      if value == "gn" then value = "groupnorm" end
      if value == "ln" then value = "layernorm" end
      assert(
        value == "none" or value == "groupnorm" or value == "layernorm",
        "--dec-norm invalide: " .. tostring(raw_dec_norm) ..
          " (attendu: none | groupnorm | layernorm)"
      )
      if resume_decoder_norm ~= nil and value ~= resume_decoder_norm then
        error(string.format(
          "Topologie decoder incompatible avec le checkpoint repris: --dec-norm=%s, graphe checkpoint=%s. " ..
          "Une reprise stricte doit conserver les couches réellement sérialisées.",
          value, resume_decoder_norm
        ))
      end
      cfg.dec_norm = value
    end
  end

  cfg.dec_gn_groups = opt_int(
    "dec-gn-groups",
    opt_int("decoder-gn-groups", opt_int("dec_gn_groups", cfg.dec_gn_groups or cfg.enc_gn_groups or 32))
  )
end

-- Blocs ResNet (optionnel). Les anciennes clés restent acceptées à la lecture.
-- Recommandé: --resnet/--use-resnet/--resnet-max-tokens.
-- Pour désactiver: --no-resnet
local configured_resnet = cfg.resnet
if configured_resnet == nil then configured_resnet = cfg.use_resnet end
if configured_resnet == nil then configured_resnet = cfg.use_attention end
if configured_resnet == nil then configured_resnet = true end
cfg.resnet = opt_bool(
  "resnet",
  opt_bool(
    "resnet-blocks",
    opt_bool("use-resnet", configured_resnet)
  )
)

-- Attention (SelfAttention) en plus des blocs ResNet.
-- Pour activer l'attention: --attn (ou --vae-attn/--self-attn).
local configured_attention = cfg.attention
if configured_attention == nil then configured_attention = cfg.use_attn end
cfg.attention = opt_bool(
  "attn",
  opt_bool(
    "vae-attn",
    opt_bool("self-attn", opt_bool("use-attn", opt_bool("use-vae-attn", configured_attention)))
  )
)

-- Sur CPU, l'attention peut être extrêmement lente; si l'utilisateur n'a pas explicitement demandé
-- l'attention, on la désactive par défaut en mode cpu-only.
do
  local attn_opt_absent =
      opts["attn"] == nil and
      opts["vae-attn"] == nil and
      opts["self-attn"] == nil and
      opts["use-attn"] == nil and
      opts["use-vae-attn"] == nil
  if CPU_ONLY and attn_opt_absent then
    cfg.attention = false
  end
end

-- `attn_heads` est utilisé par l'attention si `attention=true`.
cfg.attn_heads = opt_int("attn-heads", cfg.attn_heads or 4)

-- CPU-only: si non fourni, baisser le nombre de heads pour réduire le coût.
do
  if CPU_ONLY and opts["attn-heads"] == nil and (cfg.attention == true) then
    cfg.attn_heads = math.max(1, math.min(2, tonumber(cfg.attn_heads or 2) or 2))
  end
end

-- Garde-fou ResNet (injection de blocs) : séparé de l'attention.
cfg.resnet_max_tokens = opt_int(
  "resnet-max-tokens",
  opt_int("resnet-max", cfg.resnet_max_tokens or 0)
)

-- Garde-fou SelfAttention.
cfg.attn_max_tokens = opt_int(
  "attn-max-tokens",
  opt_int(
    "attn-max",
    opt_int("vae-attn-max-tokens", opt_int("self-attn-max-tokens", cfg.attn_max_tokens or 0))
  )
)

-- Skip connections encodeur→décodeur (style U-Net).
-- Compatible seulement avec les nouveaux checkpoints (incompatible avec les anciens).
cfg.use_skip_connections = opt_bool(
  "skip-connections",
  opt_bool("skip_connections", opt_bool("use-skip-connections", cfg.use_skip_connections or false))
)

-- État initial du contrôle live, indépendant de la présence des skips.
-- Exemple : --skip-connections=true --skip-connections-enabled=false
cfg.skip_connections_enabled = opt_bool(
  "skip-connections-enabled", cfg.skip_connections_enabled ~= false
)

-- Prior appris dans le graphe (couche Constant entraînée par backprop).
-- Quand true: un vecteur biais de dim=latent_dim est ajouté additivement à z.
-- N'utilise pas le ConditioningEncoder externe (pas de risque d'expansion d'embedding).
cfg.use_encoder_prior = opt_bool(
  "encoder-prior",
  opt_bool("use-encoder-prior", opt_bool("enc-prior", cfg.use_encoder_prior or false))
)

-- ConditioningEncoder externe (classe ConditioningEncoder C++): embeddings mag/mod/seq appris, dimension d_model.
-- 0 = désactivé. Quand > 0, le ConditioningEncoder est initialisé, sérialisé avec le checkpoint
-- et ses vecteurs peuvent être utilisés pour conditioning externe.
-- Rec: d_model=1024. Allocation: vocab_size*d_model*4 octets (~200 Mo pour vocab=47895).
cfg.d_model = opt_int(
  "d-model",
  opt_int("d_model", cfg.d_model or 0)
)

-- Latent stochastique (réparameterisation). Important pour éviter un "AE déterministe" pénalisé au KL.
-- NOTE: côté C++, Reparameterize utilise mu directement si training=false OU stochastic_latent=false.
local stochastic_opt_absent =
    opts["stochastic-latent"] == nil and
    opts["stochastic_latent"] == nil and
    opts["vae-stochastic-latent"] == nil and
    opts["vae_stochastic_latent"] == nil

cfg.stochastic_latent = opt_bool(
  "stochastic-latent",
  opt_bool(
    "stochastic_latent",
    opt_bool(
      "vae-stochastic-latent",
      opt_bool("vae_stochastic_latent", cfg.stochastic_latent or false)
    )
  )
)

-- Texte (optionnel)
cfg.text_cond = opt_bool("text-cond", cfg.text_cond or false)
cfg.seq_len = opt_int("seq-len", cfg.seq_len or 64)
cfg.text_d_model = opt_int("text-d-model", cfg.text_d_model or 64)
cfg.proj_dim = opt_int("proj-dim", cfg.proj_dim or (cfg.text_cond and 64 or 0))
cfg.align_weight = opt_num("align-weight", cfg.align_weight or 0.1)
-- Supervision spatiale active par défaut, indépendamment du conditionnement texte.
-- Le runtime affiche automatiquement son icône sur [DS] si la VIZ est ouverte.
cfg.bbox_loss_weight = opt_num("bbox-loss-weight", cfg.bbox_loss_weight or 1.0)
if cfg.bbox_loss_weight ~= cfg.bbox_loss_weight or cfg.bbox_loss_weight == math.huge or cfg.bbox_loss_weight < 0 then
  error("--bbox-loss-weight doit être fini et >= 0")
end

for role, default in pairs({primary=3.0, secondary=2.0, other=1.0}) do
  local key = "relation_" .. role .. "_weight"
  cfg[key] = opt_num("relation-" .. role .. "-weight", cfg[key] or default)
  if cfg[key] ~= cfg[key] or cfg[key] == math.huge or cfg[key] < 0 then
    error("--relation-" .. role .. "-weight doit être fini et >= 0")
  end
end

-- IMPORTANT: base tokenizer commun (si texte activé)
local base_tok_path = opt_str("base-tokenizer", BaseTok.default_path())
do
  -- Tokenizer base: requis pour text-cond (natif ou fallback runtime), optionnel sinon.
  local ok_bt, err_bt = BaseTok.load_base({
    path = base_tok_path,
    max_vocab = opt_int("max-vocab", cfg.vocab_size or 50000),
    require = false,
  })
  assert(ok_bt == true, "Base tokenizer: " .. tostring(err_bt))
end

-- NOTE: text-cond est maintenant supporté nativement par le graphe vae_conv (optionnel).
-- Compat legacy: si un ancien checkpoint image-only est repris avec --text-cond true,
-- le runtime active automatiquement un fallback tokenizer+ConditioningEncoder,
-- sans casser la compatibilité de convergence.
if cfg.text_cond then
  log("ℹ️  vae_conv: --text-cond true active le chemin multi-modal (texte optionnel) pour l'alignement image↔texte.")
  log("   En reprise d'un ancien graphe image-only, un fallback runtime tokenizer+encoder reste disponible.")
end

-- Si latent_h/w non fournis, on dérive (downsample x8 par défaut)
if (cfg.latent_h or 0) <= 0 then cfg.latent_h = math.max(1, math.floor(cfg.image_h / 8)) end
if (cfg.latent_w or 0) <= 0 then cfg.latent_w = math.max(1, math.floor(cfg.image_w / 8)) end

-- Valeurs sûres par défaut pour l'attention.
-- IMPORTANT: côté C++, `attn_max_tokens<=0` désactive le garde-fou (attention partout) => potentiellement très coûteux.
-- Ici, si l'option n'est pas fournie, on choisit un défaut = latent_w*latent_h.
if cfg.attention == true then
  local latent_tokens = math.max(1, (cfg.latent_h or 1) * (cfg.latent_w or 1))

  local attn_opt_absent =
      opts["attn-max-tokens"] == nil and
      opts["attn-max"] == nil and
      opts["vae-attn-max-tokens"] == nil and
      opts["self-attn-max-tokens"] == nil

  if attn_opt_absent then
    cfg.attn_max_tokens = latent_tokens
  end

  if (tonumber(cfg.attn_max_tokens or 0) or 0) <= 0 then
    cfg.attn_max_tokens = latent_tokens
  end
end

-- Défaut sûr: bloc ResNet au bottleneck uniquement.
-- Si l'utilisateur n'a pas fourni --resnet-max-tokens,
-- on force une limite = latent_w*latent_h
-- afin que le bloc soit effectivement injecté au latent (et skip aux upscales trop coûteux).
if cfg.resnet == true then
  local latent_tokens = math.max(1, (cfg.latent_h or 1) * (cfg.latent_w or 1))

  -- Cas 1: option absente -> auto
  if opts["resnet-max-tokens"] == nil and opts["resnet-max"] == nil then
    cfg.resnet_max_tokens = latent_tokens
  end

  -- Cas 2: option présente mais invalide / <= 0 (ex: `--resnet-max-tokens 0` ou `--resnet-max-tokens auto`)
  -- => on retombe sur l'auto.
  if (tonumber(cfg.resnet_max_tokens or 0) or 0) <= 0 then
    cfg.resnet_max_tokens = cfg.latent_w * cfg.latent_h
  end
end

-- Alerte UX: ResNet activé mais gate trop basse => aucun bloc résiduel injecté dans le graphe.
do
  local tokens = math.max(1, (cfg.latent_h or 1) * (cfg.latent_w or 1))
  local max_t = tonumber(cfg.resnet_max_tokens or 0) or 0
  if cfg.resnet == true and max_t > 0 and max_t < tokens then
    log(string.format("⚠️  Blocs ResNet activés mais skippés (latent tokens=%d > resnet_max_tokens=%d). Augmente --resnet-max-tokens (ex: %d) ou baisse latent_h/latent_w.",
      tokens, max_t, tokens))
  end
end

-- Renseigne `latent_dim` pour éviter l'inférence ambiguë (et garder la compat training/inférence).
cfg.latent_dim = math.max(1, (cfg.latent_h or 0) * (cfg.latent_w or 0) * (cfg.latent_c or 0))

-- Options d'entraînement consommées côté C++ (LuaScripting.cpp)
cfg.optimizer = opt_str("optimizer", cfg.optimizer or "adamw")
cfg.beta1 = opt_num("beta1", cfg.beta1 or 0.9)
cfg.beta2 = opt_num("beta2", cfg.beta2 or 0.999)
cfg.epsilon = opt_num("epsilon", cfg.epsilon or 1e-8)
-- VAEConv: le weight decay trop fort dégrade souvent la reconstruction.
cfg.weight_decay = opt_num("weight-decay", cfg.weight_decay or 1e-8)

cfg.decay_strategy = opt_str("decay-strategy", cfg.decay_strategy or "cosine")

cfg.kl_beta = opt_num("kl-beta", cfg.kl_beta or 0.5)
-- Stabilisation VAE (consommée côté C++ par le hook Model::trainStep)
-- Par défaut: ramp-up du KL sur ~1/2 époque (dataset ~1967 linkables)
cfg.kl_warmup_steps = opt_int(
  "kl-warmup-steps",
  opt_int("kl-warmup", opt_int("kl_warmup", cfg.kl_warmup_steps or 900))
)

-- Auto-sécurité: si l'utilisateur n'a pas explicitement fixé stochastic_latent
-- et qu'on entraîne avec KL>0, un latent déterministe a de fortes chances de
-- collaps-er (mu→0) et le décodeur "ignore" l'information.
-- En resume, on évite de changer ce comportement implicitement.
if stochastic_opt_absent and (not resume_dir_hint) and ((cfg.kl_beta or 0) > 0) then
  cfg.stochastic_latent = true
end

-- Recon loss (consommé côté C++ par le hook Model::trainStep)
cfg.recon_loss = opt_str("recon-loss", cfg.recon_loss or "charbonnier")

-- Losses additionnelles (optionnelles)
cfg.ssim_weight = opt_num("ssim-weight", cfg.ssim_weight or 0.0)
cfg.ssim_mode = opt_str("ssim-mode", cfg.ssim_mode or "ssim") -- "ssim" ou "ms_ssim"
cfg.ssim_k1 = opt_num("ssim-k1", cfg.ssim_k1 or 0.01)
cfg.ssim_k2 = opt_num("ssim-k2", cfg.ssim_k2 or 0.03)
cfg.ssim_L = opt_num("ssim-L", cfg.ssim_L or 1.2)

cfg.spectral_weight = opt_num("spectral-weight", cfg.spectral_weight or 0.0)
cfg.spectral_scales = opt_int("spectral-scales", cfg.spectral_scales or 1)

-- Perceptual loss: utilise par défaut le vgg16_feat pré-entraîné par
-- scripts/training/pretrain_vgg16_feat.lua.
cfg.perceptual_weight = opt_num("perceptual-weight", 0.0)
cfg.perceptual_arch = opt_str("perceptual-arch", cfg.perceptual_arch or "vgg16_feat")
do
  local default_pckpt = cfg.perceptual_checkpoint
  if default_pckpt == nil or tostring(default_pckpt) == "" then
    default_pckpt = "./checkpoint/vgg16_feat_pretrain"
  end
  cfg.perceptual_checkpoint = opt_str("perceptual-ckpt", default_pckpt)
end
cfg.perceptual_base_channels = opt_int("perceptual-base-channels", cfg.perceptual_base_channels or 16)

-- Compat: `vgg16_feat` force base_channels>=4 côté C++.
-- Ici on aligne dès le script pour éviter warnings/rebuild inutiles.
do
  local parch = tostring(cfg.perceptual_arch or ""):lower()
  if parch == "vgg16_feat" then
    if (tonumber(cfg.perceptual_base_channels or 0) or 0) < 4 then
      cfg.perceptual_base_channels = 8
    end
  end
end

-- Compat UX: permettre `--perceptual-ckpt checkpoint/vgg16_feat_pretrain`
-- (résout vers le dernier epoch disponible). On lit aussi base_channels dans
-- l'architecture sauvegardée afin que le modèle auxiliaire corresponde exactement
-- aux poids pré-entraînés, sauf override explicite de l'utilisateur.
do
  local p = tostring(cfg.perceptual_checkpoint or "")
  local perceptual_enabled = (tonumber(cfg.perceptual_weight or 0) or 0) > 0
  local base_channels_explicit = opts["perceptual-base-channels"] ~= nil

  if #p > 0 and Ckpt then
    local resolver = Ckpt.resolve_dir_prefer_final or Ckpt.resolve_dir
    local resolved = resolver and resolver(p) or nil
    if resolved then
      cfg.perceptual_checkpoint = resolved
      if not base_channels_explicit then
        local arch_path = FS.join(resolved, "model", "architecture.json")
        local f = io.open(arch_path, "r")
        if f then
          local json = f:read("*a") or ""
          f:close()
          local layer = json:match('"name"%s*:%s*"vgg16_feat/b1/c1"(.-)}')
          local inferred = layer and tonumber(layer:match('"out_channels"%s*:%s*(%d+)')) or nil
          if inferred and inferred >= 4 then
            cfg.perceptual_base_channels = inferred
          elseif perceptual_enabled then
            error("Cannot infer perceptual base_channels from " .. arch_path)
          end
        elseif perceptual_enabled then
          error("Missing perceptual architecture: " .. arch_path)
        end
      end
    elseif perceptual_enabled then
      error("Perceptual loss is enabled but no loadable checkpoint was found at: " .. p)
    end
  elseif perceptual_enabled then
    error("Perceptual loss is enabled but --perceptual-ckpt is empty")
  end
end

-- Si la perception est utilisee, le prior latent est obligatoire: ses poids
-- seront hydrates par EMA avec les features du modele perceptuel.
if (tonumber(cfg.perceptual_weight or 0) or 0) > 0 then
  cfg.use_encoder_prior = true
end
cfg.perceptual_prior_momentum = opt_num("perceptual-prior-momentum", cfg.perceptual_prior_momentum or 0.95)
cfg.perceptual_prior_scale = opt_num("perceptual-prior-scale", cfg.perceptual_prior_scale or 0.05)

-- Paramètres recon loss
cfg.huber_delta = opt_num("huber-delta", cfg.huber_delta or 1.0)
cfg.charbonnier_eps = opt_num("charbonnier-eps", cfg.charbonnier_eps or 3e-5)
cfg.nll_sigma = opt_num("nll-sigma", cfg.nll_sigma or 1.0)

-- Warmup LR (consommé côté C++ via Optimizer.warmup_steps)
-- Important: le scheduler applique le warmup sur `opt.initial_lr`.
cfg.warmup_steps = opt_int("lr-warmup-steps", opt_int("warmup-steps", cfg.warmup_steps or 500))

-- Autosave (consommé côté C++ dans LuaScripting::lua_trainModel)
-- 0 = désactiver. 1 = sauvegarde à chaque epoch.
local autosave_default = tonumber(
  conf_training.autosave_every_epochs or conf_training.autosave_every_epoch or
  conf_model.autosave_every_epochs or conf_model.autosave_every_epoch)
if autosave_default == nil then autosave_default = 1 end
cfg.autosave_every_epochs = opt_int(
  "autosave-every-epochs", opt_int("autosave_every_epochs", autosave_default))

-- Marqueurs (Wasserstein/Temporal) qui modulent la loss de reconstruction côté C++.
-- Par défaut: désactivé (0.0) pour conserver un training identique.
cfg.marker_wass_scale = opt_num("marker-wass-scale", cfg.marker_wass_scale or 1.0)
cfg.marker_temp_scale = opt_num("marker-temp-scale", cfg.marker_temp_scale or 1.0)
cfg.marker_warmup_steps = opt_int("marker-warmup-steps", cfg.marker_warmup_steps or 0)
cfg.marker_scale_max = opt_num("marker-scale-max", cfg.marker_scale_max or 1.0)
-- Clamp logvar plus serré => std dans ~[exp(-3), exp(1)] = [0.05, 2.7]
-- Clamp logvar (log(variance)). Pour un VAE destiné à servir de backbone à un modèle
-- de diffusion, on évite des std trop grands (latents trop bruités).
-- std = exp(0.5*logvar) => logvar_max=0 => std_max=1.
cfg.logvar_clip_min = opt_num("logvar-clip-min", cfg.logvar_clip_min or -6.0)
cfg.logvar_clip_max = opt_num("logvar-clip-max", cfg.logvar_clip_max or 1.0)
-- Clip grad global (L2) pour éviter un emballement; 1.0 est souvent trop agressif
-- sur des modèles/étapes avec pertes additionnelles (SSIM/perceptual).
cfg.grad_clip_norm = opt_num("grad-clip-norm", cfg.grad_clip_norm or 1.5)
cfg.grad_accum_steps = opt_int("grad-accum-steps", cfg.grad_accum_steps or 1)

cfg.max_items = opt_int("max-items", cfg.max_items or 0)
cfg.log_every = opt_int("log-every", cfg.log_every or 1)

-- IMPORTANT: utilisé côté C++ pour le shuffle/ordre dataset.
cfg.seed = seed

-- Viz taps (consommés côté C++ si viz active)
cfg.viz_taps_max_frames = opt_int("viz-taps-max-frames", cfg.viz_taps_max_frames or 200)
cfg.viz_taps_max_side = opt_int("viz-taps-max-side", cfg.viz_taps_max_side or 1024)
cfg.viz_taps_force_inference = opt_bool(
  "viz-taps-force-inference",
  opt_bool("viz_taps_force_inference", cfg.viz_taps_force_inference or false)
)

-- Checkpoints/validation (consommés côté C++ dans Mimir.Model.train)
cfg.checkpoint_dir = save_out_dir

-- Chemin CSV explicite (--csv-file / --csv-path / --csv_path).
-- Consommé par le bloc commun de lua_trainModel (tous les types de modèles).
do
  local csv_f = opt_str("csv-file", opt_str("csv-path", nil))
  local csv_d = opt_str("csv-dir", nil)
  if csv_f and csv_f ~= "" then cfg.csv_file = csv_f end
  if csv_d and csv_d ~= "" then cfg.csv_dir  = csv_d end
end

Args.apply_validation_config(cfg, opts, {
  validate_every_steps = 50,
  validate_items = 6,
  validate_holdout_frac = 0.1,
})
cfg.fpga_validation_enabled = opt_bool(
  "fpga-validation",
  cfg.fpga_validation_enabled ~= nil and cfg.fpga_validation_enabled or true
)
cfg.fpga_validation_every_steps = opt_int(
  "fpga-validation-every-steps",
  cfg.fpga_validation_every_steps or cfg.validate_every_steps
)

-- Détection automatique: "VAE backbone prêt" (pour diffusion text→image)
-- NOTE: dépend de la validation (validate_every_steps/items). Si la validation est désactivée,
-- le C++ désactive automatiquement la readiness.
cfg.backbone_ready = opt_bool("backbone-ready", opt_bool("vae-ready", cfg.backbone_ready ~= nil and cfg.backbone_ready or true))
cfg.backbone_ready_enabled = opt_bool("backbone-ready-enabled", cfg.backbone_ready)
cfg.backbone_ready_stop = opt_bool("backbone-ready-stop", opt_bool("vae-ready-stop", cfg.backbone_ready_stop or true))
cfg.backbone_ready_window = opt_int("backbone-ready-window", cfg.backbone_ready_window or 5)
cfg.backbone_ready_plateau_rel = opt_num("backbone-ready-plateau-rel", cfg.backbone_ready_plateau_rel or 0.01)
cfg.backbone_ready_plateau_abs = opt_num("backbone-ready-plateau-abs", cfg.backbone_ready_plateau_abs or 1e-4)
cfg.backbone_ready_recon_target = opt_num("backbone-ready-recon-target", cfg.backbone_ready_recon_target or 0.02)
cfg.backbone_ready_kl_min = opt_num("backbone-ready-kl-min", cfg.backbone_ready_kl_min or 0.01)
cfg.backbone_ready_kl_max = opt_num("backbone-ready-kl-max", cfg.backbone_ready_kl_max or 5.0)
cfg.backbone_ready_min_steps = opt_int("backbone-ready-min-steps", cfg.backbone_ready_min_steps or (cfg.kl_warmup_steps or 0))
cfg.backbone_ready_file = opt_str("backbone-ready-file", cfg.backbone_ready_file or save_out_dir .. "/vae_backbone_ready.json")

cfg.triple_fault = opt_bool("triple-fault", false)
cfg.triple_fault_every_steps = opt_int("fault-every", opt_int("triple-fault-every", 5))
cfg.dtype = opt_str("dtype", os.getenv("MIMIR_DTYPE") or "float32")


local function format_cfg_value(value, seen)
  local value_type = type(value)
  if value_type == "string" then return string.format("%q", value) end
  if value_type ~= "table" then return tostring(value) end

  seen = seen or {}
  if seen[value] then return "<cycle>" end
  seen[value] = true

  local keys = {}
  for key in pairs(value) do keys[#keys + 1] = key end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)

  local entries = {}
  for _, key in ipairs(keys) do
    entries[#entries + 1] = string.format(
      "[%s]=%s",
      format_cfg_value(key, seen),
      format_cfg_value(value[key], seen)
    )
  end
  seen[value] = nil
  return "{" .. table.concat(entries, ", ") .. "}"
end

log("VAEConv train config :")
log(string.format("  - dataset_root=%s", dataset_root))
log(string.format("  - out_dir(resume_from)=%s", out_dir))
log(string.format("  - output_model(save_to)=%s", save_out_dir))
log(string.format("  - epochs=%d", epochs))
log(string.format("  - learning_rate=%g", lr))
if cfg.bbox_loss_weight > 0 then
  log(string.format("  - bbox: activées (poids=%g), icône [DS] automatique dans VIZ", cfg.bbox_loss_weight))
else
  log("  - bbox: désactivées (poids=0)")
end
log(string.format("  - base_tokenizer=%s", base_tok_path))
log("  - cfg:")
do
  local cfg_keys = {}
  for key in pairs(cfg) do cfg_keys[#cfg_keys + 1] = key end
  table.sort(cfg_keys, function(a, b) return tostring(a) < tostring(b) end)
  for _, key in ipairs(cfg_keys) do
    log(string.format("      %s = %s", tostring(key), format_cfg_value(cfg[key])))
  end
end

-- Heuristiques de qualité: si le latent est trop petit ou si KL+latent déterministe => détails lissés.
do
  local image_dim = (cfg.image_w or 0) * (cfg.image_h or 0) * (cfg.image_c or 0)
  local latent_dim = (cfg.latent_h or 0) * (cfg.latent_w or 0) * (cfg.latent_c or 0)
  if image_dim > 0 and latent_dim > 0 then
    local ratio = latent_dim / image_dim
    if ratio < 0.05 then
      log(string.format("⚠️  Latent très compressé: latent_dim=%d (%.3g x image_dim=%d). Risque de perte de détails.", latent_dim, ratio, image_dim))
    end
  end
  if (cfg.stochastic_latent == false) and ((cfg.kl_beta or 0) > 0) then
    log("⚠️  stochastic_latent=false avec KL>0: le KL pousse mu→0 et lisse les détails. Essaye --stochastic-latent true, ou baisse --kl-beta (ex: 0.01) si tu veux garder un encodeur quasi-déterministe.")
  end
end

-- Dataset
local ok_ds, n_or_err = Mimir.Dataset.load(dataset_root, cfg.image_w, cfg.image_h, cfg.text_cond and 2 or 1, true, 'dataset_cache.json', 10240, true)
assert_ok(ok_ds, n_or_err, "Dataset.load failed")
if (tonumber(n_or_err) or 0) == 0 then
  local min_mod = cfg.text_cond and 2 or 1
  error(string.format(
    "Dataset vide: 0 item chargé depuis '%s' (min_modalities=%d).\n" ..
    "  - Vérifiez que le dossier existe et contient des images.\n" ..
    "  - Avec --text-cond true (min_modalities=2), chaque image doit avoir une annotation .json ou un fichier .txt " ..
    "avec le même stem dans le même répertoire.\n" ..
    "  - Structure attendue: dataset_root/<stem>.jpg + <stem>.json (ou <stem>.txt)",
    dataset_root, min_mod
  ))
end
log("✓ Dataset chargé: " .. tostring(n_or_err) .. " items")



-- Modèle
assert_ok(Mimir.Model.create(arch, cfg), nil, "Model.create(" .. tostring(arch) .. ") failed")
apply_dtype(cfg)

local params = Mimir.Model.total_params()
log("✓ Model créé (registry): params=" .. tostring(params))

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
assert_ok(ok_alloc, err_alloc, "Model.allocate_params failed")

local resumed_from = nil
if RESUME and Ckpt and Ckpt.resolve_dir then
  local resume_dir = Ckpt.resolve_dir(out_dir)
  if resume_dir then
    log("↩︎ Resume: chargement checkpoint: " .. tostring(resume_dir))
    local load_opts = {
      load_encoder = true,
      load_tokenizer = true,
      load_optimizer = true,
      strict_mode = true,
      validate_checksums = true
    }
    local ok_load, err_load = Mimir.Serialization.load(resume_dir, "raw_folder", load_opts)
    assert_ok(ok_load, err_load, "Serialization.load(resume) failed")
    resumed_from = resume_dir
  end
end

if not resumed_from then
  local init_method = opt_str("init", "xavier")
  local ok_init, err_init = Mimir.Model.init_weights(init_method, seed)
  assert_ok(ok_init, err_init, "Model.init_weights failed")
end

-- Sauvegarde debug JSON juste avant l'entraînement (snapshot du modèle + config).
-- NOTE: format côté C++/Lua = "debug_json" (alias: "debug", "json").
do
  local starttrain_path = save_out_dir .. "/starttrain.json"
  FS.mkdir_p(save_out_dir)
  local ok_dbg, err_dbg = Mimir.Serialization.save(starttrain_path, "debug_json", {
    save_tokenizer = true,
    save_encoder = true,
    include_git_info = true,
  })
  assert_ok(ok_dbg, err_dbg, "Serialization.save(starttrain.json, debug_json) failed")
  log("✓ Debug JSON écrit: " .. starttrain_path)
end

-- Entraînement
local ok_train, err_train = Mimir.Model.train(epochs, lr)
if ok_train == false and tostring(err_train) == "STOP_REQUESTED" then
  log("⛔ Stop demandé via Viz: autosave effectué. Sauvegarde finale (complète) puis fin du programme.")

  -- On ré-écrit dans le dernier dossier epoch_* (incluant *_stop si présent)
  local last_dir = nil
  if Ckpt and Ckpt.find_latest_epoch_dir then
    last_dir = Ckpt.find_latest_epoch_dir(save_out_dir)
  end
  if not last_dir and Ckpt and Ckpt.resolve_dir then
    last_dir = Ckpt.resolve_dir(save_out_dir)
  end
  if not last_dir then
    last_dir = save_out_dir
  end

  FS.mkdir_p(last_dir)
  local ok_save_stop, err_save_stop = Mimir.Serialization.save(last_dir, "raw_folder", {
    save_optimizer = true,
    save_tokenizer = cfg.text_cond,
    save_encoder = false,
    include_checksums = true,
    include_git_info = true,
    include_gradients = true,
    include_activations = true,
    include_optimizer_state = true,
    include_weight_deltas = true,
  })
  assert_ok(ok_save_stop, err_save_stop, "Serialization.save(stop) failed")
  log("✓ Checkpoint STOP écrit: " .. tostring(last_dir))

  -- Snapshot debug JSON de fin d'entraînement (cas STOP).
  do
    local endtrain_path = last_dir .. "/endtrain.json"
    local ok_dbg_end, err_dbg_end = Mimir.Serialization.save(endtrain_path, "debug_json", {
      save_optimizer = true,
      save_tokenizer = true,
      save_encoder = true,
      include_checksums = true,
      include_git_info = true,
      include_gradients = true,
      include_activations = true,
      include_optimizer_state = true,
      include_weight_deltas = true,
    })
    assert_ok(ok_dbg_end, err_dbg_end, "Serialization.save(endtrain.json, debug_json) failed")
    log("✓ Debug JSON écrit: " .. endtrain_path)
  end

  -- Export SafeTensors de fin d'entraînement (cas STOP).
  do
    local st_path = last_dir .. "/model.safetensors"
    local ok_st, err_st = Mimir.Serialization.save(st_path, "safetensors", {
      save_optimizer = true,
      save_tokenizer = true,
      save_encoder = true,
      include_checksums = true,
      include_git_info = true,
      include_gradients = true,
      include_activations = true,
      include_optimizer_state = true,
      include_weight_deltas = true,
    })
    assert_ok(ok_st, err_st, "Serialization.save(model.safetensors) failed")
    log("✓ SafeTensors écrit: " .. st_path)
  end
  return
end
assert_ok(ok_train, err_train, "Model.train failed")

-- Sauvegarde
FS.mkdir_p(save_out_dir)

-- Snapshot debug JSON de fin d'entraînement (run normal).
do
  local endtrain_path = save_out_dir .. "/endtrain.json"
  local ok_dbg_end, err_dbg_end = Mimir.Serialization.save(endtrain_path, "debug_json", {
    save_optimizer = true,
    save_tokenizer = false,
    save_encoder = true,
    include_checksums = true,
    include_git_info = true,
    include_gradients = true,
    include_activations = true,
    include_optimizer_state = true,
    include_weight_deltas = true,
  })
  assert_ok(ok_dbg_end, err_dbg_end, "Serialization.save(endtrain.json, debug_json) failed")
  log("✓ Debug JSON écrit: " .. endtrain_path)
end

-- Export SafeTensors de fin d'entraînement (run normal).
do
  local st_path = save_out_dir .. "/model.safetensors"
  local ok_st, err_st = Mimir.Serialization.save(st_path, "safetensors", {
    save_optimizer = false,
    save_tokenizer = false,
    save_encoder = false,
    include_checksums = true,
    include_git_info = true,
    include_gradients = false,
    include_activations = true,
    include_optimizer_state = false,
    include_weight_deltas = false,
  })
  assert_ok(ok_st, err_st, "Serialization.save(model.safetensors) failed")
  log("✓ SafeTensors écrit: " .. st_path)
end

local ok_save, err_save = Mimir.Serialization.save(save_out_dir, "raw_folder", {
  save_optimizer = true,
  save_tokenizer = false,
  save_encoder = false,
  include_checksums = true,
  include_git_info = true,
  include_gradients = true,
  include_activations = true,
  include_optimizer_state = true,
  include_weight_deltas = true,
})
assert_ok(ok_save, err_save, "Serialization.save failed")

log("✓ Checkpoint écrit: " .. save_out_dir)
