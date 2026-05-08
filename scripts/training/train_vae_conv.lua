---@diagnostic disable: undefined-global, undefined-field, inject-field
local Args = dofile("scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local Ckpt = dofile("scripts/modules/checkpoint_resume.lua")

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
  if v == nil then return d end
  if v == true or v == false then return v end
  v = tostring(v):lower()
  if v == "1" or v == "true" or v == "yes" or v == "on" then return true end
  if v == "0" or v == "false" or v == "no" or v == "off" then return false end
  return d
end

local BaseTok = dofile("scripts/modules/base_tokenizer.lua")

local function assert_ok(ok, err, msg)
  if ok == false then
    error((msg or "Operation failed") .. ": " .. tostring(err))
  end
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

local dataset_root = opt_str("dataset-root", "./dataset_2")
local arch = opt_str("arch", opt_str("model", opt_str("model-type", "vae_conv")))
if arch ~= "vae_conv" then
  error("Unknown --arch: " .. tostring(arch) .. " (expected: vae_conv)")
end

local default_out_dir = "checkpoint/vae_conv_512_latent-64-64-32_base-64"
local out_dir = opt_str("out-dir", default_out_dir)
local RESUME = opt_bool("resume", false)
local epochs = opt_int("epochs", 100)
local lr = opt_num("lr", 3e-5)
local seed = opt_int("seed", opt_int("init-seed", 4242))

-- Hint: si on resume réellement (checkpoint trouvé), on évite de changer
-- la topo du modèle par défaut (sauf si l'utilisateur force explicitement enc_norm).
local resume_dir_hint = nil
if RESUME and Ckpt and Ckpt.resolve_dir then
  resume_dir_hint = Ckpt.resolve_dir(out_dir)
end

local cfg0, err = Mimir.Architectures.default_config(arch)
assert(type(cfg0) == "table", "default_config(" .. tostring(arch) .. ") failed: " .. tostring(err))
---@cast cfg0 table
local cfg = cfg0

cfg.image_w = opt_int("image-w", 512)
cfg.image_h = opt_int("image-h", 512)
cfg.image_c = opt_int("image-c", 3)

cfg.latent_h = opt_int("latent-h", 64)
cfg.latent_w = opt_int("latent-w", 64)
cfg.latent_c = opt_int("latent-c", 32)

cfg.base_channels = opt_int("base-channels", 64)

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

-- Blocs ResNet (ex-attention) (optionnel)
-- NOTE: historiquement ce flag activait des blocs d'attention; il active maintenant des blocs ResNet.
-- Recommandé: --resnet/--use-resnet/--resnet-max-tokens.
-- Pour désactiver: --no-resnet
cfg.use_attention = opt_bool(
  "resnet",
  opt_bool(
    "resnet-blocks",
    opt_bool("use-resnet", cfg.use_attention or true)
  )
)

-- Attention (SelfAttention) en plus des blocs ResNet.
-- Pour activer l'attention: --attn (ou --vae-attn/--self-attn).
cfg.use_attn = opt_bool(
  "attn",
  opt_bool(
    "vae-attn",
    opt_bool("self-attn", opt_bool("use-attn", opt_bool("use-vae-attn", (cfg.use_attn ~= nil) and cfg.use_attn or true)))
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
    cfg.use_attn = false
  end
end

-- `attn_heads` est utilisé par l'attention si `use_attn=true`.
cfg.attn_heads = opt_int("attn-heads", cfg.attn_heads or 4)

-- CPU-only: si non fourni, baisser le nombre de heads pour réduire le coût.
do
  if CPU_ONLY and opts["attn-heads"] == nil and (cfg.use_attn == true) then
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

-- IMPORTANT: base tokenizer commun (si texte activé)
local base_tok_path = opt_str("base-tokenizer", BaseTok.default_path())
do
  local ok_bt, err_bt = BaseTok.load_base({
    path = base_tok_path,
    max_vocab = opt_int("max-vocab", cfg.vocab_size or 50000),
    require = cfg.text_cond == true,
  })
  assert(ok_bt == true, "Base tokenizer: " .. tostring(err_bt))
end

if cfg.text_cond then
  cfg.vocab_size = BaseTok.vocab_size()
  cfg.tokenizer_frozen = true

  -- IMPORTANT: `Model::trainStepVAEText` exige les projections (img_proj + text_proj)
  -- donc proj_dim doit être > 0.
  if (cfg.proj_dim or 0) <= 0 then
    cfg.proj_dim = 64
  end
end

-- Si latent_h/w non fournis, on dérive (downsample x8 par défaut)
if (cfg.latent_h or 0) <= 0 then cfg.latent_h = math.max(1, math.floor(cfg.image_h / 8)) end
if (cfg.latent_w or 0) <= 0 then cfg.latent_w = math.max(1, math.floor(cfg.image_w / 8)) end

-- Valeurs sûres par défaut pour l'attention.
-- IMPORTANT: côté C++, `attn_max_tokens<=0` désactive le garde-fou (attention partout) => potentiellement très coûteux.
-- Ici, si l'option n'est pas fournie, on choisit un défaut = latent_w*latent_h.
if cfg.use_attn == true then
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
if cfg.use_attention == true then
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

-- Alerte UX: use_attention activé mais gate trop basse => aucun bloc ResNet injecté dans le graph.
do
  local tokens = math.max(1, (cfg.latent_h or 1) * (cfg.latent_w or 1))
  local max_t = tonumber(cfg.resnet_max_tokens or 0) or 0
  if cfg.use_attention == true and max_t > 0 and max_t < tokens then
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
-- Stabilisation VAE (consommée côté C++ par Model::trainStepVAE)
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

-- Recon loss (consommé côté C++ par Model::trainStepVAE)
cfg.recon_loss = opt_str("recon-loss", cfg.recon_loss or "charbonnier")

-- Losses additionnelles (optionnelles)
cfg.ssim_weight = opt_num("ssim-weight", cfg.ssim_weight or 0.0)
cfg.ssim_mode = opt_str("ssim-mode", cfg.ssim_mode or "ms_ssim") -- "ssim" ou "ms_ssim"
cfg.ssim_k1 = opt_num("ssim-k1", cfg.ssim_k1 or 0.01)
cfg.ssim_k2 = opt_num("ssim-k2", cfg.ssim_k2 or 0.03)
cfg.ssim_L = opt_num("ssim-L", cfg.ssim_L or 1.2)

cfg.spectral_weight = opt_num("spectral-weight", cfg.spectral_weight or 0.05)
cfg.spectral_scales = opt_int("spectral-scales", cfg.spectral_scales or 1)

-- Perceptual loss: désactivée par défaut (coûteuse et peut compliquer le debug).
-- Réactiver explicitement via `--perceptual-weight <valeur>`.
cfg.perceptual_weight = opt_num("perceptual-weight", 0.0)
cfg.perceptual_arch = opt_str("perceptual-arch", cfg.perceptual_arch or "vgg16_feat")
do
  local default_pckpt = cfg.perceptual_checkpoint
  if default_pckpt == nil or tostring(default_pckpt) == "" then
    default_pckpt = "./checkpoint/vgg16_feat_pretrain"
  end
  cfg.perceptual_checkpoint = opt_str("perceptual-ckpt", default_pckpt)
end
cfg.perceptual_base_channels = opt_int("perceptual-base-channels", cfg.perceptual_base_channels or 4)

-- Compat: `vgg16_feat` force base_channels>=4 côté C++.
-- Ici on aligne dès le script pour éviter warnings/rebuild inutiles.
do
  local parch = tostring(cfg.perceptual_arch or ""):lower()
  if parch == "vgg16_feat" then
    if (tonumber(cfg.perceptual_base_channels or 0) or 0) < 4 then
      cfg.perceptual_base_channels = 4
    end
  end
end

-- Compat UX: permettre `--perceptual-ckpt checkpoint/vgg16_feat_pretrain` (résout vers final/ ou epoch_*).
do
  local p = tostring(cfg.perceptual_checkpoint or "")
  if #p > 0 and Ckpt then
    local resolver = Ckpt.resolve_dir_prefer_final or Ckpt.resolve_dir
    local resolved = resolver and resolver(p) or nil
    if resolved then
      cfg.perceptual_checkpoint = resolved
    end
  end
end

-- Paramètres recon loss
cfg.huber_delta = opt_num("huber-delta", cfg.huber_delta or 1.0)
cfg.charbonnier_eps = opt_num("charbonnier-eps", cfg.charbonnier_eps or 3e-5)
cfg.nll_sigma = opt_num("nll-sigma", cfg.nll_sigma or 1.0)

-- Warmup LR (consommé côté C++ via Optimizer.warmup_steps)
-- Important: le scheduler applique le warmup sur `opt.initial_lr`.
cfg.warmup_steps = opt_int("lr-warmup-steps", opt_int("warmup-steps", cfg.warmup_steps or 500))

-- Autosave (consommé côté C++ dans LuaScripting::lua_trainModel)
-- 0 = désactiver. 1 = sauvegarde à chaque epoch.
cfg.autosave_every_epochs = opt_int("autosave-every-epochs", opt_int("autosave_every_epochs", cfg.autosave_every_epochs or 1))

-- Marqueurs (Wasserstein/Temporal) qui modulent la loss de reconstruction côté C++.
-- Par défaut: désactivé (0.0) pour conserver un training identique.
cfg.marker_wass_scale = opt_num("marker-wass-scale", cfg.marker_wass_scale or 0.01)
cfg.marker_temp_scale = opt_num("marker-temp-scale", cfg.marker_temp_scale or 0.01)
cfg.marker_warmup_steps = opt_int("marker-warmup-steps", cfg.marker_warmup_steps or 1)
cfg.marker_scale_max = opt_num("marker-scale-max", cfg.marker_scale_max or 1.0)
-- Clamp logvar plus serré => std dans ~[exp(-3), exp(1)] = [0.05, 2.7]
-- Clamp logvar (log(variance)). Pour un VAE destiné à servir de backbone à un modèle
-- de diffusion, on évite des std trop grands (latents trop bruités).
-- std = exp(0.5*logvar) => logvar_max=0 => std_max=1.
cfg.logvar_clip_min = opt_num("logvar-clip-min", cfg.logvar_clip_min or -6.0)
cfg.logvar_clip_max = opt_num("logvar-clip-max", cfg.logvar_clip_max or 0.0)
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
cfg.checkpoint_dir = out_dir

Args.apply_validation_config(cfg, opts, {
  validate_every_steps = 50,
  validate_items = 6,
  validate_holdout_frac = 0.1,
})

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
cfg.backbone_ready_file = opt_str("backbone-ready-file", cfg.backbone_ready_file or out_dir.."vae_backbone_ready.json")

cfg.triple_fault = opt_bool("triple-fault", false)
cfg.triple_fault_every_steps = opt_int("fault-every", opt_int("triple-fault-every", 5))


log("VAEConv train config :")
log(string.format("  - dataset_root=%s", dataset_root))
log(string.format("  - out_dir=%s", out_dir))
log(string.format("  - image=%dx%dx%d", cfg.image_w, cfg.image_h, cfg.image_c))
log(string.format("  - latent=%dx%dx%d", cfg.latent_h, cfg.latent_w, cfg.latent_c))
log(string.format("  - base_channels=%d", cfg.base_channels))
log(string.format("  - enc_norm=%s enc_gn_groups=%d",
  tostring(cfg.enc_norm or "none"),
  tonumber(cfg.enc_gn_groups or 0) or 0))
log(string.format("  - resnet_blocks=%s max_tokens=%d",
  tostring(cfg.use_attention),
  tonumber(cfg.resnet_max_tokens or 0) or 0))
log(string.format("  - attn=%s heads=%d max_tokens=%d",
  tostring(cfg.use_attn),
  tonumber(cfg.attn_heads or 0) or 0,
  tonumber(cfg.attn_max_tokens or 0) or 0))
log(string.format("  - stochastic_latent=%s", tostring(cfg.stochastic_latent)))
if cfg.text_cond then
  log(string.format("  - text_cond=true seq_len=%d vocab_size=%d text_d_model=%d proj_dim=%d align_weight=%g",
    cfg.seq_len or 0, cfg.vocab_size or 0, cfg.text_d_model or 0, cfg.proj_dim or 0, cfg.align_weight or 0.0))
  log(string.format("  - base_tokenizer=%s", base_tok_path))
end
log(string.format("  - epochs=%d lr=%g kl_beta=%g", epochs, lr, cfg.kl_beta))
log(string.format("  - perceptual: weight=%g arch=%s base=%d ckpt=%s",
  tonumber(cfg.perceptual_weight or 0.0) or 0.0,
  tostring(cfg.perceptual_arch or ""),
  tonumber(cfg.perceptual_base_channels or 0) or 0,
  tostring(cfg.perceptual_checkpoint or "")))
log(string.format("  - seed=%d", seed))
log(string.format("  - warmup: lr_warmup_steps=%d kl_warmup_steps=%d", cfg.warmup_steps or 0, cfg.kl_warmup_steps or 0))
log(string.format("  - logvar_clip=[%g,%g] grad_clip_norm=%g",
  cfg.logvar_clip_min, cfg.logvar_clip_max, cfg.grad_clip_norm))
log(string.format("  - autosave_every_epochs=%d", cfg.autosave_every_epochs or 0))
log(string.format("  - viz_taps_force_inference=%s", tostring(cfg.viz_taps_force_inference)))
log(string.format("  - markers: wass_scale=%g temp_scale=%g warmup_steps=%d scale_max=%g",
  cfg.marker_wass_scale or 0.0, cfg.marker_temp_scale or 0.0, cfg.marker_warmup_steps or 0, cfg.marker_scale_max or 0.0))
log(string.format("  - grad_accum_steps=%d", cfg.grad_accum_steps))
log(string.format("  - backbone_ready=%s stop=%s window=%d recon_target=%g kl=[%g,%g] plateau(rel<=%g abs<=%g) min_steps=%d file=%s",
  tostring(cfg.backbone_ready_enabled),
  tostring(cfg.backbone_ready_stop),
  tonumber(cfg.backbone_ready_window or 0) or 0,
  tonumber(cfg.backbone_ready_recon_target or 0.0) or 0.0,
  tonumber(cfg.backbone_ready_kl_min or 0.0) or 0.0,
  tonumber(cfg.backbone_ready_kl_max or 0.0) or 0.0,
  tonumber(cfg.backbone_ready_plateau_rel or 0.0) or 0.0,
  tonumber(cfg.backbone_ready_plateau_abs or 0.0) or 0.0,
  tonumber(cfg.backbone_ready_min_steps or 0) or 0,
  tostring(cfg.backbone_ready_file or "")))
if (cfg.validate_every_steps or 0) > 0 then
  log(string.format("  - validate_every_steps=%d validate_items=%d holdout_frac=%.4g",
    cfg.validate_every_steps, cfg.validate_items, cfg.validate_holdout_frac))
end
if cfg.triple_fault then
  log(string.format("  - triple_fault=true fault_every=%d", cfg.triple_fault_every_steps))
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
log("✓ Dataset chargé: " .. tostring(n_or_err))

-- Modèle
assert_ok(Mimir.Model.create(arch, cfg), nil, "Model.create(" .. tostring(arch) .. ") failed")
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
      strict_mode = false,
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
  local starttrain_path = out_dir .. "/starttrain.json"
  os.execute("mkdir -p '" .. out_dir:gsub("'", "'\\''") .. "' 2>/dev/null")
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
    last_dir = Ckpt.find_latest_epoch_dir(out_dir)
  end
  if not last_dir and Ckpt and Ckpt.resolve_dir then
    last_dir = Ckpt.resolve_dir(out_dir)
  end
  if not last_dir then
    last_dir = out_dir
  end

  os.execute("mkdir -p '" .. tostring(last_dir):gsub("'", "'\\''") .. "' 2>/dev/null")
  local ok_save_stop, err_save_stop = Mimir.Serialization.save(last_dir, "raw_folder", {
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
  assert_ok(ok_save_stop, err_save_stop, "Serialization.save(stop) failed")
  log("✓ Checkpoint STOP écrit: " .. tostring(last_dir))
  return
end
assert_ok(ok_train, err_train, "Model.train failed")

-- Sauvegarde
os.execute("mkdir -p '" .. out_dir:gsub("'", "'\\''") .. "' 2>/dev/null")
local ok_save, err_save = Mimir.Serialization.save(out_dir, "raw_folder", {
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
assert_ok(ok_save, err_save, "Serialization.save failed")

log("✓ Checkpoint écrit: " .. out_dir)
