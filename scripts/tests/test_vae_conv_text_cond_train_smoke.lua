local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- Smoke test: VAEConv text-cond natif (multi-modal optionnel)
--
-- Objectif:
-- - construire un mini dataset image+texte,
-- - activer text_cond=true,
-- - exécuter un mini entraînement réel (1 epoch, 1 item),
-- - vérifier que la sortie forward contient bien les têtes proj texte.
--
-- Usage:
--   ./bin/mimir --lua scripts/tests/test_vae_conv_text_cond_train_smoke.lua

local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local BaseTok = dofile(ROOTWORK.."/scripts/modules/base_tokenizer.lua")

local function logx(msg)
  local l = rawget(_G, "log")
  if type(l) == "function" then l(msg) else print(msg) end
end

local function apply_dtype(cfg)
  local dtype = (type(cfg) == "table" and cfg.dtype) or os.getenv("MIMIR_DTYPE")
  if dtype == nil then return true end
  if type(Mimir) ~= "table" or type(Mimir.model) ~= "table" or type(Mimir.model.dtype) ~= "function" then
    return true
  end
  local ok, dt_or_err = Mimir.model.dtype(dtype)
  assert(ok ~= false, "dtype invalide: " .. tostring(dt_or_err))
  return true
end

local function copy_file(src, dst)
  local fi, erri = io.open(src, "rb")
  if not fi then return false, erri end
  local data = fi:read("*a")
  fi:close()

  local fo, erro = io.open(dst, "wb")
  if not fo then return false, erro end
  fo:write(data)
  fo:close()
  return true
end

local cfg, err = Mimir.Architectures.default_config("vae_conv")
assert(type(cfg) == "table", "default_config(vae_conv) failed: " .. tostring(err))
---@cast cfg table<string, any>

cfg.image_w = 32
cfg.image_h = 32
cfg.image_c = 3
cfg.latent_h = 8
cfg.latent_w = 8
cfg.latent_c = 16
cfg.base_channels = 16
cfg.resnet = false
cfg.attention = false
cfg.stochastic_latent = false

cfg.text_cond = true
cfg.seq_len = 16
cfg.text_d_model = 16
cfg.proj_dim = 8
cfg.align_weight = 0.1

cfg.latent_dim = cfg.latent_h * cfg.latent_w * cfg.latent_c

cfg.max_items = 1
cfg.log_every = 1
cfg.validate_every_steps = 0
cfg.validate_items = 0
cfg.autosave_every_epochs = 0
cfg.kl_beta = 0.01
cfg.kl_warmup_steps = 1

local tmp_root = "scripts/tests/tmp_vae_conv_text_cond_smoke"
local ds_dir = tmp_root .. "/dataset"
local ckpt_dir = tmp_root .. "/ckpt"
FS.mkdir_p(ds_dir)
FS.mkdir_p(ckpt_dir)

local img_path = ds_dir .. "/sample.png"
local txt_path = ds_dir .. "/sample.txt"

local ok_img, err_img = copy_file("logo.png", img_path)
assert(ok_img, "copy logo.png failed: " .. tostring(err_img))

local tf, terr = io.open(txt_path, "w")
assert(tf ~= nil, "write text failed: " .. tostring(terr))
tf:write("simple smoke caption")
tf:close()

local ok_bt, err_bt = BaseTok.load_base({
  max_vocab = 2048,
  require = false,
})
assert(ok_bt == true, "Base tokenizer: " .. tostring(err_bt))

local cur_vocab = tonumber(BaseTok.vocab_size()) or 0
if cur_vocab < 8 then cur_vocab = 8 end
cfg.vocab_size = cur_vocab
cfg.checkpoint_dir = ckpt_dir

local ok_ds, n_or_err = Mimir.Dataset.load(ds_dir, cfg.image_w, cfg.image_h, 2, true, ds_dir .. "/dataset_cache.json", 128, true)
assert(ok_ds == true, "Dataset.load failed: " .. tostring(n_or_err))
assert((tonumber(n_or_err) or 0) >= 1, "Dataset.load returned 0 item")

local ok_create, err_create = Mimir.Model.create("vae_conv", cfg)
assert(ok_create == true, "Model.create failed: " .. tostring(err_create))
apply_dtype(cfg)

local ok_alloc, err_alloc = Mimir.Model.allocate_params()
assert(ok_alloc == true, "Model.allocate_params failed: " .. tostring(err_alloc))

local ok_init, err_init = Mimir.Model.init_weights("he", 123)
assert(ok_init == true, "Model.init_weights failed: " .. tostring(err_init))

-- Vérifie le pack de sortie en mode text-cond: recon + z + logvar + img_proj + txt_proj
local image_dim = cfg.image_w * cfg.image_h * cfg.image_c
local latent_dim = cfg.latent_dim
local expected = image_dim + 2 * latent_dim + 2 * cfg.proj_dim

local x = {}
x[image_dim] = 0.0
for i = 1, image_dim do x[i] = 0.0 end

local packed, err_fwd = Mimir.Model.forward(x, false)
assert(type(packed) == "table", "Model.forward failed: " .. tostring(err_fwd))
assert(#packed == expected,
  string.format("unexpected output size (text_cond): got=%d expected=%d", #packed, expected))

local ok_train, steps_or_err = Mimir.Model.train(1, 1e-3)
assert(ok_train == true, "Model.train failed: " .. tostring(steps_or_err))
assert((tonumber(steps_or_err) or 0) >= 1, "Model.train returned invalid step count")

logx(string.format("[test_vae_conv_text_cond_train_smoke] OK steps=%d output=%d", tonumber(steps_or_err) or -1, #packed))
