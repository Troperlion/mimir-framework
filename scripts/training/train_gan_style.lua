#!/usr/bin/env mimir --lua
-- GAN image -> image non apparié Mímir.
-- architecture native Gan_Style : générateur et discriminateur convolutionnels.
-- Le L1 de contenu limite les changements; ce modèle n'est pas un CycleGAN.
-- Usage : ./bin/mimir --lua scripts/training/train_gan_style.lua \
--   --source-root datasets/photos --style-root datasets/style --epochs 20
-- Inférence : même script --input photo.png --checkpoint checkpoint/gan_style/epoch_0020 --output result.ppm
local root = ROOTWORK or '.'
for _,a in ipairs(arg or {}) do
  if a=='--help' or a=='-h' then
    print([[GAN image -> image Mímir
Entraînement: --source-root DIR --style-root DIR [--epochs 20] [--size 64]
  --channels 16 --d-channels 16 --lr 0.0002 --d-lr 0.001
  --dtype float32|float16|bfloat16 --content-weight 10 --seed 42 --out-dir checkpoint/gan_style --cpu
  --steps-per-epoch N : limite optionnelle (défaut: max des tailles des datasets)
  --resume DIR : checkpoint epoch_XXXX, --epochs = nombre total d'époques
Inférence: --checkpoint DIR (--input IMAGE --output result.ppm |
  --input-root DIR --output-dir DIR) [--cpu]
Chaque époque sauvegarde G, son optimiseur, D, les réglages et le curseur.
Les deux réseaux et leur entraînement appartiennent au modèle natif Gan_Style.
Les sorties sont redimensionnées à la résolution d'entraînement et écrites en PPM.]])
    return
  end
end
local opts = dofile(root..'/scripts/modules/args.lua').parse(arg)
local FS = dofile(root..'/scripts/modules/fs.lua')
local PPM = dofile(root..'/scripts/modules/image_ppm.lua')
local function num(k, d, lo, integer)
  local v = tonumber(opts[k] or d)
  assert(v and v == v and v < math.huge and v >= lo and (not integer or v%1==0), 'option invalide: '..k)
  return v
end
local inference = opts.input ~= nil or opts['input-root'] ~= nil
assert(not (opts.input and opts['input-root']), 'choisir --input ou --input-root')
assert(not (inference and opts.resume), '--resume est réservé à l’entraînement')
if inference then assert(opts.checkpoint, '--checkpoint requis') end
local side = num('size', 64, 3, true)
local seed = num('seed', 42, 1, true)
math.randomseed(seed)
assert(Mimir.Model.set_hardware(opts.cpu ~= true))
local function finite(v) assert(v == v and math.abs(v)<math.huge, 'NaN/Inf'); return v end
local function chw(image, s)
  assert(#image == s*s*3, 'image RGB invalide')
  local out = {}
  for p=0,s*s-1 do for c=0,2 do out[c*s*s+p+1]=image[p*3+c+1]/127.5-1 end end
  return out
end
local function hwc(image, s)
  local out = {}
  for p=0,s*s-1 do for c=0,2 do out[p*3+c+1]=finite(image[c*s*s+p+1]) end end
  return out
end
local function read(path)
  local item, err = Mimir.IO.readImageRGBU8(path, side, side)
  assert(item, err); return chw(item.image, side)
end
if not inference then assert(opts['source-root'] and opts['style-root'], '--source-root et --style-root requis') end
local epochs = num('epochs',20,1,true)
local steps = opts['steps-per-epoch'] and num('steps-per-epoch',1,1,true)
local lr = num('lr',0.0002,1e-12)
local dlr = num('d-lr',0.001,1e-12)
local content = num('content-weight',10,0)
local channels = num('channels',16,1,true)
local hidden = num('d-channels',16,1,true)
local outdir = opts['out-dir'] or 'checkpoint/gan_style'
if not inference then assert(FS.mkdir_p(outdir)) end
-- Retenir seulement les chemins: le dataset global peut ensuite être remplacé.
local function paths(dir)
  local ok,n = Mimir.Dataset.load(dir,side,side,1,false,'',256,true)
  assert(ok,n)
  local result={}
  for i=1,n do local item=assert(Mimir.Dataset.get(i,false)); if item.image_file and item.image_file~='' then result[#result+1]=item.image_file end end
  assert(#result>0,'aucune image: '..dir); return result
end
local sources, styles
local checkpoint = inference and opts.checkpoint or opts.resume
local completed = 0
if checkpoint then
  local file=assert(io.open(checkpoint..'/gan_state.txt','r'))
  local version=file:read('*l'); assert(version=='mimir_native_gan_style_v1','checkpoint GAN incompatible')
  local values={}
  for token in file:read('*a'):gmatch('%S+') do values[#values+1]=assert(tonumber(token),'état GAN invalide') end
  file:close()
  local function take() local value=table.remove(values,1); assert(value,'état GAN tronqué'); return finite(value) end
  side,channels,hidden,completed,seed=math.floor(take()),math.floor(take()),math.floor(take()),math.floor(take()),math.floor(take())
  lr,dlr,content=take(),take(),take()
  assert(side>=3 and channels>=1 and hidden>=1 and completed>=0,'dimensions GAN invalides')
  assert(#values==0, 'état GAN incompatible')
end
if not inference then
  sources,styles=paths(opts['source-root']),paths(opts['style-root'])
  steps=steps or math.max(#sources,#styles)
  assert(epochs>completed,'--epochs doit dépasser la dernière époque sauvegardée')
end
assert(Mimir.Model.create('Gan_Style', {image_w=side,image_h=side,image_c=3,
  base_channels=channels,discriminator_channels=hidden,
  content_weight=content,discriminator_lr=dlr}))
if opts.dtype then assert(Mimir.Model.dtype(opts.dtype)) end
assert(Mimir.Model.allocate_params())
if checkpoint then
  assert(Mimir.Serialization.load(checkpoint,'raw_folder',{strict_mode=true,load_optimizer=not inference}))
  if opts.dtype then assert(Mimir.Model.dtype(opts.dtype)) end
end
if inference then
  assert(checkpoint,'--checkpoint requis')
  if opts.input then
    local output=assert(Mimir.Model.forward(read(opts.input),false))
    assert(PPM.write(opts.output or 'gan_result.ppm',hwc(output,side),side,side))
  else
    local inputs=paths(opts['input-root'])
    local destination=assert(opts['output-dir'],'--output-dir requis')
    assert(FS.mkdir_p(destination))
    local manifest=assert(io.open(destination..'/manifest.tsv','w'))
    manifest:write('source\toutput\n')
    for i,path in ipairs(inputs) do
      local target=string.format('%s/%06d.ppm',destination,i)
      local output=assert(Mimir.Model.forward(read(path),false))
      assert(PPM.write(target,hwc(output,side),side,side))
      manifest:write(path,'\t',target,'\n'); manifest:flush()
      print(string.format('Inférence %d/%d: %s',i,#inputs,target))
    end
    manifest:close()
  end
  print('Inférence terminée'); return
end
if not checkpoint then assert(Mimir.Model.init_weights('xavier',seed)) end
local csvpath=outdir..'/loss.csv'
local append=checkpoint and FS.file_exists(csvpath)
local csv=assert(io.open(csvpath,append and 'a' or 'w'))
if not append then csv:write('epoch,step,d_real,d_fake,g_adv,content_l1\n') end
local function shuffle(n)
  local indices={}; for i=1,n do indices[i]=i end
  for i=n,2,-1 do local j=math.random(i); indices[i],indices[j]=indices[j],indices[i] end
  return indices
end
for epoch=completed+1,epochs do
  -- Seed par époque: reprise à une frontière d'époque reproductible.
  math.randomseed(seed+epoch)
  local source_order,style_order=shuffle(#sources),shuffle(#styles)
  local sum=0
  for step=1,steps do
    local x=read(sources[source_order[(step-1)%#sources+1]])
    local real=read(styles[style_order[(step-1)%#styles+1]])
    local stats=assert(Mimir.Model.vision_train_step(x,real,{learning_rate=lr,optimizer='adam'}))
    local dr,df,ga,l1=stats.d_real,stats.d_fake,stats.g_adv,stats.content_l1
    finite(stats.loss)
    sum=sum+stats.loss
    csv:write(string.format('%d,%d,%.9g,%.9g,%.9g,%.9g\n',epoch,step,dr,df,ga,l1)); csv:flush()
    if step==1 or step%10==0 then print(string.format('epoch=%d step=%d D=%.5f G_adv=%.5f L1=%.5f',epoch,step,dr+df,ga,l1)) end
  end
  local destination=string.format('%s/epoch_%04d',outdir,epoch)
  assert(not FS.file_exists(destination..'/gan_state.txt'),'checkpoint existant: '..destination)
  assert(Mimir.Serialization.save(destination,'raw_folder',{save_optimizer=true}))
  local state=assert(io.open(destination..'/gan_state.txt','w'))
  state:write('mimir_native_gan_style_v1\n')
  local function write(value) state:write(string.format('%.17g\n',finite(value))) end
  for _,value in ipairs({side,channels,hidden,epoch,seed,lr,dlr,content}) do write(value) end
  assert(state:close())
  local preview=assert(Mimir.Model.forward(read(sources[1]),false))
  assert(PPM.write(destination..'/preview.ppm',hwc(preview,side),side,side))
  print(string.format('Époque %d: loss moyenne=%.6f checkpoint=%s',epoch,sum/steps,destination))
end
csv:close()
print('Entraînement terminé: '..outdir)
