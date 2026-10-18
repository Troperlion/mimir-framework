-- Exercise the training entrypoint up to model creation, without training or user data.
ROOTWORK = ROOTWORK or "."
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local root = os.tmpname()
os.remove(root)
FS.mkdir_p(root)
local saved_mimir, saved_arg, saved_log = Mimir, arg, log
local saved_read_json, saved_json, saved_cjson = read_json, json, cjson
local function write(path, text)
  local f=assert(io.open(path,"w"));f:write(text);f:close()
end
local function report(weight)
  -- Deliberately reversed order: weights must follow vocab, not JSON order.
  return '{"classes":[{"tag":"rare","recommended_pos_weight":'..weight..'}, {"tag":"common","recommended_pos_weight":0.25}]}'
end
local automatic_items=nil
local function run(extra, expected, error_text)
  local captured
  Mimir={Dataset={load=function() return true,automatic_items and #automatic_items or 2 end,
    get=function(i) return automatic_items and automatic_items[i] end,release=function() end},
    Architectures={default_config=function() return {} end},
    Model={create=function(_, cfg) captured=cfg;error("COMPOSITION_TEST_STOP") end}}
  arg={"--dataset-root",root,"--tags-vocab",root.."/tags.txt","--out-dir",root.."/out","--pos-weight","2"}
  for _, v in ipairs(extra or {}) do arg[#arg+1]=v end
  log=function() end
  local ok,err=pcall(dofile,ROOTWORK.."/scripts/training/train_vgg16_tags_multilabel.lua")
  assert(not ok)
  if error_text then assert(tostring(err):find(error_text,1,true),tostring(err));assert(not captured);return end
  assert(tostring(err):find("COMPOSITION_TEST_STOP",1,true),tostring(err))
  if not expected then assert(captured.class_pos_weights==nil);return end
  for i,w in ipairs(expected) do assert(captured.class_pos_weights[i]==w,"incorrect weight "..i) end
  assert(captured.use_dataset_annotation_labels==true)
end
local ok,err=pcall(function()
  json=nil;cjson=nil;read_json=nil -- pure Lua decoder, as in runtimes without json/cjson
  write(root.."/tags.txt","common\nrare\nother\n")
  run({},nil)
  write(root.."/composition.json",report(7))
  run({},{.25,7,2})
  write(root.."/tags.composition.json",report(9))
  run({},{.25,9,2}) -- vocab report takes precedence over dataset report
  write(root.."/explicit.json",report(11))
  run({"--class-composition",root.."/explicit.json"},{.25,11,2})
  run({"--use-class-composition-weights","false"},nil)
  run({"--class-composition",root.."/missing.json"},nil,"composition de classes introuvable")
  write(root.."/tags.composition.json",'{"classes":')
  run({},nil,"lecture composition de classes échouée")
  write(root.."/tags.composition.json",'{"classes":[{"tag":"unknown","recommended_pos_weight":2}]}')
  run({},nil,"aucune classe du vocabulaire ne correspond")
  if saved_read_json then
    read_json=saved_read_json
    write(root.."/tags.composition.json",report(13))
    run({},{.25,13,2}) -- actual native read_json under mimir
  end
  os.remove(root.."/composition.json");os.remove(root.."/tags.composition.json")
  automatic_items={
    {image_file="a.png",label_candidates={"Common","Common","Rare"}},
    {image_file="b.png",label_candidates={"Common"}},
    {image_file="c.png",label_candidates={}},
  }
  run({"--tags-vocab",""},{.5,2})
  local f=assert(io.open(root.."/out/tags_vocab.auto.txt"));assert(f:read("*a")=="common\nrare\n");f:close()
  run({"--tags-vocab","","--use-class-composition-weights","false"},nil)
  automatic_items[2].label_candidates={"New class"}
  run({"--tags-vocab","","--resume","true"},nil,"Vocabulaire automatique modifié")
end)
Mimir=saved_mimir;arg=saved_arg;log=saved_log
read_json=saved_read_json;json=saved_json;cjson=saved_cjson
os.execute("rm -rf "..FS.quote(root))
assert(ok,err)
print("PASS VGG16 composition: auto-detection, precedence, decoding, class alignment, BCE config, opt-out and invalid files")
