#!/usr/bin/env python3
"""Native VGG16 smoke: source dataset JSON -> automatic classes -> training/checkpoint.
Usage: python3 scripts/tests/test_vgg16_dataset_annotations.py --mimir ./bin/mimir
"""
import argparse
import json, pathlib, subprocess, tempfile, os
repo=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mimir',default=str(repo/'bin/mimir'))
binary=str(pathlib.Path(parser.parse_args().mimir).resolve())
with tempfile.TemporaryDirectory(prefix='mimir-auto-tags-') as t:
 root=pathlib.Path(t)
 for mode in ['medical','annotator']:
  data=root/mode;data.mkdir()
  for i in range(3):
   name=f'{i}.ppm'
   (data/name).write_bytes(b'P6\n32 32\n255\n'+bytes([64+i*50,96,128])*1024)
   if mode=='medical':
    doc={'normalized':True,'boxes':([] if i==2 else [{'x1':.1,'y1':.1,'x2':.9,'y2':.9,'class_id':i,'label':['Mass','Calcification'][i]}])}
   else:
    doc={'image':name,'captions':{'primary':['Red cat.','Blue dog.','Red dog.'][i]},'entities':[{'id':'subject','role':'primary_subject','phrase':['red cat','blue dog','red dog'][i]}]}
   (data/f'{i}.json').write_text(json.dumps(doc))
  flags=['--dataset-root',str(data),'--out-dir',str(root/(mode+'-out')),'--image-w','32','--image-h','32','--base-channels','1','--fc-hidden','4','--epochs','1','--validate-items','0','--validate-every-steps','0','--validate-every-epochs','0','--validate-holdout','false','--autosave-every-epochs','0','--hw','false','--log-every','1']
  wrapper=root/'smoke.lua'
  wrapper.write_text('ROOTWORK='+json.dumps(str(repo))+'\narg={'+','.join(json.dumps(v) for v in flags)+'}\ndofile(ROOTWORK.."/scripts/training/train_vgg16_tags_multilabel.lua")\n')
  r=subprocess.run([binary,'--lua',str(wrapper)],cwd=root,env={**os.environ,'OMP_NUM_THREADS':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
  assert r.returncode==0,r.stdout[-5000:]
  assert 'step=3 loss=' in r.stdout,r.stdout[-5000:]
  if mode=='medical': assert 'active_labels=0 label_candidates=0' in r.stdout
  vocab=(root/(mode+'-out')/'tags_vocab.auto.txt').read_text().splitlines()
  assert ('mass' in vocab and 'calcification' in vocab) if mode=='medical' else ('red cat' in vocab and 'dog' in vocab)
  assert (root/(mode+'-out')/'final').is_dir()
  print('PASS native VGG16',mode,': JSON sidecars, auto vocab',vocab,', training and final checkpoint; no tags-vocab or composition JSON')
