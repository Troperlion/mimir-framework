#!/usr/bin/env python3
"""Differential JAX reference; actual TPU execution is a separate native test."""
import importlib.util,json,os
from pathlib import Path
import subprocess,sys,tempfile
import numpy as np
import jax
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mimir_reconstruction',root/'src/runtimes/tpu/worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
engine=worker.Engine(jax.devices('cpu')[0])
with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'reconstruction.json'
    subprocess.run([sys.argv[1],'fixture',str(path)],check=True)
    cases=json.loads(path.read_text())
for case in cases:
    memory=np.asarray(case['memory'],np.float32);n=case['count']
    arrays=[memory[:n],memory[n:2*n],memory[2*n:3*n],np.asarray(case['parameters'],np.float32),memory[12*n:],memory[5*n:6*n]]
    request={k:case[k] for k in ('op','kind','relation_only','width','height','channels')}
    _,result=engine.execute(dict(command='reconstruction_loss',**request),arrays)
    np.testing.assert_allclose(np.asarray(result[0]),[case['loss']],atol=3e-5,rtol=3e-5,err_msg=str(request))
    np.testing.assert_allclose(np.asarray(result[2]),[case['relation_loss']],atol=3e-5,rtol=3e-5,err_msg=str(request))
    np.testing.assert_allclose(np.asarray(result[1]),case['grad'],atol=2e-5,rtol=5e-4,err_msg=str(request))
print('RECONSTRUCTION_JAX_REFERENCE_OK',len(cases),'cases',file=sys.stderr)
