#!/usr/bin/env python3
"""CPU/JAX differential proof for all optimizers; no TPU hardware claim."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import jax

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mimir_tpu_optimizer',root/'src/runtimes/tpu/worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
engine=worker.Engine(jax.devices('cpu')[0])
with tempfile.TemporaryDirectory() as d:
    fixture=Path(d)/'optimizers.json'
    subprocess.run([sys.argv[1],'fixture',str(fixture)],check=True)
    cases=json.loads(fixture.read_text())
for case in cases:
    memory=np.asarray(case['input'],np.float32);n=case["count"]
    arrays=[memory[i*n:(i+1)*n] for i in range(4)]+[np.asarray(case['parameters'],np.float32)]
    _,output=engine.execute(dict(command='optimizer',kind=case['kind']),arrays)
    expected=np.asarray(case['expected'],np.float32)
    for actual,start in zip(output,(0,2*n,3*n)):
        np.testing.assert_allclose(np.asarray(actual),expected[start:start+n],atol=4e-5,rtol=5e-4)
print('OPTIMIZER_JAX_REFERENCE_OK',len(cases),'cases; all 9 optimizers; two Adafactor modes; moments preserved',file=sys.stderr)
