#!/usr/bin/env python3
"""CPU/JAX reference differential test. Does NOT establish TPU hardware parity."""
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import os
import sys
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")

worker=Path(__file__).resolve().parents[1]/"src/runtimes/tpu/worker.py"
spec=importlib.util.spec_from_file_location("mimir_tpu_worker",worker)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Target recognition is tested independently of the machine running this suite.
assert m.target_profile("TPU v5 lite",1)=="TPU-v5e1"
assert m.target_profile("TPU v5e",1,"TPU-v5e1")=="TPU-v5e1"
assert m.target_profile("TPU v6 lite",1,"v6e-1")=="TPU-v6e1"
assert m.target_profile("TPU v6e",1,"TPU-v6e1")=="TPU-v6e1"
for kind,count,target in [("TPU v5p",1,"auto"),("TPU v4",1,"auto"),
                          ("TPU v5 lite",2,"auto"),("TPU v6 lite",1,"TPU-v5e1"),
                          ("TPU v5 lite",1,"bogus")]:
    try:m.target_profile(kind,count,target)
    except ValueError:pass
    else:raise AssertionError(f"target should be rejected: {kind}, {count}, {target}")
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/"cpu.json"
    subprocess.run([sys.argv[1],str(path)],check=True)
    cases=json.loads(path.read_text())

errors=[]
covered=set()
def check(label,actual,expected):
    a=np.asarray(actual,dtype=np.float32).ravel();b=np.asarray(expected,dtype=np.float32).ravel()
    if a.shape!=b.shape or not np.allclose(a,b,rtol=4e-4,atol=3e-5,equal_nan=False):
        delta=np.max(np.abs(a-b)) if a.shape==b.shape and a.size else "shape"
        raise AssertionError(f"{label}: {a.shape} vs {b.shape}, max diff={delta}\nactual={a[:15]}\nCPU={b[:15]}")

for idx,c in enumerate(cases):
    l=c["layer"];t=l["type"];label=f"{idx}:{t} training={c['training']}"
    try:
        xs=tuple(jnp.asarray(a,dtype=jnp.float32) for a in c["inputs"]);w=jnp.asarray(c["weights"],dtype=jnp.float32)
        x=xs[0] if xs else jnp.empty((0,),jnp.float32)
        noise=jnp.asarray(c["mask"],dtype=jnp.float32) if c["mask"] else jnp.zeros_like(x)
        if t=="Reparameterize" and c["training"]:
            noise=(jnp.asarray(c["outputs"][0])-x)/jnp.exp(.5*jnp.clip(xs[1],-20,20))
        if t=="NMS":
            out,count=jax.jit(lambda a:m.nms(a,l))(xs);out=(out[:int(count)],)
        else:out=jax.jit(lambda a,b,n:m.forward(a,b,l,c["training"],n,True))(xs,w,noise)
        if len(out)!=len(c["outputs"]):raise AssertionError("output count")
        for i,(a,b) in enumerate(zip(out,c["outputs"])):check(f"{label} forward[{i}]",a,b)
        gos=tuple(jnp.asarray(a,dtype=jnp.float32) for a in c["go"])
        saved=jnp.asarray(c["outputs"][0],dtype=jnp.float32);mask=jnp.asarray(c["mask"],dtype=jnp.float32)
        gx,gw=jax.jit(lambda a,b,g,s,k:m.backward(a,b,g,s,k,l,c["training"]))(xs,w,gos,saved,mask)
        if len(gx)!=len(c["gi"]):raise AssertionError("gradient input count")
        for i,(a,b) in enumerate(zip(gx,c["gi"])):check(f"{label} backward[{i}]",a,b)
        check(f"{label} grad_weights",gw,c["gw"] or [0.]*w.size)
        gb=m.bias_gradient(gw,l)
        check(f"{label} grad_bias",gb,c["gb"])
        covered.add(t)
    except Exception as exc:
        errors.append(f"{label}: {exc}")
for error in errors:print(error,file=sys.stderr)
print(f"TPU_REFERENCE: {len(cases)-len(errors)}/{len(cases)} cases, {len(covered)}/{len(m.OPS)} layer types",file=sys.stderr)
if errors or covered!=m.OPS:sys.exit(1)

# Exercise the actual persistent worker/Engine and binary framing as well as the
# pure kernels above. Its CPU handshake is test-only and rejected by TpuRuntime.
proc=subprocess.Popen([sys.executable,str(worker),"--reference-test"],
                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                      env=dict(os.environ,JAX_PLATFORMS="cpu"))
def exact(n):
    out=b""
    while len(out)<n:
        part=proc.stdout.read(n-len(out))
        if not part:raise RuntimeError("reference worker closed its protocol")
        out+=part
    return out

def rpc(req,arrays):
    arrays=[np.asarray(a,dtype="<f4").ravel() for a in arrays]
    req=dict(req,sizes=[a.size for a in arrays])
    header=json.dumps(req,separators=(",",":")).encode()
    proc.stdin.write(struct.pack("<I",len(header)));proc.stdin.write(header)
    for a in arrays:proc.stdin.write(a.tobytes())
    proc.stdin.flush()
    meta=json.loads(exact(struct.unpack("<I",exact(4))[0]))
    out=[np.frombuffer(exact(n*4),dtype="<f4") for n in meta["sizes"]]
    if not meta["ok"]:raise RuntimeError(meta.get("error"))
    return meta,out
try:
    meta,_=rpc({"command":"hello"},[])
    assert meta["platform"]=="cpu"
    samples={"Linear":False,"Dropout":True,"Reparameterize":True,"Constant":False,"NMS":False,"Split":False}
    for t,training in samples.items():
        c=next(c for c in cases if c["layer"]["type"]==t and c["training"]==training)
        base={"layer":c["layer"],"inputs":len(c["inputs"]),"training":training}
        meta,out=rpc(dict(base,command="forward",fused=True),c["inputs"]+[c["weights"]])
        assert meta["outputs"]==len(c["outputs"]) and len(out)==len(c["outputs"])+1
        if not training:
            for a,b in zip(out,c["outputs"]):check(t+" wire forward",a,b)
        else:
            assert all(np.all(np.isfinite(a)) for a in out)
            if t=="Dropout":check(t+" sampled mask",out[0],np.asarray(c["inputs"][0])*out[-1]/(1-c["layer"]["dropout_p"]))
        arrays=c["inputs"]+[c["weights"]]+c["go"]+[c["outputs"][0],c["mask"]]
        meta,out=rpc(dict(base,command="backward",grad_outputs=len(c["go"])),arrays)
        assert meta["grad_inputs"]==len(c["gi"]) and len(out)==len(c["gi"])+2
        for a,b in zip(out,c["gi"]+[c["gw"],c["gb"]]):check(t+" wire backward",a,b)
    print("TPU_REFERENCE_PROTOCOL_OK platform=cpu",file=sys.stderr)
finally:
    proc.stdin.close();proc.terminate();proc.wait(timeout=10)
