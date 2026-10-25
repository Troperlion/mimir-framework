"""Check three optimizer updates, reset and allocation against NumPy on CPU.

Run with each framework environment's Python; unavailable adapters are skipped.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

class OptimizerAdapters(unittest.TestCase):
    def test_installed_frameworks(self):
        if importlib.util.find_spec('numpy') is None:self.skipTest('numpy unavailable')
        installed=[f for f,m in [('pytorch','torch'),('tensorflow','tensorflow'),('jax','jax')] if importlib.util.find_spec(m)]
        if not installed:self.skipTest('no competitor installed')
        for framework in installed:
            for name in ('sgd','adam','adamw'):
                with self.subTest(framework=framework,optimizer=name):
                    result=subprocess.run([sys.executable,__file__,'--worker',framework,name],capture_output=True,text=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

if __name__=='__main__' and '--worker' in sys.argv:
    import sys,importlib.util
    from pathlib import Path
    import numpy as np
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('worker',root/'scripts/benchmarks/colab_real_worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    f,name=sys.argv[2:];r=np.random.default_rng(11);h=4
    p=[r.normal(0,.04,shape).astype('float32') for shape in [(h,784),(h,),(10,h),(10,)]]
    x=r.normal(0,.1,(2,784)).astype('float32');y=np.eye(10,dtype='float32')[[1,4]]
    a=worker.make_adapter(f,'cpu',dict(hidden=h,lr=.01,optimizer=name,weight_decay=.01 if name=='adamw' else 0.),p)
    m=[np.zeros_like(t,dtype='float64') for t in p];v=[t.copy() for t in m];gold=[t.astype('float64') for t in p]
    for t in range(1,4):
     z=x@gold[0].T+gold[1];act=np.maximum(z,0);out=act@gold[2].T+gold[3];delta=2*(out-y)/20;dh=delta@gold[2]*(z>0)
     grads=[dh.T@x,dh.sum(0),delta.T@act,delta.sum(0)]
     for i,g in enumerate(grads):
      m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
      gold[i]=gold[i]*(1-.01*(.01 if name=='adamw' else 0.))-.01*(g if name=='sgd' else (m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8))
     a.train(x,y)
     for actual,expected in zip(a.parameters(),gold):np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=4e-4)
    a.reset();a.train(x,y);once=a.parameters();a.train(x,y);a.reset();a.train(x,y)
    for actual,expected in zip(a.parameters(),once):np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
    print(f,name,'three-step reference + reset + allocation OK',a.allocation())
elif __name__=='__main__':
    unittest.main()
