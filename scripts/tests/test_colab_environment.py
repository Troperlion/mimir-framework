#!/usr/bin/env python3
"""Exercise the exact notebook detection code without requiring accelerators."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[2]


class EnvironmentDetection(unittest.TestCase):
    def check_notebook(self,name):
        notebook=json.loads((ROOT/name).read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        source=next(''.join(c['source']) for c in notebook['cells'] if 'def choose_backend(request):' in ''.join(c['source']))
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='choose_backend')
        process=SimpleNamespace(run=Mock(),TimeoutExpired=subprocess.TimeoutExpired)
        paths=set()
        scope=dict(os=SimpleNamespace(environ={}),shutil=SimpleNamespace(which=Mock(return_value=None)),
                   subprocess=process,sys=SimpleNamespace(executable='python'),
                   Path=lambda p:SimpleNamespace(exists=lambda:p in paths))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),name,'exec'),scope)
        choose=scope['choose_backend']
        def reply(code=0,out=''):return SimpleNamespace(returncode=code,stdout=out)
        process.run.return_value=reply(1)
        self.assertEqual(choose('auto'),'cpu')
        self.assertEqual(choose('gpu'),'cuda')
        for backend in ('cpu','cuda','tpu'):self.assertEqual(choose(backend),backend)
        process.run.side_effect=subprocess.TimeoutExpired('probe',60)
        self.assertEqual(choose('auto'),'cpu')
        process.run.side_effect=None
        process.run.return_value=reply(out='MIMIR_TPU_COUNT=8\n')
        self.assertEqual(choose('auto'),'tpu')
        process.run.return_value=reply(1)
        scope['os'].environ['TPU_ACCELERATOR_TYPE']='v6e-1'
        self.assertEqual(choose('auto'),'tpu')
        scope['os'].environ.clear();paths.add('/dev/accel0')
        self.assertEqual(choose('auto'),'tpu');paths.clear()
        scope['shutil'].which.return_value='/usr/bin/nvidia-smi'
        process.run.return_value=reply(out='NVIDIA RTX PRO 6000, 12.0\n')
        self.assertEqual(choose('auto'),'cuda')
        process.run.side_effect=[reply(1),reply(out='GPU 0: Tesla T4 (UUID: GPU-test)')]
        self.assertEqual(choose('auto'),'cuda')
        with self.assertRaises(ValueError):choose('invalid')
        # Run the actual post-detection configuration, with its printing suppressed.
        config=source[source.index('MIMIR_GPU_RUNTIMES='):]
        for backend in ('cpu','cuda','tpu'):
            env=dict(os=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES':'0'}),BACKEND=backend,
                     REQUESTED_BACKEND='auto',REPO=Path('/tmp/repo'),THREADS=2,TPU_TARGET='auto',
                     sys=SimpleNamespace(executable='python',version='test'),platform=SimpleNamespace(platform=lambda:'test'),
                     json=json,print=lambda *a,**k:None,GPU_CUDA=True,GPU_VULKAN=False,GPU_OPENCL=False)
            exec(config,env)
            runtime=env['RUNTIME_ENV']
            self.assertEqual(runtime['MIMIR_TPU'],'1' if backend=='tpu' else '0')
            self.assertEqual(runtime['JAX_PLATFORMS'],dict(cpu='cpu',cuda='cuda',tpu='tpu')[backend])
            if backend=='cpu':self.assertEqual(runtime['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(env['ENVIRONMENT']['requested_backend'],'auto')
            # Independent Mímir switches must not change competitor GPU selection.
            import itertools
            for flags in itertools.product((False,True),repeat=3):
                env.update(GPU_CUDA=flags[0],GPU_VULKAN=flags[1],GPU_OPENCL=flags[2])
                if backend=='cuda' and not any(flags):
                    with self.assertRaises(ValueError):exec(config,env)
                    continue
                exec(config,env)
                expected=[name for name,enabled in zip(('cuda','vulkan','opencl'),flags) if enabled] if backend=='cuda' else []
                self.assertEqual(env['MIMIR_GPU_RUNTIMES'],expected)
                for name in ('cuda','vulkan','opencl'):
                    self.assertEqual(env['RUNTIME_ENV']['MIMIR_DISABLE_'+name.upper()],'0' if name in expected else '1')
                self.assertEqual(env['MIMIR_MICRO_RUNTIME'],expected[0] if expected else backend)

    def test_benchmark(self):self.check_notebook('Colab-bench-multi-framework.ipynb')
    def test_runtime(self):self.check_notebook('Colab-test.ipynb')


if __name__=='__main__':unittest.main()
