#!/usr/bin/env python3
"""Run notebook GPU validation control flow with simulated driver outcomes."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class Availability(unittest.TestCase):
    def validate(self,filename,runtimes,portable,strict=False,optimizer_unavailable=False,loss_unavailable=False,legacy_cell=False,backend='cuda',initial_env=None,cmake_gpu=True):
        n=json.loads((ROOT/filename).read_text())
        source=next(''.join(c['source']) for c in n['cells'] if 'VALIDATED=False' in ''.join(c['source']) and 'REFERENCE_TARGETS=' in ''.join(c['source']))
        source=source[:source.index('assert CPU_HASHES==')]
        if legacy_cell:
            source=source.replace("GPU_REQUIRE_ALL = globals().get('GPU_REQUIRE_ALL', False)", '')
        def run(command,**kwargs):
            if backend=='cpu' and ('PortableNativeOps' in str(command[0]) or
                    'CudaNativeOps' in str(command[0]) or command[-1] in ('CUDA','VULKAN','OPENCL','TPU')):
                self.fail('CPU validation attempted a GPU/TPU test: '+str(command))
            if command[0]=='ctest':
                self.assertIn('build-colab-reference-cpu' if backend!='cpu' or cmake_gpu else '/tmp/build',str(command[command.index('--test-dir')+1]))
                for runtime in ('CUDA','ROCM','VULKAN','OPENCL'):
                    self.assertEqual(kwargs['env']['MIMIR_DISABLE_'+runtime],'1')
            if command[0]=='cmake' and '-B' in command:
                for runtime in ('CUDA','VULKAN','OPENCL'):self.assertIn('-DENABLE_'+runtime+'=OFF',command)
            if 'PortableNativeOps' in str(command[0]):
                target=command[1]
                for runtime in ('CUDA','VULKAN','OPENCL'):
                    self.assertEqual(kwargs['env']['MIMIR_DISABLE_'+runtime],'0' if runtime==target else '1')
                self.assertEqual(kwargs['allowed_returncodes'],(77,))
                if isinstance(portable,Exception):raise portable
                return portable
            if 'ReconstructionLoss' in str(command[0]):
                if loss_unavailable and command[1]!='CUDA':return 'SKIP: loss GPU unavailable'
                return 'RECONSTRUCTION_RUNTIME_OK '+command[1]
            if 'OptimizerDispatch' in str(command[0]):
                if optimizer_unavailable and command[1]!='CUDA':return 'SKIP: native optimizer GPU unavailable'
                return 'OPTIMIZER_RUNTIME_OK '+command[1]
            return 'CUDA_NATIVE_PARITY_OK'
        scope=dict(run=run,BUILD=Path('/tmp/build'),REPO=ROOT,RUNTIME_ENV=dict(initial_env or {}),BACKEND=backend,
                   MIMIR_GPU_RUNTIMES=runtimes.copy(),GPU_REQUIRE_ALL=strict,ENVIRONMENT={},
                   MIMIR_MICRO_RUNTIME=runtimes[0] if runtimes else backend,print=lambda *a:None,JOBS=1,
                   CMAKE_ARGS=['cmake','-S',str(ROOT),'-B','/tmp/build','-DENABLE_CUDA=ON','-DENABLE_VULKAN=ON','-DENABLE_OPENCL=ON'])
        if not cmake_gpu:scope['CMAKE_ARGS']=[a.replace('=ON','=OFF') for a in scope['CMAKE_ARGS']]
        if strict is None:scope.pop('GPU_REQUIRE_ALL')
        exec(source,scope)
        return scope

    def test_cpu_transition_does_not_run_gpu_validation(self):
        stale=dict(MIMIR_DISABLE_CPU='1',MIMIR_DISABLE_CUDA='0',MIMIR_DISABLE_OPENCL='0',
                   MIMIR_DISABLE_VULKAN='0',MIMIR_BENCH_RUNTIME='opencl',MIMIR_TPU='1',
                   CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',PJRT_DEVICE='CUDA')
        for filename in ('Colab-test.ipynb','Colab-bench-multi-framework.ipynb'):
            for gpu_build in (False,True):
                for requested in ([],['cuda','vulkan','opencl']):
                    result=self.validate(filename,requested,RuntimeError('GPU must not run'),
                        strict=True,backend='cpu',initial_env=stale,cmake_gpu=gpu_build)
                    self.assertEqual(result['MIMIR_GPU_RUNTIMES'],[])
                    self.assertEqual(result['MIMIR_MICRO_RUNTIME'],'cpu')
                    self.assertEqual(result['REQUESTED_GPU_RUNTIMES'],[])
                    env=result['RUNTIME_ENV']
                    self.assertEqual(env['MIMIR_BENCH_RUNTIME'],'cpu')
                    self.assertEqual(env['MIMIR_DISABLE_CPU'],'0')
                    self.assertEqual(env['MIMIR_TPU'],'0')
                    self.assertEqual(env['JAX_PLATFORMS'],'cpu')
                    self.assertEqual(env['PJRT_DEVICE'],'CPU')
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
                    for runtime in ('CUDA','ROCM','VULKAN','OPENCL'):
                        self.assertEqual(env['MIMIR_DISABLE_'+runtime],'1')

    def test_missing_flag_in_resumed_session(self):
        for filename in ('Colab-test.ipynb','Colab-bench-multi-framework.ipynb'):
            result=self.validate(filename,['cuda','vulkan'],'SKIP: VULKAN device unavailable',strict=None)
            self.assertIs(result.get('GPU_REQUIRE_ALL',False),False)
            self.assertEqual(result['MIMIR_GPU_RUNTIMES'],['cuda'])
            legacy=self.validate(filename,['cuda','vulkan'],'SKIP: VULKAN device unavailable',strict=None,legacy_cell=True)
            self.assertEqual(legacy['MIMIR_GPU_RUNTIMES'],['cuda'])
            # Missing optional configuration must not admit a CPU-only GPU run.
            with self.assertRaisesRegex(RuntimeError,'Aucun runtime GPU'):
                self.validate(filename,['vulkan'],'SKIP: VULKAN device unavailable',strict=None)

    def test_optional_driver_absent(self):
        for filename in ('Colab-test.ipynb','Colab-bench-multi-framework.ipynb'):
            s=self.validate(filename,['cuda','vulkan'],'GPU native types: forward=72/72\nSKIP: VULKAN device unavailable')
            self.assertEqual(s['MIMIR_GPU_RUNTIMES'],['cuda'])
            self.assertEqual(s['RUNTIME_ENV']['MIMIR_DISABLE_VULKAN'],'1')
            self.assertEqual(s['RUNTIME_ENV']['MIMIR_BENCH_RUNTIME'],'cuda')
            self.assertIn('vulkan',s['ENVIRONMENT']['unavailable_gpu_runtimes'])
            with self.assertRaises(RuntimeError):self.validate(filename,['vulkan'],'SKIP: VULKAN device unavailable')
            with self.assertRaises(RuntimeError):self.validate(filename,['cuda','vulkan'],'SKIP: device unavailable',True)
            with self.assertRaises(RuntimeError):self.validate(filename,['cuda','vulkan'],RuntimeError('numerical mismatch'))
            with self.assertRaises(RuntimeError):self.validate(filename,['cuda','vulkan'],'no native proof')
            s=self.validate(filename,['cuda','vulkan'],'GPU_NATIVE_PARITY_OK Linear',optimizer_unavailable=True)
            self.assertEqual(s['MIMIR_GPU_RUNTIMES'],['cuda'])
            self.assertIn('optimizer:',s['UNAVAILABLE_GPU_RUNTIMES']['vulkan'])
            s=self.validate(filename,['cuda','opencl'],'GPU_NATIVE_PARITY_OK Linear',loss_unavailable=True)
            self.assertEqual(s['MIMIR_GPU_RUNTIMES'],['cuda'])
            self.assertIn('reconstruction_loss:',s['UNAVAILABLE_GPU_RUNTIMES']['opencl'])
            with self.assertRaises(RuntimeError):self.validate(filename,['cuda','opencl'],'GPU_NATIVE_PARITY_OK Linear',strict=True,loss_unavailable=True)
            s=self.validate(filename,['opencl'],'GPU_NATIVE_PARITY_OK Linear')
            self.assertEqual(s['RUNTIME_ENV']['MIMIR_BENCH_RUNTIME'],'opencl')


if __name__=='__main__':unittest.main()
