#!/usr/bin/env python3
"""Run notebook GPU validation control flow with simulated driver outcomes."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class Availability(unittest.TestCase):
    def validate(self,filename,runtimes,portable,strict=False,optimizer_unavailable=False,loss_unavailable=False):
        n=json.loads((ROOT/filename).read_text())
        source=next(''.join(c['source']) for c in n['cells'] if ''.join(c['source']).startswith('VALIDATED=False'))
        source=source[:source.index('assert CPU_HASHES==')]
        def run(command,**kwargs):
            if command[0]=='ctest':
                self.assertIn('build-colab-reference-cpu',str(command[command.index('--test-dir')+1]))
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
        scope=dict(run=run,BUILD=Path('/tmp/build'),REPO=ROOT,RUNTIME_ENV={},BACKEND='cuda',
                   MIMIR_GPU_RUNTIMES=runtimes.copy(),GPU_REQUIRE_ALL=strict,ENVIRONMENT={},
                   MIMIR_MICRO_RUNTIME=runtimes[0],print=lambda *a:None,JOBS=1,
                   CMAKE_ARGS=['cmake','-S',str(ROOT),'-B','/tmp/build','-DENABLE_CUDA=ON','-DENABLE_VULKAN=ON','-DENABLE_OPENCL=ON'])
        exec(source,scope)
        return scope

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
