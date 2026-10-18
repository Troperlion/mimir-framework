#!/usr/bin/env python3
"""Verify the production shell launcher forwards setup, arguments and exit status."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class Launcher(unittest.TestCase):
    def test_native_entrypoint(self):
        with tempfile.TemporaryDirectory(prefix='mimir launch ') as directory:
            binary=Path(directory)/'native driver'
            binary.write_text('#!/usr/bin/env python3\nimport os,sys,json\nprint(json.dumps(dict(args=sys.argv[1:],planner=os.environ["MIMIR_PLANNER_MODE"],fusion=os.environ["MIMIR_PLANNER_FUSION"],arena=os.environ["MALLOC_ARENA_MAX"],threads=os.environ["OMP_NUM_THREADS"])))\nsys.exit(7)\n')
            binary.chmod(0o755)
            env=dict(os.environ,MIMIR_EXECUTABLE=str(binary),OMP_NUM_THREADS='1',MIMIR_PLANNER_MODE='cost',MIMIR_PLANNER_FUSION='1',MALLOC_ARENA_MAX='2')
            result=subprocess.run(['bash',ROOT/'run_mimir.sh','cpu','path with spaces','literal;$x'],env=env,cwd=directory,text=True,capture_output=True)
            self.assertEqual(result.returncode,7,result.stderr)
            data=json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(data,dict(args=['cpu','path with spaces','literal;$x'],planner='cost',fusion='1',arena='2',threads='1'))

    def test_gpu_switches_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'probe'
            binary.write_text('#!/bin/bash\nprintf "%s %s %s" "$MIMIR_DISABLE_CUDA" "$MIMIR_DISABLE_VULKAN" "$MIMIR_DISABLE_OPENCL"\n')
            binary.chmod(0o755)
            env={k:v for k,v in os.environ.items() if not k.startswith(('MIMIR_GPU_','MIMIR_DISABLE_'))}
            env.update(MIMIR_EXECUTABLE=str(binary),MIMIR_GPU_CUDA='0',MIMIR_GPU_VULKAN='1',MIMIR_GPU_OPENCL='0')
            def run():return subprocess.run(['bash',ROOT/'run_mimir.sh'],env=env,text=True,capture_output=True)
            result=run();self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(result.stdout.endswith('1 0 1'),result.stdout)
            env['MIMIR_DISABLE_VULKAN']='1'
            self.assertTrue(run().stdout.endswith('1 1 1'))
            env['MIMIR_GPU_CUDA']='invalid'
            self.assertEqual(run().returncode,2)


if __name__=='__main__':unittest.main()
