#!/usr/bin/env python3
"""Private Vulkan loader selection, rollback, and device identity checks."""
import importlib.util
import hashlib
import textwrap
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('nvidia_vulkan',ROOT/'scripts/colab/nvidia_vulkan.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class VulkanSetup(unittest.TestCase):
    def test_embedded_helper_without_repo_file(self):
        for filename in ('Colab-test.ipynb','Colab-bench-multi-framework.ipynb'):
            n=json.loads((ROOT/filename).read_text())
            source=next(''.join(c['source']) for c in n['cells'] if '# BEGIN MIMIR VULKAN HELPER' in ''.join(c['source']))
            a=source.index('    # BEGIN MIMIR VULKAN HELPER')
            b=source.index('    # END MIMIR VULKAN HELPER',a)
            with tempfile.TemporaryDirectory() as directory:
                scope=dict(hashlib=hashlib,REPO=Path(directory)/'old-snapshot')
                exec(textwrap.dedent(source[a:b]),scope)
                self.assertTrue(callable(scope['vulkan_setup_module'].configure))
                self.assertEqual(scope['vulkan_helper_source'],(ROOT/'scripts/colab/nvidia_vulkan.py').read_text())

    def test_device_identity(self):
        self.assertTrue(v.nvidia_device('GPU0:\n vendorID = 0x10de\n deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU'))
        self.assertFalse(v.nvidia_device('NVIDIA loader\nGPU0:\n vendorID = 0x10005\n deviceType = PHYSICAL_DEVICE_TYPE_CPU'))
        self.assertFalse(v.nvidia_device('GPU0:\n vendorID = 0x10de\n deviceType = PHYSICAL_DEVICE_TYPE_CPU'))

    def test_matching_version_and_rollback(self):
        for usable in (True,False):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'580.82.07';extracted=root/'extracted';extracted.mkdir(parents=True)
                (extracted/'.mimir-complete').touch()
                (root/'NVIDIA-Linux-x86_64-580.82.07.run').write_bytes(b'cached-test')
                (extracted/'nvidia_icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path='libGLX_nvidia.so.0',api_version='1.3.0'))))
                original=dict(LD_LIBRARY_PATH='/host/cuda',MIMIR_DISABLE_CUDA='0')
                with patch.object(v,'probe',side_effect=[(False,'no ICD'),(usable,'probe output')]),patch.object(v,'stage_libraries',return_value=['libEGL_nvidia.so.580.82.07']),patch.object(v.platform,'machine',return_value='x86_64'):
                    env,report=v.configure(original,d,lambda *a,**k:'580.82.07\n')
                self.assertEqual(original,dict(LD_LIBRARY_PATH='/host/cuda',MIMIR_DISABLE_CUDA='0'))
                self.assertIn('/580.82.07/',report['url'])
                if usable:
                    self.assertEqual(report['status'],'private_ready')
                    self.assertTrue(env['LD_LIBRARY_PATH'].endswith(':/host/cuda'))
                    self.assertEqual(env['VK_DRIVER_FILES'],env['VK_ICD_FILENAMES'])
                    icd=json.loads(Path(env['VK_DRIVER_FILES']).read_text())
                    self.assertIn('libEGL_nvidia.so.580.82.07',icd['ICD']['library_path'])
                else:
                    self.assertEqual(report['status'],'unavailable');self.assertEqual(env,original)


if __name__=='__main__':unittest.main()
