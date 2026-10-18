#!/usr/bin/env python3
"""Validate the actual notebook fixtures, adapters and embedded core snapshot."""
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'Colab-bench-multi-framework.ipynb'
try:
    import numpy as np
except ImportError:
    np = None


def sources():
    return [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type'] == 'code']


def literals(source):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}


class NotebookIntegrity(unittest.TestCase):
    def test_code_and_worker_syntax(self):
        for source in sources():
            ast.parse(source)
            values = literals(source)
            if 'FRAMEWORK_WORKER' in values:
                ast.parse(values['FRAMEWORK_WORKER'])
                for operation in ('batch_matmul_backward', 'self_attention'):
                    self.assertIn(operation, values['FRAMEWORK_WORKER'])
                    self.assertIn(operation, values['CPP_SOURCE'])

    def test_runtime_helper_works_with_old_or_missing_checkout(self):
        source = next(s for s in sources() if 'MICRO_SUITE_SOURCE = ' in s)
        values = literals(source)
        self.assertEqual(values['MICRO_SUITE_SOURCE'],
                         (ROOT / 'scripts/benchmarks/colab_micro_suite.py').read_text())
        prefix = source[source.index('import types'):source.index('MICRO_ENTRIES=')]
        with tempfile.TemporaryDirectory() as directory:
            scope = dict(RUN_DIR=Path(directory), REPO=Path(directory) / 'old-snapshot')
            exec(compile(prefix, 'notebook-helper', 'exec'), scope)
            module = scope['MICRO_SUITE']
            entries = module.contenders(['mimir','jax'], 'cuda', ['vulkan','opencl'])
            self.assertEqual([e['label'] for e in entries], ['mimir-vulkan','mimir-opencl','jax'])
            for entry in entries[:2]:
                row = dict(framework='mimir', backend='cuda', case='test', runtime=entry['runtime'].upper())
                module.validate_identity(row, entry, 'cuda', 'test')
                row['runtime']='CPU'
                with self.assertRaisesRegex(RuntimeError, 'Wrong native'):
                    module.validate_identity(row, entry, 'cuda', 'test')
            with self.assertRaises(ValueError):module.contenders(['mimir'], 'cuda', [])
            self.assertEqual(module.contenders(['mimir'], 'cpu', [])[0]['label'], 'mimir')
            self.assertTrue((Path(directory) / 'colab_micro_suite.py').is_file())

    def test_embedded_optimized_sources_and_validation_targets(self):
        expected = ['MatMulBackward', 'BatchMatMulForward', 'FlatJoin', 'AttentionTiled']
        for name in ('Colab-test.ipynb', 'Colab-bench-multi-framework.ipynb'):
            cells = json.loads((ROOT / name).read_text())['cells']
            code = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
            setup = next(s for s in code if '# BEGIN MIMIR EMBEDDED SOURCES' in s)
            values = literals(setup)
            packed = base64.b64decode(values['SOURCE_PAYLOAD'], validate=True)
            self.assertEqual(hashlib.sha256(packed).hexdigest(), values['SOURCE_SHA256'])
            with tarfile.open(fileobj=io.BytesIO(gzip.decompress(packed))) as archive:
                manifest = json.load(archive.extractfile('colab-source-manifest.json'))
                for path in ('src/runtimes/cpu/MatMulKernels.hpp', 'src/runtimes/cpu/FlatJoinKernels.hpp',
                             'src/runtimes/cpu/LayerOps.hpp', 'src/runtimes/cpu/RuntimeLayerDispatch.hpp'):
                    digest = hashlib.sha256(archive.extractfile(path).read()).hexdigest()
                    self.assertEqual(digest, manifest['files'][path])
                    self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
            build = next(s for s in code if s.startswith('TEST_TARGETS='))
            validation = next(s for s in code if 'REFERENCE_TARGETS=' in s)
            for target in expected:
                self.assertIn('RuntimeTest.' + target, build)
                self.assertIn('RuntimeTest.' + target, validation)
                self.assertIn(target, validation.split("'--no-tests=error'")[1])


@unittest.skipIf(np is None, 'NumPy not installed')
class Fixtures(unittest.TestCase):
    def functions(self):
        source = next(s for s in sources() if 'def reference(f):' in s)
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        scope = dict(np=np, rng=np.random.default_rng(42))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), scope)
        return scope

    def test_portable_profile_is_shared_and_bounded(self):
        source = next(s for s in sources() if 'def reference(f):' in s)
        self.assertNotIn('MIMIR_GPU_RUNTIMES', source)
        with tempfile.TemporaryDirectory() as directory:
            scope = dict(PROFILE='portable', SEED=42, RUN_DIR=Path(directory),
                         ITERATIONS=2, WARMUP=1, TRIALS=1, json=json)
            exec(compile(source, 'portable-fixtures', 'exec'), scope)
            self.assertEqual(len(scope['CASES']), 7)
            self.assertEqual(len(scope['GOLD']), 7)
            self.assertIn('batch_matmul_backward', [c['op'] for c in scope['CASES']])
            self.assertIn('self_attention', [c['op'] for c in scope['CASES']])
            self.assertLess(max(len(c['x']) + len(c['weights']) for c in scope['CASES']), 65536)

    def test_matmul_backward_is_the_derivative_of_forward(self):
        f = self.functions()
        fixture = f['batch_matmul']('check', 2, 3, 5, 7, True)
        da, db = f['reference'](fixture)
        a, b, go = fixture['x'].astype(float), fixture['weights'].astype(float), fixture['target'].astype(float)
        eps = 1e-5
        for tensor, gradient in ((a, da), (b, db)):
            for index in (0, tensor.size - 1):
                old = tensor.flat[index]
                tensor.flat[index] = old + eps
                plus = np.sum((a @ b) * go)
                tensor.flat[index] = old - eps
                minus = np.sum((a @ b) * go)
                tensor.flat[index] = old
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * eps), places=8)

    def test_causal_attention_does_not_read_future_tokens(self):
        f = self.functions()
        fixture = f['attention']('causal', 129, 12, 3, True)
        before = f['reference'](fixture)[0].reshape(129, 12)
        fixture['x'][-1] += 100
        after = f['reference'](fixture)[0].reshape(129, 12)
        np.testing.assert_array_equal(before[:-1], after[:-1])
        self.assertTrue(np.isfinite(after).all())
        self.assertFalse(np.array_equal(before[-1], after[-1]))


if __name__ == '__main__':
    unittest.main()
