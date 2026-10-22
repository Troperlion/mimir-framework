"""Exercise native Colab diagnostics and their resource-budget contract."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('native_suite', ROOT / 'scripts/benchmarks/colab_native_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def record():
    return dict(final_loss=0.5, parameter_checksum=1, cold_trainstep_ms=2,
                total=dict(median_ms=1, p95_ms=1.5, samples_ms=[1, 1.5]))


class NativeBench(unittest.TestCase):
    def test_profiles(self):
        self.assertEqual(len(suite.cases('quick')), 2)
        self.assertEqual(len(suite.cases('standard')), 4)
        self.assertEqual(suite.cases('standard')[-1]['arguments'][0], '512')
        with self.assertRaises(ValueError):
            suite.cases('unknown')

    def test_launcher_output_and_invalid_measurement(self):
        value = record()
        self.assertEqual(suite.measurement('launcher diagnostics\n' + json.dumps(value)), value)
        value['total']['median_ms'] = float('nan')
        with self.assertRaises(ValueError):
            suite.measurement(json.dumps(value))
        with self.assertRaises(ValueError):
            suite.measurement('no JSON result')

    def test_launcher_budgets_and_persistent_failures(self):
        calls = []
        def runner(command, env, log, timeout, memory, cwd):
            calls.append((command, env, timeout, memory, cwd))
            log.write_text(json.dumps(record()))
            return dict(budget_status='completed' if len(calls) == 1 else 'memory_limit')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = suite.run_suite(ROOT, ROOT / 'build', output, suite.cases('quick'),
                                   {'OMP_NUM_THREADS': '2'}, [2, 3], timeout_s=7,
                                   memory_mib=123, runner=runner)
            self.assertEqual([r['status'] for r in rows], ['ok', 'memory_limit'])
            self.assertEqual(json.loads((output / 'records.json').read_text()), rows)
            self.assertTrue(all(not r['participates_in_comparative_score'] for r in rows))
            command, env, timeout, memory, cwd = calls[0]
            self.assertEqual(command[:5], ['taskset', '-c', '2,3', 'bash', str(ROOT / 'run_mimir.sh')])
            self.assertEqual((timeout, memory, cwd), (7, 123, output))
            self.assertEqual(env['GOMP_CPU_AFFINITY'], '2 3')
            self.assertEqual(env['MIMIR_PLANNER_MODE'], 'cost')
            self.assertEqual(env['MIMIR_EXECUTABLE'], str(ROOT / 'build/Tests/mimir_bench_whole_model'))
            self.assertNotIn('measurement', rows[1])

    def test_notebook_skips_native_diagnostic_on_tpu(self):
        notebook = json.loads((ROOT / 'Colab-bench-multi-framework.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if ''.join(c.get('source', [])).startswith('RUN_NATIVE_BENCH=True'))
        scope = dict(PROFILE='quick', BACKEND='tpu', print=lambda *a: None)
        exec(compile(source, 'native-notebook-cell', 'exec'), scope)
        self.assertEqual(scope['NATIVE_RECORDS'], [])

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            suite.run_suite(ROOT, ROOT / 'build', ROOT / 'build', [], {}, [])


if __name__ == '__main__':
    unittest.main()
