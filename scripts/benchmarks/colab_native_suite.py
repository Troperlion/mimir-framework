"""Bounded Mimir-only diagnostics; never part of the cross-framework score."""
import importlib.util
import json
import math
from pathlib import Path


def cases(profile):
    if profile not in {'quick', 'standard'}:
        raise ValueError('Expected quick or standard profile')
    steps = 5 if profile == 'quick' else 20
    result = [dict(case='linear-small-sgd', target='mimir_bench_whole_model',
                   arguments=['float32', str(steps), '0', '64', '16', '8'])]
    if profile == 'standard':
        result.append(dict(case='linear-large-adamw', target='mimir_bench_whole_model',
                           arguments=['float32', str(steps), '2', '256', '256', '4']))
    for side in ([64] if profile == 'quick' else [128, 512]):
        result.append(dict(case=f'vgg16-{side}-lion', target='mimir_bench_vgg16_whole_model',
                           arguments=[str(side), '5', '4' if profile == 'quick' else '16']))
    return result


def measurement(log):
    # The production launcher prints diagnostics before the final JSON record.
    for line in reversed(log.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and ('training' in value or 'total' in value):
            for key in ('final_loss', 'parameter_checksum', 'cold_trainstep_ms'):
                if not math.isfinite(float(value[key])):
                    raise ValueError('Non-finite ' + key)
            for key in ('training', 'inference', 'forward', 'backward_and_loss', 'optimizer_step', 'total'):
                if key in value:
                    for statistic in ('median_ms', 'p95_ms'):
                        number = float(value[key][statistic])
                        if not math.isfinite(number) or number < 0:
                            raise ValueError('Invalid latency: ' + key)
            return value
    raise ValueError('Native benchmark JSON record missing')


def run_suite(repo, build, directory, configurations, environment, affinity,
              trials=1, timeout_s=300, memory_mib=4096, runner=None):
    if not affinity or trials < 1 or timeout_s <= 0 or memory_mib <= 0:
        raise ValueError('Invalid benchmark budgets')
    repo, build, directory = map(Path, (repo, build, directory))
    directory.mkdir(parents=True, exist_ok=True)
    if runner is None:
        spec = importlib.util.spec_from_file_location('mimir_native_budgets',
            Path(__file__).with_name('colab_real_suite.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = module.constrained_run
    records = []
    for config in configurations:
        for trial in range(trials):
            stem = f'{config["case"]}.{trial}'
            log = directory / (stem + '.log')
            env = dict(environment, MIMIR_EXECUTABLE=str((build / 'Tests' / config['target']).resolve()),
                       MIMIR_PLANNER_MODE='cost', MIMIR_PLANNER_AUTOTUNE='1',
                       MIMIR_RUNTIME_TRACE='0', MIMIR_ACCEL_VERBOSE='1',
                       MIMIR_PLANNER_DUMP='0', MIMIR_PLANNER_STDOUT='0',
                       GOMP_CPU_AFFINITY=' '.join(map(str, affinity)))
            command = ['taskset', '-c', ','.join(map(str, affinity)), 'bash',
                       str(repo / 'run_mimir.sh'), *config['arguments']]
            row = dict(case=config['case'], trial=trial, kind='mimir_native_diagnostic',
                       participates_in_comparative_score=False, log=str(log))
            print('Diagnostic natif:', stem, flush=True)
            try:
                budget = runner(command, env, log, timeout_s, memory_mib, cwd=directory)
                row.update(budget)
                row['status'] = budget['budget_status']
                if row['status'] == 'completed':
                    row['measurement'] = measurement(log.read_text(errors='replace'))
                    row['status'] = 'ok'
            except Exception as error:
                row.update(status='error', error=str(error))
            records.append(row)
            (directory / (stem + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    (directory / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    return records
