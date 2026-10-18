"""Runtime identities shared by the Colab microbenchmark and its tests."""
GPU_RUNTIMES = ('cuda', 'vulkan', 'opencl')


def contenders(frameworks, backend, runtimes):
    if backend not in ('cpu', 'cuda', 'tpu'):
        raise ValueError('Invalid benchmark backend')
    if len(set(frameworks)) != len(frameworks):
        raise ValueError('Duplicate framework')
    if len(set(runtimes)) != len(runtimes) or any(r not in GPU_RUNTIMES for r in runtimes):
        raise ValueError('Invalid or duplicate GPU runtime')
    if backend == 'cuda' and 'mimir' in frameworks and not runtimes:
        raise ValueError('No validated native Mimir GPU runtime')
    result = []
    for framework in frameworks:
        if framework not in ('mimir', 'pytorch', 'tensorflow', 'jax'):
            raise ValueError('Invalid framework')
        if framework == 'mimir' and backend == 'cuda':
            result.extend(dict(label='mimir-' + runtime, framework='mimir', runtime=runtime)
                          for runtime in runtimes)
        else:
            result.append(dict(label=framework, framework=framework, runtime=backend))
    return result


def validate_identity(record, contender, backend, case):
    if (record.get('framework'), record.get('backend'), record.get('case')) != (
            contender['framework'], backend, case):
        raise RuntimeError('Incorrect benchmark result identity')
    if contender['framework'] == 'mimir' and record.get('runtime') != contender['runtime'].upper():
        raise RuntimeError('Wrong native Mimir runtime: ' + str(record.get('runtime')))
