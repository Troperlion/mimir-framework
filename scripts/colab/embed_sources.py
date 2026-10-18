#!/usr/bin/env python3
"""Refresh the self-contained Colab source snapshot from this checkout.

Run after source changes, before distributing either notebook. Only explicit
build/source directories are included; datasets, checkpoints, secrets, build
outputs and notebook files are excluded. No CPU or runtime source is changed.
"""
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ('Colab-test.ipynb', 'Colab-bench-multi-framework.ipynb')
BEGIN = '# BEGIN MIMIR EMBEDDED SOURCES'
END = '# END MIMIR EMBEDDED SOURCES'


def source_files():
    files = [ROOT / 'CMakeLists.txt', ROOT / 'VERSION', ROOT / 'run_mimir.sh']
    for directory in ('src', 'Tests', 'scripts', 'configs', 'shaders', 'fpga'):
        for path in (ROOT / directory).rglob('*'):
            if any(part in {'__pycache__', '.git', 'node_modules', 'build', '.pytest_cache'}
                   for part in path.relative_to(ROOT).parts):
                continue
            if path.suffix in {'.pyc', '.o', '.a', '.so', '.vvp', '.bin', '.log'}:
                continue
            if path.is_symlink():
                raise RuntimeError(f'Source symlink requires explicit handling: {path}')
            if path.is_file():
                files.append(path)
    return sorted(set(files))


def snapshot():
    content = {str(p.relative_to(ROOT)): p.read_bytes() for p in source_files()}
    manifest = dict(format=1,
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    provenance='Current local sources, including uncommitted and untracked build inputs',
                    files={p: hashlib.sha256(data).hexdigest() for p, data in content.items()})
    content['colab-source-manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, data in sorted(content.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o755 if name.endswith(('.sh', '.py')) else 0o644
            archive.addfile(entry, io.BytesIO(data))
    packed = gzip.compress(stream.getvalue(), mtime=0)
    return packed, hashlib.sha256(packed).hexdigest(), len(content)


def restore_code(packed, digest):
    # repr keeps the payload a Python string literal, with no shell interpolation.
    return BEGIN + '\n' + '''import base64, gzip, io, tarfile, tempfile
SOURCE_MODE = "embedded"  # embedded | checkout (explicit opt-in to REPO/BRANCH)
''' + 'SOURCE_SHA256 = ' + repr(digest) + '\nSOURCE_PAYLOAD = ' + repr(base64.b64encode(packed).decode()) + '''
if SOURCE_MODE == 'embedded':
    packed = base64.b64decode(SOURCE_PAYLOAD, validate=True)
    if hashlib.sha256(packed).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Instantané des sources corrompu : recharger le notebook')
    # A dedicated immutable source directory preserves any existing checkout.
    snapshot_root = REPO.parent / ('mimir-source-' + SOURCE_SHA256[:16])
    def verify_snapshot(root, manifest):
        return all((root / name).is_file() and
                   hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                   for name, digest in manifest['files'].items())
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(packed)), mode='r:') as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile():
                raise RuntimeError('Chemin interdit dans les sources : ' + member.name)
        manifest = json.load(archive.extractfile('colab-source-manifest.json'))
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(manifest['files']) | {'colab-source-manifest.json'}:
            raise RuntimeError('Inventaire des sources invalide')
        if snapshot_root.exists():
            if not verify_snapshot(snapshot_root, manifest):
                raise RuntimeError('Instantané local modifié : supprimer ce dossier puis relancer : ' + str(snapshot_root))
        else:
            snapshot_root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='mimir-extract-', dir=snapshot_root.parent) as temporary:
                temporary = Path(temporary)
                for member in members:
                    target = temporary / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
                    target.chmod(member.mode)
                if not verify_snapshot(temporary, manifest):
                    raise RuntimeError('Empreintes des sources invalides')
                temporary.rename(snapshot_root)
    REPO = snapshot_root
    COMMIT = manifest['commit']
    DIRTY = manifest['provenance'] + '; snapshot_sha256=' + SOURCE_SHA256
elif SOURCE_MODE == 'checkout':
    if not REPO.exists():
        run(['git', 'clone', '--branch', BRANCH, REPO_URL, REPO])
    COMMIT = run(['git', 'rev-parse', 'HEAD'], cwd=REPO).strip()
    DIRTY = run(['git', 'status', '--short'], cwd=REPO)
else:
    raise ValueError('SOURCE_MODE doit être embedded ou checkout')
BUILD = REPO / f'build-colab-{BACKEND}'
print('Sources :', REPO, '| mode :', SOURCE_MODE)
''' + END


def vulkan_helper_code():
    helper = (ROOT / 'scripts/colab/nvidia_vulkan.py').read_text()
    digest = hashlib.sha256(helper.encode()).hexdigest()
    return ('# BEGIN MIMIR VULKAN HELPER\n'
            + '    import types\n'
            + '    vulkan_helper_source = ' + repr(helper) + '\n'
            + '    vulkan_helper_sha256 = ' + repr(digest) + '\n'
            + "    if hashlib.sha256(vulkan_helper_source.encode()).hexdigest()!=vulkan_helper_sha256:raise RuntimeError('Module Vulkan embarque corrompu')\n"
            + "    vulkan_setup_module=types.ModuleType('mimir_nvidia_vulkan')\n"
            + "    exec(compile(vulkan_helper_source,'<mimir_nvidia_vulkan_embedded>','exec'),vulkan_setup_module.__dict__)\n"
            + '    # END MIMIR VULKAN HELPER')


def main():
    packed, digest, count = snapshot()
    replacement = restore_code(packed, digest)
    for name in NOTEBOOKS:
        path = ROOT / name
        notebook = json.loads(path.read_text())
        updated = 0
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and "apt=['apt-get']" in source:
                if BEGIN in source:
                    start, end = source.index(BEGIN), source.index(END) + len(END)
                else:
                    start = source.index('if not REPO.exists():')
                    end = source.index("required=['src/runtimes/cpu/CpuRuntime.cpp'")
                source = source[:start] + replacement + '\n' + source[end:]
                # The setup must also work if REPO still points to an older snapshot.
                if '# BEGIN MIMIR VULKAN HELPER' in source:
                    a = source.index('# BEGIN MIMIR VULKAN HELPER')
                    b = source.index('# END MIMIR VULKAN HELPER', a) + len('# END MIMIR VULKAN HELPER')
                    source = source[:a] + vulkan_helper_code() + source[b:]
                elif "    import importlib.util\n    spec=importlib.util.spec_from_file_location('mimir_nvidia_vulkan'" in source:
                    a = source.index('    import importlib.util', source.index('# Restore NVIDIA Vulkan'))
                    b = source.index('    RUNTIME_ENV,VULKAN_SETUP=', a)
                    source = source[:a] + '    ' + vulkan_helper_code() + '\n' + source[b:]
                source = source.replace('Synchroniser les changements actuels de Mímir avant de compiler : ',
                                        'Sources incomplètes ; en mode checkout, utiliser embedded ou synchroniser la branche : ')
                cell['source'] = source.splitlines(True)
                cell['outputs'] = []
                cell['execution_count'] = None
                updated += 1
            elif cell['cell_type'] == 'code' and 'MICRO_SUITE_SOURCE = ' in source:
                node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'MICRO_SUITE_SOURCE')
                lines = source.splitlines(True)
                helper = (ROOT / 'scripts/benchmarks/colab_micro_suite.py').read_text()
                lines[node.lineno - 1:node.end_lineno] = ['MICRO_SUITE_SOURCE = ' + repr(helper) + '\n']
                cell['source'] = ''.join(lines).splitlines(True)
                cell['outputs'] = []
                cell['execution_count'] = None
            elif cell['cell_type'] == 'markdown':
                source = source.replace('Un checkout existant est conservé. La branche distante doit contenir les changements runtime à tester.',
                    'Par défaut, les sources actuelles sont embarquées dans ce notebook et vérifiées par SHA256. Elles sont extraites dans un dossier dédié ; un checkout existant est conservé. `SOURCE_MODE="checkout"` permet de choisir explicitement la branche distante.')
                source = source.replace('Le checkout doit contenir les changements à mesurer. Les sources CPU ne sont pas modifiées.',
                    'Les sources actuelles sont embarquées et vérifiées par SHA256, y compris les fichiers CUDA et TPU non encore publiés sur la branche distante. Extraction dans un dossier dédié, sans modification des sources CPU. `SOURCE_MODE="checkout"` active explicitement un checkout externe.')
                cell['source'] = source.splitlines(True)
        if updated != 1:
            raise RuntimeError(f'{name}: expected exactly one source cell, got {updated}')
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print(f'Embedded {count} files, {len(packed)} compressed bytes, sha256={digest}')


if __name__ == '__main__':
    main()
