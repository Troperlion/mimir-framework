#!/usr/bin/env python3
"""End-to-end GAN training, exact epoch resume and folder inference (CPU)."""
from pathlib import Path
import csv
import os
import struct
import subprocess
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[2]


def png(path, offset):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\0' + b''.join(bytes([(x * 25 + offset) % 256, y * 25, 120]) for x in range(8)) for y in range(8))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 8, 8, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def run(*args):
    result = subprocess.run([str(ROOT / 'bin/mimir'), '--lua', 'scripts/training/train_gan_style.lua', '--cpu', *map(str, args)], cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': '1'}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


with tempfile.TemporaryDirectory(prefix='mimir-gan-test-') as tmp:
    base = Path(tmp)
    source, style = base / 'source', base / 'style'
    source.mkdir(); style.mkdir()
    for i in range(2):
        png(source / f'{i}.png', i * 30)
        png(style / f'{i}.png', 100 + i * 30)
    common = ['--source-root', source, '--style-root', style, '--size', 8, '--channels', 4]
    resumed, continuous = base / 'resumed', base / 'continuous'
    run(*common, '--epochs', 1, '--out-dir', resumed)
    first = resumed / 'epoch_0001'
    run(*common, '--resume', first, '--epochs', 2, '--out-dir', resumed)
    run(*common, '--epochs', 2, '--out-dir', continuous)
    second, reference = resumed / 'epoch_0002', continuous / 'epoch_0002'
    for path in reference.rglob('*.bin'):
        relative = path.relative_to(reference)
        assert path.read_bytes() == (second / relative).read_bytes(), f'resume mismatch: {relative}'
    assert (second / 'gan_state.txt').read_bytes() == (reference / 'gan_state.txt').read_bytes()
    assert (first / 'tensors/g/conv1_weights.bin').read_bytes() != (second / 'tensors/g/conv1_weights.bin').read_bytes(), 'G weights did not change'
    assert (first / 'tensors/d/conv1_weights.bin').read_bytes() != (second / 'tensors/d/conv1_weights.bin').read_bytes(), 'D weights did not change'
    rows = list(csv.DictReader((resumed / 'loss.csv').open()))
    assert [(r['epoch'], r['step']) for r in rows] == [('1', '1'), ('1', '2'), ('2', '1'), ('2', '2')]
    output = base / 'inference'
    run('--checkpoint', second, '--input-root', source, '--output-dir', output)
    assert len(list(output.glob('*.ppm'))) == 2
    run('--checkpoint', second, '--input', source / '0.png', '--output', base / 'single.ppm')
    assert (base / 'single.ppm').read_bytes() == (output / '000001.ppm').read_bytes()
    assert (base / 'single.ppm').read_bytes() == (second / 'preview.ppm').read_bytes(), 'loaded model output differs from training model'
    print('GAN_STYLE_OK: G/D updates, exact resume, dataset traversal, single/folder inference')
