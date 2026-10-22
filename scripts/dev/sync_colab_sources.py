#!/usr/bin/env python3
"""Refresh both Colab embedded source archives from their audited file manifest."""
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
NOTEBOOKS=['Colab-test.ipynb','Colab-bench-multi-framework.ipynb']


def main():
    notebook=json.loads((ROOT/NOTEBOOKS[0]).read_text())
    source=next(''.join(c['source']) for c in notebook['cells'] if '# BEGIN MIMIR EMBEDDED SOURCES' in ''.join(c['source']))
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(values['SOURCE_PAYLOAD'])))) as archive:
        manifest=json.load(archive.extractfile('colab-source-manifest.json'))
    paths=set(manifest['files'])|{'VERSION','config.sh','scripts/tests/test_config_cli.py','scripts/dev/sync_colab_sources.py'}
    contents={name:(ROOT/name).read_bytes() for name in sorted(paths)}
    manifest['files']={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    manifest['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest['provenance']='Release source snapshot from the working tree; commit identifies the base, files carry authoritative SHA-256 digests'
    manifest['version']=(ROOT/'VERSION').read_text().strip()
    contents['colab-source-manifest.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w',format=tarfile.PAX_FORMAT) as archive:
        for name,data in contents.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o755 if name.endswith('.sh') else 0o644
            archive.addfile(info,io.BytesIO(data))
    packed=gzip.compress(output.getvalue(),mtime=0)
    digest=hashlib.sha256(packed).hexdigest();payload=base64.b64encode(packed).decode()
    for name in NOTEBOOKS:
        path=ROOT/name;notebook=json.loads(path.read_text())
        cell=next(c for c in notebook['cells'] if '# BEGIN MIMIR EMBEDDED SOURCES' in ''.join(c['source']))
        source=''.join(cell['source'])
        source=re.sub(r'^SOURCE_SHA256 = .*$',lambda _: 'SOURCE_SHA256 = '+repr(digest),source,flags=re.MULTILINE)
        source=re.sub(r'^SOURCE_PAYLOAD = .*$',lambda _: 'SOURCE_PAYLOAD = '+repr(payload),source,flags=re.MULTILINE)
        cell['source']=source.splitlines(keepends=True)
        path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print(f'Colab source snapshot: {len(manifest["files"])} files, SHA-256 {digest}')


if __name__=='__main__':main()
