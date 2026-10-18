#!/usr/bin/env python3
"""Exercise the shared script scene through real WEB transport."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--executable', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
with (out/'script-ui.log').open('w') as log:
    process = subprocess.Popen([args.executable, '15'], stdout=log, stderr=log,
                               env=dict(os.environ, DISPLAY=''))
    try:
        for _ in range(100):
            text = (out/'script-ui.log').read_text()
            match = re.search(r'http://127\.0\.0\.1:\d+/[a-f0-9]+/', text)
            if match and 'configured' in text:
                break
            if process.poll() is not None:
                raise AssertionError(text)
            time.sleep(.05)
        assert match and 'configured' in text, text
        base = match.group(0).rstrip('/')
        def request(path, method='GET'):
            with urllib.request.urlopen(urllib.request.Request(base+path, method=method), timeout=3) as response:
                return response.read()
        def event(kind, x=0, y=0, value=0):
            request(f'/event?t={kind}&x={x}&y={y}&v={value}&m=0&d=0','POST')
        (out/'controls.json').write_bytes(request('/frame'))
        event(3,40,35)
        event(4,40,35)
        # Help must cover controls and suppress their activation.
        event(6,value=72)
        time.sleep(.2)
        (out/'help.json').write_bytes(request('/frame'))
        event(3,40,35)
        event(4,40,35)
        time.sleep(.2)
        event(1)
        assert process.wait(timeout=5)==0, (out/'script-ui.log').read_text()
        text = (out/'script-ui.log').read_text()
        assert text.count('"type":"click"')==1, text
        for token in ['"type":"pointer_down"','"type":"pointer_up"','"type":"key"','SCRIPT_UI_RESULT 1 1']:
            assert token in text, text
        assert (out/'controls.json').read_bytes() != (out/'help.json').read_bytes()
        print('SCRIPT_WEB_OK: configure, click, raw events, help overlay, close')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
