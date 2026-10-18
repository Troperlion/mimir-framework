#!/usr/bin/env python3
"""Isolated HTTP integration test; no access to the user's dataset."""
import base64
import copy
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import urllib.parse

APP = Path(__file__).resolve().parents[1]
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jGZkAAAAASUVORK5CYII=')


class Client:
    def __init__(self, base):
        self.base = base
        self.root = None
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        page = self.opener.open(base + '/').read().decode()
        self.token = re.search(r'name="csrf-token" content="([^"]+)"', page)[1]

    def post(self, endpoint, payload, expected=200, token=None):
        if endpoint != 'scan':
            payload = {**payload, 'dataset_root': self.root}
        req = urllib.request.Request(self.base + '/api/' + endpoint + '.php', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json', 'X-CSRF-Token': self.token if token is None else token})
        try:
            response = self.opener.open(req)
        except urllib.error.HTTPError as exc:
            response = exc
        raw = response.read()
        try:
            result = json.loads(raw)
        except ValueError:
            raise AssertionError((endpoint, response.code, raw))
        assert response.code == expected, (endpoint, response.code, result)
        if endpoint == 'scan' and response.code == 200:
            self.root = result['root']
        return result


def main():
    with tempfile.TemporaryDirectory(prefix='lumen-annotation-test-') as temp:
        root = Path(temp) / 'dataset'
        (root / 'portraits').mkdir(parents=True)
        (root / 'animals').mkdir()
        for name in ['portraits/étoile.PNG', 'portraits/second.png', 'animals/dog.png']:
            (root / name).write_bytes(PNG)
        originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.png')}
        (Path(temp) / 'outside.png').write_bytes(PNG)
        (root / 'escape.png').symlink_to(Path(temp) / 'outside.png')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        sessions = Path(temp) / 'sessions'; sessions.mkdir()
        cmd = [os.environ.get('PHP_BIN', 'php'), *shlex.split(os.environ.get('PHP_ARGS', '')), '-d', 'session.save_path=' + str(sessions), '-S', f'127.0.0.1:{port}', '-t', str(APP)]
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        client = Client(base)
                        break
                    except urllib.error.URLError:
                        if process.poll() is not None:
                            log.seek(0)
                            raise RuntimeError(log.read().decode())
                        time.sleep(.05)
                else:
                    raise RuntimeError('PHP server unavailable')
                scanned = client.post('scan', {'root': str(root)})
                assert len(scanned['images']) == 3 and len(scanned['warnings']) == 1
                relative = 'portraits/étoile.PNG'
                loaded = client.post('load', {'image': relative})
                assert loaded['revision'] is None and not loaded['exists']
                a = loaded['annotation']
                a['captions']['primary'] = 'Une femme près d’une fenêtre, éclairée par le soleil.'
                a['captions']['alternatives'] = ['Un portrait baigné de lumière.']
                a['entities'] = [{'id':'person_1','phrase':'une femme','role':'primary_subject','bbox':[.1,.2,.5,.9]},
                                 {'id':'window_1','phrase':'une fenêtre','role':'object','bbox':[.6,0,1,1]}]
                a['attributes'] = [{'entity':'person_1','property':'hair_color','value':'black'}]
                a['relations'] = [{'subject':'person_1','predicate':'near','object':'window_1'}]
                a['custom_metadata'] = {'preserved': True}
                a['entities'][0]['skeleton'] = {'name':'humain', 'points':[
                    {'name':'hanche','position':[.3,.6],'parent':None,'role':'articulation majeure'},
                    {'name':'tête','position':[.3,.2],'parent':'hanche','role':'tête'},
                    {'name':'bras gauche','position':None,'parent':'hanche'}]}

                a['entities'][0]['skeleton']['points'][1]['entities'] = ['person_1','window_1']
                a['entities'][0]['skeleton']['edges'] = [{'from':'tête','to':'bras gauche','label':'articulation'}]
                saved = client.post('save', {'image':relative, 'annotation':a, 'revision':None})
                assert (root / 'portraits/étoile.json').exists()
                assert json.loads((root / 'portraits/étoile.json').read_text()) == a
                reloaded = client.post('load', {'image':relative})
                assert reloaded['annotation'] == a and reloaded['revision'] == saved['revision']
                client.post('save', {'image':relative,'annotation':a,'revision':None}, 409)
                for mutation in ['position', 'duplicate', 'parent', 'name', 'shape', 'role']:
                    bad = copy.deepcopy(a)
                    skeleton = bad['entities'][0]['skeleton']
                    if mutation == 'position': skeleton['points'][0]['position'] = [1.1, .5]
                    if mutation == 'duplicate': skeleton['points'][1]['name'] = 'hanche'
                    if mutation == 'parent': skeleton['points'][1]['parent'] = 'absent'
                    if mutation == 'name': skeleton['name'] = ' '
                    if mutation == 'role': skeleton['points'][0]['role'] = 'invalid'
                    if mutation == 'shape': skeleton['points'][0]['position'] = [.5]
                    client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                for mutation in ['unknown_entity','duplicate_entity','bad_edge','self_edge']:
                    bad = copy.deepcopy(a); skeleton = bad['entities'][0]['skeleton']
                    if mutation == 'unknown_entity': skeleton['points'][1]['entities'] = ['missing']
                    if mutation == 'duplicate_entity': skeleton['points'][1]['entities'] = ['person_1','person_1']
                    if mutation == 'bad_edge': skeleton['edges'][0]['to'] = 'missing'
                    if mutation == 'self_edge': skeleton['edges'][0]['to'] = 'tête'
                    client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                legacy = copy.deepcopy(a); del legacy['entities'][0]['skeleton']
                legacy_saved = client.post('save', {'image':relative,'annotation':legacy,'revision':saved['revision']})
                assert client.post('load', {'image':relative})['annotation'] == legacy
                saved = client.post('save', {'image':relative,'annotation':a,'revision':legacy_saved['revision']})

                bad = copy.deepcopy(a); bad['entities'][0]['bbox'] = [-.1,0,1,1]
                client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                bad = copy.deepcopy(a); bad['relations'][0]['object'] = 'missing'
                client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                bad = copy.deepcopy(a); bad['captions']['primary'] = ' '
                client.post('save', {'image':relative,'annotation':bad,'revision':saved['revision']}, 400)
                client.post('load', {'image':'../outside.png'}, 403)
                client.post('load', {'image':'escape.png'}, 404)
                client.post('state', {}, 403, token='invalid')
                image = client.opener.open(base + '/api/image.php?image=portraits/second.png&dataset_root=' + urllib.parse.quote(str(root)))
                assert image.headers['Content-Type'] == 'image/png' and image.read() == PNG
                client.post('state', {'image':'portraits/second.png'})
                state = json.loads((root / '.annotation_state.json').read_text())
                assert state['last_image'] == 'portraits/second.png' and relative in state['annotated']
                fresh = Client(base)
                resumed = fresh.post('scan', {'root':str(root)})
                assert resumed['images'][resumed['resume_index']] == 'portraits/second.png'
                assert resumed['state']['last_annotation'] == relative and resumed['annotated'] == [relative]
                fresh.post('state', {'image':relative})
                resumed = Client(base).post('scan', {'root':str(root)})
                assert resumed['images'][resumed['resume_index']] not in resumed['annotated']
                broken = Path(temp) / 'broken'; broken.mkdir(); (broken / '.annotation_state.json').write_text('invalid')
                fresh.post('scan', {'root':str(broken)}, 400)
                assert fresh.post('load', {'image':relative})['annotation'] == a
                # A conflicting existing JSON must never be replaced with a blank annotation.
                unrelated = root / 'animals/dog.json'; unrelated.write_text('{"foreign":true}')
                fresh.post('load', {'image':'animals/dog.png'}, 400)
                assert unrelated.read_text() == '{"foreign":true}'
                unrelated.unlink()
                (root / 'portraits/second.jpg').write_bytes(PNG)
                fresh.post('load', {'image':'portraits/second.png'}, 409)
                # A sidecar symlink cannot overwrite anything outside the selected root.
                target = Path(temp) / 'outside.json'; target.write_text('{"untouched":true}')
                (root / 'animals/dog.json').symlink_to(target)
                fresh.post('save', {'image':'animals/dog.png','annotation':a,'revision':None}, 409)
                assert target.read_text() == '{"untouched":true}'
                req = urllib.request.Request(base + '/', headers={'Host':'attacker.example'})
                try:
                    fresh.opener.open(req)
                    raise AssertionError('Foreign Host accepted')
                except urllib.error.HTTPError as exc:
                    assert exc.code == 403
                other = Path(temp) / 'other'; other.mkdir(); (other / 'dog.png').write_bytes(PNG)
                client.post('scan', {'root':str(other)})
                client.root = str(root)  # Simulate the stale tab sharing this session.
                client.post('load', {'image':relative}, 409)
                for path, digest in originals.items():
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
                assert not list(root.rglob('.annotation-tmp-*'))
                print('PASS: recursive scan, UTF-8 sidecars, reload, metadata, normalized boxes, references, revisions, exact resume, CSRF, Host, traversal, symlinks, name collisions, scan rollback, dataset switch, unchanged images.')
            except Exception:
                log.seek(0)
                print(log.read().decode())
                raise
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
