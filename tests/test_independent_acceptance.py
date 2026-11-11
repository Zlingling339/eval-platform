"""独立模型真实本地 HTTP 安全及比较回归。"""
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from fastapi.testclient import TestClient
import server
from src import core


def test_real_http_standalone_security(tmp_path, monkeypatch):
    secret = 'regression-key-' + uuid.uuid4().hex
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, self.headers.get('Authorization'), data))
            raw = json.dumps({'choices': [{'message': {'content': 'yes ' + secret}}], 'usage': {'total_tokens': 4}}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(raw)
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    monkeypatch.setattr(core, 'DB', tmp_path / 'db.sqlite')
    monkeypatch.setattr(server, 'DB', core.DB)
    uploads = tmp_path / 'uploads'; uploads.mkdir()
    monkeypatch.setattr(server, 'UPLOADS', uploads)
    root = tmp_path / 'root'; (root / 'reports').mkdir(parents=True)
    monkeypatch.setattr(server, 'ROOT', root)
    client = TestClient(server.app)
    try:
        assert client.get('/api/products').json() == []
        def upload(question):
            doc = {'cases': [{'id':'a', 'input': {'question':question}, 'gold': {'answer':'yes'}, 'expect': {'must_mention':['yes']}, 'mock_reply':'yes'}]}
            r = client.post('/api/datasets', files={'file': ('data.json', json.dumps(doc), 'application/json')})
            assert r.status_code == 200, r.text
            return r.json()['id']
        did = upload('hello world')
        body = {'scope':'standalone_model', 'dataset_id':did, 'grader':'keyword_rule', 'mock':False, 'model':'local-test', 'api_key':secret, 'base_url':f'http://127.0.0.1:{http.server_port}/v1', 'version':{'change_note':secret}}
        assert client.post('/api/runs', json=body).status_code == 409
        assert client.post(f'/api/datasets/{did}/review', json={'reviewed':True}).status_code == 200
        def run(**kw):
            r = client.post('/api/runs', json={**body, **kw}); assert r.status_code == 200, r.text
            for _ in range(500):
                job = client.get('/api/jobs/' + r.json()['job_id']).json()
                if job['status'] != 'running': break
                time.sleep(.01)
            assert job['status'] == 'done', job
            return job
        a, b = run(), run()
        assert len(calls) == 2
        assert all(p == '/v1/chat/completions' and auth == 'Bearer ' + secret and data['model'] == 'local-test' for p, auth, data in calls)
        for job in (a,b):
            detail = client.get('/api/runs/' + job['run_id']); assert detail.status_code == 200
            assert detail.json()['results'][0]['verdict'] == 'pass'
            assert secret not in detail.text
            report = client.get('/api/reports/' + job['run_id'] + '.html'); assert report.status_code == 200 and secret not in report.text
        for endpoint, key in [('/api/compare','run_id'), ('/api/version-compare','version_id')]:
            r = client.get(endpoint, params={'a':a[key], 'b':b[key]}); assert r.status_code == 200, r.text
            if endpoint.endswith('version-compare'):
                assert client.get(r.json()['url']).status_code == 200
        other = upload('another question')
        assert client.post(f'/api/datasets/{other}/review', json={'reviewed':True}).status_code == 200
        for changed in (run(mock=True), run(repeat=2), run(dataset_id=other)):
            for endpoint, key in [('/api/compare','run_id'), ('/api/version-compare','version_id')]:
                r = client.get(endpoint, params={'a':a[key], 'b':changed[key]}); assert r.status_code == 400, r.text
        for path in tmp_path.rglob('*'):
            if path.is_file(): assert secret.encode() not in path.read_bytes(), str(path)
    finally:
        http.shutdown(); http.server_close(); thread.join()
