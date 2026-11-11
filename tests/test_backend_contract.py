"""端到端后端契约：本地 HTTP 服务，不调用外部模型。"""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

SECRET = 'test-secret-1739-judge'
TARGET_SECRET = 'test-secret-1739-target'

class Handler(BaseHTTPRequestHandler):
    calls = []
    def log_message(self, *args):
        pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.calls.append((self.path, dict(self.headers), body))
        if self.path == '/chat/completions':
            if body.get('model') == 'bad-json':
                text = 'not json ' + SECRET
            else:
                text = json.dumps({'scores': {'correctness': 2, 'instruction_following': 2, 'clarity': 2}, 'evidence': {'correctness': '正确'}, 'reasoning': 'OK ' + SECRET, 'confidence': .9})
            obj = {'choices': [{'message': {'content': text}}], 'usage': {'total_tokens': 10}}
        else:
            trace = {'steps': [{'type': 'tool_call', 'name': 'lookup', 'args': {'id': 'A'}, 'result': '完成'}]}
            obj = {'answer': '正确 ' + TARGET_SECRET, 'trace': trace}
            if self.path == '/no-trace':
                obj.pop('trace')
            if self.path == '/bad-trace':
                obj['trace'] = 'invalid'
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(raw)

class BackendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global server, core, models
        import server
        from src import core, models
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:%d' % cls.http.server_port

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db, self.old_server_db, self.old_config = core.DB, server.DB, models.CONFIG
        core.DB = Path(self.tmp.name) / 'db.sqlite'
        server.DB = core.DB
        models.CONFIG = Path(self.tmp.name) / 'models.yaml'
        models.save_config(models.DEFAULT_CONFIG)
        self.client = TestClient(server.app)
        server.conn().close()
        self.assertEqual(self.client.post('/api/examples/general').status_code, 200)
        self.reports = []
        Handler.calls.clear()

    def tearDown(self):
        for report in self.reports:
            report.unlink(missing_ok=True)
        core.DB, server.DB, models.CONFIG = self.old_db, self.old_server_db, self.old_config
        self.tmp.cleanup()

    def run_api(self, **kwargs):
        body = {'product': 'general-example', 'stage': 'answer', 'force': True}
        body.update(kwargs)
        response = self.client.post('/api/runs', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        jid = response.json()['job_id']
        for _ in range(300):
            job = self.client.get('/api/jobs/' + jid).json()
            if job['status'] != 'running':
                break
            time.sleep(.01)
        self.assertEqual(job['status'], 'done', job)
        self.reports.append(Path(job['report']))
        detail = self.client.get('/api/runs/' + job['run_id']).json()
        return job, detail

    def target(self, path='/agent'):
        return {'url': self.url + path, 'response_path': 'answer', 'headers': {'Authorization': 'Bearer ' + TARGET_SECRET}}

    def test_grader_selection_and_invalid(self):
        _, detail = self.run_api(mock=True, grader='trajectory_rule')
        self.assertTrue(all(r['verdict'] == 'review' for r in detail['results']))
        self.assertEqual(json.loads(detail['meta']['meta'])['grader'], 'trajectory_rule')
        _, combined = self.run_api(mock=True, grader='keyword_rule+trajectory_rule')
        self.assertTrue(all(r['verdict'] == 'review' for r in combined['results']))
        r = self.client.post('/api/runs', json={'product': 'general-example', 'stage': 'answer', 'force': True, 'mock': True, 'grader': 'not_a_grader'})
        self.assertEqual(r.status_code, 400)

    def test_http_trace_and_expected_trace_not_actual(self):
        cases = [{'id': 'one', 'input': {'question': 'lookup'}, 'gold': {'answer': '正确'}, 'trace': {'steps': [{'type':'tool_call','name':'lookup'}]}, 'expect': {'required_tools': ['lookup']}}]
        path = Path(self.tmp.name) / 'cases.json'
        path.write_text(json.dumps({'cases': cases}), encoding='utf-8')
        stage = core.StageConfig(id='t', name='t', task_type='text', dataset=str(path), grader='tool_call_rule')
        for route, verdict in [('/agent', 'pass'), ('/no-trace', 'review'), ('/bad-trace', 'review')]:
            with self.subTest(route=route):
                target = core.HttpTarget(url=self.url + route, response_path='answer')
                rid = core.run_stage({'product': 'test'}, stage, target=target)
                con = core.init_db()
                result = core.get_results(con, rid)[0]
                con.close()
                self.assertEqual(result['verdict'], verdict, result)
                self.assertIn('正确', result['output'])

    def test_rubric_real_http_key_and_redaction(self):
        cfg = models.config_raw()
        cfg['roles']['judge'].update({'adapter':'openai_compat', 'base_url':self.url, 'model':'judge-test', 'vendor':'local-judge'})
        models.save_config(cfg)
        job, detail = self.run_api(target=self.target(), grader='llm_rubric', model_api_keys={'judge': SECRET}, version={'change_note': SECRET})
        judge_calls = [c for c in Handler.calls if c[0] == '/chat/completions']
        self.assertEqual(len(judge_calls), len(detail['results']))
        self.assertEqual(judge_calls[0][1]['Authorization'], 'Bearer ' + SECRET)
        self.assertIn('判定表', judge_calls[0][2]['messages'][0]['content'])
        self.assertTrue(all(r['verdict'] == 'pass' and r['metrics']['rubric_score'] == 1 for r in detail['results']), detail)
        visible = json.dumps([job, detail], ensure_ascii=False) + Path(job['report']).read_text() + models.CONFIG.read_text() + core.DB.read_bytes().decode('latin1')
        self.assertNotIn(SECRET, visible)
        self.assertNotIn(TARGET_SECRET, visible)
        self.assertEqual(len([c for c in Handler.calls if c[0] == '/agent']), len(detail['results']))

    def test_rubric_missing_key_and_invalid_result_review(self):
        cfg = models.config_raw()
        cfg['roles']['judge'].update({'adapter': 'openai_compat', 'base_url': self.url, 'model': 'judge-test', 'vendor': 'local-judge'})
        models.save_config(cfg)
        _, detail = self.run_api(target=self.target(), grader='llm_rubric')
        self.assertTrue(all(r['verdict'] == 'review' for r in detail['results']))
        self.assertFalse(any(c[0] == '/chat/completions' for c in Handler.calls))
        cfg['roles']['judge']['model'] = 'bad-json'
        models.save_config(cfg)
        job, detail = self.run_api(target=self.target(), grader='llm_rubric', model_api_keys={'judge': SECRET})
        self.assertTrue(all(r['verdict'] == 'review' for r in detail['results']))
        self.assertNotIn(SECRET, json.dumps([job, detail], ensure_ascii=False))

    def test_model_config_secret_field_and_credential_url(self):
        r = self.client.post('/api/models', json={'roles': {'judge': {'api_key': SECRET}}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn(SECRET, models.CONFIG.read_text())
        r = self.client.post('/api/models', json={'roles': {'judge': {'base_url': self.url + '?key=' + SECRET}}})
        self.assertEqual(r.status_code, 400)
        self.assertNotIn(SECRET, r.text + models.CONFIG.read_text())
        self.assertEqual(models.config_raw()['roles']['target']['model'], 'mock-target')

if __name__ == '__main__':
    unittest.main()
