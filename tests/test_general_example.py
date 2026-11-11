"""通用示例的隔离端到端回归：python -m unittest discover -s tests -v。"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

_tmp = tempfile.TemporaryDirectory()
os.environ['EVAL_DB'] = str(Path(_tmp.name) / 'test.db')
import server
from fastapi.testclient import TestClient


class GeneralExampleTest(unittest.TestCase):
    def test_upload_run_report(self):
        from src import core
        old_db, old_server_db = core.DB, server.DB
        core.DB = server.DB = Path(_tmp.name) / 'test.db'
        self.addCleanup(lambda: setattr(core, 'DB', old_db))
        self.addCleanup(lambda: setattr(server, 'DB', old_server_db))
        client = TestClient(server.app)
        self.assertEqual(client.get('/api/products').json(), [])
        self.assertEqual(client.post('/api/examples/general').status_code, 200)
        self.assertEqual(client.post('/api/examples/general').status_code, 409)
        original_uploads = server.UPLOADS
        server.UPLOADS = Path(_tmp.name) / 'uploads'
        server.UPLOADS.mkdir(exist_ok=True)
        report = None
        try:
            raw = (server.ROOT / 'data/general_example.json').read_bytes()
            res = client.post('/api/datasets?product=general-example&stage=answer', files={'file': ('example.json', raw, 'application/json')})
            self.assertEqual(res.status_code, 200, res.text)
            did = res.json()['id']
            body = {'product': 'general-example', 'stage': 'answer', 'dataset_id': did, 'mock': True}
            self.assertEqual(client.post('/api/runs', json=body).status_code, 409)
            body['force'] = True
            res = client.post('/api/runs', json=body)
            self.assertEqual(res.status_code, 200, res.text)
            jid = res.json()['job_id']
            for _ in range(200):
                job = client.get('/api/jobs/' + jid).json()
                if job['status'] != 'running':
                    break
                time.sleep(.05)
            self.assertEqual(job['status'], 'done', job)
            report = Path(job['report'])
            rid = job['run_id']
            detail = client.get('/api/runs/' + rid).json()
            self.assertEqual(len(detail['results']), 2, detail)
            self.assertTrue(all(r['verdict'] == 'pass' and r['output'] for r in detail['results']))
            html = client.get('/api/runs/' + rid + '/report')
            self.assertEqual(html.status_code, 200)
            self.assertIn('100', html.text)
            for legacy in ('心愿基金', '金额一致率', '打扰率', 'OCR 文本'):
                self.assertNotIn(legacy, html.text)
        finally:
            server.UPLOADS = original_uploads
            if report:
                report.unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()
