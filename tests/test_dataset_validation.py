"""数据集结构校验与模型默认值回归测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
import server
from src import core, models


class DatasetValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db, self.old_server_db, self.old_uploads = core.DB, server.DB, server.UPLOADS
        core.DB = Path(self.tmp.name) / 'test.db'
        server.DB = core.DB
        server.UPLOADS = Path(self.tmp.name) / 'uploads'
        server.UPLOADS.mkdir()
        core.init_db().close()
        self.client = TestClient(server.app)

    def tearDown(self):
        core.DB, server.DB, server.UPLOADS = self.old_db, self.old_server_db, self.old_uploads
        self.tmp.cleanup()

    def upload(self, obj):
        return self.client.post('/api/datasets', files={'file': ('cases.json', json.dumps(obj), 'application/json')})

    def test_invalid_structures_are_400_and_not_persisted(self):
        cases = [([], '顶层'), (None, '顶层'), ({'cases': {}}, 'cases'),
                 ({'cases': []}, 'cases'), ({'cases': [1]}, '对象'),
                 ({'cases': [{'id': 'x'}]}, 'input'),
                 ({'cases': [{'id': 'x', 'input': None}]}, 'input'),
                 ({'cases': [{'id': 'x', 'input': {}}, {'id': 'x', 'input': {}}]}, '重复'),
                 ({'cases': [{'id': [], 'input': {}}]}, 'id')]
        for doc, message in cases:
            with self.subTest(doc=doc):
                res = self.upload(doc)
                self.assertEqual(res.status_code, 400, res.text)
                self.assertIn(message, res.json()['detail'])
        self.assertEqual(list(server.UPLOADS.iterdir()), [])
        self.assertEqual(self.client.get('/api/datasets').json(), [])

    def test_valid_and_missing_gold_warning(self):
        res = self.upload({'cases': [{'id': 'a', 'input': {}, 'gold': {'answer': 'yes'}}, {'id': 'b', 'input': 'text'}]})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()['n_cases'], 2)
        self.assertEqual(res.json()['n_problems'], 1)

    def test_csv_validation(self):
        for raw in ['id,gold\nx,{}\n', 'id,input,gold\nx,,{}\n', 'id,input,gold\nx,{},{}\nx,{},{}\n']:
            with self.subTest(raw=raw):
                res = self.client.post('/api/datasets', files={'file': ('cases.csv', raw, 'text/csv')})
                self.assertEqual(res.status_code, 400, res.text)

    def test_neutral_default_model(self):
        self.assertEqual(models.DEFAULT_CONFIG['roles']['judge']['model'], 'mock-judge')
        self.assertEqual(models.config_raw()['roles']['judge']['model'], 'mock-judge')
        with self.assertRaisesRegex(RuntimeError, '模型和 API 地址'):
            token = core.set_request_context('test-key')
            try:
                core.call_llm('test')
            finally:
                core.reset_request_context(token)


if __name__ == '__main__':
    unittest.main()
