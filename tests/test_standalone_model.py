import time
import json
from fastapi.testclient import TestClient
import server
from src import core


def test_standalone_requires_dataset_and_grader(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'DB', tmp_path / 'db')
    monkeypatch.setattr(server, 'DB', core.DB)
    client = TestClient(server.app)
    r = client.post('/api/runs', json={'scope':'standalone_model', 'mock':True})
    assert r.status_code == 400
    assert '数据集' in r.text
    r = client.post('/api/runs', json={'scope':'standalone_model', 'dataset_id':'x', 'mock':True})
    assert r.status_code == 400
    assert '评分器' in r.text


def test_standalone_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'DB', tmp_path / 'db')
    monkeypatch.setattr(server, 'DB', core.DB)
    monkeypatch.setattr(server, 'UPLOADS', tmp_path)
    root = tmp_path / 'root'
    (root / 'reports').mkdir(parents=True)
    monkeypatch.setattr(server, 'ROOT', root)
    client = TestClient(server.app)
    doc = {'cases': [{'id':'a', 'input': {'question':'hello world'}, 'gold': {'answer':'yes'}, 'expect': {'must_mention':['yes']}, 'mock_reply':'yes'}]}
    r = client.post('/api/datasets', files={'file':('data.json', json.dumps(doc), 'application/json')})
    assert r.status_code == 200, r.text
    did = r.json()['id']
    def run(**kwargs):
        body = {'scope':'standalone_model', 'dataset_id':did, 'grader':'keyword_rule', 'mock':True, 'force':True}
        body.update(kwargs)
        response = client.post('/api/runs', json=body)
        assert response.status_code == 200, response.text
        for _ in range(500):
            job = client.get('/api/jobs/' + response.json()['job_id']).json()
            if job['status'] != 'running':
                break
            time.sleep(.01)
        assert job['status'] == 'done', job
        return job
    a = run()
    b = run()
    detail = client.get('/api/runs/' + a['run_id']).json()
    assert detail['results'][0]['prompt'] == 'hello world'
    assert detail['results'][0]['output'] == 'yes'
    assert detail['results'][0]['verdict'] == 'pass'
    assert json.loads(detail['meta']['meta'])['snapshot']['stage']['grader'] == 'keyword_rule'
    assert client.get('/api/reports/' + a['run_id'] + '.html').status_code == 200
    assert client.get('/api/version-compare', params={'a':a['version_id'], 'b':b['version_id']}).status_code == 200
    c = run(repeat=2)
    assert client.get('/api/compare', params={'a':a['run_id'], 'b':c['run_id']}).status_code == 400
    assert client.get('/api/version-compare', params={'a':a['version_id'], 'b':c['version_id']}).status_code == 400


def test_render_tpl_accepts_string_input():
    """input 是纯字符串时，模板渲染不能崩（CSV/纯文本题很常见）。

    之前 render_tpl 直接对 input 调 .get()，字符串会抛 AttributeError，
    整个运行变成 job error。
    """
    out = server.render_tpl("题目：{input}\n正文：{text}", {"input": "加微信领优惠券"})
    assert "加微信领优惠券" in out
    assert "{text}" not in out and "{input}" not in out
    # 对象形式的 input 仍然可用，且能按一级字段取值
    out2 = server.render_tpl("Q: {question}\nA:", {"input": {"question": "水沸点？"}})
    assert "Q: 水沸点？" in out2
