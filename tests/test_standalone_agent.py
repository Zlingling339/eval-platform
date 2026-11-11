import json
import time

from fastapi.testclient import TestClient

import server
from src import core
import src.graders_rule  # noqa: F401  注册规则评分器


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'DB', tmp_path / 'db')
    monkeypatch.setattr(server, 'DB', core.DB)
    monkeypatch.setattr(server, 'UPLOADS', tmp_path)
    root = tmp_path / 'root'
    (root / 'reports').mkdir(parents=True)
    monkeypatch.setattr(server, 'ROOT', root)
    return TestClient(server.app)


def test_standalone_agent_requires_dataset_grader_and_target(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.post('/api/runs', json={'scope': 'standalone_agent'})
    assert r.status_code == 400 and '数据集' in r.text
    r = client.post('/api/runs', json={'scope': 'standalone_agent', 'dataset_id': 'x'})
    assert r.status_code == 400 and '评分器' in r.text
    # 既不填接口也不勾仿真：Agent 拿不到轨迹，必须当场拒绝，不能跑完一片「待复核」
    r = client.post('/api/runs', json={'scope': 'standalone_agent', 'dataset_id': 'x',
                                       'grader': 'tool_call_rule'})
    assert r.status_code == 400 and '接口地址' in r.text


def test_standalone_agent_mock_end_to_end(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    doc = {'cases': [{
        'id': 'agent-1',
        'input': {'question': '帮我查一下杭州明天会不会下雨'},
        'gold': {'answer': '调用天气工具并回答'},
        'expect': {'allowed_tools': ['get_weather'], 'required_tools': ['get_weather'],
                   'required_args': {'get_weather': ['city', 'date']},
                   'max_tool_calls': 2, 'key_path': ['get_weather']},
        'mock_reply': '明天小雨，建议带伞。',
        'mock_trace': {'steps': [
            {'type': 'tool_call', 'name': 'get_weather', 'args': {'city': '杭州', 'date': '明天'}, 'result': '小雨'},
            {'type': 'final', 'content': '明天小雨，建议带伞。'},
        ]},
    }]}
    r = client.post('/api/datasets', files={'file': ('agent.json', json.dumps(doc), 'application/json')})
    assert r.status_code == 200, r.text
    did = r.json()['id']
    client.post(f'/api/datasets/{did}/review', json={'reviewed': True, 'note': '看过'})

    res = client.post('/api/runs', json={'scope': 'standalone_agent', 'dataset_id': did,
                                        'grader': 'tool_call_rule+trajectory_rule',
                                        'mock': True, 'force': True})
    assert res.status_code == 200, res.text
    job = {}
    for _ in range(500):
        job = client.get('/api/jobs/' + res.json()['job_id']).json()
        if job['status'] != 'running':
            break
        time.sleep(.01)
    assert job['status'] == 'done', job

    detail = client.get('/api/runs/' + job['run_id']).json()
    assert detail['results'][0]['verdict'] == 'pass'
    assert detail['meta']['product'] == 'standalone_agent'
    # 不关联产品也要能出报告、记版本
    assert client.get('/api/reports/' + job['run_id'] + '.html').status_code == 200
    versions = client.get('/api/versions', params={'product': 'standalone_agent'}).json()
    assert versions and versions[0]['run_id'] == job['run_id']


def test_trajectory_ignores_steps_without_name():
    """final / message 这类收尾步骤没有工具名，不能被算成「多余步骤」。"""
    case = {'expect': {'key_path': ['lookup', 'update']},
            '_trace': {'steps': [
                {'type': 'tool_call', 'name': 'lookup', 'args': {}},
                {'type': 'tool_call', 'name': 'update', 'args': {}},
                {'type': 'final', 'content': 'done'},
            ]}}
    stage = core.StageConfig(id='s', name='s', task_type='agent', dataset='', grader='trajectory_rule')
    out = src.graders_rule.trajectory_rule(case, 'done', stage)
    assert out['verdict'] == 'pass'
    assert out['metrics']['extra_steps'] == 0
