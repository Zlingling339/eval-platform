"""通用冒烟数据的评分器契约和真实跑批回归；不评价模型能力。"""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import server
from src import core

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('filename,grader', [
    ('general-label-smoke.json', 'label_match'),
    ('general-qa-summary-smoke.json', 'keyword_rule'),
    ('general-format-smoke.json', 'format_rule'),
])
def test_dataset_contract_and_mock_batch(filename, grader, tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'DB', tmp_path / 'eval.db')
    monkeypatch.setattr(server, 'DB', core.DB)
    monkeypatch.setattr(server, 'UPLOADS', tmp_path / 'uploads')
    server.UPLOADS.mkdir()
    path = ROOT / 'examples' / filename
    doc = json.loads(path.read_text(encoding='utf-8'))
    cases = doc['cases']
    assert len({c['id'] for c in cases}) == len(cases)
    assert all(c.get('gold') and c['input'].get('question') for c in cases)
    if grader == 'label_match':
        assert all(c['gold']['label'] in ('positive', 'negative') for c in cases)
    elif grader == 'keyword_rule':
        assert all(c['expect'].get('must_mention') for c in cases)
    else:
        assert all(c['expect'].get('json') and c['expect'].get('required') for c in cases)
    client = TestClient(server.app)
    uploaded = client.post('/api/datasets', files={'file': (filename, path.read_bytes(), 'application/json')})
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()['n_cases'] == len(cases)
    stage = core.StageConfig(id='smoke', name='smoke', task_type='text', dataset=str(path), grader=grader)
    run_id = core.run_stage({'product':'smoke'}, stage, mock=True,
                            prompt_fn=lambda c: c['input']['question'],
                            mock_reply_fn=lambda c: c['mock_reply'])
    con = core.init_db()
    try:
        results = {r['case_id']: r for r in core.get_results(con, run_id)}
    finally:
        con.close()
    assert set(results) == {c['id'] for c in cases}
    for case in cases:
        assert results[case['id']]['verdict'] == case['smoke_expected_verdict'], case['id']
