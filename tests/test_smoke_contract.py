"""仅验证冒烟夹具与现有跑批/规则接口，不衡量模型或 Agent 能力。"""
import json
import tempfile
import unittest
from pathlib import Path

from src import core
import src.graders_rule  # 注册确定性规则

ROOT = Path(__file__).resolve().parents[1]


class SmokeContractTest(unittest.TestCase):
    def test_mock_runs_match_fixture_verdicts(self):
        for name, grader in (("model", "keyword_rule"), ("agent", "tool_call_rule+trajectory_rule")):
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as tmp:
                path = ROOT / "examples" / ("smoke-" + name + ".json")
                doc = json.loads(path.read_text(encoding="utf-8"))
                cases = doc["cases"]
                self.assertEqual(len({c["id"] for c in cases}), len(cases))
                self.assertTrue(all(c.get("gold") and c.get("input") for c in cases))
                old_db = core.DB
                core.DB = Path(tmp) / "test.db"
                try:
                    stage = core.StageConfig(id="smoke", name="smoke", task_type="text", dataset=str(path), grader=grader)
                    run_id = core.run_stage({"product": "smoke"}, stage, mock=True,
                                            prompt_fn=lambda c: c["input"]["question"],
                                            mock_reply_fn=lambda c: c["mock_reply"])
                    con = core.init_db()
                    try:
                        results = {r["case_id"]: r for r in core.get_results(con, run_id)}
                    finally:
                        con.close()
                    self.assertEqual(set(results), {c["id"] for c in cases})
                    for case in cases:
                        result = results[case["id"]]
                        self.assertEqual(result["verdict"], case["smoke_expected_verdict"], case["id"])
                        if case["smoke_expected_verdict"] == "review":
                            self.assertIn("轨迹", result["note"])
                finally:
                    core.DB = old_db


if __name__ == "__main__":
    unittest.main()
