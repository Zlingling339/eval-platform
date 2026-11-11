"""这轮修的几个坑：评测计划落库的 counters、接口地址脱敏、双跑换评分器。

三个断言对应三个真实故障：
1. 计划导入后 counters/derived 丢了 → tp/fp 被当普通指标平均，precision 恒 0.5；
2. run.meta 存整条接口地址 → path 里的密钥跟着进台账、进导出报告；
3. /api/dual 没接 grader → 前端在双跑页换不了评分器（或换了不生效）。
"""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

TARGET_SECRET = "path-secret-8f3a2"
TEMPLATE_SECRET = "tenant-key-4b7c1e"


class EchoHandler(BaseHTTPRequestHandler):
    """最小被测接口：把请求体里的 tenant 原样回显，用来验证它会不会被打码。"""

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        out = {"answer": "已处理 " + str(body.get("tenant", "")), "ok": True}
        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


class BackendFixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global server, core, models
        import server
        from src import core, models
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), EchoHandler)
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()
        cls.base = "http://127.0.0.1:%d" % cls.http.server_port

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db, self.old_server_db, self.old_config = core.DB, server.DB, models.CONFIG
        core.DB = Path(self.tmp.name) / "fixes.sqlite"
        server.DB = core.DB
        models.CONFIG = Path(self.tmp.name) / "models.yaml"
        models.save_config(models.DEFAULT_CONFIG)
        self.client = TestClient(server.app)
        server.conn().close()
        self.assertEqual(self.client.post("/api/examples/general").status_code, 200)
        self.reports = []

    def tearDown(self):
        for report in self.reports:
            report.unlink(missing_ok=True)
        core.DB, server.DB, models.CONFIG = self.old_db, self.old_server_db, self.old_config
        self.tmp.cleanup()

    def wait_job(self, jid):
        for _ in range(400):
            job = self.client.get("/api/jobs/" + jid).json()
            if job["status"] != "running":
                return job
            time.sleep(0.01)
        self.fail("后台任务没有在预期时间内结束")

    # ── 1. 计划导入：counters / derived 必须落库 ──
    def test_import_plan_keeps_counters_and_derived(self):
        plan = {"product": "cls-plan", "name": "分类计划", "stages": [{
            "id": "judge", "name": "判定", "grader": "label_match",
            "gates": [{"key": "accuracy", "op": ">=", "value": 0.8, "label": "准确率"}],
            "fatal": ["WRONG_LABEL"],
        }]}
        r = self.client.post("/api/products/import", json={"plan": plan})
        self.assertEqual(r.status_code, 200, r.text)
        stage = core.load_products()["cls-plan"]["stages"][0]
        self.assertEqual(stage["counters"], ["tp", "fp", "fn", "tn"])
        self.assertEqual({d["key"] for d in stage["derived"]},
                         {"accuracy", "precision", "recall", "f1"})
        # 达标门里的 accuracy 不能再被当成「评分器不产出的指标」
        self.assertFalse([w for w in r.json()["warnings"] if "不产出" in w], r.json())

    # ── 2. 接口地址与 body_template：只留 hostname，密钥登记给 redact ──
    def test_run_meta_keeps_only_hostname_and_template_secret_is_redacted(self):
        target = {"url": self.base + "/hook/" + TARGET_SECRET,
                  "response_path": "answer",
                  "headers": {"Authorization": "Bearer " + TARGET_SECRET},
                  "body_template": {"tenant": TEMPLATE_SECRET, "text": "{question}"}}
        r = self.client.post("/api/runs", json={
            "product": "general-example", "stage": "answer", "force": True,
            "target": target})
        self.assertEqual(r.status_code, 200, r.text)
        job = self.wait_job(r.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        self.reports.append(Path(job["report"]))

        detail = self.client.get("/api/runs/" + job["run_id"]).json()
        meta = json.loads(detail["meta"]["meta"])
        self.assertEqual(meta["target_url"], "127.0.0.1")
        blob = json.dumps(detail, ensure_ascii=False) + Path(job["report"]).read_text()
        self.assertNotIn(TARGET_SECRET, blob)
        self.assertNotIn(TEMPLATE_SECRET, blob)

    # ── 3. 双跑支持 grader 参数，非法名字当场 400 ──
    def test_dual_accepts_grader_and_rejects_unknown(self):
        body = {"product": "general-example", "stage": "answer",
                "target": {"url": self.base + "/hook/x", "response_path": "answer"},
                "grader": "not_a_grader", "allow_mock_baseline": True}
        r = self.client.post("/api/dual", json=body)
        self.assertEqual(r.status_code, 400, r.text)

        body["grader"] = "label_match"
        r = self.client.post("/api/dual", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        job = self.wait_job(r.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        for rep in job["reports"]:
            self.reports.append(Path(rep["path"]))
        # 通用示例数据集没有分类标注 → label_match 必须判 review，不能瞎给分
        detail = self.client.get("/api/runs/" + job["run_id"]).json()
        self.assertTrue(detail["results"], detail)
        for row in detail["results"]:
            self.assertEqual(row["verdict"], "review")
            self.assertIn("分类标注", row["note"])


    # ── 4. 组合评分器的达标门必须能命中（曾因指标加前缀而恒不达标）──
    def test_combined_grader_gates_resolve(self):
        path = Path(self.tmp.name) / "agent.json"
        path.write_text(json.dumps({"cases": [{
            "id": "a", "input": {"q": "查一下"},
            "expect": {"allowed_tools": ["search", "verify"], "key_path": ["search", "verify"],
                       "must_mention": ["完成"]},
            "mock_reply": "已完成",
            "mock_trace": {"steps": [{"type": "tool_call", "name": "search"},
                                     {"type": "tool_call", "name": "verify"}]},
        }]}, ensure_ascii=False), encoding="utf-8")
        stage = core.StageConfig(
            id="agent", name="agent", task_type="agent", dataset=str(path),
            grader="tool_call_rule+trajectory_rule",
            gates=[{"key": "path_ok", "op": ">=", "value": 1, "label": "关键路径完整"},
                   {"key": "whitelist_ok", "op": ">=", "value": 1, "label": "工具白名单合规"}],
            fatal=[])
        rid = core.run_stage({"product": "p"}, stage, mock=True,
                             prompt_fn=lambda c: c["input"]["q"],
                             mock_reply_fn=lambda c: c["mock_reply"])
        con = core.init_db()
        try:
            ev = core.evaluate(con, rid, stage)
        finally:
            con.close()
        self.assertFalse([k for k in ev["metrics"] if "." in k], ev["metrics"])
        self.assertTrue(all(c["actual"] == 1.0 for c in ev["checks"]), ev["checks"])
        self.assertTrue(ev["passed"], ev)


    # ── 5. 旧业务评分器的下线提示 ──
    def test_deprecated_grader_says_what_to_use_instead(self):
        plan = {"product": "old-plan", "stages": [{
            "id": "s1", "name": "旧环节", "grader": "advise_quality",
            "gates": [{"key": "precision", "op": ">=", "value": 0.8, "label": "精确率"}]}]}
        r = self.client.post("/api/products/import", json={"plan": plan})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("已下线", r.text)
        self.assertIn("label_match", r.text)

        r = self.client.post("/api/runs", json={"product": "general-example", "stage": "answer",
                                                "force": True, "mock": True,
                                                "grader": "advise_quality"})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("已下线", r.text)


if __name__ == "__main__":
    unittest.main()
