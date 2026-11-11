"""评分器目录与注册表必须一一对齐。

这里的断言不是「列表长什么样」，而是三条硬约束：
1. 注册表里有几个，目录就得有几个 —— 少一个，用户就选不到一个能跑的评分器；
2. 目录里的每个名字都能真的调用（在 REGISTRY 里）；
3. 旧业务评分器不再对外提供（它们绑死在某个产品上，不是通用能力）。
"""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import server
from src import core


class GraderCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.old_db, cls.old_server_db = core.DB, server.DB
        core.DB = Path(cls.tmp.name) / "catalog.sqlite"
        server.DB = core.DB
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        core.DB, server.DB = cls.old_db, cls.old_server_db
        cls.tmp.cleanup()

    def catalog(self):
        r = self.client.get("/api/graders")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_catalog_matches_registry(self):
        items = self.catalog()
        self.assertEqual(len(items), len(core.REGISTRY),
                         f"目录 {len(items)} 项 vs 注册表 {len(core.REGISTRY)} 项：{sorted(core.REGISTRY)}")
        keys = [x["key"] for x in items]
        self.assertEqual(sorted(keys), sorted(core.REGISTRY))
        # 注册名 → real 必须等于自身，组合/校验都靠这个名字对齐
        for x in items:
            self.assertEqual(x["key"], x["real"], x)
            self.assertIn(x["real"], core.REGISTRY)

    def test_generic_graders_present_and_legacy_gone(self):
        items = self.catalog()
        keys = {x["key"] for x in items}
        for name in ("keyword_rule", "format_rule", "citation_rule", "efficiency_rule",
                     "outcome_rule", "pair_compare", "retrieval_rule", "llm_rubric",
                     "rag_judge", "tool_call_rule", "tool_call_judge",
                     "trajectory_rule", "trajectory_judge", "label_match"):
            self.assertIn(name, keys, f"通用评分器 {name} 没出现在目录里")
        for legacy in ("advise_quality", "trigger_matrix", "extract_quality",
                       "chat_quality", "layout_validity"):
            self.assertNotIn(legacy, keys, f"旧业务评分器 {legacy} 不该再对外提供")
        self.assertNotIn("advise_quality", core.REGISTRY,
                         "旧业务评分器不该在注册表里，否则目录会兜底把它列出来")

    def test_every_entry_has_label_and_declared_metrics(self):
        for x in self.catalog():
            self.assertTrue(x.get("label"), x)
            self.assertTrue(x.get("desc"), x)
            for m in x.get("metrics") or []:
                self.assertEqual(len(m), 3, x)
                self.assertIn(m[2], (">=", "<="), x)
            self.assertIsInstance(x.get("counters"), list)
            self.assertIsInstance(x.get("derived"), list)

    def test_label_match_declares_confusion_matrix(self):
        item = next(x for x in self.catalog() if x["key"] == "label_match")
        self.assertEqual(item["counters"], ["tp", "fp", "fn", "tn"])
        self.assertEqual({d["key"] for d in item["derived"]},
                         {"accuracy", "precision", "recall", "f1"})
        formulas = {d["key"]: d["formula"] for d in item["derived"]}
        self.assertEqual(formulas["accuracy"], "(tp+tn)/(tp+fp+fn+tn)")
        self.assertEqual(formulas["precision"], "tp/(tp+fp)")
        self.assertEqual(formulas["recall"], "tp/(tp+fn)")
        self.assertEqual(formulas["f1"], "2*tp/(2*tp+fp+fn)")

    def test_unregistered_in_info_falls_back_to_registry_name(self):
        """注册了但没写中文说明的评分器，也必须能被选到（用注册名兜底）。"""
        name = "zz_tmp_probe_grader"

        @core.register(name)
        def _probe(case, output, stage):
            return {"metrics": {}, "verdict": "pass", "fatal": [], "note": ""}

        try:
            item = next(x for x in self.catalog() if x["key"] == name)
            self.assertEqual(item["label"], name)
            self.assertEqual(item["metrics"], [])
            self.assertEqual(item["counters"], [])
        finally:
            core.REGISTRY.pop(name, None)


if __name__ == "__main__":
    unittest.main()
