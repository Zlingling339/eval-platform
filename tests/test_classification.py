"""二分类指标：accuracy / precision / recall / F1 必须精确等于手算值。

为什么要专门测这个：
    label_match 每条只产出 tp/fp/fn/tn 中的一个 1。如果这四个 key 没进
    stage.counters，它们会被当普通指标求平均，precision 就会变成一个
    看着很像样、其实毫无意义的固定值。所以这里断言的是「算出来的数」，
    不是「有没有这个字段」。
"""
import json
import tempfile
import unittest
from pathlib import Path

from src import core
import src.graders_rule  # noqa: F401  注册确定性规则评分器
import server  # GRADER_INFO 里的 counters / derived 定义

ROOT = Path(__file__).resolve().parents[1]


def _label_match_meta():
    info = server.GRADER_INFO["label_match"]
    return list(info["counters"]), [dict(d) for d in info["derived"]]


def _case(cid, gold_label, reply):
    return {"id": cid, "input": {"q": "这条该判什么"}, "gold": {"label": gold_label},
            "mock_reply": reply, "expect": {"predict_from": "label"}}


class ClassificationMetricTest(unittest.TestCase):
    def run_cases(self, cases):
        """跑完返回 (evaluate 结果, {case_id: 单条结果})。"""
        counters, derived = _label_match_meta()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
            old_db = core.DB
            core.DB = Path(tmp) / "cls.sqlite"
            try:
                stage = core.StageConfig(
                    id="cls", name="分类", task_type="text", dataset=str(path),
                    grader="label_match", counters=counters, derived=derived,
                    gates=[{"key": "accuracy", "op": ">=", "value": 0.4, "label": "准确率"}],
                )
                rid = core.run_stage({"product": "cls"}, stage, mock=True,
                                     prompt_fn=lambda c: c["input"]["q"],
                                     mock_reply_fn=lambda c: c["mock_reply"])
                con = core.init_db()
                try:
                    ev = core.evaluate(con, rid, stage)
                    rows = {r["case_id"]: r for r in core.get_results(con, rid)}
                finally:
                    con.close()
                return ev, rows
            finally:
                core.DB = old_db

    def test_four_cells_one_case_each(self):
        """tp / fp / fn / tn 各一条 → 准确率 2/4、精确率 1/2、召回率 1/2、F1 2/4。"""
        cases = [
            _case("tp", "positive", json.dumps({"label": "positive"})),
            _case("fp", "negative", json.dumps({"label": "positive"})),
            _case("fn", "positive", json.dumps({"label": "negative"})),
            _case("tn", "negative", json.dumps({"label": "negative"})),
        ]
        ev, rows = self.run_cases(cases)

        # 单条：只贡献混淆矩阵的一格，不做任何加权
        self.assertEqual(rows["tp"]["metrics"], {"tp": 1})
        self.assertEqual(rows["fp"]["metrics"], {"fp": 1})
        self.assertEqual(rows["fn"]["metrics"], {"fn": 1})
        self.assertEqual(rows["tn"]["metrics"], {"tn": 1})
        # 单条判定：判对 pass，判错 fail
        self.assertEqual(rows["tp"]["verdict"], "pass")
        self.assertEqual(rows["tn"]["verdict"], "pass")
        self.assertEqual(rows["fp"]["verdict"], "fail")
        self.assertEqual(rows["fn"]["verdict"], "fail")

        m = ev["metrics"]
        self.assertEqual(m["tp"], 1.0)
        self.assertEqual(m["fp"], 1.0)
        self.assertEqual(m["fn"], 1.0)
        self.assertEqual(m["tn"], 1.0)
        self.assertEqual(m["accuracy"], 0.5)   # (1+1)/4
        self.assertEqual(m["precision"], 0.5)  # 1/(1+1)
        self.assertEqual(m["recall"], 0.5)     # 1/(1+1)
        self.assertEqual(m["f1"], 0.5)         # 2*1/(2*1+1+1)
        # 达标门按 key 命中，说明汇总里的 key 没有被加前缀
        self.assertEqual(ev["checks"][0]["actual"], 0.5)
        self.assertTrue(ev["checks"][0]["pass"])
        self.assertNotIn("label_match.accuracy", m)

    def test_asymmetric_counts_prove_counters_are_summed(self):
        """tp=3 fp=1 fn=1 tn=1：准确率 4/6、精确率 3/4、召回率 3/4、F1 6/8。

        这组数字刻意取不整齐：如果 tp/fp/fn/tn 被当普通指标做平均，
        算出来会全是 1.0 或 0.5，与手算值差得很远。
        """
        cases = [
            _case("tp1", True, json.dumps({"label": "positive"})),
            _case("tp2", True, json.dumps({"result": {"label": "positive"}})),
            _case("fp1", False, json.dumps({"label": "positive"})),
            _case("tp3", "positive", json.dumps({"label": "positive"})),
            _case("fn1", "positive", json.dumps({"label": "negative"})),
            _case("tn1", "negative", json.dumps({"label": "negative"})),
        ]
        cases[1]["expect"] = {"predict_from": "result.label"}
        ev, rows = self.run_cases(cases)

        m = ev["metrics"]
        self.assertEqual(m["tp"], 3.0)
        self.assertEqual(m["fp"], 1.0)
        self.assertEqual(m["fn"], 1.0)
        self.assertEqual(m["tn"], 1.0)
        self.assertEqual(m["accuracy"], round(4 / 6, 4))
        self.assertEqual(m["precision"], 0.75)
        self.assertEqual(m["recall"], 0.75)
        self.assertEqual(m["f1"], 0.75)
        # 4 条判对 2 条判错
        self.assertEqual(sum(1 for r in rows.values() if r["verdict"] == "pass"), 4)

    def test_keyword_fallback_and_missing_annotation(self):
        """没有 label 字段时用关键词判；标注缺失时一律 review，不瞎判。"""
        kw_case = {"id": "kw", "input": {"q": "买不买"}, "gold": {"label": "positive"},
                   "expect": {"positive_keywords": ["建议购买"]},
                   "mock_reply": "综合考虑，建议购买"}
        no_gold = {"id": "nogo", "input": {"q": "买不买"}, "gold": {},
                   "expect": {"predict_from": "label"},
                   "mock_reply": json.dumps({"label": "positive"})}
        ev, rows = self.run_cases([kw_case, no_gold])

        self.assertEqual(rows["kw"]["metrics"], {"tp": 1})
        self.assertEqual(rows["kw"]["verdict"], "pass")
        self.assertIn("命中正面关键词", rows["kw"]["note"])
        self.assertEqual(rows["nogo"]["verdict"], "review")
        self.assertEqual(rows["nogo"]["metrics"], {})

    def test_plain_text_label_is_recognized(self):
        """模型只回一个纯文本标签（真实 API 最常见）也要能判，不能一律 review。

        背景：之前 label_match 只认 JSON 的 label 字段或配置的关键词，
        模型老老实实回 "negative" 反而被判「找不到预测标签」。
        """
        cases = [
            {"id": "plain", "input": {"q": "x"}, "gold": {"label": "negative"},
             "mock_reply": "negative"},
            {"id": "prefix", "input": {"q": "x"}, "gold": {"label": "positive"},
             "mock_reply": "答案：positive\n理由：出现引导私信。"},
            {"id": "fenced", "input": {"q": "x"}, "gold": {"label": "positive"},
             "mock_reply": "```\npositive\n```"},
        ]
        _, rows = self.run_cases(cases)
        self.assertEqual(rows["plain"]["verdict"], "pass")
        self.assertEqual(rows["plain"]["metrics"], {"tn": 1})
        self.assertEqual(rows["prefix"]["verdict"], "pass")
        self.assertEqual(rows["prefix"]["metrics"], {"tp": 1})
        self.assertEqual(rows["fenced"]["verdict"], "pass")

    def test_no_prediction_returns_review(self):
        case = {"id": "empty", "input": {"q": "x"}, "gold": {"label": "positive"},
                "expect": {"predict_from": "label"}, "mock_reply": "无法判断"}
        _, rows = self.run_cases([case])
        self.assertEqual(rows["empty"]["verdict"], "review")
        self.assertIn("找不到预测标签", rows["empty"]["note"])

    def test_combined_grader_keeps_metric_keys_unprefixed(self):
        """组合评分器（a+b）的指标 key 不能被加前缀，否则 gates 永远匹配不到。"""
        case = {"id": "c", "input": {"q": "x"}, "gold": {"label": "positive"},
                "expect": {"must_mention": ["购买"], "predict_from": "label"},
                "mock_reply": json.dumps({"label": "positive", "text": "建议购买"})}
        stage = core.StageConfig(id="s", name="s", task_type="text", dataset="", grader="a+b")
        out = case["mock_reply"]
        # 组合：分类判定 + 关键词合规，两个评分器各出一个指标
        g = core.apply_graders("label_match+keyword_rule", case, out, stage)
        self.assertIn("mention_ok", g["metrics"])
        self.assertIn("tp", g["metrics"])
        self.assertFalse([k for k in g["metrics"] if "." in k], g["metrics"])

    def test_derived_metric_formulas_are_reusable(self):
        """GRADER_INFO 里的 derived 公式和 core 的算法一致（同一套 key 名）。"""
        _, derived = _label_match_meta()
        self.assertEqual(len(derived), 4)
        for d in derived:
            self.assertEqual(set(d), {"key", "label", "formula"})


if __name__ == "__main__":
    unittest.main()
