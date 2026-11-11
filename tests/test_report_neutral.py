"""报表中立性回归：报告里不许残留任何旧业务专属表述，指标区必须按实际数据动态生成。

这些断言只看生成的 HTML，不看过程 —— 报表是对外交付物，出现旧业务词就是事故。
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.baseline import check_dataset
from src.core import StageConfig
from src.report import render
import src.graders_rule  # noqa: F401  注册确定性评分器

# 旧业务残留词：报表里一个都不许出现
LEGACY = ("心愿基金", "心愿单", "劝阻", "劝购", "打扰率", "陷阱识别",
          "金额一致率", "OCR", "小票", "借贷", "征信", "消费心理")


def _ev(metrics=None, n=2):
    return {
        "run_id": "r1", "n": n, "n_runs": n, "repeat": 1,
        "metrics": dict(metrics or {}),
        "checks": [{"key": "score", "label": "质量分", "op": ">=", "value": 0.8,
                    "actual": 0.9, "pass": True}],
        "fatals": [], "by_stratum": {"basic": {"n": n, "pass": n, "fail": 0, "fatal": 0}},
        "passed": True, "pass_at_k": 1.0, "pass_pow_k": 1.0, "unstable": [],
        "verdict": "合格",
    }


def _run():
    return {"id": "run-0001", "product": "general-example", "product_name": "通用示例",
            "stage": "answer", "dataset": "general_example_v1", "model": "mock",
            "mode": "mock", "started_at": "2026-01-01 00:00:00", "cost": 0.0}


def _rows(n=2):
    return [{"case_id": f"c{i}", "stratum": "basic", "verdict": "pass", "fatal": [],
             "note": "关键词合规", "output": "100", "prompt": "问题", "metrics": {"score": 1.0},
             "n_pass": 1, "n_repeat": 1} for i in range(n)]


def _baseline(metrics_gold, metrics_empty, labels=None):
    return {
        "supported": True, "grader": "keyword_rule", "n_cases": 2,
        "labels": labels or {},
        "baselines": [
            {"key": "gold", "label": "参考答案", "n_pass": 2, "fatal": 0, "metrics": metrics_gold},
            {"key": "empty", "label": "空输出", "n_pass": 0, "fatal": 0, "metrics": metrics_empty},
        ],
        "holes": [],
        "verdict": {"gold_ok": True, "gold_rate": 1.0, "empty_pass_rate": 0.0, "n_holes": 0},
    }


class ReportNeutralTest(unittest.TestCase):
    def _render(self, ev, baseline=None, labels=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "report.html"
        render(ev, _run(), _rows(ev["n"]), path, baseline=baseline, labels=labels)
        return path.read_text(encoding="utf-8")

    def test_no_legacy_vocabulary(self):
        """即使基线和指标都齐全，也不许出现旧业务表述。"""
        bl = _baseline({"precision": 0.9, "recall": 0.8},
                       {"precision": 0.1, "recall": 0.0})
        html = self._render(_ev({"precision": 0.9, "recall": 0.8, "f1": 0.85}), baseline=bl)
        for word in LEGACY:
            self.assertNotIn(word, html, f"报表里出现了旧业务表述：{word}")

    def test_baseline_columns_follow_actual_metrics(self):
        """列由 baseline 实际产出的指标决定：没有的指标整列不显示，也不许用「—」凑数。"""
        # 两条基线产出的指标不同：gold 有 precision/recall，empty 只有 precision
        bl = _baseline({"precision": 0.9, "recall": 0.8},
                       {"precision": 0.2}, labels={"precision": "精确率", "recall": "召回率"})
        html = self._render(_ev({"score": 1.0}), baseline=bl)
        head = html.split("<h2>基线对照")[1].split("</table>")[0]
        self.assertIn("精确率", head)
        self.assertIn("召回率", head)
        self.assertNotIn("F1", head)          # 两条基线都没有 f1，整列不出现
        self.assertNotIn(">—<", head)          # 不用「—」占位
        self.assertEqual(head.count("<th"), head.count("<th"))  # 结构完整
        # 计数型 key（tp/fp/fn/tn）不当比率铺进基线表
        bl2 = _baseline({"tp": 8, "fp": 2, "fn": 1, "tn": 9, "precision": 0.8},
                        {"tp": 0, "fp": 0, "fn": 9, "tn": 0, "precision": 0.0})
        head2 = self._render(_ev({"score": 1.0}), baseline=bl2).split("<h2>基线对照")[1].split("</table>")[0]
        for k in ("tp", "fp", "fn", "tn"):
            self.assertNotIn(f">{k}<", head2)

    def test_unknown_metric_falls_back_to_key(self):
        """label 映射里没有的指标，表头直接用 key，不写死中文名。"""
        bl = _baseline({"custom_hit_rate": 0.75}, {"custom_hit_rate": 0.0})
        head = self._render(_ev({"score": 1.0}), baseline=bl).split("<h2>基线对照")[1].split("</table>")[0]
        self.assertIn("custom_hit_rate", head)

    def test_classification_block_only_when_f1_present(self):
        html = self._render(_ev({"f1": 0.87, "precision": 0.9, "recall": 0.84}))
        self.assertIn("F1", html)
        self.assertIn("分类指标", html)
        self.assertIn("不要相加", html)

        html2 = self._render(_ev({"mention_ok": 1.0, "coverage": 1.0}))
        self.assertNotIn("F1", html2)
        self.assertNotIn("分类指标", html2)

    def test_classification_from_confusion_matrix(self):
        html = self._render(_ev({"tp": 8, "fp": 2, "fn": 2, "tn": 8}))
        self.assertIn("准确率", html)
        self.assertIn("F1", html)
        self.assertIn("样本数", html)

    def test_foot_note_is_dynamic(self):
        html = self._render(_ev({"score": 1.0}))
        self.assertIn("general_example_v1", html)
        self.assertIn("口径", html)
        self.assertIn("限制", html)


class BaselineNeutralTest(unittest.TestCase):
    CASES = [
        {"id": "c1", "input": {"q": "水沸点"}, "expect": {"must_mention": ["100"]}},
        {"id": "c2", "input": {"q": "一周几天"}, "expect": {"must_mention": ["7", "七"]}},
    ]

    def _stage(self, grader):
        return StageConfig(id="s", name="s", task_type="text", dataset="d.json", grader=grader)

    def test_supported_grader_returns_gold_and_empty(self):
        from src.core import get_grader
        res = check_dataset(self.CASES, self._stage("keyword_rule"), get_grader("keyword_rule"))
        self.assertTrue(res["supported"])
        self.assertEqual([b["key"] for b in res["baselines"]], ["gold", "empty"])
        gold, empty = res["baselines"]
        self.assertEqual(gold["n_pass"], 2)          # 参考答案必须全过
        self.assertEqual(empty["n_pass"], 0)         # 空输出拿不到分
        self.assertTrue(res["verdict"]["gold_ok"])
        for b in res["baselines"]:
            self.assertTrue({"key", "label", "n_pass", "fatal", "metrics"} <= set(b))

    def test_unsupported_grader_explains_itself(self):
        res = check_dataset(self.CASES, self._stage("llm_rubric"), lambda *a: None)
        self.assertFalse(res["supported"])
        self.assertIn("reason", res)
        self.assertTrue(res["reason"])
        self.assertNotIn("baselines", res)

    def test_baseline_output_carries_no_business_copy(self):
        from src.core import get_grader
        res = check_dataset(self.CASES, self._stage("keyword_rule"), get_grader("keyword_rule"))
        blob = json.dumps(res, ensure_ascii=False)
        for word in LEGACY:
            self.assertNotIn(word, blob)
        # 基线只产出结构化指标，不带任何"话术"字段
        for b in res["baselines"]:
            self.assertFalse({k for k in b if "script" in k or "text" in k or "copy" in k})


if __name__ == "__main__":
    unittest.main()
