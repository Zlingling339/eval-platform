"""数据集体检：用两条通用基线看一套题**本身**是否可信。

  gold    把用例的 gold / expect 回填成「参考答案」再交给本次评分器。必须 100% 通过 ——
          过不了说明评分器配错了，或者这道题根本不可解。
  empty   空输出（不做事）。它的通过率就是**白拿的地板**：空输出也能过的题没有区分力，
          τ-bench 的教训是有 38% 的题什么都不做反而「通过」。

两条基线都不调用模型、不含任何业务话术，只复用本次评测的评分器。
所以结果只取决于「这套题 + 这份评分器配置」，与产品、prompt、模型无关。

基线只输出 metrics / n_pass / fatal，不产出任何面向业务的文案。
"""
from __future__ import annotations

import json
import re
from typing import Callable


# ────────────────────────── 通用工具 ──────────────────────────
def _expect(case) -> dict:
    """与规则评分器同一套约定：expect 优先，其次 gold.expect。"""
    return case.get("expect") or (case.get("gold") or {}).get("expect") or {}


def _must(case) -> list:
    e = _expect(case)
    g = case.get("gold") or {}
    return e.get("must_mention") or g.get("must_mention") or []


def _forbidden(case) -> list:
    e = _expect(case)
    g = case.get("gold") or {}
    return e.get("forbidden") or g.get("forbidden") or []


def _strip(text: str, words) -> str:
    """参考答案里不能出现题目自己禁止的内容，否则 gold 基线会被自身规则判死。"""
    out = text or ""
    for w in words or []:
        for alt in str(w).split("|"):
            if alt:
                out = out.replace(alt, "")
    return out


# gold 里这些 key 不是"要交付的内容"，回填时跳过
_META_KEYS = {"note", "notes", "expect", "id", "stratum", "metadata", "rationale",
              "must_mention", "forbidden"}


def _placeholder(key, expect) -> str:
    allowed = (expect.get("enum") or {}).get(key)
    if isinstance(allowed, list) and allowed:
        return str(allowed[0])
    return ""


def _gold_output(case) -> str:
    """用 gold / expect 回填一个"参考答案"输出。

    回填的是**这道题自己声明的要求**，不是编造内容：
      1. gold 里有 output/reference/answer/text 这类现成文本，直接用；
      2. 否则把 gold 的字段拼成 JSON，并按 expect.required / expect.enum 补齐声明过的字段；
      3. 关键词类题目 gold 只给了 must_mention 时，把必需内容写进输出。
    """
    g = case.get("gold") if isinstance(case.get("gold"), dict) else {}
    e = _expect(case)
    forbid = _forbidden(case)

    for key in ("output", "reference", "answer", "response", "text", "reply"):
        v = g.get(key)
        if isinstance(v, str) and v.strip():
            return _strip(v, forbid)

    fields = g.get("fields") if isinstance(g.get("fields"), dict) else {
        k: v for k, v in g.items() if k not in _META_KEYS}
    data = dict(fields)

    for key in e.get("required") or []:
        data.setdefault(key, _placeholder(key, e))

    dumped = json.dumps(data, ensure_ascii=False)
    must = _must(case)
    if must and not any(str(w).split("|")[0].lower() in dumped.lower() for w in must):
        data["required_content"] = "、".join(str(w).split("|")[0] for w in must)
        dumped = json.dumps(data, ensure_ascii=False)
    return _strip(dumped, forbid)


def _gold_case(case):
    """gold 基线用的用例副本。

    工具 / 轨迹类评分器看的是用例里的轨迹而不是输出，所以要按 expect 声明的
    allowed/required_tools 与 required_args 造一条"合规动作"的轨迹，否则这类题
    永远只能是 review，gold 基线就失去了"题目可解"的含义。
    """
    e = _expect(case)
    tools = e.get("required_tools") or e.get("allowed_tools") or []
    need = e.get("required_args") or {}
    if not tools:
        return case
    steps = [{"type": "tool_call", "name": str(t),
              "args": {k: "" for k in (need.get(t) or [])}} for t in tools]
    c = dict(case)
    c["_trace"] = {"steps": steps}
    return c


def _empty_case(case):
    """empty 基线：不做事，也没有轨迹。"""
    if "_trace" not in case:
        return case
    c = dict(case)
    c.pop("_trace", None)
    return c


def _gold(case):
    return _gold_output(case)


def _empty(case):
    return ""


# grader -> (中文名, 说明)。只有这里列出的评分器才有基线；
# 没列的返回 supported:false，由 reason 说明为什么不做。
BASELINES: dict[str, list[tuple[str, str, Callable]]] = {
    "keyword_rule": [("gold", "参考答案", _gold), ("empty", "空输出", _empty)],
    "label_match": [("gold", "金标标签", _gold), ("empty", "空输出", _empty)],
    "format_rule": [("gold", "按声明字段回填", _gold), ("empty", "空输出", _empty)],
    "citation_rule": [("gold", "金标文本（含引用）", _gold), ("empty", "空输出", _empty)],
    "tool_call_rule": [("gold", "合规动作（按 expect 构造轨迹）", _gold),
                       ("empty", "空输出", _empty)],
}

# 明确不做的评分器：reason 写清楚"为什么"，不假装支持。
_UNSUPPORTED = {
    "llm_rubric": "由模型按 rubric 打分。基线只能控制输入输出、证明不了判定标准本身对不对，"
                  "跑出来的只会是「模型给参考答案打高分」，没有诊断价值。",
    "tool_call_judge": "LLM 判定工具调用是否合理，基线构造的动作序列无法代表真实语义，结论不可信。",
    "trajectory_judge": "LLM 判定轨迹质量，需要真实执行过程，基线无法构造。",
    "rag_judge": "LLM 判定引用与忠实度，需要真实检索结果，基线无法构造。",
    "pair_compare": "两两对比评分，单条基线输出无法构成对比关系。",
}
_DEFAULT_UNSUPPORTED = ("该评分器没有在本平台声明通用的基线构造方式。"
                        "需要先约定「参考答案」长什么样（gold.output / gold.fields / expect），"
                        "否则硬造的基线只会给出假结论。")


def supports(grader: str) -> bool:
    return grader in BASELINES


def reason(grader: str) -> str:
    return _UNSUPPORTED.get(grader, _DEFAULT_UNSUPPORTED)


# ────────────────────────────── 聚合 ──────────────────────────────
def _agg(rows: list[dict], counters: list[str]) -> dict:
    """counters 求和，其余按出现条数平均。"""
    cs = set(counters or [])
    s: dict[str, float] = {}
    n: dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                s[k] = s.get(k, 0.0) + v
                n[k] = n.get(k, 0) + 1
    return {k: (v if k in cs else round(v / max(n[k], 1), 4)) for k, v in s.items()}


def _ratio(a, b):
    return round(a / b, 4) if b else None


def _derived(agg: dict) -> dict:
    """通用派生指标：只从混淆矩阵计数推，不猜业务语义。"""
    d = {}
    tp, fp, fn, tn = (agg.get(k, 0) for k in ("tp", "fp", "fn", "tn"))
    if tp + fp:
        d["precision"] = _ratio(tp, tp + fp)
    if tp + fn:
        d["recall"] = _ratio(tp, tp + fn)
    if tp + fp + fn + tn:
        d["accuracy"] = _ratio(tp + tn, tp + fp + fn + tn)
        p, r = d.get("precision"), d.get("recall")
        if p and r:
            d["f1"] = round(2 * p * r / (p + r), 4)
    return d


def _stage_derived(agg: dict, stage) -> dict:
    """环节配置里声明的集合级指标（derived[].formula），与 core.evaluate 同一套规则。"""
    out = {}
    for d in (getattr(stage, "derived", None) or []):
        if not isinstance(d, dict) or not d.get("key") or not d.get("formula"):
            continue
        e = d["formula"]
        for k in sorted(agg, key=lambda x: -len(x)):
            e = e.replace(k, str(agg[k]))
        for ident in set(re.findall(r"[A-Za-z_]\w*", e)):
            e = e.replace(ident, "0")
        if not all(c in "0123456789.+-*/() " for c in e):
            continue
        try:
            out[d["key"]] = round(float(eval(e, {"__builtins__": {}}, {})), 4)
        except Exception:
            continue
    return out


# ────────────────────────────── 体检 ──────────────────────────────
def check_dataset(cases: list[dict], stage, grader_fn) -> dict:
    """对一套题跑两条基线，返回对照结果。

    grader_fn: (case, output, stage) -> {metrics, verdict, fatal, note}
    """
    if not supports(stage.grader):
        return {"supported": False, "grader": stage.grader,
                "reason": f"`{stage.grader}` 不做基线体检：{reason(stage.grader)}"}

    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or "id" not in case:
            raise ValueError(f"第 {index} 条用例缺少 id")
        if not isinstance(case.get("gold"), dict) and not isinstance(case.get("expect"), dict):
            raise ValueError(f"第 {index} 条用例缺少 gold 或 expect，无法构造参考答案")

    labels = {}
    for d in (getattr(stage, "derived", None) or []):
        if isinstance(d, dict) and d.get("key") and d.get("label"):
            labels[d["key"]] = d["label"]

    out = {"supported": True, "grader": stage.grader, "n_cases": len(cases),
           "labels": labels, "baselines": []}
    for key, label, make in BASELINES[stage.grader]:
        is_gold = key == "gold"
        rows, passed, fatal_ids = [], [], []
        for c in cases:
            graded_case = _gold_case(c) if is_gold else _empty_case(c)
            r = grader_fn(graded_case, make(c), stage)
            rows.append(r)
            if r.get("verdict") == "pass":
                passed.append(c["id"])
            if r.get("verdict") == "fatal" or r.get("fatal"):
                fatal_ids.append(c["id"])
        agg = _agg(rows, getattr(stage, "counters", None) or [])
        agg.update(_derived(agg))
        agg.update(_stage_derived(agg, stage))
        out["baselines"].append({
            "key": key, "label": label,
            "n_pass": len(passed), "pass_rate": _ratio(len(passed), len(cases)),
            "fatal": len(fatal_ids), "fatal_ids": fatal_ids[:20],
            "metrics": agg, "passed_ids": passed,
        })

    gold_bl = next((b for b in out["baselines"] if b["key"] == "gold"), None)
    empty_bl = next((b for b in out["baselines"] if b["key"] == "empty"), None)
    holes = list(empty_bl["passed_ids"]) if empty_bl else []
    empty_rate = empty_bl["pass_rate"] if empty_bl else None
    out["holes"] = holes
    out["verdict"] = {
        "gold_ok": bool(gold_bl and gold_bl["pass_rate"] == 1.0),
        "gold_rate": gold_bl["pass_rate"] if gold_bl else None,
        "empty_pass_rate": empty_rate,
        # 旧前端字段名（null = 什么都不做），保留只为不打断已有页面，含义同 empty_pass_rate
        "null_pass_rate": empty_rate,
        "n_holes": len(holes),
    }
    return out
