"""LLM-as-Judge 评分器：只判代码判不了的东西。

三条边界（写新评分器时照这个守）：

1. **标准不来自评委。** 判定标准全在 config/rubrics/*.yaml，模型只负责
   照着判定表挑档。prompt 里禁止出现「你觉得」「综合来看」这类自由裁决措辞。

2. **跨模型执行，同源禁评。** 评委必须和被测产品来自不同厂商；
   同一个模型评自己会宽容，这个偏差不是靠 prompt 能校正的。

3. **拿不到评委就返回 review，不许静默给分。** 评测平台最危险的 bug
   是「没真的评，但报告上有个分数」。所以路由器缺失、Key 缺失、
   judge 输出不是合法 JSON —— 一律 review，交给人工。
"""
from __future__ import annotations

from .core import get_router, register
from . import rubric as rb


def _same_model_guard(router) -> str:
    """启动时校验过一次，这里是第二道闸：防止有人绕过 API 直接改配置。"""
    t, j = router.specs.get("target"), router.specs.get("judge")
    if t and j and t.fingerprint == j.fingerprint:
        return f"评委与被测同为 {j.name}，同源评分不可信"
    return ""


def judge_with_rubric(case, output, stage, rubric_name: str, role: str = "judge",
                      allow_stage_override: bool = False) -> dict:
    """执行一份 rubric，返回统一格式的判分结果。

    只有通用评分器（llm_rubric）才接受环节配置的 rubric 覆盖；
    专用评分器（tool_call_judge 等）永远用自己那份判定表 ——
    否则一个环节挂多个 judge 时，后挂的会静默改用别人的标准，分数没法解释。
    """
    name = (getattr(stage, "rubric", "") or rubric_name) if allow_stage_override else rubric_name
    if name in {"tool_call", "trajectory_process"}:
        trace = case.get("_trace")
        steps = trace.get("steps") if isinstance(trace, dict) else trace
        if not isinstance(steps, list):
            return {"metrics": {}, "verdict": "review", "fatal": [], "note": "缺少有效轨迹，无法评测"}
    router = get_router()
    if router is None:
        return {"metrics": {"rubric_score": 0.0}, "verdict": "review", "fatal": [],
                "note": "未配置模型网关，无法执行 rubric（不假装给分）"}

    guard = _same_model_guard(router)
    if guard:
        return {"metrics": {"rubric_score": 0.0}, "verdict": "fail", "fatal": ["SIBLING_MODEL"],
                "note": guard}

    name = (getattr(stage, "rubric", "") or rubric_name) if allow_stage_override else rubric_name
    try:
        rub = rb.load(name)
    except FileNotFoundError as e:
        return {"metrics": {}, "verdict": "review", "fatal": [], "note": str(e)}

    spec = router.role_of(role)
    if spec.is_mock:
        raw = rub.mock_score(case, output)
        model_used = f"{spec.name}·mock"
    else:
        prompt = rub.render(case, output)
        try:
            raw = router.call(role, prompt, temperature=0.0, timeout=90)["text"]
        except Exception as e:
            return {"metrics": {"rubric_score": 0.0}, "verdict": "review", "fatal": [],
                    "note": f"评委调用失败：{e}"}
        model_used = spec.name

    res = rub.score(raw)
    if not res["ok"]:
        return {"metrics": {"rubric_score": 0.0}, "verdict": "review", "fatal": [],
                "note": f"{model_used} 输出不合规：{res['raw'][:120]}"}

    metrics = {
        "rubric_score": res["score"],
        "gate_fail": 1 if res["gate_failed"] else 0,
        "judge_confidence": res.get("confidence") if isinstance(res.get("confidence"), (int, float)) else 0.0,
    }
    # 每个维度单独出指标，报告里能看出「哪一条拖后腿」而不只是一个总分
    for d in res["dimensions"]:
        if d["score"] is not None:
            metrics[f"dim_{d['key']}"] = d["score"]

    note = res["note"] or ""
    if res["gate_failed"]:
        note = f"否决项未达标：{'、'.join(res['gate_failed'])}"
    ev = next((f"{d['label']}：{d['evidence']}" for d in res["dimensions"]
               if d.get("evidence")), "")
    if ev:
        note = f"{note}｜依据 {ev[:80]}"

    return {"metrics": metrics, "verdict": res["verdict"], "fatal": [],
            "note": f"[{model_used}] {note}"[:300]}


# ───────────────────── 1. 通用 rubric 评分器 ─────────────────────
@register("llm_rubric")
def llm_rubric(case, output, stage):
    """按环节配置的 rubric 打分。一个评分器跑所有 rubric —— 标准是数据，不是代码。"""
    return judge_with_rubric(case, output, stage, "general_quality", allow_stage_override=True)


# ───────────────────── 2. 工具调用质量 ─────────────────────
@register("tool_call_judge")
def tool_call_judge(case, output, stage):
    """工具调用里「合不合适」的部分：选得对不对口、参数语义、结果有没有真用上。"""
    return judge_with_rubric(case, output, stage, "tool_call")


# ───────────────────── 3. 轨迹过程质量 ─────────────────────
@register("trajectory_judge")
def trajectory_judge(case, output, stage):
    """只看过程，不看结果。缺步、冗余、失败处理、声称与过程是否一致。"""
    return judge_with_rubric(case, output, stage, "trajectory_process")


# ───────────────────── 4. RAG 生成层 ─────────────────────
@register("rag_judge")
def rag_judge(case, output, stage):
    """忠实度、相关性、上下文利用、抗噪。检索层归 retrieval_rule，不在这里重复。"""
    return judge_with_rubric(case, output, stage, "rag_faithfulness")


# ───────────────────── 5. 成对比较 ─────────────────────
@register("pair_compare")
def pair_compare(case, output, stage):
    """A/B 谁更好 —— 只判「相对」，不判「绝对分」。

    位置偏差是成对比较最大的坑：同一对答案换个顺序，评委就换边站。
    所以交换顺序各判一次，两次结论一致才算数；不一致直接判 review，
    说明这一对题目本身区分力不够，不该进结论。
    """
    router = get_router()
    if router is None:
        return {"metrics": {"pair_consistent": 0}, "verdict": "review", "fatal": [],
                "note": "未配置模型网关"}

    pair = case.get("pair") or {}
    a = pair.get("a", "")
    b = pair.get("b", "")
    if not a or not b:
        return {"metrics": {"pair_consistent": 0}, "verdict": "review", "fatal": [],
                "note": "用例没有提供 A/B 两个候选（pair.a / pair.b）"}

    spec = router.role_of("judge")
    rubric_txt = (getattr(stage, "pair_rule", "")
                  or "判断哪个回答更好：更符合事实、更贴合用户真实需求、更少冗余。"
                     "只输出 JSON：{\"winner\": \"A\"|\"B\"|\"tie\", \"reason\": \"...\"}")
    keys = ("has_pair", "pair_consistent", "a_wins", "b_wins", "tie")

    def one(x, y):
        if spec.is_mock:
            # 离线：按长度与结构化线索粗略判，保证同一对两次结果可复现
            score_x = len(x) - 3 * x.count("据我所知") - 2 * x.count("一般来说")
            score_y = len(y) - 3 * y.count("据我所知") - 2 * y.count("一般来说")
            w = "A" if score_x >= score_y else "B"
            return {"winner": w}
        prompt = (f"你是一名严格的评测员，只做相对判断，不打绝对分。\n\n"
                  f"【判定规则】{rubric_txt}\n\n【用户输入】{case.get('input')}\n\n"
                  f"【候选 A】{x[:2000]}\n\n【候选 B】{y[:2000]}\n")
        try:
            return rb._extract_json(router.call("judge", prompt)["text"]) or {}
        except Exception:
            return {}

    first = one(a, b)                        # 正常顺序
    second = one(b, a)                       # 交换顺序
    w1 = str(first.get("winner", "")).upper()
    w2 = str(second.get("winner", "")).upper()
    # 交换后仍选「A」意味着它其实选了原来的 B
    flipped = {"A": "B", "B": "A", "TIE": "TIE"}.get(w2, "")

    consistent = bool(w1) and w1 == flipped
    m = {"has_pair": 1, "pair_consistent": 1 if consistent else 0,
         "a_wins": 1 if (consistent and w1 == "A") else 0,
         "b_wins": 1 if (consistent and w1 == "B") else 0,
         "tie": 1 if (consistent and w1 == "TIE") else 0}
    if not consistent:
        return {"metrics": m, "verdict": "review", "fatal": [],
                "note": f"交换顺序后结论翻转（{w1 or '空'} → {w2 or '空'}），位置偏差，判 pair 无效"}
    return {"metrics": m, "verdict": "pass", "fatal": [],
            "note": f"一致判定胜者 {w1}｜{str(first.get('reason', ''))[:80]}"}
