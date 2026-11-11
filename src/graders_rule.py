"""确定性评分器：能用代码判断的，绝不用模型。

为什么先写这一层：
    rubric 评分器贵、会抖、要校准。凡是「存在/不存在、对不对、超没超」
   这类二值事实，都必须在这一层解决 —— 否则等于花 API 的钱买一个不确定的答案。
   顺序永远是：确定性 → rubric → 人工。

统一约定：用例的 expect 字段声明这条题要检查什么。
一个 expect 支撑下面全部评分器，接新产品不用改代码。

    "expect": {
      "json": true,                          # 必须输出合法 JSON
      "required": ["amount", "category"],    # 必填字段
      "enum": {"category": ["餐饮", "交通"]}, # 枚举取值
      "max_len": 200,                        # 长度上限
      "must_mention": ["1240"],              # 必须提到的内容（支持 "a|b" 任一说中）
      "forbidden": ["绝对", "保证"],          # 禁止出现
      "citation_required": true,             # 必须给出来源/引用
      "allowed_tools": ["search"],           # 允许调用的工具白名单
      "max_tool_calls": 5,                   # 调用次数上限
      "key_path": ["verify", "query", "refund"],  # 关键步骤顺序
      "top_k": 5, "relevant": ["d1", "d3"],  # 检索层标注
      "max_tokens": 1500, "max_latency_ms": 8000
    }
"""
from __future__ import annotations

import json
import re

from .core import register


# ────────────────────────── 共用小工具 ──────────────────────────

def _json(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}|\[.*\]", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _expect(case) -> dict:
    return case.get("expect") or (case.get("gold") or {}).get("expect") or {}


def _hit(text: str, words) -> list[str]:
    """支持 "具体金额|金额|吃半个月" —— 任一说中即算命中。"""
    low = (text or "").lower()
    out = []
    for w in words or []:
        alts = [a for a in str(w).split("|") if a]
        if any(a.lower() in low for a in alts):
            out.append(w)
    return out


def _trace(case) -> dict | None:
    return case.get("_trace")


def _steps(case) -> list[dict]:
    t = _trace(case) or {}
    steps = t.get("steps") if isinstance(t, dict) else t
    return [s for s in (steps or []) if isinstance(s, dict)]


def _verdict(ok: bool, fatal=None, note="") -> dict:
    return {"metrics": {}, "verdict": "fatal" if fatal else ("pass" if ok else "fail"),
            "fatal": list(fatal or []), "note": note}


# ─────────────────────── 1. 格式 / 结构 ───────────────────────
@register("format_rule")
def format_rule(case, output, stage):
    """输出是不是合法结构：JSON、必填字段、枚举、长度、数值边界。"""
    e = _expect(case)
    m, notes = {}, []

    data = _json(output)
    schema_ok = 1
    if e.get("json") and data is None:
        schema_ok = 0
        notes.append("输出不是合法 JSON")
    m["schema_ok"] = schema_ok

    if isinstance(data, dict):
        req = e.get("required") or []
        missing = [k for k in req if k not in data]
        m["field_ok"] = 1 if not missing else 0
        m["missing"] = len(missing)
        if missing:
            notes.append("缺字段：" + "、".join(missing[:3]))

        enums = e.get("enum") or {}
        bad = [k for k, allowed in enums.items() if k in data and data[k] not in allowed]
        m["enum_ok"] = 1 if not bad else 0
        if bad:
            notes.append("枚举越界：" + "、".join(bad[:3]))
    else:
        m["field_ok"] = 1 if not e.get("required") else 0
        m["enum_ok"] = 1

    if e.get("max_len"):
        over = len(output or "") > e["max_len"]
        m["len_ok"] = 0 if over else 1
        if over:
            notes.append(f"超长 {len(output or '')}>{e['max_len']}")

    ok = all(v == 1 for k, v in m.items() if isinstance(v, int) and k.endswith("_ok"))
    return {"metrics": m, "verdict": "pass" if ok else "fail", "fatal": [],
            "note": "；".join(notes) if notes else "结构合规"}


# ───────────────────────── 2. 关键词命中 ─────────────────────────
@register("keyword_rule")
def keyword_rule(case, output, stage):
    """必须说什么、不能说什么。同一份逻辑原来散在 advise/trigger 里，抽出来通用。"""
    e = _expect(case)
    g = case.get("gold") or {}
    must = e.get("must_mention") or g.get("must_mention") or []
    forbid = e.get("forbidden") or g.get("forbidden") or []

    hit = _hit(output, must)
    bad = _hit(output, forbid)
    m = {
        "mention_ok": 1 if (not must or hit) else 0,
        "forbidden_hits": len(bad),
        "coverage": round(len(hit) / len(must), 3) if must else 1.0,
    }
    note = ""
    if must and not hit:
        note = "未提及必需内容：" + "、".join(must[:3])
    elif bad:
        note = "命中禁用词：" + "、".join(bad[:3])
    return {"metrics": m, "verdict": "fail" if (bad or (must and not hit)) else "pass",
            "fatal": bad[:3], "note": note or "关键词合规"}


# ───────────────────────── 3. 引用溯源 ─────────────────────────
@register("citation_rule")
def citation_rule(case, output, stage):
    """回答里的引用是否真实存在于给定资料中 —— 编造引用比不给引用更糟。"""
    e = _expect(case)
    docs = case.get("docs") or (case.get("input") or {}).get("docs") or []
    corpus = json.dumps(docs, ensure_ascii=False) if not isinstance(docs, str) else docs

    # 抓 [1] / ｜来源：xxx ｜ / "见 xxx 文档" 三类常见引用写法
    refs = re.findall(r"\[(\d{1,2})\]", output or "")
    quoted = re.findall(r"[「《]([^」》]{2,30})[」》]", output or "")

    m = {"has_citation": 1 if (refs or quoted) else 0}
    if e.get("citation_required") and not (refs or quoted):
        return {"metrics": m, "verdict": "fail", "fatal": [],
                "note": "要求给出来源，但没有引用"}

    bad = []
    if corpus:
        for q in quoted:
            if q not in corpus:
                bad.append(q)
        if refs:
            # 编号引用要能在资料列表里找到对应序号
            n_docs = len(docs) if isinstance(docs, list) else 999
            bad += [f"[{r}]" for r in refs if int(r) > n_docs]
    m["phantom_refs"] = len(bad)
    return {"metrics": m, "verdict": "fail" if bad else "pass", "fatal": bad[:3],
            "note": ("引用不存在于资料：" + "、".join(bad[:3])) if bad else "引用可溯源"}


# ───────────────────── 4. 工具调用（确定性部分）─────────────────────
@register("tool_call_rule")
def tool_call_rule(case, output, stage):
    """工具调用的硬事实：调了没、调的在不在白名单、参数键齐不齐、调了几次。

    「选得对不对口 / 参数语义合不合理 / 结果有没有真用上」代码判不了，
    那是 tool_call rubric 的活。两套配合，不重叠。
    """
    e = _expect(case)
    trace = _trace(case)
    if not isinstance(trace, (dict, list)) or (isinstance(trace, dict) and not isinstance(trace.get("steps"), list)):
        return {"metrics": {}, "verdict": "review", "fatal": [], "note": "缺少有效轨迹，无法评测"}
    calls = [s for s in _steps(case) if s.get("type") in ("tool_call", "tool")]
    names = [str(s.get("name") or "") for s in calls]
    allowed = e.get("allowed_tools") or []
    required = e.get("required_tools") or []

    m = {"n_calls": len(calls)}
    notes = []

    if e.get("max_tool_calls") is not None:
        over = len(calls) > e["max_tool_calls"]
        m["calls_ok"] = 0 if over else 1
        if over:
            notes.append(f"调用 {len(calls)} 次 > 上限 {e['max_tool_calls']}")

    if allowed:
        illegal = [n for n in names if n not in allowed]
        m["whitelist_ok"] = 0 if illegal else 1
        if illegal:
            notes.append("调用了未授权工具：" + "、".join(sorted(set(illegal))[:3]))

    if required:
        miss = [t for t in required if t not in names]
        m["required_ok"] = 0 if miss else 1
        if miss:
            notes.append("该调未调：" + "、".join(miss[:3]))

    # 参数必填键
    need = e.get("required_args") or {}
    arg_bad = []
    for s in calls:
        want = need.get(s.get("name"))
        if want:
            got = s.get("args") or {}
            miss = [k for k in want if k not in got]
            if miss:
                arg_bad.append(f"{s.get('name')}缺{','.join(miss)}")
    if need:
        m["args_ok"] = 0 if arg_bad else 1
        if arg_bad:
            notes.append("；".join(arg_bad[:2]))

    ok = all(v == 1 for k, v in m.items() if isinstance(v, int) and k.endswith("_ok"))
    return {"metrics": m, "verdict": "pass" if ok else "fail", "fatal": [],
            "note": "；".join(notes) if notes else f"调用 {len(calls)} 次，合规"}


# ─────────────────── 5. 轨迹对齐（关键路径）───────────────────
@register("trajectory_rule")
def trajectory_rule(case, output, stage):
    """用 LCS 把实际步骤序列和专家标注的关键路径对齐。

    为什么不直接比字符串：真实轨迹常有多余的查询步骤，
    朴素比对会把「多查了一次」判成「路径错了」。LCS 只关心关键步骤有没有按序出现。
    """
    e = _expect(case)
    trace = _trace(case)
    if not isinstance(trace, (dict, list)) or (isinstance(trace, dict) and not isinstance(trace.get("steps"), list)):
        return {"metrics": {}, "verdict": "review", "fatal": [], "note": "缺少有效轨迹，无法评测"}
    key_path = e.get("key_path") or (case.get("gold") or {}).get("key_path") or []
    # final / message 这类收尾步骤没有工具名，不算轨迹步骤；
    # 不过滤的话每条都会多出一个「多余步骤」的空名。
    actual = [str(s.get("name") or s.get("step") or "") for s in _steps(case)]
    actual = [x for x in actual if x]

    if not key_path:
        return {"metrics": {"has_trace": 0}, "verdict": "fail", "fatal": [],
                "note": "用例没有标注关键路径，无法评轨迹"}

    lcs = _lcs(actual, key_path)
    missing = [s for s in key_path if s not in actual]
    extra = [s for s in actual if s not in key_path]
    m = {
        "has_trace": 1 if actual else 0,
        "path_recall": round(lcs / len(key_path), 3),
        "missing_steps": len(missing),
        "extra_steps": len(extra),
        "path_ok": 1 if lcs == len(key_path) else 0,
    }
    note = "关键步骤齐全" if not missing else "缺步骤：" + "、".join(missing[:3])
    if extra:
        note += f"（多余 {len(extra)} 步）"
    return {"metrics": m, "verdict": "pass" if m["path_ok"] else "fail", "fatal": [],
            "note": note}


def _lcs(a: list[str], b: list[str]) -> int:
    """最长公共子序列长度 —— 允许中间有多余步骤，只要关键步骤按序出现。"""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


# ───────────────────────── 6. 效率成本 ─────────────────────────
@register("efficiency_rule")
def efficiency_rule(case, output, stage):
    """token、调用次数、延迟 —— 只看有没有越线，不做跨产品横比。

    不同产品指标同名不同义，跨产品比延迟没有意义；这里只判断「这条题是否超了它自己的预算」。
    """
    e = _expect(case)
    info = case.get("_meta") or {}
    tokens = info.get("tokens") or 0
    latency = info.get("latency_ms") or 0
    steps = len(_steps(case))

    m, notes = {}, []
    if e.get("max_tokens"):
        ok = 1 if tokens <= e["max_tokens"] else 0
        m["tokens_ok"] = ok
        if not ok:
            notes.append(f"token {tokens} > {e['max_tokens']}")
    else:
        m["tokens_ok"] = 1

    if e.get("max_latency_ms"):
        ok = 1 if latency <= e["max_latency_ms"] else 0
        m["latency_ok"] = ok
        if not ok:
            notes.append(f"延迟 {latency}ms > {e['max_latency_ms']}ms")
    else:
        m["latency_ok"] = 1

    if e.get("max_steps"):
        ok = 1 if steps <= e["max_steps"] else 0
        m["steps_ok"] = ok
        if not ok:
            notes.append(f"步骤 {steps} > {e['max_steps']}")
    else:
        m["steps_ok"] = 1

    m["tokens"] = tokens
    m["latency_ms"] = latency
    ok = all(v == 1 for k, v in m.items() if k.endswith("_ok"))
    return {"metrics": m, "verdict": "pass" if ok else "fail", "fatal": [],
            "note": "；".join(notes) if notes else "在预算内"}


# ───────────────────────── 7. 检索层 ─────────────────────────
@register("retrieval_rule")
def retrieval_rule(case, output, stage):
    """检索层指标：Hit@k / Recall@k / MRR / Precision@k。

    检索层和生成层必须分开评 —— 拿检索结果当标准答案去评生成，
    会把「没检索到」和「检索到了但没用」混成同一个低分。
    """
    e = _expect(case)
    relevant = set(e.get("relevant") or (case.get("gold") or {}).get("relevant") or [])
    k = int(e.get("top_k") or 5)

    data = _json(output)
    got = []
    if isinstance(data, list):
        got = [str(x.get("id") if isinstance(x, dict) else x) for x in data]
    elif isinstance(data, dict):
        got = [str(x.get("id") if isinstance(x, dict) else x)
               for x in (data.get("results") or data.get("docs") or [])]
    got = got[:k]

    if not relevant:
        return {"metrics": {"has_label": 0}, "verdict": "fail", "fatal": [],
                "note": "用例没有标注相关文档，无法算检索指标"}

    hit = 1 if (set(got) & relevant) else 0
    recall = len(set(got) & relevant) / len(relevant)
    precision = len(set(got) & relevant) / len(got) if got else 0.0
    mrr = 0.0
    for i, d in enumerate(got, 1):
        if d in relevant:
            mrr = 1.0 / i
            break

    m = {"hit_at_k": hit, "recall_at_k": round(recall, 3),
         "precision_at_k": round(precision, 3), "mrr": round(mrr, 3), "k": k}
    gate_recall = float(e.get("min_recall", 0.6))
    ok = hit == 1 and recall >= gate_recall
    return {"metrics": m, "verdict": "pass" if ok else "fail", "fatal": [],
            "note": f"命中 {len(set(got) & relevant)}/{len(relevant)}，MRR={m['mrr']}"}


# ───────────────────────── 8. 终态检查 ─────────────────────────
@register("outcome_rule")
def outcome_rule(case, output, stage):
    """任务结束后的真实状态对不对（不是「说得好不好」）。

    Agent 类产品的关键失败常常藏在「回复说成功了但状态没变」里，
    所以这里必须读结构化状态，不能读自然语言。
    """
    e = _expect(case)
    want = e.get("expect_state") or (case.get("gold") or {}).get("expect_state")
    if not want:
        return {"metrics": {"has_expect": 0}, "verdict": "fail", "fatal": [],
                "note": "用例没有声明期望终态"}

    data = _json(output)
    got = {}
    if isinstance(data, dict):
        got = data.get("state") or data.get("result") or data
    if not isinstance(got, dict):
        return {"metrics": {"state_ok": 0}, "verdict": "fail", "fatal": [],
                "note": "输出里没有可读的结构化终态"}

    matched, total = 0, 0
    diff = []
    for k, v in want.items():
        total += 1
        if k in got and str(got[k]).strip() == str(v).strip():
            matched += 1
        else:
            diff.append(f"{k}={got.get(k)!r}≠{v!r}")

    m = {"state_ok": 1 if matched == total else 0,
         "state_match": round(matched / total, 3) if total else 0.0}
    return {"metrics": m, "verdict": "pass" if matched == total else "fail",
            "fatal": [], "note": "终态不一致：" + "、".join(diff[:3]) if diff else "终态符合预期"}


# ─────────────────── 9. 二分类判定（accuracy / P / R / F1）───────────────────
# 每条的产物不是分数，而是「归到混淆矩阵的哪一格」：一条用例只能贡献 tp/fp/fn/tn 中的一个 1。
# 集合级的 accuracy / precision / recall / F1 由 core.evaluate 的 derived 公式从
# 求和后的四个计数算出 —— 所以这四个 key 必须写进环节的 counters，
# 否则会被当成普通指标做平均，precision 会死死卡在 0.5（这是最会骗人的错误数字）。
_POSITIVE_WORDS = {"positive", "pos", "true", "yes", "y", "1", "正", "正例", "阳性", "是"}
_NEGATIVE_WORDS = {"negative", "neg", "false", "no", "n", "0", "负", "负例", "阴性", "否"}


def _as_label(value):
    """把标注/预测归一成 positive / negative；认不出来就返回 None（不猜）。"""
    if isinstance(value, bool):
        return "positive" if value else "negative"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "positive" if value else "negative"
    s = str(value if value is not None else "").strip().lower()
    if s in _POSITIVE_WORDS:
        return "positive"
    if s in _NEGATIVE_WORDS:
        return "negative"
    return None


def _path_get(data, path: str):
    """按 "result.label" 这种点号路径取值；取不到返回 None。"""
    cur = data
    for seg in [p for p in str(path or "").split(".") if p]:
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            return None
    return cur


@register("label_match")
def label_match(case, output, stage):
    """把一条输出判成「正例 / 负例」，只记混淆矩阵的一格。

    gold 里必须有标注：gold.label 为 "positive"/"negative"（或 true/false）。
    预测标签的来源按顺序取：
      1. expect.predict_from 指定的字段路径（默认 "label"），从输出的 JSON 里取；
      2. 输出 JSON 顶层的 label 字段；
      3. expect.positive_keywords 命中（负面关键词 negative_keywords 同理）。
    任何一步都拿不到 → review。没有标注就判分等于凭空造一个准确率。
    """
    e = _expect(case)
    gold = case.get("gold") or {}

    want = None
    for key in ("label", "is_positive", "positive"):
        if key in gold:
            want = _as_label(gold[key])
            if want:
                break
    if want is None:
        return {"metrics": {}, "verdict": "review", "fatal": [],
                "note": "gold 里没有可识别的分类标注（label 应为 positive/negative 或 true/false）"}

    data = _json(output)
    path = e.get("predict_from") or "label"
    pred, basis = None, ""

    got = _path_get(data, path) if isinstance(data, dict) else None
    if got is None and isinstance(data, dict):
        got = data.get("label")
        if got is not None:
            path = "label"
    pred = _as_label(got)
    if pred:
        basis = f"取输出字段 {path}={got!r}"

    if pred is None:
        pos_kw = e.get("positive_keywords") or []
        neg_kw = e.get("negative_keywords") or []
        hit_pos = _hit(output, pos_kw)
        hit_neg = _hit(output, neg_kw)
        if hit_pos and not hit_neg:
            pred, basis = "positive", "命中正面关键词 " + "、".join(hit_pos[:3])
        elif hit_neg and not hit_pos:
            pred, basis = "negative", "命中负面关键词 " + "、".join(hit_neg[:3])
        elif hit_pos and hit_neg:
            return {"metrics": {}, "verdict": "review", "fatal": [],
                    "note": "正负关键词同时命中，无法判定预测标签"}

    if pred is None and not isinstance(data, dict):
        # 3.5) 纯文本输出：模型只回一个标签（"positive"/"negative"，这也是最常见的正确行为）
        #      应该能判。只接受「整段/某一行就是一个标签词」，或「答案：xxx」这种带前缀的行，
        #      避免从解释句里抠词造成误判。
        txt = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", (output or "").strip(), flags=re.S).strip()
        cands = [txt] + [ln.strip() for ln in txt.splitlines() if ln.strip()]
        for raw_c in cands:
            m = re.match(r"^(?:答案|标签|分类|结果|label|answer)\s*[:：]\s*(.+)$", raw_c, re.I)
            c = (m.group(1) if m else raw_c).strip()
            c = c.strip('`"\'“”‘’ \n\t').strip("。.!！\n\t ")
            if c and len(c) <= 20 and "\n" not in c:
                lab = _as_label(c)
                if lab:
                    pred, basis = lab, f"纯文本输出 {c!r}"
                    break

    if pred is None:
        return {"metrics": {}, "verdict": "review", "fatal": [],
                "note": f"输出里找不到预测标签（字段路径 {path} 缺失，也没有关键词命中）"}

    cell = {("positive", "positive"): "tp", ("negative", "positive"): "fp",
            ("positive", "negative"): "fn", ("negative", "negative"): "tn"}[(want, pred)]
    ok = cell in ("tp", "tn")
    return {"metrics": {cell: 1}, "verdict": "pass" if ok else "fail", "fatal": [],
            "note": f"{cell}：应为 {want}，判为 {pred}（{basis}）"}
