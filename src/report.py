"""HTML 报表渲染 v2。

v1 自检后修的问题（都来自实际截图评审）：
1. 顶部留白 80-100px → 收紧到 24px
2. 达标/未达标用红绿二元 → 改为「墨灰点 = 达标，赭红点 = 未达标」，去掉系统绿，
   整体变成「纸白 + 墨黑 + 单一赭红」，摆脱中后台模板感
3. 运算符用 >= 显示 → 改为 ≥ / ≤
4. 数字精度不一致 → 统一 1 位小数
5. 一票否决区左缩进与表格不对齐 → 对齐到同一条基线
6. 结论区右侧大片空白 → 改为两栏，右侧放三个关键数字（最大缺口/否决数/最差分层的）
7. 逐条编号宽度不一 → 固定宽度
8. 下一步的小方块与状态点混淆 → 改成长横线

通用化改造（不再绑定任何具体业务）：
- 基线表列由 baseline["baselines"][i]["metrics"] 里实际存在的 key 动态生成，
  中文名走传入的 label 映射（checks[].label / baseline["labels"] / 内置兜底），取不到就显示 key；
- 二分类统计单独成块，只在本次 metrics 含 tp/fp/fn/tn 或 accuracy/precision/recall/f1 时渲染；
- 口径/限制按本次数据集、评分器、样本量生成，不写死业务名。
"""
from __future__ import annotations
import html
import json
from pathlib import Path

CSS = """
:root{
  /* 手账风：纸 + 暖墨 + 批改红（与平台界面同一套 token） */
  --ink:#2a2420; --ink2:#4a423b; --ink3:#6b6660;
  --paper:#f8f5ee; --sunk:#f2ecdc; --line:#d9d2c2; --line2:#c2b7a1;
  --accent:#c2412d; --fail:#c2412d; --fatal:#8a2118; --pass:#3b7a4b;
  --desk:#a8c8bb; --pencil:#8e887f; --rule:rgba(143,176,212,.28);
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  --kai:"Kaiti SC","STKaiti","KaiTi","楷体","LXGW WenKai","AR PL UKai CN","Noto Serif CJK SC","Songti SC",serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{color:var(--ink);font:13px/1.62 var(--sans);-webkit-font-smoothing:antialiased;
  background-color:var(--desk);
  background-image:repeating-linear-gradient(58deg,rgba(255,255,255,.055) 0 2px,transparent 2px 5px),
    repeating-linear-gradient(-32deg,rgba(0,0,0,.022) 0 1px,transparent 1px 6px);
  padding:22px 0 40px}
.wrap{max-width:1060px;margin:0 auto;padding:30px 34px 56px;background-color:var(--paper);
  background-image:repeating-linear-gradient(to bottom,transparent 0 27px,var(--rule) 27px 28px);
  background-position:0 10px;
  box-shadow:0 1px 0 rgba(255,255,255,.85) inset,0 20px 44px -20px rgba(42,36,30,.5),
    2px 2px 0 #dcc8a3,-2px 0 0 #efe9dc}
.revbar{position:sticky;top:0;z-index:5;background:var(--sunk);border:1px solid var(--line2);padding:8px 12px;font-size:12.5px;margin-bottom:12px}
.revrow{font-size:11.5px;color:var(--ink3);display:flex;align-items:center;gap:8px;margin:8px 0 2px}
.revrow button{border:1px solid var(--ink2);background:#fff;padding:1px 9px;cursor:pointer;font-family:var(--kai);font-size:12px}
.revnow{color:var(--accent)}
.tiny{font-family:var(--kai);font-size:12px;color:var(--ink3);margin:10px 0 3px}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.eyebrow{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink3);font-weight:600}

/* ── 顶栏 ── */
.top{display:flex;justify-content:space-between;align-items:flex-end;
     border-bottom:2px solid var(--ink);padding-bottom:12px}
h1{font-family:var(--kai);font-size:26px;font-weight:700;letter-spacing:.02em;margin-top:4px}
.h1tag{font:400 12px/1 var(--mono);color:var(--ink3);margin-left:9px;letter-spacing:0}
td.mkey{width:46%}
.meta{font-size:10.5px;color:var(--ink3);text-align:right;line-height:1.9}

/* ── 结论区（两栏，右侧关键数字）── */
.verdict{margin-top:22px;padding:13px 0 20px;border-top:1.5px solid var(--ink);
         border-bottom:1px solid var(--line2);
         display:grid;grid-template-columns:1fr auto;gap:48px;align-items:start}
.vline{display:flex;align-items:center;gap:14px}
.vtag{font-family:var(--kai);font-size:12px;font-weight:700;letter-spacing:.14em;
      padding:4px 9px;border:2.2px solid currentColor;border-radius:4px;
      transform:rotate(-6deg);display:inline-block}
.ok{color:var(--pass)} .no{color:var(--accent)}
.vbig{font-family:var(--kai);font-size:34px;font-weight:700;letter-spacing:.01em;line-height:1}
.vsub{margin-top:11px;color:var(--ink2);font-size:12px}
.kpis{display:flex;gap:28px}
.kpi{min-width:92px}
.klabel{display:block;font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3)}
.kval{display:block;font-family:var(--mono);font-variant-numeric:tabular-nums;
      font-size:20px;font-weight:500;letter-spacing:-.03em;margin-top:3px}
.knote{display:block;font-size:10.5px;color:var(--ink3);margin-top:1px}

/* ── 区块 ── */
section{margin-top:26px}
h2{font-family:var(--kai);font-size:13px;font-weight:600;letter-spacing:-.01em;margin-top:5px;
   padding-bottom:8px;border-bottom:1px solid var(--line2);display:flex;justify-content:space-between;align-items:baseline}
h2 .c{font-family:var(--mono);font-size:11px;color:var(--ink3);font-weight:400}

/* ── 表 ── */
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--line);vertical-align:baseline}
th{font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink3);font-weight:600;
   border-bottom:1px solid var(--line2);padding-top:4px}
td{font-size:12.5px}
td.r,th.r{text-align:right;padding-right:0}
tr:last-child td{border-bottom:1px solid var(--line2)}
.dot{display:inline-block;width:6px;height:6px;margin-right:8px;vertical-align:1px}
.d-pass{background:var(--pass)} .d-fail{background:var(--fail)} .d-fatal{background:var(--fatal)}
.st-pass{color:var(--ink2)} .st-fail{color:var(--fail);font-weight:500}

/* ── 否决 ── */
.veto{margin-top:6px}
.vitem{display:grid;grid-template-columns:112px 56px 1fr;gap:12px;padding:8px 0;
       border-bottom:1px solid var(--line);align-items:baseline}
.vitem:last-child{border-bottom:1px solid var(--line2)}
.vcode{font:500 10px/1.6 var(--mono);color:var(--fatal);letter-spacing:.06em}
.vcid{font:500 11.5px/1.6 var(--mono);color:var(--ink2)}
.vnote{font-size:12.5px;color:var(--ink2)}

/* ── 案例 ── */
details{border-bottom:1px solid var(--line)}
summary{cursor:pointer;padding:7px 0;font-size:12.5px;list-style:none;
        display:grid;grid-template-columns:52px 1fr auto;gap:10px;align-items:baseline}
summary::-webkit-details-marker{display:none}
summary:hover{background:var(--sunk)}
.cid{font:500 11.5px/1.6 var(--mono);color:var(--ink3)}
.cnote{color:var(--ink2)}
.tag{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3)}
pre{background:var(--sunk);border-left:1.5px solid var(--line2);padding:10px 12px;margin:2px 0 12px 62px;
    font:11.5px/1.7 var(--mono);white-space:pre-wrap;word-break:break-word;color:var(--ink2)}

/* ── 下一步 ── */
.acts{list-style:none;margin-top:4px}
.acts li{padding:11px 0;border-bottom:1px solid var(--line);
         display:grid;grid-template-columns:198px 1fr;gap:16px;align-items:baseline}
.acts li:last-child{border-bottom:1px solid var(--line2)}
.acts b{color:var(--ink);font-weight:600;font-size:12.5px}
.acts span{color:var(--ink2);font-size:12.5px}
@media(max-width:720px){.acts li{grid-template-columns:1fr;gap:3px}}

.foot{margin-top:38px;padding-top:12px;border-top:1.5px solid var(--ink);
      font-size:10.5px;color:var(--ink3);line-height:2}
.foot b{color:var(--ink2);font-weight:600;letter-spacing:.06em}
@media(max-width:720px){
  .verdict{grid-template-columns:1fr;gap:18px}
  .kpis{flex-wrap:wrap}
  .kpi{border-left:none;padding:0 20px 0 0}
}
"""


FCODE_ZH = {}

# ── 指标 key → 中文名 ──────────────────────────────────────────────
# 只做「兜底翻译」：真正的中文名由调用方传入的 label 映射（评分器 / 环节配置里的
# gates[].label、derived[].label、baseline["labels"]）决定；查不到就直接显示 key 本身，
# 绝不写死某一条业务线的指标名。
DEFAULT_METRIC_LABELS = {
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1",
    "pass_rate": "通过率",
    "score": "得分",
    "coverage": "覆盖率",
    "mention_ok": "必需内容命中",
    "forbidden_hits": "禁用词命中数",
    "schema_ok": "结构合规",
    "field_ok": "字段完整",
    "enum_ok": "枚举合规",
    "len_ok": "长度合规",
    "calls_ok": "调用次数合规",
    "whitelist_ok": "工具白名单合规",
    "required_ok": "必调工具覆盖",
    "args_ok": "参数完整",
    "has_citation": "是否给出引用",
    "phantom_refs": "无效引用数",
    "outcome_ok": "结果达成",
}

# 纯计数型 key：它们参与分类统计，但不是「比率」，
# 直接按百分号铺进基线表只会造成误读，因此只做列名过滤、不进基线表。
COUNT_KEYS = {
    "tp", "fp", "fn", "tn", "btp", "bfn", "field_hit", "field_total",
    "__n__", "n_calls", "missing", "phantom_refs", "forbidden_hits",
}


def _label_map(labels=None, ev=None, baseline=None) -> dict:
    """合并中文名映射，优先级：显式 labels > 基线自带 labels > 环节指标 label > 内置兜底。"""
    out = dict(DEFAULT_METRIC_LABELS)
    for c in (ev or {}).get("checks", []) or []:
        if c.get("key") and c.get("label"):
            out[c["key"]] = c["label"]
    for k, v in ((baseline or {}).get("labels") or {}).items():
        if v:
            out[k] = v
    for k, v in (labels or {}).items():
        if v:
            out[k] = v
    return out


def _mlabel(key, labels):
    return labels.get(key) or key


def _dot(kind):
    return f'<span class="dot d-{kind}"></span>'


def _fmt(c):
    a = c.get("actual")
    if a is None:
        return "—"
    return f"{a:.1%}" if c.get("value", 1) <= 1 else f"{a:.1f}"


def _op(sym):
    return {"<=": "≤", ">=": "≥"}.get(sym, sym)


def _gap(c):
    """返回「离达标还差多少」，正数表示不达标。"""
    a, v = c.get("actual"), c.get("value", 0)
    if a is None:
        return 0.0
    return (v - a) if c["op"] == ">=" else (a - v)


def _suggest(ev, rows):
    out = []
    if ev.get("fatals"):
        out.append(("检查否决项", "核对命中样本与评分规则。"))
    for c in ev.get("checks", []):
        if not c.get("pass"):
            out.append((str(c.get("label") or c.get("key")), "未达到配置门槛，请检查样本和评分依据。"))
    if any(r.get("verdict") == "review" for r in rows):
        out.append(("待复核", "部分样本缺少证据或评分失败，请人工复核。"))
    return out or [("查看样本", "复核评测结果并保存回归样本。")]


def _ratio(a, b):
    return a / b if b else None


def _pct(v):
    return "—" if v is None else f"{v:.1%}"


def _classification(ev) -> dict:
    """本次指标里的二分类统计。

    来源有两种，有任何一种才出区块：
      1. 混淆矩阵计数 tp/fp/fn/tn —— 自行换算准确率/精确率/召回率/F1；
      2. 已经算好的比率 accuracy/precision/recall/f1 —— 直接用配置值。
    「样本数」优先取混淆矩阵之和，没有矩阵时退回用例数。
    """
    m = ev.get("metrics") or {}
    num = lambda k: m.get(k) if isinstance(m.get(k), (int, float)) else None

    tp, fp, fn, tn = (num("tp"), num("fp"), num("fn"), num("tn"))
    n = sum(x for x in (tp, fp, fn, tn) if x is not None) if any(
        x is not None for x in (tp, fp, fn, tn)) else ev.get("n")

    vals: dict[str, float | None] = {"accuracy": num("accuracy"), "precision": num("precision"),
                                     "recall": num("recall"), "f1": num("f1")}
    if all(x is not None for x in (tp, fp, fn, tn)):
        if n:
            vals["accuracy"] = vals["accuracy"] if vals["accuracy"] is not None else _ratio(tp + tn, n)
        vals["precision"] = vals["precision"] if vals["precision"] is not None else _ratio(tp, tp + fp)
        vals["recall"] = vals["recall"] if vals["recall"] is not None else _ratio(tp, tp + fn)
        p, r = vals["precision"], vals["recall"]
        if vals["f1"] is None and p is not None and r is not None and (p + r):
            vals["f1"] = 2 * p * r / (p + r)

    rows = [(name, _pct(vals[key])) for key, name in
            (("accuracy", "准确率"), ("precision", "精确率"), ("recall", "召回率"), ("f1", "F1"))
            if vals[key] is not None]
    if not rows:
        return {"rows": []}
    rows.append(("样本数", f"{n:g}" if isinstance(n, (int, float)) else "—"))
    return {"rows": rows}


def render(ev, run, rows, out_path, baseline=None, labels=None):
    ok = ev["passed"]
    nf = len(ev["fatals"])
    pc = sum(1 for c in ev["checks"] if c["pass"])

    # 最大缺口
    worst = max((c for c in ev["checks"] if not c["pass"]), key=_gap, default=None)
    gap_txt = f"{_gap(worst) * 100:.1f}pt" if worst else "—"
    gap_name = worst["label"] if worst else "无缺口"
    # 最差分层
    st = ev["by_stratum"]
    worst_st = min(st.items(), key=lambda kv: kv[1]["pass"] / max(kv[1]["n"], 1)) if st else None
    st_txt = f"{worst_st[1]['pass']}/{worst_st[1]['n']}" if worst_st else "—"
    st_name = worst_st[0] if worst_st else "—"

    rep = ev.get("repeat") or 1
    if rep > 1:
        pk = ev.get("pass_pow_k")
        pa = ev.get("pass_at_k")
        unstable = ev.get("unstable") or []
        stable_kpi = (f"<div class='kpi'><span class='klabel'>稳定性 {rep} 遍</span>"
                      f"<span class='kval' style='color:{'var(--pass)' if (pk or 0) >= 0.8 else 'var(--fail)'}'>"
                      f"{(pk or 0)*100:.0f}%</span>"
                      f"<span class='knote'>{len(unstable)} 题不稳</span></div>")
    else:
        stable_kpi = ""

    mrows = "".join(
        f"<tr><td class='mkey'>{html.escape(c['label'])}</td>"
        f"<td class='r num'>{_fmt(c)}</td>"
        f"<td class='r num' style='color:var(--ink3)'>{_op(c['op'])} {c['value']:.0%}</td>"
        f"<td class='r {'st-pass' if c['pass'] else 'st-fail'}'>{'达标' if c['pass'] else '未达标'}</td></tr>"
        for c in ev["checks"]
    )

    veto = ""
    if ev["fatals"]:
        items = "".join(
            f"<div class='vitem'><span class='vcode'>{html.escape(f['code'])} {FCODE_ZH.get(f['code'],'')}</span>"
            f"<span class='vcid'>{html.escape(f['case_id'])}</span>"
            f"<span class='vnote'>{html.escape(f['note'][:170])}</span></div>"
            for f in ev["fatals"])
        veto = f"<section><div class='eyebrow'>VETO</div><h2>一票否决<span class='c'>{nf}</span></h2><div class='veto'>{items}</div></section>"

    srows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='r num'>{v['n']}</td>"
        f"<td class='r num'>{v['pass']}</td><td class='r num'>{v['fail']}</td>"
        f"<td class='r num'>{v['fatal']}</td>"
        f"<td class='r num'>{v['pass'] / max(v['n'], 1):.0%}</td></tr>"
        for k, v in sorted(st.items()))

    # 逐条：同一道题跑了 N 遍就并成一条，展开能看到执行记录（发出去的 prompt）与每次输出
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["case_id"], []).append(r)
    cases = []
    for cid, gs in sorted(grouped.items(),
                          key=lambda kv: (0 if any(x["fatal"] for x in kv[1])
                                          else 1 if any(x["verdict"] != "pass" for x in kv[1]) else 2, kv[0])):
        worst = next((x for x in gs if x["verdict"] != "pass"), gs[0])
        kind = "fatal" if worst["fatal"] else ("fail" if worst["verdict"] != "pass" else "pass")
        n_ok = sum(1 for x in gs if x["verdict"] == "pass")
        rep_txt = f"<span class='tag'>{n_ok}/{len(gs)}</span>" if len(gs) > 1 else ""
        tr = gs[0].get("prompt") or ""
        blocks = []
        if tr:
            blocks.append(f"<div class='tiny'>发给模型的原文</div><pre>{html.escape(tr)[:2000]}</pre>")
        for i, x in enumerate(gs):
            head = f"第 {i+1} 次 · {x['verdict']}" if len(gs) > 1 else "输出"
            blocks.append(f"<div class='tiny'>{head}</div><pre>{html.escape((x['output'] or '(空)')[:1400])}</pre>")
        blocks.append(f"<div class='tiny'>判分说明</div><pre>{html.escape(worst['note'] or '—')}</pre>")
        human = next((x.get("human_verdict") for x in gs if x.get("human_verdict")), "")
        cases.append(
            f"<details><summary>{_dot(kind)}<span class='cid'>{html.escape(cid)}</span>"
            f"<span class='cnote'>{html.escape(worst['note'] or '')}</span>"
            f"<span class='tag'>{html.escape(worst['stratum'] or '')}</span>{rep_txt}</summary>"
            f"<div class='revrow'>人工复核（代码判分对软指标偏严，不服就改）"
            f"<button class='rev' data-cid='{html.escape(cid)}' data-h='pass'>通过</button>"
            f"<button class='rev' data-cid='{html.escape(cid)}' data-h='fail'>不通过</button>"
            f"<span class='revnow' data-cid='{html.escape(cid)}'>{('已标：' + human) if human else ''}</span></div>"
            f"{''.join(blocks)}</details>")

    acts = "".join(f"<li><b>{t}</b><span>{b}</span></li>" for t, b in _suggest(ev, rows))

    # ── 稳定性：同一道题跑 N 遍的差别 ──
    stable_block = ""
    if (ev.get("repeat") or 1) > 1:
        pa, pk = ev.get("pass_at_k"), ev.get("pass_pow_k")
        uns = ev.get("unstable") or []
        rows_html = "".join(
            f"<tr><td>{html.escape(u['case_id'])}</td><td>{html.escape(u['stratum'] or '-')}</td>"
            f"<td class='r num'>{u['pass']}/{u['n']}</td></tr>" for u in uns[:40])
        stable_block = f"""<section><div class="eyebrow">Stability</div>
<h2>跑 {ev['repeat']} 遍的差别<span class="c">pass@{ev['repeat']} {(pa or 0)*100:.0f}% · pass^{ev['repeat']} {(pk or 0)*100:.0f}%</span></h2>
<div class="note">判定要求 <b>{ev['repeat']} 次全对</b>。pass@{ev['repeat']} 是「至少对一次」——
两者的差距就是模型的不稳定性，也是「分数涨了是真变好，还是刚好多对了一道」的答案。</div>
{f"<table><thead><tr><th>题号</th><th>分层</th><th class='r'>对了</th></tr></thead><tbody>{rows_html}</tbody></table>" if uns else "<div class='note'>没有不稳定的题——每一道都是 N 次同一结果。</div>"}
</section>"""


    # ── 基线对照：gold / empty，列由 baseline 实际产出的指标 key 动态生成 ──
    baseline_block = ""
    if baseline and baseline.get("supported"):
        labels = _label_map(labels, ev, baseline)
        baselines = baseline.get("baselines") or []
        n_cases = baseline.get("n_cases") or ev["n"] or 1

        # 列 = 所有基线里真实存在（数值非空）的指标 key；某条基线没这个指标就留空，
        # 所有基线都没有的指标整列不出现。
        cols: list[str] = []
        for b in baselines:
            for k, val in (b.get("metrics") or {}).items():
                if isinstance(val, (int, float)) and k not in COUNT_KEYS and k not in cols:
                    cols.append(k)

        def _cell(b, key):
            val = (b.get("metrics") or {}).get(key)
            if not isinstance(val, (int, float)):
                return ""
            # 计数型 key 和被环节声明成计数的指标（如命中条数）直接显示原值，
            # 0~1 之间的比率才换算成百分数。
            if key in COUNT_KEYS or not 0 <= val <= 1:
                return f"{val:g}"
            return f"{val*100:.1f}%"

        head = "".join(f'<th class="r">{html.escape(_mlabel(k, labels))}</th>' for k in cols)
        body = ""
        for b in baselines:
            cells = "".join(f'<td class="r num">{_cell(b, k)}</td>' for k in cols)
            body += (f"<tr><td>{html.escape(b.get('label') or b.get('key') or '')}</td>"
                     f"<td class='r num'>{b.get('n_pass', 0)}/{n_cases}</td>{cells}"
                     f"<td class='r num'>{b.get('fatal', 0)}</td></tr>")

        v = baseline.get("verdict", {})
        empty_rate = v.get("empty_pass_rate", v.get("null_pass_rate"))
        judge = []
        if not v.get("gold_ok"):
            judge.append("<b class='no'>参考答案没过评分器</b>——分数不可信，先修评分器或检查题目是否可解。")
        else:
            judge.append("参考答案全过，题目可解、评分器配置正确。")
        if v.get("n_holes"):
            rate = f"（{empty_rate*100:.0f}%）" if isinstance(empty_rate, (int, float)) else ""
            judge.append(f"<b class='no'>{v['n_holes']} 道题在空输出下也能通过</b>{rate}，"
                         f"这部分是不用做事就能拿到的地板分。")
        else:
            judge.append("没有空输出也能过的用例。")
        baseline_block = f"""<section><div class="eyebrow">Baseline</div>
<h2>基线对照<span class="c">{len(baselines)} 条基线 · 不跑模型，只跑评分器</span></h2>
<div class="note">{' '.join(judge)}</div>
<table><thead><tr><th>基线</th><th class="r">通过</th>{head}<th class="r">否决</th></tr></thead>
<tbody>{body}</tbody></table>
<div class="note" style="border-left-color:var(--ink3);color:var(--ink3)">
列按本次基线实际产出的指标生成，没有的指标不显示。先看「空输出」那一行：
真实输出只有明显高于它，这次评测才有信息量。</div>
</section>"""

    # ── 分类指标：只在本次指标里真有二分类统计时才渲染 ──
    class_block = ""
    if ev.get("metrics"):
        labels = _label_map(labels, ev, baseline)
        cm = _classification(ev)
        if cm["rows"]:
            body = "".join(
                f"<tr><td>{html.escape(r[0])}</td><td class='r num'>{r[1]}</td></tr>"
                for r in cm["rows"])
            class_block = f"""<section><div class="eyebrow">Classification</div>
<h2>分类指标<span class="c">按用例标注的二分类统计</span></h2>
<table><thead><tr><th>指标</th><th class="r">数值</th></tr></thead><tbody>{body}</tbody></table>
<div class="note" style="border-left-color:var(--ink3);color:var(--ink3)">
基于用例标注的二分类统计，与上方质量分不是同一个口径，不要相加。</div>
</section>"""

    reviews_js = """<script>
(function(){
  const RID = __RID__;
  const bar = document.getElementById('revbar');
  const paint = d => {
    if(!d) return;
    if(!d.n_marked){ bar.textContent = '人工复核：还没标。'; return; }
    const pct = (d.agree_rate*100).toFixed(0);
    bar.innerHTML = '人工复核 <b>'+d.n_marked+'/'+d.n_cases+'</b> 条 · 与代码判定一致 <b>'+pct+'%</b>'
      + (d.disagree.length ? '　不一致：'+d.disagree.map(x=>x.case_id).join('、') : '');
  };
  fetch('/api/runs/'+RID+'/review').then(r=>r.json()).then(paint).catch(()=>{});
  document.querySelectorAll('button.rev').forEach(b=>{
    b.onclick = async ev => {
      ev.preventDefault(); ev.stopPropagation();
      const cid = b.dataset.cid, h = b.dataset.h;
      const r = await fetch('/api/runs/'+RID+'/review', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({case_id: cid, human: h})});
      const d = await r.json();
      document.querySelectorAll('.revnow[data-cid="'+cid+'"]').forEach(e=>e.textContent='已标：'+h);
      paint(d);
    };
  });
})();
</script>""".replace("__RID__", json.dumps(run["id"]))
    doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>评测报告 · {html.escape(run['product'])} / {html.escape(run['stage'])}</title>
<style>{CSS}</style></head><body><div class="wrap">

<div class="top">
  <div>
    <div class="eyebrow">Evaluation Report</div>
    <h1>{html.escape(run.get('product_name', run['product']))}<span class="h1tag">{html.escape(run['stage'])}</span></h1>
  </div>
  <div class="meta">
    <span class="num">{html.escape(run['model'])}</span> · <span class="num">{ev['n']}</span> 用例 · <span class="num">¥{run.get('cost', 0):.4f}</span><br>
    <span class="num">{html.escape(run['started_at'])}</span>
  </div>
</div>

<div class="verdict">
  <div>
    <div class="vline">
      <span class="vtag {'ok' if ok else 'no'}">{'PASS' if ok else 'FAIL'}</span>
      <span class="vbig {'ok' if ok else 'no'}">{ev['verdict']}</span>
    </div>
    <div class="vsub">指标 <b class="num">{pc}/{len(ev['checks'])}</b> 达标 · 一票否决 <b class="num">{nf}</b> 次{'　·　全部通过' if ok else ''}</div>
  </div>
  <div class="kpis">
    <div class="kpi"><span class="klabel">最大缺口</span><span class="kval">{gap_txt}</span><span class="knote">{html.escape(gap_name)}</span></div>
    <div class="kpi"><span class="klabel">否决</span><span class="kval">{nf}</span><span class="knote">优先级最高</span></div>
    <div class="kpi"><span class="klabel">最差分层</span><span class="kval">{st_txt}</span><span class="knote">{html.escape(st_name[:16])}</span></div>
    {stable_kpi}
  </div>
</div>

{stable_block}

<section><div class="eyebrow">Metrics</div><h2>指标达成<span class="c">{pc}/{len(ev['checks'])}</span></h2>
<table><thead><tr><th>指标</th><th class="r">实际</th><th class="r">要求</th><th class="r">状态</th></tr></thead>
<tbody>{mrows}</tbody></table></section>

{class_block}

{veto}

{baseline_block}

<section><div class="eyebrow">By Stratum</div><h2>分层分布</h2>
<table><thead><tr><th>分层</th><th class="r">用例</th><th class="r">通过</th><th class="r">失败</th><th class="r">否决</th><th class="r">通过率</th></tr></thead>
<tbody>{srows}</tbody></table></section>

<section><div class="eyebrow">Action</div><h2>下一步</h2><ul class="acts">{acts}</ul></section>

<section><div class="eyebrow">Cases</div><h2>逐条结果<span class="c">{ev['n']}</span></h2>
<div id="revbar" class="revbar">人工复核：还没标。<span style="color:var(--ink3)">（标几条就能看出代码判分和你差多少）</span></div>
{''.join(cases)}</section>

__REVIEW_JS__

<div class="foot">
  <b>口径</b>　本报告按数据集「{html.escape(str(run.get('dataset') or '未标注'))}」的 {len(ev['checks'])} 项指标门槛出结论，
  汇总 {len(ev.get('metrics') or {})} 项指标{'；评分器 ' + html.escape(str(baseline['grader'])) if baseline and baseline.get('grader') else ''}；
  同一用例重复运行时指标取均值、判定取全对，阈值以本次配置为准。<br>
  <b>限制</b>　本次共 {ev['n']} 条样本，结论仅适用于该数据集与本次测试条件，样本较少时应谨慎外推{'；分类指标与质量分是两套口径，不要互相换算或相加' if class_block else ''}。<br>
  <b>模式</b>　{'本地仿真（mock，用于验证链路，不代表模型真实表现）' if run.get('mode') == 'mock' else '真实 API 调用'}
</div>

</div></body></html>"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    doc = doc.replace("__REVIEW_JS__", reviews_js)
    Path(out_path).write_text(doc, encoding="utf-8")
    return str(out_path)


# ═══════════════════════ 对照报告（A/B 双跑）═══════════════════════
def _delta(a, b):
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _changed_axes(ver_a, ver_b):
    """判断两次评测之间「哪些变量变了」。归因的前提是只有一个变量变。"""
    if not ver_a or not ver_b:
        return None
    ch = []
    if (ver_a.get("prompt_ver") or "") != (ver_b.get("prompt_ver") or ""):
        ch.append("prompt")
    if (ver_a.get("product_ver") or "") != (ver_b.get("product_ver") or ""):
        ch.append("product")
    return ch


def _diagnose(ma, mb, diffs, label_b="产品", axis=None):
    evidence = [str(x.get("case_id")) for x in diffs if x.get("a_verdict") != x.get("b_verdict")]
    return {"bad": [], "good": [], "evidence": ["状态变化样本：" + ", ".join(evidence[:6])] if evidence else []}


def render_compare(a_run, b_run, a_ev, b_ev, a_rows, b_rows, out_path,
                   ver_a=None, ver_b=None):
    """对照报告。ver_a/ver_b 有值时，按「哪个变量变了」决定能不能归因。"""
    da = {r["case_id"]: r for r in a_rows}
    db = {r["case_id"]: r for r in b_rows}
    common = sorted(set(da) & set(db))

    diffs = []
    for cid in common:
        x, y = da[cid], db[cid]
        if x["verdict"] != y["verdict"]:
            diffs.append({"case_id": cid, "stratum": x["stratum"],
                          "a": x["verdict"], "b": y["verdict"],
                          "a_note": x["note"], "b_note": y["note"]})

    SKIP = set(COUNT_KEYS)
    keys = [k for k in sorted(set(a_ev["metrics"]) | set(b_ev["metrics"]))
            if k not in SKIP and a_ev["metrics"].get(k) is not None
            and b_ev["metrics"].get(k) is not None]

    axis = _changed_axes(ver_a, ver_b)
    diag = _diagnose(a_ev["metrics"], b_ev["metrics"], diffs,
                     label_b=("B" if ver_a else "产品"), axis=axis)
    mock_warn = (a_run.get("mode") == "mock")

    if axis is None:
        axis_html = ""
    elif len(axis) == 2:
        axis_html = ("<div class='warn'><b>无法归因：prompt 和产品同时改了。</b>"
                     "这次分数变化无法拆分到具体某一层——可能 prompt 改好了、产品改坏了互相抵消，也可能反过来。"
                     "<br><b>建议</b>：把改动拆成两次跑（先只改 prompt，再只改产品），各拿一个版本号。</div>")
    elif axis == ["prompt"]:
        axis_html = ("<div class='note'><b>只改了 prompt</b>（产品版本相同），差异仍需排除随机性及其他变量，不能直接认定因果。 </div>")
    else:
        axis_html = ("<div class='note'><b>只改了产品</b>（prompt 版本相同），"
                     "差异仍需排除随机性及其他变量，不能直接认定因果。 </div>")

    if ver_a or ver_b:
        va, vb = ver_a or {}, ver_b or {}
        ver_head = f"""<section><div class="eyebrow">Versions</div><h2>版本对照</h2>
        <table><thead><tr><th></th><th>A</th><th>B</th></tr></thead><tbody>
        <tr><td style="color:var(--ink3)">版本</td><td class="num">{html.escape(va.get('label') or '—')}</td><td class="num">{html.escape(vb.get('label') or '—')}</td></tr>
        <tr><td style="color:var(--ink3)">prompt</td><td>{html.escape(va.get('prompt_ver') or '默认')}</td><td>{html.escape(vb.get('prompt_ver') or '默认')}</td></tr>
        <tr><td style="color:var(--ink3)">产品</td><td>{html.escape(va.get('product_ver') or '—')}</td><td>{html.escape(vb.get('product_ver') or '—')}</td></tr>
        <tr><td style="color:var(--ink3)">改动说明</td><td>{html.escape(va.get('change_note') or '—')}</td><td>{html.escape(vb.get('change_note') or '—')}</td></tr>
        </tbody></table></section>"""
        axis_html = ver_head + axis_html

    mrows = []
    for k in keys:
        va, vb = a_ev["metrics"].get(k), b_ev["metrics"].get(k)
        dv = _delta(va, vb)
        low_is_bad = k in ("fpr", "false_positive_rate")
        better = None
        if dv is not None and abs(dv) > 0.05:
            better = "B" if (dv < 0 if low_is_bad else dv > 0) else "A"
        cls = "" if better is None else ("ok" if better == "B" else "bad")
        mrows.append(
            f"<tr><td class='num'>{html.escape(k)}</td>"
            f"<td class='r num'>{va:.1%}</td><td class='r num'>{vb:.1%}</td>"
            f"<td class='r num {cls}'>{'' if dv is None else f'{dv*100:+.1f}pt'}</td>"
            f"<td class='r' style='color:var(--ink3)'>{'' if better is None else ('B 更好' if better == 'B' else 'A 更好')}</td></tr>")

    drows = "".join(
        f"<tr><td class='num'>{html.escape(d['case_id'])}</td>"
        f"<td style='color:var(--ink3)'>{html.escape(d['stratum'])}</td>"
        f"<td class='num {'ok' if d['a'] == 'pass' else 'bad'}'>{html.escape(d['a'])}</td>"
        f"<td class='num {'ok' if d['b'] == 'pass' else 'bad'}'>{html.escape(d['b'])}</td>"
        f"<td>{html.escape(d['b_note'] or d['a_note'] or '')}</td></tr>"
        for d in diffs)

    bad_html = "".join(f"<li><b>{html.escape(t)}</b><span>{v}</span></li>" for t, v in diag["bad"]) or \
        "<li><b>没有发现产品侧明显退化</b><span>两边指标差异都在 5pt 内，说明产品层没有削弱模型能力。</span></li>"
    good_html = "".join(f"<li><b>{html.escape(t)}</b><span>{v}</span></li>" for t, v in diag["good"]) or \
        "<li><b>没有发现产品侧明显优势</b><span>产品层目前只是「转发 + 包装」，还没体现出额外价值。</span></li>"
    ev_html = ("<div class='note'>证据：" + "；".join(html.escape(e) for e in diag["evidence"]) + "</div>"
               if diag["evidence"] else "")

    rev_a = sum(1 for d in diffs if d["a"] != "pass" and d["b"] == "pass")
    rev_b = sum(1 for d in diffs if d["a"] == "pass" and d["b"] != "pass")

    doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>对照报告 · prompt 基线 vs 产品接口</title><style>{CSS}</style></head><body><div class="wrap">

<div class="top">
  <div><div class="eyebrow">Comparison Report</div>
  <h1>{html.escape(a_run.get('product_name', a_run['product']))}<span class="h1tag">prompt 基线 vs 产品接口</span></h1></div>
  <div class="meta">A <span class="num">{html.escape(a_run['id'][-4:])}</span> · <span class="num">{html.escape(a_run.get('model') or '')}</span><br>
  B <span class="num">{html.escape(b_run['id'][-4:])}</span> · <span class="num">{html.escape(b_run.get('model') or '')}</span><br>
  <span class="num">{html.escape(b_run['started_at'])}</span></div>
</div>

{"<div class='warn'><b>注意：基线用的是本地仿真</b>，不是真实模型输出。这份对照只能看链路是否打通，不能用来判断产品的真实水平。配置 API Key 后重跑才有意义。</div>" if mock_warn else ""}

<div class="verdict">
  <div>
    <div class="vline"><span class="vtag no">DELTA</span>
      <span class="vbig">{len(diffs)}<span style="font-size:16px;color:var(--ink3);font-weight:400"> 条判定翻转</span></span></div>
    <div class="vsub">共同用例 <b class="num">{len(common)}</b> · 产品变好 <b class="num ok">{rev_a}</b> · 产品变差 <b class="num bad">{rev_b}</b></div>
  </div>
  <div class="kpis">
    <div class="kpi"><span class="klabel">A 基线</span><span class="kval">{a_ev['verdict']}</span><span class="knote">{sum(1 for c in a_ev['checks'] if c['pass'])}/{len(a_ev['checks'])} 指标</span></div>
    <div class="kpi"><span class="klabel">B 产品</span><span class="kval">{b_ev['verdict']}</span><span class="knote">{sum(1 for c in b_ev['checks'] if c['pass'])}/{len(b_ev['checks'])} 指标</span></div>
    <div class="kpi"><span class="klabel">否决</span><span class="kval">{len(a_ev['fatals'])}→{len(b_ev['fatals'])}</span><span class="knote">A→B</span></div>
  </div>
</div>

<section><div class="eyebrow">Metrics</div><h2>指标差异<span class="c">只比两边都有的比率型指标</span></h2>
<table><thead><tr><th>指标</th><th class="r">A 基线</th><th class="r">B 产品</th><th class="r">差值</th><th class="r">判断</th></tr></thead>
<tbody>{''.join(mrows)}</tbody></table>
<div class="note">差异小于 5pt 一律视为同档——样本量这个级别，更小的差异是噪声。</div></section>

<section><div class="eyebrow">Diagnose</div><h2>问题在哪<span class="c">该改 prompt 还是改产品代码</span></h2>
{axis_html}
<ul class="acts">{bad_html}</ul>
{ev_html}
<div class="eyebrow" style="margin-top:22px">产品做对的地方</div>
<h2>产品层的优势</h2>
<ul class="acts">{good_html}</ul></section>

<section><div class="eyebrow">Flips</div><h2>判定翻转的用例<span class="c">{len(diffs)}</span></h2>
{('<table><thead><tr><th>用例</th><th>分层</th><th>A</th><th>B</th><th>B 侧说明</th></tr></thead><tbody>' + drows + '</tbody></table>') if diffs else '<p class="empty">两边判定完全一致。</p>'}</section>

<div class="foot">
  <b>怎么读</b>　A 是「只喂 prompt 给模型」，B 是「打你产品的真实接口」。<br>
  <b>归因前提</b>　两次必须同数据集、同用例；A/B 之间只允许「是否经过产品」这一个变量。<br>
  <b>模式</b>　A：{html.escape(a_run.get('mode', ''))} ／ B：{html.escape(b_run.get('mode', ''))}
</div>

</div></body></html>"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(doc, encoding="utf-8")
    return str(out_path)
