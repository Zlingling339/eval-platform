#!/usr/bin/env python3
"""评分器校准：它到底能不能分辨好坏？

为什么要单独跑这一步：
    报告上的「通过率」只告诉你过没过，不告诉你**评分器有没有区分力**。
    一个把所有答案都判 fail 的评分器，通过率 0%，看起来很严格，
    实际上零信息量。所以必须看分数分布与失败归因。

为什么分两层看：
    确定性评分器（代码判）和判官评分器（模型判）的**可信度不同**。
    mock 模式下判官分数只是近似，不能用来判断「rubric 好不好」——
    但规则层的结果是确定的，可以当真。
    把两层混在一起看，会得出「评分器不合格」这种错误结论。

用法：
    python3 tools/calibrate.py --product PRODUCT --stage STAGE
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core import DB, load_products, stage_of  # noqa: E402

RULE_HINTS = ("_ok", "path_recall", "state_match", "recall_at_k", "precision_at_k",
              "mrr", "hit_at_k", "coverage", "phantom_refs", "missing_steps", "extra_steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()

    cfg = load_products()[a.product]
    sc = stage_of(cfg, a.stage)
    ds = json.loads((ROOT / sc["dataset"]).read_text(encoding="utf-8"))
    tier = {c["id"]: c.get("tier", c.get("stratum", "?")) for c in ds["cases"]}

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    if a.run:
        run_id = a.run
    else:
        row = con.execute("SELECT id FROM run WHERE product=? AND stage=? "
                          "ORDER BY started_at DESC LIMIT 1", (a.product, a.stage)).fetchone()
        if not row:
            sys.exit("还没有跑过这个环节，先跑 run.py")
        run_id = row["id"]

    rows = [dict(r) for r in con.execute("SELECT * FROM result WHERE run_id=?", (run_id,))]
    if not rows:
        sys.exit("这次运行没有结果")

    is_mock = "mock" in run_id[:0] or all(r["note"] and "mock" in r["note"] for r in rows) \
        or any("mock" in (r["note"] or "") for r in rows)

    metrics_of = {r["case_id"]: json.loads(r["metrics"] or "{}") for r in rows}
    all_keys = sorted({k for m in metrics_of.values() for k in m})
    rule_keys = [k for k in all_keys if k.split(".")[-1].startswith(RULE_HINTS)]
    judge_keys = [k for k in all_keys if k.endswith("rubric_score")]

    order = [t for t in ("good", "mid", "bad") if any(tier.get(r["case_id"]) == t for r in rows)]

    def avg(t, keys):
        vals = {k: [] for k in keys}
        for r in rows:
            if tier.get(r["case_id"]) != t:
                continue
            for k in keys:
                v = metrics_of[r["case_id"]].get(k)
                if isinstance(v, (int, float)):
                    vals[k].append(v)
        return {k: (sum(v) / len(v) if v else None) for k, v in vals.items()}

    print(f"\n运行 {run_id} · {len(rows)} 条 · {sc['grader']}")
    print(f"{'人工标注':<10}{'条数':>4}{'通过':>6}{'待人工':>7}")
    for t in order:
        g = [r for r in rows if tier.get(r["case_id"]) == t]
        print(f"{t:<10}{len(g):>4}{sum(1 for r in g if r['verdict']=='pass'):>6}"
              f"{sum(1 for r in g if r['verdict']=='review'):>7}")

    # ── 规则层：mock 下也完全可信 ──
    if rule_keys:
        print(f"\n【规则层】确定性指标（mock 下同样可信）")
        a0 = {t: avg(t, rule_keys) for t in order}
        for k in rule_keys:
            vals = [a0[t][k] for t in order]
            if any(v is None for v in vals):
                continue
            mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
            gap = vals[0] - vals[-1]
            print(f"  {k:<42} {'✓' if mono else '✗'} "
                  + " ≥ ".join(f"{v:.2f}" for v in vals) + f"  间距 {gap:+.2f}")

    # ── 判官层：mock 下只是近似 ──
    if judge_keys:
        print(f"\n【判官层】rubric 分数"
              + ("（mock 近似，不代表 rubric 质量）" if is_mock else "（真实模型）"))
        a1 = {t: avg(t, judge_keys) for t in order}
        for k in judge_keys:
            vals = [a1[t][k] for t in order]
            if any(v is None for v in vals):
                continue
            mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
            print(f"  {k:<42} {'✓' if mono else '✗'} "
                  + " ≥ ".join(f"{v:.2f}" for v in vals))

    # ── 失败归因：比通过率更有用 ──
    print(f"\n【失败归因】哪一层把它判挂的")
    for t in order:
        g = [r for r in rows if tier.get(r["case_id"]) == t and r["verdict"] != "pass"]
        if not g:
            print(f"  {t:<10} 无失败")
            continue
        by = {}
        for r in g:
            who = (r["note"] or "")[1:].split("]")[0] if (r["note"] or "").startswith("[") else "未知"
            by[who or "未知"] = by.get(who or "未知", 0) + 1
        print(f"  {t:<10} " + "，".join(f"{k} {v} 条" for k, v in by.items()))

    # ── 与人工标注的一致度 ──
    passed = [r for r in rows if r["verdict"] == "pass"]
    good_ids = [r for r in rows if tier.get(r["case_id"]) == "good"]
    tp = sum(1 for r in passed if tier.get(r["case_id"]) == "good")
    recall = tp / len(good_ids) if good_ids else 0
    precision = tp / len(passed) if passed else 0
    kappa = _kappa([(tier.get(r["case_id"]) == "good", r["verdict"] == "pass") for r in rows])

    print(f"\n【与人工标注一致度】")
    print(f"  好答案被识别为通过   recall    = {recall:.1%}（{tp}/{len(good_ids)}）")
    print(f"  判通过的确实是好答案 precision = {precision:.1%}（{tp}/{len(passed)}）")
    print(f"  Cohen's kappa = {kappa:.3f}  （<0.4 不可用 / 0.4-0.6 尚可 / >0.6 可作门禁）")

    if is_mock:
        print("\n注意：本次为 mock 模式。判官分数是关键词近似的结果，"
              "\n      只能验证「链路是否通、组合会不会打架」，不能代表 rubric 的真实区分力。"
              "\n      要判断 rubric 准不准，必须接真实模型 + 人工抽样复核。")
    print()


def _kappa(pairs) -> float:
    """二分类 kappa。不用「一致率」的原因：类别不平衡时会虚高
    （全判 fail 也能有 66% 一致率）。kappa 扣掉了随机一致的那部分。"""
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


if __name__ == "__main__":
    main()
