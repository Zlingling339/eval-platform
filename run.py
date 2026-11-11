#!/usr/bin/env python3
"""通用评测平台 CLI。\n\n产品与环节来自外部配置，并通过适配器注册 prompt/mock；交付副本不内置业务示例。\n"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core import (init_db, run_stage, evaluate, load_products,
                      get_results, list_runs, StageConfig)
from src.products import PRODUCTS
import src.graders_rule  # noqa: F401  确定性评分器
import src.graders_llm   # noqa: F401  LLM-as-Judge 评分器
from src.models import load_router
from src.core import set_router
from src.report import render

# 评委必须和被测产品来自不同厂商，启动时就校验，不等到跑完才发现同源
ROUTER = load_router()
set_router(ROUTER)
for _x in ROUTER.independence():
    if _x.level != "ok":
        print(f"⚠️  模型独立性：{_x.level.upper()} {_x.role}↔{_x.peer}：{_x.reason}")
ROUTER.assert_startable()


def make_stage(s: dict) -> StageConfig:
    return StageConfig(
        id=s["id"], name=s["name"], task_type=s["task_type"],
        dataset=s["dataset"], grader=s["grader"],
        gates=s.get("gates", []), fatal=s.get("fatal", []),
        counters=s.get("counters", []), derived=s.get("derived", []),
        rubric=s.get("rubric", ""), pair_rule=s.get("pair_rule", ""),
    )


def print_summary(ev, run, path):
    line = "─" * 58
    print(f"\n{line}\n {run.get('product_name', run['product'])} · {run['stage']}")
    print(f" 用例 {ev['n']} · 模型 {run['model']} · 成本 ¥{run.get('cost', 0):.4f}\n{line}")
    mark = "✓" if ev["passed"] else "✗"
    print(f" 结论  {ev['verdict']}  {mark}")
    parts = []
    for c in ev["checks"]:
        v = "—" if c["actual"] is None else (f"{c['actual']:.1%}" if c["value"] <= 1 else f"{c['actual']:g}")
        parts.append(f"{c['label']}={v} {'✓' if c['pass'] else '✗'}")
    print(" 指标  " + " · ".join(parts))
    if ev["fatals"]:
        hit = " / ".join(f"{f['case_id']}({f['code']})" for f in ev["fatals"])
        print(f" 否决  {len(ev['fatals'])} 次：{hit}")
    print(f" 分层  " + " · ".join(f"{k} {v['pass']}/{v['n']}" for k, v in sorted(ev["by_stratum"].items())))
    print(f" 报表  {path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mock", action="store_true", help="不联网，用本地仿真输出验证链路")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="")
    a = ap.parse_args()

    prods = load_products()
    if not prods:
        sys.exit("本包没有内置产品配置（config/products/*.json 为空），CLI 无法直接跑批：\n"
                 "  1) 把产品配置放进 config/products/；\n"
                 "  2) 在 src/products.py 的 PRODUCTS 里登记 <product>/<stage> 的 prompt/mock 适配器；\n"
                 "  3) 或者用 server.py / Web 界面导入数据集，走通用产品配置。")
    if a.product not in prods:
        sys.exit(f"未找到产品 {a.product}，现有：{list(prods)}")
    cfg = prods[a.product]

    stages = cfg["stages"]
    if a.stage and not a.all:
        stages = [s for s in stages if s["id"] == a.stage]
        if not stages:
            sys.exit(f"产品 {a.product} 无环节 {a.stage}")

    con = init_db()
    for s in stages:
        if a.product not in PRODUCTS or s["id"] not in PRODUCTS[a.product]:
            sys.exit(f"产品 {a.product} 的环节 {s['id']} 没有登记 prompt/mock 适配器，"
                     f"CLI 不支持直接运行（请在 src/products.py 的 PRODUCTS 中注册后重试）。")
        pf, mf = PRODUCTS[a.product][s["id"]]
        st = make_stage(s)
        run_id = run_stage(cfg, st, model=a.model, limit=a.limit,
                           mock=a.mock, prompt_fn=pf, mock_reply_fn=mf)
        ev = evaluate(con, run_id, st)
        rows = get_results(con, run_id)
        run = next(r for r in list_runs(con) if r["id"] == run_id)
        run["product_name"] = cfg["name"]
        path = render(ev, run, rows, ROOT / "reports" / f"{run_id}.html")
        print_summary(ev, run, path)


if __name__ == "__main__":
    main()
