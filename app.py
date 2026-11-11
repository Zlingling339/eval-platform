"""Streamlit 执行台（低频操作：选配置、跑批、看判定）。

视觉定位：执行台不求美，只求不出错、够快。真正要好看的是生成的 HTML 报表。
所以这里刻意用最少的样式，不做主题魔改。
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core import (init_db, run_stage, evaluate, load_products,
                      get_results, list_runs, StageConfig)
from src.products import PRODUCTS
import src.graders_rule  # noqa: F401  确定性评分器
import src.graders_llm  # noqa: F401  LLM-as-Judge 评分器
from src.report import render

st.set_page_config(page_title="评测平台", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container{padding-top:2.2rem;max-width:1240px}
      h1{font-size:20px!important;letter-spacing:-.01em}
      div[data-testid="stMetricValue"]{font-family:ui-monospace,monospace;font-size:22px}
      .stButton>button{border-radius:0}
    </style>
    """,
    unsafe_allow_html=True,
)

prods = load_products()
con = init_db()

# ── 侧栏 ──
with st.sidebar:
    st.markdown("### 配置")
    pk = st.selectbox("产品", list(prods), format_func=lambda k: prods[k]["name"])
    cfg = prods[pk]
    sk = st.selectbox("环节", [s["id"] for s in cfg["stages"]],
                      format_func=lambda sid: next(s["name"] for s in cfg["stages"] if s["id"] == sid))
    model = st.text_input("模型", value="",
                          placeholder="填写被测模型名（留空 + 勾选本地仿真即可验证链路）")
    limit = st.number_input("限制条数（调试用，0 = 全部）", 0, 500, 0)
    mock = st.toggle("本地仿真（不联网）", value=True,
                     help="用于验证链路与报表；打开时结果不代表模型真实表现")
    run = st.button("开始评测", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### 历史")
    hist = list_runs(con, product=pk)
    if hist:
        for r in hist[:10]:
            st.caption(f"{r['started_at']} · {r['stage']} · {r['n_cases']} 条 · {r['model']}")
    else:
        st.caption("还没有运行记录")

# ── 主区 ──
st.title("LLM 产品评测平台")
st.caption(f"{cfg['name']} · v{cfg.get('version','1.0')} · 同一套引擎，换产品只改 config + dataset + grader")

if run:
    stage_cfg = next(s for s in cfg["stages"] if s["id"] == sk)
    st_obj = StageConfig(
        id=stage_cfg["id"], name=stage_cfg["name"], task_type=stage_cfg["task_type"],
        dataset=stage_cfg["dataset"], grader=stage_cfg["grader"],
        gates=stage_cfg.get("gates", []), fatal=stage_cfg.get("fatal", []),
        counters=stage_cfg.get("counters", []), derived=stage_cfg.get("derived", []),
    )
    if pk not in PRODUCTS or sk not in PRODUCTS[pk]:
        st.error(f"产品 {pk} 的环节 {sk} 没有登记 prompt/mock 适配器。"
                 "请先在 src/products.py 的 PRODUCTS 里注册，或改用 Web 界面。")
        st.stop()
    pf, mf = PRODUCTS[pk][sk]
    with st.spinner("跑批中…"):
        rid = run_stage(cfg, st_obj, model=model, limit=(limit or None),
                        mock=mock, prompt_fn=pf, mock_reply_fn=mf)
        ev = evaluate(con, rid, st_obj)
        rows = get_results(con, rid)
        meta = next(r for r in list_runs(con) if r["id"] == rid)
        meta["product_name"] = cfg["name"]
        path = render(ev, meta, rows, ROOT / "reports" / f"{rid}.html")
    st.session_state["last"] = {"ev": ev, "path": path, "meta": meta, "rows": rows}

if "last" in st.session_state:
    d = st.session_state["last"]
    ev, meta, rows = d["ev"], d["meta"], d["rows"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("结论", ev["verdict"])
    c2.metric("指标达标", f"{sum(1 for c in ev['checks'] if c['pass'])}/{len(ev['checks'])}")
    c3.metric("一票否决", len(ev["fatals"]))
    c4.metric("成本", f"¥{meta.get('cost', 0):.4f}")

    st.markdown("#### 指标达成")
    st.dataframe(
        [{"指标": c["label"],
          "实际": "—" if c["actual"] is None else round(c["actual"], 3),
          "要求": f"{c['op']} {c['value']}",
          "状态": "达标" if c["pass"] else "未达标"} for c in ev["checks"]],
        use_container_width=True, hide_index=True,
    )

    if ev["fatals"]:
        st.error("一票否决：" + " · ".join(f"{f['case_id']}({f['code']})" for f in ev["fatals"]))

    st.markdown("#### 完整报表")
    st.caption(f"已生成静态 HTML：{Path(d['path']).name}（可单独打开、导出、放进作品集）")
    st.components.v1.html(Path(d["path"]).read_text(encoding="utf-8"), height=1400, scrolling=True)
else:
    st.info("左侧选择配置后点「开始评测」。首次建议用本地仿真跑一遍，确认任务卡与阈值是否符合预期。")
