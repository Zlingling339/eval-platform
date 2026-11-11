"""本地评测平台 Web 服务（单机，不上传任何数据）。

启动：python3 server.py    →  浏览器打开 http://localhost:8765

安全边界（重要）：
- API Key 由浏览器在请求中提供，仅用于当前请求或后台任务，不写入设置。
- 调用模型或产品接口时，相关输入会发送至配置的上游服务。
- 本服务需部署在可信环境；浏览器存储、网络和反向代理日志均需自行保护。
"""
from __future__ import annotations

import json, os, sqlite3, sys, threading, time, uuid, csv, io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core import (init_db, run_stage, evaluate, get_results, list_runs,
                      StageConfig, load_products, DB, HttpTarget, call_http, _fill,
                      has_stage, stage_of, save_product_doc)
from src.products import PRODUCTS
# 不加载 src/graders.py：里面的 5 个评分器（advise_quality / trigger_matrix /
# extract_accuracy / layout_validity / chat_quality）是某个具体产品的业务逻辑，
# 不是通用能力，已经跑偏。留在这里会被注册表带进「可选评分器」列表，
# 让新接入的人以为平台只支持那一种业务。
# 通用评分器只来自下面两个模块：
import src.graders_rule  # noqa: F401  确定性评分器
import src.graders_llm   # noqa: F401  LLM-as-Judge 评分器
from src import rubric as rubric_mod
from src.models import load_router, SiblingModelError, save_config, config_raw, ModelSpec
from src.core import set_router, set_request_context, reset_request_context, redact, validate_base_url
from src.report import render, render_compare
import src.baseline as baseline_mod
from src.core import get_grader, apply_graders  # noqa: F401  apply_graders 支持 "a+b" 组合

# 评委与被测必须来自不同厂商。启动即校验，不让「同源自评」的分数流进台账。
try:
    ROUTER = set_router(load_router())
    ROUTER.assert_startable()
except SiblingModelError as _e:
    ROUTER = set_router(load_router())
    print(f"⚠️  模型独立性校验未通过：{_e}")
    print("   评分类环节会被拒绝执行，直到在「模型角色」页改成不同厂商的模型。")

UPLOADS = ROOT / "data" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
SECRET = ROOT / "data" / "secret.json"
JOBS: dict[str, dict] = {}

app = FastAPI(title="LLM 评测平台")
from src.product_ui_routes import router as product_ui_router
app.include_router(product_ui_router)


# ─────────────────────────── 存储 ───────────────────────────
def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS dataset(
      id TEXT PRIMARY KEY, product TEXT, stage TEXT, name TEXT, version TEXT,
      n_cases INT, path TEXT, created_at TEXT);
    CREATE TABLE IF NOT EXISTS eval_version(
      id TEXT PRIMARY KEY, product TEXT, stage TEXT, label TEXT,
      prompt_ver TEXT, product_ver TEXT, change_note TEXT,
      dataset_id TEXT, run_id TEXT, verdict TEXT, created_at TEXT);
    CREATE TABLE IF NOT EXISTS prompt_tpl(
      id TEXT PRIMARY KEY, product TEXT, stage TEXT, version TEXT,
      template TEXT, note TEXT, created_at TEXT);
    """)
    # 两侧各自的备注：prompt 侧改了什么 / 产品侧改了什么。自动采集拿不到这些，只能人填。
    for col in ("prompt_note", "product_note"):
        try:
            c.execute(f"ALTER TABLE eval_version ADD COLUMN {col} TEXT")
        except Exception:
            pass
    # 数据集复核状态：没复核过的题不该进正式台账。
    # lint 通过（格式没问题）不等于有人读过（内容对不对）——这两件事必须分开记。
    for col, typ in (("reviewed", "TEXT DEFAULT ''"), ("review_note", "TEXT DEFAULT ''")):
        try:
            c.execute(f"ALTER TABLE dataset ADD COLUMN {col} {typ}")
        except Exception:
            pass
    c.commit()
    return c


def rows(cur) -> list[dict]:
    return [dict(r) for r in cur]


def load_secret() -> dict:
    if SECRET.exists():
        try:
            return json.loads(SECRET.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def product_repo_setting():
    c = init_db()
    c.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = c.execute("SELECT value FROM app_settings WHERE key='product_repo'").fetchone()
    if row is not None:
        return row[0]
    return load_secret().get("product_repo") or ""


def validate_request_url(body):
    try:
        validate_base_url(body.get("base_url", ""))
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


# ─────────────────────────── 页面 ───────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


# ─────────────────────────── 元数据 ───────────────────────────
@app.get("/api/products")
def api_products(include_builtin: bool = False):
    out = []
    for k, cfg in load_products().items():
        if cfg.get("_source") == "builtin" and not include_builtin:
            continue
        out.append({
            "product": k, "name": cfg["name"], "source": cfg.get("_source", "db"),
            "stages": [{"id": s["id"], "name": s["name"], "task_type": s["task_type"],
                        "gates": s.get("gates", []), "fatal": s.get("fatal", [])}
                       for s in cfg["stages"]],
        })
    return out


@app.post("/api/examples/general")
def install_general_example():
    """按需创建通用演示，不覆盖任何现有产品。"""
    example = {
        "product": "general-example", "name": "通用问答示例", "version": "1.0",
        "_example": True,
        "stages": [{"id": "answer", "name": "问答", "task_type": "text",
                    "dataset": "data/general_example.json", "grader": "keyword_rule",
                    "prompt": "{question}",
                    "counters": [], "derived": [], "gates": [], "fatal": []}],
    }
    c = init_db()
    try:
        c.execute("CREATE TABLE IF NOT EXISTS product_doc(product TEXT PRIMARY KEY, doc TEXT, created_at TEXT)")
        if c.execute("SELECT 1 FROM product_doc WHERE product=?", (example["product"],)).fetchone():
            raise HTTPException(409, "该产品 ID 已存在，未修改任何数据")
        c.execute("INSERT INTO product_doc(product,doc,created_at) VALUES(?,?,?)",
                  (example["product"], json.dumps(example, ensure_ascii=False), time.strftime("%Y-%m-%d %H:%M:%S")))
        c.commit()
    finally:
        c.close()
    return {"product": example["product"], "source": "example"}


# ─────────────────────────── 评测集 ───────────────────────────
# ────────────────── 通用接入：评分器目录 / 评测计划 / 生成数据集 ──────────────────
# 评分器目录：只登记「通用」评分器（src/graders_rule.py + src/graders_llm.py）。
#
# 三条纪律：
# 1. 这里的 key 必须等于 src/core.py REGISTRY 里的注册名 —— 名字对不上，
#    前端选了也跑不起来，这是最贵的 bug（跑完 30 条才发现）。
# 2. metrics 只写评分器真会产出的指标 key，写错的门永远显示「—」。
# 3. 计数型指标（tp/fp/fn/tn）必须同时写进 counters：
#    它们要汇总求和再算比率，混进平均里会把 precision 固定成 0.5。
# 旧业务评分器（advise_quality / trigger_matrix / extract_quality / chat_quality /
# layout_validity）已不再对外提供：它们绑死在某个具体产品上，不是通用能力。
GRADER_INFO = {
    "keyword_rule": {
        "label": "关键词合规", "task_type": "text", "expect": "keyword",
        "desc": "必须说什么、不能说什么：命中率、覆盖率、禁用词违规数。",
        "metrics": [("mention_ok", "必需内容命中", ">="), ("coverage", "关键词覆盖率", ">="),
                    ("forbidden_hits", "禁用词命中数", "<=")],
        "counters": [], "derived": [],
    },
    "format_rule": {
        "label": "格式结构", "task_type": "text", "expect": "format",
        "desc": "输出是否是合法结构：JSON 合法、必填字段齐、枚举不越界、长度不超标。",
        "metrics": [("schema_ok", "结构合规", ">="), ("field_ok", "字段完整", ">="),
                    ("enum_ok", "枚举合规", ">="), ("len_ok", "长度合规", ">="),
                    ("missing", "缺失字段数", "<=")],
        "counters": [], "derived": [],
    },
    "citation_rule": {
        "label": "引用溯源", "task_type": "text", "expect": "citation",
        "desc": "引用是否真实存在于给定资料：有没有引用、有没有编造引用。",
        "metrics": [("has_citation", "是否给出引用", ">="), ("phantom_refs", "无效引用数", "<=")],
        "counters": [], "derived": [],
    },
    "label_match": {
        "label": "分类判定", "task_type": "text", "expect": "label",
        "desc": "二分类：把每条输出归入混淆矩阵（tp/fp/fn/tn），集合级算准确率、精确率、召回率、F1。",
        "metrics": [("accuracy", "准确率", ">="), ("precision", "精确率", ">="),
                    ("recall", "召回率", ">="), ("f1", "F1", ">="),
                    ("tp", "正例判对数", ">="), ("fp", "误报数", "<="),
                    ("fn", "漏报数", "<="), ("tn", "负例判对数", ">=")],
        "counters": ["tp", "fp", "fn", "tn"],
        "derived": [
            {"key": "accuracy", "label": "准确率", "formula": "(tp+tn)/(tp+fp+fn+tn)"},
            {"key": "precision", "label": "精确率", "formula": "tp/(tp+fp)"},
            {"key": "recall", "label": "召回率", "formula": "tp/(tp+fn)"},
            {"key": "f1", "label": "F1", "formula": "2*tp/(2*tp+fp+fn)"},
        ],
    },
    "outcome_rule": {
        "label": "终态校验", "task_type": "agent", "expect": "state",
        "desc": "任务结束后真实状态对不对：读结构化终态，不听自然语言里的「已完成」。",
        "metrics": [("state_ok", "终态一致", ">="), ("state_match", "终态字段匹配率", ">=")],
        "counters": [], "derived": [],
    },
    "retrieval_rule": {
        "label": "检索层", "task_type": "rag", "expect": "retrieval",
        "desc": "检索质量单独评：Hit@k、Recall@k、Precision@k、MRR，不与会生成混算。",
        "metrics": [("hit_at_k", "Hit@k", ">="), ("recall_at_k", "Recall@k", ">="),
                    ("precision_at_k", "Precision@k", ">="), ("mrr", "MRR", ">="),
                    ("has_label", "有相关标注", ">="), ("k", "k 值", "<=")],
        "counters": [], "derived": [],
    },
    "efficiency_rule": {
        "label": "效率成本", "task_type": "agent", "expect": "budget",
        "desc": "token、延迟、步骤是否超这条题自己的预算（不做跨产品横比）。",
        "metrics": [("tokens_ok", "token 在预算内", ">="), ("latency_ok", "延迟在预算内", ">="),
                    ("steps_ok", "步骤数在预算内", ">="), ("tokens", "token 数", "<="),
                    ("latency_ms", "延迟(ms)", "<=")],
        "counters": [], "derived": [],
    },
    "tool_call_rule": {
        "label": "工具调用（规则）", "task_type": "agent", "expect": "tool",
        "desc": "工具调用的硬事实：调没调、是否越白名单、参数键齐不齐、次数超没超。",
        "metrics": [("calls_ok", "调用次数合规", ">="), ("whitelist_ok", "工具白名单合规", ">="),
                    ("required_ok", "必调工具覆盖", ">="), ("args_ok", "参数完整", ">="),
                    ("n_calls", "调用次数", "<=")],
        "counters": [], "derived": [],
    },
    "trajectory_rule": {
        "label": "轨迹对齐（规则）", "task_type": "agent", "expect": "trajectory",
        "desc": "关键步骤是否按序出现（LCS 对齐）：路径命中、缺步、多余步骤。",
        "metrics": [("path_ok", "关键路径完整", ">="), ("path_recall", "路径召回率", ">="),
                    ("has_trace", "有轨迹", ">="), ("missing_steps", "缺失步骤数", "<="),
                    ("extra_steps", "多余步骤数", "<=")],
        "counters": [], "derived": [],
    },
    "llm_rubric": {
        "label": "判定表评分（LLM）", "task_type": "text", "expect": "rubric",
        "desc": "按判定表逐维度挑档：总分、否决项、评委置信度，标准来自 rubric 文件。",
        "metrics": [("rubric_score", "判定表总分", ">="), ("gate_fail", "否决项未达标", "<="),
                    ("judge_confidence", "评委置信度", ">=")],
        "counters": [], "derived": [],
    },
    "rag_judge": {
        "label": "RAG 生成层（LLM）", "task_type": "rag", "expect": "rubric_rag",
        "desc": "生成层判官：忠实度、相关性、上下文利用、抗噪（检索层归检索评分器）。",
        "metrics": [("rubric_score", "判定表总分", ">="), ("gate_fail", "否决项未达标", "<="),
                    ("judge_confidence", "评委置信度", ">=")],
        "counters": [], "derived": [],
    },
    "tool_call_judge": {
        "label": "工具调用质量（LLM）", "task_type": "agent", "expect": "rubric_tool",
        "desc": "工具调用里判不了的软理由：选得对不对口、参数语义、结果有没有用上。",
        "metrics": [("rubric_score", "判定表总分", ">="), ("gate_fail", "否决项未达标", "<="),
                    ("judge_confidence", "评委置信度", ">=")],
        "counters": [], "derived": [],
    },
    "trajectory_judge": {
        "label": "轨迹过程质量（LLM）", "task_type": "agent", "expect": "rubric_trajectory",
        "desc": "只看过程不看结果：缺步、冗余、失败处理、声称与过程是否一致。",
        "metrics": [("rubric_score", "判定表总分", ">="), ("gate_fail", "否决项未达标", "<="),
                    ("judge_confidence", "评委置信度", ">=")],
        "counters": [], "derived": [],
    },
    "pair_compare": {
        "label": "成对比较（LLM）", "task_type": "text", "expect": "pair",
        "desc": "A/B 谁更好，交换顺序各判一次；两次不一致判无效，避免位置偏差。",
        "metrics": [("pair_consistent", "结论一致", ">="), ("has_pair", "有 A/B 候选", ">="),
                    ("a_wins", "A 胜", ">="), ("b_wins", "B 胜", ">="), ("tie", "平局", ">=")],
        "counters": [], "derived": [],
    },
}
# 别名机制（目前没有别名，留着以备将来重命名评分器时老计划还能跑）
GRADER_ALIAS: dict[str, str] = {}
# 已下线的旧业务评分器：老计划 / 老文档里还会出现，报错要说人话，
# 只回「评分器不存在」会让人以为是打错字，而不是「这个能力已经换成通用评分器了」。
DEPRECATED_GRADERS = {
    "advise_quality": "已下线。二分类（该不该劝）请用 label_match，话术检查请用 keyword_rule 或 llm_rubric",
    "trigger_matrix": "已下线。二分类（该不该提醒）请用 label_match，文案检查请用 keyword_rule",
    "chat_quality": "已下线。多轮话术请用 keyword_rule 或 llm_rubric",
    "extract_quality": "已下线。结构化输出校验请用 format_rule",
    "extract_accuracy": "已下线。结构化输出校验请用 format_rule",
    "layout_validity": "已下线。结构化产物校验请用 format_rule",
}
# 目录索引：环节没显式写 counters / derived 时，从评分器定义里取默认值。
# 缺了这段，tp/fp 会被当普通指标平均，precision 恒 0.5 —— 数字看着正常，其实是错的。
COUNTERS_BY_GRADER = {k: list(v.get("counters") or []) for k, v in GRADER_INFO.items()}
DERIVED_BY_GRADER = {k: [dict(d) for d in (v.get("derived") or [])] for k, v in GRADER_INFO.items()}


def _split_grader_spec(spec) -> list[str]:
    """把 "a+b" / "a,b" 拆成注册名列表（已做别名对齐）。"""
    out = []
    for p in str(spec or "").replace(",", "+").split("+"):
        p = p.strip()
        if p:
            out.append(GRADER_ALIAS.get(p, p))
    return out


def _catalog_counters(reals: list[str]) -> tuple[list, list]:
    """组合评分器时把各自的 counters / derived 并起来（按声明顺序去重）。"""
    counters, derived, seen = [], [], set()
    for r in reals:
        for c in COUNTERS_BY_GRADER.get(r, []):
            if c not in seen:
                seen.add(c)
                counters.append(c)
        for d in DERIVED_BY_GRADER.get(r, []):
            if d["key"] not in seen:
                seen.add(d["key"])
                derived.append(dict(d))
    return counters, derived


def _validate_grader_spec(value) -> str:
    """校验并归一化评分器声明；不存在直接 400，不替用户挑。"""
    from src.core import REGISTRY
    names = _split_grader_spec(value)
    for n in names:
        if n in DEPRECATED_GRADERS:
            raise HTTPException(400, f"评分器 {n} {DEPRECATED_GRADERS[n]}")
    if not names or any(n not in REGISTRY for n in names):
        raise HTTPException(400, f"评分器不存在。可选：{sorted(REGISTRY)}")
    return "+".join(names)


def _grader_fn(spec: str):
    """把评分器声明（含 "a+b" 组合）包成一个 (case, output, stage) -> result 的函数。

    以前这里直接 get_grader(spec)：环节挂组合评分器时 KeyError → HTTP 500，
    而组合恰恰是这个平台的常规用法（规则判事实 + 判官判合理性）。
    """
    spec = str(spec or "")

    def run(case, output, stage):
        return apply_graders(spec, case, output, stage)

    return run


def _stage_eval_extras(sc: dict) -> tuple[list, list]:
    """环节的 counters / derived：自己写了就用，没写就按评分器目录补。

    补这一步是为了「老产品」也能算对：产品配置落库时如果没存 counters，
    tp/fp 会被当普通指标平均，precision 恒 0.5。
    """
    counters = sc.get("counters") or []
    derived = sc.get("derived") or []
    if counters and derived:
        return counters, derived
    auto_counters, auto_derived = _catalog_counters(_split_grader_spec(sc.get("grader")))
    return counters or auto_counters, derived or auto_derived


@app.get("/api/graders")
def api_graders():
    """可选评分器目录。

    遍历注册表而不是遍历目录：目录是给人看的说明，注册表才是「真能跑的东西」。
    只要一个评分器注册了，它就一定要能被选到 —— 反过来（目录里列了但没注册）
    只会让用户选到一个跑不起来的名字。
    """
    from src.core import REGISTRY
    out = []
    for real in sorted(REGISTRY):
        info = GRADER_INFO.get(real)
        if info is None:
            # 兜底：注册了但没写说明的评分器，用注册名占位，绝不隐藏。
            info = {"label": real, "desc": f"已注册评分器 {real}（尚未登记中文说明）",
                    "metrics": [], "counters": [], "derived": []}
        item = {"key": real, "real": real, **info}
        item.setdefault("counters", COUNTERS_BY_GRADER.get(real, []))
        item.setdefault("derived", DERIVED_BY_GRADER.get(real, []))
        out.append(item)
    return out


# ────────────────── 模型角色 / Rubric：多模型互评的两块基础设施 ──────────────────
@app.get("/api/models")
def api_models():
    """四个角色 + 独立性体检。Key 只回「配没配」，永不回显明文。"""
    r = load_router()
    roles = []
    for k, s in r.specs.items():
        roles.append({
            "role": k, "name": s.name, "adapter": s.adapter,
            "base_url": s.base_url, "model": s.model, "vendor": s.vendor,
            "key_env": s.key_env, "key_ready": False,
            "note": s.note,
        })
    return {"roles": roles, "strict_vendor": r.strict_vendor,
            "independence": [x.as_dict() for x in r.independence()]}


@app.post("/api/models")
def api_set_models(body: dict = Body(...)):
    """保存角色配置。保存前先跑一次同源校验 —— 配错了当场拒，不要等跑完才发现。"""
    raw = config_raw()
    raw.setdefault("roles", {})
    incoming = body.get("roles") or {}
    for role, cfg in incoming.items():
        cur = raw["roles"].get(role) or {}
        cur.update({k: v for k, v in cfg.items() if k in
                    ("name", "adapter", "base_url", "model", "key_env", "vendor", "note")})
        try:
            validate_base_url(cur.get("base_url", ""))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        raw["roles"][role] = cur
    if "strict_vendor" in body:
        raw["strict_vendor"] = bool(body["strict_vendor"])

    specs = {k: ModelSpec(key=k, **v) for k, v in raw["roles"].items()}
    from src.models import ModelRouter
    probe = ModelRouter(specs, strict_vendor=bool(raw.get("strict_vendor")))
    bad = [x for x in probe.independence() if x.level == "block"]
    if bad and body.get("force") is not True:
        raise HTTPException(400, "同源身份冲突：" + "；".join(x.reason for x in bad))

    save_config(raw)
    return {"ok": True, "independence": [x.as_dict() for x in probe.independence()]}


@app.get("/api/rubrics")
def api_rubrics():
    """可用判定表目录。rubric 是数据不是代码——改标准不用改评分器。"""
    return rubric_mod.catalog()


@app.get("/api/rubrics/{name}")
def api_rubric_detail(name: str):
    try:
        r = rubric_mod.load(name)
    except FileNotFoundError:
        raise HTTPException(404, f"没有这份 rubric：{name}")
    demo_case = {"input": "（示例）用户问：上海出差 3 晚住宿 1840 元能报多少"}
    demo_out = "（示例）已核对标准后给出金额"
    return {
        "id": r.id, "label": r.label, "version": r.version, "threshold": r.threshold,
        "applies_to": r.applies_to, "note": r.note,
        "dimensions": [{"key": d.key, "label": d.label, "weight": d.weight,
                        "gate": d.gate, "evidence": d.evidence,
                        "levels": {str(k): v for k, v in d.levels.items()}}
                       for d in r.dimensions],
        "prompt_preview": r.render(demo_case, demo_out),
    }


def _validate_plan(doc: dict) -> dict:
    """把一份评测计划校验成能直接落库的产品配置。返回 (cfg, warnings)。"""
    from src.core import REGISTRY
    warns = []
    if not (doc.get("product") or "").strip():
        raise HTTPException(400, "评测计划缺少 product（产品英文 id，如 resume）")
    stages_in = doc.get("stages") or []
    if not stages_in:
        raise HTTPException(400, "评测计划至少要有一个环节（stages）")

    stages = []
    for i, st in enumerate(stages_in):
        if not (st.get("id") or "").strip():
            raise HTTPException(400, f"第 {i+1} 个环节缺少 id")
        gkey = st.get("grader") or ""
        # 评分器可以组合：规则判事实、判官判合理性，一个环节挂多个用 + 连起来
        parts = [p.strip() for p in str(gkey).replace(",", "+").split("+") if p.strip()]
        if not parts:
            raise HTTPException(400, f"环节 {st['id']} 没有指定评分器")
        reals = []
        for p in parts:
            r = GRADER_ALIAS.get(p, p)
            if r in DEPRECATED_GRADERS:
                raise HTTPException(400, f"环节 {st['id']} 的评分器 {p!r} {DEPRECATED_GRADERS[r]}")
            if r not in REGISTRY:
                raise HTTPException(400, f"环节 {st['id']} 的评分器 {p!r} 不存在。"
                                         f"可选：{sorted(REGISTRY)}")
            reals.append(r)
        real = "+".join(reals)
        info = GRADER_INFO.get(reals[0], {})
        gates = st.get("gates") or []
        if not gates:
            # 没写门槛：给一套默认（照评分器能产出的指标），并警告
            gates = [{"key": m[0], "op": m[2], "value": 0.8 if m[2] == ">=" else 0.2,
                      "label": m[1]} for m in (info.get("metrics") or [])[:3]]
            warns.append(f"环节 {st['id']} 没有写达标门，已按「{info.get('label','')}」给了一套默认阈值，请确认。")
        for g in gates:
            if not g.get("key"):
                raise HTTPException(400, f"环节 {st['id']} 的达标门缺少 key")
            if g.get("op") not in (">=", "<="):
                raise HTTPException(400, f"环节 {st['id']} 的达标门 op 只能是 >= 或 <=")
            if not isinstance(g.get("value"), (int, float)):
                raise HTTPException(400, f"环节 {st['id']} 的达标门 value 必须是数字（0~1 的小数）")
        known = {m[0] for m in (info.get("metrics") or [])}
        known |= set(COUNTERS_BY_GRADER.get(reals[0], []))
        if known and len(reals) == 1:
            unknown = [g["key"] for g in gates if g["key"] not in known]
            if unknown:
                warns.append(f"环节 {st['id']} 的达标门里有评分器不产出的指标：{unknown}，"
                             f"这些门会永远是「—」。评分器 {gkey} 能产出：{sorted(known)}")
        fatal = st.get("fatal") or []
        if not fatal:
            warns.append(f"环节 {st['id']} 没有一票否决项。平均分掩盖不了「一条就毁掉信任」的错误，建议至少定一个。")
        # counters / derived 必须落库：丢了 counters，tp/fp 会被当普通指标做平均，
        # precision 恒 0.5；丢了 derived，accuracy/F1 根本不会算出来。
        auto_counters, auto_derived = _catalog_counters(reals)
        stages.append({
            "id": st["id"], "name": st.get("name") or st["id"],
            "task_type": st.get("task_type") or info.get("task_type") or "custom",
            "dataset": st.get("dataset") or "",
            "grader": real,
            "rubric": st.get("rubric") or "",
            "depends_on": st.get("depends_on") or [],
            "prompt": st.get("prompt") or "",
            "counters": st.get("counters") or auto_counters,
            "derived": st.get("derived") or auto_derived,
            "gates": gates, "fatal": fatal,
        })

    # 环节依赖必须指向真实存在的上游，且不能成环（A依赖B、B依赖A会永远跑不起来）
    ids = {s["id"] for s in stages}
    for s in stages:
        for up in s.get("depends_on") or []:
            if up not in ids:
                raise HTTPException(400, f"环节 {s['id']} 依赖的 {up!r} 不存在")
            if up == s["id"]:
                raise HTTPException(400, f"环节 {s['id']} 不能依赖自己")
    _check_cycle(stages)

    cfg = {"product": doc["product"].strip(), "name": doc.get("name") or doc["product"],
           "version": doc.get("version") or "1.0",
           "note": doc.get("note") or "", "stages": stages}
    return cfg, warns


def _check_cycle(stages: list) -> None:
    """拓扑检查：有环就报错，别等到跑批时才死循环。"""
    dep = {s["id"]: list(s.get("depends_on") or []) for s in stages}
    state = {}

    def walk(n, path):
        if state.get(n) == 1:
            raise HTTPException(400, f"环节依赖成环：{' → '.join(path + [n])}")
        if state.get(n) == 2:
            return
        state[n] = 1
        for m in dep.get(n, []):
            walk(m, path + [n])
        state[n] = 2

    for n in dep:
        walk(n, [])


@app.post("/api/products/import")
def api_import_plan(body: dict = Body(...)):
    """导入一份评测计划（JSON），自动识别环节并落库。

    markdown 也可以：传 {"markdown": "..."}，会用模型抽取成 JSON（需要配 API Key）。
    落库前只做结构校验，不做「猜」——评分器不存在就直接报错，不替你挑。
    """
    md = body.get("markdown")
    if md:
        plan = _plan_from_markdown(md, body.get("model_api_keys"))
    else:
        plan = body.get("plan")
        if not isinstance(plan, dict):
            raise HTTPException(400, "需要 plan（JSON 对象）或 markdown（文本）")
    cfg, warns = _validate_plan(plan)
    if body.get("dry_run"):
        return {"ok": True, "dry_run": True, "product": cfg["product"], "name": cfg["name"],
                "stages": [{"id": s["id"], "name": s["name"], "grader": s["grader"],
                            "task_type": s["task_type"], "n_gates": len(s["gates"]),
                            "n_fatal": len(s["fatal"]), "depends_on": s.get("depends_on") or [],
                            "gates": s["gates"]} for s in cfg["stages"]],
                "warnings": warns}
    try:
        save_product_doc(cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 补上老产品早就有的「prompt 版本」这一环。
    # 新产品在计划里写的模板如果只是躺在 config 里，改一版就没有版本记录，
    # 后面台账上的「P=只改prompt」就无从归因。
    tpls = []
    con = conn()
    for s in cfg["stages"]:
        tpl = s.get("prompt") or s.get("prompt_tpl") or ""
        if not tpl:
            continue
        exist = con.execute("SELECT id FROM prompt_tpl WHERE product=? AND stage=? LIMIT 1",
                            (cfg["product"], s["id"])).fetchone()
        if exist:
            continue
        pid = uuid.uuid4().hex[:10]
        con.execute("INSERT INTO prompt_tpl VALUES(?,?,?,?,?,?,?)",
                    (pid, cfg["product"], s["id"], "v1", tpl, "导入计划时创建",
                     time.strftime("%Y-%m-%d %H:%M:%S")))
        tpls.append({"stage": s["id"], "id": pid, "version": "v1"})
    con.commit()

    return {"ok": True, "product": cfg["product"], "name": cfg["name"],
            "stages": [{"id": s["id"], "name": s["name"], "grader": s["grader"],
                        "n_gates": len(s["gates"]), "n_fatal": len(s["fatal"]),
                        "depends_on": s.get("depends_on") or []} for s in cfg["stages"]],
            "prompts_created": tpls, "warnings": warns}


PLAN_SCHEMA_HINT = """输出一个 JSON 对象，结构如下（不要解释、不要代码块标记）：
{
  "product": "英文短 id，如 resume",
  "name": "产品中文名",
  "note": "一句话说明这个产品干什么",
  "stages": [
    {
      "id": "英文短 id，如 parse",
      "name": "环节中文名",
      "task_type": "text / agent / rag / custom 之一",
      "grader": "从下面清单里选一个（多个可用 + 组合）",
      "prompt": "发给模型的完整 prompt 模板，可用 {input} 占位",
      "gates": [{"key": "指标名", "op": ">=", "value": 0.85, "label": "中文名"}],
      "fatal": ["F1"],
      "n_cases": 20,
      "case_hint": "这类题应该长什么样，一句话"
    }
  ]
}
评分器与指标的对应关系（只能用这些 key）：
""" + "\n".join(f"- {k}: {v['label']} → {[m[0] for m in v['metrics']]}" for k, v in GRADER_INFO.items())


# markdown 里出现这些词，就大致能判断环节类型。
# 规则版解析的价值不在于准，而在于「没配 Key 也能跑通一次」——
# 否则这个入口永远停在「未实测」状态，谁也不知道它能不能用。
PLAN_KEYWORDS = {
    "extract": ("抽取", "解析", "识别", "提取", "结构化", "字段"),
    "advise": ("建议", "推荐", "决策", "咨询"),
    "trigger": ("提醒", "触发", "推送", "通知"),
    "chat": ("对话", "多轮", "追问", "客服", "聊天"),
    "generate_layout": ("布局", "图表", "看板", "报表", "可视化", "仪表盘"),
}
STAGE_DEFAULT_GRADER = {
    # markdown 规则解析只认类型，落到通用评分器上：
    # 结构化抽取 → 结构校验；建议/提醒/对话 → 关键词合规；布局图表 → 结构校验。
    "extract": "format_rule", "advise": "keyword_rule", "trigger": "keyword_rule",
    "chat": "keyword_rule", "generate_layout": "format_rule",
    "text": "keyword_rule", "agent": "tool_call_rule+trajectory_rule", "rag": "retrieval_rule",
}


def _rule_plan(md: str) -> dict:
    """规则版：从 markdown 的编号/列表里抽环节。不需要模型。"""
    import re
    stages, seen = [], set()
    for ln in md.splitlines():
        m = (re.match(r"^\s{0,3}(?:#{1,5}\s*)?(\d{1,2})[.、)．]\s*(.{2,40})$", ln)
             or re.match(r"^\s{0,3}[-*+]\s*(.{2,40})$", ln))
        if not m:
            continue
        name = (m.group(2) if m.lastindex == 2 else m.group(1)).strip(" 　:：。")
        if not name or len(name) < 2 or name in seen:
            continue
        low = name.lower()
        task = next((t for t, kws in PLAN_KEYWORDS.items() if any(k in name for k in kws)), None)
        if not task:
            continue  # 认不出类型的行不硬塞，避免把正文段落当环节
        seen.add(name)
        sid = f"stage{len(stages) + 1}"
        stages.append({
            "id": sid, "name": name[:24], "task_type": task,
            "grader": STAGE_DEFAULT_GRADER[task],
            "prompt": f"请完成「{name}」任务，输入如下：{{input}}",
            "gates": [], "fatal": [], "n_cases": 10,
        })
    if not stages:
        raise ValueError("没能从这份 markdown 里认出任何评测环节。"
                         "请用「1. 抽取字段」这样的编号行写环节，或直接导入 JSON。")
    title = next((ln.strip("# ").strip() for ln in md.splitlines() if ln.strip()), "未命名产品")
    return {"product": f"plan{int(time.time()) % 10000}", "name": title[:30],
            "note": "由规则解析器从 markdown 提取（未使用模型）", "stages": stages}


def _plan_from_markdown(md: str, model_api_keys=None) -> dict:
    """让模型把一份人写的评测计划读成结构化 JSON。

    走网关的 generator 角色，不走被测模型 —— 造题的模型若和被测同源，
    等于自己出题自己考。模型不可用时回退到规则解析器，保证这个入口永远可用。
    """
    from src.models import load_router
    try:
        router = load_router()
        router.api_keys = dict(model_api_keys or {})
        router.assert_startable()
        spec = router.role_of("generator")
        if spec.is_mock:
            return _rule_plan(md)
        prompt = ("""下面是一份 AI 产品的评测计划。请把它转成配置。

要求：
1. 逐个识别「评测环节」，每个环节对应一个评分器
2. 评分器只能从给定清单里选，不要发明
3. 达标门（gates）的 key 必须是该评分器能产出的指标
4. 原文没说阈值的地方，按你对该指标的常识给一个合理值，但不要编造原文没有的环节

""" + PLAN_SCHEMA_HINT + "\n\n【评测计划原文】\n" + md[:8000])
        token = set_request_context(model_api_keys=model_api_keys)
        try:
            txt = (router.call("generator", prompt)["text"] or "").strip()
        finally:
            reset_request_context(token)
        if txt.startswith("```"):
            txt = txt.split("```")[1]
            txt = txt.split("\n", 1)[1] if "\n" in txt else txt
        return json.loads(txt)
    except HTTPException:
        raise
    except Exception as e:
        try:
            plan = _rule_plan(md)
            plan["note"] = "模型解析失败，已回退规则解析"
            return plan
        except ValueError:
            raise HTTPException(400, "模型与规则都无法解析这份计划")


@app.get("/api/datasets")
def api_datasets():
    c = conn()
    return rows(c.execute("SELECT * FROM dataset ORDER BY created_at DESC"))


@app.post("/api/datasets")
async def api_upload_dataset(file: UploadFile = File(...), product: str = "", stage: str = ""):
    """支持 JSON（本项目格式）与 CSV（需表头 id,input,gold）。上传即校验。"""
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    name = file.filename or "upload"

    if name.lower().endswith(".csv"):
        rd = csv.DictReader(io.StringIO(raw))
        if not rd.fieldnames or not {"id", "input", "gold"}.issubset(rd.fieldnames):
            raise HTTPException(400, "CSV 必须包含 id、input、gold 表头")
        cases = []
        for i, r in enumerate(rd):
            try:
                inp = json.loads(r.get("input") or "{}")
                gold = json.loads(r.get("gold") or "{}")
            except Exception as e:
                raise HTTPException(400, f"第 {i+2} 行 JSON 解析失败：{e}")
            if not r.get("input"):
                raise HTTPException(400, f"第 {i+2} 行缺少 input")
            cases.append({"id": r.get("id") or f"c{i+1}", "stratum": r.get("stratum", ""),
                          "input": inp, "gold": gold})
        doc = {"dataset": name, "version": "1.0", "n_cases": len(cases), "cases": cases}
    else:
        try:
            doc = json.loads(raw)
        except Exception as e:
            raise HTTPException(400, f"JSON 解析失败：{e}")
        if not isinstance(doc, dict):
            raise HTTPException(400, "数据集顶层必须是 JSON 对象")
        cases = doc.get("cases")

    if not isinstance(cases, list) or not cases:
        raise HTTPException(400, "cases 必须是非空数组")

    problems = []
    seen_ids = set()
    for i, cs in enumerate(cases):
        if not isinstance(cs, dict):
            raise HTTPException(400, f"第 {i+1} 条用例必须是对象")
        case_id = cs.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise HTTPException(400, f"第 {i+1} 条缺少有效的字符串 id")
        if case_id in seen_ids:
            raise HTTPException(400, f"第 {i+1} 条用例 id 重复：{case_id}")
        seen_ids.add(case_id)
        if "input" not in cs or cs["input"] is None:
            raise HTTPException(400, f"第 {i+1} 条（{case_id}）缺少 input")
        if not cs.get("gold"):
            problems.append(f"第 {i+1} 条（{case_id}）缺少 gold 标注——没有标准答案就无法判分")

    did = uuid.uuid4().hex[:10]
    path = UPLOADS / f"{did}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    c = conn()
    c.execute("INSERT INTO dataset(id,product,stage,name,version,n_cases,path,created_at) VALUES(?,?,?,?,?,?,?,?)",
              (did, product, stage, doc.get("dataset") or name, doc.get("version", "1.0"),
               len(cases), str(path), time.strftime("%Y-%m-%d %H:%M:%S")))
    c.commit()
    return {"id": did, "n_cases": len(cases), "problems": problems[:20],
            "n_problems": len(problems), "strata": doc.get("strata", {})}


@app.post("/api/datasets/{did}/review")
def api_dataset_review(did: str, body: dict = Body(...)):
    """标记「这套题有人读过」。

    为什么必须单独记这件事：
      lint 通过只说明格式没问题，不说明内容对；门禁如果建立在没人读过的题上，
      测出来的分数就是「AI 自己出的题 + AI 自己判分」，看着漂亮但没有意义。
    所以复核状态与 lint 结果分开存，并且没复核过的数据集不允许直接进正式台账。
    """
    c = conn()
    row = c.execute("SELECT * FROM dataset WHERE id=?", (did,)).fetchone()
    if not row:
        raise HTTPException(404, "数据集不存在")
    ok = bool(body.get("reviewed", True))
    note = (body.get("note") or "").strip()
    if not ok and not note:
        raise HTTPException(400, "标记为未通过时必须写一句原因，否则改题的人不知道该改什么")
    c.execute("UPDATE dataset SET reviewed=?, review_note=? WHERE id=?",
              ("yes" if ok else "no", note, did))
    c.commit()
    return {"ok": True, "id": did, "reviewed": "yes" if ok else "no", "note": note}


@app.get("/api/datasets/queue")
def api_dataset_queue():
    """待复核队列：造出来就没人管过的题都在这。"""
    c = conn()
    pend = rows(c.execute("SELECT id,product,stage,name,n_cases,created_at FROM dataset "
                          "WHERE COALESCE(reviewed,'')='' ORDER BY created_at DESC"))
    done = rows(c.execute("SELECT id,product,stage,name,n_cases,created_at,reviewed,review_note "
                          "FROM dataset WHERE reviewed='yes' ORDER BY created_at DESC LIMIT 50"))
    bad = rows(c.execute("SELECT id,product,stage,name,review_note,created_at FROM dataset "
                         "WHERE reviewed='no' ORDER BY created_at DESC"))
    return {"pending": pend, "approved": done, "rejected": bad}


@app.post("/api/datasets/verify")
def api_verify_dataset(body: dict = Body(...)):
    """体检一套题：参考答案必须全过、白卷基线不该过。

    对应 Anthropic「从零到一」的第 2、3 步。不跑模型，只跑判分器，几毫秒出结果。
    """
    product, stage_id = body.get("product"), body.get("stage")
    cfg = load_products().get(product)
    if not cfg:
        raise HTTPException(404, f"产品 {product} 不存在")
    sc = stage_of(cfg, stage_id)
    if not sc:
        raise HTTPException(404, f"环节 {stage_id} 不存在")

    path = resolve_dataset(body.get("dataset_id"), product, stage_id)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") or data
    if not isinstance(cases, list) or not cases:
        raise HTTPException(400, "评测集是空的")

    from src.core import StageConfig
    spec = _validate_grader_spec(sc["grader"])
    v_counters, v_derived = _stage_eval_extras({**sc, "grader": spec})
    st = StageConfig(id=sc["id"], name=sc["name"], task_type=sc["task_type"],
                     dataset=path, grader=spec, gates=sc.get("gates", []),
                     fatal=sc.get("fatal", []), counters=v_counters,
                     derived=v_derived)
    gf = _grader_fn(spec)
    validate_request_url(body)
    token = set_request_context(body.get("api_key", ""), body.get("model_api_keys"), base_url=body.get("base_url", ""))
    try:
        try:
            res = redact(baseline_mod.check_dataset(cases, st, gf))
        except ValueError as e:
            raise HTTPException(400, redact(str(e))) from None
    finally:
        reset_request_context(token)
    res["product"], res["stage"] = product, stage_id
    res["dataset"] = Path(path).name
    return res


GEN_SPEC = {
    "advise_quality": {
        "strata": "should_advise（该劝）/ should_not_advise（不该劝）/ boundary（边界陷阱）/ forbidden_bait（诱导踩线）",
        "gold": '{"should_advise": true/false, "reason_type": "超预算/重复购买/非必需", '
                '"must_mention": ["话术里必须出现的具体证据"], "forbidden": ["不许出现的词"], '
                '"fatal_risk": "F1|F2|F3|F5 或 null"}',
        "tip": "四种分层都要有，且 should_not_advise 不能全靠「沉默」过——给它配一个必须给出的确认反馈",
    },
    "trigger_matrix": {
        "strata": "should_trigger（该提醒）/ should_not_trigger（不该提醒）/ boundary（临界）",
        "gold": '{"should_trigger": true/false, "must_mention": ["提醒里必须出现的数字"], "reason": "一句话"}',
        "tip": "临界情况最有价值：刚过线/刚没过线、今天已提醒过、深夜",
    },
    "extract_accuracy": {
        "strata": "clean（干净输入）/ noisy（有噪声）/ ambiguous（该标不确定）",
        "gold": '{"entries": [{"字段": "值"}], "no_hallucination": true, "must_flag_uncertain": false, "note": ""}',
        "tip": "至少一半要带噪声：模糊字、缺失字段、手写体识别错",
    },
    "chat_quality": {
        "strata": "not_used（用户说没怎么用）/ refuse（用户明确拒绝）/ normal（正常使用）",
        "gold": '{"must_not": ["不许说的话"], "should_offer": ["低压力下一步"], "should_stop": true/false}',
        "tip": "refuse 那类必须 should_stop=true，用来测「拒绝后会不会停」",
    },
    "layout_validity": {
        "strata": "single_intent（单一意图）/ multi_intent（多意图）/ ambiguous（模糊需求）",
        "gold": '{"expect": [{"priority": 1, "type": ["line"], "x": "字段", "y": "字段"}], '
                '"forbidden_types": [{"type": "pie", "reason": "为什么不行"}], "must_mention": []}',
        "tip": "要有明确「不该画某类图」的反例，否则测不出判断力",
    },
}

# 通用评分器没有专属配题说明时用这份兜底 —— 否则生成 prompt 里
# 「gold 怎么写」「要覆盖哪几种分层」会变成空字符串，模型就会瞎写标注。
GEN_SPEC_DEFAULT = {
    "strata": "典型（一眼能判）/ 边界（两种都说得通，但标准答案必须唯一）/ 刁钻（信息缺失或互相矛盾）",
    "gold": '{"label": "positive"/"negative"（分类任务）或 {"expect": {...}} 形式的可机检要求，'
            '写具体值、具体词，不写「表达得体」这类没法判的话}',
    "tip": "gold 里出现的每个字段都要能被某个评分器检查到；检查不到就别写进去。",
}


def _gen_prompt(stage_cfg, grader, n, input_example, instruction):
    spec = GEN_SPEC.get(grader) or GEN_SPEC_DEFAULT
    ex = json.dumps(input_example, ensure_ascii=False, indent=1) if input_example else "（未提供，请按环节 prompt 里的输入描述推断结构）"
    return f"""你在帮人构建一套 AI 产品的评测集。请生成 {n} 条评测用例。

【这一步要测什么】
环节：{stage_cfg.get('name') or stage_cfg.get('id')}
发给模型的 prompt：
{(stage_cfg.get('prompt') or '（未提供）')[:1200]}

【用例的输入结构】
{ex}

【每条用例的 gold（标准答案）写法】
{spec.get('gold','')}

【分层要求】
必须覆盖：{spec.get('strata','')}
{spec.get('tip','')}

【质量要求】
1. gold 必须是「可机检」的：写具体数字、具体词，不要写「表达得体」这种没法判的
2. 输入要像真实用户的数据，不要写得比真实数据干净
3. 至少 1/3 的用例是刁钻情况（信息缺失、互相矛盾、用户情绪化）
4. id 用 {stage_cfg.get('id','X').upper()[:2]}01 起顺序编号
5. 不要凑数：宁可少几条，也不要两条几乎一样

{('【额外要求】' + instruction) if instruction else ''}

【输出】
只输出一个 JSON 数组，每个元素形如：
{{"id": "...", "stratum": "...", "input": {{...}}, "gold": {{...}}}}
不要解释，不要 markdown 代码块标记。"""


def _lint_cases(cases, grader):
    """生成完先挑毛病。比生成本身重要——AI 写的 gold 经常自相矛盾。"""
    bad = []
    strata = {}
    for c in cases:
        cid = c.get("id") or "?"
        st = c.get("stratum") or ""
        strata[st] = strata.get(st, 0) + 1
        if not c.get("input"):
            bad.append(f"{cid}: input 是空的")
        g = c.get("gold") or {}
        if not g:
            bad.append(f"{cid}: 没有 gold")
            continue
        if grader == "advise_quality":
            if g.get("should_advise") and not (g.get("must_mention") or []):
                bad.append(f"{cid}: 该劝但 must_mention 为空——判分器没法检查话术具体度")
            if not (g.get("forbidden") or []):
                bad.append(f"{cid}: forbidden 为空")
        elif grader == "trigger_matrix":
            if g.get("should_trigger") and not (g.get("must_mention") or []):
                bad.append(f"{cid}: 该提醒但 must_mention 为空")
        elif grader == "chat_quality":
            if not (g.get("must_not") or []):
                bad.append(f"{cid}: must_not 为空——测不出二次伤害")
        elif grader == "extract_accuracy":
            if not (g.get("entries") or []):
                bad.append(f"{cid}: entries 为空")
        elif grader == "label_match":
            # 分类题没有标注就永远算不出准确率，必须在生成阶段拦住
            if "label" not in g and "is_positive" not in g:
                bad.append(f"{cid}: 分类题缺少 label 标注（positive/negative 或 true/false）")
    if len(strata) < 2:
        bad.append(f"只覆盖了 {list(strata)} 一种分层，评测会失真")
    return bad, strata


@app.post("/api/datasets/generate")
def api_generate_dataset(body: dict = Body(...)):
    """AI 辅助构建评测集：给一条真实输入 + 要求，生成 N 条带 gold 的用例。

    生成的东西**不能直接信**：先 lint（自相矛盾的 gold 挑出来），再让人过一遍，
    最后跑体检（参考答案必须全过）。三步缺一步，这套题就是废的。
    """
    from src.core import call_llm
    product, stage = body.get("product"), body.get("stage")
    cfg = load_products().get(product)
    if not cfg:
        raise HTTPException(404, f"产品 {product} 不存在")
    sc = stage_of(cfg, stage)
    if not sc:
        raise HTTPException(404, f"环节 {stage} 不存在")
    request_key = (body.get("api_key") or "").strip()
    if not request_key:
        raise HTTPException(400, "生成评测集需要 API Key（从浏览器本地配置传入）。"
                                 "不想花这个钱的话，可以直接上传手写的 JSON。")
    n = max(1, min(int(body.get("n_cases") or 20), 60))
    grader = sc["grader"]
    prompt = _gen_prompt(sc, grader, n, body.get("input_example"), body.get("instruction"))
    validate_request_url(body)
    token = set_request_context(request_key, body.get("model_api_keys"), base_url=body.get("base_url", ""))
    try:
        r = call_llm(prompt, model=body.get("model") or "")
    finally:
        reset_request_context(token)
    txt = (r.text or "").strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        txt = txt.split("\n", 1)[1] if "\n" in txt else txt
    try:
        cases = json.loads(txt)
    except Exception as e:
        raise HTTPException(400, f"模型返回的不是合法 JSON：{e}｜前 200 字：{txt[:200]}")
    if not isinstance(cases, list) or not cases:
        raise HTTPException(400, "模型没有返回用例数组")

    problems, strata = _lint_cases(cases, grader)
    did = uuid.uuid4().hex[:10]
    doc = {"dataset": f"{product}-{stage}-ai", "version": time.strftime("%m%d-%H%M"),
           "product": product, "task_type": sc.get("task_type"), "strata": strata,
           "note": "AI 生成，未人工复核前不要当正式结论", "cases": cases}
    path = UPLOADS / f"{did}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    c = conn()
    c.execute("INSERT INTO dataset(id,product,stage,name,version,n_cases,path,created_at) VALUES(?,?,?,?,?,?,?,?)",
              (did, product, stage, doc["dataset"], doc["version"], len(cases), str(path),
               time.strftime("%Y-%m-%d %H:%M:%S")))
    c.commit()
    return {"id": did, "n_cases": len(cases), "strata": strata,
            "problems": problems, "n_problems": len(problems),
            "preview": cases[:3], "cost_tokens": r.tokens}


@app.get("/api/datasets/{did}")
def api_dataset_detail(did: str):
    c = conn()
    r = c.execute("SELECT * FROM dataset WHERE id=?", (did,)).fetchone()
    if not r:
        raise HTTPException(404, "评测集不存在")
    doc = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
    return {"meta": dict(r), "cases": doc.get("cases", []), "strata": doc.get("strata", {}),
            "gold_schema": doc.get("gold_schema", {})}


# ─────────────────────────── Prompt 版本 ───────────────────────────
@app.get("/api/prompts")
def api_prompts(product: str = "", stage: str = ""):
    c = conn()
    q = "SELECT * FROM prompt_tpl"
    args = []
    if product and stage:
        q += " WHERE product=? AND stage=?"
        args = [product, stage]
    return rows(c.execute(q + " ORDER BY created_at DESC", args))


@app.post("/api/prompts")
def api_create_prompt(body: dict = Body(...)):
    for k in ("product", "stage", "version", "template"):
        if not body.get(k):
            raise HTTPException(400, f"缺少字段 {k}")
    tpl = body["template"]
    # 占位符可以是 {input}/{raw}，也可以直接写数据集 input 里的一级字段名（如 {text}、{question}）。
    # 只要有一个占位符，模型就能看到数据；一个都没有等于让模型凭空作答，直接拒绝。
    import re as _re
    placeholders = _re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", tpl)
    if not placeholders:
        raise HTTPException(400, "模板里至少要有一个占位符（如 {input}、{raw} 或 {text}），否则模型看不到数据")
    pid = uuid.uuid4().hex[:10]
    c = conn()
    c.execute("INSERT INTO prompt_tpl VALUES(?,?,?,?,?,?,?)",
              (pid, body["product"], body["stage"], body["version"], tpl,
               body.get("note", ""), time.strftime("%Y-%m-%d %H:%M:%S")))
    c.commit()
    return {"id": pid}


def render_tpl(tpl: str, case: dict) -> str:
    inp = case.get("input", {}) or {}
    # input 允许是纯字符串/数组（CSV 上传、只有一段文本的题目很常见）。
    # 统一包成对象，模板里写 {input} 或 {text} 都能用，否则 inp.get 会抛 AttributeError。
    if not isinstance(inp, dict):
        inp = {"text": inp}
    flat = {"input": json.dumps(inp, ensure_ascii=False, indent=1),
            "raw": inp.get("raw_ocr_text", ""),
            "turns": "\n".join(f"- {t['content']}" for t in case.get("turns", []))}
    for k, v in inp.items():
        flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
    s = tpl
    for k, v in flat.items():
        s = s.replace("{" + k + "}", str(v))
    return s

def resolve_prompt_fn(product, stage, Sc=None, pid=None):
    """拿到这个环节的 prompt 构造函数。三级兜底，让「不写代码接入产品」成立：

    1. 指定了 prompt_id  → 用 prompt_tpl 表里的模板（可版本管理，台账能对比）
    2. 环节配置里带 prompt → 用评测计划/向导里填的模板
    3. 内置 PRODUCTS 里的 Python 函数 → 自定义适配器
    """
    if pid:
        c = conn()
        r = c.execute("SELECT * FROM prompt_tpl WHERE id=?", (pid,)).fetchone()
        if not r:
            raise HTTPException(404, "prompt 版本不存在")
        tpl = r["template"]
        return lambda case, t=tpl: render_tpl(t, case)
    tpl = (Sc or {}).get("prompt") if isinstance(Sc, dict) else None
    if tpl:
        return lambda case, t=tpl: render_tpl(t, case)
    builtin = PRODUCTS.get(product, {}).get(stage)
    if builtin and builtin[0]:
        return builtin[0]
    raise HTTPException(400, "这个环节还没有 prompt 模板。请到「接入新产品」里填一个，"
                             "或在评测计划的环节里加 \"prompt\" 字段。")



# ─────────────────────────── 运行 ───────────────────────────
def build_target(cfg: dict) -> HttpTarget | None:
    if not cfg or not cfg.get("url"):
        return None
    from urllib.parse import urlsplit
    parts = urlsplit(cfg["url"])
    if parts.username or parts.password or parts.query or parts.fragment:
        raise HTTPException(400, "接口地址不支持凭据、查询参数或片段；请使用 headers 传鉴权")
    headers = dict(cfg.get("headers") or {})
    target_key = (cfg.get("api_key") or "").strip()
    if target_key and not any(k.lower() == "authorization" for k in headers):
        headers["Authorization"] = f"Bearer {target_key}"
    return HttpTarget(
        url=cfg["url"], method=cfg.get("method", "POST"),
        headers=headers,
        body_template=cfg.get("body_template"),
        response_path=cfg.get("response_path", ""),
        timeout=int(cfg.get("timeout", 30)),
    )


def _safe_host(url) -> str | None:
    """只留 hostname。

    接口地址的 path 里经常塞着密钥（/hook/T000/B000/xxxxxx），
    整条 URL 落进 run.meta 就等于把凭据写进台账、写进导出报告。
    """
    if not url:
        return None
    try:
        from urllib.parse import urlsplit
        return urlsplit(str(url)).hostname or ""
    except Exception:
        return ""


def _target_secrets(target) -> tuple:
    """把「发出去的东西」登记为敏感串，供 redact 使用。

    headers 里的 Authorization/token 会被回显进诊断信息；body_template 里的
    固定字符串（webhook key、租户 ID、path 里的令牌）同样会被回显。两者都要登记。

    只登记「像凭据的」字符串：无空白、纯 ASCII、长度 ≥ 8。
    直接把模板里所有文本都当密钥会把正常输出打成一片 [REDACTED]
    （模板里的中文说明与 {占位符} 都不是密钥），那等于为了脱敏毁掉可读性。
    """
    if target is None:
        return ()
    out: list[str] = []
    for v in (target.headers or {}).values():
        if v:
            out.append(str(v))
    tmpl = target.body_template

    def looks_like_token(s: str) -> bool:
        if len(s) < 8 or "{" in s or any(c.isspace() for c in s):
            return False
        return all(33 <= ord(c) < 127 for c in s)

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
        elif isinstance(node, str) and looks_like_token(node):
            out.append(node)

    walk(tmpl)
    return tuple(out)


def auto_product_version() -> str:
    """从「被测产品的代码仓库」读 git 短 sha。

    产品接口不会告诉你版本，因为代码变了接口没变。版本只能从仓库来。
    """
    repo = product_repo_setting()
    if not repo or not Path(repo).exists():
        return ""
    try:
        import subprocess
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def resolve_dataset(did, product, stage):
    """上传的评测集优先，否则用 config 里的默认集。"""
    if did:
        c = conn()
        r = c.execute("SELECT * FROM dataset WHERE id=?", (did,)).fetchone()
        if not r:
            raise HTTPException(404, "评测集不存在")
        return r["path"]
    cfg = load_products()[product]
    sc = next((x for x in cfg["stages"] if x["id"] == stage), None)
    if not sc:
        raise HTTPException(400, "环节不存在")
    return sc["dataset"]


@app.post("/api/probe")
def api_probe(body: dict = Body(...)):
    """连通性测试：拿第一条用例试打一次你的接口，把「发出去什么、收回来什么」都摊开。

    这一步不做，用户跑 30 条失败后根本不知道是网络、字段名还是响应路径错了。
    """
    product, stage = body.get("product"), body.get("stage")
    target = build_target(body.get("target"))
    if target is None:
        raise HTTPException(400, "请先填写接口地址")
    path = resolve_dataset(body.get("dataset_id"), product, stage)
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = doc.get("cases") or []
    if not cases:
        raise HTTPException(400, "评测集为空")
    case = cases[0]
    sent = (_fill(target.body_template, case.get("input", {}))
            if target.body_template is not None else case.get("input", {}))
    # 探针会把「发出去什么、收回来什么」原样回显，所以 header 与 body_template
    # 里的固定串都要登记为敏感串，否则试连一次就把密钥写进响应和日志。
    token = set_request_context(secrets=_target_secrets(target))
    sent = redact(sent)
    try:
        text, lat = call_http(target, case.get("input", {}))
    except Exception as e:
        return {"ok": False, "error": str(e), "case_id": case.get("id"),
                "sent": redact(sent), "url": target.url}
    finally:
        reset_request_context(token)
    return {"ok": True, "case_id": case.get("id"), "latency_ms": lat,
            "sent": sent, "url": target.url,
            "extracted": text[:2000], "extracted_len": len(text)}


@app.post("/api/runs")
def api_start_run(body: dict = Body(...)):
    validate_request_url(body)
    scope = body.get("scope") or "product"
    if scope not in ("product", "standalone_model", "standalone_agent"):
        raise HTTPException(400, "不支持的评测 scope")
    standalone = scope.startswith("standalone")
    product, stage = body.get("product"), body.get("stage")
    if standalone:
        is_agent = scope == "standalone_agent"
        label = "独立 Agent 评测" if is_agent else "独立模型评测"
        if not body.get("dataset_id"):
            raise HTTPException(400, f"{label}必须指定数据集")
        if not body.get("grader"):
            raise HTTPException(400, f"{label}必须指定评分器")
        if is_agent and not body.get("target") and not body.get("mock"):
            # Agent 的工具轨迹只能来自被测接口（或本地仿真的 mock_trace），
            # 只调一个文本模型是拿不到轨迹的，那样每条都会进「待复核」。
            raise HTTPException(400, "独立 Agent 评测需要填写被测 Agent 的接口地址，或勾选「本地仿真」回放 mock_trace")
        if not is_agent and body.get("target"):
            raise HTTPException(400, "独立模型评测不能使用产品接口")
        product = scope
        stage = "agent" if is_agent else "model"
        sc = {"id": stage, "name": label,
              "task_type": "agent" if is_agent else "text", "grader": body["grader"]}
        cfg = {"product": product, "name": label, "stages": [sc]}
    else:
        cfg = load_products().get(product)
        if not cfg:
            raise HTTPException(404, f"产品 {product} 不存在")
        sc = stage_of(cfg, stage)
        if not sc:
            raise HTTPException(404, f"环节 {stage} 不存在")

    force = bool(body.get("force", False))
    did = body.get("dataset_id")

    # ① 数据集复核状态：没读过的题不能直接进正式台账
    if did and not force:
        row = conn().execute("SELECT name, reviewed, review_note FROM dataset WHERE id=?",
                             (did,)).fetchone()
        if row and not (row["reviewed"] or "").strip():
            raise HTTPException(409, f"数据集「{row['name']}」还没复核过。"
                                     f"先看一眼题目和标注，点「已复核」再来跑；"
                                     f"如果只是试点，传 force=true 跳过。")
        if row and row["reviewed"] == "no":
            raise HTTPException(409, f"数据集「{row['name']}」复核未通过：{row['review_note']}")

    # ② 环节依赖：上游没跑过或明显不达标时，下游的分数没有解释力
    deps = sc.get("depends_on") or []
    dep_state = []
    if deps:
        c0 = conn()
        for up in deps:
            r = c0.execute("SELECT id FROM run WHERE product=? AND stage=? "
                           "ORDER BY started_at DESC LIMIT 1", (product, up)).fetchone()
            if not r:
                dep_state.append({"upstream": up, "status": "missing", "rate": None})
                continue
            agg = c0.execute("SELECT SUM(verdict='pass') p, COUNT(*) n FROM result WHERE run_id=?",
                             (r["id"],)).fetchone()
            rate = (agg["p"] or 0) / (agg["n"] or 1)
            # 「解析错了后面全错」：上游通过率低于 80%，下游分数说明不了下游的问题
            dep_state.append({"upstream": up, "status": "ok" if rate >= 0.8 else "weak",
                              "rate": round(rate, 3), "run_id": r["id"]})
        blocking = [d for d in dep_state if d["status"] != "ok"]
        if blocking and not force:
            bits = []
            for d in blocking:
                if d["status"] == "missing":
                    bits.append(f"{d['upstream']}（还没跑过）")
                else:
                    bits.append(f"{d['upstream']}（通过率 {d['rate']:.0%}）")
            raise HTTPException(409, f"上游环节 {'、'.join(bits)} 未达标。"
                                     f"下游的失败大概率是上游传导的，先修上游；"
                                     f"要强行跑传 force=true。")

    mock = bool(body.get("mock", False))
    example_mock = bool(cfg.get("_example") and product == "general-example" and stage == "answer")
    model = body.get("model") or ""
    dataset_path = resolve_dataset(did, product, stage)

    # 目标：自己的产品接口 或 模型
    target = build_target(body.get("target"))
    is_http = target is not None

    pid = body.get("prompt_id")
    if standalone and not pid:
        def prompt_fn(case):
            value = case.get("input", {})
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                for key in ("question", "prompt", "text", "raw_ocr_text"):
                    if value.get(key):
                        return str(value[key])
            return json.dumps(value, ensure_ascii=False)
    else:
        prompt_fn = resolve_prompt_fn(product, stage, Sc=sc, pid=pid)
    mock_fn = (PRODUCTS.get(product, {}).get(stage) or (None, None))[1]
    if mock and not mock_fn:
        mock_fn = lambda case, *args: str(case.get("mock_reply", ""))
    if example_mock:
        # 只回放用例中显式提供的输出，绝不从评分标准拼答案。
        mock_fn = lambda case, *args: str(case.get("mock_reply") or "（示例没有预置回答）")

    selected = body.get("grader") or sc["grader"]
    if not isinstance(selected, str):
        raise HTTPException(400, "grader 必须是评分器名称或以 + 分隔的名称")
    selected = _validate_grader_spec(selected)
    v_counters, v_derived = _stage_eval_extras({**sc, "grader": selected})
    st = StageConfig(
        id=sc["id"], name=sc["name"], task_type=sc["task_type"],
        dataset=("" if standalone else dataset_path), grader=selected,
        gates=sc.get("gates", []), fatal=sc.get("fatal", []),
        counters=v_counters, derived=v_derived,
        rubric=sc.get("rubric", ""), pair_rule=sc.get("pair_rule", ""),
    )
    cfg2 = dict(cfg)
    cfg2["product"] = product
    dataset_snapshot = None
    run_snapshot = None
    if standalone:
        import hashlib
        from dataclasses import asdict
        try:
            dataset_snapshot = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
            if not isinstance(dataset_snapshot, dict) or not isinstance(dataset_snapshot.get("cases"), list) or not dataset_snapshot["cases"]:
                raise ValueError("空数据集")
        except (OSError, ValueError, TypeError):
            raise HTTPException(400, "数据集无法读取或为空")
        canonical = json.dumps(dataset_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        run_snapshot = {"scope": scope, "dataset_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                        "stage": asdict(st), "dataset_id": did,
                        "prompt_id": pid, "repeat": int(body.get("repeat") or 1),
                        "limit": body.get("limit")}

    request_key = (body.get("api_key") or "").strip()
    if not is_http and not mock and not request_key:
        raise HTTPException(400, "还没有配置模型 API Key。请在浏览器本地配置中填写，或勾「本地仿真」，或改为测你自己的产品接口。")
    if is_http:
        if not body.get("target", {}).get("response_path"):
            pass  # 允许留空（整个响应体当输出）

    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "started": time.time(), "product": product,
                    "stage": stage, "mock": mock, "is_http": is_http,
                    "model": (_safe_host(target.url) if is_http else ("mock" if mock else model)),
                    "prompt_id": pid, "dataset_id": did, "run_id": None}

    def work():
        token = None
        try:
            token = set_request_context(request_key, body.get("model_api_keys"), _target_secrets(target), base_url=body.get("base_url", ""))
            rid = run_stage(cfg2, st, model=model, limit=body.get("limit"),
                            mock=mock, prompt_fn=prompt_fn, mock_reply_fn=mock_fn,
                            target=target, repeat=int(body.get("repeat") or 1),
                            dataset_snapshot=dataset_snapshot, run_snapshot=run_snapshot)
            con = init_db()
            ev = evaluate(con, rid, st)
            rs = get_results(con, rid)
            meta = next(x for x in list_runs(con) if x["id"] == rid)
            meta["product_name"] = cfg["name"]
            meta["prompt_id"] = pid
            meta["dataset_id"] = did
            meta["task_type"] = sc["task_type"]
            con.execute("UPDATE run SET meta=? WHERE id=?",
                        (json.dumps({"prompt_id": pid, "dataset_id": did,
                                     "task_type": sc["task_type"], "grader": selected,
                                     "target_url": _safe_host(target.url) if is_http else None,
                                     "repeat": int(body.get("repeat") or 1),
                                     **({"scope": scope, "snapshot": redact(run_snapshot)} if standalone else {})}),
                         rid))
            con.commit()
            # 报告里并列三个基线（参考答案 / 什么都不做 / 无脑执行）
            bl = None
            try:
                _cases = dataset_snapshot if standalone else json.loads(Path(dataset_path).read_text(encoding="utf-8"))
                _cases = _cases.get("cases") or _cases
                bl = baseline_mod.check_dataset(_cases, st, _grader_fn(selected))
            except ValueError as _e:
                JOBS[job_id]["baseline_warning"] = redact(str(_e))
                bl = None
            except Exception:
                JOBS[job_id]["baseline_warning"] = "基线体检失败，请检查数据集与评分器配置"
                bl = None
            path = render(ev, meta, rs, ROOT / "reports" / f"{rid}.html", baseline=bl)

            v = redact(body.get("version") or {})
            vid = uuid.uuid4().hex[:10]
            con.execute(
                "INSERT INTO eval_version"
                "(id,product,stage,label,prompt_ver,product_ver,change_note,"
                "dataset_id,run_id,verdict,created_at,prompt_note,product_note) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (vid, product, stage, v.get("label") or time.strftime("%m-%d %H:%M"),
                 v.get("prompt_ver") or ("默认" if not pid else str(pid)[:6]),
                 v.get("product_ver") or auto_product_version()
                 or ("—" if not is_http else _safe_host(target.url)),
                 v.get("change_note") or "", did, rid,
                 ev["verdict"], time.strftime("%Y-%m-%d %H:%M:%S"),
                 v.get("prompt_note") or "", v.get("product_note") or ""))
            con.commit()

            JOBS[job_id].update({"status": "done", "run_id": rid, "report": path,
                                 "verdict": ev["verdict"], "version_id": vid})
        except Exception as e:
            JOBS[job_id].update({"status": "error", "error": (f"{type(e).__name__}: {redact(str(e))}" if token is not None else "请求上下文初始化失败")})
        finally:
            if token is not None:
                reset_request_context(token)

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{jid}")
def api_job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "任务不存在")
    j = dict(j)
    j["elapsed"] = round(time.time() - j["started"], 1)
    return j


@app.get("/api/runs")
def api_runs(product: str = ""):
    c = init_db()
    return list_runs(c, product or None)


@app.get("/api/runs/{rid}")
def api_run_detail(rid: str):
    c = init_db()
    rs = get_results(c, rid)
    meta = next((x for x in list_runs(c) if x["id"] == rid), None)
    if not meta:
        raise HTTPException(404, "运行不存在")
    return {"meta": meta, "results": rs}


@app.post("/api/runs/{rid}/review")
def api_review_mark(rid: str, body: dict = Body(...)):
    """人工改判一条。代码判分对这类软指标系统性偏严（AgentRewardBench: 30.8% vs 专家 55.1%），
    所以要留一个「人不服」的出口，并算人工与代码的一致率。
    """
    cid, human = body.get("case_id"), body.get("human")
    if human not in ("pass", "fail", None):
        raise HTTPException(400, "human 只能是 pass / fail / null")
    con = init_db()
    r = con.execute("SELECT id FROM result WHERE run_id=? AND case_id=?", (rid, cid)).fetchone()
    if not r:
        raise HTTPException(404, "这条结果不存在")
    con.execute("UPDATE result SET human_verdict=? WHERE run_id=? AND case_id=?", (human, rid, cid))
    con.commit()
    return api_review_summary(rid)


@app.get("/api/runs/{rid}/review")
def api_review_summary(rid: str):
    """人工复核进度与一致率。判定口径：代码 = N 次全过为 pass。"""
    con = init_db()
    con.row_factory = sqlite3.Row
    rows = [dict(x) for x in con.execute(
        "SELECT case_id, verdict, human_verdict FROM result WHERE run_id=? ORDER BY case_id", (rid,))]
    if not rows:
        raise HTTPException(404, "这次运行没有结果")
    by_case: dict[str, dict] = {}
    for r in rows:
        d = by_case.setdefault(r["case_id"], {"pass": 0, "n": 0, "human": None})
        d["n"] += 1
        if r["verdict"] == "pass":
            d["pass"] += 1
        if r["human_verdict"]:
            d["human"] = r["human_verdict"]
    marked = {k: v for k, v in by_case.items() if v["human"]}
    agree = sum(1 for v in marked.values()
                if (v["human"] == "pass") == (v["n"] and v["pass"] == v["n"]))
    return {
        "n_cases": len(by_case), "n_marked": len(marked), "n_agree": agree,
        "agree_rate": round(agree / len(marked), 4) if marked else None,
        "disagree": [{"case_id": k, "code": "pass" if v["pass"] == v["n"] else "fail",
                      "human": v["human"]} for k, v in marked.items()
                     if (v["human"] == "pass") != (v["n"] and v["pass"] == v["n"])],
    }


@app.get("/api/runs/{rid}/review.csv")
def api_review_csv(rid: str):
    """导出待复核清单：人工判定留空，填完回传即可算一致率。"""
    con = init_db()
    con.row_factory = sqlite3.Row
    rows = [dict(x) for x in con.execute(
        "SELECT * FROM result WHERE run_id=? ORDER BY case_id", (rid,))]
    if not rows:
        raise HTTPException(404, "这次运行没有结果")
    by_case: dict[str, dict] = {}
    for r in rows:
        d = by_case.setdefault(r["case_id"], {"stratum": r["stratum"], "pass": 0, "n": 0,
                                              "out": r["output"], "note": r["note"],
                                              "human": r["human_verdict"]})
        d["n"] += 1
        if r["verdict"] == "pass":
            d["pass"] += 1
    buf = io.StringIO()
    buf.write("\ufeff")
    w = csv.writer(buf)
    w.writerow(["case_id", "stratum", "代码判定", "输出", "判分说明", "人工判定(pass/fail)", "人工备注"])
    for cid, d in by_case.items():
        w.writerow([cid, d["stratum"], "pass" if d["pass"] == d["n"] else "fail",
                    (d["out"] or "")[:600], (d["note"] or "")[:200], d["human"] or "", ""])
    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="review-{rid}.csv"'})


@app.get("/api/reports/{name}", response_class=HTMLResponse)
def api_report_by_name(name: str):
    if "/" in name or "\\" in name or not name.endswith(".html"):
        raise HTTPException(400, "非法文件名")
    base = (ROOT / "reports").resolve()
    p = (base / name).resolve()
    if not str(p).startswith(str(base)) or not p.exists():
        raise HTTPException(404, "报表不存在")
    return p.read_text(encoding="utf-8")


@app.get("/api/runs/{rid}/report", response_class=HTMLResponse)
def api_report(rid: str):
    p = ROOT / "reports" / f"{rid}.html"
    if not p.exists():
        raise HTTPException(404, "报表尚未生成")
    return p.read_text(encoding="utf-8")


# ─────────────────────────── A/B 对比 ───────────────────────────
def _check_comparable(ma, mb):
    a = json.loads(ma.get("meta") or "{}")
    b = json.loads(mb.get("meta") or "{}")
    if not str(a.get("scope") or "").startswith("standalone") and \
            not str(b.get("scope") or "").startswith("standalone"):
        return
    if a.get("scope") != b.get("scope"):
        raise HTTPException(400, "评测 scope 不同，不能比较")
    if ma.get("mode") != mb.get("mode"):
        raise HTTPException(400, "运行模式不同，不能比较")
    sa, sb = a.get("snapshot") or {}, b.get("snapshot") or {}
    if not sa or not sb:
        raise HTTPException(400, "缺少运行快照，无法确认比较口径")
    for key in ("dataset_sha256", "repeat", "limit"):
        if sa.get(key) != sb.get(key):
            raise HTTPException(400, f"评测口径不同：{key}")
    sta, stb = sa.get("stage") or {}, sb.get("stage") or {}
    for key in ("grader", "task_type", "gates", "fatal", "counters", "derived", "rubric", "pair_rule"):
        if sta.get(key) != stb.get(key):
            raise HTTPException(400, f"评分口径不同：{key}")


@app.get("/api/compare")
def api_compare(a: str, b: str):
    c = init_db()
    ra, rb = get_results(c, a), get_results(c, b)
    if not ra or not rb:
        raise HTTPException(404, "其中一个运行没有结果")
    ma = next(x for x in list_runs(c) if x["id"] == a)
    mb = next(x for x in list_runs(c) if x["id"] == b)
    _check_comparable(ma, mb)

    da = {r["case_id"]: r for r in ra}
    db_ = {r["case_id"]: r for r in rb}
    common = sorted(set(da) & set(db_))

    def metric_avg(rs, key):
        vals = [r["metrics"][key] for r in rs if isinstance(r["metrics"].get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    COUNTERS = {"tp", "fp", "fn", "tn", "btp", "bfn", "field_hit", "field_total"}
    keys = sorted({k for r in ra + rb for k in r["metrics"]} - COUNTERS)
    table = []
    for k in keys:
        va, vb = metric_avg(ra, k), metric_avg(rb, k)
        if va is None or vb is None:
            continue
        table.append({"metric": k, "a": va, "b": vb, "delta": round(vb - va, 4)})

    diffs = []
    for cid in common:
        x, y = da[cid], db_[cid]
        if x["verdict"] != y["verdict"]:
            diffs.append({"case_id": cid, "stratum": x["stratum"],
                          "a": x["verdict"], "b": y["verdict"],
                          "a_note": x["note"], "b_note": y["note"]})

    same = ma.get("mode") == "mock" and mb.get("mode") == "mock"
    warns = []
    if same:
        warns.append("两次都是本地仿真（mock），输出是构造出来的，A/B 差异没有意义。请关闭 mock 重跑。")
    if len(common) < 20:
        warns.append(f"共同用例只有 {len(common)} 条，统计意义很弱；建议 ≥30 条。")
    big = [t for t in table if abs(t["delta"]) > 0.05]
    if not big:
        warns.append("没有任何指标差异超过 5 个百分点。在这个样本量下，应判定为「同档」，不要把微小差异当成改进。")
    if ma.get("model") != mb.get("model"):
        warns.append("两次运行的模型不同，无法归因到 prompt——一次只改一个变量。")

    return {"a": {"id": a, "model": ma.get("model"), "mode": ma.get("mode"),
                  "started": ma.get("started_at"), "n": len(ra),
                  "prompt_id": (json.loads(ma.get("meta") or "{}") or {}).get("prompt_id")},
            "b": {"id": b, "model": mb.get("model"), "mode": mb.get("mode"),
                  "started": mb.get("started_at"), "n": len(rb),
                  "prompt_id": (json.loads(mb.get("meta") or "{}") or {}).get("prompt_id")},
            "metrics": table, "diffs": diffs, "n_common": len(common), "warnings": warns}


# 指标中文名 + 「越大越好 / 越小越好」。
# 这里必须是中性口径：同一个 key 在不同产品里含义不同（precision 可能是分类精确率，
# 也可能是检索精确率），标签写死成某个业务的说法会误导读数的人。
METRIC_LABELS = {
    "accuracy": ("准确率", True), "precision": ("精确率", True),
    "recall": ("召回率", True), "f1": ("F1", True),
    "tp": ("正例判对数", True), "fp": ("误报数", False),
    "fn": ("漏报数", False), "tn": ("负例判对数", True),
    "n": ("样本数", True), "coverage": ("关键词覆盖率", True),
    "mention_ok": ("必需内容命中", True), "forbidden_hits": ("禁用词命中数", False),
    "len_ok": ("长度合规", True), "has_citation": ("是否给出引用", True),
    "phantom_refs": ("无效引用数", False), "state_ok": ("终态一致", True),
    "state_match": ("终态字段匹配率", True),
    "schema_ok": ("结构合规", True), "field_ok": ("字段完整", True),
    "enum_ok": ("枚举合规", True), "missing": ("缺失字段数", False),
    "has_trace": ("有轨迹", True), "path_ok": ("关键路径完整", True),
    "path_recall": ("路径召回率", True), "missing_steps": ("缺失步骤数", False),
    "extra_steps": ("多余步骤数", False),
    "n_calls": ("调用次数", False), "calls_ok": ("调用次数合规", True),
    "whitelist_ok": ("工具白名单合规", True), "required_ok": ("必调工具覆盖", True),
    "args_ok": ("参数完整", True),
    "hit_at_k": ("Hit@k", True), "recall_at_k": ("Recall@k", True),
    "precision_at_k": ("Precision@k", True), "mrr": ("MRR", True),
    "k": ("k 值", False), "has_label": ("有相关标注", True),
    "tokens_ok": ("token 在预算内", True), "latency_ok": ("延迟在预算内", True),
    "steps_ok": ("步骤数在预算内", True),
    "tokens": ("token 数", False), "latency_ms": ("延迟(ms)", False),
    "rubric_score": ("判定表总分", True), "gate_fail": ("否决项未达标", False),
    "judge_confidence": ("评委置信度", True),
    "has_pair": ("有 A/B 候选", True), "pair_consistent": ("结论一致", True),
    "a_wins": ("A 胜", True), "b_wins": ("B 胜", True), "tie": ("平局", True),
}


@app.get("/api/overview")
def api_overview(include_builtin: bool = False):
    """跨产品总览：每个产品的每个环节，最近一次跑成什么样。

    为什么「有总览」和「不跨产品比分数」不矛盾：
      指标同名不同义（都叫 precision，一个环节算检索命中、另一个环节算字段抽取），
      横向比分数是错的。但「哪个环节还没跑过、哪个环节一直在挂、
      哪个产品这周没动过」跟指标口径无关，可以横向看。
      所以这里只给通过率和时间，不排名、不给综合分。
    """
    prods = load_products()
    if not include_builtin:
        prods = {k: v for k, v in prods.items() if v.get("_source") != "builtin"}
    c = conn()
    out, n_stage, n_done, n_pass = [], 0, 0, 0
    for pid, cfg in sorted(prods.items()):
        stages = []
        for st in cfg.get("stages") or []:
            n_stage += 1
            r = c.execute("SELECT * FROM run WHERE product=? AND stage=? "
                          "ORDER BY started_at DESC LIMIT 1", (pid, st["id"])).fetchone()
            if not r:
                stages.append({"id": st["id"], "name": st.get("name", st["id"]),
                               "has_run": False, "run_id": "", "when": "",
                               "n": 0, "pass_rate": None, "verdict": "未跑过",
                               "depends_on": st.get("depends_on") or [],
                               "grader": st.get("grader", "")})
                continue
            agg = c.execute("SELECT SUM(verdict='pass') p, COUNT(*) n FROM result WHERE run_id=?",
                            (r["id"],)).fetchone()
            n = agg["n"] or 1
            rate = (agg["p"] or 0) / n
            n_done += 1
            n_pass += 1 if rate >= 0.8 else 0
            ev = c.execute("SELECT verdict, label FROM eval_version WHERE run_id=? LIMIT 1",
                           (r["id"],)).fetchone()
            stages.append({
                "id": st["id"], "name": st.get("name", st["id"]), "has_run": True,
                "run_id": r["id"], "when": r["started_at"], "n": agg["n"],
                "pass_rate": round(rate, 3),
                "verdict": (ev["verdict"] if ev else ("合格" if rate >= 0.8 else "不合格")),
                "depends_on": st.get("depends_on") or [], "grader": st.get("grader", ""),
                "repeat": json.loads(r["meta"] or "{}").get("repeat", 1),
            })
        out.append({"product": pid, "name": cfg.get("name", pid), "stages": stages})
    return {"products": out,
            "summary": {"n_products": len(out), "n_stages": n_stage,
                        "n_run": n_done, "n_pass": n_pass,
                        "n_never_run": n_stage - n_done}}


@app.get("/api/ledger")
def api_ledger(product: str = "", stage: str = ""):
    """指标台账：某产品某环节下所有评测 run 横向铺开，供对比扫读。

    这是平台的主界面。报告只是某一行的详情。
    """
    c = conn()
    q = "SELECT * FROM eval_version WHERE 1=1"
    args = []
    if product:
        q += " AND product=?"
        args.append(product)
    if stage:
        q += " AND stage=?"
        args.append(stage)
    vers = rows(c.execute(q + " ORDER BY created_at ASC", args))

    con = init_db()
    allruns = {x["id"]: x for x in list_runs(con)}
    out = []
    seen_keys = []
    for v in vers:
        run = allruns.get(v["run_id"])
        if not run:
            continue
        try:
            meta, ev, rs = _load_run_ctx(v["run_id"])
        except Exception:
            continue
        vals = {k: round(x, 4) for k, x in ev["metrics"].items() if k in METRIC_LABELS}
        for k in vals:
            if k not in seen_keys:
                seen_keys.append(k)
        out.append({
            "version_id": v["id"], "run_id": v["run_id"], "label": v["label"],
            "prompt_ver": v["prompt_ver"] or "默认",
            "product_ver": v["product_ver"] or "—",
            "change_note": v["change_note"] or "",
            "prompt_note": v["prompt_note"] or "",
            "product_note": v["product_note"] or "",
            "created_at": v["created_at"], "verdict": v["verdict"],
            "mode": run.get("mode"), "n": run.get("n_cases"),
            "repeat": _repeat_of((meta or {}).get("meta")),
            "pass_pow_k": ev.get("pass_pow_k"),
            "n_fatal": len(ev["fatals"]), "values": vals,
        })

    prev = None
    for r in out:
        ch = []
        if prev:
            if prev["prompt_ver"] != r["prompt_ver"]:
                ch.append("prompt")
            if prev["product_ver"] != r["product_ver"]:
                ch.append("product")
        r["changed"] = ch
        prev = r

    # 达标门里的指标 = 核心指标。其余默认折叠，否则一张表十几列没法扫读。
    core_keys = []
    if product and stage:
        try:
            scg = next(x for x in load_products()[product]["stages"] if x["id"] == stage)
            core_keys = [g["key"] for g in scg.get("gates", [])]
        except Exception:
            core_keys = []
    metrics = [{"key": k, "label": METRIC_LABELS[k][0],
                "higher_better": METRIC_LABELS[k][1],
                "core": (k in core_keys)} for k in seen_keys]
    return {"metrics": metrics, "rows": out, "has_core": bool(core_keys),
            "auto_product_version": auto_product_version(),
            "product_repo": product_repo_setting()}


@app.get("/api/versions")
def api_versions(product: str = "", stage: str = ""):
    c = conn()
    q = "SELECT * FROM eval_version"
    args = []
    if product:
        q += " WHERE product=?"
        args.append(product)
        if stage:
            q += " AND stage=?"
            args.append(stage)
    return rows(c.execute(q + " ORDER BY created_at DESC", args))


def _repeat_of(meta) -> int:
    """run.meta 是 JSON 字符串；容错读 repeat。"""
    try:
        m = json.loads(meta) if isinstance(meta, (str, bytes)) else (meta or {})
        return int(m.get("repeat") or 1)
    except Exception:
        return 1


def _load_run_ctx(run_id):
    con = init_db()
    meta = next((x for x in list_runs(con) if x["id"] == run_id), None)
    if not meta:
        raise HTTPException(404, f"运行 {run_id} 不存在")
    stored = json.loads(meta.get("meta") or "{}")
    snap = stored.get("snapshot") or {}
    if str(stored.get("scope") or "").startswith("standalone") and snap.get("stage"):
        st = StageConfig(**snap["stage"])
        meta["product_name"] = "独立模型评测"
        return meta, evaluate(con, run_id, st), get_results(con, run_id)
    cfg = load_products()[meta["product"]]
    sc = next(x for x in cfg["stages"] if x["id"] == meta["stage"])
    # 跑的时候可能临时换过评分器（meta.grader），汇总 counters/derived 要跟它走，
    # 否则重算历史报告会按产品默认评分器去解释别人的指标。
    try:
        used = (json.loads(meta.get("meta") or "{}") or {}).get("grader") or sc["grader"]
    except Exception:
        used = sc["grader"]
    l_counters, l_derived = _stage_eval_extras({**sc, "grader": used})
    st = StageConfig(id=sc["id"], name=sc["name"], task_type=sc["task_type"],
                     dataset=meta["dataset"], grader=used,
                     gates=sc.get("gates", []), fatal=sc.get("fatal", []),
                     counters=l_counters, derived=l_derived)
    meta["product_name"] = cfg["name"]
    return meta, evaluate(con, run_id, st), get_results(con, run_id)


@app.get("/api/version-compare")
def api_version_compare(a: str, b: str):
    """任意两个评测版本对比。先判断「哪些变量变了」，再决定能不能归因。"""
    c = conn()
    va = c.execute("SELECT * FROM eval_version WHERE id=?", (a,)).fetchone()
    vb = c.execute("SELECT * FROM eval_version WHERE id=?", (b,)).fetchone()
    if not va or not vb:
        raise HTTPException(404, "版本不存在")
    if va["product"] != vb["product"] or va["stage"] != vb["stage"]:
        raise HTTPException(400, "两个版本必须属于同一产品、同一环节，否则指标不可比")

    ma, ea, ra = _load_run_ctx(va["run_id"])
    mb, eb, rb = _load_run_ctx(vb["run_id"])
    _check_comparable(ma, mb)
    name = f"vercmp-{va['id']}-{vb['id']}.html"
    render_compare(ma, mb, ea, eb, ra, rb, ROOT / "reports" / name,
                   ver_a=dict(va), ver_b=dict(vb))
    return {"url": f"/api/reports/{name}", "verdict_a": ea["verdict"], "verdict_b": eb["verdict"]}


@app.post("/api/dual")
def api_dual(body: dict = Body(...)):
    """双跑：同数据集跑两遍（prompt 基线 + 产品接口），出三份报告。

    双跑用于比较基线与产品接口的表现。
    """
    validate_request_url(body)
    product, stage = body.get("product"), body.get("stage")
    cfg = load_products().get(product)
    if not cfg:
        raise HTTPException(404, f"产品 {product} 不存在")
    sc = stage_of(cfg, stage)
    if not sc:
        raise HTTPException(404, f"环节 {stage} 不存在")

    did = body.get("dataset_id")
    dataset_path = resolve_dataset(did, product, stage)
    target = build_target(body.get("target"))
    if target is None:
        raise HTTPException(400, "双跑需要填写你的产品接口地址")

    model = body.get("model") or ""
    pid = body.get("prompt_id")
    prompt_fn = resolve_prompt_fn(product, stage, Sc=sc, pid=pid)
    mock_fn = (PRODUCTS.get(product, {}).get(stage) or (None, None))[1]

    request_key = (body.get("api_key") or "").strip()
    has_key = bool(request_key)
    baseline_mock = not has_key
    if baseline_mock and not body.get("allow_mock_baseline"):
        raise HTTPException(400, "双跑的基线需要 API Key（否则基线是仿真数据，对照没有意义）。请从浏览器传入 api_key，或勾选「允许用仿真做基线」。")

    # 双跑也允许临时换评分器：和单跑同一套校验，前端不必记两条规则
    selected = body.get("grader") or sc["grader"]
    if not isinstance(selected, str):
        raise HTTPException(400, "grader 必须是评分器名称或以 + 分隔的名称")
    selected = _validate_grader_spec(selected)
    d_counters, d_derived = _stage_eval_extras({**sc, "grader": selected})
    st = StageConfig(id=sc["id"], name=sc["name"], task_type=sc["task_type"],
                     dataset=dataset_path, grader=selected,
                     gates=sc.get("gates", []), fatal=sc.get("fatal", []),
                     counters=d_counters, derived=d_derived)
    cfg2 = dict(cfg)
    cfg2["product"] = product

    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "step": "基线", "started": time.time(),
                    "product": product, "stage": stage, "dual": True,
                    "model": f"dual:{_safe_host(target.url)}", "run_id": None, "reports": []}

    def work():
        token = None
        try:
            token = set_request_context(request_key, body.get("model_api_keys"), _target_secrets(target), base_url=body.get("base_url", ""))
            con = init_db()

            JOBS[job_id]["step"] = "基线（prompt）"
            rid_a = run_stage(cfg2, st, model=model, limit=body.get("limit"),
                              mock=baseline_mock, prompt_fn=prompt_fn,
                              mock_reply_fn=mock_fn, target=None)
            ev_a = evaluate(con, rid_a, st)
            rows_a = get_results(con, rid_a)
            meta_a = next(x for x in list_runs(con) if x["id"] == rid_a)
            meta_a["product_name"] = cfg["name"]
            path_a = render(ev_a, meta_a, rows_a, ROOT / "reports" / f"{rid_a}.html")

            JOBS[job_id]["step"] = "产品接口"
            rid_b = run_stage(cfg2, st, model=model, limit=body.get("limit"),
                              mock=False, prompt_fn=prompt_fn,
                              mock_reply_fn=mock_fn, target=target)
            ev_b = evaluate(con, rid_b, st)
            rows_b = get_results(con, rid_b)
            meta_b = next(x for x in list_runs(con) if x["id"] == rid_b)
            meta_b["product_name"] = cfg["name"]
            path_b = render(ev_b, meta_b, rows_b, ROOT / "reports" / f"{rid_b}.html")

            JOBS[job_id]["step"] = "生成对照报告"
            path_c = render_compare(meta_a, meta_b, ev_a, ev_b, rows_a, rows_b,
                                    ROOT / "reports" / f"compare-{rid_b}.html")

            con.execute("UPDATE run SET meta=? WHERE id=?",
                        (json.dumps({"prompt_id": pid, "dataset_id": did,
                                     "pair": rid_b, "role": "baseline",
                                     "target_url": None}), rid_a))
            con.execute("UPDATE run SET meta=? WHERE id=?",
                        (json.dumps({"prompt_id": pid, "dataset_id": did,
                                     "pair": rid_a, "role": "product",
                                     "target_url": _safe_host(target.url)}), rid_b))
            con.commit()

            JOBS[job_id].update({
                "status": "done", "step": "完成",
                "run_id": rid_b, "run_a": rid_a, "run_b": rid_b,
                "verdict": ev_b["verdict"], "verdict_a": ev_a["verdict"],
                "compare": path_c,
                "reports": [
                    {"kind": "baseline", "label": "① prompt 基线", "run": rid_a, "path": path_a},
                    {"kind": "product", "label": "② 产品接口", "run": rid_b, "path": path_b},
                    {"kind": "compare", "label": "③ 对照诊断", "run": f"compare-{rid_b}", "path": path_c},
                ],
            })
        except Exception as e:
            JOBS[job_id].update({"status": "error", "error": (f"{type(e).__name__}: {redact(str(e))}" if token is not None else "请求上下文初始化失败")})
        finally:
            if token is not None:
                reset_request_context(token)

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


# ─────────────────────────── 设置 ───────────────────────────
@app.get("/api/settings")
def api_get_settings():
    return {"model_configured": False,
            "product_repo": product_repo_setting(),
            "db_path": str(DB), "uploads": str(UPLOADS)}


@app.post("/api/settings")
def api_set_settings(body: dict = Body(...)):
    # 未知字段一律拒绝，避免写入凭据。
    if set(body) - {"product_repo"}:
        raise HTTPException(400, "设置仅支持 product_repo")
    if "product_repo" in body:
        value = body["product_repo"]
        if value is not None and not isinstance(value, str):
            raise HTTPException(400, "product_repo 必须为字符串")
        c = init_db()
        c.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        c.execute("INSERT OR REPLACE INTO app_settings(key,value) VALUES('product_repo',?)", ((value or "").strip(),))
        c.commit()
    return {"ok": True, "model_configured": False}


if __name__ == "__main__":
    import uvicorn
    init_db()
    print("\n  评测平台已启动 →  http://localhost:8765\n  Ctrl+C 停止\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
