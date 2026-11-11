"""评测引擎核心：存储 / 评分器注册 / 模型调用 / 跑批 / 判定。

设计约束：
- L1 通用层：与具体产品解耦
- 新增产品只改 config + dataset + grader，本文件不动
"""
from __future__ import annotations
import json, os, sqlite3, tempfile, time, hashlib, urllib.request, urllib.error
from dataclasses import dataclass, field
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
# 默认使用持久数据库；EVAL_DB 可用于隔离测试或显式指定位置。
DB = Path(os.environ.get("EVAL_DB") or (ROOT / "data" / "eval.db"))

# ──────────────────────────── 存储 ────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS run(
  id TEXT PRIMARY KEY, product TEXT, stage TEXT, dataset TEXT,
  model TEXT, mode TEXT, started_at TEXT, finished_at TEXT,
  n_cases INT, cost REAL, meta TEXT
);
CREATE TABLE IF NOT EXISTS result(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT, case_id TEXT, stratum TEXT,
  output TEXT, latency_ms INT, tokens INT,
  metrics TEXT, verdict TEXT, fatal TEXT, note TEXT,
  repeat_idx INT DEFAULT 0, prompt TEXT
);
CREATE INDEX IF NOT EXISTS idx_result_run ON result(run_id);
"""

def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    # 新库从空表开始；绝不隐式复制历史示例库。
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    # 老库补列（幂等）：repeat_idx 是第几次重复，prompt 是发给模型的原文
    for col, typ in (("repeat_idx", "INT DEFAULT 0"), ("prompt", "TEXT"),
                     ("human_verdict", "TEXT")):
        try:
            con.execute(f"ALTER TABLE result ADD COLUMN {col} {typ}")
        except Exception:
            pass
    con.commit()
    return con


def save_run(con, **kw):
    con.execute(
        "INSERT INTO run(id,product,stage,dataset,model,mode,started_at,finished_at,n_cases,cost,meta)"
        " VALUES(:id,:product,:stage,:dataset,:model,:mode,:started_at,:finished_at,:n_cases,:cost,:meta)",
        {"meta": "{}", "cost": 0.0, **kw},
    )
    con.commit()


def save_result(con, run_id, case_id, stratum, output, latency_ms, tokens, metrics, verdict, fatal, note,
                repeat_idx=0, prompt=""):
    con.execute(
        "INSERT INTO result(run_id,case_id,stratum,output,latency_ms,tokens,metrics,verdict,fatal,note,"
        "repeat_idx,prompt) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, case_id, stratum, output, latency_ms, tokens,
         json.dumps(metrics, ensure_ascii=False), verdict,
         json.dumps(fatal, ensure_ascii=False), note, repeat_idx, prompt or ""),
    )


def list_runs(con, product=None):
    q = "SELECT * FROM run" + (" WHERE product=?" if product else "") + " ORDER BY started_at DESC"
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(q, (product,) if product else ())]


def get_results(con, run_id):
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM result WHERE run_id=? ORDER BY case_id", (run_id,))]
    for r in rows:
        r["metrics"] = json.loads(r["metrics"] or "{}")
        r["fatal"] = json.loads(r["fatal"] or "[]")
    return rows

# ──────────────────────── 评分器注册表 ────────────────────────
REGISTRY: dict[str, Callable] = {}

def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

def get_grader(name):
    if name not in REGISTRY:
        raise KeyError(f"grader '{name}' 未注册。已注册：{sorted(REGISTRY)}")
    return REGISTRY[name]


# ───────────────────────── 多模型网关（全局）─────────────────────────
# 评分器需要「另一个模型」来执行 rubric。这里存一个进程级引用，
# 由 run_stage / server 在建好角色后注入。没有注入时，LLM 类评分器一律返回
# review（待人工），绝不假装给出了分数 —— 静默降级是评测平台最危险的 bug。
_ROUTER = None
_REQUEST = ContextVar("eval_request", default=None)

def validate_base_url(base_url=""):
    from urllib.parse import urlsplit
    if not base_url:
        return
    try:
        if not isinstance(base_url, str):
            raise ValueError()
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                any(c.isspace() or ord(c) < 32 for c in base_url)):
            raise ValueError()
        _ = parsed.port
    except (ValueError, TypeError):
        raise ValueError("模型 API 地址必须是无凭据、查询参数或片段的 HTTP(S) URL") from None


def set_request_context(api_key="", model_api_keys=None, secrets=(), base_url=""):
    from src.models import load_router
    keys = dict(model_api_keys or {})
    if api_key:
        keys.setdefault("judge", api_key)
    validate_base_url(base_url)
    router = load_router()
    router.api_keys = keys
    if base_url and api_key and not (model_api_keys or {}).get("judge") and "judge" in router.specs:
        router.specs["judge"].base_url = base_url
    values = [api_key, *keys.values(), *secrets]
    return _REQUEST.set((api_key, router, tuple(str(x) for x in values if x), base_url))

def reset_request_context(token):
    _REQUEST.reset(token)

def redact(value):
    context = _REQUEST.get()
    secrets = context[2] if context else ()
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
            if secret.lower().startswith("bearer "):
                value = value.replace(secret[7:], "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact(k): redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def set_router(r):
    global _ROUTER
    _ROUTER = r
    return r


def get_router():
    context = _REQUEST.get()
    return context[1] if context else _ROUTER


# ─────────────────── 评分器组合：规则 + 判官一起用 ───────────────────
# 一个复杂产品往往需要多套标准：工具调用既要看「调没调对」（代码判），
# 也要看「选得对不对口」（模型判）。所以环节可以挂多个评分器，用 + 连起来。
GRADE_PRIORITY = {"fatal": 4, "fail": 3, "review": 2, "pass": 1}


def apply_graders(spec: str, case, output, stage) -> dict:
    """按 + 组合多个评分器，合并结果。

    合并规则（顺序即优先级）：
      fatal > fail > review > pass
    为什么 review 高于 pass：一个维度证据不足时，不能因为另一个维度给了 pass
    就把这条算过 —— 那等于用「评得出来的」盖住「评不出来的」。
    """
    names = [x.strip() for x in str(spec or "").replace(",", "+").split("+") if x.strip()]
    trace = case.get("_trace")
    steps = trace.get("steps") if isinstance(trace, dict) else trace
    if any(n in {"trajectory_rule", "trajectory_judge", "tool_call_rule", "tool_call_judge"} for n in names) and not isinstance(steps, list):
        return {"metrics": {}, "verdict": "review", "fatal": [], "note": "缺少有效轨迹，无法评测"}
    if not names:
        raise KeyError("环节没有配置 grader")
    if len(names) == 1:
        return get_grader(names[0])(case, output, stage)

    metrics, fatals, notes = {}, [], []
    worst, worst_by = "pass", ""
    for n in names:
        try:
            g = get_grader(n)(case, output, stage)
        except Exception as e:
            notes.append(f"{n} 执行失败：{e}")
            g = {"metrics": {}, "verdict": "review", "fatal": [], "note": f"执行失败 {e}"}
        # 合并指标时不改 key。
        # 曾经这里加了 "评分器名." 前缀，结果环节里 gates 写的 key（如 path_ok）
        # 在汇总里永远找不到 —— 达标门恒为「—」、passed 恒 False，跑得再对也不合格。
        # 同名冲突时按声明顺序「后者覆盖前者」，但 key 必须保持原样：
        # 汇总、派生公式、gates 全部靠 key 对齐，改名等于把整条链路切断。
        for k, v in (g.get("metrics") or {}).items():
            metrics[k] = v
        fatals += list(g.get("fatal") or [])
        if g.get("note"):
            notes.append(f"{n}: {g['note']}")
        vd = g.get("verdict", "review")
        if GRADE_PRIORITY.get(vd, 0) > GRADE_PRIORITY.get(worst, 0):
            worst, worst_by = vd, n

    # 标出是哪个评分器决定了最终判定 —— 失败归因比失败本身更有用：
    # 「规则层挂」和「判官层挂」要改的东西完全不同。
    note = f"[{worst_by or '?'}] " + "｜".join(notes)[:290]
    return {"metrics": metrics, "verdict": worst, "fatal": fatals, "note": note}

# ─────────────────────────── 模型调用 ───────────────────────────
@dataclass
class LLMReply:
    text: str
    latency_ms: int = 0
    tokens: int = 0
    cost: float = 0.0


def call_llm(prompt: str | None = None, model: str = "", mock_fn: Callable | None = None,
             temperature: float = 0.0, timeout: int = 60, messages: list | None = None) -> LLMReply:
    """真实调用走 OpenAI 兼容接口；mock_fn 存在时不联网，用合成输出。

    messages 用于多轮：传入完整对话历史，prompt 被忽略。
    """
    if mock_fn is not None:
        t0 = time.time()
        text = mock_fn()
        return LLMReply(text=text, latency_ms=int((time.time() - t0) * 1000) + 800, tokens=len(text) // 2)

    context = _REQUEST.get()
    key = context[0] if context else ""
    if not key:
        raise RuntimeError("缺少请求级 API 密钥。本地验证请使用 mock。")
    if not model or not context or not context[3]:
        raise RuntimeError("真实调用需要显式指定模型和 API 地址")

    msgs = messages if messages is not None else [{"role": "user", "content": prompt}]
    body = json.dumps({
        "model": model,
        "messages": msgs,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        (context[3].rstrip("/") + "/chat/completions"),
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API 错误 HTTP {e.code}") from None
    text = redact(data["choices"][0]["message"]["content"])
    usage = data.get("usage", {})
    tok = usage.get("total_tokens", 0)
    # 未配置供应商价格时不推算成本。
    return LLMReply(text=text, latency_ms=int((time.time() - t0) * 1000), tokens=tok,
                    cost=0.0)

# 跑批
@dataclass
class StageConfig:
    id: str
    name: str
    task_type: str
    dataset: str
    grader: str
    gates: list = field(default_factory=list)
    fatal: list = field(default_factory=list)
    counters: list = field(default_factory=list)   # 这些 metric key 是计数，需要求和而非平均
    derived: list = field(default_factory=list)    # 集合级指标，由 counters 计算：{"key","formula","label"}
    rubric: str = ""                               # LLM 类评分器用的判定表名（config/rubrics/*.yaml）
    pair_rule: str = ""                            # pair_compare 的判定规则


def load_products() -> dict[str, dict]:
    """内置产品（config/products/*.json）+ 用户在界面里建的产品（product_doc 表）。

    DB 优先。这样接入一个新产品不必改代码、也不必往仓库里加文件——
    评测计划导入后就是往这张表里写一行。
    """
    out = {}
    for p in sorted((ROOT / "config" / "products").glob("*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg["_source"] = "builtin"
        out[cfg["product"]] = cfg
    try:
        con = init_db()
        con.execute("CREATE TABLE IF NOT EXISTS product_doc("
                    "product TEXT PRIMARY KEY, doc TEXT, created_at TEXT)")
        for r in con.execute("SELECT product, doc FROM product_doc"):
            try:
                cfg = json.loads(r[1])
                cfg.setdefault("product", r[0])
                cfg["_source"] = "db"
                out[cfg["product"]] = cfg
            except Exception:
                continue
    except Exception:
        pass
    return out


def save_product_doc(doc: dict) -> str:
    """写入/覆盖一个产品的配置（含所有环节）。返回 product id。"""
    product = (doc.get("product") or "").strip()
    if not product:
        raise ValueError("缺少 product")
    con = init_db()
    con.execute("CREATE TABLE IF NOT EXISTS product_doc("
                "product TEXT PRIMARY KEY, doc TEXT, created_at TEXT)")
    con.execute("INSERT OR REPLACE INTO product_doc(product, doc, created_at) VALUES(?,?,?)",
                (product, json.dumps(doc, ensure_ascii=False),
                 __import__("time").strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    con.close()
    return product


def stage_of(cfg: dict, stage_id: str) -> dict | None:
    return next((x for x in (cfg.get("stages") or []) if x.get("id") == stage_id), None)


def has_stage(product: str, stage_id: str) -> bool:
    cfg = load_products().get(product)
    return bool(cfg and stage_of(cfg, stage_id))


def load_dataset(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def classify_reply(text: str) -> str:
    """把 agent 上一句粗分成 劝 / 不劝 / 说教 / 其它。

    带分支的用户脚本靠它决定「下一句该说什么」——这就是离线模拟用户的全部机制：
    不生成，只查表。
    """
    t = text or ""
    if any(w in t for w in ("你不应该", "你怎么", "早说了", "说了多少次", "成年人应该", "智商税")):
        return "lecture"
    if any(w in t for w in ("不建议", "别买", "放一放", "再想想", "先别", "不推荐", "暂缓")):
        return "advise"
    if any(w in t for w in ("可以买", "买吧", "支持你", "没问题", "值得", "买下来")):
        return "decline"
    return "default"


def _one_case(c, model, mock, prompt_fn, mock_reply_fn, target):
    """跑一条用例，返回 (交给判分器的输出, 记下来的原文, 延迟ms, tokens, 成本)。

    第二个返回值是「执行记录」——发给模型的 prompt（多轮时是完整 messages）。
    分数告诉结果，记录才能看出它是怎么做的。
    """
    turns = c.get("turns") or []
    if target is not None:
        body = c.get("input", {})
        out, lat, trace = call_http(target, body, include_trace=True)
        return out, json.dumps(body, ensure_ascii=False)[:3000], lat, 0, 0.0, trace

    branch = c.get("branch")
    if branch:
        # 离线用户模拟：剧本是预生成的（别的平台生成后导入），平台只按 agent 的回应查表选下一句。
        # 好处是零成本、可复现（同一份剧本每次跑一样，版本对比才公平）；
        # 代价是测不到「临场反应」——剧本没写的分支就落 default。
        history = []
        if prompt_fn:
            history.append({"role": "system", "content": prompt_fn(c)})
        lat = tok = 0
        cost = 0.0
        last = ""
        for step in branch:
            if not isinstance(step, dict):
                continue
            if "say" in step:
                user_say = step["say"]
            elif "when" in step:
                opts = step["when"] or {}
                cls = classify_reply(last)
                pool = opts.get(cls) or opts.get("default") or ["…"]
                user_say = pool[0] if isinstance(pool, list) else str(pool)
            else:
                continue
            history.append({"role": "user", "content": user_say})
            mfn = (lambda cc=c, hh=history: mock_reply_fn(cc, hh)) if mock else None
            r = call_llm(None, model=model, mock_fn=mfn, messages=history)
            history.append({"role": "assistant", "content": r.text})
            last = r.text
            lat += r.latency_ms
            tok += r.tokens
            cost += r.cost
        transcript = json.dumps(history[:-1], ensure_ascii=False)[:4000]
        return history[-1]["content"], transcript, lat, tok, cost

    if turns:
        history = []
        if prompt_fn:
            history.append({"role": "system", "content": prompt_fn(c)})
        lat = tok = 0
        cost = 0.0
        for t in turns:
            history.append({"role": "user", "content": t["content"]})
            mfn = (lambda cc=c, hh=history: mock_reply_fn(cc, hh)) if mock else None
            r = call_llm(None, model=model, mock_fn=mfn, messages=history)
            history.append({"role": "assistant", "content": r.text})
            lat += r.latency_ms
            tok += r.tokens
            cost += r.cost
        transcript = json.dumps(history[:-1], ensure_ascii=False)[:4000]
        return history[-1]["content"], transcript, lat, tok, cost

    prompt = prompt_fn(c) if prompt_fn else c.get("input", {}).get("raw_ocr_text", "")
    mfn = (lambda cc=c: mock_reply_fn(cc)) if mock else None
    r = call_llm(prompt, model=model, mock_fn=mfn)
    return r.text, prompt, r.latency_ms, r.tokens, r.cost


def run_stage(product_cfg: dict, stage: StageConfig, model="", limit=None,
              mock=False, prompt_fn=None, mock_reply_fn=None, target: "HttpTarget | None" = None,
              repeat: int = 1, dataset_snapshot=None, run_snapshot=None) -> str:
    """跑一个环节。

    target 不为空时，走「你的产品接口」而不是调模型——这时的分数才是端到端的产品表现。
    repeat>1 时同一批题跑 N 遍：单次结果全存，判定时先按题聚合再算指标，
    额外给出 pass@N（至少一次对）和 pass^N（次次都对）。模型有随机性，跑一遍的分数不可信。
    """
    init_db()
    con = sqlite3.connect(DB)
    ds = dataset_snapshot if dataset_snapshot is not None else load_dataset(stage.dataset)
    cases = ds["cases"][:limit] if limit else ds["cases"]
    repeat = max(1, int(repeat or 1))
    run_id = f"{product_cfg['product']}-{stage.id}-{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.md5(str(time.time()).encode()).hexdigest()[:4]}"
    save_run(con, id=run_id, product=product_cfg["product"], stage=stage.id,
             dataset=redact(stage.dataset), model=redact("http" if target is not None else ("mock" if mock else model)),
             mode=("mock" if mock else ("http" if target is not None else "live")), started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
             finished_at="", n_cases=len(cases), cost=0.0)

    grader_spec = stage.grader  # 可能是 "a+b" 组合
    total_cost = 0.0
    for rep in range(repeat):
        for c in cases:
            try:
                result = _one_case(c, model, mock, prompt_fn, mock_reply_fn, target)
                out, transcript, lat, tok, cost = result[:5]
                trace = result[5] if len(result) > 5 else (c.get("mock_trace") if mock else None)
            except Exception as e:
                g = {"metrics": {}, "verdict": "fail", "fatal": [], "note": f"调用失败: {redact(str(e))}"}
                save_result(con, run_id, c["id"], c.get("stratum", ""), "", 0, 0,
                            g["metrics"], g["verdict"], g["fatal"], g["note"], rep, "")
                continue
            # 轨迹与元信息挂到「用例副本」上，评分器按需读取。
            # 必须用副本：同一个 case 会跑 N 遍，直接改原对象会串味。
            if trace is None and (out or "").lstrip().startswith("{"):
                try:
                    obj = json.loads(out)
                    trace = obj.get("trace") if isinstance(obj, dict) else None
                except ValueError:
                    pass
            c2 = dict(c)
            c2["_trace"] = trace
            c2["_meta"] = {"tokens": tok, "latency_ms": lat}
            out, transcript = redact(out), redact(transcript)
            g = redact(apply_graders(grader_spec, c2, out, stage))
            total_cost += cost
            save_result(con, run_id, c["id"], c.get("stratum", ""), out, lat, tok,
                        g["metrics"], g["verdict"], g["fatal"], g.get("note", ""), rep, transcript)

    con.execute("UPDATE run SET finished_at=?, cost=? WHERE id=?",
                (time.strftime("%Y-%m-%d %H:%M:%S"), total_cost, run_id))
    # 重复次数写进 meta，台账与报告要读
    try:
        row = con.execute("SELECT meta FROM run WHERE id=?", (run_id,)).fetchone()
        meta = json.loads(row[0]) if row and row[0] else {}
        meta["repeat"] = repeat
        if run_snapshot is not None:
            meta["snapshot"] = redact(run_snapshot)
        con.execute("UPDATE run SET meta=? WHERE id=?", (json.dumps(meta, ensure_ascii=False), run_id))
    except Exception:
        pass
    con.commit()
    con.close()
    return run_id

# ──────────────────────────── 判定 ────────────────────────────
def evaluate(con, run_id, stage: StageConfig) -> dict:
    """按 gate 汇总。fatal 优先于平均分。

    同一道题跑了 N 遍时，先按题聚合再算指标：
    指标取 N 次平均，判定要求「N 次全过」（pass^N）。
    另外单独给出 pass@N（至少一次对）和「不稳定的题」——后者才是改 prompt 时最该看的东西。
    """
    raw = get_results(con, run_id)
    repeat = 1
    for r in raw:
        repeat = max(repeat, int(r.get("repeat_idx") or 0) + 1)

    groups: dict[str, list] = {}
    for r in raw:
        groups.setdefault(r["case_id"], []).append(r)

    rows, n_at, n_all, unstable = [], 0, 0, []
    for cid, gs in groups.items():
        k = len(gs)
        pk = sum(1 for x in gs if x["verdict"] == "pass")
        if pk > 0:
            n_at += 1
        if k and pk == k:
            n_all += 1
        elif pk > 0:
            unstable.append({"case_id": cid, "pass": pk, "n": k, "stratum": gs[0]["stratum"]})
        # 指标：N 次平均（counters 也会变成小数，集合级比率仍成立）
        keys = set()
        for x in gs:
            keys |= set(x["metrics"].keys())
        m = {}
        for key in keys:
            vals = [x["metrics"].get(key) for x in gs
                    if isinstance(x["metrics"].get(key), (int, float))]
            if vals:
                m[key] = sum(vals) / len(vals)
        vd = ("fatal" if any(x["fatal"] or x["verdict"] == "fatal" for x in gs)
              else "fail" if any(x["verdict"] == "fail" for x in gs)
              else "review" if any(x["verdict"] == "review" for x in gs)
              else "pass" if k and pk == k else "review")
        rows.append({
            "case_id": cid, "stratum": gs[0]["stratum"], "metrics": m, "verdict": vd,
            "fatal": (gs[0]["fatal"] if vd == "fatal" else []),
            "note": next((x["note"] for x in gs if x["verdict"] != "pass"), gs[0]["note"]),
            "output": gs[0]["output"], "prompt": gs[0].get("prompt") or "",
            "latency_ms": gs[0].get("latency_ms") or 0,
            "n_pass": pk, "n_repeat": k,
        })

    n = len(rows) or 1

    # 指标汇总：counters 求和，其余取平均
    ckeys = set(stage.counters or [])
    agg_sum: dict[str, float] = {}
    agg_n: dict[str, int] = {}
    for r in rows:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)):
                agg_sum[k] = agg_sum.get(k, 0.0) + v
                agg_n[k] = agg_n.get(k, 0) + 1
    # counters 求和；其余按「该指标实际出现的条数」平均，而不是全局 n
    agg: dict[str, float] = {
        k: (v if k in ckeys else round(v / max(agg_n[k], 1), 4))
        for k, v in agg_sum.items()
    }

    # 集合级派生指标（precision / recall / bother_rate 这类不能按单条算）
    def _calc(expr: str) -> float | None:
        e = expr
        for k, v in sorted(agg.items(), key=lambda x: -len(x[0])):
            e = e.replace(k, str(v))
        if not all(c in "0123456789.+-*/() " for c in e):
            return None
        try:
            if "/0" in e.replace(" ", "") or e.strip().endswith("/0"):
                return None
            return round(float(eval(e, {"__builtins__": {}}, {})), 4)
        except Exception:
            return None

    import re as _re
    for d in (stage.derived or []):
        # 关键：分母为 0 的计数根本不会出现在 agg 里，必须先补 0，否则公式算不出来
        for ident in set(_re.findall(r"[A-Za-z_]\w*", d["formula"])):
            agg.setdefault(ident, 0)
        val = _calc(d["formula"])
        if val is not None:
            agg[d["key"]] = val

    fatals = []
    for r in rows:
        for f in r["fatal"]:
            fatals.append({"case_id": r["case_id"], "code": f, "note": r["note"]})

    checks = []
    for g in stage.gates:
        val = agg.get(g["key"])
        ok = val is not None and (val >= g["value"] if g["op"] == ">=" else val <= g["value"])
        checks.append({**g, "actual": val, "pass": bool(ok)})

    passed = bool(rows) and all(c["pass"] for c in checks) and not fatals and all(r["verdict"] == "pass" for r in rows)

    # 按层统计通过率，用于定位失效集中在哪里
    by_stratum: dict[str, dict] = {}
    for r in rows:
        s = r["stratum"] or "-"
        d = by_stratum.setdefault(s, {"n": 0, "pass": 0, "fail": 0, "fatal": 0})
        d["n"] += 1
        if r["fatal"]:
            d["fatal"] += 1
        elif r["verdict"] == "pass":
            d["pass"] += 1
        else:
            d["fail"] += 1

    return {
        "run_id": run_id, "n": len(rows), "n_runs": len(raw), "repeat": repeat,
        "metrics": agg, "checks": checks,
        "fatals": fatals, "by_stratum": by_stratum, "passed": passed,
        "pass_at_k": round(n_at / n, 4) if groups else None,
        "pass_pow_k": round(n_all / n, 4) if groups else None,
        "unstable": unstable,
        "verdict": "合格" if passed else "不合格",
    }


# ═══════════════════ HTTP 目标：直接打你自己的产品接口 ═══════════════════
# 有这一层，评测的才是「你的产品」，而不是「模型 + prompt」。
# 你的产品内部怎么调模型、怎么做规则校验、怎么后处理，全部保留 —— 这才叫端到端。
@dataclass
class HttpTarget:
    url: str
    method: str = "POST"
    headers: dict = field(default_factory=dict)
    body_template: dict | list | None = None   # None = 直接把 case.input 原样发过去
    response_path: str = ""                    # 如 "data.advice"、"result.0.text"
    timeout: int = 30


def _dig(obj, path: str):
    """按 a.b.0.c 取值，取不到返回 None。"""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _fill(tpl, inp):
    """把 body_template 里的 "{input.item}" / "{item}" 替换成真实值，保留原始类型。"""
    if isinstance(tpl, dict):
        return {k: _fill(v, inp) for k, v in tpl.items()}
    if isinstance(tpl, list):
        return [_fill(v, inp) for v in tpl]
    if isinstance(tpl, str):
        if tpl.startswith("{") and tpl.endswith("}"):
            key = tpl[1:-1]
            if key.startswith("input."):
                key = key[6:]
            if key == "input":
                return inp
            return inp.get(key, tpl)
        return tpl
    return tpl


def call_http(target: HttpTarget, case_input: dict, include_trace=False):
    """调用被测产品接口，返回 (用于判分的文本, 耗时毫秒)。"""
    body = _fill(target.body_template, case_input) if target.body_template is not None else case_input
    data = None if target.method.upper() == "GET" else json.dumps(body, ensure_ascii=False).encode()
    url = target.url
    if target.method.upper() == "GET" and body:
        from urllib.parse import urlencode
        url += ("&" if "?" in url else "?") + urlencode(
            {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
             for k, v in body.items()})

    req = urllib.request.Request(url, data=data, method=target.method.upper())
    req.add_header("Content-Type", "application/json")
    for k, v in (target.headers or {}).items():
        req.add_header(k, str(v))

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=target.timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"你的接口返回 HTTP {e.code}") from None
    except Exception as e:
        raise RuntimeError(f"产品接口连接失败：{type(e).__name__}") from None

    raw = redact(raw)
    lat = int((time.time() - t0) * 1000)
    try:
        obj = json.loads(raw)
    except Exception:
        return (raw, lat, None) if include_trace else (raw, lat)

    got = _dig(obj, target.response_path) if target.response_path else obj
    if got is None:
        raise RuntimeError(
            f"响应字段路径 '{target.response_path}' 取不到值。原始返回：{raw[:300]}")
    text = json.dumps(got, ensure_ascii=False) if isinstance(got, (dict, list)) else str(got)
    trace = obj.get("trace") if isinstance(obj, dict) else None
    return (text, lat, trace) if include_trace else (text, lat)
