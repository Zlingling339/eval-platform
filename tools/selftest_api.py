"""接口与功能自测（交付版自带，可反复运行）。

用法：
    PYTHONPATH=/sandbox/workspace/.pylibs python3 tools/selftest_api.py

做的事：
  1) 结构巡检：遍历所有 /api 路由，GET 直接请求、POST 发空体，
     断言不出现 5xx（4xx 属于正常的参数校验）。
  2) 功能巡检：三条评测路径各跑一次真实流程，外加分类指标（label_match）算数值。
  3) 内容巡检：报告与台账不得出现旧业务词，密钥不得落到报告/数据库。

脚本会先备份 data/ 与 reports/，跑完自动还原，不影响你自己的数据。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []


def rec(area: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"area": area, "name": name, "ok": bool(ok), "detail": detail})
    print(("  PASS " if ok else "  FAIL ") + f"[{area}] {name}" + (f"  << {detail}" if detail else ""))


def backup() -> tuple[Path, Path | None, Path | None]:
    tmp = Path(tempfile.mkdtemp(prefix="selftest-"))
    db = ROOT / "data" / "eval.db"
    rep = ROOT / "reports"
    b_db = tmp / "eval.db.bak"
    b_rep = tmp / "reports.bak"
    if db.exists():
        shutil.copy2(db, b_db)
    if rep.exists():
        shutil.copytree(rep, b_rep)
    return tmp, b_db, b_rep


def restore(tmp: Path, b_db: Path, b_rep: Path) -> None:
    db = ROOT / "data" / "eval.db"
    if b_db.exists():
        shutil.copy2(b_db, db)
    elif db.exists():
        db.unlink()
    rep = ROOT / "reports"
    if b_rep.exists():
        shutil.rmtree(rep, ignore_errors=True)
        shutil.copytree(b_rep, rep)
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    from fastapi.testclient import TestClient

    import server

    tmp, b_db, b_rep = backup()
    client = TestClient(server.app)

    # ── 0. 路由清单 ────────────────────────────────────────────
    routes: list[tuple[str, str]] = []
    for r in server.app.routes:
        path = getattr(r, "path", "")
        methods = getattr(r, "methods", set()) or set()
        if not path.startswith("/api"):
            continue
        for m in sorted(methods):
            if m in ("GET", "POST", "PUT", "DELETE"):
                routes.append((m, path))
    print(f"\n== 结构巡检：{len(routes)} 个 /api 路由 ==")

    # ── 1. 结构巡检 ────────────────────────────────────────────
    parametrised = ("{", "}")
    for method, path in routes:
        if any(t in path for t in parametrised):
            continue  # 参数化路由在功能巡检里覆盖
        if method == "GET":
            r = client.get(path)
        else:
            r = client.post(path, json={})
        ok = r.status_code < 500
        rec("接口", f"{method} {path}", ok, f"HTTP {r.status_code}")

    # ── 2. 准备：加载通用示例 + 上传两份数据集 ──────────────────
    print("\n== 功能巡检 ==")
    r = client.post("/api/examples/general")
    ok_example = r.status_code == 200 or "已存在" in r.text
    rec("准备", "加载通用示例", ok_example, str(r.json())[:60])

    def upload(name: str) -> str | None:
        p = ROOT / "examples" / name
        with open(p, "rb") as f:
            r = client.post(
                "/api/datasets",
                params={"product": "general-example", "stage": "answer"},
                files={"file": (name, f, "application/json")},
            )
        if r.status_code != 200:
            rec("准备", f"上传 {name}", False, f"HTTP {r.status_code} {r.text[:120]}")
            return None
        d = r.json()
        rec("准备", f"上传 {name}", True, f"{d.get('n_cases')} 条, 问题 {d.get('n_problems')}")
        did = d.get("id") or d.get("dataset_id")
        rr = client.post(f"/api/datasets/{did}/review", json={"reviewed": True})
        rec("准备", f"复核 {name}", rr.status_code == 200, f"HTTP {rr.status_code}")
        return did

    model_did = upload("smoke-model.json")
    agent_did = upload("smoke-agent.json")

    def run(product: str, stage: str, did: str, grader: str, mock: bool = True,
            limit: int | None = None, extra: dict | None = None) -> tuple[str | None, dict]:
        body = {
            "product": product, "stage": stage, "dataset_id": did,
            "grader": grader, "mock": mock, "repeat": 1,
        }
        if limit:
            body["limit"] = limit
        if extra:
            body.update(extra)
        r = client.post("/api/runs", json=body)
        if r.status_code != 200:
            return None, {"error": f"HTTP {r.status_code} {r.text[:160]}"}
        jid = r.json()["job_id"]
        for _ in range(120):
            st = client.get(f"/api/jobs/{jid}").json()
            if st.get("status") in ("done", "error"):
                return st.get("run_id"), st
            time.sleep(0.25)
        return None, {"error": "timeout"}

    def detail(rid: str) -> dict:
        return client.get(f"/api/runs/{rid}").json()

    def agg_metrics(rid: str) -> dict:
        """汇总指标在台账里（evaluate 的结果），运行详情只有逐条 metrics。"""
        led = client.get("/api/ledger", params={"product": "general-example", "stage": "answer"}).json()
        for row in led.get("rows", []):
            if row.get("run_id") == rid:
                return row.get("values") or {}
        return {}

    # ── 3. 模型路径 ────────────────────────────────────────────
    rid, st = run("general-example", "answer", model_did, "keyword_rule")
    if not rid:
        rec("模型评测", "运行", False, str(st))
    else:
        d = detail(rid)
        verdicts = [x.get("verdict") for x in d.get("results", [])]
        rec("模型评测", "运行并出结果", len(verdicts) == 2, f"verdicts={verdicts}")
        rec("模型评测", "通过/失败各一条", verdicts.count("pass") == 1 and verdicts.count("fail") == 1, str(verdicts))
        rr = client.get(f"/api/runs/{rid}/report")
        rec("模型评测", "报告可访问", rr.status_code == 200 and len(rr.text) > 2000, f"{len(rr.text)} 字符")
        MODEL_REPORT = rr.text

    # ── 4. Agent 路径（规则 + 轨迹）────────────────────────────
    rid2, st2 = run("general-example", "answer", agent_did, "tool_call_rule+trajectory_rule")
    if not rid2:
        rec("Agent评测", "运行", False, str(st2))
    else:
        d2 = detail(rid2)
        vs = [x.get("verdict") for x in d2.get("results", [])]
        rec("Agent评测", "运行并出结果", len(vs) == 3, f"verdicts={vs}")
        rec("Agent评测", "缺轨迹判待复核", vs.count("review") == 1, f"review={vs.count('review')}")
        # 缺轨迹那条本来就不该有指标，所以只看有轨迹的两条
        with_m = [x for x in d2.get("results", []) if x.get("metrics")]
        rec("Agent评测", "组合评分器未因前缀丢指标", len(with_m) == 2,
            f"有指标的条目 {len(with_m)}/3")
        rr2 = client.get(f"/api/runs/{rid2}/report")
        rec("Agent评测", "报告含轨迹段", rr2.status_code == 200 and ("轨迹" in rr2.text), "")

    # ── 5. 分类指标 label_match ────────────────────────────────
    cls = {
        "dataset": "selftest_cls", "version": "1.0",
        "cases": [
            {"id": "c-tp", "input": {"text": "a"}, "gold": {"label": "positive"},
             "mock_reply": "{\"label\":\"positive\"}"},
            {"id": "c-fp", "input": {"text": "b"}, "gold": {"label": "negative"},
             "mock_reply": "{\"label\":\"positive\"}"},
            {"id": "c-fn", "input": {"text": "c"}, "gold": {"label": "positive"},
             "mock_reply": "{\"label\":\"negative\"}"},
            {"id": "c-tn", "input": {"text": "d"}, "gold": {"label": "negative"},
             "mock_reply": "{\"label\":\"negative\"}"},
        ],
    }
    p = Path(tmp) / "cls.json"
    p.write_text(json.dumps(cls, ensure_ascii=False), encoding="utf-8")
    with open(p, "rb") as f:
        r = client.post("/api/datasets", params={"product": "general-example", "stage": "answer"},
                        files={"file": ("cls.json", f, "application/json")})
    cid = r.json().get("id") if r.status_code == 200 else None
    rec("分类指标", "上传四象限数据集", bool(cid), f"HTTP {r.status_code}")
    if cid:
        client.post(f"/api/datasets/{cid}/review", json={"reviewed": True})
        rid3, st3 = run("general-example", "answer", cid, "label_match")
        if not rid3:
            rec("分类指标", "运行", False, str(st3))
        else:
            m = agg_metrics(rid3)
            need = {k: m.get(k) for k in ("tp", "fp", "fn", "tn")}
            rec("分类指标", "混淆矩阵计数", need == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}, str(need))
            # 四格各 1：accuracy=0.5 precision=0.5 recall=0.5 f1=0.5
            vals = {k: m.get(k) for k in ("accuracy", "precision", "recall", "f1")}
            okv = all(v is not None and abs(float(v) - 0.5) < 0.01 for v in vals.values())
            rec("分类指标", "accuracy/precision/recall/f1 数值正确", okv, str(vals))
            rr3 = client.get(f"/api/runs/{rid3}/report")
            rec("分类指标", "报告出现分类指标区块", "F1" in rr3.text and "召回率" in rr3.text, "")

    # ── 6. 产品界面评测（真浏览器 + 本地页面）──────────────────
    page_dir = Path(tmp) / "site"
    page_dir.mkdir()
    (page_dir / "index.html").write_text(
        """<!doctype html><meta charset=utf-8><title>selftest</title>
<h1 id=h>演示页</h1><input id=mail placeholder=mail>
<button id=go disabled>开始</button><div id=out></div>
<script>
const m=document.getElementById('mail'), b=document.getElementById('go'), o=document.getElementById('out');
m.oninput=()=>b.disabled=!m.value.trim();
b.onclick=()=>o.textContent='已开通';
</script>""", encoding="utf-8")

    port = 8931
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                           cwd=str(page_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.2)
        r = client.post("/api/product-ui/run", json={
            "url": f"http://127.0.0.1:{port}/",
            "steps": [
                {"action": "check_text", "selector": "#h", "text": "演示页"},
                {"action": "fill", "selector": "#mail", "value": "a@b.com"},
                {"action": "check_button", "selector": "#go", "enabled": True},
                {"action": "click", "selector": "#go"},
                {"action": "check_text", "selector": "#out", "text": "已开通"},
                {"action": "screenshot"},
            ],
        })
        if r.status_code != 200:
            rec("界面评测", "运行", False, f"HTTP {r.status_code} {r.text[:160]}")
        else:
            d = r.json()
            steps = d.get("steps") or []
            rec("界面评测", "六步全部执行", len(steps) == 6, f"{len(steps)} 步")
            rec("界面评测", "全部步骤通过", all(s.get("ok") for s in steps), str([s.get("ok") for s in steps]))
            rec("界面评测", "截图已返回", bool(any(s.get("png_base64") for s in steps)), "")
            rec("界面评测", "记录了耗时", bool(d.get("elapsed_ms")), f"{d.get('elapsed_ms')} ms")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()

    # ── 7. 内容巡检 ────────────────────────────────────────────
    print("\n== 内容巡检 ==")
    banned = ["心愿基金", "劝购", "打扰率", "陷阱识别", "金额一致率", "小票", "借贷", "征信"]
    reports = list((ROOT / "reports").glob("*.html"))
    hits: list[str] = []
    for f in reports:
        t = f.read_text(encoding="utf-8", errors="ignore")
        for w in banned:
            if w in t:
                hits.append(f"{f.name}:{w}")
    rec("内容", f"{len(reports)} 份报告无旧业务词", not hits, ";".join(hits[:4]))

    ov = client.get("/api/overview")
    led = client.get("/api/ledger", params={"product": "general-example", "stage": "answer"})
    blob = json.dumps([ov.json(), led.json()], ensure_ascii=False)
    bad2 = [w for w in banned if w in blob]
    rec("内容", "总览与台账无旧业务词", not bad2, ",".join(bad2))

    # 密钥不落库、不进报告
    fake = "sk-SELFTEST-0123456789abcdef"
    rid4, st4 = run("general-example", "answer", model_did, "keyword_rule",
                    extra={"api_key": fake, "model": "fake-target",
                           "target": {"url": "http://127.0.0.1:9/x", "method": "POST"}})
    if rid4:
        rr = client.get(f"/api/runs/{rid4}/report")
        rec("安全", "假密钥不出现在报告", fake not in rr.text, "")
        rd = client.get(f"/api/runs/{rid4}")
        rec("安全", "假密钥不出现在运行详情", fake not in json.dumps(rd.json(), ensure_ascii=False), "")
        db = (ROOT / "data" / "eval.db").read_bytes()
        rec("安全", "假密钥不落数据库", fake.encode() not in db, "")

    # ── 汇总 ──────────────────────────────────────────────────
    restore(tmp, b_db, b_rep)
    passed = sum(1 for x in RESULTS if x["ok"])
    total = len(RESULTS)
    print(f"\n=== {passed}/{total} 通过 ===")
    fails = [x for x in RESULTS if not x["ok"]]
    if fails:
        print("失败项：")
        for x in fails:
            print(f"  - [{x['area']}] {x['name']}  {x['detail']}")
    out = ROOT / "selftest-report.json"
    out.write_text(json.dumps({"passed": passed, "total": total, "items": RESULTS},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
