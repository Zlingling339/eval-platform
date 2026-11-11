#!/usr/bin/env python3
"""把两版示例 prompt 写进平台（product=standalone_model, stage=model）。

为什么用独立 product 值：prompt 版本表要求 product/stage 成对出现，
而独立模型评测不属于任何业务产品。用 standalone_model + model 这两个值，
Prompt 管理页和模型评测页的「独立模型」列表都能看到它们。

用法：
    python server.py                     # 另开一个终端启动平台
    python tools/seed_prompts.py         # 默认 http://127.0.0.1:8765
    python tools/seed_prompts.py --base http://127.0.0.1:9000
    python tools/seed_prompts.py --dry-run   # 只打印，不发请求
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    ("v1-简洁直答", "压住解释性输出，只给最终答案；用于验证「短输出」是否影响格式遵循与成本",
     ROOT / "examples" / "prompts" / "prompt-v1-简洁直答.txt"),
    ("v2-结构化审慎", "先内部核对再按固定格式输出；用于验证格式约束与「信息不足」是否减少编造",
     ROOT / "examples" / "prompts" / "prompt-v2-结构化审慎.txt"),
]


def post(base, payload, dry):
    if dry:
        print("[dry-run]", json.dumps(payload, ensure_ascii=False)[:120], "...")
        return {"id": "(dry-run)"}
    req = urllib.request.Request(
        base.rstrip("/") + "/api/prompts",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"创建失败 HTTP {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        raise SystemExit(f"连不上平台（{base}）：{e.reason}\n先启动 python server.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for version, note, path in PROMPTS:
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        tpl = path.read_text(encoding="utf-8").strip()
        assert "{input}" in tpl, f"{path.name} 必须含 {{input}}"
        res = post(a.base, {"product": "standalone_model", "stage": "model",
                            "version": version, "note": note, "template": tpl}, a.dry_run)
        print(f"✓ {version} → id={res.get('id')}")

    print("\n完成。打开「开始评测 → 模型评测」，Prompt 版本下拉即可看到这两版。")


if __name__ == "__main__":
    main()
