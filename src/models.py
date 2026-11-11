"""多模型网关：被测 / 评委 / 审计 / 造题 四个角色 + 同源隔离硬校验。

设计原则（对应 docs/05-multi-model.md）：

1. **标准不来自评委。** rubric 是外部文件里的判定标准，模型只是执行者。
   换一个模型来执行同一份 rubric，问的还是同一组是非题 —— 这才让跨模型互评比
   「同模型自评」更可信。

2. **同源禁评。** 同一个模型评自己会宽容（自我强化偏差）。所以：
   - 完全同一 (base_url, model)  → BLOCK，直接拒绝启动
   - 同厂商（如 gpt-4o 与 gpt-4o-mini）→ WARN，界面标红但仍允许（多数团队只有一个备用模型）
   - 不同厂商 → OK

3. **mock 必须可复现。** 链路验证阶段不联网，但分数要有区分度，
   不能所有样本都给同一个分 —— 否则「跑通了」是假信号。

配置文件：config/models.yaml
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "models.yaml"

# 从 base_url 推断厂商。用于同源判定 —— 同一家的模型有共同偏好。
VENDOR_HINTS = [
    ("tencent", ("tencent", "hunyuan", "qq.com")),
    ("deepseek", ("deepseek",)),
    ("openai", ("openai", "azure")),
    ("aliyun", ("aliyun", "dashscope", "qwen")),
    ("zhipu", ("zhipu", "bigmodel")),
    ("moonshot", ("moonshot",)),
    ("bytedance", ("volces", "bytedance", "ark.")),
    ("anthropic", ("anthropic", "claude")),
    ("google", ("googleapis", "google")),
    ("baidu", ("baidu", "qianfan")),
]


def infer_vendor(base_url: str, model: str = "") -> str:
    hay = (base_url + " " + model).lower()
    for vendor, keys in VENDOR_HINTS:
        if any(k in hay for k in keys):
            return vendor
    host = base_url.split("//")[-1].split("/")[0]
    return host.split(".")[0] if host else "unknown"


class SiblingModelError(RuntimeError):
    """评委与被测同源 —— 这个分数不可信，必须拒绝启动。"""


@dataclass
class ModelSpec:
    key: str
    name: str
    base_url: str = ""
    model: str = ""
    key_env: str = ""
    adapter: str = "openai_compat"  # openai_compat | mock
    vendor: str = ""
    note: str = ""

    def __post_init__(self):
        if not self.vendor:
            self.vendor = infer_vendor(self.base_url, self.model)

    @property
    def fingerprint(self) -> tuple[str, str]:
        return (self.base_url.rstrip("/").lower(), self.model.lower())

    @property
    def is_mock(self) -> bool:
        return self.adapter == "mock"


@dataclass
class Independence:
    role: str
    peer: str
    level: str  # ok | warn | block
    reason: str = ""

    def as_dict(self) -> dict:
        return {"role": self.role, "peer": self.peer, "level": self.level, "reason": self.reason}


# 哪些角色之间必须保持相互独立
INDEPENDENCE_PAIRS = [
    ("judge", "target"),
    ("auditor", "target"),
    ("generator", "target"),
    ("auditor", "judge"),
]


class ModelRouter:
    """按角色发请求。所有评分侧调用都必须经过这里，避免某处偷偷用了被测模型。"""

    def __init__(self, specs: dict[str, ModelSpec], strict_vendor: bool = False):
        self.specs = specs
        self.strict_vendor = strict_vendor
        self.api_keys = {}

    # ── 独立性体检 ──
    def independence(self) -> list[Independence]:
        out = []
        for a, b in INDEPENDENCE_PAIRS:
            sa, sb = self.specs.get(a), self.specs.get(b)
            if not sa or not sb:
                continue
            if sa.fingerprint == sb.fingerprint:
                out.append(Independence(a, b, "block",
                                        f"{sa.name} 同时担任 {a} 和 {b}：同一模型评自己"))
            elif sa.vendor == sb.vendor:
                lvl = "block" if self.strict_vendor else "warn"
                out.append(Independence(a, b, lvl,
                                        f"{sa.name} 与 {sb.name} 同属 {sa.vendor}，存在同族偏好"))
            else:
                out.append(Independence(a, b, "ok",
                                        f"{sa.name}({sa.vendor}) vs {sb.name}({sb.vendor})"))
        return out

    def assert_startable(self) -> None:
        bad = [x for x in self.independence() if x.level == "block"]
        if bad:
            msg = "；".join(x.reason for x in bad)
            raise SiblingModelError(f"同源身份冲突，评测不可信：{msg}。请换成不同厂商的模型。")

    def role_of(self, role: str) -> ModelSpec:
        if role not in self.specs:
            raise KeyError(f"角色 {role} 未配置。已配置：{sorted(self.specs)}")
        return self.specs[role]

    # ── 实际调用 ──
    def call(self, role: str, messages: list[dict] | str, temperature: float = 0.0,
             timeout: int = 60, max_tokens: int | None = None,
             mock_fn=None) -> dict:
        """返回 {text, latency_ms, tokens, cost, model, vendor, mock}。"""
        spec = self.role_of(role)
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages

        if spec.is_mock or mock_fn is not None:
            t0 = time.time()
            text = mock_fn() if mock_fn is not None else _mock_reply(role, spec, msgs)
            return {
                "text": text,
                "latency_ms": int((time.time() - t0) * 1000) + 600,
                "tokens": max(1, len(text) // 2),
                "cost": 0.0,
                "model": spec.model or spec.name,
                "vendor": spec.vendor,
                "mock": True,
            }

        key = self.api_keys.get(role, "")
        if not key:
            raise RuntimeError(
                f"角色 {role}（{spec.name}）缺少请求级 API Key，"
                f"或在 config/models.yaml 把它改成 adapter: mock")
        body = {"model": spec.model, "messages": msgs, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        req = urllib.request.Request(
            spec.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"模型服务返回 HTTP {e.code}") from None
        latency = int((time.time() - t0) * 1000)
        usage = data.get("usage") or {}
        return {
            "text": __import__("src.core", fromlist=["redact"]).redact((data.get("choices") or [{}])[0].get("message", {}).get("content", "")),
            "latency_ms": latency,
            "tokens": usage.get("total_tokens", 0),
            "cost": 0.0,
            "model": spec.model,
            "vendor": spec.vendor,
            "mock": False,
        }


# ─────────────────────── mock：确定性但有区分度 ───────────────────────

def _stable_hash(*parts: str) -> int:
    h = hashlib.sha256("||".join(parts).encode()).hexdigest()
    return int(h[:12], 16)


MOCK_BANK = {
    "judge": [
        '{"faithfulness": 4, "relevance": 5, "completeness": 3, "reasoning": "事实基本可追溯，遗漏一个限定条件"}',
        '{"faithfulness": 5, "relevance": 4, "completeness": 4, "reasoning": "结论与资料一致，覆盖主要要点"}',
        '{"faithfulness": 2, "relevance": 4, "completeness": 2, "reasoning": "出现资料中没有的承诺"}',
    ],
    "auditor": [
        '{"ok": true, "issues": [], "reasoning": "题目与金标自洽"}',
        '{"ok": false, "issues": ["gold 与 forbidden 冲突"], "reasoning": "金标要求提及的点同时出现在禁用词表"}',
    ],
    "generator": [
        '{"cases": [{"id": "A01", "stratum": "normal", "input": {}, "gold": {}}]}',
    ],
}


def _mock_reply(role: str, spec: ModelSpec, msgs: list[dict]) -> str:
    """同一个 prompt + 同一个模型 → 同一结果；不同 prompt → 不同结果。

    这样链路验证时分数有分布，而不是一片同分（同分等于没测出区分力）。
    """
    last = msgs[-1]["content"] if msgs else ""
    seed = _stable_hash(spec.model or spec.name, last[:2000])
    bank = MOCK_BANK.get(role)
    if bank:
        return bank[seed % len(bank)]
    return f"[mock:{spec.name}] {seed % 1000}"


# ─────────────────────────── 配置读写 ───────────────────────────

DEFAULT_CONFIG = {
    "strict_vendor": False,
    "roles": {
        "target": {
            "name": "通用被测模型（模拟）", "adapter": "mock", "base_url": "",
            "model": "mock-target", "key_env": "", "vendor": "mock-target",
            "note": "被测产品本身，不具备任何评分权限",
        },
        "judge": {
            "name": "通用评委（模拟）", "adapter": "mock",
            "base_url": "", "model": "mock-judge",
            "key_env": "", "vendor": "mock-judge",
            "note": "执行 rubric 打分。禁止与被测同源",
        },
        "auditor": {
            "name": "Qwen（数据集审计）", "adapter": "mock",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus", "key_env": "DASHSCOPE_API_KEY", "vendor": "aliyun",
            "note": "审计造出来的题：gold 是否自洽、是否可判、是否有歧义",
        },
        "generator": {
            "name": "GLM（造题）", "adapter": "mock",
            "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-plus",
            "key_env": "ZHIPU_API_KEY", "vendor": "zhipu",
            "note": "造题。禁止与被测同源，否则等于自己出题自己考",
        },
    },
}


def load_router() -> ModelRouter:
    if CONFIG.exists():
        raw = json.loads(json.dumps(_yaml_load(CONFIG)))
    else:
        raw = DEFAULT_CONFIG
    specs = {}
    for role, cfg in (raw.get("roles") or {}).items():
        specs[role] = ModelSpec(key=role, **{k: v for k, v in cfg.items() if k != "key"} )
    return ModelRouter(specs, strict_vendor=bool(raw.get("strict_vendor")))


def save_config(raw: dict) -> None:
    import yaml
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _yaml_load(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def config_raw() -> dict:
    return _yaml_load(CONFIG) if CONFIG.exists() else DEFAULT_CONFIG
