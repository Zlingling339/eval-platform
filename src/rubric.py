"""Rubric 体系：把「什么算好」写成数据，让模型只做执行者。

一句话说清 rubric 是什么：
    不是打分表，是**判定表**。把「好不好」拆成几条互不重叠、看得见的表现，
    并为每条写清「什么样算 2 分、什么样只配 1 分、什么样直接 0 分」。
    评委（任何模型）要做的事，从「给个分」降级成「挑一档」——
    挑档是判定任务，比打分稳得多，也才能跨模型比较。

三条硬规则（写 rubric 时自查）：
1. 换一个评委评同一条，会不会得到同一档？会 → 锚点够了；不会 → 重写。
2. 档位少、档间距离大。三档（2/1/0）比五分制稳，因为「3 分和 4 分的区别」没人说得清。
3. 能写成断言的，绝不做成 rubric。「有没有引用」是断言，「引用得恰不恰当」才是 rubric。

文件位置：config/rubrics/<name>.yaml
每个维度里的 mock_hint 只服务于离线验证（让 mock 分数有区分度），真实模型时被忽略。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUBRIC_DIR = ROOT / "config" / "rubrics"

# 三档制。留出 "unknown" 给证据不足的情况 —— 逼评委硬判只会得到假分。
LEVELS = (2, 1, 0)
PASS_SCORE = 0.70

# 离线判分用的「含糊表达」表：说了但没给准数、没给结论。
# 只服务 mock，不进真实评分标准 —— 真实 judge 靠 levels 里的锚点判断。
VAGUE_WORDS = ("大概", "左右", "一般来说", "可能", "估计", "差不多", "一部分",
               "应该", "似乎", "听说", "据了解", "大概是", "大约是", "再提交一下")


@dataclass
class Dimension:
    key: str
    label: str
    weight: float
    levels: dict[int, str]
    gate: bool = False            # 这一维度 0 分 → 整体不及格，不被平均值稀释
    evidence: str = ""            # quote / number / none：要求评委给出依据
    mock_hint: dict[int, list[str]] = field(default_factory=dict)

    def norm(self, level: int) -> float:
        return max(0.0, min(1.0, level / max(LEVELS)))


@dataclass
class Rubric:
    id: str
    label: str
    version: int
    applies_to: list[str]
    dimensions: list[Dimension]
    threshold: float = PASS_SCORE
    note: str = ""

    @property
    def max_score(self) -> float:
        return sum(d.weight for d in self.dimensions) or 1.0

    def render(self, case: dict, output: str) -> str:
        """拼 judge prompt。要求只输出 JSON，且每维必须给依据。"""
        lines = [
            "你是一名严格的评测员。你的任务不是评价「感觉好不好」，",
            "而是按给定的判定表，逐条挑一档。不要自己发明标准。",
            "",
            f"# 判定表：{self.label}（{self.id} v{self.version}）",
            "",
            "## 待评对象",
            f"【用户输入】\n{_brief(case.get('input'))}",
        ]
        if case.get("gold"):
            lines.append(f"【参考答案/金标】\n{_brief(case.get('gold'))}")
        if case.get("docs"):
            lines.append(f"【参考资料】\n{_brief(case.get('docs'))}")
        lines.append(f"【被测输出】\n{_brief(output)}")

        lines += ["", "## 判定项"]
        for i, d in enumerate(self.dimensions, 1):
            tag = "（一票否决项：这一项判 0，整体直接不及格）" if d.gate else ""
            lines.append(f"### {i}. {d.label}{tag}")
            for lv in LEVELS:
                lines.append(f"  {lv} 分：{d.levels.get(lv, '')}")
            if d.evidence == "quote":
                lines.append("  ↳ 必须引用被测输出的原文片段作为依据")
            elif d.evidence == "number":
                lines.append("  ↳ 必须给出具体数字或字段名作为依据")

        keys = [d.key for d in self.dimensions]
        lines += [
            "",
            "## 输出格式（只输出 JSON，不要任何解释性文字）",
            "{",
            '  "scores": {' + ", ".join(f'"{k}": <0|1|2 或 "unknown">' for k in keys) + "},",
            '  "evidence": {' + ", ".join(f'"{k}": "<依据原文>"' for k in keys) + "},",
            '  "reasoning": "<一句话总结扣分原因>",',
            '  "confidence": <0.0-1.0>',
            "}",
            "",
            "证据不足时该维度填 \"unknown\"，不要硬判。",
        ]
        return "\n".join(lines)

    def score(self, judge_out: str) -> dict:
        """把 judge 的 JSON 变成分数。解析失败不抛异常，标 invalid 让上层决定。"""
        data = _extract_json(judge_out)
        if not isinstance(data, dict) or "scores" not in data:
            return {"ok": False, "raw": judge_out[:500], "score": 0.0,
                    "verdict": "invalid", "note": "评委输出不是合法 JSON"}

        scores = data.get("scores") or {}
        evidence = data.get("evidence") or {}
        dims, total, gate_failed, unknown = [], 0.0, [], []

        for d in self.dimensions:
            v = scores.get(d.key)
            if v == "unknown" or v is None:
                unknown.append(d.label)
                dims.append({"key": d.key, "label": d.label, "level": None,
                             "score": None, "weight": d.weight,
                             "evidence": evidence.get(d.key, "")})
                continue
            try:
                lv = int(float(v))
            except (TypeError, ValueError):
                unknown.append(d.label)
                dims.append({"key": d.key, "label": d.label, "level": None, "score": None,
                             "weight": d.weight, "evidence": evidence.get(d.key, "")})
                continue
            lv = max(0, min(max(LEVELS), lv))
            s = d.norm(lv)
            total += d.weight * s
            if d.gate and lv == 0:
                gate_failed.append(d.label)
            dims.append({"key": d.key, "label": d.label, "level": lv, "score": round(s, 3),
                         "weight": d.weight, "evidence": evidence.get(d.key, "")})

        # 缺证据的维度按 0 计入分母的权重，但不冤枉成 0 分事件 —— 交给人工
        raw_score = total / self.max_score
        verdict = "pass"
        note = str(data.get("reasoning", ""))[:120]
        if gate_failed:
            verdict, note = "fail", f"否决项未达标：{'、'.join(gate_failed)}"
        elif unknown:
            verdict = "review"
            note = f"证据不足待人工：{'、'.join(unknown)}"
        elif raw_score < self.threshold:
            verdict = "fail"

        return {
            "ok": True, "score": round(raw_score, 3), "verdict": verdict,
            "dimensions": dims, "gate_failed": gate_failed,
            "unknown": unknown, "note": note,
            "confidence": data.get("confidence"),
        }

    def mock_score(self, case: dict, output: str) -> str:
        """离线用的近似评委。

        有区分度是硬要求：如果 mock 对每条都给同一档，链路「跑通」是假信号，
        看不出评分器能不能分辨好坏。

        判分取向刻意贴近真实评委：**默认按达标处理，只有看到问题时才扣。**
        如果反过来（默认 1 分、命中好词才升），好答案会因为「没说出某句话」
        被冤枉扣分 —— 那是关键词搜索，不是评分。
        """
        low = (output or "").lower().strip()

        # trace 是确定性事实，先读它。真实 judge 也会先看轨迹再读文本，
        # 只是它用推理完成这一步；这里用规则完成，结论应当一致。
        steps = case.get("_trace") or {}
        steps = steps.get("steps") if isinstance(steps, dict) else steps
        calls = [s for s in (steps or []) if isinstance(s, dict) and s.get("type") in ("tool_call", "tool")]
        errored = any(("error" in str(s.get("result", "")).lower()) or ("失败" in str(s.get("result", "")))
                      for s in calls)
        claims_success = any(w in low for w in ("已完成", "已提交", "已经处理好了", "提交完成", "已生成"))

        scores, evidence = {}, {}
        for d in self.dimensions:
            lv, ev = 2, ""                      # 默认达标
            if not low:
                lv, ev = 0, "（无内容）"
            elif self.id == "tool_call" and not calls:
                # 整道题一个工具都没调，工具调用质量的每个维度都无从谈起。
                # 不这样处理的话，一个纯靠记忆作答的答案会因为「没说出坏话」拿到高分。
                lv, ev = 0, "全程未调用任何工具"
            elif errored and claims_success and d.key in ("result_use", "claim_consistency"):
                lv, ev = 0, "工具已失败，回复仍声称成功"
            elif self.id == "trajectory_process" and d.key == "path_coverage" and _missing_steps(case, calls):
                lv, ev = 0, "缺关键步骤：" + "、".join(_missing_steps(case, calls)[:3])
            else:
                zero = [w for w in d.mock_hint.get(0, []) if w.lower() in low]
                zero_is_empty = d.mock_hint.get(0) == ["__empty__"]
                vague = [w for w in VAGUE_WORDS if w in low]
                if zero or (zero_is_empty and not low):
                    lv, ev = 0, (zero[0] if zero else "（无内容）")
                elif vague:
                    # 含糊表达是「中等」的信号：说了但没给准数、没给结论
                    lv, ev = 1, vague[0]
                else:
                    strong = [w for w in d.mock_hint.get(2, []) if w.lower() in low]
                    ev = strong[0] if strong else ""
            scores[d.key] = lv
            evidence[d.key] = (ev or d.levels.get(lv, ""))[:60]
        return json.dumps({"scores": scores, "evidence": evidence,
                           "reasoning": "mock 判分（离线验证用，非真实模型结论）", "confidence": 0.6},
                          ensure_ascii=False)


# ─────────────────────────── 读写 ───────────────────────────

def load(name: str) -> Rubric:
    path = RUBRIC_DIR / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"rubric 不存在：{path}")
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dims = []
    for d in raw.get("dimensions") or []:
        dims.append(Dimension(
            key=d["key"], label=d.get("label", d["key"]),
            weight=float(d.get("weight", 1.0 / max(1, len(raw.get("dimensions") or [1])))),
            levels={int(k): str(v) for k, v in (d.get("levels") or {}).items()},
            gate=bool(d.get("gate", False)),
            evidence=d.get("evidence", ""),
            mock_hint={int(k): list(v) for k, v in (d.get("mock_hint") or {}).items()},
        ))
    return Rubric(
        id=raw.get("id", path.stem), label=raw.get("label", path.stem),
        version=int(raw.get("version", 1)), applies_to=raw.get("applies_to") or [],
        dimensions=dims, threshold=float(raw.get("threshold", PASS_SCORE)),
        note=raw.get("note", ""),
    )


def catalog() -> list[dict]:
    """所有可用 rubric 的目录，给界面选。"""
    out = []
    for p in sorted(RUBRIC_DIR.glob("*.yaml")):
        try:
            r = load(p.name)
            out.append({"id": r.id, "file": p.name, "label": r.label, "version": r.version,
                        "applies_to": r.applies_to, "dims": len(r.dimensions),
                        "gates": sum(1 for d in r.dimensions if d.gate), "note": r.note})
        except Exception as e:  # 坏文件不该让整个目录打不开
            out.append({"id": p.stem, "file": p.name, "label": f"（加载失败）{e}",
                        "version": 0, "applies_to": [], "dims": 0, "gates": 0, "note": ""})
    return out


def _missing_steps(case: dict, calls: list) -> list[str]:
    """关键路径里没走到的步骤。只用于 mock 近似判分。"""
    key_path = (case.get("expect") or {}).get("key_path") or []
    seen = {str(s.get("name") or "") for s in calls}
    return [k for k in key_path if k not in seen]


def _brief(v: Any, limit: int = 2400) -> str:
    if v is None:
        return "（无）"
    if isinstance(v, str):
        s = v
    else:
        s = json.dumps(v, ensure_ascii=False, indent=1)
    return s[:limit] + ("…" if len(s) > limit else "")


def _extract_json(text: str):
    """评委常把 JSON 包在 ```json 里，或前后加一句话。宽容提取。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
