# 评测台 · LLM / Agent / 页面评测

一个本地部署、不强制绑定产品的评测平台：同一套评测集和评分器，分别测**模型回答**、**Agent 工具轨迹**、**产品界面操作**，每次运行都记成可对比的版本。

> 面向个人开发者和小团队：不追求榜单排名，只回答三个问题——这次改动有没有用、坏在哪一条、能不能复现。

---

## 为什么做这个

调 prompt、换模型、改 Agent 工具，最常见的问题是「感觉变好了」：

- 分数涨了，但不知道是真变好还是随机波动；
- 报告说成功，但 Agent 其实没调工具、状态也没变；
- 换了一版 prompt，结果和上一版根本不可比（数据集、评分器、重复次数都变了）。

这个平台把「一次评测」固化成一件事：**固定的题 + 固定的判分规则 + 固定的模型/接口 + 可对比的版本**。只改一个变量时，差异才能归因。

---

## 三条评测路径

| 路径 | 测什么 | 怎么接 | 产物 |
|---|---|---|---|
| 模型评测 | 模型对题目的回答 | OpenAI 兼容 API（base_url + model + key） | 准确率/F1、关键词、格式、rubric 分 |
| Agent 评测 | 工具调用、步骤顺序、终态 | 被测 Agent 的 HTTP 接口（返回 trace / state） | 工具合规、路径召回、终态一致 |
| 产品界面评测 | 页面能不能点通 | 网址 + 步骤（fill/click/check_text/screenshot） | 每步结果、截图、耗时 |

三条路径**分开出报告，不合成一个总分**。指标同名不同义（都叫 precision，算法可能不同），横向比分数是错的。

---

## 快速开始

```bash
pip install -r requirements.txt
python server.py            # 默认 http://127.0.0.1:8765
```

Windows：

```bat
cd /d D:\eval-platform
python -m pip install -r requirements.txt
python server.py
```

空库里没有预装产品，直接可用的是**独立评测**：

1. 「准备评测 → 测试集」上传 JSON 数据集（格式见下），点「已复核」；
2. 「开始评测 → 模型评测 / Agent 评测」，选「独立…评测」；
3. 选数据集、评分器、模型或接口地址，运行；
4. 「查看结果 → 运行记录与报告」「版本记录」看结果与对比。

想体验产品环节：`curl -X POST http://127.0.0.1:8765/api/examples/general` 会装一个通用问答示例产品。

### 数据集最小格式

```json
{
  "dataset": "ads-smoke-v1",
  "cases": [
    {
      "id": "a-01",
      "input": { "text": "加微信 xxx 领优惠券，全场五折。" },
      "gold": { "label": "positive" }
    }
  ]
}
```

- `input`：题目或素材（对象；纯字符串也支持，会自动包成 `{"text": ...}`）
- `gold`：标准答案，评分器读，**不会发给模型**
- `expect`：更细的判定条件（`must_mention` / `required` / `allowed_tools` / `key_path` / `expect_state` …）

---

## 核心概念

```
评测集的一条 case
  ├─ input  ──套进 prompt 模板──▶ 模型/Agent ──▶ 输出 ──▶ 评分器
  └─ gold / expect ─────────────────────────────────────▶ 评分器
```

- **评测集**回答「测什么」，**prompt** 回答「怎么问」，**评分器**回答「怎么判」；
- 换 prompt 才是 prompt 对比；换评测集就是换题，分数不可直接比；
- 平台用数据集内容哈希、评分器、重复次数、scope、限条数校验可比性，不一致直接拒绝。

两版示例 prompt 在 `examples/prompts/`，一键导入：

```bash
python tools/seed_prompts.py
```

---

## Agent 接入契约

被测 Agent 返回：

```json
{
  "output": "杭州明天小雨，建议带伞。",
  "trace": {
    "steps": [
      {"type": "tool_call", "name": "get_weather",
       "args": {"city": "杭州", "date": "明天"}, "result": "小雨 22~27℃"},
      {"type": "final", "content": "杭州明天小雨，建议带伞。"}
    ]
  },
  "state": {"ticket_status": "open"}
}
```

页面里填：接口地址 + 方法 + 响应字段路径（评工具填 `output`，评终态填 `state`）+ 请求体模板 `{"question": "{question}"}`。

- `trace` 必须在该响应**顶层**；没有 trace 判「待复核」，不判失败；
- `allowed_tools` 是白名单，要列全 Agent 的所有工具，否则正常调用会被误判越权；
- 完整说明与排错见 `examples/agent接口契约.txt`。

示例数据：`agent-tool-real.json`（工具与路径）、`agent-guard-real.json`（边界与越权）、`agent-outcome-real.json`（终态）。

---

## 评分器（14 个）

**确定性规则**（同输入同结果，不花钱）

| 评分器 | 判什么 |
|---|---|
| `keyword_rule` | 必需词是否出现、禁用词是否避免 |
| `format_rule` | JSON 合法性、必填字段、枚举、长度 |
| `label_match` | 二分类混淆矩阵 → accuracy / precision / recall / F1 |
| `citation_rule` | 引用是否来自给定材料、有没有编造 |
| `outcome_rule` | 任务结束后的结构化终态 |
| `retrieval_rule` | Hit@k / Recall@k / Precision@k / MRR |
| `efficiency_rule` | token、延迟、步骤是否超预算 |
| `tool_call_rule` | 必需工具、必需参数、白名单、调用次数 |
| `trajectory_rule` | 关键步骤是否按序出现（LCS 对齐） |

**需要评委模型**（未配评委时判「待复核」，不假装给分）

`llm_rubric`、`rag_judge`、`tool_call_judge`、`trajectory_judge`、`pair_compare`

可用 `+` 组合，如 `tool_call_rule+trajectory_rule`。

---

## 目录结构

```
server.py              FastAPI 入口与全部 API
web/index.html         单页前端（无构建步骤）
src/core.py            运行、判分、聚合、存储
src/graders_rule.py    确定性规则评分器
src/graders_llm.py     LLM 评委评分器
src/models.py          多模型网关与同源校验
src/report.py          HTML 报告
config/                模型角色、产品、rubric 配置
examples/              数据集与 prompt 示例
tests/                 pytest
tools/                 selftest_api.py、seed_prompts.py 等
data/                  SQLite 与上传文件（运行时生成）
reports/               HTML 报告（运行时生成）
```

---

## 安全与边界

- 服务默认只监听 `127.0.0.1`，**没有登录与权限控制，不要直接暴露公网**；
- API Key 存在浏览器 localStorage，运行时随请求临时发送；localStorage 不是密钥库，同源脚本可读，共用设备请用完清空；
- 服务端不把 Key 写数据库、报告或日志，报告里接口地址只留主机名；
- 产品界面执行器没有鉴权、没有 SSRF 防护，只能测你有权测试的地址；
- 任务状态在进程内存里，重启会丢「进行中」的任务。

---

## 自测

```bash
python -m pytest tests -q
python tools/selftest_api.py          # 路由、三条路径、指标、密钥不落库
python web/test_frontend_paths.py     # 前端静态契约
```

界面评测需要浏览器：`python -m playwright install chromium`，或设置
`PRODUCT_UI_CHROMIUM_EXECUTABLE=/usr/bin/chromium` 用系统 Chromium。

---

## 已知限制

- 未内置任何真实厂商的连通性测试；接口兼容性与计费以供应商文档为准；
- 评测结果是「你的题 + 你的评分器」下的结论，样本量小（<30 条）时几个百分点的差异没有统计意义；
- `label_match` 只做二分类，多分类请用 `format_rule` 或自定义 rubric；
- `run.py` / `app.py` 两个命令行入口依赖 `src/products.py` 的 Python 适配器，本仓库该注册表为空，Web 界面不需要写代码。

---

## License

本仓库未附许可证文件。对外发布前请按你的需要补充（MIT / Apache-2.0 等），并确认示例数据不含第三方版权内容。
