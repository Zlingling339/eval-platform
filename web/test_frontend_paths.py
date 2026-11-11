"""前端静态契约检查；不替代真实后端或浏览器联调。"""
from pathlib import Path
import re, subprocess
s = (Path(__file__).parent / 'index.html').read_text()
script = re.search(r'<script>(.*?)</script>', s, re.S).group(1)
subprocess.run(['node', '-e', 'new Function(process.argv[1])', script], check=True)
assert all(f'data-v="{v}"' in s for v in ('modelRun','agentRun','uiRun','datasets','prompts','ledger','versions'))
assert 'await api(\'/api/graders\')' in s
assert '{grader: grader()}' in s
assert 'JSON.stringify(payload)' in s and "...base(), model_api_keys:roleKeys()" in s
assert "Array.isArray(d.roles) ? d.roles : []" in s
assert "uiRun:viewProductUI" in s
assert "api('/api/product-ui/run'" in s and 'JSON.stringify({url:url.href,steps})' in s
assert all(x in s for x in ('check_text', 'check_button', 'screenshot', 'png_base64', 'elapsed_ms', '未执行'))
assert 'uiExample' in s and "data-v=\"overview\"" in s
assert 'previousStage' in s
assert '实际生效需要后端支持 grader 字段' not in s
assert '双跑暂不支持指定评分器' not in s  # 双跑现已支持评分器
assert "deepseek-chat" not in s
assert 'placeholder="未配置"' in s
# 回归：被测模型必须能读到「模型与API配置」里的 target 角色卡，
# 且独立模型的 prompt 列表不能被产品过滤逻辑清空。
assert 'function targetRole()' in s and 'id="rModelPick"' in s
assert 'runModelCfg()' in s and 'pickRole()' in s
assert "x.product==='standalone_model'" in s
assert '<option value="standalone_model">独立模型评测（不关联产品）</option>' in s
assert "<option value=\"standalone_agent\">独立 Agent 评测（直接打接口）</option>" in s
assert "scopeVal === 'standalone_agent'" in s
assert "modelRun:()=>viewRun('model'), agentRun:()=>viewRun('agent'), uiRun:viewProductUI" in s
assert "api('/api/runs'" in s and "api('/api/dual'" in s
print('PASS: JavaScript 语法及导航、角色、评分器、被测模型来源和界面评测静态契约检查')
