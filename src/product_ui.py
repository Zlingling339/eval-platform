"""独立的、异步 Playwright 产品界面执行器。

安装：pip install playwright && python -m playwright install chromium
FastAPI async 路由中：result = await run_product_ui(url, steps)

步骤是字典列表，支持 fill(selector, value)、click(selector)、
check_text(selector, text)、check_button(selector, enabled)、screenshot。
所有 selector 均为 Playwright locator 表达式；检查失败即停止后续步骤。
返回 JSON 可序列化字典：ok、steps、error、elapsed_ms；每个步骤包含
index、action、ok、elapsed_ms，检查附 actual，截图附 PNG base64。
error 为 {code, message, step_index} 或 None。超时单位均为毫秒。
只适用于可信目标和可信调用方；页面可发起任意网络请求，调用方须自行
做访问控制、网络隔离和敏感数据保护。不处理登录、验证码及持久化会话。
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any
from urllib.parse import urlsplit


async def run_product_ui(
    url: str,
    steps: list[dict[str, Any]],
    *,
    timeout_ms: int = 5000,
    navigation_timeout_ms: int = 15000,
    headless: bool = True,
) -> dict[str, Any]:
    """运行一次隔离浏览器会话；从 FastAPI 的 async def 直接 await。"""
    start = time.monotonic()
    results: list[dict[str, Any]] = []
    error = None
    index = None
    browser = None
    context = None

    def elapsed() -> int:
        return round((time.monotonic() - start) * 1000)

    try:
        if not isinstance(url, str):
            raise ValueError('url 必须是 http(s) 地址')
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username is not None:
            raise ValueError('url 必须是不含凭据的 http(s) 地址')
        if not isinstance(steps, list) or len(steps) > 100:
            raise ValueError('steps 必须是最多 100 项的数组')
        for timeout in (timeout_ms, navigation_timeout_ms):
            if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 120000:
                raise ValueError('超时必须是 1 到 120000 之间的整数毫秒')
        for step in steps:
            if not isinstance(step, dict) or step.get('action') not in ('fill', 'click', 'check_text', 'check_button', 'screenshot'):
                raise ValueError('步骤 action 不受支持')
            action = step['action']
            if action != 'screenshot' and (not isinstance(step.get('selector'), str) or not step['selector']):
                raise ValueError('步骤需要非空 selector')
            if action == 'fill' and not isinstance(step.get('value'), str):
                raise ValueError('fill 需要字符串 value')
            if action == 'check_text' and not isinstance(step.get('text'), str):
                raise ValueError('check_text 需要字符串 text')
            if action == 'check_button' and not isinstance(step.get('enabled'), bool):
                raise ValueError('check_button 需要布尔值 enabled')
        try:
            from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
        except ImportError:
            return {'ok': False, 'steps': [], 'error': {'code': 'dependency_missing', 'message': '请安装 playwright 和 Chromium 浏览器', 'step_index': None}, 'elapsed_ms': elapsed()}
        async with async_playwright() as playwright:
            try:
                executable = os.environ.get('PRODUCT_UI_CHROMIUM_EXECUTABLE')
                browser = await playwright.chromium.launch(headless=headless, **({'executable_path': executable} if executable else {}))
                context = await browser.new_context()
                page = await context.new_page()
                page.set_default_timeout(timeout_ms)
                page.set_default_navigation_timeout(navigation_timeout_ms)
                await page.goto(url, wait_until='domcontentloaded')
                for index, step in enumerate(steps):
                    action = step['action']
                    step_start = time.monotonic()
                    item = {'index': index, 'action': action, 'ok': False}
                    results.append(item)
                    try:
                        if action == 'screenshot':
                            image = await page.screenshot(type='png', timeout=timeout_ms)
                            item['png_base64'] = base64.b64encode(image).decode('ascii')
                        else:
                            locator = page.locator(step['selector'])
                            if action == 'fill':
                                await locator.fill(step['value'])
                            elif action == 'click':
                                await locator.click()
                            elif action == 'check_text':
                                # inner_text 等待元素出现；不等待文本异步变化。
                                actual = await locator.inner_text(timeout=timeout_ms)
                                item['actual'] = actual
                                if step['text'] not in actual:
                                    raise AssertionError('文本未包含预期内容')
                            elif action == 'check_button':
                                await locator.wait_for(state='attached', timeout=timeout_ms)
                                actual = await locator.is_enabled(timeout=timeout_ms)
                                item['actual'] = actual
                                if actual != step['enabled']:
                                    raise AssertionError('按钮启用状态与预期不符')
                        item['ok'] = True
                    finally:
                        item['elapsed_ms'] = round((time.monotonic() - step_start) * 1000)
            finally:
                if context is not None:
                    await context.close()
                if browser is not None:
                    await browser.close()
    except ValueError as exc:
        error = {'code': 'invalid_input', 'message': str(exc), 'step_index': index}
    except AssertionError as exc:
        error = {'code': 'assertion_failed', 'message': str(exc), 'step_index': index}
    except Exception as exc:
        # 不回传 Playwright 原始异常：可能包含 URL、输入值或其他敏感信息。
        if 'PlaywrightTimeoutError' in locals() and isinstance(exc, PlaywrightTimeoutError):
            code, message = 'timeout', '页面导航或操作超时'
        else:
            code, message = 'browser_error', '浏览器启动、导航或操作失败；请检查浏览器安装及目标页面'
        error = {'code': code, 'message': message, 'step_index': index}
    return {'ok': error is None, 'steps': results, 'error': error, 'elapsed_ms': elapsed()}
