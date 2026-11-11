"""独立产品 UI 路由；挂载：app.include_router(product_ui_router)。

仅适用于可信调用方与受限网络：浏览器可以访问内网、重定向和子资源，
本路由不提供 SSRF 防护、鉴权或并发限制。截图仅随响应返回，不落盘，
仍可能包含敏感信息。持久记录严格采用非敏感元数据白名单。
记录目录可用 PRODUCT_UI_RECORD_DIR 指定，默认 data/product_ui_runs。
"""
from __future__ import annotations

import json
import os
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .product_ui import run_product_ui

router = APIRouter()
product_ui_router = router
_ACTIONS = frozenset(('fill', 'click', 'check_text', 'check_button', 'screenshot'))
_CODES = frozenset(('invalid_input', 'dependency_missing', 'assertion_failed', 'timeout', 'browser_error'))


class ProductUIRequest(BaseModel):
    url: str
    steps: list[dict[str, Any]]


def _number(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**53 else None


def _record(result: dict[str, Any]) -> str:
    """仅写入有限白名单，不保存 URL、输入、选择器、文本、异常或截图。"""
    run_id = uuid.uuid4().hex
    err = result.get('error')
    code = err.get('code') if isinstance(err, dict) else None
    code = code if isinstance(code, str) and code in _CODES else ('other' if err else None)
    steps = []
    for step in result.get('steps', []):
        if not isinstance(step, dict):
            continue
        action = step.get('action')
        steps.append({
            'index': _number(step.get('index')),
            'action': action if isinstance(action, str) and action in _ACTIONS else 'other',
            'ok': step.get('ok') is True,
            'elapsed_ms': _number(step.get('elapsed_ms')),
        })
    record = {'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
              'ok': result.get('ok') is True, 'elapsed_ms': _number(result.get('elapsed_ms')),
              'error_code': code, 'steps': steps}
    root = Path(os.environ.get('PRODUCT_UI_RECORD_DIR') or
                Path(__file__).resolve().parent.parent / 'data' / 'product_ui_runs')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    # openat + O_NOFOLLOW 防止记录目录末级及文件被符号链接替换。
    dfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(dfd)
        if info.st_uid != os.geteuid():
            raise PermissionError('record directory must be owned by current user')
        if stat.S_IMODE(info.st_mode) & 0o077:
            # 目录是我们自己管的：权限过宽时直接收紧，不要因此让整次运行失败
            # （检出、解压、不同 umask 都会把 0700 变成 0777/0755）。
            os.fchmod(dfd, 0o700)
        fd = os.open(run_id + '.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=dfd)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                fd = -1
                json.dump(record, stream, ensure_ascii=False, separators=(',', ':'))
        except BaseException:
            if fd != -1:
                os.close(fd)
            try:
                os.unlink(run_id + '.json', dir_fd=dfd)
            except OSError:
                pass
            raise
    finally:
        os.close(dfd)
    return run_id


@router.post('/api/product-ui/run')
async def product_ui_run(body: ProductUIRequest) -> dict[str, Any]:
    try:
        result = await run_product_ui(body.url, body.steps)
    except Exception:
        # 不向调用方回传原始异常，避免泄露页面信息。
        result = {'ok': False, 'steps': [], 'elapsed_ms': 0,
                  'error': {'code': 'browser_error', 'message': '界面执行失败', 'step_index': None}}
    try:
        run_id = _record(result)
    except Exception:
        # 不报告成功：调用方要求每次运行均持久化。
        raise HTTPException(status_code=500, detail='执行已结束，但本地记录保存失败') from None
    return {**result, 'run_id': run_id}


if __name__ == '__main__':
    # 自包含测试：python -m src.product_ui_routes；不触及共享测试文件。
    import tempfile
    import unittest
    from unittest.mock import AsyncMock, patch
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class RouteTests(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory()
            self.env = patch.dict(os.environ, {'PRODUCT_UI_RECORD_DIR': self.temp.name})
            self.env.start()
            app = FastAPI()
            app.include_router(router)
            self.client = TestClient(app)

        def tearDown(self):
            self.client.close()
            self.env.stop()
            self.temp.cleanup()

        def test_record_whitelist_and_screenshot_response(self):
            secret = 'SECRET_unique_123'
            fake = {'ok': True, 'elapsed_ms': 10, 'error': None, 'steps': [
                {'index': 0, 'action': 'check_text', 'ok': True, 'elapsed_ms': 2, 'actual': secret},
                {'index': 1, 'action': 'screenshot', 'ok': True, 'elapsed_ms': 3, 'png_base64': secret}]}
            with patch(__name__ + '.run_product_ui', new=AsyncMock(return_value=fake)) as run:
                response = self.client.post('/api/product-ui/run', json={
                    'url': 'https://example.com/?token=' + secret,
                    'steps': [{'action': 'fill', 'selector': secret, 'value': secret}]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['steps'][1]['png_base64'], secret)
            run.assert_awaited_once()
            path = Path(self.temp.name) / (response.json()['run_id'] + '.json')
            content = path.read_text()
            self.assertNotIn(secret, content)
            self.assertEqual(json.loads(content)['steps'][0]['action'], 'check_text')
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

        def test_error_and_storage_failure(self):
            with patch(__name__ + '.run_product_ui', new=AsyncMock(side_effect=Exception('secret'))):
                response = self.client.post('/api/product-ui/run', json={'url': 'http://localhost', 'steps': []})
            self.assertFalse(response.json()['ok'])
            self.assertNotIn('secret', response.text)
            self.assertEqual(json.loads(next(Path(self.temp.name).glob('*.json')).read_text())['error_code'], 'browser_error')
            with patch(__name__ + '._record', side_effect=OSError('secret')):
                response = self.client.post('/api/product-ui/run', json={'url': 'bad', 'steps': []})
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('secret', response.text)

        def test_invalid_json(self):
            self.assertEqual(self.client.post('/api/product-ui/run', json={'steps': []}).status_code, 422)

        def test_insecure_directory_refused(self):
            os.chmod(self.temp.name, 0o755)
            with self.assertRaises(PermissionError):
                _record({'ok': True, 'steps': []})
            os.chmod(self.temp.name, 0o700)

    unittest.main()
