"""真实前端 → FastAPI → Chromium 端到端冒烟测试。"""
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path


class ProductUIFrontendTest(unittest.TestCase):
    def test_frontend_example(self):
        executable = os.environ.get('PRODUCT_UI_CHROMIUM_EXECUTABLE')
        if not executable:
            self.skipTest('请设置 PRODUCT_UI_CHROMIUM_EXECUTABLE 进行真实前端测试')
        from playwright.sync_api import sync_playwright
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = dict(os.environ, EVAL_DB=str(Path(tmp) / 'test.db'),
                       PRODUCT_UI_RECORD_DIR=str(Path(tmp) / 'records'))
            process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', str(port)], cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                url = f'http://127.0.0.1:{port}/'
                for _ in range(300):  # 首次导入 fastapi/pydantic 较慢，给足 30s
                    if process.poll() is not None:
                        self.fail('服务提前退出')
                    try:
                        with urllib.request.urlopen(url, timeout=1):
                            break
                    except OSError:
                        time.sleep(.1)
                else:
                    self.fail('服务启动超时')
                with sync_playwright() as p:
                    browser = p.chromium.launch(executable_path=executable)
                    try:
                        page = browser.new_page()
                        page.goto(url)
                        page.locator('button[data-v="uiRun"]').click()
                        page.locator('#uiExample').click()
                        with page.expect_response(lambda r: '/api/product-ui/run' in r.url, timeout=30000) as response_info:
                            page.locator('#uiGo').click()
                        response = response_info.value
                        body = response.json()
                        self.assertEqual(response.status, 200, body)
                        self.assertTrue(body['ok'], body)
                        self.assertEqual([s['action'] for s in body['steps']], ['click', 'check_text', 'screenshot'])
                        self.assertTrue(body['steps'][2]['png_base64'].startswith('iVBOR'))
                        page.locator('#uiResults img').wait_for(timeout=5000)
                        self.assertEqual(page.locator('#uiResults .verbox').count(), 3)
                        record = (Path(tmp) / 'records' / (body['run_id'] + '.json')).read_text()
                        self.assertNotIn('png_base64', record)
                    finally:
                        browser.close()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    unittest.main()
