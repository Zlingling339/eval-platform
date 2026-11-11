"""独立界面执行器测试；真实浏览器测试在缺少 Playwright/Chromium 时跳过。"""
import asyncio
import base64
import importlib.util
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.product_ui import run_product_ui


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'''<!doctype html><input id="name"><button id="go" onclick="document.querySelector('#out').textContent=document.querySelector('#name').value">Go</button><button id="off" disabled>Off</button><p id="out">waiting</p>'''
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ProductUITest(unittest.TestCase):
    def run_ui(self, *args, **kwargs):
        return asyncio.run(run_product_ui(*args, **kwargs))

    def test_validation(self):
        for url, steps in [('file:///etc/passwd', []), ('https://x', [{'action': 'fill', 'selector': '#a'}]), ('https://x', [{'action': 'check_button', 'selector': 'b', 'enabled': 'false'}])]:
            result = self.run_ui(url, steps)
            self.assertEqual(result['error']['code'], 'invalid_input')
            self.assertFalse(result['ok'])
            self.assertGreaterEqual(result['elapsed_ms'], 0)

    def test_missing_dependency(self):
        if importlib.util.find_spec('playwright'):
            self.skipTest('Playwright 已安装')
        result = self.run_ui('http://localhost', [])
        self.assertEqual(result['error']['code'], 'dependency_missing')


class BrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not importlib.util.find_spec('playwright'):
            raise unittest.SkipTest('未安装 Playwright')
        from playwright.sync_api import sync_playwright
        try:
            with sync_playwright() as p:
                executable = os.environ.get('PRODUCT_UI_CHROMIUM_EXECUTABLE')
                browser = p.chromium.launch(**({'executable_path': executable} if executable else {}))
                browser.close()
        except Exception as exc:
            raise unittest.SkipTest('Chromium 不可用: ' + str(exc).splitlines()[0])
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:%s/' % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def run_ui(self, steps, **kwargs):
        return asyncio.run(run_product_ui(self.url, steps, **kwargs))

    def test_success(self):
        result = self.run_ui([
            {'action': 'fill', 'selector': '#name', 'value': 'hello'},
            {'action': 'check_button', 'selector': '#go', 'enabled': True},
            {'action': 'check_button', 'selector': '#off', 'enabled': False},
            {'action': 'click', 'selector': '#go'},
            {'action': 'check_text', 'selector': '#out', 'text': 'hello'},
            {'action': 'screenshot'},
        ])
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['steps'][4]['actual'], 'hello')
        self.assertTrue(base64.b64decode(result['steps'][5]['png_base64']).startswith(b'\x89PNG'))
        self.assertTrue(all('elapsed_ms' in s for s in result['steps']))

    def test_assertion_and_fail_fast(self):
        result = self.run_ui([{'action': 'check_text', 'selector': '#out', 'text': 'no'}, {'action': 'click', 'selector': '#go'}])
        self.assertEqual(result['error']['code'], 'assertion_failed')
        self.assertEqual(result['error']['step_index'], 0)
        self.assertEqual(len(result['steps']), 1)
        self.assertEqual(result['steps'][0]['actual'], 'waiting')

    def test_timeout(self):
        result = self.run_ui([{'action': 'click', 'selector': '#missing'}], timeout_ms=100)
        self.assertEqual(result['error']['code'], 'timeout')
        self.assertEqual(result['error']['step_index'], 0)
        self.assertFalse(result['steps'][0]['ok'])

    def test_navigation_failure(self):
        result = asyncio.run(run_product_ui('http://127.0.0.1:1/', [], navigation_timeout_ms=500))
        self.assertFalse(result['ok'])
        self.assertIsNone(result['error']['step_index'])


if __name__ == '__main__':
    unittest.main()
