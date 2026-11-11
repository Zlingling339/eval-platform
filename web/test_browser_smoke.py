"""Browser smoke test against a running isolated server. Requires Playwright and Chromium."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def test_browser_smoke():
    import pytest
    if not os.environ.get("EVAL_URL") or not Path('/usr/bin/chromium').is_file():
        pytest.skip("需要 EVAL_URL 指向隔离服务及 /usr/bin/chromium")
    base = os.environ.get('EVAL_URL', 'http://127.0.0.1:8765')
    examples = Path(__file__).resolve().parents[1] / 'examples'
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
        page = browser.new_page()
        errors = []
        requests = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda req: requests.append(req.post_data_json) if req.method == 'POST' and req.url.endswith('/api/runs') else None)
        page.goto(base, wait_until='networkidle')
        page.evaluate('localStorage.clear()')
        page.reload(wait_until='networkidle')
        page.locator('[data-v="settings"]').first.click()
        page.locator('#sModel').wait_for()
        assert page.locator('#sModel').input_value() == ''
        assert page.locator('#sModel').get_attribute('placeholder') == '未配置'
        page.locator('#sModel').fill('test-configured-model')
        page.locator('#skSave').click()
        page.locator('[data-v="modelRun"]').first.click()
        page.locator('#rModel').wait_for()
        assert page.locator('#rModel').input_value() == 'test-configured-model'
        page.evaluate('localStorage.clear()')
        for kind in ('model', 'agent'):
            page.locator('[data-v="datasets"]').first.click()
            page.locator('#dsFile').set_input_files(str(examples / f'smoke-{kind}.json'))
            with page.expect_response(lambda r: '/api/datasets?' in r.url and r.request.method == 'POST') as upload:
                page.locator('#dsUp').click()
            assert upload.value.ok, upload.value.text()
            page.wait_for_function("document.querySelector('#dsResult') && document.querySelector('#dsResult').innerText.includes('上传成功')")
            print(kind, 'upload:', upload.value.json())
            page.wait_for_timeout(800)
            row = page.locator('tr[data-id]').filter(has_text=f'smoke_{kind}_v1').first
            did = row.get_attribute('data-id')
            row.locator('[data-rv="yes"]').click()
            page.wait_for_function("id => document.querySelector(`tr[data-id='${id}'] .rbg.y`)", arg=did)
            print(kind, 'reviewed:', did)
            page.locator(f'[data-v="{kind}Run"]').first.click()
            page.locator('#rDs').wait_for()
            options = page.locator('#rDs option').all_text_contents()
            print(kind, 'datasets:', options)
            option = page.locator('#rDs option').filter(has_text=f'smoke_{kind}_v1').first
            page.locator('#rDs').select_option(value=option.get_attribute('value'))
            if kind == 'model':
                assert page.locator('#rModel').input_value() == ''
                page.locator('#rGrader').select_option('__custom__')
                page.locator('#rGraderCustom').fill('keyword_rule')
            else:
                page.locator('#rGrader').select_option('__custom__')
                page.locator('#rGraderCustom').fill('tool_call_rule+trajectory_rule')
            page.locator('#rMock').check()
            page.locator('#rGo').click()
            page.wait_for_timeout(4000)
            print(kind, 'intermediate:', page.locator('#rMsg').inner_text(), 'errors:', errors, flush=True)
            page.wait_for_function("document.querySelector('#rMsg') && /完成|失败|错误/.test(document.querySelector('#rMsg').innerText)", timeout=10000)
            print(kind, 'run:', page.locator('#rMsg').inner_text())
            print(kind, 'payload:', requests[-1])
            assert '完成' in page.locator('#rMsg').inner_text()
            assert requests[-1]['mock'] is True
            assert requests[-1]['grader'] == ('keyword_rule' if kind == 'model' else 'tool_call_rule+trajectory_rule')
            iframe = page.locator('#rOut iframe')
            iframe.wait_for()
            print(kind, 'report:', iframe.get_attribute('src'))
        print('pageerrors:', errors)
        assert not errors
        browser.close()

if __name__ == "__main__":
    test_browser_smoke()
