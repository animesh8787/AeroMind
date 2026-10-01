"""End-to-end acceptance test of the live demo in a real browser (Playwright + Chromium).

Skipped when Playwright or a Chromium build is not available. Set AEROMIND_CHROMIUM to a
Chromium executable to use a system browser.
"""

import os
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")
playwright = pytest.importorskip("playwright.sync_api")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(0.25)
    raise AssertionError("condition not met in time")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    from aeromind.onnx_export import export_onnx
    from aeromind.server.app import create_app
    from aeromind.train import TrainConfig, train

    d = tmp_path_factory.mktemp("e2e")
    export_onnx(train(TrainConfig(healthy_runs=10, healthy_len=200, runs_per_mode=5, life_range=(200, 300),
                                  seed=11, phases=True)), d / "model")
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(d / "model", d / "work", speed=20), port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    _wait(lambda: srv.started, 30)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(10)


@pytest.fixture(scope="module")
def page(server):
    exe = os.environ.get("AEROMIND_CHROMIUM") or next(
        (p for p in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome",) if os.path.exists(p)), None)
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        except Exception as e:  # no browser installed
            pytest.skip(f"Chromium not available: {e}")
        pg = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server)
        pg.wait_for_selector('[data-testid="aircraft-VT-AMA03"]')
        yield pg
        assert errors == []
        browser.close()


def text(pg, tid):
    return pg.inner_text(f'[data-testid="{tid}"]')


def test_component_fault_flow(page):
    page.click('[data-testid="aircraft-VT-AMA03"]')
    page.click('[data-testid="inject-bearing_wear"]')
    _wait(lambda: "bearing" in text(page, "fault"))
    r1 = float(text(page, "rul").split()[0])
    _wait(lambda: text(page, "rul").split()[0] not in ("–",) and float(text(page, "rul").split()[0]) < r1)
    _wait(lambda: "BPFO" in text(page, "evidence"))
    _wait(lambda: any(k in text(page, "decision").lower() for k in ("replace", "ground", "defer")))
    _wait(lambda: text(page, "acars").startswith("AMD1/VT-AMA03/") and len(text(page, "acars")) <= 220)


def test_sensor_failure_is_not_a_component_fault(page):
    page.click('[data-testid="aircraft-VT-AMA05"]')
    page.click('[data-testid="inject-temperature_failure"]')
    _wait(lambda: "fault" in text(page, "sensor-health"))
    _wait(lambda: "SENSOR_FAULT" in text(page, "events"))
    time.sleep(3)  # let the fleet run on with the dead sensor
    assert "no anomaly" in text(page, "fault") or "building" in text(page, "fault")
    assert text(page, "status") in ("SENSOR FAULT", "WATCH")


def test_ota_valid_then_tampered(page):
    page.click('[data-testid="ota-valid"]')
    _wait(lambda: "v2" in text(page, "model-version"))
    page.click('[data-testid="ota-tampered"]')
    _wait(lambda: "REJECTED" in text(page, "events"))
    assert "v2" in text(page, "model-version")  # previous valid model stays active


def test_roi_panel_and_whatif(page):
    page.click('[data-testid="aircraft-VT-AMA03"]')  # has a bearing advisory from the first test
    page.fill('[data-testid="whatif"]', "100")
    page.dispatch_event('[data-testid="whatif"]', "input")
    _wait(lambda: "risk before check" in text(page, "whatif-result"))
    _wait(lambda: "AeroMind" in text(page, "roi"), timeout=240)
