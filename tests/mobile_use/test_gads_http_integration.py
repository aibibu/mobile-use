"""Exercise the official Appium client against a local, deterministic Grid fixture."""

import base64
import io
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image
from pydantic import SecretStr

from minitap.mobile_use.clients.appium_client import RemoteAppiumClientWrapper
from minitap.mobile_use.clients.remote_appium_config import RemoteAppiumClientConfig
from minitap.mobile_use.controllers.ios_controller import iOSDeviceController
from minitap.mobile_use.controllers.types import CoordinatesSelectorRequest


@pytest.fixture
def grid():
    state = {"requests": [], "clicked": False, "fail_actions": False}
    png = io.BytesIO()
    Image.new("RGB", (780, 1688), "white").save(png, "PNG")
    screenshot = base64.b64encode(png.getvalue()).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):
            pass

        def reply(self, value, status=200):
            payload = json.dumps({"value": value}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            state["requests"].append(("POST", self.path, body))
            if self.path == "/grid/session":
                self.reply(
                    {
                        "sessionId": "fixture-session",
                        "capabilities": {
                            "platformName": "iOS",
                            "appium:automationName": "XCUITest",
                            "gads:deviceUdid": "fixture-iphone",
                        },
                    }
                )
            elif self.path == "/grid/session/fixture-session/actions":
                if state["fail_actions"]:
                    self.reply({"error": "unknown error", "message": "fixture input failed"}, 500)
                else:
                    state["clicked"] = True
                    self.reply(None)
            else:
                self.reply({"error": "unknown command", "message": self.path}, 404)

        def do_GET(self):
            state["requests"].append(("GET", self.path, None))
            if self.path == "/grid/session/fixture-session/window/rect":
                self.reply({"x": 0, "y": 0, "width": 390, "height": 844})
            elif self.path == "/grid/session/fixture-session/screenshot":
                self.reply(screenshot)
            elif self.path == "/grid/session/fixture-session/source":
                label = "Done" if state["clicked"] else "Continue"
                self.reply(
                    '<AppiumAUT><XCUIElementTypeApplication x="0" y="0" '
                    'width="390" height="844" enabled="true" visible="true">'
                    f'<XCUIElementTypeButton x="20" y="40" width="100" height="30" '
                    f'label="{label}" name="fixture-button" enabled="true" visible="true"/>'
                    "</XCUIElementTypeApplication></AppiumAUT>"
                )
            else:
                self.reply({"error": "unknown command", "message": self.path}, 404)

        def do_DELETE(self):
            state["requests"].append(("DELETE", self.path, None))
            if self.path == "/grid/session/fixture-session":
                self.reply(None)
            else:
                self.reply({"error": "unknown command", "message": self.path}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/grid", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def config(url):
    return RemoteAppiumClientConfig(
        hub_url=url, udid="fixture-iphone", client_secret=SecretStr("fixture-secret")
    )


@pytest.mark.asyncio
async def test_official_client_observes_clicks_and_releases_same_grid_session(grid):
    url, state = grid
    client = RemoteAppiumClientWrapper(config=config(url))
    try:
        assert await client.init_client()
        assert await client.get_screen_size() == (390, 844)
        image = Image.open(io.BytesIO(await client.screenshot()))
        assert image.size == (390, 844)
        before = await client.describe_all()
        assert any(element["label"] == "Continue" for element in before)
        assert await client.tap(70, 55)
        after = await client.describe_all()
        assert any(element["label"] == "Done" for element in after)
    finally:
        await client.cleanup()

    capabilities = state["requests"][0][2]["capabilities"]["alwaysMatch"]
    assert capabilities["gads:clientSecret"] == "fixture-secret"
    assert capabilities["appium:udid"] == "fixture-iphone"
    assert capabilities["appium:noReset"] is True
    assert capabilities["appium:fullReset"] is False
    assert capabilities["appium:autoLaunch"] is False
    assert "bstack:options" not in capabilities
    assert "appium:app" not in capabilities
    action = next(body for method, path, body in state["requests"] if path.endswith("/actions"))
    move = next(
        entry
        for device in action["actions"]
        for entry in device["actions"]
        if entry["type"] == "pointerMove"
    )
    assert (move["x"], move["y"]) == (70, 55)
    assert state["requests"][-1][:2] == ("DELETE", "/grid/session/fixture-session")


@pytest.mark.asyncio
async def test_grid_action_failure_is_not_reported_as_a_successful_tap(grid):
    url, state = grid
    client = RemoteAppiumClientWrapper(config=config(url))
    try:
        await client.init_client()
        controller = iOSDeviceController(client, "fixture-iphone", 390, 844)
        state["fail_actions"] = True
        result = await controller.tap(CoordinatesSelectorRequest(x=70, y=55))
        assert result.error is not None
        assert "fixture-secret" not in result.error
        assert state["clicked"] is False
    finally:
        await client.cleanup()
    assert state["requests"][-1][:2] == ("DELETE", "/grid/session/fixture-session")


def test_observation_example_uses_grid_without_model_or_phone(grid):
    url, state = grid
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.update(
        GADS_APPIUM_URL=url,
        GADS_DEVICE_UDID="fixture-iphone",
        GADS_CLIENT_SECRET="fixture-secret",
        MOBILE_USE_TELEMETRY_ENABLED="false",
    )
    process = subprocess.run(
        [sys.executable, "scripts/gads.py"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    output = json.loads(process.stdout.splitlines()[-1])
    assert output["width"] == 390
    assert output["height"] == 844
    assert output["element_count"] >= 1
    assert "fixture-secret" not in process.stdout + process.stderr
    assert state["clicked"] is False
    assert state["requests"][-1][:2] == ("DELETE", "/grid/session/fixture-session")
