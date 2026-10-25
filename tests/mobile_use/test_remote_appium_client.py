import asyncio
import base64
import logging
import threading
import traceback
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image
from pydantic import SecretStr, ValidationError
from appium.webdriver.webdriver import WebDriver

from minitap.mobile_use.clients import appium_client
from minitap.mobile_use.clients.appium_client import (
    RemoteAppiumClientWrapper,
    RemoteAppiumError,
    _SameOriginConnection,
)
from minitap.mobile_use.clients.browserstack_client import BrowserStackClientWrapper
from minitap.mobile_use.clients.ios_client_config import BrowserStackClientConfig
from minitap.mobile_use.clients.remote_appium_config import RemoteAppiumClientConfig

SECRET = "private-grid-secret-do-not-log"
UDID = "00008110-EXAMPLE"


def config(**kwargs):
    return RemoteAppiumClientConfig(
        hub_url="https://grid.example.test/gads/grid/",
        udid=UDID,
        client_secret=SecretStr(SECRET),
        **kwargs,
    )


def png(size):
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, "PNG")
    return output.getvalue()


class FakeDriver:
    def __init__(self):
        self.created: list[dict[str, Any]] = []
        self.session_id = "own-session-1"
        self.capabilities = {"gads:deviceUdid": UDID}
        self.size = {"width": 100, "height": 200}
        self.image = png((300, 600))
        self.page_source = (
            '<AppiumAUT><XCUIElementTypeButton name="Continue" x="10" y="20" '
            'width="30" height="40" enabled="true" visible="true"/></AppiumAUT>'
        )
        self.commands = []
        self.quit_count = 0
        self.action_started = threading.Event()
        self.action_release = threading.Event()
        self.block_actions = False
        self.fail_actions = False
        self.fail_quit = False
        self.element = SimpleNamespace(send_keys=self.send_keys, clear=self.clear)
        self.switch_to = SimpleNamespace(active_element=self.element)

    def get_window_size(self):
        return dict(self.size)

    def get_screenshot_as_base64(self):
        return base64.b64encode(self.image).decode()

    def execute(self, command, params):
        self.commands.append((command, params))
        if self.block_actions:
            self.action_started.set()
            assert self.action_release.wait(5), "test did not release fake operation"
        if self.fail_actions:
            raise RuntimeError(f"remote rejected operation with {SECRET}")
        return {"value": None}

    def execute_script(self, script, params=None):
        self.commands.append((script, params))
        if script == "mobile: activeAppInfo":
            return {"name": "Fixture", "bundleId": "test.fixture"}
        if self.fail_actions:
            raise RuntimeError(SECRET)

    def send_keys(self, value):
        self.commands.append(("send_keys", value))

    def clear(self):
        self.commands.append(("clear", None))

    def get(self, url):
        self.commands.append(("get", url))

    def quit(self):
        self.quit_count += 1
        if self.fail_quit:
            raise RuntimeError(SECRET)


@pytest.fixture
def driver(monkeypatch):
    driver = FakeDriver()
    created = []

    def create(**kwargs):
        created.append(kwargs)
        return driver

    monkeypatch.setattr(appium_client, "WebDriver", create)
    driver.created = created
    return driver


@pytest.mark.asyncio
async def test_fixed_capabilities_do_not_install_or_use_browserstack(driver):
    client = RemoteAppiumClientWrapper(config())
    await client.init_client()
    caps = driver.created[0]["options"].to_capabilities()
    assert caps == {
        "platformName": "iOS",
        "appium:automationName": "XCUITest",
        "appium:udid": UDID,
        "appium:noReset": True,
        "appium:fullReset": False,
        "appium:autoLaunch": False,
        "gads:clientSecret": SECRET,
        "gads:queueTimeout": 10,
    }
    assert driver.created[0]["client_config"].direct_connection is False
    assert SECRET not in repr(client.config)
    assert "client_secret" not in client.config.model_dump()
    await client.cleanup()
    await client.cleanup()
    assert driver.quit_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host/grid",
        "https://user:pass@host/grid",
        "https://host/grid?token=x",
        "https://host/grid#fragment",
        "https://host/grid/../elsewhere",
        "https://host/%2fgrid",
    ],
)
def test_invalid_grid_urls_are_rejected(url):
    with pytest.raises(ValidationError):
        RemoteAppiumClientConfig(hub_url=url, client_secret=SecretStr(SECRET), udid=UDID)


def test_capability_overrides_and_missing_udid_are_rejected():
    with pytest.raises(ValidationError):
        config(capabilities={"appium:noReset": False})
    with pytest.raises(ValidationError):
        RemoteAppiumClientConfig.model_validate(
            {"hub_url": "https://host/grid", "client_secret": SecretStr(SECRET)}
        )


@pytest.mark.asyncio
async def test_observations_share_logical_coordinates_and_refresh_after_rotation(driver):
    client = RemoteAppiumClientWrapper(config())
    await client.init_client()
    context = SimpleNamespace(device_width=0, device_height=0)
    client.bind_device_context(context)
    image = Image.open(BytesIO(await client.screenshot()))
    assert image.size == (100, 200)
    elements = await client.describe_all()
    assert elements == [
        {
            "type": "XCUIElementTypeButton",
            "value": "",
            "label": "Continue",
            "frame": {"x": 10, "y": 20, "width": 30, "height": 40},
            "enabled": True,
            "visible": True,
        }
    ]
    driver.size = {"width": 200, "height": 100}
    driver.image = png((600, 300))
    assert Image.open(BytesIO(await client.screenshot())).size == (200, 100)
    assert (context.device_width, context.device_height) == (200, 100)
    await client.cleanup()


@pytest.mark.asyncio
async def test_actions_are_forwarded_in_points_and_failures_are_not_success(driver):
    client = RemoteAppiumClientWrapper(config())
    await client.init_client()
    assert await client.tap(10, 20)
    actions = driver.commands[-1][1]["actions"][0]["actions"]
    assert actions[0]["x"] == 10
    assert actions[0]["y"] == 20
    assert await client.swipe(10, 20, 50, 80, duration=0.4)
    actions = driver.commands[-1][1]["actions"][0]["actions"]
    assert actions[2]["duration"] == 400
    assert await client.home()
    assert driver.commands[-1] == ("mobile: pressButton", {"name": "home"})
    assert (await client.app_current()).bundle_id == "test.fixture"
    assert await client.text("hello")
    assert await client.key(42)
    assert await client.clear_text()
    for operation in (client.key(999), client.button("power"), client.tap(300, 10)):
        with pytest.raises(RemoteAppiumError):
            await operation
    driver.fail_actions = True
    with pytest.raises(RemoteAppiumError) as error:
        await client.tap(10, 20)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    await client.cleanup()


@pytest.mark.asyncio
async def test_invalid_source_is_an_error_not_an_empty_hierarchy(driver):
    client = RemoteAppiumClientWrapper(config())
    await client.init_client()
    for source in ("not xml", "<!DOCTYPE x><x/>", '<x x="NaN"/>'):
        driver.page_source = source
        with pytest.raises(RemoteAppiumError):
            await client.describe_all()
    await client.cleanup()


@pytest.mark.asyncio
async def test_wrong_device_is_closed_before_startup_error(driver):
    driver.capabilities = {"gads:deviceUdid": "wrong-device"}
    client = RemoteAppiumClientWrapper(config())
    with pytest.raises(RemoteAppiumError):
        await client.init_client()
    assert driver.quit_count == 1
    assert client._driver is None


@pytest.mark.asyncio
async def test_cancelled_creation_waits_for_session_then_closes_it(monkeypatch):
    driver = FakeDriver()
    started, release = threading.Event(), threading.Event()

    def create(**kwargs):
        started.set()
        assert release.wait(5)
        return driver

    monkeypatch.setattr(appium_client, "WebDriver", create)
    client = RemoteAppiumClientWrapper(config())
    task = asyncio.create_task(client.init_client())
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert driver.quit_count == 1
    assert client._driver is None


@pytest.mark.asyncio
async def test_cleanup_waits_for_cancelled_inflight_action(driver):
    client = RemoteAppiumClientWrapper(config())
    await client.init_client()
    driver.block_actions = True
    action = asyncio.create_task(client.tap(10, 20))
    await asyncio.to_thread(driver.action_started.wait, 2)
    action.cancel()
    cleanup = asyncio.create_task(client.cleanup())
    await asyncio.sleep(0.02)
    action.cancel()  # repeated cancellation must not release the lock early
    assert driver.quit_count == 0
    driver.action_release.set()
    with pytest.raises(asyncio.CancelledError):
        await action
    await cleanup
    assert driver.quit_count == 1


@pytest.mark.asyncio
async def test_failed_cleanup_is_retryable(driver):
    client = RemoteAppiumClientWrapper(config())
    await client.init_client()
    driver.fail_quit = True
    with pytest.raises(RemoteAppiumError):
        await client.cleanup()
    assert client._driver is driver
    driver.fail_quit = False
    await client.cleanup()
    assert client._driver is None


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", [SECRET, 'private-"quoted\\雪\nsecret'])
async def test_secret_in_transport_debug_log_and_exception_is_redacted(monkeypatch, caplog, secret):
    def fail(**kwargs):
        logging.getLogger("selenium.webdriver.remote.remote_connection").debug(
            "POST capabilities %s", kwargs["options"].to_capabilities()
        )
        raise RuntimeError(secret)

    monkeypatch.setattr(appium_client, "WebDriver", fail)
    client = RemoteAppiumClientWrapper(
        RemoteAppiumClientConfig(
            hub_url="https://grid.example.test/grid", udid=UDID, client_secret=SecretStr(secret)
        )
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(RemoteAppiumError) as error:
        await client.init_client()
    assert secret not in caplog.text
    assert secret not in "".join(traceback.format_exception(error.value))


def test_transport_rejects_cross_origin_or_outside_prefix_without_request():
    connection = _SameOriginConnection("https://host/grid")
    for url in (
        "https://elsewhere/grid/session",
        "https://host/other/session",
        "https://host/grid/../session",
        "https://host/grid/%2e%2e/session",
    ):
        with pytest.raises(RemoteAppiumError):
            connection._request("GET", url)
    connection.close()


@pytest.mark.asyncio
async def test_browserstack_shared_operations_keep_legacy_error_fallback():
    cfg = BrowserStackClientConfig(
        username="user",
        access_key=SecretStr("key"),
        device_name="iPhone",
        platform_version="17",
        app_url="bs://fixture",
    )
    client = BrowserStackClientWrapper(cfg)
    driver = FakeDriver()
    driver.fail_actions = True
    client._driver = cast(WebDriver, driver)
    assert await client.tap(10, 20) is False
    assert await client.screenshot() is not None
    assert await client.app_current() is None
