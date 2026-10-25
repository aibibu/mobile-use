"""Shared Appium operations and an iOS remote-grid client.

The Appium Python client owns the WebDriver protocol. This module only adds
lifecycle, serialization, coordinate normalization, and safe error boundaries.
"""

import asyncio
import base64
import json
import logging
import math
import re
from functools import wraps
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from appium.options.common.base import AppiumOptions
from appium.webdriver.appium_connection import AppiumConnection
from appium.webdriver.client_config import AppiumClientConfig
from appium.webdriver.webdriver import WebDriver
from PIL import Image
from selenium.webdriver.common.actions import interaction
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.keys import Keys

from minitap.mobile_use.clients.idb_client import IOSAppInfo
from minitap.mobile_use.clients.remote_appium_config import RemoteAppiumClientConfig


class RemoteAppiumError(RuntimeError):
    """Safe public error: never includes a server response or credentials."""


async def _settle(task: asyncio.Task) -> None:
    """Wait for a non-cancellable thread-backed operation, even on repeated cancel."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if task.done() and not task.cancelled():
        task.exception()  # retrieve errors without exposing response bodies


def appium_operation(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        async with self._command_lock:
            task = asyncio.create_task(method(self, *args, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _settle(task)
                cleanup = asyncio.create_task(self._after_cancel(method.__name__))
                await _settle(cleanup)
                raise
            except Exception as error:
                return self._operation_error(
                    method.__name__, error, method.__annotations__.get("return")
                )

    return wrapped


def parse_ios_source(xml_source: str) -> list[dict[str, Any]]:
    """Translate native XCUITest XML into the existing IDB-style element contract."""
    elements = []
    for elem in ET.fromstring(xml_source).iter():
        if elem.tag == "AppiumAUT":
            continue
        frame = {key: float(elem.get(key, 0)) for key in ("x", "y", "width", "height")}
        if not all(math.isfinite(value) for value in frame.values()):
            raise ValueError("Invalid element frame")
        elements.append(
            {
                "type": elem.get("type", elem.tag),
                "value": elem.get("value", ""),
                "label": elem.get("label", elem.get("name", "")),
                "frame": frame,
                "enabled": elem.get("enabled", "false").lower() == "true",
                "visible": elem.get("visible", "true").lower() == "true",
            }
        )
    return elements


class AppiumClientOperations:
    """Reusable operations; subclasses choose connection and error policy."""

    def __init__(self):
        self._driver: WebDriver | None = None
        self._command_lock = asyncio.Lock()

    def _ensure_driver(self) -> WebDriver:
        if self._driver is None:
            raise RemoteAppiumError("Remote Appium session is not initialized")
        return self._driver

    def _operation_error(self, name: str, error: Exception, return_type) -> Any:
        if isinstance(error, RemoteAppiumError):
            raise RemoteAppiumError(str(error)) from None
        raise RemoteAppiumError(f"Remote Appium {name} failed ({type(error).__name__})") from None

    async def _after_cancel(self, name: str) -> None:
        pass

    def _check_launch_options(self, args, env) -> None:
        raise RemoteAppiumError("App launch arguments and environment are unsupported")

    def _validate_point(self, driver, x, y):
        pass

    def _validate_tap_duration(self, duration):
        pass

    @appium_operation
    async def tap(self, x: int, y: int, duration: float | None = None) -> bool:
        """Tap at the specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            duration: Optional tap duration in seconds (for long press)

        Returns:
            True if tap succeeded, False otherwise
        """
        driver = self._ensure_driver()
        self._validate_tap_duration(duration)

        def perform_tap():
            self._validate_point(driver, x, y)
            finger = PointerInput(interaction.POINTER_TOUCH, "finger")
            actions = ActionBuilder(driver, mouse=finger)
            actions.pointer_action.move_to_location(x, y)
            actions.pointer_action.pointer_down()
            if duration:
                actions.pointer_action.pause(duration)
            actions.pointer_action.pointer_up()
            actions.perform()

        await asyncio.to_thread(perform_tap)
        return True

    @appium_operation
    async def swipe(
        self,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
        duration: float | None = None,
    ) -> bool:
        """Swipe from start coordinates to end coordinates.

        Args:
            x_start: Starting X coordinate
            y_start: Starting Y coordinate
            x_end: Ending X coordinate
            y_end: Ending Y coordinate
            duration: Optional swipe duration in seconds

        Returns:
            True if swipe succeeded, False otherwise
        """
        driver = self._ensure_driver()

        swipe_duration = duration or 0.5

        def perform_swipe():
            self._validate_point(driver, x_start, y_start)
            self._validate_point(driver, x_end, y_end)
            finger = PointerInput(interaction.POINTER_TOUCH, "finger")
            actions = ActionBuilder(driver, mouse=finger)
            actions.pointer_action.move_to_location(x_start, y_start)
            actions.pointer_action.pointer_down()
            actions.pointer_action.pause(swipe_duration)
            actions.pointer_action.move_to_location(x_end, y_end)
            actions.pointer_action.pointer_up()
            actions.perform()

        await asyncio.to_thread(perform_swipe)
        return True

    @appium_operation
    async def screenshot(self, output_path: str | None = None) -> bytes | None:
        """Take a screenshot and return raw image data.

        Args:
            output_path: Optional path to save the screenshot

        Returns:
            Raw image data (PNG bytes) or None on failure
        """
        driver = self._ensure_driver()

        screenshot_base64 = await asyncio.to_thread(driver.get_screenshot_as_base64)

        import base64

        screenshot_data = base64.b64decode(screenshot_base64)

        if output_path:
            with open(output_path, "wb") as f:
                f.write(screenshot_data)

        return screenshot_data

    @appium_operation
    async def launch(
        self,
        bundle_id: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> bool:
        """Launch an application by bundle ID.

        Args:
            bundle_id: The bundle identifier of the app to launch
            args: Optional list of arguments to pass to the app (not supported on BrowserStack)
            env: Optional environment variables for the app (not supported on BrowserStack)

        Returns:
            True if launch succeeded, False otherwise
        """
        driver = self._ensure_driver()

        if args or env:
            self._check_launch_options(args, env)

        script = "mobile: launchApp"
        params = {"bundleId": bundle_id}

        await asyncio.to_thread(driver.execute_script, script, params)
        return True

    @appium_operation
    async def terminate(self, bundle_id: str) -> bool:
        """Terminate an application by bundle ID.

        Args:
            bundle_id: The bundle identifier of the app to terminate

        Returns:
            True if termination succeeded, False otherwise
        """
        driver = self._ensure_driver()

        script = "mobile: terminateApp"
        params = {"bundleId": bundle_id}

        await asyncio.to_thread(driver.execute_script, script, params)
        return True

    @appium_operation
    async def text(self, text: str) -> bool:
        """Type text using the keyboard.

        Args:
            text: The text to type

        Returns:
            True if text input succeeded, False otherwise
        """
        driver = self._ensure_driver()

        active_element = await asyncio.to_thread(lambda: driver.switch_to.active_element)
        await asyncio.to_thread(active_element.send_keys, text)
        return True

    @appium_operation
    async def open_url(self, url: str) -> bool:
        """Open a URL on the device.

        Args:
            url: The URL to open

        Returns:
            True if URL opened successfully, False otherwise
        """
        driver = self._ensure_driver()

        await asyncio.to_thread(driver.get, url)
        return True


class _SecretFilter(logging.Filter):
    def __init__(self, secret: str):
        super().__init__()
        self.forms = {secret, json.dumps(secret)[1:-1], repr(secret)[1:-1]}

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "gads:clientSecret" in message:
            record.msg = "[Remote Appium capabilities redacted]"
            record.args = ()
        else:
            for secret in self.forms:
                if secret in message:
                    message = message.replace(secret, "[REDACTED]")
                    record.msg = message
                    record.args = ()
        return True


class _SameOriginConnection(AppiumConnection):
    """Keep SDK redirects and commands on the configured grid origin and prefix."""

    def __init__(self, hub_url: str, **kwargs):
        self._hub = urlsplit(hub_url)
        if "client_config" not in kwargs:
            kwargs["remote_server_addr"] = hub_url
        super().__init__(**kwargs)

    def _request(self, method, url, body=None):
        parsed = urlsplit(url)
        prefix = self._hub.path.rstrip("/")
        if (
            (parsed.scheme, parsed.hostname, parsed.port)
            != (self._hub.scheme, self._hub.hostname, self._hub.port)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "%" in parsed.path
            or "\\" in parsed.path
            or any(part in {".", ".."} for part in parsed.path.split("/"))
            or not parsed.path.startswith(prefix + "/")
        ):
            raise RemoteAppiumError("Remote Appium refused an unsafe command destination")
        return super()._request(method, url, body)


class RemoteAppiumClientWrapper(AppiumClientOperations):
    """Operate one explicitly selected existing iOS device through its grid session.

    All commands are serialized. Cancellation waits for the current SDK command;
    session deletion never races a still-running command or session creation.
    """

    MAX_IMAGE_BYTES = 8 * 1024 * 1024
    MAX_PIXELS = 16_000_000
    MAX_XML_BYTES = 4 * 1024 * 1024
    MAX_ELEMENTS = 20_000
    _SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,199}\Z")

    def __init__(self, config: RemoteAppiumClientConfig):
        super().__init__()
        self.config = config
        self.udid = config.udid
        self.platform = "ios"
        self._device_context: Any = None
        self._secret_filter = _SecretFilter(config.client_secret.get_secret_value())
        self._filtered_loggers = [
            logging.getLogger(name)
            for name in (
                "selenium.webdriver.remote.remote_connection",
                "appium.webdriver.appium_connection",
                "urllib3.connectionpool",
            )
        ]

    def bind_device_context(self, context) -> None:
        self._device_context = context

    def _publish_size(self, size: tuple[int, int]) -> tuple[int, int]:
        if self._device_context is not None:
            self._device_context.device_width, self._device_context.device_height = size
        return size

    @staticmethod
    def _logical_size(driver) -> tuple[int, int]:
        value = driver.get_window_size()
        dimensions = (value.get("width"), value.get("height"))
        if any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
            or number != int(number)
            for number in dimensions
        ):
            raise ValueError("Invalid logical screen size")
        size = (int(dimensions[0]), int(dimensions[1]))
        if size[0] * size[1] > RemoteAppiumClientWrapper.MAX_PIXELS:
            raise ValueError("Logical screen size exceeds limit")
        return size

    def _validate_point(self, driver, x, y):
        width, height = self._publish_size(self._logical_size(driver))
        if any(isinstance(n, bool) or not isinstance(n, (int, float)) for n in (x, y)):
            raise ValueError("Invalid coordinates")
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError("Coordinates outside logical screen")

    def _validate_tap_duration(self, duration):
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or not 0 <= duration <= 10
        ):
            raise ValueError("Invalid tap duration")

    def _log_protection(self, enable: bool) -> None:
        for logger in self._filtered_loggers:
            if enable:
                logger.addFilter(self._secret_filter)
            else:
                logger.removeFilter(self._secret_filter)

    async def _quit_owned(self) -> None:
        driver = self._driver
        if driver is None:
            self._log_protection(False)
            return
        # Never interpolate an untrusted server-supplied path into a DELETE.
        if not isinstance(driver.session_id, str) or not self._SESSION_ID.fullmatch(
            driver.session_id
        ):
            executor = getattr(driver, "command_executor", None)
            if executor is not None:
                executor.close()
            self._driver = None
            self._log_protection(False)
            raise RemoteAppiumError("Remote Appium returned an invalid session ID")
        await asyncio.to_thread(driver.quit)
        self._driver = None
        self._log_protection(False)

    @appium_operation
    async def init_client(self) -> bool:
        if self._driver is not None:
            return True
        self._log_protection(True)
        options = AppiumOptions().load_capabilities(
            {
                "platformName": "iOS",
                "appium:automationName": "XCUITest",
                "appium:udid": self.udid,
                "appium:noReset": True,
                "appium:fullReset": False,
                "appium:autoLaunch": False,
                "gads:clientSecret": self.config.client_secret.get_secret_value(),
                "gads:queueTimeout": self.config.queue_timeout,
            }
        )
        client_config = AppiumClientConfig(
            remote_server_addr=self.config.hub_url,
            timeout=self.config.timeout,
            direct_connection=False,
        )
        connection = _SameOriginConnection(self.config.hub_url, client_config=client_config)
        try:
            self._driver = await asyncio.to_thread(
                WebDriver, command_executor=connection, options=options, client_config=client_config
            )
            driver = self._ensure_driver()
            if not isinstance(driver.session_id, str) or not self._SESSION_ID.fullmatch(
                driver.session_id
            ):
                raise ValueError("Invalid session ID")
            returned_udid = (
                driver.capabilities.get("gads:deviceUdid")
                or driver.capabilities.get("appium:udid")
                or driver.capabilities.get("udid")
            )
            if returned_udid != self.udid:
                raise ValueError("Grid did not confirm the requested device")
            self._publish_size(await asyncio.to_thread(self._logical_size, driver))
            return True
        except Exception:
            if self._driver is not None:
                await self._quit_owned()
            else:
                connection.close()
                self._log_protection(False)
            raise

    async def _after_cancel(self, name: str) -> None:
        if name == "init_client":
            try:
                await self._quit_owned()
            except Exception:
                # The caller receives cancellation, not a server error containing credentials.
                pass

    @appium_operation
    async def cleanup(self) -> None:
        await self._quit_owned()

    async def __aenter__(self):
        await self.init_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    @appium_operation
    async def get_screen_size(self) -> tuple[int, int]:
        return self._publish_size(
            await asyncio.to_thread(self._logical_size, self._ensure_driver())
        )

    @appium_operation
    async def screenshot(self, output_path: str | None = None) -> bytes:
        def capture():
            driver = self._ensure_driver()
            for _ in range(2):
                size = self._logical_size(driver)
                encoded = driver.get_screenshot_as_base64()
                if not isinstance(encoded, str) or len(encoded) > self.MAX_IMAGE_BYTES * 4 // 3 + 4:
                    raise ValueError("Screenshot exceeds limit")
                data = base64.b64decode(encoded, validate=True)
                if self._logical_size(driver) != size:
                    continue
                with Image.open(BytesIO(data)) as image:
                    if image.format != "PNG" or image.width * image.height > self.MAX_PIXELS:
                        raise ValueError("Invalid or oversized screenshot")
                    image.load()
                    if image.size != size:
                        image = image.resize(size, Image.Resampling.LANCZOS)
                    output = BytesIO()
                    image.save(output, format="PNG")
                    return output.getvalue(), size
            raise ValueError("Orientation changed while capturing screenshot")

        data, size = await asyncio.to_thread(capture)
        self._publish_size(size)
        if output_path:
            await asyncio.to_thread(_write_bytes, output_path, data)
        return data

    def _read_ios_source(self) -> tuple[list[dict[str, Any]], str, tuple[int, int]]:
        driver = self._ensure_driver()
        for _ in range(2):
            size = self._logical_size(driver)
            source = driver.page_source
            if not isinstance(source, str) or len(source.encode()) > self.MAX_XML_BYTES:
                raise ValueError("UI source exceeds limit")
            if "<!DOCTYPE" in source.upper() or "<!ENTITY" in source.upper():
                raise ValueError("Unsupported XML declarations")
            if self._logical_size(driver) != size:
                continue
            elements = parse_ios_source(source)
            if len(elements) > self.MAX_ELEMENTS:
                raise ValueError("UI source contains too many elements")
            return elements, source, size
        raise ValueError("Orientation changed while reading UI source")

    @appium_operation
    async def describe_all(self) -> list[dict[str, Any]]:
        elements, _, size = await asyncio.to_thread(self._read_ios_source)
        self._publish_size(size)
        return elements

    @appium_operation
    async def describe_all_with_source(self) -> tuple[list[dict[str, Any]], str]:
        """Return the parsed elements and the exact XML used for this observation."""
        elements, source, size = await asyncio.to_thread(self._read_ios_source)
        self._publish_size(size)
        return elements, source

    @appium_operation
    async def swipe(self, x_start, y_start, x_end, y_end, duration: float | None = None) -> bool:
        driver = self._ensure_driver()
        seconds = 0.5 if duration is None else duration
        if (
            not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or not 0 < seconds <= 10
        ):
            raise ValueError("Invalid swipe duration")

        def perform():
            self._validate_point(driver, x_start, y_start)
            self._validate_point(driver, x_end, y_end)
            actions = ActionBuilder(
                driver,
                mouse=PointerInput(interaction.POINTER_TOUCH, "finger"),
                duration=round(seconds * 1000),
            )
            actions.pointer_action.move_to_location(x_start, y_start)
            actions.pointer_action.pointer_down()
            actions.pointer_action.move_to_location(x_end, y_end)
            actions.pointer_action.pointer_up()
            actions.perform()

        await asyncio.to_thread(perform)
        return True

    @appium_operation
    async def key(self, key_code: int) -> bool:
        keys = {42: Keys.BACKSPACE, 40: Keys.ENTER}
        if key_code not in keys:
            raise RemoteAppiumError("Remote Appium does not support this keyboard key")
        driver = self._ensure_driver()
        await asyncio.to_thread(lambda: driver.switch_to.active_element.send_keys(keys[key_code]))
        return True

    @appium_operation
    async def button(self, button_type: Any) -> bool:
        name = getattr(button_type, "name", str(button_type)).lower()
        buttons = {
            "home": "home",
            "volume_up": "volumeUp",
            "volumeup": "volumeUp",
            "volume_down": "volumeDown",
            "volumedown": "volumeDown",
        }
        if name not in buttons:
            raise RemoteAppiumError("Remote Appium does not support this hardware button")
        await asyncio.to_thread(
            self._ensure_driver().execute_script, "mobile: pressButton", {"name": buttons[name]}
        )
        return True

    async def home(self) -> bool:
        return await self.button("home")

    @appium_operation
    async def clear_text(self) -> bool:
        driver = self._ensure_driver()
        await asyncio.to_thread(lambda: driver.switch_to.active_element.clear())
        return True

    @appium_operation
    async def app_current(self) -> IOSAppInfo:
        info = await asyncio.to_thread(
            self._ensure_driver().execute_script, "mobile: activeAppInfo"
        )
        if (
            not isinstance(info, dict)
            or not isinstance(info.get("bundleId"), str)
            or not info["bundleId"]
        ):
            raise ValueError("Active application is unavailable")
        return IOSAppInfo(name=info.get("name"), bundle_id=info["bundleId"])

    async def install(self, app_path: str):
        raise RemoteAppiumError("Remote Appium client does not install applications")


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as file:
        file.write(data)
