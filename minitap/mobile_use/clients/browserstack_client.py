import asyncio
from functools import wraps
from typing import Any

from appium.options.common.base import AppiumOptions
from appium.webdriver.webdriver import WebDriver

from minitap.mobile_use.clients.appium_client import AppiumClientOperations, parse_ios_source
from minitap.mobile_use.clients.idb_client import IOSAppInfo
from minitap.mobile_use.clients.ios_client_config import BrowserStackClientConfig
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

BROWSERSTACK_HUB_URL = "https://hub-cloud.browserstack.com/wd/hub"


def with_browserstack_client(func):
    """Decorator to handle BrowserStack client error handling.

    Note: Function must have None or bool in return type for error fallback.
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        method_name = func.__name__
        try:
            logger.debug(f"Executing BrowserStack operation: {method_name}...")
            result = await func(self, *args, **kwargs)
            logger.debug(f"{method_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to {method_name}: {e}")
            import traceback

            logger.debug(f"Traceback: {traceback.format_exc()}")

            return_type = func.__annotations__.get("return")
            if return_type is bool:
                return False
            return None

    return wrapper


class BrowserStackClientWrapper(AppiumClientOperations):
    """Wrapper around Appium WebDriver for BrowserStack iOS device automation.

    This wrapper provides an interface similar to IdbClientWrapper and WdaClientWrapper
    but uses BrowserStack's cloud infrastructure for physical iOS device automation.

    BrowserStack is used for:
    - Cloud-based physical iOS devices
    - CI/CD pipelines requiring real device testing
    - Cross-device testing without local hardware

    Prerequisites:
        1. BrowserStack account with App Automate access
        2. Valid username and access_key
        3. App uploaded to BrowserStack (app_url)

    Example:
        config = BrowserStackClientConfig(
            username="your_username",
            access_key="your_access_key",
            device_name="iPhone 14",
            platform_version="16",
            app_url="bs://your_app_hash",
        )
        wrapper = BrowserStackClientWrapper(config=config)
        await wrapper.init_client()
        await wrapper.tap(100, 200)
        await wrapper.cleanup()

        # Using context manager
        async with BrowserStackClientWrapper(config=config) as wrapper:
            await wrapper.tap(100, 200)
            screenshot = await wrapper.screenshot()
    """

    def __init__(self, config: BrowserStackClientConfig):
        """Initialize the BrowserStack client wrapper.

        Args:
            config: BrowserStack configuration with credentials and device settings
        """
        super().__init__()
        self.config = config

    def _operation_error(self, name: str, error: Exception, return_type):
        # Preserve BrowserStack's historical bool/None fallback contract.
        logger.error(f"Failed to {name}: {error}")
        return False if return_type is bool else None

    def _check_launch_options(self, args, env):
        logger.warning(
            "BrowserStack does not support app launch arguments or environment variables"
        )

    async def init_client(self) -> bool:
        """Initialize the Appium WebDriver session on BrowserStack.

        Returns:
            True if session created successfully, False otherwise
        """
        try:
            logger.info(
                f"Creating BrowserStack session for {self.config.device_name} "
                f"(iOS {self.config.platform_version})"
            )

            options = AppiumOptions()

            options.set_capability("platformName", "iOS")
            options.set_capability("appium:deviceName", self.config.device_name)
            options.set_capability("appium:platformVersion", self.config.platform_version)
            options.set_capability("appium:automationName", "XCUITest")
            options.set_capability("appium:app", self.config.app_url)

            bstack_options: dict[str, Any] = {
                "userName": self.config.username,
                "accessKey": self.config.access_key.get_secret_value(),
                "buildName": self.config.build_name or "mobile-use-session",
                "sessionName": self.config.session_name or "BrowserStack Session",
                "debug": True,
            }

            if self.config.project_name:
                bstack_options["projectName"] = self.config.project_name

            options.set_capability("bstack:options", bstack_options)

            hub_url = self.config.hub_url or BROWSERSTACK_HUB_URL

            self._driver = await asyncio.to_thread(
                WebDriver,
                command_executor=hub_url,
                options=options,
            )

            if self._driver:
                session_id = self._driver.session_id
                logger.info(f"BrowserStack session created successfully. Session ID: {session_id}")
                logger.info(
                    f"View session: https://app-automate.browserstack.com/dashboard/v2/sessions/{session_id}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to create BrowserStack session: {e}")
            self._driver = None
            return False

    async def cleanup(self) -> None:
        """Clean up BrowserStack session and quit the driver."""
        if self._driver is not None:
            try:
                logger.info("Ending BrowserStack session")
                await asyncio.to_thread(self._driver.quit)
            except Exception as e:
                logger.debug(f"Error ending BrowserStack session: {e}")
            finally:
                self._driver = None

        logger.debug("BrowserStack client cleanup completed")

    async def __aenter__(self):
        """Async context manager entry."""
        if not await self.init_client():
            raise RuntimeError("Failed to create BrowserStack session")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False

    def _ensure_driver(self) -> WebDriver:
        """Ensure a valid WebDriver session exists.

        Returns:
            The WebDriver instance

        Raises:
            RuntimeError: If no driver is available
        """
        if self._driver is None:
            raise RuntimeError(
                "BrowserStack session not initialized. "
                "Call init_client() first or use as context manager."
            )
        return self._driver

    @with_browserstack_client
    async def key(self, key_code: int) -> bool:
        """Send a key press.

        Note: Limited key support on BrowserStack/Appium.
        For delete (key_code=42), we send a backspace.

        Args:
            key_code: HID key code (42 = delete/backspace)

        Returns:
            True if key press succeeded, False otherwise
        """
        driver = self._ensure_driver()

        if key_code == 42:  # Delete/backspace
            active_element = await asyncio.to_thread(lambda: driver.switch_to.active_element)
            current_text = await asyncio.to_thread(lambda: active_element.text)
            if current_text:
                await asyncio.to_thread(active_element.clear)
                await asyncio.to_thread(active_element.send_keys, current_text[:-1])
        return True

    @with_browserstack_client
    async def button(self, button_type: Any) -> bool:
        """Press a hardware button (compatible with IDB's HIDButtonType).

        Args:
            button_type: Button type (HIDButtonType.HOME, etc.)

        Returns:
            True if button press succeeded, False otherwise
        """
        driver = self._ensure_driver()

        button_name = getattr(button_type, "name", str(button_type)).lower()

        if button_name == "home":
            script = "mobile: pressButton"
            params = {"name": "home"}
            await asyncio.to_thread(driver.execute_script, script, params)
        elif button_name in ("volume_up", "volumeup"):
            script = "mobile: pressButton"
            params = {"name": "volumeUp"}
            await asyncio.to_thread(driver.execute_script, script, params)
        elif button_name in ("volume_down", "volumedown"):
            script = "mobile: pressButton"
            params = {"name": "volumeDown"}
            await asyncio.to_thread(driver.execute_script, script, params)

        return True

    async def describe_all(self) -> list[dict[str, Any]] | None:
        """Get UI hierarchy as a flat list (compatible with IDB's describe_all).

        Returns:
            List of UI elements or None on error
        """
        try:
            driver = self._ensure_driver()
            page_source = await asyncio.to_thread(lambda: driver.page_source)
            if page_source is None:
                return None
            return self._parse_xml_to_elements(page_source)
        except Exception as e:
            logger.error(f"Failed to describe_all: {e}")
            return None

    def _parse_xml_to_elements(self, xml_source: str) -> list[dict[str, Any]]:
        """Parse Appium XML source into flat element list matching IDB format."""
        try:
            return parse_ios_source(xml_source)
        except Exception as error:
            logger.error(f"Failed to parse XML: {error}")
            return []

    async def app_current(self) -> IOSAppInfo | None:
        """Get information about the currently active app.

        Note: BrowserStack doesn't support activeAppInfo script directly.
        Returns None as this feature is not available on BrowserStack.

        Returns:
            None (not supported on BrowserStack)
        """
        logger.debug("app_current is not supported on BrowserStack")
        return None

    async def install(self, app_path: str) -> list[Any]:
        """Install an app (not supported on BrowserStack - apps must be pre-uploaded).

        Args:
            app_path: Path to the app (ignored)

        Returns:
            Empty list with warning
        """
        logger.warning(
            "App installation not supported on BrowserStack. "
            "Please upload your app to BrowserStack first and use the app_url in config."
        )
        return []
