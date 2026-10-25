import asyncio
import importlib
import sys
from io import BytesIO
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from pydantic import SecretStr

# A legacy outputter test installs a process-wide module mock during collection.
vertex_module = sys.modules.get("langchain_google_vertexai")
if vertex_module is not None and not hasattr(vertex_module, "__path__"):
    sys.modules.pop("langchain_google_vertexai", None)
    importlib.import_module("langchain_google_vertexai")

from minitap.mobile_use.clients.appium_client import RemoteAppiumClientWrapper  # noqa: E402
from minitap.mobile_use.clients.ios_client import DeviceType  # noqa: E402
from minitap.mobile_use.clients.remote_appium_config import RemoteAppiumClientConfig  # noqa: E402
from minitap.mobile_use.context import DevicePlatform  # noqa: E402
from minitap.mobile_use.controllers.controller_factory import create_device_controller  # noqa: E402
from minitap.mobile_use.controllers.ios_controller import iOSDeviceController  # noqa: E402
from minitap.mobile_use.controllers.types import CoordinatesSelectorRequest  # noqa: E402
from minitap.mobile_use.sdk import Agent  # noqa: E402
from minitap.mobile_use.sdk.builders import Builders  # noqa: E402
from minitap.mobile_use.sdk.types.agent import AgentConfig  # noqa: E402


def image_bytes(size=(390, 844)):
    stream = BytesIO()
    Image.new("RGB", size).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def remote_config():
    return RemoteAppiumClientConfig(
        hub_url="http://127.0.0.1:18100/grid",
        client_secret=SecretStr("fixture-only-secret"),
        udid="fixture-iphone",
    )


@pytest.fixture
def agent(remote_config, monkeypatch):
    config = Builders.AgentConfig.build(validate_profiles=False).model_copy(
        update={"remote_appium_config": remote_config}
    )
    instance = Agent(config=config)
    monkeypatch.setattr(instance, "_init_clients", Mock(side_effect=AssertionError("local init")))
    monkeypatch.setattr(instance, "_run_servers", Mock(side_effect=AssertionError("local server")))
    monkeypatch.setattr(RemoteAppiumClientWrapper, "init_client", AsyncMock(return_value=True))
    monkeypatch.setattr(
        RemoteAppiumClientWrapper, "get_screen_size", AsyncMock(return_value=(390, 844))
    )
    monkeypatch.setattr(RemoteAppiumClientWrapper, "cleanup", AsyncMock())
    return instance


@pytest.mark.asyncio
async def test_remote_sdk_uses_logical_dimensions_without_local_services(agent, monkeypatch):
    monkeypatch.setattr(
        "minitap.mobile_use.sdk.agent.which", Mock(side_effect=AssertionError("local discovery"))
    )
    assert await agent._init_internal()
    assert agent._ios_device_type is DeviceType.REMOTE_APPIUM
    assert agent._device_context.mobile_platform is DevicePlatform.IOS
    assert (agent._device_context.device_width, agent._device_context.device_height) == (390, 844)
    assert agent._device_context.device_id == "fixture-iphone"
    assert agent._adb_client is None
    assert agent._ui_adb_client is None
    first_client = agent._ios_client
    assert await agent._init_internal()
    assert agent._ios_client is first_client
    cast(AsyncMock, RemoteAppiumClientWrapper.init_client).assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_initialization_creates_only_one_session(agent):
    assert await asyncio.gather(agent._init_internal(), agent._init_internal()) == [True, True]
    cast(AsyncMock, RemoteAppiumClientWrapper.init_client).assert_awaited_once()


@pytest.mark.asyncio
async def test_clean_before_initialization_does_not_create_or_release_a_session(agent):
    await agent.clean()
    cast(AsyncMock, RemoteAppiumClientWrapper.init_client).assert_not_awaited()
    cast(AsyncMock, RemoteAppiumClientWrapper.cleanup).assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("device_platform", DevicePlatform.ANDROID), ("video_recording_enabled", True)]
)
async def test_mutated_remote_config_is_revalidated_before_connecting(agent, field, value):
    setattr(agent._config, field, value)
    with pytest.raises(ValueError):
        await agent._init_internal()
    cast(AsyncMock, RemoteAppiumClientWrapper.init_client).assert_not_awaited()


@pytest.mark.asyncio
async def test_remote_sdk_tasks_receive_the_same_remote_client(agent, monkeypatch):
    await agent._init_internal()
    contexts = []

    async def capture_graph(context):
        contexts.append(context)
        controller = create_device_controller(context)
        assert isinstance(controller, iOSDeviceController)
        assert controller.ios_client is agent._ios_client
        raise RuntimeError("fixture stops before any model call")

    monkeypatch.setattr("minitap.mobile_use.sdk.agent.get_graph", capture_graph)
    with pytest.raises(RuntimeError, match="fixture stops"):
        await agent.run_task(goal="Read the fixture screen")
    assert len(contexts) == 1
    assert contexts[0].device is agent._device_context
    assert contexts[0].adb_client is None


@pytest.mark.asyncio
async def test_remote_sdk_screenshot_never_calls_xcrun(agent, monkeypatch):
    await agent._init_internal()
    monkeypatch.setattr(
        RemoteAppiumClientWrapper, "screenshot", AsyncMock(return_value=image_bytes())
    )
    monkeypatch.setattr("subprocess.run", Mock(side_effect=AssertionError("local subprocess")))
    screenshot = await agent.get_screenshot()
    assert screenshot.size == (390, 844)


@pytest.mark.asyncio
@pytest.mark.parametrize("dimensions", [(0, 844), (390, -1), (True, 844)])
async def test_invalid_remote_dimensions_cleanup_without_fallback(agent, monkeypatch, dimensions):
    monkeypatch.setattr(
        RemoteAppiumClientWrapper, "get_screen_size", AsyncMock(return_value=dimensions)
    )
    with pytest.raises(ValueError, match="invalid logical"):
        await agent._init_internal()
    cast(AsyncMock, RemoteAppiumClientWrapper.cleanup).assert_awaited_once()
    assert agent._ios_client is None
    assert not agent._initialized


@pytest.mark.asyncio
async def test_cancelled_remote_initialization_releases_created_session(agent, monkeypatch):
    started = asyncio.Event()

    async def blocked_dimensions(self):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(RemoteAppiumClientWrapper, "get_screen_size", blocked_dimensions)
    task = asyncio.create_task(agent._init_internal())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cast(AsyncMock, RemoteAppiumClientWrapper.cleanup).assert_awaited_once()
    assert agent._ios_client is None
    assert not agent._initialized


@pytest.mark.asyncio
async def test_cleanup_retries_partial_initialization(agent, monkeypatch):
    monkeypatch.setattr(
        RemoteAppiumClientWrapper,
        "get_screen_size",
        AsyncMock(side_effect=RuntimeError("fixture read failure")),
    )
    cleanup = AsyncMock(side_effect=[RuntimeError("fixture close failure"), None])
    monkeypatch.setattr(RemoteAppiumClientWrapper, "cleanup", cleanup)
    with pytest.raises(RuntimeError, match="close failure"):
        await agent._init_internal()
    assert agent._ios_client is not None
    assert not agent._initialized
    await agent.clean()
    assert cleanup.await_count == 2
    assert agent._ios_client is None


@pytest.mark.asyncio
async def test_cleanup_waits_for_task_cancellation_before_releasing_session(agent, monkeypatch):
    await agent._init_internal()
    events = []
    running = asyncio.Event()

    async def execute_task():
        running.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append("task-stopped")

    async def close():
        events.append("session-closed")

    monkeypatch.setattr(agent._ios_client, "cleanup", close)
    agent._current_task = asyncio.create_task(execute_task())
    await running.wait()
    await agent.clean()
    assert events == ["task-stopped", "session-closed"]
    assert not agent._initialized


@pytest.mark.asyncio
async def test_ios_controller_does_not_report_failed_actions_as_success(remote_config, monkeypatch):
    client = RemoteAppiumClientWrapper(config=remote_config)
    monkeypatch.setattr(client, "tap", AsyncMock(return_value=False))
    monkeypatch.setattr(client, "swipe", AsyncMock(return_value=False))
    monkeypatch.setattr(client, "key", AsyncMock(return_value=False))
    controller = iOSDeviceController(client, "fixture-iphone", 390, 844)
    point = CoordinatesSelectorRequest(x=10, y=20)
    assert (await controller.tap(point)).error is not None
    assert await controller.swipe(point, point) is not None
    assert not await controller.press_enter()
    assert not await controller.erase_text(1)


@pytest.mark.asyncio
async def test_remote_controller_refreshes_rotation_and_rejects_mixed_screen(
    remote_config, monkeypatch
):
    client = RemoteAppiumClientWrapper(config=remote_config)
    monkeypatch.setattr(client, "screenshot", AsyncMock(return_value=image_bytes((844, 390))))
    monkeypatch.setattr(client, "describe_all", AsyncMock(return_value=[]))
    monkeypatch.setattr(client, "get_screen_size", AsyncMock(return_value=(844, 390)))
    controller = iOSDeviceController(client, "fixture-iphone", 390, 844)
    screen = await controller.get_screen_data()
    assert (screen.width, screen.height) == (844, 390)

    monkeypatch.setattr(client, "screenshot", AsyncMock(return_value=image_bytes((390, 844))))
    with pytest.raises(RuntimeError, match="rotated"):
        await controller.get_screen_data()


@pytest.mark.asyncio
async def test_remote_hierarchy_failure_is_not_an_empty_success(remote_config, monkeypatch):
    client = RemoteAppiumClientWrapper(config=remote_config)
    monkeypatch.setattr(
        client, "describe_all", AsyncMock(side_effect=RuntimeError("fixture failure"))
    )
    controller = iOSDeviceController(client, "fixture-iphone", 390, 844)
    with pytest.raises(RuntimeError, match="fixture failure"):
        await controller.get_ui_hierarchy()


@pytest.mark.asyncio
async def test_remote_hierarchy_preserves_hidden_elements(remote_config, monkeypatch):
    client = RemoteAppiumClientWrapper(config=remote_config)
    monkeypatch.setattr(
        client,
        "describe_all",
        AsyncMock(
            return_value=[
                {
                    "type": "XCUIElementTypeButton",
                    "label": "Hidden fixture button",
                    "visible": False,
                    "enabled": True,
                    "frame": {"x": 0, "y": 0, "width": 50, "height": 20},
                }
            ]
        ),
    )
    monkeypatch.setattr(client, "get_screen_size", AsyncMock(return_value=(390, 844)))
    controller = iOSDeviceController(client, "fixture-iphone", 390, 844)
    elements = await controller.get_ui_hierarchy()
    assert len(elements) == 1
    assert elements[0]["visible"] is False


def test_remote_provider_configuration_rejects_an_android_target(remote_config):
    original = Builders.AgentConfig.build(validate_profiles=False)
    with pytest.raises(ValueError, match="iOS only"):
        AgentConfig(
            **{
                **original.__dict__,
                "remote_appium_config": remote_config,
                "device_platform": DevicePlatform.ANDROID,
            }
        )


def test_remote_provider_rejects_local_video_recording(remote_config):
    original = Builders.AgentConfig.build(validate_profiles=False)
    with pytest.raises(ValueError, match="Video recording"):
        AgentConfig(
            **{
                **original.__dict__,
                "remote_appium_config": remote_config,
                "video_recording_enabled": True,
            }
        )
