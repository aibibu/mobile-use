import argparse
import importlib
import json
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr

# A legacy outputter test installs a process-wide module mock during collection.
vertex_module = sys.modules.get("langchain_google_vertexai")
if vertex_module is not None and not hasattr(vertex_module, "__path__"):
    sys.modules.pop("langchain_google_vertexai", None)
    importlib.import_module("langchain_google_vertexai")

from minitap.mobile_use import config as model_config  # noqa: E402
from minitap.mobile_use import sdk  # noqa: E402
from minitap.mobile_use.clients.remote_appium_config import RemoteAppiumClientConfig  # noqa: E402
from scripts import gads  # noqa: E402


def fixture_connection():
    return RemoteAppiumClientConfig(
        hub_url="http://127.0.0.1:18100/grid",
        udid="fixture-device",
        client_secret=SecretStr("fixture-only-secret"),
    )


@pytest.mark.asyncio
async def test_observe_does_not_initialize_model_configuration(monkeypatch):
    connection = fixture_connection()
    monkeypatch.setattr(gads, "connection_config", lambda: connection)
    observe = AsyncMock()
    initialize = Mock(side_effect=AssertionError("observe must not initialize an LLM"))
    agent_class = Mock(side_effect=AssertionError("observe must not create an Agent"))
    monkeypatch.setattr(gads, "observe", observe)
    monkeypatch.setattr(model_config, "initialize_llm_config", initialize)
    monkeypatch.setattr(sdk, "Agent", agent_class)

    await gads.run(argparse.Namespace(goal=None, output_dir=None, max_steps=10))

    observe.assert_awaited_once_with(connection, None)
    initialize.assert_not_called()
    agent_class.assert_not_called()


@pytest.mark.asyncio
async def test_observe_loads_project_dotenv_from_another_cwd_without_overriding_process(
    monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text(
        "GADS_APPIUM_URL=http://127.0.0.1:18100/grid\n"
        "GADS_DEVICE_UDID=fixture-file-device\n"
        "GADS_CLIENT_SECRET=fixture-file-secret\n"
    )
    other_cwd = tmp_path / "another-directory"
    other_cwd.mkdir()
    monkeypatch.setattr(gads, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(other_cwd)
    for name in ("GADS_APPIUM_URL", "GADS_CLIENT_SECRET"):
        # Track restoration even if the variable was initially absent: dotenv
        # writes directly to os.environ rather than through monkeypatch.
        monkeypatch.setenv(name, "fixture-reset")
        monkeypatch.delenv(name)
    monkeypatch.setenv("GADS_DEVICE_UDID", "fixture-process-device")
    observe = AsyncMock()
    initialize = Mock(side_effect=AssertionError("observe must not initialize an LLM"))
    monkeypatch.setattr(gads, "observe", observe)
    monkeypatch.setattr(model_config, "initialize_llm_config", initialize)

    await gads.run(argparse.Namespace(goal=None, output_dir=None, max_steps=10))

    observe.assert_awaited_once()
    assert observe.await_args is not None
    connection, output_dir = observe.await_args.args
    assert connection.hub_url == "http://127.0.0.1:18100/grid"
    assert connection.udid == "fixture-process-device"
    assert connection.client_secret == SecretStr("fixture-file-secret")
    assert output_dir is None
    initialize.assert_not_called()


@pytest.mark.asyncio
async def test_goal_loads_local_override_into_the_sdk_profile(monkeypatch, tmp_path):
    node = {
        "provider": "openai",
        "model": "fixture-default",
        "fallback": {"provider": "openai", "model": "fixture-fallback"},
    }
    default = {
        role: node for role in ("planner", "orchestrator", "contextor", "cortex", "executor")
    }
    default["utils"] = {role: node for role in ("hopper", "outputter")}
    (tmp_path / model_config.DEFAULT_LLM_CONFIG_FILENAME).write_text(
        json.dumps({"default": default})
    )
    (tmp_path / model_config.OVERRIDE_LLM_CONFIG_FILENAME).write_text(
        json.dumps({"cortex": {"provider": "openai", "model": "fixture-vision-override"}})
    )
    monkeypatch.setattr(model_config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(model_config.settings, "OPENAI_API_KEY", SecretStr("fixture-model-secret"))
    monkeypatch.setattr(
        model_config.settings, "MINITAP_API_KEY", SecretStr("fixture-other-provider")
    )
    connection = fixture_connection()
    monkeypatch.setattr(gads, "connection_config", lambda: connection)
    captured = {}
    original_agent = sdk.Agent

    class FixtureAgent:
        def __init__(self, *, config):
            self.delegate = original_agent(config=config)
            captured["config"] = config

        async def init(self, *args, **kwargs):
            captured["initialized"] = True
            return True

        def new_task(self, goal):
            return self.delegate.new_task(goal)

        async def run_task(self, request=None, **kwargs):
            captured["request"] = request
            return "fixture: no device or model calls"

        async def clean(self, force=False):
            captured["cleaned"] = True

    monkeypatch.setattr(sdk, "Agent", FixtureAgent)

    await gads.run(argparse.Namespace(goal="Read the fixture title", output_dir=None, max_steps=10))

    assert captured["initialized"] and captured["cleaned"]
    actual_config = captured["config"]
    assert actual_config.default_profile.name == "gads"
    assert actual_config.default_profile.llm_config.cortex.model == "fixture-vision-override"
    assert actual_config.default_profile.llm_config.cortex.provider == "openai"
    assert actual_config.remote_appium_config is connection
    assert captured["request"].goal == "Read the fixture title"
    assert captured["request"].max_steps == 10


def test_max_steps_help_distinguishes_graph_steps_from_phone_actions(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gads.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        gads.arguments()
    assert exit_info.value.code == 0
    assert "LangGraph execution steps" in capsys.readouterr().out
