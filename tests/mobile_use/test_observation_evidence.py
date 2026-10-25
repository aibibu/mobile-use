"""Evidence persistence tests using generated images and in-process model messages."""

import base64
import copy
import hashlib
import importlib
import json
import stat
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel, SecretStr

# A legacy outputter test installs a process-wide module mock during collection.
vertex_module = sys.modules.get("langchain_google_vertexai")
if vertex_module is not None and not hasattr(vertex_module, "__path__"):
    sys.modules.pop("langchain_google_vertexai", None)
    importlib.import_module("langchain_google_vertexai")

from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    ExecutionSetup,
    MobileUseContext,
)
from minitap.mobile_use.controllers.device_controller import ScreenDataResponse  # noqa: E402
from minitap.mobile_use.sdk import Agent  # noqa: E402
from minitap.mobile_use.sdk.builders.agent_config_builder import AgentConfigBuilder  # noqa: E402
from minitap.mobile_use.sdk.types.task import AgentProfile, Task  # noqa: E402
from minitap.mobile_use.utils import evidence  # noqa: E402


def fixture_llm_config() -> LLMConfig:
    node = LLMWithFallback(
        provider="deepseek",
        model="fixture-model",
        fallback=LLM(provider="deepseek", model="fixture-model"),
    )
    return LLMConfig(
        planner=node,
        orchestrator=node,
        contextor=node,
        cortex=node,
        executor=node,
        utils=LLMConfigUtils(outputter=node, hopper=node),
    )


def make_context(setup: ExecutionSetup | None = None) -> MobileUseContext:
    return MobileUseContext(
        trace_id="fixture-run",
        device=DeviceContext(
            host_platform="LINUX",
            mobile_platform=DevicePlatform.IOS,
            device_id="fixture-device",
            device_width=16,
            device_height=32,
        ),
        llm_config=fixture_llm_config(),
        execution_setup=setup,
    )


def image_bytes(format="PNG", color="red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 32), color).save(output, format=format)
    return output.getvalue()


def screen(raw_xml="<AppiumAUT/>\n", color="red") -> ScreenDataResponse:
    return ScreenDataResponse(
        base64=base64.b64encode(image_bytes(color=color)).decode(),
        raw_xml=raw_xml,
        elements=[{"label": "设置", "visible": False, "bounds": "[1,2][3,4]"}],
        width=16,
        height=32,
        platform="ios",
    )


@pytest.fixture
def context(tmp_path):
    root = tmp_path / "runs"
    root.mkdir(mode=0o700)
    run = root / "fixture-run"
    run.mkdir(mode=0o700)
    evidence.write_run_metadata(run, "run", {"task_id": "fixture-run"})
    return make_context(ExecutionSetup(traces_path=root, trace_name="fixture-run"))


def round_dir(context, observation_id):
    setup = context.execution_setup
    assert setup is not None and setup.traces_path is not None and setup.trace_name is not None
    return setup.traces_path / setup.trace_name / "evidence" / observation_id


@pytest.mark.asyncio
async def test_observation_keeps_exact_image_xml_and_processed_elements(context):
    captured = screen(raw_xml='<?xml version="1.0"?>\n<AppiumAUT label="设置 &amp; about"/>\n')
    observation_id = await evidence.begin_observation(context, captured)
    assert observation_id == "round-0001"
    folder = round_dir(context, observation_id)
    assert (folder / "screen.png").read_bytes() == base64.b64decode(captured.base64)
    assert captured.raw_xml is not None
    assert (folder / "source.xml").read_bytes() == captured.raw_xml.encode()
    assert json.loads((folder / "elements.json").read_text()) == captured.elements
    metadata = json.loads((folder / "observation.json").read_text())
    assert metadata["screen_sha256"] == hashlib.sha256(image_bytes()).hexdigest()
    assert metadata["element_count"] == 1
    assert metadata["raw_xml_available"] is True
    assert metadata["source"] == "source.xml"
    assert (metadata["width"], metadata["height"]) == (16, 32)


@pytest.mark.asyncio
async def test_model_input_keeps_original_jpeg_bytes_without_mutating_messages(context):
    observation_id = await evidence.begin_observation(context, screen())
    jpeg = image_bytes(format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    messages = [
        SystemMessage(content="Fixture system message"),
        HumanMessage(
            content=[
                {"type": "text", "text": "Read the current fixture screen"},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ]
        ),
    ]
    before = copy.deepcopy(messages)
    await evidence.record_model_input(
        context, observation_id, "cortex", messages, {"model": "first"}
    )
    assert messages == before
    assert messages[1].content == before[1].content

    folder = round_dir(context, observation_id)
    (input_file,) = folder.glob("cortex-*-input.json")
    payload = json.loads(input_file.read_text())
    image_ref = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert (folder / image_ref["artifact"]).read_bytes() == jpeg
    assert image_ref["sha256"] == hashlib.sha256(jpeg).hexdigest()
    assert image_ref["mime_type"] == "image/jpeg"
    assert payload["messages"][1]["content"][1]["image_url"]["detail"] == "high"
    assert data_url not in input_file.read_text()


@pytest.mark.asyncio
async def test_primary_and_fallback_inputs_in_one_round_do_not_overwrite(context, monkeypatch):
    observation_id = await evidence.begin_observation(context, screen())
    monkeypatch.setattr(evidence.time, "time_ns", lambda: 12345)
    messages = [HumanMessage(content="Fixture")]
    await evidence.record_model_input(
        context, observation_id, "cortex", messages, {"model": "primary"}
    )
    await evidence.record_model_input(
        context, observation_id, "cortex", messages, {"model": "fallback"}
    )
    folder = round_dir(context, observation_id)
    inputs = list(folder.glob("cortex-*-input.json"))
    assert len(inputs) == 2
    assert {json.loads(p.read_text())["model"]["model"] for p in inputs} == {"primary", "fallback"}
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    calls = [event for event in events if event["kind"] == "cortex_input"]
    assert len(calls) == 2
    assert len({event["payload"]["artifact"] for event in calls}) == 2
    assert all((folder / event["payload"]["artifact"]).is_file() for event in calls)


@pytest.mark.asyncio
async def test_evidence_without_trace_is_noop(tmp_path):
    context = make_context()
    captured = screen()
    captured.base64 = "not even a valid base64 image"
    assert await evidence.begin_observation(context, captured) is None
    await evidence.record_model_input(context, "round-0001", "cortex", [], SecretStr("private"))
    await evidence.record_event(context, "round-0001", "fixture", {"api_key": "private"})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_rounds_have_independent_artifacts(context):
    first = await evidence.begin_observation(context, screen("<first/>", color="red"))
    second = await evidence.begin_observation(context, screen("<second/>", color="blue"))
    assert (first, second) == ("round-0001", "round-0002")
    assert (round_dir(context, first) / "source.xml").read_text() == "<first/>"
    assert (round_dir(context, second) / "source.xml").read_text() == "<second/>"
    assert (round_dir(context, first) / "screen.png").read_bytes() == image_bytes(color="red")
    assert (round_dir(context, second) / "screen.png").read_bytes() == image_bytes(color="blue")


class SecretConfig(BaseModel):
    credential: SecretStr


@pytest.mark.asyncio
async def test_structured_secret_values_are_not_persisted(context):
    observation_id = await evidence.begin_observation(context, screen())
    secrets = {
        "api_key": "fixture-api-key",
        "apiKey": "fixture-camel-key",
        "CLIENT_SECRET": "fixture-client-secret",
        "Authorization": "Bearer fixture-token",
        "password": "fixture-password",
        "DEEPSEEK_API_KEY": "fixture-provider-key",
        "access_token": "fixture-access-token",
        "refreshToken": "fixture-refresh-token",
        "clientSecret": "fixture-camel-client",
        "nested": [SecretStr("fixture-secret-str")],
        "model_config": SecretConfig(credential=SecretStr("fixture-model-secret")),
    }
    await evidence.record_event(context, observation_id, "fixture", secrets)
    await evidence.record_model_input(context, observation_id, "cortex", [], secrets)
    folder = round_dir(context, observation_id)
    persisted = "\n".join(path.read_text() for path in folder.glob("*.json*"))
    for secret in (
        "fixture-api-key",
        "fixture-camel-key",
        "fixture-client-secret",
        "fixture-token",
        "fixture-password",
        "fixture-secret-str",
        "fixture-model-secret",
        "fixture-provider-key",
        "fixture-access-token",
        "fixture-refresh-token",
        "fixture-camel-client",
    ):
        assert secret not in persisted


@pytest.mark.asyncio
async def test_artifacts_and_created_trace_directories_are_private(context):
    observation_id = await evidence.begin_observation(context, screen())
    await evidence.record_model_input(
        context, observation_id, "cortex", [HumanMessage(content="x")], {}
    )
    setup = context.execution_setup
    assert setup is not None and setup.traces_path is not None
    assert stat.S_IMODE(setup.traces_path.stat().st_mode) == 0o700
    for path in setup.traces_path.rglob("*"):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected, path


def fixture_task(agent: Agent, context: MobileUseContext, task_id: str, root: Path) -> Task:
    request = (
        agent.new_task("Fixture goal")
        .with_name("Same task name")
        .with_trace_recording(True, str(root))
        .build()
    )
    return Task(
        id=task_id,
        device=context.device,
        status="running",
        request=request,
        created_at=datetime(2026, 10, 25, 12, 0, 0),
    )


def fixture_agent() -> Agent:
    config = (
        AgentConfigBuilder()
        .add_profile(AgentProfile(name="fixture", llm_config=fixture_llm_config()), validate=False)
        .build(validate_profiles=False)
    )
    return Agent(config=config)


def test_same_named_sdk_tasks_use_different_trace_directories(tmp_path):
    agent = fixture_agent()
    first_ctx, second_ctx = make_context(), make_context()
    first = fixture_task(agent, first_ctx, "first-task-id", tmp_path)
    second = fixture_task(agent, second_ctx, "second-task-id", tmp_path)
    agent._prepare_tracing(first, first_ctx)
    agent._prepare_tracing(second, second_ctx)
    assert (tmp_path / first.id / "evidence" / "run.json").is_file()
    assert (tmp_path / second.id / "evidence" / "run.json").is_file()
    assert first_ctx.execution_setup is not None and second_ctx.execution_setup is not None
    assert first_ctx.execution_setup.trace_name != second_ctx.execution_setup.trace_name


@pytest.mark.asyncio
async def test_gif_failure_keeps_raw_evidence_and_records_artifact_error(tmp_path, monkeypatch):
    agent = fixture_agent()
    context = make_context()
    task = fixture_task(agent, context, "retained-task-id", tmp_path)
    agent._prepare_tracing(task, context)
    observation_id = await evidence.begin_observation(context, screen("<retained/>"))

    def fail_gif(*args, **kwargs):
        raise RuntimeError("fixture GIF failure")

    monkeypatch.setattr("minitap.mobile_use.sdk.agent.create_gif_from_trace_folder", fail_gif)
    await agent._finalize_tracing(task, context)
    directories = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(directories) == 1
    folder = directories[0] / "evidence"
    assert (folder / observation_id / "source.xml").read_text() == "<retained/>"
    assert (folder / observation_id / "screen.png").read_bytes() == image_bytes()
    result = json.loads((folder / "result.json").read_text())
    assert result["artifact_error"] == "RuntimeError"
    assert result["task_id"] == task.id
