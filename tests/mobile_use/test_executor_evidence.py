"""Record the actual executor inputs and tool outcomes without changing execution."""

import asyncio
from types import SimpleNamespace
from typing import Annotated, NotRequired, TypedDict, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.runtime import Runtime
from pydantic import BaseModel

from minitap.mobile_use.agents.executor import executor
from minitap.mobile_use.agents.executor import tool_node
from minitap.mobile_use.context import MobileUseContext

KEY = "executor_messages"


class ToolState(TypedDict):
    executor_messages: Annotated[list, add_messages]
    marker: str
    observation_id: NotRequired[str | None]


def action_tool(executed):
    @tool
    async def action(
        value: int,
        agent_thought: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[ToolState, InjectedState],
    ) -> ToolMessage:
        """Record a deterministic fixture action; negative values fail."""
        executed.append((value, state["marker"]))
        return ToolMessage(
            tool_call_id=tool_call_id,
            content=f"action:{value}",
            status="success" if value >= 0 else "error",
        )

    return action


def call(index, value):
    return {
        "name": "action",
        "args": {"value": value, "agent_thought": "private rationale"},
        "id": f"c{index}",
        "type": "tool_call",
    }


def tool_input(calls) -> ToolState:
    return {
        "executor_messages": [AIMessage(content="", tool_calls=calls)],
        "marker": "injected state only",
        "observation_id": "observation-1",
    }


@pytest.mark.asyncio
async def test_tool_evidence_orders_start_end_and_marks_unexecuted_calls_skipped(monkeypatch):
    events, executed = [], []
    context = cast(MobileUseContext, object())

    async def record(ctx, observation_id, kind, payload):
        assert ctx is context and observation_id == "observation-1"
        events.append((kind, payload))

    monkeypatch.setattr(tool_node, "record_event", record)
    node = tool_node.ExecutorToolNode(
        [action_tool(executed)], messages_key=KEY, evidence_context=context
    )
    graph = StateGraph(ToolState)
    graph.add_node("tools", node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    result = await graph.compile().ainvoke(tool_input([call(1, 1), call(2, -1), call(3, 3)]))

    assert executed == [(1, "injected state only"), (-1, "injected state only")]
    assert [(kind, p["call_id"], p.get("status")) for kind, p in events] == [
        ("tool_start", "c1", None),
        ("tool_end", "c1", "success"),
        ("tool_start", "c2", None),
        ("tool_end", "c2", "error"),
        ("tool_end", "c3", "skipped"),
    ]
    assert events[1][1]["result"] == {"status": "success", "content": "action:1"}
    assert events[3][1]["result"] == {"status": "error", "content": "action:-1"}
    assert [p["arguments"] for kind, p in events if kind == "tool_start"] == [
        {"value": 1},
        {"value": -1},
    ]
    assert [m.status for m in result[KEY] if isinstance(m, ToolMessage)] == [
        "success",
        "error",
        "error",
    ]


def test_recorded_arguments_exclude_injected_context_credentials_and_reasoning():
    original = {
        "id": "call-1",
        "name": "swipe_coordinates",
        "args": {
            "start_x": 320,
            "end_x": 40,
            "agent_thought": "private rationale",
            "state": {"screenshot": "private"},
            "runtime": object(),
            "config": {"key": "private"},
            "nested": {"api_key": "private", "duration": 400, "clientSecret": "private"},
        },
    }
    assert tool_node.safe_tool_call(original) == {
        "call_id": "call-1",
        "name": "swipe_coordinates",
        "arguments": {"start_x": 320, "end_x": 40, "nested": {"duration": 400}},
    }
    assert original["args"]["nested"]["api_key"] == "private"
    assert original["args"]["agent_thought"] == "private rationale"


@pytest.mark.asyncio
async def test_tool_evidence_reads_observation_id_from_a_state_model(monkeypatch):
    class ModelState(BaseModel):
        executor_messages: list[AIMessage]
        observation_id: str

    observations = []

    async def record(_ctx, observation_id, _kind, _payload):
        observations.append(observation_id)

    async def run(*_args):
        return ToolMessage(content="ok", tool_call_id="c1", status="success")

    monkeypatch.setattr(tool_node, "record_event", record)
    node = tool_node.ExecutorToolNode(
        [action_tool([])], messages_key=KEY, evidence_context=cast(MobileUseContext, object())
    )
    monkeypatch.setattr(node, "_arun_one", run)
    monkeypatch.setattr(node, "_build_tool_runtime", lambda *_args: None)
    state = ModelState(
        executor_messages=[AIMessage(content="", tool_calls=[call(1, 1)])],
        observation_id="model-observation",
    )
    await node._afunc(state, {}, Runtime())
    assert observations == ["model-observation", "model-observation"]


@pytest.mark.asyncio
@pytest.mark.parametrize("recording_fails", [False, True])
async def test_tool_exception_is_preserved_and_remaining_calls_are_skipped(
    monkeypatch, recording_fails
):
    events = []
    original = RuntimeError("original tool failure")

    async def record(_ctx, _observation_id, kind, payload):
        if recording_fails and kind == "tool_end":
            raise OSError("evidence write failed")
        events.append((kind, payload))

    async def fail(*_args):
        raise original

    monkeypatch.setattr(tool_node, "record_event", record)
    node = tool_node.ExecutorToolNode(
        [action_tool([])], messages_key=KEY, evidence_context=cast(MobileUseContext, object())
    )
    monkeypatch.setattr(node, "_arun_one", fail)
    monkeypatch.setattr(node, "_build_tool_runtime", lambda *_args: None)
    with pytest.raises(RuntimeError) as caught:
        await node._afunc(dict(tool_input([call(1, 1), call(2, 2)])), {}, Runtime())
    assert caught.value is original
    if not recording_fails:
        assert [(kind, p.get("status")) for kind, p in events] == [
            ("tool_start", None),
            ("tool_end", "error"),
            ("tool_end", "skipped"),
        ]
        assert events[1][1]["error_type"] == "RuntimeError"
        assert "original tool failure" not in str(events)


@pytest.mark.asyncio
async def test_cancelled_tool_records_terminal_event_before_propagating(monkeypatch):
    events = []
    started = asyncio.Event()

    async def record(_ctx, _observation_id, kind, payload):
        events.append((kind, payload))

    async def wait(*_args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(tool_node, "record_event", record)
    node = tool_node.ExecutorToolNode(
        [action_tool([])], messages_key=KEY, evidence_context=cast(MobileUseContext, object())
    )
    monkeypatch.setattr(node, "_arun_one", wait)
    monkeypatch.setattr(node, "_build_tool_runtime", lambda *_args: None)
    pending = asyncio.create_task(
        node._afunc(dict(tool_input([call(1, 1), call(2, 2)])), {}, Runtime())
    )
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert [(kind, p.get("status")) for kind, p in events] == [
        ("tool_start", None),
        ("tool_end", "cancelled"),
        ("tool_end", "skipped"),
    ]


class FakeModel:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.seen = []

    def bind_tools(self, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("use_fallback,cancelled", [(False, False), (True, False), (False, True)])
async def test_executor_records_actual_messages_model_attempt_and_decision(
    monkeypatch, use_fallback, cancelled
):
    response = AIMessage(content="", tool_calls=[call(1, 1)])
    failure = asyncio.CancelledError() if cancelled else RuntimeError("model unavailable")
    primary = FakeModel(response=response, error=failure if cancelled or use_fallback else None)
    fallback = FakeModel(response=response)
    fallback_config = SimpleNamespace(
        provider="deepseek",
        model="fallback-model",
        model_dump=lambda **_: {"provider": "deepseek", "model": "fallback-model"},
    )
    ctx = SimpleNamespace(
        device=SimpleNamespace(mobile_platform=SimpleNamespace(value="ios")),
        video_recording_enabled=False,
        llm_config=SimpleNamespace(
            executor=SimpleNamespace(
                provider="deepseek",
                model="primary-model",
                fallback=fallback_config,
            )
        ),
    )
    previous = HumanMessage(content="Previous explicit tool result")
    state = SimpleNamespace(
        structured_decisions="Swipe from x=320 to x=40",
        cortex_last_thought="Move to the next home page",
        agents_thoughts=[],
        executor_messages=[previous],
        observation_id="observation-1",
        asanitize_update=AsyncMock(side_effect=lambda **kwargs: kwargs["update"]),
    )
    inputs, events = [], []

    async def record_input(context, observation_id, agent, messages, model):
        assert context is ctx and observation_id == "observation-1" and agent == "executor"
        inputs.append((messages, model))

    async def record_event(context, observation_id, kind, payload):
        assert context is ctx and observation_id == "observation-1"
        events.append((kind, payload))

    async def attempt(main_call, fallback_call):
        try:
            return await main_call()
        except RuntimeError:
            return await fallback_call()

    monkeypatch.setattr(
        executor, "get_llm", lambda **kwargs: fallback if kwargs.get("use_fallback") else primary
    )
    monkeypatch.setattr(executor, "get_tools_from_wrappers", lambda *_args: [])
    monkeypatch.setattr(executor, "record_model_input", record_input)
    monkeypatch.setattr(executor, "record_event", record_event)
    monkeypatch.setattr(executor, "with_fallback", attempt)

    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await executor.ExecutorNode(cast(MobileUseContext, ctx))(state)
        assert len(inputs) == 1
        assert inputs[0][0] is primary.seen[0]
        assert fallback.seen == []
        assert events == [
            (
                "executor_error",
                {"error_type": "CancelledError", "attempt": "primary", "status": "cancelled"},
            )
        ]
        return

    result = await executor.ExecutorNode(cast(MobileUseContext, ctx))(state)
    assert inputs[0][0] is primary.seen[0]
    assert inputs[0][0][-1] is previous
    assert inputs[0][0][-2].content == state.structured_decisions
    assert inputs[0][1]["model"] == "primary-model"
    assert len(inputs) == (2 if use_fallback else 1)
    if use_fallback:
        assert inputs[1][0] is fallback.seen[0]
        assert inputs[1][1]["model"] == "fallback-model"
        assert inputs[1][1]["attempt"] == "fallback"
    decision = [
        (
            "executor_decision",
            {
                "tool_calls": [
                    {
                        "call_id": "c1",
                        "name": "action",
                        "arguments": {"value": 1},
                    }
                ]
            },
        )
    ]
    if use_fallback:
        assert events[0] == (
            "executor_error",
            {"error_type": "RuntimeError", "attempt": "primary", "status": "error"},
        )
        assert events[1:] == decision
    else:
        assert events == decision
    assert result[executor.EXECUTOR_MESSAGES_KEY] == [response]
