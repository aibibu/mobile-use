"""DeepSeek configuration and payload tests; no API requests or device actions."""

import importlib
import sys

import pytest
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBinding, RunnableSequence
from pydantic import BaseModel, SecretStr

# A legacy outputter test installs a process-wide module mock during collection.
vertex_module = sys.modules.get("langchain_google_vertexai")
if vertex_module is not None and not hasattr(vertex_module, "__path__"):
    sys.modules.pop("langchain_google_vertexai", None)
    importlib.import_module("langchain_google_vertexai")

from minitap.mobile_use.config import (  # noqa: E402
    LLM,
    LLMConfig,
    LLMConfigUtils,
    LLMWithFallback,
    settings,
)
from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext  # noqa: E402
from minitap.mobile_use.services.llm import (  # noqa: E402
    DeepSeekChatOpenAI,
    get_deepseek_llm,
    get_llm,
    get_openai_llm,
)


class StructuredAnswer(BaseModel):
    answer: str


@pytest.fixture
def deepseek_key(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", SecretStr("fixture-deepseek-key"))


def test_deepseek_constructor_uses_official_endpoint_and_disables_thinking(deepseek_key):
    client = get_deepseek_llm(temperature=0.7)
    assert isinstance(client, DeepSeekChatOpenAI)
    assert client.model_name == "deepseek-flash"
    assert client.openai_api_base == "https://api.deepseek.com"
    assert client.temperature == 0.7
    assert client.extra_body == {"thinking": {"type": "disabled"}}
    assert client.openai_api_key == SecretStr("fixture-deepseek-key")


def test_deepseek_structured_output_uses_non_strict_function_calling(deepseek_key):
    structured = get_deepseek_llm().with_structured_output(StructuredAnswer)
    assert isinstance(structured, RunnableSequence)
    assert isinstance(structured.first, RunnableBinding)
    payload = structured.first.kwargs
    assert "response_format" not in payload
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "StructuredAnswer"}}
    assert payload["tools"][0]["function"]["strict"] is False


@pytest.mark.parametrize("options", [{"method": "json_schema"}, {"strict": True}])
def test_deepseek_rejects_unsupported_structured_output_modes(deepseek_key, options):
    with pytest.raises(ValueError):
        get_deepseek_llm().with_structured_output(StructuredAnswer, **options)


def test_deepseek_allows_explicit_json_object_mode(deepseek_key):
    structured = get_deepseek_llm().with_structured_output(StructuredAnswer, method="json_mode")
    assert isinstance(structured, RunnableSequence)
    assert isinstance(structured.first, RunnableBinding)
    assert structured.first.kwargs["response_format"] == {"type": "json_object"}


def test_deepseek_key_validation_does_not_accept_an_openai_key(monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", None)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("fixture-openai-key"))
    with pytest.raises(Exception, match="DEEPSEEK_API_KEY"):
        LLM(provider="deepseek", model="deepseek-flash").validate_provider("Cortex")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        get_deepseek_llm()


def test_deepseek_dispatch_applies_to_primary_and_fallback(deepseek_key):
    node = LLMWithFallback(
        provider="deepseek",
        model="deepseek-flash",
        fallback=LLM(provider="deepseek", model="deepseek-flash"),
    )
    context = MobileUseContext(
        trace_id="fixture-trace",
        device=DeviceContext(
            host_platform="LINUX",
            mobile_platform=DevicePlatform.IOS,
            device_id="fixture-device",
            device_width=375,
            device_height=667,
        ),
        llm_config=LLMConfig(
            planner=node,
            orchestrator=node,
            contextor=node,
            cortex=node,
            executor=node,
            utils=LLMConfigUtils(outputter=node, hopper=node),
        ),
    )
    for use_fallback in (False, True):
        client = get_llm(context, "cortex", use_fallback=use_fallback)
        assert isinstance(client, DeepSeekChatOpenAI)
        assert client.extra_body == {"thinking": {"type": "disabled"}}


def test_openai_provider_keeps_existing_structured_output_behavior(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("fixture-openai-key"))
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", None)
    client = get_openai_llm("gpt-5")
    assert type(client) is ChatOpenAI
    assert not client.extra_body
    structured = client.with_structured_output(StructuredAnswer)
    assert isinstance(structured, RunnableSequence)
    assert isinstance(structured.first, RunnableBinding)
    assert "response_format" in structured.first.kwargs
