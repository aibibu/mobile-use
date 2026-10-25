import asyncio
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai.chat_models import ChatVertexAI

from minitap.mobile_use.agents.executor.tool_node import safe_tool_call
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    get_tools_from_wrappers,
)
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.evidence import record_event, record_model_input
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutorNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Executor Agent..."),
        on_success=lambda _: logger.success("Executor Agent"),
        on_failure=lambda _: logger.error("Executor Agent"),
    )
    async def __call__(self, state: State):
        structured_decisions = state.structured_decisions
        if not structured_decisions:
            logger.warning("No structured decisions found.")
            return await state.asanitize_update(
                ctx=self.ctx,
                update={
                    "agents_thoughts": [
                        "No structured decisions found, I cannot execute anything."
                    ],
                },
                agent="executor",
            )

        system_message = Template(
            Path(__file__).parent.joinpath("executor.md").read_text(encoding="utf-8")
        ).render(platform=self.ctx.device.mobile_platform.value)
        cortex_last_thought = (
            state.cortex_last_thought if state.cortex_last_thought else state.agents_thoughts[-1]
        )
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=cortex_last_thought),
            HumanMessage(content=structured_decisions),
            *state.executor_messages,
        ]

        llm = get_llm(ctx=self.ctx, name="executor")
        llm_fallback = get_llm(ctx=self.ctx, name="executor", use_fallback=True)

        executor_wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            executor_wrappers.extend(VIDEO_RECORDING_WRAPPERS)

        llm_bind_tools_kwargs: dict = {
            "tools": get_tools_from_wrappers(self.ctx, executor_wrappers),
        }

        # ChatGoogleGenerativeAI does not support the "parallel_tool_calls" keyword
        if not isinstance(llm, ChatGoogleGenerativeAI | ChatVertexAI):
            llm_bind_tools_kwargs["parallel_tool_calls"] = True

        llm = llm.bind_tools(**llm_bind_tools_kwargs)
        llm_fallback = llm_fallback.bind_tools(**llm_bind_tools_kwargs)

        async def invoke(model, *, fallback: bool = False):
            configured = self.ctx.llm_config.executor
            selected = configured.fallback if fallback else configured
            await record_model_input(
                self.ctx,
                state.observation_id,
                "executor",
                messages,
                {
                    "provider": selected.provider,
                    "model": selected.model,
                    "fallback": configured.fallback.model_dump(mode="json"),
                    "attempt": "fallback" if fallback else "primary",
                },
            )
            try:
                return await invoke_llm_with_timeout_message(model.ainvoke(messages))
            except BaseException as error:
                try:
                    await record_event(
                        self.ctx,
                        state.observation_id,
                        "executor_error",
                        {
                            "error_type": type(error).__name__,
                            "attempt": "fallback" if fallback else "primary",
                            "status": "cancelled"
                            if isinstance(error, asyncio.CancelledError)
                            else "error",
                        },
                    )
                except BaseException:
                    logger.warning("Could not record interrupted executor model call")
                raise

        response = await with_fallback(
            main_call=lambda: invoke(llm),
            fallback_call=lambda: invoke(llm_fallback, fallback=True),
        )
        await record_event(
            self.ctx,
            state.observation_id,
            "executor_decision",
            {"tool_calls": [safe_tool_call(call) for call in response.tool_calls]},
        )
        return await state.asanitize_update(
            ctx=self.ctx,
            update={
                "cortex_last_thought": cortex_last_thought,
                EXECUTOR_MESSAGES_KEY: [response],
            },
            agent="executor",
        )
