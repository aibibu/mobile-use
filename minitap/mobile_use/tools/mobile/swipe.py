from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import Field

from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.types import (
    CoordinatesSelectorRequest,
    PercentagesSelectorRequest,
    SwipeRequest,
    SwipeStartEndCoordinatesRequest,
    SwipeStartEndPercentagesRequest,
)
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.tool_wrapper import CompositeToolWrapper


def get_swipe_tool(ctx: MobileUseContext) -> BaseTool:
    @tool
    async def swipe(
        agent_thought: str,
        swipe_request: SwipeRequest,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
    ) -> Command:
        """Swipe from start to end position on screen.

        Supports percentage-based or coordinate-based positioning.
        """
        controller = UnifiedMobileController(ctx)
        output = await controller.swipe_request(swipe_request)
        has_failed = output is not None

        agent_outcome = (
            swipe_wrapper.on_success_fn() if not has_failed else swipe_wrapper.on_failure_fn()
        )

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            additional_kwargs={"error": output} if has_failed else {},
            status="error" if has_failed else "success",
        )

        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, agent_outcome],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                },
                agent="executor",
            ),
        )

    return swipe


def get_composite_swipe_tools(ctx: MobileUseContext) -> list[BaseTool]:
    """
    Returns composite swipe tools with flattened arguments.
    Each tool handles a specific swipe mode to avoid complex Union type issues.
    """

    async def _execute_swipe(
        tool_call_id: str,
        state: State,
        agent_thought: str,
        swipe_request: SwipeRequest,
    ) -> Command:
        """Shared swipe execution logic."""
        controller = UnifiedMobileController(ctx)
        output = await controller.swipe_request(swipe_request)
        has_failed = output is not None

        agent_outcome = (
            swipe_wrapper.on_success_fn() if not has_failed else swipe_wrapper.on_failure_fn()
        )

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            additional_kwargs={"error": output} if has_failed else {},
            status="error" if has_failed else "success",
        )

        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, agent_outcome],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                },
                agent="executor",
            ),
        )

    @tool
    async def swipe_coordinates(
        agent_thought: str,
        start_x: int = Field(description="Start X coordinate in pixels"),
        start_y: int = Field(description="Start Y coordinate in pixels"),
        end_x: int = Field(description="End X coordinate in pixels"),
        end_y: int = Field(description="End Y coordinate in pixels"),
        duration: int = Field(description="Duration in ms", ge=1, le=10000, default=400),
        tool_call_id: Annotated[str, InjectedToolCallId] = None,  # type: ignore
        state: Annotated[State, InjectedState] = None,  # type: ignore
    ) -> Command:
        """Swipe using pixel coordinates from start position to end position."""
        swipe_request = SwipeRequest(
            swipe_mode=SwipeStartEndCoordinatesRequest(
                start=CoordinatesSelectorRequest(x=start_x, y=start_y),
                end=CoordinatesSelectorRequest(x=end_x, y=end_y),
            ),
            duration=duration,
        )
        return await _execute_swipe(tool_call_id, state, agent_thought, swipe_request)

    @tool
    async def swipe_percentages(
        agent_thought: str,
        start_x_percent: int = Field(
            description=(
                "Finger start X, integer 0-100 (0=left edge, 100=right edge). "
                "Use 80 for 80%, not 0.8. With Y unchanged, start X 80 to end X 20 "
                "swipes LEFT; start X 20 to end X 80 swipes RIGHT."
            ),
            ge=0,
            le=100,
        ),
        start_y_percent: int = Field(
            description=(
                "Finger start Y, integer 0-100 (0=top edge, 100=bottom edge). "
                "Use 80 for 80%, not 0.8. With X unchanged, start Y 80 to end Y 20 "
                "swipes UP; start Y 20 to end Y 80 swipes DOWN."
            ),
            ge=0,
            le=100,
        ),
        end_x_percent: int = Field(
            description=(
                "Finger end X, integer 0-100 (0=left edge, 100=right edge). "
                "Less than start_x_percent moves the finger LEFT; "
                "greater than start_x_percent moves the finger RIGHT."
            ),
            ge=0,
            le=100,
        ),
        end_y_percent: int = Field(
            description=(
                "Finger end Y, integer 0-100 (0=top edge, 100=bottom edge). "
                "Less than start_y_percent moves the finger UP; "
                "greater than start_y_percent moves the finger DOWN."
            ),
            ge=0,
            le=100,
        ),
        duration: int = Field(description="Duration in ms", ge=1, le=10000, default=400),
        tool_call_id: Annotated[str, InjectedToolCallId] = None,  # type: ignore
        state: Annotated[State, InjectedState] = None,  # type: ignore
    ) -> Command:
        """Swipe using integer screen percentages from the finger's start to end position.

        Directions describe finger movement, not the destination page or content.
        The origin is the top-left corner; X increases rightward and Y downward.
        """
        swipe_request = SwipeRequest(
            swipe_mode=SwipeStartEndPercentagesRequest(
                start=PercentagesSelectorRequest(
                    x_percent=start_x_percent, y_percent=start_y_percent
                ),
                end=PercentagesSelectorRequest(x_percent=end_x_percent, y_percent=end_y_percent),
            ),
            duration=duration,
        )
        return await _execute_swipe(tool_call_id, state, agent_thought, swipe_request)

    return [swipe_coordinates, swipe_percentages]


swipe_wrapper = CompositeToolWrapper(
    tool_fn_getter=get_swipe_tool,
    composite_tools_fn_getter=get_composite_swipe_tools,
    on_success_fn=lambda: "Swipe is successful.",
    on_failure_fn=lambda: "Failed to swipe.",
)
