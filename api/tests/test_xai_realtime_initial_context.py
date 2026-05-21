from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import LLMSettings
from pipecat.services.xai.realtime.events import SessionUpdateEvent

from api.services.pipecat.realtime.xai_realtime import DograhXAIRealtimeLLMService

# Non-secret placeholder passed to the constructor in unit tests. Named so it
# doesn't trip secret scanners — no real credential is involved.
FAKE_CRED = "unit-test"


def _make_service() -> DograhXAIRealtimeLLMService:
    service = DograhXAIRealtimeLLMService(api_key=FAKE_CRED)
    service._create_response = AsyncMock()
    service._process_completed_function_calls = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_initial_context_triggers_response_when_context_was_prepopulated():
    service = _make_service()
    context = LLMContext()
    service._context = context

    await service._handle_context(context)

    assert service._handled_initial_context is True
    assert service._context is context
    service._create_response.assert_awaited_once()
    service._process_completed_function_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_updated_context_uses_tool_result_path_after_initial_context():
    service = _make_service()
    context = LLMContext()
    service._handled_initial_context = True

    await service._handle_context(context)

    assert service._context is context
    service._create_response.assert_not_awaited()
    service._process_completed_function_calls.assert_awaited_once_with(
        send_new_results=True
    )


@pytest.mark.asyncio
async def test_tts_greeting_uses_initial_context_handler():
    service = _make_service()
    service._context = LLMContext()
    service._handle_context = AsyncMock()

    await service.process_frame(
        TTSSpeakFrame("hello", append_to_context=True),
        FrameDirection.DOWNSTREAM,
    )

    service._handle_context.assert_awaited_once_with(service._context)
    service._create_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_function_call_runs_immediately_without_deferral():
    # Tool calls must run as soon as the arguments arrive — we deliberately do
    # NOT defer until the bot stops speaking, because that stalls on Grok (the
    # model says "let me check" and the call never fires). Guard against the
    # defer machinery being re-introduced.
    service = _make_service()
    service._context = LLMContext()
    service.run_function_calls = AsyncMock()
    service._pending_function_calls["call-1"] = SimpleNamespace(name="customer_support")

    await service._handle_evt_function_call_arguments_done(
        SimpleNamespace(
            call_id="call-1", name="customer_support", arguments='{"department":"sales"}'
        )
    )

    service.run_function_calls.assert_awaited_once()
    assert not hasattr(service, "_deferred_function_calls")
    assert not hasattr(service, "_bot_is_speaking")


def _tool(name: str) -> FunctionSchema:
    return FunctionSchema(
        name=name, description=f"{name} tool", properties={}, required=[]
    )


@pytest.mark.asyncio
async def test_node_transition_sends_session_update_with_new_prompt_and_tools():
    """A node transition (prompt change + tool replacement) is delivered as a
    single session.update carrying the fresh instructions and the new node's
    tools — no reconnect, mirroring how the engine drives OpenAI Realtime."""
    service = _make_service()
    # Restore the real _process_completed_function_calls; this test exercises
    # the session-update path, which _make_service does not stub.
    del service._process_completed_function_calls
    # Simulate an open, ready session.
    sentinel_ws = object()
    service._websocket = sentinel_ws
    service._api_session_ready = True
    service.send_client_event = AsyncMock()

    context = LLMContext()
    service._context = context

    # --- Node A: prompt + collect_name tool ---
    context.set_tools(ToolsSchema(standard_tools=[_tool("collect_name")]))
    await service._update_settings(LLMSettings(system_instruction="You are at node A"))

    evt_a = service.send_client_event.await_args_list[-1].args[0]
    assert isinstance(evt_a, SessionUpdateEvent)
    instr_a = evt_a.session.instructions
    tools_a = {t["name"] for t in evt_a.session.tools if t.get("type") == "function"}
    assert instr_a == "You are at node A"
    assert tools_a == {"collect_name"}

    # --- Node B: new prompt + book_appointment tool (replaces collect_name) ---
    context.set_tools(ToolsSchema(standard_tools=[_tool("book_appointment")]))
    await service._update_settings(LLMSettings(system_instruction="You are at node B"))

    evt_b = service.send_client_event.await_args_list[-1].args[0]
    instr_b = evt_b.session.instructions
    tools_b = {t["name"] for t in evt_b.session.tools if t.get("type") == "function"}
    assert instr_b == "You are at node B"
    assert tools_b == {"book_appointment"}  # collect_name is gone (replaced)

    # The WebSocket was never torn down — the update happened in-session.
    assert service._websocket is sentinel_ws


@pytest.mark.asyncio
async def test_deferred_connect_holds_until_system_instruction_set():
    service = _make_service()

    with patch(
        "api.services.pipecat.realtime.xai_realtime.GrokRealtimeLLMService._connect",
        new_callable=AsyncMock,
    ) as super_connect:
        # No system_instruction yet -> connect is deferred (parent not called).
        service._settings.system_instruction = None
        await service._connect()
        super_connect.assert_not_awaited()

        # Once set, _connect delegates to the parent to open the WebSocket.
        service._settings.system_instruction = "be helpful"
        await service._connect()
        super_connect.assert_awaited_once()
