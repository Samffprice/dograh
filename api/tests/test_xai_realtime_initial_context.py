from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

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
async def test_function_call_executes_immediately_when_bot_is_not_speaking():
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
    assert service._deferred_function_calls == []


@pytest.mark.asyncio
async def test_function_call_is_deferred_until_bot_stops_speaking():
    service = _make_service()
    service._context = LLMContext()
    service.run_function_calls = AsyncMock()
    service._bot_is_speaking = True
    service._pending_function_calls["call-1"] = SimpleNamespace(name="customer_support")

    await service._handle_evt_function_call_arguments_done(
        SimpleNamespace(
            call_id="call-1", name="customer_support", arguments='{"department":"sales"}'
        )
    )

    service.run_function_calls.assert_not_awaited()
    assert len(service._deferred_function_calls) == 1

    await service._run_pending_function_calls()

    service.run_function_calls.assert_awaited_once()
    assert service._deferred_function_calls == []


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
