"""Reproducer for the Gemini Live tool-call session-swap race.

The bug: when `_update_settings(system_instruction=...)` is invoked between a
`tool_call` arriving and its `tool_response` being delivered (the ordering that
dograh's workflow `transition_func` produces — see
api/services/workflow/pipecat_engine.py:271-299), the synchronous reconnect
path in `GeminiLiveLLMService` closes the originating session before the
tool-result is sent. The subsequent `send_tool_response` then lands on a new
session that never emitted the matching `tool_call`. Under Gemini 3.x's
sync-only function-call contract, the model is stranded.

This test drives the production dispatch sequence as faithfully as possible
without standing up a full `Pipeline`:

    REAL  (1) `_handle_msg_tool_call(msg)` while `_bot_is_responding=True`
              → call is deferred onto `_pending_function_calls`.
    REAL  (2) `_set_bot_is_responding(False)`
              → triggers `_run_pending_function_calls()`
              → `run_function_calls(...)`
              → registered handler runs as an asyncio task.
    REAL  (3) Handler mimics dograh's `transition_func`:
              calls `_update_settings(LLMSettings(system_instruction=...))`,
              then `params.result_callback(result)`.
    MANUAL (4) `_tool_result(call_id, ...)` — stands in for the aggregator
              path that would normally translate `FunctionCallResultFrame`
              into a `_tool_result` call. Without a `Pipeline` there are no
              downstream processors, so the broadcast in step 3 is a no-op
              and the test has to invoke `_tool_result` directly.

Only step 4 is hand-rolled; steps 1–3 run the same code paths production uses.
The harness substitutes a `MockGeminiSession` for the real websocket so the
session-identity assertion is mechanical: each `_connect()` produces a fresh
labelled instance, and the assertion checks that `send_tool_response` reached
the instance that emitted the matching `tool_call`.

On the unfixed service the test fails with a diagnostic naming the swap.
With the proposed `_pending_tool_responses` gating in `_update_settings`,
the reconnect is deferred and the test passes.
"""

import asyncio
import os
from types import SimpleNamespace
from typing import List

import pytest
from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.settings import LLMSettings
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class MockGeminiSession:
    """Stand-in for google-genai's AsyncSession that records every send and
    is uniquely labelled so tests can prove which session received a call.
    """

    _next_id = 0

    def __init__(self):
        type(self)._next_id += 1
        self.label = f"session-{type(self)._next_id}"
        self.sent_tool_responses: list = []
        self.sent_realtime_inputs: list = []
        self.sent_client_contents: list = []
        self.closed = False

    async def send_tool_response(self, function_responses):
        self.sent_tool_responses.append(function_responses)

    async def send_realtime_input(self, **kwargs):
        self.sent_realtime_inputs.append(kwargs)

    async def send_client_content(self, **kwargs):
        self.sent_client_contents.append(kwargs)

    async def close(self):
        self.closed = True

    def receive(self):
        # The test never drives the receive loop — _connect is overridden to
        # skip the connection_task. Return an empty async iterator so any
        # accidental access doesn't hang.
        async def _empty():
            if False:
                yield None

        return _empty()


class HarnessedGeminiLiveLLMService(GeminiLiveLLMService):
    """Replaces the real websocket with a `MockGeminiSession`.

    Skips `_connection_task_handler` / receive loop so the test owns
    message-arrival timing. Every other code path — `_update_settings`,
    `_tool_result`, `_disconnect`, `_reconnect`, `_handle_msg_tool_call`,
    `_set_bot_is_responding`, `_run_pending_function_calls`,
    `run_function_calls` — runs unmodified.
    """

    def __init__(self, *, sessions_log: List[MockGeminiSession], **kwargs):
        super().__init__(**kwargs)
        self._sessions_log = sessions_log

    async def _connect(self, session_resumption_handle: str | None = None):
        if self._session:
            return
        session = MockGeminiSession()
        self._sessions_log.append(session)
        await self._handle_session_ready(session)


def _make_tool_call_message(call_id: str, name: str, args: dict | None = None):
    """Duck-type a `LiveServerMessage` carrying a single tool_call.

    `_handle_msg_tool_call` only reads `message.tool_call.function_calls` and
    each call's `.id`, `.name`, `.args` — `SimpleNamespace` is sufficient.
    """
    return SimpleNamespace(
        tool_call=SimpleNamespace(
            function_calls=[SimpleNamespace(id=call_id, name=name, args=args or {})]
        )
    )


class TestGeminiLiveToolCallRace:
    @pytest.mark.asyncio
    async def test_send_tool_response_reaches_originating_session(self):
        """Wire-protocol invariant: `tool_response(id=X)` reaches the session that emitted `tool_call(id=X)`.

        Drives the production dispatch path: a tool_call arrives while the
        bot is mid-utterance and is deferred; when the bot's utterance ends,
        `_set_bot_is_responding(False)` triggers the function-call runner,
        which invokes our registered handler. The handler mirrors dograh's
        `transition_func` (see `api/services/workflow/pipecat_engine.py:271-299`)
        by calling `_update_settings(system_instruction=...)` and then
        `params.result_callback(result)`. The aggregator path that would
        normally translate the broadcast result into `_tool_result(...)` is
        invoked manually at the end of the test.

        On the unfixed service, `_update_settings` reconnects synchronously
        (`gemini_live/llm.py:606`), swapping `self._session` before the
        manual `_tool_result` runs, so `send_tool_response` lands on the
        wrong instance.
        """
        sessions: List[MockGeminiSession] = []
        service = HarnessedGeminiLiveLLMService(
            sessions_log=sessions,
            api_key=os.getenv("GOOGLE_API_KEY", "stub-value-mock-connect-bypasses-auth"),
            system_instruction="initial node prompt",
        )
        # Use a Gemini 3.x model so `_is_gemini_3` matches production (the bug
        # is reachable on 2.5 too, but only 3.x's sync-only contract makes it
        # a permanent hang rather than a recoverable lossy turn).
        service._settings.model = "models/gemini-3.1-flash-live-preview"

        # `_handle_msg_tool_call` requires a context object; an empty one is fine.
        service._context = LLMContext()

        # `run_function_calls` schedules handlers via `self.task_manager`,
        # which is populated by `FrameProcessor.setup`. Production calls
        # `setup` when the service receives a `StartFrame`; we replicate the
        # minimal initialization here so the function-call dispatch path runs.
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        await service.setup(
            FrameProcessorSetup(clock=SystemClock(), task_manager=task_manager)
        )

        # Register a handler that mirrors dograh's transition_func ordering:
        # `set_node` (ends up at `_update_settings`) before `result_callback`.
        handler_done = asyncio.Event()
        handler_trace: List[str] = []

        async def transition_func(params: FunctionCallParams):
            handler_trace.append("entered")
            await service._update_settings(
                LLMSettings(system_instruction="new node prompt")
            )
            handler_trace.append("update_settings_returned")
            # Without a Pipeline this just broadcasts (no downstream consumers).
            # The real aggregator path is simulated by the manual `_tool_result`
            # call below.
            await params.result_callback({"status": "ok"})
            handler_trace.append("result_callback_returned")
            handler_done.set()

        service.register_function("advance_to_agent", transition_func)

        # --- Open Session A ---
        await service._connect()
        assert len(sessions) == 1
        session_a = sessions[0]
        assert service._session is session_a

        # --- Server emits tool_call(X) on Session A while bot is mid-utterance ---
        # `_bot_is_responding=True` causes `_handle_msg_tool_call` to take the
        # deferred path, parking the call in `_pending_function_calls`.
        service._bot_is_responding = True
        await service._handle_msg_tool_call(
            _make_tool_call_message(call_id="X-call-id", name="advance_to_agent")
        )

        # --- Bot's utterance ends; production dispatch path runs ---
        # `_set_bot_is_responding(False)` calls `_run_pending_function_calls()`
        # which calls `run_function_calls(...)` which schedules our handler as
        # an asyncio task. The handler runs `_update_settings` (the race
        # trigger on unfixed code) then `result_callback`.
        await service._set_bot_is_responding(False)
        await asyncio.wait_for(handler_done.wait(), timeout=2.0)

        session_at_result_time = service._session

        # --- Aggregator stand-in: deliver tool_result for the call ---
        # In production, the `FunctionCallResultFrame` broadcast inside
        # `result_callback` propagates through `LLMContextAggregator` to a
        # context frame, which `_handle_context` consumes to call
        # `_tool_result`. Without a Pipeline we invoke it directly.
        await service._tool_result(
            tool_call_id="X-call-id",
            tool_name="advance_to_agent",
            tool_result_message={"status": "ok"},
        )

        # === Wire-protocol invariant ===
        assert session_a.sent_tool_responses, (
            "\n\n*** WIRE-PROTOCOL INVARIANT VIOLATED ***\n"
            "send_tool_response(id=X) did not reach Session A — the session "
            "that emitted tool_call(id=X).\n"
            "\nDispatch trace (production code paths exercised):\n"
            f"  Handler progress: {handler_trace}\n"
            f"  Sessions opened during the test: {[s.label for s in sessions]}\n"
            f"  Session A label:                 {session_a.label}\n"
            f"  Session A closed:                {session_a.closed}\n"
            f"  Per-session sent_tool_responses: "
            f"{[(s.label, len(s.sent_tool_responses)) for s in sessions]}\n"
            f"  service._session at _tool_result time: "
            f"{getattr(session_at_result_time, 'label', None)}\n"
            "\nRoot cause: `_update_settings` (invoked from the registered "
            "handler — the same call dograh's `set_node` makes at "
            "`api/services/workflow/pipecat_engine.py:271`) hit the "
            "synchronous reconnect branch at `gemini_live/llm.py:606`, "
            "closing Session A before `_tool_result` ran. Under Gemini 3.x's "
            "sync-only function-call contract, the model on the new session "
            "has no record of tool_call X and hangs indefinitely.\n"
        )
