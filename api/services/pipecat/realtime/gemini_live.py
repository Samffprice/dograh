"""Dograh subclass of pipecat's Gemini Live LLM service.

Layers Dograh engine integration quirks onto upstream-pristine
:class:`GeminiLiveLLMService`:

- **Deferred connect.** Connection is held back until ``system_instruction``
  is set via :meth:`_update_settings`, so pre-call-fetch template variables
  land before the live session opens.
- **Reconnect on node transitions.** Gemini Live cannot update
  ``system_instruction`` mid-session, so a setting change triggers a
  reconnect (deferred until the bot turn ends if currently responding).
- **Function-call deferral.** Tool calls emitted mid-turn are queued and run
  when the bot stops speaking, to avoid racing the turn's audio.
- **User-mute audio gating.** ``UserMuteStarted/StoppedFrame`` from the
  user aggregator gates whether incoming audio is forwarded to Gemini.
- **TTSSpeakFrame as greeting trigger.** The engine queues a TTSSpeakFrame
  to kick off the first response after node setup; the service intercepts
  it and runs the initial-context path.
- **Finalize-pending on transcriptions.** Marks the transcription emitted
  immediately after VAD-stop as finalized, distinguishing it from
  mid-turn partials.
"""

from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_gemini_live


class DograhGeminiLiveLLMService(GeminiLiveLLMService):
    """Gemini Live with Dograh engine integration quirks. See module docstring."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # User-mute state, driven by broadcast UserMute{Started,Stopped}Frames.
        # Audio is not forwarded to Gemini while muted.
        self._user_is_muted: bool = False
        # Guards initial-response triggering against double-firing across the
        # initial TTSSpeakFrame and any LLMContextFrame that may arrive.
        self._handled_initial_context: bool = False
        # When a system_instruction change arrives mid-bot-turn, the reconnect
        # is queued and drained when the turn ends.
        self._reconnect_pending: bool = False
        # Function calls emitted by Gemini mid-bot-turn are deferred here and
        # invoked when the turn ends, so they don't race the turn's audio.
        self._pending_function_calls: list[FunctionCallFromLLM] = []
        # tool_call_ids the server has emitted but for which we have not yet
        # delivered a tool_response on the CURRENT session. _handle_changed_settings
        # gates reconnects on this being empty — reconnecting before the response
        # lands strands the call on Session A (the resumed Session B doesn't
        # recognize the call_id and the orphan destabilizes the model).
        self._pending_tool_responses: set[str] = set()
        # Tracks whether the next transcription to arrive should be marked as
        # the finalized transcription for the current user turn.
        self._finalize_pending: bool = False

    # ------------------------------------------------------------------
    # Hooks from upstream GeminiLiveLLMService
    # ------------------------------------------------------------------

    def _should_connect_on_start(self) -> bool:
        # Hold the connection until the engine sets a system_instruction. This
        # lets pre-call fetch populate template variables first.
        return bool(self._settings.system_instruction)

    async def _handle_changed_settings(self, changed: dict[str, Any]) -> set[str]:
        if "system_instruction" not in changed:
            return set()
        if not self._session:
            # First-time setting after deferred-connect.
            await self._connect()
        elif self._bot_is_responding or self._pending_tool_responses:
            # Defer reconnect: mid-utterance would cut the bot off, and
            # mid-tool-call would strand the tool_response on a session
            # the resumed model doesn't recognize (Gemini 3.x ignores
            # tool_responses for call_ids issued on a different session,
            # destabilizing the model's state).
            self._reconnect_pending = True
        else:
            await self._reconnect()
        return {"system_instruction"}

    async def _run_or_defer_function_calls(
        self, function_calls_llm: list[FunctionCallFromLLM]
    ):
        # Record the in-flight obligation immediately, regardless of whether
        # dispatch happens now or after the bot turn ends. _handle_changed_settings
        # gates reconnects on this set being empty.
        for fc in function_calls_llm:
            self._pending_tool_responses.add(fc.tool_call_id)
        if self._bot_is_responding:
            # Latest batch wins; Gemini emits tool calls as one batch per
            # tool_call message, so this overwrite is intentional.
            self._pending_function_calls = function_calls_llm
            logger.debug(
                f"{self}: deferring {len(function_calls_llm)} function call(s) "
                "until bot turn ends"
            )
            return
        await super()._run_or_defer_function_calls(function_calls_llm)

    # ------------------------------------------------------------------
    # State-transition side effects
    # ------------------------------------------------------------------

    async def _set_bot_is_responding(self, responding: bool):
        was_responding = self._bot_is_responding
        await super()._set_bot_is_responding(responding)
        if was_responding and not responding:
            await self._run_pending_function_calls()
            # Only fire the deferred reconnect when no tool_response is
            # still owed on this session. Otherwise the reconnect would
            # close Session A before the response is delivered, stranding
            # the call_id on the resumed session that doesn't recognize it.
            # The fired-from-_tool_result alternative is unsafe: it would
            # close the session before the model's post-tool turn arrives.
            if self._reconnect_pending and not self._pending_tool_responses:
                self._reconnect_pending = False
                await self._reconnect()

    async def _run_pending_function_calls(self):
        """Run any function calls deferred during the bot's last turn."""
        if not self._pending_function_calls:
            return
        fcs = self._pending_function_calls
        self._pending_function_calls = []
        logger.debug(
            f"{self}: executing {len(fcs)} deferred function call(s) "
            "after bot turn ended"
        )
        await self.run_function_calls(fcs)

    async def _tool_result(self, tool_call_id, tool_name, tool_result_message):
        # On successful send we discard from _pending_tool_responses so the
        # next _set_bot_is_responding(False) can fire the deferred reconnect.
        # On queued/failed send (returns False), the upstream queue
        # (_pending_tool_results) holds the response for drain — but
        # because _handle_changed_settings defers reconnects while
        # _pending_tool_responses is non-empty, the queued path is
        # effectively unreachable for engine-driven reconnects with this fix.
        delivered = await super()._tool_result(tool_call_id, tool_name, tool_result_message)
        if delivered:
            self._pending_tool_responses.discard(tool_call_id)
        return delivered

    # ------------------------------------------------------------------
    # Frame handling: mute, TTSSpeakFrame, BotStoppedSpeakingFrame flush
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserMuteStartedFrame):
            self._user_is_muted = True
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, UserMuteStoppedFrame):
            self._user_is_muted = False
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSSpeakFrame):
            # Greeting trigger: the engine queues a TTSSpeakFrame to start the
            # bot's first turn after node setup. Gemini Live renders its own
            # audio, so we don't pass the frame through — we re-enter
            # _handle_context to kick off the initial response.
            if not self._handled_initial_context:
                await self._handle_context(self._context)
            else:
                logger.warning(
                    f"{self}: TTSSpeakFrame after initial context already "
                    "handled — Gemini Live owns audio generation, ignoring"
                )
            return
        if isinstance(frame, BotStoppedSpeakingFrame):
            # Belt-and-suspenders: the main drain happens in
            # _set_bot_is_responding(False), but if Gemini delays turn_complete
            # past the audible end of the turn, flushing here ensures pending
            # function calls fire promptly.
            await self._run_pending_function_calls()
            # Fall through to super for the actual push.
        await super().process_frame(frame, direction)

    async def _send_user_audio(self, frame):
        if self._user_is_muted:
            return
        await super()._send_user_audio(frame)

    # ------------------------------------------------------------------
    # Context lifecycle: Dograh pre-populates self._context via the engine,
    # so upstream's "first arrival === self._context is None" check doesn't
    # work. We gate on _handled_initial_context instead and skip the
    # init-instruction reconciliation (Dograh updates system_instruction at
    # runtime via _update_settings, not via init).
    # ------------------------------------------------------------------

    async def _handle_context(self, context: LLMContext):
        if not self._handled_initial_context:
            self._handled_initial_context = True
            self._context = context
            await self._create_initial_response()
        else:
            self._context = context
            await self._process_completed_function_calls(send_new_results=True)

    # ------------------------------------------------------------------
    # Session lifecycle: drop upstream's automatic reconnect-seed and
    # initial-context-seed paths. The TTSSpeakFrame trigger and the
    # function-call-result LLMContextFrame are the only paths that should
    # kick off bot turns in the Dograh flow.
    # ------------------------------------------------------------------

    @traced_gemini_live(operation="llm_setup")
    async def _handle_session_ready(self, session):
        logger.debug(
            f"In _handle_session_ready self._run_llm_when_session_ready: {self._run_llm_when_session_ready}"
        )
        self._session = session
        self._ready_for_realtime_input = True
        if self._run_llm_when_session_ready:
            # Context arrived before session was ready — fulfil the queued
            # initial response now.
            self._run_llm_when_session_ready = False
            await self._create_initial_response()
        await self._drain_pending_tool_results()
        # Otherwise: no automatic seed. Reconnect after a session-resumption
        # update relies on the server-side restored state; reconnects without
        # a handle (e.g. node transitions before any handle was issued) are
        # followed by a function-call-result LLMContextFrame which feeds the
        # updated-context branch in _handle_context.

    # ------------------------------------------------------------------
    # Transcription: broadcast (so downstream voicemail detector and
    # logs buffer both see it) and set finalized= for turn-boundary
    # semantics.
    # ------------------------------------------------------------------

    async def _handle_user_started_speaking(self, frame):
        await super()._handle_user_started_speaking(frame)
        # A new VAD start invalidates any pending finalize from a prior stop
        # that hasn't been paired with a transcription yet.
        self._finalize_pending = False

    async def _handle_user_stopped_speaking(self, frame):
        await super()._handle_user_stopped_speaking(frame)
        self._finalize_pending = True

    async def _push_user_transcription(self, text: str, result=None):
        await self._handle_user_transcription(text, True, self._settings.language)
        finalized = self._finalize_pending
        self._finalize_pending = False
        await self.broadcast_frame(
            TranscriptionFrame,
            text=text,
            user_id="",
            timestamp=time_now_iso8601(),
            result=result,
            finalized=finalized,
        )
