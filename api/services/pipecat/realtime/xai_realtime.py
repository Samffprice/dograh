"""Dograh subclass of pipecat's xAI Grok Realtime LLM service.

Layers Dograh engine integration quirks onto upstream-pristine
:class:`GrokRealtimeLLMService`. Like the OpenAI Realtime subclass (and unlike
Gemini Live), Grok supports runtime ``session.update`` for both
``system_instruction`` and tools, so no reconnect machinery is needed — node
transitions just push a settings delta.

Adds:

- **Deferred connect.** The WebSocket is held back until ``system_instruction``
  is set via :meth:`_update_settings`, so pre-call-fetch template variables land
  before the live session opens and the first ``session.update`` carries the
  real instructions/tools. (Upstream connects eagerly in ``start()``.) This also
  avoids the ``conversation.created`` → ``_send_session_update`` path asserting
  on an unset ``system_instruction`` once Dograh has pre-populated the context.
- **Initial-response gating via TTSSpeakFrame.** Dograh pre-populates
  ``self._context`` through the engine, so upstream's "first context arrival
  means ``self._context`` is None" trigger no longer fires. We gate on
  ``_handled_initial_context`` and let the engine's greeting ``TTSSpeakFrame``
  drive the initial response instead.
- **Function-call deferral.** Tool calls emitted mid-bot-turn are queued and run
  when the bot stops speaking, so they don't race the turn's audio.
- **User-mute audio gating.** ``UserMuteStarted/StoppedFrame`` gate whether
  incoming audio is forwarded to Grok.
- **LLMMessagesAppendFrame handling** for one-off ephemeral prompts (e.g.
  user-idle checks); upstream Grok leaves this unimplemented.
- **Transcription broadcast** with ``finalized=True`` so the downstream
  voicemail detector and logs buffer both observe completed user
  transcriptions (every completed-transcription event is final by
  construction), for parity with the OpenAI service.
"""

import json
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.xai.realtime import events
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


class DograhXAIRealtimeLLMService(GrokRealtimeLLMService):
    """xAI Grok Realtime with Dograh engine integration quirks. See module docstring."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # User-mute state, driven by broadcast UserMute{Started,Stopped}Frames.
        # Audio is not forwarded to Grok while muted.
        self._user_is_muted: bool = False
        # Dograh pre-populates self._context via the engine before the first
        # LLMContextFrame arrives, so upstream's "first arrival means
        # self._context is None" check no longer works. Gate on this instead.
        self._handled_initial_context: bool = False
        # Track bot speech locally so tool calls can be deferred until the bot
        # has finished speaking, matching the Dograh OpenAI/Gemini behavior.
        self._bot_is_speaking: bool = False
        self._deferred_function_calls: list[FunctionCallFromLLM] = []

    # ------------------------------------------------------------------
    # Deferred connect: hold the WebSocket until system_instruction is set.
    # ------------------------------------------------------------------

    async def _connect(self):
        # Pre-call fetch populates template variables into system_instruction
        # via _update_settings; hold the connection until that lands so the
        # first session.update carries the real instructions/tools.
        if not self._settings.system_instruction:
            logger.debug(
                f"{self}: deferring Grok connect until system_instruction is set"
            )
            return
        await super()._connect()

    async def _update_settings(self, delta) -> set[str]:
        changed = await super()._update_settings(delta)
        # First time system_instruction is set after a deferred connect: open
        # the session now. Subsequent changes are pushed via session.update by
        # the parent's _update_settings, so no reconnect is needed.
        if not self._websocket and self._settings.system_instruction:
            await self._connect()
        return changed

    # ------------------------------------------------------------------
    # Frame handling: mute, TTSSpeakFrame greeting trigger, bot-turn flush.
    #
    # Upstream's process_frame calls super() first and then unconditionally
    # re-pushes every frame, so we must intercept the frames the engine owns
    # (mute, TTSSpeakFrame) and return BEFORE delegating to super().
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
            # bot's first turn after node setup. Grok Realtime renders its own
            # audio, so we don't forward the frame — we re-enter _handle_context
            # to kick off the initial response.
            if not self._handled_initial_context:
                await self._handle_context(self._context)
            else:
                logger.warning(
                    f"{self}: TTSSpeakFrame after initial context already "
                    "handled — Grok Realtime owns audio generation, ignoring"
                )
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_is_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_is_speaking = False
            await self._run_pending_function_calls()
        await super().process_frame(frame, direction)

    async def _send_user_audio(self, frame):
        if self._user_is_muted:
            return
        await super()._send_user_audio(frame)

    # ------------------------------------------------------------------
    # Context lifecycle: gate on _handled_initial_context instead of upstream's
    # "self._context is None" check, since Dograh pre-populates self._context.
    # ------------------------------------------------------------------

    async def _handle_context(self, context):
        if not self._handled_initial_context:
            if context is None:
                logger.warning(
                    f"{self}: received initial context trigger before context was set"
                )
                return
            self._handled_initial_context = True
            self._context = context
            await self._create_response()
        else:
            self._context = context
            await self._process_completed_function_calls(send_new_results=True)

    # ------------------------------------------------------------------
    # Function-call deferral: queue tool calls emitted mid-bot-turn and run
    # them when the bot stops speaking, so they don't race the turn's audio.
    # ------------------------------------------------------------------

    async def _handle_evt_function_call_arguments_done(self, evt):
        try:
            args = json.loads(evt.arguments)

            function_call_item = self._pending_function_calls.get(evt.call_id)
            if not function_call_item:
                logger.warning(
                    f"No tracked function call found for call_id: {evt.call_id}"
                )
                return
            del self._pending_function_calls[evt.call_id]

            function_calls = [
                FunctionCallFromLLM(
                    context=self._context,
                    tool_call_id=evt.call_id,
                    function_name=evt.name,
                    arguments=args,
                )
            ]

            if self._bot_is_speaking:
                self._deferred_function_calls.extend(function_calls)
                logger.debug(
                    f"{self}: deferring function call {evt.name} until bot stops speaking"
                )
            else:
                await self.run_function_calls(function_calls)
                logger.debug(f"Processed function call: {evt.name}")
        except Exception as e:
            logger.error(f"Failed to process function call arguments: {e}")

    async def _run_pending_function_calls(self):
        if not self._deferred_function_calls:
            return
        function_calls = self._deferred_function_calls
        self._deferred_function_calls = []
        logger.debug(
            f"{self}: executing {len(function_calls)} deferred function call(s) "
            "after bot turn ended"
        )
        await self.run_function_calls(function_calls)

    # ------------------------------------------------------------------
    # One-off ephemeral prompts (user-idle checks, etc.). Upstream Grok leaves
    # LLMMessagesAppendFrame unimplemented; port the OpenAI behavior, appending
    # the items without mutating Dograh's local LLMContext.
    # ------------------------------------------------------------------

    async def _handle_messages_append(self, frame: LLMMessagesAppendFrame):
        if self._disconnecting:
            return

        if not self._api_session_ready:
            if frame.run_llm:
                logger.debug(
                    f"{self}: LLMMessagesAppendFrame received before session ready; "
                    "deferring response until the session is initialized"
                )
                self._run_llm_when_api_session_ready = True
            return

        appended_any = False
        for message in frame.messages:
            item = self._message_to_conversation_item(message)
            if item is None:
                continue
            evt = events.ConversationItemCreateEvent(item=item)
            self._messages_added_manually[evt.item.id] = True
            await self.send_client_event(evt)
            appended_any = True

        if frame.run_llm and appended_any:
            await self._send_manual_response_create()

    async def _send_manual_response_create(self):
        """Trigger inference after manually appending conversation items."""
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        await self.send_client_event(
            events.ResponseCreateEvent(
                response=events.ResponseProperties(modalities=["text", "audio"])
            )
        )

    def _message_to_conversation_item(
        self, message: dict[str, Any]
    ) -> "events.ConversationItem | None":
        if not isinstance(message, dict):
            logger.warning(
                f"{self}: skipping unsupported appended message payload {message!r}"
            )
            return None

        role = message.get("role")
        if role not in {"user", "system", "developer"}:
            logger.warning(
                f"{self}: skipping unsupported appended message role {role!r}"
            )
            return None

        text = self._extract_text_content(message.get("content"))
        if not text:
            logger.warning(
                f"{self}: skipping appended message with unsupported content {message!r}"
            )
            return None

        item_role = "system" if role in {"system", "developer"} else "user"
        return events.ConversationItem(
            type="message",
            role=item_role,
            content=[events.ItemContent(type="input_text", text=text)],
        )

    @staticmethod
    def _extract_text_content(content: Any) -> "str | None":
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    return None
                if part.get("type") != "text":
                    return None
                text = part.get("text")
                if not isinstance(text, str):
                    return None
                parts.append(text)
            return "\n".join(parts) if parts else None
        return None

    # ------------------------------------------------------------------
    # Transcription: broadcast (so the downstream voicemail detector and logs
    # buffer both see it) with finalized=True. Every completed-transcription
    # event from Grok is final by construction.
    # ------------------------------------------------------------------

    async def _handle_evt_input_audio_transcription_completed(self, evt):
        await self._call_event_handler("on_conversation_item_updated", evt.item_id, None)

        transcript = evt.transcript.strip() if evt.transcript else ""
        if not transcript:
            return

        await self.broadcast_frame(
            TranscriptionFrame,
            text=transcript,
            user_id="",
            timestamp=time_now_iso8601(),
            result=evt,
            finalized=True,
        )
        await self._handle_user_transcription(transcript, True, Language.EN)

    @traced_stt
    async def _handle_user_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing (parity with OpenAI service)."""
        text = transcript.strip()
        if not text:
            return
        if is_final:
            logger.debug(f"[Transcription:user] [{text}]")
