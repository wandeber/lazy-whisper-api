"""OpenAI-like realtime transcription over WebSocket."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .auth import require_api_key_value
from .config import Settings
from .model_manager import LoadedModel, ModelManager
from .transcription import ensure_timestamp_segments_for_pcm, transcribe_pcm16_sync


PCM_SAMPLE_RATE_HZ = 24_000
PCM_FRAME_MS = 20
PCM_FRAME_SIZE_BYTES = PCM_SAMPLE_RATE_HZ * PCM_FRAME_MS // 1000 * 2
PARTIAL_INTERVAL_SECONDS = 0.75
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_PREFIX_PADDING_MS = 300
DEFAULT_VAD_SILENCE_DURATION_MS = 500
WS_CLOSE_POLICY_VIOLATION = 1008


@dataclass
class ServerVADConfig:
    """Runtime settings for server-side VAD."""

    threshold: float = DEFAULT_VAD_THRESHOLD
    prefix_padding_ms: int = DEFAULT_VAD_PREFIX_PADDING_MS
    silence_duration_ms: int = DEFAULT_VAD_SILENCE_DURATION_MS

    @property
    def prefix_frame_limit(self) -> int:
        return max(1, self.prefix_padding_ms // PCM_FRAME_MS)

    @property
    def silence_frame_limit(self) -> int:
        return max(1, self.silence_duration_ms // PCM_FRAME_MS)


@dataclass
class RealtimeSessionConfig:
    """Active configuration for one realtime WebSocket session."""

    model: str
    language: str | None = None
    prompt: str | None = None
    turn_detection: ServerVADConfig | None = field(default_factory=ServerVADConfig)


@dataclass
class RealtimeTurn:
    """One in-flight realtime transcription turn."""

    item_id: str
    model_name: str
    language: str | None
    prompt: str | None
    turn_detection: ServerVADConfig | None
    pcm_buffer: bytearray = field(default_factory=bytearray)
    frame_remainder: bytearray = field(default_factory=bytearray)
    prefix_frames: deque[bytes] = field(default_factory=deque)
    speech_started: bool = False
    trailing_silence_frames: int = 0
    committed: bool = False
    closed: bool = False
    update_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_emitted_text: str = ""
    last_processed_size: int = 0
    worker_task: asyncio.Task[None] | None = None
    lease_cm: Any | None = None
    lease_entry: LoadedModel | None = None


def new_id(prefix: str) -> str:
    """Generate a stable-looking OpenAI-style identifier."""
    return f"{prefix}_{uuid.uuid4().hex}"


def common_prefix_length(left: str, right: str) -> int:
    """Return the length of the shared prefix between two strings."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def frame_has_voice(frame: bytes, threshold: float) -> bool:
    """Very small VAD helper based on normalized peak amplitude."""
    if not frame:
        return False

    peak = 0
    for index in range(0, len(frame), 2):
        sample = int.from_bytes(frame[index : index + 2], "little", signed=True)
        absolute = abs(sample)
        if absolute > peak:
            peak = absolute
    return (peak / 32768.0) >= threshold


class RealtimeTranscriptionServer:
    """A transcription-only WebSocket server that mimics OpenAI Realtime."""

    def __init__(self, *, settings: Settings, model_manager: ModelManager) -> None:
        self.settings = settings
        self.model_manager = model_manager
        self.session_id = new_id("sess")
        self.send_lock = asyncio.Lock()
        self.session = RealtimeSessionConfig(model=settings.default_model)
        self.active_turns: dict[str, RealtimeTurn] = {}
        self.current_turn_id: str | None = None
        self.websocket: WebSocket | None = None

    async def run(self, websocket: WebSocket) -> None:
        """Handle one realtime transcription socket."""
        if not self._authenticate(websocket):
            await websocket.close(code=WS_CLOSE_POLICY_VIOLATION)
            return

        await websocket.accept()
        self.websocket = websocket
        await self.send_event("session.created", {"session": self.session_payload()})

        try:
            while True:
                message = await websocket.receive_json()
                await self.handle_client_event(message)
        except WebSocketDisconnect:
            pass
        finally:
            await self.shutdown()
            self.websocket = None

    def _authenticate(self, websocket: WebSocket) -> bool:
        """Validate the configured API key for WebSocket connections."""
        try:
            require_api_key_value(
                self.settings.api_key,
                authorization=websocket.headers.get("authorization"),
                query_api_key=websocket.query_params.get("api_key"),
            )
        except HTTPException:
            return False
        return True

    def session_payload(self) -> dict[str, Any]:
        """Render the session object returned to clients."""
        turn_detection: dict[str, Any] | None
        if self.session.turn_detection is None:
            turn_detection = None
        else:
            turn_detection = {
                "type": "server_vad",
                "threshold": self.session.turn_detection.threshold,
                "prefix_padding_ms": self.session.turn_detection.prefix_padding_ms,
                "silence_duration_ms": self.session.turn_detection.silence_duration_ms,
            }

        return {
            "id": self.session_id,
            "object": "realtime.session",
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": PCM_SAMPLE_RATE_HZ,
                    },
                    "transcription": {
                        "model": self.session.model,
                        "language": self.session.language,
                        "prompt": self.session.prompt,
                    },
                    "turn_detection": turn_detection,
                }
            },
        }

    async def send_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Serialize one server event."""
        if self.websocket is None:
            raise RuntimeError("Realtime WebSocket is not connected.")
        payload["type"] = event_type
        payload.setdefault("event_id", new_id("event"))
        async with self.send_lock:
            await self.websocket.send_json(payload)

    async def send_error(
        self,
        *,
        message: str,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        event_id: str | None = None,
        item_id: str | None = None,
    ) -> None:
        """Emit a realtime error event while keeping the socket alive."""
        error: dict[str, Any] = {
            "type": error_type,
            "message": message,
        }
        if code is not None:
            error["code"] = code
        if event_id is not None:
            error["event_id"] = event_id
        if item_id is not None:
            error["item_id"] = item_id
        await self.send_event("error", {"error": error})

    async def handle_client_event(self, message: dict[str, Any]) -> None:
        """Dispatch one client event."""
        event_type = message.get("type")
        event_id = message.get("event_id")
        if not isinstance(event_type, str):
            await self.send_error(
                message="Client event is missing a string 'type'.",
                code="invalid_event",
                event_id=event_id,
            )
            return

        if event_type == "session.update":
            await self.handle_session_update(message)
        elif event_type == "input_audio_buffer.append":
            await self.handle_input_audio_append(message)
        elif event_type == "input_audio_buffer.commit":
            await self.handle_input_audio_commit(message)
        elif event_type == "input_audio_buffer.clear":
            await self.handle_input_audio_clear(message)
        else:
            await self.send_error(
                message=f"Unsupported realtime event '{event_type}'.",
                code="unsupported_event",
                event_id=event_id,
            )

    async def handle_session_update(self, message: dict[str, Any]) -> None:
        """Apply supported session updates and reject the rest."""
        event_id = message.get("event_id")
        allowed_root_keys = {"type", "event_id", "session"}
        extra_root_keys = set(message) - allowed_root_keys
        if extra_root_keys:
            await self.send_error(
                message=f"Unsupported session.update keys: {sorted(extra_root_keys)}",
                code="unsupported_field",
                event_id=event_id,
            )
            return

        session_update = message.get("session")
        if not isinstance(session_update, dict):
            await self.send_error(
                message="session.update requires a 'session' object.",
                code="invalid_event",
                event_id=event_id,
            )
            return

        try:
            self.session = self.apply_session_update(session_update)
        except ValueError as exc:
            await self.send_error(
                message=str(exc),
                code="unsupported_field",
                event_id=event_id,
            )
            return

        await self.send_event("session.updated", {"session": self.session_payload()})

    def apply_session_update(self, session_update: dict[str, Any]) -> RealtimeSessionConfig:
        """Validate and apply a session.update payload."""
        allowed_session_keys = {"type", "audio"}
        extra_session_keys = set(session_update) - allowed_session_keys
        if extra_session_keys:
            raise ValueError(f"Unsupported session fields: {sorted(extra_session_keys)}")

        updated = RealtimeSessionConfig(
            model=self.session.model,
            language=self.session.language,
            prompt=self.session.prompt,
            turn_detection=self.session.turn_detection,
        )

        if "type" in session_update and session_update["type"] != "transcription":
            raise ValueError("Realtime v1 only supports session type 'transcription'.")

        if "audio" not in session_update:
            return updated

        audio = session_update["audio"]
        if not isinstance(audio, dict):
            raise ValueError("'audio' must be an object.")
        extra_audio_keys = set(audio) - {"input"}
        if extra_audio_keys:
            raise ValueError(f"Unsupported audio fields: {sorted(extra_audio_keys)}")

        audio_input = audio.get("input")
        if audio_input is None:
            return updated
        if not isinstance(audio_input, dict):
            raise ValueError("'audio.input' must be an object.")
        extra_input_keys = set(audio_input) - {"format", "transcription", "turn_detection"}
        if extra_input_keys:
            raise ValueError(f"Unsupported audio.input fields: {sorted(extra_input_keys)}")

        if "format" in audio_input:
            fmt = audio_input["format"]
            if not isinstance(fmt, dict):
                raise ValueError("'audio.input.format' must be an object.")
            extra_format_keys = set(fmt) - {"type", "rate"}
            if extra_format_keys:
                raise ValueError(
                    f"Unsupported audio.input.format fields: {sorted(extra_format_keys)}"
                )
            if fmt.get("type", "audio/pcm") != "audio/pcm":
                raise ValueError("Realtime v1 only supports audio.input.format.type='audio/pcm'.")
            if int(fmt.get("rate", PCM_SAMPLE_RATE_HZ)) != PCM_SAMPLE_RATE_HZ:
                raise ValueError(
                    f"Realtime v1 only supports audio.input.format.rate={PCM_SAMPLE_RATE_HZ}."
                )

        if "transcription" in audio_input:
            transcription = audio_input["transcription"]
            if not isinstance(transcription, dict):
                raise ValueError("'audio.input.transcription' must be an object.")
            extra_transcription_keys = set(transcription) - {"model", "language", "prompt"}
            if extra_transcription_keys:
                raise ValueError(
                    "Unsupported audio.input.transcription fields: "
                    f"{sorted(extra_transcription_keys)}"
                )
            if "model" in transcription:
                try:
                    updated.model = self.settings.resolve_model_name(str(transcription["model"]))
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported model '{transcription['model']}'."
                    ) from exc
                if not self.settings.model_settings[updated.model].supports("realtime"):
                    raise ValueError(
                        f"Model '{transcription['model']}' does not support realtime transcription."
                    )
            if "language" in transcription:
                updated.language = (
                    None if transcription["language"] in {None, ""} else str(transcription["language"])
                )
            if "prompt" in transcription:
                updated.prompt = None if transcription["prompt"] in {None, ""} else str(
                    transcription["prompt"]
                )

        if "turn_detection" in audio_input:
            turn_detection = audio_input["turn_detection"]
            if turn_detection is None:
                updated.turn_detection = None
            else:
                if not isinstance(turn_detection, dict):
                    raise ValueError("'audio.input.turn_detection' must be null or an object.")
                extra_turn_keys = set(turn_detection) - {
                    "type",
                    "threshold",
                    "prefix_padding_ms",
                    "silence_duration_ms",
                }
                if extra_turn_keys:
                    raise ValueError(
                        "Unsupported audio.input.turn_detection fields: "
                        f"{sorted(extra_turn_keys)}"
                    )
                if turn_detection.get("type", "server_vad") != "server_vad":
                    raise ValueError(
                        "Realtime v1 only supports turn_detection=null or type='server_vad'."
                    )
                threshold = float(turn_detection.get("threshold", DEFAULT_VAD_THRESHOLD))
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError("turn_detection.threshold must be between 0 and 1.")
                prefix_padding_ms = int(
                    turn_detection.get("prefix_padding_ms", DEFAULT_VAD_PREFIX_PADDING_MS)
                )
                silence_duration_ms = int(
                    turn_detection.get(
                        "silence_duration_ms",
                        DEFAULT_VAD_SILENCE_DURATION_MS,
                    )
                )
                if prefix_padding_ms < 0 or silence_duration_ms < 1:
                    raise ValueError(
                        "turn_detection prefix_padding_ms and silence_duration_ms must be positive."
                    )
                updated.turn_detection = ServerVADConfig(
                    threshold=threshold,
                    prefix_padding_ms=prefix_padding_ms,
                    silence_duration_ms=silence_duration_ms,
                )

        return updated

    def create_turn(self) -> RealtimeTurn:
        """Create a new turn from the current session configuration."""
        turn = RealtimeTurn(
            item_id=new_id("item"),
            model_name=self.session.model,
            language=self.session.language,
            prompt=self.session.prompt,
            turn_detection=(
                None
                if self.session.turn_detection is None
                else ServerVADConfig(
                    threshold=self.session.turn_detection.threshold,
                    prefix_padding_ms=self.session.turn_detection.prefix_padding_ms,
                    silence_duration_ms=self.session.turn_detection.silence_duration_ms,
                )
            ),
        )
        if turn.turn_detection is not None:
            turn.prefix_frames = deque(maxlen=turn.turn_detection.prefix_frame_limit)
        self.active_turns[turn.item_id] = turn
        self.current_turn_id = turn.item_id
        return turn

    def current_turn(self) -> RealtimeTurn | None:
        """Return the currently open input turn."""
        if self.current_turn_id is None:
            return None
        return self.active_turns.get(self.current_turn_id)

    async def ensure_turn_started(self, turn: RealtimeTurn) -> bool:
        """Reserve model capacity for one live turn and start its worker."""
        if turn.closed:
            return False
        if turn.lease_entry is not None:
            return True

        turn.lease_cm = self.model_manager.lease(turn.model_name)
        try:
            turn.lease_entry = turn.lease_cm.__enter__()
        except HTTPException as exc:
            if turn.lease_cm is not None:
                with contextlib.suppress(Exception):
                    turn.lease_cm.__exit__(None, None, None)
            await self.send_error(
                message=exc.detail["message"] if isinstance(exc.detail, dict) else str(exc.detail),
                error_type=(
                    exc.detail.get("type", "server_error")
                    if isinstance(exc.detail, dict)
                    else "server_error"
                ),
                code="turn_unavailable",
                item_id=turn.item_id,
            )
            await self.close_turn(turn)
            return False

        turn.worker_task = asyncio.create_task(self.run_turn(turn))
        return True

    async def close_turn(self, turn: RealtimeTurn) -> None:
        """Stop a turn and release its resources."""
        turn.closed = True
        turn.update_event.set()
        if self.current_turn_id == turn.item_id:
            self.current_turn_id = None

        if turn.worker_task is None:
            if turn.lease_cm is not None:
                with contextlib.suppress(Exception):
                    turn.lease_cm.__exit__(None, None, None)
                turn.lease_cm = None
                turn.lease_entry = None
            self.active_turns.pop(turn.item_id, None)

    async def handle_input_audio_append(self, message: dict[str, Any]) -> None:
        """Append audio to the current input buffer."""
        event_id = message.get("event_id")
        allowed_keys = {"type", "event_id", "audio"}
        extra_keys = set(message) - allowed_keys
        if extra_keys:
            await self.send_error(
                message=f"Unsupported input_audio_buffer.append keys: {sorted(extra_keys)}",
                code="unsupported_field",
                event_id=event_id,
            )
            return

        audio_field = message.get("audio")
        if not isinstance(audio_field, str):
            await self.send_error(
                message="input_audio_buffer.append requires an 'audio' base64 string.",
                code="invalid_event",
                event_id=event_id,
            )
            return

        try:
            audio_bytes = base64.b64decode(audio_field, validate=True)
        except Exception:
            await self.send_error(
                message="input_audio_buffer.append contains invalid base64 audio.",
                code="invalid_audio",
                event_id=event_id,
            )
            return

        if not audio_bytes:
            return

        turn = self.current_turn()
        if turn is None:
            turn = self.create_turn()

        if turn.turn_detection is None:
            await self.append_manual_audio(turn, audio_bytes)
            return

        await self.append_vad_audio(audio_bytes)

    async def append_manual_audio(self, turn: RealtimeTurn, audio_bytes: bytes) -> None:
        """Append audio for a manual-commit turn."""
        if not await self.ensure_turn_started(turn):
            return
        turn.pcm_buffer.extend(audio_bytes)
        turn.update_event.set()

    async def append_vad_audio(self, audio_bytes: bytes) -> None:
        """Append audio while applying a simple server-side VAD."""
        turn = self.current_turn()
        if turn is None:
            turn = self.create_turn()

        data = bytes(turn.frame_remainder) + audio_bytes
        turn.frame_remainder.clear()

        while len(data) >= PCM_FRAME_SIZE_BYTES:
            frame = data[:PCM_FRAME_SIZE_BYTES]
            data = data[PCM_FRAME_SIZE_BYTES:]
            turn = self.current_turn() or self.create_turn()
            auto_committed = await self.append_vad_frame(turn, frame)
            if auto_committed:
                turn = self.current_turn() or self.create_turn()

        current_turn = self.current_turn()
        if current_turn is not None:
            current_turn.frame_remainder.extend(data)

    async def append_vad_frame(self, turn: RealtimeTurn, frame: bytes) -> bool:
        """Append one full PCM frame with VAD state transitions."""
        assert turn.turn_detection is not None

        if not turn.speech_started:
            turn.prefix_frames.append(frame)
            if not frame_has_voice(frame, turn.turn_detection.threshold):
                return False

            turn.speech_started = True
            while turn.prefix_frames:
                turn.pcm_buffer.extend(turn.prefix_frames.popleft())
            if not await self.ensure_turn_started(turn):
                return False
            turn.update_event.set()
            return False

        turn.pcm_buffer.extend(frame)
        turn.update_event.set()
        if frame_has_voice(frame, turn.turn_detection.threshold):
            turn.trailing_silence_frames = 0
            return False

        turn.trailing_silence_frames += 1
        if turn.trailing_silence_frames >= turn.turn_detection.silence_frame_limit:
            await self.commit_turn(turn, auto=True)
            return True
        return False

    async def handle_input_audio_commit(self, message: dict[str, Any]) -> None:
        """Commit the current audio buffer for final transcription."""
        event_id = message.get("event_id")
        allowed_keys = {"type", "event_id"}
        extra_keys = set(message) - allowed_keys
        if extra_keys:
            await self.send_error(
                message=f"Unsupported input_audio_buffer.commit keys: {sorted(extra_keys)}",
                code="unsupported_field",
                event_id=event_id,
            )
            return

        turn = self.current_turn()
        if turn is None:
            await self.send_error(
                message="No current audio buffer to commit.",
                code="empty_buffer",
                event_id=event_id,
            )
            return

        await self.commit_turn(turn, auto=False)

    async def commit_turn(self, turn: RealtimeTurn, *, auto: bool) -> None:
        """Seal a turn and trigger its final transcription pass."""
        if turn.committed or turn.closed:
            return

        if turn.turn_detection is not None and not turn.speech_started and turn.prefix_frames:
            while turn.prefix_frames:
                turn.pcm_buffer.extend(turn.prefix_frames.popleft())

        if turn.frame_remainder:
            turn.pcm_buffer.extend(turn.frame_remainder)
            turn.frame_remainder.clear()

        if not turn.pcm_buffer:
            await self.send_error(
                message="No audio buffered to commit.",
                code="empty_buffer",
                item_id=turn.item_id,
            )
            await self.close_turn(turn)
            return

        if not await self.ensure_turn_started(turn):
            return

        turn.committed = True
        if self.current_turn_id == turn.item_id:
            self.current_turn_id = None

        await self.send_event(
            "input_audio_buffer.committed",
            {
                "item_id": turn.item_id,
                "commit_mode": "server_vad" if auto else "manual",
            },
        )
        turn.update_event.set()

    async def handle_input_audio_clear(self, message: dict[str, Any]) -> None:
        """Clear the current in-progress turn without closing the socket."""
        event_id = message.get("event_id")
        allowed_keys = {"type", "event_id"}
        extra_keys = set(message) - allowed_keys
        if extra_keys:
            await self.send_error(
                message=f"Unsupported input_audio_buffer.clear keys: {sorted(extra_keys)}",
                code="unsupported_field",
                event_id=event_id,
            )
            return

        turn = self.current_turn()
        if turn is None:
            return
        await self.close_turn(turn)

    async def run_turn(self, turn: RealtimeTurn) -> None:
        """Emit partial and final transcription events for one active turn."""
        try:
            while not turn.closed:
                if turn.committed:
                    await self.transcribe_and_emit(turn, final=True)
                    return

                try:
                    await asyncio.wait_for(turn.update_event.wait(), timeout=PARTIAL_INTERVAL_SECONDS)
                except TimeoutError:
                    pass

                if turn.closed:
                    return
                turn.update_event.clear()
                if len(turn.pcm_buffer) == 0 or len(turn.pcm_buffer) == turn.last_processed_size:
                    continue
                await self.transcribe_and_emit(turn, final=False)
        except Exception as exc:  # pragma: no cover - exercised in integration flow
            await self.send_error(
                message=str(exc),
                error_type="server_error",
                code="transcription_failed",
                item_id=turn.item_id,
            )
        finally:
            self.active_turns.pop(turn.item_id, None)
            if self.current_turn_id == turn.item_id:
                self.current_turn_id = None
            if turn.lease_cm is not None:
                with contextlib.suppress(Exception):
                    turn.lease_cm.__exit__(None, None, None)
                turn.lease_cm = None
                turn.lease_entry = None

    async def transcribe_and_emit(self, turn: RealtimeTurn, *, final: bool) -> None:
        """Run one partial or final transcription pass for a turn."""
        if turn.lease_entry is None or turn.lease_entry.runtime is None:
            raise RuntimeError(f"Turn '{turn.item_id}' has no active model lease.")

        pcm_snapshot = bytes(turn.pcm_buffer)
        transcription = await run_in_threadpool(
            transcribe_pcm16_sync,
            runtime=turn.lease_entry.runtime,
            pcm_bytes=pcm_snapshot,
            sample_rate_hz=PCM_SAMPLE_RATE_HZ,
            language=turn.language,
            task="transcribe",
            prompt=turn.prompt,
            temperature=0.0,
            word_timestamps=False,
        )
        text = transcription.text.strip()
        turn.last_processed_size = len(pcm_snapshot)

        if final:
            segments = transcription.segments
            if turn.lease_entry.spec.supports("timestamps") and not segments:
                segments = await run_in_threadpool(
                    ensure_timestamp_segments_for_pcm,
                    lease=turn.lease_entry,
                    pcm_bytes=pcm_snapshot,
                    sample_rate_hz=PCM_SAMPLE_RATE_HZ,
                    transcription=transcription,
                )
            await self.send_event(
                "conversation.item.input_audio_transcription.completed",
                {
                    "item_id": turn.item_id,
                    "transcript": text,
                    "model": turn.model_name,
                    "device": turn.lease_entry.actual_device,
                    "language": transcription.info.language,
                    "segments": [
                        {
                            "id": segment.id,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip(),
                            "words": [
                                {
                                    "start": word.start,
                                    "end": word.end,
                                    "word": word.word,
                                    "probability": word.probability,
                                }
                                for word in (segment.words or [])
                            ],
                        }
                        for segment in segments
                    ],
                },
            )
            turn.closed = True
            return

        if text == turn.last_emitted_text:
            return

        prefix_length = common_prefix_length(turn.last_emitted_text, text)
        delta = text[prefix_length:]
        turn.last_emitted_text = text
        await self.send_event(
            "conversation.item.input_audio_transcription.delta",
            {
                "item_id": turn.item_id,
                "delta": delta,
                "text": text,
            },
        )

    async def shutdown(self) -> None:
        """Release all active turns for a closing socket."""
        turns = list(self.active_turns.values())
        for turn in turns:
            turn.closed = True
            turn.update_event.set()
            if turn.worker_task is None:
                self.active_turns.pop(turn.item_id, None)
                if turn.lease_cm is not None:
                    with contextlib.suppress(Exception):
                        turn.lease_cm.__exit__(None, None, None)
                    turn.lease_cm = None
                    turn.lease_entry = None

        tasks = [turn.worker_task for turn in turns if turn.worker_task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
