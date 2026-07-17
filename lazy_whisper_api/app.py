"""FastAPI application assembly for the lazy multi-backend ASR API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from starlette.concurrency import run_in_threadpool

from .auth import build_api_key_dependency
from .config import configure_logging, load_settings
from .diarization import DiarizationManager
from .errors import http_exception_handler
from .model_manager import ModelManager
from .realtime import RealtimeTranscriptionServer
from .responses import build_transcription_response
from .streaming import create_transcription_stream_response
from .transcription import TranscriptionRequest, transcribe_upload


def create_app() -> FastAPI:
    """Create the fully configured API application."""
    settings = load_settings()
    configure_logging(settings.log_level)

    model_manager = ModelManager(settings)
    diarization_manager = DiarizationManager(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Tie every loaded model and worker to this FastAPI instance."""
        try:
            yield
        finally:
            # Worker shutdown can wait briefly for a subprocess. Keep those waits
            # off the event loop, and always unload ASR even if diarization cleanup
            # encounters an unexpected process-level failure.
            try:
                await run_in_threadpool(diarization_manager.unload)
            finally:
                await run_in_threadpool(model_manager.unload_all)

    app = FastAPI(title="Local ASR API", version="3.0.0", lifespan=lifespan)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.diarization_manager = diarization_manager

    require_api_key = build_api_key_dependency(settings.api_key)
    v1_router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

    @app.get("/healthz", dependencies=[Depends(require_api_key)])
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "default_model": settings.default_model,
            "cuda_available": torch.cuda.is_available(),
            "loaded_models": model_manager.snapshot(),
            "diarization": diarization_manager.snapshot(),
        }

    @v1_router.get("/models")
    def list_models() -> dict[str, Any]:
        data = []
        for model_id in settings.supported_model_ids:
            canonical = settings.model_alias_map.get(model_id, model_id)
            spec = settings.model_settings[canonical]
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "local-asr",
                    "family": spec.family,
                    "backend": spec.backend,
                }
            )
        return {
            "object": "list",
            "data": data,
        }

    async def handle_audio_request(
        *,
        file: UploadFile,
        model: str,
        task: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        stream: bool,
        temperature: float,
        timestamp_granularities: list[str] | None,
        diarize: bool,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ):
        payload = TranscriptionRequest(
            file=file,
            model=model,
            task=task,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            diarize=diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        if stream:
            if diarize:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Diarization is not supported for streaming responses.",
                        "type": "invalid_request_error",
                    },
                )
            if task != "transcribe":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Streaming is only supported for /v1/audio/transcriptions.",
                        "type": "invalid_request_error",
                    },
                )
            return await create_transcription_stream_response(
                settings=settings,
                model_manager=model_manager,
                payload=payload,
            )
        result = await transcribe_upload(
            settings=settings,
            model_manager=model_manager,
            diarization_manager=diarization_manager,
            payload=payload,
        )
        return build_transcription_response(result)

    @v1_router.post("/audio/transcriptions", response_model=None)
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        stream: bool = Form(default=False),
        temperature: float = Form(default=0.0),
        timestamp_granularities: list[str] | None = Form(
            default=None,
            alias="timestamp_granularities[]",
        ),
        diarize: bool = Form(default=False),
        num_speakers: int | None = Form(default=None),
        min_speakers: int | None = Form(default=None),
        max_speakers: int | None = Form(default=None),
    ):
        return await handle_audio_request(
            file=file,
            model=model,
            task="transcribe",
            language=language,
            prompt=prompt,
            response_format=response_format,
            stream=stream,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            diarize=diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    @v1_router.post("/audio/translations", response_model=None)
    async def create_translation(
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        stream: bool = Form(default=False),
        temperature: float = Form(default=0.0),
        timestamp_granularities: list[str] | None = Form(
            default=None,
            alias="timestamp_granularities[]",
        ),
        diarize: bool = Form(default=False),
        num_speakers: int | None = Form(default=None),
        min_speakers: int | None = Form(default=None),
        max_speakers: int | None = Form(default=None),
    ):
        return await handle_audio_request(
            file=file,
            model=model,
            task="translate",
            language=None,
            prompt=prompt,
            response_format=response_format,
            stream=stream,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            diarize=diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    @app.websocket("/v1/realtime")
    async def realtime_transcription(websocket: WebSocket) -> None:
        server = RealtimeTranscriptionServer(
            settings=settings,
            model_manager=model_manager,
        )
        await server.run(websocket)

    app.include_router(v1_router)
    return app


app = create_app()
