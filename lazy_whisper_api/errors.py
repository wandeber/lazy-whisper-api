"""Shared API error helpers and formatting."""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.requests import Request


def api_error(status_code: int, message: str, *, error_type: str) -> HTTPException:
    """Create a JSON-shaped HTTPException compatible with the OpenAI client."""
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "type": error_type,
        },
    )


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Normalize FastAPI exceptions into OpenAI-style error payloads."""
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    error_type = detail.get("type")
    if error_type is None:
        if exc.status_code == 401:
            error_type = "invalid_api_key"
        elif exc.status_code >= 500:
            error_type = "server_error"
        else:
            error_type = "invalid_request_error"

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": detail.get("message", ""), "type": error_type}},
    )
