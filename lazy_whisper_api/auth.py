"""Authentication helpers for protected API routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Header

from .errors import api_error


def build_api_key_dependency(api_key: str) -> Callable[..., None]:
    """Return a FastAPI dependency that enforces a fixed API key."""

    def require_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if not api_key:
            return

        provided_key: str | None = None
        if authorization and authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()
        elif x_api_key:
            provided_key = x_api_key.strip()

        if provided_key != api_key:
            raise api_error(
                401,
                "Invalid API key provided.",
                error_type="invalid_api_key",
            )

    return require_api_key
