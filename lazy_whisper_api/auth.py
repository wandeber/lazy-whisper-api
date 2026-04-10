"""Authentication helpers for protected API routes and WebSockets."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Header

from .errors import api_error


def extract_api_key(
    *,
    authorization: str | None = None,
    x_api_key: str | None = None,
    query_api_key: str | None = None,
) -> str | None:
    """Extract an API key from supported HTTP or WebSocket inputs."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    if x_api_key:
        return x_api_key.strip()
    if query_api_key:
        return query_api_key.strip()
    return None


def require_api_key_value(
    expected_api_key: str,
    *,
    authorization: str | None = None,
    x_api_key: str | None = None,
    query_api_key: str | None = None,
) -> None:
    """Raise if the provided API key does not match the configured one."""
    if not expected_api_key:
        return

    provided_key = extract_api_key(
        authorization=authorization,
        x_api_key=x_api_key,
        query_api_key=query_api_key,
    )
    if provided_key != expected_api_key:
        raise api_error(
            401,
            "Invalid API key provided.",
            error_type="invalid_api_key",
        )


def build_api_key_dependency(api_key: str) -> Callable[..., None]:
    """Return a FastAPI dependency that enforces a fixed API key."""

    def require_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        require_api_key_value(
            api_key,
            authorization=authorization,
            x_api_key=x_api_key,
        )

    return require_api_key
