"""Shared line-delimited JSON transport for isolated Python worker processes."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("whisper_api")


def _identity(message: str) -> str:
    return message


class JsonLineWorkerClient:
    """Own one subprocess and exchange serialized request/response messages.

    Qwen and pyannote use different isolated environments but the same transport:
    a startup message followed by one JSON object per stdin/stdout line. Keeping
    that machinery here gives every worker identical timeout, pipe-draining,
    protocol validation, and shutdown behavior while domain proxies remain small.
    """

    def __init__(
        self,
        *,
        args: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        label: str,
        startup_timeout_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
        sanitize_message: Callable[[str], str] = _identity,
    ) -> None:
        self.label = label
        self.request_timeout_seconds = request_timeout_seconds
        self._sanitize_message = sanitize_message
        self._io_lock = threading.RLock()
        self._stdout_queue: queue.Queue[dict[str, Any] | Exception | None] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._closed = False

        self.process = subprocess.Popen(
            list(args),
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            # Uvicorn receives Ctrl+C as the foreground process. A separate
            # session prevents that terminal signal from reaching workers first,
            # so FastAPI's lifespan can ask them to shut down cleanly.
            start_new_session=True,
        )
        self._start_stdout_drain()
        self._start_stderr_drain()
        try:
            ready = self._read_message(
                timeout_seconds=startup_timeout_seconds,
                operation="startup",
            )
            if ready.get("type") != "ready":
                error = ready.get("error")
                message = error.get("message") if isinstance(error, dict) else None
                raise RuntimeError(
                    self._sanitize_message(message or f"{self.label} failed to initialize.")
                )
        except Exception:
            self._terminate_process()
            raise

        self.worker_pid = int(ready.get("pid", self.process.pid or 0)) or self.process.pid

    def _start_stdout_drain(self) -> None:
        """Continuously drain stdout so startup and requests can use timeouts."""
        if self.process.stdout is None:
            return

        def drain() -> None:
            assert self.process.stdout is not None
            try:
                for line in self.process.stdout:
                    try:
                        payload = json.loads(line)
                        if not isinstance(payload, dict):
                            raise TypeError("worker messages must be JSON objects")
                        self._stdout_queue.put(payload)
                    except (json.JSONDecodeError, TypeError) as exc:
                        self._stdout_queue.put(
                            RuntimeError(f"{self.label} emitted invalid JSON: {exc}")
                        )
            finally:
                # Wake a waiter immediately when the process exits rather than
                # leaving it blocked until a potentially long inference timeout.
                self._stdout_queue.put(None)

        self._stdout_thread = threading.Thread(
            target=drain,
            daemon=True,
            name=f"{self.label}-stdout",
        )
        self._stdout_thread.start()

    def _start_stderr_drain(self) -> None:
        if self.process.stderr is None:
            return

        def drain() -> None:
            assert self.process.stderr is not None
            for line in self.process.stderr:
                LOGGER.info("[%s] %s", self.label, self._sanitize_message(line.rstrip()))

        self._stderr_thread = threading.Thread(
            target=drain,
            daemon=True,
            name=f"{self.label}-stderr",
        )
        self._stderr_thread.start()

    def _read_message(
        self,
        *,
        timeout_seconds: float | None,
        operation: str,
    ) -> dict[str, Any]:
        try:
            if timeout_seconds is None:
                payload = self._stdout_queue.get()
            else:
                payload = self._stdout_queue.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            self._terminate_process()
            raise RuntimeError(
                f"{self.label} timed out during {operation} after {timeout_seconds:g} seconds."
            ) from exc

        if payload is None:
            raise RuntimeError(
                f"{self.label} exited during {operation} (exit={self.process.poll()})."
            )
        if isinstance(payload, Exception):
            self._terminate_process()
            raise RuntimeError(self._sanitize_message(str(payload))) from payload
        return payload

    def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Send one request and return its object-valued result."""
        effective_timeout = (
            self.request_timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        with self._io_lock:
            if self._closed or self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.label} is not running (exit={self.process.returncode})."
                )
            if self.process.stdin is None:
                raise RuntimeError(f"{self.label} has a broken stdin pipe.")

            request_id = uuid.uuid4().hex
            encoded = json.dumps(
                {"id": request_id, "method": method, "params": params},
                ensure_ascii=False,
            ) + "\n"
            try:
                self.process.stdin.write(encoded)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate_process()
                raise RuntimeError(f"{self.label} closed its input pipe.") from exc

            response = self._read_message(
                timeout_seconds=effective_timeout,
                operation=method,
            )

            if response.get("id") != request_id:
                self._terminate_process()
                raise RuntimeError(f"{self.label} returned a mismatched response id.")
            if not response.get("ok", False):
                error = response.get("error", {})
                message = error.get("message") if isinstance(error, dict) else None
                raise RuntimeError(
                    self._sanitize_message(message or f"Worker error in {method}.")
                )
            result = response.get("result")
            if not isinstance(result, dict):
                self._terminate_process()
                raise RuntimeError(f"{self.label} returned a non-object result for {method}.")
            return result

    def is_running(self) -> bool:
        return not self._closed and self.process.poll() is None

    def _terminate_process(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def close(self) -> None:
        """Request graceful shutdown, then guarantee that the process is gone."""
        with self._io_lock:
            if self._closed:
                return
            try:
                if self.process.poll() is None:
                    try:
                        self.request("shutdown", {}, timeout_seconds=10)
                        self.process.wait(timeout=2)
                    except Exception:
                        pass
                self._terminate_process()
            finally:
                # Report a closed client only after the process is confirmed gone.
                # If process control itself fails, the exception remains visible and
                # callers can retry cleanup instead of masking a live worker.
                self._closed = self.process.poll() is not None
