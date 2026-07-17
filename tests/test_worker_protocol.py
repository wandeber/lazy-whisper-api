from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from lazy_whisper_api.worker_protocol import JsonLineWorkerClient


WORKER_SOURCE = r"""
import json
import os
import sys
import time

print(json.dumps({"type": "ready", "pid": os.getpid()}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    request_id = request["id"]
    method = request["method"]
    params = request.get("params", {})
    if method == "shutdown":
        print(json.dumps({"id": request_id, "ok": True, "result": {"shutdown": True}}), flush=True)
        break
    if method == "sleep":
        time.sleep(float(params["seconds"]))
    if method == "fail":
        print(json.dumps({"id": request_id, "ok": False, "error": {"message": params["message"]}}), flush=True)
        continue
    print(json.dumps({"id": request_id, "ok": True, "result": params}), flush=True)
"""


def build_client(tmp_path: Path, **kwargs) -> JsonLineWorkerClient:
    return JsonLineWorkerClient(
        args=[sys.executable, "-u", "-c", WORKER_SOURCE],
        cwd=tmp_path,
        env=os.environ.copy(),
        label="test-worker",
        startup_timeout_seconds=2,
        request_timeout_seconds=2,
        **kwargs,
    )


def test_json_line_worker_client_round_trips_and_closes(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    assert client.request("echo", {"message": "hello"}) == {"message": "hello"}
    assert client.is_running() is True

    client.close()

    assert client.is_running() is False


def test_json_line_worker_client_isolates_worker_process_signals(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    try:
        # Workers run in their own POSIX session so Ctrl+C reaches Uvicorn first;
        # FastAPI's lifespan then closes each child through the JSON protocol.
        assert os.getsid(client.process.pid) == client.process.pid
    finally:
        client.close()


def test_json_line_worker_client_sanitizes_worker_errors(tmp_path: Path) -> None:
    client = build_client(
        tmp_path,
        sanitize_message=lambda message: message.replace("secret", "<redacted>"),
    )
    try:
        with pytest.raises(RuntimeError, match="<redacted>"):
            client.request("fail", {"message": "secret"})
        assert client.is_running() is True
    finally:
        client.close()


def test_json_line_worker_client_terminates_after_timeout(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    with pytest.raises(RuntimeError, match="timed out during sleep"):
        client.request("sleep", {"seconds": 1}, timeout_seconds=0.05)

    assert client.is_running() is False
    client.close()
