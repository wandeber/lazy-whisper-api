PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install install-gpu install-cpu install-macos setup-macos install-qwen-runtime install-qwen-cuda-runtime install-qwen-mlx-runtime install-diarization-runtime smoke-diarization test run start stop restart status logs

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -e .

install-cpu: venv
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cpu -e '.[cpu]'

install-macos: venv
	$(PIP) install -e '.[macos]'

setup-macos:
	./setup-macos.sh

install-gpu: venv
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cu126 -e '.[gpu-cu126]'

install-qwen-runtime:
	./setup-qwen-runtime.sh

install-qwen-cuda-runtime:
	./setup-qwen-runtime.sh

install-qwen-mlx-runtime:
	./setup-qwen-mlx-runtime.sh

install-diarization-runtime:
	./setup-diarization-runtime.sh

smoke-diarization:
	./smoke-test-diarization.sh

test:
	$(PY) -m pytest

run:
	./whisper-api.sh

start:
	./whisper-service.sh start

stop:
	./whisper-service.sh stop

restart:
	./whisper-service.sh restart

status:
	./whisper-service.sh status

logs:
	./whisper-service.sh logs
