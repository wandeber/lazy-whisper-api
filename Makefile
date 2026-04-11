PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install install-gpu install-cpu install-qwen-runtime test run start stop restart status logs

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -e .

install-cpu: venv
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cpu -e '.[cpu]'

install-gpu: venv
	$(PIP) install --extra-index-url https://download.pytorch.org/whl/cu126 -e '.[gpu-cu126]'

install-qwen-runtime:
	./setup-qwen-runtime.sh

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
