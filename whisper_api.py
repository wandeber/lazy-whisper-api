#!/usr/bin/env python3
"""Compatibility entrypoint for uvicorn and existing shell scripts."""

from lazy_whisper_api import app

__all__ = ["app"]
