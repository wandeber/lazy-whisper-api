"""Environment-driven configuration for the lazy Whisper API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_ROOT = PROJECT_ROOT / ".cache" / "faster-whisper"


def parse_mapping(raw_value: str) -> dict[str, str]:
    """Parse comma-separated key=value pairs."""
    mapping: dict[str, str] = {}
    for entry in raw_value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid mapping entry: {entry}")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def parse_int_mapping(raw_value: str) -> dict[str, int]:
    """Parse key=value pairs where values must be integers."""
    return {key: int(value) for key, value in parse_mapping(raw_value).items()}


def parse_bool(raw_value: str | None, default: bool = False) -> bool:
    """Parse a permissive env-var boolean."""
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(level_name: str) -> None:
    """Keep application logging aligned with the configured env var."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)


def normalize_model_source(raw_value: str, project_root: Path) -> str:
    """Resolve local model paths relative to the project root."""
    candidate = Path(raw_value)
    if raw_value.startswith(".") or candidate.is_absolute() or "/" in raw_value:
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        return str(candidate)
    return raw_value


@dataclass(frozen=True)
class ModelSettings:
    """Static config for a single canonical model."""

    name: str
    source: str
    preferred_device: str
    compute_type: str
    idle_seconds: int
    vad_filter: bool


@dataclass(frozen=True)
class Settings:
    """Full runtime settings for the API process."""

    project_root: Path
    download_root: str
    default_model: str
    default_device: str
    api_key: str
    hf_token: str | None
    cpu_threads: int
    log_level: str
    max_loaded_models_cpu: int
    max_loaded_models_gpu: int
    max_concurrent_requests_per_model: int
    upload_chunk_size: int
    model_alias_map: dict[str, str]
    model_settings: dict[str, ModelSettings]
    supported_model_ids: tuple[str, ...]

    def resolve_model_name(self, requested_model: str) -> str:
        canonical = self.model_alias_map.get(requested_model, requested_model)
        if canonical not in self.model_settings:
            raise KeyError(requested_model)
        return canonical


def load_settings() -> Settings:
    """Load and validate all env-based configuration."""
    default_device = os.environ.get("WHISPER_DEFAULT_DEVICE", "cpu")
    raw_default_model = os.environ.get("WHISPER_DEFAULT_MODEL", "turbo")
    api_key = os.environ.get("WHISPER_API_KEY", "").strip()
    hf_token = os.environ.get("HF_TOKEN") or None
    cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", "0"))
    log_level = os.environ.get("WHISPER_LOG_LEVEL", "INFO").upper()
    max_loaded_models_cpu = int(os.environ.get("WHISPER_MAX_LOADED_MODELS_CPU", "1"))
    max_loaded_models_gpu = int(os.environ.get("WHISPER_MAX_LOADED_MODELS_GPU", "2"))
    max_concurrent_requests_per_model = int(
        os.environ.get("WHISPER_MAX_CONCURRENT_REQUESTS_PER_MODEL", "2")
    )
    upload_chunk_size = int(os.environ.get("WHISPER_UPLOAD_CHUNK_SIZE", str(1024 * 1024)))

    model_alias_map = parse_mapping(
        os.environ.get(
            "WHISPER_MODEL_ALIAS_MAP",
            (
                "whisper-1=turbo,"
                "turbo=turbo,"
                "large-v3=large-v3,"
                "distil=distil-multi4,"
                "distil-multi4=distil-multi4"
            ),
        )
    )
    model_source_map = {
        key: normalize_model_source(value, PROJECT_ROOT)
        for key, value in parse_mapping(
            os.environ.get(
                "WHISPER_MODEL_SOURCE_MAP",
                (
                    f"turbo=turbo,"
                    f"large-v3=large-v3,"
                    f"distil-multi4={PROJECT_ROOT / 'models' / 'distil-multi4-ct2'}"
                ),
            )
        ).items()
    }
    model_device_map = parse_mapping(
        os.environ.get(
            "WHISPER_MODEL_DEVICE_MAP",
            "turbo=cuda,large-v3=cpu,distil-multi4=cuda",
        )
    )
    model_compute_type_map = parse_mapping(
        os.environ.get(
            "WHISPER_MODEL_COMPUTE_TYPE_MAP",
            "turbo=int8,large-v3=int8,distil-multi4=int8",
        )
    )
    model_idle_seconds_map = parse_int_mapping(
        os.environ.get(
            "WHISPER_MODEL_IDLE_SECONDS_MAP",
            "turbo=5400,large-v3=600,distil-multi4=5400",
        )
    )
    model_vad_map = {
        key: parse_bool(value, default=False)
        for key, value in parse_mapping(
            os.environ.get(
                "WHISPER_MODEL_VAD_MAP",
                "turbo=false,large-v3=false,distil-multi4=false",
            )
        ).items()
    }

    if max_loaded_models_cpu < 1:
        raise ValueError("WHISPER_MAX_LOADED_MODELS_CPU must be >= 1")
    if max_loaded_models_gpu < 1:
        raise ValueError("WHISPER_MAX_LOADED_MODELS_GPU must be >= 1")
    if max_concurrent_requests_per_model < 1:
        raise ValueError("WHISPER_MAX_CONCURRENT_REQUESTS_PER_MODEL must be >= 1")
    if upload_chunk_size < 1:
        raise ValueError("WHISPER_UPLOAD_CHUNK_SIZE must be >= 1")

    model_settings = {
        model_name: ModelSettings(
            name=model_name,
            source=model_source_map[model_name],
            preferred_device=model_device_map.get(model_name, default_device),
            compute_type=model_compute_type_map.get(model_name, "default"),
            idle_seconds=model_idle_seconds_map.get(model_name, 5400),
            vad_filter=model_vad_map.get(model_name, False),
        )
        for model_name in model_source_map
    }

    invalid_aliases = sorted(
        alias for alias, canonical in model_alias_map.items() if canonical not in model_settings
    )
    if invalid_aliases:
        raise ValueError(
            "WHISPER_MODEL_ALIAS_MAP references unknown canonical models: "
            + ", ".join(invalid_aliases)
        )

    default_model = model_alias_map.get(raw_default_model, raw_default_model)
    if default_model not in model_settings:
        raise ValueError(f"WHISPER_DEFAULT_MODEL points to an unknown model: {raw_default_model}")

    supported_model_ids = tuple(sorted(set(model_alias_map) | set(model_settings)))

    return Settings(
        project_root=PROJECT_ROOT,
        download_root=str(DOWNLOAD_ROOT),
        default_model=default_model,
        default_device=default_device,
        api_key=api_key,
        hf_token=hf_token,
        cpu_threads=cpu_threads,
        log_level=log_level,
        max_loaded_models_cpu=max_loaded_models_cpu,
        max_loaded_models_gpu=max_loaded_models_gpu,
        max_concurrent_requests_per_model=max_concurrent_requests_per_model,
        upload_chunk_size=upload_chunk_size,
        model_alias_map=model_alias_map,
        model_settings=model_settings,
        supported_model_ids=supported_model_ids,
    )
