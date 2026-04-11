"""Environment-driven configuration for the local multi-backend ASR API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_ROOT = PROJECT_ROOT / ".cache" / "faster-whisper"


def getenv_alias(name: str, legacy_name: str | None = None, default: str | None = None) -> str | None:
    """Read a generic ASR env var first, then its legacy Whisper alias."""
    if name in os.environ:
        return os.environ[name]
    if legacy_name and legacy_name in os.environ:
        return os.environ[legacy_name]
    return default


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
    """Parse key=value pairs with integer values."""
    return {key: int(value) for key, value in parse_mapping(raw_value).items()}


def parse_capabilities_mapping(raw_value: str) -> dict[str, frozenset[str]]:
    """Parse key=cap1|cap2 maps into frozensets."""
    mapping: dict[str, frozenset[str]] = {}
    for key, value in parse_mapping(raw_value).items():
        capabilities = {
            capability.strip()
            for capability in value.split("|")
            if capability.strip()
        }
        mapping[key] = frozenset(capabilities)
    return mapping


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


def normalize_pathish(raw_value: str, project_root: Path) -> str:
    """Resolve local paths relative to the project root while leaving repo ids intact."""
    candidate = Path(raw_value).expanduser()
    if (
        candidate.is_absolute()
        or raw_value.startswith("./")
        or raw_value.startswith("../")
        or raw_value.startswith("~")
        or (project_root / raw_value).exists()
    ):
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        return str(candidate)
    return raw_value


@dataclass(frozen=True)
class ModelSettings:
    """Static config for a single canonical model."""

    name: str
    family: str
    backend: str
    source: str
    preferred_device: str
    compute_type: str
    idle_seconds: int
    capabilities: frozenset[str]
    vad_filter: bool
    runtime_python: str | None
    gpu_memory_reservation_mb: int
    max_concurrent_requests: int
    aligner_source: str | None
    aligner_device: str | None
    aligner_dtype: str | None

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


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
    gpu_memory_budget_mb: int
    upload_chunk_size: int
    max_concurrent_requests_per_model: int
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
    default_device = getenv_alias("ASR_DEFAULT_DEVICE", "WHISPER_DEFAULT_DEVICE", "cpu") or "cpu"
    raw_default_model = getenv_alias("ASR_DEFAULT_MODEL", "WHISPER_DEFAULT_MODEL", "turbo") or "turbo"
    api_key = (getenv_alias("ASR_API_KEY", "WHISPER_API_KEY", "") or "").strip()
    hf_token = os.environ.get("HF_TOKEN") or None
    cpu_threads = int(getenv_alias("ASR_CPU_THREADS", "WHISPER_CPU_THREADS", "0") or "0")
    log_level = (getenv_alias("ASR_LOG_LEVEL", "WHISPER_LOG_LEVEL", "INFO") or "INFO").upper()
    max_loaded_models_cpu = int(
        getenv_alias("ASR_MAX_LOADED_MODELS_CPU", "WHISPER_MAX_LOADED_MODELS_CPU", "1") or "1"
    )
    max_concurrent_requests_per_model = int(
        getenv_alias(
            "ASR_MAX_CONCURRENT_REQUESTS_PER_MODEL",
            "WHISPER_MAX_CONCURRENT_REQUESTS_PER_MODEL",
            "2",
        )
        or "2"
    )
    upload_chunk_size = int(
        getenv_alias("ASR_UPLOAD_CHUNK_SIZE", "WHISPER_UPLOAD_CHUNK_SIZE", str(1024 * 1024))
        or str(1024 * 1024)
    )
    gpu_memory_budget_mb = int(
        getenv_alias("ASR_GPU_MEMORY_BUDGET_MB", "WHISPER_GPU_MEMORY_BUDGET_MB", "8192") or "8192"
    )

    model_alias_map = parse_mapping(
        getenv_alias(
            "ASR_MODEL_ALIAS_MAP",
            "WHISPER_MODEL_ALIAS_MAP",
            (
                "whisper-1=turbo,"
                "turbo=turbo,"
                "large-v3=large-v3,"
                "distil=distil-multi4,"
                "distil-multi4=distil-multi4,"
                "qwen3-asr-0.6b=qwen3-asr-0.6b,"
                "qwen3-asr-1.7b=qwen3-asr-1.7b,"
                "qwen-0.6b=qwen3-asr-0.6b,"
                "qwen-1.7b=qwen3-asr-1.7b"
            ),
        )
        or ""
    )
    model_source_map = {
        key: normalize_pathish(value, PROJECT_ROOT)
        for key, value in parse_mapping(
            getenv_alias(
                "ASR_MODEL_SOURCE_MAP",
                "WHISPER_MODEL_SOURCE_MAP",
                (
                    f"turbo=turbo,"
                    f"large-v3=large-v3,"
                    f"distil-multi4={PROJECT_ROOT / 'models' / 'distil-multi4-ct2'},"
                    "qwen3-asr-0.6b=Qwen/Qwen3-ASR-0.6B,"
                    "qwen3-asr-1.7b=Qwen/Qwen3-ASR-1.7B"
                ),
            )
            or ""
        ).items()
    }
    model_family_map = parse_mapping(
        getenv_alias(
            "ASR_MODEL_FAMILY_MAP",
            None,
            (
                "turbo=whisper,"
                "large-v3=whisper,"
                "distil-multi4=whisper,"
                "qwen3-asr-0.6b=qwen,"
                "qwen3-asr-1.7b=qwen"
            ),
        )
        or ""
    )
    model_backend_map = parse_mapping(
        getenv_alias(
            "ASR_MODEL_BACKEND_MAP",
            None,
            (
                "turbo=faster-whisper,"
                "large-v3=faster-whisper,"
                "distil-multi4=faster-whisper,"
                "qwen3-asr-0.6b=qwen-worker,"
                "qwen3-asr-1.7b=qwen-worker"
            ),
        )
        or ""
    )
    model_device_map = parse_mapping(
        getenv_alias(
            "ASR_MODEL_DEVICE_MAP",
            "WHISPER_MODEL_DEVICE_MAP",
            "turbo=cuda,large-v3=cpu,distil-multi4=cuda,qwen3-asr-0.6b=cuda,qwen3-asr-1.7b=cuda",
        )
        or ""
    )
    model_compute_type_map = parse_mapping(
        getenv_alias(
            "ASR_MODEL_COMPUTE_TYPE_MAP",
            "WHISPER_MODEL_COMPUTE_TYPE_MAP",
            "turbo=int8,large-v3=int8,distil-multi4=int8,qwen3-asr-0.6b=float16,qwen3-asr-1.7b=float16",
        )
        or ""
    )
    model_idle_seconds_map = parse_int_mapping(
        getenv_alias(
            "ASR_MODEL_IDLE_SECONDS_MAP",
            "WHISPER_MODEL_IDLE_SECONDS_MAP",
            "turbo=5400,large-v3=600,distil-multi4=5400,qwen3-asr-0.6b=5400,qwen3-asr-1.7b=5400",
        )
        or ""
    )
    model_vad_map = {
        key: parse_bool(value, default=False)
        for key, value in parse_mapping(
            getenv_alias(
                "ASR_MODEL_VAD_MAP",
                "WHISPER_MODEL_VAD_MAP",
                "turbo=false,large-v3=false,distil-multi4=false,qwen3-asr-0.6b=false,qwen3-asr-1.7b=false",
            )
            or ""
        ).items()
    }
    model_capabilities_map = parse_capabilities_mapping(
        getenv_alias(
            "ASR_MODEL_CAPABILITIES_MAP",
            None,
            (
                "turbo=transcribe|translate|timestamps|stream|realtime,"
                "large-v3=transcribe|translate|timestamps|stream|realtime,"
                "distil-multi4=transcribe|translate|timestamps|stream|realtime,"
                "qwen3-asr-0.6b=transcribe|timestamps|stream|realtime,"
                "qwen3-asr-1.7b=transcribe|timestamps|stream|realtime"
            ),
        )
        or ""
    )
    family_runtime_python_map = {
        key: normalize_pathish(value, PROJECT_ROOT)
        for key, value in parse_mapping(
            getenv_alias(
                "ASR_FAMILY_RUNTIME_PYTHON_MAP",
                None,
                f"qwen={PROJECT_ROOT / '.venv-qwen' / 'bin' / 'python'}",
            )
            or ""
        ).items()
    }
    model_gpu_memory_map = parse_int_mapping(
        getenv_alias(
            "ASR_MODEL_GPU_MEMORY_RESERVATION_MB_MAP",
            None,
            "turbo=5200,large-v3=0,distil-multi4=4200,qwen3-asr-0.6b=6500,qwen3-asr-1.7b=7800",
        )
        or ""
    )
    model_concurrency_map = parse_int_mapping(
        getenv_alias(
            "ASR_MODEL_MAX_CONCURRENT_REQUESTS_MAP",
            None,
            "turbo=2,large-v3=2,distil-multi4=2,qwen3-asr-0.6b=1,qwen3-asr-1.7b=1",
        )
        or ""
    )
    model_aligner_source_map = {
        key: normalize_pathish(value, PROJECT_ROOT)
        for key, value in parse_mapping(
            getenv_alias(
                "ASR_MODEL_ALIGNER_SOURCE_MAP",
                None,
                "qwen3-asr-0.6b=Qwen/Qwen3-ForcedAligner-0.6B,qwen3-asr-1.7b=Qwen/Qwen3-ForcedAligner-0.6B",
            )
            or ""
        ).items()
    }
    model_aligner_device_map = parse_mapping(
        getenv_alias(
            "ASR_MODEL_ALIGNER_DEVICE_MAP",
            None,
            "qwen3-asr-0.6b=cpu,qwen3-asr-1.7b=cpu",
        )
        or ""
    )
    model_aligner_dtype_map = parse_mapping(
        getenv_alias(
            "ASR_MODEL_ALIGNER_DTYPE_MAP",
            None,
            "qwen3-asr-0.6b=float32,qwen3-asr-1.7b=float32",
        )
        or ""
    )

    if max_loaded_models_cpu < 1:
        raise ValueError("ASR_MAX_LOADED_MODELS_CPU must be >= 1")
    if max_concurrent_requests_per_model < 1:
        raise ValueError("ASR_MAX_CONCURRENT_REQUESTS_PER_MODEL must be >= 1")
    if upload_chunk_size < 1:
        raise ValueError("ASR_UPLOAD_CHUNK_SIZE must be >= 1")
    if gpu_memory_budget_mb < 1:
        raise ValueError("ASR_GPU_MEMORY_BUDGET_MB must be >= 1")

    model_settings = {
        model_name: ModelSettings(
            name=model_name,
            family=model_family_map.get(model_name, "whisper"),
            backend=model_backend_map.get(model_name, "faster-whisper"),
            source=model_source_map[model_name],
            preferred_device=model_device_map.get(model_name, default_device),
            compute_type=model_compute_type_map.get(model_name, "default"),
            idle_seconds=model_idle_seconds_map.get(model_name, 5400),
            capabilities=model_capabilities_map.get(model_name, frozenset({"transcribe"})),
            vad_filter=model_vad_map.get(model_name, False),
            runtime_python=family_runtime_python_map.get(model_family_map.get(model_name, "whisper")),
            gpu_memory_reservation_mb=model_gpu_memory_map.get(model_name, 0),
            max_concurrent_requests=model_concurrency_map.get(
                model_name,
                max_concurrent_requests_per_model,
            ),
            aligner_source=model_aligner_source_map.get(model_name),
            aligner_device=model_aligner_device_map.get(model_name),
            aligner_dtype=model_aligner_dtype_map.get(model_name),
        )
        for model_name in model_source_map
    }

    invalid_aliases = sorted(
        alias for alias, canonical in model_alias_map.items() if canonical not in model_settings
    )
    if invalid_aliases:
        raise ValueError(
            "ASR_MODEL_ALIAS_MAP references unknown canonical models: "
            + ", ".join(invalid_aliases)
        )

    default_model = model_alias_map.get(raw_default_model, raw_default_model)
    if default_model not in model_settings:
        raise ValueError(f"ASR_DEFAULT_MODEL points to an unknown model: {raw_default_model}")

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
        gpu_memory_budget_mb=gpu_memory_budget_mb,
        upload_chunk_size=upload_chunk_size,
        max_concurrent_requests_per_model=max_concurrent_requests_per_model,
        model_alias_map=model_alias_map,
        model_settings=model_settings,
        supported_model_ids=supported_model_ids,
    )
