"""Environment-driven configuration for the local multi-backend ASR API."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_ROOT = PROJECT_ROOT / ".cache" / "faster-whisper"
SUBTITLE_PROFILE_NAME = "subtitles-v1"
EDIT_MAX_PROFILE_NAME = "edit-max-v1"
EDIT_MAX_MODEL_ID = "qwen-1.7b-edit-max"
EDIT_MAX_CANONICAL_MODEL = "qwen3-asr-1.7b"


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


def parse_env_int(name: str, default: int) -> int:
    """Parse one integer option while keeping startup errors actionable."""
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def parse_env_float(name: str, default: float) -> float:
    """Parse one floating-point option while naming the invalid variable."""
    try:
        return float(os.environ.get(name, str(default)) or str(default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


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
            # Keep virtualenv interpreter symlinks intact. Executing the real
            # Homebrew/Python target would bypass the venv site-packages, which
            # breaks isolated worker runtimes such as `.venv-qwen-mlx`.
            candidate = (project_root / candidate).absolute()
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
class EditMaxSettings:
    """Versioned acoustic settings for the edit-oriented transcription profile.

    Silero works in 32 ms frames, while the local energy pass searches around
    those coarse transitions at a finer resolution. All thresholds live here
    so boundary behavior is reproducible and can be tuned without hiding magic
    constants inside the fusion algorithm.
    """

    sample_rate_hz: int
    vad_start_threshold: float
    vad_end_threshold: float
    min_speech_ms: int
    min_silence_ms: int
    energy_window_ms: int
    energy_search_ms: int
    energy_silence_run_ms: int
    energy_speech_run_ms: int
    energy_noise_percentile: float
    energy_noise_multiplier: float
    energy_min_dbfs: float
    energy_max_dbfs: float
    word_association_ms: int
    outer_word_snap_ms: int
    vad_only_min_peak: float
    vad_only_min_mean: float


@dataclass(frozen=True)
class ModelProfileSettings:
    """Behavior layered over a canonical model without duplicating its runtime."""

    name: str
    mode: str
    edit_max: EditMaxSettings | None = None

    @property
    def is_edit_max(self) -> bool:
        return self.mode == "edit"


@dataclass(frozen=True)
class ModelRoute:
    """Resolved public ID, canonical scheduler key, and behavioral profile."""

    requested_model: str
    canonical_model: str
    profile: ModelProfileSettings


@dataclass(frozen=True)
class DiarizationSettings:
    """Static config for the optional local speaker diarization backend."""

    enabled: bool
    backend: str
    model_id: str
    model_path: str
    device: str
    idle_seconds: int
    runtime_python: str
    startup_timeout_seconds: int
    request_timeout_seconds: int


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
    model_profile_map: dict[str, str]
    model_profiles: dict[str, ModelProfileSettings]
    model_settings: dict[str, ModelSettings]
    supported_model_ids: tuple[str, ...]
    diarization: DiarizationSettings

    def resolve_model_name(self, requested_model: str) -> str:
        canonical = self.model_alias_map.get(requested_model, requested_model)
        if canonical not in self.model_settings:
            raise KeyError(requested_model)
        return canonical

    def resolve_model_route(self, requested_model: str) -> ModelRoute:
        """Resolve an API ID without losing its behavioral profile.

        Most existing IDs deliberately fall back to the subtitle profile. The
        edit-max ID is reserved and fail-closed: a configuration bug must never
        turn a precision-editing request into an ordinary subtitle request.
        """
        canonical = self.resolve_model_name(requested_model)
        profile_name = self.model_profile_map.get(requested_model, SUBTITLE_PROFILE_NAME)
        try:
            profile = self.model_profiles[profile_name]
        except KeyError as exc:
            raise KeyError(requested_model) from exc

        if requested_model == EDIT_MAX_MODEL_ID and (
            canonical != EDIT_MAX_CANONICAL_MODEL or profile.name != EDIT_MAX_PROFILE_NAME
        ):
            raise KeyError(requested_model)
        return ModelRoute(
            requested_model=requested_model,
            canonical_model=canonical,
            profile=profile,
        )


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
    diarization_enabled = parse_bool(os.environ.get("ASR_DIARIZATION_ENABLED"), default=False)
    diarization_backend = os.environ.get("ASR_DIARIZATION_BACKEND", "pyannote").strip()
    diarization_model_id = os.environ.get(
        "ASR_DIARIZATION_MODEL_ID",
        "pyannote/speaker-diarization-community-1",
    ).strip() or "pyannote/speaker-diarization-community-1"
    diarization_model_path = normalize_pathish(
        os.environ.get(
            "ASR_DIARIZATION_MODEL_PATH",
            str(PROJECT_ROOT / "models" / "pyannote-speaker-diarization-community-1"),
        ),
        PROJECT_ROOT,
    )
    diarization_device = os.environ.get("ASR_DIARIZATION_DEVICE", "cpu").strip() or "cpu"
    diarization_idle_seconds = int(os.environ.get("ASR_DIARIZATION_IDLE_SECONDS", "1800") or "1800")
    diarization_runtime_python = normalize_pathish(
        os.environ.get(
            "ASR_DIARIZATION_RUNTIME_PYTHON",
            str(PROJECT_ROOT / ".venv-diarization" / "bin" / "python"),
        ),
        PROJECT_ROOT,
    )
    diarization_startup_timeout_seconds = int(
        os.environ.get("ASR_DIARIZATION_STARTUP_TIMEOUT_SECONDS", "300") or "300"
    )
    diarization_request_timeout_seconds = int(
        os.environ.get("ASR_DIARIZATION_REQUEST_TIMEOUT_SECONDS", "3600") or "3600"
    )

    alias_map_is_explicit = (
        "ASR_MODEL_ALIAS_MAP" in os.environ or "WHISPER_MODEL_ALIAS_MAP" in os.environ
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
                "qwen-1.7b=qwen3-asr-1.7b,"
                f"{EDIT_MAX_MODEL_ID}={EDIT_MAX_CANONICAL_MODEL}"
            ),
        )
        or ""
    )
    profile_map_is_explicit = "ASR_MODEL_PROFILE_MAP" in os.environ
    if profile_map_is_explicit:
        model_profile_map = parse_mapping(os.environ.get("ASR_MODEL_PROFILE_MAP", ""))
    elif alias_map_is_explicit:
        # Environment maps replace the defaults in this project. An older local
        # alias map therefore remains valid and simply does not expose edit-max.
        model_profile_map = {}
    else:
        model_profile_map = {EDIT_MAX_MODEL_ID: EDIT_MAX_PROFILE_NAME}

    edit_max = EditMaxSettings(
        sample_rate_hz=16_000,
        vad_start_threshold=parse_env_float("ASR_EDIT_MAX_VAD_START_THRESHOLD", 0.50),
        vad_end_threshold=parse_env_float("ASR_EDIT_MAX_VAD_END_THRESHOLD", 0.35),
        min_speech_ms=parse_env_int("ASR_EDIT_MAX_MIN_SPEECH_MS", 64),
        min_silence_ms=parse_env_int("ASR_EDIT_MAX_MIN_SILENCE_MS", 96),
        energy_window_ms=parse_env_int("ASR_EDIT_MAX_ENERGY_WINDOW_MS", 10),
        energy_search_ms=parse_env_int("ASR_EDIT_MAX_ENERGY_SEARCH_MS", 192),
        energy_silence_run_ms=parse_env_int("ASR_EDIT_MAX_ENERGY_SILENCE_RUN_MS", 30),
        energy_speech_run_ms=parse_env_int("ASR_EDIT_MAX_ENERGY_SPEECH_RUN_MS", 20),
        energy_noise_percentile=parse_env_float(
            "ASR_EDIT_MAX_ENERGY_NOISE_PERCENTILE", 20.0
        ),
        energy_noise_multiplier=parse_env_float(
            "ASR_EDIT_MAX_ENERGY_NOISE_MULTIPLIER", 3.0
        ),
        energy_min_dbfs=parse_env_float("ASR_EDIT_MAX_ENERGY_MIN_DBFS", -60.0),
        energy_max_dbfs=parse_env_float("ASR_EDIT_MAX_ENERGY_MAX_DBFS", -35.0),
        word_association_ms=parse_env_int("ASR_EDIT_MAX_WORD_ASSOCIATION_MS", 240),
        outer_word_snap_ms=parse_env_int("ASR_EDIT_MAX_OUTER_WORD_SNAP_MS", 240),
        vad_only_min_peak=parse_env_float("ASR_EDIT_MAX_VAD_ONLY_MIN_PEAK", 0.80),
        vad_only_min_mean=parse_env_float("ASR_EDIT_MAX_VAD_ONLY_MIN_MEAN", 0.60),
    )
    model_profiles = {
        SUBTITLE_PROFILE_NAME: ModelProfileSettings(
            name=SUBTITLE_PROFILE_NAME,
            mode="subtitles",
        ),
        EDIT_MAX_PROFILE_NAME: ModelProfileSettings(
            name=EDIT_MAX_PROFILE_NAME,
            mode="edit",
            edit_max=edit_max,
        ),
    }
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
    model_runtime_python_map = {
        key: normalize_pathish(value, PROJECT_ROOT)
        for key, value in parse_mapping(
            getenv_alias(
                "ASR_MODEL_RUNTIME_PYTHON_MAP",
                None,
                "",
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
    if diarization_backend != "pyannote":
        raise ValueError("ASR_DIARIZATION_BACKEND currently supports only 'pyannote'")
    if diarization_idle_seconds < 0:
        raise ValueError("ASR_DIARIZATION_IDLE_SECONDS must be >= 0")
    if diarization_startup_timeout_seconds < 1:
        raise ValueError("ASR_DIARIZATION_STARTUP_TIMEOUT_SECONDS must be >= 1")
    if diarization_request_timeout_seconds < 1:
        raise ValueError("ASR_DIARIZATION_REQUEST_TIMEOUT_SECONDS must be >= 1")

    finite_options = {
        "ASR_EDIT_MAX_VAD_START_THRESHOLD": edit_max.vad_start_threshold,
        "ASR_EDIT_MAX_VAD_END_THRESHOLD": edit_max.vad_end_threshold,
        "ASR_EDIT_MAX_ENERGY_NOISE_PERCENTILE": edit_max.energy_noise_percentile,
        "ASR_EDIT_MAX_ENERGY_NOISE_MULTIPLIER": edit_max.energy_noise_multiplier,
        "ASR_EDIT_MAX_ENERGY_MIN_DBFS": edit_max.energy_min_dbfs,
        "ASR_EDIT_MAX_ENERGY_MAX_DBFS": edit_max.energy_max_dbfs,
        "ASR_EDIT_MAX_VAD_ONLY_MIN_PEAK": edit_max.vad_only_min_peak,
        "ASR_EDIT_MAX_VAD_ONLY_MIN_MEAN": edit_max.vad_only_min_mean,
    }
    for option_name, option_value in finite_options.items():
        if not math.isfinite(option_value):
            raise ValueError(f"{option_name} must be finite")
    probability_options = {
        "ASR_EDIT_MAX_VAD_START_THRESHOLD": edit_max.vad_start_threshold,
        "ASR_EDIT_MAX_VAD_END_THRESHOLD": edit_max.vad_end_threshold,
        "ASR_EDIT_MAX_VAD_ONLY_MIN_PEAK": edit_max.vad_only_min_peak,
        "ASR_EDIT_MAX_VAD_ONLY_MIN_MEAN": edit_max.vad_only_min_mean,
    }
    for option_name, option_value in probability_options.items():
        if not 0.0 <= option_value <= 1.0:
            raise ValueError(f"{option_name} must be between 0 and 1")
    if edit_max.vad_start_threshold <= edit_max.vad_end_threshold:
        raise ValueError(
            "ASR_EDIT_MAX_VAD_START_THRESHOLD must be greater than "
            "ASR_EDIT_MAX_VAD_END_THRESHOLD"
        )
    positive_duration_options = {
        "ASR_EDIT_MAX_MIN_SPEECH_MS": edit_max.min_speech_ms,
        "ASR_EDIT_MAX_MIN_SILENCE_MS": edit_max.min_silence_ms,
        "ASR_EDIT_MAX_ENERGY_WINDOW_MS": edit_max.energy_window_ms,
        "ASR_EDIT_MAX_ENERGY_SEARCH_MS": edit_max.energy_search_ms,
        "ASR_EDIT_MAX_ENERGY_SILENCE_RUN_MS": edit_max.energy_silence_run_ms,
        "ASR_EDIT_MAX_ENERGY_SPEECH_RUN_MS": edit_max.energy_speech_run_ms,
    }
    for option_name, option_value in positive_duration_options.items():
        if option_value < 1:
            raise ValueError(f"{option_name} must be >= 1")
    if not 0.0 <= edit_max.energy_noise_percentile <= 100.0:
        raise ValueError("ASR_EDIT_MAX_ENERGY_NOISE_PERCENTILE must be between 0 and 100")
    if edit_max.energy_noise_multiplier <= 0.0:
        raise ValueError("ASR_EDIT_MAX_ENERGY_NOISE_MULTIPLIER must be > 0")
    if edit_max.energy_min_dbfs >= edit_max.energy_max_dbfs:
        raise ValueError(
            "ASR_EDIT_MAX_ENERGY_MIN_DBFS must be lower than "
            "ASR_EDIT_MAX_ENERGY_MAX_DBFS"
        )
    non_negative_duration_options = {
        "ASR_EDIT_MAX_WORD_ASSOCIATION_MS": edit_max.word_association_ms,
        "ASR_EDIT_MAX_OUTER_WORD_SNAP_MS": edit_max.outer_word_snap_ms,
    }
    for option_name, option_value in non_negative_duration_options.items():
        if option_value < 0:
            raise ValueError(f"{option_name} must be >= 0")

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
            runtime_python=(
                model_runtime_python_map.get(model_name)
                or family_runtime_python_map.get(model_family_map.get(model_name, "whisper"))
            ),
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

    edit_alias_target = model_alias_map.get(EDIT_MAX_MODEL_ID)
    edit_profile_name = model_profile_map.get(EDIT_MAX_MODEL_ID)
    if edit_alias_target is None:
        if edit_profile_name is not None:
            raise ValueError(
                f"ASR_MODEL_PROFILE_MAP configures reserved ID '{EDIT_MAX_MODEL_ID}' "
                "but ASR_MODEL_ALIAS_MAP does not expose it."
            )
    elif (
        edit_alias_target != EDIT_MAX_CANONICAL_MODEL
        or edit_profile_name != EDIT_MAX_PROFILE_NAME
    ):
        raise ValueError(
            f"Reserved model ID '{EDIT_MAX_MODEL_ID}' must map exactly to "
            f"'{EDIT_MAX_CANONICAL_MODEL}' with profile '{EDIT_MAX_PROFILE_NAME}'."
        )

    supported_ids = set(model_alias_map) | set(model_settings)
    invalid_profile_ids = sorted(set(model_profile_map) - supported_ids)
    if invalid_profile_ids:
        raise ValueError(
            "ASR_MODEL_PROFILE_MAP references unknown public model IDs: "
            + ", ".join(invalid_profile_ids)
        )
    invalid_profile_names = sorted(
        model_id
        for model_id, profile_name in model_profile_map.items()
        if profile_name not in model_profiles
    )
    if invalid_profile_names:
        raise ValueError(
            "ASR_MODEL_PROFILE_MAP references unknown profiles for: "
            + ", ".join(invalid_profile_names)
        )

    for model_id, profile_name in model_profile_map.items():
        profile = model_profiles[profile_name]
        if not profile.is_edit_max:
            continue
        canonical = model_alias_map.get(model_id, model_id)
        spec = model_settings[canonical]
        if (
            spec.family != "qwen"
            or not spec.supports("transcribe")
            or not spec.supports("timestamps")
            or not spec.aligner_source
        ):
            raise ValueError(
                f"Profile '{profile_name}' requires a Qwen transcription model with "
                f"timestamps and a configured aligner: {model_id}"
            )

    default_model = model_alias_map.get(raw_default_model, raw_default_model)
    if default_model not in model_settings:
        raise ValueError(f"ASR_DEFAULT_MODEL points to an unknown model: {raw_default_model}")
    if model_profile_map.get(raw_default_model) == EDIT_MAX_PROFILE_NAME:
        raise ValueError(
            "ASR_DEFAULT_MODEL cannot use the edit-max profile because realtime "
            "sessions use the default model."
        )

    supported_model_ids = tuple(sorted(supported_ids))

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
        model_profile_map=model_profile_map,
        model_profiles=model_profiles,
        model_settings=model_settings,
        supported_model_ids=supported_model_ids,
        diarization=DiarizationSettings(
            enabled=diarization_enabled,
            backend=diarization_backend,
            model_id=diarization_model_id,
            model_path=diarization_model_path,
            device=diarization_device,
            idle_seconds=diarization_idle_seconds,
            runtime_python=diarization_runtime_python,
            startup_timeout_seconds=diarization_startup_timeout_seconds,
            request_timeout_seconds=diarization_request_timeout_seconds,
        ),
    )
