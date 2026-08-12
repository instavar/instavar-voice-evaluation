from __future__ import annotations

import importlib.metadata
import platform
import re
from pathlib import Path
from typing import Any

from .extraction import (
    EXECUTED_ASR_EXTRACTION_SCHEMA_VERSION,
    build_extractor_identity,
    observation_document_sha256,
    verify_observation_audio,
)
from .speaker_references import canonical_sha256

FASTER_WHISPER_BACKEND = "instavar_voice_lab.faster_whisper_v1"
DEVICE_RE = re.compile(r"^(cpu|cuda)$")
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")
COMPUTE_TYPES = {
    "auto",
    "default",
    "int8",
    "int8_bfloat16",
    "int8_float16",
    "int8_float32",
    "int16",
    "float16",
    "float32",
    "bfloat16",
}


def faster_whisper_artifacts(model_dir: Path) -> dict[str, tuple[Path, str]]:
    return {
        "model": (model_dir, "tree"),
        "runner": (Path(__file__), "file"),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("faster-whisper", "ctranslate2", "tokenizers", "av"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required optional package is not installed: {package}") from error
    return versions


def _load_model(model_dir: Path, *, device: str, device_index: int, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("faster-whisper is not installed") from error
    return WhisperModel(
        str(model_dir),
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        local_files_only=True,
    )


def _transcribe(model: Any, path: Path, *, language: str, beam_size: int) -> str:
    segments, _information = model.transcribe(
        str(path),
        language=language,
        task="transcribe",
        beam_size=beam_size,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
        vad_filter=False,
    )
    parts = [str(segment.text).strip() for segment in segments if str(segment.text).strip()]
    return " ".join(parts)


def _result_document_sha256(document: dict[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in document.items() if key != "execution_receipt_sha256"})


def build_faster_whisper_results(
    observations: Any,
    *,
    audio_base_dir: Path,
    model_dir: Path,
    model_name: str,
    model_revision: str,
    device: str = "cpu",
    device_index: int = 0,
    compute_type: str = "int8",
    language: str = "en",
    beam_size: int = 5,
) -> dict[str, Any]:
    if not isinstance(device, str) or not DEVICE_RE.fullmatch(device):
        raise ValueError("device must be cpu or cuda")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    if device == "cpu" and device_index != 0:
        raise ValueError("CPU execution requires device_index 0")
    if compute_type not in COMPUTE_TYPES:
        raise ValueError(f"compute_type must be one of: {', '.join(sorted(COMPUTE_TYPES))}")
    if not isinstance(language, str) or not LANGUAGE_RE.fullmatch(language):
        raise ValueError("language must be a lowercase language code such as en or zh")
    if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
        raise ValueError("beam_size must be a positive integer")

    source_sha256 = observation_document_sha256(observations)
    rows = observations
    artifacts = faster_whisper_artifacts(model_dir)
    extractor = build_extractor_identity(
        kind="asr",
        name=model_name,
        revision=model_revision,
        artifacts=artifacts,
    )
    runtime_versions = _package_versions()
    if set(runtime_versions) != {"faster-whisper", "ctranslate2", "tokenizers", "av"} or any(
        not isinstance(value, str) or not value.strip() for value in runtime_versions.values()
    ):
        raise ValueError("package_versions must contain non-empty faster-whisper runtime versions")
    execution = {
        "backend": FASTER_WHISPER_BACKEND,
        "device": device,
        "device_index": device_index,
        "compute_type": compute_type,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": runtime_versions,
        "model_loading": {"local_files_only": True},
        "decoding": {
            "language": language,
            "task": "transcribe",
            "beam_size": beam_size,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "word_timestamps": False,
            "vad_filter": False,
        },
    }

    verified_audio: dict[str, tuple[Path, str]] = {}
    for index, row in enumerate(rows):
        if row["valid"] is True:
            verified_audio[row["sample_id"]] = verify_observation_audio(row, audio_base_dir, index)

    try:
        model = _load_model(
            model_dir,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )
    except Exception as error:
        raise RuntimeError(f"failed to initialize faster-whisper on {device}:{device_index}: {error}") from error

    results: list[dict[str, Any]] = []
    for row in rows:
        if row["valid"] is not True:
            continue
        sample_id = row["sample_id"]
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "audio_sha256": verified_audio[sample_id][1],
            "status": "complete",
        }
        try:
            result["values"] = {
                "hypothesis_text": _transcribe(
                    model,
                    verified_audio[sample_id][0],
                    language=language,
                    beam_size=beam_size,
                )
            }
        except Exception as error:  # noqa: BLE001 - preserve backend failures as evaluation evidence
            result.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error) or type(error).__name__,
                }
            )
        results.append(result)

    if (
        build_extractor_identity(
            kind="asr",
            name=model_name,
            revision=model_revision,
            artifacts=artifacts,
        )
        != extractor
    ):
        raise ValueError("faster-whisper model or runner artifacts changed during extraction")
    for index, row in enumerate(rows):
        if (
            row["valid"] is True
            and verify_observation_audio(row, audio_base_dir, index) != verified_audio[row["sample_id"]]
        ):
            raise ValueError(f"observation audio changed during extraction: {row['sample_id']}")

    document = {
        "schema_version": EXECUTED_ASR_EXTRACTION_SCHEMA_VERSION,
        "source_observations_sha256": source_sha256,
        "extractor": extractor,
        "execution": execution,
        "results": results,
    }
    return {**document, "execution_receipt_sha256": _result_document_sha256(document)}


def validate_execution_receipt(extraction: dict[str, Any]) -> dict[str, Any]:
    execution = extraction.get("execution")
    expected_fields = {
        "backend",
        "device",
        "device_index",
        "compute_type",
        "python_version",
        "platform",
        "package_versions",
        "model_loading",
        "decoding",
    }
    if not isinstance(execution, dict) or set(execution) != expected_fields:
        raise ValueError("ASR execution does not contain the exact required provenance fields")
    if execution.get("backend") != FASTER_WHISPER_BACKEND:
        raise ValueError("unsupported ASR execution backend")
    device = execution.get("device")
    if not isinstance(device, str) or not DEVICE_RE.fullmatch(device):
        raise ValueError("ASR execution device is invalid")
    device_index = execution.get("device_index")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("ASR execution device_index is invalid")
    if device == "cpu" and device_index != 0:
        raise ValueError("ASR CPU execution requires device_index 0")
    if execution.get("compute_type") not in COMPUTE_TYPES:
        raise ValueError("ASR execution compute_type is invalid")
    for field in ("python_version", "platform"):
        if not isinstance(execution.get(field), str) or not execution[field].strip():
            raise ValueError(f"ASR execution {field} must be non-empty")
    versions = execution.get("package_versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != {"faster-whisper", "ctranslate2", "tokenizers", "av"}
        or any(not isinstance(value, str) or not value.strip() for value in versions.values())
    ):
        raise ValueError("ASR execution package versions are invalid")
    if execution.get("model_loading") != {"local_files_only": True}:
        raise ValueError("ASR execution must use local-only model loading")
    decoding = execution.get("decoding")
    if not isinstance(decoding, dict) or set(decoding) != {
        "language",
        "task",
        "beam_size",
        "temperature",
        "condition_on_previous_text",
        "word_timestamps",
        "vad_filter",
    }:
        raise ValueError("ASR decoding provenance is invalid")
    if (
        not isinstance(decoding.get("language"), str)
        or not LANGUAGE_RE.fullmatch(decoding["language"])
        or decoding.get("task") != "transcribe"
        or isinstance(decoding.get("beam_size"), bool)
        or not isinstance(decoding.get("beam_size"), int)
        or decoding["beam_size"] < 1
        or decoding.get("temperature") != 0.0
        or decoding.get("condition_on_previous_text") is not False
        or decoding.get("word_timestamps") is not False
        or decoding.get("vad_filter") is not False
    ):
        raise ValueError("ASR decoding settings are invalid")
    digest = extraction.get("execution_receipt_sha256")
    if digest != _result_document_sha256(extraction):
        raise ValueError("execution_receipt_sha256 does not match the extractor result document")
    return execution
