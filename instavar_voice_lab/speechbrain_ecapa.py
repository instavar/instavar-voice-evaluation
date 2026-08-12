from __future__ import annotations

import importlib.metadata
import math
import platform
import re
from pathlib import Path
from typing import Any

from .extraction import (
    EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
    build_extractor_identity,
    build_speaker_reference_catalog,
    observation_document_sha256,
    verify_observation_audio,
)
from .speaker_reference_plans import validate_speaker_reference_assignment_plan
from .speaker_references import REFERENCE_AGGREGATION, canonical_sha256

SPEECHBRAIN_ECAPA_NAME = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_ECAPA_BACKEND = "instavar_voice_lab.speechbrain_ecapa_v1"
DEVICE_RE = re.compile(r"^(cpu|cuda(?::[0-9]+)?)$")


def speechbrain_ecapa_artifacts(model_dir: Path) -> dict[str, tuple[Path, str]]:
    return {
        "model": (model_dir, "tree"),
        "runner": (Path(__file__), "file"),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("speechbrain", "torch", "torchaudio"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required optional package is not installed: {package}") from error
    return versions


def _load_classifier(model_dir: Path, device: str) -> Any:
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("CUDA execution requires Torch") from error
        _validate_torch_device(torch, device)
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError as error:
            raise RuntimeError("SpeechBrain does not expose EncoderClassifier") from error
    return EncoderClassifier.from_hparams(
        source=str(model_dir),
        run_opts={"device": device},
        overrides={"pretrained_path": str(model_dir)},
    )


def _validate_torch_device(torch_module: Any, device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but Torch reports no available CUDA device")
    device_index = int(device.split(":", 1)[1]) if ":" in device else 0
    device_count = int(torch_module.cuda.device_count())
    if device_index >= device_count:
        raise RuntimeError(f"CUDA device index {device_index} is unavailable; detected {device_count} devices")


def _encode_file(classifier: Any, path: Path) -> list[float]:
    signal = classifier.load_audio(str(path))
    embedding = classifier.encode_batch(signal.unsqueeze(0)).squeeze().detach().cpu().tolist()
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("SpeechBrain returned an empty or non-vector speaker embedding")
    values = [float(value) for value in embedding]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("SpeechBrain returned a non-finite speaker embedding")
    return values


def _result_document_sha256(document: dict[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in document.items() if key != "execution_receipt_sha256"})


def build_speechbrain_ecapa_results(
    observations: Any,
    *,
    audio_base_dir: Path,
    model_dir: Path,
    model_revision: str,
    catalog_id: str,
    speaker_references: dict[str, tuple[Path, Path]],
    speaker_reference_plan: dict[str, Any],
    generation_plan: dict[str, Any],
    device: str = "cpu",
    trusted_model_checkpoints: bool = False,
) -> dict[str, Any]:
    if not isinstance(device, str) or not DEVICE_RE.fullmatch(device):
        raise ValueError("device must be cpu, cuda, or cuda followed by a device index")
    if trusted_model_checkpoints is not True:
        raise ValueError("SpeechBrain checkpoints may contain pickle data; pass explicit checkpoint trust")
    source_sha256 = observation_document_sha256(observations)
    rows = observations
    artifacts = speechbrain_ecapa_artifacts(model_dir)
    extractor = build_extractor_identity(
        kind="speaker_encoder",
        name=SPEECHBRAIN_ECAPA_NAME,
        revision=model_revision,
        artifacts=artifacts,
    )
    reference_catalog = build_speaker_reference_catalog(catalog_id=catalog_id, references=speaker_references)
    assignment_plan = validate_speaker_reference_assignment_plan(
        speaker_reference_plan,
        generation_plan=generation_plan,
        reference_catalog=reference_catalog,
    )
    runtime_versions = _package_versions()
    if set(runtime_versions) != {"speechbrain", "torch", "torchaudio"} or any(
        not isinstance(value, str) or not value.strip() for value in runtime_versions.values()
    ):
        raise ValueError("package_versions must contain non-empty speechbrain, torch, and torchaudio versions")
    execution = {
        "backend": SPEECHBRAIN_ECAPA_BACKEND,
        "device": device,
        "trusted_model_checkpoints": True,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": runtime_versions,
    }

    verified_audio: dict[str, tuple[Path, str]] = {}
    for index, row in enumerate(rows):
        if row["valid"] is True:
            verified_audio[row["sample_id"]] = verify_observation_audio(row, audio_base_dir, index)

    try:
        classifier = _load_classifier(model_dir, device)
    except Exception as error:
        raise RuntimeError(f"failed to initialize SpeechBrain ECAPA on {device}: {error}") from error
    reference_by_id = {reference_id: paths[0] for reference_id, paths in speaker_references.items()}
    reference_embeddings: dict[str, list[float]] = {}
    results: list[dict[str, Any]] = []
    for row in rows:
        if row["valid"] is not True:
            continue
        sample_id = row["sample_id"]
        reference_ids = assignment_plan["assignments"][(row["prompt_id"], row["seed"])]
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "audio_sha256": verified_audio[sample_id][1],
            "status": "complete",
            "reference_ids": reference_ids,
        }
        try:
            selected_embeddings: list[dict[str, Any]] = []
            for reference_id in reference_ids:
                if reference_id not in reference_embeddings:
                    reference_embeddings[reference_id] = _encode_file(classifier, reference_by_id[reference_id])
                selected_embeddings.append(
                    {"reference_id": reference_id, "embedding": reference_embeddings[reference_id]}
                )
            result["values"] = {
                "reference_speaker_embeddings": selected_embeddings,
                "speaker_embedding": _encode_file(classifier, verified_audio[sample_id][0]),
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

    if build_extractor_identity(
        kind="speaker_encoder",
        name=SPEECHBRAIN_ECAPA_NAME,
        revision=model_revision,
        artifacts=artifacts,
    ) != extractor:
        raise ValueError("SpeechBrain model or runner artifacts changed during extraction")
    if build_speaker_reference_catalog(catalog_id=catalog_id, references=speaker_references) != reference_catalog:
        raise ValueError("speaker reference catalog changed during extraction")
    for index, row in enumerate(rows):
        if row["valid"] is True and verify_observation_audio(row, audio_base_dir, index) != verified_audio[row["sample_id"]]:
            raise ValueError(f"observation audio changed during extraction: {row['sample_id']}")

    document = {
        "schema_version": EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
        "source_observations_sha256": source_sha256,
        "extractor": extractor,
        "reference_catalog": reference_catalog,
        "reference_aggregation": REFERENCE_AGGREGATION,
        "reference_assignment_plan_sha256": assignment_plan["sha256"],
        "execution": execution,
        "results": results,
    }
    return {**document, "execution_receipt_sha256": _result_document_sha256(document)}


def validate_execution_receipt(extraction: dict[str, Any]) -> dict[str, Any]:
    execution = extraction.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "backend",
        "device",
        "trusted_model_checkpoints",
        "python_version",
        "platform",
        "package_versions",
    }:
        raise ValueError("extractor execution must contain exactly backend, device, Python, platform, and packages")
    if execution.get("backend") != SPEECHBRAIN_ECAPA_BACKEND:
        raise ValueError("unsupported extractor execution backend")
    device = execution.get("device")
    if not isinstance(device, str) or not DEVICE_RE.fullmatch(device):
        raise ValueError("extractor execution device is invalid")
    if execution.get("trusted_model_checkpoints") is not True:
        raise ValueError("extractor execution must explicitly acknowledge trusted model checkpoints")
    for field in ("python_version", "platform"):
        if not isinstance(execution.get(field), str) or not execution[field].strip():
            raise ValueError(f"extractor execution {field} must be non-empty")
    versions = execution.get("package_versions")
    if not isinstance(versions, dict) or set(versions) != {"speechbrain", "torch", "torchaudio"} or any(
        not isinstance(value, str) or not value.strip() for value in versions.values()
    ):
        raise ValueError("extractor execution package versions are invalid")
    digest = extraction.get("execution_receipt_sha256")
    if digest != _result_document_sha256(extraction):
        raise ValueError("execution_receipt_sha256 does not match the extractor result document")
    return execution
