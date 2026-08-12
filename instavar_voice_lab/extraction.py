from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from copy import deepcopy
from pathlib import Path
from typing import Any

from .audio_probe import probe_wav
from .observations import validate_objective_observations


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_REVISIONS = {"latest", "main", "master", "head", "unknown", "unversioned"}
EXTRACTOR_FIELDS = {
    "asr": {"hypothesis_text"},
    "speaker_encoder": {"reference_speaker_embedding", "speaker_embedding"},
    "audio_probe": {"sample_rate_hz", "silence_fraction", "clipping_fraction"},
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_file(row: dict[str, Any], base_dir: Path, index: int) -> tuple[Path, str]:
    raw_path = row.get("audio_path")
    expected_sha = row.get("audio_sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"observation {index} audio_path must be a non-empty string")
    if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError(f"observation {index} audio_sha256 must be a lowercase SHA-256")
    declared = Path(raw_path)
    path = declared if declared.is_absolute() else base_dir / declared
    if declared.is_absolute() is False:
        resolved_root = base_dir.resolve()
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"observation {index} audio_path escapes the audio base directory") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"observation {index} audio_path must be a regular non-symlink file")
    if path.stat().st_size <= 0:
        raise ValueError(f"observation {index} audio_path must not be empty")
    actual_sha = _file_sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"observation {index} audio_sha256 does not match the live audio file")
    return path, actual_sha


def _validated_source_rows(observations: Any) -> list[dict[str, Any]]:
    errors = validate_objective_observations(
        observations,
        require_version=True,
        require_seed=True,
        require_runtime=True,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return observations


def observation_document_sha256(observations: Any) -> str:
    return _canonical_sha256(_validated_source_rows(observations))


def _validate_values(kind: str, values: dict[str, Any], sample_id: str) -> None:
    if kind == "asr":
        if not isinstance(values["hypothesis_text"], str):
            raise ValueError(f"ASR hypothesis_text must be a string for sample_id: {sample_id}")
        return
    if kind == "speaker_encoder":
        reference = values["reference_speaker_embedding"]
        candidate = values["speaker_embedding"]
        if (
            not isinstance(reference, list)
            or not isinstance(candidate, list)
            or not reference
            or len(reference) != len(candidate)
        ):
            raise ValueError(f"speaker embeddings must be non-empty equal-length arrays for sample_id: {sample_id}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in [*reference, *candidate]
        ):
            raise ValueError(f"speaker embeddings must contain finite numbers for sample_id: {sample_id}")
        return
    sample_rate = values["sample_rate_hz"]
    silence = values["silence_fraction"]
    clipping = values["clipping_fraction"]
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"sample_rate_hz must be a positive integer for sample_id: {sample_id}")
    for name, value in (("silence_fraction", silence), ("clipping_fraction", clipping)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"{name} must be a finite number between zero and one for sample_id: {sample_id}")


def build_audio_probe_results(
    observations: Any,
    *,
    audio_base_dir: Path,
    extractor_revision: str,
) -> dict[str, Any]:
    rows = _validated_source_rows(observations)
    revision = extractor_revision.strip() if isinstance(extractor_revision, str) else ""
    if not revision or revision.casefold() in MUTABLE_REVISIONS:
        raise ValueError("extractor revision must be non-empty and immutable")

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["valid"] is not True:
            continue
        path, audio_sha = _audio_file(row, audio_base_dir, index)
        result: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "audio_sha256": audio_sha,
            "status": "complete",
        }
        try:
            probe = probe_wav(path)
            result["values"] = {
                "sample_rate_hz": probe["sample_rate_hz"],
                "silence_fraction": probe["silence_fraction"],
                "clipping_fraction": probe["clipping_fraction"],
            }
        except (OSError, ValueError, EOFError, wave.Error) as error:
            result.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        results.append(result)
    return {
        "schema_version": "1.0.0",
        "source_observations_sha256": observation_document_sha256(rows),
        "extractor": {
            "kind": "audio_probe",
            "name": "instavar_voice_lab.audio_probe",
            "revision": revision,
        },
        "results": results,
    }


def apply_extractor_results(
    observations: Any,
    extraction: Any,
    *,
    audio_base_dir: Path,
) -> list[dict[str, Any]]:
    rows = _validated_source_rows(observations)
    if not isinstance(extraction, dict) or extraction.get("schema_version") != "1.0.0":
        raise ValueError("extractor results must be a version 1.0.0 object")
    source_sha = extraction.get("source_observations_sha256")
    if source_sha != observation_document_sha256(rows):
        raise ValueError("extractor results do not match the source observation document")
    extractor = extraction.get("extractor")
    if not isinstance(extractor, dict):
        raise ValueError("extractor results must declare extractor provenance")
    kind = extractor.get("kind")
    name = extractor.get("name")
    revision = extractor.get("revision")
    if kind not in EXTRACTOR_FIELDS:
        raise ValueError("unsupported extractor kind")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("extractor name must be non-empty")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision.strip().casefold() in MUTABLE_REVISIONS
    ):
        raise ValueError("extractor revision must be non-empty and immutable")

    raw_results = extraction.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("extractor results must contain a results array")
    result_by_id: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(raw_results):
        if not isinstance(result, dict):
            raise ValueError(f"extractor result {index} must be an object")
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"extractor result {index} sample_id must be non-empty")
        if sample_id in result_by_id:
            raise ValueError(f"duplicate extractor result for sample_id: {sample_id}")
        result_by_id[sample_id] = result

    valid_rows = {row["sample_id"]: (index, row) for index, row in enumerate(rows) if row["valid"] is True}
    if set(result_by_id) != set(valid_rows):
        missing = sorted(set(valid_rows) - set(result_by_id))
        unexpected = sorted(set(result_by_id) - set(valid_rows))
        raise ValueError(
            f"extractor results must exactly cover valid observations; missing={missing}; unexpected={unexpected}"
        )

    output = deepcopy(rows)
    for sample_id, (index, source_row) in valid_rows.items():
        _, live_sha = _audio_file(source_row, audio_base_dir, index)
        result = result_by_id[sample_id]
        if result.get("audio_sha256") != live_sha:
            raise ValueError(f"extractor result audio_sha256 mismatch for sample_id: {sample_id}")
        status = result.get("status")
        if status not in {"complete", "failed"}:
            raise ValueError(f"extractor result status must be complete or failed for sample_id: {sample_id}")
        target = output[index]
        evidence = target.setdefault("evidence", {})
        if kind in evidence:
            raise ValueError(f"observation already contains evidence.{kind} for sample_id: {sample_id}")
        history = target.setdefault("augmentation_history", [])
        if not isinstance(history, list):
            raise ValueError(f"observation augmentation_history must be an array for sample_id: {sample_id}")
        if any(isinstance(entry, dict) and entry.get("kind") == kind for entry in history):
            raise ValueError(f"observation already has {kind} augmentation history for sample_id: {sample_id}")
        existing_failures = target.get("extractor_failures", {})
        if not isinstance(existing_failures, dict):
            raise ValueError(f"observation extractor_failures must be an object for sample_id: {sample_id}")
        if kind in existing_failures:
            raise ValueError(f"observation already has an {kind} failure for sample_id: {sample_id}")
        provenance = {
            "extractor": name.strip(),
            "revision": revision.strip(),
            "input_audio_sha256": live_sha,
        }
        if status == "complete":
            values = result.get("values")
            required_fields = EXTRACTOR_FIELDS[kind]
            if not isinstance(values, dict) or set(values) != required_fields:
                raise ValueError(
                    f"complete {kind} result must contain exactly {sorted(required_fields)} for sample_id: {sample_id}"
                )
            _validate_values(kind, values, sample_id)
            conflicts = sorted(required_fields & target.keys())
            if conflicts:
                raise ValueError(f"extractor result would overwrite fields for sample_id {sample_id}: {conflicts}")
            target.update(values)
            evidence[kind] = provenance
        else:
            if "values" in result:
                raise ValueError(f"failed extractor result must not contain values for sample_id: {sample_id}")
            error_type = result.get("error_type")
            error = result.get("error")
            if (
                not isinstance(error_type, str)
                or not error_type.strip()
                or not isinstance(error, str)
                or not error.strip()
            ):
                raise ValueError(f"failed extractor result must record error_type and error for sample_id: {sample_id}")
            failures = target.setdefault("extractor_failures", {})
            failures[kind] = {**provenance, "error_type": error_type.strip(), "error": error.strip()}
        history.append({"kind": kind, "status": status, **provenance})
    return output
