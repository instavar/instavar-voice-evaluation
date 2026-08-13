from __future__ import annotations

import hashlib
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any

from .extraction import (
    BUILTIN_PROSODY_PROXY_NAME,
    build_extractor_identity,
    builtin_prosody_proxy_artifacts,
)
from .prosody_probe import probe_prosody_proxy


HISTORICAL_PROSODY_MANIFEST_SCHEMA_VERSION = "instavar_voice_historical_prosody_manifest/v1"
HISTORICAL_PROSODY_REPORT_SCHEMA_VERSION = "instavar_voice_historical_prosody_report/v1"
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAMPLE_FIELDS = {
    "sample_id",
    "candidate_id",
    "prompt_id",
    "audio_path",
    "audio_sha256",
    "requested_text",
    "seed",
    "runtime_id",
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


def _historical_prosody_artifacts() -> dict[str, tuple[Path, str]]:
    return {
        **builtin_prosody_proxy_artifacts(),
        "historical_batch_runner": (Path(__file__), "file"),
    }


def _content_addressed_extractor() -> dict[str, Any]:
    identity = build_extractor_identity(
        kind="prosody_proxy",
        name=BUILTIN_PROSODY_PROXY_NAME,
        revision="content-addressed",
        artifacts=_historical_prosody_artifacts(),
    )
    return {
        **identity,
        "revision": f"artifact-set-sha256:{identity['artifact_set_sha256']}",
        "revision_basis": "artifact_set_sha256",
    }


def _validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    expected_fields = {"schema_version", "batch_id", "purpose", "samples"}
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError(f"historical prosody manifest must contain exactly {sorted(expected_fields)}")
    if manifest["schema_version"] != HISTORICAL_PROSODY_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"historical prosody manifest schema_version must equal {HISTORICAL_PROSODY_MANIFEST_SCHEMA_VERSION}"
        )
    if not isinstance(manifest["batch_id"], str) or not IDENTIFIER_RE.fullmatch(manifest["batch_id"]):
        raise ValueError("historical prosody manifest batch_id must be a stable lowercase identifier")
    if manifest["purpose"] != "historical_unmatched_triage":
        raise ValueError("historical prosody manifest purpose must equal historical_unmatched_triage")
    samples = manifest["samples"]
    if not isinstance(samples, list) or not samples or len(samples) > 10000:
        raise ValueError("historical prosody manifest samples must contain between 1 and 10000 rows")

    seen: set[str] = set()
    for index, row in enumerate(samples):
        if not isinstance(row, dict) or set(row) != SAMPLE_FIELDS:
            raise ValueError(f"historical prosody sample {index} must contain exactly {sorted(SAMPLE_FIELDS)}")
        for field in ("sample_id", "candidate_id", "prompt_id"):
            if not isinstance(row[field], str) or not IDENTIFIER_RE.fullmatch(row[field]):
                raise ValueError(f"historical prosody sample {index} {field} must be a stable lowercase identifier")
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate historical prosody sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        raw_path = row["audio_path"]
        declared_path = Path(raw_path) if isinstance(raw_path, str) else Path()
        if (
            not isinstance(raw_path, str)
            or not raw_path.strip()
            or declared_path.is_absolute()
            or ".." in declared_path.parts
            or "\\" in raw_path
        ):
            raise ValueError(f"historical prosody sample {index} audio_path must be a contained relative path")
        if not isinstance(row["audio_sha256"], str) or not SHA256_RE.fullmatch(row["audio_sha256"]):
            raise ValueError(f"historical prosody sample {index} audio_sha256 must be a lowercase SHA-256")
        requested_text = row["requested_text"]
        if requested_text is not None and (not isinstance(requested_text, str) or not requested_text.strip()):
            raise ValueError(f"historical prosody sample {index} requested_text must be null or non-empty")
        seed = row["seed"]
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
            raise ValueError(f"historical prosody sample {index} seed must be null or a non-negative integer")
        runtime_id = row["runtime_id"]
        if runtime_id is not None and (not isinstance(runtime_id, str) or not IDENTIFIER_RE.fullmatch(runtime_id)):
            raise ValueError(f"historical prosody sample {index} runtime_id must be null or a stable identifier")
    return samples


def _resolve_audio(base_dir: Path, row: dict[str, Any], index: int) -> Path:
    if base_dir.is_symlink() or not base_dir.is_dir():
        raise ValueError("audio base directory must be a regular non-symlink directory")
    root = base_dir.resolve()
    relative = Path(row["audio_path"])
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"historical prosody sample {index} audio path must not traverse symlinks")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"historical prosody sample {index} audio path escapes the audio base directory") from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"historical prosody sample {index} audio must be a non-empty regular file")
    if _file_sha256(resolved) != row["audio_sha256"]:
        raise ValueError(f"historical prosody sample {index} audio_sha256 does not match the live audio file")
    return resolved


def audit_historical_prosody_batch(
    manifest: Any,
    *,
    audio_base_dir: Path,
) -> dict[str, Any]:
    samples = _validate_manifest(manifest)
    extractor = _content_addressed_extractor()
    results: list[dict[str, Any]] = []
    resolved_audio: list[tuple[Path, str]] = []
    for index, row in enumerate(samples):
        path = _resolve_audio(audio_base_dir, row, index)
        resolved_audio.append((path, row["audio_sha256"]))
        result: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "audio_path": row["audio_path"],
            "input_audio_sha256": row["audio_sha256"],
            "requested_text": row["requested_text"],
            "seed": row["seed"],
            "runtime_id": row["runtime_id"],
            "status": "complete",
        }
        try:
            probe = probe_prosody_proxy(path)
            if probe["status"] != "complete":
                raise ValueError("prosody proxy found fewer than five active frames")
            result["probe"] = probe
        except (OSError, ValueError, EOFError, wave.Error) as error:
            result.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        if _file_sha256(path) != row["audio_sha256"]:
            raise ValueError(f"historical prosody sample {index} audio changed during analysis")
        results.append(result)

    final_extractor = _content_addressed_extractor()
    if final_extractor != extractor:
        raise ValueError("prosody extractor artifacts changed during historical batch analysis")
    for index, (path, expected_sha) in enumerate(resolved_audio):
        if _file_sha256(path) != expected_sha:
            raise ValueError(f"historical prosody sample {index} audio changed before report publication")

    complete = [result for result in results if result["status"] == "complete"]
    return {
        "schema_version": HISTORICAL_PROSODY_REPORT_SCHEMA_VERSION,
        "status": "analysis_complete" if len(complete) == len(results) else "analysis_complete_with_failures",
        "batch_id": manifest["batch_id"],
        "purpose": manifest["purpose"],
        "source_manifest_sha256": _canonical_sha256(manifest),
        "extractor": extractor,
        "execution": {
            "python_version": sys.version.split()[0],
            "implementation": "instavar_voice_lab.historical_prosody",
        },
        "coverage": {
            "sample_count": len(results),
            "complete_count": len(complete),
            "failed_count": len(results) - len(complete),
            "seed_not_recorded_count": sum(result["seed"] is None for result in results),
            "runtime_not_recorded_count": sum(result["runtime_id"] is None for result in results),
            "requested_text_not_recorded_count": sum(result["requested_text"] is None for result in results),
            "long_form_eligible_count": sum(
                result["status"] == "complete" and result["probe"]["metrics"]["eligible_for_long_form"]
                for result in results
            ),
        },
        "results": results,
        "quality_direction_established": False,
        "winner": None,
        "eligible_for_matched_adaptation_comparison": False,
        "proves_adaptation_benefit": False,
        "evidence_boundary": (
            "This historical audit binds explicit manifest metadata, live audio bytes, and exact proxy source "
            "artifacts while preserving unknown seed, runtime, or text fields as null. It reports per-file signal "
            "features and failures only. Without a frozen generation plan and matched baseline and adapted rows, it "
            "does not support candidate ranking, quality direction, adaptation benefit, cadence, accent, naturalness, "
            "speaker identity, preference, or long-form claims."
        ),
    }
