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
from .lineage import KINDS, fingerprint_artifact
from .observations import validate_objective_observations
from .speaker_reference_plans import (
    speaker_reference_assignment_sha256,
    validate_speaker_reference_assignment_plan,
)
from .speaker_references import (
    REFERENCE_AGGREGATION,
    canonical_sha256,
    reference_set_sha256,
    speaker_measurement_sha256,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MUTABLE_REVISIONS = {"latest", "main", "master", "head", "unknown", "unversioned"}
EXTRACTION_SCHEMA_VERSION = "1.1.0"
MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION = "1.2.0"
PLANNED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION = "1.3.0"
EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION = "1.4.0"
EXECUTED_ASR_EXTRACTION_SCHEMA_VERSION = "1.5.0"
EXTRACTION_SCHEMA_VERSIONS = {
    EXTRACTION_SCHEMA_VERSION,
    MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
    PLANNED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
    EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
    EXECUTED_ASR_EXTRACTION_SCHEMA_VERSION,
}
EXTRACTOR_FIELDS = {
    "asr": {"hypothesis_text"},
    "speaker_encoder": {"reference_speaker_embedding", "speaker_embedding"},
    "audio_probe": {"sample_rate_hz", "silence_fraction", "clipping_fraction"},
}
BUILTIN_AUDIO_PROBE_NAME = "instavar_voice_lab.audio_probe"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_revision(value: Any) -> str:
    revision = value.strip() if isinstance(value, str) else ""
    if not revision or revision.casefold() in MUTABLE_REVISIONS:
        raise ValueError("extractor revision must be non-empty and immutable")
    return revision


def _extractor_artifact_records(
    artifacts: dict[str, tuple[Path, str]],
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("extractor artifacts must contain at least one artifact")
    records: list[dict[str, Any]] = []
    for role, declaration in sorted(artifacts.items()):
        if not isinstance(role, str) or not IDENTIFIER_RE.fullmatch(role):
            raise ValueError("extractor artifact roles must be stable lowercase identifiers")
        if (
            not isinstance(declaration, tuple)
            or len(declaration) != 2
            or not isinstance(declaration[0], Path)
            or declaration[1] not in KINDS
        ):
            raise ValueError(f"extractor artifact {role} must declare a Path and file or tree kind")
        record = fingerprint_artifact(declaration[0], role=role, kind=declaration[1])
        if record["bytes"] <= 0:
            raise ValueError(f"extractor artifact must not be empty: {role}")
        records.append(record)
    return records, _canonical_sha256(records)


def build_extractor_identity(
    *,
    kind: str,
    name: str,
    revision: str,
    artifacts: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    if kind not in EXTRACTOR_FIELDS:
        raise ValueError("unsupported extractor kind")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("extractor name must be non-empty")
    records, artifact_set_sha256 = _extractor_artifact_records(artifacts)
    return {
        "kind": kind,
        "name": name.strip(),
        "revision": _immutable_revision(revision),
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": records,
    }


def _reference_file(path: Path, label: str) -> dict[str, Any]:
    record = fingerprint_artifact(path, role=label, kind="file")
    if record["bytes"] <= 0:
        raise ValueError(f"speaker reference {label} must not be empty")
    return {"sha256": record["sha256"], "bytes": record["bytes"]}


def build_speaker_reference_binding(
    *,
    reference_id: str,
    audio_path: Path,
    transcript_path: Path,
) -> dict[str, Any]:
    if not isinstance(reference_id, str) or not IDENTIFIER_RE.fullmatch(reference_id):
        raise ValueError("speaker reference id must be a stable lowercase identifier")
    return {
        "reference_id": reference_id,
        "audio": _reference_file(audio_path, "reference_audio"),
        "transcript": _reference_file(transcript_path, "reference_transcript"),
    }


def build_speaker_reference_catalog(
    *,
    catalog_id: str,
    references: dict[str, tuple[Path, Path]],
) -> dict[str, Any]:
    if not isinstance(catalog_id, str) or not IDENTIFIER_RE.fullmatch(catalog_id):
        raise ValueError("speaker reference catalog id must be a stable lowercase identifier")
    if not isinstance(references, dict) or not references:
        raise ValueError("speaker reference catalog must contain at least one reference")
    records: list[dict[str, Any]] = []
    for reference_id, paths in sorted(references.items()):
        if not isinstance(reference_id, str) or not IDENTIFIER_RE.fullmatch(reference_id):
            raise ValueError("speaker reference ids must be stable lowercase identifiers")
        if (
            not isinstance(paths, tuple)
            or len(paths) != 2
            or not isinstance(paths[0], Path)
            or not isinstance(paths[1], Path)
        ):
            raise ValueError(f"speaker reference {reference_id} must declare audio and transcript paths")
        binding = build_speaker_reference_binding(
            reference_id=reference_id,
            audio_path=paths[0],
            transcript_path=paths[1],
        )
        records.append(binding)
    catalog_payload = {"catalog_id": catalog_id, "references": records}
    return {**catalog_payload, "catalog_sha256": canonical_sha256(catalog_payload)}


def _selected_reference_records(catalog: dict[str, Any], reference_ids: list[str]) -> list[dict[str, str]]:
    by_id = {item["reference_id"]: item for item in catalog["references"]}
    records: list[dict[str, str]] = []
    for reference_id in reference_ids:
        raw = by_id[reference_id]
        records.append(
            {
                "reference_id": reference_id,
                "reference_audio_sha256": raw["audio"]["sha256"],
                "reference_transcript_sha256": raw["transcript"]["sha256"],
            }
        )
    return records


def _builtin_audio_probe_artifacts() -> dict[str, tuple[Path, str]]:
    return {"implementation": (Path(__file__).with_name("audio_probe.py"), "file")}


def verify_observation_audio(row: dict[str, Any], base_dir: Path, index: int) -> tuple[Path, str]:
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


def _validate_values(
    kind: str,
    values: dict[str, Any],
    sample_id: str,
    *,
    reference_ids: list[str] | None = None,
) -> None:
    if kind == "asr":
        if not isinstance(values["hypothesis_text"], str):
            raise ValueError(f"ASR hypothesis_text must be a string for sample_id: {sample_id}")
        return
    if kind == "speaker_encoder":
        if reference_ids is not None:
            references = values["reference_speaker_embeddings"]
            candidate = values["speaker_embedding"]
            if not isinstance(references, list) or not isinstance(candidate, list):
                raise ValueError(f"multi-reference speaker embeddings must be arrays for sample_id: {sample_id}")
            observed_ids: list[str] = []
            for index, item in enumerate(references):
                if not isinstance(item, dict) or set(item) != {"reference_id", "embedding"}:
                    raise ValueError(
                        f"reference_speaker_embeddings[{index}] must contain exactly reference_id and embedding "
                        f"for sample_id: {sample_id}"
                    )
                observed_ids.append(item["reference_id"])
                embedding = item["embedding"]
                if not isinstance(embedding, list) or not embedding or len(embedding) != len(candidate):
                    raise ValueError(
                        f"speaker embeddings must be non-empty equal-length arrays for sample_id: {sample_id}"
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                    for value in embedding
                ):
                    raise ValueError(f"speaker embeddings must contain finite numbers for sample_id: {sample_id}")
            if observed_ids != reference_ids:
                raise ValueError(
                    f"reference_speaker_embeddings must exactly match reference_ids for sample_id: {sample_id}"
                )
            if not candidate or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in candidate
            ):
                raise ValueError(f"speaker embeddings must contain finite numbers for sample_id: {sample_id}")
            return
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
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
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
    extractor = build_extractor_identity(
        kind="audio_probe",
        name=BUILTIN_AUDIO_PROBE_NAME,
        revision=extractor_revision,
        artifacts=_builtin_audio_probe_artifacts(),
    )

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["valid"] is not True:
            continue
        path, audio_sha = verify_observation_audio(row, audio_base_dir, index)
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
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "source_observations_sha256": observation_document_sha256(rows),
        "extractor": extractor,
        "results": results,
    }


def apply_extractor_results(
    observations: Any,
    extraction: Any,
    *,
    audio_base_dir: Path,
    extractor_artifacts: dict[str, tuple[Path, str]] | None = None,
    reference_audio_path: Path | None = None,
    reference_transcript_path: Path | None = None,
    speaker_references: dict[str, tuple[Path, Path]] | None = None,
    speaker_reference_plan: dict[str, Any] | None = None,
    generation_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = _validated_source_rows(observations)
    if not isinstance(extraction, dict) or extraction.get("schema_version") not in EXTRACTION_SCHEMA_VERSIONS:
        raise ValueError(
            f"extractor results schema_version must be one of: {', '.join(sorted(EXTRACTION_SCHEMA_VERSIONS))}"
        )
    extraction_version = extraction["schema_version"]
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
    if extraction_version == EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION and kind != "speaker_encoder":
        raise ValueError("schema 1.4 extractor results are reserved for executed speaker-encoder evidence")
    if extraction_version == EXECUTED_ASR_EXTRACTION_SCHEMA_VERSION and kind != "asr":
        raise ValueError("schema 1.5 extractor results are reserved for executed ASR evidence")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("extractor name must be non-empty")
    revision = _immutable_revision(revision)
    if extractor_artifacts is None:
        if kind == "audio_probe" and name.strip() == BUILTIN_AUDIO_PROBE_NAME:
            extractor_artifacts = _builtin_audio_probe_artifacts()
        else:
            raise ValueError("external extractor artifacts are required for live identity verification")
    expected_extractor = build_extractor_identity(
        kind=kind,
        name=name,
        revision=revision,
        artifacts=extractor_artifacts,
    )
    if extractor != expected_extractor:
        raise ValueError("extractor identity does not match the live artifact set")

    reference: dict[str, Any] | None = None
    reference_catalog: dict[str, Any] | None = None
    reference_assignment_plan: dict[str, Any] | None = None
    if kind == "speaker_encoder":
        if extraction_version == EXTRACTION_SCHEMA_VERSION:
            raw_reference = extraction.get("reference")
            if not isinstance(raw_reference, dict):
                raise ValueError("speaker extractor results must bind a speaker reference")
            if reference_audio_path is None or reference_transcript_path is None:
                raise ValueError("speaker reference audio and transcript paths are required")
            if speaker_references is not None:
                raise ValueError("legacy speaker results cannot declare a multi-reference catalog")
            if speaker_reference_plan is not None:
                raise ValueError("speaker reference assignment plans require schema 1.3 speaker results")
            if generation_plan is not None:
                raise ValueError("speaker generation plans require schema 1.3 speaker results")
            reference_id = raw_reference.get("reference_id")
            reference = build_speaker_reference_binding(
                reference_id=reference_id,
                audio_path=reference_audio_path,
                transcript_path=reference_transcript_path,
            )
            if raw_reference != reference:
                raise ValueError("speaker reference identity does not match the live audio and transcript")
        else:
            if reference_audio_path is not None or reference_transcript_path is not None:
                raise ValueError("multi-reference speaker results require --speaker-reference declarations")
            raw_catalog = extraction.get("reference_catalog")
            if not isinstance(raw_catalog, dict):
                raise ValueError("multi-reference speaker results must bind a reference catalog")
            if speaker_references is None:
                raise ValueError("live speaker reference declarations are required")
            reference_catalog = build_speaker_reference_catalog(
                catalog_id=raw_catalog.get("catalog_id"),
                references=speaker_references,
            )
            if raw_catalog != reference_catalog:
                raise ValueError("speaker reference catalog does not match the live audio and transcripts")
            if extraction.get("reference_aggregation") != REFERENCE_AGGREGATION:
                raise ValueError(f"reference_aggregation must equal {REFERENCE_AGGREGATION}")
            if extraction_version in {
                PLANNED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
                EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
            }:
                if speaker_reference_plan is None:
                    raise ValueError("planned speaker results require a live speaker reference assignment plan")
                if generation_plan is None:
                    raise ValueError("planned speaker results require the live generation plan")
                reference_assignment_plan = validate_speaker_reference_assignment_plan(
                    speaker_reference_plan,
                    generation_plan=generation_plan,
                    reference_catalog=reference_catalog,
                )
                if extraction.get("reference_assignment_plan_sha256") != reference_assignment_plan["sha256"]:
                    raise ValueError("speaker extractor results do not match the reference assignment plan")
            elif speaker_reference_plan is not None:
                raise ValueError("speaker reference assignment plans require schema 1.3 speaker results")
            elif generation_plan is not None:
                raise ValueError("speaker generation plans require schema 1.3 speaker results")
    elif any(
        field in extraction
        for field in (
            "reference",
            "reference_catalog",
            "reference_aggregation",
            "reference_assignment_plan_sha256",
        )
    ):
        raise ValueError("speaker reference binding is only valid for speaker_encoder results")
    elif (
        reference_audio_path is not None
        or reference_transcript_path is not None
        or speaker_references is not None
        or speaker_reference_plan is not None
        or generation_plan is not None
    ):
        raise ValueError("speaker reference paths are only valid for speaker_encoder results")
    expected_document_fields = {
        "schema_version",
        "source_observations_sha256",
        "extractor",
        "results",
    }
    if kind == "speaker_encoder" and extraction_version == EXTRACTION_SCHEMA_VERSION:
        expected_document_fields.add("reference")
    elif kind == "speaker_encoder":
        expected_document_fields.update({"reference_catalog", "reference_aggregation"})
        if extraction_version in {
            PLANNED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
            EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
        }:
            expected_document_fields.add("reference_assignment_plan_sha256")
        if extraction_version == EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION:
            from .speechbrain_ecapa import validate_execution_receipt

            expected_document_fields.update({"execution", "execution_receipt_sha256"})
            validate_execution_receipt(extraction)
    elif kind == "asr" and extraction_version == EXECUTED_ASR_EXTRACTION_SCHEMA_VERSION:
        from .faster_whisper import validate_execution_receipt

        expected_document_fields.update({"execution", "execution_receipt_sha256"})
        validate_execution_receipt(extraction)
    if set(extraction) != expected_document_fields:
        raise ValueError(f"extractor result document must contain exactly {sorted(expected_document_fields)}")

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
        _, live_sha = verify_observation_audio(source_row, audio_base_dir, index)
        result = result_by_id[sample_id]
        if result.get("audio_sha256") != live_sha:
            raise ValueError(f"extractor result audio_sha256 mismatch for sample_id: {sample_id}")
        status = result.get("status")
        if status not in {"complete", "failed"}:
            raise ValueError(f"extractor result status must be complete or failed for sample_id: {sample_id}")
        expected_result_fields = {"sample_id", "audio_sha256", "status"}
        reference_ids: list[str] | None = None
        selected_references: list[dict[str, str]] | None = None
        if kind == "speaker_encoder" and extraction_version in {
            MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
            PLANNED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
            EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
        }:
            expected_result_fields.add("reference_ids")
            raw_reference_ids = result.get("reference_ids")
            if not isinstance(raw_reference_ids, list) or not raw_reference_ids:
                raise ValueError(f"speaker result reference_ids must be a non-empty array for sample_id: {sample_id}")
            if any(
                not isinstance(reference_id, str) or not IDENTIFIER_RE.fullmatch(reference_id)
                for reference_id in raw_reference_ids
            ):
                raise ValueError(f"speaker result reference_ids must be stable lowercase identifiers: {sample_id}")
            reference_ids = list(raw_reference_ids)
            if reference_ids != sorted(set(reference_ids)):
                raise ValueError(f"speaker result reference_ids must be unique and sorted for sample_id: {sample_id}")
            assert reference_catalog is not None
            known_ids = {item["reference_id"] for item in reference_catalog["references"]}
            unknown = sorted(set(reference_ids) - known_ids)
            if unknown:
                raise ValueError(f"speaker result references are absent from the live catalog: {unknown}")
            if reference_assignment_plan is not None:
                assignment_key = (source_row.get("prompt_id"), source_row.get("seed"))
                planned_reference_ids = reference_assignment_plan["assignments"].get(assignment_key)
                if planned_reference_ids is None:
                    raise ValueError(
                        f"speaker reference assignment plan has no assignment for sample_id: {sample_id}"
                    )
                if reference_ids != planned_reference_ids:
                    raise ValueError(
                        f"speaker result reference_ids do not match the frozen assignment for sample_id: {sample_id}"
                    )
            selected_references = _selected_reference_records(reference_catalog, reference_ids)
        if status == "complete":
            expected_result_fields.add("values")
        else:
            expected_result_fields.update({"error_type", "error"})
        if set(result) != expected_result_fields:
            raise ValueError(
                f"extractor result must contain exactly {sorted(expected_result_fields)} for sample_id: {sample_id}"
            )
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
            "revision": revision,
            "extractor_artifact_set_sha256": expected_extractor["artifact_set_sha256"],
            "input_audio_sha256": live_sha,
        }
        if extraction_version in {
            EXECUTED_MULTI_REFERENCE_EXTRACTION_SCHEMA_VERSION,
            EXECUTED_ASR_EXTRACTION_SCHEMA_VERSION,
        }:
            provenance.update(
                {
                    "extractor_execution": extraction["execution"],
                    "extractor_execution_receipt_sha256": extraction["execution_receipt_sha256"],
                }
            )
        if reference is not None:
            provenance.update(
                {
                    "reference_id": reference["reference_id"],
                    "reference_audio_sha256": reference["audio"]["sha256"],
                    "reference_transcript_sha256": reference["transcript"]["sha256"],
                }
            )
        elif selected_references is not None:
            provenance.update(
                {
                    "reference_aggregation": REFERENCE_AGGREGATION,
                    "reference_catalog_sha256": reference_catalog["catalog_sha256"],
                    "reference_set_sha256": reference_set_sha256(
                        selected_references,
                        aggregation=REFERENCE_AGGREGATION,
                    ),
                    "references": selected_references,
                }
            )
            if reference_assignment_plan is not None:
                provenance.update(
                    {
                        "reference_assignment_plan_sha256": reference_assignment_plan["sha256"],
                        "reference_assignment_sha256": speaker_reference_assignment_sha256(
                            assignment_plan_sha256=reference_assignment_plan["sha256"],
                            prompt_id=source_row["prompt_id"],
                            seed=source_row["seed"],
                            reference_ids=reference_ids,
                        ),
                    }
                )
        if status == "complete":
            values = result.get("values")
            required_fields = (
                {"reference_speaker_embeddings", "speaker_embedding"}
                if kind == "speaker_encoder" and reference_ids is not None
                else EXTRACTOR_FIELDS[kind]
            )
            if not isinstance(values, dict) or set(values) != required_fields:
                raise ValueError(
                    f"complete {kind} result must contain exactly {sorted(required_fields)} for sample_id: {sample_id}"
                )
            _validate_values(kind, values, sample_id, reference_ids=reference_ids)
            conflicts = sorted(required_fields & target.keys())
            if conflicts:
                raise ValueError(f"extractor result would overwrite fields for sample_id {sample_id}: {conflicts}")
            target.update(values)
            if kind == "speaker_encoder":
                provenance["speaker_measurement_sha256"] = speaker_measurement_sha256(target, provenance)
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
