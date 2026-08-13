from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .extraction import verify_observation_audio
from .observations import validate_objective_observations

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_REVISIONS = {"latest", "main", "master", "head", "unknown", "unversioned"}
ATTEMPT_RECEIPT_SCHEMA_VERSION = "1.0.0"
RUNTIME_METRIC_FIELDS = ("generation_seconds", "audio_duration_seconds", "peak_memory_bytes")
DERIVED_OBSERVATION_FIELDS = {
    "hypothesis_text",
    "reference_speaker_embedding",
    "reference_speaker_embeddings",
    "speaker_embedding",
    "sample_rate_hz",
    "silence_fraction",
    "clipping_fraction",
    "augmentation_history",
    "extractor_failures",
}
RUNTIME_BINDING_FIELDS = {
    "generation_plan_sha256",
    "planned_sample_sha256",
    "source_generation_observation_sha256",
    "output_audio_sha256",
    "attempt_sha256",
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _immutable_revision(value: Any) -> str:
    revision = value.strip() if isinstance(value, str) else ""
    if not revision or revision.casefold() in MUTABLE_REVISIONS:
        raise ValueError("generation producer revision must be non-empty and immutable")
    return revision


def generation_observation_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in row.items()
        if key != "evidence" and key not in DERIVED_OBSERVATION_FIELDS
    }


def generation_observation_sha256(row: dict[str, Any]) -> str:
    return canonical_sha256(generation_observation_payload(row))


def _validated_rows(observations: Any) -> list[dict[str, Any]]:
    errors = validate_objective_observations(
        observations,
        require_version=True,
        require_seed=True,
        require_runtime=True,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return observations


def _plan_samples(plan: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    if not isinstance(plan, dict) or plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("generation plan must be a version 1.0.0 or 1.1.0 object")
    samples = plan.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("generation plan must contain samples")
    if plan.get("sample_count") is not None and plan.get("sample_count") != len(samples):
        raise ValueError("generation plan sample_count must equal the number of samples")
    requirements = plan.get("generation_requirements")
    if not isinstance(requirements, dict) or any(
        requirements.get(key) is not True
        for key in ("same_transcripts", "frozen_generation_settings", "record_failures_as_observations")
    ):
        raise ValueError("generation plan must require same transcripts, frozen settings, and failure observations")
    prompt_pack = plan.get("prompt_pack")
    if (
        not isinstance(prompt_pack, dict)
        or not isinstance(prompt_pack.get("sha256"), str)
        or not SHA256_RE.fullmatch(prompt_pack["sha256"])
    ):
        raise ValueError("generation plan must bind a prompt_pack sha256")
    by_id: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"generation-plan sample {index} must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"generation-plan sample {index} must contain sample_id")
        if sample_id in by_id:
            raise ValueError(f"duplicate generation-plan sample id: {sample_id}")
        by_id[sample_id] = sample
    return canonical_sha256(plan), by_id


def _validate_row_against_plan(row: dict[str, Any], planned: dict[str, Any], index: int) -> None:
    sample_id = row["sample_id"]
    for key in ("candidate_id", "prompt_id", "seed"):
        if row.get(key) != planned.get(key):
            raise ValueError(f"observation {index} {key} does not match generation plan for {sample_id}")
    if str(row.get("requested_text", "")).strip() != str(planned.get("text", "")).strip():
        raise ValueError(f"observation {index} requested_text does not match generation plan for {sample_id}")


def _runtime_metrics(row: dict[str, Any], index: int) -> dict[str, int | float]:
    metrics = {name: row[name] for name in RUNTIME_METRIC_FIELDS if name in row}
    if "generation_seconds" not in metrics:
        raise ValueError(f"observation {index} must record generation_seconds for attempt binding")
    if float(metrics["generation_seconds"]) < 0:
        raise ValueError(f"observation {index} generation_seconds must be non-negative")
    if "audio_duration_seconds" in metrics and float(metrics["audio_duration_seconds"]) <= 0:
        raise ValueError(f"observation {index} audio_duration_seconds must be positive")
    if "peak_memory_bytes" in metrics and float(metrics["peak_memory_bytes"]) < 0:
        raise ValueError(f"observation {index} peak_memory_bytes must be non-negative")
    return metrics


def _output_audio_sha(row: dict[str, Any], audio_base_dir: Path, index: int) -> str | None:
    has_path = "audio_path" in row
    has_sha = "audio_sha256" in row
    if has_path != has_sha:
        raise ValueError(f"observation {index} audio_path and audio_sha256 must be supplied together")
    if row.get("valid") is True and not has_path:
        raise ValueError(f"observation {index} valid generation must declare output audio")
    if not has_path:
        return None
    _, live_sha = verify_observation_audio(row, audio_base_dir, index)
    return live_sha


def _attempt_core(
    *,
    row: dict[str, Any],
    planned: dict[str, Any],
    plan_sha256: str,
    output_audio_sha256: str | None,
    producer_name: str,
    producer_revision: str,
) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "generation_plan_sha256": plan_sha256,
        "planned_sample_sha256": canonical_sha256(planned),
        "source_generation_observation_sha256": generation_observation_sha256(row),
        "output_audio_sha256": output_audio_sha256,
        "runtime_metrics": {name: row[name] for name in RUNTIME_METRIC_FIELDS if name in row},
        "producer": {"name": producer_name, "revision": producer_revision},
    }


def build_generation_attempt_receipt(
    observations: Any,
    *,
    plan: Any,
    audio_base_dir: Path,
    producer_name: str,
    producer_revision: str,
) -> dict[str, Any]:
    rows = _validated_rows(observations)
    name = producer_name.strip() if isinstance(producer_name, str) else ""
    if not name:
        raise ValueError("generation producer name must be non-empty")
    revision = _immutable_revision(producer_revision)
    plan_sha, planned_by_id = _plan_samples(plan)
    attempts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        evidence = row.get("evidence")
        if isinstance(evidence, dict) and "runtime" in evidence:
            raise ValueError(f"observation {index} already contains evidence.runtime")
        planned = planned_by_id.get(row["sample_id"])
        if planned is None:
            raise ValueError(f"observation {index} sample_id is absent from the generation plan")
        _validate_row_against_plan(row, planned, index)
        _runtime_metrics(row, index)
        audio_sha = _output_audio_sha(row, audio_base_dir, index)
        core = _attempt_core(
            row=row,
            planned=planned,
            plan_sha256=plan_sha,
            output_audio_sha256=audio_sha,
            producer_name=name,
            producer_revision=revision,
        )
        attempts.append({**core, "attempt_sha256": canonical_sha256(core)})
    return {
        "schema_version": ATTEMPT_RECEIPT_SCHEMA_VERSION,
        "generation_plan_sha256": plan_sha,
        "producer": {"name": name, "revision": revision},
        "attempts": attempts,
        "evidence_boundary": (
            "This receipt binds declared runtime metrics to a planned row and live output bytes. "
            "It does not prove that the producer measured honestly, loaded declared model artifacts, "
            "or ran on a trustworthy host."
        ),
    }


def apply_generation_attempt_receipt(
    observations: Any,
    receipt: Any,
    *,
    plan: Any,
    audio_base_dir: Path,
) -> list[dict[str, Any]]:
    rows = _validated_rows(observations)
    if not isinstance(receipt, dict) or receipt.get("schema_version") != ATTEMPT_RECEIPT_SCHEMA_VERSION:
        raise ValueError("generation attempt receipt must be a version 1.0.0 object")
    allowed_receipt_fields = {
        "schema_version",
        "generation_plan_sha256",
        "producer",
        "attempts",
        "evidence_boundary",
    }
    if set(receipt) != allowed_receipt_fields:
        raise ValueError("generation attempt receipt contains missing or unknown fields")
    if not isinstance(receipt.get("evidence_boundary"), str) or not receipt["evidence_boundary"].strip():
        raise ValueError("generation attempt receipt evidence_boundary must be non-empty")
    producer = receipt.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("generation attempt receipt producer must be an object")
    if set(producer) != {"name", "revision"}:
        raise ValueError("generation attempt receipt producer contains missing or unknown fields")
    name = producer.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("generation producer name must be non-empty")
    revision = _immutable_revision(producer.get("revision"))
    plan_sha, planned_by_id = _plan_samples(plan)
    if receipt.get("generation_plan_sha256") != plan_sha:
        raise ValueError("generation attempt receipt does not match the generation plan")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("generation attempt receipt attempts must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if not isinstance(attempt, dict) or not isinstance(attempt.get("sample_id"), str):
            raise ValueError("generation attempt receipt contains an invalid attempt")
        if attempt["sample_id"] in by_id:
            raise ValueError(f"generation attempt receipt duplicates sample_id: {attempt['sample_id']}")
        by_id[attempt["sample_id"]] = attempt
    if set(by_id) != {row["sample_id"] for row in rows}:
        raise ValueError("generation attempt receipt must exactly cover source observations")
    receipt_sha = canonical_sha256(receipt)
    augmented = deepcopy(rows)
    for index, row in enumerate(augmented):
        evidence = row.setdefault("evidence", {})
        if not isinstance(evidence, dict) or "runtime" in evidence:
            raise ValueError(f"observation {index} already contains evidence.runtime")
        planned = planned_by_id.get(row["sample_id"])
        if planned is None:
            raise ValueError(f"observation {index} sample_id is absent from the generation plan")
        _validate_row_against_plan(row, planned, index)
        _runtime_metrics(row, index)
        audio_sha = _output_audio_sha(row, audio_base_dir, index)
        core = _attempt_core(
            row=row,
            planned=planned,
            plan_sha256=plan_sha,
            output_audio_sha256=audio_sha,
            producer_name=name.strip(),
            producer_revision=revision,
        )
        expected = {**core, "attempt_sha256": canonical_sha256(core)}
        if by_id[row["sample_id"]] != expected:
            raise ValueError(f"generation attempt receipt does not match source observation: {row['sample_id']}")
        evidence["runtime"] = {
            "extractor": name.strip(),
            "revision": revision,
            "generation_plan_sha256": plan_sha,
            "planned_sample_sha256": expected["planned_sample_sha256"],
            "source_generation_observation_sha256": expected["source_generation_observation_sha256"],
            "output_audio_sha256": audio_sha,
            "attempt_sha256": expected["attempt_sha256"],
            "attempt_receipt_sha256": receipt_sha,
        }
    return augmented


def runtime_attempt_is_content_bound(
    row: dict[str, Any],
    *,
    index: int,
    generation_plan_sha256: str | None = None,
    planned_sample: dict[str, Any] | None = None,
) -> bool:
    evidence = row.get("evidence")
    runtime = evidence.get("runtime") if isinstance(evidence, dict) else None
    if not isinstance(runtime, dict):
        return False
    present = RUNTIME_BINDING_FIELDS.intersection(runtime)
    if not present:
        return False
    required = RUNTIME_BINDING_FIELDS - {"output_audio_sha256"}
    if not required <= runtime.keys():
        raise ValueError(f"observation {index} evidence.runtime attempt binding must be complete")
    for field_name in required:
        value = runtime.get(field_name)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"observation {index} evidence.runtime.{field_name} must be a lowercase SHA-256")
    producer_name = runtime.get("extractor")
    producer_revision = runtime.get("revision")
    if not isinstance(producer_name, str) or not producer_name.strip():
        raise ValueError(f"observation {index} evidence.runtime.extractor must be non-empty")
    revision = _immutable_revision(producer_revision)
    output_sha = runtime.get("output_audio_sha256")
    if output_sha is not None and (not isinstance(output_sha, str) or not SHA256_RE.fullmatch(output_sha)):
        raise ValueError(f"observation {index} evidence.runtime.output_audio_sha256 must be a lowercase SHA-256")
    receipt_sha = runtime.get("attempt_receipt_sha256")
    if receipt_sha is not None and (not isinstance(receipt_sha, str) or not SHA256_RE.fullmatch(receipt_sha)):
        raise ValueError(f"observation {index} evidence.runtime.attempt_receipt_sha256 must be a lowercase SHA-256")
    if row.get("valid") is True and output_sha != row.get("audio_sha256"):
        raise ValueError(f"observation {index} evidence.runtime.output_audio_sha256 must match audio_sha256")
    if row.get("valid") is not True and output_sha is not None and output_sha != row.get("audio_sha256"):
        raise ValueError(f"observation {index} evidence.runtime.output_audio_sha256 must match audio_sha256")
    source_sha = generation_observation_sha256(row)
    if runtime["source_generation_observation_sha256"] != source_sha:
        raise ValueError(f"observation {index} runtime attempt does not match generation observation content")
    if generation_plan_sha256 is not None and runtime["generation_plan_sha256"] != generation_plan_sha256:
        raise ValueError(f"observation {index} runtime attempt does not match the generation plan")
    if planned_sample is not None and runtime["planned_sample_sha256"] != canonical_sha256(planned_sample):
        raise ValueError(f"observation {index} runtime attempt does not match its planned sample")
    core = {
        "sample_id": row["sample_id"],
        "generation_plan_sha256": runtime["generation_plan_sha256"],
        "planned_sample_sha256": runtime["planned_sample_sha256"],
        "source_generation_observation_sha256": source_sha,
        "output_audio_sha256": output_sha,
        "runtime_metrics": {name: row[name] for name in RUNTIME_METRIC_FIELDS if name in row},
        "producer": {"name": producer_name.strip(), "revision": revision},
    }
    if runtime["attempt_sha256"] != canonical_sha256(core):
        raise ValueError(f"observation {index} evidence.runtime.attempt_sha256 does not match attempt content")
    return True
