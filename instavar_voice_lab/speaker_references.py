from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from statistics import mean
from typing import Any

IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REFERENCE_AGGREGATION = "mean_cosine_similarity_v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reference_set_sha256(references: list[dict[str, str]], *, aggregation: str) -> str:
    if aggregation != REFERENCE_AGGREGATION:
        raise ValueError(f"unsupported speaker reference aggregation: {aggregation}")
    normalized = sorted(references, key=lambda item: item["reference_id"])
    return canonical_sha256({"aggregation": aggregation, "references": normalized})


def validate_reference_records(value: Any, *, context: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.references must be a non-empty array")
    references: list[dict[str, str]] = []
    seen: set[str] = set()
    for reference_index, raw in enumerate(value):
        path = f"{context}.references[{reference_index}]"
        if not isinstance(raw, dict) or set(raw) != {
            "reference_id",
            "reference_audio_sha256",
            "reference_transcript_sha256",
        }:
            raise ValueError(f"{path} must contain exactly the reference id, audio hash, and transcript hash")
        reference_id = raw.get("reference_id")
        if not isinstance(reference_id, str) or not IDENTIFIER_RE.fullmatch(reference_id):
            raise ValueError(f"{path}.reference_id must be a stable lowercase identifier")
        if reference_id in seen:
            raise ValueError(f"{context}.references contains duplicate reference_id: {reference_id}")
        seen.add(reference_id)
        for field in ("reference_audio_sha256", "reference_transcript_sha256"):
            digest = raw.get(field)
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ValueError(f"{path}.{field} must be a lowercase SHA-256")
        references.append(
            {
                "reference_id": reference_id,
                "reference_audio_sha256": raw["reference_audio_sha256"],
                "reference_transcript_sha256": raw["reference_transcript_sha256"],
            }
        )
    if [item["reference_id"] for item in references] != sorted(seen):
        raise ValueError(f"{context}.references must be sorted by reference_id")
    return references


def validate_speaker_reference_evidence(record: dict[str, Any], *, context: str) -> dict[str, Any]:
    legacy_fields = ("reference_id", "reference_audio_sha256", "reference_transcript_sha256")
    set_fields = ("reference_aggregation", "reference_set_sha256", "references")
    legacy_values = [record.get(field) for field in legacy_fields]
    set_values = [record.get(field) for field in set_fields]
    has_legacy = any(value is not None for value in legacy_values)
    has_set = any(value is not None for value in set_values)
    if has_legacy and has_set:
        raise ValueError(f"{context} cannot mix legacy and reference-set bindings")
    if has_legacy:
        if not all(value is not None for value in legacy_values):
            raise ValueError(f"{context} reference binding must be complete")
        reference_id = legacy_values[0]
        if not isinstance(reference_id, str) or not IDENTIFIER_RE.fullmatch(reference_id):
            raise ValueError(f"{context}.reference_id must be a stable lowercase identifier")
        for field, value in zip(legacy_fields[1:], legacy_values[1:]):
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise ValueError(f"{context}.{field} must be a lowercase SHA-256")
        return {
            "mode": "legacy_single_reference",
            "content_bound": True,
            "reference_id": reference_id,
            "reference_audio_sha256": legacy_values[1],
            "reference_transcript_sha256": legacy_values[2],
            "aggregation": None,
            "reference_set_sha256": None,
            "references": [],
        }
    if has_set:
        if not all(value is not None for value in set_values):
            raise ValueError(f"{context} reference-set binding must be complete")
        aggregation = record.get("reference_aggregation")
        if aggregation != REFERENCE_AGGREGATION:
            raise ValueError(f"{context}.reference_aggregation must equal {REFERENCE_AGGREGATION}")
        digest = record.get("reference_set_sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"{context}.reference_set_sha256 must be a lowercase SHA-256")
        references = validate_reference_records(record.get("references"), context=context)
        expected = reference_set_sha256(references, aggregation=aggregation)
        if digest != expected:
            raise ValueError(f"{context}.reference_set_sha256 does not match its reference records")
        return {
            "mode": "content_addressed_reference_set",
            "content_bound": True,
            "reference_id": None,
            "reference_audio_sha256": None,
            "reference_transcript_sha256": None,
            "aggregation": aggregation,
            "reference_set_sha256": digest,
            "references": references,
        }
    return {
        "mode": "unbound",
        "content_bound": False,
        "reference_id": None,
        "reference_audio_sha256": None,
        "reference_transcript_sha256": None,
        "aggregation": None,
        "reference_set_sha256": None,
        "references": [],
    }


def cosine_similarity(reference: Iterable[float], candidate: Iterable[float]) -> float:
    left = [float(value) for value in reference]
    right = [float(value) for value in candidate]
    if not left or len(left) != len(right):
        raise ValueError("speaker embeddings must be non-empty and have equal dimensions")
    if any(not math.isfinite(value) for value in [*left, *right]):
        raise ValueError("speaker embeddings must contain finite numbers")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("speaker embeddings must have non-zero norm")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def aggregate_reference_similarities(
    reference_embeddings: Any,
    candidate_embedding: Any,
    *,
    evidence: dict[str, Any],
    context: str,
) -> tuple[float, list[dict[str, Any]]]:
    binding = validate_speaker_reference_evidence(evidence, context=context)
    if binding["mode"] != "content_addressed_reference_set":
        raise ValueError(f"{context} must use a content-addressed reference set")
    if not isinstance(reference_embeddings, list) or not reference_embeddings:
        raise ValueError("reference_speaker_embeddings must be a non-empty array")
    if not isinstance(candidate_embedding, list):
        raise ValueError("speaker_embedding must be an array")
    expected_ids = [item["reference_id"] for item in binding["references"]]
    observed_ids: list[str] = []
    scores: list[dict[str, Any]] = []
    for index, raw in enumerate(reference_embeddings):
        if not isinstance(raw, dict) or set(raw) != {"reference_id", "embedding"}:
            raise ValueError(f"reference_speaker_embeddings[{index}] must contain exactly reference_id and embedding")
        reference_id = raw.get("reference_id")
        embedding = raw.get("embedding")
        if not isinstance(reference_id, str):
            raise ValueError(f"reference_speaker_embeddings[{index}].reference_id must be a string")
        if not isinstance(embedding, list):
            raise ValueError(f"reference_speaker_embeddings[{index}].embedding must be an array")
        observed_ids.append(reference_id)
        scores.append(
            {
                "reference_id": reference_id,
                "cosine_similarity": cosine_similarity(embedding, candidate_embedding),
            }
        )
    if observed_ids != expected_ids:
        raise ValueError("reference_speaker_embeddings must exactly match the bound sorted reference set")
    return mean(item["cosine_similarity"] for item in scores), scores


def speaker_measurement_sha256(row: dict[str, Any], evidence: dict[str, Any]) -> str:
    binding = validate_speaker_reference_evidence(evidence, context="speaker measurement evidence")
    candidate_embedding = row.get("speaker_embedding")
    if not isinstance(candidate_embedding, list):
        raise ValueError("speaker measurement must contain speaker_embedding")
    if binding["mode"] == "content_addressed_reference_set":
        reference_values = row.get("reference_speaker_embeddings")
        if not isinstance(reference_values, list):
            raise ValueError("speaker measurement must contain reference_speaker_embeddings")
        reference_binding: dict[str, Any] = {
            "mode": binding["mode"],
            "reference_set_sha256": binding["reference_set_sha256"],
            "reference_aggregation": binding["aggregation"],
        }
    elif binding["mode"] == "legacy_single_reference":
        reference_values = row.get("reference_speaker_embedding")
        if not isinstance(reference_values, list):
            raise ValueError("speaker measurement must contain reference_speaker_embedding")
        reference_binding = {
            "mode": binding["mode"],
            "reference_id": binding["reference_id"],
            "reference_audio_sha256": binding["reference_audio_sha256"],
            "reference_transcript_sha256": binding["reference_transcript_sha256"],
        }
    else:
        raise ValueError("speaker measurement must bind a speaker reference")
    sample_id = row.get("sample_id")
    input_audio_sha256 = evidence.get("input_audio_sha256")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("speaker measurement sample_id must be non-empty")
    if not isinstance(input_audio_sha256, str) or not SHA256_RE.fullmatch(input_audio_sha256):
        raise ValueError("speaker measurement input_audio_sha256 must be a lowercase SHA-256")
    return canonical_sha256(
        {
            "sample_id": sample_id,
            "input_audio_sha256": input_audio_sha256,
            "reference_binding": reference_binding,
            "reference_values": reference_values,
            "speaker_embedding": candidate_embedding,
        }
    )


def speaker_measurement_is_content_bound(
    row: dict[str, Any],
    evidence: dict[str, Any],
    *,
    context: str,
) -> bool:
    digest = evidence.get("speaker_measurement_sha256")
    if digest is None:
        return False
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{context}.speaker_measurement_sha256 must be a lowercase SHA-256")
    expected = speaker_measurement_sha256(row, evidence)
    if digest != expected:
        raise ValueError(f"{context}.speaker_measurement_sha256 does not match the speaker embeddings")
    return True
