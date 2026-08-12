from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .speaker_references import REFERENCE_AGGREGATION, SHA256_RE, canonical_sha256

IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ASSIGNMENT_PLAN_SCHEMA_VERSION = "1.0.0"
SELECTION_STAGE = "before_candidate_generation_or_scoring"


def validate_speaker_reference_catalog(catalog: Any) -> dict[str, Any]:
    if not isinstance(catalog, dict) or set(catalog) != {"catalog_id", "references", "catalog_sha256"}:
        raise ValueError("speaker reference catalog must contain exactly catalog_id, references, and catalog_sha256")
    catalog_id = catalog.get("catalog_id")
    if not isinstance(catalog_id, str) or not IDENTIFIER_RE.fullmatch(catalog_id):
        raise ValueError("speaker reference catalog id must be a stable lowercase identifier")
    raw_references = catalog.get("references")
    if not isinstance(raw_references, list) or not raw_references:
        raise ValueError("speaker reference catalog must contain at least one reference")
    reference_ids: list[str] = []
    for index, reference in enumerate(raw_references):
        context = f"speaker reference catalog references[{index}]"
        if not isinstance(reference, dict) or set(reference) != {"reference_id", "audio", "transcript"}:
            raise ValueError(f"{context} must contain exactly reference_id, audio, and transcript")
        reference_id = reference.get("reference_id")
        if not isinstance(reference_id, str) or not IDENTIFIER_RE.fullmatch(reference_id):
            raise ValueError(f"{context}.reference_id must be a stable lowercase identifier")
        reference_ids.append(reference_id)
        for role in ("audio", "transcript"):
            fingerprint = reference.get(role)
            if not isinstance(fingerprint, dict) or set(fingerprint) != {"sha256", "bytes"}:
                raise ValueError(f"{context}.{role} must contain exactly sha256 and bytes")
            if not isinstance(fingerprint.get("sha256"), str) or not SHA256_RE.fullmatch(fingerprint["sha256"]):
                raise ValueError(f"{context}.{role}.sha256 must be a lowercase SHA-256")
            byte_count = fingerprint.get("bytes")
            if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 1:
                raise ValueError(f"{context}.{role}.bytes must be a positive integer")
    if reference_ids != sorted(set(reference_ids)):
        raise ValueError("speaker reference catalog references must be unique and sorted by reference_id")
    payload = {"catalog_id": catalog_id, "references": raw_references}
    expected_sha = canonical_sha256(payload)
    if catalog.get("catalog_sha256") != expected_sha:
        raise ValueError("speaker reference catalog_sha256 does not match the catalog contents")
    return {
        "catalog_id": catalog_id,
        "catalog_sha256": expected_sha,
        "reference_ids": reference_ids,
    }


def _generation_assignment_keys(generation_plan: Any) -> list[tuple[str, int]]:
    if not isinstance(generation_plan, dict) or generation_plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("generation plan must be a version 1.0.0 or 1.1.0 object")
    samples = generation_plan.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("generation plan must contain samples")
    keys: set[tuple[str, int]] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"generation-plan sample {index} must be an object")
        prompt_id = sample.get("prompt_id")
        seed = sample.get("seed")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"generation-plan sample {index} prompt_id must be non-empty")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"generation-plan sample {index} seed must be a non-negative integer")
        keys.add((prompt_id.strip(), seed))
    return sorted(keys)


def _normalize_dimensions(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError("speaker reference stratification dimensions must be an array")
    dimensions = list(values)
    if not dimensions:
        raise ValueError("speaker reference selection policy must declare at least one stratification dimension")
    if any(not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value) for value in dimensions):
        raise ValueError("speaker reference stratification dimensions must be stable lowercase identifiers")
    if dimensions != sorted(set(dimensions)):
        raise ValueError("speaker reference stratification dimensions must be unique and sorted")
    return dimensions


def _normalize_assignments(
    assignments: Any,
    *,
    known_reference_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[str]]]:
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("speaker reference assignment plan must contain assignments")
    normalized: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], list[str]] = {}
    for index, assignment in enumerate(assignments):
        context = f"speaker reference assignments[{index}]"
        if not isinstance(assignment, dict) or set(assignment) != {"prompt_id", "seed", "reference_ids"}:
            raise ValueError(f"{context} must contain exactly prompt_id, seed, and reference_ids")
        prompt_id = assignment.get("prompt_id")
        seed = assignment.get("seed")
        reference_ids = assignment.get("reference_ids")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"{context}.prompt_id must be non-empty")
        prompt_id = prompt_id.strip()
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"{context}.seed must be a non-negative integer")
        if not isinstance(reference_ids, list) or not reference_ids:
            raise ValueError(f"{context}.reference_ids must be a non-empty array")
        if any(not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value) for value in reference_ids):
            raise ValueError(f"{context}.reference_ids must contain stable lowercase identifiers")
        if reference_ids != sorted(set(reference_ids)):
            raise ValueError(f"{context}.reference_ids must be unique and sorted")
        if known_reference_ids is not None:
            unknown = sorted(set(reference_ids) - known_reference_ids)
            if unknown:
                raise ValueError(f"{context} contains references absent from the catalog: {unknown}")
        key = (prompt_id, seed)
        if key in by_key:
            raise ValueError(f"duplicate speaker reference assignment for prompt {prompt_id}, seed {seed}")
        copied_ids = list(reference_ids)
        by_key[key] = copied_ids
        normalized.append({"prompt_id": prompt_id, "seed": seed, "reference_ids": copied_ids})
    if [(item["prompt_id"], item["seed"]) for item in normalized] != sorted(by_key):
        raise ValueError("speaker reference assignments must be sorted by prompt_id and seed")
    return normalized, by_key


def build_speaker_reference_assignment_plan(
    *,
    plan_id: str,
    generation_plan: dict[str, Any],
    reference_catalog: dict[str, Any],
    assignments: dict[tuple[str, int], list[str]],
    policy_id: str,
    stratification_dimensions: Iterable[str],
    rationale: str,
) -> dict[str, Any]:
    if not isinstance(plan_id, str) or not IDENTIFIER_RE.fullmatch(plan_id):
        raise ValueError("speaker reference assignment plan id must be a stable lowercase identifier")
    if not isinstance(policy_id, str) or not IDENTIFIER_RE.fullmatch(policy_id):
        raise ValueError("speaker reference policy id must be a stable lowercase identifier")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("speaker reference selection policy rationale must be non-empty")
    dimensions = _normalize_dimensions(stratification_dimensions)
    catalog_binding = validate_speaker_reference_catalog(reference_catalog)
    expected_keys = _generation_assignment_keys(generation_plan)
    if not isinstance(assignments, dict):
        raise ValueError("speaker reference assignments must be a mapping")
    assignment_rows: list[dict[str, Any]] = []
    for key, reference_ids in assignments.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("speaker reference assignment keys must contain prompt_id and seed")
        prompt_id, seed = key
        assignment_rows.append({"prompt_id": prompt_id, "seed": seed, "reference_ids": reference_ids})
    assignment_rows.sort(key=lambda item: (str(item["prompt_id"]), item["seed"] if isinstance(item["seed"], int) else -1))
    normalized_assignments, by_key = _normalize_assignments(
        assignment_rows,
        known_reference_ids=set(catalog_binding["reference_ids"]),
    )
    if set(by_key) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(by_key))
        unexpected = sorted(set(by_key) - set(expected_keys))
        raise ValueError(
            f"speaker reference assignments must exactly cover generation prompt and seed pairs; "
            f"missing={missing}; unexpected={unexpected}"
        )
    payload = {
        "schema_version": ASSIGNMENT_PLAN_SCHEMA_VERSION,
        "plan_id": plan_id,
        "generation_plan_sha256": canonical_sha256(generation_plan),
        "reference_catalog_sha256": catalog_binding["catalog_sha256"],
        "reference_aggregation": REFERENCE_AGGREGATION,
        "selection_policy": {
            "policy_id": policy_id,
            "selection_stage": SELECTION_STAGE,
            "stratification_dimensions": dimensions,
            "rationale": rationale.strip(),
        },
        "assignments": normalized_assignments,
    }
    return {**payload, "assignment_plan_sha256": canonical_sha256(payload)}


def validate_speaker_reference_assignment_plan(
    plan: Any,
    *,
    generation_plan: dict[str, Any] | None = None,
    reference_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "plan_id",
        "generation_plan_sha256",
        "reference_catalog_sha256",
        "reference_aggregation",
        "selection_policy",
        "assignments",
        "assignment_plan_sha256",
    }
    if not isinstance(plan, dict) or set(plan) != expected_fields:
        raise ValueError(f"speaker reference assignment plan must contain exactly {sorted(expected_fields)}")
    if plan.get("schema_version") != ASSIGNMENT_PLAN_SCHEMA_VERSION:
        raise ValueError(f"speaker reference assignment plan schema_version must equal {ASSIGNMENT_PLAN_SCHEMA_VERSION}")
    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str) or not IDENTIFIER_RE.fullmatch(plan_id):
        raise ValueError("speaker reference assignment plan id must be a stable lowercase identifier")
    for field in ("generation_plan_sha256", "reference_catalog_sha256", "assignment_plan_sha256"):
        value = plan.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(f"speaker reference assignment plan {field} must be a lowercase SHA-256")
    if plan.get("reference_aggregation") != REFERENCE_AGGREGATION:
        raise ValueError(f"speaker reference assignment plan reference_aggregation must equal {REFERENCE_AGGREGATION}")
    selection_policy = plan.get("selection_policy")
    policy_fields = {"policy_id", "selection_stage", "stratification_dimensions", "rationale"}
    if not isinstance(selection_policy, dict) or set(selection_policy) != policy_fields:
        raise ValueError(f"speaker reference selection policy must contain exactly {sorted(policy_fields)}")
    policy_id = selection_policy.get("policy_id")
    if not isinstance(policy_id, str) or not IDENTIFIER_RE.fullmatch(policy_id):
        raise ValueError("speaker reference policy id must be a stable lowercase identifier")
    if selection_policy.get("selection_stage") != SELECTION_STAGE:
        raise ValueError(f"speaker reference selection_stage must equal {SELECTION_STAGE}")
    dimensions = _normalize_dimensions(selection_policy.get("stratification_dimensions", []))
    rationale = selection_policy.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("speaker reference selection policy rationale must be non-empty")
    known_reference_ids: set[str] | None = None
    if reference_catalog is not None:
        catalog_binding = validate_speaker_reference_catalog(reference_catalog)
        if plan["reference_catalog_sha256"] != catalog_binding["catalog_sha256"]:
            raise ValueError("speaker reference assignment plan does not match the reference catalog")
        known_reference_ids = set(catalog_binding["reference_ids"])
    _, by_key = _normalize_assignments(
        plan.get("assignments"),
        known_reference_ids=known_reference_ids,
    )
    if generation_plan is not None:
        generation_sha = canonical_sha256(generation_plan)
        if plan["generation_plan_sha256"] != generation_sha:
            raise ValueError("speaker reference assignment plan does not match the generation plan")
        expected_keys = set(_generation_assignment_keys(generation_plan))
        if set(by_key) != expected_keys:
            missing = sorted(expected_keys - set(by_key))
            unexpected = sorted(set(by_key) - expected_keys)
            raise ValueError(
                f"speaker reference assignments must exactly cover generation prompt and seed pairs; "
                f"missing={missing}; unexpected={unexpected}"
            )
    payload = {key: plan[key] for key in expected_fields - {"assignment_plan_sha256"}}
    expected_sha = canonical_sha256(payload)
    if plan["assignment_plan_sha256"] != expected_sha:
        raise ValueError("speaker reference assignment_plan_sha256 does not match the plan contents")
    return {
        "plan_id": plan_id,
        "sha256": expected_sha,
        "generation_plan_sha256": plan["generation_plan_sha256"],
        "reference_catalog_sha256": plan["reference_catalog_sha256"],
        "reference_aggregation": REFERENCE_AGGREGATION,
        "selection_policy": {
            "policy_id": policy_id,
            "selection_stage": SELECTION_STAGE,
            "stratification_dimensions": dimensions,
            "rationale": rationale.strip(),
        },
        "assignments": by_key,
        "assignment_count": len(by_key),
    }


def speaker_reference_assignment_sha256(
    *,
    assignment_plan_sha256: str,
    prompt_id: str,
    seed: int,
    reference_ids: list[str],
) -> str:
    if not isinstance(assignment_plan_sha256, str) or not SHA256_RE.fullmatch(assignment_plan_sha256):
        raise ValueError("speaker reference assignment plan digest must be a lowercase SHA-256")
    normalized, _ = _normalize_assignments(
        [{"prompt_id": prompt_id, "seed": seed, "reference_ids": reference_ids}]
    )
    return canonical_sha256(
        {
            "assignment_plan_sha256": assignment_plan_sha256,
            **normalized[0],
        }
    )
