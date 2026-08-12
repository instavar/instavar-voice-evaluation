from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable


IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_prompt_pack(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["prompt pack must be an object"]

    required = {
        "id",
        "version",
        "locale",
        "purpose",
        "generation",
        "prompts",
        "objective_metrics",
        "listening_criteria",
        "measurement_limits",
    }
    for key in sorted(required - document.keys()):
        errors.append(f"{key} is required")

    for key in ("id", "version", "locale", "purpose"):
        if not isinstance(document.get(key), str) or not document[key].strip():
            errors.append(f"{key} must be a non-empty string")
    pack_id = document.get("id")
    if isinstance(pack_id, str) and pack_id and not IDENTIFIER_RE.fullmatch(pack_id):
        errors.append("id must contain only lowercase letters, digits, dots, underscores, and hyphens")

    generation = document.get("generation")
    if not isinstance(generation, dict):
        errors.append("generation must be an object")
        generation = {}
    seeds = generation.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("generation.seeds must be a non-empty array of frozen integer seeds")
        seeds = []
    elif any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        errors.append("generation.seeds must contain non-negative integers")
    elif len(set(seeds)) != len(seeds):
        errors.append("generation.seeds must be unique")
    if generation.get("seeds_per_candidate") != len(seeds):
        errors.append("generation.seeds_per_candidate must equal the number of frozen seeds")
    for name in ("require_same_transcripts", "require_frozen_generation_settings"):
        if not isinstance(generation.get(name), bool):
            errors.append(f"generation.{name} must be a boolean")

    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        errors.append("prompts must be a non-empty array")
        prompts = []
    prompt_ids: list[str] = []
    categories: set[str] = set()
    for index, value in enumerate(prompts):
        path = f"prompts[{index}]"
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            continue
        for key in ("id", "category", "text"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                errors.append(f"{path}.{key} must be a non-empty string")
        prompt_id = value.get("id")
        if isinstance(prompt_id, str) and prompt_id:
            prompt_ids.append(prompt_id)
            if not IDENTIFIER_RE.fullmatch(prompt_id):
                errors.append(f"{path}.id has an invalid format")
        category = value.get("category")
        if isinstance(category, str) and category:
            categories.add(category)
        instruction = value.get("instruction")
        if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
            errors.append(f"{path}.instruction must be a non-empty string when present")
    duplicates = sorted(prompt_id for prompt_id, count in Counter(prompt_ids).items() if count > 1)
    if duplicates:
        errors.append(f"prompt ids must be unique: {', '.join(duplicates)}")
    if "long_form_cadence" not in categories:
        errors.append("prompts must include a long_form_cadence category")
    if "pronunciation" not in categories:
        errors.append("prompts must include a pronunciation category")

    for key in ("objective_metrics", "listening_criteria", "measurement_limits"):
        values = document.get(key)
        if not isinstance(values, list) or not values:
            errors.append(f"{key} must be a non-empty array")
        elif any(not isinstance(value, str) or not value.strip() for value in values):
            errors.append(f"{key} must contain non-empty strings")
        elif len(set(values)) != len(values):
            errors.append(f"{key} must not contain duplicates")
    return errors


def build_generation_plan(
    prompt_pack: Any,
    candidate_ids: Iterable[str],
    *,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    errors = validate_prompt_pack(prompt_pack)
    if errors:
        raise ValueError("; ".join(errors))

    candidates = list(candidate_ids)
    if not candidates:
        raise ValueError("at least one candidate id is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate ids must be unique")
    for candidate_id in candidates:
        if not isinstance(candidate_id, str) or not IDENTIFIER_RE.fullmatch(candidate_id):
            raise ValueError(f"invalid candidate id: {candidate_id!r}")

    frozen_seeds = list(prompt_pack["generation"]["seeds"] if seeds is None else seeds)
    if not frozen_seeds or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in frozen_seeds):
        raise ValueError("seeds must be non-negative integers")
    if len(set(frozen_seeds)) != len(frozen_seeds):
        raise ValueError("seeds must be unique")

    samples: list[dict[str, Any]] = []
    for candidate_id in candidates:
        for prompt in prompt_pack["prompts"]:
            for seed in frozen_seeds:
                sample_id = f"{candidate_id}--{prompt['id']}--seed-{seed}"
                row = {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "seed": seed,
                    "text": prompt["text"],
                    "expected_audio_path": f"audio/{candidate_id}/{prompt['id']}/seed-{seed}.wav",
                }
                if prompt.get("instruction"):
                    row["instruction"] = prompt["instruction"]
                samples.append(row)

    return {
        "schema_version": "1.0.0",
        "prompt_pack": {
            "id": prompt_pack["id"],
            "version": prompt_pack["version"],
            "sha256": _canonical_sha256(prompt_pack),
        },
        "candidate_ids": candidates,
        "seeds": frozen_seeds,
        "prompt_count": len(prompt_pack["prompts"]),
        "sample_count": len(samples),
        "samples": samples,
        "generation_requirements": {
            "same_transcripts": bool(prompt_pack["generation"]["require_same_transcripts"]),
            "frozen_generation_settings": bool(prompt_pack["generation"]["require_frozen_generation_settings"]),
            "record_failures_as_observations": True,
        },
    }


def check_suite_coverage(plan: Any, observations: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema_version") != "1.0.0":
        raise ValueError("generation plan must be a version 1.0.0 object")
    expected_rows = plan.get("samples")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError("generation plan must contain samples")
    if not isinstance(observations, list):
        raise ValueError("observations must be an array")

    expected: dict[str, dict[str, Any]] = {}
    for row in expected_rows:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise ValueError("every generation-plan sample must contain sample_id")
        if row["sample_id"] in expected:
            raise ValueError(f"duplicate generation-plan sample id: {row['sample_id']}")
        expected[row["sample_id"]] = row

    observed_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed_count = 0
    for row in observations:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            malformed_count += 1
            continue
        observed_rows[row["sample_id"]].append(row)

    missing = sorted(set(expected) - set(observed_rows))
    unexpected = sorted(set(observed_rows) - set(expected))
    duplicates = sorted(sample_id for sample_id, rows in observed_rows.items() if len(rows) > 1)
    invalid = sorted(
        sample_id
        for sample_id, rows in observed_rows.items()
        if sample_id in expected and len(rows) == 1 and rows[0].get("valid") is not True
    )
    mismatched: list[dict[str, Any]] = []
    for sample_id, rows in sorted(observed_rows.items()):
        if sample_id not in expected or len(rows) != 1:
            continue
        planned = expected[sample_id]
        observed = rows[0]
        mismatched_fields = [
            observed_name
            for observed_name, planned_name in (
                ("candidate_id", "candidate_id"),
                ("prompt_id", "prompt_id"),
                ("seed", "seed"),
                ("requested_text", "text"),
            )
            if observed.get(observed_name) != planned.get(planned_name)
        ]
        if not isinstance(observed.get("valid"), bool):
            mismatched_fields.append("valid")
        if mismatched_fields:
            mismatched.append({"sample_id": sample_id, "fields": sorted(mismatched_fields)})

    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_id in plan.get("candidate_ids", []):
        candidate_expected = [row for row in expected.values() if row.get("candidate_id") == candidate_id]
        candidate_ids = {row["sample_id"] for row in candidate_expected}
        candidate_observed = candidate_ids & set(observed_rows)
        by_candidate[candidate_id] = {
            "expected": len(candidate_ids),
            "observed": len(candidate_observed),
            "missing": len(candidate_ids - candidate_observed),
            "invalid": sum(sample_id in invalid for sample_id in candidate_ids),
            "prompt_count": len({row.get("prompt_id") for row in candidate_expected}),
            "seed_count": len({row.get("seed") for row in candidate_expected}),
        }

    by_category: dict[str, dict[str, int]] = {}
    categories = sorted({str(row.get("category")) for row in expected.values()})
    for category in categories:
        category_ids = {row["sample_id"] for row in expected.values() if row.get("category") == category}
        by_category[category] = {
            "expected": len(category_ids),
            "observed": len(category_ids & set(observed_rows)),
            "missing": len(category_ids - set(observed_rows)),
            "invalid": sum(sample_id in invalid for sample_id in category_ids),
        }

    coverage_complete = (
        not missing
        and not unexpected
        and not duplicates
        and not mismatched
        and malformed_count == 0
    )
    return {
        "schema_version": "1.0.0",
        "status": "passed" if coverage_complete else "failed",
        "coverage_complete": coverage_complete,
        "generation_complete_without_invalid_outputs": coverage_complete and not invalid,
        "expected_sample_count": len(expected),
        "observed_sample_count": sum(len(rows) for rows in observed_rows.values()),
        "missing_sample_ids": missing,
        "unexpected_sample_ids": unexpected,
        "duplicate_sample_ids": duplicates,
        "mismatched_observations": mismatched,
        "invalid_sample_ids": invalid,
        "malformed_observation_count": malformed_count,
        "by_candidate": by_candidate,
        "by_category": by_category,
        "evidence_boundary": (
            "Complete coverage proves that every planned sample has one recorded observation. "
            "It does not prove intelligibility, speaker identity, accent fidelity, cadence, or perceptual quality."
        ),
    }
