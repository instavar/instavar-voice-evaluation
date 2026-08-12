from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .metrics import bootstrap_mean_interval

LISTENING_ASSIGNMENT_PLAN_SCHEMA_VERSION = "1.0.0"
LISTENING_ROUTING_SCHEMA_VERSION = "1.0.0"
_ROUTING_SELECTORS = {"all_samples", "categories", "categories_or_lexical_anchors", "lexical_anchors"}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_routing(routing: Any) -> list[dict[str, Any]]:
    if not isinstance(routing, dict) or routing.get("schema_version") != LISTENING_ROUTING_SCHEMA_VERSION:
        raise ValueError(f"listening routing schema_version must equal {LISTENING_ROUTING_SCHEMA_VERSION}")
    raw_routes = routing.get("routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise ValueError("listening routing routes must be a non-empty array")
    routes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_routes):
        if not isinstance(raw, dict):
            raise ValueError(f"listening routing route {index} must be an object")
        criterion = raw.get("criterion")
        selector = raw.get("selector")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"listening routing route {index} criterion must be a non-empty string")
        criterion = criterion.strip()
        if criterion in seen:
            raise ValueError(f"listening routing criterion must be unique: {criterion}")
        seen.add(criterion)
        if selector not in _ROUTING_SELECTORS:
            raise ValueError(
                f"listening routing route {criterion} selector must be one of: {', '.join(sorted(_ROUTING_SELECTORS))}"
            )
        normalized = {"criterion": criterion, "selector": selector}
        categories = raw.get("categories")
        if selector in {"categories", "categories_or_lexical_anchors"}:
            if (
                not isinstance(categories, list)
                or not categories
                or any(not isinstance(value, str) or not value.strip() for value in categories)
            ):
                raise ValueError(f"listening routing route {criterion} categories must be a non-empty string array")
            normalized_categories = [value.strip() for value in categories]
            if len(set(normalized_categories)) != len(normalized_categories):
                raise ValueError(f"listening routing route {criterion} categories must be unique")
            normalized["categories"] = normalized_categories
        elif categories is not None:
            raise ValueError(
                f"listening routing route {criterion} categories require a category-based selector"
            )
        routes.append(normalized)
    return routes


def _generation_rows(generation_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(generation_plan, dict) or generation_plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("generation plan schema_version must equal 1.0.0 or 1.1.0")
    raw_samples = generation_plan.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("generation plan samples must be a non-empty array")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    candidate_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
    prompt_shapes: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"generation plan sample {index} must be an object")
        required = {"sample_id", "candidate_id", "prompt_id", "seed", "category"}
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"generation plan sample {index} is missing: {', '.join(sorted(missing))}")
        sample_id = raw["sample_id"]
        candidate_id = raw["candidate_id"]
        prompt_id = raw["prompt_id"]
        category = raw["category"]
        seed = raw["seed"]
        identifiers = (sample_id, candidate_id, prompt_id, category)
        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError(f"generation plan sample {index} contains an empty identifier")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"generation plan sample {index} seed must be a non-negative integer")
        if sample_id in seen_ids:
            raise ValueError(f"generation plan sample_id must be unique: {sample_id}")
        seen_ids.add(sample_id)
        key = (prompt_id, seed)
        if key in candidate_keys[candidate_id]:
            raise ValueError(
                f"generation plan repeats prompt and seed for candidate {candidate_id}: {prompt_id}/{seed}"
            )
        candidate_keys[candidate_id].add(key)
        shape = (category, _digest(raw.get("lexical_anchors")))
        if prompt_id in prompt_shapes and prompt_shapes[prompt_id] != shape:
            raise ValueError(f"generation plan route-relevant fields drift across candidates or seeds: {prompt_id}")
        prompt_shapes[prompt_id] = shape
        rows.append(dict(raw))
    key_sets = list(candidate_keys.values())
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("generation plan candidates must cover the same prompt and seed keys for listening routing")
    return rows


def _assignment_payload(generation_plan: dict[str, Any], routing: dict[str, Any]) -> dict[str, Any]:
    rows = _generation_rows(generation_plan)
    routes = _normalized_routing(routing)
    assignments: list[dict[str, Any]] = []
    matched_by_criterion: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row in rows:
        assigned: list[str] = []
        for route in routes:
            selector = route["selector"]
            matches = (
                selector == "all_samples"
                or (selector == "categories" and row["category"] in route["categories"])
                or (selector == "lexical_anchors" and bool(row.get("lexical_anchors")))
                or (
                    selector == "categories_or_lexical_anchors"
                    and (row["category"] in route["categories"] or bool(row.get("lexical_anchors")))
                )
            )
            if matches:
                assigned.append(route["criterion"])
                matched_by_criterion[route["criterion"]].append(
                    (str(row["candidate_id"]), str(row["prompt_id"]), int(row["seed"]))
                )
        if not assigned:
            raise ValueError(f"listening routing leaves sample without any criterion: {row['sample_id']}")
        assignments.append(
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "criteria": assigned,
            }
        )
    candidate_ids = sorted({str(row["candidate_id"]) for row in rows})
    for route in routes:
        criterion = route["criterion"]
        matched = matched_by_criterion[criterion]
        if not matched:
            raise ValueError(f"listening routing criterion matches no generation-plan samples: {criterion}")
        target_sets = {
            candidate_id: {(prompt_id, seed) for candidate, prompt_id, seed in matched if candidate == candidate_id}
            for candidate_id in candidate_ids
        }
        if any(targets != target_sets[candidate_ids[0]] for targets in target_sets.values()):
            raise ValueError(f"listening routing criterion is candidate-asymmetric: {criterion}")
    normalized_routing = {"schema_version": LISTENING_ROUTING_SCHEMA_VERSION, "routes": routes}
    return {
        "schema_version": LISTENING_ASSIGNMENT_PLAN_SCHEMA_VERSION,
        "generation_plan_sha256": _digest(generation_plan),
        "criteria": [route["criterion"] for route in routes],
        "routing": normalized_routing,
        "assignments": assignments,
        "evidence_boundary": (
            "The artifact freezes which planned samples support each listening criterion. Its hashes do not prove "
            "that it was created before generation without an external server-stamped chronology record."
        ),
    }


def build_listening_assignment_plan(generation_plan: dict[str, Any], routing: dict[str, Any]) -> dict[str, Any]:
    payload = _assignment_payload(generation_plan, routing)
    return {**payload, "assignment_plan_sha256": _digest(payload)}


def validate_listening_assignment_plan(
    plan: Any,
    *,
    generation_plan: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("listening assignment plan must be an object")
    if plan.get("schema_version") != LISTENING_ASSIGNMENT_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"listening assignment plan schema_version must equal {LISTENING_ASSIGNMENT_PLAN_SCHEMA_VERSION}"
        )
    claimed_hash = plan.get("assignment_plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "assignment_plan_sha256"}
    if not isinstance(claimed_hash, str) or claimed_hash != _digest(payload):
        raise ValueError("listening assignment plan self-hash does not match its content")
    expected = build_listening_assignment_plan(generation_plan, plan.get("routing"))
    if plan != expected:
        raise ValueError("listening assignment plan does not match the supplied generation plan and routing")
    return {
        "schema_version": LISTENING_ASSIGNMENT_PLAN_SCHEMA_VERSION,
        "generation_plan_sha256": plan["generation_plan_sha256"],
        "assignment_plan_sha256": claimed_hash,
        "sample_count": len(plan["assignments"]),
        "criterion_count": len(plan["criteria"]),
    }


def build_blind_pack(
    samples: list[dict[str, Any]],
    criteria: list[str] | None,
    *,
    seed: int,
    assignment_plan: dict[str, Any] | None = None,
    generation_plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    required = {"sample_id", "candidate_id", "prompt_id", "audio_path"}
    for index, sample in enumerate(samples):
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"sample {index} is missing: {', '.join(sorted(missing))}")
    criteria_by_sample: dict[str, list[str]] = {}
    assignment_binding: dict[str, Any] | None = None
    if assignment_plan is not None:
        if generation_plan is None:
            raise ValueError("generation_plan is required with assignment_plan")
        assignment_binding = validate_listening_assignment_plan(assignment_plan, generation_plan=generation_plan)
        criteria = list(assignment_plan["criteria"])
        planned = {row["sample_id"]: row for row in assignment_plan["assignments"]}
        observed_ids = {str(sample.get("sample_id", "")) for sample in samples}
        if len(observed_ids) != len(samples) or observed_ids != set(planned):
            raise ValueError("blind-review samples must cover every listening assignment exactly once")
        for sample in samples:
            sample_id = str(sample["sample_id"])
            row = planned[sample_id]
            for field in ("candidate_id", "prompt_id", "seed"):
                if sample.get(field) != row[field]:
                    raise ValueError(f"blind-review sample {sample_id} does not match assignment field {field}")
            criteria_by_sample[sample_id] = list(row["criteria"])
    if not criteria or any(not isinstance(value, str) or not value.strip() for value in criteria):
        raise ValueError("criteria must contain non-empty strings")
    if len(set(criteria)) != len(criteria):
        raise ValueError("criteria must be unique")
    suffixes = {Path(str(sample["audio_path"])).suffix.lower() for sample in samples}
    if len(suffixes) != 1:
        raise ValueError("all blind-review audio files must use the same extension to avoid candidate leakage")

    randomized = list(samples)
    random.Random(seed).shuffle(randomized)
    blind_rows: list[dict[str, Any]] = []
    reveal_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(randomized, start=1):
        blind_id = f"sample-{index:04d}"
        suffix = Path(str(sample["audio_path"])).suffix.lower()
        if not suffix or len(suffix) > 10:
            suffix = ".wav"
        blind_row = {
            "blind_id": blind_id,
            "prompt_id": str(sample["prompt_id"]),
            "audio_path": f"blind_audio/{blind_id}{suffix}",
        }
        if assignment_plan is not None:
            blind_row["criteria"] = criteria_by_sample[str(sample["sample_id"])]
        blind_rows.append(blind_row)
        reveal_row = {
            "blind_id": blind_id,
            "sample_id": str(sample["sample_id"]),
            "candidate_id": str(sample["candidate_id"]),
            "prompt_id": str(sample["prompt_id"]),
            "source_audio_path": str(sample["audio_path"]),
        }
        if "seed" in sample:
            reveal_row["seed"] = sample["seed"]
        reveal_rows.append(reveal_row)
    review = {
        "schema_version": "1.0.0",
        "blind": True,
        "criteria": criteria,
        "samples": blind_rows,
        "instructions": (
            "Rate one criterion at a time. Review only the blind_audio paths and do not open the reveal mapping "
            "until every rating is recorded."
        ),
    }
    if assignment_binding is not None:
        review["assignment_plan"] = assignment_binding
        review["instructions"] = (
            "Rate only the criteria assigned to each sample, one criterion at a time. Review only the blind_audio "
            "paths and do not open the reveal mapping until every assigned rating is recorded."
        )
    reveal = {
        "schema_version": "1.0.0",
        "seed": seed,
        "review_sha256": _digest(review),
        "mapping": reveal_rows,
    }
    return review, reveal


def stage_blind_audio(review: dict[str, Any], reveal: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if _digest(review) != reveal.get("review_sha256"):
        raise ValueError("reveal mapping does not match the review document")
    review_paths = {
        row["blind_id"]: row["audio_path"]
        for row in review.get("samples", [])
        if isinstance(row, dict) and {"blind_id", "audio_path"} <= row.keys()
    }
    reveal_rows = {
        row["blind_id"]: row
        for row in reveal.get("mapping", [])
        if isinstance(row, dict) and {"blind_id", "source_audio_path"} <= row.keys()
    }
    if set(review_paths) != set(reveal_rows):
        raise ValueError("review and reveal documents must cover the same blind ids")

    staged = []
    if output_root.is_symlink():
        raise ValueError(f"blind audio output root must not be a symlink: {output_root}")
    for blind_id in sorted(review_paths):
        relative = Path(str(review_paths[blind_id]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe blind audio path for {blind_id}: {relative}")
        source = Path(str(reveal_rows[blind_id]["source_audio_path"]))
        if not source.is_file():
            raise ValueError(f"source audio does not exist for {blind_id}: {source}")
        destination = output_root / relative
        current = output_root
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"blind audio destination parent must not be a symlink: {current}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if destination.is_symlink():
            raise ValueError(f"blind audio destination must not be a symlink: {destination}")
        if destination.exists():
            destination_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if destination_digest != source_digest:
                raise ValueError(f"blind audio destination already exists with different content: {destination}")
        else:
            shutil.copyfile(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        staged.append(
            {
                "blind_id": blind_id,
                "audio_path": str(relative),
                "sha256": digest,
                "bytes": destination.stat().st_size,
            }
        )
    return {
        "schema_version": "1.0.0",
        "review_sha256": reveal["review_sha256"],
        "file_count": len(staged),
        "files": staged,
        "evidence_boundary": (
            "Staging removes candidate identifiers from visible paths. It does not strip embedded audio metadata "
            "or prove that reviewers remained blind to outside information."
        ),
    }


def _krippendorff_alpha_interval(values_by_item: dict[str, list[float]]) -> float | None:
    observed_numerator = 0.0
    observed_denominator = 0
    all_values: list[float] = []
    for values in values_by_item.values():
        all_values.extend(values)
        if len(values) < 2:
            continue
        observed_denominator += len(values)
        observed_numerator += sum(
            2 * (left - right) ** 2 / (len(values) - 1)
            for left_index, left in enumerate(values)
            for right in values[left_index + 1 :]
        )
    if observed_denominator == 0 or len(all_values) < 2:
        return None
    observed = observed_numerator / observed_denominator
    expected = sum(
        2 * (left - right) ** 2 / (len(all_values) - 1)
        for left_index, left in enumerate(all_values)
        for right in all_values[left_index + 1 :]
    ) / len(all_values)
    if expected == 0:
        return 1.0
    return 1.0 - observed / expected


def aggregate_listening_results(
    review: dict[str, Any],
    reveal: dict[str, Any],
    ratings_document: dict[str, Any],
    *,
    seed: int = 20260812,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    if _digest(review) != reveal.get("review_sha256"):
        raise ValueError("reveal mapping does not match the review document")
    criteria = review.get("criteria")
    if (
        not isinstance(criteria, list)
        or not criteria
        or any(not isinstance(value, str) or not value.strip() for value in criteria)
        or len(set(criteria)) != len(criteria)
    ):
        raise ValueError("review document has no criteria")
    scale = ratings_document.get("scale")
    ratings = ratings_document.get("ratings")
    if not isinstance(scale, dict) or not isinstance(ratings, list):
        raise ValueError("ratings document must contain scale and ratings")
    minimum = scale.get("min")
    maximum = scale.get("max")
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)) or minimum >= maximum:
        raise ValueError("ratings scale must contain numeric min smaller than max")
    expected_rater_values = ratings_document.get("expected_rater_ids")
    if expected_rater_values is None and not allow_incomplete:
        raise ValueError("ratings document must declare expected_rater_ids for a complete matrix")
    if expected_rater_values is not None:
        if (
            not isinstance(expected_rater_values, list)
            or not expected_rater_values
            or any(not isinstance(value, str) or not value.strip() for value in expected_rater_values)
        ):
            raise ValueError("expected_rater_ids must be a non-empty array of strings")
        expected_rater_ids = {value.strip() for value in expected_rater_values}
        if len(expected_rater_ids) != len(expected_rater_values):
            raise ValueError("expected_rater_ids must be unique")
    else:
        expected_rater_ids = set()

    mapping = {
        row["blind_id"]: row
        for row in reveal.get("mapping", [])
        if isinstance(row, dict) and {"blind_id", "candidate_id", "sample_id"} <= row.keys()
    }
    review_rows = review.get("samples")
    if not isinstance(review_rows, list) or not review_rows:
        raise ValueError("review document has no samples")
    review_blind_ids = {row["blind_id"] for row in review_rows if isinstance(row, dict) and "blind_id" in row}
    if len(review_blind_ids) != len(review_rows):
        raise ValueError("review document samples must have unique blind ids")
    if set(mapping) != review_blind_ids:
        raise ValueError("reveal mapping must cover every blind sample exactly once")
    criteria_by_blind: dict[str, list[str]] = {}
    for index, row in enumerate(review_rows):
        assigned = row.get("criteria", criteria)
        if (
            not isinstance(assigned, list)
            or not assigned
            or any(not isinstance(value, str) or value not in criteria for value in assigned)
            or len(set(assigned)) != len(assigned)
        ):
            raise ValueError(f"review sample {index} criteria must be a non-empty unique subset of review criteria")
        criteria_by_blind[row["blind_id"]] = assigned

    seen: set[tuple[str, str, str]] = set()
    candidate_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    agreement_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rater_ids: set[str] = set()
    for index, rating in enumerate(ratings):
        if not isinstance(rating, dict):
            raise ValueError(f"rating {index} must be an object")
        required = {"rater_id", "blind_id", "criterion", "score"}
        missing = required - rating.keys()
        if missing:
            raise ValueError(f"rating {index} is missing: {', '.join(sorted(missing))}")
        rater_id = str(rating["rater_id"]).strip()
        blind_id = str(rating["blind_id"]).strip()
        criterion = str(rating["criterion"]).strip()
        score = rating["score"]
        if not rater_id or blind_id not in mapping or criterion not in criteria:
            raise ValueError(f"rating {index} contains an unknown or empty identifier")
        if criterion not in criteria_by_blind[blind_id]:
            raise ValueError(f"rating {index} uses a criterion not assigned to sample {blind_id}: {criterion}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not minimum <= score <= maximum:
            raise ValueError(f"rating {index} score is outside the declared scale")
        key = (rater_id, blind_id, criterion)
        if key in seen:
            raise ValueError(f"duplicate rating for rater, sample, and criterion: {key}")
        seen.add(key)
        candidate_id = str(mapping[blind_id]["candidate_id"])
        candidate_values[(candidate_id, criterion)].append(float(score))
        sample_values[(blind_id, criterion)].append(float(score))
        agreement_values[criterion][blind_id].append(float(score))
        rater_ids.add(rater_id)

    if expected_rater_ids:
        unexpected_raters = sorted(rater_ids - expected_rater_ids)
        if unexpected_raters:
            raise ValueError(f"ratings contain undeclared raters: {', '.join(unexpected_raters)}")
    else:
        expected_rater_ids = set(rater_ids)

    expected_keys = {
        (rater_id, blind_id, criterion)
        for rater_id in expected_rater_ids
        for blind_id in review_blind_ids
        for criterion in criteria_by_blind[blind_id]
    }
    missing_keys = sorted(expected_keys - seen)
    if missing_keys and not allow_incomplete:
        preview = ", ".join("/".join(key) for key in missing_keys[:5])
        suffix = "" if len(missing_keys) <= 5 else f" and {len(missing_keys) - 5} more"
        raise ValueError(f"ratings matrix is incomplete: {preview}{suffix}")

    candidates: dict[str, dict[str, Any]] = defaultdict(dict)
    for index, ((candidate_id, criterion), values) in enumerate(sorted(candidate_values.items())):
        candidates[candidate_id][criterion] = {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "mean_95pct_bootstrap_ci": bootstrap_mean_interval(values, seed=seed + index),
        }
    agreement = {
        criterion: {
            "krippendorff_alpha_interval": _krippendorff_alpha_interval(values_by_item),
            "rated_items": len(values_by_item),
            "items_with_multiple_raters": sum(len(values) >= 2 for values in values_by_item.values()),
        }
        for criterion, values_by_item in sorted(agreement_values.items())
    }
    review_by_id = {
        row["blind_id"]: row
        for row in review.get("samples", [])
        if isinstance(row, dict) and "blind_id" in row
    }
    per_sample = []
    for (blind_id, criterion), values in sorted(sample_values.items()):
        reveal_row = mapping[blind_id]
        row = {
            "blind_id": blind_id,
            "sample_id": reveal_row["sample_id"],
            "candidate_id": reveal_row["candidate_id"],
            "prompt_id": review_by_id[blind_id].get("prompt_id"),
            "criterion": criterion,
            "rating_count": len(values),
            "mean": mean(values),
            "median": median(values),
        }
        if "seed" in reveal_row:
            row["seed"] = reveal_row["seed"]
        per_sample.append(row)
    return {
        "schema_version": "1.0.0",
        "revealed": True,
        "rating_scale": {"min": minimum, "max": maximum},
        "rater_count": len(rater_ids),
        "expected_rater_count": len(expected_rater_ids),
        "rating_count": len(ratings),
        "coverage": {
            "assignment_mode": "plan_bound" if "assignment_plan" in review else "all_criteria_per_sample",
            "status": "complete" if not missing_keys else "incomplete",
            "expected_rating_count": len(expected_keys),
            "observed_rating_count": len(seen),
            "missing_rating_count": len(missing_keys),
            "missing_ratings": [
                {"rater_id": rater_id, "blind_id": blind_id, "criterion": criterion}
                for rater_id, blind_id, criterion in missing_keys
            ],
        },
        "candidates": dict(sorted(candidates.items())),
        "samples": per_sample,
        "agreement": agreement,
        "evidence_boundary": (
            "Agreement measures rating consistency, not truth. Criterion summaries remain separate and do not "
            "prove general quality."
        ),
    }
