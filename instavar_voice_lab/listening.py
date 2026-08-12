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

LISTENING_ASSIGNMENT_PLAN_SCHEMA_VERSION = "1.2.0"
LISTENING_ROUTING_SCHEMA_VERSION = "1.1.0"
LISTENING_PRESENTATION_SCHEDULE_SCHEMA_VERSION = "1.0.0"
LISTENING_RATER_PACKET_SCHEMA_VERSION = "1.0.0"
LISTENING_RATER_SUBMISSION_SCHEMA_VERSION = "1.0.0"
_ROUTING_SELECTORS = {"all_samples", "categories", "categories_or_lexical_anchors", "lexical_anchors"}
_CRITERION_DIRECTIONS = {"higher_is_better", "lower_is_better"}


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
        direction = raw.get("direction")
        if direction not in _CRITERION_DIRECTIONS:
            raise ValueError(
                f"listening routing route {criterion} direction must be one of: "
                f"{', '.join(sorted(_CRITERION_DIRECTIONS))}"
            )
        normalized = {"criterion": criterion, "selector": selector, "direction": direction}
        for field in ("review_prompt", "low_label", "high_label"):
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"listening routing route {criterion} {field} must be a non-empty string")
            normalized[field] = value.strip()
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


def _review_stimulus(raw: dict[str, Any], *, index: int) -> dict[str, Any]:
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"generation plan sample {index} text must be a non-empty string")
    stimulus: dict[str, Any] = {"text": text}
    instruction = raw.get("instruction")
    if instruction is not None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"generation plan sample {index} instruction must be a non-empty string when present")
        stimulus["instruction"] = instruction
    anchors = raw.get("lexical_anchors")
    if anchors is not None:
        if not isinstance(anchors, list) or not anchors:
            raise ValueError(f"generation plan sample {index} lexical_anchors must be a non-empty array when present")
        targets: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for anchor_index, anchor in enumerate(anchors):
            if not isinstance(anchor, dict):
                raise ValueError(
                    f"generation plan sample {index} lexical anchor {anchor_index} must be an object"
                )
            anchor_id = anchor.get("anchor_id")
            surface = anchor.get("surface")
            if not isinstance(anchor_id, str) or not anchor_id.strip():
                raise ValueError(
                    f"generation plan sample {index} lexical anchor {anchor_index} anchor_id must be non-empty"
                )
            if not isinstance(surface, str) or not surface.strip():
                raise ValueError(
                    f"generation plan sample {index} lexical anchor {anchor_index} surface must be non-empty"
                )
            if anchor_id in seen_ids:
                raise ValueError(f"generation plan sample {index} lexical anchor ids must be unique")
            seen_ids.add(anchor_id)
            targets.append({"anchor_id": anchor_id, "surface": surface})
        stimulus["lexical_targets"] = targets
    return stimulus


def _generation_rows(generation_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(generation_plan, dict) or generation_plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("generation plan schema_version must equal 1.0.0 or 1.1.0")
    raw_samples = generation_plan.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("generation plan samples must be a non-empty array")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    candidate_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
    prompt_shapes: dict[str, tuple[str, str, str]] = {}
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
        stimulus = _review_stimulus(raw, index=index)
        shape = (category, _digest(stimulus), _digest(raw.get("lexical_anchors")))
        if prompt_id in prompt_shapes and prompt_shapes[prompt_id] != shape:
            raise ValueError(f"generation plan listening-relevant fields drift across candidates or seeds: {prompt_id}")
        prompt_shapes[prompt_id] = shape
        row = dict(raw)
        row["_review_stimulus"] = stimulus
        rows.append(row)
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
                "stimulus": row["_review_stimulus"],
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
    criterion_definitions = [
        {
            field: route[field]
            for field in ("criterion", "direction", "review_prompt", "low_label", "high_label")
        }
        for route in routes
    ]
    return {
        "schema_version": LISTENING_ASSIGNMENT_PLAN_SCHEMA_VERSION,
        "generation_plan_sha256": _digest(generation_plan),
        "criteria": [route["criterion"] for route in routes],
        "criterion_definitions": criterion_definitions,
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


def _presentation_schedule_audit(
    schedules: list[dict[str, Any]],
    reveal_rows: Any,
    *,
    seed: Any,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("counterbalance audit seed must be a non-negative integer")
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("counterbalance audit schedules must be a non-empty array")
    if not isinstance(reveal_rows, list) or not reveal_rows:
        raise ValueError("counterbalance audit mapping must be a non-empty array")
    mapping: dict[str, tuple[tuple[str, int], str]] = {}
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for index, row in enumerate(reveal_rows):
        if not isinstance(row, dict):
            raise ValueError(f"counterbalance audit mapping row {index} must be an object")
        seed_value = row.get("seed")
        candidate_id = str(row.get("candidate_id", "")).strip()
        prompt_id = str(row.get("prompt_id", "")).strip()
        blind_id = str(row.get("blind_id", "")).strip()
        if (
            isinstance(seed_value, bool)
            or not isinstance(seed_value, int)
            or seed_value < 0
            or not candidate_id
            or not prompt_id
            or not blind_id
            or blind_id in mapping
        ):
            raise ValueError(f"counterbalance audit mapping row {index} is invalid")
        key = (prompt_id, seed_value)
        mapping[blind_id] = (key, candidate_id)
        if candidate_id in groups[key]:
            raise ValueError(f"counterbalance audit repeats candidate for {prompt_id}/{seed_value}")
        groups[key].add(candidate_id)
    candidates = sorted({candidate_id for _, candidate_id in mapping.values()})
    block_keys = sorted(groups)
    if len(candidates) < 2 or not block_keys:
        raise ValueError("counterbalance audit requires at least two candidates and one prompt-seed block")
    if any(groups[key] != set(candidates) for key in block_keys):
        raise ValueError("counterbalance audit requires symmetric candidate coverage")
    block_count = len(block_keys)
    candidate_count = len(candidates)
    position_counts = {
        key: {candidate_id: [0] * candidate_count for candidate_id in candidates} for key in block_keys
    }
    rater_pass_rows: list[dict[str, Any]] = []
    separations: list[int] = []
    for schedule_index, schedule in enumerate(schedules):
        if not isinstance(schedule, dict):
            raise ValueError(f"counterbalance audit schedule {schedule_index} must be an object")
        sample_order = schedule.get("sample_order")
        if not isinstance(sample_order, list) or len(sample_order) != len(mapping) or set(sample_order) != set(mapping):
            raise ValueError(f"counterbalance audit schedule {schedule_index} must cover every mapped sample")
        pass_counts = {candidate_id: [0] * candidate_count for candidate_id in candidates}
        keys_by_pass = [set() for _ in candidates]
        positions_by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
        for absolute_position, blind_id in enumerate(sample_order):
            if blind_id not in mapping:
                raise ValueError(f"counterbalance audit schedule {schedule_index} contains an unknown blind id")
            candidate_position = absolute_position // block_count
            if candidate_position >= candidate_count:
                raise ValueError(f"counterbalance audit schedule {schedule_index} has too many listening passes")
            key, candidate_id = mapping[blind_id]
            if key in keys_by_pass[candidate_position]:
                raise ValueError(f"counterbalance audit schedule {schedule_index} repeats a prompt block in one pass")
            keys_by_pass[candidate_position].add(key)
            position_counts[key][candidate_id][candidate_position] += 1
            pass_counts[candidate_id][candidate_position] += 1
            positions_by_key[key].append(absolute_position)
        if any(keys != set(block_keys) for keys in keys_by_pass):
            raise ValueError(f"counterbalance audit schedule {schedule_index} does not cover every block in each pass")
        for key in block_keys:
            positions = sorted(positions_by_key[key])
            if len(positions) != candidate_count:
                raise ValueError(f"counterbalance audit schedule {schedule_index} has incomplete matched candidates")
            separations.extend(right - left for left, right in zip(positions, positions[1:]))
        rater_pass_rows.append(
            {"rater_id": schedule["rater_id"], "candidate_pass_counts": pass_counts}
        )

    position_rows = []
    imbalances: list[int] = []
    for prompt_id, seed_value in block_keys:
        counts = position_counts[(prompt_id, seed_value)]
        position_rows.append(
            {"prompt_id": prompt_id, "seed": seed_value, "candidate_position_counts": counts}
        )
        imbalances.extend(max(counts[candidate_id]) - min(counts[candidate_id]) for candidate_id in candidates)
    max_imbalance = max(imbalances, default=0)
    pass_imbalances = [
        max(counts) - min(counts)
        for row in rater_pass_rows
        for counts in row["candidate_pass_counts"].values()
    ]
    max_pass_imbalance = max(pass_imbalances, default=0)
    if max_imbalance > 1 or max_pass_imbalance > 1:
        raise ValueError("counterbalance audit imbalance exceeds one")
    return {
        "schema_version": LISTENING_PRESENTATION_SCHEDULE_SCHEMA_VERSION,
        "method": "interleaved_prompt_seed_blocks_with_cyclic_candidate_precedence",
        "schedule_seed": seed,
        "rater_count": len(schedules),
        "candidate_count": candidate_count,
        "prompt_seed_count": block_count,
        "max_candidate_position_imbalance": max_imbalance,
        "max_candidate_pass_imbalance": max_pass_imbalance,
        "master_order_matched_candidate_minimum_separation": min(separations, default=0),
        "status": "passed",
        "candidate_position_counts": position_rows,
        "rater_candidate_pass_counts": rater_pass_rows,
        "evidence_boundary": (
            "The audit balances candidate precedence within each prompt and seed and candidate exposure across each "
            "rater's listening passes to within one. It does not eliminate sequence, learning, fatigue, or carryover "
            "effects or prove that reviewers followed the schedule."
        ),
    }


def _counterbalanced_presentation_schedules(
    review_rows: list[dict[str, Any]],
    reveal_rows: list[dict[str, Any]],
    criteria: list[str],
    rater_ids: list[str],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        not isinstance(rater_ids, list)
        or not rater_ids
        or any(not isinstance(value, str) or not value.strip() for value in rater_ids)
    ):
        raise ValueError("rater_ids must be a non-empty array of non-empty strings")
    normalized_raters = sorted(value.strip() for value in rater_ids)
    if len(set(normalized_raters)) != len(normalized_raters):
        raise ValueError("rater_ids must be unique")

    review_by_id = {row["blind_id"]: row for row in review_rows}
    groups: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    candidate_ids: set[str] = set()
    for index, row in enumerate(reveal_rows):
        seed_value = row.get("seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
            raise ValueError(f"counterbalanced sample {index} must contain a non-negative seed")
        candidate_id = str(row.get("candidate_id", "")).strip()
        prompt_id = str(row.get("prompt_id", "")).strip()
        blind_id = str(row.get("blind_id", "")).strip()
        if not candidate_id or not prompt_id or blind_id not in review_by_id:
            raise ValueError(f"counterbalanced sample {index} contains an unknown or empty identifier")
        key = (prompt_id, seed_value)
        if candidate_id in groups[key]:
            raise ValueError(
                f"counterbalanced prompt and seed repeats candidate {candidate_id}: {prompt_id}/{seed_value}"
            )
        groups[key][candidate_id] = blind_id
        candidate_ids.add(candidate_id)

    candidates = sorted(candidate_ids)
    if len(candidates) < 2:
        raise ValueError("counterbalanced presentation requires at least two candidates")
    expected_candidates = set(candidates)
    for prompt_id, seed_value in sorted(groups):
        if set(groups[(prompt_id, seed_value)]) != expected_candidates:
            raise ValueError(
                f"counterbalanced presentation requires symmetric candidates for {prompt_id}/{seed_value}"
            )

    block_keys = sorted(groups)
    stable_block_indexes = {key: index for index, key in enumerate(block_keys)}
    schedules: list[dict[str, Any]] = []
    all_blind_ids = set(review_by_id)
    for rater_index, rater_id in enumerate(normalized_raters):
        rater_blocks = list(block_keys)
        rater_seed = int(_digest({"schedule_seed": seed, "rater_id": rater_id})[:16], 16)
        random.Random(rater_seed).shuffle(rater_blocks)
        sample_order: list[str] = []
        for candidate_position in range(len(candidates)):
            for key in rater_blocks:
                candidate_index = (candidate_position + rater_index + stable_block_indexes[key]) % len(candidates)
                candidate_id = candidates[candidate_index]
                sample_order.append(groups[key][candidate_id])
        if len(sample_order) != len(all_blind_ids) or set(sample_order) != all_blind_ids:
            raise ValueError(f"counterbalanced presentation does not cover every sample for rater {rater_id}")
        criterion_orders = {
            criterion: [
                blind_id for blind_id in sample_order if criterion in review_by_id[blind_id].get("criteria", criteria)
            ]
            for criterion in criteria
        }
        schedule_payload = {
            "rater_id": rater_id,
            "sample_order": sample_order,
            "criterion_orders": criterion_orders,
        }
        schedules.append({**schedule_payload, "schedule_sha256": _digest(schedule_payload)})
    audit = _presentation_schedule_audit(schedules, reveal_rows, seed=seed)
    return schedules, audit


def _validate_presentation_schedules(
    review: dict[str, Any], criteria: list[str], review_rows: list[dict[str, Any]]
) -> set[str] | None:
    raw_schedules = review.get("presentation_schedules")
    if raw_schedules is None:
        return None
    if review.get("presentation_schedule_schema_version") != LISTENING_PRESENTATION_SCHEDULE_SCHEMA_VERSION:
        raise ValueError(
            "review presentation_schedule_schema_version must equal "
            f"{LISTENING_PRESENTATION_SCHEDULE_SCHEMA_VERSION}"
        )
    if not isinstance(raw_schedules, list) or not raw_schedules:
        raise ValueError("review presentation_schedules must be a non-empty array")
    review_by_id = {row["blind_id"]: row for row in review_rows}
    expected_blind_ids = set(review_by_id)
    scheduled_raters: set[str] = set()
    for index, schedule in enumerate(raw_schedules):
        if not isinstance(schedule, dict):
            raise ValueError(f"review presentation schedule {index} must be an object")
        rater_id = schedule.get("rater_id")
        if not isinstance(rater_id, str) or not rater_id.strip() or rater_id in scheduled_raters:
            raise ValueError(f"review presentation schedule {index} has an empty or duplicate rater_id")
        scheduled_raters.add(rater_id)
        payload = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
        if schedule.get("schedule_sha256") != _digest(payload):
            raise ValueError(f"review presentation schedule {index} self-hash does not match its content")
        sample_order = schedule.get("sample_order")
        if (
            not isinstance(sample_order, list)
            or len(sample_order) != len(expected_blind_ids)
            or set(sample_order) != expected_blind_ids
        ):
            raise ValueError(f"review presentation schedule {index} must cover every blind sample exactly once")
        criterion_orders = schedule.get("criterion_orders")
        if not isinstance(criterion_orders, dict) or set(criterion_orders) != set(criteria):
            raise ValueError(f"review presentation schedule {index} criterion_orders must cover every criterion")
        for criterion in criteria:
            expected_order = [
                blind_id
                for blind_id in sample_order
                if criterion in review_by_id[blind_id].get("criteria", criteria)
            ]
            if criterion_orders[criterion] != expected_order:
                raise ValueError(
                    f"review presentation schedule {index} criterion order does not match assigned samples: {criterion}"
                )
    return scheduled_raters


def build_rater_review_packet(review: dict[str, Any], rater_id: str) -> dict[str, Any]:
    if not isinstance(review, dict):
        raise ValueError("review must be an object")
    if not isinstance(rater_id, str) or not rater_id.strip():
        raise ValueError("rater_id must be a non-empty string")
    rater_id = rater_id.strip()
    criteria = review.get("criteria")
    review_rows = review.get("samples")
    if (
        not isinstance(criteria, list)
        or not criteria
        or any(not isinstance(value, str) or not value.strip() for value in criteria)
        or len(set(criteria)) != len(criteria)
        or not isinstance(review_rows, list)
        or not review_rows
    ):
        raise ValueError("review must contain criteria and samples")
    scheduled_raters = _validate_presentation_schedules(review, criteria, review_rows)
    if scheduled_raters is None:
        raise ValueError("review does not contain per-rater presentation schedules")
    if rater_id not in scheduled_raters:
        raise ValueError(f"review has no presentation schedule for rater_id: {rater_id}")
    schedule = next(row for row in review["presentation_schedules"] if row["rater_id"] == rater_id)
    review_by_id = {row["blind_id"]: row for row in review_rows}
    rating_order = [
        {"criterion": criterion, "blind_id": blind_id}
        for criterion in criteria
        for blind_id in schedule["criterion_orders"][criterion]
    ]
    payload = {
        "schema_version": LISTENING_RATER_PACKET_SCHEMA_VERSION,
        "master_review_sha256": _digest(review),
        "presentation_schedule_schema_version": LISTENING_PRESENTATION_SCHEDULE_SCHEMA_VERSION,
        "rater_id": rater_id,
        "schedule_sha256": schedule["schedule_sha256"],
        "criteria": list(criteria),
        "criterion_definitions": review.get("criterion_definitions"),
        "instructions": review.get("instructions"),
        "sample_order": list(schedule["sample_order"]),
        "criterion_orders": dict(schedule["criterion_orders"]),
        "rating_order": rating_order,
        "samples": [dict(review_by_id[blind_id]) for blind_id in schedule["sample_order"]],
        "evidence_boundary": (
            "This packet contains only one pseudonymous rater schedule and blind review items. It does not prove "
            "that the intended reviewer received it or followed its order."
        ),
    }
    return {**payload, "packet_sha256": _digest(payload)}


def _validate_rater_packet(packet: Any) -> tuple[str, dict[str, list[str]]]:
    if not isinstance(packet, dict) or packet.get("schema_version") != LISTENING_RATER_PACKET_SCHEMA_VERSION:
        raise ValueError(f"rater packet schema_version must equal {LISTENING_RATER_PACKET_SCHEMA_VERSION}")
    payload = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != _digest(payload):
        raise ValueError("rater packet self-hash does not match its content")
    rater_id = packet.get("rater_id")
    if not isinstance(rater_id, str) or not rater_id.strip():
        raise ValueError("rater packet rater_id must be a non-empty string")
    sample_order = packet.get("sample_order")
    samples = packet.get("samples")
    if (
        not isinstance(sample_order, list)
        or not sample_order
        or any(not isinstance(value, str) or not value for value in sample_order)
        or not isinstance(samples, list)
    ):
        raise ValueError("rater packet must contain sample_order and samples")
    sample_by_id = {
        row.get("blind_id"): row
        for row in samples
        if isinstance(row, dict) and isinstance(row.get("blind_id"), str)
    }
    if len(sample_by_id) != len(samples) or sample_order != [row.get("blind_id") for row in samples]:
        raise ValueError("rater packet samples must exactly follow sample_order")
    criteria = packet.get("criteria")
    if (
        not isinstance(criteria, list)
        or not criteria
        or any(not isinstance(value, str) or not value.strip() for value in criteria)
        or len(set(criteria)) != len(criteria)
    ):
        raise ValueError("rater packet criteria must be a non-empty unique array")
    criterion_orders = packet.get("criterion_orders")
    if not isinstance(criterion_orders, dict) or set(criterion_orders) != set(criteria):
        raise ValueError("rater packet criterion_orders must cover every criterion")
    criteria_by_blind: dict[str, list[str]] = {}
    for blind_id in sample_order:
        row = sample_by_id.get(blind_id)
        assigned = row.get("criteria", criteria) if row else None
        if (
            not isinstance(assigned, list)
            or not assigned
            or any(not isinstance(value, str) or value not in criteria for value in assigned)
            or len(set(assigned)) != len(assigned)
        ):
            raise ValueError(f"rater packet sample has invalid criteria: {blind_id}")
        criteria_by_blind[blind_id] = assigned
    for criterion in criteria:
        expected = [blind_id for blind_id in sample_order if criterion in criteria_by_blind[blind_id]]
        if criterion_orders[criterion] != expected:
            raise ValueError(f"rater packet criterion order does not match assigned samples: {criterion}")
    expected_rating_order = [
        {"criterion": criterion, "blind_id": blind_id}
        for criterion in criteria
        for blind_id in criterion_orders[criterion]
    ]
    if packet.get("rating_order") != expected_rating_order:
        raise ValueError("rater packet rating_order must follow criterion_orders")
    return rater_id, criteria_by_blind


def build_rater_submission(
    packet: dict[str, Any], ratings_document: dict[str, Any], *, allow_incomplete: bool = False
) -> dict[str, Any]:
    rater_id, criteria_by_blind = _validate_rater_packet(packet)
    if not isinstance(ratings_document, dict):
        raise ValueError("rater ratings must be an object")
    scale = ratings_document.get("scale")
    ratings = ratings_document.get("ratings")
    presentation_log = ratings_document.get("presentation_log")
    if not isinstance(scale, dict) or not isinstance(ratings, list) or not isinstance(presentation_log, list):
        raise ValueError("rater ratings must contain scale, ratings, and presentation_log")
    minimum = scale.get("min")
    maximum = scale.get("max")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
        or minimum >= maximum
    ):
        raise ValueError("rater ratings scale must contain numeric min smaller than max")
    rating_order = packet["rating_order"]
    if (
        any(
            not isinstance(value, dict)
            or set(value) != {"criterion", "blind_id"}
            or not isinstance(value["criterion"], str)
            or not isinstance(value["blind_id"], str)
            for value in presentation_log
        )
        or len({_digest(value) for value in presentation_log}) != len(presentation_log)
        or presentation_log != rating_order[: len(presentation_log)]
    ):
        raise ValueError("rater presentation_log must be an exact prefix of the assigned rating order")
    if not allow_incomplete and presentation_log != rating_order:
        raise ValueError("rater presentation_log is incomplete")
    presented = {(row["blind_id"], row["criterion"]) for row in presentation_log}
    normalized_ratings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, rating in enumerate(ratings):
        if not isinstance(rating, dict) or "rater_id" in rating:
            raise ValueError(f"rater rating {index} must be an object without rater_id")
        required = {"blind_id", "criterion", "score"}
        if set(rating) != required:
            raise ValueError(f"rater rating {index} must contain exactly blind_id, criterion, and score")
        blind_id = rating["blind_id"]
        criterion = rating["criterion"]
        score = rating["score"]
        if (blind_id, criterion) not in presented or criterion not in criteria_by_blind.get(blind_id, []):
            raise ValueError(f"rater rating {index} is outside the presented assigned matrix")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not minimum <= score <= maximum:
            raise ValueError(f"rater rating {index} score is outside the declared scale")
        key = (blind_id, criterion)
        if key in seen:
            raise ValueError(f"duplicate rater rating for sample and criterion: {key}")
        seen.add(key)
        normalized_ratings.append({"blind_id": blind_id, "criterion": criterion, "score": score})
    rating_positions = {
        (row["blind_id"], row["criterion"]): index for index, row in enumerate(rating_order)
    }
    normalized_ratings.sort(key=lambda row: rating_positions[(row["blind_id"], row["criterion"])])
    expected = {(row["blind_id"], row["criterion"]) for row in rating_order}
    missing = sorted(expected - seen)
    if missing and not allow_incomplete:
        raise ValueError("rater ratings matrix is incomplete")
    payload = {
        "schema_version": LISTENING_RATER_SUBMISSION_SCHEMA_VERSION,
        "rater_id": rater_id,
        "packet_sha256": packet["packet_sha256"],
        "schedule_sha256": packet["schedule_sha256"],
        "scale": {"min": minimum, "max": maximum},
        "presentation_log": list(presentation_log),
        "ratings": normalized_ratings,
        "coverage": {
            "status": "complete" if presentation_log == rating_order and not missing else "incomplete",
            "expected_rating_count": len(expected),
            "presented_rating_count": len(presented),
            "observed_rating_count": len(seen),
            "missing_rating_count": len(missing),
        },
        "compliance_boundary": (
            "The presentation log is a self-attested receipt bound to this packet. It does not independently "
            "prove delivery identity, listening order, attention, or reviewer independence."
        ),
    }
    return {**payload, "submission_sha256": _digest(payload)}


def _ratings_from_rater_submissions(
    review: dict[str, Any], ratings_document: dict[str, Any], *, allow_incomplete: bool
) -> dict[str, Any] | None:
    if not isinstance(ratings_document, dict):
        raise ValueError("ratings document must be an object")
    submissions = ratings_document.get("submissions")
    if submissions is None:
        return None
    if ratings_document.get("schema_version") != "1.1.0" or not isinstance(submissions, list):
        raise ValueError("rater submission bundle must use schema 1.1.0 and contain a submissions array")
    bundle_scale = ratings_document.get("scale")
    if not isinstance(bundle_scale, dict):
        raise ValueError("rater submission bundle must declare one shared scale")
    if not submissions and not allow_incomplete:
        raise ValueError("rater submission bundle must contain every scheduled rater")
    schedules = review.get("presentation_schedules")
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("rater submissions require review presentation schedules")
    expected_raters: set[str] = set()
    for index, schedule in enumerate(schedules):
        rater_id = schedule.get("rater_id") if isinstance(schedule, dict) else None
        if not isinstance(rater_id, str) or not rater_id.strip() or rater_id in expected_raters:
            raise ValueError(f"review presentation schedule {index} has an empty or duplicate rater_id")
        expected_raters.add(rater_id)
    seen_raters: set[str] = set()
    scales: set[str] = set()
    normalized: list[dict[str, Any]] = []
    ordered_submissions = sorted(
        submissions,
        key=lambda row: str(row.get("rater_id", "")) if isinstance(row, dict) else "",
    )
    canonical_submissions: list[dict[str, Any]] = []
    for index, submission in enumerate(ordered_submissions):
        if not isinstance(submission, dict):
            raise ValueError(f"rater submission {index} must be an object")
        rater_id = submission.get("rater_id")
        if rater_id not in expected_raters or rater_id in seen_raters:
            raise ValueError(f"rater submission {index} has an unknown or duplicate rater_id")
        seen_raters.add(rater_id)
        packet = build_rater_review_packet(review, rater_id)
        source = {
            "scale": submission.get("scale"),
            "presentation_log": submission.get("presentation_log"),
            "ratings": submission.get("ratings"),
        }
        expected_submission = build_rater_submission(packet, source, allow_incomplete=allow_incomplete)
        if submission != expected_submission:
            raise ValueError(f"rater submission {index} does not match its review packet")
        if expected_submission["scale"] != bundle_scale:
            raise ValueError("rater submissions must match the bundle rating scale")
        canonical_submissions.append(expected_submission)
        scales.add(_digest(expected_submission["scale"]))
        normalized.extend({"rater_id": rater_id, **row} for row in expected_submission["ratings"])
    if len(scales) > 1:
        raise ValueError("rater submissions must use one shared rating scale")
    if not allow_incomplete and seen_raters != expected_raters:
        raise ValueError("rater submission bundle must contain every scheduled rater")
    return {
        "scale": bundle_scale,
        "expected_rater_ids": sorted(expected_raters),
        "ratings": normalized,
        "submission_receipts_sha256": _digest(canonical_submissions),
    }


def build_blind_pack(
    samples: list[dict[str, Any]],
    criteria: list[str] | None,
    *,
    seed: int,
    assignment_plan: dict[str, Any] | None = None,
    generation_plan: dict[str, Any] | None = None,
    rater_ids: list[str] | None = None,
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
    elif rater_ids is not None:
        raise ValueError("assignment_plan is required with rater_ids")
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
            blind_row["stimulus"] = planned[str(sample["sample_id"])]["stimulus"]
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
        review["criterion_definitions"] = assignment_plan["criterion_definitions"]
        review["stimulus_boundary"] = (
            "Stimuli reproduce generation-plan text, instructions, and lexical target surfaces. Accepted ASR forms "
            "are intentionally excluded, and a target surface does not prescribe its correct pronunciation."
        )
        review["instructions"] = (
            "Rate only the criteria assigned to each sample, one criterion at a time. Review only the blind_audio "
            "paths and do not open the reveal mapping until every assigned rating is recorded."
        )
    counterbalance_audit: dict[str, Any] | None = None
    if rater_ids is not None:
        schedules, counterbalance_audit = _counterbalanced_presentation_schedules(
            blind_rows,
            reveal_rows,
            criteria,
            rater_ids,
            seed=seed,
        )
        review["schema_version"] = "1.1.0"
        review["presentation_schedule_schema_version"] = LISTENING_PRESENTATION_SCHEDULE_SCHEMA_VERSION
        review["presentation_schedules"] = schedules
        review["counterbalance_audit_sha256"] = _digest(counterbalance_audit)
        review["presentation_boundary"] = (
            "Schedules use pseudonymous rater ids and blind ids only. They counterbalance candidate precedence but "
            "cannot prove reviewer compliance or eliminate all order and carryover effects."
        )
        review["instructions"] = (
            "Use only the presentation schedule for your assigned pseudonymous rater id. Rate one criterion at a "
            "time in its criterion_orders sequence. Review only blind_audio paths and do not open the reveal mapping "
            "until every assigned rating is recorded."
        )
    reveal = {
        "schema_version": "1.0.0",
        "seed": seed,
        "review_sha256": _digest(review),
        "mapping": reveal_rows,
    }
    if counterbalance_audit is not None:
        reveal["counterbalance_audit"] = counterbalance_audit
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
    if not isinstance(review, dict) or not isinstance(reveal, dict):
        raise ValueError("review and reveal documents must be objects")
    if _digest(review) != reveal.get("review_sha256"):
        raise ValueError("reveal mapping does not match the review document")
    normalized_submissions = _ratings_from_rater_submissions(
        review, ratings_document, allow_incomplete=allow_incomplete
    )
    uses_rater_submissions = normalized_submissions is not None
    if normalized_submissions is not None:
        ratings_document = normalized_submissions
    criteria = review.get("criteria")
    if (
        not isinstance(criteria, list)
        or not criteria
        or any(not isinstance(value, str) or not value.strip() for value in criteria)
        or len(set(criteria)) != len(criteria)
    ):
        raise ValueError("review document has no criteria")
    raw_definitions = review.get("criterion_definitions")
    if raw_definitions is None:
        criterion_definitions = {
            criterion: {"criterion": criterion, "direction": "unspecified"} for criterion in criteria
        }
    else:
        if not isinstance(raw_definitions, list) or len(raw_definitions) != len(criteria):
            raise ValueError("review criterion_definitions must cover every criterion exactly once")
        criterion_definitions = {}
        for index, definition in enumerate(raw_definitions):
            if not isinstance(definition, dict):
                raise ValueError(f"review criterion definition {index} must be an object")
            criterion = definition.get("criterion")
            direction = definition.get("direction")
            if criterion not in criteria or criterion in criterion_definitions:
                raise ValueError(f"review criterion definition {index} has an unknown or duplicate criterion")
            if direction not in _CRITERION_DIRECTIONS:
                raise ValueError(f"review criterion definition {index} has an invalid direction")
            for field in ("review_prompt", "low_label", "high_label"):
                if not isinstance(definition.get(field), str) or not definition[field].strip():
                    raise ValueError(f"review criterion definition {index} has an invalid {field}")
            criterion_definitions[criterion] = dict(definition)
    scale = ratings_document.get("scale")
    ratings = ratings_document.get("ratings")
    if not isinstance(scale, dict) or not isinstance(ratings, list):
        raise ValueError("ratings document must contain scale and ratings")
    minimum = scale.get("min")
    maximum = scale.get("max")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
        or minimum >= maximum
    ):
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
    scheduled_rater_ids = _validate_presentation_schedules(review, criteria, review_rows)
    if scheduled_rater_ids is not None:
        counterbalance_audit = reveal.get("counterbalance_audit")
        expected_audit = _presentation_schedule_audit(
            review["presentation_schedules"],
            reveal.get("mapping", []),
            seed=reveal.get("seed"),
        )
        if (
            not isinstance(counterbalance_audit, dict)
            or review.get("counterbalance_audit_sha256") != _digest(counterbalance_audit)
            or counterbalance_audit.get("status") != "passed"
            or counterbalance_audit != expected_audit
        ):
            raise ValueError("reveal counterbalance audit does not match the review document")
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
    if scheduled_rater_ids is not None and expected_rater_ids != scheduled_rater_ids:
        raise ValueError("ratings expected_rater_ids must exactly match the review presentation schedules")

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
        "criterion_definitions": [criterion_definitions[criterion] for criterion in criteria],
        "presentation": {
            "mode": "counterbalanced_per_rater" if scheduled_rater_ids is not None else "shared_global_order",
            "scheduled_rater_count": len(scheduled_rater_ids or set()),
            "counterbalance_audit_sha256": review.get("counterbalance_audit_sha256"),
            **(
                {
                    "submission_receipts_sha256": ratings_document["submission_receipts_sha256"],
                    "receipt_evidence_boundary": (
                        "Receipt hashes bind declared content to reconstructed packets but are not signatures and "
                        "do not prove reviewer identity, delivery, order compliance, attention, or independence."
                    ),
                }
                if uses_rater_submissions
                else {}
            ),
        },
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
            "prove general quality. Direction metadata prevents inverted interpretation but does not justify "
            "combining distinct criteria into a composite."
        ),
    }
