from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any

from .metrics import bootstrap_mean_interval, score_objective_observations


METRIC_DIRECTIONS = {
    "asr_word_error_rate": "lower_is_better",
    "speaker_embedding_similarity": "higher_is_better",
    "real_time_factor": "lower_is_better",
    "generation_seconds": "lower_is_better",
    "peak_memory_bytes": "lower_is_better",
    "audio_duration_seconds": "context_only",
}

METRIC_EVIDENCE_KINDS = {
    "asr_word_error_rate": "asr",
    "speaker_embedding_similarity": "speaker_encoder",
    "real_time_factor": "runtime",
    "generation_seconds": "runtime",
    "peak_memory_bytes": "runtime",
    "audio_duration_seconds": "runtime",
}


def _pair_key(row: dict[str, Any], index: int) -> tuple[str, int]:
    prompt_id = row.get("prompt_id")
    seed = row.get("seed")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError(f"observation {index} prompt_id must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"observation {index} seed must be a non-negative integer for matched comparison")
    return prompt_id.strip(), seed


def _evidence_signature(row: dict[str, Any], kind: str, index: int) -> tuple[str, str] | None:
    evidence = row.get("evidence")
    if not isinstance(evidence, dict) or kind not in evidence:
        return None
    record = evidence[kind]
    if not isinstance(record, dict):
        raise ValueError(f"observation {index} evidence.{kind} must be an object")
    extractor = record.get("extractor")
    revision = record.get("revision")
    if not isinstance(extractor, str) or not extractor.strip():
        raise ValueError(f"observation {index} evidence.{kind}.extractor must be non-empty")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"observation {index} evidence.{kind}.revision must be non-empty")
    return extractor.strip(), revision.strip()


def _provenance(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    by_kind: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for index, row in enumerate(rows):
        for kind in set(METRIC_EVIDENCE_KINDS.values()):
            signature = _evidence_signature(row, kind, index)
            if signature is not None:
                by_kind[kind].add(signature)
    mixed = {kind: sorted(signatures) for kind, signatures in by_kind.items() if len(signatures) > 1}
    if mixed:
        details = "; ".join(
            f"{kind}={','.join(f'{extractor}@{revision}' for extractor, revision in signatures)}"
            for kind, signatures in sorted(mixed.items())
        )
        raise ValueError(f"matched comparison cannot mix extractor provenance: {details}")
    return {
        kind: {"extractor": next(iter(signatures))[0], "revision": next(iter(signatures))[1]}
        for kind, signatures in sorted(by_kind.items())
        if signatures
    }


def compare_matched_candidates(
    rows: list[dict[str, Any]],
    *,
    baseline_candidate_id: str,
    adapted_candidate_id: str,
    seed: int = 20260812,
) -> dict[str, Any]:
    if not baseline_candidate_id or not adapted_candidate_id:
        raise ValueError("baseline and adapted candidate ids must be non-empty")
    if baseline_candidate_id == adapted_candidate_id:
        raise ValueError("baseline and adapted candidate ids must differ")
    selected = [row for row in rows if row.get("candidate_id") in {baseline_candidate_id, adapted_candidate_id}]
    if not selected:
        raise ValueError("no observations matched the requested candidates")

    grouped: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        baseline_candidate_id: {},
        adapted_candidate_id: {},
    }
    for index, row in enumerate(selected):
        if not isinstance(row, dict):
            raise ValueError(f"observation {index} must be an object")
        candidate_id = str(row.get("candidate_id", ""))
        key = _pair_key(row, index)
        if key in grouped[candidate_id]:
            raise ValueError(f"duplicate matched observation for {candidate_id}, prompt {key[0]}, seed {key[1]}")
        grouped[candidate_id][key] = row

    baseline_keys = set(grouped[baseline_candidate_id])
    adapted_keys = set(grouped[adapted_candidate_id])
    if baseline_keys != adapted_keys:
        missing_adapted = sorted(baseline_keys - adapted_keys)
        missing_baseline = sorted(adapted_keys - baseline_keys)
        raise ValueError(
            "matched comparison requires identical prompt and seed coverage; "
            f"missing adapted={missing_adapted}; missing baseline={missing_baseline}"
        )
    if not baseline_keys:
        raise ValueError("matched comparison requires at least one prompt and seed pair")

    for key in sorted(baseline_keys):
        baseline_text = str(grouped[baseline_candidate_id][key].get("requested_text", "")).strip()
        adapted_text = str(grouped[adapted_candidate_id][key].get("requested_text", "")).strip()
        if not baseline_text or baseline_text != adapted_text:
            raise ValueError(f"requested_text mismatch for prompt {key[0]}, seed {key[1]}")

    provenance = _provenance(selected)
    scored = score_objective_observations(selected, seed=seed)
    scored_by_id = {sample["sample_id"]: sample for sample in scored["samples"]}

    paired_deltas: dict[str, list[float]] = defaultdict(list)
    pair_rows: list[dict[str, Any]] = []
    baseline_invalid = 0
    adapted_invalid = 0
    for key in sorted(baseline_keys):
        baseline_row = grouped[baseline_candidate_id][key]
        adapted_row = grouped[adapted_candidate_id][key]
        baseline_valid = baseline_row.get("valid") is True
        adapted_valid = adapted_row.get("valid") is True
        baseline_invalid += not baseline_valid
        adapted_invalid += not adapted_valid
        baseline_metrics = scored_by_id[str(baseline_row["sample_id"])]["metrics"]
        adapted_metrics = scored_by_id[str(adapted_row["sample_id"])]["metrics"]
        metrics: dict[str, dict[str, float]] = {}
        for metric in sorted(set(baseline_metrics) & set(adapted_metrics)):
            baseline_value = float(baseline_metrics[metric])
            adapted_value = float(adapted_metrics[metric])
            delta = adapted_value - baseline_value
            metrics[metric] = {
                "baseline": baseline_value,
                "adapted": adapted_value,
                "adapted_minus_baseline": delta,
            }
            paired_deltas[metric].append(delta)
        pair_rows.append(
            {
                "prompt_id": key[0],
                "seed": key[1],
                "baseline_sample_id": baseline_row["sample_id"],
                "adapted_sample_id": adapted_row["sample_id"],
                "baseline_valid": baseline_valid,
                "adapted_valid": adapted_valid,
                "metrics": metrics,
            }
        )

    metric_rows: list[dict[str, Any]] = []
    for index, (metric, values) in enumerate(sorted(paired_deltas.items())):
        direction = METRIC_DIRECTIONS.get(metric, "context_only")
        mean_delta = mean(values)
        if direction == "lower_is_better":
            mean_improvement = -mean_delta
        elif direction == "higher_is_better":
            mean_improvement = mean_delta
        else:
            mean_improvement = None
        evidence_kind = METRIC_EVIDENCE_KINDS.get(metric)
        metric_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "matched_pair_count": len(values),
                "mean_adapted_minus_baseline": mean_delta,
                "median_adapted_minus_baseline": median(values),
                "mean_directional_improvement": mean_improvement,
                "mean_delta_95pct_bootstrap_ci": bootstrap_mean_interval(
                    values,
                    seed=seed + index,
                ),
                "provenance": provenance.get(evidence_kind) if evidence_kind else None,
            }
        )

    pair_count = len(baseline_keys)
    return {
        "schema_version": "1.0.0",
        "status": "passed",
        "baseline_candidate_id": baseline_candidate_id,
        "adapted_candidate_id": adapted_candidate_id,
        "pairing_keys": ["prompt_id", "seed"],
        "pair_count": pair_count,
        "bootstrap_seed": seed,
        "validity": {
            "baseline_invalid_count": baseline_invalid,
            "adapted_invalid_count": adapted_invalid,
            "baseline_invalid_output_rate": baseline_invalid / pair_count,
            "adapted_invalid_output_rate": adapted_invalid / pair_count,
            "adapted_minus_baseline_invalid_output_rate": (adapted_invalid - baseline_invalid) / pair_count,
        },
        "metric_provenance": provenance,
        "metrics": metric_rows,
        "pairs": pair_rows,
        "proves_adaptation_benefit": False,
        "evidence_boundary": (
            "A passed matched comparison proves exact prompt and seed pairing plus consistent extractor provenance. "
            "Objective deltas remain proxies and do not establish speaker identity, accent fidelity, cadence, naturalness, or preference."
        ),
    }
