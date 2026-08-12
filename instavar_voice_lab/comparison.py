from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .attempts import runtime_attempt_is_content_bound
from .metrics import bootstrap_mean_interval, score_objective_observations
from .runtime_artifacts import exact_runtime_binding, verify_runtime_artifact_manifest
from .speaker_reference_plans import (
    speaker_reference_assignment_sha256,
    validate_speaker_reference_assignment_plan,
)
from .speaker_references import speaker_measurement_is_content_bound, validate_speaker_reference_evidence

METRIC_DIRECTIONS = {
    "asr_word_error_rate": "lower_is_better",
    "speaker_embedding_similarity": "higher_is_better",
    "real_time_factor": "lower_is_better",
    "generation_seconds": "lower_is_better",
    "peak_memory_bytes": "lower_is_better",
    "audio_duration_seconds": "context_only",
    "sample_rate_hz": "context_only",
    "silence_fraction": "context_only",
    "clipping_fraction": "lower_is_better",
}

METRIC_EVIDENCE_KINDS = {
    "asr_word_error_rate": "asr",
    "speaker_embedding_similarity": "speaker_encoder",
    "real_time_factor": "runtime",
    "generation_seconds": "runtime",
    "peak_memory_bytes": "runtime",
    "audio_duration_seconds": "runtime",
    "sample_rate_hz": "audio_probe",
    "silence_fraction": "audio_probe",
    "clipping_fraction": "audio_probe",
}

REQUIRED_METRIC_OUTPUTS = {
    "asr_word_error_rate": "asr_word_error_rate",
    "speaker_embedding_similarity": "speaker_embedding_similarity",
    "invalid_output_rate": None,
    "duration_seconds": "audio_duration_seconds",
    "sample_rate_hz": "sample_rate_hz",
    "silence_fraction": "silence_fraction",
    "clipping_fraction": "clipping_fraction",
    "real_time_factor": "real_time_factor",
    "peak_memory_bytes": "peak_memory_bytes",
}
CONTENT_BOUND_EVIDENCE = {
    "asr_word_error_rate": "asr",
    "speaker_embedding_similarity": "speaker_encoder",
    "sample_rate_hz": "audio_probe",
    "silence_fraction": "audio_probe",
    "clipping_fraction": "audio_probe",
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


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_plan_binding(
    plan: Any,
    rows: list[dict[str, Any]],
    candidate_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("generation plan must be a version 1.0.0 or 1.1.0 object")
    plan_rows = plan.get("samples")
    if not isinstance(plan_rows, list) or not plan_rows:
        raise ValueError("generation plan must contain samples")
    if plan.get("sample_count") is not None and plan.get("sample_count") != len(plan_rows):
        raise ValueError("generation plan sample_count must equal the number of samples")
    declared_candidates = plan.get("candidate_ids")
    if (
        not isinstance(declared_candidates, list)
        or any(not isinstance(value, str) for value in declared_candidates)
        or not candidate_ids <= set(declared_candidates)
    ):
        raise ValueError("generation plan candidate_ids must declare both requested candidates")
    requirements = plan.get("generation_requirements")
    if not isinstance(requirements, dict) or any(
        requirements.get(key) is not True
        for key in ("same_transcripts", "frozen_generation_settings", "record_failures_as_observations")
    ):
        raise ValueError("generation plan must require same transcripts, frozen settings, and failure observations")
    expected: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(plan_rows):
        if not isinstance(row, dict):
            raise ValueError(f"generation-plan sample {index} must be an object")
        if row.get("candidate_id") not in candidate_ids:
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"generation-plan sample {index} must contain sample_id")
        if sample_id in expected:
            raise ValueError(f"duplicate generation-plan sample id: {sample_id}")
        expected[sample_id] = row
    if not expected:
        raise ValueError("generation plan does not contain both requested candidates")

    observed_ids = {str(row.get("sample_id", "")) for row in rows}
    if observed_ids != set(expected):
        missing = sorted(set(expected) - observed_ids)
        unexpected = sorted(observed_ids - set(expected))
        raise ValueError(
            f"matched comparison observations must exactly cover the plan; missing={missing}; unexpected={unexpected}"
        )
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        planned = expected[sample_id]
        for key in ("candidate_id", "prompt_id", "seed"):
            if row.get(key) != planned.get(key):
                raise ValueError(f"observation {index} {key} does not match generation plan for {sample_id}")
        if str(row.get("requested_text", "")).strip() != str(planned.get("text", "")).strip():
            raise ValueError(f"observation {index} requested_text does not match generation plan for {sample_id}")
    prompt_pack = plan.get("prompt_pack")
    if (
        not isinstance(prompt_pack, dict)
        or not isinstance(prompt_pack.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", prompt_pack["sha256"])
    ):
        raise ValueError("generation plan must bind a prompt_pack sha256")
    required_metrics = plan.get("required_objective_metrics", [])
    if plan.get("schema_version") == "1.1.0":
        if not isinstance(required_metrics, list) or not required_metrics:
            raise ValueError("generation plan 1.1.0 must declare required_objective_metrics")
        if any(not isinstance(metric, str) or metric not in REQUIRED_METRIC_OUTPUTS for metric in required_metrics):
            raise ValueError("generation plan contains unsupported required_objective_metrics")
        if len(required_metrics) != len(set(required_metrics)):
            raise ValueError("generation plan required_objective_metrics must be unique")
    elif required_metrics:
        raise ValueError("generation plan 1.0.0 cannot declare required_objective_metrics")
    return {
        "sha256": _canonical_sha256(plan),
        "prompt_pack": prompt_pack,
        "sample_count": len(expected),
        "required_objective_metrics": required_metrics,
        "required_metric_coverage_enforced": plan.get("schema_version") == "1.1.0",
    }


def _evidence_signature(
    row: dict[str, Any],
    kind: str,
    index: int,
) -> tuple[str, str, str, str, str, str, str, str] | None:
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
    artifact_sha = record.get("extractor_artifact_set_sha256")
    if artifact_sha is not None and (
        not isinstance(artifact_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha)
    ):
        raise ValueError(
            f"observation {index} evidence.{kind}.extractor_artifact_set_sha256 must be a lowercase SHA-256"
        )
    if kind == "speaker_encoder":
        reference_binding = validate_speaker_reference_evidence(
            record,
            context=f"observation {index} evidence.speaker_encoder",
        )
        reference_id = reference_binding["reference_id"]
        reference_audio_sha = reference_binding["reference_audio_sha256"]
        reference_transcript_sha = reference_binding["reference_transcript_sha256"]
        reference_aggregation = reference_binding["aggregation"]
        reference_catalog_sha = reference_binding.get("reference_catalog_sha256")
    elif any(
        record.get(field) is not None
        for field in (
            "reference_id",
            "reference_audio_sha256",
            "reference_transcript_sha256",
            "reference_aggregation",
            "reference_set_sha256",
            "references",
            "reference_catalog_sha256",
            "reference_assignment_plan_sha256",
            "reference_assignment_sha256",
        )
    ):
        raise ValueError(f"observation {index} evidence.{kind} must not declare a speaker reference")
    else:
        reference_id = None
        reference_audio_sha = None
        reference_transcript_sha = None
        reference_aggregation = None
        reference_catalog_sha = None
    return (
        extractor.strip(),
        revision.strip(),
        artifact_sha or "",
        reference_id or "",
        reference_audio_sha or "",
        reference_transcript_sha or "",
        reference_aggregation or "",
        reference_catalog_sha or "",
    )


def _provenance(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_kind: dict[str, set[tuple[str, str, str, str, str, str, str, str]]] = defaultdict(set)
    for index, row in enumerate(rows):
        for kind in set(METRIC_EVIDENCE_KINDS.values()):
            signature = _evidence_signature(row, kind, index)
            if signature is not None:
                by_kind[kind].add(signature)
    mixed = {kind: sorted(signatures) for kind, signatures in by_kind.items() if len(signatures) > 1}
    if mixed:
        rendered: list[str] = []
        for kind, signatures in sorted(mixed.items()):
            values = ",".join(f"{signature[0]}@{signature[1]}#{signature[2] or 'unbound'}" for signature in signatures)
            rendered.append(f"{kind}={values}")
        details = "; ".join(rendered)
        raise ValueError(f"matched comparison cannot mix extractor provenance: {details}")
    return {
        kind: {
            "extractor": next(iter(signatures))[0],
            "revision": next(iter(signatures))[1],
            "extractor_artifact_set_sha256": next(iter(signatures))[2] or None,
            "reference_id": next(iter(signatures))[3] or None,
            "reference_audio_sha256": next(iter(signatures))[4] or None,
            "reference_transcript_sha256": next(iter(signatures))[5] or None,
            "reference_aggregation": next(iter(signatures))[6] or None,
            "reference_catalog_sha256": next(iter(signatures))[7] or None,
        }
        for kind, signatures in sorted(by_kind.items())
        if signatures
    }


def _validate_content_bound_required_metrics(
    row: dict[str, Any],
    required_metrics: list[str],
    index: int,
    *,
    generation_plan_sha256: str,
    planned_sample: dict[str, Any],
    speaker_reference_plan: dict[str, Any] | None,
) -> None:
    required_kinds = {kind for metric in required_metrics if (kind := CONTENT_BOUND_EVIDENCE.get(metric)) is not None}
    if not required_kinds:
        return
    audio_sha = row.get("audio_sha256")
    if required_kinds - {"runtime"} and (
        not isinstance(audio_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", audio_sha)
    ):
        raise ValueError(f"observation {index} must bind required metrics to audio_sha256")
    evidence = row.get("evidence")
    for kind in sorted(required_kinds):
        if kind == "runtime":
            if not runtime_attempt_is_content_bound(
                row,
                index=index,
                generation_plan_sha256=generation_plan_sha256,
                planned_sample=planned_sample,
            ):
                raise ValueError(f"observation {index} plan-required runtime metrics must bind a generation attempt")
            continue
        record = evidence.get(kind) if isinstance(evidence, dict) else None
        if not isinstance(record, dict) or record.get("input_audio_sha256") != audio_sha:
            raise ValueError(f"observation {index} evidence.{kind}.input_audio_sha256 must match audio_sha256")
        artifact_sha = record.get("extractor_artifact_set_sha256")
        if not isinstance(artifact_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha):
            raise ValueError(f"observation {index} evidence.{kind} must bind extractor_artifact_set_sha256")
        if kind == "speaker_encoder":
            reference_binding = validate_speaker_reference_evidence(
                record,
                context=f"observation {index} evidence.speaker_encoder",
            )
            if not reference_binding["content_bound"]:
                raise ValueError(f"observation {index} evidence.speaker_encoder must bind a speaker reference")
            if reference_binding["mode"] != "content_addressed_reference_set":
                raise ValueError(
                    f"observation {index} plan-required speaker metrics must bind a content-addressed reference set"
                )
            if speaker_reference_plan is None:
                raise ValueError(
                    f"observation {index} plan-required speaker metrics must bind a frozen reference assignment plan"
                )
            assignment_key = (planned_sample.get("prompt_id"), planned_sample.get("seed"))
            expected_reference_ids = speaker_reference_plan["assignments"].get(assignment_key)
            if expected_reference_ids is None:
                raise ValueError(f"observation {index} has no frozen speaker reference assignment")
            observed_reference_ids = [item["reference_id"] for item in reference_binding["references"]]
            if observed_reference_ids != expected_reference_ids:
                raise ValueError(
                    f"observation {index} speaker reference set does not match the frozen assignment plan"
                )
            if reference_binding["reference_catalog_sha256"] != speaker_reference_plan["reference_catalog_sha256"]:
                raise ValueError(
                    f"observation {index} speaker reference catalog does not match the frozen assignment plan"
                )
            if reference_binding["reference_assignment_plan_sha256"] != speaker_reference_plan["sha256"]:
                raise ValueError(
                    f"observation {index} speaker evidence does not bind the frozen reference assignment plan"
                )
            expected_assignment_sha = speaker_reference_assignment_sha256(
                assignment_plan_sha256=speaker_reference_plan["sha256"],
                prompt_id=planned_sample["prompt_id"],
                seed=planned_sample["seed"],
                reference_ids=expected_reference_ids,
            )
            if reference_binding["reference_assignment_sha256"] != expected_assignment_sha:
                raise ValueError(
                    f"observation {index} speaker evidence does not bind its frozen reference assignment"
                )
            if not speaker_measurement_is_content_bound(
                row,
                record,
                context=f"observation {index} evidence.speaker_encoder",
            ):
                raise ValueError(f"observation {index} plan-required speaker metrics must bind the speaker measurement")


def _matched_speaker_reference_set(
    baseline_row: dict[str, Any],
    adapted_row: dict[str, Any],
    *,
    prompt_id: str,
    seed: int,
) -> tuple[str | None, str | None]:
    bindings: list[dict[str, Any] | None] = []
    for label, row in (("baseline", baseline_row), ("adapted", adapted_row)):
        evidence = row.get("evidence")
        record = evidence.get("speaker_encoder") if isinstance(evidence, dict) else None
        if record is None:
            bindings.append(None)
            continue
        if not isinstance(record, dict):
            raise ValueError(f"{label} speaker evidence must be an object for prompt {prompt_id}, seed {seed}")
        bindings.append(
            validate_speaker_reference_evidence(
                record,
                context=f"{label} evidence.speaker_encoder for prompt {prompt_id}, seed {seed}",
            )
        )
    baseline_binding, adapted_binding = bindings
    if baseline_binding is None or adapted_binding is None:
        return None, None
    baseline_mode = baseline_binding["mode"]
    adapted_mode = adapted_binding["mode"]
    if baseline_mode != adapted_mode:
        raise ValueError(
            f"matched speaker comparison requires the same reference binding mode for prompt {prompt_id}, seed {seed}"
        )
    if baseline_mode != "content_addressed_reference_set":
        return None, None
    baseline_sha = baseline_binding["reference_set_sha256"]
    adapted_sha = adapted_binding["reference_set_sha256"]
    if baseline_sha != adapted_sha:
        raise ValueError(
            f"matched speaker comparison requires the same reference set for prompt {prompt_id}, seed {seed}"
        )
    baseline_assignment_sha = baseline_binding["reference_assignment_sha256"]
    adapted_assignment_sha = adapted_binding["reference_assignment_sha256"]
    baseline_plan_sha = baseline_binding["reference_assignment_plan_sha256"]
    adapted_plan_sha = adapted_binding["reference_assignment_plan_sha256"]
    if (baseline_assignment_sha, baseline_plan_sha) != (adapted_assignment_sha, adapted_plan_sha):
        raise ValueError(
            f"matched speaker comparison requires the same frozen reference assignment for prompt {prompt_id}, "
            f"seed {seed}"
        )
    return baseline_sha, baseline_assignment_sha


def compare_matched_candidates(
    rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    baseline_candidate_id: str,
    adapted_candidate_id: str,
    seed: int = 20260812,
    speaker_reference_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not baseline_candidate_id or not adapted_candidate_id:
        raise ValueError("baseline and adapted candidate ids must be non-empty")
    if baseline_candidate_id == adapted_candidate_id:
        raise ValueError("baseline and adapted candidate ids must differ")
    selected = [row for row in rows if row.get("candidate_id") in {baseline_candidate_id, adapted_candidate_id}]
    if not selected:
        raise ValueError("no observations matched the requested candidates")
    plan_binding = _validate_plan_binding(
        plan,
        selected,
        {baseline_candidate_id, adapted_candidate_id},
    )
    speaker_reference_plan_binding: dict[str, Any] | None = None
    requires_speaker_metric = "speaker_embedding_similarity" in plan_binding["required_objective_metrics"]
    if speaker_reference_plan is not None:
        speaker_reference_plan_binding = validate_speaker_reference_assignment_plan(
            speaker_reference_plan,
            generation_plan=plan,
        )
    elif requires_speaker_metric:
        raise ValueError("plan-required speaker metrics require a frozen speaker reference assignment plan")

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

    for pair_index, key in enumerate(sorted(baseline_keys)):
        baseline_text = str(grouped[baseline_candidate_id][key].get("requested_text", "")).strip()
        adapted_text = str(grouped[adapted_candidate_id][key].get("requested_text", "")).strip()
        if not baseline_text or baseline_text != adapted_text:
            raise ValueError(f"requested_text mismatch for prompt {key[0]}, seed {key[1]}")

    provenance = _provenance(selected)
    scored = score_objective_observations(selected, seed=seed, generation_plan=plan)
    if "asr" in provenance:
        provenance["asr"]["reference_text"] = scored["metric_provenance"]["asr"]["reference_text"]
    scored_by_id = {sample["sample_id"]: sample for sample in scored["samples"]}
    planned_by_id = {
        row["sample_id"]: row
        for row in plan["samples"]
        if isinstance(row, dict) and row.get("sample_id") in {item["sample_id"] for item in selected}
    }

    paired_deltas: dict[str, list[float]] = defaultdict(list)
    pair_rows: list[dict[str, Any]] = []
    baseline_invalid = 0
    adapted_invalid = 0
    for key in sorted(baseline_keys):
        baseline_row = grouped[baseline_candidate_id][key]
        adapted_row = grouped[adapted_candidate_id][key]
        speaker_reference_set_sha256, speaker_reference_assignment_sha256_value = _matched_speaker_reference_set(
            baseline_row,
            adapted_row,
            prompt_id=key[0],
            seed=key[1],
        )
        baseline_valid = baseline_row.get("valid") is True
        adapted_valid = adapted_row.get("valid") is True
        baseline_invalid += not baseline_valid
        adapted_invalid += not adapted_valid
        baseline_metrics = scored_by_id[str(baseline_row["sample_id"])]["metrics"]
        adapted_metrics = scored_by_id[str(adapted_row["sample_id"])]["metrics"]
        required_outputs = {
            output
            for metric in plan_binding["required_objective_metrics"]
            if (output := REQUIRED_METRIC_OUTPUTS[metric]) is not None
        }
        if baseline_valid and adapted_valid:
            missing_required_baseline = sorted(required_outputs - set(baseline_metrics))
            missing_required_adapted = sorted(required_outputs - set(adapted_metrics))
            if missing_required_baseline or missing_required_adapted:
                raise ValueError(
                    "matched comparison is missing plan-required metrics for a valid pair; "
                    f"prompt={key[0]}; seed={key[1]}; missing baseline={missing_required_baseline}; "
                    f"missing adapted={missing_required_adapted}"
                )
            _validate_content_bound_required_metrics(
                baseline_row,
                plan_binding["required_objective_metrics"],
                pair_index * 2,
                generation_plan_sha256=plan_binding["sha256"],
                planned_sample=planned_by_id[baseline_row["sample_id"]],
                speaker_reference_plan=speaker_reference_plan_binding,
            )
            _validate_content_bound_required_metrics(
                adapted_row,
                plan_binding["required_objective_metrics"],
                pair_index * 2 + 1,
                generation_plan_sha256=plan_binding["sha256"],
                planned_sample=planned_by_id[adapted_row["sample_id"]],
                speaker_reference_plan=speaker_reference_plan_binding,
            )
        if baseline_valid and adapted_valid and set(baseline_metrics) != set(adapted_metrics):
            missing_adapted = sorted(set(baseline_metrics) - set(adapted_metrics))
            missing_baseline = sorted(set(adapted_metrics) - set(baseline_metrics))
            raise ValueError(
                "matched comparison requires symmetric metric availability for valid pairs; "
                f"prompt={key[0]}; seed={key[1]}; missing adapted={missing_adapted}; "
                f"missing baseline={missing_baseline}"
            )
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
                "speaker_reference_set_sha256": speaker_reference_set_sha256,
                "speaker_reference_assignment_sha256": speaker_reference_assignment_sha256_value,
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
        "generation_plan": plan_binding,
        "speaker_reference_assignment_plan": (
            {
                "plan_id": speaker_reference_plan_binding["plan_id"],
                "sha256": speaker_reference_plan_binding["sha256"],
                "reference_catalog_sha256": speaker_reference_plan_binding["reference_catalog_sha256"],
                "reference_aggregation": speaker_reference_plan_binding["reference_aggregation"],
                "selection_policy": speaker_reference_plan_binding["selection_policy"],
                "assignment_count": speaker_reference_plan_binding["assignment_count"],
            }
            if speaker_reference_plan_binding is not None
            else None
        ),
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
            "A passed matched comparison proves exact prompt and seed pairing, plan-required and symmetric metric "
            "availability for valid pairs, consistent extractor provenance, and frozen reference assignments when "
            "speaker similarity is plan-required. "
            "Objective deltas remain proxies and do not establish speaker identity, accent fidelity, cadence, "
            "naturalness, or preference."
        ),
    }


def compare_runtime_candidates(
    rows: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    artifact_manifest: dict[str, Any],
    artifact_binding_plan: dict[str, Any],
    artifact_base_dir: Path,
    reference_candidate_id: str,
    candidate_candidate_id: str,
    reference_runtime_id: str,
    candidate_runtime_id: str,
    seed: int = 20260812,
    speaker_reference_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if reference_runtime_id == candidate_runtime_id:
        raise ValueError("runtime comparison requires two distinct runtime ids")
    verification = verify_runtime_artifact_manifest(
        artifact_manifest,
        artifact_binding_plan,
        base_dir=artifact_base_dir,
    )
    binding = exact_runtime_binding(artifact_manifest, {reference_runtime_id, candidate_runtime_id})
    selected = [row for row in rows if row.get("candidate_id") in {reference_candidate_id, candidate_candidate_id}]
    if not selected:
        raise ValueError("no observations matched the requested runtime candidates")
    expected_runtime = {
        reference_candidate_id: reference_runtime_id,
        candidate_candidate_id: candidate_runtime_id,
    }
    for index, row in enumerate(selected):
        candidate_id = row.get("candidate_id")
        if row.get("runtime_id") != expected_runtime.get(candidate_id):
            raise ValueError(f"observation {index} runtime_id does not match its requested candidate binding")
        if row.get("artifact_set_id") != binding["artifact_set_id"]:
            raise ValueError(f"observation {index} artifact_set_id does not match the runtime artifact manifest")
        if row.get("artifact_set_sha256") != binding["source_artifact_set_sha256"]:
            raise ValueError(f"observation {index} artifact_set_sha256 does not match the runtime artifact manifest")

    objective = compare_matched_candidates(
        rows,
        plan=plan,
        baseline_candidate_id=reference_candidate_id,
        adapted_candidate_id=candidate_candidate_id,
        seed=seed,
        speaker_reference_plan=speaker_reference_plan,
    )
    return {
        "schema_version": "1.0.0",
        "status": "passed",
        "reference": {"candidate_id": reference_candidate_id, "runtime_id": reference_runtime_id},
        "candidate": {"candidate_id": candidate_candidate_id, "runtime_id": candidate_runtime_id},
        "artifact_binding": binding,
        "artifact_verification": verification,
        "objective_comparison": objective,
        "proves_shared_artifact_identity": True,
        "proves_runtime_equivalence": False,
        "evidence_boundary": (
            "A passed runtime comparison proves current exact artifact fingerprints plus matched prompt, seed, text, "
            "observation coverage, and extractor provenance. It does not prove that either runtime loaded the declared "
            "bytes, numerical equivalence, speaker identity, accent fidelity, cadence, naturalness, or preference."
        ),
    }
