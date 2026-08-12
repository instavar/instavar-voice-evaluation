from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from statistics import mean, median
from typing import Any

from .attempts import runtime_attempt_is_content_bound
from .observations import validate_objective_observations
from .speaker_references import (
    aggregate_reference_similarities,
    canonical_sha256,
    cosine_similarity,
    speaker_measurement_is_content_bound,
    validate_speaker_reference_evidence,
)

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_token in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_tokens = _tokens(reference)
    if not reference_tokens:
        raise ValueError("reference text must contain at least one token")
    return _edit_distance(reference_tokens, _tokens(hypothesis)) / len(reference_tokens)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(values: list[float], *, seed: int, iterations: int = 2000) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return {"low": values[0], "high": values[0], "confidence": 0.95}
    generator = random.Random(seed)
    samples = [mean(generator.choices(values, k=len(values))) for _ in range(iterations)]
    return {
        "low": _percentile(samples, 0.025),
        "high": _percentile(samples, 0.975),
        "confidence": 0.95,
    }


def _number(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number when present")
    return float(value)


def _metric_summary(values: list[float], *, seed: int) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "mean_95pct_bootstrap_ci": bootstrap_mean_interval(values, seed=seed),
    }


def _asr_reference_text_provenance(rows: list[dict[str, Any]], generation_plan: Any | None) -> dict[str, Any]:
    observed = [row for row in rows if row.get("hypothesis_text") is not None]
    scored = [row for row in observed if row.get("valid") is True]
    if generation_plan is None:
        return {
            "mode": "declared_observation",
            "generation_plan_sha256": None,
            "observed_reference_count": len(observed),
            "scored_reference_count": len(scored),
            "plan_bound_reference_count": 0,
            "all_scored_references_plan_bound": False if scored else None,
            "evidence_boundary": (
                "The scorer used requested_text declared by each observation. Content-bound ASR execution does not "
                "independently verify that reference text."
            ),
        }
    if not isinstance(generation_plan, dict) or generation_plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("ASR reference generation plan must be a version 1.0.0 or 1.1.0 object")
    raw_samples = generation_plan.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("ASR reference generation plan must contain samples")
    planned_by_id: dict[str, dict[str, Any]] = {}
    for index, planned in enumerate(raw_samples):
        if not isinstance(planned, dict) or not isinstance(planned.get("sample_id"), str):
            raise ValueError(f"ASR reference generation plan sample {index} must declare sample_id")
        if planned["sample_id"] in planned_by_id:
            raise ValueError(f"ASR reference generation plan duplicates sample_id: {planned['sample_id']}")
        planned_by_id[planned["sample_id"]] = planned
    for index, row in enumerate(observed):
        planned = planned_by_id.get(row.get("sample_id"))
        if planned is None:
            raise ValueError(f"ASR observation {index} is absent from the reference generation plan")
        mismatches = [
            observation_field
            for observation_field, plan_field in (
                ("candidate_id", "candidate_id"),
                ("prompt_id", "prompt_id"),
                ("seed", "seed"),
                ("requested_text", "text"),
            )
            if row.get(observation_field) != planned.get(plan_field)
        ]
        if mismatches:
            raise ValueError(
                f"ASR observation {row.get('sample_id')} does not match reference generation plan fields: "
                + ", ".join(mismatches)
            )
    return {
        "mode": "generation_plan",
        "generation_plan_sha256": canonical_sha256(generation_plan),
        "observed_reference_count": len(observed),
        "scored_reference_count": len(scored),
        "plan_bound_reference_count": len(scored),
        "all_scored_references_plan_bound": True if scored else None,
        "evidence_boundary": (
            "The live generation plan binds requested_text to each scored sample. This establishes reference identity, "
            "not ASR validity, perceptual quality, or honest model execution."
        ),
    }


def _plan_categories(rows: list[dict[str, Any]], generation_plan: Any | None) -> dict[str, Any]:
    if generation_plan is None:
        return {
            "mode": "unavailable",
            "generation_plan_sha256": None,
            "planned_sample_count": None,
            "planned_categorized_sample_count": None,
            "planned_uncategorized_sample_count": None,
            "categorized_sample_count": 0,
            "uncategorized_sample_count": len(rows),
            "categories": {},
            "evidence_boundary": "Category strata require a live generation plan.",
        }
    if not isinstance(generation_plan, dict) or generation_plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("category generation plan must be a version 1.0.0 or 1.1.0 object")
    raw_samples = generation_plan.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("category generation plan must contain samples")
    planned_by_id: dict[str, dict[str, Any]] = {}
    category_values_by_prompt: dict[str, set[str | None]] = defaultdict(set)
    planned_categorized = 0
    for index, planned in enumerate(raw_samples):
        if not isinstance(planned, dict) or not isinstance(planned.get("sample_id"), str):
            raise ValueError(f"category generation plan sample {index} must declare sample_id")
        if planned["sample_id"] in planned_by_id:
            raise ValueError(f"category generation plan duplicates sample_id: {planned['sample_id']}")
        category = planned.get("category")
        if category is not None and (not isinstance(category, str) or not category.strip()):
            raise ValueError(f"category generation plan sample {index} category must be non-empty when present")
        normalized_category = category.strip() if isinstance(category, str) else None
        planned_categorized += normalized_category is not None
        if isinstance(planned.get("prompt_id"), str):
            category_values_by_prompt[planned["prompt_id"]].add(normalized_category)
        planned_by_id[planned["sample_id"]] = planned
    for prompt_id, category_values in sorted(category_values_by_prompt.items()):
        if len(category_values) > 1:
            raise ValueError(f"category generation plan prompt {prompt_id} must use one category across samples")

    categories: dict[str, str] = {}
    uncategorized = 0
    for index, row in enumerate(rows):
        planned = planned_by_id.get(row.get("sample_id"))
        if planned is None:
            raise ValueError(f"observation {index} is absent from the category generation plan")
        mismatches = [
            observation_field
            for observation_field, plan_field in (
                ("candidate_id", "candidate_id"),
                ("prompt_id", "prompt_id"),
                ("seed", "seed"),
                ("requested_text", "text"),
            )
            if row.get(observation_field) != planned.get(plan_field)
        ]
        if mismatches:
            raise ValueError(
                f"observation {row.get('sample_id')} does not match category generation plan fields: "
                + ", ".join(mismatches)
            )
        category = planned.get("category")
        if isinstance(category, str):
            categories[str(row["sample_id"])] = category.strip()
        else:
            uncategorized += 1
    categorized = len(rows) - uncategorized
    planned_uncategorized = len(raw_samples) - planned_categorized
    return {
        "mode": "generation_plan" if uncategorized == 0 and planned_uncategorized == 0 else "partial_generation_plan",
        "generation_plan_sha256": canonical_sha256(generation_plan),
        "planned_sample_count": len(raw_samples),
        "planned_categorized_sample_count": planned_categorized,
        "planned_uncategorized_sample_count": planned_uncategorized,
        "categorized_sample_count": categorized,
        "uncategorized_sample_count": uncategorized,
        "categories": categories,
        "evidence_boundary": (
            "Category strata are bound to generation-plan labels. They expose heterogeneous proxy results but do not "
            "establish perceptual quality or explain why a category differs."
        ),
    }


def _category_strata(samples: list[dict[str, Any]], plan_categories: dict[str, Any], *, seed: int) -> dict[str, Any]:
    public_binding = {key: value for key, value in plan_categories.items() if key != "categories"}
    if not plan_categories["categories"]:
        return {**public_binding, "candidates": []}

    metric_names = (
        "asr_word_error_rate",
        "speaker_embedding_similarity",
        "real_time_factor",
        "generation_seconds",
        "audio_duration_seconds",
        "peak_memory_bytes",
        "sample_rate_hz",
        "silence_fraction",
        "clipping_fraction",
    )
    quality_metrics = {
        "asr_word_error_rate",
        "speaker_embedding_similarity",
        "real_time_factor",
        "audio_duration_seconds",
        "sample_rate_hz",
        "silence_fraction",
        "clipping_fraction",
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        category = plan_categories["categories"].get(sample["sample_id"])
        if category is not None:
            grouped[(sample["candidate_id"], category)].append(sample)

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stratum_index, ((candidate_id, category), category_samples) in enumerate(sorted(grouped.items())):
        valid_count = sum(sample["valid"] is True for sample in category_samples)
        total = len(category_samples)
        metrics: dict[str, Any] = {}
        coverage: dict[str, Any] = {}
        for metric_index, metric in enumerate(metric_names):
            values = [float(sample["metrics"][metric]) for sample in category_samples if metric in sample["metrics"]]
            eligible = valid_count if metric in quality_metrics else total
            metrics[metric] = _metric_summary(values, seed=seed + stratum_index * 100 + metric_index)
            coverage[metric] = {
                "observed": len(values),
                "eligible": eligible,
                "rate": len(values) / eligible if eligible else None,
            }
        by_candidate[candidate_id].append(
            {
                "category": category,
                "sample_count": total,
                "valid_sample_count": valid_count,
                "invalid_output_rate": (total - valid_count) / total,
                "metrics": metrics,
                "metric_coverage": coverage,
            }
        )
    return {
        **public_binding,
        "candidates": [
            {"candidate_id": candidate_id, "categories": categories}
            for candidate_id, categories in sorted(by_candidate.items())
        ],
    }


def score_objective_observations(
    rows: list[dict[str, Any]],
    *,
    seed: int = 20260812,
    generation_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract_errors = validate_objective_observations(rows)
    if contract_errors:
        raise ValueError("; ".join(contract_errors))
    asr_reference_text = _asr_reference_text_provenance(rows, generation_plan)
    plan_categories = _plan_categories(rows, generation_plan)

    required = {"sample_id", "candidate_id", "prompt_id", "requested_text", "valid"}
    seen: set[str] = set()
    per_candidate: dict[str, dict[str, Any]] = {}
    per_sample: list[dict[str, Any]] = []
    evidence_signatures: dict[str, set[tuple[str, str, str, str, str, str, str]]] = {}
    evidence_content_binding: dict[str, dict[str, int]] = {}
    speaker_reference_sets: set[str] = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"observation {index} must be an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"observation {index} is missing: {', '.join(sorted(missing))}")
        sample_id = str(row["sample_id"]).strip()
        candidate_id = str(row["candidate_id"]).strip()
        prompt_id = str(row["prompt_id"]).strip()
        requested_text = str(row["requested_text"]).strip()
        if not sample_id or not candidate_id or not prompt_id or not requested_text:
            raise ValueError(f"observation {index} contains an empty identifier or requested_text")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if not isinstance(row["valid"], bool):
            raise ValueError(f"observation {index} valid must be a boolean")

        candidate = per_candidate.setdefault(
            candidate_id,
            {
                "total": 0,
                "valid": 0,
                "word_error_rate": [],
                "speaker_embedding_similarity": [],
                "real_time_factor": [],
                "generation_seconds": [],
                "audio_duration_seconds": [],
                "peak_memory_bytes": [],
                "sample_rate_hz": [],
                "silence_fraction": [],
                "clipping_fraction": [],
            },
        )
        candidate["total"] += 1
        sample_result: dict[str, Any] = {
            "sample_id": sample_id,
            "candidate_id": candidate_id,
            "prompt_id": prompt_id,
            "valid": row["valid"],
            "metrics": {},
            "diagnostics": {},
            "excluded_quality_metrics": [],
            "evidence": row.get("evidence", {}),
        }
        if sample_id in plan_categories["categories"]:
            sample_result["category"] = plan_categories["categories"][sample_id]
        evidence = row.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError(f"observation {index} evidence must be an object")

        def require_evidence(kind: str) -> None:
            record = evidence.get(kind)
            if not isinstance(record, dict):
                raise ValueError(f"observation {index} requires evidence.{kind}")
            if not isinstance(record.get("extractor"), str) or not record["extractor"].strip():
                raise ValueError(f"observation {index} evidence.{kind}.extractor must be non-empty")
            if not isinstance(record.get("revision"), str) or not record["revision"].strip():
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
                speaker_measurement_bound = speaker_measurement_is_content_bound(
                    row,
                    record,
                    context=f"observation {index} evidence.speaker_encoder",
                )
                if reference_binding["reference_set_sha256"] is not None:
                    speaker_reference_sets.add(reference_binding["reference_set_sha256"])
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
                speaker_measurement_bound = False
            evidence_signatures.setdefault(kind, set()).add(
                (
                    record["extractor"].strip(),
                    record["revision"].strip(),
                    artifact_sha or "",
                    reference_id or "",
                    reference_audio_sha or "",
                    reference_transcript_sha or "",
                    reference_aggregation or "",
                )
            )
            binding = evidence_content_binding.setdefault(kind, {"bound": 0, "unbound": 0})
            if kind == "runtime":
                if runtime_attempt_is_content_bound(row, index=index):
                    binding["bound"] += 1
                else:
                    binding["unbound"] += 1
                return
            input_audio_sha = record.get("input_audio_sha256")
            audio_bound = input_audio_sha is not None
            if audio_bound:
                audio_sha = row.get("audio_sha256")
                if (
                    not isinstance(input_audio_sha, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", input_audio_sha)
                    or input_audio_sha != audio_sha
                ):
                    raise ValueError(f"observation {index} evidence.{kind}.input_audio_sha256 must match audio_sha256")
            reference_bound = kind != "speaker_encoder" or (
                reference_binding["content_bound"] and speaker_measurement_bound
            )
            if audio_bound and artifact_sha is not None and reference_bound:
                binding["bound"] += 1
            else:
                binding["unbound"] += 1

        if row["valid"]:
            candidate["valid"] += 1

        hypothesis = row.get("hypothesis_text")
        if hypothesis is not None:
            if not isinstance(hypothesis, str):
                raise ValueError(f"observation {index} hypothesis_text must be a string")
            require_evidence("asr")
            sample_result["diagnostics"]["asr_reference_text_binding"] = asr_reference_text["mode"]
            if row["valid"]:
                value = word_error_rate(requested_text, hypothesis)
                sample_result["metrics"]["asr_word_error_rate"] = value
                candidate["word_error_rate"].append(value)
            else:
                sample_result["excluded_quality_metrics"].append("asr_word_error_rate")

        reference_embedding = row.get("reference_speaker_embedding")
        reference_embeddings = row.get("reference_speaker_embeddings")
        candidate_embedding = row.get("speaker_embedding")
        if reference_embedding is not None and reference_embeddings is not None:
            raise ValueError(f"observation {index} cannot mix legacy and multi-reference speaker embeddings")
        if reference_embedding is not None or reference_embeddings is not None or candidate_embedding is not None:
            require_evidence("speaker_encoder")
            if reference_embeddings is not None:
                record = evidence["speaker_encoder"]
                value, reference_scores = aggregate_reference_similarities(
                    reference_embeddings,
                    candidate_embedding,
                    evidence=record,
                    context=f"observation {index} evidence.speaker_encoder",
                )
                sample_result["diagnostics"]["speaker_reference_scores"] = reference_scores
                sample_result["diagnostics"]["speaker_reference_aggregation"] = record["reference_aggregation"]
            else:
                if not isinstance(reference_embedding, list) or not isinstance(candidate_embedding, list):
                    raise ValueError(f"observation {index} must provide both speaker embeddings as arrays")
                value = cosine_similarity(reference_embedding, candidate_embedding)
            if row["valid"]:
                sample_result["metrics"]["speaker_embedding_similarity"] = value
                candidate["speaker_embedding_similarity"].append(value)
            else:
                sample_result["excluded_quality_metrics"].append("speaker_embedding_similarity")

        generation_seconds = _number(row, "generation_seconds")
        audio_duration_seconds = _number(row, "audio_duration_seconds")
        peak_memory_bytes = _number(row, "peak_memory_bytes")
        sample_rate_hz = _number(row, "sample_rate_hz")
        silence_fraction = _number(row, "silence_fraction")
        clipping_fraction = _number(row, "clipping_fraction")
        if generation_seconds is not None or audio_duration_seconds is not None or peak_memory_bytes is not None:
            require_evidence("runtime")
        if sample_rate_hz is not None or silence_fraction is not None or clipping_fraction is not None:
            require_evidence("audio_probe")
        if generation_seconds is not None:
            if generation_seconds < 0:
                raise ValueError(f"observation {index} generation_seconds must be non-negative")
            sample_result["metrics"]["generation_seconds"] = generation_seconds
            candidate["generation_seconds"].append(generation_seconds)
        if audio_duration_seconds is not None:
            if audio_duration_seconds <= 0:
                raise ValueError(f"observation {index} audio_duration_seconds must be positive")
            if row["valid"]:
                sample_result["metrics"]["audio_duration_seconds"] = audio_duration_seconds
                candidate["audio_duration_seconds"].append(audio_duration_seconds)
            else:
                sample_result["diagnostics"]["invalid_audio_duration_seconds"] = audio_duration_seconds
                sample_result["excluded_quality_metrics"].append("audio_duration_seconds")
        if row["valid"] and generation_seconds is not None and audio_duration_seconds is not None:
            value = generation_seconds / audio_duration_seconds
            sample_result["metrics"]["real_time_factor"] = value
            candidate["real_time_factor"].append(value)
        if peak_memory_bytes is not None:
            if peak_memory_bytes < 0:
                raise ValueError(f"observation {index} peak_memory_bytes must be non-negative")
            sample_result["metrics"]["peak_memory_bytes"] = peak_memory_bytes
            candidate["peak_memory_bytes"].append(peak_memory_bytes)
        if sample_rate_hz is not None:
            if sample_rate_hz <= 0 or not sample_rate_hz.is_integer():
                raise ValueError(f"observation {index} sample_rate_hz must be a positive integer")
            if row["valid"]:
                sample_result["metrics"]["sample_rate_hz"] = sample_rate_hz
                candidate["sample_rate_hz"].append(sample_rate_hz)
            else:
                sample_result["excluded_quality_metrics"].append("sample_rate_hz")
        for name, value in (
            ("silence_fraction", silence_fraction),
            ("clipping_fraction", clipping_fraction),
        ):
            if value is None:
                continue
            if value < 0 or value > 1:
                raise ValueError(f"observation {index} {name} must be between zero and one")
            if row["valid"]:
                sample_result["metrics"][name] = value
                candidate[name].append(value)
            else:
                sample_result["excluded_quality_metrics"].append(name)
        per_sample.append(sample_result)

    candidates: list[dict[str, Any]] = []
    for candidate_index, (candidate_id, values) in enumerate(sorted(per_candidate.items())):
        candidate_seed = seed + candidate_index * 1000
        total = int(values["total"])
        valid = int(values["valid"])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "sample_count": total,
                "valid_sample_count": valid,
                "invalid_output_rate": (total - valid) / total,
                "asr_word_error_rate": _metric_summary(values["word_error_rate"], seed=candidate_seed + 1),
                "speaker_embedding_similarity": _metric_summary(
                    values["speaker_embedding_similarity"], seed=candidate_seed + 2
                ),
                "real_time_factor": _metric_summary(values["real_time_factor"], seed=candidate_seed + 3),
                "generation_seconds": _metric_summary(values["generation_seconds"], seed=candidate_seed + 4),
                "audio_duration_seconds": _metric_summary(values["audio_duration_seconds"], seed=candidate_seed + 5),
                "peak_memory_bytes": _metric_summary(values["peak_memory_bytes"], seed=candidate_seed + 6),
                "sample_rate_hz": _metric_summary(values["sample_rate_hz"], seed=candidate_seed + 7),
                "silence_fraction": _metric_summary(values["silence_fraction"], seed=candidate_seed + 8),
                "clipping_fraction": _metric_summary(values["clipping_fraction"], seed=candidate_seed + 9),
                "metric_coverage": {
                    "asr_word_error_rate": {
                        "observed": len(values["word_error_rate"]),
                        "eligible": valid,
                    },
                    "speaker_embedding_similarity": {
                        "observed": len(values["speaker_embedding_similarity"]),
                        "eligible": valid,
                    },
                    "real_time_factor": {
                        "observed": len(values["real_time_factor"]),
                        "eligible": valid,
                    },
                    "generation_seconds": {
                        "observed": len(values["generation_seconds"]),
                        "eligible": total,
                    },
                    "audio_duration_seconds": {
                        "observed": len(values["audio_duration_seconds"]),
                        "eligible": valid,
                    },
                    "peak_memory_bytes": {
                        "observed": len(values["peak_memory_bytes"]),
                        "eligible": total,
                    },
                    "sample_rate_hz": {
                        "observed": len(values["sample_rate_hz"]),
                        "eligible": valid,
                    },
                    "silence_fraction": {
                        "observed": len(values["silence_fraction"]),
                        "eligible": valid,
                    },
                    "clipping_fraction": {
                        "observed": len(values["clipping_fraction"]),
                        "eligible": valid,
                    },
                },
            }
        )

    for candidate in candidates:
        for coverage in candidate["metric_coverage"].values():
            eligible = int(coverage["eligible"])
            coverage["rate"] = coverage["observed"] / eligible if eligible else None

    metric_provenance = {
        kind: {
            "consistent": len(signatures) == 1,
            "extractors": [
                {
                    "extractor": extractor,
                    "revision": revision,
                    "extractor_artifact_set_sha256": artifact_sha or None,
                    "reference_id": reference_id or None,
                    "reference_audio_sha256": reference_audio_sha or None,
                    "reference_transcript_sha256": reference_transcript_sha or None,
                    "reference_aggregation": reference_aggregation or None,
                }
                for (
                    extractor,
                    revision,
                    artifact_sha,
                    reference_id,
                    reference_audio_sha,
                    reference_transcript_sha,
                    reference_aggregation,
                ) in sorted(signatures)
            ],
            "reference_set_sha256s": sorted(speaker_reference_sets) if kind == "speaker_encoder" else [],
            "reference_set_count": len(speaker_reference_sets) if kind == "speaker_encoder" else 0,
            "content_bound_count": evidence_content_binding[kind]["bound"],
            "unbound_count": evidence_content_binding[kind]["unbound"],
            "all_content_bound": evidence_content_binding[kind]["unbound"] == 0,
        }
        for kind, signatures in sorted(evidence_signatures.items())
    }
    if "asr" in metric_provenance:
        metric_provenance["asr"]["reference_text"] = asr_reference_text

    return {
        "schema_version": "1.0.0",
        "evaluation_scope": "objective_proxies_from_versioned_external_observations",
        "invalid_sample_policy": (
            "Invalid samples contribute invalid-output rate and operational attempt metrics only. "
            "They are excluded from ASR, speaker, audio-duration, and real-time-factor quality summaries."
        ),
        "proves_perceptual_quality": False,
        "observation_contract": {
            "version": "1.0.0",
            "versioned_sample_count": sum(row.get("observation_schema_version") == "1.0.0" for row in rows),
            "unversioned_sample_count": sum("observation_schema_version" not in row for row in rows),
        },
        "bootstrap_seed": seed,
        "metric_provenance": metric_provenance,
        "plan_category_stratification": _category_strata(per_sample, plan_categories, seed=seed + 100000),
        "candidates": candidates,
        "samples": per_sample,
    }
