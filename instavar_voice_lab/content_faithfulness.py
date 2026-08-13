from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .extraction import build_speaker_reference_catalog, observation_document_sha256
from .metrics import word_error_rate
from .speaker_reference_plans import (
    speaker_reference_assignment_sha256,
    validate_speaker_reference_assignment_plan,
)
from .speaker_references import SHA256_RE, canonical_sha256
from .suite import check_suite_coverage

CONTENT_FAITHFULNESS_SCHEMA_VERSION = "instavar_voice_content_faithfulness/v1"
TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
MAX_REFERENCE_COUNT = 128
MAX_REFERENCE_IDS_PER_SAMPLE = 32
MAX_TOTAL_REFERENCE_TOKENS = 50000
MAX_TRANSCRIPT_BYTES = 64 * 1024
MAX_REPORTED_NGRAM_HITS = 100


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def _ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def _ngram_sha256(ngram: tuple[str, ...]) -> str:
    return hashlib.sha256(" ".join(ngram).encode("utf-8")).hexdigest()


def _read_reference_transcript(path: Path, reference_id: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"speaker reference transcript must be a regular non-symlink file: {reference_id}")
    byte_count = path.stat().st_size
    if byte_count <= 0 or byte_count > MAX_TRANSCRIPT_BYTES:
        raise ValueError(
            f"speaker reference transcript must contain between 1 and {MAX_TRANSCRIPT_BYTES} bytes: {reference_id}"
        )
    text = path.read_text(encoding="utf-8")
    if not _tokens(text):
        raise ValueError(f"speaker reference transcript must contain at least one token: {reference_id}")
    return text


def _validate_threshold(name: str, value: float, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return float(value)


def _validate_asr_binding(row: dict[str, Any], index: int) -> None:
    audio_sha = row.get("audio_sha256")
    if not isinstance(audio_sha, str) or not SHA256_RE.fullmatch(audio_sha):
        raise ValueError(f"observation {index} content diagnostics require audio_sha256")
    evidence = row.get("evidence")
    asr = evidence.get("asr") if isinstance(evidence, dict) else None
    if not isinstance(asr, dict):
        raise ValueError(f"observation {index} content diagnostics require evidence.asr")
    for field in ("extractor", "revision"):
        if not isinstance(asr.get(field), str) or not asr[field].strip():
            raise ValueError(f"observation {index} evidence.asr.{field} must be non-empty")
    if asr.get("input_audio_sha256") != audio_sha:
        raise ValueError(f"observation {index} evidence.asr.input_audio_sha256 must match audio_sha256")
    artifact_sha = asr.get("extractor_artifact_set_sha256")
    if not isinstance(artifact_sha, str) or not SHA256_RE.fullmatch(artifact_sha):
        raise ValueError(f"observation {index} content diagnostics require bound ASR extractor artifacts")


def _sample_diagnostics(
    *,
    requested_text: str,
    hypothesis_text: str,
    reference_texts: list[str],
    ngram_size: int,
    repetition_excess_fraction_threshold: float,
    minimum_reference_ngram_hits: int,
    word_error_rate_threshold: float,
) -> dict[str, Any]:
    requested = _tokens(requested_text)
    hypothesis = _tokens(hypothesis_text)
    requested_ngrams = Counter(_ngrams(requested, ngram_size))
    hypothesis_ngrams = Counter(_ngrams(hypothesis, ngram_size))

    repeated_excess = sum(
        max(0, count - max(1, requested_ngrams.get(ngram, 0)))
        for ngram, count in hypothesis_ngrams.items()
    )
    hypothesis_window_count = max(0, len(hypothesis) - ngram_size + 1)
    repetition_fraction = repeated_excess / hypothesis_window_count if hypothesis_window_count else 0.0

    reference_exclusive: set[tuple[str, ...]] = set()
    for text in reference_texts:
        reference_exclusive.update(_ngrams(_tokens(text), ngram_size))
    reference_exclusive.difference_update(requested_ngrams)
    reference_hits = sorted(reference_exclusive & set(hypothesis_ngrams))

    wer = word_error_rate(requested_text, hypothesis_text)
    flags = {
        "high_word_error_rate": wer > word_error_rate_threshold,
        "repetition_excess": repetition_fraction > repetition_excess_fraction_threshold,
        "reference_transcript_overlap": len(reference_hits) >= minimum_reference_ngram_hits,
    }
    return {
        "content_gate_status": "failed" if any(flags.values()) else "not_flagged",
        "flags": flags,
        "word_error_rate": wer,
        "requested_word_count": len(requested),
        "hypothesis_word_count": len(hypothesis),
        "hypothesis_to_requested_word_ratio": len(hypothesis) / len(requested),
        "ngram_size": ngram_size,
        "hypothesis_ngram_window_count": hypothesis_window_count,
        "repeated_ngram_excess_count": repeated_excess,
        "repeated_ngram_excess_fraction": repetition_fraction,
        "reference_exclusive_ngram_count": len(reference_exclusive),
        "reference_exclusive_ngram_hit_count": len(reference_hits),
        "reference_exclusive_ngram_hit_rate": (
            len(reference_hits) / len(reference_exclusive) if reference_exclusive else None
        ),
        "reference_exclusive_ngram_hit_sha256": [
            _ngram_sha256(ngram) for ngram in reference_hits[:MAX_REPORTED_NGRAM_HITS]
        ],
        "reference_exclusive_ngram_hit_hashes_truncated": len(reference_hits) > MAX_REPORTED_NGRAM_HITS,
    }


def build_content_faithfulness_report(
    observations: Any,
    *,
    generation_plan: dict[str, Any],
    reference_catalog: dict[str, Any],
    reference_assignment_plan: dict[str, Any],
    speaker_references: dict[str, tuple[Path, Path]],
    ngram_size: int = 4,
    repetition_excess_fraction_threshold: float = 0.05,
    minimum_reference_ngram_hits: int = 2,
    word_error_rate_threshold: float = 0.1,
) -> dict[str, Any]:
    if isinstance(ngram_size, bool) or not isinstance(ngram_size, int) or not 2 <= ngram_size <= 12:
        raise ValueError("ngram_size must be an integer between 2 and 12")
    if (
        isinstance(minimum_reference_ngram_hits, bool)
        or not isinstance(minimum_reference_ngram_hits, int)
        or minimum_reference_ngram_hits < 1
    ):
        raise ValueError("minimum_reference_ngram_hits must be a positive integer")
    repetition_threshold = _validate_threshold(
        "repetition_excess_fraction_threshold",
        repetition_excess_fraction_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    wer_threshold = _validate_threshold(
        "word_error_rate_threshold",
        word_error_rate_threshold,
        minimum=0.0,
        maximum=10.0,
    )

    source_sha = observation_document_sha256(observations)
    coverage = check_suite_coverage(generation_plan, observations)
    if coverage["status"] != "passed":
        raise ValueError("content diagnostics require exact generation-plan observation coverage")

    if not isinstance(reference_catalog, dict):
        raise ValueError("speaker reference catalog must be an object")
    if not isinstance(speaker_references, dict) or not 1 <= len(speaker_references) <= MAX_REFERENCE_COUNT:
        raise ValueError(f"speaker references must contain between 1 and {MAX_REFERENCE_COUNT} entries")
    live_catalog = build_speaker_reference_catalog(
        catalog_id=reference_catalog.get("catalog_id"),
        references=speaker_references,
    )
    if live_catalog != reference_catalog:
        raise ValueError("speaker reference catalog does not match the live audio and transcripts")
    assignment = validate_speaker_reference_assignment_plan(
        reference_assignment_plan,
        generation_plan=generation_plan,
        reference_catalog=live_catalog,
    )

    reference_texts = {
        reference_id: _read_reference_transcript(paths[1], reference_id)
        for reference_id, paths in sorted(speaker_references.items())
    }
    if sum(len(_tokens(text)) for text in reference_texts.values()) > MAX_TOTAL_REFERENCE_TOKENS:
        raise ValueError(f"speaker reference transcripts exceed {MAX_TOTAL_REFERENCE_TOKENS} total tokens")
    planned_by_id = {sample["sample_id"]: sample for sample in generation_plan["samples"]}
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(observations):
        planned = planned_by_id[row["sample_id"]]
        mismatches = [
            observed_field
            for observed_field, planned_field in (
                ("candidate_id", "candidate_id"),
                ("prompt_id", "prompt_id"),
                ("seed", "seed"),
                ("requested_text", "text"),
            )
            if row.get(observed_field) != planned.get(planned_field)
        ]
        if mismatches:
            raise ValueError(
                f"observation {row['sample_id']} does not match generation plan fields: {', '.join(mismatches)}"
            )
        reference_ids = assignment["assignments"][(row["prompt_id"], row["seed"])]
        if len(reference_ids) > MAX_REFERENCE_IDS_PER_SAMPLE:
            raise ValueError(
                f"speaker reference assignment exceeds {MAX_REFERENCE_IDS_PER_SAMPLE} references: {row['sample_id']}"
            )
        result: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "reference_ids": reference_ids,
            "reference_assignment_sha256": speaker_reference_assignment_sha256(
                assignment_plan_sha256=assignment["sha256"],
                prompt_id=row["prompt_id"],
                seed=row["seed"],
                reference_ids=reference_ids,
            ),
        }
        if row["valid"] is not True:
            result.update(
                {
                    "content_gate_status": "failed",
                    "flags": {"invalid_output": True},
                    "diagnostic_reason": "invalid_output",
                }
            )
        elif not isinstance(row.get("hypothesis_text"), str):
            result.update(
                {
                    "content_gate_status": "not_evaluable",
                    "flags": {"asr_unavailable": True},
                    "diagnostic_reason": "asr_unavailable",
                }
            )
        else:
            _validate_asr_binding(row, index)
            result.update(
                _sample_diagnostics(
                    requested_text=planned["text"],
                    hypothesis_text=row["hypothesis_text"],
                    reference_texts=[reference_texts[reference_id] for reference_id in reference_ids],
                    ngram_size=ngram_size,
                    repetition_excess_fraction_threshold=repetition_threshold,
                    minimum_reference_ngram_hits=minimum_reference_ngram_hits,
                    word_error_rate_threshold=wer_threshold,
                )
            )
        samples.append(result)

    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_candidate[sample["candidate_id"]].append(sample)
    candidate_summaries = []
    for candidate_id, candidate_samples in sorted(by_candidate.items()):
        status_counts = Counter(sample["content_gate_status"] for sample in candidate_samples)
        evaluable = [sample for sample in candidate_samples if "word_error_rate" in sample]
        flag_names = (
            "high_word_error_rate",
            "repetition_excess",
            "reference_transcript_overlap",
        )
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "sample_count": len(candidate_samples),
                "evaluable_sample_count": len(evaluable),
                "status_counts": {
                    status: status_counts.get(status, 0)
                    for status in ("failed", "not_evaluable", "not_flagged")
                },
                "content_gate_status": (
                    "failed"
                    if status_counts.get("failed", 0)
                    else "incomplete"
                    if status_counts.get("not_evaluable", 0)
                    else "not_flagged"
                ),
                "flag_counts": {
                    flag: sum(sample.get("flags", {}).get(flag) is True for sample in candidate_samples)
                    for flag in flag_names
                },
                "mean_word_error_rate": (
                    sum(sample["word_error_rate"] for sample in evaluable) / len(evaluable) if evaluable else None
                ),
                "mean_repeated_ngram_excess_fraction": (
                    sum(sample["repeated_ngram_excess_fraction"] for sample in evaluable) / len(evaluable)
                    if evaluable
                    else None
                ),
                "reference_exclusive_ngram_hit_count": sum(
                    sample["reference_exclusive_ngram_hit_count"] for sample in evaluable
                ),
            }
        )
    payload = {
        "schema_version": CONTENT_FAITHFULNESS_SCHEMA_VERSION,
        "source_observations_sha256": source_sha,
        "generation_plan_sha256": canonical_sha256(generation_plan),
        "reference_catalog_sha256": live_catalog["catalog_sha256"],
        "reference_assignment_plan_sha256": assignment["sha256"],
        "parameters": {
            "ngram_size": ngram_size,
            "minimum_reference_ngram_hits": minimum_reference_ngram_hits,
            "repetition_excess_fraction_threshold": repetition_threshold,
            "word_error_rate_threshold": wer_threshold,
        },
        "candidates": candidate_summaries,
        "samples": samples,
        "proves_content_faithfulness": False,
        "evidence_boundary": (
            "This deterministic ASR-text diagnostic can flag requested-text error, repeated n-gram excess, and "
            "reference-exclusive n-gram overlap. It does not prove perceptual quality, pronunciation, accent, "
            "causality, honest runtime execution, or absence of leakage when no flag fires."
        ),
    }
    return {**payload, "report_sha256": canonical_sha256(payload)}
