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


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_blind_pack(samples: list[dict[str, Any]], criteria: list[str], *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    required = {"sample_id", "candidate_id", "prompt_id", "audio_path"}
    for index, sample in enumerate(samples):
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"sample {index} is missing: {', '.join(sorted(missing))}")
    if not criteria or any(not isinstance(value, str) or not value.strip() for value in criteria):
        raise ValueError("criteria must contain non-empty strings")
    suffixes = {Path(str(sample["audio_path"])).suffix.lower() for sample in samples}
    if len(suffixes) != 1:
        raise ValueError("all blind-review audio files must use the same extension to avoid candidate leakage")

    randomized = list(samples)
    random.Random(seed).shuffle(randomized)
    blind_rows: list[dict[str, str]] = []
    reveal_rows: list[dict[str, str]] = []
    for index, sample in enumerate(randomized, start=1):
        blind_id = f"sample-{index:04d}"
        suffix = Path(str(sample["audio_path"])).suffix.lower()
        if not suffix or len(suffix) > 10:
            suffix = ".wav"
        blind_rows.append(
            {
                "blind_id": blind_id,
                "prompt_id": str(sample["prompt_id"]),
                "audio_path": f"blind_audio/{blind_id}{suffix}",
            }
        )
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
    if not isinstance(criteria, list) or not criteria:
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
    review_blind_ids = {
        row["blind_id"] for row in review.get("samples", []) if isinstance(row, dict) and "blind_id" in row
    }
    if set(mapping) != review_blind_ids:
        raise ValueError("reveal mapping must cover every blind sample exactly once")

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
        for criterion in criteria
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
            "Agreement measures rating consistency, not truth. Criterion summaries remain separate and do not prove general quality."
        ),
    }
