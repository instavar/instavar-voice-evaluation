from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


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

    randomized = list(samples)
    random.Random(seed).shuffle(randomized)
    blind_rows: list[dict[str, str]] = []
    reveal_rows: list[dict[str, str]] = []
    for index, sample in enumerate(randomized, start=1):
        blind_id = f"sample-{index:04d}"
        blind_rows.append(
            {
                "blind_id": blind_id,
                "prompt_id": str(sample["prompt_id"]),
                "audio_path": str(sample["audio_path"]),
            }
        )
        reveal_rows.append(
            {
                "blind_id": blind_id,
                "sample_id": str(sample["sample_id"]),
                "candidate_id": str(sample["candidate_id"]),
            }
        )
    review = {
        "schema_version": "1.0.0",
        "blind": True,
        "criteria": criteria,
        "samples": blind_rows,
        "instructions": "Rate one criterion at a time. Do not open the reveal mapping until every rating is recorded.",
    }
    reveal = {
        "schema_version": "1.0.0",
        "seed": seed,
        "review_sha256": _digest(review),
        "mapping": reveal_rows,
    }
    return review, reveal
