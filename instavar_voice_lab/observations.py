from __future__ import annotations

import math
import re
from typing import Any


IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OBSERVATION_SCHEMA_VERSION = "1.0.0"


def validate_objective_observations(
    document: Any,
    *,
    require_version: bool = False,
    require_seed: bool = False,
    require_runtime: bool = False,
) -> list[str]:
    if not isinstance(document, list):
        return ["objective observations must be a JSON array"]
    if not document:
        return ["at least one objective observation is required"]

    errors: list[str] = []
    seen: set[str] = set()
    required = {"sample_id", "candidate_id", "prompt_id", "requested_text", "valid"}
    for index, value in enumerate(document):
        path = f"observations[{index}]"
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            continue
        missing = required - value.keys()
        if missing:
            errors.append(f"{path} is missing: {', '.join(sorted(missing))}")

        version = value.get("observation_schema_version")
        if require_version and version is None:
            errors.append(f"{path}.observation_schema_version is required")
        elif version is not None and version != OBSERVATION_SCHEMA_VERSION:
            errors.append(
                f"{path}.observation_schema_version must equal {OBSERVATION_SCHEMA_VERSION}"
            )

        for name in ("sample_id", "candidate_id", "prompt_id"):
            identifier = value.get(name)
            if not isinstance(identifier, str) or not IDENTIFIER_RE.fullmatch(identifier):
                errors.append(f"{path}.{name} must be a stable lowercase identifier")
        sample_id = value.get("sample_id")
        if isinstance(sample_id, str):
            if sample_id in seen:
                errors.append(f"{path}.sample_id duplicates {sample_id}")
            seen.add(sample_id)

        requested_text = value.get("requested_text")
        if not isinstance(requested_text, str) or not requested_text.strip():
            errors.append(f"{path}.requested_text must be a non-empty string")
        if not isinstance(value.get("valid"), bool):
            errors.append(f"{path}.valid must be a boolean")

        seed = value.get("seed")
        if require_seed and seed is None:
            errors.append(f"{path}.seed is required")
        elif seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            errors.append(f"{path}.seed must be a non-negative integer")

        runtime_id = value.get("runtime_id")
        if require_runtime and runtime_id is None:
            errors.append(f"{path}.runtime_id is required")
        elif runtime_id is not None and (
            not isinstance(runtime_id, str) or not IDENTIFIER_RE.fullmatch(runtime_id)
        ):
            errors.append(f"{path}.runtime_id must be a stable lowercase identifier")

        artifact_id = value.get("artifact_set_id")
        artifact_sha = value.get("artifact_set_sha256")
        if (artifact_id is None) != (artifact_sha is None):
            errors.append(
                f"{path}.artifact_set_id and artifact_set_sha256 must be supplied together"
            )
        if artifact_id is not None and (
            not isinstance(artifact_id, str) or not IDENTIFIER_RE.fullmatch(artifact_id)
        ):
            errors.append(f"{path}.artifact_set_id must be a stable lowercase identifier")
        if artifact_sha is not None and (
            not isinstance(artifact_sha, str) or not SHA256_RE.fullmatch(artifact_sha)
        ):
            errors.append(f"{path}.artifact_set_sha256 must be a lowercase SHA-256")

        for name in ("generation_seconds", "audio_duration_seconds", "peak_memory_bytes"):
            number = value.get(name)
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
            ):
                errors.append(f"{path}.{name} must be a finite number when present")

        prosody_nullable = {
            "prosody_active_rms_db_std",
            "prosody_pause_duration_cv",
            "prosody_phrase_duration_cv",
            "prosody_window_rms_db_std",
            "prosody_zero_crossing_rate_hz_std",
        }
        prosody_bounded = {"prosody_active_frame_fraction", "prosody_pause_fraction"}
        prosody_fields = {
            "prosody_analysis_duration_seconds",
            "prosody_active_frame_fraction",
            "prosody_active_rms_db_std",
            "prosody_leading_inactive_seconds",
            "prosody_pause_duration_cv",
            "prosody_pause_fraction",
            "prosody_pause_rate_per_minute",
            "prosody_phrase_duration_cv",
            "prosody_trailing_inactive_seconds",
            "prosody_window_rms_db_std",
            "prosody_zero_crossing_rate_hz_std",
        }
        for name in prosody_fields:
            number = value.get(name)
            if number is None and (name not in value or name in prosody_nullable):
                continue
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) < 0
                or name in prosody_bounded and float(number) > 1
            ):
                errors.append(f"{path}.{name} must be a finite non-negative number in its allowed range")
        eligible = value.get("prosody_eligible_for_long_form")
        if eligible is not None and not isinstance(eligible, bool):
            errors.append(f"{path}.prosody_eligible_for_long_form must be a boolean when present")

        evidence = value.get("evidence")
        if evidence is not None and not isinstance(evidence, dict):
            errors.append(f"{path}.evidence must be an object when present")
    return errors
