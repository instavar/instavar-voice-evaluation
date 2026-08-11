from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_path(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return value["path"].strip()
    return ""


def audit_corpus(
    splits: dict[str, Path],
    *,
    audio_field: str = "audio",
    text_field: str = "text",
    group_field: str | None = None,
) -> dict[str, Any]:
    required_splits = {"train", "validation", "test"}
    errors: list[str] = []
    warnings: list[str] = []
    if set(splits) != required_splits:
        errors.append("splits must declare exactly train, validation, and test")

    seen_audio: dict[str, tuple[str, int]] = {}
    seen_group: dict[str, tuple[str, int]] = {}
    seen_text: dict[str, tuple[str, int]] = {}
    split_reports: dict[str, dict[str, Any]] = {}

    for split_name in sorted(splits):
        path = splits[split_name]
        report = {
            "path": str(path),
            "manifest_sha256": "",
            "rows": 0,
            "valid_rows": 0,
            "groups": 0,
        }
        split_reports[split_name] = report
        if not path.is_file():
            errors.append(f"{split_name}: manifest not found: {path}")
            continue
        report["manifest_sha256"] = _manifest_sha256(path)
        split_groups: set[str] = set()
        with path.open(encoding="utf-8") as source:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip():
                    continue
                report["rows"] += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as error:
                    errors.append(f"{split_name}:{line_number}: invalid JSON: {error.msg}")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"{split_name}:{line_number}: row must be an object")
                    continue

                text = row.get(text_field)
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"{split_name}:{line_number}: {text_field} must be non-empty text")
                    continue
                normalized_text = " ".join(text.split()).casefold()
                previous_text = seen_text.get(normalized_text)
                if previous_text and previous_text[0] != split_name:
                    warnings.append(
                        f"{split_name}:{line_number}: text duplicates {previous_text[0]}:{previous_text[1]}"
                    )
                else:
                    seen_text.setdefault(normalized_text, (split_name, line_number))

                audio_value = _audio_path(row.get(audio_field))
                if not audio_value:
                    errors.append(f"{split_name}:{line_number}: {audio_field} must name an audio path")
                    continue
                audio_path = Path(audio_value)
                if not audio_path.is_absolute():
                    audio_path = path.parent / audio_path
                resolved_audio = str(audio_path.resolve())
                if not audio_path.is_file():
                    errors.append(f"{split_name}:{line_number}: audio file not found: {audio_path}")
                    continue
                previous_audio = seen_audio.get(resolved_audio)
                if previous_audio:
                    errors.append(
                        f"{split_name}:{line_number}: audio duplicates {previous_audio[0]}:{previous_audio[1]}"
                    )
                    continue
                seen_audio[resolved_audio] = (split_name, line_number)

                if group_field:
                    group = row.get(group_field)
                    if not isinstance(group, str) or not group.strip():
                        errors.append(f"{split_name}:{line_number}: {group_field} must be a non-empty string")
                        continue
                    group = group.strip()
                    split_groups.add(group)
                    previous_group = seen_group.get(group)
                    if previous_group and previous_group[0] != split_name:
                        errors.append(
                            f"{split_name}:{line_number}: group {group!r} leaks from "
                            f"{previous_group[0]}:{previous_group[1]}"
                        )
                        continue
                    seen_group.setdefault(group, (split_name, line_number))

                report["valid_rows"] += 1

        report["groups"] = len(split_groups)
        if report["rows"] == 0:
            errors.append(f"{split_name}: manifest has no rows")
        elif report["valid_rows"] == 0:
            errors.append(f"{split_name}: manifest has no valid rows")

    return {
        "schema_version": "1.0.0",
        "status": "passed" if not errors else "failed",
        "audio_field": audio_field,
        "text_field": text_field,
        "group_field": group_field,
        "grouped_split_verified": bool(group_field) and not errors,
        "splits": split_reports,
        "errors": errors,
        "warnings": warnings,
    }
