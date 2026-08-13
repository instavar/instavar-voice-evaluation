from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Iterator


DEFAULT_MAX_MANIFEST_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_MANIFEST_LINE_BYTES = 8 * 1024 * 1024
_MANIFEST_READ_CHUNK_BYTES = 64 * 1024


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _stable_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(source.fileno())
    current = path.stat()
    if _stat_fingerprint(before) != _stat_fingerprint(after) or _stat_fingerprint(after) != _stat_fingerprint(current):
        raise ValueError("audio file changed while its content hash was computed")
    return digest.hexdigest()


class _StableManifestSource:
    def __init__(self, path: Path, *, max_bytes: int, max_line_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.max_line_bytes = max_line_bytes
        self.total_bytes = 0
        self._digest = hashlib.sha256()
        self._source: BinaryIO | None = None
        self._before: os.stat_result | None = None
        self._finished = False

    def __enter__(self) -> _StableManifestSource:
        self._source = self.path.open("rb")
        self._before = os.fstat(self._source.fileno())
        if self._before.st_size > self.max_bytes:
            self._source.close()
            self._source = None
            raise ValueError(f"manifest declares {self._before.st_size} bytes, above the {self.max_bytes}-byte limit")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._source is not None:
            self._source.close()

    def lines(self) -> Iterator[tuple[int, str | None, str | None]]:
        if self._source is None or self._finished:
            raise RuntimeError("manifest source is not open for one-pass reading")
        line_number = 0
        while True:
            chunks: list[bytes] = []
            line_bytes = 0
            found_data = False
            while True:
                chunk = self._source.readline(_MANIFEST_READ_CHUNK_BYTES)
                if not chunk:
                    break
                found_data = True
                self._digest.update(chunk)
                self.total_bytes += len(chunk)
                if self.total_bytes > self.max_bytes:
                    raise ValueError(f"manifest exceeded the {self.max_bytes}-byte limit while reading")
                line_bytes += len(chunk)
                if line_bytes <= self.max_line_bytes:
                    chunks.append(chunk)
                if chunk.endswith(b"\n"):
                    break
            if not found_data:
                break
            line_number += 1
            if line_bytes > self.max_line_bytes:
                yield (
                    line_number,
                    None,
                    f"manifest line exceeds the {self.max_line_bytes}-byte limit",
                )
                continue
            raw = b"".join(chunks)
            try:
                yield line_number, raw.decode("utf-8"), None
            except UnicodeDecodeError as error:
                yield (
                    line_number,
                    None,
                    f"invalid UTF-8 at byte {error.start}: {error.reason}",
                )
        self._finished = True

    def finish(self) -> str:
        if self._source is None or self._before is None or not self._finished:
            raise RuntimeError("manifest source must be fully consumed before finishing")
        after = os.fstat(self._source.fileno())
        current = self.path.stat()
        if (
            self.total_bytes != after.st_size
            or _stat_fingerprint(self._before) != _stat_fingerprint(after)
            or _stat_fingerprint(after) != _stat_fingerprint(current)
        ):
            raise ValueError("manifest file changed while it was audited")
        return self._digest.hexdigest()


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
    max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
    max_manifest_line_bytes: int = DEFAULT_MAX_MANIFEST_LINE_BYTES,
) -> dict[str, Any]:
    for name, value, ceiling in (
        ("max_manifest_bytes", max_manifest_bytes, DEFAULT_MAX_MANIFEST_BYTES),
        (
            "max_manifest_line_bytes",
            max_manifest_line_bytes,
            DEFAULT_MAX_MANIFEST_LINE_BYTES,
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value > ceiling:
            raise ValueError(f"{name} cannot exceed the {ceiling}-byte safety ceiling")

    required_split_order = ("train", "validation", "test")
    required_splits = set(required_split_order)
    errors: list[str] = []
    warnings: list[str] = []
    if set(splits) != required_splits:
        errors.append("splits must declare exactly train, validation, and test")

    seen_audio: dict[str, tuple[str, int]] = {}
    seen_audio_content: dict[str, tuple[str, int]] = {}
    seen_group: dict[str, tuple[str, int]] = {}
    seen_text: dict[str, tuple[str, int]] = {}
    split_reports: dict[str, dict[str, Any]] = {}

    split_order = [name for name in required_split_order if name in splits]
    split_order.extend(sorted(set(splits) - required_splits))
    for split_name in split_order:
        path = splits[split_name]
        report = {
            "path": str(path),
            "manifest_sha256": "",
            "manifest_bytes": 0,
            "rows": 0,
            "valid_rows": 0,
            "groups": 0,
        }
        split_reports[split_name] = report
        if not path.is_file():
            errors.append(f"{split_name}: manifest not found: {path}")
            continue
        split_groups: set[str] = set()
        try:
            with _StableManifestSource(
                path,
                max_bytes=max_manifest_bytes,
                max_line_bytes=max_manifest_line_bytes,
            ) as source:
                for line_number, raw, line_error in source.lines():
                    if line_error is not None:
                        report["rows"] += 1
                        errors.append(f"{split_name}:{line_number}: {line_error}")
                        continue
                    assert raw is not None
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
                    normalized_text = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
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
                    try:
                        audio_sha256 = _stable_file_sha256(audio_path)
                    except OSError as error:
                        errors.append(f"{split_name}:{line_number}: cannot hash audio file: {error}")
                        continue
                    except ValueError as error:
                        errors.append(f"{split_name}:{line_number}: {error}")
                        continue
                    previous_content = seen_audio_content.get(audio_sha256)
                    if previous_content:
                        errors.append(
                            f"{split_name}:{line_number}: audio content duplicates "
                            f"{previous_content[0]}:{previous_content[1]} under a different path"
                        )
                        continue
                    seen_audio_content[audio_sha256] = (split_name, line_number)

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
                report["manifest_sha256"] = source.finish()
                report["manifest_bytes"] = source.total_bytes
        except (OSError, ValueError) as error:
            errors.append(f"{split_name}: cannot audit manifest: {error}")

        report["groups"] = len(split_groups)
        if report["rows"] == 0:
            errors.append(f"{split_name}: manifest has no rows")
        elif report["valid_rows"] == 0:
            errors.append(f"{split_name}: manifest has no valid rows")

    return {
        "schema_version": "1.1.0",
        "status": "passed" if not errors else "failed",
        "audio_field": audio_field,
        "text_field": text_field,
        "group_field": group_field,
        "limits": {
            "max_manifest_bytes": max_manifest_bytes,
            "max_manifest_line_bytes": max_manifest_line_bytes,
        },
        "grouped_split_verified": bool(group_field) and not errors,
        "splits": split_reports,
        "errors": errors,
        "warnings": warnings,
    }
