from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

KINDS = {"file", "tree"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: Path) -> Path:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"lineage artifact must not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"lineage artifact file not found: {resolved}")
    return resolved


def _safe_tree(path: Path) -> Path:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"lineage artifact tree must not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"lineage artifact tree not found: {resolved}")
    return resolved


def fingerprint_artifact(path: Path, *, role: str, kind: str) -> dict[str, Any]:
    if not role.strip():
        raise ValueError("lineage artifact role must be non-empty")
    if kind not in KINDS:
        raise ValueError(f"unsupported lineage artifact kind: {kind}")
    if kind == "file":
        resolved = _safe_file(path)
        return {
            "role": role,
            "kind": kind,
            "sha256": _sha256(resolved),
            "bytes": resolved.stat().st_size,
        }

    root = _safe_tree(path)
    files: list[dict[str, Any]] = []
    for artifact in sorted(root.rglob("*")):
        if artifact.is_symlink():
            raise ValueError(f"lineage artifact tree contains a symlink: {artifact}")
        if artifact.is_file():
            files.append(
                {
                    "path": artifact.relative_to(root).as_posix(),
                    "sha256": _sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )
        elif not artifact.is_dir():
            raise ValueError(f"lineage artifact tree contains an unsupported entry: {artifact}")
    if not files:
        raise ValueError(f"lineage artifact tree contains no files: {root}")
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return {
        "role": role,
        "kind": kind,
        "sha256": digest.hexdigest(),
        "bytes": sum(record["bytes"] for record in files),
        "file_count": len(files),
    }


def build_dataset_lineage(
    *,
    lineage_id: str,
    producer_repository: str,
    producer_revision: str,
    inputs: dict[str, tuple[Path, str]],
    outputs: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    if not lineage_id.strip():
        raise ValueError("lineage_id must be non-empty")
    if not producer_repository.strip():
        raise ValueError("producer_repository must be non-empty")
    if len(producer_revision) != 40 or any(character not in "0123456789abcdef" for character in producer_revision):
        raise ValueError("producer_revision must be a lowercase 40-character git SHA")
    if not inputs or not outputs:
        raise ValueError("dataset lineage must contain at least one input and one output")
    overlap = sorted(set(inputs).intersection(outputs))
    if overlap:
        raise ValueError("dataset lineage roles must be unique across inputs and outputs: " + ", ".join(overlap))
    return {
        "schema_version": "1.0.0",
        "lineage_id": lineage_id,
        "producer": {"repository": producer_repository, "revision": producer_revision},
        "inputs": [fingerprint_artifact(path, role=role, kind=kind) for role, (path, kind) in sorted(inputs.items())],
        "outputs": [fingerprint_artifact(path, role=role, kind=kind) for role, (path, kind) in sorted(outputs.items())],
        "evidence_boundary": (
            "Content fingerprints bind the declared raw inputs to the declared prepared outputs. "
            "They detect substitution or mutation, but do not prove semantic correctness of the preparation algorithm."
        ),
    }


def validate_dataset_lineage(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["dataset lineage must be an object"]
    if document.get("schema_version") != "1.0.0":
        errors.append("schema_version must equal 1.0.0")
    for name in ("lineage_id", "evidence_boundary"):
        if not isinstance(document.get(name), str) or not document[name].strip():
            errors.append(f"{name} must be a non-empty string")
    producer = document.get("producer")
    if not isinstance(producer, dict):
        errors.append("producer must be an object")
    else:
        if not isinstance(producer.get("repository"), str) or not producer["repository"].strip():
            errors.append("producer.repository must be a non-empty string")
        revision = producer.get("revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            errors.append("producer.revision must be a lowercase 40-character git SHA")

    roles: set[str] = set()
    for collection_name in ("inputs", "outputs"):
        records = document.get(collection_name)
        if not isinstance(records, list) or not records:
            errors.append(f"{collection_name} must be a non-empty array")
            continue
        for index, record in enumerate(records):
            prefix = f"{collection_name}[{index}]"
            if not isinstance(record, dict):
                errors.append(f"{prefix} must be an object")
                continue
            role = record.get("role")
            if not isinstance(role, str) or not role.strip():
                errors.append(f"{prefix}.role must be a non-empty string")
            elif role in roles:
                errors.append(f"{prefix}.role must be unique across inputs and outputs")
            else:
                roles.add(role)
            kind = record.get("kind")
            if kind not in KINDS:
                errors.append(f"{prefix}.kind must equal file or tree")
            digest = record.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                errors.append(f"{prefix}.sha256 must be a lowercase SHA-256 digest")
            size = record.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                errors.append(f"{prefix}.bytes must be a non-negative integer")
            if kind == "tree":
                count = record.get("file_count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    errors.append(f"{prefix}.file_count must be a positive integer for a tree")
            elif "file_count" in record:
                errors.append(f"{prefix}.file_count is only valid for a tree")
    return errors


def verify_dataset_lineage(
    document: Any,
    *,
    producer_revision: str,
    inputs: dict[str, tuple[Path, str]],
    outputs: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    errors = validate_dataset_lineage(document)
    if errors:
        raise ValueError("invalid dataset lineage: " + "; ".join(errors))
    if document["producer"]["revision"] != producer_revision:
        raise ValueError("dataset lineage producer revision does not match the companion checkout")
    expected_roles = set(inputs).union(outputs)
    observed_records = {record["role"]: record for record in document["inputs"] + document["outputs"]}
    if set(observed_records) != expected_roles:
        missing = sorted(expected_roles - set(observed_records))
        extra = sorted(set(observed_records) - expected_roles)
        raise ValueError(f"dataset lineage role mismatch: missing={missing}, extra={extra}")
    for role, (path, kind) in sorted({**inputs, **outputs}.items()):
        current = fingerprint_artifact(path, role=role, kind=kind)
        if current != observed_records[role]:
            raise ValueError(f"dataset lineage artifact does not match current content: {role}")
    return {
        "schema_version": "1.0.0",
        "status": "passed",
        "lineage_id": document["lineage_id"],
        "producer": document["producer"],
        "input_roles": sorted(inputs),
        "output_roles": sorted(outputs),
        "evidence_boundary": document["evidence_boundary"],
    }
