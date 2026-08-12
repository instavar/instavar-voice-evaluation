from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .lineage import KINDS, fingerprint_artifact


RELATIONS = {"exact", "derived"}
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MUTABLE_REVISIONS = {"head", "latest", "main", "master", "unknown", "unversioned"}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _revision(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character git SHA")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{name} must contain only lowercase letters, digits, dots, underscores, and hyphens")
    return value


def _pinned_conversion_revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty pinned revision")
    normalized = value.strip()
    if normalized.lower() in MUTABLE_REVISIONS:
        raise ValueError(f"{name} must not use a mutable revision alias")
    return normalized


def _artifact_declarations(value: Any, *, path: str, base_dir: Path) -> dict[str, tuple[Path, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty array")
    declarations: dict[str, tuple[Path, str]] = {}
    for index, record in enumerate(value):
        record_path = f"{path}[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{record_path} must be an object")
        role = record.get("role")
        kind = record.get("kind")
        raw_path = record.get("path")
        _identifier(role, f"{record_path}.role")
        if role in declarations:
            raise ValueError(f"{path} contains a duplicate role: {role}")
        if kind not in KINDS:
            raise ValueError(f"{record_path}.kind must equal file or tree")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{record_path}.path must be a non-empty string")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        declarations[role] = (candidate, kind)
    return declarations


def _fingerprint_set(declarations: dict[str, tuple[Path, str]]) -> tuple[list[dict[str, Any]], str]:
    records = [
        fingerprint_artifact(path, role=role, kind=kind)
        for role, (path, kind) in sorted(declarations.items())
    ]
    empty_roles = [record["role"] for record in records if record["bytes"] <= 0]
    if empty_roles:
        raise ValueError("runtime artifacts must be non-empty: " + ", ".join(empty_roles))
    return records, _canonical_sha256(records)


def build_runtime_artifact_manifest(plan: Any, *, base_dir: Path) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("runtime artifact binding plan must be an object")
    artifact_set_id = _identifier(plan.get("artifact_set_id"), "artifact_set_id")
    producer = plan.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("producer must be an object")
    repository = producer.get("repository")
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("producer.repository must be a non-empty string")
    revision = _revision(producer.get("revision"), "producer.revision")

    source_declarations = _artifact_declarations(
        plan.get("source_artifacts"),
        path="source_artifacts",
        base_dir=base_dir,
    )
    source_records, source_set_sha256 = _fingerprint_set(source_declarations)

    raw_bindings = plan.get("runtime_bindings")
    if not isinstance(raw_bindings, list) or len(raw_bindings) < 2:
        raise ValueError("runtime_bindings must contain at least two runtimes")
    bindings: list[dict[str, Any]] = []
    runtime_ids: set[str] = set()
    for index, raw_binding in enumerate(raw_bindings):
        prefix = f"runtime_bindings[{index}]"
        if not isinstance(raw_binding, dict):
            raise ValueError(f"{prefix} must be an object")
        runtime_id = _identifier(raw_binding.get("runtime_id"), f"{prefix}.runtime_id")
        if runtime_id in runtime_ids:
            raise ValueError(f"runtime_bindings contains a duplicate runtime_id: {runtime_id}")
        runtime_ids.add(runtime_id)
        relation = raw_binding.get("relation")
        if relation not in RELATIONS:
            raise ValueError(f"{prefix}.relation must equal exact or derived")
        declarations = _artifact_declarations(
            raw_binding.get("artifacts"),
            path=f"{prefix}.artifacts",
            base_dir=base_dir,
        )
        records, artifact_set_sha256 = _fingerprint_set(declarations)
        binding: dict[str, Any] = {
            "runtime_id": runtime_id,
            "relation": relation,
            "artifact_set_sha256": artifact_set_sha256,
            "artifacts": records,
        }
        if relation == "exact":
            if records != source_records:
                raise ValueError(f"{prefix} claims exact relation but its artifacts differ from the source set")
        else:
            conversion = raw_binding.get("conversion")
            if not isinstance(conversion, dict):
                raise ValueError(f"{prefix}.conversion must be an object for a derived binding")
            tool = conversion.get("tool")
            conversion_revision = _pinned_conversion_revision(
                conversion.get("revision"),
                f"{prefix}.conversion.revision",
            )
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError(f"{prefix}.conversion.tool must be a non-empty string")
            binding["derived_from_sha256"] = source_set_sha256
            binding["conversion"] = {"tool": tool.strip(), "revision": conversion_revision}
        bindings.append(binding)

    return {
        "schema_version": "1.0.0",
        "artifact_set_id": artifact_set_id,
        "producer": {"repository": repository.strip(), "revision": revision},
        "source_artifact_set_sha256": source_set_sha256,
        "source_artifacts": source_records,
        "runtime_bindings": bindings,
        "evidence_boundary": (
            "Exact bindings prove that the declared runtime artifacts had the same content fingerprints as the source set. "
            "They do not prove that a runtime loaded those bytes or produced numerically or perceptually equivalent audio. "
            "Derived bindings record conversion provenance but do not establish equivalence with the source artifacts."
        ),
    }


def _validate_artifacts(value: Any, *, path: str) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(value, list) or not value:
        return [], [f"{path} must be a non-empty array"]
    roles: set[str] = set()
    records: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        prefix = f"{path}[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be an object")
            continue
        role = record.get("role")
        try:
            _identifier(role, f"{prefix}.role")
        except ValueError as error:
            errors.append(str(error))
        if isinstance(role, str) and role in roles:
            errors.append(f"{path} contains a duplicate role: {role}")
        elif isinstance(role, str):
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
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            errors.append(f"{prefix}.bytes must be a positive integer")
        count = record.get("file_count")
        if kind == "tree" and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
            errors.append(f"{prefix}.file_count must be a positive integer for a tree")
        if kind == "file" and "file_count" in record:
            errors.append(f"{prefix}.file_count is only valid for a tree")
        records.append(record)
    return records, errors


def validate_runtime_artifact_manifest(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["runtime artifact manifest must be an object"]
    if document.get("schema_version") != "1.0.0":
        errors.append("schema_version must equal 1.0.0")
    try:
        _identifier(document.get("artifact_set_id"), "artifact_set_id")
    except ValueError as error:
        errors.append(str(error))
    if not isinstance(document.get("evidence_boundary"), str) or not document["evidence_boundary"].strip():
        errors.append("evidence_boundary must be a non-empty string")
    producer = document.get("producer")
    if not isinstance(producer, dict):
        errors.append("producer must be an object")
    else:
        if not isinstance(producer.get("repository"), str) or not producer["repository"].strip():
            errors.append("producer.repository must be a non-empty string")
        try:
            _revision(producer.get("revision"), "producer.revision")
        except ValueError as error:
            errors.append(str(error))

    source_records, source_errors = _validate_artifacts(document.get("source_artifacts"), path="source_artifacts")
    errors.extend(source_errors)
    source_digest = document.get("source_artifact_set_sha256")
    if source_records and source_digest != _canonical_sha256(source_records):
        errors.append("source_artifact_set_sha256 does not match source_artifacts")

    bindings = document.get("runtime_bindings")
    if not isinstance(bindings, list) or len(bindings) < 2:
        errors.append("runtime_bindings must contain at least two runtimes")
        return errors
    runtime_ids: set[str] = set()
    for index, binding in enumerate(bindings):
        prefix = f"runtime_bindings[{index}]"
        if not isinstance(binding, dict):
            errors.append(f"{prefix} must be an object")
            continue
        runtime_id = binding.get("runtime_id")
        try:
            _identifier(runtime_id, f"{prefix}.runtime_id")
        except ValueError as error:
            errors.append(str(error))
        if isinstance(runtime_id, str) and runtime_id in runtime_ids:
            errors.append(f"runtime_bindings contains a duplicate runtime_id: {runtime_id}")
        elif isinstance(runtime_id, str):
            runtime_ids.add(runtime_id)
        relation = binding.get("relation")
        if relation not in RELATIONS:
            errors.append(f"{prefix}.relation must equal exact or derived")
        records, record_errors = _validate_artifacts(binding.get("artifacts"), path=f"{prefix}.artifacts")
        errors.extend(record_errors)
        digest = binding.get("artifact_set_sha256")
        if records and digest != _canonical_sha256(records):
            errors.append(f"{prefix}.artifact_set_sha256 does not match artifacts")
        if relation == "exact":
            if records and source_records and records != source_records:
                errors.append(f"{prefix} claims exact relation but its artifacts differ from the source set")
            if "conversion" in binding or "derived_from_sha256" in binding:
                errors.append(f"{prefix} exact binding must not declare conversion fields")
        elif relation == "derived":
            if binding.get("derived_from_sha256") != source_digest:
                errors.append(f"{prefix}.derived_from_sha256 must equal source_artifact_set_sha256")
            conversion = binding.get("conversion")
            if not isinstance(conversion, dict):
                errors.append(f"{prefix}.conversion must be an object for a derived binding")
            else:
                if not isinstance(conversion.get("tool"), str) or not conversion["tool"].strip():
                    errors.append(f"{prefix}.conversion.tool must be a non-empty string")
                try:
                    _pinned_conversion_revision(
                        conversion.get("revision"),
                        f"{prefix}.conversion.revision",
                    )
                except ValueError as error:
                    errors.append(str(error))
    return errors


def verify_runtime_artifact_manifest(document: Any, plan: Any, *, base_dir: Path) -> dict[str, Any]:
    errors = validate_runtime_artifact_manifest(document)
    if errors:
        raise ValueError("invalid runtime artifact manifest: " + "; ".join(errors))
    current = build_runtime_artifact_manifest(plan, base_dir=base_dir)
    if current != document:
        raise ValueError("runtime artifact manifest does not match current binding plan or artifact content")
    return {
        "schema_version": "1.0.0",
        "status": "passed",
        "artifact_set_id": document["artifact_set_id"],
        "source_artifact_set_sha256": document["source_artifact_set_sha256"],
        "exact_runtime_ids": sorted(
            binding["runtime_id"] for binding in document["runtime_bindings"] if binding["relation"] == "exact"
        ),
        "derived_runtime_ids": sorted(
            binding["runtime_id"] for binding in document["runtime_bindings"] if binding["relation"] == "derived"
        ),
        "evidence_boundary": document["evidence_boundary"],
    }


def exact_runtime_binding(document: Any, runtime_ids: set[str]) -> dict[str, Any]:
    errors = validate_runtime_artifact_manifest(document)
    if errors:
        raise ValueError("invalid runtime artifact manifest: " + "; ".join(errors))
    bindings = {binding["runtime_id"]: binding for binding in document["runtime_bindings"]}
    missing = sorted(runtime_ids - set(bindings))
    if missing:
        raise ValueError(f"runtime artifact manifest is missing runtimes: {missing}")
    nonexact = sorted(runtime_id for runtime_id in runtime_ids if bindings[runtime_id]["relation"] != "exact")
    if nonexact:
        raise ValueError(f"runtime comparison requires exact artifact bindings; non-exact runtimes={nonexact}")
    return {
        "artifact_set_id": document["artifact_set_id"],
        "source_artifact_set_sha256": document["source_artifact_set_sha256"],
        "runtime_ids": sorted(runtime_ids),
    }
