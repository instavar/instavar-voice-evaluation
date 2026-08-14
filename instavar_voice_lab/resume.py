from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .lineage import KINDS, fingerprint_artifact


SCHEMA_VERSION = "1.0.0"
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ADAPTATION_MODES = {"lora", "full_sft", "partial_sft", "prompt_adapter"}
CORE_STATE_ROLES = {
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "trainer_state",
    "rng_state",
}
IDENTITY_FIELDS = (
    "producer_repository",
    "producer_revision",
    "backend_id",
    "adaptation_mode",
    "base_artifact_sha256",
    "dataset_lineage_sha256",
    "training_controls_sha256",
    "initial_state_sha256",
    "target_updates",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{name} must contain only lowercase letters, digits, dots, underscores, and hyphens")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not GIT_SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40-character git SHA")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_path(raw_path: Any, *, name: str, base_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(raw_path).expanduser()
    return candidate if candidate.is_absolute() else base_dir / candidate


def _read_receipt(path: Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = fingerprint_artifact(path, role=name, kind="file")
    if record["bytes"] < 1:
        raise ValueError(f"{name} must be non-empty")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value, record


def _validate_receipt(receipt: dict[str, Any], *, expected_mode: str, name: str) -> dict[str, Any]:
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{name}.schema_version must equal {SCHEMA_VERSION}")
    _identifier(receipt.get("run_id"), f"{name}.run_id")
    repository = receipt.get("producer_repository")
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError(f"{name}.producer_repository must be a non-empty string")
    _git_sha(receipt.get("producer_revision"), f"{name}.producer_revision")
    _identifier(receipt.get("backend_id"), f"{name}.backend_id")
    if receipt.get("adaptation_mode") not in ADAPTATION_MODES:
        raise ValueError(f"{name}.adaptation_mode must be one of: {', '.join(sorted(ADAPTATION_MODES))}")
    for field in (
        "base_artifact_sha256",
        "dataset_lineage_sha256",
        "training_controls_sha256",
        "initial_state_sha256",
    ):
        _sha256(receipt.get(field), f"{name}.{field}")
    target_updates = _positive_integer(receipt.get("target_updates"), f"{name}.target_updates")
    completed_updates = _positive_integer(receipt.get("completed_updates"), f"{name}.completed_updates")
    if completed_updates != target_updates:
        raise ValueError(f"{name}.completed_updates must equal target_updates")
    if receipt.get("execution_mode") != expected_mode:
        raise ValueError(f"{name}.execution_mode must equal {expected_mode}")

    if expected_mode == "uninterrupted":
        if "resume" in receipt:
            raise ValueError(f"{name}.resume must be absent for an uninterrupted run")
        return receipt

    resume = receipt.get("resume")
    if not isinstance(resume, dict):
        raise ValueError(f"{name}.resume must be an object")
    if resume.get("interruption_observed") is not True:
        raise ValueError(f"{name}.resume.interruption_observed must equal true")
    checkpoint_updates = _positive_integer(
        resume.get("checkpoint_completed_updates"),
        f"{name}.resume.checkpoint_completed_updates",
    )
    resumed_updates = _positive_integer(
        resume.get("resumed_from_completed_updates"),
        f"{name}.resume.resumed_from_completed_updates",
    )
    if checkpoint_updates != resumed_updates:
        raise ValueError(
            f"{name}.resume.resumed_from_completed_updates must equal checkpoint_completed_updates"
        )
    if checkpoint_updates >= target_updates:
        raise ValueError(f"{name}.resume checkpoint must precede target_updates")
    signal = resume.get("interruption_signal")
    if not isinstance(signal, str) or not signal.strip():
        raise ValueError(f"{name}.resume.interruption_signal must be a non-empty string")
    _sha256(
        resume.get("interruption_receipt_sha256"),
        f"{name}.resume.interruption_receipt_sha256",
    )
    return receipt


def _artifact_declarations(value: Any, *, name: str, base_dir: Path) -> dict[str, tuple[Path, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    declarations: dict[str, tuple[Path, str]] = {}
    for index, raw_record in enumerate(value):
        prefix = f"{name}[{index}]"
        if not isinstance(raw_record, dict):
            raise ValueError(f"{prefix} must be an object")
        role = _identifier(raw_record.get("role"), f"{prefix}.role")
        if role in declarations:
            raise ValueError(f"{name} contains a duplicate role: {role}")
        kind = raw_record.get("kind")
        if kind not in KINDS:
            raise ValueError(f"{prefix}.kind must equal file or tree")
        declarations[role] = (
            _resolve_path(raw_record.get("path"), name=f"{prefix}.path", base_dir=base_dir),
            kind,
        )
    return declarations


def _identity_tokens(path: Path, *, kind: str) -> set[tuple[int, int]]:
    if path.is_symlink():
        raise ValueError(f"resume artifact must not be a symlink: {path}")
    resolved = path.resolve()
    if kind == "file":
        if not resolved.is_file():
            raise FileNotFoundError(f"resume artifact file not found: {resolved}")
        stat = resolved.stat()
        return {(stat.st_dev, stat.st_ino)}
    if not resolved.is_dir():
        raise FileNotFoundError(f"resume artifact tree not found: {resolved}")
    tokens: set[tuple[int, int]] = set()
    for artifact in sorted(resolved.rglob("*")):
        if artifact.is_symlink():
            raise ValueError(f"resume artifact tree contains a symlink: {artifact}")
        if artifact.is_file():
            stat = artifact.stat()
            token = (stat.st_dev, stat.st_ino)
            if token in tokens:
                raise ValueError(f"resume artifact tree contains a hardlink alias: {artifact}")
            tokens.add(token)
        elif not artifact.is_dir():
            raise ValueError(f"resume artifact tree contains an unsupported entry: {artifact}")
    if not tokens:
        raise ValueError(f"resume artifact tree contains no files: {resolved}")
    return tokens


def _fingerprint_run_artifacts(
    declarations: dict[str, tuple[Path, str]],
    *,
    run_name: str,
) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    records: list[dict[str, Any]] = []
    all_tokens: set[tuple[int, int]] = set()
    for role, (path, kind) in sorted(declarations.items()):
        tokens = _identity_tokens(path, kind=kind)
        overlap = all_tokens.intersection(tokens)
        if overlap:
            raise ValueError(f"{run_name} artifact roles must not share files or hardlinks: {role}")
        all_tokens.update(tokens)
        record = fingerprint_artifact(path, role=role, kind=kind)
        if record["bytes"] < 1:
            raise ValueError(f"{run_name} artifact must be non-empty: {role}")
        records.append(record)
    return records, all_tokens


def compare_resume_artifacts(plan: Any, *, base_dir: Path) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("resume comparison plan must be an object")
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    comparison_id = _identifier(plan.get("comparison_id"), "comparison_id")

    required_roles = plan.get("required_artifact_roles")
    if not isinstance(required_roles, list) or not required_roles:
        raise ValueError("required_artifact_roles must be a non-empty array")
    normalized_roles = [
        _identifier(role, f"required_artifact_roles[{index}]")
        for index, role in enumerate(required_roles)
    ]
    if len(set(normalized_roles)) != len(normalized_roles):
        raise ValueError("required_artifact_roles must not contain duplicates")
    missing_core = sorted(CORE_STATE_ROLES - set(normalized_roles))
    if missing_core:
        raise ValueError("required_artifact_roles must include core state roles: " + ", ".join(missing_core))

    runs: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    receipt_records: dict[str, dict[str, Any]] = {}
    receipt_paths: dict[str, Path] = {}
    receipt_tokens: dict[str, set[tuple[int, int]]] = {}
    artifact_declarations: dict[str, dict[str, tuple[Path, str]]] = {}
    artifact_records: dict[str, list[dict[str, Any]]] = {}
    artifact_tokens: dict[str, set[tuple[int, int]]] = {}
    for run_name, expected_mode in (("uninterrupted", "uninterrupted"), ("resumed", "interrupted_resumed")):
        run = plan.get(run_name)
        if not isinstance(run, dict):
            raise ValueError(f"{run_name} must be an object")
        runs[run_name] = run
        receipt_path = _resolve_path(run.get("receipt"), name=f"{run_name}.receipt", base_dir=base_dir)
        receipt_paths[run_name] = receipt_path
        receipt_tokens[run_name] = _identity_tokens(receipt_path, kind="file")
        receipt, receipt_record = _read_receipt(receipt_path, name=f"{run_name}_receipt")
        receipts[run_name] = _validate_receipt(receipt, expected_mode=expected_mode, name=f"{run_name}.receipt")
        receipt_records[run_name] = receipt_record
        declarations = _artifact_declarations(
            run.get("artifacts"),
            name=f"{run_name}.artifacts",
            base_dir=base_dir,
        )
        artifact_declarations[run_name] = declarations
        if set(declarations) != set(normalized_roles):
            missing = sorted(set(normalized_roles) - set(declarations))
            extra = sorted(set(declarations) - set(normalized_roles))
            raise ValueError(f"{run_name}.artifacts role mismatch: missing={missing}, extra={extra}")
        records, tokens = _fingerprint_run_artifacts(declarations, run_name=run_name)
        artifact_records[run_name] = records
        artifact_tokens[run_name] = tokens

    if receipts["uninterrupted"]["run_id"] == receipts["resumed"]["run_id"]:
        raise ValueError("uninterrupted and resumed run_id values must differ")
    if receipt_tokens["uninterrupted"].intersection(receipt_tokens["resumed"]):
        raise ValueError("uninterrupted and resumed receipts must not share files or hardlinks")
    for field in IDENTITY_FIELDS:
        if receipts["uninterrupted"][field] != receipts["resumed"][field]:
            raise ValueError(f"run conditioning mismatch: {field}")
    if artifact_tokens["uninterrupted"].intersection(artifact_tokens["resumed"]):
        raise ValueError("uninterrupted and resumed artifacts must not share files or hardlinks")
    all_artifact_tokens = artifact_tokens["uninterrupted"].union(artifact_tokens["resumed"])
    all_receipt_tokens = receipt_tokens["uninterrupted"].union(receipt_tokens["resumed"])
    if all_artifact_tokens.intersection(all_receipt_tokens):
        raise ValueError("run receipts must not alias compared artifacts")

    interruption_receipt_path = _resolve_path(
        plan.get("interruption_receipt"),
        name="interruption_receipt",
        base_dir=base_dir,
    )
    interruption_record = fingerprint_artifact(
        interruption_receipt_path,
        role="interruption_receipt",
        kind="file",
    )
    if interruption_record["bytes"] < 1:
        raise ValueError("interruption_receipt must be non-empty")
    interruption_tokens = _identity_tokens(interruption_receipt_path, kind="file")
    if interruption_tokens.intersection(all_artifact_tokens.union(all_receipt_tokens)):
        raise ValueError("interruption_receipt must not alias run receipts or compared artifacts")
    expected_interruption_sha = receipts["resumed"]["resume"]["interruption_receipt_sha256"]
    if interruption_record["sha256"] != expected_interruption_sha:
        raise ValueError("interruption_receipt does not match the resumed run receipt")

    uninterrupted_by_role = {record["role"]: record for record in artifact_records["uninterrupted"]}
    resumed_by_role = {record["role"]: record for record in artifact_records["resumed"]}
    role_results = []
    for role in sorted(normalized_roles):
        reference = uninterrupted_by_role[role]
        candidate = resumed_by_role[role]
        if reference["kind"] != candidate["kind"]:
            raise ValueError(f"artifact kind mismatch for role: {role}")
        exact = reference == candidate
        role_results.append(
            {
                "role": role,
                "kind": reference["kind"],
                "byte_exact": exact,
                "uninterrupted_sha256": reference["sha256"],
                "resumed_sha256": candidate["sha256"],
                "uninterrupted_bytes": reference["bytes"],
                "resumed_bytes": candidate["bytes"],
            }
        )
    mismatched_roles = [row["role"] for row in role_results if not row["byte_exact"]]
    exact = not mismatched_roles

    for run_name in ("uninterrupted", "resumed"):
        expected_by_role = {record["role"]: record for record in artifact_records[run_name]}
        for role, (path, kind) in sorted(artifact_declarations[run_name].items()):
            if fingerprint_artifact(path, role=role, kind=kind) != expected_by_role[role]:
                raise ValueError(f"{run_name} artifact mutated during comparison: {role}")
        if (
            fingerprint_artifact(receipt_paths[run_name], role=f"{run_name}_receipt", kind="file")
            != receipt_records[run_name]
        ):
            raise ValueError(f"{run_name} receipt mutated during comparison")
    if (
        fingerprint_artifact(interruption_receipt_path, role="interruption_receipt", kind="file")
        != interruption_record
    ):
        raise ValueError("interruption_receipt mutated during comparison")

    report = {
        "schema_version": SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "status": "passed" if exact else "negative_result",
        "claim_tier": "byte_exact_declared_artifact_set" if exact else "artifact_mismatch",
        "conditioning": {field: receipts["uninterrupted"][field] for field in IDENTITY_FIELDS},
        "runs": {
            run_name: {
                "run_id": receipts[run_name]["run_id"],
                "receipt_sha256": receipt_records[run_name]["sha256"],
                "artifact_set_sha256": _canonical_sha256(artifact_records[run_name]),
            }
            for run_name in ("uninterrupted", "resumed")
        },
        "interruption": {
            **receipts["resumed"]["resume"],
            "receipt_verified": True,
        },
        "independent_artifact_storage_verified": True,
        "required_artifact_roles": sorted(normalized_roles),
        "role_results": role_results,
        "matched_roles": [row["role"] for row in role_results if row["byte_exact"]],
        "mismatched_roles": mismatched_roles,
        "exact_resume_artifact_equivalence": exact,
        "proves_numerical_resume_equivalence": False,
        "proves_training_semantics": False,
        "proves_model_quality": False,
        "evidence_boundary": (
            "This report rehashes independent declared final artifacts and an interruption receipt, and checks that "
            "the two run receipts declare identical conditioning. Exact bytes qualify only the named artifact roles. "
            "The report does not independently prove that a trainer honored the receipts, that omitted or hidden "
            "state is equal, that floating-point trajectories are numerically equivalent, or that model quality "
            "is equal."
        ),
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report
