from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .lineage import KINDS, fingerprint_artifact


SCHEMA_VERSION = "1.0.0"
LIVE_CONDITIONING_SCHEMA_VERSION = "1.1.0"
SCHEMA_VERSIONS = {SCHEMA_VERSION, LIVE_CONDITIONING_SCHEMA_VERSION}
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
IDENTITY_ARTIFACT_FIELDS = {
    "base_artifact": "base_artifact_sha256",
    "dataset_lineage": "dataset_lineage_sha256",
    "training_controls": "training_controls_sha256",
    "initial_state": "initial_state_sha256",
}


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


def _reject_output_alias(
    output_path: Path,
    artifacts: dict[str, tuple[Path, str]],
    *,
    name: str,
) -> None:
    output = output_path.expanduser().resolve()
    for role, (path, kind) in sorted(artifacts.items()):
        resolved = path.expanduser().resolve()
        aliases = output == resolved or (kind == "tree" and output.is_relative_to(resolved))
        if aliases:
            raise ValueError(f"{name} must not overwrite or be created inside input artifact: {role}")


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


def _validate_fingerprint_record(record: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{name} must be an object")
    role = _identifier(record.get("role"), f"{name}.role")
    kind = record.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{name}.kind must equal file or tree")
    _sha256(record.get("sha256"), f"{name}.sha256")
    size = _positive_integer(record.get("bytes"), f"{name}.bytes")
    expected_keys = {"role", "kind", "sha256", "bytes"}
    if kind == "tree":
        _positive_integer(record.get("file_count"), f"{name}.file_count")
        expected_keys.add("file_count")
    if set(record) != expected_keys:
        raise ValueError(f"{name} fields must equal: {', '.join(sorted(expected_keys))}")
    return {**record, "role": role, "bytes": size}


def _validate_identity_artifacts(receipt: dict[str, Any], *, name: str) -> list[dict[str, Any]]:
    raw_records = receipt.get("identity_artifacts")
    if not isinstance(raw_records, list):
        raise ValueError(f"{name}.identity_artifacts must be an array")
    records = [
        _validate_fingerprint_record(record, name=f"{name}.identity_artifacts[{index}]")
        for index, record in enumerate(raw_records)
    ]
    by_role = {record["role"]: record for record in records}
    if len(by_role) != len(records):
        raise ValueError(f"{name}.identity_artifacts must not contain duplicate roles")
    if set(by_role) != set(IDENTITY_ARTIFACT_FIELDS):
        missing = sorted(set(IDENTITY_ARTIFACT_FIELDS) - set(by_role))
        extra = sorted(set(by_role) - set(IDENTITY_ARTIFACT_FIELDS))
        raise ValueError(f"{name}.identity_artifacts role mismatch: missing={missing}, extra={extra}")
    for role, field in IDENTITY_ARTIFACT_FIELDS.items():
        if by_role[role]["sha256"] != receipt[field]:
            raise ValueError(f"{name}.{field} does not match identity_artifacts role {role}")
    return records


def _validate_receipt(receipt: dict[str, Any], *, expected_mode: str, name: str) -> dict[str, Any]:
    schema_version = receipt.get("schema_version")
    if schema_version not in SCHEMA_VERSIONS:
        raise ValueError(f"{name}.schema_version must be one of: {', '.join(sorted(SCHEMA_VERSIONS))}")
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
    if schema_version == LIVE_CONDITIONING_SCHEMA_VERSION:
        _validate_identity_artifacts(receipt, name=name)
    elif "identity_artifacts" in receipt:
        raise ValueError(f"{name}.identity_artifacts requires schema_version {LIVE_CONDITIONING_SCHEMA_VERSION}")

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


def build_resume_run_receipt(
    *,
    run_id: str,
    producer_repository: str,
    producer_revision: str,
    backend_id: str,
    adaptation_mode: str,
    target_updates: int,
    completed_updates: int,
    execution_mode: str,
    identity_artifacts: dict[str, tuple[Path, str]],
    interruption_receipt: Path | None = None,
    checkpoint_completed_updates: int | None = None,
    resumed_from_completed_updates: int | None = None,
    interruption_signal: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if execution_mode not in {"uninterrupted", "interrupted_resumed"}:
        raise ValueError("execution_mode must equal uninterrupted or interrupted_resumed")
    if set(identity_artifacts) != set(IDENTITY_ARTIFACT_FIELDS):
        missing = sorted(set(IDENTITY_ARTIFACT_FIELDS) - set(identity_artifacts))
        extra = sorted(set(identity_artifacts) - set(IDENTITY_ARTIFACT_FIELDS))
        raise ValueError(f"identity_artifacts role mismatch: missing={missing}, extra={extra}")
    if output_path is not None:
        _reject_output_alias(output_path, identity_artifacts, name="receipt output")
        if (
            interruption_receipt is not None
            and output_path.expanduser().resolve() == interruption_receipt.expanduser().resolve()
        ):
            raise ValueError("receipt output must not overwrite interruption_receipt")

    records: list[dict[str, Any]] = []
    tokens: set[tuple[int, int]] = set()
    for role, (path, kind) in sorted(identity_artifacts.items()):
        current_tokens = _identity_tokens(path, kind=kind)
        if tokens.intersection(current_tokens):
            raise ValueError(f"identity_artifacts must not share files or hardlinks: {role}")
        tokens.update(current_tokens)
        record = fingerprint_artifact(path, role=role, kind=kind)
        if record["bytes"] < 1:
            raise ValueError(f"identity artifact must be non-empty: {role}")
        records.append(record)

    receipt: dict[str, Any] = {
        "schema_version": LIVE_CONDITIONING_SCHEMA_VERSION,
        "run_id": run_id,
        "producer_repository": producer_repository,
        "producer_revision": producer_revision,
        "backend_id": backend_id,
        "adaptation_mode": adaptation_mode,
        **{
            field: next(record["sha256"] for record in records if record["role"] == role)
            for role, field in IDENTITY_ARTIFACT_FIELDS.items()
        },
        "identity_artifacts": records,
        "target_updates": target_updates,
        "completed_updates": completed_updates,
        "execution_mode": execution_mode,
    }
    if execution_mode == "interrupted_resumed":
        if interruption_receipt is None:
            raise ValueError("interruption_receipt is required for interrupted_resumed")
        interruption_tokens = _identity_tokens(interruption_receipt, kind="file")
        if tokens.intersection(interruption_tokens):
            raise ValueError("interruption_receipt must not alias an identity artifact")
        interruption_record = fingerprint_artifact(
            interruption_receipt,
            role="interruption_receipt",
            kind="file",
        )
        if interruption_record["bytes"] < 1:
            raise ValueError("interruption_receipt must be non-empty")
        receipt["resume"] = {
            "interruption_observed": True,
            "checkpoint_completed_updates": checkpoint_completed_updates,
            "resumed_from_completed_updates": resumed_from_completed_updates,
            "interruption_signal": interruption_signal,
            "interruption_receipt_sha256": interruption_record["sha256"],
        }
    else:
        unexpected = {
            "interruption_receipt": interruption_receipt,
            "checkpoint_completed_updates": checkpoint_completed_updates,
            "resumed_from_completed_updates": resumed_from_completed_updates,
            "interruption_signal": interruption_signal,
        }
        present = sorted(name for name, value in unexpected.items() if value is not None)
        if present:
            raise ValueError("uninterrupted receipt must not include resume inputs: " + ", ".join(present))

    _validate_receipt(receipt, expected_mode=execution_mode, name="receipt")
    expected_by_role = {record["role"]: record for record in records}
    for role, (path, kind) in sorted(identity_artifacts.items()):
        if fingerprint_artifact(path, role=role, kind=kind) != expected_by_role[role]:
            raise ValueError(f"identity artifact mutated while building receipt: {role}")
    if execution_mode == "interrupted_resumed":
        assert interruption_receipt is not None
        if (
            fingerprint_artifact(interruption_receipt, role="interruption_receipt", kind="file")["sha256"]
            != receipt["resume"]["interruption_receipt_sha256"]
        ):
            raise ValueError("interruption_receipt mutated while building receipt")
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


def compare_resume_artifacts(
    plan: Any,
    *,
    base_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("resume comparison plan must be an object")
    plan_schema_version = plan.get("schema_version")
    if plan_schema_version not in SCHEMA_VERSIONS:
        raise ValueError(f"schema_version must be one of: {', '.join(sorted(SCHEMA_VERSIONS))}")
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

    conditioning_declarations: dict[str, tuple[Path, str]] = {}
    conditioning_records: list[dict[str, Any]] = []
    conditioning_tokens: set[tuple[int, int]] = set()
    if plan_schema_version == LIVE_CONDITIONING_SCHEMA_VERSION:
        conditioning_declarations = _artifact_declarations(
            plan.get("conditioning_artifacts"),
            name="conditioning_artifacts",
            base_dir=base_dir,
        )
        if set(conditioning_declarations) != set(IDENTITY_ARTIFACT_FIELDS):
            missing = sorted(set(IDENTITY_ARTIFACT_FIELDS) - set(conditioning_declarations))
            extra = sorted(set(conditioning_declarations) - set(IDENTITY_ARTIFACT_FIELDS))
            raise ValueError(f"conditioning_artifacts role mismatch: missing={missing}, extra={extra}")
        conditioning_records, conditioning_tokens = _fingerprint_run_artifacts(
            conditioning_declarations,
            run_name="conditioning",
        )
    elif "conditioning_artifacts" in plan:
        raise ValueError(
            f"conditioning_artifacts requires schema_version {LIVE_CONDITIONING_SCHEMA_VERSION}"
        )
    if output_path is not None:
        _reject_output_alias(output_path, conditioning_declarations, name="comparison output")

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
        receipt_path = _resolve_path(run.get("receipt"), name=f"{run_name}.receipt", base_dir=base_dir)
        receipt_paths[run_name] = receipt_path
        receipt_tokens[run_name] = _identity_tokens(receipt_path, kind="file")
        receipt, receipt_record = _read_receipt(receipt_path, name=f"{run_name}_receipt")
        receipts[run_name] = _validate_receipt(receipt, expected_mode=expected_mode, name=f"{run_name}.receipt")
        if plan_schema_version == LIVE_CONDITIONING_SCHEMA_VERSION:
            if receipts[run_name]["schema_version"] != LIVE_CONDITIONING_SCHEMA_VERSION:
                raise ValueError(
                    f"{run_name}.receipt must use schema_version {LIVE_CONDITIONING_SCHEMA_VERSION} "
                    "for live conditioning"
                )
            receipt_identity = _validate_identity_artifacts(
                receipts[run_name],
                name=f"{run_name}.receipt",
            )
            if (
                {record["role"]: record for record in receipt_identity}
                != {record["role"]: record for record in conditioning_records}
            ):
                raise ValueError(f"{run_name}.receipt identity_artifacts do not match live conditioning")
        receipt_records[run_name] = receipt_record
        declarations = _artifact_declarations(
            run.get("artifacts"),
            name=f"{run_name}.artifacts",
            base_dir=base_dir,
        )
        artifact_declarations[run_name] = declarations
        if output_path is not None:
            _reject_output_alias(output_path, declarations, name="comparison output")
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
    if conditioning_tokens.intersection(all_artifact_tokens.union(all_receipt_tokens)):
        raise ValueError("conditioning artifacts must not alias run receipts or compared artifacts")

    interruption_receipt_path = _resolve_path(
        plan.get("interruption_receipt"),
        name="interruption_receipt",
        base_dir=base_dir,
    )
    if output_path is not None:
        comparison_output = output_path.expanduser().resolve()
        protected_files = {
            "interruption_receipt": interruption_receipt_path,
            **{f"{run_name}_receipt": path for run_name, path in receipt_paths.items()},
        }
        for protected_name, path in protected_files.items():
            if comparison_output == path.expanduser().resolve():
                raise ValueError(f"comparison output must not overwrite input file: {protected_name}")
    interruption_record = fingerprint_artifact(
        interruption_receipt_path,
        role="interruption_receipt",
        kind="file",
    )
    if interruption_record["bytes"] < 1:
        raise ValueError("interruption_receipt must be non-empty")
    interruption_tokens = _identity_tokens(interruption_receipt_path, kind="file")
    if interruption_tokens.intersection(
        all_artifact_tokens.union(all_receipt_tokens).union(conditioning_tokens)
    ):
        raise ValueError(
            "interruption_receipt must not alias conditioning artifacts, run receipts, or compared artifacts"
        )
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
    expected_conditioning = {record["role"]: record for record in conditioning_records}
    for role, (path, kind) in sorted(conditioning_declarations.items()):
        if fingerprint_artifact(path, role=role, kind=kind) != expected_conditioning[role]:
            raise ValueError(f"conditioning artifact mutated during comparison: {role}")
    if (
        fingerprint_artifact(interruption_receipt_path, role="interruption_receipt", kind="file")
        != interruption_record
    ):
        raise ValueError("interruption_receipt mutated during comparison")

    report = {
        "schema_version": LIVE_CONDITIONING_SCHEMA_VERSION,
        "plan_schema_version": plan_schema_version,
        "comparison_id": comparison_id,
        "status": "passed" if exact else "negative_result",
        "claim_tier": (
            "byte_exact_live_conditioned_artifact_set"
            if exact and plan_schema_version == LIVE_CONDITIONING_SCHEMA_VERSION
            else "byte_exact_declared_artifact_set"
            if exact
            else "artifact_mismatch"
        ),
        "conditioning": {field: receipts["uninterrupted"][field] for field in IDENTITY_FIELDS},
        "conditioning_artifacts_verified": plan_schema_version == LIVE_CONDITIONING_SCHEMA_VERSION,
        "conditioning_artifacts": conditioning_records,
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
            "This report rehashes independent declared final artifacts and an interruption receipt, checks that "
            "the two run receipts declare identical conditioning, and records whether those conditioning artifacts "
            "were rehashed from a schema 1.1 plan. Exact bytes qualify only the named artifact roles. The report does "
            "not independently prove that a trainer honored the receipts, that omitted or hidden "
            "state is equal, that floating-point trajectories are numerically equivalent, or that model quality "
            "is equal."
        ),
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report
