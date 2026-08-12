from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import validate_capability_manifest, validate_experiment_manifest


STAGES = ("preflight", "train", "infer", "evaluate", "package")
SENSITIVE_ENV_FRAGMENTS = ("secret", "token", "password", "api_key", "credential")
RESERVED_ENV_NAMES = {
    "INSTAVAR_VOICE_EXPERIMENT_MANIFEST",
    "INSTAVAR_VOICE_STAGE",
    "INSTAVAR_VOICE_STAGE_RESULT",
    "INSTAVAR_VOICE_WORK_DIR",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_command(command: list[Any], *, work_dir: Path, stage_result: Path) -> list[str]:
    replacements = {
        "{python}": sys.executable,
        "{work_dir}": str(work_dir),
        "{stage_result}": str(stage_result),
    }
    expanded: list[str] = []
    for value in command:
        if not isinstance(value, str) or not value:
            raise ValueError("backend commands must contain non-empty string arguments")
        expanded.append(replacements.get(value, value))
    return expanded


def _artifact_record(path: Path, *, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"lifecycle artifacts must not be symlinks: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"expected lifecycle artifact not found: {path}")
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"lifecycle artifact escapes the work directory: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"lifecycle artifacts must not be empty: {path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "bytes": size,
    }


def _verify_artifact_records(records: list[dict[str, Any]], *, root: Path) -> None:
    for record in records:
        current = _artifact_record(root / record["path"], root=root)
        if current != record:
            raise ValueError(f"lifecycle artifact changed after hashing: {record['path']}")


def _validate_capability_binding(spec: dict[str, Any], spec_path: Path | None, errors: list[str]) -> None:
    starting_error_count = len(errors)
    binding = spec.get("capability_binding")
    if not isinstance(binding, dict):
        errors.append("capability_binding must be an object")
        return
    manifest_text = binding.get("manifest")
    adaptation = binding.get("adaptation")
    runtime_ids = binding.get("runtime_ids")
    if not isinstance(manifest_text, str) or not manifest_text:
        errors.append("capability_binding.manifest must be a non-empty relative path")
    else:
        manifest_path = Path(manifest_text)
        if manifest_path.is_absolute() or ".." in manifest_path.parts:
            errors.append("capability_binding.manifest must be a safe relative path")
    if not isinstance(adaptation, str) or not adaptation:
        errors.append("capability_binding.adaptation must be a non-empty string")
    if (
        not isinstance(runtime_ids, list)
        or not runtime_ids
        or any(not isinstance(runtime_id, str) or not runtime_id for runtime_id in runtime_ids)
    ):
        errors.append("capability_binding.runtime_ids must be a non-empty string array")
    elif len(runtime_ids) != len(set(runtime_ids)):
        errors.append("capability_binding.runtime_ids must not contain duplicates")
    if len(errors) > starting_error_count or spec_path is None:
        return
    manifest_path = spec_path.parent / manifest_text
    if manifest_path.is_symlink():
        errors.append("capability_binding.manifest must not be a symlink")
        return
    try:
        if not manifest_path.resolve().is_relative_to(spec_path.parent.resolve()):
            errors.append("capability_binding.manifest resolves outside the backend directory")
            return
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"capability_binding.manifest could not be read: {error}")
        return
    contract_errors = validate_capability_manifest(manifest)
    if contract_errors:
        errors.append("capability_binding.manifest is invalid: " + "; ".join(str(error) for error in contract_errors))
        return
    adaptation_record = manifest.get("adaptation", {}).get(adaptation)
    if not isinstance(adaptation_record, dict) or adaptation_record.get("status") not in {"supported", "experimental"}:
        errors.append("capability_binding.adaptation must name a supported or experimental adaptation")
    runtime_records = {
        runtime.get("id"): runtime
        for runtime in manifest.get("runtimes", [])
        if isinstance(runtime, dict) and isinstance(runtime.get("id"), str)
    }
    for runtime_id in runtime_ids:
        if runtime_id not in runtime_records:
            errors.append(f"capability_binding.runtime_ids names an unknown runtime: {runtime_id}")
        elif runtime_records[runtime_id].get("status") not in {"supported", "experimental", "unverified_for_adapter"}:
            errors.append(f"capability_binding.runtime_ids names an unsupported runtime: {runtime_id}")


def validate_backend_spec(spec: Any, *, spec_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["backend spec must be an object"]
    if spec.get("schema_version") not in {"1.0.0", "1.1.0", "1.2.0"}:
        errors.append("schema_version must equal 1.0.0, 1.1.0, or 1.2.0")
    if not isinstance(spec.get("backend_id"), str) or not spec["backend_id"].strip():
        errors.append("backend_id must be a non-empty string")
    commands = spec.get("commands")
    if not isinstance(commands, dict):
        errors.append("commands must be an object")
        return errors
    for stage in STAGES:
        command = commands.get(stage)
        if not isinstance(command, list) or not command:
            errors.append(f"commands.{stage} must be a non-empty argument array")
    expected = spec.get("expected_artifacts", {})
    if not isinstance(expected, dict):
        errors.append("expected_artifacts must be an object when present")
    else:
        owned_paths: set[str] = set()
        for stage in STAGES:
            paths = expected.get(stage)
            if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
                errors.append(f"expected_artifacts.{stage} must be an array of paths for a known stage")
                continue
            if len(paths) != len(set(paths)):
                errors.append(f"expected_artifacts.{stage} must not contain duplicates")
            for path_text in paths:
                artifact_path = Path(path_text)
                if artifact_path.is_absolute() or ".." in artifact_path.parts:
                    errors.append(f"expected_artifacts.{stage} contains an unsafe path: {path_text}")
                    continue
                if len(artifact_path.parts) < 2 or artifact_path.parts[0] != stage:
                    errors.append(f"expected_artifacts.{stage} must contain only paths owned by the {stage} stage: {path_text}")
                if artifact_path.name in {"stage-result.json", "stdout.log", "stderr.log"}:
                    errors.append(f"expected_artifacts.{stage} contains a runner-owned path: {path_text}")
                normalized = artifact_path.as_posix()
                if normalized in owned_paths:
                    errors.append(f"expected artifact path is declared more than once: {path_text}")
                owned_paths.add(normalized)
    environment = spec.get("environment", {})
    if not isinstance(environment, dict):
        errors.append("environment must be an object when present")
    else:
        for name, value in environment.items():
            if name in RESERVED_ENV_NAMES:
                errors.append(f"environment.{name} is runner-owned and must not be overridden")
            if any(fragment in name.casefold() for fragment in SENSITIVE_ENV_FRAGMENTS):
                errors.append(f"environment.{name} looks sensitive and must be supplied outside the spec")
            if not isinstance(value, str):
                errors.append(f"environment.{name} must be a string")
    timeouts = spec.get("timeout_seconds", {})
    if not isinstance(timeouts, dict):
        errors.append("timeout_seconds must be an object when present")
    elif spec.get("schema_version") in {"1.1.0", "1.2.0"}:
        for stage in STAGES:
            timeout = timeouts.get(stage)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append(f"timeout_seconds.{stage} must be a positive number")
    if spec.get("schema_version") == "1.2.0":
        required_environment = spec.get("required_environment")
        if not isinstance(required_environment, list):
            errors.append("required_environment must be an array")
        else:
            names: list[str] = []
            for index, requirement in enumerate(required_environment):
                if not isinstance(requirement, dict):
                    errors.append(f"required_environment[{index}] must be an object")
                    continue
                name = requirement.get("name")
                purpose = requirement.get("purpose")
                if (
                    not isinstance(name, str)
                    or not name
                    or not name.isascii()
                    or not name.replace("_", "").isalnum()
                    or not name[0].isalpha()
                ):
                    errors.append(f"required_environment[{index}].name must be a portable environment variable name")
                else:
                    names.append(name)
                    if name in RESERVED_ENV_NAMES:
                        errors.append(f"required_environment[{index}].name is runner-owned")
                if not isinstance(purpose, str) or not purpose.strip():
                    errors.append(f"required_environment[{index}].purpose must be a non-empty string")
            if len(names) != len(set(names)):
                errors.append("required_environment names must not contain duplicates")
            for name in sorted(set(names).intersection(environment)):
                errors.append(f"required_environment and environment both declare {name}")
        _validate_capability_binding(spec, spec_path, errors)
    return errors


def run_lifecycle(spec_path: Path, experiment_path: Path, work_dir: Path) -> dict[str, Any]:
    spec = _read_json(spec_path)
    errors = validate_backend_spec(spec, spec_path=spec_path)
    if errors:
        raise ValueError("; ".join(errors))
    experiment = _read_json(experiment_path)
    experiment_errors = validate_experiment_manifest(experiment)
    if experiment_errors:
        raise ValueError("invalid experiment manifest: " + "; ".join(str(error) for error in experiment_errors))
    experiment_id = str(experiment["experiment_id"]).strip()
    binding = spec.get("capability_binding")
    if isinstance(binding, dict) and experiment.get("adaptation_mode") != binding.get("adaptation"):
        raise ValueError("experiment adaptation_mode does not match capability_binding.adaptation")
    missing_environment = [
        requirement["name"]
        for requirement in spec.get("required_environment", [])
        if not os.environ.get(requirement["name"], "").strip()
    ]
    if missing_environment:
        raise ValueError("missing required backend environment: " + ", ".join(sorted(missing_environment)))

    if work_dir.is_symlink():
        raise ValueError(f"lifecycle work directory must not be a symlink: {work_dir}")
    work_dir = work_dir.resolve()
    if work_dir.exists() and not work_dir.is_dir():
        raise ValueError(f"lifecycle work directory is not a directory: {work_dir}")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError(f"lifecycle work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    lifecycle_started = _now()
    stage_reports: list[dict[str, Any]] = []
    lifecycle_artifacts: list[dict[str, Any]] = []
    lifecycle_status = "passed"

    for stage in STAGES:
        stage_dir = work_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_result_path = stage_dir / "stage-result.json"
        stdout_path = stage_dir / "stdout.log"
        stderr_path = stage_dir / "stderr.log"
        command = _expand_command(spec["commands"][stage], work_dir=work_dir, stage_result=stage_result_path)
        environment = os.environ.copy()
        environment.update(spec.get("environment", {}))
        package_root = str(Path(__file__).parents[1])
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        environment.update(
            {
                "INSTAVAR_VOICE_EXPERIMENT_MANIFEST": str(experiment_path.resolve()),
                "INSTAVAR_VOICE_WORK_DIR": str(work_dir),
                "INSTAVAR_VOICE_STAGE": stage,
                "INSTAVAR_VOICE_STAGE_RESULT": str(stage_result_path),
            }
        )
        started = _now()
        timeout_seconds = spec.get("timeout_seconds", {}).get(stage, 3600)
        timed_out = False
        completed: subprocess.CompletedProcess[str] | None = None
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=spec_path.parent,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
        report: dict[str, Any] = {
            "stage": stage,
            "started_at": started,
            "finished_at": _now(),
            "command": command,
            "exit_code": None if completed is None else completed.returncode,
            "timeout_seconds": timeout_seconds,
            "stdout": str(stdout_path.relative_to(work_dir)),
            "stderr": str(stderr_path.relative_to(work_dir)),
            "status": "failed",
            "artifacts": [],
        }
        if timed_out:
            report["error"] = "backend command exceeded its stage timeout"
        elif completed is None:
            report["error"] = "backend command did not return a process result"
        elif completed.returncode != 0:
            report["error"] = "backend command returned a non-zero exit code"
        elif not stage_result_path.is_file():
            report["error"] = "backend command did not write stage-result.json"
        elif stage_result_path.is_symlink():
            report["error"] = "backend stage result must not be a symlink"
        else:
            try:
                stage_result = _read_json(stage_result_path)
                if (
                    not isinstance(stage_result, dict)
                    or stage_result.get("schema_version") != "1.0.0"
                    or stage_result.get("stage") != stage
                ):
                    raise ValueError("stage result must name the active stage")
                if stage_result.get("status") != "passed":
                    raise ValueError("stage result status must equal passed")
                _verify_artifact_records(lifecycle_artifacts, root=work_dir)
                artifacts = [
                    _artifact_record(work_dir / relative_path, root=work_dir)
                    for relative_path in spec.get("expected_artifacts", {}).get(stage, [])
                ]
                artifacts.append(_artifact_record(stage_result_path, root=work_dir))
                lifecycle_artifacts.extend(artifacts)
                report["status"] = "passed"
                report["artifacts"] = artifacts
                report["backend_result"] = stage_result
            except (OSError, json.JSONDecodeError, ValueError) as error:
                report["error"] = str(error)
        stage_reports.append(report)
        if report["status"] != "passed":
            lifecycle_status = "failed"
            break

    result = {
        "schema_version": "1.0.0",
        "backend_id": spec["backend_id"],
        "capability_binding": binding,
        "required_environment": [
            requirement["name"] for requirement in spec.get("required_environment", [])
        ],
        "experiment_id": experiment_id,
        "status": lifecycle_status,
        "started_at": lifecycle_started,
        "finished_at": _now(),
        "stages": stage_reports,
        "evidence_boundary": (
            "A passed lifecycle proves command invocation, declared artifact creation, and hashing only. "
            "Model quality and runtime equivalence require separate evidence."
        ),
    }
    _write_json(work_dir / "lifecycle-report.json", result)
    return result
