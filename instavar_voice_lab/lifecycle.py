from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = ("preflight", "train", "infer", "evaluate", "package")
SENSITIVE_ENV_FRAGMENTS = ("secret", "token", "password", "api_key", "credential")


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
    if not path.is_file():
        raise FileNotFoundError(f"expected lifecycle artifact not found: {path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def validate_backend_spec(spec: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["backend spec must be an object"]
    if spec.get("schema_version") != "1.0.0":
        errors.append("schema_version must equal 1.0.0")
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
        for stage, paths in expected.items():
            if stage not in STAGES or not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
                errors.append(f"expected_artifacts.{stage} must be an array of paths for a known stage")
    environment = spec.get("environment", {})
    if not isinstance(environment, dict):
        errors.append("environment must be an object when present")
    else:
        for name, value in environment.items():
            if any(fragment in name.casefold() for fragment in SENSITIVE_ENV_FRAGMENTS):
                errors.append(f"environment.{name} looks sensitive and must be supplied outside the spec")
            if not isinstance(value, str):
                errors.append(f"environment.{name} must be a string")
    return errors


def run_lifecycle(spec_path: Path, experiment_path: Path, work_dir: Path) -> dict[str, Any]:
    spec = _read_json(spec_path)
    errors = validate_backend_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    experiment = _read_json(experiment_path)
    experiment_id = str(experiment.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("experiment manifest must contain experiment_id")

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    lifecycle_started = _now()
    stage_reports: list[dict[str, Any]] = []
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
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=spec_path.parent, env=environment, stdout=stdout, stderr=stderr, check=False)
        report: dict[str, Any] = {
            "stage": stage,
            "started_at": started,
            "finished_at": _now(),
            "command": command,
            "exit_code": completed.returncode,
            "stdout": str(stdout_path.relative_to(work_dir)),
            "stderr": str(stderr_path.relative_to(work_dir)),
            "status": "failed",
            "artifacts": [],
        }
        if completed.returncode != 0:
            report["error"] = "backend command returned a non-zero exit code"
        elif not stage_result_path.is_file():
            report["error"] = "backend command did not write stage-result.json"
        else:
            try:
                stage_result = _read_json(stage_result_path)
                if not isinstance(stage_result, dict) or stage_result.get("stage") != stage:
                    raise ValueError("stage result must name the active stage")
                if stage_result.get("status") != "passed":
                    raise ValueError("stage result status must equal passed")
                artifacts = [
                    _artifact_record(work_dir / relative_path, root=work_dir)
                    for relative_path in spec.get("expected_artifacts", {}).get(stage, [])
                ]
                artifacts.append(_artifact_record(stage_result_path, root=work_dir))
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
