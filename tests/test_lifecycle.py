from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lifecycle import (
    _terminate_stage_processes,
    resolve_backend_spec,
    run_lifecycle,
    run_registered_lifecycle,
    validate_backend_registry,
    validate_backend_spec,
)

ROOT = Path(__file__).parents[1]


def write_backend_fixture(directory: Path, filename: str, spec: dict[str, object]) -> Path:
    spec_path = directory / filename
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    (directory / "capability-manifest.json").write_text(
        (ROOT / "examples" / "capability-manifest.json").read_text(), encoding="utf-8"
    )
    return spec_path


def write_registry_fixture(directory: Path, entries: list[dict[str, str]]) -> Path:
    registry_path = directory / "backend-registry.json"
    registry_path.write_text(
        json.dumps({"schema_version": "1.0.0", "backends": entries}),
        encoding="utf-8",
    )
    return registry_path


class LifecycleTests(unittest.TestCase):
    def test_non_posix_timeout_fallback_does_not_claim_process_tree_cleanup(self) -> None:
        class ResistantProcess:
            terminated = False
            killed = False
            wait_count = 0

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: float | None = None) -> int:
                self.wait_count += 1
                if self.wait_count == 1:
                    raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)
                return -9

            def kill(self) -> None:
                self.killed = True

        process = ResistantProcess()
        with patch("instavar_voice_lab.lifecycle.os.name", "nt"):
            result = _terminate_stage_processes(process, grace_seconds=0.01)  # type: ignore[arg-type]
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(result["mode"], "direct_process_only")
        self.assertTrue(result["kill_signal_sent"])
        self.assertFalse(result["process_tree_termination_verified"])

    def test_fake_backend_runs_every_stage_and_hashes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_lifecycle(
                ROOT / "examples" / "fake-backend.json",
                ROOT / "examples" / "experiment-manifest.json",
                Path(temporary),
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual([stage["stage"] for stage in result["stages"]], ["preflight", "train", "infer", "evaluate", "package"])
            self.assertTrue(all(stage["status"] == "passed" for stage in result["stages"]))
            self.assertTrue((Path(temporary) / "lifecycle-report.json").is_file())
            report = json.loads((Path(temporary) / "lifecycle-report.json").read_text())
            self.assertIn("Model quality", report["evidence_boundary"])
            self.assertEqual(report["capability_binding"]["adaptation"], "lora")
            self.assertEqual(report["required_environment"], [])
            self.assertEqual(
                {record["role"] for record in report["control_inputs"]},
                {"backend_spec", "capability_manifest", "experiment_manifest"},
            )
            self.assertTrue(all(len(record["sha256"]) == 64 for record in report["control_inputs"]))

    def test_rejects_nonempty_work_directory_instead_of_reusing_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary) / "work"
            work_dir.mkdir()
            (work_dir / "stale-stage-result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                run_lifecycle(
                    ROOT / "examples" / "fake-backend.json",
                    ROOT / "examples" / "experiment-manifest.json",
                    work_dir,
                )

    def test_rejects_symlinked_work_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            target.mkdir()
            work_dir = Path(temporary) / "work-link"
            work_dir.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                run_lifecycle(
                    ROOT / "examples" / "fake-backend.json",
                    ROOT / "examples" / "experiment-manifest.json",
                    work_dir,
                )

    def test_rejects_symlinked_backend_and_experiment_control_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary)
            linked_spec = fixture_dir / "backend.json"
            linked_spec.symlink_to(ROOT / "examples" / "fake-backend.json")
            with self.assertRaisesRegex(ValueError, "control input must not be a symlink"):
                run_lifecycle(
                    linked_spec,
                    ROOT / "examples" / "experiment-manifest.json",
                    fixture_dir / "spec-work",
                )
            self.assertFalse((fixture_dir / "spec-work").exists())

            linked_experiment = fixture_dir / "experiment.json"
            linked_experiment.symlink_to(ROOT / "examples" / "experiment-manifest.json")
            with self.assertRaisesRegex(ValueError, "control input must not be a symlink"):
                run_lifecycle(
                    ROOT / "examples" / "fake-backend.json",
                    linked_experiment,
                    fixture_dir / "experiment-work",
                )
            self.assertFalse((fixture_dir / "experiment-work").exists())

    def test_rejects_semantically_invalid_experiment_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = json.loads((ROOT / "examples" / "experiment-manifest.json").read_text())
            experiment["rights"]["consent"] = ""
            experiment_path = Path(temporary) / "invalid-experiment.json"
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid experiment manifest"):
                run_lifecycle(
                    ROOT / "examples" / "fake-backend.json",
                    experiment_path,
                    Path(temporary) / "work",
                )

    def test_backend_spec_rejects_artifact_path_escape(self) -> None:
        spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
        spec["expected_artifacts"]["package"] = ["../../escaped.bin"]
        errors = validate_backend_spec(spec)
        self.assertTrue(any("unsafe path" in error for error in errors))

    def test_backend_spec_rejects_cross_stage_and_runner_owned_artifacts(self) -> None:
        spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
        spec["expected_artifacts"]["train"] = ["infer/candidate.wav", "train/stdout.log"]
        errors = validate_backend_spec(spec)
        self.assertTrue(any("owned by the train stage" in error for error in errors))
        self.assertTrue(any("runner-owned path" in error for error in errors))

    def test_backend_spec_binds_declared_adaptation_and_runtime(self) -> None:
        spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
        self.assertEqual(validate_backend_spec(spec, spec_path=ROOT / "examples" / "fake-backend.json"), [])
        spec["capability_binding"]["runtime_ids"] = ["missing-runtime"]
        errors = validate_backend_spec(spec, spec_path=ROOT / "examples" / "fake-backend.json")
        self.assertTrue(any("unknown runtime" in error for error in errors))

    def test_backend_spec_rejects_runner_owned_or_predefined_required_environment(self) -> None:
        spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
        spec["required_environment"] = [
            {"name": "MODEL_PATH", "purpose": "Model path supplied by the operator."},
            {"name": "INSTAVAR_VOICE_WORK_DIR", "purpose": "Invalid attempt to own runner state."},
        ]
        spec["environment"] = {"MODEL_PATH": "/invalid/constant", "INSTAVAR_VOICE_STAGE": "train"}
        errors = validate_backend_spec(spec, spec_path=ROOT / "examples" / "fake-backend.json")
        self.assertTrue(any("both declare MODEL_PATH" in error for error in errors))
        self.assertTrue(any("runner-owned" in error for error in errors))

    def test_missing_required_environment_fails_before_work_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["required_environment"] = [
                {"name": "INSTAVAR_TEST_MODEL_PATH", "purpose": "Path to a test-only model fixture."}
            ]
            fixture_dir = Path(temporary)
            spec["capability_binding"]["manifest"] = "capability-manifest.json"
            spec_path = write_backend_fixture(fixture_dir, "required-environment-backend.json", spec)
            work_dir = Path(temporary) / "work"
            with self.assertRaisesRegex(ValueError, "INSTAVAR_TEST_MODEL_PATH"):
                run_lifecycle(spec_path, ROOT / "examples" / "experiment-manifest.json", work_dir)
            self.assertFalse(work_dir.exists())

    def test_experiment_adaptation_must_match_backend_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = json.loads((ROOT / "examples" / "experiment-manifest.json").read_text())
            experiment["adaptation_mode"] = "full_sft"
            experiment_path = Path(temporary) / "experiment.json"
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_lifecycle(
                    ROOT / "examples" / "fake-backend.json",
                    experiment_path,
                    Path(temporary) / "work",
                )

    def test_empty_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["environment"] = {"INSTAVAR_FAKE_EMPTY_PREFLIGHT": "1"}
            spec_path = write_backend_fixture(Path(temporary), "empty-artifact-backend.json", spec)
            result = run_lifecycle(
                spec_path,
                ROOT / "examples" / "experiment-manifest.json",
                Path(temporary) / "work",
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn("must not be empty", result["stages"][0]["error"])

    def test_later_stage_cannot_mutate_hashed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["environment"] = {"INSTAVAR_FAKE_MUTATE_CHECKPOINT": "1"}
            spec_path = write_backend_fixture(Path(temporary), "mutating-backend.json", spec)
            result = run_lifecycle(
                spec_path,
                ROOT / "examples" / "experiment-manifest.json",
                Path(temporary) / "work",
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stages"][-1]["stage"], "infer")
            self.assertIn("changed after hashing", result["stages"][-1]["error"])

    def test_stage_timeout_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["commands"]["preflight"] = [
                sys.executable,
                "-c",
                "import time; time.sleep(0.2)",
            ]
            spec["timeout_seconds"]["preflight"] = 0.01
            spec_path = write_backend_fixture(Path(temporary), "timeout-backend.json", spec)
            result = run_lifecycle(
                spec_path,
                ROOT / "examples" / "experiment-manifest.json",
                Path(temporary) / "work",
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn("timeout", result["stages"][0]["error"])

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_stage_timeout_terminates_descendant_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "child-ready"
            terminated = root / "child-terminated"
            child_code = """
import signal
import sys
import time
from pathlib import Path

ready = Path(sys.argv[1])
terminated = Path(sys.argv[2])

def stop(*_args):
    terminated.write_text("terminated", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
ready.write_text("ready", encoding="utf-8")
while True:
    time.sleep(1)
"""
            parent_code = """
import subprocess
import sys
import time
from pathlib import Path

subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]])
ready = Path(sys.argv[2])
deadline = time.monotonic() + 10
while not ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
time.sleep(60)
"""
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["commands"]["preflight"] = [
                sys.executable,
                "-c",
                parent_code,
                child_code,
                str(ready),
                str(terminated),
            ]
            spec["timeout_seconds"]["preflight"] = 1
            spec_path = write_backend_fixture(root, "descendant-timeout-backend.json", spec)
            with patch("instavar_voice_lab.lifecycle.PROCESS_TERMINATION_GRACE_SECONDS", 0.5):
                result = run_lifecycle(spec_path, ROOT / "examples" / "experiment-manifest.json", root / "work")
            stage = result["stages"][0]
            self.assertEqual(stage["error"], "backend command exceeded its stage timeout")
            self.assertEqual(stage["process_termination"]["mode"], "posix_process_group")
            self.assertTrue(stage["process_termination"]["term_signal_sent"])
            self.assertTrue(terminated.is_file())

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_successful_parent_cannot_leave_background_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "child-ready"
            terminated = root / "child-terminated"
            child_code = """
import signal
import sys
import time
from pathlib import Path

ready = Path(sys.argv[1])
terminated = Path(sys.argv[2])

def stop(*_args):
    terminated.write_text("terminated", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
ready.write_text("ready", encoding="utf-8")
while True:
    time.sleep(1)
"""
            parent_code = """
import subprocess
import sys
import time
from pathlib import Path

subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]])
ready = Path(sys.argv[2])
deadline = time.monotonic() + 10
while not ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["commands"]["preflight"] = [
                sys.executable,
                "-c",
                parent_code,
                child_code,
                str(ready),
                str(terminated),
            ]
            spec_path = write_backend_fixture(root, "descendant-exit-backend.json", spec)
            with patch("instavar_voice_lab.lifecycle.PROCESS_TERMINATION_GRACE_SECONDS", 0.5):
                result = run_lifecycle(spec_path, ROOT / "examples" / "experiment-manifest.json", root / "work")
            stage = result["stages"][0]
            self.assertEqual(stage["error"], "backend command left descendant processes running")
            self.assertTrue(stage["process_termination"]["term_signal_sent"])
            self.assertTrue(terminated.is_file())

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_stage_timeout_escalates_when_descendant_ignores_sigterm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "child-ready"
            child_code = """
import signal
import sys
import time
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text("ready", encoding="utf-8")
while True:
    time.sleep(1)
"""
            parent_code = """
import subprocess
import sys
import time
from pathlib import Path

subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])
ready = Path(sys.argv[2])
deadline = time.monotonic() + 10
while not ready.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
time.sleep(60)
"""
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["commands"]["preflight"] = [
                sys.executable,
                "-c",
                parent_code,
                child_code,
                str(ready),
            ]
            spec["timeout_seconds"]["preflight"] = 1
            spec_path = write_backend_fixture(root, "sigterm-resistant-backend.json", spec)
            with patch("instavar_voice_lab.lifecycle.PROCESS_TERMINATION_GRACE_SECONDS", 0.1):
                result = run_lifecycle(spec_path, ROOT / "examples" / "experiment-manifest.json", root / "work")
            stage = result["stages"][0]
            self.assertEqual(stage["error"], "backend command exceeded its stage timeout")
            self.assertTrue(stage["process_termination"]["kill_signal_sent"])

    def test_backend_registry_validates_and_runs_unique_adaptation(self) -> None:
        self.assertEqual(
            validate_backend_registry(
                json.loads((ROOT / "examples" / "backend-registry.json").read_text()),
                registry_path=ROOT / "examples" / "backend-registry.json",
            ),
            [],
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_registered_lifecycle(
                ROOT / "examples" / "backend-registry.json",
                ROOT / "examples" / "experiment-manifest.json",
                Path(temporary),
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["backend_registry"]["selected_backend_id"], "instavar-fake-backend")
            self.assertEqual(result["backend_registry"]["selected_spec"], "fake-backend.json")
            self.assertEqual(len(result["backend_registry"]["selected_spec_sha256"]), 64)

    def test_backend_registry_rejects_duplicates_unsafe_paths_and_id_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary)
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            write_backend_fixture(fixture_dir, "fake-backend.json", spec)
            registry_path = write_registry_fixture(
                fixture_dir,
                [
                    {"backend_id": "wrong-id", "spec": "fake-backend.json"},
                    {"backend_id": "wrong-id", "spec": "fake-backend.json"},
                    {"backend_id": "escaped", "spec": "../escaped.json"},
                ],
            )
            errors = validate_backend_registry(json.loads(registry_path.read_text()), registry_path=registry_path)
            self.assertTrue(any("does not match" in error for error in errors))
            self.assertTrue(any("backend_ids must not contain duplicates" in error for error in errors))
            self.assertTrue(any("spec paths must not contain duplicates" in error for error in errors))
            self.assertTrue(any("safe relative path" in error for error in errors))

    def test_backend_registry_rejects_symlinked_registry_and_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary)
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            real_spec = write_backend_fixture(fixture_dir, "real-backend.json", spec)
            linked_spec = fixture_dir / "linked-backend.json"
            linked_spec.symlink_to(real_spec)
            registry_path = write_registry_fixture(
                fixture_dir,
                [{"backend_id": spec["backend_id"], "spec": linked_spec.name}],
            )
            errors = validate_backend_registry(json.loads(registry_path.read_text()), registry_path=registry_path)
            self.assertTrue(any("spec path must not contain symlinks" in error for error in errors))

            nested_target = fixture_dir / "nested-target"
            nested_target.mkdir()
            write_backend_fixture(nested_target, "nested.json", spec)
            linked_directory = fixture_dir / "linked-directory"
            linked_directory.symlink_to(nested_target, target_is_directory=True)
            nested_registry = write_registry_fixture(
                fixture_dir,
                [{"backend_id": spec["backend_id"], "spec": "linked-directory/nested.json"}],
            )
            errors = validate_backend_registry(json.loads(nested_registry.read_text()), registry_path=nested_registry)
            self.assertTrue(any("spec path must not contain symlinks" in error for error in errors))

            linked_registry = fixture_dir / "linked-registry.json"
            linked_registry.symlink_to(registry_path)
            errors = validate_backend_registry(json.loads(linked_registry.read_text()), registry_path=linked_registry)
            self.assertTrue(any("registry must not be a symlink" in error for error in errors))

    def test_backend_registry_requires_explicit_selection_when_adaptation_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary)
            first = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            second = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            first["backend_id"] = "fake-lora-a"
            second["backend_id"] = "fake-lora-b"
            write_backend_fixture(fixture_dir, "a.json", first)
            write_backend_fixture(fixture_dir, "b.json", second)
            registry_path = write_registry_fixture(
                fixture_dir,
                [
                    {"backend_id": "fake-lora-a", "spec": "a.json"},
                    {"backend_id": "fake-lora-b", "spec": "b.json"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "multiple recipes"):
                resolve_backend_spec(registry_path, ROOT / "examples" / "experiment-manifest.json")
            selected = resolve_backend_spec(
                registry_path,
                ROOT / "examples" / "experiment-manifest.json",
                backend_id="fake-lora-b",
            )
            self.assertEqual(selected.name, "b.json")

    def test_explicit_backend_selection_cannot_override_experiment_adaptation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = json.loads((ROOT / "examples" / "experiment-manifest.json").read_text())
            experiment["adaptation_mode"] = "full_sft"
            experiment_path = Path(temporary) / "experiment.json"
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
            work_dir = Path(temporary) / "work"
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_registered_lifecycle(
                    ROOT / "examples" / "backend-registry.json",
                    experiment_path,
                    work_dir,
                    backend_id="instavar-fake-backend",
                )
            self.assertFalse(work_dir.exists())

    def test_registered_lifecycle_marks_report_failed_when_registry_mutates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_dir = Path(temporary)
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            write_backend_fixture(fixture_dir, "fake-backend.json", spec)
            registry_path = write_registry_fixture(
                fixture_dir,
                [{"backend_id": spec["backend_id"], "spec": "fake-backend.json"}],
            )
            work_dir = fixture_dir / "work"

            def mutate_registry(_spec_path: Path, _experiment_path: Path, target_work_dir: Path) -> dict[str, object]:
                target_work_dir.mkdir()
                registry_path.write_text(registry_path.read_text() + "\n", encoding="utf-8")
                return {"backend_id": spec["backend_id"], "status": "passed"}

            with (
                patch("instavar_voice_lab.lifecycle.run_lifecycle", side_effect=mutate_registry),
                self.assertRaisesRegex(ValueError, "changed during"),
            ):
                run_registered_lifecycle(
                    registry_path,
                    ROOT / "examples" / "experiment-manifest.json",
                    work_dir,
                )
            report = json.loads((work_dir / "lifecycle-report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("changed during", report["error"])

    def test_lifecycle_fails_when_a_control_input_mutates_during_a_stage(self) -> None:
        for role in ("backend_spec", "experiment_manifest", "capability_manifest"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                fixture_dir = Path(temporary)
                spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
                spec_path = write_backend_fixture(fixture_dir, "backend.json", spec)
                experiment_path = fixture_dir / "experiment.json"
                experiment_path.write_text(
                    (ROOT / "examples" / "experiment-manifest.json").read_text(),
                    encoding="utf-8",
                )
                targets = {
                    "backend_spec": spec_path,
                    "experiment_manifest": experiment_path,
                    "capability_manifest": fixture_dir / "capability-manifest.json",
                }

                class MutatingProcess:
                    pid = 2_147_483_647
                    returncode = 0

                    def wait(self, *_args: object, target: Path = targets[role], **_kwargs: object) -> int:
                        target.write_text(target.read_text() + "\n", encoding="utf-8")
                        return 0

                    def poll(self) -> int:
                        return 0

                work_dir = fixture_dir / "work"
                with patch("instavar_voice_lab.lifecycle.subprocess.Popen", return_value=MutatingProcess()):
                    result = run_lifecycle(spec_path, experiment_path, work_dir)
                self.assertEqual(result["status"], "failed")
                self.assertIn(f"after locking: {role}", result["stages"][0]["error"])
                report = json.loads((work_dir / "lifecycle-report.json").read_text())
                self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
