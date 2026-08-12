from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from instavar_voice_lab.lifecycle import run_lifecycle, validate_backend_spec


ROOT = Path(__file__).parents[1]


def write_backend_fixture(directory: Path, filename: str, spec: dict[str, object]) -> Path:
    spec_path = directory / filename
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    (directory / "capability-manifest.json").write_text(
        (ROOT / "examples" / "capability-manifest.json").read_text(), encoding="utf-8"
    )
    return spec_path


class LifecycleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
