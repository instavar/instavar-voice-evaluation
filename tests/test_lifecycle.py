from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from instavar_voice_lab.lifecycle import run_lifecycle, validate_backend_spec


ROOT = Path(__file__).parents[1]


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

    def test_stage_timeout_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = json.loads((ROOT / "examples" / "fake-backend.json").read_text())
            spec["commands"]["preflight"] = [
                sys.executable,
                "-c",
                "import time; time.sleep(0.2)",
            ]
            spec["timeout_seconds"]["preflight"] = 0.01
            spec_path = Path(temporary) / "timeout-backend.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            result = run_lifecycle(
                spec_path,
                ROOT / "examples" / "experiment-manifest.json",
                Path(temporary) / "work",
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn("timeout", result["stages"][0]["error"])


if __name__ == "__main__":
    unittest.main()
