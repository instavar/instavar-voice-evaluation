from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from instavar_voice_lab.lifecycle import run_lifecycle


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


if __name__ == "__main__":
    unittest.main()
