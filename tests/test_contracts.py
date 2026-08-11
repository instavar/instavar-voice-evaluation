from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from instavar_voice_lab.contracts import validate_document


ROOT = Path(__file__).parents[1]


def load_example(name: str):
    return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))


class ContractTests(unittest.TestCase):
    def test_examples_are_valid(self) -> None:
        examples = {
            "capability": "capability-manifest.json",
            "experiment": "experiment-manifest.json",
            "evaluation": "evaluation-report.json",
            "package": "artifact-package.json",
        }
        for kind, filename in examples.items():
            with self.subTest(kind=kind):
                self.assertEqual(validate_document(kind, load_example(filename)), [])

    def test_supported_capability_requires_evidence(self) -> None:
        manifest = load_example("capability-manifest.json")
        manifest["adaptation"]["lora"]["evidence"] = []
        errors = validate_document("capability", manifest)
        self.assertTrue(any(error.path == "$.adaptation.lora.evidence" for error in errors))

    def test_duplicate_runtime_identifier_is_rejected(self) -> None:
        manifest = load_example("capability-manifest.json")
        manifest["runtimes"].append(deepcopy(manifest["runtimes"][0]))
        errors = validate_document("capability", manifest)
        self.assertTrue(any("unique" in error.message for error in errors))

    def test_evaluation_rejects_composite_score(self) -> None:
        report = load_example("evaluation-report.json")
        report["composite_score"] = 0.91
        errors = validate_document("evaluation", report)
        self.assertTrue(any(error.path == "$.composite_score" for error in errors))

    def test_experiment_rejects_split_hash_reuse(self) -> None:
        manifest = load_example("experiment-manifest.json")
        manifest["corpus"]["split_hashes"]["test"] = manifest["corpus"]["split_hashes"]["train"]
        errors = validate_document("experiment", manifest)
        self.assertTrue(any(error.path == "$.corpus.split_hashes" for error in errors))


if __name__ == "__main__":
    unittest.main()
