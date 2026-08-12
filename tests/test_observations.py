from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.observations import validate_objective_observations


class ObservationContractTests(unittest.TestCase):
    def row(self) -> dict:
        return {
            "observation_schema_version": "1.0.0",
            "sample_id": "adapter--p1--seed-42",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "seed": 42,
            "requested_text": "hello world",
            "valid": True,
            "runtime_id": "pytorch",
        }

    def test_accepts_strict_versioned_runtime_observation(self) -> None:
        self.assertEqual(
            validate_objective_observations(
                [self.row()],
                require_version=True,
                require_seed=True,
                require_runtime=True,
            ),
            [],
        )

    def test_rejects_duplicate_and_unstable_identifiers(self) -> None:
        first = self.row()
        second = self.row()
        second["candidate_id"] = "../adapter"
        errors = validate_objective_observations([first, second])
        self.assertTrue(any("duplicates" in error for error in errors))
        self.assertTrue(any("candidate_id" in error for error in errors))

    def test_rejects_partial_artifact_binding_and_mutable_version(self) -> None:
        row = self.row()
        row["observation_schema_version"] = "latest"
        row["artifact_set_id"] = "voice-1"
        errors = validate_objective_observations([row])
        self.assertTrue(any("must equal 1.0.0" in error for error in errors))
        self.assertTrue(any("must be supplied together" in error for error in errors))

    def test_cli_validates_strict_producer_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            observations = Path(temporary) / "observations.json"
            observations.write_text(
                '[{"observation_schema_version":"1.0.0","sample_id":"adapter--p1--seed-42",'
                '"candidate_id":"adapter","prompt_id":"p1","seed":42,"requested_text":"hello",'
                '"valid":true,"runtime_id":"pytorch"}]\n',
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "validate-observations",
                        str(observations),
                        "--require-version",
                        "--require-seed",
                        "--require-runtime",
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
