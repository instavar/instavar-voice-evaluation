from __future__ import annotations

import json
import unittest
from pathlib import Path

from instavar_voice_lab.suite import build_generation_plan, check_suite_coverage, validate_prompt_pack


ROOT = Path(__file__).parents[1]


class SuiteTests(unittest.TestCase):
    def prompt_pack(self):
        return json.loads((ROOT / "reference" / "singapore-english-v1.json").read_text(encoding="utf-8"))

    def test_prompt_pack_has_frozen_unique_seeds(self) -> None:
        self.assertEqual(validate_prompt_pack(self.prompt_pack()), [])

    def test_generation_plan_is_complete_and_deterministic(self) -> None:
        first = build_generation_plan(self.prompt_pack(), ["base", "adapter"])
        second = build_generation_plan(self.prompt_pack(), ["base", "adapter"])
        self.assertEqual(first, second)
        self.assertEqual(first["prompt_count"], 7)
        self.assertEqual(first["sample_count"], 42)
        self.assertEqual(len({row["sample_id"] for row in first["samples"]}), 42)

    def test_coverage_preserves_invalid_outputs_as_evidence(self) -> None:
        plan = build_generation_plan(self.prompt_pack(), ["adapter"], seeds=[42])
        observations = [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "valid": row["prompt_id"] != "local-context",
            }
            for row in plan["samples"]
        ]
        result = check_suite_coverage(plan, observations)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["coverage_complete"])
        self.assertFalse(result["generation_complete_without_invalid_outputs"])
        self.assertEqual(len(result["invalid_sample_ids"]), 1)

    def test_coverage_fails_on_missing_or_duplicate_observations(self) -> None:
        plan = build_generation_plan(self.prompt_pack(), ["adapter"], seeds=[42])
        observations = [
            {"sample_id": row["sample_id"], "valid": True}
            for row in plan["samples"][:-1]
        ]
        observations.append(dict(observations[0]))
        result = check_suite_coverage(plan, observations)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["missing_sample_ids"]), 1)
        self.assertEqual(len(result["duplicate_sample_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
