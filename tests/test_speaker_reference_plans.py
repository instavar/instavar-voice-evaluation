from __future__ import annotations

import unittest
from copy import deepcopy

from instavar_voice_lab.speaker_reference_plans import (
    build_speaker_reference_assignment_plan,
    speaker_reference_assignment_sha256,
    validate_speaker_reference_assignment_plan,
)
from instavar_voice_lab.speaker_references import canonical_sha256


def generation_plan() -> dict:
    rows = [
        {
            "sample_id": f"{candidate}--{prompt}--seed-{seed}",
            "candidate_id": candidate,
            "prompt_id": prompt,
            "seed": seed,
            "text": "hello",
        }
        for candidate in ("base", "adapter")
        for prompt in ("p1", "p2")
        for seed in (7, 42)
    ]
    return {
        "schema_version": "1.1.0",
        "prompt_pack": {"id": "test", "version": "1.0.0", "sha256": "a" * 64},
        "candidate_ids": ["base", "adapter"],
        "sample_count": len(rows),
        "required_objective_metrics": ["speaker_embedding_similarity"],
        "samples": rows,
        "generation_requirements": {
            "same_transcripts": True,
            "frozen_generation_settings": True,
            "record_failures_as_observations": True,
        },
    }


def reference_catalog() -> dict:
    payload = {
        "catalog_id": "voice-1-catalog",
        "references": [
            {
                "reference_id": reference_id,
                "audio": {"sha256": digit * 64, "bytes": 100},
                "transcript": {"sha256": "d" * 64, "bytes": 20},
            }
            for reference_id, digit in (("phone", "b"), ("studio", "c"))
        ],
    }
    return {**payload, "catalog_sha256": canonical_sha256(payload)}


class SpeakerReferenceAssignmentPlanTests(unittest.TestCase):
    def build(self) -> tuple[dict, dict, dict]:
        generation = generation_plan()
        catalog = reference_catalog()
        assignments = {
            (prompt, seed): (["phone", "studio"] if prompt == "p1" else ["studio"])
            for prompt in ("p1", "p2")
            for seed in (7, 42)
        }
        plan = build_speaker_reference_assignment_plan(
            plan_id="voice-1-eval",
            generation_plan=generation,
            reference_catalog=catalog,
            assignments=assignments,
            policy_id="stratified-v1",
            stratification_dimensions=["accent", "channel"],
            rationale="Freeze channel and accent coverage before candidate generation or scoring.",
        )
        return generation, catalog, plan

    def test_builds_and_validates_complete_frozen_assignments(self) -> None:
        generation, catalog, plan = self.build()
        binding = validate_speaker_reference_assignment_plan(
            plan,
            generation_plan=generation,
            reference_catalog=catalog,
        )
        self.assertEqual(binding["assignment_count"], 4)
        self.assertEqual(binding["assignments"][("p1", 7)], ["phone", "studio"])
        self.assertRegex(binding["sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_missing_pair_unknown_reference_and_post_freeze_mutation(self) -> None:
        generation, catalog, plan = self.build()
        missing = deepcopy(plan)
        missing["assignments"].pop()
        payload = {key: value for key, value in missing.items() if key != "assignment_plan_sha256"}
        missing["assignment_plan_sha256"] = canonical_sha256(payload)
        with self.assertRaisesRegex(ValueError, "exactly cover generation prompt and seed pairs"):
            validate_speaker_reference_assignment_plan(missing, generation_plan=generation, reference_catalog=catalog)

        unknown = deepcopy(plan)
        unknown["assignments"][0]["reference_ids"] = ["unknown"]
        payload = {key: value for key, value in unknown.items() if key != "assignment_plan_sha256"}
        unknown["assignment_plan_sha256"] = canonical_sha256(payload)
        with self.assertRaisesRegex(ValueError, "absent from the catalog"):
            validate_speaker_reference_assignment_plan(unknown, generation_plan=generation, reference_catalog=catalog)

        mutated = deepcopy(plan)
        mutated["selection_policy"]["rationale"] = "Selected after viewing results."
        with self.assertRaisesRegex(ValueError, "does not match the plan contents"):
            validate_speaker_reference_assignment_plan(mutated, generation_plan=generation, reference_catalog=catalog)

    def test_assignment_digest_binds_pair_and_reference_membership(self) -> None:
        _, _, plan = self.build()
        first = speaker_reference_assignment_sha256(
            assignment_plan_sha256=plan["assignment_plan_sha256"],
            prompt_id="p1",
            seed=7,
            reference_ids=["phone", "studio"],
        )
        changed_pair = speaker_reference_assignment_sha256(
            assignment_plan_sha256=plan["assignment_plan_sha256"],
            prompt_id="p1",
            seed=42,
            reference_ids=["phone", "studio"],
        )
        changed_set = speaker_reference_assignment_sha256(
            assignment_plan_sha256=plan["assignment_plan_sha256"],
            prompt_id="p1",
            seed=7,
            reference_ids=["studio"],
        )
        self.assertEqual(len({first, changed_pair, changed_set}), 3)


if __name__ == "__main__":
    unittest.main()
