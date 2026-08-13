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
        self.assertEqual(first["schema_version"], "1.1.0")
        self.assertEqual(first["selected_prompt_ids"], [prompt["id"] for prompt in self.prompt_pack()["prompts"]])
        self.assertEqual(first["required_objective_metrics"], self.prompt_pack()["objective_metrics"])
        self.assertEqual(len({row["sample_id"] for row in first["samples"]}), 42)
        local_rows = [row for row in first["samples"] if row["prompt_id"] == "local-context"]
        self.assertEqual(len(local_rows), 6)
        self.assertEqual(local_rows[0]["lexical_anchors"][0]["anchor_id"], "paiseh")
        self.assertEqual(
            local_rows[0]["lexical_anchors"][0]["accepted_asr_forms"],
            ["paiseh", "pai seh", "pie say"],
        )

    def test_generation_plan_can_preregister_a_focused_prompt_slice(self) -> None:
        prompt_pack = self.prompt_pack()
        full_hash = build_generation_plan(prompt_pack, ["base"], seeds=[42])["prompt_pack"]["sha256"]
        plan = build_generation_plan(
            prompt_pack,
            ["base", "adapter"],
            seeds=[20260812],
            prompt_ids=["cadence-two-minute"],
        )
        self.assertEqual(plan["prompt_pack"]["sha256"], full_hash)
        self.assertEqual(plan["selected_prompt_ids"], ["cadence-two-minute"])
        self.assertEqual(plan["prompt_count"], 1)
        self.assertEqual(plan["sample_count"], 2)
        self.assertEqual({row["prompt_id"] for row in plan["samples"]}, {"cadence-two-minute"})

    def test_generation_plan_rejects_empty_duplicate_or_unknown_prompt_selection(self) -> None:
        for prompt_ids, message in (
            ([], "at least one prompt id is required"),
            (["neutral-brief", "neutral-brief"], "prompt ids must be unique"),
            (["not-in-pack"], "unknown prompt ids: not-in-pack"),
        ):
            with self.subTest(prompt_ids=prompt_ids):
                with self.assertRaisesRegex(ValueError, message):
                    build_generation_plan(self.prompt_pack(), ["base"], prompt_ids=prompt_ids)

    def test_coverage_preserves_invalid_outputs_as_evidence(self) -> None:
        plan = build_generation_plan(self.prompt_pack(), ["adapter"], seeds=[42])
        observations = [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "requested_text": row["text"],
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
        observations = [{"sample_id": row["sample_id"], "valid": True} for row in plan["samples"][:-1]]
        observations.append(dict(observations[0]))
        result = check_suite_coverage(plan, observations)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["missing_sample_ids"]), 1)
        self.assertEqual(len(result["duplicate_sample_ids"]), 1)

    def test_coverage_rejects_sample_id_with_spoofed_plan_fields(self) -> None:
        plan = build_generation_plan(self.prompt_pack(), ["adapter"], seeds=[42])
        observations = [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "requested_text": row["text"],
                "valid": True,
            }
            for row in plan["samples"]
        ]
        observations[0]["requested_text"] = "different text"
        observations[0]["seed"] = 314159
        result = check_suite_coverage(plan, observations)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["mismatched_observations"],
            [{"sample_id": observations[0]["sample_id"], "fields": ["requested_text", "seed"]}],
        )

    def test_prompt_pack_rejects_unimplemented_objective_metric(self) -> None:
        prompt_pack = self.prompt_pack()
        prompt_pack["objective_metrics"].append("subjective_magic_score")
        errors = validate_prompt_pack(prompt_pack)
        self.assertTrue(any("unsupported metrics" in error for error in errors))

    def test_prompt_pack_rejects_ambiguous_or_unbound_lexical_anchors(self) -> None:
        prompt_pack = self.prompt_pack()
        anchors = prompt_pack["prompts"][1]["lexical_anchors"]
        anchors[0]["surface"] = "not in the prompt"
        anchors[0]["accepted_asr_forms"] = ["not in the prompt", "pie say", "pie-say"]
        anchors.append(
            {
                "anchor_id": "second-anchor",
                "surface": "Tanjong Pagar",
                "accepted_asr_forms": ["tanjong pagar", "pie say"],
            }
        )
        errors = validate_prompt_pack(prompt_pack)
        self.assertTrue(any("surface must occur" in error for error in errors))
        self.assertTrue(any("unique after normalization" in error for error in errors))
        self.assertTrue(any("overlaps another anchor" in error for error in errors))

    def test_prompt_pack_rejects_duplicate_lexical_anchor_ids_across_prompts(self) -> None:
        prompt_pack = self.prompt_pack()
        prompt_pack["prompts"][2]["lexical_anchors"] = [
            {
                "anchor_id": "paiseh",
                "surface": "Sze Min",
                "accepted_asr_forms": ["sze min"],
            }
        ]
        errors = validate_prompt_pack(prompt_pack)
        self.assertTrue(any("unique across prompts" in error for error in errors))

    def test_prompt_pack_rejects_repeated_surface_and_prompt_colliding_alias(self) -> None:
        prompt_pack = self.prompt_pack()
        local_prompt = prompt_pack["prompts"][1]
        local_prompt["text"] += " I was still paiseh later."
        local_prompt["lexical_anchors"][0]["accepted_asr_forms"].append("tanjong pagar")
        errors = validate_prompt_pack(prompt_pack)
        self.assertTrue(any("surface must occur exactly once" in error for error in errors))
        self.assertTrue(any("must not already occur" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
