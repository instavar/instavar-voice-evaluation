from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.listening import (
    aggregate_listening_results,
    build_blind_pack,
    build_listening_assignment_plan,
    stage_blind_audio,
    validate_listening_assignment_plan,
)


class ListeningPackTests(unittest.TestCase):
    @staticmethod
    def generation_plan() -> dict:
        samples = []
        prompts = [
            ("brief", "neutral", False),
            ("local", "local_context", True),
            ("long", "long_form", False),
        ]
        for candidate in ("base", "adapter"):
            for prompt_id, category, has_anchor in prompts:
                row = {
                    "sample_id": f"{candidate}--{prompt_id}--seed-42",
                    "candidate_id": candidate,
                    "prompt_id": prompt_id,
                    "category": category,
                    "seed": 42,
                    "text": f"Text for {prompt_id}",
                }
                if prompt_id == "long":
                    row["instruction"] = "Read steadily."
                if has_anchor:
                    row["lexical_anchors"] = [
                        {"anchor_id": "paiseh", "surface": "paiseh", "accepted_asr_forms": ["paiseh"]}
                    ]
                samples.append(row)
        return {"schema_version": "1.1.0", "samples": samples}

    @staticmethod
    def routing() -> dict:
        return {
            "schema_version": "1.0.0",
            "routes": [
                {"criterion": "identity", "selector": "all_samples"},
                {"criterion": "lexical_pronunciation", "selector": "lexical_anchors"},
                {"criterion": "cadence", "selector": "categories", "categories": ["long_form"]},
            ],
        }

    def test_builds_deterministic_blind_pack(self) -> None:
        samples = [
            {"sample_id": "base-1", "candidate_id": "base", "prompt_id": "p1", "audio_path": "base.wav"},
            {"sample_id": "adapter-1", "candidate_id": "adapter", "prompt_id": "p1", "audio_path": "adapter.wav"},
        ]
        first = build_blind_pack(samples, ["speaker_identity"], seed=42)
        second = build_blind_pack(samples, ["speaker_identity"], seed=42)
        self.assertEqual(first, second)
        review, reveal = first
        self.assertTrue(review["blind"])
        self.assertNotIn("candidate_id", review["samples"][0])
        self.assertNotIn("base", review["samples"][0]["audio_path"])
        self.assertNotIn("adapter", review["samples"][0]["audio_path"])
        self.assertIn("source_audio_path", reveal["mapping"][0])
        self.assertEqual(len(reveal["review_sha256"]), 64)
        self.assertEqual(len(reveal["mapping"]), 2)

    def test_aggregates_revealed_ratings_per_criterion(self) -> None:
        samples = [
            {"sample_id": "base-1", "candidate_id": "base", "prompt_id": "p1", "audio_path": "base.wav"},
            {"sample_id": "adapter-1", "candidate_id": "adapter", "prompt_id": "p1", "audio_path": "adapter.wav"},
        ]
        review, reveal = build_blind_pack(samples, ["speaker_identity"], seed=42)
        ratings = {
            "scale": {"min": 1, "max": 5},
            "expected_rater_ids": ["a", "b"],
            "ratings": [
                {"rater_id": "a", "blind_id": "sample-0001", "criterion": "speaker_identity", "score": 4},
                {"rater_id": "b", "blind_id": "sample-0001", "criterion": "speaker_identity", "score": 5},
                {"rater_id": "a", "blind_id": "sample-0002", "criterion": "speaker_identity", "score": 2},
                {"rater_id": "b", "blind_id": "sample-0002", "criterion": "speaker_identity", "score": 3},
            ],
        }
        result = aggregate_listening_results(review, reveal, ratings, seed=7)
        self.assertEqual(result["rater_count"], 2)
        self.assertEqual(set(result["candidates"]), {"base", "adapter"})
        self.assertNotIn("composite_score", result)
        self.assertIsNotNone(result["agreement"]["speaker_identity"]["krippendorff_alpha_interval"])
        self.assertEqual(result["coverage"]["status"], "complete")
        self.assertEqual(result["coverage"]["assignment_mode"], "all_criteria_per_sample")

    def test_builds_plan_bound_criterion_assignments(self) -> None:
        generation_plan = self.generation_plan()
        first = build_listening_assignment_plan(generation_plan, self.routing())
        second = build_listening_assignment_plan(generation_plan, self.routing())
        self.assertEqual(first, second)
        self.assertEqual(validate_listening_assignment_plan(first, generation_plan=generation_plan)["sample_count"], 6)
        by_prompt = {row["prompt_id"]: row["criteria"] for row in first["assignments"]}
        self.assertEqual(by_prompt["brief"], ["identity"])
        self.assertEqual(by_prompt["local"], ["identity", "lexical_pronunciation"])
        self.assertEqual(by_prompt["long"], ["identity", "cadence"])
        long_form = next(row for row in first["assignments"] if row["prompt_id"] == "long")
        self.assertEqual(long_form["stimulus"]["instruction"], "Read steadily.")
        local = next(row for row in first["assignments"] if row["prompt_id"] == "local")
        self.assertEqual(local["stimulus"]["text"], "Text for local")
        self.assertEqual(
            local["stimulus"]["lexical_targets"],
            [{"anchor_id": "paiseh", "surface": "paiseh"}],
        )
        self.assertNotIn("accepted_asr_forms", str(local["stimulus"]))

    def test_plan_bound_pack_requires_only_assigned_ratings(self) -> None:
        generation_plan = self.generation_plan()
        assignment_plan = build_listening_assignment_plan(generation_plan, self.routing())
        samples = [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "audio_path": f"{row['sample_id']}.wav",
            }
            for row in generation_plan["samples"]
        ]
        review, reveal = build_blind_pack(
            samples,
            None,
            seed=42,
            assignment_plan=assignment_plan,
            generation_plan=generation_plan,
        )
        self.assertEqual(review["assignment_plan"]["assignment_plan_sha256"], assignment_plan["assignment_plan_sha256"])
        local_review = next(row for row in review["samples"] if row["prompt_id"] == "local")
        self.assertEqual(local_review["stimulus"]["text"], "Text for local")
        self.assertNotIn("accepted_asr_forms", str(local_review))
        ratings = {
            "scale": {"min": 1, "max": 5},
            "expected_rater_ids": ["rater"],
            "ratings": [
                {
                    "rater_id": "rater",
                    "blind_id": row["blind_id"],
                    "criterion": criterion,
                    "score": 4,
                }
                for row in review["samples"]
                for criterion in row["criteria"]
            ],
        }
        result = aggregate_listening_results(review, reveal, ratings)
        self.assertEqual(result["coverage"]["assignment_mode"], "plan_bound")
        self.assertEqual(result["coverage"]["expected_rating_count"], 10)
        brief = next(row for row in review["samples"] if row["prompt_id"] == "brief")
        ratings["ratings"].append(
            {"rater_id": "rater", "blind_id": brief["blind_id"], "criterion": "cadence", "score": 3}
        )
        with self.assertRaisesRegex(ValueError, "criterion not assigned"):
            aggregate_listening_results(review, reveal, ratings)

    def test_assignment_plan_rejects_tampering_and_ood_routing(self) -> None:
        generation_plan = self.generation_plan()
        plan = build_listening_assignment_plan(generation_plan, self.routing())
        plan["assignments"][0]["criteria"].append("cadence")
        with self.assertRaisesRegex(ValueError, "self-hash"):
            validate_listening_assignment_plan(plan, generation_plan=generation_plan)

        unmatched = self.routing()
        unmatched["routes"][2]["categories"] = ["not_present"]
        with self.assertRaisesRegex(ValueError, "matches no"):
            build_listening_assignment_plan(generation_plan, unmatched)

        drifted = self.generation_plan()
        drifted["samples"][3]["category"] = "different_for_adapter"
        with self.assertRaisesRegex(ValueError, "drift across candidates"):
            build_listening_assignment_plan(drifted, self.routing())

        text_drift = self.generation_plan()
        text_drift["samples"][3]["text"] = "Candidate-specific wording"
        with self.assertRaisesRegex(ValueError, "drift across candidates"):
            build_listening_assignment_plan(text_drift, self.routing())

    def test_assignment_plan_rejects_missing_or_malformed_review_stimuli(self) -> None:
        missing_text = self.generation_plan()
        del missing_text["samples"][0]["text"]
        with self.assertRaisesRegex(ValueError, "text must be a non-empty string"):
            build_listening_assignment_plan(missing_text, self.routing())

        empty_instruction = self.generation_plan()
        empty_instruction["samples"][0]["instruction"] = " "
        with self.assertRaisesRegex(ValueError, "instruction must be a non-empty string"):
            build_listening_assignment_plan(empty_instruction, self.routing())

        malformed_anchor = self.generation_plan()
        malformed_anchor["samples"][1]["lexical_anchors"] = [{"anchor_id": "paiseh"}]
        with self.assertRaisesRegex(ValueError, "surface must be non-empty"):
            build_listening_assignment_plan(malformed_anchor, self.routing())

    def test_plan_bound_pack_rejects_sample_identity_drift(self) -> None:
        generation_plan = self.generation_plan()
        assignment_plan = build_listening_assignment_plan(generation_plan, self.routing())
        samples = [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "audio_path": f"{row['sample_id']}.wav",
            }
            for row in generation_plan["samples"]
        ]
        samples[0]["candidate_id"] = "spoofed"
        with self.assertRaisesRegex(ValueError, "does not match assignment field candidate_id"):
            build_blind_pack(
                samples,
                None,
                seed=42,
                assignment_plan=assignment_plan,
                generation_plan=generation_plan,
            )

    def test_cli_builds_assignment_plan_and_plan_bound_pack(self) -> None:
        generation_plan = self.generation_plan()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "generation-plan.json"
            routing_path = root / "routing.json"
            assignments_path = root / "assignments.json"
            samples_path = root / "samples.json"
            review_path = root / "review.json"
            reveal_path = root / "reveal.json"
            plan_path.write_text(json.dumps(generation_plan), encoding="utf-8")
            routing_path.write_text(json.dumps(self.routing()), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-listening-assignment-plan",
                        str(plan_path),
                        "--routing",
                        str(routing_path),
                        "--output",
                        str(assignments_path),
                    ]
                ),
                0,
            )
            samples = [
                {
                    "sample_id": row["sample_id"],
                    "candidate_id": row["candidate_id"],
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "audio_path": f"{row['sample_id']}.wav",
                }
                for row in generation_plan["samples"]
            ]
            samples_path.write_text(json.dumps(samples), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-listening-pack",
                        str(samples_path),
                        "--assignment-plan",
                        str(assignments_path),
                        "--generation-plan",
                        str(plan_path),
                        "--review-output",
                        str(review_path),
                        "--reveal-output",
                        str(reveal_path),
                        "--seed",
                        "42",
                    ]
                ),
                0,
            )
            review = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertTrue(all({"criteria", "stimulus"} <= row.keys() for row in review["samples"]))

    def test_rejects_incomplete_rating_matrix_by_default(self) -> None:
        samples = [
            {"sample_id": "base-1", "candidate_id": "base", "prompt_id": "p1", "audio_path": "base.wav"},
            {"sample_id": "adapter-1", "candidate_id": "adapter", "prompt_id": "p1", "audio_path": "adapter.wav"},
        ]
        review, reveal = build_blind_pack(samples, ["speaker_identity"], seed=42)
        ratings = {
            "scale": {"min": 1, "max": 5},
            "expected_rater_ids": ["a", "b"],
            "ratings": [
                {"rater_id": "a", "blind_id": "sample-0001", "criterion": "speaker_identity", "score": 4},
                {"rater_id": "b", "blind_id": "sample-0001", "criterion": "speaker_identity", "score": 5},
                {"rater_id": "a", "blind_id": "sample-0002", "criterion": "speaker_identity", "score": 3},
            ],
        }
        with self.assertRaisesRegex(ValueError, "ratings matrix is incomplete"):
            aggregate_listening_results(review, reveal, ratings)
        partial = aggregate_listening_results(review, reveal, ratings, allow_incomplete=True)
        self.assertEqual(partial["coverage"]["status"], "incomplete")

    def test_expected_rater_with_zero_submissions_remains_visible(self) -> None:
        samples = [
            {"sample_id": "base-1", "candidate_id": "base", "prompt_id": "p1", "audio_path": "base.wav"},
            {"sample_id": "adapter-1", "candidate_id": "adapter", "prompt_id": "p1", "audio_path": "adapter.wav"},
        ]
        review, reveal = build_blind_pack(samples, ["speaker_identity"], seed=42)
        ratings = {
            "scale": {"min": 1, "max": 5},
            "expected_rater_ids": ["a", "never-submitted"],
            "ratings": [
                {"rater_id": "a", "blind_id": "sample-0001", "criterion": "speaker_identity", "score": 4},
                {"rater_id": "a", "blind_id": "sample-0002", "criterion": "speaker_identity", "score": 3},
            ],
        }
        partial = aggregate_listening_results(review, reveal, ratings, allow_incomplete=True)
        self.assertEqual(partial["expected_rater_count"], 2)
        self.assertEqual(partial["rater_count"], 1)
        self.assertEqual(partial["coverage"]["missing_rating_count"], 2)

    def test_rejects_mixed_audio_extensions_that_can_unblind_candidates(self) -> None:
        samples = [
            {"sample_id": "base-1", "candidate_id": "base", "prompt_id": "p1", "audio_path": "base.wav"},
            {"sample_id": "adapter-1", "candidate_id": "adapter", "prompt_id": "p1", "audio_path": "adapter.mp3"},
        ]
        with self.assertRaisesRegex(ValueError, "same extension"):
            build_blind_pack(samples, ["speaker_identity"], seed=42)

    def test_stages_audio_under_identity_neutral_paths(self) -> None:
        with TemporaryDirectory() as source_dir, TemporaryDirectory() as output_dir:
            base = Path(source_dir) / "base.wav"
            adapter = Path(source_dir) / "adapter.wav"
            base.write_bytes(b"base-audio")
            adapter.write_bytes(b"adapter-audio")
            samples = [
                {"sample_id": "base-1", "candidate_id": "base", "prompt_id": "p1", "audio_path": str(base)},
                {
                    "sample_id": "adapter-1",
                    "candidate_id": "adapter",
                    "prompt_id": "p1",
                    "audio_path": str(adapter),
                },
            ]
            review, reveal = build_blind_pack(samples, ["speaker_identity"], seed=42)
            result = stage_blind_audio(review, reveal, Path(output_dir))
            self.assertEqual(result["file_count"], 2)
            self.assertTrue(all((Path(output_dir) / row["audio_path"]).is_file() for row in result["files"]))

    def test_staging_refuses_to_overwrite_different_audio(self) -> None:
        with TemporaryDirectory() as source_dir, TemporaryDirectory() as output_dir:
            base = Path(source_dir) / "base.wav"
            adapter = Path(source_dir) / "adapter.wav"
            base.write_bytes(b"base-audio")
            adapter.write_bytes(b"adapter-audio")
            samples = [
                {"sample_id": "base-1", "candidate_id": "base", "prompt_id": "p1", "audio_path": str(base)},
                {
                    "sample_id": "adapter-1",
                    "candidate_id": "adapter",
                    "prompt_id": "p1",
                    "audio_path": str(adapter),
                },
            ]
            review, reveal = build_blind_pack(samples, ["speaker_identity"], seed=42)
            destination = Path(output_dir) / review["samples"][0]["audio_path"]
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"stale-audio")
            with self.assertRaisesRegex(ValueError, "different content"):
                stage_blind_audio(review, reveal, Path(output_dir))


if __name__ == "__main__":
    unittest.main()
