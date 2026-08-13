from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.listening import (
    aggregate_listening_results,
    build_blind_pack,
    build_listening_assignment_plan,
    build_rater_review_packet,
    build_rater_submission,
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
            "schema_version": "1.1.0",
            "routes": [
                {
                    "criterion": "identity",
                    "selector": "all_samples",
                    "direction": "higher_is_better",
                    "review_prompt": "How close is identity?",
                    "low_label": "No match",
                    "high_label": "Strong match",
                },
                {
                    "criterion": "lexical_pronunciation",
                    "selector": "lexical_anchors",
                    "direction": "higher_is_better",
                    "review_prompt": "How accurate is pronunciation?",
                    "low_label": "Wrong",
                    "high_label": "Accurate",
                },
                {
                    "criterion": "cadence",
                    "selector": "categories",
                    "categories": ["long_form"],
                    "direction": "lower_is_better",
                    "review_prompt": "How monotonous is cadence?",
                    "low_label": "No monotony",
                    "high_label": "Severe monotony",
                },
            ],
        }

    def scheduled_pack(self, rater_ids: list[str] | None = None) -> tuple[dict, dict]:
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
        return build_blind_pack(
            samples,
            None,
            seed=42,
            assignment_plan=assignment_plan,
            generation_plan=generation_plan,
            rater_ids=rater_ids or ["rater-a", "rater-b"],
        )

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
        self.assertEqual(
            result["criterion_definitions"],
            [{"criterion": "speaker_identity", "direction": "unspecified"}],
        )

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
        definitions = {row["criterion"]: row for row in first["criterion_definitions"]}
        self.assertEqual(definitions["identity"]["direction"], "higher_is_better")
        self.assertEqual(definitions["cadence"]["direction"], "lower_is_better")

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
        definitions = {row["criterion"]: row for row in result["criterion_definitions"]}
        self.assertEqual(definitions["identity"]["direction"], "higher_is_better")
        self.assertEqual(definitions["cadence"]["direction"], "lower_is_better")
        self.assertNotIn("composite_score", result)
        brief = next(row for row in review["samples"] if row["prompt_id"] == "brief")
        ratings["ratings"].append(
            {"rater_id": "rater", "blind_id": brief["blind_id"], "criterion": "cadence", "score": 3}
        )
        with self.assertRaisesRegex(ValueError, "criterion not assigned"):
            aggregate_listening_results(review, reveal, ratings)

    def test_builds_deterministic_counterbalanced_rater_schedules(self) -> None:
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
        first = build_blind_pack(
            samples,
            None,
            seed=42,
            assignment_plan=assignment_plan,
            generation_plan=generation_plan,
            rater_ids=["rater-b", "rater-a"],
        )
        second = build_blind_pack(
            samples,
            None,
            seed=42,
            assignment_plan=assignment_plan,
            generation_plan=generation_plan,
            rater_ids=["rater-a", "rater-b"],
        )
        self.assertEqual(first, second)
        review, reveal = first
        self.assertEqual(review["schema_version"], "1.1.0")
        schedules = review["presentation_schedules"]
        self.assertEqual([row["rater_id"] for row in schedules], ["rater-a", "rater-b"])
        self.assertNotEqual(schedules[0]["sample_order"], schedules[1]["sample_order"])
        blind_ids = {row["blind_id"] for row in review["samples"]}
        criteria_by_id = {row["blind_id"]: row["criteria"] for row in review["samples"]}
        for schedule in schedules:
            self.assertEqual(set(schedule["sample_order"]), blind_ids)
            self.assertEqual(len(schedule["sample_order"]), len(blind_ids))
            for criterion, order in schedule["criterion_orders"].items():
                expected = [blind_id for blind_id in schedule["sample_order"] if criterion in criteria_by_id[blind_id]]
                self.assertEqual(order, expected)
        audit = reveal["counterbalance_audit"]
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["max_candidate_position_imbalance"], 0)
        self.assertEqual(audit["max_candidate_pass_imbalance"], 1)
        self.assertEqual(audit["master_order_matched_candidate_minimum_separation"], 3)
        for row in audit["candidate_position_counts"]:
            self.assertTrue(all(counts == [1, 1] for counts in row["candidate_position_counts"].values()))
        for row in audit["rater_candidate_pass_counts"]:
            self.assertTrue(all(max(counts) - min(counts) <= 1 for counts in row["candidate_pass_counts"].values()))
        self.assertNotIn("candidate_id", str(review))
        self.assertNotIn("source_audio_path", str(review))

    def test_counterbalanced_schedules_reject_ood_inputs_and_rater_drift(self) -> None:
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
        for malformed in ([], ["rater", "rater"], [" "], {"rater": "a"}):
            with self.subTest(rater_ids=malformed), self.assertRaisesRegex(ValueError, "rater_ids must"):
                build_blind_pack(
                    samples,
                    None,
                    seed=42,
                    assignment_plan=assignment_plan,
                    generation_plan=generation_plan,
                    rater_ids=malformed,
                )
        with self.assertRaisesRegex(ValueError, "assignment_plan is required with rater_ids"):
            build_blind_pack(samples[:2], ["identity"], seed=42, rater_ids=["rater"])

        review, reveal = build_blind_pack(
            samples,
            None,
            seed=42,
            assignment_plan=assignment_plan,
            generation_plan=generation_plan,
            rater_ids=["rater-a", "rater-b"],
        )
        tampered_reveal = json.loads(json.dumps(reveal))
        tampered_reveal["counterbalance_audit"]["status"] = "failed"
        empty_ratings = {
            "scale": {"min": 1, "max": 5},
            "expected_rater_ids": ["rater-a", "rater-b"],
            "ratings": [],
        }
        with self.assertRaisesRegex(ValueError, "counterbalance audit does not match"):
            aggregate_listening_results(review, tampered_reveal, empty_ratings)

        forged_review = json.loads(json.dumps(review))
        forged_reveal = json.loads(json.dumps(reveal))
        forged_reveal["counterbalance_audit"]["max_candidate_pass_imbalance"] = 0
        canonical_audit = json.dumps(
            forged_reveal["counterbalance_audit"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        forged_review["counterbalance_audit_sha256"] = hashlib.sha256(canonical_audit).hexdigest()
        canonical_review = json.dumps(
            forged_review,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        forged_reveal["review_sha256"] = hashlib.sha256(canonical_review).hexdigest()
        with self.assertRaisesRegex(ValueError, "counterbalance audit does not match"):
            aggregate_listening_results(forged_review, forged_reveal, empty_ratings)

        ratings = {
            "scale": {"min": 1, "max": 5},
            "expected_rater_ids": ["rater-a", "rater-c"],
            "ratings": [
                {
                    "rater_id": rater_id,
                    "blind_id": row["blind_id"],
                    "criterion": criterion,
                    "score": 3,
                }
                for rater_id in ("rater-a", "rater-c")
                for row in review["samples"]
                for criterion in row["criteria"]
            ],
        }
        with self.assertRaisesRegex(ValueError, "exactly match the review presentation schedules"):
            aggregate_listening_results(review, reveal, ratings)

        ratings["expected_rater_ids"] = ["rater-a", "rater-b"]
        ratings["ratings"] = [
            {
                "rater_id": rater_id,
                "blind_id": row["blind_id"],
                "criterion": criterion,
                "score": 3,
            }
            for rater_id in ("rater-a", "rater-b")
            for row in review["samples"]
            for criterion in row["criteria"]
        ]
        result = aggregate_listening_results(review, reveal, ratings)
        self.assertEqual(result["presentation"]["mode"], "counterbalanced_per_rater")
        self.assertEqual(result["presentation"]["scheduled_rater_count"], 2)

    def test_counterbalancing_generalizes_to_three_candidates(self) -> None:
        generation_plan = self.generation_plan()
        third_candidate = []
        for row in generation_plan["samples"][:3]:
            clone = dict(row)
            clone["candidate_id"] = "challenger"
            clone["sample_id"] = row["sample_id"].replace("base--", "challenger--")
            third_candidate.append(clone)
        generation_plan["samples"].extend(third_candidate)
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
            rater_ids=["rater-c", "rater-a", "rater-b"],
        )
        self.assertEqual(len(review["presentation_schedules"]), 3)
        audit = reveal["counterbalance_audit"]
        self.assertEqual(audit["candidate_count"], 3)
        self.assertEqual(audit["max_candidate_position_imbalance"], 0)
        self.assertEqual(audit["max_candidate_pass_imbalance"], 0)
        for row in audit["candidate_position_counts"]:
            self.assertTrue(all(counts == [1, 1, 1] for counts in row["candidate_position_counts"].values()))

    def test_exports_one_rater_packet_and_aggregates_bound_submissions(self) -> None:
        review, reveal = self.scheduled_pack()
        submissions = []
        for rater_id in ("rater-a", "rater-b"):
            packet = build_rater_review_packet(review, rater_id)
            self.assertEqual(packet["rater_id"], rater_id)
            self.assertNotIn("presentation_schedules", packet)
            other_rater = "rater-b" if rater_id == "rater-a" else "rater-a"
            self.assertNotIn(other_rater, str(packet))
            self.assertEqual(packet["sample_order"], [row["blind_id"] for row in packet["samples"]])
            ratings = {
                "scale": {"min": 1, "max": 5},
                "presentation_log": packet["rating_order"],
                "ratings": [
                    {"blind_id": row["blind_id"], "criterion": criterion, "score": 3}
                    for row in packet["samples"]
                    for criterion in row["criteria"]
                ],
            }
            submission = build_rater_submission(packet, ratings)
            reversed_ratings = dict(ratings)
            reversed_ratings["ratings"] = list(reversed(ratings["ratings"]))
            self.assertEqual(submission, build_rater_submission(packet, reversed_ratings))
            self.assertEqual(submission["coverage"]["status"], "complete")
            self.assertEqual(submission["packet_sha256"], packet["packet_sha256"])
            submissions.append(submission)
        result = aggregate_listening_results(
            review,
            reveal,
            {"schema_version": "1.1.0", "scale": {"min": 1, "max": 5}, "submissions": list(reversed(submissions))},
        )
        forward = aggregate_listening_results(
            review,
            reveal,
            {"schema_version": "1.1.0", "scale": {"min": 1, "max": 5}, "submissions": submissions},
        )
        self.assertEqual(result, forward)
        self.assertEqual(result["coverage"]["status"], "complete")
        self.assertEqual(result["rater_count"], 2)
        self.assertEqual(len(result["presentation"]["submission_receipts_sha256"]), 64)

    def test_rater_delivery_contract_rejects_ood_and_forged_inputs(self) -> None:
        review, reveal = self.scheduled_pack()
        with self.assertRaisesRegex(ValueError, "no presentation schedule"):
            build_rater_review_packet(review, "unknown")
        packet = build_rater_review_packet(review, "rater-a")
        ratings = {
            "scale": {"min": 1, "max": 5},
            "presentation_log": packet["rating_order"],
            "ratings": [
                {"blind_id": row["blind_id"], "criterion": criterion, "score": 3}
                for row in packet["samples"]
                for criterion in row["criteria"]
            ],
        }
        tampered_packet = json.loads(json.dumps(packet))
        tampered_packet["sample_order"].reverse()
        with self.assertRaisesRegex(ValueError, "self-hash"):
            build_rater_submission(tampered_packet, ratings)
        wrong_order = dict(ratings)
        wrong_order["presentation_log"] = list(reversed(packet["rating_order"]))
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            build_rater_submission(packet, wrong_order)
        with_rater_id = json.loads(json.dumps(ratings))
        with_rater_id["ratings"][0]["rater_id"] = "rater-a"
        with self.assertRaisesRegex(ValueError, "without rater_id"):
            build_rater_submission(packet, with_rater_id)
        submission = build_rater_submission(packet, ratings)
        forged = json.loads(json.dumps(submission))
        forged["coverage"]["observed_rating_count"] -= 1
        forged_payload = {key: value for key, value in forged.items() if key != "submission_sha256"}
        forged["submission_sha256"] = hashlib.sha256(
            json.dumps(forged_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "does not match its review packet"):
            aggregate_listening_results(
                review,
                reveal,
                {"schema_version": "1.1.0", "scale": {"min": 1, "max": 5}, "submissions": [forged]},
                allow_incomplete=True,
            )
        with self.assertRaisesRegex(ValueError, "every scheduled rater"):
            aggregate_listening_results(
                review,
                reveal,
                {"schema_version": "1.1.0", "scale": {"min": 1, "max": 5}, "submissions": [submission]},
            )

    def test_incomplete_rater_receipt_records_attrition_without_claiming_compliance(self) -> None:
        review, reveal = self.scheduled_pack()
        packet = build_rater_review_packet(review, "rater-a")
        first_cell = packet["rating_order"][0]
        submission = build_rater_submission(
            packet,
            {
                "scale": {"min": 1, "max": 5},
                "presentation_log": [first_cell],
                "ratings": [{"blind_id": first_cell["blind_id"], "criterion": first_cell["criterion"], "score": 3}],
            },
            allow_incomplete=True,
        )
        self.assertEqual(submission["coverage"]["status"], "incomplete")
        self.assertIn("self-attested", submission["compliance_boundary"])
        result = aggregate_listening_results(
            review,
            reveal,
            {"schema_version": "1.1.0", "scale": {"min": 1, "max": 5}, "submissions": [submission]},
            allow_incomplete=True,
        )
        self.assertEqual(result["coverage"]["status"], "incomplete")
        self.assertEqual(result["expected_rater_count"], 2)
        self.assertEqual(result["rater_count"], 1)

        no_returns = aggregate_listening_results(
            review,
            reveal,
            {"schema_version": "1.1.0", "scale": {"min": 1, "max": 5}, "submissions": []},
            allow_incomplete=True,
        )
        self.assertEqual(no_returns["rater_count"], 0)
        self.assertEqual(no_returns["expected_rater_count"], 2)
        self.assertEqual(no_returns["coverage"]["status"], "incomplete")

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

        focused = build_listening_assignment_plan(
            generation_plan,
            unmatched,
            allow_unmatched_routes=True,
        )
        self.assertEqual(focused["route_coverage_policy"], "allow_unmatched_for_focused_plan")
        self.assertEqual(focused["excluded_unmatched_criteria"], ["cadence"])
        self.assertNotIn("cadence", focused["criteria"])
        self.assertNotIn("cadence", focused["assignments"][0]["criteria"])
        validation = validate_listening_assignment_plan(focused, generation_plan=generation_plan)
        self.assertEqual(validation["criterion_count"], 2)

        focused["route_coverage_policy"] = "require_every_route"
        with self.assertRaisesRegex(ValueError, "self-hash"):
            validate_listening_assignment_plan(focused, generation_plan=generation_plan)

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

        missing_direction = self.routing()
        del missing_direction["routes"][0]["direction"]
        with self.assertRaisesRegex(ValueError, "direction must be one of"):
            build_listening_assignment_plan(self.generation_plan(), missing_direction)

        invalid_direction = self.routing()
        invalid_direction["routes"][0]["direction"] = "larger_number_is_vaguer"
        with self.assertRaisesRegex(ValueError, "direction must be one of"):
            build_listening_assignment_plan(self.generation_plan(), invalid_direction)

        for field in ("review_prompt", "low_label", "high_label"):
            malformed = self.routing()
            malformed["routes"][0][field] = " "
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"{field} must be a non-empty string"):
                build_listening_assignment_plan(self.generation_plan(), malformed)

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
            rater_ids_path = root / "rater-ids.json"
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
            rater_ids_path.write_text(json.dumps(["rater-b", "rater-a"]), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-listening-pack",
                        str(samples_path),
                        "--assignment-plan",
                        str(assignments_path),
                        "--generation-plan",
                        str(plan_path),
                        "--rater-ids",
                        str(rater_ids_path),
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
            self.assertEqual(
                [row["rater_id"] for row in review["presentation_schedules"]],
                ["rater-a", "rater-b"],
            )
            packet_path = root / "rater-a-packet.json"
            ratings_path = root / "rater-a-ratings.json"
            submission_path = root / "rater-a-submission.json"
            self.assertEqual(
                main(
                    [
                        "export-rater-listening-packet",
                        str(review_path),
                        "--rater-id",
                        "rater-a",
                        "--output",
                        str(packet_path),
                    ]
                ),
                0,
            )
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            ratings_path.write_text(
                json.dumps(
                    {
                        "scale": {"min": 1, "max": 5},
                        "presentation_log": packet["rating_order"],
                        "ratings": [
                            {"blind_id": row["blind_id"], "criterion": criterion, "score": 3}
                            for row in packet["samples"]
                            for criterion in row["criteria"]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "build-rater-listening-submission",
                        str(packet_path),
                        str(ratings_path),
                        "--output",
                        str(submission_path),
                    ]
                ),
                0,
            )
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
            self.assertEqual(submission["rater_id"], "rater-a")
            self.assertEqual(submission["coverage"]["status"], "complete")

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
