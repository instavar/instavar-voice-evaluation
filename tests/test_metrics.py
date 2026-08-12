from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.metrics import cosine_similarity, score_objective_observations, word_error_rate
from instavar_voice_lab.speaker_references import canonical_sha256, reference_set_sha256, speaker_measurement_sha256


class MetricTests(unittest.TestCase):
    def test_word_error_rate_uses_word_edits(self) -> None:
        self.assertEqual(word_error_rate("one two three", "one two three"), 0.0)
        self.assertAlmostEqual(word_error_rate("one two three", "one four three"), 1 / 3)

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_scores_separate_objective_proxies(self) -> None:
        result = score_objective_observations(
            [
                {
                    "sample_id": "sample-1",
                    "candidate_id": "adapter",
                    "prompt_id": "p1",
                    "requested_text": "hello world",
                    "hypothesis_text": "hello world",
                    "valid": True,
                    "generation_seconds": 0.5,
                    "audio_duration_seconds": 1.0,
                    "peak_memory_bytes": 100,
                    "sample_rate_hz": 24000,
                    "silence_fraction": 0.1,
                    "clipping_fraction": 0.0,
                    "reference_speaker_embedding": [1, 0],
                    "speaker_embedding": [1, 0],
                    "evidence": {
                        "asr": {"extractor": "test", "revision": "1"},
                        "speaker_encoder": {"extractor": "test", "revision": "1"},
                        "runtime": {"extractor": "test", "revision": "1"},
                        "audio_probe": {"extractor": "test-probe", "revision": "1"},
                    },
                },
                {
                    "sample_id": "sample-2",
                    "candidate_id": "adapter",
                    "prompt_id": "p2",
                    "requested_text": "another line",
                    "valid": False,
                },
            ],
            seed=7,
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["invalid_output_rate"], 0.5)
        self.assertEqual(candidate["asr_word_error_rate"]["mean"], 0.0)
        self.assertEqual(candidate["speaker_embedding_similarity"]["mean"], 1.0)
        self.assertEqual(candidate["real_time_factor"]["mean"], 0.5)
        self.assertEqual(candidate["metric_coverage"]["asr_word_error_rate"]["rate"], 1.0)
        self.assertEqual(candidate["metric_coverage"]["generation_seconds"]["rate"], 0.5)
        self.assertEqual(candidate["sample_rate_hz"]["mean"], 24000)
        self.assertEqual(candidate["silence_fraction"]["mean"], 0.1)
        self.assertEqual(candidate["clipping_fraction"]["mean"], 0.0)
        self.assertFalse(result["metric_provenance"]["asr"]["all_content_bound"])
        reference_text = result["metric_provenance"]["asr"]["reference_text"]
        self.assertEqual(reference_text["mode"], "declared_observation")
        self.assertEqual(reference_text["plan_bound_reference_count"], 0)
        self.assertFalse(reference_text["all_scored_references_plan_bound"])
        self.assertNotIn("composite_score", result)
        self.assertFalse(result["proves_perceptual_quality"])

    def test_binds_asr_reference_text_to_generation_plan(self) -> None:
        row = {
            "sample_id": "adapter--p1--seed-42",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "seed": 42,
            "requested_text": "hello world",
            "hypothesis_text": "hello world",
            "valid": True,
            "evidence": {"asr": {"extractor": "test-asr", "revision": "1"}},
        }
        plan = {
            "schema_version": "1.0.0",
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "candidate_id": row["candidate_id"],
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "text": row["requested_text"],
                }
            ],
        }
        result = score_objective_observations([row], generation_plan=plan)
        reference_text = result["metric_provenance"]["asr"]["reference_text"]
        self.assertEqual(reference_text["mode"], "generation_plan")
        self.assertEqual(reference_text["generation_plan_sha256"], canonical_sha256(plan))
        self.assertEqual(reference_text["plan_bound_reference_count"], 1)
        self.assertTrue(reference_text["all_scored_references_plan_bound"])
        self.assertEqual(result["samples"][0]["diagnostics"]["asr_reference_text_binding"], "generation_plan")

    def test_reports_plan_bound_category_strata_without_composite_score(self) -> None:
        rows = []
        plan_samples = []
        fixtures = (
            ("neutral", "neutral_singapore_english", "hello world", "hello world", True),
            ("pronunciation", "pronunciation", "Sze Min paiseh", "see min", True),
            ("cadence", "long_form_cadence", "a longer passage", None, False),
        )
        for prompt_id, category, requested_text, hypothesis, valid in fixtures:
            sample_id = f"adapter--{prompt_id}--seed-42"
            row = {
                "sample_id": sample_id,
                "candidate_id": "adapter",
                "prompt_id": prompt_id,
                "seed": 42,
                "requested_text": requested_text,
                "valid": valid,
            }
            if hypothesis is not None:
                row["hypothesis_text"] = hypothesis
                row["evidence"] = {"asr": {"extractor": "test-asr", "revision": "1"}}
            rows.append(row)
            plan_samples.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": "adapter",
                    "prompt_id": prompt_id,
                    "category": category,
                    "seed": 42,
                    "text": requested_text,
                }
            )
        plan = {"schema_version": "1.0.0", "samples": plan_samples}
        result = score_objective_observations(rows, generation_plan=plan)
        strata = result["plan_category_stratification"]
        self.assertEqual(strata["schema_version"], "1.0.0")
        self.assertEqual(strata["mode"], "generation_plan")
        self.assertEqual(strata["generation_plan_sha256"], canonical_sha256(plan))
        self.assertEqual(strata["categorized_sample_count"], 3)
        categories = {
            category["category"]: category
            for category in strata["candidates"][0]["categories"]
        }
        self.assertEqual(categories["neutral_singapore_english"]["metrics"]["asr_word_error_rate"]["mean"], 0.0)
        self.assertAlmostEqual(categories["pronunciation"]["metrics"]["asr_word_error_rate"]["mean"], 2 / 3)
        self.assertEqual(categories["long_form_cadence"]["invalid_output_rate"], 1.0)
        self.assertNotIn("composite_score", strata)

    def test_category_strata_reject_unbound_or_malformed_plan_rows(self) -> None:
        row = {
            "sample_id": "adapter--p1--seed-42",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "seed": 42,
            "requested_text": "hello",
            "valid": True,
        }
        missing = {
            "schema_version": "1.0.0",
            "samples": [
                {
                    "sample_id": "different",
                    "candidate_id": "adapter",
                    "prompt_id": "p1",
                    "category": "pronunciation",
                    "seed": 42,
                    "text": "hello",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "absent from the category generation plan"):
            score_objective_observations([row], generation_plan=missing)
        malformed = deepcopy(missing)
        malformed["samples"][0]["sample_id"] = row["sample_id"]
        malformed["samples"][0]["category"] = ""
        with self.assertRaisesRegex(ValueError, "category must be non-empty"):
            score_objective_observations([row], generation_plan=malformed)

        selective_omission = {
            "schema_version": "1.0.0",
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "candidate_id": "adapter",
                    "prompt_id": "p1",
                    "category": "pronunciation",
                    "seed": 42,
                    "text": "hello",
                },
                {
                    "sample_id": "adapter--p1--seed-43",
                    "candidate_id": "adapter",
                    "prompt_id": "p1",
                    "seed": 43,
                    "text": "hello",
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "must use one category"):
            score_objective_observations([row], generation_plan=selective_omission)

    def test_reports_plan_bound_lexical_anchor_hits_and_coverage(self) -> None:
        rows = []
        samples = []
        fixtures = (
            (42, True, "I felt pie say today", "hit"),
            (43, True, "I felt pleased today", "miss"),
            (44, True, None, "asr_unavailable"),
            (45, False, None, "invalid_output"),
        )
        anchors = [
            {
                "anchor_id": "paiseh",
                "surface": "paiseh",
                "accepted_asr_forms": ["paiseh", "pai seh", "pie say"],
            }
        ]
        for seed, valid, hypothesis, expected_status in fixtures:
            sample_id = f"adapter--local-context--seed-{seed}"
            row = {
                "sample_id": sample_id,
                "candidate_id": "adapter",
                "prompt_id": "local-context",
                "seed": seed,
                "requested_text": "I felt paiseh today",
                "valid": valid,
            }
            if hypothesis is not None:
                row["hypothesis_text"] = hypothesis
                row["evidence"] = {"asr": {"extractor": "test-asr", "revision": "1"}}
            rows.append(row)
            samples.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": "adapter",
                    "prompt_id": "local-context",
                    "category": "natural_local_context",
                    "seed": seed,
                    "text": row["requested_text"],
                    "lexical_anchors": deepcopy(anchors),
                }
            )
        plan = {"schema_version": "1.0.0", "samples": samples}
        result = score_objective_observations(rows, generation_plan=plan)
        evidence = result["plan_lexical_anchor_evidence"]
        self.assertEqual(evidence["schema_version"], "1.0.0")
        self.assertEqual(evidence["mode"], "generation_plan")
        self.assertEqual(evidence["generation_plan_sha256"], canonical_sha256(plan))
        self.assertEqual(evidence["planned_anchor_instance_count"], 4)
        anchor = evidence["candidates"][0]["anchors"][0]
        self.assertEqual(anchor["evaluable_instance_count"], 2)
        self.assertEqual(anchor["hit_count"], 1)
        self.assertEqual(anchor["hit_rate"], 0.5)
        statuses = [
            sample["diagnostics"]["lexical_anchors"][0]["status"]
            for sample in result["samples"]
        ]
        self.assertEqual(statuses, [fixture[3] for fixture in fixtures])
        self.assertIn("recognition evidence only", evidence["evidence_boundary"])

    def test_lexical_anchor_matching_uses_token_boundaries(self) -> None:
        row = {
            "sample_id": "adapter--p1--seed-42",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "seed": 42,
            "requested_text": "He will go",
            "hypothesis_text": "The will go",
            "valid": True,
            "evidence": {"asr": {"extractor": "test-asr", "revision": "1"}},
        }
        plan = {
            "schema_version": "1.0.0",
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "candidate_id": row["candidate_id"],
                    "prompt_id": row["prompt_id"],
                    "category": "pronunciation",
                    "seed": row["seed"],
                    "text": row["requested_text"],
                    "lexical_anchors": [
                        {"anchor_id": "he", "surface": "He", "accepted_asr_forms": ["he"]}
                    ],
                }
            ],
        }
        result = score_objective_observations([row], generation_plan=plan)
        anchor = result["samples"][0]["diagnostics"]["lexical_anchors"][0]
        self.assertEqual(anchor["status"], "miss")

    def test_lexical_anchors_reject_candidate_specific_alias_drift(self) -> None:
        rows = [
            {
                "sample_id": f"{candidate}--p1--seed-42",
                "candidate_id": candidate,
                "prompt_id": "p1",
                "seed": 42,
                "requested_text": "I felt paiseh",
                "valid": True,
            }
            for candidate in ("base", "adapter")
        ]
        samples = [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "category": "pronunciation",
                "seed": row["seed"],
                "text": row["requested_text"],
                "lexical_anchors": [
                    {
                        "anchor_id": "paiseh",
                        "surface": "paiseh",
                        "accepted_asr_forms": (
                            ["paiseh", "pie say"] if row["candidate_id"] == "base" else ["paiseh"]
                        ),
                    }
                ],
            }
            for row in rows
        ]
        with self.assertRaisesRegex(ValueError, "must use one anchor set"):
            score_objective_observations(
                rows,
                generation_plan={"schema_version": "1.0.0", "samples": samples},
            )

    def test_lexical_anchors_reject_repeated_surface_and_prompt_colliding_alias(self) -> None:
        row = {
            "sample_id": "adapter--p1--seed-42",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "seed": 42,
            "requested_text": "I felt paiseh and remained paiseh near Tanjong Pagar",
            "valid": True,
        }
        sample = {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "category": "pronunciation",
            "seed": row["seed"],
            "text": row["requested_text"],
            "lexical_anchors": [
                {
                    "anchor_id": "paiseh",
                    "surface": "paiseh",
                    "accepted_asr_forms": ["paiseh", "tanjong pagar"],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "surface must occur exactly once"):
            score_objective_observations(
                [row],
                generation_plan={"schema_version": "1.0.0", "samples": [sample]},
            )
        sample["text"] = "I felt paiseh near Tanjong Pagar"
        row["requested_text"] = sample["text"]
        with self.assertRaisesRegex(ValueError, "must not already occur"):
            score_objective_observations(
                [row],
                generation_plan={"schema_version": "1.0.0", "samples": [sample]},
            )

    def test_rejects_asr_reference_text_drift_from_generation_plan(self) -> None:
        row = {
            "sample_id": "adapter--p1--seed-42",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "seed": 42,
            "requested_text": "hello world",
            "hypothesis_text": "hello world",
            "valid": True,
            "evidence": {"asr": {"extractor": "test-asr", "revision": "1"}},
        }
        plan = {
            "schema_version": "1.0.0",
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "candidate_id": row["candidate_id"],
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "text": "different text",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "requested_text"):
            score_objective_observations([row], generation_plan=plan)
        plan["samples"][0]["sample_id"] = "missing-sample"
        with self.assertRaisesRegex(ValueError, "absent from the reference generation plan"):
            score_objective_observations([row], generation_plan=plan)

    def test_cli_binds_asr_reference_text_to_generation_plan(self) -> None:
        row = {
            "sample_id": "adapter--p1--seed-42",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "seed": 42,
            "requested_text": "hello world",
            "hypothesis_text": "hello world",
            "valid": True,
            "evidence": {"asr": {"extractor": "test-asr", "revision": "1"}},
        }
        plan = {
            "schema_version": "1.0.0",
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "candidate_id": row["candidate_id"],
                    "prompt_id": row["prompt_id"],
                    "seed": row["seed"],
                    "text": row["requested_text"],
                }
            ],
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations_path = root / "observations.json"
            plan_path = root / "generation-plan.json"
            output_path = root / "scores.json"
            observations_path.write_text(json.dumps([row]), encoding="utf-8")
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "score-objective",
                        str(observations_path),
                        "--generation-plan",
                        str(plan_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["metric_provenance"]["asr"]["reference_text"]["mode"], "generation_plan")

    def test_reports_mixed_metric_provenance_instead_of_hiding_it(self) -> None:
        rows = []
        for revision in ("rev-1", "rev-2"):
            rows.append(
                {
                    "sample_id": revision,
                    "candidate_id": "adapter",
                    "prompt_id": revision,
                    "requested_text": "hello",
                    "hypothesis_text": "hello",
                    "valid": True,
                    "evidence": {"asr": {"extractor": "test-asr", "revision": revision}},
                }
            )
        result = score_objective_observations(rows)
        self.assertFalse(result["metric_provenance"]["asr"]["consistent"])
        self.assertEqual(len(result["metric_provenance"]["asr"]["extractors"]), 2)

    def test_invalid_sample_cannot_improve_quality_aggregates(self) -> None:
        rows = [
            {
                "sample_id": "valid",
                "candidate_id": "adapter",
                "prompt_id": "p1",
                "requested_text": "hello world",
                "hypothesis_text": "hello world",
                "valid": True,
                "reference_speaker_embedding": [1, 0],
                "speaker_embedding": [1, 0],
                "evidence": {
                    "asr": {"extractor": "asr", "revision": "1"},
                    "speaker_encoder": {"extractor": "speaker", "revision": "1"},
                },
            },
            {
                "sample_id": "invalid",
                "candidate_id": "adapter",
                "prompt_id": "p2",
                "requested_text": "different words",
                "hypothesis_text": "different words",
                "valid": False,
                "audio_duration_seconds": 0.04,
                "reference_speaker_embedding": [1, 0],
                "speaker_embedding": [1, 0],
                "evidence": {
                    "asr": {"extractor": "asr", "revision": "1"},
                    "speaker_encoder": {"extractor": "speaker", "revision": "1"},
                    "runtime": {"extractor": "runtime", "revision": "1"},
                },
            },
        ]
        result = score_objective_observations(rows)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["asr_word_error_rate"]["count"], 1)
        self.assertEqual(candidate["speaker_embedding_similarity"]["count"], 1)
        invalid = next(sample for sample in result["samples"] if sample["sample_id"] == "invalid")
        self.assertIn("asr_word_error_rate", invalid["excluded_quality_metrics"])
        self.assertEqual(invalid["diagnostics"]["invalid_audio_duration_seconds"], 0.04)

    def test_rejects_partial_artifact_binding(self) -> None:
        row = {
            "sample_id": "sample-1",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "requested_text": "hello",
            "valid": True,
            "artifact_set_id": "voice-1",
        }
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            score_objective_observations([row])

    def test_rejects_out_of_range_audio_probe_metrics(self) -> None:
        row = {
            "sample_id": "sample-1",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "requested_text": "hello",
            "valid": True,
            "sample_rate_hz": 24000.5,
            "silence_fraction": 1.1,
            "evidence": {"audio_probe": {"extractor": "probe", "revision": "1"}},
        }
        with self.assertRaisesRegex(ValueError, "sample_rate_hz must be a positive integer"):
            score_objective_observations([row])

    def test_reports_and_validates_content_bound_metric_evidence(self) -> None:
        row = {
            "sample_id": "sample-1",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "requested_text": "hello",
            "hypothesis_text": "hello",
            "valid": True,
            "audio_sha256": "a" * 64,
            "evidence": {
                "asr": {
                    "extractor": "asr",
                    "revision": "1",
                    "input_audio_sha256": "a" * 64,
                    "extractor_artifact_set_sha256": "c" * 64,
                }
            },
        }
        result = score_objective_observations([row])
        self.assertTrue(result["metric_provenance"]["asr"]["all_content_bound"])
        row["evidence"]["asr"]["input_audio_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "must match audio_sha256"):
            score_objective_observations([row])

    def test_speaker_metric_is_unbound_without_complete_reference_identity(self) -> None:
        row = {
            "sample_id": "sample-1",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "requested_text": "hello",
            "valid": True,
            "audio_sha256": "a" * 64,
            "reference_speaker_embedding": [1, 0],
            "speaker_embedding": [1, 0],
            "evidence": {
                "speaker_encoder": {
                    "extractor": "speaker",
                    "revision": "1",
                    "input_audio_sha256": "a" * 64,
                    "extractor_artifact_set_sha256": "b" * 64,
                }
            },
        }
        result = score_objective_observations([row])
        self.assertFalse(result["metric_provenance"]["speaker_encoder"]["all_content_bound"])
        row["evidence"]["speaker_encoder"].update(
            {
                "reference_id": "voice-1",
                "reference_audio_sha256": "c" * 64,
                "reference_transcript_sha256": "d" * 64,
            }
        )
        result = score_objective_observations([row])
        self.assertFalse(result["metric_provenance"]["speaker_encoder"]["all_content_bound"])
        row["evidence"]["speaker_encoder"]["speaker_measurement_sha256"] = speaker_measurement_sha256(
            row,
            row["evidence"]["speaker_encoder"],
        )
        result = score_objective_observations([row])
        self.assertTrue(result["metric_provenance"]["speaker_encoder"]["all_content_bound"])

    def test_rejects_partial_speaker_reference_identity(self) -> None:
        row = {
            "sample_id": "sample-1",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "requested_text": "hello",
            "valid": True,
            "reference_speaker_embedding": [1, 0],
            "speaker_embedding": [1, 0],
            "evidence": {
                "speaker_encoder": {
                    "extractor": "speaker",
                    "revision": "1",
                    "reference_id": "voice-1",
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "reference binding must be complete"):
            score_objective_observations([row])

    def test_scores_each_bound_reference_then_uses_fixed_mean(self) -> None:
        references = [
            {
                "reference_id": "phone",
                "reference_audio_sha256": "c" * 64,
                "reference_transcript_sha256": "d" * 64,
            },
            {
                "reference_id": "studio",
                "reference_audio_sha256": "e" * 64,
                "reference_transcript_sha256": "f" * 64,
            },
        ]
        aggregation = "mean_cosine_similarity_v1"
        row = {
            "sample_id": "sample-1",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "requested_text": "hello",
            "valid": True,
            "audio_sha256": "a" * 64,
            "reference_speaker_embeddings": [
                {"reference_id": "phone", "embedding": [0.0, 1.0]},
                {"reference_id": "studio", "embedding": [1.0, 0.0]},
            ],
            "speaker_embedding": [1.0, 0.0],
            "evidence": {
                "speaker_encoder": {
                    "extractor": "speaker",
                    "revision": "1",
                    "input_audio_sha256": "a" * 64,
                    "extractor_artifact_set_sha256": "b" * 64,
                    "reference_aggregation": aggregation,
                    "reference_set_sha256": reference_set_sha256(references, aggregation=aggregation),
                    "references": references,
                }
            },
        }
        row["evidence"]["speaker_encoder"]["speaker_measurement_sha256"] = speaker_measurement_sha256(
            row,
            row["evidence"]["speaker_encoder"],
        )
        result = score_objective_observations([row])
        sample = result["samples"][0]
        self.assertAlmostEqual(sample["metrics"]["speaker_embedding_similarity"], 0.5)
        self.assertEqual(
            [item["cosine_similarity"] for item in sample["diagnostics"]["speaker_reference_scores"]],
            [0.0, 1.0],
        )
        self.assertEqual(result["metric_provenance"]["speaker_encoder"]["reference_set_count"], 1)
        self.assertTrue(result["metric_provenance"]["speaker_encoder"]["all_content_bound"])
        row["reference_speaker_embeddings"][1]["embedding"] = [1.0, 1.0]
        with self.assertRaisesRegex(ValueError, "does not match the speaker embeddings"):
            score_objective_observations([row])

    def test_rejects_reference_embedding_or_set_digest_substitution(self) -> None:
        references = [
            {
                "reference_id": "studio",
                "reference_audio_sha256": "c" * 64,
                "reference_transcript_sha256": "d" * 64,
            }
        ]
        row = {
            "sample_id": "sample-1",
            "candidate_id": "adapter",
            "prompt_id": "p1",
            "requested_text": "hello",
            "valid": True,
            "audio_sha256": "a" * 64,
            "reference_speaker_embeddings": [{"reference_id": "other", "embedding": [1.0, 0.0]}],
            "speaker_embedding": [1.0, 0.0],
            "evidence": {
                "speaker_encoder": {
                    "extractor": "speaker",
                    "revision": "1",
                    "input_audio_sha256": "a" * 64,
                    "extractor_artifact_set_sha256": "b" * 64,
                    "reference_aggregation": "mean_cosine_similarity_v1",
                    "reference_set_sha256": reference_set_sha256(
                        references,
                        aggregation="mean_cosine_similarity_v1",
                    ),
                    "references": references,
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "exactly match the bound sorted reference set"):
            score_objective_observations([row])
        row["reference_speaker_embeddings"][0]["reference_id"] = "studio"
        row["evidence"]["speaker_encoder"]["reference_set_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match its reference records"):
            score_objective_observations([row])


if __name__ == "__main__":
    unittest.main()
