from __future__ import annotations

import unittest
from copy import deepcopy

from instavar_voice_lab.comparison import compare_matched_candidates
from instavar_voice_lab.speaker_references import reference_set_sha256, speaker_measurement_sha256


def observation(candidate_id: str, prompt_id: str, seed: int, *, valid: bool = True) -> dict:
    row = {
        "sample_id": f"{candidate_id}-{prompt_id}-{seed}",
        "candidate_id": candidate_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "requested_text": "hello world",
        "valid": valid,
    }
    if valid:
        row.update(
            {
                "hypothesis_text": "hello world",
                "generation_seconds": 0.5 if candidate_id == "base" else 0.4,
                "audio_duration_seconds": 1.0,
                "peak_memory_bytes": 100,
                "reference_speaker_embedding": [1, 0],
                "speaker_embedding": [1, 0],
                "evidence": {
                    "asr": {"extractor": "asr", "revision": "rev-1"},
                    "speaker_encoder": {"extractor": "speaker", "revision": "rev-1"},
                    "runtime": {"extractor": "runtime", "revision": "rev-1"},
                },
            }
        )
    return row


def generation_plan(rows: list[dict]) -> dict:
    return {
        "schema_version": "1.0.0",
        "prompt_pack": {"id": "test-pack", "version": "1.0.0", "sha256": "a" * 64},
        "candidate_ids": sorted({row["candidate_id"] for row in rows}),
        "sample_count": len(rows),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "text": row["requested_text"],
            }
            for row in rows
        ],
        "generation_requirements": {
            "same_transcripts": True,
            "frozen_generation_settings": True,
            "record_failures_as_observations": True,
        },
    }


class MatchedComparisonTests(unittest.TestCase):
    @staticmethod
    def bind_reference_set(row: dict, reference_id: str, audio_digit: str) -> None:
        references = [
            {
                "reference_id": reference_id,
                "reference_audio_sha256": audio_digit * 64,
                "reference_transcript_sha256": "d" * 64,
            }
        ]
        del row["reference_speaker_embedding"]
        row["reference_speaker_embeddings"] = [{"reference_id": reference_id, "embedding": [1.0, 0.0]}]
        row["speaker_embedding"] = [1.0, 0.0]
        row["audio_sha256"] = "a" * 64
        row["evidence"]["speaker_encoder"]["input_audio_sha256"] = "a" * 64
        row["evidence"]["speaker_encoder"]["extractor_artifact_set_sha256"] = "b" * 64
        for field in ("reference_id", "reference_audio_sha256", "reference_transcript_sha256"):
            row["evidence"]["speaker_encoder"].pop(field, None)
        row["evidence"]["speaker_encoder"].update(
            {
                "reference_aggregation": "mean_cosine_similarity_v1",
                "reference_set_sha256": reference_set_sha256(
                    references,
                    aggregation="mean_cosine_similarity_v1",
                ),
                "references": references,
            }
        )
        row["evidence"]["speaker_encoder"]["speaker_measurement_sha256"] = speaker_measurement_sha256(
            row,
            row["evidence"]["speaker_encoder"],
        )

    def test_compares_exact_prompt_and_seed_pairs(self) -> None:
        rows = [
            observation(candidate, prompt, seed)
            for candidate in ("base", "adapter")
            for prompt in ("p1", "p2")
            for seed in (42, 314159)
        ]
        result = compare_matched_candidates(
            rows,
            plan=generation_plan(rows),
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
            seed=7,
        )
        self.assertEqual(result["pair_count"], 4)
        self.assertEqual(result["validity"]["adapted_minus_baseline_invalid_output_rate"], 0.0)
        rtf = next(metric for metric in result["metrics"] if metric["metric"] == "real_time_factor")
        self.assertGreater(rtf["mean_directional_improvement"], 0)
        self.assertFalse(result["proves_adaptation_benefit"])

    def test_rejects_missing_pair(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p2", 42)]
        with self.assertRaisesRegex(ValueError, "identical prompt and seed coverage"):
            compare_matched_candidates(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_transcript_mismatch(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        rows[1]["requested_text"] = "different text"
        with self.assertRaisesRegex(ValueError, "requested_text mismatch"):
            compare_matched_candidates(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_mixed_extractor_revisions(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        rows[1]["evidence"]["asr"]["revision"] = "rev-2"
        with self.assertRaisesRegex(ValueError, "cannot mix extractor provenance"):
            compare_matched_candidates(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_selective_metric_omission_from_valid_pair(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        del rows[1]["hypothesis_text"]
        del rows[1]["reference_speaker_embedding"]
        del rows[1]["speaker_embedding"]
        del rows[1]["evidence"]["asr"]
        del rows[1]["evidence"]["speaker_encoder"]
        with self.assertRaisesRegex(ValueError, "symmetric metric availability"):
            compare_matched_candidates(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_bilateral_omission_of_plan_required_metrics(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        for row in rows:
            del row["hypothesis_text"]
            del row["reference_speaker_embedding"]
            del row["speaker_embedding"]
            del row["evidence"]["asr"]
            del row["evidence"]["speaker_encoder"]
        bound_plan = generation_plan(rows)
        bound_plan["schema_version"] = "1.1.0"
        bound_plan["required_objective_metrics"] = [
            "asr_word_error_rate",
            "speaker_embedding_similarity",
            "invalid_output_rate",
        ]
        with self.assertRaisesRegex(ValueError, "missing plan-required metrics"):
            compare_matched_candidates(
                rows,
                plan=bound_plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_legacy_plan_reports_that_metric_coverage_is_not_enforced(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        result = compare_matched_candidates(
            rows,
            plan=generation_plan(rows),
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
        )
        self.assertFalse(result["generation_plan"]["required_metric_coverage_enforced"])

    def test_plan_required_external_metrics_must_bind_to_audio_hash(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        bound_plan = generation_plan(rows)
        bound_plan["schema_version"] = "1.1.0"
        bound_plan["required_objective_metrics"] = ["asr_word_error_rate"]
        with self.assertRaisesRegex(ValueError, "must bind required metrics to audio_sha256"):
            compare_matched_candidates(
                rows,
                plan=bound_plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

        for row in rows:
            row["audio_sha256"] = "a" * 64
            row["evidence"]["asr"]["input_audio_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "must bind extractor_artifact_set_sha256"):
            compare_matched_candidates(
                rows,
                plan=bound_plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )
        for row in rows:
            row["evidence"]["asr"]["extractor_artifact_set_sha256"] = "b" * 64
        result = compare_matched_candidates(
            rows,
            plan=bound_plan,
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
        )
        self.assertEqual(result["status"], "passed")

    def test_plan_required_speaker_metric_must_bind_reference_identity(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        bound_plan = generation_plan(rows)
        bound_plan["schema_version"] = "1.1.0"
        bound_plan["required_objective_metrics"] = ["speaker_embedding_similarity"]
        for row in rows:
            row["audio_sha256"] = "a" * 64
            row["evidence"]["speaker_encoder"].update(
                {
                    "input_audio_sha256": "a" * 64,
                    "extractor_artifact_set_sha256": "b" * 64,
                }
            )
        with self.assertRaisesRegex(ValueError, "must bind a speaker reference"):
            compare_matched_candidates(
                rows,
                plan=bound_plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )
        for row in rows:
            row["evidence"]["speaker_encoder"].update(
                {
                    "reference_id": "voice-1",
                    "reference_audio_sha256": "c" * 64,
                    "reference_transcript_sha256": "d" * 64,
                }
            )
        with self.assertRaisesRegex(ValueError, "content-addressed reference set"):
            compare_matched_candidates(
                rows,
                plan=bound_plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )
        for row in rows:
            self.bind_reference_set(row, "voice-1", "c")
        result = compare_matched_candidates(
            rows,
            plan=bound_plan,
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
        )
        self.assertEqual(result["status"], "passed")

    def test_rejects_same_extractor_revision_with_different_artifacts(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        rows[0]["evidence"]["asr"]["extractor_artifact_set_sha256"] = "a" * 64
        rows[1]["evidence"]["asr"]["extractor_artifact_set_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "cannot mix extractor provenance"):
            compare_matched_candidates(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_candidate_specific_reference_set_cherry_picking(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        self.bind_reference_set(rows[0], "studio", "c")
        self.bind_reference_set(rows[1], "phone", "e")
        with self.assertRaisesRegex(ValueError, "same reference set"):
            compare_matched_candidates(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_allows_per_prompt_reference_sets_when_each_pair_matches(self) -> None:
        rows = [observation(candidate, prompt, 42) for candidate in ("base", "adapter") for prompt in ("p1", "p2")]
        for row in rows:
            if row["prompt_id"] == "p1":
                self.bind_reference_set(row, "studio", "c")
            else:
                self.bind_reference_set(row, "phone", "e")
        result = compare_matched_candidates(
            rows,
            plan=generation_plan(rows),
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
        )
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(
            len({pair["speaker_reference_set_sha256"] for pair in result["pairs"]}),
            2,
        )

    def test_preserves_invalid_pair_in_validity_delta(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42, valid=False)]
        result = compare_matched_candidates(
            rows,
            plan=generation_plan(rows),
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
        )
        self.assertEqual(result["validity"]["adapted_invalid_count"], 1)
        self.assertEqual(result["pairs"][0]["metrics"], {})

    def test_rejects_duplicate_pair_even_with_different_sample_ids(self) -> None:
        duplicate = deepcopy(observation("adapter", "p1", 42))
        duplicate["sample_id"] = "another-id"
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42), duplicate]
        with self.assertRaisesRegex(ValueError, "duplicate matched observation"):
            compare_matched_candidates(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_observations_not_bound_to_the_plan(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        bound_plan = generation_plan(rows)
        rows[1]["sample_id"] = "unplanned-sample"
        with self.assertRaisesRegex(ValueError, "exactly cover the plan"):
            compare_matched_candidates(
                rows,
                plan=bound_plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_plan_without_frozen_generation_requirements(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        bound_plan = generation_plan(rows)
        bound_plan["generation_requirements"]["frozen_generation_settings"] = False
        with self.assertRaisesRegex(ValueError, "frozen settings"):
            compare_matched_candidates(
                rows,
                plan=bound_plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )


if __name__ == "__main__":
    unittest.main()
