from __future__ import annotations

import unittest

from instavar_voice_lab.metrics import cosine_similarity, score_objective_observations, word_error_rate


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
                    "reference_speaker_embedding": [1, 0],
                    "speaker_embedding": [1, 0],
                    "evidence": {
                        "asr": {"extractor": "test", "revision": "1"},
                        "speaker_encoder": {"extractor": "test", "revision": "1"},
                        "runtime": {"extractor": "test", "revision": "1"},
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
        self.assertNotIn("composite_score", result)
        self.assertFalse(result["proves_perceptual_quality"])

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


if __name__ == "__main__":
    unittest.main()
