from __future__ import annotations

import unittest

from instavar_voice_lab.listening import aggregate_listening_results, build_blind_pack


class ListeningPackTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
