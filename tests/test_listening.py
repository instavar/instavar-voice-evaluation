from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.listening import aggregate_listening_results, build_blind_pack, stage_blind_audio


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
