from __future__ import annotations

import unittest

from instavar_voice_lab.listening import build_blind_pack


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


if __name__ == "__main__":
    unittest.main()
