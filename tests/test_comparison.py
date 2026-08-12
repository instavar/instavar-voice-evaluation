from __future__ import annotations

import unittest
from copy import deepcopy

from instavar_voice_lab.comparison import compare_matched_candidates


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


class MatchedComparisonTests(unittest.TestCase):
    def test_compares_exact_prompt_and_seed_pairs(self) -> None:
        rows = [
            observation(candidate, prompt, seed)
            for candidate in ("base", "adapter")
            for prompt in ("p1", "p2")
            for seed in (42, 314159)
        ]
        result = compare_matched_candidates(
            rows,
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
            compare_matched_candidates(rows, baseline_candidate_id="base", adapted_candidate_id="adapter")

    def test_rejects_transcript_mismatch(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        rows[1]["requested_text"] = "different text"
        with self.assertRaisesRegex(ValueError, "requested_text mismatch"):
            compare_matched_candidates(rows, baseline_candidate_id="base", adapted_candidate_id="adapter")

    def test_rejects_mixed_extractor_revisions(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42)]
        rows[1]["evidence"]["asr"]["revision"] = "rev-2"
        with self.assertRaisesRegex(ValueError, "cannot mix extractor provenance"):
            compare_matched_candidates(rows, baseline_candidate_id="base", adapted_candidate_id="adapter")

    def test_preserves_invalid_pair_in_validity_delta(self) -> None:
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42, valid=False)]
        result = compare_matched_candidates(rows, baseline_candidate_id="base", adapted_candidate_id="adapter")
        self.assertEqual(result["validity"]["adapted_invalid_count"], 1)
        self.assertEqual(result["pairs"][0]["metrics"], {})

    def test_rejects_duplicate_pair_even_with_different_sample_ids(self) -> None:
        duplicate = deepcopy(observation("adapter", "p1", 42))
        duplicate["sample_id"] = "another-id"
        rows = [observation("base", "p1", 42), observation("adapter", "p1", 42), duplicate]
        with self.assertRaisesRegex(ValueError, "duplicate matched observation"):
            compare_matched_candidates(rows, baseline_candidate_id="base", adapted_candidate_id="adapter")


if __name__ == "__main__":
    unittest.main()
