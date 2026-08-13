from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from instavar_voice_lab.cli import main
from instavar_voice_lab.comparison import compare_matched_prosody
from instavar_voice_lab.prosody_probe import PROSODY_OBSERVATION_FIELDS


def observation(candidate: str, prompt: str, seed: int) -> dict:
    audio_sha = ("a" if candidate == "base" else "b") * 64
    row = {
        "observation_schema_version": "1.0.0",
        "sample_id": f"{candidate}-{prompt}-{seed}",
        "candidate_id": candidate,
        "prompt_id": prompt,
        "seed": seed,
        "requested_text": "The same matched passage.",
        "valid": True,
        "runtime_id": "pytorch",
        "audio_path": f"{candidate}-{prompt}-{seed}.wav",
        "audio_sha256": audio_sha,
        "evidence": {
            "prosody_proxy": {
                "extractor": "instavar_voice_lab.prosody_probe",
                "revision": "proxy-rev-1",
                "extractor_artifact_set_sha256": "c" * 64,
                "input_audio_sha256": audio_sha,
            }
        },
        "prosody_analysis_duration_seconds": 32.0,
        "prosody_eligible_for_long_form": True,
        "prosody_active_frame_fraction": 0.8,
        "prosody_active_rms_db_std": 2.0 if candidate == "base" else 2.5,
        "prosody_window_rms_db_std": None,
        "prosody_zero_crossing_rate_hz_std": 4.0,
        "prosody_pause_rate_per_minute": 8.0,
        "prosody_pause_fraction": 0.1,
        "prosody_pause_duration_cv": None,
        "prosody_phrase_duration_cv": 0.2,
        "prosody_leading_inactive_seconds": 0.1,
        "prosody_trailing_inactive_seconds": 0.2,
    }
    assert PROSODY_OBSERVATION_FIELDS <= set(row)
    return row


def generation_plan(rows: list[dict]) -> dict:
    return {
        "schema_version": "1.0.0",
        "prompt_pack": {"id": "prosody-pack", "version": "1.0.0", "sha256": "d" * 64},
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


def prosody_failure(row: dict, *, complete: bool = True) -> dict:
    failure = {
        "extractor": "instavar_voice_lab.prosody_probe",
        "revision": "proxy-rev-1",
        "extractor_artifact_set_sha256": "c" * 64,
        "input_audio_sha256": row["audio_sha256"],
        "error_type": "ValueError",
    }
    if complete:
        failure["error"] = "quiet"
    return failure


class MatchedProsodyComparisonTests(unittest.TestCase):
    def rows(self) -> list[dict]:
        return [
            observation(candidate, prompt, seed)
            for candidate in ("base", "adapter")
            for prompt, seed in (("short", 7), ("long", 42))
        ]

    def test_compares_signed_proxies_without_a_quality_winner(self) -> None:
        rows = self.rows()
        result = compare_matched_prosody(
            rows,
            plan=generation_plan(rows),
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
            seed=11,
        )
        self.assertEqual(result["coverage"]["complete_matched_pairs"], 2)
        active_rms = next(metric for metric in result["metrics"] if metric["proxy"] == "prosody_active_rms_db_std")
        self.assertEqual(active_rms["mean_adapted_minus_baseline"], 0.5)
        nullable = next(metric for metric in result["metrics"] if metric["proxy"] == "prosody_window_rms_db_std")
        self.assertIsNone(nullable["mean_adapted_minus_baseline"])
        self.assertEqual(nullable["null_delta_pair_count"], 2)
        self.assertFalse(result["quality_direction_established"])
        self.assertIsNone(result["winner"])
        self.assertFalse(result["proves_adaptation_benefit"])
        self.assertNotIn("mean_directional_improvement", json.dumps(result))

    def test_records_invalid_and_failed_pairs_separately(self) -> None:
        rows = self.rows()
        rows[0]["valid"] = False
        for field in PROSODY_OBSERVATION_FIELDS:
            rows[0].pop(field)
        rows[0].pop("evidence")
        failed = rows[2]
        for field in PROSODY_OBSERVATION_FIELDS:
            failed.pop(field)
        failed.pop("evidence")
        failed["extractor_failures"] = {"prosody_proxy": prosody_failure(failed)}
        result = compare_matched_prosody(
            rows,
            plan=generation_plan(rows),
            baseline_candidate_id="base",
            adapted_candidate_id="adapter",
        )
        self.assertEqual(result["coverage"]["baseline_invalid_output"], 1)
        self.assertEqual(result["coverage"]["adapted_extractor_failed"], 1)
        self.assertEqual(result["coverage"]["complete_matched_pairs"], 1)

    def test_rejects_no_complete_pair_instead_of_empty_pass(self) -> None:
        rows = self.rows()
        for row in rows:
            for field in PROSODY_OBSERVATION_FIELDS:
                row.pop(field)
            row.pop("evidence")
            row["extractor_failures"] = {"prosody_proxy": prosody_failure(row)}
        with self.assertRaisesRegex(ValueError, "no pair with complete"):
            compare_matched_prosody(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_proxy_results_on_an_invalid_output(self) -> None:
        rows = self.rows()
        rows[0]["valid"] = False
        with self.assertRaisesRegex(ValueError, "invalid output must not contain prosody results"):
            compare_matched_prosody(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_malformed_extractor_failure(self) -> None:
        rows = self.rows()
        for field in PROSODY_OBSERVATION_FIELDS:
            rows[0].pop(field)
        rows[0].pop("evidence")
        rows[0]["extractor_failures"] = {"prosody_proxy": prosody_failure(rows[0], complete=False)}
        with self.assertRaisesRegex(ValueError, "must record error_type and error"):
            compare_matched_prosody(
                rows,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_missing_fields_and_audio_hash_drift(self) -> None:
        rows = self.rows()
        missing = deepcopy(rows)
        missing[0].pop("prosody_pause_fraction")
        with self.assertRaisesRegex(ValueError, "missing prosody proxy fields"):
            compare_matched_prosody(
                missing,
                plan=generation_plan(missing),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )
        drifted = deepcopy(rows)
        drifted[0]["evidence"]["prosody_proxy"]["input_audio_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "must match audio_sha256"):
            compare_matched_prosody(
                drifted,
                plan=generation_plan(drifted),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_mixed_provenance_and_pair_identity_drift(self) -> None:
        rows = self.rows()
        mixed = deepcopy(rows)
        mixed[-1]["evidence"]["prosody_proxy"]["revision"] = "proxy-rev-2"
        with self.assertRaisesRegex(ValueError, "cannot mix extractor provenance"):
            compare_matched_prosody(
                mixed,
                plan=generation_plan(mixed),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_rejects_candidate_specific_category_drift(self) -> None:
        rows = self.rows()
        plan = generation_plan(rows)
        plan["samples"][0]["category"] = "long_form_cadence"
        plan["samples"][2]["category"] = "pronunciation"
        with self.assertRaisesRegex(ValueError, "must use one category"):
            compare_matched_prosody(
                rows,
                plan=plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )
        text_drift = deepcopy(rows)
        text_drift[-1]["requested_text"] = "Different text."
        with self.assertRaisesRegex(ValueError, "requested_text does not match generation plan"):
            compare_matched_prosody(
                text_drift,
                plan=generation_plan(rows),
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )

    def test_cli_writes_comparison(self) -> None:
        rows = self.rows()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations_path = root / "observations.json"
            plan_path = root / "plan.json"
            output_path = root / "comparison.json"
            observations_path.write_text(json.dumps(rows), encoding="utf-8")
            plan_path.write_text(json.dumps(generation_plan(rows)), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "compare-matched-prosody",
                        str(observations_path),
                        "--plan",
                        str(plan_path),
                        "--baseline",
                        "base",
                        "--adapted",
                        "adapter",
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            self.assertIsNone(json.loads(output_path.read_text(encoding="utf-8"))["winner"])


if __name__ == "__main__":
    unittest.main()
