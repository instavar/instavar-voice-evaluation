from __future__ import annotations

import hashlib
import json
import math
import struct
import unittest
import wave
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.attempts import (
    apply_generation_attempt_receipt,
    build_generation_attempt_receipt,
    runtime_attempt_is_content_bound,
)
from instavar_voice_lab.cli import main
from instavar_voice_lab.comparison import compare_matched_candidates


class GenerationAttemptTests(unittest.TestCase):
    @staticmethod
    def write_tone(path: Path, *, frequency: int = 440, sample_rate: int = 8000) -> None:
        samples = [
            int(0.4 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(sample_rate)
        ]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fixture(self, root: Path) -> tuple[list[dict], dict]:
        rows = []
        for candidate, elapsed, frequency in (("base", 0.6, 440), ("adapter", 0.4, 550)):
            audio = root / f"{candidate}.wav"
            self.write_tone(audio, frequency=frequency)
            rows.append(
                {
                    "observation_schema_version": "1.0.0",
                    "sample_id": f"{candidate}-p1-42",
                    "candidate_id": candidate,
                    "prompt_id": "p1",
                    "seed": 42,
                    "requested_text": "hello world",
                    "valid": True,
                    "runtime_id": "pytorch",
                    "audio_path": audio.name,
                    "audio_sha256": self.sha256(audio),
                    "generation_seconds": elapsed,
                    "audio_duration_seconds": 1.0,
                    "peak_memory_bytes": 100 if candidate == "base" else 90,
                }
            )
        plan = {
            "schema_version": "1.1.0",
            "prompt_pack": {"id": "test-pack", "version": "1.0.0", "sha256": "a" * 64},
            "candidate_ids": ["base", "adapter"],
            "sample_count": 2,
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
            "required_objective_metrics": ["real_time_factor", "peak_memory_bytes"],
        }
        return rows, plan

    def bind(self, rows: list[dict], plan: dict, root: Path) -> tuple[dict, list[dict]]:
        receipt = build_generation_attempt_receipt(
            rows,
            plan=plan,
            audio_base_dir=root,
            producer_name="test-runner",
            producer_revision="abc123",
        )
        return receipt, apply_generation_attempt_receipt(
            rows,
            receipt,
            plan=plan,
            audio_base_dir=root,
        )

    def test_binds_runtime_metrics_to_plan_row_and_live_audio(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, plan = self.fixture(root)
            receipt, bound = self.bind(rows, plan, root)
            self.assertEqual(receipt["schema_version"], "1.0.0")
            self.assertTrue(runtime_attempt_is_content_bound(bound[0], index=0))
            result = compare_matched_candidates(
                bound,
                plan=plan,
                baseline_candidate_id="base",
                adapted_candidate_id="adapter",
            )
            self.assertEqual(result["status"], "passed")

    def test_rejects_runtime_metric_substitution_after_binding(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, plan = self.fixture(root)
            _, bound = self.bind(rows, plan, root)
            bound[0]["generation_seconds"], bound[1]["generation_seconds"] = (
                bound[1]["generation_seconds"],
                bound[0]["generation_seconds"],
            )
            with self.assertRaisesRegex(ValueError, "does not match generation observation content"):
                compare_matched_candidates(
                    bound,
                    plan=plan,
                    baseline_candidate_id="base",
                    adapted_candidate_id="adapter",
                )

    def test_rejects_receipt_reuse_after_plan_or_audio_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, plan = self.fixture(root)
            receipt = build_generation_attempt_receipt(
                rows,
                plan=plan,
                audio_base_dir=root,
                producer_name="test-runner",
                producer_revision="abc123",
            )
            changed_plan = deepcopy(plan)
            changed_plan["samples"][0]["text"] = "different"
            with self.assertRaisesRegex(ValueError, "does not match the generation plan"):
                apply_generation_attempt_receipt(rows, receipt, plan=changed_plan, audio_base_dir=root)
            self.write_tone(root / "base.wav", frequency=660)
            with self.assertRaisesRegex(ValueError, "does not match the live audio file"):
                apply_generation_attempt_receipt(rows, receipt, plan=plan, audio_base_dir=root)

    def test_rejects_mutable_producer_revision_and_existing_runtime_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, plan = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "immutable"):
                build_generation_attempt_receipt(
                    rows,
                    plan=plan,
                    audio_base_dir=root,
                    producer_name="test-runner",
                    producer_revision="latest",
                )
            rows[0]["evidence"] = {"runtime": {"extractor": "old", "revision": "1"}}
            with self.assertRaisesRegex(ValueError, "already contains evidence.runtime"):
                build_generation_attempt_receipt(
                    rows,
                    plan=plan,
                    audio_base_dir=root,
                    producer_name="test-runner",
                    producer_revision="abc123",
                )

    def test_external_metric_augmentation_does_not_break_attempt_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, plan = self.fixture(root)
            _, bound = self.bind(rows, plan, root)
            bound[0]["hypothesis_text"] = "hello world"
            bound[0]["evidence"]["asr"] = {
                "extractor": "asr",
                "revision": "model-1",
                "input_audio_sha256": bound[0]["audio_sha256"],
                "extractor_artifact_set_sha256": "b" * 64,
            }
            self.assertTrue(runtime_attempt_is_content_bound(bound[0], index=0))

    def test_cli_builds_and_applies_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, plan = self.fixture(root)
            observations_path = root / "observations.json"
            plan_path = root / "plan.json"
            receipt_path = root / "attempts.json"
            output_path = root / "bound.json"
            observations_path.write_text(json.dumps(rows), encoding="utf-8")
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-generation-attempt-receipt",
                        str(observations_path),
                        "--plan",
                        str(plan_path),
                        "--audio-base-dir",
                        str(root),
                        "--producer-name",
                        "test-runner",
                        "--producer-revision",
                        "abc123",
                        "--output",
                        str(receipt_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "apply-generation-attempt-receipt",
                        str(observations_path),
                        str(receipt_path),
                        "--plan",
                        str(plan_path),
                        "--audio-base-dir",
                        str(root),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            self.assertIn("attempt_sha256", json.loads(output_path.read_text())[0]["evidence"]["runtime"])
