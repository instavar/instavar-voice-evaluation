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

from instavar_voice_lab.cli import main
from instavar_voice_lab.extraction import (
    apply_extractor_results,
    build_audio_probe_results,
    observation_document_sha256,
)


class ExtractionTests(unittest.TestCase):
    @staticmethod
    def write_tone(path: Path, *, sample_rate: int = 24000) -> None:
        samples = [
            int(0.5 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
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

    def observations(self, audio: Path) -> list[dict]:
        return [
            {
                "observation_schema_version": "1.0.0",
                "sample_id": "adapter--p1--seed-42",
                "candidate_id": "adapter",
                "prompt_id": "p1",
                "seed": 42,
                "requested_text": "hello world",
                "valid": True,
                "runtime_id": "pytorch",
                "audio_path": audio.name,
                "audio_sha256": self.sha256(audio),
            }
        ]

    def test_builds_and_applies_content_addressed_audio_probe_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="e5297b6cda702c99267a2cd95c6ebbeeedd4ecd1",
            )
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            row = augmented[0]
            self.assertEqual(row["sample_rate_hz"], 24000)
            self.assertEqual(
                row["evidence"]["audio_probe"]["input_audio_sha256"],
                row["audio_sha256"],
            )
            self.assertEqual(row["augmentation_history"][0]["status"], "complete")
            self.assertNotIn("sample_rate_hz", observations[0])

    def test_rejects_audio_mutation_after_extraction(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            self.write_tone(audio, sample_rate=16000)
            with self.assertRaisesRegex(ValueError, "does not match the live audio file"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_rejects_source_observation_mutation_and_incomplete_coverage(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            changed = deepcopy(observations)
            changed[0]["requested_text"] = "different text"
            with self.assertRaisesRegex(ValueError, "do not match the source observation"):
                apply_extractor_results(changed, results, audio_base_dir=root)
            results["results"] = []
            with self.assertRaisesRegex(ValueError, "exactly cover valid observations"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_preserves_extractor_failure_instead_of_dropping_sample(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "not-pcm.wav"
            audio.write_bytes(b"not a wav")
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            self.assertEqual(results["results"][0]["status"], "failed")
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            self.assertIn("audio_probe", augmented[0]["extractor_failures"])
            self.assertNotIn("sample_rate_hz", augmented[0])

    def test_refuses_to_overwrite_existing_metric_or_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            observations[0]["sample_rate_hz"] = 16000
            results["source_observations_sha256"] = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )["source_observations_sha256"]
            with self.assertRaisesRegex(ValueError, "would overwrite fields"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_cli_builds_and_applies_audio_probe_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            source = root / "observations.json"
            results = root / "probe-results.json"
            output = root / "augmented.json"
            self.write_tone(audio)
            source.write_text(json.dumps(self.observations(audio)), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-audio-probe-results",
                        str(source),
                        "--audio-base-dir",
                        str(root),
                        "--extractor-revision",
                        "probe-1",
                        "--output",
                        str(results),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "apply-extractor-results",
                        str(source),
                        str(results),
                        "--audio-base-dir",
                        str(root),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            augmented = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(augmented[0]["sample_rate_hz"], 24000)

    def test_applies_external_asr_and_speaker_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            source_sha = observation_document_sha256(observations)
            audio_sha = observations[0]["audio_sha256"]
            asr_results = {
                "schema_version": "1.0.0",
                "source_observations_sha256": source_sha,
                "extractor": {"kind": "asr", "name": "test-asr", "revision": "asr-1"},
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": audio_sha,
                        "status": "complete",
                        "values": {"hypothesis_text": "hello world"},
                    }
                ],
            }
            with_asr = apply_extractor_results(observations, asr_results, audio_base_dir=root)
            speaker_results = {
                "schema_version": "1.0.0",
                "source_observations_sha256": observation_document_sha256(with_asr),
                "extractor": {
                    "kind": "speaker_encoder",
                    "name": "test-speaker",
                    "revision": "speaker-1",
                },
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": audio_sha,
                        "status": "complete",
                        "values": {
                            "reference_speaker_embedding": [1.0, 0.0],
                            "speaker_embedding": [0.9, 0.1],
                        },
                    }
                ],
            }
            augmented = apply_extractor_results(with_asr, speaker_results, audio_base_dir=root)
            self.assertEqual(augmented[0]["hypothesis_text"], "hello world")
            self.assertEqual(
                augmented[0]["evidence"]["speaker_encoder"]["input_audio_sha256"],
                audio_sha,
            )

    def test_rejects_malformed_extractor_values_at_application(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            results["results"][0]["values"]["silence_fraction"] = 1.5
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_rejects_symlink_escape_and_duplicate_extractor_application(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            link = root / "link.wav"
            self.write_tone(audio)
            link.symlink_to(audio)
            observations = self.observations(audio)
            observations[0]["audio_path"] = link.name
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                build_audio_probe_results(
                    observations,
                    audio_base_dir=root,
                    extractor_revision="probe-1",
                )

            nested = root / "nested"
            nested.mkdir()
            observations[0]["audio_path"] = "../tone.wav"
            with self.assertRaisesRegex(ValueError, "escapes the audio base directory"):
                build_audio_probe_results(
                    observations,
                    audio_base_dir=nested,
                    extractor_revision="probe-1",
                )

            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            retry = build_audio_probe_results(
                augmented,
                audio_base_dir=root,
                extractor_revision="probe-2",
            )
            with self.assertRaisesRegex(ValueError, "already contains evidence.audio_probe"):
                apply_extractor_results(augmented, retry, audio_base_dir=root)


if __name__ == "__main__":
    unittest.main()
