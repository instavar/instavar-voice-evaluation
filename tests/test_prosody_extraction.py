from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import wave

from instavar_voice_lab.cli import main
from instavar_voice_lab.extraction import (
    PROSODY_EXTRACTION_SCHEMA_VERSION,
    apply_extractor_results,
    build_prosody_probe_results,
)
from instavar_voice_lab.prosody_probe import PROSODY_OBSERVATION_FIELDS


class ProsodyExtractionTests(unittest.TestCase):
    @staticmethod
    def _write_pattern(path: Path, pattern: list[tuple[float, float]], sample_rate: int = 16000) -> None:
        samples: list[int] = []
        phase = 0
        for duration, amplitude in pattern:
            for _ in range(round(duration * sample_rate)):
                samples.append(round(amplitude * 32767 * math.sin(2 * math.pi * 180 * phase / sample_rate)))
                phase += 1
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _observations(self, audio: Path) -> list[dict]:
        return [
            {
                "observation_schema_version": "1.0.0",
                "sample_id": "adapter--cadence--seed-42",
                "candidate_id": "adapter",
                "prompt_id": "cadence",
                "seed": 42,
                "requested_text": "A matched long-form passage.",
                "valid": True,
                "runtime_id": "pytorch",
                "audio_path": audio.name,
                "audio_sha256": self._sha256(audio),
            }
        ]

    def test_builds_and_applies_content_bound_prosody_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "candidate.wav"
            self._write_pattern(audio, [(0.5, 0.3), (0.2, 0.0), (0.8, 0.5)])
            observations = self._observations(audio)
            results = build_prosody_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="bdefbd76101653b2d05cc97d863f66b421316652",
            )
            self.assertEqual(results["schema_version"], PROSODY_EXTRACTION_SCHEMA_VERSION)
            self.assertEqual(results["extractor"]["kind"], "prosody_proxy")
            self.assertEqual(len(results["extractor"]["artifacts"]), 2)
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            self.assertEqual(set(augmented[0]) & PROSODY_OBSERVATION_FIELDS, PROSODY_OBSERVATION_FIELDS)
            evidence = augmented[0]["evidence"]["prosody_proxy"]
            self.assertEqual(evidence["input_audio_sha256"], observations[0]["audio_sha256"])
            self.assertRegex(evidence["extractor_artifact_set_sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("prosody_pause_fraction", observations[0])

    def test_silence_is_preserved_as_extractor_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "silent.wav"
            self._write_pattern(audio, [(1.0, 0.0)])
            observations = self._observations(audio)
            results = build_prosody_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="prosody-1",
            )
            self.assertEqual(results["results"][0]["status"], "failed")
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            self.assertIn("prosody_proxy", augmented[0]["extractor_failures"])
            self.assertFalse(set(augmented[0]) & PROSODY_OBSERVATION_FIELDS)

    def test_rejects_audio_and_proxy_value_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "candidate.wav"
            self._write_pattern(audio, [(1.0, 0.4)])
            observations = self._observations(audio)
            results = build_prosody_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="prosody-1",
            )
            changed = deepcopy(results)
            changed["results"][0]["values"]["prosody_pause_fraction"] = 2.0
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                apply_extractor_results(observations, changed, audio_base_dir=root)
            self._write_pattern(audio, [(1.0, 0.2)])
            with self.assertRaisesRegex(ValueError, "does not match the live audio file"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_schema_1_6_is_reserved_for_prosody_proxy(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "candidate.wav"
            self._write_pattern(audio, [(1.0, 0.4)])
            observations = self._observations(audio)
            results = build_prosody_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="prosody-1",
            )
            results["extractor"]["kind"] = "audio_probe"
            with self.assertRaisesRegex(ValueError, "reserved for prosody"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_prosody_proxy_rejects_legacy_schema_downgrade(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "candidate.wav"
            self._write_pattern(audio, [(1.0, 0.4)])
            observations = self._observations(audio)
            results = build_prosody_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="prosody-1",
            )
            results["schema_version"] = "1.1.0"
            with self.assertRaisesRegex(ValueError, "requires extractor schema 1.6"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_cli_builds_and_applies_prosody_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "candidate.wav"
            source = root / "observations.json"
            receipt = root / "prosody-results.json"
            output = root / "augmented.json"
            self._write_pattern(audio, [(0.5, 0.3), (0.2, 0.0), (0.8, 0.5)])
            source.write_text(json.dumps(self._observations(audio)), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-prosody-proxy-results",
                        str(source),
                        "--audio-base-dir",
                        str(root),
                        "--extractor-revision",
                        "prosody-1",
                        "--output",
                        str(receipt),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "apply-extractor-results",
                        str(source),
                        str(receipt),
                        "--audio-base-dir",
                        str(root),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            augmented = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("prosody_pause_fraction", augmented[0])
            self.assertIn("prosody_proxy", augmented[0]["evidence"])


if __name__ == "__main__":
    unittest.main()
