from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from instavar_voice_lab.audio_probe import compare_wav_probes, probe_wav


class AudioProbeTests(unittest.TestCase):
    @staticmethod
    def _write_tone(path: Path, *, sample_rate: int = 24000, amplitude: float = 0.5) -> None:
        samples = [
            int(amplitude * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
            for index in range(sample_rate)
        ]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    def test_probes_pcm_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            sample_rate = 24000
            self._write_tone(path, sample_rate=sample_rate)
            result = probe_wav(path)
            self.assertEqual(result["sample_rate_hz"], sample_rate)
            self.assertAlmostEqual(result["duration_seconds"], 1.0)
            self.assertGreater(result["rms"], 0.3)
            self.assertLess(result["clipping_fraction"], 0.001)

    def test_compares_diagnostics_without_equivalence_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.wav"
            candidate = Path(temporary) / "candidate.wav"
            self._write_tone(reference, amplitude=0.5)
            self._write_tone(candidate, amplitude=0.25)

            result = compare_wav_probes(reference, candidate)

            self.assertFalse(result["proves_runtime_equivalence"])
            self.assertTrue(all(result["format_match"].values()))
            self.assertLess(result["candidate_minus_reference"]["rms"], 0)

    def test_streams_stereo_pcm_without_changing_frame_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo.wav"
            frames = 70_000
            samples = [1000, -1000] * frames
            with wave.open(str(path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(20_000)
                output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

            result = probe_wav(path)

            self.assertEqual(result["channels"], 2)
            self.assertEqual(result["frame_count"], frames)
            self.assertAlmostEqual(result["duration_seconds"], 3.5)
            self.assertAlmostEqual(result["dc_offset"], 0.0)

    def test_rejects_pcm_payload_over_explicit_resource_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            self._write_tone(path)
            with self.assertRaisesRegex(ValueError, "analysis limit"):
                probe_wav(path, max_pcm_bytes=1_000)

    def test_rejects_truncated_pcm_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            self._write_tone(path)
            path.write_bytes(path.read_bytes()[:-100])
            with self.assertRaisesRegex(ValueError, "truncated"):
                probe_wav(path)

    def test_rejects_nonfinite_thresholds_and_invalid_resource_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            self._write_tone(path)
            for threshold in (float("nan"), float("inf"), -0.1, 1.1):
                with self.subTest(silence_threshold=threshold):
                    with self.assertRaisesRegex(ValueError, "silence threshold"):
                        probe_wav(path, silence_threshold=threshold)
            for limit in (0, -1, True):
                with self.subTest(max_pcm_bytes=limit):
                    with self.assertRaisesRegex(ValueError, "max PCM bytes"):
                        probe_wav(path, max_pcm_bytes=limit)


if __name__ == "__main__":
    unittest.main()
