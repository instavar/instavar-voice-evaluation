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


if __name__ == "__main__":
    unittest.main()
