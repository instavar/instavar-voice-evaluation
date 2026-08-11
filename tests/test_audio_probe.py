from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from instavar_voice_lab.audio_probe import probe_wav


class AudioProbeTests(unittest.TestCase):
    def test_probes_pcm_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            sample_rate = 24000
            samples = [int(0.5 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate)) for index in range(sample_rate)]
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            result = probe_wav(path)
            self.assertEqual(result["sample_rate_hz"], sample_rate)
            self.assertAlmostEqual(result["duration_seconds"], 1.0)
            self.assertGreater(result["rms"], 0.3)
            self.assertLess(result["clipping_fraction"], 0.001)


if __name__ == "__main__":
    unittest.main()
