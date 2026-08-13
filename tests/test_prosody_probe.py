import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

from instavar_voice_lab.prosody_probe import (
    PROSODY_PROXY_SCHEMA_VERSION,
    compare_prosody_proxies,
    probe_prosody_proxy,
)


def _write_pattern(path: Path, pattern: list[tuple[float, float]], sample_rate: int = 16000) -> None:
    samples: list[int] = []
    phase = 0
    for duration, amplitude in pattern:
        count = round(duration * sample_rate)
        for _ in range(count):
            value = amplitude * math.sin(2 * math.pi * 180 * phase / sample_rate)
            samples.append(round(value * 32767))
            phase += 1
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class ProsodyProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_detects_pause_and_phrase_variation(self) -> None:
        path = self.root / "varied.wav"
        _write_pattern(path, [(0.5, 0.3), (0.2, 0.0), (0.9, 0.6), (0.4, 0.0), (0.4, 0.2)])
        result = probe_prosody_proxy(path)
        metrics = result["metrics"]
        self.assertEqual(result["schema_version"], PROSODY_PROXY_SCHEMA_VERSION)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(metrics["sample_rate_hz"], 16000)
        self.assertEqual(metrics["pause_count"], 2)
        self.assertGreater(metrics["pause_duration_cv"], 0)
        self.assertGreater(metrics["phrase_duration_cv"], 0)
        self.assertFalse(result["claims"]["proves_long_form_monotony"])

    def test_uniform_signal_has_no_detected_pause(self) -> None:
        path = self.root / "uniform.wav"
        _write_pattern(path, [(1.0, 0.4)])
        metrics = probe_prosody_proxy(path)["metrics"]
        self.assertEqual(metrics["pause_count"], 0)
        self.assertEqual(metrics["pause_fraction"], 0)
        self.assertIsNone(metrics["pause_duration_cv"])

    def test_boundary_padding_is_not_an_internal_pause(self) -> None:
        path = self.root / "padded.wav"
        _write_pattern(path, [(0.3, 0.0), (0.8, 0.4), (0.4, 0.0)])
        metrics = probe_prosody_proxy(path)["metrics"]
        self.assertEqual(metrics["pause_count"], 0)
        self.assertGreater(metrics["leading_inactive_seconds"], 0.2)
        self.assertGreater(metrics["trailing_inactive_seconds"], 0.3)

    def test_silence_reports_insufficient_activity(self) -> None:
        path = self.root / "silent.wav"
        _write_pattern(path, [(1.0, 0.0)])
        result = probe_prosody_proxy(path)
        self.assertEqual(result["status"], "insufficient_activity")
        self.assertEqual(result["metrics"]["active_frame_count"], 0)

    def test_long_form_eligibility_is_explicit(self) -> None:
        path = self.root / "short.wav"
        _write_pattern(path, [(1.0, 0.4)])
        self.assertFalse(probe_prosody_proxy(path)["metrics"]["eligible_for_long_form"])
        self.assertTrue(
            probe_prosody_proxy(path, long_form_seconds=0.5)["metrics"]["eligible_for_long_form"]
        )

    def test_probe_is_deterministic(self) -> None:
        path = self.root / "same.wav"
        _write_pattern(path, [(0.5, 0.2), (0.2, 0.0), (0.5, 0.4)])
        self.assertEqual(probe_prosody_proxy(path), probe_prosody_proxy(path))

    def test_streaming_crosses_one_second_chunk_boundaries(self) -> None:
        path = self.root / "chunked.wav"
        _write_pattern(path, [(0.95, 0.3), (0.2, 0.0), (0.95, 0.3)])
        result = probe_prosody_proxy(path)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["metrics"]["pause_count"], 1)

    def test_comparison_emits_only_proxy_deltas(self) -> None:
        reference = self.root / "reference.wav"
        candidate = self.root / "candidate.wav"
        _write_pattern(reference, [(0.5, 0.2), (0.2, 0.0), (0.5, 0.4)])
        _write_pattern(candidate, [(0.3, 0.4), (0.4, 0.0), (0.5, 0.4)])
        result = compare_prosody_proxies(reference, candidate)
        self.assertIn("pause_fraction", result["candidate_minus_reference"])
        self.assertFalse(result["claims"]["proves_accent_fidelity"])
        self.assertFalse(result["claims"]["proves_cadence_quality"])
        self.assertFalse(result["claims"]["proves_matched_text"])
        self.assertTrue(result["format_match"]["sample_rate_hz"])

    def test_comparison_rejects_silent_input(self) -> None:
        reference = self.root / "reference.wav"
        candidate = self.root / "candidate.wav"
        _write_pattern(reference, [(1.0, 0.0)])
        _write_pattern(candidate, [(1.0, 0.4)])
        with self.assertRaisesRegex(ValueError, "five active frames"):
            compare_prosody_proxies(reference, candidate)

    def test_comparison_exposes_sample_rate_mismatch(self) -> None:
        reference = self.root / "reference.wav"
        candidate = self.root / "candidate.wav"
        _write_pattern(reference, [(1.0, 0.4)], sample_rate=16000)
        _write_pattern(candidate, [(1.0, 0.4)], sample_rate=8000)
        result = compare_prosody_proxies(reference, candidate)
        self.assertFalse(result["format_match"]["sample_rate_hz"])

    def test_rejects_stereo_and_invalid_configuration(self) -> None:
        path = self.root / "stereo.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<400h", *([1] * 400)))
        with self.assertRaisesRegex(ValueError, "mono"):
            probe_prosody_proxy(path)
        mono = self.root / "mono.wav"
        _write_pattern(mono, [(1.0, 0.4)])
        with self.assertRaisesRegex(ValueError, "hop_ms"):
            probe_prosody_proxy(mono, frame_ms=20, hop_ms=40)

    def test_cli_probe_and_compare(self) -> None:
        reference = self.root / "reference.wav"
        candidate = self.root / "candidate.wav"
        _write_pattern(reference, [(0.5, 0.2), (0.2, 0.0), (0.5, 0.4)])
        _write_pattern(candidate, [(0.4, 0.4), (0.3, 0.0), (0.5, 0.3)])
        probe_output = self.root / "probe.json"
        compare_output = self.root / "compare.json"
        subprocess.run(
            [sys.executable, "-m", "instavar_voice_lab.cli", "probe-prosody", str(reference), "--output", str(probe_output)],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "instavar_voice_lab.cli",
                "compare-prosody",
                str(reference),
                str(candidate),
                "--output",
                str(compare_output),
            ],
            check=True,
        )
        self.assertIn(PROSODY_PROXY_SCHEMA_VERSION, probe_output.read_text())
        self.assertIn("candidate_minus_reference", compare_output.read_text())


if __name__ == "__main__":
    unittest.main()
