from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

import instavar_voice_lab.historical_prosody as historical_prosody_module
from instavar_voice_lab.cli import main
from instavar_voice_lab.historical_prosody import (
    HISTORICAL_PROSODY_MANIFEST_SCHEMA_VERSION,
    HISTORICAL_PROSODY_REPORT_SCHEMA_VERSION,
    audit_historical_prosody_batch,
)


class HistoricalProsodyTests(unittest.TestCase):
    @staticmethod
    def _write_tone(path: Path, *, duration: float = 1.0, amplitude: float = 0.4) -> None:
        sample_rate = 16000
        samples = [
            round(amplitude * 32767 * math.sin(2 * math.pi * 180 * index / sample_rate))
            for index in range(round(duration * sample_rate))
        ]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _manifest(self, audio: Path) -> dict:
        return {
            "schema_version": HISTORICAL_PROSODY_MANIFEST_SCHEMA_VERSION,
            "batch_id": "legacy-neutral-brief",
            "purpose": "historical_unmatched_triage",
            "samples": [
                {
                    "sample_id": "legacy-base-neutral",
                    "candidate_id": "legacy-base",
                    "prompt_id": "neutral-brief",
                    "audio_path": audio.name,
                    "audio_sha256": self._sha256(audio),
                    "requested_text": "A historical passage.",
                    "seed": None,
                    "runtime_id": None,
                }
            ],
        }

    def test_audits_content_bound_batch_and_preserves_unknowns(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            self._write_tone(audio)
            report = audit_historical_prosody_batch(
                self._manifest(audio),
                audio_base_dir=root,
                extractor_revision="historical-prosody-1",
            )
            self.assertEqual(report["schema_version"], HISTORICAL_PROSODY_REPORT_SCHEMA_VERSION)
            self.assertEqual(report["coverage"]["sample_count"], 1)
            self.assertEqual(report["coverage"]["seed_not_recorded_count"], 1)
            self.assertEqual(report["coverage"]["runtime_not_recorded_count"], 1)
            self.assertEqual(report["coverage"]["long_form_eligible_count"], 0)
            self.assertEqual(len(report["extractor"]["artifacts"]), 3)
            self.assertIsNone(report["results"][0]["seed"])
            self.assertFalse(report["eligible_for_matched_adaptation_comparison"])
            self.assertIsNone(report["winner"])
            self.assertFalse(report["proves_adaptation_benefit"])

    def test_preserves_probe_failure_as_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "silence.wav"
            self._write_tone(audio, amplitude=0.0)
            report = audit_historical_prosody_batch(
                self._manifest(audio),
                audio_base_dir=root,
                extractor_revision="historical-prosody-1",
            )
            self.assertEqual(report["status"], "analysis_complete_with_failures")
            self.assertEqual(report["coverage"]["failed_count"], 1)
            self.assertEqual(report["results"][0]["status"], "failed")
            self.assertIn("five active frames", report["results"][0]["error"])

    def test_rejects_hash_drift_escape_symlinks_and_duplicate_ids(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            self._write_tone(audio)
            manifest = self._manifest(audio)
            drifted = deepcopy(manifest)
            drifted["samples"][0]["audio_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "does not match the live audio"):
                audit_historical_prosody_batch(
                    drifted,
                    audio_base_dir=root,
                    extractor_revision="historical-prosody-1",
                )
            escaped = deepcopy(manifest)
            escaped["samples"][0]["audio_path"] = "../sample.wav"
            with self.assertRaisesRegex(ValueError, "contained relative path"):
                audit_historical_prosody_batch(
                    escaped,
                    audio_base_dir=root,
                    extractor_revision="historical-prosody-1",
                )
            link = root / "linked.wav"
            link.symlink_to(audio)
            linked = deepcopy(manifest)
            linked["samples"][0]["audio_path"] = link.name
            with self.assertRaisesRegex(ValueError, "must not traverse symlinks"):
                audit_historical_prosody_batch(
                    linked,
                    audio_base_dir=root,
                    extractor_revision="historical-prosody-1",
                )
            duplicated = deepcopy(manifest)
            duplicated["samples"].append(deepcopy(duplicated["samples"][0]))
            with self.assertRaisesRegex(ValueError, "duplicate historical"):
                audit_historical_prosody_batch(
                    duplicated,
                    audio_base_dir=root,
                    extractor_revision="historical-prosody-1",
                )

    def test_rejects_implicit_unknowns_and_mutation_during_analysis(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            self._write_tone(audio)
            manifest = self._manifest(audio)
            implicit = deepcopy(manifest)
            del implicit["samples"][0]["seed"]
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                audit_historical_prosody_batch(
                    implicit,
                    audio_base_dir=root,
                    extractor_revision="historical-prosody-1",
                )
            expected = manifest["samples"][0]["audio_sha256"]
            with patch(
                "instavar_voice_lab.historical_prosody._file_sha256",
                side_effect=[expected, "f" * 64],
            ):
                with self.assertRaisesRegex(ValueError, "changed during analysis"):
                    audit_historical_prosody_batch(
                        manifest,
                        audio_base_dir=root,
                        extractor_revision="historical-prosody-1",
                    )

    def test_rejects_extractor_source_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            self._write_tone(audio)
            manifest = self._manifest(audio)
            original = historical_prosody_module.build_extractor_identity
            first = original(
                kind="prosody_proxy",
                name="instavar_voice_lab.prosody_probe",
                revision="historical-prosody-1",
                artifacts=historical_prosody_module._historical_prosody_artifacts(),
            )
            second = deepcopy(first)
            second["artifact_set_sha256"] = "f" * 64
            with patch(
                "instavar_voice_lab.historical_prosody.build_extractor_identity",
                side_effect=[first, second],
            ):
                with self.assertRaisesRegex(ValueError, "extractor artifacts changed"):
                    audit_historical_prosody_batch(
                        manifest,
                        audio_base_dir=root,
                        extractor_revision="historical-prosody-1",
                    )

    def test_cli_writes_report(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            self._write_tone(audio)
            manifest_path.write_text(json.dumps(self._manifest(audio)), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "audit-historical-prosody",
                        str(manifest_path),
                        "--audio-base-dir",
                        str(root),
                        "--extractor-revision",
                        "historical-prosody-1",
                        "--output",
                        str(report_path),
                    ]
                ),
                0,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["batch_id"], "legacy-neutral-brief")


if __name__ == "__main__":
    unittest.main()
