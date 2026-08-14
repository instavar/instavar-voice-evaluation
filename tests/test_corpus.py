from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from instavar_voice_lab.corpus import (
    DEFAULT_MAX_MANIFEST_BYTES,
    _StableManifestSource,
    audit_corpus,
)
from instavar_voice_lab.pcm_similarity import fingerprint_pcm_wav


class CorpusAuditTests(unittest.TestCase):
    def write_pcm_wav(
        self,
        path: Path,
        *,
        sample_rate: int,
        frequency: float,
        scale: float = 1.0,
        leading_silence: float = 0.0,
        trailing_silence: float = 0.0,
        sample_width: int = 2,
        envelope: tuple[float, ...] = (0.2, 0.8, 0.4, 1.0, 0.35, 0.7, 0.25, 0.9),
    ) -> None:
        samples: list[int] = [0] * round(sample_rate * leading_silence)
        active_frames = sample_rate * 2
        for index in range(active_frames):
            envelope_index = min(len(envelope) - 1, index * len(envelope) // active_frames)
            amplitude = 20_000 * scale * envelope[envelope_index]
            samples.append(round(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate)))
        samples.extend([0] * round(sample_rate * trailing_silence))
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(sample_width)
            target.setframerate(sample_rate)
            if sample_width == 1:
                target.writeframes(bytes(max(0, min(255, round(sample / 256 + 128))) for sample in samples))
            else:
                target.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    def fixture(self, root: Path, *, leak_group: bool = False) -> dict[str, Path]:
        splits: dict[str, Path] = {}
        groups = {"train": "recording-a", "validation": "recording-b", "test": "recording-c"}
        if leak_group:
            groups["test"] = groups["train"]
        for split in ("train", "validation", "test"):
            audio = root / f"{split}.wav"
            audio.write_bytes(f"fixture:{split}".encode())
            manifest = root / f"{split}.jsonl"
            row = {
                "audio": audio.name,
                "text": f"Unique text for {split}.",
                "recording_id": groups[split],
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            splits[split] = manifest
        return splits

    def test_passes_distinct_grouped_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_corpus(
                self.fixture(Path(temporary)),
                group_field="recording_id",
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["schema_version"], "1.2.0")
            self.assertTrue(result["grouped_split_verified"])
            self.assertGreater(result["splits"]["train"]["manifest_bytes"], 0)
            self.assertEqual(len(result["splits"]["train"]["manifest_sha256"]), 64)

    def test_rejects_group_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_corpus(
                self.fixture(Path(temporary), leak_group=True),
                group_field="recording_id",
            )
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("leaks from" in error for error in result["errors"]))

    def test_audio_path_group_regex_rejects_cross_convention_parent_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            train_audio = root / "0042.wav_0000000000_0000100000.wav"
            test_audio = root / "vocal_0042.wav.reformatted.wav_10.wav_0000001000_0000099000.wav"
            train_audio.write_bytes(b"train segment")
            test_audio.write_bytes(b"reformatted segment")
            train_row = json.loads(splits["train"].read_text(encoding="utf-8"))
            test_row = json.loads(splits["test"].read_text(encoding="utf-8"))
            train_row["audio"] = train_audio.name
            test_row["audio"] = test_audio.name
            splits["train"].write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            splits["test"].write_text(json.dumps(test_row) + "\n", encoding="utf-8")

            result = audit_corpus(splits, audio_path_group_regex=r"^(?:vocal_)?([0-9]{4})\.wav")

            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["grouped_split_verified"])
            self.assertTrue(any("group '0042' leaks from train:1" in error for error in result["errors"]))

    def test_audio_path_group_regex_contract_rejects_ambiguous_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            splits = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                audit_corpus(
                    splits,
                    group_field="recording_id",
                    audio_path_group_regex=r"([0-9]+)",
                )
            for pattern in ("", "no-capture", "(one)(two)", "("):
                with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, "audio_path_group_regex"):
                    audit_corpus(splits, audio_path_group_regex=pattern)

    def test_rejects_missing_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            splits = self.fixture(Path(temporary))
            splits.pop("test")
            result = audit_corpus(splits)
            self.assertEqual(result["status"], "failed")
            self.assertIn("splits must declare exactly train, validation, and test", result["errors"])

    def test_missing_group_cannot_be_reported_as_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            row = json.loads(splits["validation"].read_text(encoding="utf-8"))
            row.pop("recording_id")
            splits["validation"].write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = audit_corpus(splits, group_field="recording_id")

            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["grouped_split_verified"])

    def test_rejects_copied_audio_content_under_a_different_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            (root / "test.wav").write_bytes((root / "train.wav").read_bytes())

            result = audit_corpus(splits, group_field="recording_id")

            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("audio content duplicates train:1" in error for error in result["errors"]))

    def test_nfkc_equivalent_transcripts_emit_duplicate_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            train = json.loads(splits["train"].read_text(encoding="utf-8"))
            validation = json.loads(splits["validation"].read_text(encoding="utf-8"))
            train["text"] = "ＡＢＣ voice sample"
            validation["text"] = "ABC voice sample"
            splits["train"].write_text(json.dumps(train) + "\n", encoding="utf-8")
            splits["validation"].write_text(json.dumps(validation) + "\n", encoding="utf-8")

            result = audit_corpus(splits, group_field="recording_id")

            self.assertEqual(result["status"], "passed")
            self.assertTrue(any("text duplicates train:1" in warning for warning in result["warnings"]))

    def test_invalid_utf8_is_a_structured_manifest_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            splits["validation"].write_bytes(b'{"audio":"validation.wav","text":"bad:\xff"}\n')

            result = audit_corpus(splits, group_field="recording_id")

            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("invalid UTF-8" in error for error in result["errors"]))

    def test_oversized_manifest_line_is_rejected_without_decoding_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            row = {"audio": "test.wav", "text": "x" * 1024, "recording_id": "recording-c"}
            splits["test"].write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = audit_corpus(
                splits,
                group_field="recording_id",
                max_manifest_line_bytes=256,
            )

            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("manifest line exceeds" in error for error in result["errors"]))

    def test_oversized_manifest_is_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            splits["train"].write_bytes(b" " * 1024)

            result = audit_corpus(splits, max_manifest_bytes=512)

            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("manifest declares 1024 bytes" in error for error in result["errors"]))

    def test_manifest_replacement_during_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.jsonl"
            manifest.write_text("first\nsecond\n", encoding="utf-8")
            replacement = root / "replacement.jsonl"
            replacement.write_text("third\nfourth\n", encoding="utf-8")

            with _StableManifestSource(
                manifest,
                max_bytes=DEFAULT_MAX_MANIFEST_BYTES,
                max_line_bytes=256,
            ) as source:
                lines = source.lines()
                self.assertEqual(next(lines)[1], "first\n")
                replacement.replace(manifest)
                list(lines)
                with self.assertRaisesRegex(ValueError, "manifest file changed"):
                    source.finish()

    def test_manifest_limits_may_only_tighten_the_safety_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            splits = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                audit_corpus(splits, max_manifest_bytes=0)
            with self.assertRaisesRegex(ValueError, "safety ceiling"):
                audit_corpus(splits, max_manifest_bytes=DEFAULT_MAX_MANIFEST_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                audit_corpus(splits, check_pcm_near_duplicates=1)  # type: ignore[arg-type]

    def test_pcm_similarity_flags_level_silence_and_resampling_variant_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            self.write_pcm_wav(root / "train.wav", sample_rate=8_000, frequency=240)
            self.write_pcm_wav(root / "validation.wav", sample_rate=8_000, frequency=1_300)
            self.write_pcm_wav(
                root / "test.wav",
                sample_rate=12_000,
                frequency=240,
                scale=0.45,
                leading_silence=0.2,
                trailing_silence=0.15,
                sample_width=1,
            )

            result = audit_corpus(
                splits,
                group_field="recording_id",
                check_pcm_near_duplicates=True,
            )

            review = result["pcm_near_duplicate_review"]
            self.assertEqual(result["status"], "passed")
            self.assertEqual(review["eligible_rows"], 3)
            self.assertEqual(review["candidate_count"], 1)
            self.assertEqual(review["candidates"][0]["earlier"], {"split": "train", "line": 1})
            self.assertEqual(review["candidates"][0]["later"], {"split": "test", "line": 1})
            self.assertFalse(review["proves_duplicate_audio"])

    def test_pcm_similarity_does_not_flag_distinct_frequency_and_envelope_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            self.write_pcm_wav(root / "train.wav", sample_rate=8_000, frequency=180)
            self.write_pcm_wav(
                root / "validation.wav",
                sample_rate=8_000,
                frequency=850,
                envelope=(1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4),
            )
            self.write_pcm_wav(
                root / "test.wav",
                sample_rate=12_000,
                frequency=1_500,
                envelope=(0.1, 1.0, 0.2, 0.9, 0.3, 0.8, 0.4, 0.7),
            )

            result = audit_corpus(splits, check_pcm_near_duplicates=True)

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["pcm_near_duplicate_review"]["candidate_count"], 0)

    def test_pcm_similarity_reports_unsupported_rows_without_changing_audit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_corpus(self.fixture(Path(temporary)), check_pcm_near_duplicates=True)

            review = result["pcm_near_duplicate_review"]
            self.assertEqual(result["status"], "passed")
            self.assertEqual(review["eligible_rows"], 0)
            self.assertEqual(review["skipped_rows"], 3)
            self.assertEqual(review["candidate_count"], 0)

    def test_pcm_similarity_rejects_replacement_after_content_hash_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio.wav"
            replacement = root / "replacement.wav"
            self.write_pcm_wav(audio, sample_rate=8_000, frequency=240)
            self.write_pcm_wav(replacement, sample_rate=8_000, frequency=480)
            observed = audio.stat()
            expected = (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns)
            replacement.replace(audio)

            with self.assertRaisesRegex(ValueError, "between content hashing"):
                fingerprint_pcm_wav(audio, expected_stat_fingerprint=expected)


if __name__ == "__main__":
    unittest.main()
