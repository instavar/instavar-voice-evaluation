from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from instavar_voice_lab.corpus import (
    DEFAULT_MAX_MANIFEST_BYTES,
    _StableManifestSource,
    audit_corpus,
)


class CorpusAuditTests(unittest.TestCase):
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
            self.assertEqual(result["schema_version"], "1.1.0")
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


if __name__ == "__main__":
    unittest.main()
