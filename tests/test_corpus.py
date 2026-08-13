from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from instavar_voice_lab.corpus import audit_corpus


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
            self.assertTrue(result["grouped_split_verified"])

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
            self.assertTrue(
                any("audio content duplicates train:1" in error for error in result["errors"])
            )

    def test_nfkc_equivalent_transcripts_emit_duplicate_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.fixture(root)
            train = json.loads(splits["train"].read_text(encoding="utf-8"))
            validation = json.loads(splits["validation"].read_text(encoding="utf-8"))
            train["text"] = "ＡＢＣ voice sample"
            validation["text"] = "ABC voice sample"
            splits["train"].write_text(json.dumps(train) + "\n", encoding="utf-8")
            splits["validation"].write_text(
                json.dumps(validation) + "\n", encoding="utf-8"
            )

            result = audit_corpus(splits, group_field="recording_id")

            self.assertEqual(result["status"], "passed")
            self.assertTrue(
                any("text duplicates train:1" in warning for warning in result["warnings"])
            )


if __name__ == "__main__":
    unittest.main()
