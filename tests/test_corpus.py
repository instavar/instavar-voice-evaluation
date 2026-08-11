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
            audio.write_bytes(b"fixture")
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


if __name__ == "__main__":
    unittest.main()
