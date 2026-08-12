from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from instavar_voice_lab.cli import main
from instavar_voice_lab.lineage import build_dataset_lineage, validate_dataset_lineage, verify_dataset_lineage

REVISION = "a" * 40


class DatasetLineageTests(unittest.TestCase):
    def test_build_and_verify_file_and_tree_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            raw.write_text('{"id":"one"}\n', encoding="utf-8")
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "codes.bin").write_bytes(b"codes")
            document = build_dataset_lineage(
                lineage_id="example-v1",
                producer_repository="instavar/example",
                producer_revision=REVISION,
                inputs={"raw_train": (raw, "file")},
                outputs={"prepared_train": (prepared, "tree")},
            )
            self.assertEqual(validate_dataset_lineage(document), [])
            report = verify_dataset_lineage(
                document,
                producer_revision=REVISION,
                inputs={"raw_train": (raw, "file")},
                outputs={"prepared_train": (prepared, "tree")},
            )
            self.assertEqual(report["status"], "passed")

    def test_mutated_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            raw.write_text("raw\n", encoding="utf-8")
            prepared = root / "prepared.jsonl"
            prepared.write_text("prepared\n", encoding="utf-8")
            document = build_dataset_lineage(
                lineage_id="mutation-v1",
                producer_repository="instavar/example",
                producer_revision=REVISION,
                inputs={"raw_train": (raw, "file")},
                outputs={"prepared_train": (prepared, "file")},
            )
            prepared.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prepared_train"):
                verify_dataset_lineage(
                    document,
                    producer_revision=REVISION,
                    inputs={"raw_train": (raw, "file")},
                    outputs={"prepared_train": (prepared, "file")},
                )

    def test_duplicate_roles_and_wrong_revision_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.write_text("source", encoding="utf-8")
            output.write_text("output", encoding="utf-8")
            document = build_dataset_lineage(
                lineage_id="revision-v1",
                producer_repository="instavar/example",
                producer_revision=REVISION,
                inputs={"source": (source, "file")},
                outputs={"output": (output, "file")},
            )
            with self.assertRaisesRegex(ValueError, "producer revision"):
                verify_dataset_lineage(
                    document,
                    producer_revision="b" * 40,
                    inputs={"source": (source, "file")},
                    outputs={"output": (output, "file")},
                )
            duplicate = json.loads(json.dumps(document))
            duplicate["outputs"][0]["role"] = "source"
            self.assertTrue(any("unique" in error for error in validate_dataset_lineage(duplicate)))

    def test_symlinked_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("content", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                build_dataset_lineage(
                    lineage_id="symlink-v1",
                    producer_repository="instavar/example",
                    producer_revision=REVISION,
                    inputs={"source": (link, "file")},
                    outputs={"output": (target, "file")},
                )

    def test_cli_builds_and_verifies_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            receipt = root / "lineage.json"
            report = root / "report.json"
            source.write_text("source", encoding="utf-8")
            output.write_text("output", encoding="utf-8")
            shared = [
                "--producer-revision",
                REVISION,
                "--input",
                f"raw_train=file={source}",
                "--output-artifact",
                f"prepared_train=file={output}",
            ]
            self.assertEqual(
                main(
                    [
                        "build-dataset-lineage",
                        "--lineage-id",
                        "cli-v1",
                        "--producer-repository",
                        "instavar/example",
                        *shared,
                        "--receipt",
                        str(receipt),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "verify-dataset-lineage",
                        str(receipt),
                        *shared,
                        "--report",
                        str(report),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(report.read_text())["status"], "passed")


if __name__ == "__main__":
    unittest.main()
