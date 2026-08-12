from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.runtime_artifacts import (
    build_runtime_artifact_manifest,
    exact_runtime_binding,
    validate_runtime_artifact_manifest,
    verify_runtime_artifact_manifest,
)


REVISION = "a" * 40


def binding_plan(*, derived: bool = False) -> dict:
    candidate_binding = {
        "runtime_id": "candidate",
        "relation": "derived" if derived else "exact",
        "artifacts": [{"role": "model", "kind": "file", "path": "model.bin"}],
    }
    if derived:
        candidate_binding["artifacts"][0]["path"] = "converted.bin"
        candidate_binding["conversion"] = {"tool": "converter", "revision": "1.0"}
    return {
        "artifact_set_id": "voice-checkpoint-1",
        "producer": {"repository": "instavar/test", "revision": REVISION},
        "source_artifacts": [{"role": "model", "kind": "file", "path": "model.bin"}],
        "runtime_bindings": [
            {
                "runtime_id": "reference",
                "relation": "exact",
                "artifacts": [{"role": "model", "kind": "file", "path": "model.bin"}],
            },
            candidate_binding,
        ],
    }


class RuntimeArtifactTests(unittest.TestCase):
    def test_cli_builds_and_verifies_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.bin").write_bytes(b"source")
            plan_path = root / "plan.json"
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            plan_path.write_text(json.dumps(binding_plan()), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-runtime-artifact-manifest",
                        str(plan_path),
                        "--output",
                        str(manifest_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "verify-runtime-artifact-manifest",
                        str(manifest_path),
                        str(plan_path),
                        "--report",
                        str(report_path),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["status"], "passed")

    def test_builds_and_verifies_exact_and_derived_bindings(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.bin").write_bytes(b"source")
            (root / "converted.bin").write_bytes(b"converted")
            plan = binding_plan(derived=True)
            manifest = build_runtime_artifact_manifest(plan, base_dir=root)
            self.assertEqual(validate_runtime_artifact_manifest(manifest), [])
            result = verify_runtime_artifact_manifest(manifest, plan, base_dir=root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["exact_runtime_ids"], ["reference"])
            self.assertEqual(result["derived_runtime_ids"], ["candidate"])
            with self.assertRaisesRegex(ValueError, "non-exact runtimes"):
                exact_runtime_binding(manifest, {"reference", "candidate"})

    def test_exact_binding_must_match_source_roles_and_content(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.bin").write_bytes(b"source")
            (root / "other.bin").write_bytes(b"other")
            plan = binding_plan()
            plan["runtime_bindings"][1]["artifacts"][0]["path"] = "other.bin"
            with self.assertRaisesRegex(ValueError, "claims exact relation"):
                build_runtime_artifact_manifest(plan, base_dir=root)

    def test_empty_artifact_and_mutable_conversion_revision_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.bin").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "must be non-empty"):
                build_runtime_artifact_manifest(binding_plan(), base_dir=root)

            (root / "model.bin").write_bytes(b"source")
            (root / "converted.bin").write_bytes(b"converted")
            plan = binding_plan(derived=True)
            plan["runtime_bindings"][1]["conversion"]["revision"] = "latest"
            with self.assertRaisesRegex(ValueError, "mutable revision alias"):
                build_runtime_artifact_manifest(plan, base_dir=root)

    def test_public_identifiers_reject_paths_and_whitespace(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.bin").write_bytes(b"source")
            plan = binding_plan()
            plan["artifact_set_id"] = "/private/model"
            with self.assertRaisesRegex(ValueError, "lowercase letters"):
                build_runtime_artifact_manifest(plan, base_dir=root)

    def test_verification_rejects_artifact_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "model.bin"
            artifact.write_bytes(b"source")
            plan = binding_plan()
            manifest = build_runtime_artifact_manifest(plan, base_dir=root)
            artifact.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "does not match current"):
                verify_runtime_artifact_manifest(manifest, plan, base_dir=root)

    def test_validation_rejects_tampered_fingerprint_and_relation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.bin").write_bytes(b"source")
            manifest = build_runtime_artifact_manifest(binding_plan(), base_dir=root)
            tampered = deepcopy(manifest)
            tampered["runtime_bindings"][1]["artifacts"][0]["sha256"] = "b" * 64
            errors = validate_runtime_artifact_manifest(tampered)
            self.assertTrue(any("artifact_set_sha256 does not match" in error for error in errors))
            self.assertTrue(any("claims exact relation" in error for error in errors))

    def test_symlinked_artifact_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.bin"
            target.write_bytes(b"source")
            (root / "model.bin").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                build_runtime_artifact_manifest(binding_plan(), base_dir=root)


if __name__ == "__main__":
    unittest.main()
