from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from instavar_voice_lab.cli import main
from instavar_voice_lab.resume import (
    CORE_STATE_ROLES,
    build_resume_run_receipt,
    compare_resume_artifacts,
)


REVISION = "a" * 40
ROOT = Path(__file__).parents[1]
DIGESTS = {
    "base_artifact_sha256": "b" * 64,
    "dataset_lineage_sha256": "c" * 64,
    "training_controls_sha256": "d" * 64,
    "initial_state_sha256": "e" * 64,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(run_id: str, mode: str, interruption_sha256: str | None = None) -> dict:
    value = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "producer_repository": "instavar/test-tts",
        "producer_revision": REVISION,
        "backend_id": "test-lora-pytorch",
        "adaptation_mode": "lora",
        **DIGESTS,
        "target_updates": 2,
        "completed_updates": 2,
        "execution_mode": mode,
    }
    if mode == "interrupted_resumed":
        value["resume"] = {
            "interruption_observed": True,
            "checkpoint_completed_updates": 1,
            "resumed_from_completed_updates": 1,
            "interruption_signal": "SIGTERM",
            "interruption_receipt_sha256": interruption_sha256,
        }
    return value


def build_fixture(root: Path) -> tuple[dict, Path]:
    interruption = root / "interruption.txt"
    interruption.write_text("observed SIGTERM after update 1\n", encoding="utf-8")
    roles = sorted(CORE_STATE_ROLES)
    for run_name in ("uninterrupted", "resumed"):
        run_dir = root / run_name
        run_dir.mkdir()
        for role in roles:
            (run_dir / f"{role}.bin").write_bytes(f"state:{role}\n".encode())
    uninterrupted_receipt = root / "uninterrupted-receipt.json"
    resumed_receipt = root / "resumed-receipt.json"
    uninterrupted_receipt.write_text(
        json.dumps(receipt("run-uninterrupted", "uninterrupted")),
        encoding="utf-8",
    )
    resumed_receipt.write_text(
        json.dumps(receipt("run-resumed", "interrupted_resumed", sha256(interruption))),
        encoding="utf-8",
    )
    plan = {
        "schema_version": "1.0.0",
        "comparison_id": "test-resume",
        "required_artifact_roles": roles,
        "interruption_receipt": interruption.name,
        "uninterrupted": {
            "receipt": uninterrupted_receipt.name,
            "artifacts": [
                {"role": role, "kind": "file", "path": f"uninterrupted/{role}.bin"}
                for role in roles
            ],
        },
        "resumed": {
            "receipt": resumed_receipt.name,
            "artifacts": [
                {"role": role, "kind": "file", "path": f"resumed/{role}.bin"}
                for role in roles
            ],
        },
    }
    return plan, root / "plan.json"


def build_live_conditioning_fixture(root: Path) -> tuple[dict, Path]:
    plan, plan_path = build_fixture(root)
    conditioning_dir = root / "conditioning"
    conditioning_dir.mkdir()
    base_dir = conditioning_dir / "base"
    base_dir.mkdir()
    (base_dir / "model.bin").write_bytes(b"base model\n")
    for name in ("dataset-lineage.json", "training-controls.json", "initial-state.bin"):
        (conditioning_dir / name).write_bytes(f"{name}\n".encode())
    identity_artifacts = {
        "base_artifact": (base_dir, "tree"),
        "dataset_lineage": (conditioning_dir / "dataset-lineage.json", "file"),
        "training_controls": (conditioning_dir / "training-controls.json", "file"),
        "initial_state": (conditioning_dir / "initial-state.bin", "file"),
    }
    interruption = root / "interruption.txt"
    uninterrupted = build_resume_run_receipt(
        run_id="run-uninterrupted",
        producer_repository="instavar/test-tts",
        producer_revision=REVISION,
        backend_id="test-lora-pytorch",
        adaptation_mode="lora",
        target_updates=2,
        completed_updates=2,
        execution_mode="uninterrupted",
        identity_artifacts=identity_artifacts,
    )
    resumed = build_resume_run_receipt(
        run_id="run-resumed",
        producer_repository="instavar/test-tts",
        producer_revision=REVISION,
        backend_id="test-lora-pytorch",
        adaptation_mode="lora",
        target_updates=2,
        completed_updates=2,
        execution_mode="interrupted_resumed",
        identity_artifacts=identity_artifacts,
        interruption_receipt=interruption,
        checkpoint_completed_updates=1,
        resumed_from_completed_updates=1,
        interruption_signal="SIGTERM",
    )
    (root / "uninterrupted-receipt.json").write_text(json.dumps(uninterrupted), encoding="utf-8")
    (root / "resumed-receipt.json").write_text(json.dumps(resumed), encoding="utf-8")
    plan["schema_version"] = "1.1.0"
    plan["conditioning_artifacts"] = [
        {"role": role, "kind": kind, "path": str(path.relative_to(root))}
        for role, (path, kind) in sorted(identity_artifacts.items())
    ]
    return plan, plan_path


class ResumeComparisonTests(unittest.TestCase):
    def test_cli_builds_byte_exact_report(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, plan_path = build_fixture(root)
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "report.json"
            self.assertEqual(
                main(["compare-resume-artifacts", str(plan_path), "--output", str(output)]),
                0,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["exact_resume_artifact_equivalence"])
            self.assertTrue(report["independent_artifact_storage_verified"])
            self.assertFalse(report["proves_numerical_resume_equivalence"])
            self.assertFalse(report["proves_model_quality"])
            report_without_digest = dict(report)
            report_digest = report_without_digest.pop("report_sha256")
            encoded = json.dumps(
                report_without_digest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(report_digest, hashlib.sha256(encoded).hexdigest())

    def test_reference_schemas_are_parseable(self) -> None:
        for filename in (
            "resume-run-receipt.schema.json",
            "resume-comparison-plan.schema.json",
            "resume-artifact-comparison.schema.json",
        ):
            with self.subTest(filename=filename):
                schema = json.loads((ROOT / "reference" / filename).read_text(encoding="utf-8"))
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_builds_and_compares_live_conditioned_receipts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, plan_path = build_live_conditioning_fixture(root)
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "report.json"
            self.assertEqual(
                main(["compare-resume-artifacts", str(plan_path), "--output", str(output)]),
                0,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["conditioning_artifacts_verified"])
            self.assertEqual(report["plan_schema_version"], "1.1.0")
            self.assertEqual(report["claim_tier"], "byte_exact_live_conditioned_artifact_set")
            self.assertEqual(len(report["conditioning_artifacts"]), 4)

    def test_cli_builds_live_conditioned_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            conditioning = root / "conditioning"
            conditioning.mkdir()
            for role in ("base_artifact", "dataset_lineage", "training_controls", "initial_state"):
                (conditioning / f"{role}.bin").write_bytes(f"{role}\n".encode())
            output = root / "receipt.json"
            arguments = [
                "build-resume-run-receipt",
                "--run-id",
                "run-uninterrupted",
                "--producer-repository",
                "instavar/test-tts",
                "--producer-revision",
                REVISION,
                "--backend-id",
                "test-lora-pytorch",
                "--adaptation-mode",
                "lora",
                "--target-updates",
                "2",
                "--completed-updates",
                "2",
                "--execution-mode",
                "uninterrupted",
            ]
            for role in ("base_artifact", "dataset_lineage", "training_controls", "initial_state"):
                arguments.extend(
                    ["--identity-artifact", f"{role}=file={conditioning / f'{role}.bin'}"]
                )
            arguments.extend(["--output", str(output)])
            self.assertEqual(main(arguments), 0)
            receipt_value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt_value["schema_version"], "1.1.0")
            self.assertEqual(len(receipt_value["identity_artifacts"]), 4)
            original_bytes = output.read_bytes()
            self.assertEqual(main(arguments), 2)
            self.assertEqual(output.read_bytes(), original_bytes)

    def test_outputs_cannot_overwrite_or_mutate_evidence_inputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, plan_path = build_live_conditioning_fixture(root)
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "report.json"
            self.assertEqual(
                main(["compare-resume-artifacts", str(plan_path), "--output", str(output)]),
                0,
            )
            original_bytes = output.read_bytes()
            self.assertEqual(
                main(["compare-resume-artifacts", str(plan_path), "--output", str(output)]),
                2,
            )
            self.assertEqual(output.read_bytes(), original_bytes)

            with self.assertRaisesRegex(ValueError, "must not overwrite or be created inside"):
                compare_resume_artifacts(
                    plan,
                    base_dir=root,
                    output_path=root / "conditioning/base/report.json",
                )

            identity = {
                row["role"]: (root / row["path"], row["kind"])
                for row in plan["conditioning_artifacts"]
            }
            with self.assertRaisesRegex(ValueError, "must not overwrite or be created inside"):
                build_resume_run_receipt(
                    run_id="run-new",
                    producer_repository="instavar/test-tts",
                    producer_revision=REVISION,
                    backend_id="test-lora-pytorch",
                    adaptation_mode="lora",
                    target_updates=2,
                    completed_updates=2,
                    execution_mode="uninterrupted",
                    identity_artifacts=identity,
                    output_path=root / "conditioning/base/run-receipt.json",
                )

    def test_live_conditioning_mutation_and_legacy_receipts_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_live_conditioning_fixture(root)
            (root / "conditioning/training-controls.json").write_bytes(b"changed controls\n")
            with self.assertRaisesRegex(ValueError, "do not match live conditioning"):
                compare_resume_artifacts(plan, base_dir=root)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            conditioning = root / "conditioning"
            conditioning.mkdir()
            for role in ("base_artifact", "dataset_lineage", "training_controls", "initial_state"):
                (conditioning / f"{role}.bin").write_bytes(f"{role}\n".encode())
            plan["schema_version"] = "1.1.0"
            plan["conditioning_artifacts"] = [
                {"role": role, "kind": "file", "path": f"conditioning/{role}.bin"}
                for role in ("base_artifact", "dataset_lineage", "training_controls", "initial_state")
            ]
            with self.assertRaisesRegex(ValueError, "receipt must use schema_version 1.1.0"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_receipt_builder_rejects_alias_and_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared.bin"
            shared.write_bytes(b"shared\n")
            identity = {
                "base_artifact": (shared, "file"),
                "dataset_lineage": (shared, "file"),
                "training_controls": (root / "controls.bin", "file"),
                "initial_state": (root / "initial.bin", "file"),
            }
            (root / "controls.bin").write_bytes(b"controls\n")
            (root / "initial.bin").write_bytes(b"initial\n")
            with self.assertRaisesRegex(ValueError, "must not share files or hardlinks"):
                build_resume_run_receipt(
                    run_id="run-uninterrupted",
                    producer_repository="instavar/test-tts",
                    producer_revision=REVISION,
                    backend_id="test-lora-pytorch",
                    adaptation_mode="lora",
                    target_updates=2,
                    completed_updates=2,
                    execution_mode="uninterrupted",
                    identity_artifacts=identity,
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {}
            for role in ("base_artifact", "dataset_lineage", "training_controls", "initial_state"):
                path = root / f"{role}.bin"
                path.write_bytes(f"{role}\n".encode())
                identity[role] = (path, "file")
            target = root / "training_controls.bin"
            from instavar_voice_lab import resume as resume_module

            original = resume_module.fingerprint_artifact
            mutated = False

            def mutate_after_first_hash(path: Path, *, role: str, kind: str):
                nonlocal mutated
                result = original(path, role=role, kind=kind)
                if Path(path) == target and role == "training_controls" and not mutated:
                    target.write_bytes(b"mutated controls\n")
                    mutated = True
                return result

            with patch("instavar_voice_lab.resume.fingerprint_artifact", side_effect=mutate_after_first_hash):
                with self.assertRaisesRegex(ValueError, "identity artifact mutated"):
                    build_resume_run_receipt(
                        run_id="run-uninterrupted",
                        producer_repository="instavar/test-tts",
                        producer_revision=REVISION,
                        backend_id="test-lora-pytorch",
                        adaptation_mode="lora",
                        target_updates=2,
                        completed_updates=2,
                        execution_mode="uninterrupted",
                        identity_artifacts=identity,
                    )

    def test_artifact_mismatch_is_retained_as_negative_result(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, plan_path = build_fixture(root)
            (root / "resumed/model_state.bin").write_bytes(b"different model state\n")
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "report.json"
            self.assertEqual(
                main(["compare-resume-artifacts", str(plan_path), "--output", str(output)]),
                1,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "negative_result")
            self.assertEqual(report["mismatched_roles"], ["model_state"])

    def test_missing_core_state_role_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            plan["required_artifact_roles"].remove("optimizer_state")
            for run_name in ("uninterrupted", "resumed"):
                plan[run_name]["artifacts"] = [
                    row for row in plan[run_name]["artifacts"] if row["role"] != "optimizer_state"
                ]
            with self.assertRaisesRegex(ValueError, "core state roles: optimizer_state"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_conditioning_drift_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            resumed_receipt = root / "resumed-receipt.json"
            value = json.loads(resumed_receipt.read_text(encoding="utf-8"))
            value["training_controls_sha256"] = "f" * 64
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conditioning mismatch: training_controls_sha256"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_incomplete_run_and_post_target_checkpoint_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            resumed_receipt = root / "resumed-receipt.json"
            value = json.loads(resumed_receipt.read_text(encoding="utf-8"))
            value["completed_updates"] = 1
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed_updates must equal target_updates"):
                compare_resume_artifacts(plan, base_dir=root)

            value["completed_updates"] = 2
            value["resume"]["checkpoint_completed_updates"] = 2
            value["resume"]["resumed_from_completed_updates"] = 2
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint must precede target_updates"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_missing_interruption_and_tampered_receipt_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            resumed_receipt = root / "resumed-receipt.json"
            value = json.loads(resumed_receipt.read_text(encoding="utf-8"))
            value["resume"]["interruption_observed"] = False
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "interruption_observed must equal true"):
                compare_resume_artifacts(plan, base_dir=root)

            value["resume"]["interruption_observed"] = True
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            (root / "interruption.txt").write_text("mutated receipt\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "interruption_receipt does not match"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_same_run_id_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            resumed_receipt = root / "resumed-receipt.json"
            value = json.loads(resumed_receipt.read_text(encoding="utf-8"))
            value["run_id"] = "run-uninterrupted"
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_id values must differ"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_shared_path_and_hardlink_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            plan["resumed"]["artifacts"][0]["path"] = plan["uninterrupted"]["artifacts"][0]["path"]
            with self.assertRaisesRegex(ValueError, "must not share files or hardlinks"):
                compare_resume_artifacts(plan, base_dir=root)

        if os.name == "posix":
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan, _ = build_fixture(root)
                role = plan["resumed"]["artifacts"][0]["role"]
                resumed_path = root / plan["resumed"]["artifacts"][0]["path"]
                resumed_path.unlink()
                os.link(root / f"uninterrupted/{role}.bin", resumed_path)
                with self.assertRaisesRegex(ValueError, "must not share files or hardlinks"):
                    compare_resume_artifacts(plan, base_dir=root)

    def test_symlink_and_role_asymmetry_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            role = plan["resumed"]["artifacts"][0]["role"]
            resumed_path = root / plan["resumed"]["artifacts"][0]["path"]
            resumed_path.unlink()
            resumed_path.symlink_to(root / f"uninterrupted/{role}.bin")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                compare_resume_artifacts(plan, base_dir=root)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            plan["resumed"]["artifacts"].pop()
            with self.assertRaisesRegex(ValueError, "artifacts role mismatch"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_kind_mismatch_and_control_file_alias_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            role = plan["resumed"]["artifacts"][0]["role"]
            state_dir = root / "resumed/state-tree"
            state_dir.mkdir()
            (state_dir / "state.bin").write_bytes((root / f"resumed/{role}.bin").read_bytes())
            plan["resumed"]["artifacts"][0]["kind"] = "tree"
            plan["resumed"]["artifacts"][0]["path"] = "resumed/state-tree"
            with self.assertRaisesRegex(ValueError, "artifact kind mismatch"):
                compare_resume_artifacts(plan, base_dir=root)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            plan["interruption_receipt"] = plan["resumed"]["artifacts"][0]["path"]
            resumed_receipt = root / "resumed-receipt.json"
            value = json.loads(resumed_receipt.read_text(encoding="utf-8"))
            value["resume"]["interruption_receipt_sha256"] = sha256(root / plan["interruption_receipt"])
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not alias"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_artifact_mutation_during_comparison_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            target = root / "uninterrupted/model_state.bin"
            from instavar_voice_lab import resume as resume_module

            original = resume_module.fingerprint_artifact
            mutated = False

            def mutate_after_first_hash(path: Path, *, role: str, kind: str):
                nonlocal mutated
                result = original(path, role=role, kind=kind)
                if Path(path) == target and role == "model_state" and not mutated:
                    target.write_bytes(b"mutated after first fingerprint\n")
                    mutated = True
                return result

            with patch("instavar_voice_lab.resume.fingerprint_artifact", side_effect=mutate_after_first_hash):
                with self.assertRaisesRegex(ValueError, "artifact mutated during comparison"):
                    compare_resume_artifacts(plan, base_dir=root)

    def test_tree_artifacts_are_supported_and_internal_hardlinks_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            for run_name in ("uninterrupted", "resumed"):
                state_dir = root / run_name / "model-tree"
                state_dir.mkdir()
                (state_dir / "a.bin").write_bytes(b"tree model state\n")
                plan[run_name]["artifacts"] = [
                    {
                        **row,
                        "kind": "tree",
                        "path": f"{run_name}/model-tree",
                    }
                    if row["role"] == "model_state"
                    else row
                    for row in plan[run_name]["artifacts"]
                ]
            self.assertTrue(compare_resume_artifacts(plan, base_dir=root)["exact_resume_artifact_equivalence"])

            alias = root / "resumed/model-tree/b.bin"
            os.link(root / "resumed/model-tree/a.bin", alias)
            with self.assertRaisesRegex(ValueError, "hardlink alias"):
                compare_resume_artifacts(plan, base_dir=root)

    def test_wrong_execution_mode_and_resume_boundary_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            uninterrupted_receipt = root / "uninterrupted-receipt.json"
            value = json.loads(uninterrupted_receipt.read_text(encoding="utf-8"))
            value["execution_mode"] = "interrupted_resumed"
            uninterrupted_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "execution_mode must equal uninterrupted"):
                compare_resume_artifacts(plan, base_dir=root)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = build_fixture(root)
            resumed_receipt = root / "resumed-receipt.json"
            value = json.loads(resumed_receipt.read_text(encoding="utf-8"))
            value["resume"]["resumed_from_completed_updates"] = 2
            resumed_receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must equal checkpoint_completed_updates"):
                compare_resume_artifacts(plan, base_dir=root)


if __name__ == "__main__":
    unittest.main()
