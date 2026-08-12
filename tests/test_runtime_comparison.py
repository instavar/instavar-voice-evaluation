from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.comparison import compare_runtime_candidates
from instavar_voice_lab.runtime_artifacts import build_runtime_artifact_manifest


def observation(candidate_id: str, runtime_id: str, artifact_set_id: str, artifact_set_sha256: str) -> dict:
    return {
        "sample_id": f"{candidate_id}-p1-42",
        "candidate_id": candidate_id,
        "runtime_id": runtime_id,
        "artifact_set_id": artifact_set_id,
        "artifact_set_sha256": artifact_set_sha256,
        "prompt_id": "p1",
        "seed": 42,
        "requested_text": "hello world",
        "hypothesis_text": "hello world",
        "valid": True,
        "generation_seconds": 0.5,
        "audio_duration_seconds": 1.0,
        "peak_memory_bytes": 100,
        "reference_speaker_embedding": [1, 0],
        "speaker_embedding": [1, 0],
        "evidence": {
            "asr": {"extractor": "asr", "revision": "rev-1"},
            "speaker_encoder": {"extractor": "speaker", "revision": "rev-1"},
            "runtime": {"extractor": "runtime", "revision": "rev-1"},
        },
    }


def generation_plan(rows: list[dict]) -> dict:
    return {
        "schema_version": "1.0.0",
        "prompt_pack": {"id": "test-pack", "version": "1.0.0", "sha256": "a" * 64},
        "candidate_ids": [row["candidate_id"] for row in rows],
        "sample_count": len(rows),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "text": row["requested_text"],
            }
            for row in rows
        ],
        "generation_requirements": {
            "same_transcripts": True,
            "frozen_generation_settings": True,
            "record_failures_as_observations": True,
        },
    }


class RuntimeComparisonTests(unittest.TestCase):
    def fixture(self, root: Path, *, derived_candidate: bool = False) -> tuple[dict, dict, list[dict], dict]:
        (root / "model.bin").write_bytes(b"source")
        (root / "converted.bin").write_bytes(b"converted")
        candidate = {
            "runtime_id": "mlx",
            "relation": "derived" if derived_candidate else "exact",
            "artifacts": [
                {
                    "role": "model",
                    "kind": "file",
                    "path": "converted.bin" if derived_candidate else "model.bin",
                }
            ],
        }
        if derived_candidate:
            candidate["conversion"] = {"tool": "converter", "revision": "1.0"}
        binding_plan = {
            "artifact_set_id": "voice-checkpoint-1",
            "producer": {"repository": "instavar/test", "revision": "a" * 40},
            "source_artifacts": [{"role": "model", "kind": "file", "path": "model.bin"}],
            "runtime_bindings": [
                {
                    "runtime_id": "pytorch",
                    "relation": "exact",
                    "artifacts": [{"role": "model", "kind": "file", "path": "model.bin"}],
                },
                candidate,
            ],
        }
        manifest = build_runtime_artifact_manifest(binding_plan, base_dir=root)
        rows = [
            observation(
                "voice-pytorch",
                "pytorch",
                manifest["artifact_set_id"],
                manifest["source_artifact_set_sha256"],
            ),
            observation(
                "voice-mlx",
                "mlx",
                manifest["artifact_set_id"],
                manifest["source_artifact_set_sha256"],
            ),
        ]
        return binding_plan, manifest, rows, generation_plan(rows)

    def test_passes_shared_artifact_identity_without_equivalence_claim(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding_plan, manifest, rows, plan = self.fixture(root)
            result = compare_runtime_candidates(
                rows,
                plan=plan,
                artifact_manifest=manifest,
                artifact_binding_plan=binding_plan,
                artifact_base_dir=root,
                reference_candidate_id="voice-pytorch",
                candidate_candidate_id="voice-mlx",
                reference_runtime_id="pytorch",
                candidate_runtime_id="mlx",
            )
            self.assertTrue(result["proves_shared_artifact_identity"])
            self.assertFalse(result["proves_runtime_equivalence"])
            self.assertEqual(result["artifact_verification"]["status"], "passed")
            self.assertEqual(result["objective_comparison"]["pair_count"], 1)

    def test_cli_compares_runtimes_with_live_artifact_recheck(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding_plan, manifest, rows, plan = self.fixture(root)
            paths = {
                "binding": root / "binding.json",
                "manifest": root / "manifest.json",
                "rows": root / "rows.json",
                "plan": root / "generation-plan.json",
                "output": root / "comparison.json",
            }
            for name, value in (
                ("binding", binding_plan),
                ("manifest", manifest),
                ("rows", rows),
                ("plan", plan),
            ):
                paths[name].write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "compare-runtimes",
                        str(paths["rows"]),
                        "--plan",
                        str(paths["plan"]),
                        "--artifact-manifest",
                        str(paths["manifest"]),
                        "--artifact-binding-plan",
                        str(paths["binding"]),
                        "--reference-candidate",
                        "voice-pytorch",
                        "--candidate",
                        "voice-mlx",
                        "--reference-runtime",
                        "pytorch",
                        "--candidate-runtime",
                        "mlx",
                        "--output",
                        str(paths["output"]),
                    ]
                ),
                0,
            )
            result = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertTrue(result["proves_shared_artifact_identity"])
            self.assertFalse(result["proves_runtime_equivalence"])

    def test_rejects_artifact_mutation_before_comparison(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding_plan, manifest, rows, plan = self.fixture(root)
            (root / "model.bin").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "does not match current"):
                compare_runtime_candidates(
                    rows,
                    plan=plan,
                    artifact_manifest=manifest,
                    artifact_binding_plan=binding_plan,
                    artifact_base_dir=root,
                    reference_candidate_id="voice-pytorch",
                    candidate_candidate_id="voice-mlx",
                    reference_runtime_id="pytorch",
                    candidate_runtime_id="mlx",
                )

    def test_rejects_derived_runtime_as_exact_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding_plan, manifest, rows, plan = self.fixture(root, derived_candidate=True)
            with self.assertRaisesRegex(ValueError, "non-exact runtimes"):
                compare_runtime_candidates(
                    rows,
                    plan=plan,
                    artifact_manifest=manifest,
                    artifact_binding_plan=binding_plan,
                    artifact_base_dir=root,
                    reference_candidate_id="voice-pytorch",
                    candidate_candidate_id="voice-mlx",
                    reference_runtime_id="pytorch",
                    candidate_runtime_id="mlx",
                )

    def test_rejects_observation_runtime_or_artifact_mismatch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding_plan, manifest, rows, plan = self.fixture(root)
            wrong_runtime = deepcopy(rows)
            wrong_runtime[1]["runtime_id"] = "pytorch"
            with self.assertRaisesRegex(ValueError, "runtime_id"):
                compare_runtime_candidates(
                    wrong_runtime,
                    plan=plan,
                    artifact_manifest=manifest,
                    artifact_binding_plan=binding_plan,
                    artifact_base_dir=root,
                    reference_candidate_id="voice-pytorch",
                    candidate_candidate_id="voice-mlx",
                    reference_runtime_id="pytorch",
                    candidate_runtime_id="mlx",
                )
            wrong_hash = deepcopy(rows)
            wrong_hash[1]["artifact_set_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "artifact_set_sha256"):
                compare_runtime_candidates(
                    wrong_hash,
                    plan=plan,
                    artifact_manifest=manifest,
                    artifact_binding_plan=binding_plan,
                    artifact_base_dir=root,
                    reference_candidate_id="voice-pytorch",
                    candidate_candidate_id="voice-mlx",
                    reference_runtime_id="pytorch",
                    candidate_runtime_id="mlx",
                )


if __name__ == "__main__":
    unittest.main()
