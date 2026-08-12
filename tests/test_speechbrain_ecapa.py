from __future__ import annotations

import hashlib
import math
import struct
import unittest
import wave
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from instavar_voice_lab.extraction import apply_extractor_results, build_speaker_reference_catalog
from instavar_voice_lab.metrics import score_objective_observations
from instavar_voice_lab.speaker_reference_plans import build_speaker_reference_assignment_plan
from instavar_voice_lab.speechbrain_ecapa import (
    _validate_torch_device,
    build_speechbrain_ecapa_results,
    speechbrain_ecapa_artifacts,
)


class FakeTensor:
    def __init__(self, values: list[float]):
        self.values = values

    def unsqueeze(self, _dimension: int) -> FakeTensor:
        return self

    def squeeze(self) -> FakeTensor:
        return self

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[float]:
        return self.values


class FakeClassifier:
    def __init__(self, vectors: dict[str, list[float]], *, fail_on: str | None = None):
        self.vectors = vectors
        self.fail_on = fail_on

    def load_audio(self, path: str) -> FakeTensor:
        name = Path(path).name
        if name == self.fail_on:
            raise RuntimeError("synthetic encoder failure")
        return FakeTensor(self.vectors[name])

    def encode_batch(self, signal: FakeTensor) -> FakeTensor:
        return signal


class SpeechBrainEcapaTests(unittest.TestCase):
    @staticmethod
    def write_tone(path: Path, *, frequency: int = 440) -> None:
        sample_rate = 16000
        samples = [
            int(0.5 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(sample_rate)
        ]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fixture(self, root: Path) -> dict:
        candidate = root / "candidate.wav"
        phone = root / "phone.wav"
        studio = root / "studio.wav"
        phone_text = root / "phone.txt"
        studio_text = root / "studio.txt"
        model = root / "model"
        model.mkdir()
        (model / "embedding_model.ckpt").write_bytes(b"pinned-ecapa-model")
        self.write_tone(candidate, frequency=420)
        self.write_tone(phone, frequency=430)
        self.write_tone(studio, frequency=440)
        phone_text.write_text("phone reference", encoding="utf-8")
        studio_text.write_text("studio reference", encoding="utf-8")
        observations = [
            {
                "observation_schema_version": "1.0.0",
                "sample_id": "adapter--p1--seed-42",
                "candidate_id": "adapter",
                "prompt_id": "p1",
                "seed": 42,
                "requested_text": "hello world",
                "valid": True,
                "runtime_id": "pytorch",
                "audio_path": candidate.name,
                "audio_sha256": self.sha256(candidate),
            }
        ]
        references = {"phone": (phone, phone_text), "studio": (studio, studio_text)}
        catalog = build_speaker_reference_catalog(catalog_id="voice-1-catalog", references=references)
        generation_plan = {
            "schema_version": "1.1.0",
            "prompt_pack": {"id": "test", "version": "1.0.0", "sha256": "a" * 64},
            "candidate_ids": ["adapter"],
            "sample_count": 1,
            "required_objective_metrics": ["speaker_embedding_similarity"],
            "samples": [
                {
                    "sample_id": observations[0]["sample_id"],
                    "candidate_id": "adapter",
                    "prompt_id": "p1",
                    "seed": 42,
                    "text": "hello world",
                }
            ],
            "generation_requirements": {
                "same_transcripts": True,
                "frozen_generation_settings": True,
                "record_failures_as_observations": True,
            },
        }
        assignment_plan = build_speaker_reference_assignment_plan(
            plan_id="voice-1-eval",
            generation_plan=generation_plan,
            reference_catalog=catalog,
            assignments={("p1", 42): ["phone", "studio"]},
            policy_id="stratified-v1",
            stratification_dimensions=["channel"],
            rationale="Freeze references before generation or scoring.",
        )
        return {
            "candidate": candidate,
            "model": model,
            "observations": observations,
            "references": references,
            "generation_plan": generation_plan,
            "assignment_plan": assignment_plan,
        }

    @staticmethod
    def versions() -> dict[str, str]:
        return {"speechbrain": "1.1.0", "torch": "2.6.0", "torchaudio": "2.6.0"}

    def build(self, root: Path, fixture: dict, *, classifier: FakeClassifier | None = None) -> dict:
        vectors = {
            "candidate.wav": [0.9, 0.1, 0.0],
            "phone.wav": [1.0, 0.0, 0.0],
            "studio.wav": [0.8, 0.2, 0.0],
        }
        active = classifier or FakeClassifier(vectors)
        with (
            patch("instavar_voice_lab.speechbrain_ecapa._load_classifier", return_value=active),
            patch("instavar_voice_lab.speechbrain_ecapa._package_versions", return_value=self.versions()),
        ):
            return build_speechbrain_ecapa_results(
                fixture["observations"],
                audio_base_dir=root,
                model_dir=fixture["model"],
                model_revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                catalog_id="voice-1-catalog",
                speaker_references=fixture["references"],
                speaker_reference_plan=fixture["assignment_plan"],
                generation_plan=fixture["generation_plan"],
                trusted_model_checkpoints=True,
            )

    def test_builds_applies_and_scores_executed_speaker_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            results = self.build(root, fixture)
            self.assertEqual(results["schema_version"], "1.4.0")
            self.assertEqual(results["execution"]["backend"], "instavar_voice_lab.speechbrain_ecapa_v1")
            self.assertRegex(results["execution_receipt_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(results["results"][0]["values"]["speaker_embedding"]), 3)

            augmented = apply_extractor_results(
                fixture["observations"],
                results,
                audio_base_dir=root,
                extractor_artifacts=speechbrain_ecapa_artifacts(fixture["model"]),
                speaker_references=fixture["references"],
                speaker_reference_plan=fixture["assignment_plan"],
                generation_plan=fixture["generation_plan"],
            )
            evidence = augmented[0]["evidence"]["speaker_encoder"]
            self.assertEqual(
                evidence["extractor_execution_receipt_sha256"],
                results["execution_receipt_sha256"],
            )
            scored = score_objective_observations(augmented)
            self.assertGreater(scored["candidates"][0]["speaker_embedding_similarity"]["mean"], 0.9)

    def test_rejects_embedding_or_runtime_receipt_tampering(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            for mutation in ("embedding", "runtime"):
                results = self.build(root, fixture)
                if mutation == "embedding":
                    results["results"][0]["values"]["speaker_embedding"][0] = 0.1
                else:
                    results["execution"]["package_versions"]["torch"] = "substituted"
                with self.assertRaisesRegex(ValueError, "execution_receipt_sha256 does not match"):
                    apply_extractor_results(
                        fixture["observations"],
                        results,
                        audio_base_dir=root,
                        extractor_artifacts=speechbrain_ecapa_artifacts(fixture["model"]),
                        speaker_references=fixture["references"],
                        speaker_reference_plan=fixture["assignment_plan"],
                        generation_plan=fixture["generation_plan"],
                    )

    def test_rejects_model_mutation_during_extraction(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            vectors = {
                "candidate.wav": [0.9, 0.1],
                "phone.wav": [1.0, 0.0],
                "studio.wav": [0.8, 0.2],
            }

            def mutate_model(model: Path, _device: str) -> FakeClassifier:
                (model / "embedding_model.ckpt").write_bytes(b"changed-during-run")
                return FakeClassifier(vectors)

            with (
                patch("instavar_voice_lab.speechbrain_ecapa._load_classifier", side_effect=mutate_model),
                patch("instavar_voice_lab.speechbrain_ecapa._package_versions", return_value=self.versions()),
                self.assertRaisesRegex(ValueError, "artifacts changed during extraction"),
            ):
                build_speechbrain_ecapa_results(
                    fixture["observations"],
                    audio_base_dir=root,
                    model_dir=fixture["model"],
                    model_revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                    catalog_id="voice-1-catalog",
                    speaker_references=fixture["references"],
                    speaker_reference_plan=fixture["assignment_plan"],
                    generation_plan=fixture["generation_plan"],
                    trusted_model_checkpoints=True,
                )

    def test_preserves_per_sample_encoder_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            classifier = FakeClassifier(
                {
                    "candidate.wav": [0.9, 0.1],
                    "phone.wav": [1.0, 0.0],
                    "studio.wav": [0.8, 0.2],
                },
                fail_on="candidate.wav",
            )
            results = self.build(root, fixture, classifier=classifier)
            self.assertEqual(results["results"][0]["status"], "failed")
            augmented = apply_extractor_results(
                fixture["observations"],
                results,
                audio_base_dir=root,
                extractor_artifacts=speechbrain_ecapa_artifacts(fixture["model"]),
                speaker_references=fixture["references"],
                speaker_reference_plan=fixture["assignment_plan"],
                generation_plan=fixture["generation_plan"],
            )
            self.assertEqual(
                augmented[0]["extractor_failures"]["speaker_encoder"]["error_type"],
                "RuntimeError",
            )

    def test_rejects_unsupported_device_and_plan_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "explicit checkpoint trust"):
                build_speechbrain_ecapa_results(
                    fixture["observations"],
                    audio_base_dir=root,
                    model_dir=fixture["model"],
                    model_revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                    catalog_id="voice-1-catalog",
                    speaker_references=fixture["references"],
                    speaker_reference_plan=fixture["assignment_plan"],
                    generation_plan=fixture["generation_plan"],
                )
            changed_plan = deepcopy(fixture["generation_plan"])
            changed_plan["samples"][0]["text"] = "changed after reference assignment"
            with self.assertRaisesRegex(ValueError, "does not match the generation plan"):
                build_speechbrain_ecapa_results(
                    fixture["observations"],
                    audio_base_dir=root,
                    model_dir=fixture["model"],
                    model_revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                    catalog_id="voice-1-catalog",
                    speaker_references=fixture["references"],
                    speaker_reference_plan=fixture["assignment_plan"],
                    generation_plan=changed_plan,
                    trusted_model_checkpoints=True,
                )
            with self.assertRaisesRegex(ValueError, "device must be"):
                build_speechbrain_ecapa_results(
                    fixture["observations"],
                    audio_base_dir=root,
                    model_dir=fixture["model"],
                    model_revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                    catalog_id="voice-1-catalog",
                    speaker_references=fixture["references"],
                    speaker_reference_plan=fixture["assignment_plan"],
                    generation_plan=fixture["generation_plan"],
                    device="mps",
                    trusted_model_checkpoints=True,
                )

    def test_cuda_device_validation_fails_clearly(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 1

        class FakeTorch:
            cuda = FakeCuda()

        _validate_torch_device(FakeTorch(), "cpu")
        _validate_torch_device(FakeTorch(), "cuda")
        with self.assertRaisesRegex(RuntimeError, "index 2 is unavailable"):
            _validate_torch_device(FakeTorch(), "cuda:2")

        FakeCuda.is_available = staticmethod(lambda: False)
        with self.assertRaisesRegex(RuntimeError, "no available CUDA device"):
            _validate_torch_device(FakeTorch(), "cuda")


if __name__ == "__main__":
    unittest.main()
