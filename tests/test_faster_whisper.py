from __future__ import annotations

import hashlib
import json
import math
import struct
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from instavar_voice_lab.cli import main
from instavar_voice_lab.extraction import apply_extractor_results
from instavar_voice_lab.faster_whisper import (
    build_faster_whisper_results,
    faster_whisper_artifacts,
)
from instavar_voice_lab.metrics import score_objective_observations


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeModel:
    def __init__(self, hypotheses: dict[str, str], *, fail_on: str | None = None):
        self.hypotheses = hypotheses
        self.fail_on = fail_on
        self.calls: list[dict] = []

    def transcribe(self, path: str, **settings):
        name = Path(path).name
        self.calls.append(settings)
        if name == self.fail_on:
            raise RuntimeError("synthetic transcription failure")
        return iter([FakeSegment(f" {self.hypotheses[name]} ")]), object()


class FasterWhisperTests(unittest.TestCase):
    @staticmethod
    def write_tone(path: Path, *, frequency: int = 440) -> None:
        sample_rate = 16000
        samples = [
            int(0.5 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate)) for index in range(sample_rate)
        ]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def versions() -> dict[str, str]:
        return {
            "faster-whisper": "1.2.1",
            "ctranslate2": "4.6.0",
            "tokenizers": "0.21.4",
            "av": "15.1.0",
        }

    def fixture(self, root: Path) -> dict:
        first = root / "first.wav"
        second = root / "second.wav"
        model = root / "model"
        model.mkdir()
        (model / "model.bin").write_bytes(b"pinned-ctranslate2-model")
        (model / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
        self.write_tone(first, frequency=420)
        self.write_tone(second, frequency=430)
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
                "audio_path": first.name,
                "audio_sha256": self.sha256(first),
            },
            {
                "observation_schema_version": "1.0.0",
                "sample_id": "adapter--p2--seed-42",
                "candidate_id": "adapter",
                "prompt_id": "p2",
                "seed": 42,
                "requested_text": "testing one two",
                "valid": True,
                "runtime_id": "pytorch",
                "audio_path": second.name,
                "audio_sha256": self.sha256(second),
            },
        ]
        return {"model": model, "observations": observations}

    def build(self, root: Path, fixture: dict, *, model: FakeModel | None = None) -> tuple[dict, FakeModel]:
        active = model or FakeModel({"first.wav": "hello world", "second.wav": "testing one two"})
        with (
            patch("instavar_voice_lab.faster_whisper._load_model", return_value=active),
            patch("instavar_voice_lab.faster_whisper._package_versions", return_value=self.versions()),
        ):
            results = build_faster_whisper_results(
                fixture["observations"],
                audio_base_dir=root,
                model_dir=fixture["model"],
                model_name="Systran/faster-whisper-tiny.en",
                model_revision="0d3d19a32d3338f10357c0889762bd8d64bbdeba",
            )
        return results, active

    def test_builds_applies_and_scores_executed_asr_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            results, model = self.build(root, fixture)
            self.assertEqual(results["schema_version"], "1.5.0")
            self.assertEqual(results["execution"]["backend"], "instavar_voice_lab.faster_whisper_v1")
            self.assertEqual(results["execution"]["model_loading"], {"local_files_only": True})
            self.assertRegex(results["execution_receipt_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(results["results"][0]["values"]["hypothesis_text"], "hello world")
            self.assertEqual(model.calls[0]["temperature"], 0.0)
            self.assertFalse(model.calls[0]["condition_on_previous_text"])

            augmented = apply_extractor_results(
                fixture["observations"],
                results,
                audio_base_dir=root,
                extractor_artifacts=faster_whisper_artifacts(fixture["model"]),
            )
            evidence = augmented[0]["evidence"]["asr"]
            self.assertEqual(
                evidence["extractor_execution_receipt_sha256"],
                results["execution_receipt_sha256"],
            )
            scored = score_objective_observations(augmented)
            self.assertEqual(scored["candidates"][0]["asr_word_error_rate"]["mean"], 0.0)

    def test_rejects_hypothesis_runtime_or_decoding_tampering(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            for mutation in ("hypothesis", "runtime", "decoding"):
                results, _model = self.build(root, fixture)
                if mutation == "hypothesis":
                    results["results"][0]["values"]["hypothesis_text"] = "substituted"
                elif mutation == "runtime":
                    results["execution"]["package_versions"]["ctranslate2"] = "substituted"
                else:
                    results["execution"]["decoding"]["beam_size"] = 7
                with self.assertRaisesRegex(ValueError, "execution_receipt_sha256 does not match"):
                    apply_extractor_results(
                        fixture["observations"],
                        results,
                        audio_base_dir=root,
                        extractor_artifacts=faster_whisper_artifacts(fixture["model"]),
                    )

    def test_rejects_model_mutation_during_extraction(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            fake = FakeModel({"first.wav": "hello world", "second.wav": "testing one two"})

            def mutate_model(model_dir: Path, **_settings) -> FakeModel:
                (model_dir / "model.bin").write_bytes(b"changed-during-run")
                return fake

            with (
                patch("instavar_voice_lab.faster_whisper._load_model", side_effect=mutate_model),
                patch("instavar_voice_lab.faster_whisper._package_versions", return_value=self.versions()),
                self.assertRaisesRegex(ValueError, "artifacts changed during extraction"),
            ):
                build_faster_whisper_results(
                    fixture["observations"],
                    audio_base_dir=root,
                    model_dir=fixture["model"],
                    model_name="Systran/faster-whisper-tiny.en",
                    model_revision="0d3d19a32d3338f10357c0889762bd8d64bbdeba",
                )

    def test_preserves_per_sample_transcription_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            model = FakeModel(
                {"first.wav": "hello world", "second.wav": "testing one two"},
                fail_on="second.wav",
            )
            results, _model = self.build(root, fixture, model=model)
            self.assertEqual(results["results"][1]["status"], "failed")
            augmented = apply_extractor_results(
                fixture["observations"],
                results,
                audio_base_dir=root,
                extractor_artifacts=faster_whisper_artifacts(fixture["model"]),
            )
            self.assertEqual(
                augmented[1]["extractor_failures"]["asr"]["error_type"],
                "RuntimeError",
            )

    def test_rejects_unsupported_execution_settings_and_symlinked_model(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            common = {
                "observations": fixture["observations"],
                "audio_base_dir": root,
                "model_dir": fixture["model"],
                "model_name": "Systran/faster-whisper-tiny.en",
                "model_revision": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
            }
            with self.assertRaisesRegex(ValueError, "device must be"):
                build_faster_whisper_results(**common, device="mps")
            with self.assertRaisesRegex(ValueError, "CPU execution requires"):
                build_faster_whisper_results(**common, device_index=1)
            with self.assertRaisesRegex(ValueError, "compute_type must be"):
                build_faster_whisper_results(**common, compute_type="int4")
            with self.assertRaisesRegex(ValueError, "lowercase language code"):
                build_faster_whisper_results(**common, language="en-SG")
            with self.assertRaisesRegex(ValueError, "beam_size must be"):
                build_faster_whisper_results(**common, beam_size=0)

            linked = root / "linked-model"
            linked.symlink_to(fixture["model"], target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                build_faster_whisper_results(**{**common, "model_dir": linked})

    def test_model_initialization_failure_is_clear_and_fatal(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            with (
                patch("instavar_voice_lab.faster_whisper._load_model", side_effect=RuntimeError("unsupported int8")),
                patch("instavar_voice_lab.faster_whisper._package_versions", return_value=self.versions()),
                self.assertRaisesRegex(RuntimeError, "failed to initialize faster-whisper on cpu:0"),
            ):
                build_faster_whisper_results(
                    fixture["observations"],
                    audio_base_dir=root,
                    model_dir=fixture["model"],
                    model_name="Systran/faster-whisper-tiny.en",
                    model_revision="0d3d19a32d3338f10357c0889762bd8d64bbdeba",
                )

    def test_cli_builds_and_applies_executed_asr_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            observations_path = root / "observations.json"
            results_path = root / "results.json"
            augmented_path = root / "augmented.json"
            observations_path.write_text(json.dumps(fixture["observations"]), encoding="utf-8")
            model = FakeModel({"first.wav": "hello world", "second.wav": "testing one two"})
            with (
                patch("instavar_voice_lab.faster_whisper._load_model", return_value=model),
                patch("instavar_voice_lab.faster_whisper._package_versions", return_value=self.versions()),
            ):
                self.assertEqual(
                    main(
                        [
                            "build-faster-whisper-results",
                            str(observations_path),
                            "--audio-base-dir",
                            str(root),
                            "--model-dir",
                            str(fixture["model"]),
                            "--model-name",
                            "Systran/faster-whisper-tiny.en",
                            "--model-revision",
                            "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
                            "--output",
                            str(results_path),
                        ]
                    ),
                    0,
                )
            self.assertEqual(
                main(
                    [
                        "apply-extractor-results",
                        str(observations_path),
                        str(results_path),
                        "--audio-base-dir",
                        str(root),
                        "--faster-whisper-model-dir",
                        str(fixture["model"]),
                        "--output",
                        str(augmented_path),
                    ]
                ),
                0,
            )
            augmented = json.loads(augmented_path.read_text(encoding="utf-8"))
            self.assertEqual(augmented[0]["hypothesis_text"], "hello world")


if __name__ == "__main__":
    unittest.main()
