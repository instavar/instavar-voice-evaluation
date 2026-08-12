from __future__ import annotations

import hashlib
import json
import math
import struct
import unittest
import wave
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.extraction import (
    apply_extractor_results,
    build_audio_probe_results,
    build_extractor_identity,
    build_speaker_reference_binding,
    build_speaker_reference_catalog,
    observation_document_sha256,
)
from instavar_voice_lab.metrics import score_objective_observations
from instavar_voice_lab.speaker_reference_plans import build_speaker_reference_assignment_plan


class ExtractionTests(unittest.TestCase):
    @staticmethod
    def write_tone(path: Path, *, sample_rate: int = 24000) -> None:
        samples = [int(0.5 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate)) for index in range(sample_rate)]
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def observations(self, audio: Path) -> list[dict]:
        return [
            {
                "observation_schema_version": "1.0.0",
                "sample_id": "adapter--p1--seed-42",
                "candidate_id": "adapter",
                "prompt_id": "p1",
                "seed": 42,
                "requested_text": "hello world",
                "valid": True,
                "runtime_id": "pytorch",
                "audio_path": audio.name,
                "audio_sha256": self.sha256(audio),
            }
        ]

    def test_builds_and_applies_content_addressed_audio_probe_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="e5297b6cda702c99267a2cd95c6ebbeeedd4ecd1",
            )
            self.assertEqual(results["schema_version"], "1.1.0")
            self.assertRegex(results["extractor"]["artifact_set_sha256"], r"^[0-9a-f]{64}$")
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            row = augmented[0]
            self.assertEqual(row["sample_rate_hz"], 24000)
            self.assertEqual(
                row["evidence"]["audio_probe"]["input_audio_sha256"],
                row["audio_sha256"],
            )
            self.assertEqual(row["augmentation_history"][0]["status"], "complete")
            self.assertNotIn("sample_rate_hz", observations[0])

    def test_rejects_audio_mutation_after_extraction(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            self.write_tone(audio, sample_rate=16000)
            with self.assertRaisesRegex(ValueError, "does not match the live audio file"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_rejects_source_observation_mutation_and_incomplete_coverage(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            changed = deepcopy(observations)
            changed[0]["requested_text"] = "different text"
            with self.assertRaisesRegex(ValueError, "do not match the source observation"):
                apply_extractor_results(changed, results, audio_base_dir=root)
            results["results"] = []
            with self.assertRaisesRegex(ValueError, "exactly cover valid observations"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_preserves_extractor_failure_instead_of_dropping_sample(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "not-pcm.wav"
            audio.write_bytes(b"not a wav")
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            self.assertEqual(results["results"][0]["status"], "failed")
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            self.assertIn("audio_probe", augmented[0]["extractor_failures"])
            self.assertNotIn("sample_rate_hz", augmented[0])

    def test_refuses_to_overwrite_existing_metric_or_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            observations[0]["sample_rate_hz"] = 16000
            results["source_observations_sha256"] = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )["source_observations_sha256"]
            with self.assertRaisesRegex(ValueError, "would overwrite fields"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_cli_builds_and_applies_audio_probe_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            source = root / "observations.json"
            results = root / "probe-results.json"
            output = root / "augmented.json"
            self.write_tone(audio)
            source.write_text(json.dumps(self.observations(audio)), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-audio-probe-results",
                        str(source),
                        "--audio-base-dir",
                        str(root),
                        "--extractor-revision",
                        "probe-1",
                        "--output",
                        str(results),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "apply-extractor-results",
                        str(source),
                        str(results),
                        "--audio-base-dir",
                        str(root),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            augmented = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(augmented[0]["sample_rate_hz"], 24000)

    def test_cli_fingerprints_external_extractor_and_speaker_reference(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "speaker.bin"
            reference_audio = root / "reference.wav"
            transcript = root / "reference.txt"
            identity_output = root / "extractor.json"
            reference_output = root / "reference.json"
            model.write_bytes(b"speaker-model")
            self.write_tone(reference_audio)
            transcript.write_text("reference voice", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-extractor-identity",
                        "--kind",
                        "speaker_encoder",
                        "--name",
                        "test-speaker",
                        "--revision",
                        "speaker-1",
                        "--artifact",
                        f"model=file={model}",
                        "--output",
                        str(identity_output),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "build-speaker-reference",
                        "--reference-id",
                        "voice-reference-1",
                        "--audio",
                        str(reference_audio),
                        "--transcript",
                        str(transcript),
                        "--output",
                        str(reference_output),
                    ]
                ),
                0,
            )
            identity = json.loads(identity_output.read_text(encoding="utf-8"))
            reference = json.loads(reference_output.read_text(encoding="utf-8"))
            self.assertEqual(identity["kind"], "speaker_encoder")
            self.assertEqual(identity["artifacts"][0]["role"], "model")
            self.assertEqual(reference["reference_id"], "voice-reference-1")
            self.assertEqual(reference["audio"]["sha256"], self.sha256(reference_audio))

    def test_cli_applies_external_speaker_results_with_live_bindings(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            reference_audio = root / "reference.wav"
            transcript = root / "reference.txt"
            model = root / "speaker.bin"
            source_path = root / "observations.json"
            results_path = root / "speaker-results.json"
            output_path = root / "augmented.json"
            self.write_tone(audio)
            self.write_tone(reference_audio)
            transcript.write_text("reference voice", encoding="utf-8")
            model.write_bytes(b"speaker-model")
            observations = self.observations(audio)
            artifacts = {"model": (model, "file")}
            results = {
                "schema_version": "1.1.0",
                "source_observations_sha256": observation_document_sha256(observations),
                "extractor": build_extractor_identity(
                    kind="speaker_encoder",
                    name="test-speaker",
                    revision="speaker-1",
                    artifacts=artifacts,
                ),
                "reference": build_speaker_reference_binding(
                    reference_id="voice-reference-1",
                    audio_path=reference_audio,
                    transcript_path=transcript,
                ),
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": observations[0]["audio_sha256"],
                        "status": "complete",
                        "values": {
                            "reference_speaker_embedding": [1.0, 0.0],
                            "speaker_embedding": [0.9, 0.1],
                        },
                    }
                ],
            }
            source_path.write_text(json.dumps(observations), encoding="utf-8")
            results_path.write_text(json.dumps(results), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "apply-extractor-results",
                        str(source_path),
                        str(results_path),
                        "--audio-base-dir",
                        str(root),
                        "--extractor-artifact",
                        f"model=file={model}",
                        "--reference-audio",
                        str(reference_audio),
                        "--reference-transcript",
                        str(transcript),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            augmented = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                augmented[0]["evidence"]["speaker_encoder"]["reference_id"],
                "voice-reference-1",
            )

    def test_applies_external_asr_and_speaker_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            reference_audio = root / "reference.wav"
            reference_transcript = root / "reference.txt"
            asr_model = root / "asr.bin"
            speaker_model = root / "speaker.bin"
            self.write_tone(audio)
            self.write_tone(reference_audio, sample_rate=16000)
            reference_transcript.write_text("reference voice", encoding="utf-8")
            asr_model.write_bytes(b"asr-model")
            speaker_model.write_bytes(b"speaker-model")
            observations = self.observations(audio)
            source_sha = observation_document_sha256(observations)
            audio_sha = observations[0]["audio_sha256"]
            asr_artifacts = {"model": (asr_model, "file")}
            asr_results = {
                "schema_version": "1.1.0",
                "source_observations_sha256": source_sha,
                "extractor": build_extractor_identity(
                    kind="asr",
                    name="test-asr",
                    revision="asr-1",
                    artifacts=asr_artifacts,
                ),
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": audio_sha,
                        "status": "complete",
                        "values": {"hypothesis_text": "hello world"},
                    }
                ],
            }
            with_asr = apply_extractor_results(
                observations,
                asr_results,
                audio_base_dir=root,
                extractor_artifacts=asr_artifacts,
            )
            speaker_artifacts = {"model": (speaker_model, "file")}
            reference = build_speaker_reference_binding(
                reference_id="voice-reference-1",
                audio_path=reference_audio,
                transcript_path=reference_transcript,
            )
            speaker_results = {
                "schema_version": "1.1.0",
                "source_observations_sha256": observation_document_sha256(with_asr),
                "extractor": build_extractor_identity(
                    kind="speaker_encoder",
                    name="test-speaker",
                    revision="speaker-1",
                    artifacts=speaker_artifacts,
                ),
                "reference": reference,
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": audio_sha,
                        "status": "complete",
                        "values": {
                            "reference_speaker_embedding": [1.0, 0.0],
                            "speaker_embedding": [0.9, 0.1],
                        },
                    }
                ],
            }
            augmented = apply_extractor_results(
                with_asr,
                speaker_results,
                audio_base_dir=root,
                extractor_artifacts=speaker_artifacts,
                reference_audio_path=reference_audio,
                reference_transcript_path=reference_transcript,
            )
            self.assertEqual(augmented[0]["hypothesis_text"], "hello world")
            self.assertEqual(
                augmented[0]["evidence"]["speaker_encoder"]["input_audio_sha256"],
                audio_sha,
            )
            self.assertEqual(
                augmented[0]["evidence"]["speaker_encoder"]["reference_audio_sha256"],
                reference["audio"]["sha256"],
            )
            self.assertEqual(
                augmented[0]["evidence"]["speaker_encoder"]["extractor_artifact_set_sha256"],
                speaker_results["extractor"]["artifact_set_sha256"],
            )

    def test_rejects_external_extractor_artifact_substitution(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            model = root / "model.bin"
            self.write_tone(audio)
            model.write_bytes(b"expected-model")
            observations = self.observations(audio)
            artifacts = {"model": (model, "file")}
            results = {
                "schema_version": "1.1.0",
                "source_observations_sha256": observation_document_sha256(observations),
                "extractor": build_extractor_identity(
                    kind="asr",
                    name="test-asr",
                    revision="asr-1",
                    artifacts=artifacts,
                ),
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": observations[0]["audio_sha256"],
                        "status": "complete",
                        "values": {"hypothesis_text": "hello world"},
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "external extractor artifacts are required"):
                apply_extractor_results(observations, results, audio_base_dir=root)
            model.write_bytes(b"substituted-model")
            with self.assertRaisesRegex(ValueError, "identity does not match the live artifact set"):
                apply_extractor_results(
                    observations,
                    results,
                    audio_base_dir=root,
                    extractor_artifacts=artifacts,
                )

    def test_rejects_speaker_reference_audio_or_transcript_substitution(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            reference_audio = root / "reference.wav"
            transcript = root / "reference.txt"
            model = root / "speaker.bin"
            self.write_tone(audio)
            self.write_tone(reference_audio)
            transcript.write_text("intended reference", encoding="utf-8")
            model.write_bytes(b"speaker-model")
            observations = self.observations(audio)
            artifacts = {"model": (model, "file")}
            results = {
                "schema_version": "1.1.0",
                "source_observations_sha256": observation_document_sha256(observations),
                "extractor": build_extractor_identity(
                    kind="speaker_encoder",
                    name="test-speaker",
                    revision="speaker-1",
                    artifacts=artifacts,
                ),
                "reference": build_speaker_reference_binding(
                    reference_id="voice-reference-1",
                    audio_path=reference_audio,
                    transcript_path=transcript,
                ),
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": observations[0]["audio_sha256"],
                        "status": "complete",
                        "values": {
                            "reference_speaker_embedding": [1.0, 0.0],
                            "speaker_embedding": [1.0, 0.0],
                        },
                    }
                ],
            }
            transcript.write_text("wrong reference", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference identity does not match"):
                apply_extractor_results(
                    observations,
                    results,
                    audio_base_dir=root,
                    extractor_artifacts=artifacts,
                    reference_audio_path=reference_audio,
                    reference_transcript_path=transcript,
                )
            transcript.write_text("intended reference", encoding="utf-8")
            self.write_tone(reference_audio, sample_rate=16000)
            with self.assertRaisesRegex(ValueError, "reference identity does not match"):
                apply_extractor_results(
                    observations,
                    results,
                    audio_base_dir=root,
                    extractor_artifacts=artifacts,
                    reference_audio_path=reference_audio,
                    reference_transcript_path=transcript,
                )

    def test_applies_frozen_content_addressed_multi_reference_results(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            studio_audio = root / "studio.wav"
            phone_audio = root / "phone.wav"
            studio_transcript = root / "studio.txt"
            phone_transcript = root / "phone.txt"
            model = root / "speaker.bin"
            for path, sample_rate in ((audio, 24000), (studio_audio, 24000), (phone_audio, 16000)):
                self.write_tone(path, sample_rate=sample_rate)
            studio_transcript.write_text("studio reference", encoding="utf-8")
            phone_transcript.write_text("phone reference", encoding="utf-8")
            model.write_bytes(b"speaker-model")
            observations = self.observations(audio)
            artifacts = {"model": (model, "file")}
            live_references = {
                "phone": (phone_audio, phone_transcript),
                "studio": (studio_audio, studio_transcript),
            }
            catalog = build_speaker_reference_catalog(
                catalog_id="voice-1-catalog",
                references=live_references,
            )
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
            reference_plan = build_speaker_reference_assignment_plan(
                plan_id="voice-1-eval",
                generation_plan=generation_plan,
                reference_catalog=catalog,
                assignments={("p1", 42): ["phone", "studio"]},
                policy_id="stratified-v1",
                stratification_dimensions=["channel"],
                rationale="Freeze studio and phone references before generation or scoring.",
            )
            results = {
                "schema_version": "1.3.0",
                "source_observations_sha256": observation_document_sha256(observations),
                "extractor": build_extractor_identity(
                    kind="speaker_encoder",
                    name="test-speaker",
                    revision="speaker-1",
                    artifacts=artifacts,
                ),
                "reference_catalog": catalog,
                "reference_aggregation": "mean_cosine_similarity_v1",
                "reference_assignment_plan_sha256": reference_plan["assignment_plan_sha256"],
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": observations[0]["audio_sha256"],
                        "status": "complete",
                        "reference_ids": ["phone", "studio"],
                        "values": {
                            "reference_speaker_embeddings": [
                                {"reference_id": "phone", "embedding": [0.0, 1.0]},
                                {"reference_id": "studio", "embedding": [1.0, 0.0]},
                            ],
                            "speaker_embedding": [1.0, 0.0],
                        },
                    }
                ],
            }
            augmented = apply_extractor_results(
                observations,
                results,
                audio_base_dir=root,
                extractor_artifacts=artifacts,
                speaker_references=live_references,
                speaker_reference_plan=reference_plan,
                generation_plan=generation_plan,
            )
            evidence = augmented[0]["evidence"]["speaker_encoder"]
            self.assertEqual(evidence["reference_aggregation"], "mean_cosine_similarity_v1")
            self.assertRegex(evidence["reference_set_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(evidence["reference_assignment_plan_sha256"], reference_plan["assignment_plan_sha256"])
            self.assertRegex(evidence["speaker_measurement_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual([item["reference_id"] for item in evidence["references"]], ["phone", "studio"])
            self.assertEqual(
                [item["reference_id"] for item in augmented[0]["reference_speaker_embeddings"]],
                ["phone", "studio"],
            )

            source_path = root / "observations.json"
            results_path = root / "speaker-results-v1.3.json"
            output_path = root / "augmented.json"
            catalog_path = root / "catalog.json"
            generation_plan_path = root / "generation-plan.json"
            reference_plan_path = root / "reference-plan.json"
            source_path.write_text(json.dumps(observations), encoding="utf-8")
            results_path.write_text(json.dumps(results), encoding="utf-8")
            generation_plan_path.write_text(json.dumps(generation_plan), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "build-speaker-reference-catalog",
                        "--catalog-id",
                        "voice-1-catalog",
                        "--reference",
                        f"phone={phone_audio}={phone_transcript}",
                        "--reference",
                        f"studio={studio_audio}={studio_transcript}",
                        "--output",
                        str(catalog_path),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(catalog_path.read_text(encoding="utf-8")), catalog)
            self.assertEqual(
                main(
                    [
                        "build-speaker-reference-assignment-plan",
                        "--plan-id",
                        "voice-1-eval",
                        "--generation-plan",
                        str(generation_plan_path),
                        "--reference-catalog",
                        str(catalog_path),
                        "--policy-id",
                        "stratified-v1",
                        "--stratification-dimension",
                        "channel",
                        "--rationale",
                        "Freeze studio and phone references before generation or scoring.",
                        "--assignment",
                        "p1=42=phone,studio",
                        "--output",
                        str(reference_plan_path),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(reference_plan_path.read_text(encoding="utf-8")), reference_plan)
            self.assertEqual(
                main(
                    [
                        "apply-extractor-results",
                        str(source_path),
                        str(results_path),
                        "--audio-base-dir",
                        str(root),
                        "--extractor-artifact",
                        f"model=file={model}",
                        "--speaker-reference",
                        f"phone={phone_audio}={phone_transcript}",
                        "--speaker-reference",
                        f"studio={studio_audio}={studio_transcript}",
                        "--speaker-reference-plan",
                        str(reference_plan_path),
                        "--generation-plan",
                        str(generation_plan_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))[0]["evidence"]["speaker_encoder"],
                evidence,
            )

            changed_results = deepcopy(results)
            changed_results["results"][0]["reference_ids"] = ["studio"]
            changed_results["results"][0]["values"]["reference_speaker_embeddings"] = [
                {"reference_id": "studio", "embedding": [1.0, 0.0]}
            ]
            with self.assertRaisesRegex(ValueError, "do not match the frozen assignment"):
                apply_extractor_results(
                    observations,
                    changed_results,
                    audio_base_dir=root,
                    extractor_artifacts=artifacts,
                    speaker_references=live_references,
                    speaker_reference_plan=reference_plan,
                    generation_plan=generation_plan,
                )

            changed_generation_plan = deepcopy(generation_plan)
            changed_generation_plan["samples"][0]["text"] = "changed after preregistration"
            with self.assertRaisesRegex(ValueError, "does not match the generation plan"):
                apply_extractor_results(
                    observations,
                    results,
                    audio_base_dir=root,
                    extractor_artifacts=artifacts,
                    speaker_references=live_references,
                    speaker_reference_plan=reference_plan,
                    generation_plan=changed_generation_plan,
                )

            augmented[0]["reference_speaker_embeddings"][0]["embedding"] = [1.0, 1.0]
            with self.assertRaisesRegex(ValueError, "does not match the speaker embeddings"):
                score_objective_observations(augmented)

            phone_transcript.write_text("substituted phone reference", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "catalog does not match"):
                apply_extractor_results(
                    observations,
                    results,
                    audio_base_dir=root,
                    extractor_artifacts=artifacts,
                    speaker_references=live_references,
                    speaker_reference_plan=reference_plan,
                    generation_plan=generation_plan,
                )

    def test_rejects_multi_reference_membership_and_order_substitution(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            reference_audio = root / "reference.wav"
            transcript = root / "reference.txt"
            model = root / "speaker.bin"
            self.write_tone(audio)
            self.write_tone(reference_audio)
            transcript.write_text("reference", encoding="utf-8")
            model.write_bytes(b"speaker-model")
            observations = self.observations(audio)
            artifacts = {"model": (model, "file")}
            live_references = {"studio": (reference_audio, transcript)}
            results = {
                "schema_version": "1.2.0",
                "source_observations_sha256": observation_document_sha256(observations),
                "extractor": build_extractor_identity(
                    kind="speaker_encoder",
                    name="test-speaker",
                    revision="speaker-1",
                    artifacts=artifacts,
                ),
                "reference_catalog": build_speaker_reference_catalog(
                    catalog_id="voice-1-catalog",
                    references=live_references,
                ),
                "reference_aggregation": "mean_cosine_similarity_v1",
                "results": [
                    {
                        "sample_id": observations[0]["sample_id"],
                        "audio_sha256": observations[0]["audio_sha256"],
                        "status": "complete",
                        "reference_ids": ["studio"],
                        "values": {
                            "reference_speaker_embeddings": [
                                {"reference_id": "other", "embedding": [1.0, 0.0]},
                            ],
                            "speaker_embedding": [1.0, 0.0],
                        },
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "exactly match reference_ids"):
                apply_extractor_results(
                    observations,
                    results,
                    audio_base_dir=root,
                    extractor_artifacts=artifacts,
                    speaker_references=live_references,
                )

    def test_rejects_empty_or_symlinked_identity_artifacts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty_model = root / "empty.bin"
            reference_audio = root / "reference.wav"
            reference_link = root / "reference-link.wav"
            transcript = root / "reference.txt"
            empty_model.write_bytes(b"")
            self.write_tone(reference_audio)
            reference_link.symlink_to(reference_audio)
            transcript.write_text("reference voice", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                build_extractor_identity(
                    kind="asr",
                    name="test-asr",
                    revision="asr-1",
                    artifacts={"model": (empty_model, "file")},
                )
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                build_speaker_reference_binding(
                    reference_id="voice-reference-1",
                    audio_path=reference_link,
                    transcript_path=transcript,
                )

    def test_rejects_malformed_extractor_values_at_application(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            results["results"][0]["values"]["silence_fraction"] = 1.5
            with self.assertRaisesRegex(ValueError, "between zero and one"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_rejects_unknown_result_document_and_row_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            self.write_tone(audio)
            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            results["unreviewed"] = True
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                apply_extractor_results(observations, results, audio_base_dir=root)
            del results["unreviewed"]
            results["results"][0]["unreviewed"] = True
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                apply_extractor_results(observations, results, audio_base_dir=root)

    def test_rejects_symlink_escape_and_duplicate_extractor_application(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "tone.wav"
            link = root / "link.wav"
            self.write_tone(audio)
            link.symlink_to(audio)
            observations = self.observations(audio)
            observations[0]["audio_path"] = link.name
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                build_audio_probe_results(
                    observations,
                    audio_base_dir=root,
                    extractor_revision="probe-1",
                )

            nested = root / "nested"
            nested.mkdir()
            observations[0]["audio_path"] = "../tone.wav"
            with self.assertRaisesRegex(ValueError, "escapes the audio base directory"):
                build_audio_probe_results(
                    observations,
                    audio_base_dir=nested,
                    extractor_revision="probe-1",
                )

            observations = self.observations(audio)
            results = build_audio_probe_results(
                observations,
                audio_base_dir=root,
                extractor_revision="probe-1",
            )
            augmented = apply_extractor_results(observations, results, audio_base_dir=root)
            retry = build_audio_probe_results(
                augmented,
                audio_base_dir=root,
                extractor_revision="probe-2",
            )
            with self.assertRaisesRegex(ValueError, "already contains evidence.audio_probe"):
                apply_extractor_results(augmented, retry, audio_base_dir=root)


if __name__ == "__main__":
    unittest.main()
