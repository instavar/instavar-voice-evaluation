from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from instavar_voice_lab.cli import main
from instavar_voice_lab.content_faithfulness import build_content_faithfulness_report
from instavar_voice_lab.extraction import build_speaker_reference_catalog
from instavar_voice_lab.speaker_reference_plans import build_speaker_reference_assignment_plan


class ContentFaithfulnessTests(unittest.TestCase):
    def fixtures(self, root: Path, hypothesis: str) -> tuple[list[dict], dict, dict, dict, dict]:
        reference_audio = root / "reference.wav"
        reference_transcript = root / "reference.txt"
        reference_audio.write_bytes(b"reference-audio")
        reference_transcript.write_text(
            "A government that is honest and responsible should answer difficult questions clearly.",
            encoding="utf-8",
        )
        references = {"studio": (reference_audio, reference_transcript)}
        catalog = build_speaker_reference_catalog(catalog_id="catalog", references=references)
        requested = "Today we discuss careful testing and why exact evidence matters for reliable systems."
        sample = {
            "sample_id": "candidate--p1--seed-42",
            "candidate_id": "candidate",
            "prompt_id": "p1",
            "category": "long_form_cadence",
            "seed": 42,
            "text": requested,
            "expected_audio_path": "audio/candidate/p1/seed-42.wav",
        }
        plan = {
            "schema_version": "1.1.0",
            "prompt_pack": {"id": "pack", "version": "1.0.0", "sha256": "f" * 64},
            "candidate_ids": ["candidate"],
            "seeds": [42],
            "selected_prompt_ids": ["p1"],
            "prompt_count": 1,
            "sample_count": 1,
            "required_objective_metrics": ["asr_word_error_rate"],
            "samples": [sample],
            "generation_requirements": {
                "same_transcripts": True,
                "frozen_generation_settings": True,
                "record_failures_as_observations": True,
            },
        }
        assignment = build_speaker_reference_assignment_plan(
            plan_id="speaker-plan",
            generation_plan=plan,
            reference_catalog=catalog,
            assignments={("p1", 42): ["studio"]},
            policy_id="fixed",
            stratification_dimensions=["channel"],
            rationale="Use the same retained reference for every candidate.",
        )
        audio_sha = hashlib.sha256(b"candidate-audio").hexdigest()
        observations = [
            {
                "observation_schema_version": "1.0.0",
                "sample_id": sample["sample_id"],
                "candidate_id": sample["candidate_id"],
                "prompt_id": sample["prompt_id"],
                "seed": sample["seed"],
                "requested_text": requested,
                "valid": True,
                "runtime_id": "pytorch",
                "audio_sha256": audio_sha,
                "hypothesis_text": hypothesis,
                "evidence": {
                    "asr": {
                        "extractor": "test-asr",
                        "revision": "asr-1",
                        "input_audio_sha256": audio_sha,
                        "extractor_artifact_set_sha256": "a" * 64,
                    }
                },
            }
        ]
        return observations, plan, catalog, assignment, references

    def build(self, root: Path, hypothesis: str, **kwargs):
        observations, plan, catalog, assignment, references = self.fixtures(root, hypothesis)
        return build_content_faithfulness_report(
            observations,
            generation_plan=plan,
            reference_catalog=catalog,
            reference_assignment_plan=assignment,
            speaker_references=references,
            **kwargs,
        )

    def test_not_flagged_does_not_claim_content_proof(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.build(
                root,
                "Today we discuss careful testing and why exact evidence matters for reliable systems.",
            )
            sample = result["samples"][0]
            self.assertEqual(sample["content_gate_status"], "not_flagged")
            self.assertEqual(sample["reference_exclusive_ngram_hit_count"], 0)
            self.assertFalse(result["proves_content_faithfulness"])
            self.assertRegex(result["report_sha256"], r"^[0-9a-f]{64}$")

    def test_flags_reference_overlap_repetition_and_high_wer_separately(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            leaked = (
                "A government that is honest and responsible should answer difficult questions clearly. "
                "A government that is honest and responsible should answer difficult questions clearly."
            )
            result = self.build(
                root,
                leaked,
                ngram_size=3,
                minimum_reference_ngram_hits=1,
                repetition_excess_fraction_threshold=0.01,
            )
            sample = result["samples"][0]
            self.assertEqual(sample["content_gate_status"], "failed")
            self.assertTrue(sample["flags"]["high_word_error_rate"])
            self.assertTrue(sample["flags"]["repetition_excess"])
            self.assertTrue(sample["flags"]["reference_transcript_overlap"])
            self.assertGreater(sample["reference_exclusive_ngram_hit_count"], 0)
            self.assertGreater(sample["repeated_ngram_excess_fraction"], 0)
            self.assertNotIn("government", json.dumps(sample))
            self.assertEqual(result["candidates"][0]["flag_counts"]["reference_transcript_overlap"], 1)

    def test_requested_text_overlap_is_excluded_from_reference_leakage(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations, plan, catalog, assignment, references = self.fixtures(root, "placeholder")
            shared = "A government that is honest and responsible should answer difficult questions clearly."
            plan["samples"][0]["text"] = shared
            observations[0]["requested_text"] = shared
            observations[0]["hypothesis_text"] = shared
            assignment = build_speaker_reference_assignment_plan(
                plan_id="speaker-plan",
                generation_plan=plan,
                reference_catalog=catalog,
                assignments={("p1", 42): ["studio"]},
                policy_id="fixed",
                stratification_dimensions=["channel"],
                rationale="Use the same retained reference for every candidate.",
            )
            result = build_content_faithfulness_report(
                observations,
                generation_plan=plan,
                reference_catalog=catalog,
                reference_assignment_plan=assignment,
                speaker_references=references,
                ngram_size=3,
                minimum_reference_ngram_hits=1,
            )
            sample = result["samples"][0]
            self.assertEqual(sample["reference_exclusive_ngram_count"], 0)
            self.assertFalse(sample["flags"]["reference_transcript_overlap"])
            self.assertEqual(sample["content_gate_status"], "not_flagged")

    def test_rejects_unbound_asr_and_drifted_reference_bytes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations, plan, catalog, assignment, references = self.fixtures(root, "different words")
            observations[0]["evidence"]["asr"].pop("extractor_artifact_set_sha256")
            with self.assertRaisesRegex(ValueError, "bound ASR extractor artifacts"):
                build_content_faithfulness_report(
                    observations,
                    generation_plan=plan,
                    reference_catalog=catalog,
                    reference_assignment_plan=assignment,
                    speaker_references=references,
                )
            references["studio"][1].write_text("mutated reference transcript", encoding="utf-8")
            observations[0]["evidence"]["asr"]["extractor_artifact_set_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "catalog does not match"):
                build_content_faithfulness_report(
                    observations,
                    generation_plan=plan,
                    reference_catalog=catalog,
                    reference_assignment_plan=assignment,
                    speaker_references=references,
                )

    def test_invalid_and_asr_unavailable_rows_cannot_pass(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations, plan, catalog, assignment, references = self.fixtures(root, "different words")
            invalid = deepcopy(observations)
            invalid[0]["valid"] = False
            invalid[0].pop("hypothesis_text")
            invalid[0].pop("evidence")
            result = build_content_faithfulness_report(
                invalid,
                generation_plan=plan,
                reference_catalog=catalog,
                reference_assignment_plan=assignment,
                speaker_references=references,
            )
            self.assertEqual(result["samples"][0]["content_gate_status"], "failed")

            unavailable = deepcopy(observations)
            unavailable[0].pop("hypothesis_text")
            unavailable[0].pop("evidence")
            result = build_content_faithfulness_report(
                unavailable,
                generation_plan=plan,
                reference_catalog=catalog,
                reference_assignment_plan=assignment,
                speaker_references=references,
            )
            self.assertEqual(result["samples"][0]["content_gate_status"], "not_evaluable")
            self.assertEqual(result["candidates"][0]["content_gate_status"], "incomplete")

    def test_cli_builds_report_with_live_reference_bindings(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations, plan, catalog, assignment, references = self.fixtures(
                root,
                "Today we discuss careful testing and why exact evidence matters for reliable systems.",
            )
            paths = {
                "observations": root / "observations.json",
                "plan": root / "plan.json",
                "catalog": root / "catalog.json",
                "assignment": root / "assignment.json",
                "output": root / "report.json",
            }
            for name, value in (
                ("observations", observations),
                ("plan", plan),
                ("catalog", catalog),
                ("assignment", assignment),
            ):
                paths[name].write_text(json.dumps(value), encoding="utf-8")
            audio, transcript = references["studio"]
            self.assertEqual(
                main(
                    [
                        "build-content-faithfulness-report",
                        str(paths["observations"]),
                        "--generation-plan",
                        str(paths["plan"]),
                        "--reference-catalog",
                        str(paths["catalog"]),
                        "--speaker-reference-plan",
                        str(paths["assignment"]),
                        "--speaker-reference",
                        f"studio={audio}={transcript}",
                        "--output",
                        str(paths["output"]),
                    ]
                ),
                0,
            )
            report = json.loads(paths["output"].read_text())
            self.assertEqual(report["schema_version"], "instavar_voice_content_faithfulness/v1")


if __name__ == "__main__":
    unittest.main()
