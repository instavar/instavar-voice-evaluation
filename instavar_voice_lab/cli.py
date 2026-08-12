from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audio_probe import compare_wav_probes, probe_wav
from .comparison import compare_matched_candidates, compare_runtime_candidates
from .contracts import VALIDATORS, validate_document
from .corpus import audit_corpus
from .extraction import (
    EXTRACTOR_FIELDS,
    apply_extractor_results,
    build_audio_probe_results,
    build_extractor_identity,
    build_speaker_reference_binding,
    observation_document_sha256,
)
from .lifecycle import (
    run_lifecycle,
    run_registered_lifecycle,
    validate_backend_registry,
    validate_backend_spec,
)
from .lineage import build_dataset_lineage, validate_dataset_lineage, verify_dataset_lineage
from .listening import aggregate_listening_results, build_blind_pack, stage_blind_audio
from .metrics import score_objective_observations
from .observations import validate_objective_observations
from .runtime_artifacts import (
    build_runtime_artifact_manifest,
    validate_runtime_artifact_manifest,
    verify_runtime_artifact_manifest,
)
from .suite import build_generation_plan, check_suite_coverage, validate_prompt_pack


def _read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_declarations(values: list[str]) -> dict[str, tuple[Path, str]]:
    declarations: dict[str, tuple[Path, str]] = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"invalid artifact declaration: {value}")
        role, kind, raw_path = parts
        if role in declarations:
            raise ValueError(f"duplicate artifact role: {role}")
        declarations[role] = (Path(raw_path), kind)
    return declarations


def _validate(kind: str, path: Path) -> int:
    try:
        document = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        print(f"{path}: {error}", file=sys.stderr)
        return 2
    errors = validate_document(kind, document)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"valid {kind} contract: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and generate Instavar Voice evidence artifacts.")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a contract JSON document")
    validate.add_argument("kind", choices=sorted(VALIDATORS))
    validate.add_argument("path", type=Path)

    repository = commands.add_parser(
        "validate-repository", help="validate instavar-voice-capabilities.json in a repository"
    )
    repository.add_argument("root", type=Path)

    audit = commands.add_parser("audit-corpus", help="audit train, validation, and test JSONL manifests")
    audit.add_argument(
        "--split",
        action="append",
        required=True,
        help="split declaration in the form train=/path/to/train.jsonl",
    )
    audit.add_argument("--audio-field", default="audio")
    audit.add_argument("--text-field", default="text")
    audit.add_argument("--group-field", default="")
    audit.add_argument("--output", type=Path)

    build_lineage = commands.add_parser(
        "build-dataset-lineage",
        help="fingerprint raw and prepared dataset artifacts into a lineage receipt",
    )
    build_lineage.add_argument("--lineage-id", required=True)
    build_lineage.add_argument("--producer-repository", required=True)
    build_lineage.add_argument("--producer-revision", required=True)
    build_lineage.add_argument("--input", action="append", required=True, help="ROLE=file|tree=PATH")
    build_lineage.add_argument("--output-artifact", action="append", required=True, help="ROLE=file|tree=PATH")
    build_lineage.add_argument("--receipt", type=Path, required=True)

    verify_lineage = commands.add_parser(
        "verify-dataset-lineage",
        help="verify a dataset lineage receipt against current artifacts",
    )
    verify_lineage.add_argument("receipt", type=Path)
    verify_lineage.add_argument("--producer-revision", required=True)
    verify_lineage.add_argument("--input", action="append", required=True, help="ROLE=file|tree=PATH")
    verify_lineage.add_argument("--output-artifact", action="append", required=True, help="ROLE=file|tree=PATH")
    verify_lineage.add_argument("--report", type=Path)

    build_runtime_artifacts = commands.add_parser(
        "build-runtime-artifact-manifest",
        help="fingerprint source and runtime artifact sets from a local binding plan",
    )
    build_runtime_artifacts.add_argument("binding_plan", type=Path)
    build_runtime_artifacts.add_argument("--output", type=Path, required=True)

    verify_runtime_artifacts = commands.add_parser(
        "verify-runtime-artifact-manifest",
        help="verify a runtime artifact manifest against a local binding plan and current files",
    )
    verify_runtime_artifacts.add_argument("manifest", type=Path)
    verify_runtime_artifacts.add_argument("binding_plan", type=Path)
    verify_runtime_artifacts.add_argument("--report", type=Path)

    probe = commands.add_parser("probe-audio", help="record deterministic diagnostics for a PCM WAV file")
    probe.add_argument("wav", type=Path)
    probe.add_argument("--output", type=Path)

    compare = commands.add_parser(
        "compare-audio",
        help="compare deterministic diagnostics for reference and candidate PCM WAV files",
    )
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)

    blind = commands.add_parser("build-listening-pack", help="create blind review and reveal mapping documents")
    blind.add_argument(
        "samples", type=Path, help="JSON array of sample_id, candidate_id, prompt_id, and audio_path rows"
    )
    blind.add_argument("--criteria", type=Path, required=True, help="JSON array of criterion names")
    blind.add_argument("--review-output", type=Path, required=True)
    blind.add_argument("--reveal-output", type=Path, required=True)
    blind.add_argument("--seed", type=int, required=True)
    blind.add_argument(
        "--stage-root",
        type=Path,
        help="copy audio to identity-neutral blind_audio paths under this directory",
    )
    blind.add_argument("--stage-manifest", type=Path)

    objective = commands.add_parser(
        "score-objective",
        help="score versioned ASR, speaker, validity, runtime, and memory observations",
    )
    objective.add_argument("observations", type=Path)
    objective.add_argument("--output", type=Path, required=True)
    objective.add_argument("--seed", type=int, default=20260812)

    observation_contract = commands.add_parser(
        "validate-observations",
        help="validate objective observation rows before scoring or comparison",
    )
    observation_contract.add_argument("observations", type=Path)
    observation_contract.add_argument("--require-version", action="store_true")
    observation_contract.add_argument("--require-seed", action="store_true")
    observation_contract.add_argument("--require-runtime", action="store_true")

    audio_probe_results = commands.add_parser(
        "build-audio-probe-results",
        help="probe live WAV files and bind result rows to observation and audio hashes",
    )
    audio_probe_results.add_argument("observations", type=Path)
    audio_probe_results.add_argument("--audio-base-dir", type=Path, required=True)
    audio_probe_results.add_argument("--extractor-revision", required=True)
    audio_probe_results.add_argument("--output", type=Path, required=True)

    extractor_identity = commands.add_parser(
        "build-extractor-identity",
        help="fingerprint the exact implementation or model artifacts used by an extractor",
    )
    extractor_identity.add_argument("--kind", choices=sorted(EXTRACTOR_FIELDS), required=True)
    extractor_identity.add_argument("--name", required=True)
    extractor_identity.add_argument("--revision", required=True)
    extractor_identity.add_argument("--artifact", action="append", required=True, help="ROLE=file|tree=PATH")
    extractor_identity.add_argument("--output", type=Path, required=True)

    speaker_reference = commands.add_parser(
        "build-speaker-reference",
        help="fingerprint the exact reference audio and transcript used by speaker scoring",
    )
    speaker_reference.add_argument("--reference-id", required=True)
    speaker_reference.add_argument("--audio", type=Path, required=True)
    speaker_reference.add_argument("--transcript", type=Path, required=True)
    speaker_reference.add_argument("--output", type=Path, required=True)

    apply_results = commands.add_parser(
        "apply-extractor-results",
        help="immutably augment observations with content-addressed extractor results",
    )
    apply_results.add_argument("observations", type=Path)
    apply_results.add_argument("results", type=Path)
    apply_results.add_argument("--audio-base-dir", type=Path, required=True)
    apply_results.add_argument(
        "--extractor-artifact",
        action="append",
        default=[],
        help="live ROLE=file|tree=PATH declaration; required for external extractors",
    )
    apply_results.add_argument("--reference-audio", type=Path)
    apply_results.add_argument("--reference-transcript", type=Path)
    apply_results.add_argument("--output", type=Path, required=True)

    fingerprint_observations = commands.add_parser(
        "fingerprint-observations",
        help="print the canonical SHA-256 used by content-addressed extractor results",
    )
    fingerprint_observations.add_argument("observations", type=Path)

    listening = commands.add_parser(
        "aggregate-listening",
        help="reveal and aggregate completed blind ratings without a composite score",
    )
    listening.add_argument("review", type=Path)
    listening.add_argument("reveal", type=Path)
    listening.add_argument("ratings", type=Path)
    listening.add_argument("--output", type=Path, required=True)
    listening.add_argument("--seed", type=int, default=20260812)
    listening.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="emit explicit incomplete coverage instead of failing on missing ratings",
    )

    matched = commands.add_parser(
        "compare-matched",
        help="compare baseline and adapted observations paired by prompt and seed",
    )
    matched.add_argument("observations", type=Path)
    matched.add_argument("--plan", type=Path, required=True)
    matched.add_argument("--baseline", required=True)
    matched.add_argument("--adapted", required=True)
    matched.add_argument("--output", type=Path, required=True)
    matched.add_argument("--seed", type=int, default=20260812)

    runtime_comparison = commands.add_parser(
        "compare-runtimes",
        help="compare matched runtime observations bound to one exact artifact set",
    )
    runtime_comparison.add_argument("observations", type=Path)
    runtime_comparison.add_argument("--plan", type=Path, required=True)
    runtime_comparison.add_argument("--artifact-manifest", type=Path, required=True)
    runtime_comparison.add_argument("--artifact-binding-plan", type=Path, required=True)
    runtime_comparison.add_argument("--reference-candidate", required=True)
    runtime_comparison.add_argument("--candidate", required=True)
    runtime_comparison.add_argument("--reference-runtime", required=True)
    runtime_comparison.add_argument("--candidate-runtime", required=True)
    runtime_comparison.add_argument("--seed", type=int, default=20260812)
    runtime_comparison.add_argument("--output", type=Path, required=True)

    backend = commands.add_parser("validate-backend", help="validate a lifecycle backend specification")
    backend.add_argument("spec", type=Path)

    backend_registry = commands.add_parser(
        "validate-backend-registry",
        help="validate a repository registry of lifecycle backend specifications",
    )
    backend_registry.add_argument("registry", type=Path)

    lifecycle = commands.add_parser(
        "run-lifecycle",
        help="run fail-closed preflight, train, infer, evaluate, and package stages",
    )
    lifecycle.add_argument("backend", type=Path)
    lifecycle.add_argument("experiment", type=Path)
    lifecycle.add_argument("--work-dir", type=Path, required=True)

    registered_lifecycle = commands.add_parser(
        "run-registered-lifecycle",
        help="select and run a backend from a validated repository registry",
    )
    registered_lifecycle.add_argument("registry", type=Path)
    registered_lifecycle.add_argument("experiment", type=Path)
    registered_lifecycle.add_argument("--backend-id")
    registered_lifecycle.add_argument("--work-dir", type=Path, required=True)

    prompt_pack = commands.add_parser("validate-prompt-pack", help="validate a frozen prompt pack")
    prompt_pack.add_argument("path", type=Path)

    plan = commands.add_parser("build-generation-plan", help="expand candidates, prompts, and seeds into a frozen plan")
    plan.add_argument("prompt_pack", type=Path)
    plan.add_argument("--candidate", action="append", required=True)
    plan.add_argument("--seed", action="append", type=int)
    plan.add_argument("--output", type=Path, required=True)

    coverage = commands.add_parser("check-suite-coverage", help="fail when planned samples lack observations")
    coverage.add_argument("plan", type=Path)
    coverage.add_argument("observations", type=Path)
    coverage.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        return _validate(args.kind, args.path)
    if args.command == "validate-repository":
        return _validate("capability", args.root / "instavar-voice-capabilities.json")
    if args.command == "audit-corpus":
        splits: dict[str, Path] = {}
        for declaration in args.split:
            if "=" not in declaration:
                print(f"invalid --split declaration: {declaration}", file=sys.stderr)
                return 2
            name, raw_path = declaration.split("=", 1)
            if name in splits or not name or not raw_path:
                print(f"invalid or duplicate --split declaration: {declaration}", file=sys.stderr)
                return 2
            splits[name] = Path(raw_path)
        result = audit_corpus(
            splits,
            audio_field=args.audio_field,
            text_field=args.text_field,
            group_field=args.group_field or None,
        )
        if args.output:
            _write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    if args.command == "build-dataset-lineage":
        try:
            result = build_dataset_lineage(
                lineage_id=args.lineage_id,
                producer_repository=args.producer_repository,
                producer_revision=args.producer_revision,
                inputs=_artifact_declarations(args.input),
                outputs=_artifact_declarations(args.output_artifact),
            )
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.receipt, result)
        return 0
    if args.command == "verify-dataset-lineage":
        try:
            document = _read_json(args.receipt)
            contract_errors = validate_dataset_lineage(document)
            if contract_errors:
                raise ValueError("invalid dataset lineage: " + "; ".join(contract_errors))
            result = verify_dataset_lineage(
                document,
                producer_revision=args.producer_revision,
                inputs=_artifact_declarations(args.input),
                outputs=_artifact_declarations(args.output_artifact),
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        if args.report:
            _write_json(args.report, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "build-runtime-artifact-manifest":
        try:
            result = build_runtime_artifact_manifest(
                _read_json(args.binding_plan),
                base_dir=args.binding_plan.parent.resolve(),
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "verify-runtime-artifact-manifest":
        try:
            document = _read_json(args.manifest)
            contract_errors = validate_runtime_artifact_manifest(document)
            if contract_errors:
                raise ValueError("invalid runtime artifact manifest: " + "; ".join(contract_errors))
            result = verify_runtime_artifact_manifest(
                document,
                _read_json(args.binding_plan),
                base_dir=args.binding_plan.parent.resolve(),
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        if args.report:
            _write_json(args.report, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "probe-audio":
        try:
            result = probe_wav(args.wav)
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        if args.output:
            _write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "compare-audio":
        try:
            result = compare_wav_probes(args.reference, args.candidate)
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        if args.output:
            _write_json(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "build-listening-pack":
        try:
            samples = _read_json(args.samples)
            criteria = _read_json(args.criteria)
            if not isinstance(samples, list):
                raise ValueError("samples must be a JSON array")
            normalized_samples = []
            for sample in samples:
                if not isinstance(sample, dict):
                    normalized_samples.append(sample)
                    continue
                normalized = dict(sample)
                audio_path = Path(str(normalized.get("audio_path", "")))
                if not audio_path.is_absolute():
                    normalized["audio_path"] = str((args.samples.parent / audio_path).resolve())
                normalized_samples.append(normalized)
            review, reveal = build_blind_pack(normalized_samples, criteria, seed=args.seed)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.review_output, review)
        _write_json(args.reveal_output, reveal)
        if args.stage_root:
            try:
                stage_manifest = stage_blind_audio(review, reveal, args.stage_root)
            except (OSError, ValueError) as error:
                print(error, file=sys.stderr)
                return 2
            stage_manifest_path = args.stage_manifest or args.stage_root / "blind-audio-manifest.json"
            _write_json(stage_manifest_path, stage_manifest)
        return 0
    if args.command == "score-objective":
        try:
            rows = _read_json(args.observations)
            if not isinstance(rows, list):
                raise ValueError("objective observations must be a JSON array")
            result = score_objective_observations(rows, seed=args.seed)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "validate-observations":
        try:
            errors = validate_objective_observations(
                _read_json(args.observations),
                require_version=args.require_version,
                require_seed=args.require_seed,
                require_runtime=args.require_runtime,
            )
        except (OSError, json.JSONDecodeError) as error:
            print(error, file=sys.stderr)
            return 2
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"valid objective observations: {args.observations}")
        return 0
    if args.command == "build-audio-probe-results":
        try:
            result = build_audio_probe_results(
                _read_json(args.observations),
                audio_base_dir=args.audio_base_dir,
                extractor_revision=args.extractor_revision,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "build-extractor-identity":
        try:
            result = build_extractor_identity(
                kind=args.kind,
                name=args.name,
                revision=args.revision,
                artifacts=_artifact_declarations(args.artifact),
            )
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "build-speaker-reference":
        try:
            result = build_speaker_reference_binding(
                reference_id=args.reference_id,
                audio_path=args.audio,
                transcript_path=args.transcript,
            )
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "apply-extractor-results":
        try:
            artifacts = _artifact_declarations(args.extractor_artifact) if args.extractor_artifact else None
            result = apply_extractor_results(
                _read_json(args.observations),
                _read_json(args.results),
                audio_base_dir=args.audio_base_dir,
                extractor_artifacts=artifacts,
                reference_audio_path=args.reference_audio,
                reference_transcript_path=args.reference_transcript,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "fingerprint-observations":
        try:
            result = observation_document_sha256(_read_json(args.observations))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        print(result)
        return 0
    if args.command == "aggregate-listening":
        try:
            result = aggregate_listening_results(
                _read_json(args.review),
                _read_json(args.reveal),
                _read_json(args.ratings),
                seed=args.seed,
                allow_incomplete=args.allow_incomplete,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "compare-matched":
        try:
            rows = _read_json(args.observations)
            if not isinstance(rows, list):
                raise ValueError("objective observations must be a JSON array")
            result = compare_matched_candidates(
                rows,
                plan=_read_json(args.plan),
                baseline_candidate_id=args.baseline,
                adapted_candidate_id=args.adapted,
                seed=args.seed,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "compare-runtimes":
        try:
            rows = _read_json(args.observations)
            if not isinstance(rows, list):
                raise ValueError("objective observations must be a JSON array")
            result = compare_runtime_candidates(
                rows,
                plan=_read_json(args.plan),
                artifact_manifest=_read_json(args.artifact_manifest),
                artifact_binding_plan=_read_json(args.artifact_binding_plan),
                artifact_base_dir=args.artifact_binding_plan.parent.resolve(),
                reference_candidate_id=args.reference_candidate,
                candidate_candidate_id=args.candidate,
                reference_runtime_id=args.reference_runtime,
                candidate_runtime_id=args.candidate_runtime,
                seed=args.seed,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "validate-backend":
        try:
            errors = validate_backend_spec(_read_json(args.spec), spec_path=args.spec)
        except (OSError, json.JSONDecodeError) as error:
            print(error, file=sys.stderr)
            return 2
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"valid lifecycle backend: {args.spec}")
        return 0
    if args.command == "validate-backend-registry":
        try:
            errors = validate_backend_registry(_read_json(args.registry), registry_path=args.registry)
        except (OSError, json.JSONDecodeError) as error:
            print(error, file=sys.stderr)
            return 2
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"valid lifecycle backend registry: {args.registry}")
        return 0
    if args.command == "run-lifecycle":
        try:
            result = run_lifecycle(args.backend, args.experiment, args.work_dir)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    if args.command == "run-registered-lifecycle":
        try:
            result = run_registered_lifecycle(
                args.registry,
                args.experiment,
                args.work_dir,
                backend_id=args.backend_id,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    if args.command == "validate-prompt-pack":
        try:
            errors = validate_prompt_pack(_read_json(args.path))
        except (OSError, json.JSONDecodeError) as error:
            print(error, file=sys.stderr)
            return 2
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(f"valid prompt pack: {args.path}")
        return 0
    if args.command == "build-generation-plan":
        try:
            result = build_generation_plan(
                _read_json(args.prompt_pack),
                args.candidate,
                seeds=args.seed,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0
    if args.command == "check-suite-coverage":
        try:
            result = check_suite_coverage(_read_json(args.plan), _read_json(args.observations))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.output, result)
        return 0 if result["status"] == "passed" else 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
