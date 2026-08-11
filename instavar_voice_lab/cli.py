from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audio_probe import probe_wav
from .contracts import VALIDATORS, validate_document
from .corpus import audit_corpus
from .listening import build_blind_pack


def _read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

    repository = commands.add_parser("validate-repository", help="validate instavar-voice-capabilities.json in a repository")
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

    probe = commands.add_parser("probe-audio", help="record deterministic diagnostics for a PCM WAV file")
    probe.add_argument("wav", type=Path)
    probe.add_argument("--output", type=Path)

    blind = commands.add_parser("build-listening-pack", help="create blind review and reveal mapping documents")
    blind.add_argument("samples", type=Path, help="JSON array of sample_id, candidate_id, prompt_id, and audio_path rows")
    blind.add_argument("--criteria", type=Path, required=True, help="JSON array of criterion names")
    blind.add_argument("--review-output", type=Path, required=True)
    blind.add_argument("--reveal-output", type=Path, required=True)
    blind.add_argument("--seed", type=int, required=True)
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
    if args.command == "build-listening-pack":
        try:
            samples = _read_json(args.samples)
            criteria = _read_json(args.criteria)
            review, reveal = build_blind_pack(samples, criteria, seed=args.seed)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(error, file=sys.stderr)
            return 2
        _write_json(args.review_output, review)
        _write_json(args.reveal_output, reveal)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
