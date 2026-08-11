from __future__ import annotations

import json
import math
import os
import struct
import sys
import wave
from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    samples = [int(0.15 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate)) for index in range(sample_rate)]
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def run(stage: str) -> int:
    work_dir = Path(os.environ["INSTAVAR_VOICE_WORK_DIR"])
    result_path = Path(os.environ["INSTAVAR_VOICE_STAGE_RESULT"])
    if stage == "preflight":
        _write_json(work_dir / "preflight" / "preflight.json", {"finite_loss": True, "reload": True})
    elif stage == "train":
        checkpoint = work_dir / "train" / "checkpoint.bin"
        checkpoint.write_bytes(b"instavar-voice-fake-checkpoint\n")
    elif stage == "infer":
        _write_wav(work_dir / "infer" / "candidate.wav")
    elif stage == "evaluate":
        _write_json(
            work_dir / "evaluate" / "objective-observations.json",
            [
                {
                    "sample_id": "fake-1",
                    "candidate_id": "fake-adapter",
                    "prompt_id": "fake-prompt",
                    "requested_text": "The fake backend verifies the lifecycle.",
                    "hypothesis_text": "The fake backend verifies the lifecycle.",
                    "valid": True,
                    "generation_seconds": 0.1,
                    "audio_duration_seconds": 1.0,
                    "peak_memory_bytes": 1024,
                    "reference_speaker_embedding": [1.0, 0.0],
                    "speaker_embedding": [1.0, 0.0],
                    "evidence": {
                        "asr": {"extractor": "fake", "revision": "test-only"},
                        "speaker_encoder": {"extractor": "fake", "revision": "test-only"},
                        "runtime": {"extractor": "fake", "revision": "test-only"},
                    },
                }
            ],
        )
    elif stage == "package":
        package = work_dir / "package" / "adapter-package.bin"
        package.write_bytes(b"instavar-voice-fake-package\n")
    else:
        raise ValueError(f"unknown fake backend stage: {stage}")
    _write_json(result_path, {"schema_version": "1.0.0", "stage": stage, "status": "passed"})
    return 0


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: python -m instavar_voice_lab.fake_backend STAGE", file=sys.stderr)
        return 2
    try:
        return run(values[0])
    except (KeyError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
