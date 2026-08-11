from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def _decode_samples(data: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [value - 128 for value in data]
    if sample_width == 2:
        return list(struct.unpack(f"<{len(data) // 2}h", data))
    if sample_width == 3:
        samples: list[int] = []
        for offset in range(0, len(data), 3):
            raw = int.from_bytes(data[offset : offset + 3], "little", signed=False)
            samples.append(raw - (1 << 24) if raw & (1 << 23) else raw)
        return samples
    if sample_width == 4:
        return list(struct.unpack(f"<{len(data) // 4}i", data))
    raise ValueError(f"unsupported PCM sample width: {sample_width}")


def probe_wav(path: Path, *, silence_threshold: float = 0.01, clipping_threshold: float = 0.999) -> dict[str, float | int | str]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        compression = source.getcomptype()
        data = source.readframes(frame_count)

    if compression != "NONE":
        raise ValueError("only uncompressed PCM WAV files are supported")
    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV must contain at least one valid audio frame")
    samples = _decode_samples(data, sample_width)
    if not samples:
        raise ValueError("WAV contains no PCM samples")

    full_scale = float((1 << (sample_width * 8 - 1)) - 1)
    normalized = [sample / full_scale for sample in samples]
    peak = max(abs(sample) for sample in normalized)
    rms = math.sqrt(sum(sample * sample for sample in normalized) / len(normalized))
    dc_offset = sum(normalized) / len(normalized)
    silence_fraction = sum(abs(sample) <= silence_threshold for sample in normalized) / len(normalized)
    clipping_fraction = sum(abs(sample) >= clipping_threshold for sample in normalized) / len(normalized)
    return {
        "path": str(path),
        "duration_seconds": frame_count / sample_rate,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "peak": peak,
        "rms": rms,
        "dc_offset": dc_offset,
        "silence_fraction": silence_fraction,
        "clipping_fraction": clipping_fraction,
    }
