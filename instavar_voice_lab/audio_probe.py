from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


DEFAULT_MAX_PCM_BYTES = 512 * 1024 * 1024
READ_FRAMES = 65_536


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


def probe_wav(
    path: Path,
    *,
    silence_threshold: float = 0.01,
    clipping_threshold: float = 0.999,
    max_pcm_bytes: int = DEFAULT_MAX_PCM_BYTES,
) -> dict[str, float | int | str]:
    if not math.isfinite(silence_threshold) or not 0 <= silence_threshold <= 1:
        raise ValueError("silence threshold must be finite and between 0 and 1")
    if not math.isfinite(clipping_threshold) or not 0 < clipping_threshold <= 1:
        raise ValueError("clipping threshold must be finite, greater than 0, and at most 1")
    if isinstance(max_pcm_bytes, bool) or not isinstance(max_pcm_bytes, int) or max_pcm_bytes < 1:
        raise ValueError("max PCM bytes must be a positive integer")

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        compression = source.getcomptype()
        if compression != "NONE":
            raise ValueError("only uncompressed PCM WAV files are supported")
        if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
            raise ValueError("WAV must contain at least one valid audio frame")
        if sample_width not in {1, 2, 3, 4}:
            raise ValueError(f"unsupported PCM sample width: {sample_width}")

        expected_pcm_bytes = frame_count * channels * sample_width
        if expected_pcm_bytes > max_pcm_bytes:
            raise ValueError(
                f"WAV PCM payload exceeds the {max_pcm_bytes}-byte analysis limit"
            )

        full_scale = float((1 << (sample_width * 8 - 1)) - 1)
        sample_count = 0
        peak = 0.0
        sum_samples = 0.0
        sum_squares = 0.0
        silent_count = 0
        clipped_count = 0
        while True:
            data = source.readframes(READ_FRAMES)
            if not data:
                break
            if len(data) % sample_width:
                raise ValueError("WAV contains a partial PCM sample")
            samples = _decode_samples(data, sample_width)
            for integer_sample in samples:
                sample = integer_sample / full_scale
                absolute = abs(sample)
                sample_count += 1
                peak = max(peak, absolute)
                sum_samples += sample
                sum_squares += sample * sample
                silent_count += absolute <= silence_threshold
                clipped_count += absolute >= clipping_threshold

    expected_sample_count = frame_count * channels
    if sample_count != expected_sample_count:
        raise ValueError(
            "WAV PCM payload is truncated: "
            f"expected {expected_sample_count} samples, decoded {sample_count}"
        )

    rms = math.sqrt(sum_squares / sample_count)
    dc_offset = sum_samples / sample_count
    silence_fraction = silent_count / sample_count
    clipping_fraction = clipped_count / sample_count
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


def compare_wav_probes(reference: Path, candidate: Path) -> dict[str, object]:
    """Compare deterministic WAV diagnostics without claiming runtime equivalence."""
    reference_probe = probe_wav(reference)
    candidate_probe = probe_wav(candidate)
    delta_fields = (
        "duration_seconds",
        "peak",
        "rms",
        "dc_offset",
        "silence_fraction",
        "clipping_fraction",
    )
    deltas = {
        field: float(candidate_probe[field]) - float(reference_probe[field])
        for field in delta_fields
    }
    return {
        "comparison_scope": "deterministic_container_and_level_diagnostics_only",
        "proves_runtime_equivalence": False,
        "reference": reference_probe,
        "candidate": candidate_probe,
        "format_match": {
            field: candidate_probe[field] == reference_probe[field]
            for field in ("sample_rate_hz", "channels", "sample_width_bytes")
        },
        "candidate_minus_reference": deltas,
    }
