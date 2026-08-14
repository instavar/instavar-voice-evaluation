from __future__ import annotations

import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_PCM_BYTES = 512 * 1024 * 1024
_SOURCE_BINS = 256
_FINGERPRINT_BINS = 64
_BAND_SIZE = 8
_SILENCE_FRACTION = 0.02
_MIN_ACTIVE_BINS = 8
_ANALYSIS_RATE_HZ = 2_000
_MAX_ZCR_HZ = _ANALYSIS_RATE_HZ / 2


@dataclass(frozen=True)
class PcmSimilarityFingerprint:
    energy: tuple[int, ...]
    zero_crossing: tuple[int, ...]
    active_duration_seconds: float
    bands: tuple[str, ...]


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _decode_sample(raw: bytes, offset: int, width: int) -> int:
    if width == 1:
        return raw[offset] - 128
    if width == 2:
        return int.from_bytes(raw[offset : offset + 2], "little", signed=True)
    if width == 3:
        value = int.from_bytes(raw[offset : offset + 3], "little", signed=False)
        return value - (1 << 24) if value & (1 << 23) else value
    if width == 4:
        return int.from_bytes(raw[offset : offset + 4], "little", signed=True)
    raise ValueError("PCM sample width must be between 1 and 4 bytes")


def _resample_relative(values: list[float], count: int) -> list[float]:
    if len(values) == 1:
        return [values[0]] * count
    result: list[float] = []
    for target in range(count):
        position = target * (len(values) - 1) / (count - 1)
        left = int(math.floor(position))
        right = min(left + 1, len(values) - 1)
        fraction = position - left
        result.append(values[left] * (1.0 - fraction) + values[right] * fraction)
    return result


def _band_hashes(energy: tuple[int, ...], zero_crossing: tuple[int, ...]) -> tuple[str, ...]:
    bands: list[str] = []
    for start in range(0, len(energy), _BAND_SIZE):
        energy_band = energy[start : start + _BAND_SIZE]
        zcr_band = zero_crossing[start : start + _BAND_SIZE]
        coarse_energy = round(sum(energy_band) / len(energy_band) / 4)
        coarse_zcr = round(sum(zcr_band) / len(zcr_band) / 4)
        bands.append(f"e{coarse_energy}:z{coarse_zcr}")
    return tuple(bands)


def fingerprint_pcm_wav(
    path: Path,
    *,
    max_pcm_bytes: int = DEFAULT_MAX_PCM_BYTES,
    expected_stat_fingerprint: tuple[int, int, int, int] | None = None,
) -> PcmSimilarityFingerprint:
    if isinstance(max_pcm_bytes, bool) or not isinstance(max_pcm_bytes, int) or max_pcm_bytes <= 0:
        raise ValueError("max_pcm_bytes must be a positive integer")
    if max_pcm_bytes > DEFAULT_MAX_PCM_BYTES:
        raise ValueError(f"max_pcm_bytes cannot exceed the {DEFAULT_MAX_PCM_BYTES}-byte safety ceiling")

    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if expected_stat_fingerprint is not None and _stat_fingerprint(before) != expected_stat_fingerprint:
            raise ValueError("audio file changed between content hashing and PCM similarity review")
        try:
            reader = wave.open(source, "rb")
        except (EOFError, wave.Error) as error:
            raise ValueError(f"not a supported PCM WAV: {error}") from error
        with reader:
            if reader.getcomptype() != "NONE":
                raise ValueError("not an uncompressed PCM WAV")
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
                raise ValueError("PCM WAV has invalid channel, sample-rate, or frame-count metadata")
            if sample_width not in {1, 2, 3, 4}:
                raise ValueError("PCM WAV sample width must be between 1 and 4 bytes")
            declared_pcm_bytes = frame_count * channels * sample_width
            if declared_pcm_bytes > max_pcm_bytes:
                raise ValueError(
                    f"PCM payload declares {declared_pcm_bytes} bytes, above the {max_pcm_bytes}-byte limit"
                )

            energy_sums = [0.0] * _SOURCE_BINS
            frame_counts = [0] * _SOURCE_BINS
            zero_crossings = [0.0] * _SOURCE_BINS
            max_magnitude = float(1 << (sample_width * 8 - 1))
            frame_index = 0
            previous_signs = [0] * channels
            analysis_stride = max(1, sample_rate // _ANALYSIS_RATE_HZ)
            while frame_index < frame_count:
                requested = min(4096, frame_count - frame_index)
                raw = reader.readframes(requested)
                frame_bytes = channels * sample_width
                if len(raw) != requested * frame_bytes:
                    raise ValueError("PCM WAV payload is truncated")
                first_local_frame = (-frame_index) % analysis_stride
                for local_frame in range(first_local_frame, requested, analysis_stride):
                    frame_offset = local_frame * frame_bytes
                    magnitude_sum = 0.0
                    frame_crossings = 0
                    for channel in range(channels):
                        sample_offset = frame_offset + channel * sample_width
                        sample = _decode_sample(raw, sample_offset, sample_width) / max_magnitude
                        magnitude_sum += abs(sample)
                        sign = 1 if sample > 0 else -1 if sample < 0 else previous_signs[channel]
                        if previous_signs[channel] and sign and sign != previous_signs[channel]:
                            frame_crossings += 1
                        if sign:
                            previous_signs[channel] = sign
                    energy = magnitude_sum / channels
                    observed_frame = frame_index + local_frame
                    bucket = min(_SOURCE_BINS - 1, observed_frame * _SOURCE_BINS // frame_count)
                    energy_sums[bucket] += energy
                    frame_counts[bucket] += 1
                    zero_crossings[bucket] += frame_crossings / channels
                frame_index += requested

        after = os.fstat(source.fileno())
    current = path.stat()
    if _stat_fingerprint(before) != _stat_fingerprint(after) or _stat_fingerprint(after) != _stat_fingerprint(current):
        raise ValueError("audio file changed while its PCM similarity fingerprint was computed")

    energy_values = [
        energy_sums[index] / frame_counts[index] if frame_counts[index] else 0.0 for index in range(_SOURCE_BINS)
    ]
    peak = max(energy_values)
    if peak <= 0.0:
        raise ValueError("PCM WAV has no measurable signal activity")
    active = [index for index, value in enumerate(energy_values) if value >= peak * _SILENCE_FRACTION]
    if len(active) < _MIN_ACTIVE_BINS:
        raise ValueError("PCM WAV has too little active signal for similarity review")
    first, last = active[0], active[-1]
    trimmed_energy = energy_values[first : last + 1]
    zcr_values: list[float] = []
    for index in range(_SOURCE_BINS):
        first_frame = index * frame_count // _SOURCE_BINS
        next_frame = (index + 1) * frame_count // _SOURCE_BINS
        duration = (next_frame - first_frame) / sample_rate
        zcr_values.append(zero_crossings[index] / duration if duration > 0 else 0.0)
    trimmed_zcr = zcr_values[first : last + 1]
    normalized_energy = _resample_relative([value / peak for value in trimmed_energy], _FINGERPRINT_BINS)
    normalized_zcr = _resample_relative([min(1.0, value / _MAX_ZCR_HZ) for value in trimmed_zcr], _FINGERPRINT_BINS)
    quantized_energy = tuple(max(0, min(31, round(value * 31))) for value in normalized_energy)
    quantized_zcr = tuple(max(0, min(31, round(value * 31))) for value in normalized_zcr)
    active_duration = frame_count / sample_rate * (last - first + 1) / _SOURCE_BINS
    return PcmSimilarityFingerprint(
        energy=quantized_energy,
        zero_crossing=quantized_zcr,
        active_duration_seconds=active_duration,
        bands=_band_hashes(quantized_energy, quantized_zcr),
    )


def compare_pcm_fingerprints(
    left: PcmSimilarityFingerprint,
    right: PcmSimilarityFingerprint,
) -> dict[str, Any]:
    duration_ratio = min(left.active_duration_seconds, right.active_duration_seconds) / max(
        left.active_duration_seconds,
        right.active_duration_seconds,
    )
    energy_distance = sum(abs(a - b) for a, b in zip(left.energy, right.energy)) / (31 * len(left.energy))
    zcr_distance = sum(abs(a - b) for a, b in zip(left.zero_crossing, right.zero_crossing)) / (
        31 * len(left.zero_crossing)
    )
    distance = 0.6 * energy_distance + 0.4 * zcr_distance
    matching_bands = sum(a == b for a, b in zip(left.bands, right.bands))
    return {
        "similarity": round(1.0 - distance, 6),
        "duration_ratio": round(duration_ratio, 6),
        "matching_bands": matching_bands,
        "review_candidate": duration_ratio >= 0.92 and distance <= 0.08 and matching_bands >= 3,
    }
