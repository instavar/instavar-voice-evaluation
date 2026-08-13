from __future__ import annotations

import math
from pathlib import Path
import statistics
import wave

from .audio_probe import _decode_samples


PROSODY_PROXY_SCHEMA_VERSION = "instavar_voice_prosody_proxy/v1"
PROXY_FIELDS = (
    "active_frame_fraction",
    "active_rms_db_std",
    "window_rms_db_std",
    "zero_crossing_rate_hz_std",
    "pause_rate_per_minute",
    "pause_fraction",
    "pause_duration_cv",
    "phrase_duration_cv",
)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) >= 2 else None


def _cv(values: list[float]) -> float | None:
    mean = _mean(values)
    if mean is None or mean <= 0 or len(values) < 2:
        return None
    return statistics.pstdev(values) / mean


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _runs(mask: list[bool], hop_seconds: float, selected: bool) -> list[float]:
    durations: list[float] = []
    count = 0
    for value in mask + [not selected]:
        if value is selected:
            count += 1
        elif count:
            durations.append(count * hop_seconds)
            count = 0
    return durations


def _pause_mask(active: list[bool], hop_seconds: float, minimum_pause_seconds: float) -> list[bool]:
    pauses = [False] * len(active)
    start: int | None = None
    seen_active = False
    future_active = [False] * (len(active) + 1)
    for index in range(len(active) - 1, -1, -1):
        future_active[index] = future_active[index + 1] or active[index]
    for index, is_active in enumerate(active + [True]):
        if not is_active and start is None:
            start = index
        elif is_active and start is not None:
            if (
                seen_active
                and future_active[index]
                and (index - start) * hop_seconds >= minimum_pause_seconds
            ):
                pauses[start:index] = [True] * (index - start)
            start = None
        if is_active and index < len(active):
            seen_active = True
    return pauses


def probe_prosody_proxy(
    path: Path,
    *,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    minimum_pause_ms: float = 120.0,
    window_seconds: float = 2.0,
    long_form_seconds: float = 30.0,
) -> dict[str, object]:
    """Measure deterministic waveform proxies without making a perceptual claim."""
    if frame_ms <= 0 or hop_ms <= 0 or hop_ms > frame_ms:
        raise ValueError("frame_ms and hop_ms must be positive, with hop_ms <= frame_ms")
    if minimum_pause_ms < hop_ms:
        raise ValueError("minimum_pause_ms must be at least one hop")
    if window_seconds <= 0 or long_form_seconds <= 0:
        raise ValueError("window_seconds and long_form_seconds must be positive")

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        compression = source.getcomptype()
        if compression != "NONE":
            raise ValueError("only uncompressed PCM WAV files are supported")
        if channels != 1:
            raise ValueError("prosody proxy requires mono PCM WAV input")
        if sample_rate <= 0 or frame_count <= 0:
            raise ValueError("WAV must contain at least one valid audio frame")
        frame_samples = max(1, round(frame_ms * sample_rate / 1000.0))
        hop_samples = max(1, round(hop_ms * sample_rate / 1000.0))
        window_samples = max(1, round(window_seconds * sample_rate))
        if frame_count < frame_samples:
            raise ValueError("WAV is shorter than one analysis frame")

        full_scale = float((1 << (sample_width * 8 - 1)) - 1)
        buffer: list[float] = []
        buffer_start = 0
        frame_rms: list[float] = []
        frame_zcr: list[float] = []
        window_db: list[float] = []
        window_sum_squares = 0.0
        window_count = 0
        decoded_count = 0
        while True:
            data = source.readframes(sample_rate)
            if not data:
                break
            decoded = _decode_samples(data, sample_width)
            decoded_count += len(decoded)
            normalized = [sample / full_scale for sample in decoded]
            for sample in normalized:
                window_sum_squares += sample * sample
                window_count += 1
                if window_count == window_samples:
                    window_db.append(_dbfs(math.sqrt(window_sum_squares / window_count)))
                    window_sum_squares = 0.0
                    window_count = 0
            buffer.extend(normalized)
            while len(buffer) - buffer_start >= frame_samples:
                frame = buffer[buffer_start : buffer_start + frame_samples]
                rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
                crossings = sum(
                    (left < 0 <= right) or (right < 0 <= left)
                    for left, right in zip(frame, frame[1:])
                )
                frame_rms.append(rms)
                frame_zcr.append(crossings / (len(frame) / sample_rate))
                buffer_start += hop_samples
            if buffer_start:
                buffer = buffer[buffer_start:]
                buffer_start = 0
        if decoded_count != frame_count:
            raise ValueError("WAV frame count does not match the decoded PCM payload")
        if window_count >= window_samples // 2:
            window_db.append(_dbfs(math.sqrt(window_sum_squares / window_count)))

    sorted_rms = sorted(frame_rms)
    noise_index = min(len(sorted_rms) - 1, int(len(sorted_rms) * 0.2))
    noise_floor = sorted_rms[noise_index]
    peak_rms = max(frame_rms)
    adaptive_noise_threshold = min(noise_floor * 3.0, peak_rms * 0.5)
    threshold = max(10 ** (-50 / 20), adaptive_noise_threshold, peak_rms * 0.04)
    active = [value > threshold for value in frame_rms]
    active_indices = [index for index, value in enumerate(active) if value]
    duration_seconds = frame_count / sample_rate
    hop_seconds = hop_samples / sample_rate
    pause_mask = _pause_mask(active, hop_seconds, minimum_pause_ms / 1000.0)
    pause_durations = _runs(pause_mask, hop_seconds, True)
    first_active = active_indices[0] if active_indices else None
    last_active = active_indices[-1] if active_indices else None
    phrase_mask = [
        first_active is not None
        and last_active is not None
        and first_active <= index <= last_active
        and not is_pause
        for index, is_pause in enumerate(pause_mask)
    ]
    phrase_durations = _runs(phrase_mask, hop_seconds, True)
    active_db = [_dbfs(frame_rms[index]) for index in active_indices]
    active_zcr = [frame_zcr[index] for index in active_indices]
    status = "complete" if len(active_indices) >= 5 else "insufficient_activity"

    metrics: dict[str, float | int | bool | None] = {
        "analysis_duration_seconds": duration_seconds,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "eligible_for_long_form": duration_seconds >= long_form_seconds,
        "analyzed_frame_count": len(frame_rms),
        "active_frame_count": len(active_indices),
        "active_frame_fraction": len(active_indices) / len(frame_rms),
        "activity_threshold_dbfs": _dbfs(threshold),
        "active_rms_db_mean": _mean(active_db),
        "active_rms_db_std": _std(active_db),
        "window_rms_db_std": _std(window_db),
        "zero_crossing_rate_hz_mean": _mean(active_zcr),
        "zero_crossing_rate_hz_std": _std(active_zcr),
        "pause_count": len(pause_durations),
        "pause_rate_per_minute": len(pause_durations) * 60.0 / duration_seconds,
        "pause_fraction": sum(pause_mask) / len(pause_mask),
        "pause_duration_mean": _mean(pause_durations),
        "pause_duration_std": _std(pause_durations),
        "pause_duration_cv": _cv(pause_durations),
        "leading_inactive_seconds": first_active * hop_seconds if first_active is not None else duration_seconds,
        "trailing_inactive_seconds": (
            (len(active) - 1 - last_active) * hop_seconds if last_active is not None else duration_seconds
        ),
        "phrase_count": len(phrase_durations),
        "phrase_duration_mean": _mean(phrase_durations),
        "phrase_duration_std": _std(phrase_durations),
        "phrase_duration_cv": _cv(phrase_durations),
    }
    return {
        "schema_version": PROSODY_PROXY_SCHEMA_VERSION,
        "status": status,
        "path": str(path),
        "configuration": {
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "minimum_pause_ms": minimum_pause_ms,
            "window_seconds": window_seconds,
            "long_form_seconds": long_form_seconds,
        },
        "metrics": metrics,
        "claims": {
            "proves_matched_text": False,
            "proves_accent_fidelity": False,
            "proves_naturalness": False,
            "proves_cadence_quality": False,
            "proves_long_form_monotony": False,
        },
        "evidence_boundary": (
            "Waveform energy, pause, phrase-duration, and zero-crossing proxies only. "
            "Use matched text and blinded listening for cadence, monotony, accent, and naturalness claims."
        ),
    }


def compare_prosody_proxies(reference: Path, candidate: Path) -> dict[str, object]:
    reference_probe = probe_prosody_proxy(reference)
    candidate_probe = probe_prosody_proxy(candidate)
    if reference_probe["status"] != "complete" or candidate_probe["status"] != "complete":
        raise ValueError("both prosody proxies require at least five active frames")
    reference_metrics = reference_probe["metrics"]
    candidate_metrics = candidate_probe["metrics"]
    assert isinstance(reference_metrics, dict) and isinstance(candidate_metrics, dict)
    deltas: dict[str, float | None] = {}
    for field in PROXY_FIELDS:
        reference_value = reference_metrics[field]
        candidate_value = candidate_metrics[field]
        deltas[field] = (
            float(candidate_value) - float(reference_value)
            if isinstance(reference_value, (int, float))
            and not isinstance(reference_value, bool)
            and isinstance(candidate_value, (int, float))
            and not isinstance(candidate_value, bool)
            else None
        )
    return {
        "schema_version": PROSODY_PROXY_SCHEMA_VERSION,
        "comparison_scope": "matched_waveform_prosody_proxies_only",
        "reference": reference_probe,
        "candidate": candidate_probe,
        "candidate_minus_reference": deltas,
        "claims": {
            "proves_matched_text": False,
            "proves_accent_fidelity": False,
            "proves_naturalness": False,
            "proves_cadence_quality": False,
            "proves_long_form_monotony": False,
        },
        "format_match": {
            field: reference_metrics[field] == candidate_metrics[field]
            for field in ("sample_rate_hz", "sample_width_bytes")
        },
        "evidence_boundary": (
            "The caller must bind identical requested text and comparable rendering conditions. "
            "Proxy deltas prioritize review and do not establish a perceptual difference or cause."
        ),
    }
