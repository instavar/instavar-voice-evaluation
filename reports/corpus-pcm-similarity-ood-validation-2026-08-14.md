# Corpus PCM similarity OOD validation

Date: 2026-08-14, Asia/Singapore

## Finding

Exact path and byte hashing cannot surface the same utterance after ordinary
WAV transformations. A recording can cross train, validation, and test after a
level change, silence padding, sample-rate conversion, or bit-depth conversion
while producing a new SHA-256 digest. This leaves held-out adaptation evidence
vulnerable to a class of accidental split leakage.

Version 0.46 adds an opt-in, dependency-free PCM similarity review to
`audit-corpus`. It computes a bounded relative energy and per-channel
zero-crossing envelope from uncompressed PCM WAV files at a fixed analysis
cadence no higher than 2 kHz. Low-activity boundary bins are trimmed, the active
envelope is mapped to 64 relative-time bins, and energy is normalized before
conservative cross-split candidate retrieval. The exact content hash still
reads every byte.
Detailed comparisons run only when two rows have an active-duration ratio of
at least 0.92 and share at least three coarse envelope bands. This reduces
normal work from an unconditional all-pairs scan while a 100,000-comparison
ceiling bounds pathological collisions.

The result is deliberately advisory. Candidate rows are emitted as
`review_required_not_proven_duplicate`, remain warnings, and do not fail an
otherwise valid corpus audit. The report also records eligible rows, skipped
rows grouped by reason, candidate and comparison ceilings, and whether either
ceiling was reached. It always emits `proves_duplicate_audio: false`.

## OOD controls

Dependency-free tests cover:

- one synthetic utterance transformed across level, leading and trailing
  silence, 16-bit to 8-bit sample width, and 8 kHz to 12 kHz sample-rate
  changes;
- distinct frequency and amplitude-envelope controls that must not be flagged;
- unsupported non-WAV rows, which remain explicit skips without changing the
  exact audit verdict;
- exact byte-copy rejection, which remains a fail-closed error before the
  advisory check;
- manifest mutation, malformed UTF-8, resource ceilings, group leakage, and
  transcript normalization from the pre-existing corpus audit suite; and
- filename-derived group extraction across raw and `vocal_...reformatted`
  naming conventions; and
- the full evaluator test suite, which passed 255 tests locally.

## Real corpus scale qualification

The final implementation audited the retained 90/5/5 FEMALE_01 VoxCPM2 split
on `desktop_tailscale`:

- 10,776 train rows, 599 validation rows, and 598 test rows;
- 11,973 eligible PCM WAV files and zero skipped files;
- 56.67 seconds wall time, 55.90 seconds user CPU, and 58,940 KiB peak RSS;
- 64,390 detailed pair comparisons, below the 100,000 ceiling;
- 441 advisory candidates, below the 1,000-candidate ceiling; and
- no reached ceiling, so the advisory scan completed its declared search.

The exact audit passed with zero errors. It also retained 849 pre-existing text
duplicate warnings. The PCM review added 441 warnings. Post-hoc inspection of
filename provenance found that all 441 candidate pairs carried the same
four-digit source recording ID across splits. Every candidate pair also had
overlapping source sample intervals. The shorter interval overlap ranged from
0.913242 to 1.0, with a median of 1.0. Two pairs had exactly identical interval
bounds. This post-hoc mapping supports the candidates, but it is not part of
the generic acoustic heuristic.

A separate fail-closed audit used the dataset-specific pattern
`^(?:(?:vocal|instrument)_)?([0-9]{4})\.wav`. It found 1,036 cross-split row errors spanning
988 unique derived groups: 503 validation rows and 533 test rows. The audit
finished in 2.41 seconds with 35,044 KiB peak RSS. This contradicts the older
split receipt's zero parent-group-overlap statement. The older receipt grouped
by stripping only the final slice suffix, so raw names such as `5325.wav` and
reformatted names such as `vocal_5325.wav.reformatted.wav_10.wav` remained
different groups even though they share source identity.

Retained evidence:

- PCM audit:
  `/mnt/work/chee-wei-jie/voice-models/instavar-voice-eval-pcm-20260814/real-corpus-audit.json`,
  SHA-256 `4792bdf09160ab151281069b048ba06cf7f4cc3f9241e0cb379f0192d114f541`
- derived-group audit:
  `/mnt/work/chee-wei-jie/voice-models/instavar-voice-eval-pcm-20260814/real-corpus-group-audit.json`,
  SHA-256 `82267d54089657fb233a967bf506a6b0a37bddd8c7ce23e718d7f4da3db6c7de`

The result invalidates a broad held-out-split claim for these exact manifests.
It does not show that every candidate was used in training, quantify model
memorization, or establish the effect on any particular metric. Regenerate the
split under a source-ID rule that unifies raw and reformatted naming families,
then rerun training and matched evaluation before using held-out adaptation
claims from this corpus.

## Scope and boundary

The implementation applies to local uncompressed PCM WAV inputs with sample
widths from 8 to 32 bits and a declared PCM payload no larger than 512 MiB. It
streams sample frames and verifies file identity before and after fingerprint
construction. It does not decode MP3, AAC, Opus, FLAC, or compressed WAV. It
does not establish that a candidate pair contains the same speaker, words, or
recording.

Relative energy and zero-crossing envelopes can collide for unrelated speech
with similar activity patterns. Conversely, codec loss, internal edits, noise,
time stretching, pitch changes, channel transformations, and aggressive
resampling can evade the detector. A clean result means only that this bounded
heuristic did not surface a candidate. Candidate confirmation still requires
stronger acoustic matching, provenance review, source-group metadata, or human
inspection.

The method generalizes as a conservative triage layer for TTS, ASR, speaker,
and other speech-corpus splits. Threshold adequacy, false-positive rate, and
false-negative rate remain dataset-specific and are not established by the
synthetic controls.
