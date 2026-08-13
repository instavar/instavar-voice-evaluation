# Matched prosody comparison validation

Date: 2026-08-13

## Scope

This report covers package version 0.30.0 and implementation commit
`2a98ab0a378850478e20cda39a1fc001e160aa24`. The change adds
`compare-matched-prosody` for plan-matched, content-bound cadence proxy
measurements.

The command reports signed signal deltas. It does not assign a quality
direction, emit a winner or composite score, or claim an adaptation benefit.

## Validation

- 194 unit and integration tests passed on Python 3.11.
- 194 unit and integration tests passed on Python 3.14.
- Ruff 0.12.12 passed for `instavar_voice_lab` and `tests`.
- Python 3.11 and 3.14 bytecode compilation passed.
- The 0.30.0 source distribution and wheel built successfully.
- GitHub Actions Quality run `31663581095` passed for the implementation commit.
  The server-stamped creation time was `2026-08-13T03:19:36Z`.

## Evaluated failure and OOD cases

The focused comparison suite verifies that the command rejects or preserves the
following cases:

- an empty comparison when every pair lacks complete bilateral proxy evidence;
- missing proxy fields on only one candidate;
- null variation fields being converted to zero;
- changed input-audio hashes;
- mixed extractor revisions or artifact-set hashes;
- a different requested text, prompt, seed, or plan row;
- candidate-specific plan category drift;
- proxy measurements attached to an invalid generated output;
- a row that declares both successful proxy evidence and extractor failure;
- malformed or content-unbound extractor failure records;
- selective disappearance of a planned candidate row;
- accidental introduction of a winner or directional-improvement field.

Invalid generated outputs and proxy extraction failures remain separate
coverage outcomes. A failed extractor record retains its error and recorded
extractor provenance in the per-pair report. At least one pair must have
complete content-bound evidence on both candidates before the command can pass.

## Evidence boundary

The report proves that the compared rows match the frozen generation plan by
sample, prompt, seed, requested text, and optional category. For complete rows,
it checks that proxy fields are present, the recorded input-audio digest matches
the observation, and all compared rows name one extractor revision and artifact
set.

The report does not reopen live audio or recompute the proxy. That work belongs
to `build-prosody-proxy-results` and `apply-extractor-results`, which bind and
recheck the source observation document, live WAV bytes, probe source, and PCM
decoder. The comparison consumes their augmented observations.

The proxy fields are not calibrated quality measures. More waveform energy
variation, pauses, phrase-duration variation, or zero crossings can be helpful,
harmful, or irrelevant depending on the text and speaking style. Signed deltas
can prioritize matched samples for criterion-specific blind listening. They do
not establish cadence quality, monotony, accent fidelity, speaker identity,
naturalness, preference, listening fatigue, or causation.
