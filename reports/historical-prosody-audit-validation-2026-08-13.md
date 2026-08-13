# Historical prosody audit validation

Date: 2026-08-13

## Scope

This report covers evaluator 0.32.0 and the public implementation commits:

- `6a9586139ba423bea8f15997127657a981f4636f`, which introduced the historical
  unmatched-batch audit; and
- `f199b3ab877543be5a0702db17a321d4cb372267`, which replaced the
  caller-supplied extractor revision with a revision derived from the exact
  artifact-set digest.

The command exists for retained audio that predates strict objective
observations and a frozen generation plan. It preserves unknown metadata as
explicit null and cannot produce a ranking or adaptation comparison.

## Code validation

- 200 tests passed on Python 3.11.
- 200 tests passed on Python 3.14.
- Ruff 0.12.12 passed for `instavar_voice_lab` and `tests`.
- Focused Ruff formatting passed for the new runner, tests, and CLI.
- Python 3.11 and 3.14 bytecode compilation passed.
- The 0.32.0 source distribution and wheel built successfully.
- GitHub Actions Quality runs `31664365741` and `31664468941` passed. Their
  server-stamped creation times were `2026-08-13T03:35:02Z` and
  `2026-08-13T03:37:06Z`.

The focused OOD suite covers live audio hash drift, path escape, symlink
traversal, duplicate sample identity, omitted unknown fields, audio mutation
during analysis, extractor artifact drift, silent-input failure retention, and
CLI publication.

## Retained six-audio audit

The retained neutral-brief package named six audio files produced on 12 August
2026. Before analysis, every live WAV still matched its historical SHA-256.
The immutable unmatched manifest retained the known requested text and set the
unrecorded seed and runtime IDs to null. Nothing was inferred from the blind-pack
shuffle seed or narrative descriptions.

The final content-bound report recorded:

- source-manifest canonical digest
  `d1008b59ecfd294218d3deed317ee54cfe5d3af9307d3de5e83c2da126c9a1f3`;
- manifest file SHA-256
  `82383b940592537c018b3eb54745c02649c43d5e84089e1dc49f7f5069812428`;
- report file SHA-256
  `9c224f09173e09305836d087864a9d89df2045e05d6c4fdcd6e1c53433e405da`;
- extractor artifact-set SHA-256
  `3c85b04088b5f9362557b504dd8599053c0b2d9b0b924d36309ba4cc0e2f8807`;
- six complete analyses and zero extraction failures;
- six unknown seeds and six unknown runtime IDs; and
- zero long-form-eligible clips.

The six audio durations ranged from about 7.98 to 9.44 seconds. The report
contains per-file waveform proxy values but no mean, rank, quality direction,
winner, or adaptation-benefit field. Audio8 remains a base-runtime control while
the other five rows name selected adapted artifacts. That asymmetry is another
reason the archive cannot be treated as an adaptation ranking.

## OOD correction during real execution

The first real execution used a manually typed Git SHA as an extractor revision.
The typed value did not equal the actual commit, even though the report carried
the correct source-file hashes. An immutable-looking free-form label can still
misidentify exact bytes.

Evaluator 0.32 therefore derives the revision as
`artifact-set-sha256:<digest>` from the proxy, PCM decoder, and historical batch
runner. The report includes `revision_basis: artifact_set_sha256`, and tests
require the revision and artifact-set digest to agree. The corrected retained
archive report uses that automatic identity.

This finding generalizes to provenance records that already possess a stronger
machine-derived identity. Do not ask an operator to retype a commit, digest, or
version as a parallel authority when the exact content identity can be derived.
It does not eliminate the need for a signed build or trusted host when hostile
execution is in scope.

## Evidence boundary

This run establishes that the retained files existed at analysis time, matched
the declared digests, were processed by the exact recorded source artifacts,
and produced the stated deterministic proxy report. It is useful as historical
signal triage and as proof that the archive is too short for the current
long-form gate.

It does not establish the original generation seed, exact runtime ID, frozen
plan chronology, matched base-versus-adapted coverage, perceptual quality,
cadence quality, monotony, accent fidelity, speaker identity, naturalness,
preference, or adaptation benefit. The real long-form and blind-listening gates
remain open.
