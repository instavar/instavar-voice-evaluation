# Instavar Voice adaptation and evaluation contract

This directory implements the shared evidence layer for Instavar's TTS companion repositories. It does not contain a universal trainer. Model-specific repositories continue to own preprocessing, codec handling, LoRA or full-SFT training, checkpoint loading, and runtime integration.

The public distribution is [instavar/instavar-voice-evaluation](https://github.com/instavar/instavar-voice-evaluation). The copy under the private Instavar product repository keeps the application and its pinned contract version reviewable together.

The shared layer provides:

- versioned capability, experiment, evaluation, and artifact-package contracts;
- semantic validation using only the Python standard library;
- a frozen Singapore English prompt and listening-criteria pack;
- deterministic PCM WAV diagnostics;
- deterministic waveform prosody proxies with explicit perceptual boundaries;
- deterministic blind-review labels, identity-neutral staged filenames, and a separately stored reveal mapping;
- objective proxy scoring from versioned ASR, speaker-encoder, and runtime observations;
- a versioned objective-observation contract with stable identifiers and complete runtime-artifact bindings;
- criterion-level listening aggregation with bootstrap intervals and interval agreement; and
- a fail-closed lifecycle runner for model-specific preflight, training, inference, evaluation, and packaging;
- a frozen candidate by prompt by seed generation plan with completeness accounting;
- plan-bound objective metric requirements that reject bilateral metric omission;
- plan-bound lexical anchors with frozen ASR aliases and matched hit-rate deltas;
- exact baseline-versus-adapted pairing with extractor-provenance checks and paired bootstrap intervals;
- content-addressed extractor implementation or model artifacts plus speaker-reference audio and transcript bindings;
- deterministic multi-reference speaker scoring with per-sample reference-set binding;
- frozen per-prompt and per-seed speaker-reference assignments bound to generation plans and reference catalogs;
- an optional first-party SpeechBrain ECAPA execution path with CPU and CUDA routing, runtime provenance, and a content-addressed execution receipt;
- an optional first-party faster-whisper execution path with offline model loading, frozen decoding, and content-addressed ASR receipts;
- a plan-bound content-faithfulness report that keeps requested-text error, repeated n-gram excess, retained-reference transcript overlap, and spoken conditioning-instruction overlap separate;
- content-addressed generation-attempt receipts for runtime timing, duration, and memory evidence;
- exact-versus-derived cross-runtime artifact manifests with live content rechecks;
- lifecycle-stage and matched-comparison declarations for every supported adaptation path;
- examples and unit tests for every contract.

## Build and verify a multi-prompt generation plan

The Singapore English pack freezes seven prompts and three explicit seeds. Build
the complete matrix before generation so failed or omitted samples remain
visible:

```bash
instavar-voice-eval validate-prompt-pack reference/singapore-english-v1.json
instavar-voice-eval build-generation-plan reference/singapore-english-v1.json \
  --candidate base-model \
  --candidate selected-adapter \
  --output evaluation/generation-plan.json
```

For a bounded preregistered slice, repeat `--prompt` with exact prompt IDs from
the validated source pack and optionally repeat `--seed`:

```bash
instavar-voice-eval build-generation-plan reference/singapore-english-v1.json \
  --candidate base-model \
  --candidate selected-adapter \
  --prompt cadence-two-minute \
  --seed 20260812 \
  --output evaluation/long-form-plan.json
```

The builder validates the complete source pack before selecting prompts, binds
the complete pack hash, and records `selected_prompt_ids`. Unknown, duplicate,
or empty selections fail closed. A focused slice reduces execution cost but
does not establish full-suite coverage.

After every attempt has an objective-observation row, verify coverage:

```bash
instavar-voice-eval check-suite-coverage \
  evaluation/generation-plan.json \
  evaluation/objective-observations.json \
  --output evaluation/suite-coverage.json
```

Coverage passes when every planned sample has exactly one observation, including
failed generations recorded with `valid: false`. Coverage also verifies that
candidate, prompt, seed, and requested text match the plan instead of trusting a
matching sample ID alone. It does not convert signal, ASR, speaker, or runtime
measurements into a perceptual quality claim.

Generation plan 1.1 also carries the prompt pack's required objective metrics.
Matched comparison requires every valid pair to contain them. This prevents two
candidates from omitting the same difficult metric and producing a superficially
complete empty or timing-only comparison. Legacy plan 1.0 remains readable, but
its report states that required metric coverage was not enforced.

Prompt-pack version 1.2 can freeze optional lexical anchors on selected prompts.
Each anchor declares a token-bound surface present in the requested text and a
set of accepted ASR forms chosen before generation. The generation plan carries
the exact anchor set across candidates and seeds. Empty forms, normalized
duplicates, overlapping aliases, selective omission, and candidate-specific
alias drift fail closed. A surface must occur exactly once, and alternate aliases
must not already appear elsewhere in the prompt, so one unrelated phrase cannot
satisfy the anchor automatically.

## Validate contracts

Install the command in a virtual environment:

```bash
python3 -m pip install -e .
```

Then validate the example contracts:

```bash
instavar-voice-eval validate capability examples/capability-manifest.json
instavar-voice-eval validate experiment examples/experiment-manifest.json
instavar-voice-eval validate evaluation examples/evaluation-report.json
instavar-voice-eval validate package examples/artifact-package.json
instavar-voice-eval validate historical examples/historical-run.json
```

Validate a companion repository after it adds `instavar-voice-capabilities.json`:

```bash
instavar-voice-eval validate-repository /path/to/companion-repository
```

The checked-in JSON Schemas provide editor and ecosystem interoperability. The Python validator adds semantic checks that are awkward or misleading in schema alone, including evidence requirements for supported capabilities, unique runtime identifiers, distinct corpus split hashes, baseline presence, and the ban on a universal composite evaluation score.

Capability contract 1.1 adds deployment profiles, device, interface, precision,
batching, and explicit runtime-conformance coverage. A runtime that has not run
must record zero prompts and seeds. A completed smoke or validation must name
its report and record the tested prompt and seed counts. This keeps upstream
runtime availability separate from adapted-artifact evidence.

Capability contract 1.2 adds a seven-stage adaptation lifecycle and a matched
baseline-versus-adapted comparison declaration. Every supported or experimental
adaptation must state the evidence boundary for corpus audit, training,
checkpoint save, fresh-process reload, held-out inference, evaluation, and
packaging. Missing evidence stays visible as `not_recorded` or `blocked` rather
than being implied by a repository-level `supported` label.

Historical runs often predate the strict experiment and package contracts. Import them with the historical-run contract instead of inventing missing hashes. The record preserves stage-specific evidence and names the exact blockers that prevent migration into a complete experiment manifest or deployable artifact package.

## Probe generated audio

```bash
instavar-voice-eval probe-audio output.wav --output evaluation/output.probe.json
```

The deterministic probe reports duration, sample rate, channels, sample width,
peak, RMS, DC offset, silence fraction, and clipping fraction for uncompressed
PCM WAV files. Version 0.40 processes PCM in bounded chunks, rejects truncated
payloads, validates finite thresholds, and fails before decoding when the
declared PCM payload exceeds 512 MiB. Callers using the Python API can set a
smaller positive `max_pcm_bytes` limit. The default is a resource-safety bound,
not a statement about a useful maximum speech duration. The probe does not
measure intelligibility, speaker identity, accent fidelity, cadence, or
naturalness.

Implementation and OOD validation are recorded in
[`reports/audio-probe-streaming-ood-validation-2026-08-13.md`](reports/audio-probe-streaming-ood-validation-2026-08-13.md).

Compare the diagnostics from a reference runtime and a candidate runtime:

```bash
instavar-voice-eval compare-audio pytorch.wav alternative-runtime.wav \
  --output evaluation/runtime-comparison.json
```

The comparison records format matches and candidate-minus-reference deltas. It deliberately emits `proves_runtime_equivalence: false`: matching container and signal-level diagnostics cannot establish text, speaker, accent, cadence, or perceptual equivalence.

For matched-text cadence triage, measure waveform energy, pause, phrase-duration,
and zero-crossing variation:

```bash
instavar-voice-eval probe-prosody candidate.wav \
  --output evaluation/candidate.prosody-proxy.json

instavar-voice-eval compare-prosody baseline.wav candidate.wav \
  --output evaluation/matched.prosody-proxy.json
```

The proxy emits no composite score or good/bad threshold. It does not estimate
accent, phonemes, pitch, stress, naturalness, listening fatigue, or a monotony
verdict. Use it to surface matched outputs whose energy contours, pause timing,
phrase timing, or zero-crossing variation differ enough to prioritize blinded
listening. `eligible_for_long_form` means only that the WAV meets the default
30-second analysis duration; it is not a quality gate. Mono uncompressed PCM is
required so channel mixing cannot silently change the result.

Implementation `bdefbd76101653b2d05cc97d863f66b421316652` passed hosted
Quality run `31662790912` on 2026-08-13. A 120-second 48 kHz synthetic control
completed locally in 1.76 seconds with about 30 MB maximum resident memory.
These controls do not establish correlation with human cadence or monotony
ratings; that requires matched real speech and blinded listening.

For a batch that can be joined to the objective observation pipeline, bind every
proxy row to the source observation document, live audio bytes, and exact probe
implementation:

```bash
instavar-voice-eval build-prosody-proxy-results observations.json \
  --audio-base-dir evaluation/audio \
  --extractor-revision <immutable-evaluator-revision> \
  --output evaluation/prosody-results.json

instavar-voice-eval apply-extractor-results \
  observations.json evaluation/prosody-results.json \
  --audio-base-dir evaluation/audio \
  --output evaluation/observations-with-prosody.json
```

Extractor schema 1.6 records the probe and shared PCM decoder hashes. Applying
the receipt rechecks the observation document and each live WAV hash, refuses
field or evidence overwrite, and retains insufficient-activity rows as explicit
extractor failures. Content binding makes the measurements attributable; it
does not upgrade proxy measurements into perceptual evidence.

Compare an augmented baseline and adapted batch only after both candidates use
the same frozen generation plan:

```bash
instavar-voice-eval compare-matched-prosody \
  evaluation/observations-with-prosody.json \
  --plan evaluation/generation-plan.json \
  --baseline base-model \
  --adapted selected-adapter \
  --output evaluation/matched-prosody-comparison.json \
  --seed 20260812
```

The command requires exact prompt and seed coverage, identical planned text,
matching extractor revisions and artifact hashes, complete proxy fields on both
sides of every compared pair, and an input-audio hash match for every complete
row. Invalid generated outputs and prosody-extractor failures are counted
separately. The command fails instead of emitting an empty success when no pair
has complete evidence on both sides. Nullable variation fields remain null when
there is too little signal structure to calculate them; null is never converted
to zero.

Every numeric result is an adapted-minus-baseline signal delta with
`direction: not_established`. The report deliberately contains no composite
score, winner, or directional-improvement field. A larger pause, energy,
phrase-duration, or zero-crossing variation is not universally better. The
comparison can prioritize matched samples for blind review, but it cannot prove
better cadence, less monotony, accent fidelity, naturalness, preference, or an
adaptation benefit.

Implementation `2a98ab0a378850478e20cda39a1fc001e160aa24` passed hosted
Quality run `31663581095` on 2026-08-13. The focused validation and OOD cases are
recorded in
[`reports/prosody-comparison-validation-2026-08-13.md`](reports/prosody-comparison-validation-2026-08-13.md).

Historical audio often predates strict observation fields. Do not invent a seed,
runtime ID, or generation-plan binding to force that archive through schema 1.6.
Instead, create an explicit unmatched-triage manifest:

```json
{
  "schema_version": "instavar_voice_historical_prosody_manifest/v1",
  "batch_id": "legacy-neutral-brief",
  "purpose": "historical_unmatched_triage",
  "samples": [
    {
      "sample_id": "legacy-base-neutral",
      "candidate_id": "legacy-base",
      "prompt_id": "neutral-brief",
      "audio_path": "archive/legacy-base-neutral.wav",
      "audio_sha256": "<lowercase-sha256>",
      "requested_text": "A known historical passage.",
      "seed": null,
      "runtime_id": null
    }
  ]
}
```

Run the content-bound audit with a common parent directory for every relative
audio path:

```bash
instavar-voice-eval audit-historical-prosody historical-prosody-manifest.json \
  --audio-base-dir /absolute/archive/root \
  --output historical-prosody-report.json
```

The command rejects path escape, symlink traversal, unknown manifest fields,
duplicate sample IDs, hash drift, and audio or extractor mutation during
analysis. Missing historical metadata must be explicit null, not omitted or
guessed. The extractor revision is derived from the exact artifact-set digest,
so it cannot disagree with the proxy, PCM decoder, or historical batch runner.
The report also binds the manifest and live WAV bytes. It preserves per-file
failures and emits no ranking or aggregate quality score. Its output is
ineligible for matched adaptation comparison because a historical archive is
not a frozen generation plan.

Implementation `6a9586139ba423bea8f15997127657a981f4636f` and automatic
content-revision correction `f199b3ab877543be5a0702db17a321d4cb372267`
passed hosted Quality runs `31664365741` and `31664468941` on 2026-08-13. The
retained six-audio audit, OOD correction, exact digests, and claim boundary are
recorded in
[`reports/historical-prosody-audit-validation-2026-08-13.md`](reports/historical-prosody-audit-validation-2026-08-13.md).

## Bind artifacts across runtimes

Before comparing runtimes, write a local binding plan that names the source
artifact components and the files each runtime will consume:

```json
{
  "artifact_set_id": "female01-step-14000",
  "producer": {
    "repository": "instavar/indextts2-finetuning",
    "revision": "0123456789abcdef0123456789abcdef01234567"
  },
  "source_artifacts": [
    { "role": "checkpoint", "kind": "tree", "path": "/models/step-14000" }
  ],
  "runtime_bindings": [
    {
      "runtime_id": "pytorch",
      "relation": "exact",
      "artifacts": [
        { "role": "checkpoint", "kind": "tree", "path": "/models/step-14000" }
      ]
    },
    {
      "runtime_id": "mlx",
      "relation": "derived",
      "artifacts": [
        { "role": "checkpoint", "kind": "tree", "path": "/models/step-14000-mlx" }
      ],
      "conversion": { "tool": "example-converter", "revision": "1.0.0" }
    }
  ]
}
```

Build and verify the path-free public manifest:

```bash
instavar-voice-eval build-runtime-artifact-manifest runtime-binding-plan.json \
  --output runtime-artifact-manifest.json

instavar-voice-eval verify-runtime-artifact-manifest \
  runtime-artifact-manifest.json runtime-binding-plan.json \
  --report runtime-artifact-verification.json
```

An `exact` binding must have the same roles, kinds, sizes, and content hashes as
the source set. A `derived` binding records converter provenance and is barred
from exact-artifact comparison even if its output happens to have the same
bytes. The local plan may contain absolute paths; the emitted manifest does not.

Every runtime observation used below must record `runtime_id`,
`artifact_set_id`, and `artifact_set_sha256`. Compare matched rows with a live
artifact recheck:

```bash
instavar-voice-eval compare-runtimes objective-observations.json \
  --plan generation-plan.json \
  --artifact-manifest runtime-artifact-manifest.json \
  --artifact-binding-plan runtime-binding-plan.json \
  --reference-candidate voice-pytorch \
  --candidate voice-cuda-graph \
  --reference-runtime pytorch \
  --candidate-runtime cuda-graph \
  --output runtime-objective-comparison.json
```

A passing report proves current exact artifact fingerprints, frozen sample
pairing, and consistent extractor provenance. It still cannot prove that either
runtime loaded the declared bytes or that the audio is numerically or
perceptually equivalent. Converted MLX, ONNX, TensorRT, quantized, or merged
representations remain derived until evaluated under an explicit conversion
contract.

## Audit a training corpus

Audit file presence, non-empty text, duplicate audio, and parent or recording leakage before a training preflight:

```bash
instavar-voice-eval audit-corpus \
  --split train=data/train.jsonl \
  --split validation=data/validation.jsonl \
  --split test=data/test.jsonl \
  --group-field recording_id \
  --output evaluation/corpus-audit.json
```

The audit hashes each manifest and fails if a recording group crosses splits.
Version 0.41 also hashes every referenced audio file and rejects byte-identical
content even when copies, hard links, or different filenames make the paths
look distinct. The file identity is checked before and after hashing so a
concurrent replacement cannot leave a passing content digest. Transcript
duplicate warnings use NFKC normalization, whitespace folding, and casefolding
so compatibility-width Unicode cannot evade the warning. It checks that
referenced audio files exist, but it does not decode every audio format, detect
near-duplicate recordings, identify the same speaker across different clips,
or prove that the transcript matches the recording.

Version 0.42 emits corpus-audit schema 1.1 and hashes and parses each JSONL
manifest through the same open descriptor. It rejects replacement or mutation
during the audit, malformed UTF-8, manifests above 512 MiB, and logical JSONL
lines above 8 MiB. The Python API can tighten those two limits but cannot raise
them above the defaults. The CLI intentionally exposes no relaxation flags.

Implementation and OOD validation are recorded in
[`reports/corpus-content-leakage-ood-validation-2026-08-13.md`](reports/corpus-content-leakage-ood-validation-2026-08-13.md).
Manifest streaming and mutation validation are recorded in
[`reports/corpus-manifest-streaming-ood-validation-2026-08-13.md`](reports/corpus-manifest-streaming-ood-validation-2026-08-13.md).

Bind audited raw splits to model-ready artifacts with a content-addressed lineage receipt:

```bash
instavar-voice-eval build-dataset-lineage \
  --lineage-id female01-audio8-v1 \
  --producer-repository instavar/audio8-tts-lora-finetuning \
  --producer-revision "$COMPANION_REVISION" \
  --input raw_train=file=/data/raw-train.jsonl \
  --input raw_validation=file=/data/raw-validation.jsonl \
  --input raw_test=file=/data/raw-test.jsonl \
  --output-artifact prepared_train=file=/data/train.prepared.jsonl \
  --output-artifact prepared_validation=file=/data/validation.prepared.jsonl \
  --receipt dataset-lineage.json

instavar-voice-eval verify-dataset-lineage dataset-lineage.json \
  --producer-revision "$COMPANION_REVISION" \
  --input raw_train=file=/data/raw-train.jsonl \
  --input raw_validation=file=/data/raw-validation.jsonl \
  --input raw_test=file=/data/raw-test.jsonl \
  --output-artifact prepared_train=file=/data/train.prepared.jsonl \
  --output-artifact prepared_validation=file=/data/validation.prepared.jsonl
```

The receipt detects substitution and mutation of declared inputs and outputs. It does not prove that the preparation algorithm was semantically correct, so preparation code review and model-specific validation remain separate gates.

## Build a blind listening pack

For a preregistered study, first bind each listening criterion to prompts that
can support it. The checked-in Singapore English routing sends pronunciation
review to the dedicated pronunciation prompt and prompts with lexical anchors,
cadence and fatigue review to long-form prompts, emotion review to instructed
emotion prompts, and broadly applicable criteria to all samples:

```bash
instavar-voice-eval build-listening-assignment-plan evaluation/generation-plan.json \
  --routing reference/listening-routing-v1.json \
  --output evaluation/listening-assignment-plan.json
```

A deliberately focused generation plan can leave some full-suite routes without
samples. The default remains fail closed. For that explicit case, add
`--allow-unmatched-routes-for-focused-plan`. Assignment schema 1.3 retains the
complete routing, records `route_coverage_policy`, and lists every excluded
criterion in `excluded_unmatched_criteria`. Excluded criteria do not appear in
review assignments and cannot support a claim from that slice.

The builder rejects unmatched criteria, candidate-specific or seed-specific
text, instruction, category, or lexical-anchor drift, candidate-asymmetric
coverage, duplicate samples, and any sample left without a criterion. The
output binds the routing, assignments, exact requested text, optional
instruction, and reviewer-visible lexical target surfaces to the exact
generation-plan hash and includes a self-hash. Accepted ASR forms are excluded
from reviewer stimuli because recognition tolerances are not pronunciation
guidance. Hashes make later mutation detectable, but do not prove that
preregistration happened before generation without external server-stamped
chronology.

Every routed criterion also declares a reviewer question, low and high scale
anchors, and an explicit `higher_is_better` or `lower_is_better` direction.
This matters because a high naturalness score is favorable while a high
artifact-severity, monotony, or listening-fatigue score is unfavorable. The
same definitions are copied into the blind review and aggregate result. Raw
scores are not silently inverted, and distinct criteria are never collapsed
into an unvalidated composite.

Prepare a JSON array containing `sample_id`, `candidate_id`, `prompt_id`,
`seed`, and `audio_path` for every planned sample. Then run:

```bash
instavar-voice-eval build-listening-pack samples.json \
  --assignment-plan evaluation/listening-assignment-plan.json \
  --generation-plan evaluation/generation-plan.json \
  --rater-ids evaluation/pseudonymous-rater-ids.json \
  --review-output listening-review.json \
  --reveal-output reveal-mapping.json \
  --stage-root evaluation/listening \
  --seed 20260812
```

The rater file is a JSON array such as `["rater-001", "rater-002"]`. Use
stable pseudonyms rather than names or email addresses. The review file contains
no candidate identifiers or source filenames. Each
blind sample lists its generation-plan-bound stimulus and assigned criteria, so
a reviewer can judge exact wording and instruction obedience without consulting
an uncontrolled external prompt file. A visible lexical target does not
prescribe its correct pronunciation. A complete ratings matrix no longer
requires meaningless pronunciation scores for prompts without a target or
cadence scores for short clips. The aggregator rejects ratings for criteria
that were not assigned to that sample. With
`--stage-root`, audio is copied to paths such as
`blind_audio/sample-0001.wav`, and a hash manifest is written beside the staged
files. Preserve the reveal mapping separately and do not open it until all
ratings are recorded. File staging does not strip embedded audio metadata, so
inspect or normalize metadata separately if the source format can carry
identity-bearing tags. The builder rejects mixed audio extensions because a
format difference can itself reveal which runtime produced a sample.

With `--rater-ids`, the review also contains a deterministic schedule for each
rater. The scheduler randomizes prompt-and-seed block order per rater, separates
matched candidates into interleaved passes, and rotates candidate precedence
within every prompt and seed. Its private reveal audit requires every candidate
to occupy each within-prompt position with a count difference no larger than
one. Criterion-specific orders preserve the same master sequence while omitting
samples that were not assigned to that criterion. Aggregation requires
`expected_rater_ids` to match the scheduled pseudonyms exactly. Aggregation also
recomputes the counterbalance audit from the schedules and private reveal
mapping, so replacing the audit counts and refreshing their hashes is rejected.

Counterbalancing reduces systematic candidate-order confounding. It does not
eliminate sequence, learning, fatigue, or carryover effects, validate the number
of raters, or prove that reviewers followed their assigned schedules. A study
coordinator must distribute the correct schedule and retain the reveal file
outside the review surface.

Export one reviewer packet from the coordinator-only master review:

```bash
instavar-voice-eval export-rater-listening-packet listening-review.json \
  --rater-id rater-001 \
  --output rater-001-packet.json
```

The packet contains the blind items and only that pseudonymous rater's schedule.
After review, bind the scores and a declared presentation log to the packet:

```bash
instavar-voice-eval build-rater-listening-submission \
  rater-001-packet.json rater-001-ratings.json \
  --output rater-001-submission.json
```

The ratings input contains `scale`, `presentation_log`, and `ratings`. Rating
rows contain exactly `blind_id`, `criterion`, and `score`; the packet supplies
the rater identity. The builder rejects out-of-order logs, unpresented samples,
unassigned criteria, duplicate cells, and incomplete matrices by default. It
canonicalizes row order so equivalent inputs produce the same receipt.

To aggregate receipts, create a schema 1.1.0 document whose `scale` declares the
shared rating scale and whose `submissions` array contains every returned
submission, then pass that document as the ratings argument to
`aggregate-listening`. Aggregation reconstructs the expected packet
and submission for each scheduled rater, rejects rehashed forged metadata, and
records a digest of the canonical receipt set. `--allow-incomplete` preserves
missing raters and rating cells as explicit attrition evidence.

Each presentation-log entry contains one `criterion` and `blind_id`, following
the packet's criterion-major rating order. The log is self-attested. It does not
prove who received the packet, that audio was heard in the declared order, that
the reviewer remained attentive, or that reviewers acted independently.

For compatibility with an older study that intentionally assigned every
criterion to every sample, omit the two plan options and pass
`--criteria criteria.json`. The resulting coverage report labels that mode
`all_criteria_per_sample` rather than presenting it as plan-bound evidence.

After reviewers finish, aggregate criterion-level results with the matching review and reveal files:

```bash
instavar-voice-eval aggregate-listening listening-review.json reveal-mapping.json ratings.json \
  --output listening-results.json \
  --seed 20260812
```

The ratings document must declare `expected_rater_ids`, including invited
reviewers who submitted nothing. The output keeps every criterion separate,
reports deterministic bootstrap intervals, and calculates interval
Krippendorff alpha where multiple raters overlap. It fails on an incomplete
rater by sample by criterion matrix by default. Use `--allow-incomplete` only
when an explicitly incomplete coverage report is the intended artifact.
Agreement measures rating consistency, not correctness or perceptual truth.
Legacy all-criteria review files remain readable, but their aggregate marks
criterion direction as `unspecified` because no anchored semantics were bound.

## Score objective observations

New runners should emit `observation_schema_version: "1.0.0"`, a non-negative
`seed`, and a stable `runtime_id` on every row. Validate the strict producer
contract before scoring:

```bash
instavar-voice-eval validate-observations examples/objective-observations.json \
  --require-version \
  --require-seed \
  --require-runtime
```

Artifact-set ID and SHA-256 fields are optional for single-runtime scoring, but
they must appear together. Runtime comparisons require them through the
stronger exact-artifact binding gate. The Python scorer still accepts legacy
unversioned rows so historical evidence can be inspected, and reports how many
rows were versioned rather than silently treating them as current-contract
evidence.

The core package does not bundle model weights. It accepts external extractors and provides optional first-party execution paths for local faster-whisper ASR models and the pinned SpeechBrain ECAPA speaker architecture. Each sample records the extractor name and revision that produced its transcript, speaker embedding, runtime, and memory observations. Score those versioned observations with:

```bash
instavar-voice-eval score-objective examples/objective-observations.json \
  --output objective-results.json \
  --seed 20260812
```

When the observations come from a preregistered generation plan, bind that plan
while scoring so the report can distinguish authoritative generation text from
a reference string copied into an observation:

```bash
instavar-voice-eval score-objective examples/objective-observations.json \
  --generation-plan evaluation/generation-plan.json \
  --output objective-results.json \
  --seed 20260812
```

The result reports ASR word error rate, speaker-embedding cosine similarity,
invalid-output rate, real-time factor, generation time, audio duration, sample
rate, silence fraction, clipping fraction, and peak memory independently.
Per-candidate metric coverage makes selective omission visible. Sample rate,
silence, and clipping observations require versioned `audio_probe` evidence.
These are objective proxies. They do not establish accent fidelity, cadence,
naturalness, or listening fatigue.

The scorer also reports every extractor, revision, artifact-set digest, and
speaker-reference identity observed for ASR, speaker encoding, and runtime
probes. It counts fully content-bound and unbound evidence for each extractor
kind. Mixed provenance remains visible in an ordinary score report and is
rejected by the matched-comparison command.

ASR reference-text provenance is reported separately. Without
`--generation-plan`, word error rate uses the `requested_text` declared in each
observation and reports `declared_observation`. With a matching plan, it reports
`generation_plan`, the plan digest, and the number of scored references bound to
that plan. The binding establishes which text was requested. It does not prove
that the ASR transcript is correct, that the audio came from an honest TTS run,
or that the speech is perceptually good.

When a generation plan contains prompt categories, the same score report also
includes `plan_category_stratification`. Each candidate receives separate
invalid-output rate, metric summaries, and metric coverage for categories such
as `pronunciation`, `natural_local_context`, and `long_form_cadence`.
`compare-matched` reports matched deltas and validity by the same plan-bound
categories and rejects candidate-specific category drift. Legacy plans without
categories remain valid but report stratification as unavailable. Category
strata can expose a localized proxy regression hidden by an overall mean. They
do not define category weights, explain a difference, measure cadence from ASR,
or replace criterion-specific blind listening.

When the plan contains lexical anchors, scoring also emits
`plan_lexical_anchor_evidence`. It reports phrase hits, misses, ASR-unavailable
rows, invalid outputs, per-anchor coverage, and matched baseline-versus-adapted
hit-rate deltas. Matching uses token phrases, so an anchor such as `he` does not
match inside `the`. Accepted aliases are frozen in the plan and cannot be added
after seeing a candidate's transcript. A hit means only that the configured ASR
hypothesis contained one accepted phrase. It does not establish correct
pronunciation, accent fidelity, naturalness, alignment to the intended word, or
human acceptability. The plan digest makes later mutation visible but does not
prove that the aliases were chosen before generation without an external commit
or trusted timestamp.

## Bind runtime metrics to generation attempts

Do not treat timing and peak-memory fields copied into an observation as
attempt evidence. First build a receipt from the raw generation rows, frozen
plan, and live output audio. Then apply that receipt without overwriting other
extractor evidence:

```bash
instavar-voice-eval build-generation-attempt-receipt \
  generation-observations.json \
  --plan generation-plan.json \
  --audio-base-dir evaluation \
  --producer-name qwen3-evaluation-runner \
  --producer-revision "$RUNNER_REVISION" \
  --output generation-attempt-receipt.json

instavar-voice-eval apply-generation-attempt-receipt \
  generation-observations.json \
  generation-attempt-receipt.json \
  --plan generation-plan.json \
  --audio-base-dir evaluation \
  --output observations-with-runtime-evidence.json
```

The builder requires `generation_seconds` for every attempt. It binds the
complete generation-side row, the exact planned sample, the complete plan,
runtime metrics, producer revision, and live output bytes when an output
exists. Application rechecks the plan and live audio, rejects stale or reused
receipts, and adds a per-attempt digest plus the receipt digest to
`evidence.runtime`. Later ASR, speaker, audio-probe, and prosody-proxy
augmentation does not change the generation identity.

A version 1.1 plan that requires real-time factor, generation time, audio
duration, or peak memory rejects unbound runtime evidence. Legacy plans and
standalone score reports remain readable, but report those rows as unbound.
The receipt detects accidental or post-hoc substitution of declared fields. It
does not prove honest measurement, that a process loaded declared model bytes,
hardware isolation, clock validity, or host trust.

Runtime-attempt identity excludes later extractor-owned fields, including ASR
hypotheses, single-reference and multi-reference speaker embeddings, audio and
prosody probes, and augmentation history. This permits schema 1.4 speaker and
schema 1.6 prosody evidence to augment a bound attempt without weakening the
original generation receipt.
Changing a generation-side field such as timing, artifact identity, seed, or
audio hash still invalidates the binding.

## Bind extractor results to generated audio

Do not add ASR, speaker, or audio-probe values directly to generation rows.
Produce a content-addressed extractor result document, then apply it without
rewriting the producer fields. The built-in PCM WAV probe provides the first
complete path:

```bash
instavar-voice-eval build-audio-probe-results generation-observations.json \
  --audio-base-dir evaluation \
  --extractor-revision "$EVALUATOR_REVISION" \
  --output audio-probe-results.json

instavar-voice-eval apply-extractor-results \
  generation-observations.json \
  audio-probe-results.json \
  --audio-base-dir evaluation \
  --output observations-with-audio-probes.json
```

The result document binds the complete source observation array and every
result to SHA-256. Application rehashes each live WAV, rejects missing,
unexpected, duplicate, stale, symlinked, empty, or overwriting results, and adds
`input_audio_sha256` and `extractor_artifact_set_sha256` to extractor
provenance. A failed extraction stays on its sample as explicit
`extractor_failures` evidence rather than disappearing from coverage. The same
version 1.1 result contract supports `asr` and `speaker_encoder` patches from
external tools. Those tools can obtain the exact source binding with:

```bash
instavar-voice-eval fingerprint-observations generation-observations.json
```

External tools must also fingerprint the exact implementation or model files
used for extraction. Speaker encoders additionally bind the exact reference WAV
and transcript bytes:

```bash
instavar-voice-eval build-extractor-identity \
  --kind speaker_encoder \
  --name speaker-model \
  --revision "$SPEAKER_EXTRACTOR_REVISION" \
  --artifact model=tree=/models/speaker-model \
  --output speaker-extractor-identity.json

instavar-voice-eval build-speaker-reference \
  --reference-id target-voice-1 \
  --audio reference.wav \
  --transcript reference.txt \
  --output speaker-reference.json
```

The extractor producer places those two objects in the result document. The
consumer then rechecks the same live paths while applying it:

```bash
instavar-voice-eval apply-extractor-results \
  observations-with-audio-probes.json \
  speaker-results.json \
  --audio-base-dir evaluation \
  --extractor-artifact model=tree=/models/speaker-model \
  --reference-audio reference.wav \
  --reference-transcript reference.txt \
  --output observations-with-speaker-metrics.json
```

This protects source, extractor-artifact, and speaker-reference identity on the
observed host. It does not prove that an extractor actually loaded the declared
bytes, is scientifically valid, is robust to accent variation, or is free from
hostile-host tampering.

### Execute the optional faster-whisper ASR path

Schema 1.5 runs one local CTranslate2 faster-whisper model directly instead of
requiring a manually assembled ASR result document. Install `faster-whisper` in
an isolated environment. Supply an immutable model revision and a symlink-free
local snapshot. Hugging Face snapshots normally contain symlinks, so make a
dereferenced copy before fingerprinting it:

```bash
cp -RL /cache/models--Systran--faster-whisper-tiny.en/snapshots/$MODEL_REVISION \
  evaluation/faster-whisper-model

instavar-voice-eval build-faster-whisper-results \
  generation-observations.json \
  --audio-base-dir evaluation \
  --model-dir evaluation/faster-whisper-model \
  --model-name Systran/faster-whisper-tiny.en \
  --model-revision "$MODEL_REVISION" \
  --device cpu \
  --device-index 0 \
  --compute-type int8 \
  --language en \
  --beam-size 5 \
  --output asr-results-v1.5.json

instavar-voice-eval apply-extractor-results \
  generation-observations.json \
  asr-results-v1.5.json \
  --audio-base-dir evaluation \
  --faster-whisper-model-dir evaluation/faster-whisper-model \
  --output observations-with-executed-asr.json
```

Use `--device cuda --device-index N` only when the installed CTranslate2 build
and host support that device. The runner passes `local_files_only=True`, binds
the exact model tree and runner file, records Python and faster-whisper runtime
package versions, freezes language and decoding settings, preserves per-sample
failures, and rechecks model, runner, and audio bytes after transcription. The
complete result document is bound by `execution_receipt_sha256` and revalidated
when applied.

The result can support WER only when `requested_text` is an independently
trusted reference for the generated sample. A coherent transcript from a human
recording is plumbing evidence. It does not establish TTS intelligibility,
Singapore English coverage, pronunciation quality, accent fidelity, cadence,
or naturalness. The receipt remains unsigned and cannot defeat a malicious
host or prove that the dependency behaved honestly.

### Flag semantic corruption after ASR

Audio can pass duration, clipping, silence, and file-validity checks while the
spoken content repeats, omits requested words, includes text from the
conditioning transcript, or speaks a style instruction that should only guide
delivery. Version 0.39 provides a deterministic diagnostic that binds the exact
requested text and optional instruction in the generation plan, the
content-addressed ASR hypothesis in the observation, and the retained reference
transcript selected by the frozen speaker-reference assignment.

```bash
instavar-voice-eval build-content-faithfulness-report \
  observations-with-executed-asr.json \
  --generation-plan generation-plan.json \
  --reference-catalog speaker-reference-catalog.json \
  --speaker-reference-plan speaker-reference-assignment-plan.json \
  --speaker-reference studio=references/studio.wav=references/studio.txt \
  --ngram-size 4 \
  --minimum-reference-ngram-hits 2 \
  --instruction-ngram-size 2 \
  --minimum-instruction-ngram-hits 1 \
  --repetition-excess-fraction-threshold 0.05 \
  --word-error-rate-threshold 0.1 \
  --output content-faithfulness-report.json
```

Freeze the thresholds before using the report for a planned decision. A
post-hoc run may characterize a discovered failure but is not preregistered
confirmation. The report requires exact plan coverage, live reference bytes,
the frozen assignment plan, and ASR evidence bound to the candidate WAV and
extractor artifacts. It excludes reference n-grams that also occur in the
requested text, so correctly speaking shared words is not mislabeled as
reference leakage. Instruction n-grams found in requested text are excluded for
the same reason. Stable hashes identify both kinds of hit without copying raw
diagnostic text into the report. Those hashes are audit labels, not
anonymization.

Requested and ASR hypothesis text use the same NFKC and case-folded token
normalization for WER and n-gram checks. Each diagnostic text is limited to 64
KiB and 4,096 normalized tokens, and one report is limited to 20 million total
WER matrix cells. These fail-closed limits prevent malformed plans or unusually
large ASR output from turning the dependency-free edit-distance calculation
into unbounded work. An empty ASR hypothesis remains evaluable as a complete
omission, while requested text must contain at least one normalized token.

The four flags remain separate:

- high WER reports requested-text mismatch;
- repeated n-gram excess reports repeated hypothesis windows beyond the count
  expected from the requested text; and
- reference transcript overlap reports reference-exclusive n-grams also found
  in the ASR hypothesis; and
- spoken instruction overlap reports instruction-exclusive n-grams also found
  in the ASR hypothesis.

Instruction overlap has its own n-gram size and minimum-hit threshold so it can
be preregistered independently of retained-reference leakage. A sample without
an instruction is `not_applicable`. An instruction with no n-grams exclusive of
the requested text is `no_exclusive_ngrams`. Neither status is evidence that an
applicable leakage test passed. One-token instructions can be checked only when
the caller explicitly freezes `--instruction-ngram-size 1`, which has a higher
false-positive risk.

An invalid output fails the content gate. Missing ASR is `not_evaluable` and
cannot pass. `not_flagged` means only that the configured deterministic checks
did not fire. It does not prove content faithfulness, perceptual quality,
pronunciation, accent fidelity, runtime honesty, or absence of leakage.

The first post-hoc real-artifact characterization, exact report hashes, small
contrast set, OOD controls, and limits are recorded in
[`reports/content-faithfulness-validation-2026-08-13.md`](reports/content-faithfulness-validation-2026-08-13.md).

### Freeze multiple speaker references before evaluating candidates

Extractor result schema 1.2 supports a content-addressed catalog of reference
recordings. Build the catalog from every live audio and transcript pair:

```bash
instavar-voice-eval build-speaker-reference-catalog \
  --catalog-id target-voice-1 \
  --reference phone=references/phone.wav=references/phone.txt \
  --reference studio=references/studio.wav=references/studio.txt \
  --output speaker-reference-catalog.json
```

Schema 1.2 prevents baseline and adapted candidates from using different sets,
but it cannot show that a shared favorable set was chosen before results were
observed. For a plan-required speaker metric, freeze one assignment for every
prompt and seed pair before candidate generation or scoring. The same
assignment applies to every candidate in that pair:

This compact example assumes the generation plan contains only the shown
`greeting` and `long-form` pairs at seed `20260812`:

```bash
instavar-voice-eval build-speaker-reference-assignment-plan \
  --plan-id target-voice-1-eval \
  --generation-plan generation-plan.json \
  --reference-catalog speaker-reference-catalog.json \
  --policy-id stratified-v1 \
  --stratification-dimension accent \
  --stratification-dimension channel \
  --rationale "Freeze representative accent and channel coverage before generation." \
  --assignment greeting=20260812=phone,studio \
  --assignment long-form=20260812=phone,studio \
  --output speaker-reference-assignment-plan.json
```

The assignment declarations must exactly cover every unique prompt and seed in
the generation plan. Dimensions, assignments, and reference IDs must be unique
and sorted. Unknown references, missing pairs, extra pairs, a changed generation
plan, a changed catalog, or any post-freeze policy mutation fails validation.

Each schema 1.3 speaker result supplies one named embedding per planned
reference and binds the assignment-plan digest. The consumer rehashes the live
catalog, validates the assignment plan against the live generation plan, and
requires each result's sorted `reference_ids` to equal its frozen assignment:

```bash
instavar-voice-eval apply-extractor-results \
  observations-with-audio-probes.json \
  speaker-results-v1.3.json \
  --audio-base-dir evaluation \
  --extractor-artifact model=tree=/models/speaker-model \
  --speaker-reference phone=references/phone.wav=references/phone.txt \
  --speaker-reference studio=references/studio.wav=references/studio.txt \
  --speaker-reference-plan speaker-reference-assignment-plan.json \
  --generation-plan generation-plan.json \
  --output observations-with-multi-reference-speaker-metrics.json
```

The evaluator calculates cosine similarity separately for every bound
reference and uses the fixed `mean_cosine_similarity_v1` aggregation. It emits
the per-reference scores as diagnostics. A matched baseline and adapted pair
must use the same reference set, while different prompts may use different
sets only when the frozen plan says so. Per-sample assignment and measurement
digests bind the generation pair, planned membership, selected set, catalog,
and both sides of every similarity calculation. This rejects candidate-specific
selection, outcome-selected shared sets, and post-application embedding or
centroid substitution.

Schema 1.1 single-reference and schema 1.2 unplanned multi-reference evidence
remain readable for standalone migration reports. A plan-required speaker
metric now requires the schema 1.3 assignment contract and passes the same plan
to comparison:

```bash
instavar-voice-eval compare-matched objective-observations.json \
  --plan generation-plan.json \
  --speaker-reference-plan speaker-reference-assignment-plan.json \
  --baseline base-model \
  --adapted selected-adapter \
  --output matched-comparison.json
```

The hashes make a captured plan tamper-evident, but they do not prove when it
was created. Commit or otherwise timestamp the assignment plan before running
candidate generation for stronger chronology evidence. They also do not prove
that a speaker encoder honestly derived its embeddings from the declared audio
and model bytes, that the chosen strata are representative, or that the metric
tracks human perception.

### Execute the optional SpeechBrain ECAPA path

Schema 1.4 removes the manual vector-assembly step for one well-defined learned
speaker metric. Install compatible SpeechBrain, Torch, and Torchaudio packages
in an isolated environment. Supply a local, immutable model tree with regular
files. Hugging Face snapshots normally contain symlinks, so make a dereferenced
copy before fingerprinting it:

```bash
cp -RL /cache/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/$MODEL_REVISION \
  evaluation/ecapa-model

instavar-voice-eval build-speechbrain-ecapa-results \
  generation-observations.json \
  --audio-base-dir evaluation \
  --model-dir evaluation/ecapa-model \
  --model-revision "$MODEL_REVISION" \
  --catalog-id target-voice-1 \
  --speaker-reference phone=references/phone.wav=references/phone.txt \
  --speaker-reference studio=references/studio.wav=references/studio.txt \
  --speaker-reference-plan speaker-reference-assignment-plan.json \
  --generation-plan generation-plan.json \
  --device cpu \
  --trust-model-checkpoints \
  --output speaker-results-v1.4.json

instavar-voice-eval apply-extractor-results \
  generation-observations.json \
  speaker-results-v1.4.json \
  --audio-base-dir evaluation \
  --speechbrain-ecapa-model-dir evaluation/ecapa-model \
  --speaker-reference phone=references/phone.wav=references/phone.txt \
  --speaker-reference studio=references/studio.wav=references/studio.txt \
  --speaker-reference-plan speaker-reference-assignment-plan.json \
  --generation-plan generation-plan.json \
  --output observations-with-executed-speaker-metrics.json
```

Use `--device cuda` or an indexed device such as `--device cuda:1` when the
installed Torch build and host support CUDA. The runner loads the exact model
tree, overrides SpeechBrain's checkpoint prefix to that local tree to prevent a
hidden Hub fetch, encodes every frozen reference and candidate audio file,
records Python and package versions, preserves per-sample backend failures, and rechecks the
model, runner, reference catalog, and candidate audio after execution. The
consumer verifies the entire result document against
`execution_receipt_sha256` and rechecks the same live artifacts.

SpeechBrain and older Torch combinations can deserialize checkpoint pickle
data. The command therefore refuses to load a model until
`--trust-model-checkpoints` explicitly acknowledges that the checkpoint files
came from a trusted source. Content hashes identify bytes but do not make
untrusted checkpoint bytes safe.

This path materially reduces accidental and manual provenance gaps, but its
receipt is unsigned and produced on the same host. It cannot defeat a malicious
host, prove that a dependency behaved honestly, establish metric validity for a
new accent or recording channel, or replace blind human evaluation. A
same-speaker score from two excerpts of one recording validates plumbing only.
Evaluate synthesized held-out passages before drawing model-quality conclusions.

## Compare a matched baseline and adapted candidate

Generate the base model and adapted artifact from the same prompt pack and
frozen seeds. Every observation used for comparison must include `seed` as well
as `prompt_id`:

```bash
instavar-voice-eval compare-matched objective-observations.json \
  --plan generation-plan.json \
  --speaker-reference-plan speaker-reference-assignment-plan.json \
  --baseline base-model \
  --adapted selected-adapter \
  --output matched-comparison.json \
  --seed 20260812
```

The command binds every observation to the frozen generation plan and fails if
either candidate is missing a planned sample, an unplanned sample is added, a
prompt or seed differs, the requested transcripts differ, a pair is duplicated,
ASR, speaker, or runtime extractor provenance is mixed, or a valid candidate
selectively omits metrics available for its matched peer. With a generation
plan 1.1, both candidates must also supply every objective metric declared by
the frozen prompt pack. Invalid generations
remain in the validity delta but cannot improve WER, speaker similarity,
audio-duration, or real-time-factor summaries. Metric deltas use exact pairs
and keep directionality explicit, but the report sets
`proves_adaptation_benefit` to false because objective proxies cannot decide
perceptual improvement.

## Run the common lifecycle

A backend specification supplies argument arrays for five model-specific stages: preflight, train, infer, evaluate, and package. Commands are executed directly without a shell. Every stage must return success, write its stage result, and produce all declared artifacts before the next stage runs.

Backend specification 1.1 requires a positive timeout for every stage. Version
1.2 additionally binds the recipe to one supported or experimental adaptation
and one or more runtime IDs in a valid capability manifest. It declares every
required non-secret environment input with a purpose, checks those inputs before
creating the work directory, and rejects an experiment whose adaptation mode
does not match the recipe. The runner never records required environment values.

The
runner also requires a new or empty non-symlink work directory, validates the
complete experiment manifest before invoking a backend, rejects absolute or
parent-traversing artifact paths, and refuses symlinked stage results or
artifacts. Declared artifacts must be non-empty, live under the stage that owns
them, and must not reuse runner-owned result or log paths. Before accepting each
later stage, the runner rehashes all earlier evidence and fails if a checkpoint,
result, or other artifact changed after it was recorded. These checks prevent
stale, cross-stage, empty, mutated, or external files from satisfying a later
run. Use a unique work directory for every experiment attempt rather than
reusing or manually cleaning an old run directory.

The backend specification, experiment manifest, and bound capability manifest
are immutable control inputs for one lifecycle. The runner rejects symlinked
control files, records their sizes and SHA-256 hashes, and verifies the snapshots
before and after every stage. A backend that mutates its experiment or a
concurrent process that replaces a recipe cannot leave a passing lifecycle
report. This lock proves file identity for the observed run, not the truth of
claims inside the files.

A repository with more than one adaptation or runtime recipe can declare a
backend registry instead of relying on one conventionally named file:

```json
{
  "schema_version": "1.0.0",
  "backends": [
    {
      "backend_id": "example-lora-pytorch",
      "spec": "instavar-voice-backend.json"
    },
    {
      "backend_id": "example-full-sft-pytorch",
      "spec": "instavar-voice-backend-full-sft.json"
    }
  ]
}
```

Registry paths are relative to the registry file. Validation rejects path
traversal, symlinks, missing or invalid specifications, duplicate IDs, duplicate
paths, and an ID that differs from the referenced specification. Registered
lifecycle execution automatically selects the unique recipe bound to the
experiment's adaptation mode. If more than one recipe matches, selection fails
closed until the caller supplies an explicit backend ID. An explicit selection
still cannot override the experiment's adaptation mode.

Validate and exercise the included lightweight backend:

```bash
instavar-voice-eval validate-backend examples/fake-backend.json
instavar-voice-eval validate-backend-registry examples/backend-registry.json
instavar-voice-eval run-lifecycle \
  examples/fake-backend.json \
  examples/experiment-manifest.json \
  --work-dir /tmp/instavar-voice-fake-lifecycle
instavar-voice-eval run-registered-lifecycle \
  examples/backend-registry.json \
  examples/experiment-manifest.json \
  --work-dir /tmp/instavar-voice-registered-lifecycle
```

The lifecycle report records commands, exit codes, timeouts, logs, artifact
hashes, immutable control-input records, and the fail-closed stage boundary. A
registry-based report also records the registry hash, selected backend ID,
selected specification path, and specification hash. A passed fake lifecycle
proves
orchestration and evidence generation only. It does not prove that a real model
trains, synthesizes correct speech, or sounds good.

## Test

```bash
python3 -m unittest discover -s tests -v
```

These tests validate contract behavior, deterministic artifact generation, proxy calculations, listening aggregation, the SpeechBrain and faster-whisper runners through dependency-free test doubles, and a complete lightweight lifecycle. They do not run heavyweight model training, real ASR, real speaker encoders, or human listening. Release evidence should record separate real-model smoke tests on the intended host and dependency set.
