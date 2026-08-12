# Instavar Voice adaptation and evaluation contract

This directory implements the shared evidence layer for Instavar's TTS companion repositories. It does not contain a universal trainer. Model-specific repositories continue to own preprocessing, codec handling, LoRA or full-SFT training, checkpoint loading, and runtime integration.

The public distribution is [instavar/instavar-voice-evaluation](https://github.com/instavar/instavar-voice-evaluation). The copy under the private Instavar product repository keeps the application and its pinned contract version reviewable together.

The shared layer provides:

- versioned capability, experiment, evaluation, and artifact-package contracts;
- semantic validation using only the Python standard library;
- a frozen Singapore English prompt and listening-criteria pack;
- deterministic PCM WAV diagnostics;
- deterministic blind-review labels, identity-neutral staged filenames, and a separately stored reveal mapping;
- objective proxy scoring from versioned ASR, speaker-encoder, and runtime observations;
- a versioned objective-observation contract with stable identifiers and complete runtime-artifact bindings;
- criterion-level listening aggregation with bootstrap intervals and interval agreement; and
- a fail-closed lifecycle runner for model-specific preflight, training, inference, evaluation, and packaging;
- a frozen candidate by prompt by seed generation plan with completeness accounting;
- plan-bound objective metric requirements that reject bilateral metric omission;
- exact baseline-versus-adapted pairing with extractor-provenance checks and paired bootstrap intervals;
- content-addressed extractor implementation or model artifacts plus speaker-reference audio and transcript bindings;
- deterministic multi-reference speaker scoring with per-sample reference-set binding;
- frozen per-prompt and per-seed speaker-reference assignments bound to generation plans and reference catalogs;
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

The deterministic probe reports duration, sample rate, channels, sample width, peak, RMS, DC offset, silence fraction, and clipping fraction for uncompressed PCM WAV files. It does not measure intelligibility, speaker identity, accent fidelity, cadence, or naturalness.

Compare the diagnostics from a reference runtime and a candidate runtime:

```bash
instavar-voice-eval compare-audio pytorch.wav alternative-runtime.wav \
  --output evaluation/runtime-comparison.json
```

The comparison records format matches and candidate-minus-reference deltas. It deliberately emits `proves_runtime_equivalence: false`: matching container and signal-level diagnostics cannot establish text, speaker, accent, cadence, or perceptual equivalence.

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

The audit hashes each manifest and fails if a recording group crosses splits. It checks that referenced audio files exist, but it does not decode every audio format or prove that the transcript matches the recording.

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

Prepare a JSON array containing `sample_id`, `candidate_id`, `prompt_id`, and `audio_path`, plus a JSON array of criterion names. Then run:

```bash
instavar-voice-eval build-listening-pack samples.json \
  --criteria criteria.json \
  --review-output listening-review.json \
  --reveal-output reveal-mapping.json \
  --stage-root evaluation/listening \
  --seed 20260812
```

The review file contains no candidate identifiers or source filenames. With
`--stage-root`, audio is copied to paths such as
`blind_audio/sample-0001.wav`, and a hash manifest is written beside the staged
files. Preserve the reveal mapping separately and do not open it until all
ratings are recorded. File staging does not strip embedded audio metadata, so
inspect or normalize metadata separately if the source format can carry
identity-bearing tags. The builder rejects mixed audio extensions because a
format difference can itself reveal which runtime produced a sample.

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

The core package does not bundle a preferred ASR model or speaker encoder. Instead, each sample records the extractor name and revision that produced its transcript, speaker embedding, runtime, and memory observations. Score those versioned observations with:

```bash
instavar-voice-eval score-objective examples/objective-observations.json \
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
`evidence.runtime`. Later ASR, speaker, and audio-probe augmentation does not
change the generation identity.

A version 1.1 plan that requires real-time factor, generation time, audio
duration, or peak memory rejects unbound runtime evidence. Legacy plans and
standalone score reports remain readable, but report those rows as unbound.
The receipt detects accidental or post-hoc substitution of declared fields. It
does not prove honest measurement, that a process loaded declared model bytes,
hardware isolation, clock validity, or host trust.

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

These tests validate contract behavior, deterministic artifact generation, proxy calculations, listening aggregation, and a complete lightweight lifecycle. They do not run heavyweight model training, real ASR, real speaker encoders, or human listening.
