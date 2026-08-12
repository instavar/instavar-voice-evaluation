# Instavar Voice adaptation and evaluation contract

This directory implements the shared evidence layer for Instavar's TTS companion repositories. It does not contain a universal trainer. Model-specific repositories continue to own preprocessing, codec handling, LoRA or full-SFT training, checkpoint loading, and runtime integration.

The public distribution is [instavar/instavar-voice-evaluation](https://github.com/instavar/instavar-voice-evaluation). The copy under the private Instavar product repository keeps the application and its pinned contract version reviewable together.

The shared layer provides:

- versioned capability, experiment, evaluation, and artifact-package contracts;
- semantic validation using only the Python standard library;
- a frozen Singapore English prompt and listening-criteria pack;
- deterministic PCM WAV diagnostics;
- deterministic blind-review labels and a separately stored reveal mapping; and
- objective proxy scoring from versioned ASR, speaker-encoder, and runtime observations;
- criterion-level listening aggregation with bootstrap intervals and interval agreement; and
- a fail-closed lifecycle runner for model-specific preflight, training, inference, evaluation, and packaging;
- a frozen candidate by prompt by seed generation plan with completeness accounting;
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
failed generations recorded with `valid: false`. It does not convert signal,
ASR, speaker, or runtime measurements into a perceptual quality claim.

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

## Build a blind listening pack

Prepare a JSON array containing `sample_id`, `candidate_id`, `prompt_id`, and `audio_path`, plus a JSON array of criterion names. Then run:

```bash
instavar-voice-eval build-listening-pack samples.json \
  --criteria criteria.json \
  --review-output listening-review.json \
  --reveal-output reveal-mapping.json \
  --seed 20260812
```

The review file contains no candidate identifiers. Preserve the reveal mapping separately and do not open it until all ratings are recorded.

After reviewers finish, aggregate criterion-level results with the matching review and reveal files:

```bash
instavar-voice-eval aggregate-listening listening-review.json reveal-mapping.json ratings.json \
  --output listening-results.json \
  --seed 20260812
```

The output keeps every criterion separate, reports deterministic bootstrap intervals, and calculates interval Krippendorff alpha where multiple raters overlap. Agreement measures rating consistency, not correctness or perceptual truth.

## Score objective observations

The core package does not bundle a preferred ASR model or speaker encoder. Instead, each sample records the extractor name and revision that produced its transcript, speaker embedding, runtime, and memory observations. Score those versioned observations with:

```bash
instavar-voice-eval score-objective examples/objective-observations.json \
  --output objective-results.json \
  --seed 20260812
```

The result reports ASR word error rate, speaker-embedding cosine similarity, invalid-output rate, real-time factor, generation time, audio duration, and peak memory independently. These are objective proxies. They do not establish accent fidelity, cadence, naturalness, or listening fatigue.

## Run the common lifecycle

A backend specification supplies argument arrays for five model-specific stages: preflight, train, infer, evaluate, and package. Commands are executed directly without a shell. Every stage must return success, write its stage result, and produce all declared artifacts before the next stage runs.

Validate and exercise the included lightweight backend:

```bash
instavar-voice-eval validate-backend examples/fake-backend.json
instavar-voice-eval run-lifecycle \
  examples/fake-backend.json \
  examples/experiment-manifest.json \
  --work-dir /tmp/instavar-voice-fake-lifecycle
```

The lifecycle report records commands, exit codes, logs, artifact hashes, and the fail-closed stage boundary. A passed fake lifecycle proves orchestration and evidence generation only. It does not prove that a real model trains, synthesizes correct speech, or sounds good.

## Test

```bash
python3 -m unittest discover -s tests -v
```

These tests validate contract behavior, deterministic artifact generation, proxy calculations, listening aggregation, and a complete lightweight lifecycle. They do not run heavyweight model training, real ASR, real speaker encoders, or human listening.
