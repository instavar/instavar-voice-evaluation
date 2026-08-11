# Instavar Voice adaptation and evaluation contract

This directory implements the shared evidence layer for Instavar's TTS companion repositories. It does not contain a universal trainer. Model-specific repositories continue to own preprocessing, codec handling, LoRA or full-SFT training, checkpoint loading, and runtime integration.

The public distribution is [instavar/instavar-voice-evaluation](https://github.com/instavar/instavar-voice-evaluation). The copy under the private Instavar product repository keeps the application and its pinned contract version reviewable together.

The shared layer provides:

- versioned capability, experiment, evaluation, and artifact-package contracts;
- semantic validation using only the Python standard library;
- a frozen Singapore English prompt and listening-criteria pack;
- deterministic PCM WAV diagnostics;
- deterministic blind-review labels and a separately stored reveal mapping; and
- examples and unit tests for every contract.

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
```

Validate a companion repository after it adds `instavar-voice-capabilities.json`:

```bash
instavar-voice-eval validate-repository /path/to/companion-repository
```

The checked-in JSON Schemas provide editor and ecosystem interoperability. The Python validator adds semantic checks that are awkward or misleading in schema alone, including evidence requirements for supported capabilities, unique runtime identifiers, distinct corpus split hashes, baseline presence, and the ban on a universal composite evaluation score.

## Probe generated audio

```bash
instavar-voice-eval probe-audio output.wav --output evaluation/output.probe.json
```

The deterministic probe reports duration, sample rate, channels, sample width, peak, RMS, DC offset, silence fraction, and clipping fraction for uncompressed PCM WAV files. It does not measure intelligibility, speaker identity, accent fidelity, cadence, or naturalness.

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

## Test

```bash
python3 -m unittest discover -s tests -v
```

These tests validate contract behavior and deterministic artifact generation. They do not run model training, inference, ASR, speaker embeddings, or human listening.
