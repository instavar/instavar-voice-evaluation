# Runtime receipt stability after prosody augmentation

Date: 2026-08-13

## Result

Instavar Voice evaluation 0.36.0 preserves a valid generation-attempt receipt
after schema 1.6 prosody results are applied. The runtime identity now derives
its complete prosody field set from `PROSODY_OBSERVATION_FIELDS`, the same
canonical set used by extraction and comparison.

## Failure reproduced

A frozen CosyVoice3 Base versus epoch-12 long-form evaluation first bound each
generation attempt, then applied audio and prosody probes. The next objective
score failed with `runtime attempt does not match generation observation
content`. Timing, model identity, seed, requested text, and audio bytes had not
changed. Only extractor-owned `prosody_*` fields had been added.

The runtime identity filter already excluded ASR hypotheses, speaker
embeddings, audio probes, augmentation history, and extractor failures. It did
not exclude the canonical prosody observation fields. This contradicted the
documented contract that later extractor augmentation must not invalidate an
already-bound generation attempt.

## Fix and OOD coverage

`instavar_voice_lab.attempts` now imports the canonical prosody field set and
includes every member in `DERIVED_OBSERVATION_FIELDS`. The existing external
augmentation regression test now adds all prosody fields plus prosody extractor
evidence before rechecking the runtime receipt. Generation-owned mutation tests
remain unchanged and continue to fail closed.

Validation completed with:

- 6 focused generation-attempt tests;
- 202 full-suite tests and 18 subtests;
- Python bytecode compilation and whitespace checks.

These tests establish receipt stability for the represented extractor fields.
They do not prove honest runtime measurement, loader behavior, trustworthy host
execution, prosody validity, or perceptual quality.
