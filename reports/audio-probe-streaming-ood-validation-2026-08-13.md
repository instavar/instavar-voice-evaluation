# Streaming audio-probe OOD validation

Date: 2026-08-13

## Finding

The deterministic WAV probe previously read the complete PCM payload and then
created complete integer and normalized-float sample lists. Valid but large
audio, an oversized declared payload, or a hostile fixture could therefore use
memory proportional to the entire decoded recording before producing evidence.

Version 0.40 replaces the full-file materialization with 65,536-frame chunks
and scalar accumulators. It preserves the public probe fields and definitions,
while adding three fail-closed boundaries:

- the declared decoded PCM payload must be no larger than 512 MiB by default;
- the decoded sample count must match the WAV header, so truncated payloads do
  not become apparently valid shorter observations; and
- silence and clipping thresholds must be finite and within their documented
  ranges.

The Python API accepts a smaller positive `max_pcm_bytes` when an evaluator
needs a tighter resource budget. Raising the built-in ceiling should be an
explicit code review because the current CLI intentionally exposes no switch
that lets an untrusted input relax it.

## OOD controls

Dependency-free tests cover:

- a stereo payload larger than one read chunk;
- a valid WAV rejected under a deliberately small PCM budget;
- a truncated payload whose header still declares the original frame count;
- NaN, infinity, negative, and greater-than-one silence thresholds; and
- zero, negative, and Boolean resource limits.

The complete dependency-free suite passed 219 tests locally. `uv build`
produced the 0.40 source distribution and wheel, and focused Ruff E/F checks
passed. Implementation commit
`f04ccd31e64c0c1f4d0145be7e88242f25b964b4` passed hosted Quality run
`31681987993` on 2026-08-13.

## Scope and boundary

The resource bound generalizes to uncompressed PCM WAV probing in the evaluator
and to batch audio-probe extraction that calls the same function. It does not
cover compressed containers, RF64, ASR decoders, speaker encoders, or prosody
proxy internals. Streaming arithmetic controls evaluator memory but does not
make an arbitrarily long recording operationally cheap, establish semantic
correctness, or turn signal diagnostics into a perceptual quality result.
