# Content-faithfulness diagnostic validation, 2026-08-13

## Result

Evaluator 0.37 adds a deterministic, plan-bound diagnostic for three separate
TTS content failures: high requested-text WER, repeated n-gram excess, and
retained-reference transcript overlap. It reproduced the known CosyVoice3
long-form failure and did not flag the six retained F5, IndexTTS2, and Qwen3-TTS
long-form samples in this small post-hoc contrast set.

This is implementation and characterization evidence. The thresholds were
chosen after the CosyVoice3 outputs had been inspected, so these runs are not a
preregistered confirmation or an estimate of sensitivity, specificity, or
model-family quality.

## Contract

The command requires:

- exact observation coverage of a frozen generation plan;
- a content-addressed ASR hypothesis bound to the candidate WAV and extractor
  artifact set;
- the frozen speaker-reference assignment plan;
- the content-addressed reference catalog; and
- the live retained reference audio and transcript bytes.

The report does not copy requested, hypothesis, or reference text. It reports
counts and stable hashes for reference-exclusive n-gram hits. Those hashes are
audit labels and not anonymization. Reference n-grams that also occur in the
requested text are excluded before overlap checking.

The post-hoc settings were:

- n-gram size: 4
- minimum reference-exclusive n-gram hits: 2
- repeated n-gram excess fraction threshold: 0.05
- WER threshold: 0.1

Repeated excess counts hypothesis n-gram occurrences beyond the greater of one
or the count expected from the requested text. This avoids labeling one new
n-gram occurrence as repetition. Invalid outputs fail the gate; missing ASR is
not evaluable. A `not_flagged` result never becomes proof of content
faithfulness.

## Real retained observations

| Candidate | WER | Repeated excess fraction | Reference-exclusive hits | Result |
| --- | ---: | ---: | ---: | --- |
| CosyVoice3 Base | 0.281385 | 0.099265 | 6 | all three flags |
| CosyVoice3 epoch 12 | 0.515152 | 0.117647 | 0 | high WER and repetition |
| F5 Base | 0 | 0 | 0 | not flagged |
| F5 LoRA 1250 | 0 | 0 | 0 | not flagged |
| IndexTTS2 Base | 0.012987 | 0 | 0 | not flagged |
| IndexTTS2 step 14000 | 0.025974 | 0.012987 | 0 | not flagged |
| Qwen3-TTS Base clone | 0.008658 | 0 | 0 | not flagged |
| Qwen3-TTS epoch 10 | 0.004329 | 0 | 0 | not flagged |

The exact diagnostic refines the earlier qualitative description. Reference
transcript overlap was flagged for CosyVoice3 Base, not for epoch 12 under the
frozen four-gram and two-hit rule. Both Cosy candidates were flagged for high
WER and repetition. The adapter remains the worse requested-text result, while
Base has the additional exact reference-overlap signal.

Report file and self-hash pairs:

- CosyVoice3: file
  `10d52d3b339d2943788512c53cc77ac29e3c4d4c9129fb5971fff4c5931996c8`,
  self `7cb8aa323fcf14acc9377b4700f9ea18d8fa72a1b94e780026e1c42a9cf37e03`
- F5: file
  `4c49bf91cbf90ea677b47473e99f5e2a026fc6a49b078df8ad2abcfcb5019435`,
  self `eab47536102d0d33cb0556123edd6188a463b0dbc141eea041b51444298cd0ea`
- IndexTTS2: file
  `1f9a985fde2ed6501cae1e9ef1c4d5084e905893df6eb43e98bcee4898c1d457`,
  self `78a6968c26047f5d8f24b8d8ae4983019080981f91768ee620d4fc5e5a76e170`
- Qwen3-TTS: file
  `6f9a5857e106de80d5eb855cf566dfe0d79b94e6b8ae0e93ab4ca5cd8335b12a`,
  self `c3ed5717861276fa721c91a87f963a20d80f16a451180ece38674dd4e51f4b00`

All four reports were generated from clean revision
`b7ebd871eba620bbafbbd64d62d964ec85386880` on the RTX host. The retained
reference audio SHA-256 was
`2dc2a3d83dab1e5569d1adac7828c907acc78271cb495d80228b15ca6e460237`;
the transcript SHA-256 was
`7b5f531abde272946e3638bbd35736923e1b3562779deff69aed968bf471ba1e`.

## OOD controls and limits

Dependency-free tests cover exact requested speech, repeated reference leakage,
requested and reference overlap, missing ASR, invalid output, unbound ASR,
mutated reference bytes, CLI binding, and report non-claiming. Inputs are
bounded by reference count, references per sample, transcript bytes, and total
reference tokens. Hit hashes are capped and explicitly marked when truncated.

The diagnostic uses exact normalized token n-grams. ASR errors, paraphrases,
short leaks, homophones, or non-English tokenization can hide a real leak.
Common exact phrases can create false overlap. WER and thresholds remain
task-dependent. The six unflagged contrast samples are too few and too similar
to validate false-positive behavior outside this retained long-form slice.
Human listening and broader adversarial fixtures remain separate gates.
