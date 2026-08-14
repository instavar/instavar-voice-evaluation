# Singapore English lexical-anchor expansion validation

Date: 2026-08-15, Asia/Singapore

## Result

Prompt-pack version 1.3 freezes four local lexical anchors that already occur
in the existing prompt text: `paiseh`, `Tanjong Pagar`, `Sze Min`, and
`Jalan Membina`. This promotes the exact anchor set preregistered before the
VoxCPM2 seven-prompt, three-seed matched run into the shared cross-model pack.

The expansion does not change any requested text, category, seed, objective
metric, or listening criterion. It only makes all four existing local phrases
visible to deterministic ASR hit-rate accounting. Tests require the complete
anchor lists and accepted forms to remain identical across both candidates and
all three seeds in the 42-row generation plan.

## Evidence and boundary

The motivating VoxCPM2 run found zero accepted ASR hits for all four anchors in
both Base and one-step LoRA across three matched seeds. The public evidence is
recorded in
[`instavar/voxcpm`](https://github.com/instavar/voxcpm/blob/main/reports/voxcpm2-clean-split-full-matched-base-lora-2026-08-15.md).

That negative result justifies preserving the probes, not declaring every
rendered pronunciation wrong. ASR can fail on accented or unfamiliar speech,
and an accepted phrase hit still does not establish correct pronunciation.
Human listening remains required for pronunciation and accent claims.

The accepted alternatives are intentionally narrow and preregistered. The pack
accepts both `Tanjong Pagar` and the common ASR spelling `Tanjung Pagar`, while
the other new anchors accept only their written forms. New aliases require a
separate evidence-backed version change rather than post-generation tuning.

## Verification

- prompt-pack validation passed;
- the deterministic plan contains 42 samples;
- each of the two anchored prompts has one identical anchor set across both
  candidates and all three seeds;
- the focused suite tests passed 11 of 11;
- the full evaluator suite passed 255 of 255;
- prompt-pack file SHA-256:
  `3761962cbcee05934121c8ce61790ac389223e44a8460d57a113951f4c7372cc`;
- canonical prompt-pack SHA-256:
  `6799b4d2a692cc0a1e88d176a1090011d8723b094869ba529aed7755ad43df29`.
