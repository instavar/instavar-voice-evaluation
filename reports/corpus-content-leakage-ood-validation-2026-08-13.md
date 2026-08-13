# Corpus content-leakage OOD validation

Date: 2026-08-13

## Finding

The corpus audit previously detected repeated resolved audio paths, but not
byte-identical copies stored under distinct paths. The same recording could
therefore cross train, validation, and test while every manifest path remained
unique. Transcript duplicate warnings also used whitespace folding and
casefolding without Unicode compatibility normalization, allowing full-width
and compatibility-equivalent text to evade the warning.

Version 0.41 computes a stable SHA-256 digest for each referenced audio file and
rejects any repeated content digest. It processes splits in semantic order,
train then validation then test, so diagnostics identify the earlier training
source consistently. The hash uses one open file descriptor and verifies
device, inode, size, and nanosecond modification time before and after reading,
then rechecks the path identity. A file changed or replaced during hashing
cannot satisfy the audit.

Transcript duplicate keys now apply NFKC before the existing whitespace fold
and casefold. These remain warnings because repeated text can be legitimate
when recordings or speakers differ.

## OOD controls

Dependency-free tests cover:

- copied audio bytes under different train and test paths;
- full-width and ASCII transcript forms that normalize to the same text;
- distinct audio content across all required splits;
- recording-group leakage; and
- missing split and missing group metadata failures.

The complete dependency-free suite passed 221 tests locally. `uv build`
produced the 0.41 source distribution and wheel, and focused Ruff E/F checks
passed. Implementation commit
`0372dec202fca23386ab5cf3018ec425c5e60c13` passed hosted Quality run
`31682337739` on 2026-08-13.

## Scope and boundary

Exact content hashing generalizes to local regular files referenced by JSONL
corpus manifests and catches copies and hard links regardless of filename. It
does not detect transcoded, trimmed, level-shifted, resampled, or otherwise
near-duplicate audio. It also does not prove speaker independence, transcript
accuracy, recording rights, consent, or absence of semantic overlap. Those
remain separate corpus, legal, and evaluation checks.
