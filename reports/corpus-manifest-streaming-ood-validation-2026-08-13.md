# Corpus manifest streaming OOD validation

Date: 2026-08-13

## Finding

The corpus audit previously hashed each JSONL manifest in one read and parsed
it in a second read. A concurrent rewrite or path replacement could therefore
bind `manifest_sha256` to bytes other than those actually validated. Invalid
UTF-8 also escaped the structured audit result, and a single unbounded JSONL
line could consume memory before JSON parsing rejected it.

Version 0.42 reads, hashes, decodes, and parses each manifest through one open
descriptor. It verifies device, inode, size, and nanosecond modification time
before and after the stream, then rechecks the path identity. The report now
records the exact byte count and emits schema 1.1 so consumers can distinguish
the stronger evidence contract.

## Resource and failure controls

The default maximum is 512 MiB per manifest and 8 MiB per logical JSONL line.
Reading occurs in 64 KiB chunks. Once a line exceeds its bound, the audit keeps
hashing and discarding that line instead of retaining it in memory, records a
structured error, and continues to preserve later diagnostics. The Python API
may lower either bound but rejects attempts to raise the safety ceilings. The
CLI has no relaxation flags.

Malformed UTF-8, a too-large declared manifest, growth beyond the total bound,
and mutation or replacement during reading all produce failed audit evidence
instead of uncaught decode errors or mismatched hashes.

## Validation

Dependency-free tests cover:

- invalid UTF-8 as a structured row error;
- a logical line larger than a tightened limit;
- a manifest larger than a tightened total limit;
- path replacement after the first streamed line;
- invalid and above-ceiling API limits; and
- the existing group, copied-audio, Unicode transcript, and split controls.

The complete dependency-free suite passed 226 tests locally. Focused Ruff E/F
and formatter checks passed, and `uv build` produced the 0.42 source
distribution and wheel. Implementation commit
`16414af7996e01d715d218b15d5ad2bd4905b198` contains the bounded streaming
contract and its OOD tests.

## Scope and boundary

The protection applies to local JSONL manifests audited by this process. It
prevents ordinary concurrent mutation and replacement from silently producing
a hash-to-parse mismatch. It is not a cryptographic defense against an
attacker capable of rewriting bytes while preserving every checked filesystem
attribute. It does not make remote object stores transactional, validate
near-duplicate audio, prove speaker separation, or establish transcript truth,
consent, or rights.
