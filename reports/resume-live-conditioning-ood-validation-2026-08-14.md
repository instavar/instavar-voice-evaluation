# Resume live-conditioning OOD validation

Date: 2026-08-14, Asia/Singapore

## Finding

Evaluator 0.44 rehashed final model, optimizer, scheduler, trainer, and RNG
state, but its schema 1.0 run receipts still accepted manually declared Base,
dataset-lineage, training-control, and initial-state hashes. That boundary was
explicit, but rolling it into every companion would have standardized a weak
provenance step.

Version 0.45 adds `build-resume-run-receipt`. It fingerprints those four live
conditioning artifacts into a schema 1.1 receipt, rejects aliases and mutation,
and binds the live interruption receipt for an interrupted-resumed run. A
schema 1.1 comparison plan names the same four paths. The comparison rehashes
them and requires both run receipts to match the live records before comparing
the independently stored final states.

The report separates two claim tiers:

- `byte_exact_live_conditioned_artifact_set` means the schema 1.1
  conditioning and final-state bytes were rechecked;
- `byte_exact_declared_artifact_set` preserves schema 1.0 compatibility while
  keeping `conditioning_artifacts_verified: false`.

Neither tier claims numerical execution equivalence, training semantics, model
quality, or hidden-state coverage.

## OOD controls

The focused suite now covers 19 cases, including the evaluator 0.44 controls
and these additional paths:

- building a schema 1.1 receipt from one tree and three files;
- comparing two schema 1.1 receipts against live conditioning;
- mutating training controls after receipt creation;
- supplying schema 1.0 receipts under a schema 1.1 plan;
- reusing one conditioning file for two semantic roles;
- mutating a conditioning file while the builder hashes it;
- trying to overwrite an existing receipt or report;
- placing a receipt output inside the Base artifact tree; and
- placing a comparison report inside the Base artifact tree.

Receipt and comparison outputs use exclusive creation. This matters because an
output written inside a hashed tree would otherwise mutate the evidence after
the final recheck and could make the newly written receipt immediately stale.

The complete dependency-free suite passed 249 tests on macOS with Python 3.11.
Bytecode compilation passed for the package and tests. Legacy schema 1.0 and
live-conditioned schema 1.1 fixtures both validated against the final Draft
2020-12 run-receipt, comparison-plan, and comparison-report schemas.

Python 3.11 built the version 0.45 wheel with SHA-256
`fe774efa51ab4c9d084395d3313c58426b6cd349a4177827fbed8afff633ba78`.
The local environment did not provide Ruff, so no Ruff result is claimed.

## Scope and boundary

Live fingerprinting prevents a hand-entered digest from being mistaken for an
observed file hash. It still cannot independently prove that the trainer loaded
those exact files or honored every receipt field. A backend must emit receipts
during a fresh controlled run; rebuilding them after outcome inspection does
not repair chronology and must not retroactively upgrade historical evidence.

The two-pass fingerprint checks detect a file that changes during receipt or
comparison construction. They do not freeze a large tree against a privileged
concurrent writer, and they do not provide a filesystem snapshot. Run inputs
should therefore be immutable or mounted read-only while receipts are built.

Opaque monolithic checkpoints remain outside complete state coverage. A future
inventory-aware tier may qualify a container only if its internal state roles
are independently enumerated and verified.
