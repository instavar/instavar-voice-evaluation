# Resume artifact comparison OOD validation

Date: 2026-08-14, Asia/Singapore

## Finding

Five owned TTS companions expose guarded resume paths, but the shared evaluator
previously had no common way to compare a real interrupted and resumed run with
an uninterrupted control. Model-specific tests could show that a checkpoint
loads without proving that the final optimizer, scheduler, trainer, RNG, and
model states match an independently executed control.

Version 0.44 adds `compare-resume-artifacts`. It consumes two run receipts, a
live interruption receipt, and two independently stored final artifact sets.
Both receipts must declare the same repository revision, backend, adaptation
mode, Base artifact, dataset lineage, training controls, initial state, and
target update count. The resumed receipt must bind an observed interruption at
a checkpoint strictly before the target update.

The comparison requires these semantic roles:

- `model_state`
- `optimizer_state`
- `scheduler_state`
- `trainer_state`
- `rng_state`

Every declared file or tree is hashed again before the report is emitted. A
content difference is retained as a negative result instead of being converted
into an invalid input error.

## OOD controls

The dependency-free suite covers:

- a byte-exact uninterrupted and interrupted-resumed pair;
- a model-state mismatch retained as `negative_result` with CLI exit 1;
- omission of a core state role;
- training-control identity drift;
- a run that stops before the target update;
- a resume checkpoint at or after the target update;
- a missing observed-interruption flag;
- a mutated interruption receipt;
- reuse of the same run ID;
- reuse of the same state path across both runs;
- cross-run hardlink aliasing;
- symlinked artifacts;
- asymmetric role sets;
- file-versus-tree kind drift for the same role;
- an interruption receipt that aliases a state artifact;
- state mutation during comparison;
- tree artifacts and a hardlink alias hidden inside a tree;
- an incorrect execution mode; and
- a resumed-from boundary that differs from the saved checkpoint boundary.

The focused resume suite passed 14 tests. The complete dependency-free suite
passed 244 tests on macOS with Python 3.11. Bytecode compilation passed for the
package and tests. All three JSON Schemas passed Draft 2020-12 schema checks,
and generated uninterrupted and resumed receipts, a plan, and a report
validated against them.

Python 3.11 built the version 0.44 wheel with SHA-256
`350c05df6047cca94e992195abc45335dcb52724483861b657f53c8e8aa7b2d3`.
The local environment did not provide Ruff, so no Ruff result is claimed.

## Evidence boundary

A passing report establishes byte equality only for the named live final-state
roles under the two declared run receipts. It rejects obvious shared-storage
shortcuts by inode identity, but it does not prove that the trainer honored the
receipt fields, that hidden or omitted state is equal, or that floating-point
execution followed the same numerical trajectory. It also establishes nothing
about synthesis quality, speaker identity, accent fidelity, cadence, or runtime
equivalence.

Version 0.44 deliberately excludes opaque monolithic checkpoints whose
internal model, optimizer, scheduler, trainer, and RNG components are not
independently exposed. A future inventory-aware tier may support those formats,
but container equality alone must not be relabeled as complete state coverage.

Historical resume evidence is not upgraded retroactively. A companion must
emit the new receipts during a fresh controlled run before using this claim
tier. Reconstructing a receipt after seeing the outcome would weaken chronology
and is not accepted as equivalent evidence.
