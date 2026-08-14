# Lifecycle process-tree OOD validation

Date: 2026-08-14

## Finding

The shared lifecycle previously used `subprocess.run(..., timeout=...)` for
each preflight, training, inference, evaluation, and packaging stage. Python
terminates the direct process when that timeout expires, but a trainer can
spawn data workers, launchers, loggers, or other descendants. Those descendants
could outlive the failed stage, retain GPU memory, continue writing files, and
interfere with a later run.

Version 0.43 runs each POSIX stage in a new process session. A timeout now sends
`SIGTERM` to the complete process group, waits up to five seconds, and sends
`SIGKILL` if the group remains. The runner reaps the direct child before testing
the remaining group so a dead group leader is not mistaken for a live worker.
It also checks the process group after an apparently successful direct-process
exit. If a background descendant remains, the stage is cleaned up and fails
instead of accepting its artifacts.

Every attempted cleanup records:

- `mode`, distinguishing POSIX process-group cleanup from a direct-process
  fallback;
- whether `SIGTERM` and `SIGKILL` were sent; and
- whether process-group termination was observed before the runner continued.

## OOD controls

Dependency-free tests cover:

- a direct stage timeout with no descendants;
- a timed-out parent whose child handles `SIGTERM` and records receipt of the
  group signal;
- a child that ignores `SIGTERM`, requiring `SIGKILL` escalation;
- a parent that exits with code zero while leaving a background child; and
- the non-POSIX direct-process fallback, including its explicit refusal to
  claim process-tree verification; and
- the existing control-input mutation, artifact mutation, timeout, registry,
  environment, and stage-result invariants after replacing the subprocess
  implementation.

The complete dependency-free suite passed 230 tests locally on macOS. The
focused lifecycle suite passed 25 tests. The package has no Ruff dependency in
the local environment or hosted workflow, so no Ruff result is claimed. Python
3.11 built the 0.43 wheel successfully with SHA-256
`4bfa09fdc310d7d74970f8b9bbae0a41ce8a7ca8d3d85d71c896a7f7141a266b`.
The default Python 3.14 environment could not import `setuptools.build_meta`;
that host dependency failure is not recorded as a repository build failure.

## Scope and boundary

The POSIX cleanup applies to every model-specific backend executed through the
shared lifecycle, including LoRA and full-SFT recipes. It is independent of the
model family, accelerator, and stage command as long as descendants remain in
the new session's process group.

The non-POSIX fallback terminates only the direct process and explicitly does
not claim process-tree cleanup. A privileged or deliberately hostile
descendant may create a new session and escape the group. Process termination
does not roll back external writes, remote jobs, cloud operations, or files
already published before failure. Backends must still write into unique work
directories, publish only after validation, and make external side effects
idempotent or separately reversible.
