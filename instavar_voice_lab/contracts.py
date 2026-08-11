from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")

CAPABILITY_STATUSES = {
    "supported",
    "experimental",
    "upstream_only",
    "unverified_for_adapter",
    "unsupported",
}
EVIDENCE_LEVELS = {
    "repository_declared",
    "upstream_benchmark",
    "smoke_tested",
    "validated",
    "promoted",
    "negative_result",
}
EXPERIMENT_STATUSES = {
    "planned",
    "preflight_passed",
    "trained_unreviewed",
    "evaluation_limited",
    "evaluation_failed",
    "validated",
    "promoted",
    "blocked",
}
ADAPTATION_MODES = {
    "zero_shot",
    "lora",
    "full_sft",
    "partial_sft",
    "prompt_adapter",
}


@dataclass(frozen=True)
class ContractError:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _mapping(value: Any, path: str, errors: list[ContractError]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(ContractError(path, "must be an object"))
        return {}
    return value


def _list(value: Any, path: str, errors: list[ContractError]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(ContractError(path, "must be an array"))
        return []
    return value


def _required(obj: dict[str, Any], keys: set[str], path: str, errors: list[ContractError]) -> None:
    for key in sorted(keys - obj.keys()):
        errors.append(ContractError(f"{path}.{key}", "is required"))


def _nonempty_string(value: Any, path: str, errors: list[ContractError]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(ContractError(path, "must be a non-empty string"))
        return ""
    return value.strip()


def _enum(value: Any, allowed: set[str], path: str, errors: list[ContractError]) -> str:
    text = _nonempty_string(value, path, errors)
    if text and text not in allowed:
        errors.append(ContractError(path, f"must be one of: {', '.join(sorted(allowed))}"))
    return text


def _sha(value: Any, path: str, errors: list[ContractError], *, git: bool = False) -> None:
    text = _nonempty_string(value, path, errors)
    matcher = GIT_SHA_RE if git else SHA256_RE
    if text and not matcher.fullmatch(text):
        errors.append(ContractError(path, "must be a lowercase 40-character git SHA" if git else "must be a lowercase SHA-256 digest"))


def _evidence(value: Any, path: str, errors: list[ContractError]) -> list[Any]:
    rows = _list(value, path, errors)
    for index, row_value in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _mapping(row_value, row_path, errors)
        _required(row, {"level", "ref", "claim"}, row_path, errors)
        _enum(row.get("level"), EVIDENCE_LEVELS, f"{row_path}.level", errors)
        _nonempty_string(row.get("ref"), f"{row_path}.ref", errors)
        _nonempty_string(row.get("claim"), f"{row_path}.claim", errors)
    return rows


def validate_capability_manifest(document: Any) -> list[ContractError]:
    errors: list[ContractError] = []
    root = _mapping(document, "$", errors)
    _required(
        root,
        {"schema_version", "repository", "model", "adaptation", "runtimes", "evaluation", "rights", "known_gaps"},
        "$",
        errors,
    )
    if root.get("schema_version") != "1.0.0":
        errors.append(ContractError("$.schema_version", "must equal 1.0.0"))

    repository = _mapping(root.get("repository"), "$.repository", errors)
    _required(repository, {"slug", "url", "evidence_revision"}, "$.repository", errors)
    slug = _nonempty_string(repository.get("slug"), "$.repository.slug", errors)
    if slug and not SLUG_RE.fullmatch(slug):
        errors.append(ContractError("$.repository.slug", "must be a lowercase repository slug"))
    url = _nonempty_string(repository.get("url"), "$.repository.url", errors)
    if url and not url.startswith("https://github.com/"):
        errors.append(ContractError("$.repository.url", "must be an HTTPS GitHub URL"))
    _sha(repository.get("evidence_revision"), "$.repository.evidence_revision", errors, git=True)

    model = _mapping(root.get("model"), "$.model", errors)
    _required(model, {"family", "upstream_repository"}, "$.model", errors)
    _nonempty_string(model.get("family"), "$.model.family", errors)
    _nonempty_string(model.get("upstream_repository"), "$.model.upstream_repository", errors)

    adaptation = _mapping(root.get("adaptation"), "$.adaptation", errors)
    if not adaptation:
        errors.append(ContractError("$.adaptation", "must declare at least one adaptation mode"))
    for mode, capability_value in adaptation.items():
        capability_path = f"$.adaptation.{mode}"
        if mode not in ADAPTATION_MODES:
            errors.append(ContractError(capability_path, "uses an unknown adaptation mode"))
        capability = _mapping(capability_value, capability_path, errors)
        _required(capability, {"status", "evidence"}, capability_path, errors)
        status = _enum(capability.get("status"), CAPABILITY_STATUSES, f"{capability_path}.status", errors)
        evidence = _evidence(capability.get("evidence"), f"{capability_path}.evidence", errors)
        if status in {"supported", "experimental"} and not evidence:
            errors.append(ContractError(f"{capability_path}.evidence", "is required for supported or experimental capabilities"))

    runtimes = _list(root.get("runtimes"), "$.runtimes", errors)
    runtime_ids: set[str] = set()
    for index, runtime_value in enumerate(runtimes):
        runtime_path = f"$.runtimes[{index}]"
        runtime = _mapping(runtime_value, runtime_path, errors)
        _required(runtime, {"id", "status", "artifact_mode", "streaming", "evidence"}, runtime_path, errors)
        runtime_id = _nonempty_string(runtime.get("id"), f"{runtime_path}.id", errors)
        if runtime_id in runtime_ids:
            errors.append(ContractError(f"{runtime_path}.id", "must be unique"))
        runtime_ids.add(runtime_id)
        status = _enum(runtime.get("status"), CAPABILITY_STATUSES, f"{runtime_path}.status", errors)
        _enum(runtime.get("artifact_mode"), {"base", "adapter", "merged", "checkpoint"}, f"{runtime_path}.artifact_mode", errors)
        if not isinstance(runtime.get("streaming"), bool):
            errors.append(ContractError(f"{runtime_path}.streaming", "must be a boolean"))
        evidence = _evidence(runtime.get("evidence"), f"{runtime_path}.evidence", errors)
        if status == "supported" and not evidence:
            errors.append(ContractError(f"{runtime_path}.evidence", "is required for a supported runtime"))

    evaluation = _mapping(root.get("evaluation"), "$.evaluation", errors)
    _required(evaluation, {"prompt_pack", "objective_metrics", "listening_criteria"}, "$.evaluation", errors)
    prompt_pack = _mapping(evaluation.get("prompt_pack"), "$.evaluation.prompt_pack", errors)
    _required(prompt_pack, {"id", "version", "sha256"}, "$.evaluation.prompt_pack", errors)
    _nonempty_string(prompt_pack.get("id"), "$.evaluation.prompt_pack.id", errors)
    _nonempty_string(prompt_pack.get("version"), "$.evaluation.prompt_pack.version", errors)
    _sha(prompt_pack.get("sha256"), "$.evaluation.prompt_pack.sha256", errors)
    for name in ("objective_metrics", "listening_criteria"):
        values = _list(evaluation.get(name), f"$.evaluation.{name}", errors)
        if not values:
            errors.append(ContractError(f"$.evaluation.{name}", "must not be empty"))
        for index, value in enumerate(values):
            _nonempty_string(value, f"$.evaluation.{name}[{index}]", errors)

    rights = _mapping(root.get("rights"), "$.rights", errors)
    _required(rights, {"code", "weights", "dataset", "generated_output"}, "$.rights", errors)
    for key in ("code", "weights", "dataset", "generated_output"):
        _nonempty_string(rights.get(key), f"$.rights.{key}", errors)

    gaps = _list(root.get("known_gaps"), "$.known_gaps", errors)
    for index, value in enumerate(gaps):
        _nonempty_string(value, f"$.known_gaps[{index}]", errors)
    return errors


def validate_experiment_manifest(document: Any) -> list[ContractError]:
    errors: list[ContractError] = []
    root = _mapping(document, "$", errors)
    _required(
        root,
        {
            "schema_version",
            "experiment_id",
            "objective",
            "intended_use",
            "rights",
            "corpus",
            "backend",
            "adaptation_mode",
            "hardware",
            "preflight",
            "checkpoints",
            "evaluation_suite",
            "status",
            "promotion_rationale",
        },
        "$",
        errors,
    )
    if root.get("schema_version") != "1.0.0":
        errors.append(ContractError("$.schema_version", "must equal 1.0.0"))
    for key in ("experiment_id", "objective", "intended_use", "hardware", "promotion_rationale"):
        _nonempty_string(root.get(key), f"$.{key}", errors)
    _enum(root.get("adaptation_mode"), ADAPTATION_MODES, "$.adaptation_mode", errors)
    _enum(root.get("status"), EXPERIMENT_STATUSES, "$.status", errors)

    rights = _mapping(root.get("rights"), "$.rights", errors)
    _required(rights, {"model", "dataset", "consent", "distribution"}, "$.rights", errors)
    for key in ("model", "dataset", "consent", "distribution"):
        _nonempty_string(rights.get(key), f"$.rights.{key}", errors)

    corpus = _mapping(root.get("corpus"), "$.corpus", errors)
    _required(corpus, {"id", "sha256", "split_hashes"}, "$.corpus", errors)
    _nonempty_string(corpus.get("id"), "$.corpus.id", errors)
    _sha(corpus.get("sha256"), "$.corpus.sha256", errors)
    split_hashes = _mapping(corpus.get("split_hashes"), "$.corpus.split_hashes", errors)
    _required(split_hashes, {"train", "validation", "test"}, "$.corpus.split_hashes", errors)
    for key in ("train", "validation", "test"):
        _sha(split_hashes.get(key), f"$.corpus.split_hashes.{key}", errors)
    if len({split_hashes.get("train"), split_hashes.get("validation"), split_hashes.get("test")}) != 3:
        errors.append(ContractError("$.corpus.split_hashes", "train, validation, and test hashes must differ"))

    backend = _mapping(root.get("backend"), "$.backend", errors)
    _required(backend, {"name", "upstream_revision", "instavar_revision"}, "$.backend", errors)
    _nonempty_string(backend.get("name"), "$.backend.name", errors)
    _sha(backend.get("upstream_revision"), "$.backend.upstream_revision", errors, git=True)
    _sha(backend.get("instavar_revision"), "$.backend.instavar_revision", errors, git=True)

    preflight = _mapping(root.get("preflight"), "$.preflight", errors)
    _required(preflight, {"status", "criteria", "artifacts"}, "$.preflight", errors)
    _enum(preflight.get("status"), {"passed", "failed", "not_run"}, "$.preflight.status", errors)
    criteria = _list(preflight.get("criteria"), "$.preflight.criteria", errors)
    if not criteria:
        errors.append(ContractError("$.preflight.criteria", "must not be empty"))
    _list(preflight.get("artifacts"), "$.preflight.artifacts", errors)

    checkpoints = _list(root.get("checkpoints"), "$.checkpoints", errors)
    for index, checkpoint_value in enumerate(checkpoints):
        checkpoint_path = f"$.checkpoints[{index}]"
        checkpoint = _mapping(checkpoint_value, checkpoint_path, errors)
        _required(checkpoint, {"id", "path", "sha256", "status"}, checkpoint_path, errors)
        _nonempty_string(checkpoint.get("id"), f"{checkpoint_path}.id", errors)
        _nonempty_string(checkpoint.get("path"), f"{checkpoint_path}.path", errors)
        _sha(checkpoint.get("sha256"), f"{checkpoint_path}.sha256", errors)
        _enum(checkpoint.get("status"), {"candidate", "selected", "rejected", "quarantined"}, f"{checkpoint_path}.status", errors)

    evaluation_suite = _mapping(root.get("evaluation_suite"), "$.evaluation_suite", errors)
    _required(evaluation_suite, {"id", "revision"}, "$.evaluation_suite", errors)
    _nonempty_string(evaluation_suite.get("id"), "$.evaluation_suite.id", errors)
    _sha(evaluation_suite.get("revision"), "$.evaluation_suite.revision", errors, git=True)
    return errors


def validate_evaluation_report(document: Any) -> list[ContractError]:
    errors: list[ContractError] = []
    root = _mapping(document, "$", errors)
    _required(
        root,
        {"schema_version", "evaluation_id", "experiment_id", "prompt_pack", "candidates", "objective_results", "listening", "verdict"},
        "$",
        errors,
    )
    if "composite_score" in root:
        errors.append(ContractError("$.composite_score", "is forbidden because the criteria measure different properties"))
    if root.get("schema_version") != "1.0.0":
        errors.append(ContractError("$.schema_version", "must equal 1.0.0"))
    for key in ("evaluation_id", "experiment_id", "verdict"):
        _nonempty_string(root.get(key), f"$.{key}", errors)

    prompt_pack = _mapping(root.get("prompt_pack"), "$.prompt_pack", errors)
    _required(prompt_pack, {"id", "version", "sha256"}, "$.prompt_pack", errors)
    _nonempty_string(prompt_pack.get("id"), "$.prompt_pack.id", errors)
    _nonempty_string(prompt_pack.get("version"), "$.prompt_pack.version", errors)
    _sha(prompt_pack.get("sha256"), "$.prompt_pack.sha256", errors)

    candidates = _list(root.get("candidates"), "$.candidates", errors)
    if len(candidates) < 2:
        errors.append(ContractError("$.candidates", "must include at least a baseline and one candidate"))
    candidate_ids: set[str] = set()
    for index, candidate_value in enumerate(candidates):
        path = f"$.candidates[{index}]"
        candidate = _mapping(candidate_value, path, errors)
        _required(candidate, {"id", "artifact_ref", "revision", "role"}, path, errors)
        candidate_id = _nonempty_string(candidate.get("id"), f"{path}.id", errors)
        if candidate_id in candidate_ids:
            errors.append(ContractError(f"{path}.id", "must be unique"))
        candidate_ids.add(candidate_id)
        _nonempty_string(candidate.get("artifact_ref"), f"{path}.artifact_ref", errors)
        _sha(candidate.get("revision"), f"{path}.revision", errors, git=True)
        _enum(candidate.get("role"), {"baseline", "candidate"}, f"{path}.role", errors)
    if candidates and not any(isinstance(value, dict) and value.get("role") == "baseline" for value in candidates):
        errors.append(ContractError("$.candidates", "must include a baseline"))

    objective_results = _list(root.get("objective_results"), "$.objective_results", errors)
    for index, result_value in enumerate(objective_results):
        path = f"$.objective_results[{index}]"
        result = _mapping(result_value, path, errors)
        _required(result, {"candidate_id", "metric", "value", "scope", "evidence_ref"}, path, errors)
        candidate_id = _nonempty_string(result.get("candidate_id"), f"{path}.candidate_id", errors)
        if candidate_id and candidate_id not in candidate_ids:
            errors.append(ContractError(f"{path}.candidate_id", "must refer to a declared candidate"))
        _nonempty_string(result.get("metric"), f"{path}.metric", errors)
        if not isinstance(result.get("value"), (int, float)) or isinstance(result.get("value"), bool):
            errors.append(ContractError(f"{path}.value", "must be numeric"))
        _nonempty_string(result.get("scope"), f"{path}.scope", errors)
        _nonempty_string(result.get("evidence_ref"), f"{path}.evidence_ref", errors)

    listening = _mapping(root.get("listening"), "$.listening", errors)
    _required(listening, {"status", "blind", "criteria", "review_export", "reveal_mapping"}, "$.listening", errors)
    _enum(listening.get("status"), {"not_run", "complete", "limited"}, "$.listening.status", errors)
    if listening.get("blind") is not True:
        errors.append(ContractError("$.listening.blind", "must be true for a promotable evaluation"))
    criteria = _list(listening.get("criteria"), "$.listening.criteria", errors)
    if not criteria:
        errors.append(ContractError("$.listening.criteria", "must not be empty"))
    _nonempty_string(listening.get("review_export"), "$.listening.review_export", errors)
    _nonempty_string(listening.get("reveal_mapping"), "$.listening.reveal_mapping", errors)
    return errors


def validate_package_manifest(document: Any) -> list[ContractError]:
    errors: list[ContractError] = []
    root = _mapping(document, "$", errors)
    _required(
        root,
        {
            "schema_version",
            "package_id",
            "base_model",
            "artifact",
            "code_revision",
            "inference_configuration",
            "rights_note",
            "known_failure_modes",
            "evaluation_report",
            "smoke_command",
        },
        "$",
        errors,
    )
    if root.get("schema_version") != "1.0.0":
        errors.append(ContractError("$.schema_version", "must equal 1.0.0"))
    for key in ("package_id", "base_model", "rights_note", "evaluation_report", "smoke_command"):
        _nonempty_string(root.get(key), f"$.{key}", errors)
    artifact = _mapping(root.get("artifact"), "$.artifact", errors)
    _required(artifact, {"path", "sha256", "bytes", "kind"}, "$.artifact", errors)
    _nonempty_string(artifact.get("path"), "$.artifact.path", errors)
    _sha(artifact.get("sha256"), "$.artifact.sha256", errors)
    if not isinstance(artifact.get("bytes"), int) or artifact.get("bytes", 0) <= 0:
        errors.append(ContractError("$.artifact.bytes", "must be a positive integer"))
    _enum(artifact.get("kind"), {"adapter", "merged_model", "checkpoint"}, "$.artifact.kind", errors)
    _sha(root.get("code_revision"), "$.code_revision", errors, git=True)
    _mapping(root.get("inference_configuration"), "$.inference_configuration", errors)
    failures = _list(root.get("known_failure_modes"), "$.known_failure_modes", errors)
    for index, value in enumerate(failures):
        _nonempty_string(value, f"$.known_failure_modes[{index}]", errors)
    return errors


VALIDATORS: dict[str, Callable[[Any], list[ContractError]]] = {
    "capability": validate_capability_manifest,
    "experiment": validate_experiment_manifest,
    "evaluation": validate_evaluation_report,
    "package": validate_package_manifest,
}


def validate_document(kind: str, document: Any) -> list[ContractError]:
    try:
        validator = VALIDATORS[kind]
    except KeyError as error:
        raise ValueError(f"unknown contract kind: {kind}") from error
    return validator(document)
