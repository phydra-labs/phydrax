from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phydrax.nn.operator.adapters import (
    OperatorCheckpointManifest,
    verify_operator_checkpoint,
)

from .matrix import OperatorBenchmarkAggregate


@dataclass(frozen=True)
class ExternalOperatorCandidate:
    """Procurement metadata recorded before importing an external checkpoint."""

    name: str
    source_uri: str
    checkpoint_uri: str
    revision: str
    code_license: str | None
    weights_license: str | None
    input_schema_declared: bool
    output_schema_declared: bool
    preprocessing_declared: bool
    normalization_declared: bool
    dataset_provenance_declared: bool
    checkpoint_sha256: str | None


@dataclass(frozen=True)
class ExternalCandidateAudit:
    candidate: str
    eligible: bool
    reasons: tuple[str, ...]
    artifact_verified: bool = False


@dataclass(frozen=True)
class ExternalCandidateDecision:
    candidate: str
    integrated: bool
    reasons: tuple[str, ...]
    compared_regimes: int
    accuracy_passed: bool
    robustness_passed: bool
    efficiency_passed: bool
    complexity_passed: bool


def audit_external_candidate(
    candidate: ExternalOperatorCandidate,
    /,
    *,
    allowed_code_licenses: tuple[str, ...] = (
        "Apache-2.0",
        "BSD-3-Clause",
        "MIT",
    ),
    allowed_weights_licenses: tuple[str, ...] = (
        "Apache-2.0",
        "BSD-3-Clause",
        "MIT",
    ),
) -> ExternalCandidateAudit:
    """Reject candidates lacking an explicit reproducibility or license field."""
    reasons = []
    required_strings = {
        "source URI": candidate.source_uri,
        "checkpoint URI": candidate.checkpoint_uri,
        "immutable revision": candidate.revision,
    }
    reasons.extend(
        f"missing {name}" for name, value in required_strings.items() if not value
    )
    if candidate.code_license not in allowed_code_licenses:
        reasons.append("code license is absent or not approved")
    if candidate.weights_license not in allowed_weights_licenses:
        reasons.append("weights license is absent or not approved")
    declarations = {
        "input schema": candidate.input_schema_declared,
        "output schema": candidate.output_schema_declared,
        "preprocessing": candidate.preprocessing_declared,
        "normalization": candidate.normalization_declared,
        "dataset provenance": candidate.dataset_provenance_declared,
    }
    reasons.extend(
        f"missing {name} contract"
        for name, declared in declarations.items()
        if not declared
    )
    checksum = candidate.checkpoint_sha256
    if (
        checksum is None
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum.lower())
    ):
        reasons.append("missing valid checkpoint SHA-256")
    return ExternalCandidateAudit(
        candidate=candidate.name,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def verify_external_candidate_artifact(
    candidate: ExternalOperatorCandidate,
    manifest: OperatorCheckpointManifest,
    checkpoint_path: str | Path,
    /,
) -> ExternalCandidateAudit:
    """Verify candidate identity and bytes against a loadable PhydraX manifest."""
    audit = audit_external_candidate(candidate)
    reasons = list(audit.reasons)
    identity = {
        "name": (candidate.name, manifest.architecture),
        "source URI": (candidate.source_uri, manifest.source_uri),
        "checkpoint URI": (candidate.checkpoint_uri, manifest.checkpoint_uri),
        "revision": (candidate.revision, manifest.revision),
        "code license": (candidate.code_license, manifest.code_license),
        "weights license": (candidate.weights_license, manifest.weights_license),
        "checkpoint SHA-256": (
            candidate.checkpoint_sha256,
            manifest.checkpoint_sha256,
        ),
    }
    reasons.extend(
        f"manifest {name} mismatch"
        for name, (expected, actual) in identity.items()
        if expected != actual
    )
    if not verify_operator_checkpoint(checkpoint_path, manifest):
        reasons.append("checkpoint bytes do not match manifest SHA-256")
    return ExternalCandidateAudit(
        candidate.name,
        not reasons,
        tuple(reasons),
        artifact_verified=not reasons,
    )


def select_benchmark_superior_external(
    candidate: ExternalOperatorCandidate,
    audit: ExternalCandidateAudit,
    candidate_results: tuple[OperatorBenchmarkAggregate, ...],
    native_results: tuple[OperatorBenchmarkAggregate, ...],
    /,
    *,
    maximum_error_ratio: float = 0.95,
    maximum_latency_ratio: float = 2.0,
    maximum_parameter_ratio: float = 2.0,
    maximum_robustness_ratio: float = 1.05,
) -> ExternalCandidateDecision:
    """Approve only candidates beating the best native model in every tested regime."""
    if not 0.0 < float(maximum_error_ratio) < 1.0:
        raise ValueError("maximum_error_ratio must lie strictly between zero and one.")
    if float(maximum_latency_ratio) <= 0.0:
        raise ValueError("maximum_latency_ratio must be positive.")
    if float(maximum_parameter_ratio) <= 0.0:
        raise ValueError("maximum_parameter_ratio must be positive.")
    if float(maximum_robustness_ratio) <= 0.0:
        raise ValueError("maximum_robustness_ratio must be positive.")
    reasons = list(audit.reasons)
    if not audit.eligible:
        reasons.append("candidate failed provenance and license audit")
    if not audit.artifact_verified:
        reasons.append("candidate checkpoint artifact was not verified")
    accuracy_passed = True
    robustness_passed = True
    efficiency_passed = True
    complexity_passed = True
    native_by_regime: dict[tuple[str, str], list[OperatorBenchmarkAggregate]] = {}
    for result in native_results:
        native_by_regime.setdefault((result.scenario, result.evaluation), []).append(
            result
        )
    compared = 0
    for result in candidate_results:
        regime = (result.scenario, result.evaluation)
        native = native_by_regime.get(regime, [])
        if not native:
            reasons.append(f"no native comparison for {regime}")
            continue
        compared += 1
        best = min(native, key=lambda row: row.relative_l2_mean)
        if result.relative_l2_mean > maximum_error_ratio * best.relative_l2_mean:
            reasons.append(f"does not improve relative L2 for {regime}")
            accuracy_passed = False
        if (
            result.inference_seconds_mean
            > maximum_latency_ratio * best.inference_seconds_mean
        ):
            reasons.append(f"exceeds latency budget for {regime}")
            efficiency_passed = False
        if (
            result.parameter_count_mean
            > maximum_parameter_ratio * best.parameter_count_mean
        ):
            reasons.append(f"exceeds parameter-count budget for {regime}")
            complexity_passed = False
    native_base = {}
    candidate_base = {}
    for result in native_results:
        if result.shift == "in_distribution":
            native_base.setdefault(result.scenario, []).append(result)
    for result in candidate_results:
        if result.shift == "in_distribution":
            candidate_base.setdefault(result.scenario, []).append(result)
    for result in candidate_results:
        if result.shift == "in_distribution":
            continue
        native_rows = native_by_regime.get((result.scenario, result.evaluation), ())
        native_baselines = native_base.get(result.scenario, ())
        candidate_baselines = candidate_base.get(result.scenario, ())
        if not native_rows or not native_baselines or not candidate_baselines:
            continue
        candidate_degradation = result.relative_l2_mean / max(
            min(row.relative_l2_mean for row in candidate_baselines),
            1e-12,
        )
        native_degradation = min(row.relative_l2_mean for row in native_rows) / max(
            min(row.relative_l2_mean for row in native_baselines),
            1e-12,
        )
        if candidate_degradation > maximum_robustness_ratio * native_degradation:
            reasons.append(
                f"degrades more severely under shift for "
                f"{(result.scenario, result.evaluation)}"
            )
            robustness_passed = False
    expected_regimes = set(native_by_regime)
    candidate_regimes = {
        (result.scenario, result.evaluation) for result in candidate_results
    }
    missing = sorted(expected_regimes.difference(candidate_regimes))
    reasons.extend(f"missing benchmark regime {regime}" for regime in missing)
    if not candidate_results:
        reasons.append("no candidate benchmark results")
    return ExternalCandidateDecision(
        candidate=candidate.name,
        integrated=not reasons,
        reasons=tuple(reasons),
        compared_regimes=compared,
        accuracy_passed=accuracy_passed,
        robustness_passed=robustness_passed,
        efficiency_passed=efficiency_passed,
        complexity_passed=complexity_passed,
    )


__all__ = [
    "ExternalCandidateAudit",
    "ExternalCandidateDecision",
    "ExternalOperatorCandidate",
    "audit_external_candidate",
    "select_benchmark_superior_external",
    "verify_external_candidate_artifact",
]
