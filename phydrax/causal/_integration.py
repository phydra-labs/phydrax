#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._array_archive import (
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..uq import (
    ExpectedUtilityResult,
    ExperimentalBatchConstraints,
    ExperimentalBatchPlan,
    ExperimentalDesignCandidate,
    select_experimental_batch,
)
from ._estimate import (
    CausalDiagnostic,
    CausalEstimate,
    DiagnosticOutcome,
    EstimationStatus,
    EstimatorKind,
)
from ._identify import IdentificationCertificate
from ._scm import MechanismRegime


class CausalClaimBasis(enum.StrEnum):
    DESIGN_IDENTIFIED = "design_identified"
    GRAPH_IDENTIFIED = "graph_identified"
    EQUIVALENCE_CLASS_CONDITIONAL = "equivalence_class_conditional"
    MODEL_BASED = "model_based"
    COUNTERFACTUAL_MODEL_BASED = "counterfactual_model_based"


class CausalQualificationStatus(enum.StrEnum):
    ELIGIBLE_CONDITIONAL = "eligible_conditional"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class CausalInterventionCandidate(StrictModule, NonTrainableState):
    regime: MechanismRegime
    observation_plan_id: str = eqx.field(static=True)
    candidate: ExperimentalDesignCandidate = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        regime: MechanismRegime,
        observation_plan_id: str,
        cost: float,
        feasibility_group: str,
        prediction_source_id: str,
        setup_id: str = "",
        setup_cost: float = 0.0,
        diversity_group: str | None = None,
        mandatory_control: bool = False,
    ) -> None:
        observation = str(observation_plan_id).strip()
        if not observation:
            raise ValueError("observation_plan_id must be non-empty.")
        candidate_id = canonical_fingerprint(
            {
                "regime_id": regime.regime_id,
                "observation_plan_id": observation,
            }
        )
        candidate = ExperimentalDesignCandidate(
            candidate_id,
            regime.regime_id,
            cost,
            feasibility_group,
            prediction_source_id,
            setup_id=setup_id,
            setup_cost=setup_cost,
            diversity_group=diversity_group,
            mandatory_control=mandatory_control,
        )
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "observation_plan_id", observation)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "candidate_id", candidate_id)


class CausalQualificationReport(StrictModule, NonTrainableState):
    status: CausalQualificationStatus = eqx.field(static=True)
    basis: CausalClaimBasis = eqx.field(static=True)
    estimate_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    diagnostic_ids: tuple[str, ...] = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    @property
    def eligible(self) -> bool:
        return self.status is CausalQualificationStatus.ELIGIBLE_CONDITIONAL


def select_causal_intervention_batch(
    candidates: Sequence[CausalInterventionCandidate],
    utility: ExpectedUtilityResult,
    constraints: ExperimentalBatchConstraints,
    *,
    objective_id: str,
    model_ids: Sequence[str],
    analysis_id: str,
) -> ExperimentalBatchPlan:
    if not candidates:
        raise ValueError("At least one causal intervention candidate is required.")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("Causal intervention candidate identities must be unique.")
    return select_experimental_batch(
        tuple(candidate.candidate for candidate in candidates),
        utility,
        constraints,
        objective_id=objective_id,
        model_ids=model_ids,
        analysis_id=analysis_id,
    )


def assess_causal_estimate(
    estimate: CausalEstimate,
    certificate: IdentificationCertificate,
    diagnostics: Sequence[CausalDiagnostic] = (),
) -> CausalQualificationReport:
    """Assess release eligibility conditional on declared causal assumptions."""
    if estimate.certificate_id != certificate.certificate_id:
        raise ValueError("Estimate and identification certificate identities differ.")
    diagnostic_ids = tuple(diagnostic.diagnostic_id for diagnostic in diagnostics)
    rejected = any(
        diagnostic.outcome is DiagnosticOutcome.REJECTED for diagnostic in diagnostics
    )
    inconclusive = any(
        diagnostic.outcome
        in {
            DiagnosticOutcome.INCONCLUSIVE,
            DiagnosticOutcome.UNSUPPORTED,
            DiagnosticOutcome.NOT_RUN,
        }
        for diagnostic in diagnostics
    )
    overlap_evidence = any(
        diagnostic.criterion == "finite_sample_overlap"
        and diagnostic.outcome is DiagnosticOutcome.NOT_REJECTED
        for diagnostic in diagnostics
    )
    uncertainty_available = bool(jnp.isfinite(estimate.standard_error))
    if int(estimate.status) != int(EstimationStatus.SUCCESS) or rejected:
        status = CausalQualificationStatus.REJECTED
        reason = "The causal estimate or a required diagnostic failed."
    elif inconclusive or not overlap_evidence or not uncertainty_available:
        status = CausalQualificationStatus.INCONCLUSIVE
        reason = "Required overlap or inferential-uncertainty evidence is unresolved."
    else:
        status = CausalQualificationStatus.ELIGIBLE_CONDITIONAL
        reason = (
            "Numerical and diagnostic gates passed conditional on the certificate's "
            "declared assumptions; reviewed scientific evidence is still required."
        )
    if certificate.basis.value == "randomized_design":
        basis = CausalClaimBasis.DESIGN_IDENTIFIED
    elif certificate.basis.value == "equivalence_class_adjustment":
        basis = CausalClaimBasis.EQUIVALENCE_CLASS_CONDITIONAL
    else:
        basis = CausalClaimBasis.GRAPH_IDENTIFIED
    payload = {
        "status": status.value,
        "basis": basis.value,
        "estimate_id": estimate.result_id,
        "certificate_id": certificate.certificate_id,
        "diagnostic_ids": diagnostic_ids,
        "reason": reason,
    }
    return CausalQualificationReport(
        status=status,
        basis=basis,
        estimate_id=estimate.result_id,
        certificate_id=certificate.certificate_id,
        diagnostic_ids=diagnostic_ids,
        reason=reason,
        report_id=canonical_fingerprint(payload),
    )


def export_causal_estimate(
    path: str | Path,
    estimate: CausalEstimate,
) -> Path:
    manifest = {
        "kind": "phydrax-causal-estimate",
        "estimator_kind": estimate.estimator_kind.value,
        "certificate_id": estimate.certificate_id,
        "data_id": estimate.data_id,
        "nuisance_id": estimate.nuisance_id,
        "overlap_id": estimate.overlap_id,
        "uncertainty_basis": estimate.uncertainty_basis,
        "result_id": estimate.result_id,
    }
    arrays = {
        "status": estimate.status,
        "active_mean": estimate.active_mean,
        "reference_mean": estimate.reference_mean,
        "effect": estimate.effect,
        "standard_error": estimate.standard_error,
        "interval_lower": estimate.interval_lower,
        "interval_upper": estimate.interval_upper,
        "influence": estimate.influence,
    }
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def read_causal_estimate(
    path: str | Path,
    *,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> CausalEstimate:
    manifest, arrays = read_array_archive(path, limits=limits)
    expected_manifest = {
        "kind",
        "estimator_kind",
        "certificate_id",
        "data_id",
        "nuisance_id",
        "overlap_id",
        "uncertainty_basis",
        "result_id",
        "arrays",
    }
    expected_arrays = {
        "status",
        "active_mean",
        "reference_mean",
        "effect",
        "standard_error",
        "interval_lower",
        "interval_upper",
        "influence",
    }
    if set(manifest) != expected_manifest or set(arrays) != expected_arrays:
        raise ValueError("Causal estimate archive has invalid members.")
    if manifest["kind"] != "phydrax-causal-estimate":
        raise ValueError("Archive does not contain a causal estimate.")
    estimate = CausalEstimate(
        status=jnp.asarray(arrays["status"]),
        active_mean=jnp.asarray(arrays["active_mean"]),
        reference_mean=jnp.asarray(arrays["reference_mean"]),
        effect=jnp.asarray(arrays["effect"]),
        standard_error=jnp.asarray(arrays["standard_error"]),
        interval_lower=jnp.asarray(arrays["interval_lower"]),
        interval_upper=jnp.asarray(arrays["interval_upper"]),
        influence=jnp.asarray(arrays["influence"]),
        estimator_kind=EstimatorKind(manifest["estimator_kind"]),
        certificate_id=_manifest_id(manifest, "certificate_id"),
        data_id=_manifest_id(manifest, "data_id"),
        nuisance_id=_manifest_id(manifest, "nuisance_id"),
        overlap_id=_manifest_id(manifest, "overlap_id"),
        uncertainty_basis=_manifest_id(manifest, "uncertainty_basis"),
        result_id=_manifest_id(manifest, "result_id"),
    )
    expected_id = canonical_fingerprint(
        {
            "status": int(estimate.status),
            "kind": estimate.estimator_kind.value,
            "active_mean": estimate.active_mean,
            "reference_mean": estimate.reference_mean,
            "effect": estimate.effect,
            "standard_error": estimate.standard_error,
            "interval_lower": estimate.interval_lower,
            "interval_upper": estimate.interval_upper,
            "certificate_id": estimate.certificate_id,
            "data_id": estimate.data_id,
            "nuisance_id": estimate.nuisance_id,
            "overlap_id": estimate.overlap_id,
            "uncertainty_basis": estimate.uncertainty_basis,
        }
    )
    if estimate.result_id != expected_id:
        raise ValueError("Causal estimate archive identity is corrupt.")
    return estimate


def _manifest_id(manifest: dict[str, Any], name: str) -> str:
    value = manifest[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Archive field {name!r} must be a non-empty string.")
    return value


__all__ = [
    "CausalClaimBasis",
    "CausalInterventionCandidate",
    "CausalQualificationReport",
    "CausalQualificationStatus",
    "assess_causal_estimate",
    "export_causal_estimate",
    "read_causal_estimate",
    "select_causal_intervention_batch",
]
