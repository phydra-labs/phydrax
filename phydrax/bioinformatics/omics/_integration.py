#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...transport import AbstractBalancedTransportPlan
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


INTEGRATION_SUCCESS = 0
INTEGRATION_TRANSPORT_NONCONVERGED = 1
INTEGRATION_PROVENANCE_MISMATCH = 2
INTEGRATION_INVALID_MARGINAL = 3
INTEGRATION_NONFINITE = 4
INTEGRATION_CONFIRMATORY_USE_FORBIDDEN = 5
INTEGRATION_EMPTY_TRAINING_FIT = 6


def integration_status_name(status: int, /) -> str:
    """Return the stable name of a transport-integration status code."""
    names = (
        "success",
        "transport_nonconverged",
        "fitted_provenance_mismatch",
        "marginal_diagnostic_failed",
        "nonfinite_embedding",
        "confirmatory_use_forbidden",
        "empty_training_fit",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown transport-integration status {code}.")
    return names[code]


def _integration_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exploratory_regularized_transport_integration",
        MethodKind.RELAXED_OBJECTIVE,
        ExecutionKind.ITERATIVE_TOLERANCE,
        DifferentiationKind.IMPLICIT,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Integrated coordinates condition on a converged balanced transport plan "
            "fitted under the declared training provenance."
        ),
        truncation_statement="Transport atoms and embedding coordinates are not truncated.",
        capacity_semantics="Transport capacity is inherited from the native plan.",
        assumptions=(
            "Source and target coordinates inhabit a shared exploratory latent space.",
            "Corrected coordinates are not confirmatory test inputs.",
        ),
        nondifferentiable_outputs=("status", "valid"),
    )


class TransportIntegrationPlan(StrictModule):
    """Training split, expected marginals, and exploratory-use contract."""

    source_training_mask: Array
    target_training_mask: Array
    expected_source_marginal: Array
    expected_target_marginal: Array
    fitted_on_split_id: str = eqx.field(static=True)
    marginal_tolerance: float = eqx.field(static=True)
    exploratory_only: bool = eqx.field(static=True)
    transport_approximation: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_training_mask: ArrayLike,
        target_training_mask: ArrayLike,
        expected_source_marginal: ArrayLike,
        expected_target_marginal: ArrayLike,
        /,
        *,
        fitted_on_split_id: str,
        marginal_tolerance: float = 1e-5,
        exploratory_only: bool = True,
        transport_approximation: str = "entropic_relaxation",
    ):
        source_training = jnp.asarray(source_training_mask, dtype=bool)
        target_training = jnp.asarray(target_training_mask, dtype=bool)
        source_marginal = jnp.asarray(expected_source_marginal)
        target_marginal = jnp.asarray(expected_target_marginal)
        if (
            source_training.ndim != 1
            or target_training.ndim != 1
            or source_marginal.shape != source_training.shape
            or target_marginal.shape != target_training.shape
        ):
            raise ValueError(
                "Training masks and expected marginals must be matching vectors."
            )
        tolerance = float(marginal_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("marginal_tolerance must be finite and nonnegative.")
        if not fitted_on_split_id or not transport_approximation:
            raise ValueError("Integration provenance strings must be non-empty.")
        self.source_training_mask = source_training
        self.target_training_mask = target_training
        self.expected_source_marginal = source_marginal
        self.expected_target_marginal = target_marginal
        self.fitted_on_split_id = str(fitted_on_split_id)
        self.marginal_tolerance = tolerance
        self.exploratory_only = bool(exploratory_only)
        self.transport_approximation = str(transport_approximation)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transport-integration-plan",
                "fitted_on_split_id": self.fitted_on_split_id,
                "marginal_tolerance": tolerance,
                "exploratory_only": self.exploratory_only,
                "transport_approximation": self.transport_approximation,
                "arrays": array_tree_fingerprint(
                    (
                        source_training,
                        target_training,
                        source_marginal,
                        target_marginal,
                    )
                ),
            }
        )


class TransportIntegrationEvidence(StrictModule):
    """Native transport convergence, marginal, cycle, and provenance diagnostics."""

    transport_converged: Array
    source_marginal_residual: Array
    target_marginal_residual: Array
    source_cycle_residual: Array
    target_cycle_residual: Array
    regularized_objective: Array
    source_training_count: Array
    target_training_count: Array
    fitted_provenance_match: Array
    exploratory_use: Array
    approximation: str = eqx.field(static=True)


class TransportIntegrationResult(StrictModule):
    """Exploratory barycentric coordinates with no confirmatory interpretation."""

    integrated_source: Array
    integrated_target: Array
    source_transport_projection: Array
    target_transport_projection: Array
    valid: Array
    status: Array
    evidence: TransportIntegrationEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def transport_exploratory_integration(
    transport_plan: AbstractBalancedTransportPlan,
    source_coordinates: ArrayLike,
    target_coordinates: ArrayLike,
    plan: TransportIntegrationPlan,
    /,
    *,
    expected_split_id: str,
    requested_use: str = "exploratory",
    method_contract: BioinformaticsMethodContract | None = None,
) -> TransportIntegrationResult:
    """Barycentrically integrate modalities using a native balanced transport plan."""
    if not isinstance(transport_plan, AbstractBalancedTransportPlan):
        raise TypeError("transport_plan must implement AbstractBalancedTransportPlan.")
    if not isinstance(plan, TransportIntegrationPlan):
        raise TypeError("plan must be TransportIntegrationPlan.")
    source = jnp.asarray(source_coordinates)
    target = jnp.asarray(target_coordinates)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Transport coordinates must be rank two with a shared width.")
    if (
        source.shape[0] != plan.source_training_mask.shape[0]
        or target.shape[0] != plan.target_training_mask.shape[0]
    ):
        raise ValueError("Coordinate atom axes must match integration plan marginals.")
    if not expected_split_id or requested_use not in ("exploratory", "confirmatory"):
        raise ValueError("expected_split_id and requested_use must be valid.")

    source_projection = transport_plan.barycentric_target_to_source(target)
    target_projection = transport_plan.barycentric_source_to_target(source)
    if source_projection.shape != source.shape or target_projection.shape != target.shape:
        raise ValueError("Transport barycentric projections must preserve atom layouts.")
    integrated_source = 0.5 * (source + source_projection)
    integrated_target = 0.5 * (target + target_projection)
    source_cycle = transport_plan.barycentric_target_to_source(target_projection)
    target_cycle = transport_plan.barycentric_source_to_target(source_projection)
    source_cycle_residual = jnp.sqrt(jnp.mean((source_cycle - source) ** 2))
    target_cycle_residual = jnp.sqrt(jnp.mean((target_cycle - target) ** 2))

    actual_source_marginal = transport_plan.source_marginal()
    actual_target_marginal = transport_plan.target_marginal()
    if (
        actual_source_marginal.shape != plan.expected_source_marginal.shape
        or actual_target_marginal.shape != plan.expected_target_marginal.shape
    ):
        raise ValueError("Expected marginals must match the native transport plan.")
    source_marginal_residual = jnp.max(
        jnp.abs(actual_source_marginal - plan.expected_source_marginal)
    )
    target_marginal_residual = jnp.max(
        jnp.abs(actual_target_marginal - plan.expected_target_marginal)
    )
    marginal_valid = (source_marginal_residual <= plan.marginal_tolerance) & (
        target_marginal_residual <= plan.marginal_tolerance
    )
    finite = (
        jnp.all(jnp.isfinite(integrated_source))
        & jnp.all(jnp.isfinite(integrated_target))
        & jnp.isfinite(source_cycle_residual)
        & jnp.isfinite(target_cycle_residual)
    )
    source_training_count = jnp.sum(plan.source_training_mask, dtype=jnp.int32)
    target_training_count = jnp.sum(plan.target_training_mask, dtype=jnp.int32)
    training_present = (source_training_count > 0) & (target_training_count > 0)
    provenance_match = jnp.asarray(plan.fitted_on_split_id == str(expected_split_id))
    exploratory_use = jnp.asarray(requested_use == "exploratory")
    use_valid = exploratory_use | jnp.asarray(not plan.exploratory_only)
    converged = jnp.asarray(transport_plan.converged)
    valid = (
        converged
        & provenance_match
        & marginal_valid
        & finite
        & use_valid
        & training_present
    )
    status = jnp.where(
        ~converged,
        INTEGRATION_TRANSPORT_NONCONVERGED,
        jnp.where(
            ~provenance_match,
            INTEGRATION_PROVENANCE_MISMATCH,
            jnp.where(
                ~marginal_valid,
                INTEGRATION_INVALID_MARGINAL,
                jnp.where(
                    ~finite,
                    INTEGRATION_NONFINITE,
                    jnp.where(
                        ~use_valid,
                        INTEGRATION_CONFIRMATORY_USE_FORBIDDEN,
                        jnp.where(
                            training_present,
                            INTEGRATION_SUCCESS,
                            INTEGRATION_EMPTY_TRAINING_FIT,
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = TransportIntegrationEvidence(
        converged,
        source_marginal_residual,
        target_marginal_residual,
        source_cycle_residual,
        target_cycle_residual,
        transport_plan.regularized_objective(),
        source_training_count,
        target_training_count,
        provenance_match,
        exploratory_use,
        plan.transport_approximation,
    )
    return TransportIntegrationResult(
        integrated_source,
        integrated_target,
        source_projection,
        target_projection,
        valid,
        status,
        evidence,
        method_contract if method_contract is not None else _integration_contract(),
        "exploratory_regularized_transport",
    )


__all__ = [
    "INTEGRATION_CONFIRMATORY_USE_FORBIDDEN",
    "INTEGRATION_EMPTY_TRAINING_FIT",
    "INTEGRATION_INVALID_MARGINAL",
    "INTEGRATION_NONFINITE",
    "INTEGRATION_PROVENANCE_MISMATCH",
    "INTEGRATION_SUCCESS",
    "INTEGRATION_TRANSPORT_NONCONVERGED",
    "TransportIntegrationEvidence",
    "TransportIntegrationPlan",
    "TransportIntegrationResult",
    "integration_status_name",
    "transport_exploratory_integration",
]
