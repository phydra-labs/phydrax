#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._actions import (
    BFSSConfiguration,
    PreparedBFSSAction,
    PreparedTwistedSYMAction,
    transform_bfss_configuration,
    transform_twisted_configuration,
    TwistedSYMConfiguration,
)


class PfaffianControlPlan(StrictModule):
    maximum_dimension: int = eqx.field(static=True)
    antisymmetry_tolerance: float = eqx.field(static=True)
    determinant_tolerance: float = eqx.field(static=True)
    minimum_magnitude: float = eqx.field(static=True)
    maximum_phase_magnitude: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_dimension: int = 12,
        antisymmetry_tolerance: float = 1e-10,
        determinant_tolerance: float = 1e-8,
        minimum_magnitude: float = 1e-14,
        maximum_phase_magnitude: float = np.pi,
    ):
        maximum = int(maximum_dimension)
        antisymmetry = float(antisymmetry_tolerance)
        determinant = float(determinant_tolerance)
        minimum = float(minimum_magnitude)
        phase = float(maximum_phase_magnitude)
        if maximum < 2 or maximum % 2:
            raise ValueError("maximum_dimension must be a positive even dimension.")
        if antisymmetry < 0.0 or determinant < 0.0 or minimum < 0.0:
            raise ValueError("Pfaffian tolerances must be nonnegative.")
        if not 0.0 <= phase <= np.pi:
            raise ValueError("maximum_phase_magnitude must lie in [0, pi].")
        if not all(
            np.isfinite(value) for value in (antisymmetry, determinant, minimum, phase)
        ):
            raise ValueError("Pfaffian tolerances must be finite.")
        self.maximum_dimension = maximum
        self.antisymmetry_tolerance = antisymmetry
        self.determinant_tolerance = determinant
        self.minimum_magnitude = minimum
        self.maximum_phase_magnitude = phase
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-pfaffian-control-plan",
                "maximum_dimension": maximum,
                "antisymmetry_tolerance": antisymmetry,
                "determinant_tolerance": determinant,
                "minimum_magnitude": minimum,
                "maximum_phase_magnitude": phase,
                "algorithm": "recursive-perfect-matching-reference",
            }
        )


class PfaffianEvidence(StrictModule):
    pfaffian: Array
    phase: Array
    magnitude: Array
    antisymmetry_residual: Array
    determinant_identity_residual: Array
    antisymmetric: Array
    determinant_identity: Array
    nonzero: Array
    phase_controlled: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _pfaffian_recursive(matrix: Array, /) -> Array:
    dimension = matrix.shape[0]
    if dimension == 0:
        return jnp.asarray(1.0, dtype=matrix.dtype)
    value = jnp.asarray(0.0, dtype=matrix.dtype)
    for column in range(1, dimension):
        keep = tuple(index for index in range(1, dimension) if index != column)
        indices = jnp.asarray(keep, dtype=jnp.int32)
        minor = matrix[indices[:, None], indices[None, :]]
        sign = 1.0 if column % 2 else -1.0
        value = value + sign * matrix[0, column] * _pfaffian_recursive(minor)
    return value


def finite_pfaffian(matrix: ArrayLike, plan: PfaffianControlPlan, /) -> Array:
    """Compute a guarded exact-matching Pfaffian for a small dense matrix."""
    if not isinstance(plan, PfaffianControlPlan):
        raise TypeError("plan must be PfaffianControlPlan.")
    value = jnp.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Pfaffian input must be one square matrix.")
    dimension = int(value.shape[0])
    if dimension % 2:
        raise ValueError("Pfaffian input dimension must be even.")
    if dimension > plan.maximum_dimension:
        raise ValueError(
            f"Pfaffian dimension {dimension} exceeds capacity {plan.maximum_dimension}."
        )
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(float)
    return _pfaffian_recursive(value)


def pfaffian_evidence(
    matrix: ArrayLike, plan: PfaffianControlPlan, /
) -> PfaffianEvidence:
    if not isinstance(plan, PfaffianControlPlan):
        raise TypeError("plan must be PfaffianControlPlan.")
    value = jnp.asarray(matrix)
    pfaffian = finite_pfaffian(value, plan)
    matrix_scale = jnp.maximum(1.0, jnp.max(jnp.abs(value)))
    antisymmetry_residual = jnp.max(jnp.abs(value + jnp.swapaxes(value, -1, -2)))
    antisymmetry_residual = antisymmetry_residual / matrix_scale
    determinant = jnp.linalg.det(value)
    determinant_scale = jnp.maximum(1.0, jnp.abs(determinant))
    identity_residual = jnp.abs(pfaffian * pfaffian - determinant) / determinant_scale
    magnitude = jnp.abs(pfaffian)
    phase = jnp.angle(pfaffian)
    antisymmetric = antisymmetry_residual <= plan.antisymmetry_tolerance
    determinant_identity = identity_residual <= plan.determinant_tolerance
    nonzero = magnitude >= plan.minimum_magnitude
    phase_controlled = jnp.abs(phase) <= plan.maximum_phase_magnitude
    finite = (
        jnp.isfinite(antisymmetry_residual)
        & jnp.isfinite(identity_residual)
        & jnp.isfinite(magnitude)
        & jnp.isfinite(phase)
    )
    return PfaffianEvidence(
        pfaffian=pfaffian,
        phase=phase,
        magnitude=magnitude,
        antisymmetry_residual=antisymmetry_residual,
        determinant_identity_residual=identity_residual,
        antisymmetric=antisymmetric,
        determinant_identity=determinant_identity,
        nonzero=nonzero,
        phase_controlled=phase_controlled,
        accepted=finite
        & antisymmetric
        & determinant_identity
        & nonzero
        & phase_controlled,
        plan_id=plan.plan_id,
        claim="finite-matrix-reference-only",
    )


class WardIdentityPlan(StrictModule):
    expected_bosonic_action: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    standard_error_multiplier: float = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        expected_bosonic_action: float,
        /,
        *,
        absolute_tolerance: float,
        standard_error_multiplier: float = 3.0,
        maximum_samples: int = 4096,
    ):
        expected = float(expected_bosonic_action)
        absolute = float(absolute_tolerance)
        multiplier = float(standard_error_multiplier)
        maximum = int(maximum_samples)
        if absolute < 0.0 or multiplier < 0.0 or maximum < 1:
            raise ValueError("Ward tolerances must be nonnegative and capacity positive.")
        if not all(np.isfinite(value) for value in (expected, absolute, multiplier)):
            raise ValueError("Ward plan scalars must be finite.")
        self.expected_bosonic_action = expected
        self.absolute_tolerance = absolute
        self.standard_error_multiplier = multiplier
        self.maximum_samples = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-ward-identity-plan",
                "expected_bosonic_action": expected,
                "absolute_tolerance": absolute,
                "standard_error_multiplier": multiplier,
                "maximum_samples": maximum,
            }
        )


class GaugeInvarianceEvidence(StrictModule):
    original_action: Array
    transformed_action: Array
    absolute_residual: Array
    relative_residual: Array
    tolerance: Array
    invariant: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def assess_complexified_gauge_invariance(
    action: PreparedTwistedSYMAction | PreparedBFSSAction,
    configuration: TwistedSYMConfiguration | BFSSConfiguration,
    gauge: ArrayLike,
    /,
    *,
    inverse_gauge: ArrayLike | None = None,
    tolerance: float = 1e-9,
) -> GaugeInvarianceEvidence:
    """Compare an action before and after one explicit finite GL(N,C) transform."""
    if not isinstance(action, (PreparedTwistedSYMAction, PreparedBFSSAction)):
        raise TypeError("action must be a prepared twisted-SYM or BFSS action.")
    tolerance_ = float(tolerance)
    if tolerance_ < 0.0 or not np.isfinite(tolerance_):
        raise ValueError("tolerance must be finite and nonnegative.")
    if isinstance(action, PreparedTwistedSYMAction) and isinstance(
        configuration, TwistedSYMConfiguration
    ):
        transformed = transform_twisted_configuration(
            configuration, gauge, inverse_gauge=inverse_gauge
        )
    elif isinstance(action, PreparedBFSSAction) and isinstance(
        configuration, BFSSConfiguration
    ):
        transformed = transform_bfss_configuration(
            configuration, gauge, inverse_gauge=inverse_gauge
        )
    else:
        raise TypeError("Action and configuration kinds must match.")
    original_action = action.action(configuration)
    transformed_action = action.action(transformed)
    absolute = jnp.abs(transformed_action - original_action)
    relative = absolute / jnp.maximum(1.0, jnp.abs(original_action))
    finite = jnp.isfinite(original_action) & jnp.isfinite(transformed_action)
    return GaugeInvarianceEvidence(
        original_action=original_action,
        transformed_action=transformed_action,
        absolute_residual=absolute,
        relative_residual=relative,
        tolerance=jnp.asarray(tolerance_),
        invariant=finite & (relative <= tolerance_),
        prepared_id=action.prepared_id,
        claim="finite-complexified-gauge-orbit-reference-only",
    )


class WardPfaffianEvidence(StrictModule):
    bosonic_actions: Array
    bosonic_mean: Array
    bosonic_standard_error: Array
    ward_residual: Array
    ward_threshold: Array
    ward_satisfied: Array
    pfaffians: tuple[PfaffianEvidence, ...]
    pfaffians_accepted: Array
    accepted: Array
    ward_plan_id: str = eqx.field(static=True)
    pfaffian_plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def assess_ward_pfaffian(
    action: PreparedTwistedSYMAction | PreparedBFSSAction,
    configurations: Sequence[object],
    fermion_operators: Sequence[ArrayLike],
    ward_plan: WardIdentityPlan,
    pfaffian_plan: PfaffianControlPlan,
    /,
) -> WardPfaffianEvidence:
    """Evaluate finite-sample Ward and dense-Pfaffian controls without extrapolation."""
    if not isinstance(action, (PreparedTwistedSYMAction, PreparedBFSSAction)):
        raise TypeError("action must be a prepared twisted-SYM or BFSS action.")
    if not isinstance(ward_plan, WardIdentityPlan):
        raise TypeError("ward_plan must be WardIdentityPlan.")
    if not isinstance(pfaffian_plan, PfaffianControlPlan):
        raise TypeError("pfaffian_plan must be PfaffianControlPlan.")
    configurations_ = tuple(configurations)
    operators = tuple(fermion_operators)
    if not configurations_ or len(configurations_) != len(operators):
        raise ValueError("Configurations and fermion operators must be equally nonempty.")
    if len(configurations_) > ward_plan.maximum_samples:
        raise ValueError("Ward sample count exceeds maximum_samples.")
    actions = jnp.stack(tuple(action.action(value) for value in configurations_))
    mean = jnp.mean(actions)
    standard_error = jnp.where(
        actions.size > 1,
        jnp.std(actions, ddof=1) / jnp.sqrt(actions.size),
        jnp.asarray(0.0, dtype=mean.dtype),
    )
    residual = jnp.abs(mean - ward_plan.expected_bosonic_action)
    threshold = ward_plan.absolute_tolerance + (
        ward_plan.standard_error_multiplier * standard_error
    )
    pfaffians = tuple(pfaffian_evidence(value, pfaffian_plan) for value in operators)
    pfaffians_accepted = jnp.all(jnp.stack(tuple(value.accepted for value in pfaffians)))
    ward_satisfied = jnp.isfinite(residual) & (residual <= threshold)
    return WardPfaffianEvidence(
        bosonic_actions=actions,
        bosonic_mean=mean,
        bosonic_standard_error=standard_error,
        ward_residual=residual,
        ward_threshold=threshold,
        ward_satisfied=ward_satisfied,
        pfaffians=pfaffians,
        pfaffians_accepted=pfaffians_accepted,
        accepted=ward_satisfied & pfaffians_accepted,
        ward_plan_id=ward_plan.plan_id,
        pfaffian_plan_id=pfaffian_plan.plan_id,
        claim="finite-sample-research-only-no-continuum-or-large-n-claim",
    )


__all__ = [
    "GaugeInvarianceEvidence",
    "PfaffianControlPlan",
    "PfaffianEvidence",
    "WardIdentityPlan",
    "WardPfaffianEvidence",
    "assess_complexified_gauge_invariance",
    "assess_ward_pfaffian",
    "finite_pfaffian",
    "pfaffian_evidence",
]
