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
from ._fermions import materialize_twisted_fermion_reference, TwistedKahlerDiracOperator
from ._rhmc import (
    PreparedTwistedN2RHMC,
    TwistedN2RHMCEvidence,
    TwistedN2RHMCRun,
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
    if isinstance(action, PreparedTwistedSYMAction):
        if not isinstance(configuration, TwistedSYMConfiguration):
            raise TypeError("Twisted-SYM actions require TwistedSYMConfiguration.")
        transformed = transform_twisted_configuration(
            configuration, gauge, inverse_gauge=inverse_gauge
        )
        original_action = action.action(configuration)
        transformed_action = action.action(transformed)
    elif isinstance(action, PreparedBFSSAction):
        if not isinstance(configuration, BFSSConfiguration):
            raise TypeError("BFSS actions require BFSSConfiguration.")
        transformed = transform_bfss_configuration(
            configuration, gauge, inverse_gauge=inverse_gauge
        )
        original_action = action.action(configuration)
        transformed_action = action.action(transformed)
    else:
        raise TypeError("Action and configuration kinds must match.")
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
    if isinstance(action, PreparedTwistedSYMAction):
        if any(
            not isinstance(value, TwistedSYMConfiguration) for value in configurations_
        ):
            raise TypeError(
                "Twisted-SYM Ward samples require TwistedSYMConfiguration values."
            )
        actions = jnp.stack(
            tuple(
                action.action(value)
                for value in configurations_
                if isinstance(value, TwistedSYMConfiguration)
            )
        )
    else:
        if any(not isinstance(value, BFSSConfiguration) for value in configurations_):
            raise TypeError("BFSS Ward samples require BFSSConfiguration values.")
        actions = jnp.stack(
            tuple(
                action.action(value)
                for value in configurations_
                if isinstance(value, BFSSConfiguration)
            )
        )
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


class TwistedN2ChainEvidence(StrictModule):
    """Finite-chain Ward, Pfaffian-phase, and reweighting evidence."""

    rhmc: TwistedN2RHMCEvidence
    ward_pfaffian: WardPfaffianEvidence
    pfaffian_phases: Array
    average_phase: Array
    phase_effective_samples: Array
    minimum_phase_effective_samples: Array
    phase_overlap_sufficient: Array
    accepted: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def assess_twisted_n2_chain(
    prepared: PreparedTwistedN2RHMC,
    run: TwistedN2RHMCRun,
    ward_plan: WardIdentityPlan,
    pfaffian_plan: PfaffianControlPlan,
    /,
    *,
    minimum_phase_effective_samples: float,
    maximum_dense_elements: int,
) -> TwistedN2ChainEvidence:
    """Audit every retained finite configuration without continuum promotion."""
    if not isinstance(prepared, PreparedTwistedN2RHMC):
        raise TypeError("prepared must be PreparedTwistedN2RHMC.")
    if not isinstance(run, TwistedN2RHMCRun):
        raise TypeError("run must be TwistedN2RHMCRun.")
    if run.prepared_id != prepared.prepared_id:
        raise ValueError("Run and prepared twisted-SYM workflow identities differ.")
    minimum = float(minimum_phase_effective_samples)
    maximum = int(maximum_dense_elements)
    if not np.isfinite(minimum) or minimum < 0.0 or maximum < 1:
        raise ValueError("Phase-overlap and dense-reference bounds are invalid.")
    configurations = tuple(
        prepared.layout.unpack(run.samples.configurations[index])
        for index in range(run.samples.num_draws)
    )
    matrices = tuple(
        materialize_twisted_fermion_reference(
            TwistedKahlerDiracOperator(
                prepared.theory,
                prepared.layout,
                run.samples.configurations[index],
            ),
            maximum_elements=maximum,
        )
        for index in range(run.samples.num_draws)
    )
    ward_pfaffian = assess_ward_pfaffian(
        prepared.bosonic_action,
        configurations,
        matrices,
        ward_plan,
        pfaffian_plan,
    )
    phases = jnp.stack(tuple(value.phase for value in ward_pfaffian.pfaffians))
    weights = jnp.exp(1.0j * phases)
    average = jnp.mean(weights)
    effective = jnp.abs(jnp.sum(weights)) ** 2 / jnp.sum(jnp.abs(weights) ** 2)
    overlap = (
        jnp.isfinite(effective)
        & (effective >= minimum)
        & jnp.isfinite(jnp.real(average))
        & jnp.isfinite(jnp.imag(average))
    )
    accepted = run.evidence.successful & ward_pfaffian.accepted & overlap
    return TwistedN2ChainEvidence(
        rhmc=run.evidence,
        ward_pfaffian=ward_pfaffian,
        pfaffian_phases=phases,
        average_phase=average,
        phase_effective_samples=effective,
        minimum_phase_effective_samples=jnp.asarray(minimum),
        phase_overlap_sufficient=overlap,
        accepted=accepted,
        prepared_id=prepared.prepared_id,
        claim="finite-regulated-chain-phase-reweighting-evidence-no-continuum-supersymmetry-claim",
    )


__all__ = [
    "GaugeInvarianceEvidence",
    "PfaffianControlPlan",
    "PfaffianEvidence",
    "WardIdentityPlan",
    "WardPfaffianEvidence",
    "TwistedN2ChainEvidence",
    "assess_complexified_gauge_invariance",
    "assess_ward_pfaffian",
    "assess_twisted_n2_chain",
    "finite_pfaffian",
    "pfaffian_evidence",
]
