#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Near-boundary scalar fits and declared holographic stress-tensor evidence."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._ads_scalar import ConformalAdSScalarPlan, ConformalAdSScalarState


class AdSBoundaryScalarObservablePlan(StrictModule):
    delta_minus: float = eqx.field(static=True)
    delta_plus: float = eqx.field(static=True)
    fit_points: int = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    normalization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        delta_minus: float,
        delta_plus: float,
        /,
        *,
        fit_points: int,
        maximum_condition_number: float = 1e12,
        residual_tolerance: float = 1e-4,
        normalization: float = 1.0,
    ):
        lower = float(delta_minus)
        upper = float(delta_plus)
        count = int(fit_points)
        condition = float(maximum_condition_number)
        tolerance = float(residual_tolerance)
        normalization_value = float(normalization)
        if (
            not all(
                np.isfinite(value)
                for value in (lower, upper, condition, tolerance, normalization_value)
            )
            or lower < 0.0
            or upper <= lower
            or count < 2
            or condition <= 1.0
            or tolerance < 0.0
            or normalization_value == 0.0
        ):
            raise ValueError("AdS scalar boundary-observable parameters are invalid.")
        self.delta_minus = lower
        self.delta_plus = upper
        self.fit_points = count
        self.maximum_condition_number = condition
        self.residual_tolerance = tolerance
        self.normalization = normalization_value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ads-boundary-scalar-observable-plan",
                "delta_minus": lower,
                "delta_plus": upper,
                "fit_points": count,
                "maximum_condition_number": condition,
                "residual_tolerance": tolerance,
                "normalization": normalization_value,
            }
        )


class AdSBoundaryScalarEvidence(StrictModule):
    source_coefficient: Array
    response_coefficient: Array
    normalized_one_point_function: Array
    fit_residual: Array
    condition_number: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    scalar_plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def extract_ads_boundary_scalar(
    observable: AdSBoundaryScalarObservablePlan,
    scalar_plan: ConformalAdSScalarPlan,
    state: ConformalAdSScalarState,
    /,
) -> AdSBoundaryScalarEvidence:
    if not isinstance(observable, AdSBoundaryScalarObservablePlan):
        raise TypeError("observable must be AdSBoundaryScalarObservablePlan.")
    if not isinstance(scalar_plan, ConformalAdSScalarPlan):
        raise TypeError("scalar_plan must be ConformalAdSScalarPlan.")
    if (
        not isinstance(state, ConformalAdSScalarState)
        or state.plan_id != scalar_plan.plan_id
    ):
        raise TypeError("state must match scalar_plan.")
    if observable.fit_points >= scalar_plan.point_count:
        raise ValueError(
            "fit_points must leave the exact boundary point outside the fit."
        )
    radial = scalar_plan.radial_points[-observable.fit_points - 1 : -1]
    values = state.field[-observable.fit_points - 1 : -1]
    defining = jnp.cos(radial)
    design = jnp.stack(
        (
            defining**observable.delta_minus,
            defining**observable.delta_plus,
        ),
        axis=1,
    )
    gram = design.T @ design
    rhs = design.T @ values
    determinant = gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0]
    determinant = eqx.error_if(
        determinant,
        jnp.abs(determinant) <= jnp.finfo(gram.dtype).eps,
        "Near-boundary scalar fit is rank deficient.",
    )
    coefficients = jnp.stack(
        (
            (gram[1, 1] * rhs[0] - gram[0, 1] * rhs[1]) / determinant,
            (-gram[1, 0] * rhs[0] + gram[0, 0] * rhs[1]) / determinant,
        )
    )
    trace = gram[0, 0] + gram[1, 1]
    discriminant = jnp.sqrt(jnp.maximum(0.0, trace**2 - 4.0 * determinant))
    largest = 0.5 * (trace + discriminant)
    smallest = 0.5 * (trace - discriminant)
    condition = largest / smallest
    reconstructed = design @ coefficients
    residual = jnp.linalg.norm(reconstructed - values) / jnp.maximum(
        1.0, jnp.linalg.norm(values)
    )
    finite = (
        jnp.all(jnp.isfinite(coefficients))
        & jnp.isfinite(residual)
        & jnp.isfinite(condition)
    )
    accepted = (
        finite
        & (condition <= observable.maximum_condition_number)
        & (residual <= observable.residual_tolerance)
    )
    return AdSBoundaryScalarEvidence(
        source_coefficient=coefficients[0],
        response_coefficient=coefficients[1],
        normalized_one_point_function=observable.normalization * coefficients[1],
        fit_residual=residual,
        condition_number=condition,
        finite=finite,
        accepted=accepted,
        plan_id=observable.plan_id,
        scalar_plan_id=scalar_plan.plan_id,
        claim="finite-near-boundary-source-response-fit-requires-declared-renormalization",
    )


class HolographicStressTensorPlan(StrictModule):
    boundary_dimension: int = eqx.field(static=True)
    normalization: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary_dimension: int,
        normalization: float,
        source_id: str,
        /,
        *,
        tolerance: float = 1e-8,
    ):
        dimension = int(boundary_dimension)
        normalization_value = float(normalization)
        source = str(source_id)
        tolerance_value = float(tolerance)
        if (
            dimension < 2
            or not np.isfinite(normalization_value)
            or normalization_value == 0.0
            or not source
            or not np.isfinite(tolerance_value)
            or tolerance_value < 0.0
        ):
            raise ValueError("Holographic stress-tensor plan parameters are invalid.")
        self.boundary_dimension = dimension
        self.normalization = normalization_value
        self.source_id = source
        self.tolerance = tolerance_value
        self.plan_id = canonical_fingerprint(
            {
                "kind": "holographic-stress-tensor-plan",
                "boundary_dimension": dimension,
                "normalization": normalization_value,
                "source_id": source,
                "tolerance": tolerance_value,
            }
        )


class HolographicStressTensorEvidence(StrictModule):
    stress_tensor: Array
    trace: Array
    divergence: Array
    trace_residual: Array
    divergence_residual: Array
    symmetry_residual: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def evaluate_holographic_stress_tensor(
    plan: HolographicStressTensorPlan,
    fefferman_graham_coefficient: ArrayLike,
    counterterm_tensor: ArrayLike,
    inverse_boundary_metric: ArrayLike,
    divergence: ArrayLike,
    /,
    *,
    expected_trace: ArrayLike = 0.0,
) -> HolographicStressTensorEvidence:
    if not isinstance(plan, HolographicStressTensorPlan):
        raise TypeError("plan must be HolographicStressTensorPlan.")
    coefficient = jnp.asarray(fefferman_graham_coefficient)
    counterterm = jnp.asarray(counterterm_tensor, dtype=coefficient.dtype)
    inverse = jnp.asarray(inverse_boundary_metric, dtype=coefficient.dtype)
    divergence_value = jnp.asarray(divergence, dtype=coefficient.dtype)
    dimension = plan.boundary_dimension
    if coefficient.ndim < 2 or coefficient.shape[:2] != (dimension, dimension):
        raise ValueError("Fefferman-Graham coefficient has wrong tensor axes.")
    if counterterm.shape != coefficient.shape or inverse.shape != coefficient.shape:
        raise ValueError("Boundary metric/coefficient/counterterm shapes must match.")
    if divergence_value.shape != (dimension,) + coefficient.shape[2:]:
        raise ValueError("Stress divergence has the wrong shape.")
    stress = plan.normalization * (coefficient + counterterm)
    trace = ein.contract("ab...,ab...->...", inverse, stress)
    expected = jnp.asarray(expected_trace, dtype=trace.dtype)
    if expected.shape not in ((), trace.shape):
        raise ValueError("expected_trace must be scalar or match the boundary grid.")
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(stress)))
    trace_residual = jnp.max(jnp.abs(trace - expected)) / scale
    divergence_residual = jnp.max(jnp.abs(divergence_value)) / scale
    symmetry = jnp.max(jnp.abs(stress - jnp.swapaxes(stress, 0, 1))) / scale
    finite = (
        jnp.all(jnp.isfinite(stress))
        & jnp.all(jnp.isfinite(trace))
        & jnp.all(jnp.isfinite(divergence_value))
    )
    accepted = (
        finite
        & (trace_residual <= plan.tolerance)
        & (divergence_residual <= plan.tolerance)
        & (symmetry <= plan.tolerance)
    )
    return HolographicStressTensorEvidence(
        stress_tensor=stress,
        trace=trace,
        divergence=divergence_value,
        trace_residual=trace_residual,
        divergence_residual=divergence_residual,
        symmetry_residual=symmetry,
        finite=finite,
        accepted=accepted,
        plan_id=plan.plan_id,
        claim="declared-fefferman-graham-coefficient-and-counterterm-evidence-not-automatic-renormalization",
    )


__all__ = [
    "AdSBoundaryScalarEvidence",
    "AdSBoundaryScalarObservablePlan",
    "HolographicStressTensorEvidence",
    "HolographicStressTensorPlan",
    "evaluate_holographic_stress_tensor",
    "extract_ads_boundary_scalar",
]
