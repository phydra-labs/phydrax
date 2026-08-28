#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._dfsph import DFSPHFactorState, PreparedDFSPH
from ._iisph import PreparedIISPH
from ._qualification import ParticleConstraintResiduals


class IISPHAssembledOracle(StrictModule):
    matrix: Array
    action_error: Array
    successful: Array


def assemble_iisph_operator(
    dynamics: PreparedIISPH,
    position: ArrayLike,
    step_size: ArrayLike,
    /,
) -> IISPHAssembledOracle:
    position_ = jnp.asarray(position)
    count = dynamics.particles.capacity
    basis = jnp.eye(count, dtype=position_.dtype)
    columns = jax.vmap(
        lambda pressure: dynamics.pressure_action(
            position_, pressure, jnp.asarray(step_size)
        )
    )(basis)
    matrix = columns.T
    probe = jnp.linspace(0.1, 1.0, count, dtype=position_.dtype)
    direct = dynamics.pressure_action(position_, probe, jnp.asarray(step_size))
    assembled = matrix @ probe
    error = jnp.max(jnp.abs(direct - assembled))
    return IISPHAssembledOracle(matrix, error, jnp.isfinite(error))


class IISPHOperatorDiagnostics(StrictModule):
    symmetry_defect: Array
    diagonal_minimum: Array
    row_sum_defect: Array
    minimum_quadratic_form: Array
    finite: Array


def diagnose_iisph_operator(oracle: IISPHAssembledOracle, /) -> IISPHOperatorDiagnostics:
    matrix = oracle.matrix
    scale = jnp.maximum(jnp.max(jnp.abs(matrix)), jnp.finfo(matrix.dtype).tiny)
    symmetry = jnp.max(jnp.abs(matrix - matrix.T)) / scale
    diagonal = jnp.min(jnp.diag(matrix))
    row_sum = jnp.max(jnp.abs(jnp.sum(matrix, axis=1))) / scale
    probes = jnp.stack(
        (
            jnp.ones((matrix.shape[0],), matrix.dtype),
            jnp.linspace(-1.0, 1.0, matrix.shape[0], dtype=matrix.dtype),
        )
    )
    quadratic = jax.vmap(lambda value: value @ matrix @ value)(probes)
    return IISPHOperatorDiagnostics(
        symmetry,
        diagonal,
        row_sum,
        jnp.min(quadratic),
        jnp.all(jnp.isfinite(matrix)),
    )


def pressure_complementarity_residual(
    pressure: ArrayLike,
    constraint: ArrayLike,
    /,
    *,
    atmospheric_pressure: ArrayLike = 0.0,
    active_mask: ArrayLike | None = None,
) -> Array:
    pressure_ = jnp.asarray(pressure)
    constraint_ = jnp.asarray(constraint)
    gap = pressure_ - jnp.asarray(atmospheric_pressure, pressure_.dtype)
    mask = (
        jnp.ones(gap.shape, bool)
        if active_mask is None
        else jnp.asarray(active_mask, bool)
    )
    residual = (
        jnp.maximum(-gap, 0.0)
        + jnp.maximum(constraint_, 0.0)
        + jnp.abs(gap * constraint_)
    )
    return jnp.max(jnp.where(mask, residual, 0.0))


def rescale_pressure_warm_start(
    pressure: ArrayLike, old_step_size: ArrayLike, new_step_size: ArrayLike, /
) -> Array:
    old = jnp.asarray(old_step_size)
    new = jnp.asarray(new_step_size)
    return jnp.asarray(pressure) * (old / new) ** 2


class FrozenActiveProjectionDerivative(StrictModule):
    tangent: Array
    residual: Array
    successful: Array


def iisph_frozen_active_tangent(
    oracle: IISPHAssembledOracle,
    right_hand_side_derivative: ArrayLike,
    active_mask: ArrayLike,
    /,
) -> FrozenActiveProjectionDerivative:
    matrix = oracle.matrix
    active = jnp.asarray(active_mask, bool)
    constrained = jnp.where(active[:, None] & active[None, :], matrix, 0.0)
    constrained = constrained + jnp.diag((~active).astype(matrix.dtype))
    rhs = jnp.where(active, -jnp.asarray(right_hand_side_derivative), 0.0)
    result = solve(
        LinearSystem(DenseLinearOperator(constrained)),
        rhs,
        policy=LinearSolvePolicy(DenseLU()),
    )
    residual = constrained @ result.value - rhs
    return FrozenActiveProjectionDerivative(
        result.value,
        jnp.max(jnp.abs(residual)),
        result.successful,
    )


class DFSPHFactorOracle(StrictModule):
    factor: DFSPHFactorState
    finite: Array
    minimum_denominator: Array


def dfsph_factor_oracle(
    dynamics: PreparedDFSPH, position: ArrayLike, /
) -> DFSPHFactorOracle:
    factor = dynamics.factor(jnp.asarray(position))
    return DFSPHFactorOracle(
        factor,
        jnp.all(jnp.isfinite(factor.alpha) & jnp.isfinite(factor.denominator)),
        jnp.min(factor.denominator),
    )


def dfsph_constraints_satisfied(
    constraints: ParticleConstraintResiduals,
    /,
    *,
    density_tolerance: float = 1e-3,
    divergence_tolerance: float = 1e-3,
) -> Array:
    return (
        (constraints.relative_density_linf <= density_tolerance)
        & (constraints.relative_density_l2 <= density_tolerance)
        & (constraints.relative_divergence_linf <= divergence_tolerance)
        & (constraints.relative_divergence_l2 <= divergence_tolerance)
    )


def rescale_dfsph_warm_start(
    density_multiplier: ArrayLike,
    divergence_multiplier: ArrayLike,
    old_step_size: ArrayLike,
    new_step_size: ArrayLike,
    /,
) -> tuple[Array, Array]:
    ratio = jnp.asarray(old_step_size) / jnp.asarray(new_step_size)
    return (
        jnp.asarray(density_multiplier) * ratio**2,
        jnp.asarray(divergence_multiplier) * ratio,
    )


ProjectionAccelerationKind: TypeAlias = Literal["reference", "chebyshev", "anderson"]


class ProjectedIterationAccelerationPlan(StrictModule, NonTrainableState):
    kind: ProjectionAccelerationKind = eqx.field(static=True)
    relaxation_minimum: float = eqx.field(static=True)
    relaxation_maximum: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ProjectionAccelerationKind = "reference",
        /,
        *,
        relaxation_minimum: float = 0.3,
        relaxation_maximum: float = 0.9,
    ):
        if kind not in ("reference", "chebyshev", "anderson"):
            raise ValueError("Unknown projection acceleration kind.")
        if not 0.0 < relaxation_minimum <= relaxation_maximum <= 1.0:
            raise ValueError("Projection relaxation bounds are invalid.")
        self.kind = kind
        self.relaxation_minimum = float(relaxation_minimum)
        self.relaxation_maximum = float(relaxation_maximum)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "projected-iteration-acceleration",
                "method": kind,
                "relaxation_minimum": relaxation_minimum,
                "relaxation_maximum": relaxation_maximum,
            }
        )

    def relaxation(self, iteration: ArrayLike, maximum_iterations: int, /) -> Array:
        fraction = jnp.asarray(iteration) / jnp.maximum(maximum_iterations - 1, 1)
        if self.kind == "reference":
            return jnp.asarray(self.relaxation_minimum)
        if self.kind == "chebyshev":
            phase = jnp.pi * (fraction + 0.5 / maximum_iterations)
            return self.relaxation_minimum + (
                self.relaxation_maximum - self.relaxation_minimum
            ) * 0.5 * (1.0 - jnp.cos(phase))
        return self.relaxation_maximum


class ProductionProjectedSolvePlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    acceleration: ProjectedIterationAccelerationPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 100,
        tolerance: float = 1e-8,
        acceleration: ProjectedIterationAccelerationPlan | None = None,
    ):
        if maximum_iterations <= 0 or tolerance <= 0.0:
            raise ValueError("Projected solve controls are invalid.")
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.acceleration = (
            ProjectedIterationAccelerationPlan() if acceleration is None else acceleration
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "production-projected-solve",
                "maximum_iterations": maximum_iterations,
                "tolerance": tolerance,
                "acceleration": self.acceleration.plan_id,
            }
        )


class ProductionProjectedSolveResult(StrictModule):
    value: Array
    residual: Array
    complementarity: Array
    iterations: Array
    converged: Array
    successful: Array


def solve_projected_pressure(
    plan: ProductionProjectedSolvePlan,
    oracle: IISPHAssembledOracle,
    right_hand_side: ArrayLike,
    initial_pressure: ArrayLike,
    /,
    *,
    atmospheric_pressure: float = 0.0,
) -> ProductionProjectedSolveResult:
    matrix = oracle.matrix
    rhs = jnp.asarray(right_hand_side)
    diagonal = jnp.diag(matrix)
    safe_diagonal = jnp.where(jnp.abs(diagonal) > 1e-14, diagonal, 1.0)

    def body(iteration, carry):
        pressure, _ = carry
        residual = matrix @ pressure - rhs
        relaxation = plan.acceleration.relaxation(iteration, plan.maximum_iterations)
        candidate = jnp.maximum(
            pressure - relaxation * residual / safe_diagonal,
            atmospheric_pressure,
        )
        return candidate, jnp.max(jnp.abs(residual))

    pressure, residual = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        body,
        (jnp.asarray(initial_pressure), jnp.asarray(jnp.inf, matrix.dtype)),
    )
    constraint = matrix @ pressure - rhs
    complementarity = pressure_complementarity_residual(
        pressure, constraint, atmospheric_pressure=atmospheric_pressure
    )
    converged = (residual <= plan.tolerance) & (complementarity <= plan.tolerance)
    return ProductionProjectedSolveResult(
        pressure,
        residual,
        complementarity,
        jnp.asarray(plan.maximum_iterations, jnp.int32),
        converged,
        converged & jnp.all(jnp.isfinite(pressure)),
    )


__all__ = [
    "DFSPHFactorOracle",
    "FrozenActiveProjectionDerivative",
    "IISPHAssembledOracle",
    "IISPHOperatorDiagnostics",
    "ProductionProjectedSolvePlan",
    "ProductionProjectedSolveResult",
    "ProjectedIterationAccelerationPlan",
    "assemble_iisph_operator",
    "dfsph_constraints_satisfied",
    "dfsph_factor_oracle",
    "diagnose_iisph_operator",
    "iisph_frozen_active_tangent",
    "pressure_complementarity_residual",
    "rescale_dfsph_warm_start",
    "rescale_pressure_warm_start",
    "solve_projected_pressure",
]
