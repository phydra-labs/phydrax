#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import (
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._scft import PreparedSCFT, SCFTEvaluation


class IsotropicSCFTCellPlan(StrictModule, NonTrainableState):
    minimum_scale: float = eqx.field(static=True)
    maximum_scale: float = eqx.field(static=True)
    target_log_scale_derivative: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_scale: float,
        maximum_scale: float,
        target_log_scale_derivative: float = 0.0,
        absolute_tolerance: float = 1.0e-8,
        maximum_iterations: int = 32,
    ):
        lower = float(minimum_scale)
        upper = float(maximum_scale)
        target = float(target_log_scale_derivative)
        tolerance = float(absolute_tolerance)
        iterations = int(maximum_iterations)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0.0 < lower < upper
            or not math.isfinite(target)
            or not math.isfinite(tolerance)
            or tolerance < 0.0
            or iterations <= 0
        ):
            raise ValueError("Isotropic SCFT cell controls are invalid.")
        self.minimum_scale = lower
        self.maximum_scale = upper
        self.target_log_scale_derivative = target
        self.absolute_tolerance = tolerance
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isotropic-scft-cell-plan",
                "minimum_scale": lower,
                "maximum_scale": upper,
                "target_log_scale_derivative": target,
                "absolute_tolerance": tolerance,
                "maximum_iterations": iterations,
            }
        )


class VariableCellSCFTResult(StrictModule):
    evaluation: SCFTEvaluation
    nonlinear: NonlinearResult
    cell_scale: Array
    log_scale_derivative: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    cell_plan_id: str = eqx.field(static=True)


def solve_isotropic_cell_scft(
    prepared: PreparedSCFT,
    initial_fields: ArrayLike,
    initial_scale: float,
    plan: IsotropicSCFTCellPlan,
    /,
) -> VariableCellSCFTResult:
    if not isinstance(prepared, PreparedSCFT):
        raise TypeError("prepared must be PreparedSCFT.")
    if not isinstance(plan, IsotropicSCFTCellPlan):
        raise TypeError("plan must be IsotropicSCFTCellPlan.")
    scale = float(initial_scale)
    if not plan.minimum_scale <= scale <= plan.maximum_scale:
        raise ValueError("initial_scale lies outside the cell plan bounds.")
    fields = prepared.project_initial_fields(initial_fields)
    initial = (fields, jnp.asarray(math.log(scale), dtype=fields.dtype))

    def residual(state, _):
        current_fields, log_scale = state
        current_scale = jnp.exp(log_scale)
        evaluation = prepared.evaluate(current_fields, cell_scale=current_scale)

        def free_energy_at(value):
            return prepared.evaluate(
                current_fields, cell_scale=jnp.exp(value)
            ).free_energy

        derivative = jax.grad(free_energy_at)(log_scale)
        in_bounds = (current_scale >= plan.minimum_scale) & (
            current_scale <= plan.maximum_scale
        )
        successful = evaluation.successful & jnp.isfinite(derivative) & in_bounds
        return (
            jnp.where(successful, evaluation.residual, jnp.nan),
            jnp.where(
                successful,
                derivative - plan.target_log_scale_derivative,
                jnp.nan,
            ),
        )

    nonlinear = NewtonKrylov().solve(
        NonlinearSystemProblem(
            residual, problem_id=f"{prepared.prepared_id}:isotropic-cell-root"
        ),
        initial,
        termination=NonlinearTermination(
            absolute_residual=plan.absolute_tolerance,
            relative_residual=prepared.plan.relative_tolerance,
            maximum_steps=plan.maximum_iterations,
        ),
    )
    solved_fields, solved_log_scale = nonlinear.state
    solved_scale = jnp.exp(solved_log_scale)
    evaluation = prepared.evaluate(solved_fields, cell_scale=solved_scale)
    derivative = jax.grad(
        lambda value: (
            prepared.evaluate(solved_fields, cell_scale=jnp.exp(value)).free_energy
        )
    )(solved_log_scale)
    successful = (
        nonlinear.successful
        & evaluation.successful
        & (solved_scale >= plan.minimum_scale)
        & (solved_scale <= plan.maximum_scale)
        & (
            jnp.abs(derivative - plan.target_log_scale_derivative)
            <= plan.absolute_tolerance
        )
    )
    return VariableCellSCFTResult(
        evaluation,
        nonlinear,
        solved_scale,
        derivative,
        successful,
        prepared.prepared_id,
        plan.plan_id,
    )


__all__ = [
    "IsotropicSCFTCellPlan",
    "VariableCellSCFTResult",
    "solve_isotropic_cell_scft",
]
