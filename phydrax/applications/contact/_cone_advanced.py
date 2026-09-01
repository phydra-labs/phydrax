#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cone import (
    ContactConeEvidence,
    ContactConeProgram,
    ContactConeResult,
    project_coulomb_cone,
)


class SAPContactSolverPlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    acceleration: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 300,
        tolerance: float = 1.0e-10,
        acceleration: bool = True,
    ):
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        if iterations <= 0 or tolerance_ <= 0.0:
            raise ValueError("SAP solver controls are invalid.")
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.acceleration = bool(acceleration)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sap-contact-solver-plan",
                "maximum_iterations": iterations,
                "tolerance": tolerance_.hex(),
                "acceleration": bool(acceleration),
            }
        )


class SemismoothContactSolverPlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    maximum_linear_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 50,
        maximum_linear_iterations: int = 200,
        tolerance: float = 1.0e-10,
        regularization: float = 1.0e-10,
    ):
        nonlinear = int(maximum_iterations)
        linear = int(maximum_linear_iterations)
        tolerance_ = float(tolerance)
        regularization_ = float(regularization)
        if nonlinear <= 0 or linear <= 0 or tolerance_ <= 0.0 or regularization_ <= 0.0:
            raise ValueError("Semismooth contact solver controls are invalid.")
        self.maximum_iterations = nonlinear
        self.maximum_linear_iterations = linear
        self.tolerance = tolerance_
        self.regularization = regularization_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "semismooth-contact-solver-plan",
                "maximum_iterations": nonlinear,
                "maximum_linear_iterations": linear,
                "tolerance": tolerance_.hex(),
                "regularization": regularization_.hex(),
            }
        )


class PrimalDualContactSolverPlan(StrictModule, NonTrainableState):
    outer_iterations: int = eqx.field(static=True)
    inner_iterations: int = eqx.field(static=True)
    maximum_linear_iterations: int = eqx.field(static=True)
    initial_barrier: float = eqx.field(static=True)
    barrier_reduction: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        outer_iterations: int = 8,
        inner_iterations: int = 20,
        maximum_linear_iterations: int = 200,
        initial_barrier: float = 1.0e-2,
        barrier_reduction: float = 0.2,
        tolerance: float = 1.0e-10,
        regularization: float = 1.0e-10,
    ):
        outer = int(outer_iterations)
        inner = int(inner_iterations)
        linear = int(maximum_linear_iterations)
        barrier = float(initial_barrier)
        reduction = float(barrier_reduction)
        tolerance_ = float(tolerance)
        regularization_ = float(regularization)
        if (
            outer <= 0
            or inner <= 0
            or linear <= 0
            or barrier <= 0.0
            or not 0.0 < reduction < 1.0
            or tolerance_ <= 0.0
            or regularization_ <= 0.0
        ):
            raise ValueError("Primal-dual contact solver controls are invalid.")
        self.outer_iterations = outer
        self.inner_iterations = inner
        self.maximum_linear_iterations = linear
        self.initial_barrier = barrier
        self.barrier_reduction = reduction
        self.tolerance = tolerance_
        self.regularization = regularization_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "primal-dual-contact-solver-plan",
                "outer_iterations": outer,
                "inner_iterations": inner,
                "maximum_linear_iterations": linear,
                "initial_barrier": barrier.hex(),
                "barrier_reduction": reduction.hex(),
                "tolerance": tolerance_.hex(),
                "regularization": regularization_.hex(),
            }
        )


def _matrix(program):
    return program.effective_mass + jnp.diag(program.compliance)


def _natural_residual(program, impulse):
    gradient = (
        _matrix(program) @ impulse.reshape((-1,)) + program.free_velocity.reshape((-1,))
    ).reshape(impulse.shape)
    projected = project_coulomb_cone(impulse - gradient, program.friction)
    projected = jnp.where(program.valid[:, None], projected, 0.0)
    return impulse - projected


def _cg_normal(matrix, right, maximum_iterations, tolerance):
    value = jnp.zeros_like(right)
    residual = right - matrix(value)
    direction = residual
    residual_squared = jnp.sum(residual * residual)

    def body(_, state):
        value_, residual_, direction_, squared_, converged_ = state
        action = matrix(direction_)
        denominator = jnp.sum(direction_ * action)
        alpha = squared_ / jnp.maximum(denominator, jnp.finfo(right.dtype).eps)
        candidate_value = value_ + alpha * direction_
        candidate_residual = residual_ - alpha * action
        candidate_squared = jnp.sum(candidate_residual * candidate_residual)
        beta = candidate_squared / jnp.maximum(squared_, jnp.finfo(right.dtype).eps)
        candidate_direction = candidate_residual + beta * direction_
        now = jnp.sqrt(candidate_squared) <= tolerance
        return (
            jnp.where(converged_, value_, candidate_value),
            jnp.where(converged_, residual_, candidate_residual),
            jnp.where(converged_, direction_, candidate_direction),
            jnp.where(converged_, squared_, candidate_squared),
            converged_ | now,
        )

    value, _, _, _, _ = jax.lax.fori_loop(
        0,
        maximum_iterations,
        body,
        (
            value,
            residual,
            direction,
            residual_squared,
            jnp.asarray(False),
        ),
    )
    return value


def _evidence(program, impulse, residual_norm, iterations, tolerance, plan_id):
    post = (
        program.effective_mass @ impulse.reshape((-1,))
        + program.free_velocity.reshape((-1,))
    ).reshape(impulse.shape)
    normal = impulse[:, 0]
    tangent = impulse[:, 1:]
    tangent_norm = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1))
    cone_defect = jnp.max(
        jnp.where(
            program.valid,
            jnp.maximum(tangent_norm - program.friction * normal, 0.0),
            0.0,
        ),
        initial=0.0,
    )
    complementarity = jnp.max(
        jnp.where(
            program.valid,
            jnp.abs(jnp.minimum(normal, post[:, 0])),
            0.0,
        ),
        initial=0.0,
    )
    minimum_normal = jnp.min(jnp.where(program.valid, normal, jnp.inf), initial=0.0)
    dissipated = -jnp.sum(tangent * program.free_velocity[:, 1:])
    finite = jnp.all(jnp.isfinite(impulse)) & jnp.all(jnp.isfinite(post))
    tolerance_ = jnp.asarray(tolerance, dtype=residual_norm.dtype)
    converged = residual_norm <= tolerance_
    successful = (
        converged & finite & (minimum_normal >= 0.0) & (cone_defect <= tolerance_)
    )
    evidence = ContactConeEvidence(
        converged,
        jnp.asarray(iterations, dtype=jnp.int32),
        residual_norm,
        complementarity,
        cone_defect,
        minimum_normal,
        dissipated,
        finite,
        successful,
        program.program_id,
        plan_id,
    )
    return ContactConeResult(impulse, post, evidence)


def solve_contact_sap(
    program: ContactConeProgram,
    /,
    *,
    solver: SAPContactSolverPlan | None = None,
    initial_impulse=None,
) -> ContactConeResult:
    solver_ = SAPContactSolverPlan() if solver is None else solver
    count = program.contact_count
    dimension = program.local_dimension
    impulse = (
        jnp.zeros((count, dimension), dtype=program.free_velocity.dtype)
        if initial_impulse is None
        else jnp.asarray(initial_impulse, dtype=program.free_velocity.dtype)
    )
    matrix = _matrix(program)
    lipschitz = jnp.max(jnp.sum(jnp.abs(matrix), axis=1), initial=1.0)
    step = 1.0 / jnp.maximum(lipschitz, jnp.finfo(matrix.dtype).eps)
    previous = impulse
    momentum = jnp.asarray(1.0, dtype=matrix.dtype)

    def body(_, state):
        value, previous_, momentum_ = state
        extrapolated = value + jnp.where(
            solver_.acceleration,
            ((momentum_ - 1.0) / (momentum_ + 2.0)) * (value - previous_),
            0.0,
        )
        gradient = (
            matrix @ extrapolated.reshape((-1,)) + program.free_velocity.reshape((-1,))
        ).reshape(value.shape)
        candidate = project_coulomb_cone(extrapolated - step * gradient, program.friction)
        candidate = jnp.where(program.valid[:, None], candidate, 0.0)
        return candidate, value, momentum_ + 1.0

    impulse, _, _ = jax.lax.fori_loop(
        0,
        solver_.maximum_iterations,
        body,
        (impulse, previous, momentum),
    )
    residual_norm = jnp.sqrt(jnp.sum(_natural_residual(program, impulse) ** 2))
    return _evidence(
        program,
        impulse,
        residual_norm,
        solver_.maximum_iterations,
        solver_.tolerance,
        solver_.plan_id,
    )


def solve_contact_semismooth(
    program: ContactConeProgram,
    /,
    *,
    solver: SemismoothContactSolverPlan | None = None,
    initial_impulse=None,
) -> ContactConeResult:
    solver_ = SemismoothContactSolverPlan() if solver is None else solver
    impulse = (
        jnp.zeros(
            (program.contact_count, program.local_dimension),
            dtype=program.free_velocity.dtype,
        )
        if initial_impulse is None
        else jnp.asarray(initial_impulse, dtype=program.free_velocity.dtype)
    )
    impulse = project_coulomb_cone(impulse, program.friction)
    iterations = 0
    for iteration in range(solver_.maximum_iterations):
        residual = _natural_residual(program, impulse)
        residual_norm = jnp.sqrt(jnp.sum(residual * residual))
        iterations = iteration + 1
        if bool(residual_norm <= solver_.tolerance):
            break
        flat = impulse.reshape((-1,))

        def residual_flat(value):
            return _natural_residual(program, value.reshape(impulse.shape)).reshape((-1,))

        jacobian = jax.jacfwd(residual_flat)(flat)
        normal_matrix = jacobian.T @ jacobian + solver_.regularization * jnp.eye(
            jacobian.shape[1], dtype=jacobian.dtype
        )
        right = -(jacobian.T @ residual.reshape((-1,)))
        direction = _cg_normal(
            lambda value: normal_matrix @ value,
            right,
            solver_.maximum_linear_iterations,
            solver_.tolerance,
        ).reshape(impulse.shape)
        rate = 1.0
        candidate = impulse
        for _ in range(20):
            trial = project_coulomb_cone(impulse + rate * direction, program.friction)
            trial = jnp.where(program.valid[:, None], trial, 0.0)
            trial_norm = jnp.sqrt(jnp.sum(_natural_residual(program, trial) ** 2))
            if bool(trial_norm < residual_norm):
                candidate = trial
                break
            rate *= 0.5
        impulse = candidate
    residual_norm = jnp.sqrt(jnp.sum(_natural_residual(program, impulse) ** 2))
    return _evidence(
        program,
        impulse,
        residual_norm,
        iterations,
        solver_.tolerance,
        solver_.plan_id,
    )


def solve_contact_primal_dual(
    program: ContactConeProgram,
    /,
    *,
    solver: PrimalDualContactSolverPlan | None = None,
) -> ContactConeResult:
    solver_ = PrimalDualContactSolverPlan() if solver is None else solver
    dtype = program.free_velocity.dtype
    normal = jnp.where(program.valid, 1.0e-3, 0.0)
    tangent = jnp.zeros((program.contact_count, program.tangent_dimension), dtype=dtype)
    impulse = jnp.concatenate((normal[:, None], tangent), axis=-1)
    barrier = solver_.initial_barrier
    iterations = 0
    matrix = _matrix(program)

    def objective(value, barrier_value):
        flat = value.reshape((-1,))
        quadratic = (
            0.5 * flat @ matrix @ flat + program.free_velocity.reshape((-1,)) @ flat
        )
        normal_value = value[:, 0]
        tangent_value = value[:, 1:]
        cone_margin = (program.friction * normal_value) ** 2 - jnp.sum(
            tangent_value * tangent_value, axis=-1
        )
        safe_normal = jnp.where(
            program.valid,
            jnp.maximum(normal_value, jnp.finfo(dtype).tiny),
            1.0,
        )
        safe_margin = jnp.where(
            program.valid,
            jnp.maximum(cone_margin, jnp.finfo(dtype).tiny),
            1.0,
        )
        return quadratic - barrier_value * jnp.sum(
            jnp.log(safe_normal) + jnp.log(safe_margin)
        )

    for _ in range(solver_.outer_iterations):
        for _ in range(solver_.inner_iterations):
            iterations += 1
            gradient = jax.grad(objective)(impulse, barrier)
            gradient_norm = jnp.sqrt(jnp.sum(gradient * gradient))
            if bool(gradient_norm <= solver_.tolerance):
                break
            hessian_action = jax.linearize(
                lambda value: jax.grad(objective)(value, barrier), impulse
            )[1]
            direction = _cg_normal(
                lambda value: (
                    hessian_action(value.reshape(impulse.shape)).reshape((-1,))
                    + solver_.regularization * value
                ),
                -gradient.reshape((-1,)),
                solver_.maximum_linear_iterations,
                solver_.tolerance,
            ).reshape(impulse.shape)
            rate = 1.0
            current_value = objective(impulse, barrier)
            for _ in range(30):
                trial = impulse + rate * direction
                trial_normal = trial[:, 0]
                trial_tangent = trial[:, 1:]
                margin = (program.friction * trial_normal) ** 2 - jnp.sum(
                    trial_tangent * trial_tangent, axis=-1
                )
                feasible = jnp.all(
                    (~program.valid) | ((trial_normal > 0.0) & (margin > 0.0))
                )
                if bool(feasible & (objective(trial, barrier) < current_value)):
                    impulse = jnp.where(program.valid[:, None], trial, 0.0)
                    break
                rate *= 0.5
        barrier *= solver_.barrier_reduction
    impulse = project_coulomb_cone(impulse, program.friction)
    residual_norm = jnp.sqrt(jnp.sum(_natural_residual(program, impulse) ** 2))
    return _evidence(
        program,
        impulse,
        residual_norm,
        iterations,
        solver_.tolerance,
        solver_.plan_id,
    )


__all__ = [
    "PrimalDualContactSolverPlan",
    "SAPContactSolverPlan",
    "SemismoothContactSolverPlan",
    "solve_contact_primal_dual",
    "solve_contact_sap",
    "solve_contact_semismooth",
]
