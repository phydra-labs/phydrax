#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Alternative iterations for the operator-valued cone program.

Every method consumes only response actions ``W p`` of the program, iterates
in the congruently scaled variables of its block layout, and hands its
candidate to the common physical certificate of :mod:`._cone`.  Loops are
bounded JAX loops without host decisions.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import phydrax.linalg as la

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import ConvertibleToArray
from ._cone import (
    block_scale,
    build_contact_cone_result,
    ContactBlockLaw,
    ContactConeProgram,
    ContactConeResult,
    project_contact_blocks,
)


class SAPContactSolverPlan(StrictModule, NonTrainableState):
    """Accelerated projected gradient with descent-lemma backtracking."""

    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    acceleration: bool = eqx.field(static=True)
    maximum_step_reductions: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 300,
        tolerance: float = 1.0e-10,
        acceleration: bool = True,
        maximum_step_reductions: int = 64,
    ) -> None:
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        reductions = int(maximum_step_reductions)
        if iterations <= 0 or tolerance_ <= 0.0 or reductions < 0:
            raise ValueError("SAP solver controls are invalid.")
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.acceleration = bool(acceleration)
        self.maximum_step_reductions = reductions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sap-contact-solver-plan",
                "maximum_iterations": iterations,
                "tolerance": tolerance_.hex(),
                "acceleration": bool(acceleration),
                "maximum_step_reductions": reductions,
                "step": "descent-lemma-backtracking",
            }
        )

    @property
    def absolute_tolerance(self) -> float:
        return self.tolerance

    @property
    def relative_tolerance(self) -> float:
        return 0.0

    @property
    def relaxation(self) -> float:
        return 1.0


class SemismoothContactSolverPlan(StrictModule, NonTrainableState):
    """Semismooth Newton on the natural map with matrix-free GMRES directions.

    ``regularization`` shifts the generalized Jacobian, ``J + r I``; the
    direction is accepted only through a strict natural-residual decrease.
    """

    maximum_iterations: int = eqx.field(static=True)
    maximum_linear_iterations: int = eqx.field(static=True)
    maximum_line_search: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 50,
        maximum_linear_iterations: int = 200,
        maximum_line_search: int = 20,
        tolerance: float = 1.0e-10,
        regularization: float = 1.0e-10,
    ) -> None:
        nonlinear = int(maximum_iterations)
        linear = int(maximum_linear_iterations)
        line_search = int(maximum_line_search)
        tolerance_ = float(tolerance)
        regularization_ = float(regularization)
        if (
            nonlinear <= 0
            or linear <= 0
            or line_search <= 0
            or tolerance_ <= 0.0
            or regularization_ < 0.0
        ):
            raise ValueError("Semismooth contact solver controls are invalid.")
        self.maximum_iterations = nonlinear
        self.maximum_linear_iterations = linear
        self.maximum_line_search = line_search
        self.tolerance = tolerance_
        self.regularization = regularization_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "semismooth-contact-solver-plan",
                "maximum_iterations": nonlinear,
                "maximum_linear_iterations": linear,
                "maximum_line_search": line_search,
                "tolerance": tolerance_.hex(),
                "regularization": regularization_.hex(),
                "linear": "gmres",
            }
        )

    @property
    def absolute_tolerance(self) -> float:
        return self.tolerance

    @property
    def relative_tolerance(self) -> float:
        return 0.0

    @property
    def relaxation(self) -> float:
        return 1.0


class PrimalDualContactSolverPlan(StrictModule, NonTrainableState):
    """Log-barrier interior iteration with matrix-free conjugate-gradient steps."""

    outer_iterations: int = eqx.field(static=True)
    inner_iterations: int = eqx.field(static=True)
    maximum_linear_iterations: int = eqx.field(static=True)
    maximum_line_search: int = eqx.field(static=True)
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
        maximum_line_search: int = 30,
        initial_barrier: float = 1.0e-2,
        barrier_reduction: float = 0.2,
        tolerance: float = 1.0e-10,
        regularization: float = 1.0e-10,
    ) -> None:
        outer = int(outer_iterations)
        inner = int(inner_iterations)
        linear = int(maximum_linear_iterations)
        line_search = int(maximum_line_search)
        barrier = float(initial_barrier)
        reduction = float(barrier_reduction)
        tolerance_ = float(tolerance)
        regularization_ = float(regularization)
        if (
            outer <= 0
            or inner <= 0
            or linear <= 0
            or line_search <= 0
            or barrier <= 0.0
            or not 0.0 < reduction < 1.0
            or tolerance_ <= 0.0
            or regularization_ < 0.0
        ):
            raise ValueError("Primal-dual contact solver controls are invalid.")
        self.outer_iterations = outer
        self.inner_iterations = inner
        self.maximum_linear_iterations = linear
        self.maximum_line_search = line_search
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
                "maximum_line_search": line_search,
                "initial_barrier": barrier.hex(),
                "barrier_reduction": reduction.hex(),
                "tolerance": tolerance_.hex(),
                "regularization": regularization_.hex(),
                "linear": "conjugate-gradient",
            }
        )

    @property
    def maximum_iterations(self) -> int:
        return self.outer_iterations * self.inner_iterations

    @property
    def absolute_tolerance(self) -> float:
        return self.tolerance

    @property
    def relative_tolerance(self) -> float:
        return 0.0

    @property
    def relaxation(self) -> float:
        return 1.0


class _ScaledProgram(StrictModule):
    """Scaled response action, bias and projection of one program."""

    program: ContactConeProgram
    scale: Array
    bias: Array
    mask: Array

    def action(self, value: Array, /) -> Array:
        impulse = self.scale[:, None] * value
        return self.scale[:, None] * (
            self.program.response.mv(impulse) + self.program.compliance * impulse
        )

    def project(self, value: Array, /) -> Array:
        return project_contact_blocks(
            value,
            self.program.dynamic_friction,
            self.program.layout,
            self.program.valid,
        )

    def natural_residual(self, value: Array, /) -> Array:
        return value - self.project(value - (self.action(value) + self.bias))

    def unscale(self, value: Array, /) -> Array:
        return self.scale[:, None] * value


def _scaled(program: ContactConeProgram, /) -> _ScaledProgram:
    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    scale = block_scale(program)
    return _ScaledProgram(
        program,
        scale,
        scale[:, None] * program.law_free_velocity,
        program.layout.component_mask & program.valid[:, None],
    )


def _initial(
    scaled: _ScaledProgram, initial_impulse: ConvertibleToArray | None, /
) -> Array:
    program = scaled.program
    shape = program.free_velocity.shape
    dtype = program.free_velocity.dtype
    value = (
        jnp.zeros(shape, dtype=dtype)
        if initial_impulse is None
        else jnp.asarray(initial_impulse, dtype=dtype)
    )
    if value.shape != shape:
        raise ValueError("initial_impulse has invalid shape.")
    return scaled.project(value / scaled.scale[:, None])


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


def solve_contact_sap(
    program: ContactConeProgram,
    /,
    *,
    solver: SAPContactSolverPlan | None = None,
    initial_impulse: ConvertibleToArray | None = None,
) -> ContactConeResult:
    """Accelerated projected gradient (FISTA) with backtracking on the cone QP."""

    scaled = _scaled(program)
    solver_ = SAPContactSolverPlan() if solver is None else solver
    if not isinstance(solver_, SAPContactSolverPlan):
        raise TypeError("solver must be SAPContactSolverPlan or None.")
    start = _initial(scaled, initial_impulse)
    dtype = start.dtype
    slack = jnp.sqrt(jnp.finfo(dtype).eps)
    tiny = jnp.finfo(dtype).tiny
    tolerance = jnp.asarray(solver_.tolerance, dtype=dtype)
    descent = scaled.natural_residual(start)
    curvature = jnp.sum(descent * scaled.action(descent))
    squared = jnp.sum(descent * descent)
    initial_step = jnp.where(
        (squared > 0.0) & (curvature > tiny),
        squared / jnp.maximum(curvature, tiny),
        jnp.asarray(1.0, dtype=dtype),
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False)
    acceleration = jnp.asarray(1.0 if solver_.acceleration else 0.0, dtype=dtype)

    _State = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]

    def condition(state: _State) -> Array:
        index, _, _, _, _, converged, _, _, exhausted = state
        return (index < solver_.maximum_iterations) & ~converged & ~exhausted

    def body(state: _State) -> _State:
        index, value, previous, momentum, step, _, _, reductions, _ = state
        residual = _norm(scaled.natural_residual(value))
        now = residual <= tolerance
        extrapolated = value + acceleration * (
            (momentum - 1.0) / (momentum + 2.0)
        ) * (value - previous)
        extrapolated = scaled.project(extrapolated)
        gradient = scaled.action(extrapolated) + scaled.bias

        def trial(step_: Array) -> tuple[Array, Array]:
            candidate = scaled.project(extrapolated - step_ * gradient)
            difference = candidate - extrapolated
            accepted = jnp.sum(difference * scaled.action(difference)) <= (
                (1.0 + slack) * jnp.sum(difference * difference) / step_
            )
            return candidate, accepted

        first, first_accepted = trial(step)

        def search_condition(search: tuple[Array, Array, Array, Array]) -> Array:
            _, _, accepted, count = search
            return (~accepted) & (count < solver_.maximum_step_reductions)

        def search_body(
            search: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            step_, _, _, count = search
            halved = 0.5 * step_
            candidate, accepted = trial(halved)
            return halved, candidate, accepted, count + 1

        final_step, candidate, accepted, count = jax.lax.while_loop(
            search_condition,
            search_body,
            (step, first, first_accepted, reductions),
        )
        advance = (~now) & accepted
        return (
            index + 1,
            jnp.where(advance, candidate, value),
            jnp.where(advance, value, previous),
            jnp.where(advance, momentum + 1.0, momentum),
            jnp.where(now, step, final_step),
            now,
            residual,
            jnp.where(now, reductions, count),
            (~now) & ~accepted,
        )

    (
        iterations,
        value,
        _,
        _,
        step,
        converged,
        residual,
        reductions,
        exhausted,
    ) = jax.lax.while_loop(
        condition,
        body,
        (
            zero,
            start,
            start,
            jnp.asarray(1.0, dtype=dtype),
            initial_step,
            false,
            jnp.asarray(jnp.inf, dtype=dtype),
            zero,
            false,
        ),
    )
    return build_contact_cone_result(
        program,
        solver_,
        scaled.unscale(value),
        converged=converged,
        iterations=iterations,
        fixed_point_residual=residual,
        iteration_tolerance=tolerance,
        step_size=step,
        step_reductions=reductions,
        step_budget_exhausted=exhausted,
    )


def solve_contact_semismooth(
    program: ContactConeProgram,
    /,
    *,
    solver: SemismoothContactSolverPlan | None = None,
    initial_impulse: ConvertibleToArray | None = None,
) -> ContactConeResult:
    """Semismooth Newton with prepared linearizations and restarted GMRES."""

    scaled = _scaled(program)
    solver_ = SemismoothContactSolverPlan() if solver is None else solver
    if not isinstance(solver_, SemismoothContactSolverPlan):
        raise TypeError("solver must be SemismoothContactSolverPlan or None.")
    start = _initial(scaled, initial_impulse)
    dtype = start.dtype
    space = la.ArraySpace(start.shape, dtype=dtype)
    tolerance = jnp.asarray(solver_.tolerance, dtype=dtype)
    regularization = jnp.asarray(solver_.regularization, dtype=dtype)
    linear_policy = la.LinearSolvePolicy(
        la.GMRES(),
        tolerance=la.TolerancePolicy(
            relative=1.0e-2 * solver_.tolerance,
            absolute=0.0,
            max_steps=solver_.maximum_linear_iterations,
        ),
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False)

    _State = tuple[Array, Array, Array, Array, Array, Array, Array]

    def condition(state: _State) -> Array:
        index, _, _, converged, _, _, stalled = state
        return (index < solver_.maximum_iterations) & ~converged & ~stalled

    def body(state: _State) -> _State:
        index, value, _, _, rate, reductions, _ = state
        linearization = la.prepare_linearization(
            scaled.natural_residual, value, source=space, target=space
        )
        residual = linearization.primal
        residual_norm = _norm(residual)
        now = residual_norm <= tolerance

        def shifted(direction: Array) -> Array:
            return linearization.jvp(direction) + regularization * direction

        def shifted_transpose(direction: Array) -> Array:
            return linearization.vjp(direction) + regularization * direction

        operator = la.FunctionLinearOperator(
            shifted,
            source=space,
            target=space,
            transpose_action=shifted_transpose,
            operator_id="semismooth-contact-newton-operator",
        )
        solution = la.solve(la.LinearSystem(operator), -residual, policy=linear_policy)
        direction = jnp.where(
            jnp.all(jnp.isfinite(solution.value)), solution.value, -residual
        )

        def trial(rate_: Array) -> tuple[Array, Array]:
            candidate = scaled.project(value + rate_ * direction)
            return candidate, _norm(scaled.natural_residual(candidate)) < residual_norm

        first, first_accepted = trial(jnp.asarray(1.0, dtype=dtype))

        def search_condition(search: tuple[Array, Array, Array, Array]) -> Array:
            _, _, accepted, count = search
            return (~accepted) & (count < solver_.maximum_line_search)

        def search_body(
            search: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            rate_, _, _, count = search
            halved = 0.5 * rate_
            candidate, accepted = trial(halved)
            return halved, candidate, accepted, count + 1

        final_rate, candidate, accepted, count = jax.lax.while_loop(
            search_condition,
            search_body,
            (jnp.asarray(1.0, dtype=dtype), first, first_accepted, zero),
        )
        advance = (~now) & accepted
        return (
            index + 1,
            jnp.where(advance, candidate, value),
            residual_norm,
            now,
            jnp.where(advance, final_rate, rate),
            reductions + jnp.where(now, 0, count),
            (~now) & ~accepted,
        )

    iterations, value, residual, converged, rate, reductions, stalled = (
        jax.lax.while_loop(
            condition,
            body,
            (
                zero,
                start,
                jnp.asarray(jnp.inf, dtype=dtype),
                false,
                jnp.asarray(1.0, dtype=dtype),
                zero,
                false,
            ),
        )
    )
    return build_contact_cone_result(
        program,
        solver_,
        scaled.unscale(value),
        converged=converged,
        iterations=iterations,
        fixed_point_residual=residual,
        iteration_tolerance=tolerance,
        step_size=rate,
        step_reductions=reductions,
        step_budget_exhausted=stalled,
    )


def _barrier_terms(
    scaled: _ScaledProgram, value: Array, /
) -> tuple[Array, Array, Array, Array]:
    """Return normal, cone margin, interior flags and the barrier-safe values."""

    program = scaled.program
    dtype = value.dtype
    coulomb = (
        (program.layout.law == int(ContactBlockLaw.COULOMB))
        & program.valid
        & (program.dynamic_friction > 0.0)
    )
    normal = value[:, 0]
    tangent = jnp.where(scaled.mask[:, 1:], value[:, 1:], 0.0)
    margin = (program.dynamic_friction * normal) ** 2 - jnp.sum(tangent**2, axis=-1)
    interior = (~program.valid) | ((normal > 0.0) & ((~coulomb) | (margin > 0.0)))
    safe_normal = jnp.where(
        program.valid, jnp.maximum(normal, jnp.finfo(dtype).tiny), 1.0
    )
    safe_margin = jnp.where(coulomb, jnp.maximum(margin, jnp.finfo(dtype).tiny), 1.0)
    return safe_normal, safe_margin, interior, coulomb


def solve_contact_primal_dual(
    program: ContactConeProgram,
    /,
    *,
    solver: PrimalDualContactSolverPlan | None = None,
) -> ContactConeResult:
    """Log-barrier path following on the cone QP with matrix-free CG Newton steps.

    Normal-only blocks carry only the normal barrier; padded and invalid
    components stay exactly zero.
    """

    scaled = _scaled(program)
    solver_ = PrimalDualContactSolverPlan() if solver is None else solver
    if not isinstance(solver_, PrimalDualContactSolverPlan):
        raise TypeError("solver must be PrimalDualContactSolverPlan or None.")
    dtype = program.free_velocity.dtype
    shape = program.free_velocity.shape
    space = la.ArraySpace(shape, dtype=dtype)
    frictional = (program.layout.law == int(ContactBlockLaw.COULOMB)) & (
        program.dynamic_friction > 0.0
    )
    normal_component = (jnp.arange(program.local_dimension) == 0)[None, :]
    mask = scaled.mask & (frictional[:, None] | normal_component)
    start = jnp.zeros(shape, dtype=dtype).at[:, 0].set(
        jnp.where(program.valid, 1.0e-3, 0.0)
    )
    tolerance = jnp.asarray(solver_.tolerance, dtype=dtype)
    regularization = jnp.asarray(solver_.regularization, dtype=dtype)
    linear_policy = la.LinearSolvePolicy(
        la.ConjugateGradient(),
        tolerance=la.TolerancePolicy(
            relative=1.0e-2 * solver_.tolerance,
            absolute=0.0,
            max_steps=solver_.maximum_linear_iterations,
        ),
    )
    zero = jnp.asarray(0, dtype=jnp.int32)

    def objective(value: Array, barrier: Array) -> Array:
        quadratic = 0.5 * jnp.sum(value * scaled.action(value)) + jnp.sum(
            scaled.bias * value
        )
        safe_normal, safe_margin, _, _ = _barrier_terms(scaled, value)
        return quadratic - barrier * jnp.sum(jnp.log(safe_normal) + jnp.log(safe_margin))

    def masked_gradient(value: Array, barrier: Array) -> Array:
        return jnp.where(mask, jax.grad(objective)(value, barrier), 0.0)

    def newton_step(
        state: tuple[Array, Array, Array, Array], barrier: Array
    ) -> tuple[Array, Array, Array, Array]:
        value, iterations, reductions, stalled = state
        gradient, hessian = jax.linearize(
            lambda point: masked_gradient(point, barrier), value
        )
        gradient_norm = _norm(gradient)
        done = gradient_norm <= tolerance

        def action(direction: Array) -> Array:
            masked = jnp.where(mask, direction, 0.0)
            return (
                jnp.where(mask, hessian(masked), 0.0)
                + jnp.where(mask, 0.0, direction)
                + regularization * direction
            )

        operator = la.FunctionLinearOperator(
            action,
            source=space,
            target=space,
            transpose_action=action,
            properties=la.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id="primal-dual-contact-barrier-hessian",
        )
        solution = la.solve(la.LinearSystem(operator), -gradient, policy=linear_policy)
        direction = jnp.where(mask, solution.value, 0.0)
        current = objective(value, barrier)

        def trial(rate: Array) -> tuple[Array, Array]:
            candidate = jnp.where(mask, value + rate * direction, 0.0)
            _, _, interior, _ = _barrier_terms(scaled, candidate)
            accepted = (
                jnp.all(interior)
                & jnp.all(jnp.isfinite(candidate))
                & (objective(candidate, barrier) < current)
            )
            return candidate, accepted

        first, first_accepted = trial(jnp.asarray(1.0, dtype=dtype))

        def search_condition(search: tuple[Array, Array, Array, Array]) -> Array:
            _, _, accepted, count = search
            return (~accepted) & (count < solver_.maximum_line_search)

        def search_body(
            search: tuple[Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array]:
            rate, _, _, count = search
            halved = 0.5 * rate
            candidate, accepted = trial(halved)
            return halved, candidate, accepted, count + 1

        _, candidate, accepted, count = jax.lax.while_loop(
            search_condition,
            search_body,
            (jnp.asarray(1.0, dtype=dtype), first, first_accepted, zero),
        )
        active = (~done) & ~stalled
        advance = active & accepted
        return (
            jnp.where(advance, candidate, value),
            iterations + jnp.where(active, 1, 0),
            reductions + jnp.where(active, count, 0),
            stalled | (active & ~accepted),
        )

    def barrier_level(
        level: Array, state: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        barrier = solver_.initial_barrier * jnp.asarray(
            solver_.barrier_reduction, dtype=dtype
        ) ** level.astype(dtype)
        value, iterations, reductions, _ = state
        return jax.lax.fori_loop(
            0,
            solver_.inner_iterations,
            lambda _, inner: newton_step(inner, barrier),
            (value, iterations, reductions, jnp.asarray(False)),
        )

    value, iterations, reductions, stalled = jax.lax.fori_loop(
        0,
        solver_.outer_iterations,
        barrier_level,
        (start, zero, zero, jnp.asarray(False)),
    )
    value = scaled.project(value)
    residual = _norm(scaled.natural_residual(value))
    return build_contact_cone_result(
        program,
        solver_,
        scaled.unscale(value),
        converged=residual <= tolerance,
        iterations=iterations,
        fixed_point_residual=residual,
        iteration_tolerance=tolerance,
        step_size=jnp.asarray(1.0, dtype=dtype),
        step_reductions=reductions,
        step_budget_exhausted=stalled,
    )


__all__ = [
    "PrimalDualContactSolverPlan",
    "SAPContactSolverPlan",
    "SemismoothContactSolverPlan",
    "solve_contact_primal_dual",
    "solve_contact_sap",
    "solve_contact_semismooth",
]
