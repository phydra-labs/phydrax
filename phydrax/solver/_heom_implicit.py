#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._temporal_precision import TemporalPrecisionPolicy
from ..linalg import (
    ArraySpace,
    DiagonalPreconditioner,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    PreconditioningPolicy,
    solve,
)
from ._bdf_method import bdf_shift_offset
from ._heom import HEOMProblem, HEOMSolution


def _heom_temporal_precision(state: Array) -> TemporalPrecisionPolicy:
    real_dtype = jnp.real(state).dtype
    return TemporalPrecisionPolicy(
        coefficient_dtype=real_dtype,
        state_dtype=state.dtype,
        stage_dtype=state.dtype,
        accumulation_dtype=state.dtype,
        residual_dtype=state.dtype,
        decision_dtype=real_dtype,
        checkpoint_dtype=state.dtype,
        output_dtype=state.dtype,
    )


class HEOMImplicitEvidence(StrictModule):
    linear_residuals: Array
    successful_steps: Array
    valid: Array

    def __init__(self, linear_residuals: ArrayLike, successful_steps: ArrayLike, /):
        self.linear_residuals = jnp.asarray(linear_residuals)
        self.successful_steps = jnp.asarray(successful_steps, dtype=bool)
        self.valid = jnp.all(jnp.isfinite(self.linear_residuals)) & jnp.all(
            self.successful_steps
        )


class HEOMImplicitResult(StrictModule):
    solution: HEOMSolution
    evidence: HEOMImplicitEvidence
    valid: Array

    def __init__(self, solution: HEOMSolution, evidence: HEOMImplicitEvidence, /):
        self.solution = solution
        self.evidence = evidence
        self.valid = solution.valid & evidence.valid


def solve_heom_backward_euler(
    problem: HEOMProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> HEOMImplicitResult:
    """Matrix-free backward Euler for the linear HEOM generator."""
    step = jnp.asarray(step_size, dtype=problem.initial_state.real.dtype).reshape(())
    if int(steps) <= 0 or not bool(jnp.isfinite(step)) or float(step) <= 0.0:
        raise ValueError("Backward Euler steps and step_size must be positive.")
    state = problem.initial_state
    shape = state.shape
    space = ArraySpace(shape, dtype=state.dtype, space_id=f"{problem.problem_id}:ado")
    residuals = []
    successful = []
    roots = [state[0]]
    for _ in range(int(steps)):
        state, result = _backward_euler_step(problem, state, step, space)
        residuals.append(result.diagnostics.residual_norm)
        successful.append(result.successful)
        roots.append(state[0])
    solution = HEOMSolution(
        problem,
        jnp.stack(roots),
        state,
        step * jnp.arange(int(steps) + 1),
        step_size=step,
        temporal_precision=_heom_temporal_precision(state),
        geometry_precision=problem.geometry_precision,
        hermitian_precision=problem.hermitian_precision,
    )
    evidence = HEOMImplicitEvidence(jnp.stack(residuals), jnp.stack(successful))
    return HEOMImplicitResult(solution, evidence)


class HEOMTierBlockPreconditioner(StrictModule):
    diagonal: Array

    def __init__(
        self,
        problem: HEOMProblem,
        shift: ArrayLike,
        /,
    ):
        decay = jnp.real(problem.hierarchy.multi_indices @ problem.expansion.exponents)
        self.diagonal = jnp.asarray(shift) + decay

    def apply(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        return values / self.diagonal[:, None, None]

    def as_preconditioner(
        self,
        space: ArraySpace,
        shape: tuple[int, ...],
        /,
    ) -> DiagonalPreconditioner:
        diagonal = jnp.broadcast_to(self.diagonal[:, None, None], shape).reshape(-1)
        return DiagonalPreconditioner(
            diagonal,
            space=space,
            preconditioner_id="heom-tier-block",
        )


class HEOMBDFEvidence(StrictModule):
    linear_residuals: Array
    successful_steps: Array
    orders: Array
    preconditioned_rhs_norms: Array
    valid: Array

    def __init__(
        self,
        linear_residuals: ArrayLike,
        successful_steps: ArrayLike,
        orders: ArrayLike,
        preconditioned_rhs_norms: ArrayLike,
        /,
    ):
        self.linear_residuals = jnp.asarray(linear_residuals)
        self.successful_steps = jnp.asarray(successful_steps, dtype=bool)
        self.orders = jnp.asarray(orders, dtype=jnp.int32)
        self.preconditioned_rhs_norms = jnp.asarray(preconditioned_rhs_norms)
        self.valid = (
            jnp.all(jnp.isfinite(self.linear_residuals))
            & jnp.all(self.successful_steps)
            & jnp.all(jnp.isfinite(self.preconditioned_rhs_norms))
        )


class HEOMBDFResult(StrictModule):
    solution: HEOMSolution
    evidence: HEOMBDFEvidence
    valid: Array

    def __init__(self, solution: HEOMSolution, evidence: HEOMBDFEvidence, /):
        self.solution = solution
        self.evidence = evidence
        self.valid = solution.valid & evidence.valid


def solve_heom_bdf(
    problem: HEOMProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    maximum_order: int = 5,
) -> HEOMBDFResult:
    """Fixed-step variable-order BDF1–5 with exact linear HEOM action."""
    order_limit = int(maximum_order)
    if not 1 <= order_limit <= 5:
        raise ValueError("maximum_order must lie in [1,5].")
    step = jnp.asarray(step_size, dtype=problem.initial_state.real.dtype).reshape(())
    if int(steps) <= 0 or not bool(jnp.isfinite(step)) or float(step) <= 0.0:
        raise ValueError("HEOM BDF steps and step_size must be positive.")
    state = problem.initial_state
    space = ArraySpace(
        state.shape,
        dtype=state.dtype,
        space_id=f"{problem.problem_id}:bdf-ado",
    )
    history = jnp.broadcast_to(state, (5,) + state.shape)
    history_times = -step * jnp.arange(5)
    roots = [state[0]]
    residuals = []
    successful = []
    orders = []
    preconditioned = []
    for index in range(int(steps)):
        target_time = (index + 1) * step
        order = min(index + 1, order_limit)
        shift, offset = bdf_shift_offset(
            history,
            history_times,
            target_time,
            jnp.asarray(order, dtype=jnp.int32),
        )
        operator = FunctionLinearOperator(
            lambda value: shift * value - problem.rhs(value),
            source=space,
            target=space,
            operator_id=f"{problem.problem_id}:bdf:{index}:order-{order}",
        )
        right_hand_side = -offset
        preconditioner = HEOMTierBlockPreconditioner(problem, shift)
        policy = LinearSolvePolicy(
            preconditioning=PreconditioningPolicy(
                preconditioner.as_preconditioner(space, state.shape),
                side="right",
            )
        )
        result = solve(
            LinearSystem(
                operator,
                problem_id=f"{problem.problem_id}:bdf-system",
            ),
            right_hand_side,
            policy=policy,
        )
        state = result.value
        preconditioned.append(jnp.linalg.norm(preconditioner.apply(right_hand_side)))
        residuals.append(result.diagnostics.residual_norm)
        successful.append(result.successful)
        orders.append(order)
        roots.append(state[0])
        history = jnp.concatenate((state[None, ...], history[:-1]), axis=0)
        history_times = jnp.concatenate((target_time[None], history_times[:-1]), axis=0)
    solution = HEOMSolution(
        problem,
        jnp.stack(roots),
        state,
        step * jnp.arange(int(steps) + 1),
        step_size=step,
        temporal_precision=_heom_temporal_precision(state),
        geometry_precision=problem.geometry_precision,
        hermitian_precision=problem.hermitian_precision,
    )
    evidence = HEOMBDFEvidence(
        jnp.stack(residuals) if residuals else jnp.zeros((0,)),
        jnp.stack(successful) if successful else jnp.zeros((0,), dtype=bool),
        jnp.asarray(orders),
        jnp.stack(preconditioned) if preconditioned else jnp.zeros((0,)),
    )
    return HEOMBDFResult(solution, evidence)


class HEOMAdaptiveBDFEvidence(StrictModule):
    attempted_step_sizes: Array
    accepted_steps: Array
    error_ratios: Array
    linear_residuals: Array
    capacity_saturated: Array
    valid: Array

    def __init__(
        self,
        attempted_step_sizes: ArrayLike,
        accepted_steps: ArrayLike,
        error_ratios: ArrayLike,
        linear_residuals: ArrayLike,
        capacity_saturated: ArrayLike,
        /,
    ):
        self.attempted_step_sizes = jnp.asarray(attempted_step_sizes)
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=bool)
        self.error_ratios = jnp.asarray(error_ratios)
        self.linear_residuals = jnp.asarray(linear_residuals)
        self.capacity_saturated = jnp.asarray(capacity_saturated, dtype=bool)
        count = self.attempted_step_sizes.shape[0]
        if (
            self.attempted_step_sizes.shape != (count,)
            or self.accepted_steps.shape != (count,)
            or self.error_ratios.shape != (count,)
            or self.linear_residuals.shape != (count, 3)
            or self.capacity_saturated.shape != ()
        ):
            raise ValueError("Adaptive HEOM evidence shapes are inconsistent.")
        self.valid = (
            (count > 0)
            & jnp.all(jnp.isfinite(self.attempted_step_sizes))
            & jnp.all(self.attempted_step_sizes > 0.0)
            & jnp.all(jnp.isfinite(self.error_ratios))
            & jnp.all(jnp.isfinite(self.linear_residuals))
            & ~self.capacity_saturated
            & jnp.any(self.accepted_steps)
        )


class HEOMAdaptiveBDFResult(StrictModule):
    solution: HEOMSolution
    evidence: HEOMAdaptiveBDFEvidence
    accepted_step_count: int
    rejected_step_count: int
    maximum_attempts: int
    valid: Array

    def __init__(
        self,
        solution: HEOMSolution,
        evidence: HEOMAdaptiveBDFEvidence,
        /,
        *,
        maximum_attempts: int,
    ):
        self.solution = solution
        self.evidence = evidence
        self.accepted_step_count = int(jnp.sum(evidence.accepted_steps))
        self.rejected_step_count = int(jnp.sum(~evidence.accepted_steps))
        self.maximum_attempts = int(maximum_attempts)
        self.valid = solution.valid & evidence.valid


def _backward_euler_step(
    problem: HEOMProblem,
    state: Array,
    step: Array,
    space: ArraySpace,
    /,
):
    shift = jnp.reciprocal(step)
    operator = FunctionLinearOperator(
        lambda value: shift * value - problem.rhs(value),
        source=space,
        target=space,
        operator_id=f"{problem.problem_id}:adaptive-bdf1",
    )
    right_hand_side = shift * state
    preconditioner = HEOMTierBlockPreconditioner(problem, shift)
    policy = LinearSolvePolicy(
        preconditioning=PreconditioningPolicy(
            preconditioner.as_preconditioner(space, state.shape),
            side="right",
        )
    )
    result = solve(
        LinearSystem(
            operator,
            problem_id=f"{problem.problem_id}:adaptive-bdf1-system",
        ),
        right_hand_side,
        policy=policy,
    )
    return result.value, result


def solve_heom_adaptive_bdf(
    problem: HEOMProblem,
    /,
    *,
    final_time: ArrayLike,
    initial_step: ArrayLike,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-8,
    minimum_step: float = 1e-8,
    maximum_step: float | None = None,
    maximum_attempts: int = 10_000,
) -> HEOMAdaptiveBDFResult:
    """Adaptive matrix-free BDF1 using a full-step/two-half-step estimator."""
    final = float(jnp.asarray(final_time).reshape(()))
    initial = float(jnp.asarray(initial_step).reshape(()))
    relative = float(relative_tolerance)
    absolute = float(absolute_tolerance)
    minimum = float(minimum_step)
    maximum = final if maximum_step is None else float(maximum_step)
    attempt_limit = int(maximum_attempts)
    if (
        not all(
            bool(jnp.isfinite(value))
            for value in (final, initial, relative, absolute, minimum, maximum)
        )
        or final <= 0.0
        or initial <= 0.0
        or relative <= 0.0
        or absolute <= 0.0
        or minimum <= 0.0
        or maximum < minimum
        or attempt_limit <= 0
    ):
        raise ValueError("Adaptive HEOM policy values are invalid.")
    state = problem.initial_state
    space = ArraySpace(
        state.shape,
        dtype=state.dtype,
        space_id=f"{problem.problem_id}:adaptive-bdf-ado",
    )
    time = 0.0
    step = min(max(initial, minimum), maximum, final)
    times = [jnp.asarray(0.0, dtype=state.real.dtype)]
    roots = [state[0]]
    attempted_steps = []
    accepted = []
    error_ratios = []
    residuals = []
    attempts = 0
    while time < final and attempts < attempt_limit:
        step = min(step, maximum, final - time)
        step_array = jnp.asarray(step, dtype=state.real.dtype)
        full, full_result = _backward_euler_step(problem, state, step_array, space)
        half, first_half_result = _backward_euler_step(
            problem, state, 0.5 * step_array, space
        )
        refined, second_half_result = _backward_euler_step(
            problem, half, 0.5 * step_array, space
        )
        scale = absolute + relative * jnp.maximum(
            jnp.linalg.norm(refined), jnp.linalg.norm(state)
        )
        error_ratio = jnp.linalg.norm(refined - full) / scale
        linear_success = (
            jnp.all(full_result.successful)
            & jnp.all(first_half_result.successful)
            & jnp.all(second_half_result.successful)
        )
        accept = bool(linear_success & (error_ratio <= 1.0))
        attempted_steps.append(step_array)
        accepted.append(jnp.asarray(accept))
        error_ratios.append(error_ratio)
        residuals.append(
            jnp.stack(
                (
                    full_result.diagnostics.residual_norm,
                    first_half_result.diagnostics.residual_norm,
                    second_half_result.diagnostics.residual_norm,
                )
            )
        )
        attempts += 1
        if accept:
            state = refined
            time += step
            times.append(jnp.asarray(time))
            roots.append(state[0])
        factor = float(
            jnp.clip(
                0.9 * jnp.maximum(error_ratio, 1e-12) ** -0.5,
                0.2,
                2.0,
            )
        )
        next_step = min(maximum, max(minimum, step * factor))
        if not accept and step <= minimum and next_step <= minimum:
            break
        step = next_step
    saturated = time < final
    evidence = HEOMAdaptiveBDFEvidence(
        jnp.stack(attempted_steps),
        jnp.stack(accepted),
        jnp.stack(error_ratios),
        jnp.stack(residuals),
        saturated,
    )
    accepted_step = jnp.max(
        jnp.where(evidence.accepted_steps, evidence.attempted_step_sizes, 0.0)
    )
    solution = HEOMSolution(
        problem,
        jnp.stack(roots),
        state,
        jnp.stack(times),
        step_size=accepted_step,
        temporal_precision=_heom_temporal_precision(state),
        geometry_precision=problem.geometry_precision,
        hermitian_precision=problem.hermitian_precision,
        maximum_time_step=maximum,
    )
    return HEOMAdaptiveBDFResult(
        solution,
        evidence,
        maximum_attempts=attempt_limit,
    )


__all__ = [
    "HEOMAdaptiveBDFEvidence",
    "HEOMAdaptiveBDFResult",
    "HEOMBDFEvidence",
    "HEOMBDFResult",
    "HEOMImplicitEvidence",
    "HEOMImplicitResult",
    "HEOMTierBlockPreconditioner",
    "solve_heom_adaptive_bdf",
    "solve_heom_backward_euler",
    "solve_heom_bdf",
]
