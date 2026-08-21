#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array

from ..._strict import StrictModule
from .._plans import _certified_rank, LinearSolvePlan
from .._policies import (
    FGMRES,
    GeneralizedLSMR,
    GMRES,
    LinearSolveControl,
    MINRES,
    PCG,
    ProjectedPCG,
)
from .._preconditioners import AbstractPreconditioner
from .._problems import LeastSquaresProblem
from .._results import LinearSolveStatus
from ..krylov._results import KrylovBreakdownStatus


class NativeKrylovState(StrictModule):
    problem: Any
    preconditioner: AbstractPreconditioner | None


class NativeKrylovBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


class _LSMRState(NamedTuple):
    iteration: Array
    alpha: Array
    u_operator: Array
    u_regularizer: Array
    v: Array
    alphabar: Array
    rho: Array
    rhobar: Array
    zeta: Array
    sbar: Array
    cbar: Array
    zetabar: Array
    hbar: Array
    h: Array
    x: Array
    betadd: Array
    thetatilde: Array
    rhodold: Array
    betad: Array
    tautildeold: Array
    accumulated_residual: Array
    norm_a_squared: Array
    maximum_rbar: Array
    minimum_rbar: Array
    normal_residual: Array
    residual: Array
    norm_a: Array
    condition: Array
    active: Array
    breakdown: Array


def prepare_native_krylov(
    problem: Any,
    plan: LinearSolvePlan,
    /,
    *,
    preconditioner: AbstractPreconditioner | None = None,
) -> NativeKrylovState:
    if plan.backend != "native-krylov":
        raise ValueError("Native Krylov preparation requires a native-krylov plan.")
    if problem.operator.batch_shape:
        raise ValueError("Native Krylov preparation requires an unbatched operator.")
    if (plan.preconditioner_plan is None) != (preconditioner is None):
        raise ValueError("Prepared preconditioning must match the symbolic solve plan.")
    return NativeKrylovState(problem, preconditioner)


def _runtime_controls(
    problem: Any,
    plan: LinearSolvePlan,
    control: LinearSolveControl | None,
    dtype: Any,
    /,
) -> tuple[Array, Array, Array, int]:
    # The planned bound fixes loop/basis structure; each invocation may only
    # tighten it.
    tolerance = plan.policy.tolerance
    structural_max_steps = tolerance.max_steps or (
        max(problem.operator.source.size, problem.operator.target.size)
        if isinstance(problem, LeastSquaresProblem)
        else max(1, problem.operator.source.size)
    )
    relative = jnp.asarray(
        (
            tolerance.relative
            if control is None or control.relative_tolerance is None
            else control.relative_tolerance
        ),
        dtype=dtype,
    )
    absolute = jnp.asarray(
        (
            tolerance.absolute
            if control is None or control.absolute_tolerance is None
            else control.absolute_tolerance
        ),
        dtype=dtype,
    )
    max_steps = jnp.asarray(
        (
            structural_max_steps
            if control is None or control.maximum_steps is None
            else control.maximum_steps
        ),
    )
    max_steps = eqx.error_if(
        max_steps,
        max_steps > structural_max_steps,
        "Runtime maximum_steps cannot exceed the prepared structural limit.",
    ).astype(jnp.int32)
    return relative, absolute, max_steps, structural_max_steps


def solve_native_krylov(
    state: NativeKrylovState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    initial_guess: Array | None = None,
    control: LinearSolveControl | None = None,
) -> NativeKrylovBackendOutput:
    if rhs.ndim != 2:
        raise ValueError("Native Krylov right-hand sides must have shape (m, k).")
    if control is not None and not isinstance(control, LinearSolveControl):
        raise TypeError("control must be a LinearSolveControl or None.")
    problem = state.problem
    if plan.policy.differentiation.mode == "rhs-only":
        problem = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
            problem,
        )
    method = plan.policy.method
    method_name = plan.method if method.name == "auto" else method.name
    (
        relative_tolerance,
        absolute_tolerance,
        maximum_steps,
        structural_maximum_steps,
    ) = _runtime_controls(problem, plan, control, rhs.real.dtype)
    guesses = (
        jnp.zeros((problem.operator.source.size, rhs.shape[1]), dtype=rhs.dtype)
        if initial_guess is None
        else initial_guess
    )
    if guesses.shape != (problem.operator.source.size, rhs.shape[1]):
        raise ValueError("initial_guess must match canonical solution and RHS axes.")

    def solve_column(target, guess):
        if method_name == PCG().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "pcg",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
            )
        if method_name == ProjectedPCG().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "projected-pcg",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
            )
        if method_name == MINRES().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "minres",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
            )
        if method_name == FGMRES().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "fgmres",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
            )
        if method_name == GMRES().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "gmres",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
            )
        if method_name == GeneralizedLSMR().name:
            return _least_squares_solve(
                problem,
                target,
                guess,
                plan,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
            )
        raise ValueError(f"Unsupported native Krylov method {method_name!r}.")

    value, auxiliary = jax.vmap(solve_column, in_axes=(1, 1), out_axes=(1, 0))(
        rhs, guesses
    )
    (
        iterations,
        residual,
        normal_residual,
        condition,
        breakdown,
        matvec_count,
        adjoint_matvec_count,
    ) = auxiliary
    tolerance = (relative_tolerance, absolute_tolerance)
    rhs_norms = jax.vmap(
        lambda column: _space_norm(problem.operator.target, column), in_axes=1
    )(rhs)
    converged = residual <= tolerance[1] + tolerance[0] * rhs_norms
    if isinstance(problem, LeastSquaresProblem):

        def normal_reference(column):
            _, adjoint, _, source_inner, target = _least_squares_actions(problem, column)
            return _norm(adjoint(target), source_inner)

        normal_reference = jax.vmap(normal_reference, in_axes=1)(rhs)
        converged = normal_residual <= (tolerance[1] + tolerance[0] * normal_reference)
        adjoint_matvec_count = adjoint_matvec_count + 1
    status = jnp.full(rhs_norms.shape, int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~converged,
        int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    if method_name == GeneralizedLSMR().name:
        selected_lsmr = (
            method if isinstance(method, GeneralizedLSMR) else GeneralizedLSMR()
        )
        status = jnp.where(
            (~converged) & (condition >= selected_lsmr.condition_limit),
            int(LinearSolveStatus.CONDITION_LIMIT_REACHED),
            status,
        )
    status = jnp.where(
        breakdown == int(KrylovBreakdownStatus.NONFINITE_ACTION),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    status = jnp.where(
        breakdown == int(KrylovBreakdownStatus.STAGNATION),
        int(LinearSolveStatus.STAGNATION),
        status,
    )
    status = jnp.where(
        (breakdown != int(KrylovBreakdownStatus.NONE))
        & (breakdown != int(KrylovBreakdownStatus.HAPPY))
        & (breakdown != int(KrylovBreakdownStatus.NONFINITE_ACTION))
        & (breakdown != int(KrylovBreakdownStatus.STAGNATION)),
        int(LinearSolveStatus.BREAKDOWN),
        status,
    )
    if plan.policy.differentiation.mode == "none":
        value = jax.lax.stop_gradient(value)
    if method_name == ProjectedPCG().name:
        rank = (
            problem.operator.source.size
            - problem.nullspace_policy.certificate.right.dimension
        ).astype(jnp.int32)
    else:
        certified_rank = _certified_rank(problem.operator)
        rank = jnp.asarray(
            -1 if certified_rank is None else certified_rank,
            dtype=jnp.int32,
        )
    return NativeKrylovBackendOutput(
        value=value,
        status=status,
        iterations=iterations,
        matvec_count=matvec_count,
        adjoint_matvec_count=adjoint_matvec_count,
        rank=rank,
        condition_estimate=condition,
        singular_values=None,
    )


def _square_solve(
    problem,
    rhs,
    initial,
    plan,
    method: str,
    /,
    *,
    preconditioner: AbstractPreconditioner | None,
    relative: Array,
    absolute: Array,
    max_steps: Array,
    structural_max_steps: int,
):
    operator = problem.operator
    action = lambda vector: _action_coordinates(operator, vector)
    inner = lambda left, right: _space_inner(operator.source, left, right)
    precondition = _preconditioner_action(preconditioner, operator.source)
    tolerance = (relative, absolute)
    projected = method == "projected-pcg"
    if projected:
        nullspace = problem.nullspace_policy
        certificate = nullspace.certificate
        complement = lambda vector: vector - nullspace.right.project_coordinates(vector)
        rhs = eqx.error_if(
            complement(rhs),
            (~certificate.valid) | (certificate.right.dimension < 1),
            "ProjectedPCG requires a valid nonempty kernel certificate.",
        )
        initial = complement(initial)
        unprojected_action = action
        action = lambda vector: complement(unprojected_action(complement(vector)))
        unprojected_precondition = precondition
        precondition = lambda vector, iteration: complement(
            unprojected_precondition(complement(vector), iteration)
        )
        method = "pcg"

    def run(selected_action, target):
        if method == "pcg":
            value, auxiliary = _pcg_raw(
                selected_action,
                target,
                initial,
                inner,
                precondition,
                structural_max_steps,
                tolerance[0],
                tolerance[1],
                step_limit=max_steps,
            )
            matvec_count = auxiliary[0] + 2
        elif method == "minres":
            value, auxiliary = _minres_raw(
                selected_action,
                target,
                initial,
                inner,
                precondition,
                structural_max_steps,
                tolerance[0],
                tolerance[1],
                step_limit=max_steps,
            )
            matvec_count = auxiliary[0] + 2
        else:
            if method == "gmres":
                selected_method = (
                    plan.policy.method
                    if isinstance(plan.policy.method, GMRES)
                    else GMRES()
                )
                selected_preconditioner = lambda vector, _: precondition(
                    vector, jnp.asarray(0, dtype=jnp.int32)
                )
            else:
                selected_method = (
                    plan.policy.method
                    if isinstance(plan.policy.method, FGMRES)
                    else FGMRES()
                )
                selected_preconditioner = precondition
            restart = min(selected_method.restart, structural_max_steps)
            value, auxiliary = _fgmres_raw(
                selected_action,
                target,
                initial,
                inner,
                selected_preconditioner,
                structural_max_steps,
                restart,
                selected_method.stagnation_iterations,
                tolerance[0],
                tolerance[1],
                step_limit=max_steps,
                identity_preconditioner=preconditioner is None,
            )
            *auxiliary, executed_cycles = auxiliary
            matvec_count = auxiliary[0] + executed_cycles + 1
        return value, (
            *auxiliary,
            jnp.asarray(matvec_count, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    value, auxiliary = run(action, rhs)
    return (complement(value) if projected else value), auxiliary


def _pcg_raw(
    action,
    rhs,
    initial,
    inner,
    precondition,
    max_steps: int,
    relative: Array,
    absolute: Array,
    *,
    step_limit: Array | None = None,
):
    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    residual = rhs - action(initial)
    transformed = precondition(residual, jnp.asarray(0, dtype=jnp.int32))
    direction = transformed
    rho = jnp.real(inner(residual, transformed))
    rhs_norm = _norm(rhs, inner)
    threshold = absolute + relative * rhs_norm
    residual_norm = _norm(residual, inner)
    state = (
        initial,
        residual,
        transformed,
        direction,
        rho,
        jnp.asarray(0, dtype=jnp.int32),
        residual_norm > threshold,
        jnp.asarray(int(KrylovBreakdownStatus.NONE), dtype=jnp.int32),
    )
    epsilon = jnp.finfo(rhs.real.dtype).eps

    def step(index, current):
        x, r, z, p, rho_, iterations, active, breakdown = current

        def execute(operand):
            x_, r_, z_, p_, rho_i, _, _, _ = operand
            image = action(p_)
            denominator = jnp.real(inner(p_, image))
            invalid = (
                ~jnp.isfinite(denominator)
                | (
                    jnp.abs(denominator)
                    <= epsilon * _norm(p_, inner) * _norm(image, inner)
                )
                | (rho_i <= 0.0)
            )
            safe_denominator = jnp.where(invalid, 1.0, denominator)
            alpha = rho_i / safe_denominator
            candidate_x = x_ + alpha * p_
            candidate_r = r_ - alpha * image
            candidate_z = precondition(
                candidate_r, jnp.asarray(index + 1, dtype=jnp.int32)
            )
            next_rho = jnp.real(inner(candidate_r, candidate_z))
            beta = next_rho / jnp.where(rho_i == 0.0, 1.0, rho_i)
            candidate_p = candidate_z + beta * p_
            norm = _norm(candidate_r, inner)
            converged = norm <= threshold
            finite = jnp.all(jnp.isfinite(candidate_x)) & jnp.isfinite(norm)
            breakdown_i = jnp.where(
                finite,
                jnp.where(
                    invalid & ~converged,
                    int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                    jnp.where(
                        converged,
                        int(KrylovBreakdownStatus.HAPPY),
                        int(KrylovBreakdownStatus.NONE),
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            return (
                candidate_x,
                candidate_r,
                candidate_z,
                candidate_p,
                next_rho,
                jnp.asarray(index + 1, dtype=jnp.int32),
                finite & ~invalid & ~converged,
                breakdown_i,
            )

        return jax.lax.cond(
            active & (index < step_limit),
            execute,
            lambda operand: operand,
            current,
        )

    x, residual, _, _, _, iterations, _, breakdown = jax.lax.fori_loop(
        0, max_steps, step, state
    )
    residual_norm = _norm(rhs - action(x), inner)
    auxiliary = (
        iterations,
        residual_norm,
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        breakdown,
    )
    return x, auxiliary


def _minres_raw(
    action,
    rhs,
    initial,
    inner,
    precondition,
    max_steps: int,
    relative: Array,
    absolute: Array,
    *,
    step_limit: Array | None = None,
):
    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    residual = rhs - action(initial)
    y = precondition(residual, jnp.asarray(0, dtype=jnp.int32))
    beta_one_squared = jnp.real(inner(residual, y))
    beta_one = jnp.sqrt(jnp.maximum(beta_one_squared, 0.0))
    rhs_norm = _norm(rhs, inner)
    threshold = absolute + relative * rhs_norm
    state = (
        initial,
        residual,
        residual,
        y,
        jnp.asarray(0.0, dtype=rhs.real.dtype),
        beta_one,
        jnp.asarray(0.0, dtype=rhs.real.dtype),
        jnp.asarray(0.0, dtype=rhs.real.dtype),
        beta_one,
        jnp.asarray(-1.0, dtype=rhs.real.dtype),
        jnp.asarray(0.0, dtype=rhs.real.dtype),
        jnp.zeros_like(rhs),
        jnp.zeros_like(rhs),
        jnp.asarray(0, dtype=jnp.int32),
        beta_one > threshold,
        jnp.where(
            beta_one_squared >= 0.0,
            int(KrylovBreakdownStatus.NONE),
            int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
        ).astype(jnp.int32),
    )
    epsilon = jnp.finfo(rhs.real.dtype).eps

    def step(index, current):
        (
            x,
            r1,
            r2,
            y_,
            old_beta,
            beta,
            dbar,
            epsln,
            phibar,
            cosine,
            sine,
            w,
            w2,
            iterations,
            active,
            breakdown,
        ) = current

        def execute(operand):
            (
                x_,
                r1_,
                r2_,
                y_i,
                old_beta_i,
                beta_i,
                dbar_i,
                epsln_i,
                phibar_i,
                cosine_i,
                sine_i,
                w_i,
                w2_i,
                _,
                _,
                _,
            ) = operand
            safe_beta = jnp.where(beta_i > epsilon, beta_i, 1.0)
            v = y_i / safe_beta
            next_y = action(v)
            next_y = jax.lax.cond(
                index > 0,
                lambda value: (
                    value
                    - (beta_i / jnp.where(old_beta_i > epsilon, old_beta_i, 1.0)) * r1_
                ),
                lambda value: value,
                next_y,
            )
            alpha = jnp.real(inner(v, next_y))
            next_y = next_y - (alpha / safe_beta) * r2_
            next_r1 = r2_
            next_r2 = next_y
            preconditioned = precondition(
                next_r2, jnp.asarray(index + 1, dtype=jnp.int32)
            )
            beta_squared = jnp.real(inner(next_r2, preconditioned))
            next_beta = jnp.sqrt(jnp.maximum(beta_squared, 0.0))
            old_epsilon = epsln_i
            delta = cosine_i * dbar_i + sine_i * alpha
            gbar = sine_i * dbar_i - cosine_i * alpha
            next_epsln = sine_i * next_beta
            next_dbar = -cosine_i * next_beta
            gamma = jnp.sqrt(gbar * gbar + next_beta * next_beta)
            safe_gamma = jnp.maximum(gamma, epsilon)
            next_cosine = gbar / safe_gamma
            next_sine = next_beta / safe_gamma
            phi = next_cosine * phibar_i
            next_phibar = next_sine * phibar_i
            w1 = w2_i
            next_w2 = w_i
            next_w = (v - old_epsilon * w1 - delta * next_w2) / safe_gamma
            next_x = x_ + phi * next_w
            residual_estimate = jnp.abs(next_phibar)
            converged = residual_estimate <= threshold
            invalid = (
                (beta_squared < 0.0)
                | ~jnp.isfinite(residual_estimate)
                | jnp.any(~jnp.isfinite(next_x))
            )
            status = jnp.where(
                invalid,
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
                jnp.where(
                    converged,
                    int(KrylovBreakdownStatus.HAPPY),
                    int(KrylovBreakdownStatus.NONE),
                ),
            ).astype(jnp.int32)
            return (
                next_x,
                next_r1,
                next_r2,
                preconditioned,
                beta_i,
                next_beta,
                next_dbar,
                next_epsln,
                next_phibar,
                next_cosine,
                next_sine,
                next_w,
                next_w2,
                jnp.asarray(index + 1, dtype=jnp.int32),
                ~invalid & ~converged,
                status,
            )

        return jax.lax.cond(
            active & (index < step_limit),
            execute,
            lambda operand: operand,
            current,
        )

    result = jax.lax.fori_loop(0, max_steps, step, state)
    x, *_, iterations, _, breakdown = result
    residual_norm = _norm(rhs - action(x), inner)
    return x, (
        iterations,
        residual_norm,
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        breakdown,
    )


def _fgmres_raw(
    action,
    rhs,
    initial,
    inner,
    precondition,
    max_steps: int,
    restart: int,
    stagnation_iterations: int,
    relative: Array,
    absolute: Array,
    *,
    step_limit: Array | None = None,
    identity_preconditioner: bool = False,
):
    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    rhs_norm = _norm(rhs, inner)
    threshold = absolute + relative * rhs_norm
    residual = rhs - action(initial)
    residual_norm = _norm(residual, inner)
    finite = jnp.isfinite(residual_norm) & jnp.all(jnp.isfinite(initial))
    active = finite & (residual_norm > threshold)
    initial_breakdown = jnp.where(
        finite,
        int(KrylovBreakdownStatus.NONE),
        int(KrylovBreakdownStatus.NONFINITE_ACTION),
    ).astype(jnp.int32)
    initial_state = (
        initial,
        residual,
        residual_norm,
        jnp.asarray(0, dtype=jnp.int32),
        initial_breakdown,
        residual_norm,
        jnp.asarray(0, dtype=jnp.int32),
        active,
        jnp.asarray(0, dtype=jnp.int32),
    )
    cycles = (max_steps + restart - 1) // restart
    columns = jnp.arange(restart)

    def reduced_solve(hessenberg, reduced_rhs, steps):
        active_columns = columns < steps
        upper = jnp.where(
            active_columns[:, None] & active_columns[None, :],
            hessenberg[:-1],
            0,
        )
        upper = upper.at[columns, columns].add((~active_columns).astype(rhs.dtype))
        target = jnp.where(active_columns, reduced_rhs[:-1], 0)
        coefficients = jsp.linalg.solve_triangular(
            upper,
            target,
            lower=False,
        )
        return jnp.where(active_columns, coefficients, 0)

    def cycle_step(_, state):
        (
            x,
            current_residual,
            current_norm,
            iterations,
            breakdown,
            best_norm,
            stagnant_steps,
            cycle_active,
            executed_cycles,
        ) = state

        def execute_cycle(operand):
            (
                cycle_base,
                cycle_residual,
                cycle_norm,
                starting_iterations,
                _,
                previous_best,
                previous_stagnant,
                _,
                previous_cycles,
            ) = operand
            safe_norm = jnp.where(cycle_norm > 0.0, cycle_norm, 1.0)
            basis = jnp.zeros((restart + 1, rhs.size), dtype=rhs.dtype)
            basis = basis.at[0].set(cycle_residual / safe_norm)
            preconditioned_basis = jnp.zeros(
                (0 if identity_preconditioner else restart, rhs.size),
                dtype=rhs.dtype,
            )
            hessenberg = jnp.zeros((restart + 1, restart), dtype=rhs.dtype)
            cosines = jnp.zeros((restart,), dtype=rhs.real.dtype)
            sines = jnp.zeros((restart,), dtype=rhs.dtype)
            reduced_rhs = jnp.zeros((restart + 1,), dtype=rhs.dtype)
            reduced_rhs = reduced_rhs.at[0].set(cycle_norm)
            inner_state = (
                basis,
                preconditioned_basis,
                hessenberg,
                cosines,
                sines,
                reduced_rhs,
                starting_iterations,
                jnp.asarray(True),
                jnp.asarray(int(KrylovBreakdownStatus.NONE), dtype=jnp.int32),
                previous_best,
                previous_stagnant,
            )

            def arnoldi_step(local_index, current):
                (
                    basis_,
                    preconditioned_,
                    hessenberg_,
                    cosines_,
                    sines_,
                    reduced_rhs_,
                    iteration_,
                    inner_active,
                    inner_breakdown,
                    inner_best,
                    inner_stagnant,
                ) = current
                can_execute = (
                    inner_active & (iteration_ < step_limit) & (iteration_ < max_steps)
                )

                def execute_arnoldi(inner_operand):
                    (
                        basis_i,
                        preconditioned_i,
                        hessenberg_i,
                        cosines_i,
                        sines_i,
                        reduced_rhs_i,
                        iteration_i,
                        _,
                        _,
                        best_i,
                        stagnant_i,
                    ) = inner_operand
                    vector = basis_i[local_index]
                    transformed = (
                        vector
                        if identity_preconditioner
                        else precondition(vector, iteration_i)
                    )
                    image = action(transformed)
                    projection = jnp.zeros((restart,), dtype=rhs.dtype)

                    def orthogonalize(index, state):
                        remainder, coefficients = state
                        coefficient = inner(basis_i[index], remainder)
                        return (
                            remainder - coefficient * basis_i[index],
                            coefficients.at[index].add(coefficient),
                        )

                    orthogonal, projection = jax.lax.fori_loop(
                        0,
                        local_index + 1,
                        orthogonalize,
                        (image, projection),
                    )
                    orthogonal, projection = jax.lax.fori_loop(
                        0,
                        local_index + 1,
                        orthogonalize,
                        (orthogonal, projection),
                    )
                    next_norm = _norm(orthogonal, inner)
                    near_breakdown = next_norm <= jnp.sqrt(
                        jnp.finfo(rhs.real.dtype).eps
                    ) * jnp.maximum(_norm(image, inner), 1.0)
                    basis_i = basis_i.at[local_index + 1].set(
                        orthogonal / jnp.where(near_breakdown, 1.0, next_norm)
                    )
                    if not identity_preconditioner:
                        preconditioned_i = preconditioned_i.at[local_index].set(
                            transformed
                        )
                    column = hessenberg_i[:, local_index]
                    column = column.at[:-1].set(projection)
                    column = column.at[local_index + 1].set(next_norm)

                    def apply_previous_rotation(index, value):
                        upper = value[index]
                        lower = value[index + 1]
                        cosine = cosines_i[index]
                        sine = sines_i[index]
                        value = value.at[index].set(cosine * upper + sine * lower)
                        return value.at[index + 1].set(
                            -jnp.conj(sine) * upper + cosine * lower
                        )

                    column = jax.lax.fori_loop(
                        0,
                        local_index,
                        apply_previous_rotation,
                        column,
                    )
                    upper = column[local_index]
                    lower = column[local_index + 1]
                    upper_abs = jnp.abs(upper)
                    rotation_scale = jnp.hypot(upper_abs, jnp.abs(lower))
                    safe_scale = jnp.where(rotation_scale > 0.0, rotation_scale, 1.0)
                    safe_upper_abs = jnp.where(upper_abs > 0.0, upper_abs, 1.0)
                    phase = upper / safe_upper_abs
                    phase = jnp.where(
                        upper_abs > 0.0,
                        phase,
                        jnp.ones((), dtype=rhs.dtype),
                    )
                    cosine = jnp.where(
                        rotation_scale > 0.0,
                        upper_abs / safe_scale,
                        1.0,
                    )
                    sine = jnp.where(
                        rotation_scale > 0.0,
                        phase * jnp.conj(lower) / safe_scale,
                        jnp.zeros((), dtype=rhs.dtype),
                    )
                    column = column.at[local_index].set(cosine * upper + sine * lower)
                    column = column.at[local_index + 1].set(0)
                    hessenberg_i = hessenberg_i.at[:, local_index].set(column)
                    cosines_i = cosines_i.at[local_index].set(cosine)
                    sines_i = sines_i.at[local_index].set(sine)
                    reduced_upper = reduced_rhs_i[local_index]
                    reduced_lower = reduced_rhs_i[local_index + 1]
                    reduced_rhs_i = reduced_rhs_i.at[local_index].set(
                        cosine * reduced_upper + sine * reduced_lower
                    )
                    reduced_rhs_i = reduced_rhs_i.at[local_index + 1].set(
                        -jnp.conj(sine) * reduced_upper + cosine * reduced_lower
                    )
                    estimated_norm = jnp.abs(reduced_rhs_i[local_index + 1])
                    finite_step = (
                        jnp.isfinite(next_norm)
                        & jnp.isfinite(rotation_scale)
                        & jnp.isfinite(estimated_norm)
                        & jnp.all(jnp.isfinite(image))
                        & jnp.all(jnp.isfinite(transformed))
                        & jnp.all(jnp.isfinite(column))
                    )
                    converged = estimated_norm <= threshold
                    improvement = estimated_norm < best_i * (
                        1.0 - jnp.sqrt(jnp.finfo(rhs.real.dtype).eps)
                    )
                    next_best = jnp.minimum(best_i, estimated_norm)
                    next_stagnant = jnp.where(
                        improvement,
                        jnp.asarray(0, dtype=jnp.int32),
                        stagnant_i + 1,
                    )
                    stagnated = next_stagnant >= stagnation_iterations
                    step_breakdown = jnp.where(
                        finite_step,
                        jnp.where(
                            converged,
                            int(KrylovBreakdownStatus.HAPPY),
                            jnp.where(
                                stagnated,
                                int(KrylovBreakdownStatus.STAGNATION),
                                jnp.where(
                                    near_breakdown,
                                    int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                                    int(KrylovBreakdownStatus.NONE),
                                ),
                            ),
                        ),
                        int(KrylovBreakdownStatus.NONFINITE_ACTION),
                    ).astype(jnp.int32)
                    return (
                        basis_i,
                        preconditioned_i,
                        hessenberg_i,
                        cosines_i,
                        sines_i,
                        reduced_rhs_i,
                        iteration_i + 1,
                        finite_step & ~converged & ~near_breakdown & ~stagnated,
                        step_breakdown,
                        next_best,
                        next_stagnant,
                    )

                return jax.lax.cond(
                    can_execute,
                    execute_arnoldi,
                    lambda inner_operand: inner_operand,
                    current,
                )

            def inner_condition(current):
                iteration = current[6]
                return (
                    current[7]
                    & (iteration - starting_iterations < restart)
                    & (iteration < step_limit)
                    & (iteration < max_steps)
                )

            def inner_body(current):
                return arnoldi_step(
                    current[6] - starting_iterations,
                    current,
                )

            (
                basis,
                preconditioned_basis,
                hessenberg,
                _,
                _,
                reduced_rhs,
                final_iterations,
                _,
                inner_breakdown,
                inner_best,
                inner_stagnant,
            ) = jax.lax.while_loop(inner_condition, inner_body, inner_state)
            update_basis = basis[:-1] if identity_preconditioner else preconditioned_basis
            cycle_steps = final_iterations - starting_iterations
            coefficients = reduced_solve(
                hessenberg,
                reduced_rhs,
                cycle_steps,
            )
            candidate = cycle_base + jnp.sum(
                coefficients[:, None] * update_basis,
                axis=0,
            )
            candidate_residual = rhs - action(candidate)
            candidate_norm = _norm(candidate_residual, inner)
            finite_candidate = (
                jnp.isfinite(candidate_norm)
                & jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(coefficients))
            )
            converged = candidate_norm <= threshold
            next_best = jnp.minimum(inner_best, candidate_norm)
            next_stagnant = inner_stagnant
            stagnated = next_stagnant >= stagnation_iterations
            final_breakdown = jnp.where(
                finite_candidate,
                jnp.where(
                    converged,
                    int(KrylovBreakdownStatus.HAPPY),
                    jnp.where(
                        inner_breakdown != int(KrylovBreakdownStatus.NONE),
                        inner_breakdown,
                        jnp.where(
                            stagnated,
                            int(KrylovBreakdownStatus.STAGNATION),
                            int(KrylovBreakdownStatus.NONE),
                        ),
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            blocking_breakdown = (final_breakdown != int(KrylovBreakdownStatus.NONE)) & (
                final_breakdown != int(KrylovBreakdownStatus.HAPPY)
            )
            next_active = (
                finite_candidate
                & ~converged
                & ~blocking_breakdown
                & (final_iterations < step_limit)
                & (final_iterations < max_steps)
            )
            return (
                candidate,
                candidate_residual,
                candidate_norm,
                final_iterations,
                final_breakdown,
                next_best,
                next_stagnant,
                next_active,
                previous_cycles + 1,
            )

        return jax.lax.cond(
            cycle_active & (iterations < step_limit) & (iterations < max_steps),
            execute_cycle,
            lambda operand: operand,
            state,
        )

    def cycle_condition(state):
        return (
            state[7]
            & (state[3] < step_limit)
            & (state[3] < max_steps)
            & (state[8] < cycles)
        )

    (
        x,
        _,
        residual_norm,
        iterations,
        breakdown,
        _,
        _,
        _,
        executed_cycles,
    ) = jax.lax.while_loop(
        cycle_condition,
        lambda state: cycle_step(None, state),
        initial_state,
    )
    return x, (
        iterations,
        residual_norm,
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        breakdown,
        executed_cycles,
    )


def _least_squares_solve(
    problem,
    rhs,
    initial,
    plan,
    *,
    relative: Array,
    absolute: Array,
    max_steps: Array,
    structural_max_steps: int,
):
    action, adjoint, target_inner, source_inner, right = _least_squares_actions(
        problem, rhs
    )
    selected = plan.policy.method
    method = selected if isinstance(selected, GeneralizedLSMR) else GeneralizedLSMR()
    max_steps_ = structural_max_steps
    value, auxiliary = _lsmr_raw(
        action,
        adjoint,
        right,
        initial,
        source_inner,
        target_inner,
        max_steps_,
        relative,
        absolute,
        method.condition_limit,
        method.damping,
        step_limit=max_steps,
    )
    iterations = auxiliary[0]
    return value, (
        *auxiliary,
        jnp.asarray(iterations + 3, dtype=jnp.int32),
        jnp.asarray(iterations + 2, dtype=jnp.int32),
    )


def _lsmr_raw(
    action,
    adjoint,
    rhs,
    initial,
    source_inner,
    target_inner,
    max_steps: int,
    relative: Array,
    absolute: Array,
    condition_limit: float,
    damping: float,
    *,
    step_limit: Array | None = None,
):
    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    residual = _target_subtract(rhs, action(initial))
    beta = _target_norm(residual, target_inner)
    u_operator, u_regularizer = _target_scale(
        residual, 1.0 / jnp.where(beta > 0.0, beta, 1.0)
    )
    v = adjoint((u_operator, u_regularizer))
    alpha = _norm(v, source_inner)
    v = v / jnp.where(alpha > 0.0, alpha, 1.0)
    norm_b = _target_norm(rhs, target_inner)
    state = _LSMRState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        alpha=alpha,
        u_operator=u_operator,
        u_regularizer=u_regularizer,
        v=v,
        alphabar=alpha,
        rho=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        rhobar=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        zeta=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        sbar=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        cbar=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        zetabar=alpha * beta,
        hbar=jnp.zeros_like(initial),
        h=v,
        x=initial,
        betadd=beta,
        thetatilde=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        rhodold=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        betad=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        tautildeold=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        accumulated_residual=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        norm_a_squared=alpha * alpha,
        maximum_rbar=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        minimum_rbar=jnp.asarray(jnp.inf, dtype=rhs[0].real.dtype),
        normal_residual=alpha * beta,
        residual=beta,
        norm_a=alpha,
        condition=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        active=(norm_b > 0.0) & (alpha * beta > 0.0),
        breakdown=jnp.asarray(int(KrylovBreakdownStatus.NONE), dtype=jnp.int32),
    )

    def step(index, current):
        def execute(value):
            image_operator, image_regularizer = action(value.v)
            next_u_operator = image_operator - value.alpha * value.u_operator
            next_u_regularizer = image_regularizer - value.alpha * value.u_regularizer
            next_beta = _target_norm((next_u_operator, next_u_regularizer), target_inner)
            next_u_operator, next_u_regularizer = _target_scale(
                (next_u_operator, next_u_regularizer),
                1.0 / jnp.where(next_beta > 0.0, next_beta, 1.0),
            )
            next_v = adjoint((next_u_operator, next_u_regularizer)) - next_beta * value.v
            next_alpha = _norm(next_v, source_inner)
            next_v = next_v / jnp.where(next_alpha > 0.0, next_alpha, 1.0)
            chat, shat, alphahat = _symmetric_orthogonalization(
                value.alphabar, jnp.asarray(damping, dtype=value.alphabar.dtype)
            )
            cosine, sine, rho = _symmetric_orthogonalization(alphahat, next_beta)
            theta_new = sine * next_alpha
            alpha_bar = cosine * next_alpha
            theta_bar = value.sbar * rho
            rho_temp = value.cbar * rho
            cbar, sbar, rho_bar = _symmetric_orthogonalization(rho_temp, theta_new)
            zeta = cbar * value.zetabar
            zeta_bar = -sbar * value.zetabar
            safe_denominator = jnp.where(
                value.rho * value.rhobar == 0.0,
                1.0,
                value.rho * value.rhobar,
            )
            hbar = value.h - value.hbar * (theta_bar * rho / safe_denominator)
            x = (
                value.x
                + (zeta / jnp.where(rho * rho_bar == 0.0, 1.0, rho * rho_bar)) * hbar
            )
            h = next_v - value.h * (theta_new / jnp.where(rho == 0.0, 1.0, rho))
            beta_acute = chat * value.betadd
            beta_check = -shat * value.betadd
            beta_hat = cosine * beta_acute
            beta_dd = -sine * beta_acute
            ctilde, stilde, rho_tilde = _symmetric_orthogonalization(
                value.rhodold, theta_bar
            )
            theta_tilde = stilde * rho_bar
            rho_d = ctilde * rho_bar
            beta_d = -stilde * value.betad + ctilde * beta_hat
            tau_tilde = (value.zeta - value.thetatilde * value.tautildeold) / jnp.where(
                rho_tilde == 0.0, 1.0, rho_tilde
            )
            tau_d = (zeta - theta_tilde * tau_tilde) / jnp.where(rho_d == 0.0, 1.0, rho_d)
            accumulated = value.accumulated_residual + beta_check * beta_check
            residual_norm = jnp.sqrt(
                accumulated + (beta_d - tau_d) ** 2 + beta_dd * beta_dd
            )
            norm_a_squared = value.norm_a_squared + next_beta**2
            norm_a = jnp.sqrt(norm_a_squared)
            norm_a_squared = norm_a_squared + next_alpha**2
            maximum = jnp.maximum(value.maximum_rbar, value.rhobar)
            minimum = jnp.minimum(value.minimum_rbar, jnp.abs(value.rhobar))
            condition = jnp.maximum(maximum, jnp.abs(rho_temp)) / jnp.maximum(
                jnp.minimum(minimum, jnp.abs(rho_temp)),
                jnp.finfo(rho_temp.dtype).tiny,
            )
            normal_residual = jnp.abs(zeta_bar)
            iteration = value.iteration + 1
            converged = (residual_norm <= absolute + relative * norm_b) | (
                normal_residual
                <= absolute + relative * norm_a * jnp.maximum(residual_norm, 1.0)
            )
            condition_limited = condition >= condition_limit
            finite = (
                jnp.all(jnp.isfinite(x))
                & jnp.isfinite(residual_norm)
                & jnp.isfinite(normal_residual)
            )
            recurrence_breakdown = (rho == 0.0) | (rho_bar == 0.0)
            breakdown = jnp.where(
                finite,
                jnp.where(
                    converged,
                    int(KrylovBreakdownStatus.HAPPY),
                    jnp.where(
                        recurrence_breakdown,
                        int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                        int(KrylovBreakdownStatus.NONE),
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            return _LSMRState(
                iteration,
                next_alpha,
                next_u_operator,
                next_u_regularizer,
                next_v,
                alpha_bar,
                rho,
                rho_bar,
                zeta,
                sbar,
                cbar,
                zeta_bar,
                hbar,
                h,
                x,
                beta_dd,
                theta_tilde,
                rho_d,
                beta_d,
                tau_tilde,
                accumulated,
                norm_a_squared,
                maximum,
                minimum,
                normal_residual,
                residual_norm,
                norm_a,
                condition,
                finite & ~converged & ~condition_limited & ~recurrence_breakdown,
                breakdown,
            )

        return jax.lax.cond(
            current.active & (index < step_limit),
            execute,
            lambda value: value,
            current,
        )

    state = jax.lax.fori_loop(0, max_steps, step, state)
    true_residual = _target_norm(_target_subtract(rhs, action(state.x)), target_inner)
    true_normal = _norm(adjoint(_target_subtract(action(state.x), rhs)), source_inner)
    return state.x, (
        state.iteration,
        true_residual,
        true_normal,
        state.condition,
        state.breakdown,
    )


def _least_squares_actions(problem, rhs):
    operator = problem.operator
    regularizer = (
        problem.regularizer if isinstance(problem, LeastSquaresProblem) else None
    )
    weights = None
    if isinstance(problem, LeastSquaresProblem) and problem.weights is not None:
        weights = jnp.asarray(problem.weights).reshape((-1,))
        if weights.shape != (operator.target.size,):
            raise ValueError(
                "GeneralizedLSMR weights must have one target coordinate entry."
            )

    def action(vector):
        primary = _action_coordinates(operator, vector)
        secondary = (
            jnp.zeros((0,), dtype=primary.dtype)
            if regularizer is None
            else _action_coordinates(regularizer, vector)
        )
        return primary, secondary

    def adjoint(value):
        primary, secondary = value
        weighted = primary if weights is None else weights * primary
        result = _adjoint_coordinates(operator, weighted)
        if regularizer is not None:
            result = result + _adjoint_coordinates(regularizer, secondary)
        return result

    def target_inner(left, right):
        left_primary, left_secondary = left
        right_primary, right_secondary = right
        weighted_right = right_primary if weights is None else weights * right_primary
        value = _space_inner(operator.target, left_primary, weighted_right)
        if regularizer is not None:
            value = value + _space_inner(
                regularizer.target, left_secondary, right_secondary
            )
        return value

    source_inner = lambda left, right: _space_inner(operator.source, left, right)
    right = (
        rhs,
        jnp.zeros(
            (0 if regularizer is None else regularizer.target.size,), dtype=rhs.dtype
        ),
    )
    return action, adjoint, target_inner, source_inner, right


def _preconditioner_action(preconditioner, space):
    if preconditioner is None:
        return lambda vector, iteration: vector

    def apply(vector, iteration):
        return space.flatten(
            preconditioner.apply(space.unflatten(vector), iteration=iteration)
        )

    return apply


def _action_coordinates(operator, vector):
    return operator.target.flatten(operator.mv(operator.source.unflatten(vector)))


def _adjoint_coordinates(operator, vector):
    return operator.source.flatten(operator.adjoint_mv(operator.target.unflatten(vector)))


def _space_inner(space, left, right):
    return space.inner(space.unflatten(left), space.unflatten(right))


def _space_norm(space, vector):
    return _norm(vector, lambda left, right: _space_inner(space, left, right))


def _norm(vector, inner):
    return jnp.sqrt(jnp.maximum(jnp.real(inner(vector, vector)), 0.0))


def _target_norm(value, inner):
    return jnp.sqrt(jnp.maximum(jnp.real(inner(value, value)), 0.0))


def _target_subtract(left, right):
    return left[0] - right[0], left[1] - right[1]


def _target_scale(value, scalar):
    return scalar * value[0], scalar * value[1]


def _symmetric_orthogonalization(left, right):
    radius = jnp.hypot(left, right)
    safe = jnp.where(radius == 0.0, 1.0, radius)
    return left / safe, right / safe, radius


__all__ = [
    "NativeKrylovBackendOutput",
    "NativeKrylovState",
    "prepare_native_krylov",
    "solve_native_krylov",
]
