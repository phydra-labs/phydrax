#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...backends import (
    import_backend_module,
    mpax_availability,
    MPAXPlan,
    prepare_mpax,
)
from ._policy import ConvexSolvePolicy, MPAXr2HPDHG, MPAXraPDHG
from ._problem import LinearProgram
from ._quadratic import (
    _diagnostics,
    ConvexProgramResult,
    QuadraticProgram,
)
from ._types import ConvexWarmStart


def _flat_count(batch_shape: tuple[int, ...], /) -> int:
    return int(np.prod(batch_shape)) if batch_shape else 1


def _audit_problem(program: LinearProgram | QuadraticProgram, /) -> QuadraticProgram:
    return (
        program.as_quadratic_program() if isinstance(program, LinearProgram) else program
    )


def _provider_plan(policy: ConvexSolvePolicy, /) -> MPAXPlan:
    method = policy.method
    if isinstance(method, MPAXraPDHG):
        algorithm = "rapdhg"
    elif isinstance(method, MPAXr2HPDHG):
        algorithm = "r2hpdhg"
    else:
        raise TypeError("MPAX adapter requires an MPAX method.")
    source = method.plan
    return MPAXPlan(
        algorithm,
        eps_abs=max(policy.termination.absolute, 1e-12),
        eps_rel=max(policy.termination.relative, 1e-12),
        eps_primal_infeasible=max(policy.termination.primal_infeasible, 1e-12),
        eps_dual_infeasible=max(policy.termination.dual_infeasible, 1e-12),
        iteration_limit=policy.termination.maximum_steps,
        warm_start=source.warm_start,
        feasibility_polishing=source.feasibility_polishing,
        unroll=source.unroll,
    )


def _flatten_provider_data(problem: QuadraticProgram, /):
    count = _flat_count(problem.batch_shape)
    quadratic = problem.quadratic.reshape(
        (count, problem.num_variables, problem.num_variables)
    )
    linear = problem.linear.reshape((count, problem.num_variables))
    return (
        quadratic,
        linear,
        problem.equality_matrix.reshape(
            (count, problem.num_equalities, problem.num_variables)
        ),
        problem.equality_rhs.reshape((count, problem.num_equalities)),
        problem.inequality_matrix.reshape(
            (count, problem.num_inequalities, problem.num_variables)
        ),
        problem.inequality_rhs.reshape((count, problem.num_inequalities)),
        jnp.full_like(linear, -jnp.inf),
        jnp.full_like(linear, jnp.inf),
    )


def _flatten_warm_start(
    problem: QuadraticProgram,
    warm_start: ConvexWarmStart | None,
    /,
) -> tuple[Array | None, Array | None]:
    if warm_start is None:
        return None, None
    if warm_start.structure_id != problem.structure_id:
        raise ValueError("Warm start does not match the MPAX program structure.")
    expected_primal = problem.batch_shape + (problem.num_variables,)
    expected_equality = problem.batch_shape + (problem.num_user_equalities,)
    expected_inequality = problem.batch_shape + (problem.num_user_inequalities,)
    if tuple(warm_start.primal.shape) != expected_primal:
        raise ValueError(f"Warm-start primal must have shape {expected_primal}.")
    if tuple(warm_start.equality_dual.shape) != expected_equality:
        raise ValueError(f"Warm-start equality dual must have shape {expected_equality}.")
    if tuple(warm_start.inequality_dual.shape) != expected_inequality:
        raise ValueError(
            f"Warm-start inequality dual must have shape {expected_inequality}."
        )
    if tuple(warm_start.inequality_slack.shape) != expected_inequality:
        raise ValueError(
            f"Warm-start inequality slack must have shape {expected_inequality}."
        )
    if tuple(warm_start.lower_bound_dual.shape) != expected_primal:
        raise ValueError(
            f"Warm-start lower-bound dual must have shape {expected_primal}."
        )
    if tuple(warm_start.upper_bound_dual.shape) != expected_primal:
        raise ValueError(
            f"Warm-start upper-bound dual must have shape {expected_primal}."
        )

    fixed_indices = jnp.asarray(problem.fixed_bound_indices, dtype=jnp.int32)
    lower_indices = jnp.asarray(problem.lower_bound_indices, dtype=jnp.int32)
    upper_indices = jnp.asarray(problem.upper_bound_indices, dtype=jnp.int32)
    fixed_dual = jnp.take(
        warm_start.upper_bound_dual - warm_start.lower_bound_dual,
        fixed_indices,
        axis=-1,
    )
    full_equality_dual = jnp.concatenate((warm_start.equality_dual, fixed_dual), axis=-1)
    full_inequality_dual = jnp.concatenate(
        (
            warm_start.inequality_dual,
            jnp.take(warm_start.lower_bound_dual, lower_indices, axis=-1),
            jnp.take(warm_start.upper_bound_dual, upper_indices, axis=-1),
        ),
        axis=-1,
    )
    count = _flat_count(problem.batch_shape)
    primal = warm_start.primal.reshape((count, problem.num_variables))
    dual = jnp.concatenate((-full_equality_dual, full_inequality_dual), axis=-1).reshape(
        (count, problem.num_equalities + problem.num_inequalities)
    )
    return primal, dual


def _expanded_duals_and_slacks(
    problem: QuadraticProgram,
    primal: Array,
    provider_dual: Array,
    /,
) -> tuple[Array, Array, Array]:
    equality_dual = -provider_dual[..., : problem.num_equalities]
    inequality_dual = provider_dual[
        ..., problem.num_equalities : problem.num_equalities + problem.num_inequalities
    ]
    slack = problem.inequality_rhs - jnp.einsum(
        "...ij,...j->...i",
        problem.inequality_matrix,
        primal,
    )
    return slack, inequality_dual, equality_dual


def prepare_mpax_policy(policy: ConvexSolvePolicy, /):
    """Prepare the MPAX algorithm selected by one convex solve policy."""

    return prepare_mpax(_provider_plan(policy))


def solve_mpax_program(
    program: LinearProgram | QuadraticProgram,
    policy: ConvexSolvePolicy,
    /,
    *,
    warm_start: ConvexWarmStart | None = None,
    prepared_backend=None,
) -> ConvexProgramResult:
    """Solve one dense/batched LP or QP through MPAX and audit original data."""

    problem = _audit_problem(program)
    method = policy.method
    if isinstance(method, MPAXr2HPDHG) and not isinstance(program, LinearProgram):
        raise ValueError("MPAXr2HPDHG supports LinearProgram only.")
    if isinstance(method, MPAXr2HPDHG) and policy.regularization != 0.0:
        raise ValueError("MPAXr2HPDHG requires zero quadratic regularization.")
    plan = _provider_plan(policy)
    if warm_start is not None and not plan.warm_start:
        raise ValueError("MPAX warm-start data require method warm_start=True.")
    module = import_backend_module(
        mpax_availability(),
        "optimization.linear-program"
        if isinstance(program, LinearProgram)
        else "optimization.quadratic-program",
        "mpax",
    )
    status_module = import_backend_module(
        mpax_availability(),
        "optimization.linear-program",
        "mpax.utils",
    )
    prepared = prepare_mpax(plan) if prepared_backend is None else prepared_backend
    if prepared.plan.plan_id != plan.plan_id:
        raise ValueError("Prepared MPAX state does not match the solve policy.")
    (
        quadratic,
        linear,
        equality,
        equality_rhs,
        inequality,
        inequality_rhs,
        lower,
        upper,
    ) = _flatten_provider_data(problem)
    if (
        warm_start is not None
        and isinstance(program, LinearProgram)
        and warm_start.structure_id == program.structure_id
    ):
        warm_start = ConvexWarmStart(
            primal=warm_start.primal,
            equality_dual=warm_start.equality_dual,
            inequality_dual=warm_start.inequality_dual,
            inequality_slack=warm_start.inequality_slack,
            lower_bound_dual=warm_start.lower_bound_dual,
            upper_bound_dual=warm_start.upper_bound_dual,
            structure_id=problem.structure_id,
        )
    warm_primal, warm_dual = _flatten_warm_start(problem, warm_start)
    regularized = quadratic + policy.regularization * jnp.eye(
        problem.num_variables, dtype=quadratic.dtype
    )

    linear_execution = isinstance(program, LinearProgram) and policy.regularization == 0.0
    requires_lift = problem.num_equalities + problem.num_inequalities == 0

    def solve_one(q, c, a, b, g, h, lo, hi, initial_x, initial_y):
        if requires_lift:
            q = jnp.pad(q, ((0, 1), (0, 1)))
            c = jnp.pad(c, (0, 1))
            a = (
                jnp.zeros((1, problem.num_variables + 1), dtype=a.dtype)
                .at[0, -1]
                .set(1.0)
            )
            b = jnp.zeros((1,), dtype=b.dtype)
            g = jnp.empty((0, problem.num_variables + 1), dtype=g.dtype)
            lo = jnp.full((problem.num_variables + 1,), -jnp.inf, dtype=lo.dtype)
            hi = jnp.full((problem.num_variables + 1,), jnp.inf, dtype=hi.dtype)
            initial_x = jnp.pad(initial_x, (0, 1))
            initial_y = jnp.zeros((1,), dtype=initial_y.dtype)
        if linear_execution:
            model = module.create_lp(
                c,
                a,
                b,
                -g,
                -h,
                lo,
                hi,
                use_sparse_matrix=False,
            )
        else:
            model = replace(
                module.create_qp(
                    q,
                    c,
                    a,
                    b,
                    -g,
                    -h,
                    lo,
                    hi,
                    use_sparse_matrix=False,
                ),
                is_lp=False,
            )
        output = prepared.solver.optimize(
            model,
            initial_primal_solution=initial_x,
            initial_dual_solution=initial_y,
        )
        primal_solution = output.primal_solution
        dual_solution = output.dual_solution
        if requires_lift:
            primal_solution = primal_solution[:-1]
            dual_solution = dual_solution[:-1]
        return (
            primal_solution,
            dual_solution,
            output.termination_status,
            output.iteration_count,
        )

    count = linear.shape[0]
    initial_primal = jnp.zeros_like(linear) if warm_primal is None else warm_primal
    initial_dual = (
        jnp.zeros(
            (count, problem.num_equalities + problem.num_inequalities),
            dtype=linear.dtype,
        )
        if warm_dual is None
        else warm_dual
    )
    primal, provider_dual, provider_status, iterations = jax.vmap(solve_one)(
        regularized,
        linear,
        equality,
        equality_rhs,
        inequality,
        inequality_rhs,
        lower,
        upper,
        initial_primal,
        initial_dual,
    )
    primal = primal.reshape(problem.batch_shape + (problem.num_variables,))
    provider_dual = provider_dual.reshape(
        problem.batch_shape + (problem.num_equalities + problem.num_inequalities,)
    )
    slack, inequality_dual, equality_dual = _expanded_duals_and_slacks(
        problem, primal, provider_dual
    )
    optimal_code = int(status_module.TerminationStatus.OPTIMAL)
    backend_converged = provider_status.reshape(problem.batch_shape) == optimal_code
    return _diagnostics(
        problem,
        primal,
        slack,
        inequality_dual,
        equality_dual,
        backend_converged,
        iterations.reshape(problem.batch_shape),
        method=method.method_id,
        backend=f"mpax-{prepared.backend_version}",
        tolerance=policy.termination.absolute,
        max_iterations=policy.termination.maximum_steps,
        relative_tolerance=policy.termination.relative,
        regularization=policy.regularization,
        policy_id=policy.policy_id,
        primal_infeasible_tolerance=policy.termination.primal_infeasible,
        dual_infeasible_tolerance=policy.termination.dual_infeasible,
    )


__all__ = ["prepare_mpax_policy", "solve_mpax_program"]
