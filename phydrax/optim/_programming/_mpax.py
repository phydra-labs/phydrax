#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jax.experimental import sparse as jsparse
from jaxtyping import Array

from ...backends import (
    import_backend_module,
    mpax_availability,
    MPAXPlan,
    prepare_mpax,
)
from ._clarabel import _audit_result
from ._cones import NonnegativeCone, ProductCone, ZeroCone
from ._policy import ConvexSolvePolicy, MPAXr2HPDHG, MPAXraPDHG
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
    ConicProgram,
    LinearProgram,
)
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
        representation=source.representation,
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
    slack = problem.inequality_rhs - oe.contract(
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
                use_sparse_matrix=plan.representation == "sparse",
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
                    use_sparse_matrix=plan.representation == "sparse",
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


def _storage_bcoo(operator, selected_rows, /):
    storage = operator.sparse_storage()
    if storage.batch_shape:
        raise ValueError("Sparse MPAX ConicProgram currently requires unbatched values.")
    indptr = np.asarray(storage.indptr)
    rows = np.repeat(np.arange(storage.shape[0]), np.diff(indptr))
    selected = np.asarray(selected_rows, dtype=np.int64)
    row_map = np.full((storage.shape[0],), -1, dtype=np.int32)
    row_map[selected] = np.arange(selected.size, dtype=np.int32)
    keep = row_map[rows] >= 0
    indices = jnp.stack(
        (
            jnp.asarray(row_map[rows[keep]], dtype=jnp.int32),
            storage.indices[keep].astype(jnp.int32),
        ),
        axis=-1,
    )
    return jsparse.BCOO(
        (storage.values[keep], indices),
        shape=(selected.size, storage.shape[1]),
        indices_sorted=False,
        unique_indices=storage.canonical,
    )


def _quadratic_bcoo(program):
    if program.quadratic is None:
        indices = jnp.empty((0, 2), dtype=jnp.int32)
        return jsparse.BCOO(
            (jnp.empty((0,), dtype=program.linear.dtype), indices),
            shape=(program.num_variables, program.num_variables),
        )
    if program.quadratic_is_sparse:
        return _storage_bcoo(program.quadratic, np.arange(program.num_variables))
    return jsparse.BCOO.fromdense(program.quadratic)


def solve_mpax_conic_program(
    program: ConicProgram,
    policy: ConvexSolvePolicy,
    /,
    *,
    prepared_backend=None,
) -> ConvexProgramResult:
    """Solve an unbatched sparse LP/QP with zero/nonnegative cone rows."""
    if program.batch_shape or not program.constraint_is_sparse:
        raise ValueError("Sparse MPAX ConicProgram must be unbatched and sparse.")
    blocks = (
        program.cone.cones if isinstance(program.cone, ProductCone) else (program.cone,)
    )
    slices = (
        program.cone.slices
        if isinstance(program.cone, ProductCone)
        else (slice(0, program.num_constraints),)
    )
    equality_rows = (
        np.concatenate(
            [
                np.arange(item.start, item.stop)
                for block, item in zip(blocks, slices, strict=True)
                if isinstance(block, ZeroCone)
            ]
        )
        if any(isinstance(block, ZeroCone) for block in blocks)
        else np.empty((0,), dtype=np.int64)
    )
    inequality_rows = (
        np.concatenate(
            [
                np.arange(item.start, item.stop)
                for block, item in zip(blocks, slices, strict=True)
                if isinstance(block, NonnegativeCone)
            ]
        )
        if any(isinstance(block, NonnegativeCone) for block in blocks)
        else np.empty((0,), dtype=np.int64)
    )
    plan = _provider_plan(policy)
    module = import_backend_module(
        mpax_availability(), "optimization.linear-program", "mpax"
    )
    status_module = import_backend_module(
        mpax_availability(), "optimization.linear-program", "mpax.utils"
    )
    prepared = prepare_mpax(plan) if prepared_backend is None else prepared_backend
    a = _storage_bcoo(program.constraint_matrix, equality_rows)
    g = _storage_bcoo(program.constraint_matrix, inequality_rows)
    b = program.constraint_rhs[jnp.asarray(equality_rows)]
    h = program.constraint_rhs[jnp.asarray(inequality_rows)]
    q = _quadratic_bcoo(program)
    if policy.regularization:
        diagonal = jnp.arange(program.num_variables, dtype=jnp.int32)
        regularizer = jsparse.BCOO(
            (
                jnp.full(
                    (program.num_variables,),
                    policy.regularization,
                    dtype=program.linear.dtype,
                ),
                jnp.stack((diagonal, diagonal), axis=-1),
            ),
            shape=(program.num_variables, program.num_variables),
            indices_sorted=True,
            unique_indices=True,
        )
        q = q + regularizer
    linear_execution = program.quadratic is None and policy.regularization == 0.0
    model = (
        module.create_lp(
            program.linear,
            a,
            b,
            -g,
            -h,
            program.lower_bounds,
            program.upper_bounds,
            use_sparse_matrix=True,
        )
        if linear_execution
        else replace(
            module.create_qp(
                q,
                program.linear,
                a,
                b,
                -g,
                -h,
                program.lower_bounds,
                program.upper_bounds,
                use_sparse_matrix=True,
            ),
            is_lp=False,
        )
    )
    output = prepared.solver.optimize(model)
    primal = output.primal_solution
    provider_dual = output.dual_solution
    equality_dual = -provider_dual[: equality_rows.size]
    inequality_dual = provider_dual[equality_rows.size :]
    cone_dual = jnp.zeros((program.num_constraints,), dtype=primal.dtype)
    cone_dual = cone_dual.at[jnp.asarray(equality_rows)].set(equality_dual)
    cone_dual = cone_dual.at[jnp.asarray(inequality_rows)].set(inequality_dual)
    slack = program.constraint_rhs - _conic_matrix_mv(program.constraint_matrix, primal)
    gradient = (
        _conic_quadratic_mv(program.quadratic, primal)
        + program.linear
        + _conic_matrix_transpose_mv(program.constraint_matrix, cone_dual)
    )
    tolerance = max(policy.termination.absolute, 1e-8)
    lower_active = jnp.isfinite(program.lower_bounds) & (
        primal - program.lower_bounds <= tolerance
    )
    upper_active = jnp.isfinite(program.upper_bounds) & (
        program.upper_bounds - primal <= tolerance
    )
    lower_dual = jnp.where(lower_active, jnp.maximum(gradient, 0.0), 0.0)
    upper_dual = jnp.where(upper_active, jnp.maximum(-gradient, 0.0), 0.0)
    backend_converged = output.termination_status == int(
        status_module.TerminationStatus.OPTIMAL
    )
    return _audit_result(
        program,
        primal,
        slack,
        cone_dual,
        lower_dual,
        upper_dual,
        backend_converged,
        output.iteration_count,
        policy,
        prepared.backend_version,
        backend="mpax",
    )


__all__ = [
    "prepare_mpax_policy",
    "solve_mpax_conic_program",
    "solve_mpax_program",
]
