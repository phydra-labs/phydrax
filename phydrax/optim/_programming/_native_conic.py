#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from phydrax._strict import StrictModule

from ..._bounds import Bounds
from ._barrier import cone_barrier_oracle, ConeBarrierOracle
from ._clarabel import _audit_result
from ._cones import NonnegativeCone, ProductCone, ZeroCone
from ._native_hsd import solve_homogeneous_conic
from ._native_hsd_factorized import FactorizedNewtonPlan, prepare_factorized_newton
from ._policy import ConvexSolvePolicy, NativeHomogeneousConic
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
    ConicProgram,
)
from ._quadratic import ConvexProgramResult
from ._types import ConvexProgramStatus, ConvexWarmStart


class _NativeConicState(StrictModule):
    primal: Array
    dual: Array
    active: Array
    iterations: Array


def _maximum_abs(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value), axis=-1, initial=0.0)


def _initial_primal(
    program: ConicProgram, warm_start: ConvexWarmStart | None, /
) -> Array:
    if warm_start is not None:
        primal = jnp.asarray(warm_start.primal, dtype=program.linear.dtype)
        if primal.shape != program.batch_shape + (program.num_variables,):
            raise ValueError("Native conic warm-start primal has the wrong shape.")
        return primal
    lower = jnp.where(jnp.isfinite(program.lower_bounds), program.lower_bounds, -1.0)
    upper = jnp.where(jnp.isfinite(program.upper_bounds), program.upper_bounds, 1.0)
    return jnp.minimum(jnp.maximum(jnp.zeros_like(program.linear), lower), upper)


def _initial_dual(program: ConicProgram, warm_start: ConvexWarmStart | None, /) -> Array:
    if warm_start is None:
        return jnp.zeros(
            program.batch_shape + (program.num_constraints,), dtype=program.linear.dtype
        )
    dual = jnp.asarray(warm_start.inequality_dual, dtype=program.linear.dtype)
    if dual.shape != program.batch_shape + (program.num_constraints,):
        raise ValueError(
            "Native general-conic warm starts store cone duals in inequality_dual."
        )
    residual = dual - program.cone.project_dual(dual)
    return eqx.error_if(
        dual,
        jnp.any(jnp.abs(residual) > 0.0),
        "Native conic warm-start dual is outside the declared dual cone.",
    )


def _augment_dense_bounds(
    program: ConicProgram,
) -> tuple[ConicProgram, Array, Array, Array]:
    fixed = jnp.asarray(program.fixed_bound_indices, dtype=jnp.int32)
    lower = jnp.asarray(program.lower_bound_indices, dtype=jnp.int32)
    upper = jnp.asarray(program.upper_bound_indices, dtype=jnp.int32)
    identity = jnp.eye(program.num_variables, dtype=program.linear.dtype)
    # Bound lowering is only selected for dense constraint matrices.
    dense_matrix = cast("Array", program.constraint_matrix)
    matrix = jnp.concatenate(
        (
            dense_matrix,
            identity[jnp.asarray(fixed)],
            -identity[jnp.asarray(lower)],
            identity[jnp.asarray(upper)],
        ),
        axis=0,
    )
    rhs = jnp.concatenate(
        (
            program.constraint_rhs,
            program.lower_bounds[jnp.asarray(fixed)],
            -program.lower_bounds[jnp.asarray(lower)],
            program.upper_bounds[jnp.asarray(upper)],
        )
    )
    blocks = (
        program.cone.cones if isinstance(program.cone, ProductCone) else (program.cone,)
    )
    if fixed.size:
        blocks = (*blocks, ZeroCone(fixed.size))
    if lower.size:
        blocks = (*blocks, NonnegativeCone(lower.size))
    if upper.size:
        blocks = (*blocks, NonnegativeCone(upper.size))
    augmented = ConicProgram(
        program.quadratic,
        program.linear,
        matrix,
        rhs,
        ProductCone(blocks),
        bounds=Bounds(),
        problem_id=f"{program.problem_id}:native-bound-lowering",
        convexity_evidence=program.convexity_evidence,
    )
    return augmented, fixed, lower, upper


class PreparedNativeConic(StrictModule):
    """Coefficient-independent native conic state for one program structure.

    ``newton`` is the reduced-KKT symbolic plan, present exactly when the
    homogeneous embedding runs on a sparse program with the factorized route.
    """

    barrier: ConeBarrierOracle
    newton: FactorizedNewtonPlan | None


def _uses_homogeneous_embedding(program: ConicProgram, /) -> bool:
    has_bounds = bool(
        program.fixed_bound_indices
        or program.lower_bound_indices
        or program.upper_bound_indices
    )
    return not program.batch_shape and (
        not program.constraint_is_sparse or not has_bounds
    )


def prepare_native_conic(
    program: ConicProgram, policy: ConvexSolvePolicy, /
) -> PreparedNativeConic:
    """Prepare the barrier oracle and, when selected, the reduced-KKT analysis.

    Host-only symbolic preparation over concrete topology, reused by every
    numeric binding of the same program structure. A reduced pattern exceeding
    the declared factorization budget refuses here.
    """
    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    method = policy.method
    if not isinstance(method, NativeHomogeneousConic):
        raise TypeError("policy method must be NativeHomogeneousConic.")
    sparse = program.constraint_is_sparse or program.quadratic_is_sparse
    newton = (
        prepare_factorized_newton(program, method.factorization)
        if sparse
        and method.newton == "factorized"
        and _uses_homogeneous_embedding(program)
        else None
    )
    return PreparedNativeConic(cone_barrier_oracle(program.cone), newton)


@eqx.filter_jit
def solve_native_conic_program(
    program: ConicProgram,
    policy: ConvexSolvePolicy,
    /,
    *,
    prepared: PreparedNativeConic | None = None,
    warm_start: ConvexWarmStart | None = None,
) -> ConvexProgramResult:
    """Execute a fixed-capacity JAX-native primal-dual conic iteration.

    The independent original-coordinate audit remains authoritative for every
    optimality or ray status; iteration residuals are never trusted as certificates.
    Sparse factorized execution requires ``prepared`` from `prepare_native_conic`,
    because its symbolic analysis reads concrete topology outside tracing.
    """
    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    method = policy.method
    if not isinstance(method, NativeHomogeneousConic):
        raise TypeError("policy method must be NativeHomogeneousConic.")
    if prepared is not None and not isinstance(prepared, PreparedNativeConic):
        raise TypeError("prepared must be a PreparedNativeConic or None.")
    barrier_ = cone_barrier_oracle(program.cone) if prepared is None else prepared.barrier
    if barrier_.cone.cone_id != program.cone.cone_id:
        raise ValueError("Prepared barrier oracle does not match the program cone.")
    has_bounds = bool(
        program.fixed_bound_indices
        or program.lower_bound_indices
        or program.upper_bound_indices
    )
    if _uses_homogeneous_embedding(program):
        if has_bounds:
            embedded, fixed, lower, upper = _augment_dense_bounds(program)
            embedded_barrier = cone_barrier_oracle(embedded.cone)
        else:
            embedded, fixed, lower, upper = (
                program,
                jnp.empty((0,), dtype=jnp.int32),
                jnp.empty((0,), dtype=jnp.int32),
                jnp.empty((0,), dtype=jnp.int32),
            )
            embedded_barrier = barrier_
        homogeneous = solve_homogeneous_conic(
            embedded,
            embedded_barrier,
            maximum_steps=policy.termination.maximum_steps,
            tolerance=policy.termination.absolute,
            policy=policy,
            newton=None if prepared is None else prepared.newton,
        )
        original = program.num_constraints
        cone_dual = homogeneous.dual[:original]
        cone_slack = homogeneous.slack[:original]
        lower_dual = jnp.zeros_like(homogeneous.primal)
        upper_dual = jnp.zeros_like(homogeneous.primal)
        cursor = original
        if fixed.size:
            signed = homogeneous.dual[cursor : cursor + fixed.size]
            lower_dual = lower_dual.at[jnp.asarray(fixed)].set(jnp.maximum(-signed, 0.0))
            upper_dual = upper_dual.at[jnp.asarray(fixed)].set(jnp.maximum(signed, 0.0))
            cursor += fixed.size
        if lower.size:
            lower_dual = lower_dual.at[jnp.asarray(lower)].set(
                homogeneous.dual[cursor : cursor + lower.size]
            )
            cursor += lower.size
        if upper.size:
            upper_dual = upper_dual.at[jnp.asarray(upper)].set(
                homogeneous.dual[cursor : cursor + upper.size]
            )
        audited = _audit_result(
            program,
            homogeneous.primal,
            cone_slack,
            cone_dual,
            lower_dual,
            upper_dual,
            ~homogeneous.active,
            homogeneous.iterations,
            policy,
            "native-jax-hsd",
            backend="phydrax",
        )
        # An uncertified stop after a failed Newton direction is a stalled
        # method, not an exhausted budget; certificates and optimality keep
        # their audited status.
        stalled = homogeneous.direction_failed & (
            audited.status == int(ConvexProgramStatus.ITERATION_LIMIT)
        )
        return eqx.tree_at(
            lambda value: value.status,
            audited,
            jnp.where(
                stalled, int(ConvexProgramStatus.NUMERICAL_FAILURE), audited.status
            ).astype(jnp.int32),
        )
    primal = _initial_primal(program, warm_start)
    dual = _initial_dual(program, warm_start)
    state = _NativeConicState(
        primal,
        dual,
        jnp.ones(program.batch_shape, dtype=jnp.bool_),
        jnp.zeros(program.batch_shape, dtype=jnp.int32),
    )
    tolerance = policy.termination.absolute

    def step(_: Array, current: _NativeConicState) -> _NativeConicState:
        quadratic_primal = _conic_quadratic_mv(program.quadratic, current.primal)
        gradient = (
            quadratic_primal
            + program.linear
            + _conic_matrix_transpose_mv(program.constraint_matrix, current.dual)
            + policy.regularization * current.primal
        )
        candidate_primal = jnp.clip(
            current.primal - method.primal_step * gradient,
            program.lower_bounds,
            program.upper_bounds,
        )
        extrapolated = candidate_primal + method.extrapolation * (
            candidate_primal - current.primal
        )
        violation = (
            _conic_matrix_mv(program.constraint_matrix, extrapolated)
            - program.constraint_rhs
        )
        candidate_dual = program.cone.project_dual(
            current.dual + method.dual_step * violation
        )
        affine_slack = program.cone.project(
            program.constraint_rhs
            - _conic_matrix_mv(program.constraint_matrix, candidate_primal)
        )
        interior_slack = affine_slack + jnp.sqrt(
            jnp.finfo(candidate_primal.dtype).eps
        ) * barrier_.interior_reference(candidate_primal.dtype)
        affine_mu = jnp.sum(interior_slack * candidate_dual, axis=-1) / max(
            barrier_.parameter, 1.0
        )
        central_dual = -affine_mu[..., None] * barrier_.gradient(interior_slack)
        corrected_dual = program.cone.project_dual(
            candidate_dual - 0.1 * method.dual_step * (candidate_dual - central_dual)
        )
        corrected_gradient = (
            _conic_quadratic_mv(program.quadratic, candidate_primal)
            + program.linear
            + _conic_matrix_transpose_mv(program.constraint_matrix, corrected_dual)
            + policy.regularization * candidate_primal
        )
        candidate_primal = jnp.clip(
            candidate_primal - 0.1 * method.primal_step * corrected_gradient,
            program.lower_bounds,
            program.upper_bounds,
        )
        candidate_dual = corrected_dual
        slack = program.cone.project(
            program.constraint_rhs
            - _conic_matrix_mv(program.constraint_matrix, candidate_primal)
        )
        primal_residual = _maximum_abs(
            _conic_matrix_mv(program.constraint_matrix, candidate_primal)
            + slack
            - program.constraint_rhs
        )
        dual_residual = _maximum_abs(
            _conic_quadratic_mv(program.quadratic, candidate_primal)
            + program.linear
            + _conic_matrix_transpose_mv(program.constraint_matrix, candidate_dual)
        )
        cone_complementarity = (
            program.cone.block_complementarity(slack, candidate_dual)
            if isinstance(program.cone, ProductCone)
            else program.cone.complementarity(slack, candidate_dual)[..., None]
        )
        complementarity = _maximum_abs(cone_complementarity)
        converged = (
            jnp.maximum(jnp.maximum(primal_residual, dual_residual), complementarity)
            <= tolerance
        )
        mask = current.active[..., None]
        return _NativeConicState(
            jnp.where(mask, candidate_primal, current.primal),
            jnp.where(mask, candidate_dual, current.dual),
            current.active & ~converged,
            current.iterations + current.active.astype(jnp.int32),
        )

    state = jax.lax.fori_loop(0, policy.termination.maximum_steps, step, state)
    primal = state.primal
    dual = program.cone.project_dual(state.dual)
    slack = program.cone.project(
        program.constraint_rhs - _conic_matrix_mv(program.constraint_matrix, primal)
    )
    unconstrained_gradient = (
        _conic_quadratic_mv(program.quadratic, primal)
        + program.linear
        + _conic_matrix_transpose_mv(program.constraint_matrix, dual)
    )
    activity_tolerance = jnp.asarray(
        max(policy.termination.absolute, 1e-8), dtype=primal.dtype
    )
    lower_active = jnp.isfinite(program.lower_bounds) & (
        primal - program.lower_bounds <= activity_tolerance
    )
    upper_active = jnp.isfinite(program.upper_bounds) & (
        program.upper_bounds - primal <= activity_tolerance
    )
    lower_dual = jnp.where(lower_active, jnp.maximum(unconstrained_gradient, 0.0), 0.0)
    upper_dual = jnp.where(upper_active, jnp.maximum(-unconstrained_gradient, 0.0), 0.0)
    return _audit_result(
        program,
        primal,
        slack,
        dual,
        lower_dual,
        upper_dual,
        ~state.active,
        state.iterations,
        policy,
        "native-jax",
        backend="phydrax",
    )


__all__ = [
    "PreparedNativeConic",
    "prepare_native_conic",
    "solve_native_conic_program",
]
