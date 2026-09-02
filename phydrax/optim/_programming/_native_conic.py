#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._bounds import Bounds
from ._barrier import cone_barrier_oracle, ConeBarrierOracle
from ._clarabel import _audit_result
from ._cones import NonnegativeCone, ProductCone, ZeroCone
from ._native_hsd import solve_homogeneous_conic
from ._policy import ConvexSolvePolicy, NativeHomogeneousConic
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
    ConicProgram,
)
from ._types import ConvexWarmStart


class _NativeConicState(eqx.Module):
    primal: Array
    dual: Array
    active: Array
    iterations: Array


def _maximum_abs(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value), axis=-1, initial=0.0)


def _initial_primal(program: ConicProgram, warm_start, /) -> Array:
    if warm_start is not None:
        primal = jnp.asarray(warm_start.primal, dtype=program.linear.dtype)
        if primal.shape != program.batch_shape + (program.num_variables,):
            raise ValueError("Native conic warm-start primal has the wrong shape.")
        return primal
    lower = jnp.where(jnp.isfinite(program.lower_bounds), program.lower_bounds, -1.0)
    upper = jnp.where(jnp.isfinite(program.upper_bounds), program.upper_bounds, 1.0)
    return jnp.minimum(jnp.maximum(jnp.zeros_like(program.linear), lower), upper)


def _initial_dual(program: ConicProgram, warm_start, /) -> Array:
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


def _augment_dense_bounds(program):
    fixed = jnp.asarray(program.fixed_bound_indices, dtype=jnp.int32)
    lower = jnp.asarray(program.lower_bound_indices, dtype=jnp.int32)
    upper = jnp.asarray(program.upper_bound_indices, dtype=jnp.int32)
    identity = jnp.eye(program.num_variables, dtype=program.linear.dtype)
    matrix = jnp.concatenate(
        (
            program.constraint_matrix,
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
        blocks = (*blocks, ZeroCone(int(fixed.size)))
    if lower.size:
        blocks = (*blocks, NonnegativeCone(int(lower.size)))
    if upper.size:
        blocks = (*blocks, NonnegativeCone(int(upper.size)))
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


@eqx.filter_jit
def solve_native_conic_program(
    program: ConicProgram,
    policy: ConvexSolvePolicy,
    /,
    *,
    barrier: ConeBarrierOracle | None = None,
    warm_start: ConvexWarmStart | None = None,
):
    """Execute a fixed-capacity JAX-native primal-dual conic iteration.

    The independent original-coordinate audit remains authoritative for every
    optimality or ray status; iteration residuals are never trusted as certificates.
    """
    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    method = policy.method
    if not isinstance(method, NativeHomogeneousConic):
        raise TypeError("policy method must be NativeHomogeneousConic.")
    barrier_ = cone_barrier_oracle(program.cone) if barrier is None else barrier
    if (
        not isinstance(barrier_, ConeBarrierOracle)
        or barrier_.cone.cone_id != program.cone.cone_id
    ):
        raise ValueError("Prepared barrier oracle does not match the program cone.")
    has_bounds = bool(
        program.fixed_bound_indices
        or program.lower_bound_indices
        or program.upper_bound_indices
    )
    if not program.batch_shape and (not program.constraint_is_sparse or not has_bounds):
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
        return _audit_result(
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
    primal = _initial_primal(program, warm_start)
    dual = _initial_dual(program, warm_start)
    state = _NativeConicState(
        primal,
        dual,
        jnp.ones(program.batch_shape, dtype=bool),
        jnp.zeros(program.batch_shape, dtype=jnp.int32),
    )
    tolerance = policy.termination.absolute

    def step(_, current):
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


__all__ = ["solve_native_conic_program"]
