#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._barrier import ConeBarrierOracle
from ._cones import ProductCone, ZeroCone
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
    ConicProgram,
)


class HomogeneousConicState(eqx.Module):
    primal: Array
    dual: Array
    slack: Array
    tau: Array
    kappa: Array
    active: Array
    iterations: Array


def _split(cone, value):
    return cone.split(value) if isinstance(cone, ProductCone) else (value,)


def _blocks(cone):
    return cone.cones if isinstance(cone, ProductCone) else (cone,)


def _centrality(cone, barrier, slack, dual, mu):
    pieces = []
    for block, slack_block, dual_block in zip(
        _blocks(cone), _split(cone, slack), _split(cone, dual), strict=True
    ):
        if isinstance(block, ZeroCone):
            pieces.append(slack_block)
        else:
            local = barrier if not isinstance(cone, ProductCone) else None
            if local is None:
                from ._barrier import cone_barrier_oracle

                local = cone_barrier_oracle(block)
            pieces.append(dual_block + mu * local.gradient(slack_block))
    return jnp.concatenate(tuple(pieces))


def _embedding_residual(program, barrier, vector, mu):
    n, m = program.num_variables, program.num_constraints
    x = vector[:n]
    z = vector[n : n + m]
    s = vector[n + m : n + 2 * m]
    tau = vector[-2]
    kappa = vector[-1]
    px = _conic_quadratic_mv(program.quadratic, x)
    stationarity = (
        px
        + _conic_matrix_transpose_mv(program.constraint_matrix, z)
        + program.linear * tau
    )
    primal = (
        _conic_matrix_mv(program.constraint_matrix, x) + s - program.constraint_rhs * tau
    )
    gap = (
        jnp.sum(x * px) / tau
        + jnp.sum(program.linear * x)
        + jnp.sum(program.constraint_rhs * z)
        + kappa
    )
    centrality = _centrality(program.cone, barrier, s, z, mu)
    scalar_centrality = tau * kappa - mu
    return jnp.concatenate(
        (stationarity, primal, gap[None], centrality, scalar_centrality[None])
    )


def _positive_step(value, direction):
    candidate = jnp.where(direction < 0.0, -value / direction, jnp.inf)
    return jnp.minimum(1.0, jnp.min(candidate, initial=jnp.inf))


def _dual_step(cone, point, direction):
    lower = jnp.asarray(0.0, dtype=point.dtype)
    upper = jnp.asarray(1.0, dtype=point.dtype)

    def body(_, state):
        lo, hi = state
        middle = 0.5 * (lo + hi)
        candidate = point + middle * direction
        interior = (
            cone.dual_residual(candidate) <= 64.0 * jnp.finfo(point.dtype).eps
        ) & (cone.dual_projection_smoothness_margin(candidate) > 0.0)
        return jnp.where(interior, middle, lo), jnp.where(interior, hi, middle)

    endpoint = point + direction
    accepted = (cone.dual_residual(endpoint) <= 64.0 * jnp.finfo(point.dtype).eps) & (
        cone.dual_projection_smoothness_margin(endpoint) > 0.0
    )
    lower, _ = jax.lax.fori_loop(0, 64, body, (lower, upper))
    return jnp.where(accepted, 1.0, 0.995 * lower)


def _direction(program, barrier, vector, mu):
    residual = _embedding_residual(program, barrier, vector, mu)
    jacobian = jax.jacfwd(lambda value: _embedding_residual(program, barrier, value, mu))(
        vector
    )
    result = solve(
        LinearSystem(DenseLinearOperator(jacobian), problem_id="native-hsd-newton"),
        -residual,
        policy=LinearSolvePolicy(DenseLU()),
    )
    return result.value, result.successful


def _step_bound(program, barrier, vector, direction):
    n, m = program.num_variables, program.num_constraints
    z = vector[n : n + m]
    s = vector[n + m : n + 2 * m]
    dz = direction[n : n + m]
    ds = direction[n + m : n + 2 * m]
    tau, kappa = vector[-2], vector[-1]
    dtau, dkappa = direction[-2], direction[-1]
    primal_step = barrier.maximum_interior_step(s, ds)
    dual_step = _dual_step(program.cone, z, dz)
    return jnp.minimum(
        jnp.minimum(primal_step, dual_step),
        jnp.minimum(_positive_step(tau, dtau), _positive_step(kappa, dkappa)),
    )


def solve_homogeneous_conic(
    program: ConicProgram,
    barrier: ConeBarrierOracle,
    /,
    *,
    maximum_steps: int,
    tolerance: float,
):
    """Monotone homogeneous embedding with affine and centered Newton solves."""
    if program.batch_shape:
        raise ValueError("Homogeneous conic kernel currently requires one case.")
    reference = barrier.interior_reference(program.linear.dtype)
    dual = -barrier.gradient(reference)
    vector = jnp.concatenate(
        (
            jnp.zeros((program.num_variables,), dtype=program.linear.dtype),
            dual,
            reference,
            jnp.ones((2,), dtype=program.linear.dtype),
        )
    )
    active = jnp.asarray(True)
    iterations = jnp.asarray(0, dtype=jnp.int32)

    def iteration(_, state):
        vector_, active_, iterations_ = state
        n, m = program.num_variables, program.num_constraints
        slack = vector_[n + m : n + 2 * m]
        dual_ = vector_[n : n + m]
        tau, kappa = vector_[-2], vector_[-1]
        mu = (jnp.sum(slack * dual_) + tau * kappa) / (barrier.parameter + 1.0)
        affine, affine_ok = _direction(
            program, barrier, vector_, jnp.asarray(0.0, dtype=mu.dtype)
        )
        affine_step = _step_bound(program, barrier, vector_, affine)
        affine_vector = vector_ + affine_step * affine
        affine_mu = (
            jnp.sum(affine_vector[n + m : n + 2 * m] * affine_vector[n : n + m])
            + affine_vector[-2] * affine_vector[-1]
        ) / (barrier.parameter + 1.0)
        sigma = jnp.clip(
            (affine_mu / jnp.maximum(mu, jnp.finfo(mu.dtype).tiny)) ** 3, 0.0, 1.0
        )
        corrected, corrected_ok = _direction(program, barrier, vector_, sigma * mu)
        step = _step_bound(program, barrier, vector_, corrected)
        candidate = vector_ + step * corrected
        residual = jnp.max(
            jnp.abs(
                _embedding_residual(
                    program, barrier, candidate, jnp.asarray(0.0, dtype=mu.dtype)
                )
            ),
            initial=0.0,
        )
        converged = (residual <= tolerance) & (mu <= tolerance)
        accepted = active_ & affine_ok & corrected_ok & jnp.all(jnp.isfinite(candidate))
        next_vector = jax.lax.cond(
            accepted,
            lambda _: candidate,
            lambda _: vector_,
            operand=None,
        )
        return (
            next_vector,
            active_ & accepted & ~converged,
            iterations_ + active_.astype(jnp.int32),
        )

    vector, active, iterations = jax.lax.fori_loop(
        0, int(maximum_steps), iteration, (vector, active, iterations)
    )
    n, m = program.num_variables, program.num_constraints
    slack = vector[n + m : n + 2 * m]
    dual = vector[n : n + m]
    mu = (jnp.sum(slack * dual) + vector[-2] * vector[-1]) / (barrier.parameter + 1.0)
    residual = jnp.max(
        jnp.abs(
            _embedding_residual(
                program, barrier, vector, jnp.asarray(0.0, dtype=vector.dtype)
            )
        ),
        initial=0.0,
    )
    active = ~(
        jnp.all(jnp.isfinite(vector)) & (residual <= tolerance) & (mu <= tolerance)
    )
    tau = vector[-2]
    safe_tau = jnp.maximum(tau, jnp.sqrt(jnp.finfo(tau.dtype).eps))
    return HomogeneousConicState(
        vector[:n] / safe_tau,
        vector[n : n + m] / safe_tau,
        vector[n + m : n + 2 * m] / safe_tau,
        tau,
        vector[-1],
        active,
        iterations,
    )


__all__ = ["HomogeneousConicState", "solve_homogeneous_conic"]
