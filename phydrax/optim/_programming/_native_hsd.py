#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from phydrax._strict import StrictModule

from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    JacobianLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSystem,
    prepare_linearization,
    solve,
    TolerancePolicy,
)
from ._barrier import ConeBarrierOracle
from ._cones import AbstractConvexCone, ProductCone, ZeroCone
from ._policy import ConvexSolvePolicy, ConvexTermination, NativeHomogeneousConic
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
    ConicProgram,
)


class HomogeneousConicState(StrictModule):
    primal: Array
    dual: Array
    slack: Array
    tau: Array
    kappa: Array
    active: Array
    iterations: Array
    last_linear_status: Array


def _split(cone: AbstractConvexCone, value: Array) -> tuple[Array, ...]:
    return cone.split(value) if isinstance(cone, ProductCone) else (value,)


def _blocks(cone: AbstractConvexCone) -> tuple[AbstractConvexCone, ...]:
    return cone.cones if isinstance(cone, ProductCone) else (cone,)


def _centrality(
    cone: AbstractConvexCone,
    barrier: ConeBarrierOracle,
    slack: Array,
    dual: Array,
    mu: Array,
) -> Array:
    pieces: list[Array] = []
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


def _embedding_residual(
    program: ConicProgram, barrier: ConeBarrierOracle, vector: Array, mu: Array
) -> Array:
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
    # Align equation blocks with (x, z, s, tau, kappa). A one-row gap shift
    # otherwise creates a long artificial permutation cycle in matrix-free
    # Krylov coordinates, although it is invisible to a dense direct solve.
    return jnp.concatenate(
        (stationarity, primal, centrality, gap[None], scalar_centrality[None])
    )


def _positive_step(value: Array, direction: Array) -> Array:
    candidate = jnp.where(direction < 0.0, -value / direction, jnp.inf)
    return jnp.minimum(1.0, jnp.min(candidate, initial=jnp.inf))


def _dual_step(cone: AbstractConvexCone, point: Array, direction: Array) -> Array:
    lower = jnp.asarray(0.0, dtype=point.dtype)
    upper = jnp.asarray(1.0, dtype=point.dtype)

    def body(_: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
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


def _direction(
    program: ConicProgram,
    barrier: ConeBarrierOracle,
    vector: Array,
    mu: Array,
    *,
    linear: LinearSolvePolicy | None = None,
) -> LinearSolveResult:
    if program.constraint_is_sparse or program.quadratic_is_sparse:
        coordinates = ArraySpace(
            vector.shape,
            dtype=vector.dtype,
            space_id=f"native-hsd:{program.structure_id}:coordinates",
        )

        def residual(value: Array) -> Array:
            return _embedding_residual(program, barrier, value, mu)

        linearization = prepare_linearization(
            residual,
            vector,
            source=coordinates,
            target=coordinates,
            linearization_id=f"native-hsd:{program.structure_id}:newton",
        )
        operator = JacobianLinearOperator(linearization)
        selected = (
            LinearSolvePolicy(
                GMRES(restart=min(256, vector.size)),
                tolerance=TolerancePolicy(
                    relative=1e-10, absolute=1e-12, max_steps=max(64, 8 * vector.size)
                ),
                differentiation=DifferentiationPolicy("none"),
                failure=FailurePolicy("status"),
            )
            if linear is None
            else linear
        )
        if not isinstance(selected.method, GMRES):
            raise TypeError(
                "Sparse homogeneous Newton directions require native matrix-free GMRES."
            )
        return solve(
            LinearSystem(operator, problem_id="native-hsd-sparse-newton"),
            -linearization.primal,
            policy=selected,
        )
    residual = _embedding_residual(program, barrier, vector, mu)
    jacobian = jax.jacfwd(lambda value: _embedding_residual(program, barrier, value, mu))(
        vector
    )
    return solve(
        LinearSystem(DenseLinearOperator(jacobian), problem_id="native-hsd-newton"),
        -residual,
        policy=LinearSolvePolicy(DenseLU()) if linear is None else linear,
    )


def _normalized_kkt_converged(
    program: ConicProgram,
    vector: Array,
    policy: ConvexSolvePolicy,
) -> Array:
    # Reuse the canonical original-coordinate audit, including cone-block
    # complementarity aggregation and requested relative/absolute thresholds.
    # Unused ray/provenance outputs are eliminated from this scalar JAX action.
    from ._clarabel import _audit_result

    n, m = program.num_variables, program.num_constraints
    tau = vector[-2]
    safe_tau = jnp.maximum(tau, jnp.sqrt(jnp.finfo(vector.dtype).eps))
    primal = vector[:n] / safe_tau
    dual = vector[n : n + m] / safe_tau
    slack = vector[n + m : n + 2 * m] / safe_tau
    zero_bounds = jnp.zeros_like(primal)
    audit = _audit_result(
        program,
        primal,
        slack,
        dual,
        zero_bounds,
        zero_bounds,
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        policy,
        "native-hsd-convergence",
        backend="phydrax",
    )
    return (tau > jnp.sqrt(jnp.finfo(vector.dtype).eps)) & audit.successful


def _step_bound(
    program: ConicProgram, barrier: ConeBarrierOracle, vector: Array, direction: Array
) -> Array:
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
    policy: ConvexSolvePolicy | None = None,
) -> HomogeneousConicState:
    """Monotone homogeneous embedding with affine and centered Newton solves."""
    if program.batch_shape:
        raise ValueError("Homogeneous conic kernel currently requires one case.")
    audit_policy = (
        ConvexSolvePolicy(
            NativeHomogeneousConic(),
            termination=ConvexTermination(
                absolute=tolerance, maximum_steps=maximum_steps
            ),
            failure=FailurePolicy("status"),
        )
        if policy is None
        else policy
    )
    linear_policy = None
    if program.constraint_is_sparse or program.quadratic_is_sparse:
        size = program.num_variables + 2 * program.num_constraints + 2
        linear_policy = LinearSolvePolicy(
            GMRES(restart=min(256, size)),
            tolerance=TolerancePolicy(
                relative=min(1e-10, max(tolerance * 0.01, 1e-14)),
                absolute=min(1e-12, max(tolerance * 0.01, 1e-14)),
                max_steps=max(64, 8 * size),
            ),
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
            resources=audit_policy.resources,
        )
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
    last_linear_status = jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)

    def iteration(
        _: Array, state: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        vector_, active_, iterations_, _ = state
        n, m = program.num_variables, program.num_constraints
        slack = vector_[n + m : n + 2 * m]
        dual_ = vector_[n : n + m]
        tau, kappa = vector_[-2], vector_[-1]
        mu = (jnp.sum(slack * dual_) + tau * kappa) / (barrier.parameter + 1.0)
        affine_result = _direction(
            program,
            barrier,
            vector_,
            jnp.asarray(0.0, dtype=mu.dtype),
            linear=linear_policy,
        )
        affine, affine_ok = affine_result.value, affine_result.successful
        affine_step = _step_bound(program, barrier, vector_, affine)
        affine_vector = vector_ + affine_step * affine
        affine_mu = (
            jnp.sum(affine_vector[n + m : n + 2 * m] * affine_vector[n : n + m])
            + affine_vector[-2] * affine_vector[-1]
        ) / (barrier.parameter + 1.0)
        sigma = jnp.clip(
            (affine_mu / jnp.maximum(mu, jnp.finfo(mu.dtype).tiny)) ** 3, 0.0, 1.0
        )
        corrected_result = _direction(
            program, barrier, vector_, sigma * mu, linear=linear_policy
        )
        corrected, corrected_ok = corrected_result.value, corrected_result.successful
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
        converged = (
            (residual <= tolerance)
            & (mu <= tolerance)
            & _normalized_kkt_converged(program, candidate, audit_policy)
        )
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
            jnp.where(affine_ok, corrected_result.status, affine_result.status).astype(
                jnp.int32
            ),
        )

    def keep_iterating(state: tuple[Array, Array, Array, Array]) -> Array:
        _, active_, count_, _ = state
        return active_ & (count_ < maximum_steps)

    def advance(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        return iteration(state[2], state)

    vector, active, iterations, last_linear_status = jax.lax.while_loop(
        keep_iterating,
        advance,
        (vector, active, iterations, last_linear_status),
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
        jnp.all(jnp.isfinite(vector))
        & (residual <= tolerance)
        & (mu <= tolerance)
        & _normalized_kkt_converged(program, vector, audit_policy)
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
        last_linear_status,
    )


__all__ = ["HomogeneousConicState", "solve_homogeneous_conic"]
