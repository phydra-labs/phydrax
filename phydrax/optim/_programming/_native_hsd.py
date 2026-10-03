#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import assert_never

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
from ._cones import AbstractConvexCone, NonnegativeCone, ProductCone, ZeroCone
from ._native_hsd_factorized import (
    factor_reduced_newton,
    factorized_direction,
    FactorizedNewtonFactor,
    FactorizedNewtonPlan,
)
from ._policy import ConvexSolvePolicy, ConvexTermination, NativeHomogeneousConic
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
    ConicProgram,
)


type _LoopState = tuple[Array, Array, Array, Array, Array]


class HomogeneousConicState(StrictModule):
    """Terminal homogeneous iterate recovered in original conic coordinates.

    ``primal``, ``dual`` and ``slack`` are one consistent witness divided by the
    safeguarded homogeneous scale. They are an optimal candidate when ``tau`` is
    bounded away from zero and ray directions otherwise; the original-coordinate
    audit decides which interpretation is certified. ``direction_failed`` records
    that iteration stopped because a Newton direction failed or was non-finite,
    which distinguishes a stalled method from an exhausted iteration budget.
    """

    primal: Array
    dual: Array
    slack: Array
    tau: Array
    kappa: Array
    active: Array
    iterations: Array
    last_linear_status: Array
    direction_failed: Array


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
        elif isinstance(block, NonnegativeCone):
            # Primal-dual form s_i z_i = mu. The barrier form z + mu grad F(s)
            # has Jacobian entries mu / s_i^2 = z_i^2 / mu that diverge on
            # active rows as mu -> 0, stalling Newton at every active orthant
            # constraint; the bilinear form stays bounded on the central path.
            pieces.append(slack_block * dual_block - mu)
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


def _newton_row_scale(program: ConicProgram, vector: Array) -> Array:
    """Constant row scaling 1/(s_i + z_i) on orthant centrality rows.

    Row scaling leaves the exact Newton direction unchanged. The linearized row
    ``z_i ds_i + s_i dz_i`` divided by ``s_i + z_i`` has bounded coefficients in
    both strict-complementarity limits: ``ds_i`` dominates on active rows
    (``s_i -> 0``) and ``dz_i`` on inactive rows (``z_i -> 0``). This keeps the
    dense LU and matrix-free Krylov systems well scaled near optimality.
    """
    n, m = program.num_variables, program.num_constraints
    dual = vector[n : n + m]
    slack = vector[n + m : n + 2 * m]
    pieces: list[Array] = []
    for block, slack_block, dual_block in zip(
        _blocks(program.cone),
        _split(program.cone, slack),
        _split(program.cone, dual),
        strict=True,
    ):
        if isinstance(block, NonnegativeCone):
            pieces.append(1.0 / (slack_block + dual_block))
        else:
            pieces.append(jnp.ones_like(slack_block))
    ones = jnp.ones((n + m,), dtype=vector.dtype)
    tail = jnp.ones((2,), dtype=vector.dtype)
    return jnp.concatenate((ones, *pieces, tail))


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

        scale = _newton_row_scale(program, vector)

        def residual(value: Array) -> Array:
            return scale * _embedding_residual(program, barrier, value, mu)

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
    scale = _newton_row_scale(program, vector)
    residual = scale * _embedding_residual(program, barrier, vector, mu)
    jacobian = jax.jacfwd(
        lambda value: scale * _embedding_residual(program, barrier, value, mu)
    )(vector)
    return solve(
        LinearSystem(DenseLinearOperator(jacobian), problem_id="native-hsd-newton"),
        -residual,
        policy=LinearSolvePolicy(DenseLU()) if linear is None else linear,
    )


def _normalized_audit(
    program: ConicProgram,
    vector: Array,
    policy: ConvexSolvePolicy,
) -> tuple[Array, Array]:
    # Reuse the canonical original-coordinate audit, including cone-block
    # complementarity aggregation, requested relative/absolute thresholds and
    # the scale-normalized primal/dual ray certificates. Returns the optimality
    # and certified-infeasibility decisions for the recovered witness.
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
    converged = (tau > jnp.sqrt(jnp.finfo(vector.dtype).eps)) & audit.successful
    certified = audit.certificate.primal_ray_valid | audit.certificate.dual_ray_valid
    return converged, ~converged & certified


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
    newton: FactorizedNewtonPlan | None = None,
) -> HomogeneousConicState:
    """Monotone homogeneous embedding with affine and centered Newton solves.

    Sparse programs solve Newton systems through the policy's explicit route:
    ``"factorized"`` requires the structure's prepared ``newton`` plan and
    ``"matrix-free"`` uses native GMRES. Dense programs use dense LU.
    """
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
    method = audit_policy.method
    if not isinstance(method, NativeHomogeneousConic):
        raise TypeError("Homogeneous conic execution requires NativeHomogeneousConic.")
    sparse = program.constraint_is_sparse or program.quadratic_is_sparse
    relative_tolerance = min(1e-10, max(tolerance * 0.01, 1e-14))
    absolute_tolerance = min(1e-12, max(tolerance * 0.01, 1e-14))
    linear_policy = None
    factorized: FactorizedNewtonPlan | None = None
    if sparse:
        match method.newton:
            case "factorized":
                if newton is None:
                    raise ValueError(
                        "Factorized sparse homogeneous Newton systems require the "
                        "prepared reduced-KKT plan of this program structure."
                    )
                if newton.structure_id != program.structure_id:
                    raise ValueError(
                        "Prepared reduced-KKT plan does not match the program structure."
                    )
                factorized = newton
            case "matrix-free":
                size = program.num_variables + 2 * program.num_constraints + 2
                linear_policy = LinearSolvePolicy(
                    GMRES(restart=min(256, size)),
                    tolerance=TolerancePolicy(
                        relative=relative_tolerance,
                        absolute=absolute_tolerance,
                        max_steps=max(64, 8 * size),
                    ),
                    differentiation=DifferentiationPolicy("none"),
                    failure=FailurePolicy("status"),
                    resources=audit_policy.resources,
                )
            case _ as unreachable:
                assert_never(unreachable)
    elif newton is not None:
        raise ValueError("Dense homogeneous programs do not use a reduced-KKT plan.")

    def newton_direction(
        vector_: Array, mu_: Array, factor: FactorizedNewtonFactor | None
    ) -> tuple[Array, Array, Array]:
        if factorized is None or factor is None:
            result = _direction(program, barrier, vector_, mu_, linear=linear_policy)
            return result.value, result.successful, result.status.astype(jnp.int32)
        return factorized_direction(
            factorized,
            factor,
            program,
            vector_,
            mu_,
            lambda value: _embedding_residual(program, barrier, value, mu_),
            _newton_row_scale(program, vector_),
            relative_tolerance,
            absolute_tolerance,
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
    direction_failed = jnp.asarray(False)

    def iteration(_: Array, state: _LoopState) -> _LoopState:
        vector_, active_, iterations_, _, _ = state
        n, m = program.num_variables, program.num_constraints
        slack = vector_[n + m : n + 2 * m]
        dual_ = vector_[n : n + m]
        tau, kappa = vector_[-2], vector_[-1]
        mu = (jnp.sum(slack * dual_) + tau * kappa) / (barrier.parameter + 1.0)
        zero_mu = jnp.asarray(0.0, dtype=mu.dtype)
        affine_factor = (
            None
            if factorized is None
            else factor_reduced_newton(factorized, program, vector_, zero_mu)
        )
        affine, affine_ok, affine_status = newton_direction(
            vector_, zero_mu, affine_factor
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
        # Orthant and zero-cone weights do not depend on mu, so the affine factor
        # is reused; barrier-form cone blocks refactor at the centering mu.
        corrected_factor = (
            affine_factor
            if factorized is None or factorized.multiplier_independent
            else factor_reduced_newton(factorized, program, vector_, sigma * mu)
        )
        corrected, corrected_ok, corrected_status = newton_direction(
            vector_, sigma * mu, corrected_factor
        )
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
        optimal, certified = _normalized_audit(program, candidate, audit_policy)
        converged = (residual <= tolerance) & (mu <= tolerance) & optimal
        accepted = active_ & affine_ok & corrected_ok & jnp.all(jnp.isfinite(candidate))
        next_vector = jax.lax.cond(
            accepted,
            lambda _: candidate,
            lambda _: vector_,
            operand=None,
        )
        # A certified primal or dual ray terminates as early as optimality does;
        # the final audit re-derives the certificate from the retained iterate.
        return (
            next_vector,
            active_ & accepted & ~converged & ~certified,
            iterations_ + active_.astype(jnp.int32),
            jnp.where(affine_ok, corrected_status, affine_status).astype(jnp.int32),
            active_ & ~accepted,
        )

    def keep_iterating(state: _LoopState) -> Array:
        _, active_, count_, _, _ = state
        return active_ & (count_ < maximum_steps)

    def advance(state: _LoopState) -> _LoopState:
        return iteration(state[2], state)

    vector, active, iterations, last_linear_status, direction_failed = jax.lax.while_loop(
        keep_iterating,
        advance,
        (vector, active, iterations, last_linear_status, direction_failed),
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
    optimal, _ = _normalized_audit(program, vector, audit_policy)
    active = ~(
        jnp.all(jnp.isfinite(vector))
        & (residual <= tolerance)
        & (mu <= tolerance)
        & optimal
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
        direction_failed,
    )


__all__ = ["HomogeneousConicState", "solve_homogeneous_conic"]
