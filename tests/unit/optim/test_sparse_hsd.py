# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    LinearSolvePolicy,
    OperatorProperties,
    SolveResourcePolicy,
    SparseFactorizationPolicy,
    TolerancePolicy,
)
from phydrax.optim import (
    cone_barrier_oracle,
    ConicProgram,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexTermination,
    NativeConicNewtonRoute,
    NativeHomogeneousConic,
    NonnegativeCone,
    ProductCone,
    solve_conic_program,
    ZeroCone,
)
from phydrax.optim._programming._native_hsd import _direction, _embedding_residual
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _positive_metric_program(blocks: int = 1) -> ConicProgram:
    size = 2 * blocks
    variables = ArraySpace(
        (size,), dtype=jnp.float64, space_id=f"sparse-hsd-regression:metric:{blocks}"
    )
    constraints = ArraySpace(
        (2 * size,), dtype=jnp.float64, space_id=f"sparse-hsd-regression:moments:{blocks}"
    )
    base = jnp.arange(blocks, dtype=jnp.int32) * 2
    columns = jnp.stack((base, base + 1, base, base + 1, base, base + 1), axis=1).reshape(
        -1
    )
    rows = jnp.stack(
        (base, base, base + 1, base + 1, size + base, size + base + 1), axis=1
    ).reshape(-1)
    relation = EdgeRelation(columns, rows, source_size=size, target_size=2 * size)
    matrix = SparseCoordinateOperator(
        relation,
        jnp.tile(jnp.asarray([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0]), blocks),
        source=variables,
        target=constraints,
    )
    diagonal = EdgeRelation(
        jnp.arange(size, dtype=jnp.int32),
        jnp.arange(size, dtype=jnp.int32),
        source_size=size,
        target_size=size,
    )
    quadratic = SparseCoordinateOperator(
        diagonal,
        jnp.full((size,), np.exp((1 / 1.1) ** 2), dtype=jnp.float64),
        source=variables,
        target=variables,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    return ConicProgram(
        quadratic,
        jnp.zeros((size,), dtype=jnp.float64),
        matrix,
        jnp.concatenate((jnp.tile(jnp.asarray([0.0, 2.0]), blocks), jnp.zeros((size,)))),
        ProductCone((ZeroCone(size), NonnegativeCone(size))),
        problem_id=f"positive-metric-hsd-regression:{blocks}",
        convexity_evidence="construction",
    )


def _initial_vector(program: ConicProgram) -> Array:
    barrier = cone_barrier_oracle(program.cone)
    slack = barrier.interior_reference(program.linear.dtype)
    return jnp.concatenate(
        (
            jnp.zeros_like(program.linear),
            -barrier.gradient(slack),
            slack,
            jnp.ones((2,), dtype=slack.dtype),
        )
    )


@pytest.mark.parametrize("route", ("factorized", "matrix-free"))
@pytest.mark.parametrize("blocks", (1, 32))
def test_sparse_feasible_conic_stops_only_after_original_kkt_complementarity(
    blocks: int, route: NativeConicNewtonRoute
) -> None:
    program = _positive_metric_program(blocks)
    tolerance = 1e-7
    result = solve_conic_program(
        program,
        policy=ConvexSolvePolicy(
            NativeHomogeneousConic(newton=route),
            termination=ConvexTermination(absolute=tolerance, maximum_steps=64),
            failure=FailurePolicy("status"),
        ),
    )
    assert bool(result.successful)
    assert bool(jnp.all(result.primal > 0))
    np.testing.assert_allclose(
        result.primal, jnp.ones_like(program.linear), atol=tolerance
    )
    left, right = result.primal[::2], result.primal[1::2]
    equations = jnp.stack((right - left, left + right), axis=1).reshape(-1)
    constraint_values = jnp.concatenate((equations, -result.primal))
    np.testing.assert_allclose(
        constraint_values + result.cone_slack, program.constraint_rhs, atol=tolerance
    )
    assert float(result.kkt_residual_norm) <= tolerance
    assert abs(float(result.complementarity_gap)) <= tolerance


def test_factorized_newton_refuses_reduced_pattern_over_budget() -> None:
    program = _positive_metric_program(32)
    starved = NativeHomogeneousConic(
        factorization=SparseFactorizationPolicy(
            "lu", ordering="approximate-minimum-degree", max_factor_nnz=8
        )
    )
    with pytest.raises(ValueError):
        solve_conic_program(
            program,
            policy=ConvexSolvePolicy(starved, failure=FailurePolicy("status")),
        )
    with pytest.raises(ValueError, match="symmetric indefinite"):
        NativeHomogeneousConic(factorization=SparseFactorizationPolicy("cholesky"))


def test_sparse_newton_direction_satisfies_consumer_equation_and_work_refusal() -> None:
    program = _positive_metric_program()
    barrier = cone_barrier_oracle(program.cone)
    vector = _initial_vector(program)
    mu = jnp.asarray(1.0, dtype=vector.dtype)
    result = _direction(program, barrier, vector, mu)
    assert bool(result.successful)
    epsilon = 1e-6
    action = (
        _embedding_residual(program, barrier, vector + epsilon * result.value, mu)
        - _embedding_residual(program, barrier, vector - epsilon * result.value, mu)
    ) / (2 * epsilon)
    residual = _embedding_residual(program, barrier, vector, mu)
    np.testing.assert_allclose(action + residual, jnp.zeros_like(vector), atol=1e-7)
    insufficient = LinearSolvePolicy(
        GMRES(restart=1),
        tolerance=TolerancePolicy(relative=1e-12, absolute=1e-12, max_steps=1),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    failed = _direction(program, barrier, vector, mu, linear=insufficient)
    assert not bool(failed.successful)
    assert int(failed.diagnostics.iterations) == 1
    denied = LinearSolvePolicy(
        GMRES(restart=12),
        tolerance=TolerancePolicy(max_steps=64),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
        resources=SolveResourcePolicy(krylov_basis_bytes=0),
    )
    with pytest.raises(ValueError, match="Krylov|krylov|basis|budget"):
        _direction(program, barrier, vector, mu, linear=denied)


def _orthant_ray_program(blocks: int, kind: str) -> tuple[ConicProgram, np.ndarray]:
    """Sparse LP pairs with x >= 0 that are primal infeasible or unbounded.

    ``primal``: x0 + x1 = -1 per block. ``dual``: min -(x0 + x1) with x0 = x1.
    Returns the program and its dense constraint matrix for independent audits.
    """
    size = 2 * blocks
    variables = ArraySpace(
        (size,), dtype=jnp.float64, space_id=f"sparse-hsd-ray:{kind}:x:{blocks}"
    )
    constraints = ArraySpace(
        (blocks + size,),
        dtype=jnp.float64,
        space_id=f"sparse-hsd-ray:{kind}:c:{blocks}",
    )
    base = np.arange(blocks) * 2
    columns = np.concatenate((np.stack((base, base + 1), 1).reshape(-1), np.arange(size)))
    rows = np.concatenate((np.repeat(np.arange(blocks), 2), blocks + np.arange(size)))
    second = 1.0 if kind == "primal" else -1.0
    values = np.concatenate((np.tile([1.0, second], blocks), -np.ones(size)))
    dense = np.zeros((blocks + size, size))
    dense[rows, columns] = values
    matrix = SparseCoordinateOperator(
        EdgeRelation(
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=size,
            target_size=blocks + size,
        ),
        jnp.asarray(values),
        source=variables,
        target=constraints,
    )
    rhs = np.concatenate(
        (np.full(blocks, -1.0 if kind == "primal" else 0.0), np.zeros(size))
    )
    program = ConicProgram(
        None,
        jnp.full((size,), 1.0 if kind == "primal" else -1.0, dtype=jnp.float64),
        matrix,
        jnp.asarray(rhs),
        ProductCone((ZeroCone(blocks), NonnegativeCone(size))),
        problem_id=f"sparse-hsd-ray-{kind}:{blocks}",
        convexity_evidence="construction",
    )
    return program, dense


def _ray_policy(maximum_steps: int) -> ConvexSolvePolicy:
    return ConvexSolvePolicy(
        NativeHomogeneousConic(),
        termination=ConvexTermination(absolute=1e-7, maximum_steps=maximum_steps),
        failure=FailurePolicy("status"),
    )


def test_sparse_primal_infeasible_program_stops_on_audited_dual_ray() -> None:
    blocks = 32
    program, dense = _orthant_ray_program(blocks, "primal")
    result = solve_conic_program(program, policy=_ray_policy(64))
    assert int(result.status) == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
    assert not bool(result.successful)
    assert int(result.iterations) < 64
    certificate = result.certificate
    assert bool(certificate.dual_ray_valid)
    # Farkas audit independent of the solver: A^T y = 0, y in K*, b^T y < 0.
    ray = np.asarray(certificate.inequality_dual_ray)
    assert np.max(np.abs(dense.T @ ray)) <= 1e-8
    assert np.min(ray[blocks:]) >= -1e-12
    assert float(np.asarray(program.constraint_rhs) @ ray) < -1e-8


def test_sparse_unbounded_program_stops_on_audited_primal_ray() -> None:
    blocks = 32
    program, dense = _orthant_ray_program(blocks, "dual")
    result = solve_conic_program(program, policy=_ray_policy(64))
    assert int(result.status) == int(ConvexProgramStatus.DUAL_INFEASIBLE)
    assert int(result.iterations) < 64
    certificate = result.certificate
    assert bool(certificate.primal_ray_valid)
    # Recession audit: -A d in K (zero rows vanish, orthant rows >= 0), q^T d < 0.
    ray = np.asarray(certificate.primal_ray)
    recession = -dense @ ray
    assert np.max(np.abs(recession[:blocks])) <= 1e-8
    assert np.min(recession[blocks:]) >= -1e-8
    assert float(np.asarray(program.linear) @ ray) < -1e-8


def test_sparse_infeasibility_without_certificate_stays_unresolved() -> None:
    # Same infeasible program, but the requested certificates demand a ray
    # objective below -1e3, which no normalized ray of this program can reach.
    # Without an admissible certificate the bounded run stays unresolved.
    program, _ = _orthant_ray_program(32, "primal")
    policy = ConvexSolvePolicy(
        NativeHomogeneousConic(),
        termination=ConvexTermination(
            absolute=1e-7,
            primal_infeasible=1e3,
            dual_infeasible=1e3,
            maximum_steps=2,
        ),
        failure=FailurePolicy("status"),
    )
    result = solve_conic_program(program, policy=policy)
    assert int(result.status) == int(ConvexProgramStatus.ITERATION_LIMIT)
    assert int(result.iterations) == 2
    assert not bool(result.successful)
    assert not bool(result.certificate.dual_ray_valid)
    assert not bool(result.certificate.primal_ray_valid)
    assert float(result.certificate.dual_ray_objective) < 0.0
