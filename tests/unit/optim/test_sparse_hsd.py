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
    TolerancePolicy,
)
from phydrax.optim import (
    cone_barrier_oracle,
    ConicProgram,
    ConvexSolvePolicy,
    ConvexTermination,
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


@pytest.mark.parametrize("blocks", (1, 32))
def test_sparse_feasible_conic_stops_only_after_original_kkt_complementarity(
    blocks: int,
) -> None:
    program = _positive_metric_program(blocks)
    tolerance = 1e-7
    result = solve_conic_program(
        program,
        policy=ConvexSolvePolicy(
            NativeHomogeneousConic(),
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
