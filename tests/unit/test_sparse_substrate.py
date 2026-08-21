#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.linalg import ArraySpace, DiagonalPairing


def test_edge_linear_map_matches_dense_forward_transpose_and_adjoint():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2, 0], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=3,
        target_size=2,
        valid=jnp.asarray([True, False, True, True]),
    )
    coefficients = jnp.asarray([2.0 + 1.0j, jnp.nan + 0.0j, -1.0j, 4.0])
    action = phx.sparse.SparseLinearMap(relation, coefficients)
    source = jnp.asarray([1.0 - 1.0j, jnp.nan + 0.0j, 2.0 + 0.5j])
    target = jnp.asarray([3.0 + 2.0j, -1.0 + 0.5j])
    dense = action.as_dense()

    forward = jax.jit(lambda values: action.mv(values))(source)
    transpose = jax.jit(lambda values: action.transpose_mv(values))(target)
    adjoint = jax.jit(lambda values: action.adjoint_mv(values))(target)

    assert jnp.all(jnp.isfinite(forward))
    assert jnp.allclose(forward, dense @ source.at[1].set(0.0))
    assert jnp.allclose(transpose, dense.T @ target)
    assert jnp.allclose(adjoint, jnp.conj(dense).T @ target)


def test_row_linear_map_preserves_cases_payloads_and_dense_adjoint():
    relation = phx.sparse.RowRelation(
        jnp.asarray(
            [
                [[0, 2], [1, 0]],
                [[2, 1], [0, 2]],
            ],
            dtype=jnp.int32,
        ),
        source_size=3,
        valid=jnp.asarray(
            [
                [[True, True], [True, False]],
                [[True, False], [True, True]],
            ]
        ),
        case_shape=(2,),
    )
    coefficients = jnp.asarray(
        [
            [[0.25, 0.75], [2.0, jnp.nan]],
            [[-1.0, jnp.nan], [0.5, 0.5]],
        ]
    )
    action = phx.sparse.SparseLinearMap(relation, coefficients)
    source = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [jnp.nan, jnp.nan], [9.0, 10.0]],
        ]
    )
    target = jnp.asarray(
        [
            [[1.0, -1.0], [2.0, 3.0]],
            [[-2.0, 0.5], [4.0, -3.0]],
        ]
    )
    dense = action.as_dense()

    forward = action.mv(source)
    adjoint = action.adjoint_mv(target)
    safe_source = source.at[1, 1].set(0.0)

    assert forward.shape == (2, 2, 2)
    assert adjoint.shape == (2, 3, 2)
    assert jnp.all(jnp.isfinite(forward))
    assert jnp.allclose(
        forward.reshape((4, 2)),
        dense @ safe_source.reshape((6, 2)),
    )
    assert jnp.allclose(
        adjoint.reshape((6, 2)),
        jnp.conj(dense).T @ target.reshape((4, 2)),
    )


def test_route_reductions_keep_invalid_nan_routes_inert_and_empty_targets_zero():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        source_size=3,
        target_size=3,
        valid=jnp.asarray([True, False, True]),
    )
    route_values = jnp.asarray([[2.0, 4.0], [jnp.nan, jnp.nan], [-1.0, 5.0]])

    summed = phx.sparse.route_reduce(relation, route_values)
    maximum = phx.sparse.route_reduce(relation, route_values, reduction="max")

    expected = jnp.asarray([[2.0, 4.0], [-1.0, 5.0], [0.0, 0.0]])
    assert jnp.array_equal(summed, expected)
    assert jnp.array_equal(maximum, expected)


def test_cochain_incidence_exposes_sparse_boundary_and_derivative_actions():
    boundary = jnp.asarray(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
        ]
    )
    incidence = phx.graph.CochainIncidence.from_dense(1, boundary)
    lower = jnp.asarray([2.0, 3.0, 5.0])
    upper = jnp.asarray([7.0, 11.0])

    derivative_action = incidence.exterior_derivative_map()
    boundary_action = incidence.boundary_map()

    assert jnp.allclose(derivative_action(lower), boundary.T @ lower)
    assert jnp.allclose(boundary_action(upper), boundary @ upper)
    assert jnp.array_equal(derivative_action.as_dense(), boundary.T)
    assert jnp.array_equal(boundary_action.as_dense(), boundary)


def test_sparse_coordinate_operator_adjoint_respects_declared_pairings():
    source = ArraySpace(
        (3,),
        dtype=jnp.complex128,
        pairing=DiagonalPairing(jnp.asarray([2.0, 3.0, 5.0])),
    )
    target = ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=DiagonalPairing(jnp.asarray([7.0, 11.0])),
    )
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2, 0], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=3,
        target_size=2,
    )
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.asarray([1.0 + 2.0j, -3.0j, 4.0 - 1.0j, 2.0]),
        source=source,
        target=target,
    )
    left = jnp.asarray([1.0 - 1.0j, 2.0, -0.5 + 3.0j])
    right = jnp.asarray([0.25 + 2.0j, -1.0j])

    assert jnp.allclose(
        target.inner(operator.mv(left), right),
        source.inner(left, operator.adjoint_mv(right)),
    )

    row_operator = phx.sparse.SparseCoordinateOperator(
        phx.sparse.RowRelation(
            jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32),
            source_size=3,
        ),
        jnp.asarray([[1.0 + 1.0j, 2.0], [3.0, 4.0 - 2.0j]]),
        source=source,
        target=target,
    )
    assert jnp.allclose(
        row_operator.mv(left),
        row_operator.as_dense() @ left,
    )
    assert jnp.allclose(
        target.inner(row_operator.mv(left), right),
        source.inner(left, row_operator.adjoint_mv(right)),
    )


def test_sparse_plans_reuse_global_asdex_jacobian_and_hessian_patterns():
    space = ArraySpace((4,), dtype=jnp.float64)
    target = ArraySpace((3,), dtype=jnp.float64)

    def residual(values, _):
        return (values[1:] - values[:-1]) ** 2

    first = jnp.asarray([0.0, 1.0, 3.0, 6.0])
    second = jnp.asarray([1.0, 1.5, 2.5, 4.0])
    jacobian_plan = phx.sparse.compile_sparse_jacobian(
        residual,
        first,
        source=space,
        target=target,
        compiler="asdex",
    )
    first_operator = jacobian_plan.operator(first)
    second_operator = jacobian_plan.operator(second)

    assert jacobian_plan.nnz == 6
    assert jacobian_plan.num_colors == 2
    assert jnp.allclose(first_operator.as_dense(), jax.jacfwd(residual)(first, None))
    assert jnp.allclose(second_operator.as_dense(), jax.jacfwd(residual)(second, None))
    assert jnp.allclose(
        jax.jit(lambda vector: second_operator.mv(vector))(jnp.ones_like(second)),
        second_operator.as_dense() @ jnp.ones_like(second),
    )
    dynamic_action = jax.jit(
        lambda point, vector: jacobian_plan.operator(point).mv(vector)
    )
    assert jnp.allclose(
        dynamic_action(second, jnp.ones_like(second)),
        second_operator.as_dense() @ jnp.ones_like(second),
    )

    def energy(values, _):
        return jnp.sum((values[1:] - values[:-1]) ** 2) + jnp.sum(values**2)

    hessian_plan = phx.sparse.compile_sparse_hessian(
        energy,
        first,
        space=space,
        compiler="asdex",
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    hessian = hessian_plan.operator(second)
    hessian_matrix = jax.hessian(energy)(second, None)
    assert hessian_plan.num_colors < space.size
    assert jnp.allclose(hessian.as_dense(), hessian_matrix)
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    result = phx.linalg.solve(phx.linalg.LinearSystem(hessian), rhs)
    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(hessian_matrix, rhs))


def test_sparse_diagonal_assembly_and_numeric_refresh_are_jit_safe():
    indices = jnp.arange(3, dtype=jnp.int32)
    relation = phx.sparse.EdgeRelation(
        indices,
        indices,
        source_size=3,
        target_size=3,
    )
    space = ArraySpace((3,), dtype=jnp.float64)
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )

    def operator(coefficients):
        return phx.sparse.SparseCoordinateOperator(
            relation,
            coefficients,
            source=space,
            target=space,
            properties=properties,
            operator_id="jit-sparse-refresh-operator",
        )

    initial_coefficients = jnp.asarray([2.0, 3.0, 4.0])
    problem = phx.linalg.LinearSystem(
        operator(initial_coefficients),
        problem_id="jit-sparse-refresh-system",
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.ConjugateGradient(),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.JacobiPreconditionerBuilder()
        ),
    )
    prepared = phx.linalg.prepare(problem, policy)
    right_hand_side = jnp.asarray([1.0, -2.0, 3.0])

    def refresh_and_solve(coefficients):
        refreshed_problem = phx.linalg.LinearSystem(
            operator(coefficients),
            problem_id=problem.problem_id,
        )
        refreshed = phx.linalg.refresh(prepared, refreshed_problem)
        diagonal = phx.linalg.assemble_diagonal(refreshed_problem.operator)
        result = phx.linalg.solve(refreshed, right_hand_side)
        return result.value, diagonal

    coefficients = jnp.asarray([2.5, 3.5, 4.5])
    value, diagonal = jax.jit(refresh_and_solve)(coefficients)
    assert jnp.allclose(diagonal, coefficients)
    assert jnp.allclose(value, right_hand_side / coefficients)
