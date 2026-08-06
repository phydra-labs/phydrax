#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


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
