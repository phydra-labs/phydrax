"""Independent metric solve, active restriction, and simplicial dual contracts."""

from __future__ import annotations

import math
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization._cell_complex import simplicial_cell_complex
from phydrax.discretization._cochain_hodge import (
    DiagonalHodge,
    simplicial_dual_hodges,
    SparseHodge,
)
from phydrax.discretization._topology import CellComplexTopology
from phydrax.linalg import (
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    OperatorPairing,
    PCG,
    solve,
    TolerancePolicy,
)


_ROWS = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int32)
_COLUMNS = np.asarray([0, 1, 2, 1, 2, 2], dtype=np.int32)
_VALUES = jnp.asarray([4.0, 1.0, 0.5, 3.0, 1.0, 2.0], dtype=jnp.float64)


def _dense(values: Array) -> Array:
    diagonal = jnp.diag(values[jnp.asarray([0, 3, 5], dtype=jnp.int32)])
    upper = jnp.zeros((3, 3), dtype=values.dtype)
    upper = upper.at[0, 1].set(values[1]).at[0, 2].set(values[2]).at[1, 2].set(values[4])
    return diagonal + upper + upper.T


def _simplex(dimension: int) -> CellComplexTopology:
    return simplicial_cell_complex(
        tuple(
            np.asarray(
                tuple(combinations(range(dimension + 1), degree + 1)), dtype=np.int32
            )
            for degree in range(dimension + 1)
        )
    )


def test_sparse_riesz_jit_refresh_has_value_and_rhs_implicit_gradients() -> None:
    hodge = SparseHodge(_ROWS, _COLUMNS, _VALUES, 3)
    rhs = jnp.asarray([1.0, -2.0, 0.75], dtype=jnp.float64)
    probe = jnp.asarray([0.5, 1.0, -0.25], dtype=jnp.float64)

    def objective(values: Array, right: Array) -> Array:
        updated = hodge.refresh(values)
        space, _ = updated.make_space(space_id="dynamic-metric")
        return jnp.vdot(probe, space.inverse_riesz(right))

    def reference(values: Array, right: Array) -> Array:
        return jnp.vdot(probe, jnp.linalg.solve(_dense(values), right))

    evaluate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    expected_value, expected_gradient = jax.value_and_grad(reference, argnums=(0, 1))(
        _VALUES, rhs
    )
    actual_value, actual_gradient = evaluate(_VALUES, rhs)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        actual_gradient[0], expected_gradient[0], rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        actual_gradient[1], expected_gradient[1], rtol=1e-8, atol=1e-9
    )
    refreshed = _VALUES * jnp.asarray(1.4, dtype=jnp.float64)
    refreshed_value, _ = evaluate(refreshed, rhs)
    np.testing.assert_allclose(
        refreshed_value, reference(refreshed, rhs), rtol=1e-9, atol=1e-10
    )


def test_sparse_active_restriction_inverts_principal_metric_not_masked_inverse() -> None:
    hodge = SparseHodge(_ROWS, _COLUMNS, _VALUES, 3)
    active = np.asarray([0, 2], dtype=np.int32)
    restricted = hodge.restrict(active)
    space, mass = restricted.make_space(space_id="relative-metric")
    rhs = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    solution = eqx.filter_jit(lambda right: space.inverse_riesz(right))(rhs)
    principal = np.asarray(_dense(_VALUES))[np.ix_(active, active)]
    expected = np.linalg.solve(principal, np.asarray(rhs))
    np.testing.assert_allclose(solution, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mass.mv(solution), rhs, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        principal @ np.asarray(solution), rhs, rtol=1e-10, atol=1e-10
    )
    full_rhs = np.asarray([1.0, 0.0, -0.5], dtype=np.float64)
    masked_inverse = np.linalg.solve(np.asarray(_dense(_VALUES)), full_rhs)[active]
    assert not np.allclose(principal @ masked_inverse, rhs, rtol=1e-10, atol=1e-10)


def test_sparse_validity_rejects_indefinite_positive_diagonal_after_refresh() -> None:
    rows = np.asarray([0, 0, 1], dtype=np.int32)
    columns = np.asarray([0, 1, 1], dtype=np.int32)
    hodge = SparseHodge(
        rows, columns, jnp.asarray([3.0, 0.5, 2.0], dtype=jnp.float64), 2
    ).admit()
    invalid = hodge.refresh(jnp.asarray([1.0, 2.0, 1.0], dtype=jnp.float64))
    assert not bool(
        jax.jit(lambda values: hodge.refresh(values).valid)(invalid.upper_values)
    )
    with pytest.raises(ValueError, match="positive definite"):
        invalid.admit()


def test_diagonal_restriction_and_refresh_preserve_dynamic_metric_derivative() -> None:
    hodge = DiagonalHodge(jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float64))
    active = np.asarray([True, False, True], dtype=np.bool_)
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    def energy(weights: Array) -> Array:
        space, _ = (
            hodge.refresh(weights)
            .restrict(active)
            .make_space(space_id="diagonal-relative")
        )
        return jnp.vdot(rhs, space.inverse_riesz(rhs))

    value, gradient = jax.jit(jax.value_and_grad(energy))(hodge.weights)
    np.testing.assert_allclose(value, 1.0 / 2.0 + 4.0 / 5.0)
    np.testing.assert_allclose(gradient, [-1.0 / 4.0, 0.0, -4.0 / 25.0])


def test_triangle_barycentric_dual_has_closed_form_vertex_and_edge_weights() -> None:
    vertices = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    hodges = simplicial_dual_hodges(_simplex(2), vertices, dual="barycentric")
    expected_edges = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
    points = np.asarray(vertices)
    midpoints = points[expected_edges].mean(axis=1)
    lengths = np.linalg.norm(
        points[expected_edges[:, 1]] - points[expected_edges[:, 0]], axis=1
    )
    expected = np.linalg.norm(points.mean(axis=0) - midpoints, axis=1) / lengths
    np.testing.assert_allclose(hodges[0].weights, np.full(3, 1.0 / 3.0), rtol=1e-12)
    np.testing.assert_allclose(hodges[1].weights, expected, rtol=1e-12)
    np.testing.assert_allclose(hodges[2].weights, [1.0], rtol=1e-12)


def test_four_dimensional_barycentric_volume_and_geometry_gradient() -> None:
    topology = _simplex(4)
    vertices = jnp.asarray(
        np.concatenate((np.zeros((1, 4)), np.eye(4))), dtype=jnp.float64
    )

    def vertex_volume(scale: Array) -> Array:
        return jnp.sum(
            simplicial_dual_hodges(topology, vertices * scale, dual="barycentric")[
                0
            ].weights
        )

    volume, derivative = jax.jit(jax.value_and_grad(vertex_volume))(
        jnp.asarray(2.0, dtype=jnp.float64)
    )
    np.testing.assert_allclose(volume, 2.0**4 / math.factorial(4), rtol=1e-10)
    np.testing.assert_allclose(derivative, 4.0 * 2.0**3 / math.factorial(4), rtol=1e-9)
    hodges = simplicial_dual_hodges(topology, vertices, dual="barycentric")
    np.testing.assert_allclose(hodges[0].weights, np.full(5, 1.0 / 120.0), rtol=1e-10)
    np.testing.assert_allclose(hodges[4].weights, [24.0], rtol=1e-10)


def test_circumcentric_dual_refuses_non_well_centered_triangle() -> None:
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="well-centered"):
        simplicial_dual_hodges(
            _simplex(2),
            jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
            dual="circumcentric",
        )


def test_sparse_riesz_preserves_failed_native_solve_status() -> None:
    policy = LinearSolvePolicy(
        PCG(),
        tolerance=TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=1),
        failure=FailurePolicy("status"),
    )
    space, _ = SparseHodge(_ROWS, _COLUMNS, _VALUES, 3, policy=policy).make_space(
        space_id="underresolved-metric"
    )
    pairing = space.pairing
    assert isinstance(pairing, OperatorPairing)
    prepared = pairing.prepared_inverse
    assert prepared is not None
    rhs = jnp.asarray([1.0, -2.0, 0.75], dtype=jnp.float64)
    result = solve(prepared, rhs)
    assert not bool(result.successful)
    assert int(result.status) == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    inverse = eqx.filter_jit(lambda right: space.inverse_riesz(right))
    with pytest.raises(eqx.EquinoxRuntimeError):
        inverse(rhs).block_until_ready()


def test_circumcentric_regular_tetrahedron_has_exact_dual_volume_weights() -> None:
    edges = np.linalg.cholesky(
        (np.eye(3, dtype=np.float64) + np.ones((3, 3), dtype=np.float64)) / 2.0
    )
    vertices = jnp.asarray(
        np.concatenate((np.zeros((1, 3), dtype=np.float64), edges)), dtype=jnp.float64
    )
    hodges = simplicial_dual_hodges(_simplex(3), vertices, dual="circumcentric")
    np.testing.assert_allclose(
        hodges[0].weights, np.full(4, np.sqrt(2.0) / 48.0), rtol=1e-11
    )
    np.testing.assert_allclose(hodges[3].weights, [6.0 * np.sqrt(2.0)], rtol=1e-11)
