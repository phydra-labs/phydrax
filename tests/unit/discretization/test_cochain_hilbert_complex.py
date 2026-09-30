#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_complex import interval_cell_complex
from phydrax.discretization._cochain import CochainDiscretization
from phydrax.discretization._cochain_hodge import DiagonalHodge, SparseHodge


_GRAM0 = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.4], [0.5, 0.4, 2.0]])
_GRAM1 = np.array([[2.0, 0.3], [0.3, 1.5]])
_D = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])


def _sparse(matrix: np.ndarray) -> SparseHodge:
    rows, columns = np.triu_indices(matrix.shape[0])
    return SparseHodge(rows, columns, matrix[rows, columns], matrix.shape[0])


def _relative_complex() -> CochainDiscretization:
    topology = interval_cell_complex(np.array([[0, 1], [1, 2]], dtype=np.int32), 3)
    return CochainDiscretization(
        topology,
        (_sparse(_GRAM0), _sparse(_GRAM1)),
        boundary_masks=(np.array([True, False, False]), np.zeros(2, dtype=np.bool_)),
        numeric_revision="relative-pairing-binding",
    )


def test_relative_adjoint_uses_restricted_inverse_not_full_inverse() -> None:
    complex_ = _relative_complex()
    values = jnp.array([0.7, -0.4])
    expected_active = np.linalg.solve(
        _GRAM0[1:, 1:], _D[:, 1:].T @ _GRAM1 @ np.asarray(values)
    )
    expected = np.concatenate(([0.0], expected_active))
    actual = eqx.filter_jit(
        lambda vector: complex_.codifferential(1, vector, boundary="relative")
    )(values)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    wrong = np.linalg.solve(_GRAM0, _D.T @ _GRAM1 @ np.asarray(values))[1:]
    assert np.linalg.norm(wrong - expected_active) > 1e-3
    assert complex_.hilbert_complex(boundary="relative").space(0).size == 2


def test_relative_adjoint_respects_complex_sesquilinear_duality() -> None:
    complex_ = _relative_complex()
    potential = jnp.array([13.0 + 4.0j, 0.2 - 0.5j, -0.7 + 0.3j])
    flux = jnp.array([0.4 + 0.1j, -0.6 + 0.8j])
    derivative = complex_.exterior_derivative(0, potential, boundary="relative")
    adjoint = complex_.codifferential(1, flux, boundary="relative")
    left = np.vdot(np.asarray(derivative), _GRAM1 @ np.asarray(flux))
    right = np.vdot(np.asarray(potential)[1:], _GRAM0[1:, 1:] @ np.asarray(adjoint)[1:])
    np.testing.assert_allclose(left, right, atol=1e-11, rtol=1e-11)
    assert adjoint[0] == 0.0


def test_sparse_refresh_inside_scan_updates_pairing_and_value_gradients() -> None:
    complex_ = _relative_complex()
    values = jnp.array([0.7, -0.4])

    def objective(scales: jax.Array) -> jax.Array:
        def step(
            state: CochainDiscretization, scale: jax.Array
        ) -> tuple[CochainDiscretization, jax.Array]:
            refreshed = state.with_metric(
                (
                    state.hodges[0],
                    complex_.hodges[1].refresh(scale * _GRAM1[np.triu_indices(2)]),
                ),
                numeric_revision=state.numeric_revision,
            )
            return refreshed, jnp.sum(
                refreshed.codifferential(1, values, boundary="relative")
            )

        _, outputs = jax.lax.scan(step, complex_, scales)
        return jnp.sum(outputs)

    scales = jnp.array([0.8, 1.1, 1.7])
    actual, gradient = eqx.filter_jit(jax.value_and_grad(objective))(scales)
    coefficient = np.sum(
        np.linalg.solve(_GRAM0[1:, 1:], _D[:, 1:].T @ _GRAM1 @ np.asarray(values))
    )
    np.testing.assert_allclose(
        actual, coefficient * np.sum(np.asarray(scales)), atol=1e-10
    )
    np.testing.assert_allclose(gradient, np.full(3, coefficient), atol=1e-10)


def test_traced_constructor_requires_binding_and_differentiates_metric() -> None:
    topology = interval_cell_complex(np.array([[0, 1]], dtype=np.int32), 2)

    def objective(scale: jax.Array) -> jax.Array:
        complex_ = CochainDiscretization(
            topology,
            (
                DiagonalHodge(jnp.array([2.0, 4.0])),
                DiagonalHodge(jnp.reshape(scale, (1,))),
            ),
            numeric_revision="jit-metric-binding",
        )
        return complex_.codifferential(1, jnp.array([3.0]))[1]

    value, derivative = jax.jit(jax.value_and_grad(objective))(jnp.asarray(5.0))
    np.testing.assert_allclose(value, 15.0 / 4.0)
    np.testing.assert_allclose(derivative, 3.0 / 4.0)
    with pytest.raises(ValueError, match="numeric_revision"):
        jax.jit(
            lambda scale: CochainDiscretization(
                topology,
                (DiagonalHodge(jnp.ones(2)), DiagonalHodge(jnp.reshape(scale, (1,)))),
            ).codifferential(1, jnp.ones(1))
        )(jnp.asarray(1.0))
