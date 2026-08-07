#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from phydrax._interpolation import (
    bspline_cross_gram,
    bspline_evaluate,
    bspline_mass_matrix,
    bspline_projection_matrix,
    BSplineGrid,
    BSplineGridTransfer,
    project_bspline_coefficients,
)


def _open_grid(degree, interior):
    knots = jnp.asarray([*([-1.0] * (degree + 1)), *interior, *([1.0] * (degree + 1))])
    return BSplineGrid(knots, degree)


def _span_quadrature(*grids, order=8):
    reference_points, reference_weights = np.polynomial.legendre.leggauss(order)
    breakpoints = np.unique(
        np.concatenate(tuple(np.asarray(grid.breakpoints) for grid in grids))
    )
    points = []
    weights = []
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        points.extend(midpoint + half_width * reference_points)
        weights.extend(half_width * reference_weights)
    return np.asarray(points), np.asarray(weights)


def test_mass_and_cross_gram_match_scipy_basis_matrices():
    old_grid = _open_grid(3, [-0.72, -0.1, -0.1, 0.63])
    new_grid = _open_grid(2, [-0.85, -0.25, 0.4, 0.78])
    points, weights = _span_quadrature(old_grid, new_grid)
    old_basis = SciPyBSpline.design_matrix(
        points, np.asarray(old_grid.knots), old_grid.degree
    ).toarray()
    new_basis = SciPyBSpline.design_matrix(
        points, np.asarray(new_grid.knots), new_grid.degree
    ).toarray()

    expected_mass = new_basis.T @ (weights[:, None] * new_basis)
    expected_cross = new_basis.T @ (weights[:, None] * old_basis)

    assert np.allclose(
        np.asarray(bspline_mass_matrix(new_grid)), expected_mass, atol=2e-12
    )
    assert np.allclose(
        np.asarray(bspline_cross_gram(old_grid, new_grid)),
        expected_cross,
        atol=2e-12,
    )


@pytest.mark.parametrize("degree", range(1, 6))
def test_exact_nested_knot_insertion_preserves_functions(degree):
    old_grid = BSplineGrid.open_uniform(degree, 4)
    inserted = jnp.asarray([-0.73, -0.25, -0.25, 0.41])
    new_grid = BSplineGrid(jnp.sort(jnp.concatenate((old_grid.knots, inserted))), degree)
    transfer = BSplineGridTransfer(old_grid, new_grid)
    coefficients = (
        jnp.arange(2 * old_grid.coefficient_count * 3, dtype=float)
        .reshape((2, old_grid.coefficient_count, 3))
        .astype(complex)
    )
    coefficients = coefficients + 0.3j * coefficients[::-1]
    projected = project_bspline_coefficients(
        coefficients,
        transfer,
        coefficient_axis=1,
    )
    query = jnp.linspace(-1.0, 1.0, 201)

    actual = bspline_evaluate(
        new_grid.knots,
        jnp.moveaxis(projected, 1, 0),
        query,
        degree=degree,
        case_shape=(),
    ).values
    expected = bspline_evaluate(
        old_grid.knots,
        jnp.moveaxis(coefficients, 1, 0),
        query,
        degree=degree,
        case_shape=(),
    ).values

    assert transfer.method == "exact"
    assert transfer.projection_error_bound == 0.0
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=3e-11)
    assert np.allclose(
        np.asarray(transfer.matrix),
        np.asarray(bspline_projection_matrix(old_grid, new_grid, method="exact")),
    )


def test_l2_projection_preserves_global_affine_functions():
    old_grid = _open_grid(3, [-0.75, -0.22, 0.48])
    new_grid = _open_grid(3, [-0.88, -0.41, 0.05, 0.67])
    transfer = BSplineGridTransfer(old_grid, new_grid)

    constant = transfer(jnp.ones((old_grid.coefficient_count,)))
    identity = transfer(old_grid.greville_abscissae)

    assert transfer.method == "l2"
    assert transfer.condition_estimate > 1.0
    assert transfer.projection_error_bound >= 0.0
    assert np.allclose(np.asarray(constant), 1.0, atol=2e-12)
    assert np.allclose(
        np.asarray(identity), np.asarray(new_grid.greville_abscissae), atol=2e-12
    )


def test_l2_projection_error_bound_controls_observed_l2_error():
    old_grid = _open_grid(3, [-0.76, -0.28, 0.46])
    new_grid = _open_grid(3, [-0.9, -0.52, 0.14, 0.72])
    transfer = BSplineGridTransfer(old_grid, new_grid)
    coefficients = jnp.sin(jnp.arange(old_grid.coefficient_count, dtype=float))
    projected = transfer(coefficients)
    points, weights = BSplineGrid.open_uniform(1, 200).quadrature(3)
    old_values = bspline_evaluate(
        old_grid.knots, coefficients, points, degree=old_grid.degree
    ).values
    new_values = bspline_evaluate(
        new_grid.knots, projected, points, degree=new_grid.degree
    ).values
    observed_error = jnp.sqrt(jnp.sum(weights * (new_values - old_values) ** 2))
    certified_error = transfer.projection_error_bound * jnp.linalg.norm(coefficients)

    assert float(observed_error) <= float(certified_error) + 2e-11


def test_transfer_is_jittable_and_has_the_expected_linear_gradient():
    old_grid = BSplineGrid.open_uniform(3, 5)
    new_grid = _open_grid(3, [-0.81, -0.3, 0.19, 0.58])
    transfer = BSplineGridTransfer(old_grid, new_grid)
    coefficients = jnp.linspace(-1.0, 1.0, old_grid.coefficient_count)

    projected = eqx.filter_jit(transfer)(coefficients)
    gradient = jax.grad(lambda values: jnp.sum(transfer(values) ** 2))(coefficients)
    expected_gradient = 2.0 * transfer.matrix.T @ (transfer.matrix @ coefficients)

    assert projected.shape == (new_grid.coefficient_count,)
    assert np.allclose(np.asarray(gradient), np.asarray(expected_gradient), atol=2e-12)


def test_projection_failures_are_explicit():
    old_grid = BSplineGrid.open_uniform(3, 4)
    shifted_grid = BSplineGrid.open_uniform(3, 4, interval=(0.0, 2.0))
    nonnested_grid = _open_grid(3, [-0.8, -0.1, 0.55])

    with pytest.raises(ValueError, match="same active interval"):
        BSplineGridTransfer(old_grid, shifted_grid)
    with pytest.raises(ValueError, match="nested knot vector"):
        BSplineGridTransfer(old_grid, nonnested_grid, method="exact")
    with pytest.raises(ValueError, match="ill-conditioned"):
        BSplineGridTransfer(old_grid, nonnested_grid, maximum_condition=1.01)
    with pytest.raises(ValueError, match="coefficient axis"):
        BSplineGridTransfer(old_grid, nonnested_grid)(jnp.ones((3,)))
    with pytest.raises(eqx.EquinoxRuntimeError, match="must be finite"):
        projected = BSplineGridTransfer(old_grid, nonnested_grid)(
            jnp.full((old_grid.coefficient_count,), jnp.nan)
        )
        jax.block_until_ready(projected)
