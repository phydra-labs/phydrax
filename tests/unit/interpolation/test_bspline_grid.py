#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._interpolation import (
    bspline_batched_evaluate,
    bspline_evaluate,
    BSplineGrid,
    BSplineGridBank,
    TrainableBSplineGrid,
)


def test_open_uniform_grid_preserves_kan_grid_contract():
    grid = BSplineGrid.open_uniform(3, 8)

    assert grid.degree == 3
    assert grid.num_intervals == 8
    assert grid.coefficient_count == 11
    assert grid.is_uniform
    assert np.array_equal(np.asarray(grid.active_interval), np.asarray([-1.0, 1.0]))
    assert np.allclose(np.asarray(grid.breakpoints), np.linspace(-1.0, 1.0, 9))
    assert np.count_nonzero(np.asarray(grid.knots) == -1.0) == 4
    assert np.count_nonzero(np.asarray(grid.knots) == 1.0) == 4
    assert phx.nn.BSplineGrid is BSplineGrid


def test_homogeneous_grid_bank_matches_independent_grid_evaluation():
    grids = (
        BSplineGrid.open_uniform(3, 4),
        BSplineGrid(
            jnp.asarray([-1.0, -1.0, -1.0, -1.0, -0.82, -0.27, 0.46, 1.0, 1.0, 1.0, 1.0]),
            3,
        ),
    )
    bank = BSplineGridBank.from_grids(grids)
    coefficients = jax.random.normal(jax.random.key(20), (3, 2, 7))
    query = jax.random.uniform(jax.random.key(21), (3, 2, 9), minval=-1.0, maxval=1.0)

    actual = bspline_batched_evaluate(
        bank.knots,
        coefficients,
        query,
        degree=bank.degree,
        derivative_order=1,
    ).values
    expected = jnp.stack(
        tuple(
            bspline_evaluate(
                grid.knots,
                coefficients[:, index],
                query[:, index],
                degree=grid.degree,
                derivative_order=1,
                case_shape=(3,),
            ).values
            for index, grid in enumerate(grids)
        ),
        axis=1,
    )

    assert bank.num_grids == 2
    assert bank.coefficient_count == 7
    assert phx.nn.BSplineGridBank is BSplineGridBank
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=2e-12)
    assert np.all(
        np.isfinite(
            np.asarray(
                jax.jacrev(
                    lambda values: (
                        bspline_batched_evaluate(
                            bank.knots,
                            coefficients,
                            values,
                            degree=bank.degree,
                        ).values
                    )
                )(query)
            )
        )
    )


def test_trainable_grid_is_ordered_bounded_and_differentiable():
    grid = TrainableBSplineGrid(
        jnp.asarray([-8.0, -2.0, 0.0, 1.0, 7.0]),
        3,
        minimum_span=0.02,
    )
    coefficients = jnp.asarray([0.2, -0.7, 1.1, 0.4, -0.3, 0.8, -0.1, 0.5])
    query = jnp.asarray(0.17)

    def evaluate(logits):
        candidate = eqx.tree_at(lambda value: value.raw_span_logits, grid, logits)
        return bspline_evaluate(
            candidate.knots,
            coefficients,
            query,
            degree=candidate.degree,
        ).values

    gradient = jax.grad(evaluate)(grid.raw_span_logits)
    direction = jnp.asarray([0.3, -0.2, 0.4, -0.1, -0.4])
    epsilon = 1.0e-5
    finite_difference = (
        evaluate(grid.raw_span_logits + epsilon * direction)
        - evaluate(grid.raw_span_logits - epsilon * direction)
    ) / (2.0 * epsilon)

    assert np.all(np.diff(np.asarray(grid.breakpoints)) >= grid.minimum_span)
    assert np.count_nonzero(np.asarray(grid.knots) == -1.0) == 4
    assert np.count_nonzero(np.asarray(grid.knots) == 1.0) == 4
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.vdot(gradient, direction)) == pytest.approx(
        float(finite_difference), rel=2e-6, abs=2e-7
    )


def test_trainable_grid_quadrature_and_regularization_follow_live_spans():
    fixed = BSplineGrid(
        jnp.asarray(
            [-1.0, -1.0, -1.0, -1.0, -0.75, -0.2, 0.18, 0.81, 1.0, 1.0, 1.0, 1.0]
        ),
        3,
    )
    grid = TrainableBSplineGrid.from_grid(fixed, minimum_span=0.01)
    points, weights = grid.quadrature(7)
    uniform = TrainableBSplineGrid.open_uniform(3, 5)

    for power in range(8):
        assert float(jnp.sum(weights * points**power)) == pytest.approx(
            (1.0 - (-1.0) ** (power + 1)) / (power + 1),
            abs=3e-13,
        )
    assert np.allclose(np.asarray(grid.breakpoints), np.asarray(fixed.breakpoints))
    assert float(uniform.regularization()) == pytest.approx(0.0, abs=1e-14)
    assert float(grid.regularization()) > 0.0
    assert np.all(
        np.isfinite(
            np.asarray(
                jax.grad(
                    lambda logits: eqx.tree_at(
                        lambda value: value.raw_span_logits, grid, logits
                    ).regularization()
                )(grid.raw_span_logits)
            )
        )
    )


def test_nonuniform_repeated_and_unclamped_grid_metadata():
    knots = jnp.asarray([-1.0, 0.0, 0.0, 0.3, 0.3, 0.8, 1.0, 1.0, 2.0])
    grid = BSplineGrid(knots, 2)

    assert grid.coefficient_count == 6
    assert np.array_equal(np.asarray(grid.active_interval), np.asarray([0.0, 1.0]))
    assert np.allclose(np.asarray(grid.breakpoints), [0.0, 0.3, 0.8, 1.0])
    assert grid.continuity_orders == (0, 1)
    assert not grid.is_uniform


def test_span_quadrature_integrates_polynomials_exactly():
    grid = BSplineGrid(
        jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.13, 0.4, 0.4, 0.87, 1.0, 1.0, 1.0, 1.0]),
        3,
    )
    points, weights = grid.quadrature(7)

    for power in range(8):
        actual = jnp.sum(weights * points**power)
        expected = 1.0 / (power + 1)
        assert float(actual) == pytest.approx(expected, abs=2e-13)

    derivative_points, derivative_weights = grid.derivative_quadrature(2)
    assert derivative_points.shape == derivative_weights.shape
    assert derivative_points.size == grid.num_intervals * 2
    assert float(jnp.sum(derivative_weights)) == pytest.approx(1.0, abs=1e-13)


def test_kan_basis_accepts_an_explicit_nonuniform_grid():
    grid = BSplineGrid(
        jnp.asarray([-1.0, -1.0, -1.0, -0.7, -0.15, 0.55, 1.0, 1.0, 1.0]),
        2,
    )
    basis = phx.nn.BSplineEdgeBasis(grid=grid, regularization_order=1)
    coefficients = basis.initialize_coefficients(
        1, 1, "identity", jnp.asarray([0, 0], dtype=jnp.uint32)
    )
    query = jnp.linspace(-1.0, 1.0, 31)
    values = jax.vmap(lambda value: basis.evaluate(coefficients, value.reshape((1, 1))))(
        query
    ).reshape(query.shape)

    assert basis.grid is grid
    assert np.allclose(np.asarray(values), np.asarray(query), atol=2e-12)


@pytest.mark.parametrize(
    ("knots", "degree", "error", "message"),
    [
        ([0.0, 0.0, 1.0, 1.0], 1.5, TypeError, "integer"),
        ([0.0, 0.0, 1.0, 1.0], -1, ValueError, "nonnegative"),
        ([[0.0, 0.0], [1.0, 1.0]], 1, ValueError, "rank-one"),
        ([0.0, 0.0, 1.0 + 1.0j, 1.0], 1, TypeError, "real-valued"),
        ([0.0, 0.0, np.nan, 1.0], 1, ValueError, "finite"),
        ([0.0, 0.7, 0.4, 1.0], 1, ValueError, "nondecreasing"),
        ([0.0, 0.0, 0.0, 1.0], 1, ValueError, "multiplicity"),
        ([0.0, 0.0, 1.0], 1, ValueError, r"degree \+ 1 coefficients"),
        ([0.0, 0.0, 0.0, 0.0], 1, ValueError, "multiplicity"),
    ],
)
def test_grid_validation_rejects_invalid_knot_contracts(knots, degree, error, message):
    with pytest.raises(error, match=message):
        BSplineGrid(jnp.asarray(knots), degree)


def test_grid_constructor_validation():
    with pytest.raises(TypeError, match="integer"):
        BSplineGrid.open_uniform(3, 2.5)
    with pytest.raises(ValueError, match="positive"):
        BSplineGrid.open_uniform(3, 0)
    with pytest.raises(ValueError, match="finite and increasing"):
        BSplineGrid.open_uniform(3, 4, interval=(1.0, 1.0))
    with pytest.raises(ValueError, match="cannot be combined"):
        phx.nn.BSplineEdgeBasis(
            grid=BSplineGrid.open_uniform(3, 4),
            num_intervals=4,
        )
