#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.ml import _numerics as numerics
from phydrax.ml._contracts import (
    ML_INFEASIBLE,
    ML_RANK_DEFICIENT,
    ML_SUCCESS,
)


def test_weighted_reductions_mask_zero_weight_nonfinite_values():
    values = jnp.array([[1.0, 3.0], [jnp.nan, jnp.nan], [5.0, 7.0]])
    weights = jnp.array([1.0, 0.0, 3.0])

    assert jnp.allclose(numerics.weighted_mean(values, weights), jnp.array([4.0, 6.0]))
    totals, mass = numerics.segmented_weighted_sum(
        values,
        jnp.array([0, 0, 1]),
        weights,
        num_segments=3,
    )
    assert jnp.allclose(totals, jnp.array([[1.0, 3.0], [15.0, 21.0], [0.0, 0.0]]))
    assert jnp.allclose(mass, jnp.array([1.0, 3.0, 0.0]))
    means, mean_mass = numerics.segmented_weighted_mean(
        values,
        jnp.array([0, 0, 1]),
        weights,
        num_segments=3,
    )
    assert jnp.allclose(means, jnp.array([[1.0, 3.0], [5.0, 7.0], [0.0, 0.0]]))
    assert jnp.array_equal(mean_mass, mass)


def test_augmented_svd_solves_weighted_multioutput_and_differentiates():
    design = jnp.array([[-2.0, 1.0], [-1.0, 2.0], [1.0, 1.0], [2.0, -1.0]])
    coefficients = jnp.array([[2.0, -1.0], [0.5, 3.0]])
    intercept = jnp.array([1.5, -2.0])
    target = design @ coefficients + intercept
    weights = jnp.array([1.0, 2.0, 3.0, 4.0])

    result = numerics.solve_weighted_least_squares(design, target, weights)
    assert bool(result.valid)
    assert int(result.status) == ML_SUCCESS
    assert jnp.allclose(result.coefficients, coefficients, atol=1e-10)
    assert jnp.allclose(result.intercept, intercept, atol=1e-10)
    assert jnp.allclose(result.residual_sum_squares, 0.0, atol=1e-20)

    gradient = jax.grad(
        lambda values: jnp.sum(
            numerics.solve_weighted_least_squares(
                design,
                values,
                weights,
                ridge=0.1,
            ).coefficients
        )
    )(target)
    assert gradient.shape == target.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_least_squares_reports_invalid_weights_and_unidentified_intercept():
    design = jnp.ones((3, 1))
    target = jnp.arange(3.0)

    invalid = numerics.solve_weighted_least_squares(
        design,
        target,
        jnp.array([1.0, -1.0, 1.0]),
    )
    assert not bool(invalid.valid)
    assert int(invalid.status) == ML_INFEASIBLE

    unidentified = numerics.solve_weighted_least_squares(
        design,
        target,
        jnp.zeros((3,)),
        ridge=1.0,
        regularize_intercept=False,
    )
    assert not bool(unidentified.valid)
    assert int(unidentified.status) == ML_RANK_DEFICIENT


def test_weighted_subspace_reconstructs_and_ignores_zero_weight_nan_rows():
    values = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 0.0, 3.0],
            [jnp.nan, jnp.nan, jnp.nan],
        ]
    )
    weights = jnp.array([1.0, 1.0, 1.0, 0.0])
    result = numerics.fit_weighted_subspace(values, weights, rank=1)
    centered = values[:3] - result.offset
    reconstructed = (centered @ jnp.conj(result.components).T) @ result.components

    assert bool(result.valid)
    assert int(result.status) == ML_SUCCESS
    assert jnp.allclose(reconstructed, centered, atol=1e-10)
    assert jnp.allclose(result.retained_energy, 1.0, atol=1e-12)
    assert jnp.allclose(result.orthogonality_error, 0.0, atol=1e-12)

    projector_gradient = jax.grad(
        lambda matrix: jnp.sum(
            numerics.fit_weighted_subspace(matrix, weights, rank=1).components ** 2
        )
    )(values.at[-1].set(0.0))
    assert jnp.all(jnp.isfinite(projector_gradient))


def test_pairwise_chunking_and_assignment_surfaces_are_consistent():
    left = jnp.array([[0.0, 0.0], [1.0, 0.0], [3.0, 4.0]])
    right = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    dense = numerics.pairwise_distances(left, right, metric="squared-euclidean")
    reduced = numerics.chunked_pairwise_apply(
        left,
        right,
        lambda distances, start: jnp.min(distances, axis=-1) + 0 * start,
        metric="squared-euclidean",
        chunk_size=2,
    )

    assert jnp.allclose(dense, jnp.array([[0.0, 4.0], [1.0, 1.0], [25.0, 17.0]]))
    assert jnp.allclose(reduced, jnp.min(dense, axis=-1))
    assert jnp.array_equal(numerics.hard_assignments(dense), jnp.array([0, 0, 1]))
    probabilities = numerics.soft_assignments(dense, temperature=0.5)
    assert jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0)


def test_histogram_statistics_and_xgboost_newton_formulas():
    bins = jnp.array([[0, 1], [1, 0], [1, 1]])
    gradients = jnp.array([1.0, jnp.nan, -2.0])
    hessians = jnp.array([2.0, jnp.nan, 4.0])
    weights = jnp.array([1.0, 0.0, 2.0])
    gradient_sum, hessian_sum, count = numerics.histogram_gradient_statistics(
        bins,
        gradients,
        hessians,
        weights,
        num_bins=2,
    )

    assert jnp.allclose(gradient_sum, jnp.array([[1.0, -4.0], [0.0, -3.0]]))
    assert jnp.allclose(hessian_sum, jnp.array([[2.0, 8.0], [0.0, 10.0]]))
    assert jnp.allclose(count, jnp.array([[1.0, 2.0], [0.0, 3.0]]))
    assert jnp.allclose(
        numerics.xgboost_leaf_weight(-4.0, 8.0, l2_regularization=2.0),
        0.4,
    )
    gain = numerics.xgboost_split_gain(
        -3.0,
        4.0,
        1.0,
        2.0,
        l2_regularization=1.0,
    )
    expected = 0.5 * (9.0 / 5.0 + 1.0 / 3.0 - 4.0 / 7.0)
    assert jnp.allclose(gain, expected)


def test_proximal_and_fixed_iteration_primitives_have_fixed_shapes():
    projected = numerics.project_simplex(jnp.array([0.2, -0.5, 2.0]))
    assert jnp.all(projected >= 0.0)
    assert jnp.allclose(jnp.sum(projected), 1.0)
    assert jnp.allclose(
        numerics.soft_threshold(jnp.array([-2.0, 0.5, 3.0]), jnp.array(1.0)),
        jnp.array([-1.0, 0.0, 2.0]),
    )

    result = numerics.run_fixed_iterations(
        jnp.array(8.0),
        lambda value, iteration: (
            value / 2.0,
            value * value + 0 * iteration,
            jnp.abs(value / 2.0),
        ),
        max_iterations=8,
        tolerance=0.5,
        method="halving",
    )
    assert result.objective_history.shape == (8,)
    assert bool(result.converged)
    assert int(result.iterations) == 4
    assert jnp.allclose(result.value, 0.5)
