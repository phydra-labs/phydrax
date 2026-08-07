#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable


def test_exact_fit_preserves_unsorted_nodes_payloads_and_derivatives():
    ordered = jnp.linspace(-1.7, 2.3, 10)
    permutation = jnp.asarray([4, 0, 8, 2, 6, 1, 9, 5, 3, 7])
    nodes = ordered[permutation]

    def target(value):
        polynomial = value**3 - 2.0 * value + 0.4
        return jnp.asarray(
            [
                [polynomial + 1j * value, 2.0 * polynomial - 0.5j * value**2],
                [value**2, polynomial - 2j],
            ]
        )

    values = jax.vmap(target)(nodes)
    interpolant = phx.operators.fit_bspline(nodes, values)
    query = jnp.linspace(-1.7, 2.3, 31)
    actual = jax.vmap(interpolant)(query)
    expected = jax.vmap(target)(query)
    first = jax.jacfwd(lambda value: interpolant(value))(jnp.asarray(0.17))
    second = jax.jacfwd(jax.jacfwd(lambda value: interpolant(value)))(jnp.asarray(0.17))

    assert interpolant.output_shape == (2, 2)
    assert interpolant.coefficients.shape == (10, 2, 2)
    assert interpolant.diagnostics.matrix_rank == 10
    assert interpolant.diagnostics.weighted_residual_norm < 1e-12
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=2e-11)
    assert np.allclose(
        np.asarray(first),
        np.asarray(jax.jacfwd(target)(jnp.asarray(0.17))),
        atol=2e-10,
    )
    assert np.allclose(
        np.asarray(second),
        np.asarray(jax.jacfwd(jax.jacfwd(target))(jnp.asarray(0.17))),
        atol=2e-9,
    )


def test_weighted_least_squares_recovers_polynomial_with_duplicate_nodes():
    nodes = jnp.concatenate((jnp.linspace(-1.0, 1.0, 40), jnp.asarray([0.0, 0.0])))
    target = lambda value: 0.7 * value**3 - 0.2 * value**2 + 1.3 * value - 0.1
    values = target(nodes)
    values = values.at[-2:].add(jnp.asarray([3.0, -4.0]))
    weights = jnp.ones(nodes.shape).at[-2:].set(0.0)
    interpolant = phx.operators.fit_bspline(
        nodes,
        values,
        plan=phx.operators.BSplineInterpolationPlan(
            degree=3,
            num_intervals=4,
            mode="least_squares",
        ),
        sample_weights=weights,
    )
    query = jnp.linspace(-1.0, 1.0, 101)

    assert interpolant.coefficients.shape == (7,)
    assert interpolant.diagnostics.num_observations == 42
    assert np.allclose(
        np.asarray(interpolant(query)), np.asarray(target(query)), atol=2e-12
    )


def test_smoothing_reduces_sobolev_energy_for_noisy_data():
    nodes = jnp.linspace(-1.0, 1.0, 80)
    clean = jnp.sin(2.5 * jnp.pi * nodes)
    noisy = clean + 0.12 * jax.random.normal(jax.random.key(0), nodes.shape)
    least_squares = phx.operators.fit_bspline(
        nodes,
        noisy,
        plan=phx.operators.BSplineInterpolationPlan(
            degree=3,
            num_intervals=12,
            mode="least_squares",
        ),
    )
    smoothed = phx.operators.fit_bspline(
        nodes,
        noisy,
        plan=phx.operators.BSplineInterpolationPlan(
            degree=3,
            num_intervals=12,
            mode="smooth",
            smoothing=2.0e-3,
            regularization_order=2,
        ),
    )
    quadrature_points, quadrature_weights = smoothed.grid.derivative_quadrature(2)
    least_squares_energy = jnp.sum(
        quadrature_weights * jnp.abs(least_squares.derivative(quadrature_points, 2)) ** 2
    )

    assert smoothed.diagnostics.regularization_energy < float(least_squares_energy)
    assert smoothed.diagnostics.weighted_residual_norm > 0.0
    assert np.linalg.norm(np.asarray(smoothed(nodes) - clean)) < np.linalg.norm(
        np.asarray(noisy - clean)
    )


def test_natural_periodic_and_explicit_boundary_jets_are_exact():
    nodes = jnp.linspace(0.0, 1.0, 12)
    values = jnp.sin(2.0 * jnp.pi * nodes)
    natural = phx.operators.fit_bspline(
        nodes,
        values,
        plan=phx.operators.BSplineInterpolationPlan(boundary="natural"),
    )
    periodic = phx.operators.fit_bspline(
        nodes,
        values,
        plan=phx.operators.BSplineInterpolationPlan(
            mode="least_squares",
            num_intervals=8,
            boundary="periodic",
        ),
    )
    explicit = phx.operators.fit_bspline(
        nodes,
        values,
        constraints=(
            phx.operators.BSplineBoundaryConstraint("lower", 1, 2.0 * jnp.pi),
            phx.operators.BSplineBoundaryConstraint("upper", 1, 2.0 * jnp.pi),
        ),
    )

    assert np.allclose(np.asarray(natural(nodes)), np.asarray(values), atol=2e-12)
    assert np.allclose(
        np.asarray(natural.derivative(jnp.asarray([0.0, 1.0]), 2)),
        0.0,
        atol=2e-11,
    )
    for order in range(3):
        assert float(periodic.derivative(0.0, order)) == pytest.approx(
            float(periodic.derivative(1.0, order)), abs=2e-10
        )
    assert float(explicit.derivative(0.0)) == pytest.approx(2.0 * np.pi, abs=2e-11)
    assert float(explicit.derivative(1.0)) == pytest.approx(2.0 * np.pi, abs=2e-11)
    assert explicit.diagnostics.constraint_residual_norm < 3e-11


def test_fit_validation_rejects_ambiguous_or_rank_deficient_systems():
    nodes = jnp.linspace(-1.0, 1.0, 8)
    values = nodes**2
    with pytest.raises(ValueError, match="distinct"):
        phx.operators.fit_bspline(
            nodes.at[3].set(nodes[2]),
            values,
        )
    with pytest.raises(ValueError, match="rank deficient"):
        phx.operators.fit_bspline(
            nodes,
            values,
            plan=phx.operators.BSplineInterpolationPlan(
                mode="least_squares",
                num_intervals=12,
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        phx.operators.fit_bspline(nodes.at[2].set(jnp.nan), values)
    with pytest.raises(ValueError, match="finite"):
        phx.operators.fit_bspline(nodes, values.at[2].set(jnp.inf))
    with pytest.raises(ValueError, match="nonnegative"):
        phx.operators.fit_bspline(
            nodes,
            values,
            plan=phx.operators.BSplineInterpolationPlan(mode="least_squares"),
            sample_weights=jnp.ones(8).at[0].set(-1.0),
        )


def test_domain_function_adapter_is_fixed_jittable_and_differentiable():
    interval = phx.domain.ScalarInterval(-2.0, 3.0, label="x")
    source = interval.Function("x")(lambda x: jnp.asarray([x**3 - 2.0 * x, x**2 + 0.5]))
    approximation = phx.operators.interpolate_bspline(
        source,
        jnp.linspace(-2.0, 3.0, 11),
    )
    query = jnp.asarray(0.23)

    def evaluate(value):
        return approximation({"x": value}).data

    trainable, _ = partition_trainable(approximation)
    leaves = jax.tree.leaves(trainable)
    actual = eqx.filter_jit(evaluate)(query)
    jacobian = jax.jacrev(evaluate)(query)

    assert isinstance(approximation.func, phx.operators.BSplineInterpolant)
    assert approximation.domain is interval
    assert approximation.deps == ("x",)
    assert not any(eqx.is_inexact_array(leaf) for leaf in leaves)
    assert np.allclose(np.asarray(actual), np.asarray(source({"x": query}).data))
    assert np.allclose(
        np.asarray(jacobian), np.asarray([3.0 * query**2 - 2.0, 2.0 * query])
    )
