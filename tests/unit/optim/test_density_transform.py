from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_conic_filter_has_positive_normalized_rows_and_preserves_box() -> None:
    plan = phx.optim.ConicDensityFilterPlan(
        jnp.asarray([[0.0], [0.4], [1.1], [2.0]]),
        1.25,
        jnp.ones((4,), dtype=bool),
        None,
        jnp.asarray([0.2, 0.4, 0.7, 1.1]),
    )
    prepared = plan.prepare()

    coefficients = np.asarray(prepared.operator.coefficients)
    targets = np.asarray(prepared.operator.relation.target_indices)
    row_sums = np.bincount(targets, weights=coefficients, minlength=4)
    assert np.all(coefficients > 0.0)
    np.testing.assert_allclose(row_sums, 1.0, rtol=0.0, atol=2.0e-15)

    constant = prepared.apply(jnp.full((4,), 0.37))
    bounded = prepared.apply(jnp.asarray([0.0, 0.2, 0.8, 1.0]))
    np.testing.assert_allclose(constant, 0.37, rtol=0.0, atol=2.0e-15)
    assert bool(jnp.all((bounded >= 0.0) & (bounded <= 1.0)))


def test_conic_filter_uses_physical_distance_and_nonuniform_measures() -> None:
    prepared = phx.optim.ConicDensityFilterPlan(
        jnp.asarray([[0.0], [0.5], [2.0]]),
        1.1,
        jnp.ones((3,), dtype=bool),
        None,
        jnp.asarray([1.0, 3.0, 2.0]),
    ).prepare()

    filtered = prepared.apply(jnp.asarray([0.0, 1.0, 0.25]))
    expected = jnp.asarray(
        [
            (0.6 * 3.0) / (1.1 * 1.0 + 0.6 * 3.0),
            (1.1 * 3.0) / (0.6 * 1.0 + 1.1 * 3.0),
            0.25,
        ]
    )
    np.testing.assert_allclose(filtered, expected, rtol=2.0e-14, atol=2.0e-14)


def test_fixed_region_is_context_not_a_design_input_and_is_restored_exactly() -> None:
    filter_plan = phx.optim.ConicDensityFilterPlan(
        jnp.asarray([[0.0], [1.0], [2.0]]),
        1.5,
        jnp.asarray([False, True, False]),
        jnp.asarray([1.0, 0.0, 0.25]),
    )
    prepared_filter = filter_plan.prepare()

    first = prepared_filter.apply(jnp.asarray([0.1, 0.0, 0.9]))
    second = prepared_filter.apply(jnp.asarray([0.9, 0.0, 0.1]))
    expected_design_value = (0.5 * 1.0 + 0.5 * 0.25) / (0.5 + 1.5 + 0.5)
    np.testing.assert_allclose(first, [1.0, expected_design_value, 0.25])
    np.testing.assert_allclose(second, first)

    transform = phx.optim.DensityTransformPlan(
        filter_plan,
        phx.optim.TanhDensityProjectionPlan(jnp.asarray(0.5)),
    ).prepare()
    physical = transform.apply(jnp.asarray([0.6, 0.0, 0.8]), jnp.asarray(5.0))
    assert float(physical[0]) == 1.0
    assert float(physical[2]) == 0.25


def test_omitted_fixed_density_is_explicit_fixed_void() -> None:
    prepared = phx.optim.ConicDensityFilterPlan(
        jnp.asarray([[0.0], [1.0]]),
        0.0,
        jnp.asarray([True, False]),
    ).prepare()
    np.testing.assert_array_equal(prepared.apply(jnp.asarray([0.4, 0.9])), [0.4, 0.0])


def test_zero_radius_is_identity_on_the_design_region() -> None:
    prepared = phx.optim.ConicDensityFilterPlan(
        jnp.asarray([[0.0], [0.3], [1.8]]),
        0.0,
        jnp.asarray([True, False, True]),
        jnp.asarray([0.0, 0.75, 0.0]),
        jnp.asarray([0.2, 2.0, 0.5]),
    ).prepare()
    result = prepared.apply(jnp.asarray([0.1, 0.2, 0.9]))
    np.testing.assert_array_equal(result, [0.1, 0.75, 0.9])


def test_tanh_projection_is_monotone_bounded_and_supports_dynamic_beta() -> None:
    projection = phx.optim.TanhDensityProjectionPlan(jnp.asarray(0.5))
    density = jnp.linspace(0.0, 1.0, 41)
    low_beta = projection.apply(density, jnp.asarray(1.0))
    high_beta = projection.apply(density, jnp.asarray(12.0))

    assert bool(jnp.all(jnp.diff(low_beta) >= 0.0))
    assert bool(jnp.all(jnp.diff(high_beta) >= 0.0))
    endpoints = jnp.asarray([0, -1])
    np.testing.assert_allclose(low_beta[endpoints], [0.0, 1.0], atol=2.0e-15)
    np.testing.assert_allclose(high_beta[endpoints], [0.0, 1.0], atol=2.0e-15)
    assert float(high_beta[10]) < float(low_beta[10])


def test_radius_eta_beta_and_measures_are_validated() -> None:
    coordinates = jnp.asarray([[0.0], [1.0]])
    mask = jnp.ones((2,), dtype=bool)
    for radius in (-1.0, jnp.nan, jnp.inf):
        with pytest.raises(ValueError, match="radius"):
            phx.optim.ConicDensityFilterPlan(coordinates, radius, mask)

    for measures in (jnp.asarray([1.0, 0.0]), jnp.asarray([1.0, -0.1])):
        with pytest.raises(ValueError, match="measures"):
            phx.optim.ConicDensityFilterPlan(coordinates, 1.0, mask, None, measures)

    for eta in (0.0, 1.0, -0.1, 1.1, jnp.nan, jnp.inf):
        with pytest.raises(ValueError, match="eta"):
            phx.optim.TanhDensityProjectionPlan(eta)

    projection = phx.optim.TanhDensityProjectionPlan(0.5)
    for beta in (0.0, -1.0, jnp.nan, jnp.inf):
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="beta"):
            projection.apply(jnp.asarray([0.25, 0.75]), beta)
    with pytest.raises(ValueError, match="scalar"):
        projection.apply(jnp.asarray([0.25, 0.75]), jnp.asarray([2.0]))


def test_sparse_resource_limit_fails_before_route_materialization() -> None:
    plan = phx.optim.ConicDensityFilterPlan(
        jnp.asarray([[0.0], [0.1], [0.2]]),
        1.0,
        jnp.ones((3,), dtype=bool),
        maximum_connections=4,
    )
    with pytest.raises(ValueError, match="maximum_connections"):
        plan.prepare()


def test_threshold_density_is_binary_inclusive_and_forward_only() -> None:
    density = jnp.asarray([0.0, 0.49, 0.5, 0.9, 1.0])
    binary = phx.optim.threshold_density(density, jnp.asarray(0.5))
    np.testing.assert_array_equal(binary, [0.0, 0.0, 1.0, 1.0, 1.0])

    gradient = jax.grad(lambda value: jnp.sum(phx.optim.threshold_density(value)))(
        density
    )
    np.testing.assert_array_equal(gradient, jnp.zeros_like(density))

    for eta in (-0.1, 1.1, jnp.nan):
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="eta"):
            phx.optim.threshold_density(density, eta)


def test_density_transform_vjp_matches_centered_directional_difference() -> None:
    prepared = phx.optim.DensityTransformPlan(
        phx.optim.ConicDensityFilterPlan(
            jnp.asarray([[0.0], [0.5], [1.4], [2.2]]),
            1.6,
            jnp.ones((4,), dtype=bool),
            None,
            jnp.asarray([0.4, 0.7, 1.1, 0.6]),
        ),
        phx.optim.TanhDensityProjectionPlan(jnp.asarray(0.45)),
    ).prepare()
    density = jnp.asarray([0.22, 0.41, 0.63, 0.78])
    direction = jnp.asarray([0.3, -0.2, 0.1, 0.25])
    cotangent = jnp.asarray([0.7, -0.4, 0.5, 0.2])
    beta = jnp.asarray(3.5)

    _, pullback = jax.vjp(lambda value: prepared.apply(value, beta), density)
    (gradient,) = pullback(cotangent)
    directional_vjp = jnp.vdot(gradient, direction)

    step = 1.0e-4
    upper = jnp.vdot(cotangent, prepared.apply(density + step * direction, beta))
    lower = jnp.vdot(cotangent, prepared.apply(density - step * direction, beta))
    finite_difference = (upper - lower) / (2.0 * step)
    np.testing.assert_allclose(
        directional_vjp,
        finite_difference,
        rtol=2.0e-3,
        atol=2.0e-5,
    )
