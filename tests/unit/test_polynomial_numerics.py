#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import (
    barycentric_differentiation_matrix,
    barycentric_interpolate,
)
from phydrax._polynomial._chebyshev import chebyshev_lobatto_data
from phydrax._polynomial._cubature import (
    CubatureRuleData,
    lebedev_rule_data,
    periodic_circle_rule_data,
    radial_ball_rule_data,
    radial_disk_rule_data,
    xiao_gimbutas_rule_data,
)
from phydrax._polynomial._lebedev_cubature_data import LEBEDEV_RULES
from phydrax._polynomial._orthogonal import (
    legendre_rule_data,
    standard_affine_coefficients,
    standard_normal_hermite_rule_data,
    standard_series_value,
)
from phydrax._polynomial._simplex_cubature_data import (
    TETRAHEDRON_RULES,
    TRIANGLE_RULES,
)


_FAMILIES = ("chebyshev", "legendre", "hermite", "hermite_e", "laguerre")


@pytest.mark.parametrize("family", _FAMILIES)
def test_standard_family_affine_coefficients_are_exact_and_differentiable(family):
    intercept = jnp.asarray(-0.7)
    slope = jnp.asarray(1.3)
    coefficients = standard_affine_coefficients(family, intercept, slope)
    evaluate = jax.jit(lambda x: standard_series_value(family, coefficients, x))
    points = jnp.asarray([-0.8, -0.1, 0.0, 0.4, 1.1])

    values = jax.vmap(evaluate)(points)
    derivatives = jax.vmap(jax.grad(evaluate))(points)

    assert jnp.allclose(values, intercept + slope * points, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(derivatives, slope, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("family", "expected"),
    (
        ("chebyshev", lambda x: 2.0 * x**2 - 1.0),
        ("legendre", lambda x: 0.5 * (3.0 * x**2 - 1.0)),
        ("hermite", lambda x: 4.0 * x**2 - 2.0),
        ("hermite_e", lambda x: x**2 - 1.0),
        ("laguerre", lambda x: 1.0 - 2.0 * x + 0.5 * x**2),
    ),
)
def test_standard_family_quadratic_mode_matches_classical_definition(family, expected):
    point = jnp.asarray(0.37)
    value = standard_series_value(family, jnp.asarray([0.0, 0.0, 1.0]), point)
    assert value == pytest.approx(float(expected(point)), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("kind", "count", "exact_degree", "endpoint_policy"),
    (
        ("gauss", 4, 7, "none"),
        ("radau", 4, 6, "left"),
        ("lobatto", 4, 5, "both"),
    ),
)
def test_legendre_rules_preserve_mass_endpoints_and_declared_moments(
    kind, count, exact_degree, endpoint_policy
):
    rule = legendre_rule_data(count, kind)

    assert rule.exact_degree == exact_degree
    assert rule.integration_measure == "lebesgue"
    assert rule.measure_mass == 2.0
    assert rule.endpoint_policy == endpoint_policy
    assert jnp.all(jnp.diff(rule.nodes) > 0.0)
    assert jnp.all(rule.weights > 0.0)
    if endpoint_policy in ("left", "both"):
        assert rule.nodes[0] == -1.0
    if endpoint_policy == "both":
        assert rule.nodes[-1] == 1.0

    for degree in range(exact_degree + 1):
        observed = jnp.sum(rule.weights * rule.nodes**degree)
        expected = 0.0 if degree % 2 else 2.0 / float(degree + 1)
        assert observed == pytest.approx(expected, rel=2e-11, abs=2e-11)


def test_legendre_rules_handle_minimum_orders_and_reject_invalid_requests():
    gauss = legendre_rule_data(1, "gauss")
    radau = legendre_rule_data(1, "radau")
    lobatto = legendre_rule_data(2, "lobatto")

    assert jnp.array_equal(gauss.nodes, jnp.asarray([0.0]))
    assert jnp.array_equal(gauss.weights, jnp.asarray([2.0]))
    assert jnp.array_equal(radau.nodes, jnp.asarray([-1.0]))
    assert jnp.array_equal(radau.weights, jnp.asarray([2.0]))
    assert jnp.array_equal(lobatto.nodes, jnp.asarray([-1.0, 1.0]))
    assert jnp.array_equal(lobatto.weights, jnp.asarray([1.0, 1.0]))
    with pytest.raises(TypeError, match="integer"):
        legendre_rule_data(True, "gauss")
    with pytest.raises(ValueError, match="at least two"):
        legendre_rule_data(1, "lobatto")
    with pytest.raises(ValueError, match="kind"):
        legendre_rule_data(3, "typo")


def test_standard_normal_hermite_rule_preserves_probability_moments():
    rule = standard_normal_hermite_rule_data(5)

    assert rule.integration_measure == "standard-normal"
    assert rule.measure_mass == 1.0
    assert jnp.sum(rule.weights) == pytest.approx(1.0, rel=1e-13, abs=1e-13)
    for degree in range(rule.exact_degree + 1):
        observed = jnp.sum(rule.weights * rule.nodes**degree)
        expected = 0.0 if degree % 2 else float(math.prod(range(1, degree, 2)))
        assert observed == pytest.approx(expected, rel=2e-11, abs=2e-11)


def test_high_order_rules_remain_finite_positive_and_symmetric():
    legendre = legendre_rule_data(64, "gauss")
    hermite = standard_normal_hermite_rule_data(64)

    for rule in (legendre, hermite):
        assert jnp.all(jnp.isfinite(rule.nodes))
        assert jnp.all(jnp.isfinite(rule.weights))
        assert jnp.all(rule.weights > 0.0)
        assert np.allclose(np.asarray(rule.nodes), -np.asarray(rule.nodes)[::-1])
        assert np.allclose(np.asarray(rule.weights), np.asarray(rule.weights)[::-1])


def test_chebyshev_lobatto_data_differentiates_and_interpolates_polynomials():
    data = chebyshev_lobatto_data(17, maximum_derivative_order=2)
    nodes = data.nodes
    values = nodes**8 - 2.0 * nodes**3 + nodes
    first = data.differentiation_matrix(1) @ values
    second = data.differentiation_matrix(2) @ values

    assert nodes[0] == -1.0
    assert nodes[-1] == 1.0
    assert jnp.sum(data.quadrature_weights) == pytest.approx(2.0, abs=1e-13)
    assert jnp.allclose(first, 8.0 * nodes**7 - 6.0 * nodes**2 + 1.0, atol=2e-10)
    assert jnp.allclose(second, 56.0 * nodes**6 - 12.0 * nodes, atol=2e-8)
    interpolated = jax.vmap(
        lambda point: barycentric_interpolate(
            point,
            nodes,
            data.barycentric_weights,
            values,
        )
    )(nodes)
    assert jnp.allclose(interpolated, values, rtol=1e-12, atol=1e-12)


def test_chebyshev_lobatto_data_handles_payloads_dtype_and_budgets():
    data = chebyshev_lobatto_data(
        9,
        maximum_derivative_order=1,
        dtype=jnp.float32,
    )
    payload = jnp.stack((data.nodes, data.nodes**2), axis=-1)
    differentiated = jax.jit(lambda values: data.differentiation_matrix(1) @ values)(
        payload
    )

    assert data.nodes.dtype == jnp.float32
    assert jnp.allclose(differentiated[:, 0], 1.0, atol=2e-5)
    assert jnp.allclose(differentiated[:, 1], 2.0 * data.nodes, atol=2e-5)
    with pytest.raises(ValueError, match="maximum_construction_bytes"):
        chebyshev_lobatto_data(
            17,
            maximum_derivative_order=2,
            maximum_construction_bytes=64,
        )
    with pytest.raises(ValueError, match="prepared range"):
        data.differentiation_matrix(2)


def test_generic_barycentric_differentiation_preserves_irregular_polynomials():
    nodes = jnp.asarray([-1.0, -0.4, 0.1, 0.8, 1.3])
    matrix = barycentric_differentiation_matrix(nodes)
    values = nodes**4 - 3.0 * nodes**2 + 2.0

    assert jnp.allclose(matrix @ values, 4.0 * nodes**3 - 6.0 * nodes, atol=1e-10)


def _multiindices(dimension: int, degree: int):
    return np.asarray(
        [
            exponent
            for exponent in np.ndindex(*((degree + 1,) * dimension))
            if sum(exponent) <= degree
        ],
        dtype=np.int32,
    )


def _reference_moments(reference: str, exponents: np.ndarray):
    values = []
    for exponent in exponents:
        total = int(np.sum(exponent))
        if reference in ("triangle", "tetrahedron"):
            numerator = math.prod(math.factorial(int(value)) for value in exponent)
            values.append(numerator / math.factorial(total + exponent.size))
        elif np.any(exponent % 2):
            values.append(0.0)
        else:
            numerator = math.prod(math.gamma((int(value) + 1) / 2) for value in exponent)
            if reference in ("circle", "sphere"):
                values.append(2.0 * numerator / math.gamma((total + exponent.size) / 2))
            else:
                values.append(numerator / math.gamma((total + exponent.size) / 2 + 1))
    return np.asarray(values)


def _assert_cubature_exact(rule: CubatureRuleData):
    points = np.asarray(rule.points)
    weights = np.asarray(rule.weights)
    exponents = _multiindices(points.shape[1], rule.exact_degree)
    values = np.prod(points[:, None, :] ** exponents[None, :, :], axis=-1)
    observed = weights @ values
    expected = _reference_moments(rule.reference_domain, exponents)
    assert np.allclose(observed, expected, rtol=5e-10, atol=5e-11)


@pytest.mark.parametrize("degree", tuple(TRIANGLE_RULES))
def test_xiao_gimbutas_triangle_rules_have_certified_total_degree(degree):
    _assert_cubature_exact(xiao_gimbutas_rule_data("triangle", degree))


@pytest.mark.parametrize("degree", tuple(TETRAHEDRON_RULES))
def test_xiao_gimbutas_tetrahedron_rules_have_certified_total_degree(degree):
    _assert_cubature_exact(xiao_gimbutas_rule_data("tetrahedron", degree))


@pytest.mark.parametrize("degree", tuple(LEBEDEV_RULES))
def test_positive_lebedev_rules_have_certified_total_degree(degree):
    _assert_cubature_exact(lebedev_rule_data(degree))


@pytest.mark.parametrize(
    "factory",
    (periodic_circle_rule_data, radial_disk_rule_data, radial_ball_rule_data),
)
@pytest.mark.parametrize("degree", (0, 1, 2, 5, 8))
def test_procedural_radial_rules_have_certified_total_degree(factory, degree):
    _assert_cubature_exact(factory(degree))


def test_cubature_data_identity_storage_and_validation_are_explicit():
    first = radial_disk_rule_data(6)
    second = radial_disk_rule_data(6)
    assert first.rule_id == second.rule_id
    assert first.storage_bytes == first.points.nbytes + first.weights.nbytes
    assert jnp.all(first.weights > 0.0)

    with pytest.raises(TypeError, match="integer"):
        radial_disk_rule_data(True)
    with pytest.raises(ValueError, match="maximum_rule_bytes"):
        CubatureRuleData(
            first.points,
            first.weights,
            exact_degree=first.exact_degree,
            family="radial-product",
            reference_domain="disk",
            backend="test",
            source_id="test",
            maximum_rule_bytes=1,
        )
