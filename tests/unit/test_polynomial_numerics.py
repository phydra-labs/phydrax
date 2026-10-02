#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

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
from phydrax._polynomial._endpoint import EndpointJetBasis
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


def test_polynomial_numerics_scenario_1() -> None:
    for family in _FAMILIES:
        intercept = jnp.asarray(-0.7)
        slope = jnp.asarray(1.3)
        coefficients = standard_affine_coefficients(family, intercept, slope)
        evaluate = jax.jit(lambda x: standard_series_value(family, coefficients, x))
        points = jnp.asarray([-0.8, -0.1, 0.0, 0.4, 1.1])

        values = jax.vmap(evaluate)(points)
        derivatives = jax.vmap(jax.grad(evaluate))(points)

        assert jnp.allclose(values, intercept + slope * points, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(derivatives, slope, rtol=1e-12, atol=1e-12)
    for family, expected in (
        ("chebyshev", lambda x: 2.0 * x**2 - 1.0),
        ("legendre", lambda x: 0.5 * (3.0 * x**2 - 1.0)),
        ("hermite", lambda x: 4.0 * x**2 - 2.0),
        ("hermite_e", lambda x: x**2 - 1.0),
        ("laguerre", lambda x: 1.0 - 2.0 * x + 0.5 * x**2),
    ):
        point = jnp.asarray(0.37)
        value = standard_series_value(family, jnp.asarray([0.0, 0.0, 1.0]), point)
        assert value == pytest.approx(float(expected(point)), rel=1e-12, abs=1e-12)
    rule = standard_normal_hermite_rule_data(5)

    assert rule.integration_measure == "standard-normal"
    assert rule.measure_mass == 1.0
    assert jnp.sum(rule.weights) == pytest.approx(1.0, rel=1e-13, abs=1e-13)
    for degree in range(rule.exact_degree + 1):
        observed = jnp.sum(rule.weights * rule.nodes**degree)
        expected = 0.0 if degree % 2 else float(math.prod(range(1, degree, 2)))
        assert observed == pytest.approx(expected, rel=2e-11, abs=2e-11)
    for kind, count, exact_degree, endpoint_policy in (
        ("gauss", 4, 7, "none"),
        ("radau", 4, 6, "left"),
        ("lobatto", 4, 5, "both"),
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
        # ty: ignore[invalid-argument-type]
        legendre_rule_data(3, "typo")
    legendre = legendre_rule_data(64, "gauss")
    hermite = standard_normal_hermite_rule_data(64)

    for rule in (legendre, hermite):
        assert jnp.all(jnp.isfinite(rule.nodes))
        assert jnp.all(jnp.isfinite(rule.weights))
        assert jnp.all(rule.weights > 0.0)
        assert np.allclose(np.asarray(rule.nodes), -np.asarray(rule.nodes)[::-1])
        assert np.allclose(np.asarray(rule.weights), np.asarray(rule.weights)[::-1])


def test_polynomial_numerics_scenario_2() -> None:
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
    nodes = jnp.asarray([-1.0, -0.4, 0.1, 0.8, 1.3])
    matrix = barycentric_differentiation_matrix(nodes)
    values = nodes**4 - 3.0 * nodes**2 + 2.0

    assert jnp.allclose(matrix @ values, 4.0 * nodes**3 - 6.0 * nodes, atol=1e-10)
    for degree in tuple(TRIANGLE_RULES):
        _assert_cubature_exact(xiao_gimbutas_rule_data("triangle", degree))
    for degree in tuple(TETRAHEDRON_RULES):
        _assert_cubature_exact(xiao_gimbutas_rule_data("tetrahedron", degree))


def _multiindices(dimension: int, degree: int) -> Any:
    return np.asarray(
        [
            exponent
            for exponent in np.ndindex(*((degree + 1,) * dimension))
            if sum(exponent) <= degree
        ],
        dtype=np.int32,
    )


def _reference_moments(reference: str, exponents: np.ndarray) -> Any:
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


def _assert_cubature_exact(rule: CubatureRuleData) -> None:
    points = np.asarray(rule.points)
    weights = np.asarray(rule.weights)
    exponents = _multiindices(points.shape[1], rule.exact_degree)
    values = np.prod(points[:, None, :] ** exponents[None, :, :], axis=-1)
    observed = weights @ values
    expected = _reference_moments(rule.reference_domain, exponents)
    assert np.allclose(observed, expected, rtol=5e-10, atol=5e-11)


def test_polynomial_numerics_scenario_3() -> None:
    for degree in tuple(LEBEDEV_RULES):
        _assert_cubature_exact(lebedev_rule_data(degree))
    for factory in (
        periodic_circle_rule_data,
        radial_disk_rule_data,
        radial_ball_rule_data,
    ):
        for degree in (0, 1, 2, 5, 8):
            _assert_cubature_exact(factory(degree))
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


@pytest.mark.strict_jax
def test_endpoint_basis_integer_coordinates_keep_fractional_bernoulli_values() -> None:
    basis = EndpointJetBasis(2, 2.0)
    points = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    observed = eqx.filter_jit(basis.values)(points)
    expected = np.asarray(
        [
            [[1.0, -0.5, 1.0 / 6.0, 0.0], [1.0, 0.5, 1.0 / 6.0, 0.0]],
            [[1.0, 0.5, 1.0 / 6.0, 0.0], [1.0, -0.5, 1.0 / 6.0, 0.0]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)
    assert observed.dtype == jnp.float64


@pytest.mark.parametrize("length", (0.125, 8.0), ids=("short", "long"))
@pytest.mark.strict_jax
def test_endpoint_horner_derivatives_match_physical_endpoint_jet_jumps(
    length: float,
) -> None:
    basis = EndpointJetBasis(3, length)
    evaluate: Callable[[Array], Array] = lambda x: basis.values(x / length)
    endpoints = jnp.asarray((0.0, length), dtype=jnp.float64)
    expected_jumps = np.eye(4, 5, k=1, dtype=np.float64)
    for order in range(4):
        actual = np.asarray(jax.vmap(evaluate)(endpoints))
        np.testing.assert_allclose(
            actual,
            basis.endpoint_jets(length)[:, order, :],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual[1] - actual[0], expected_jumps[order], rtol=0.0, atol=1e-12
        )
        evaluate = jax.jacfwd(evaluate)


@pytest.mark.parametrize("order", (1.5, True), ids=("fractional", "boolean"))
def test_endpoint_basis_refuses_noninteger_order(order: float) -> None:
    with pytest.raises(TypeError, match="max_order must be an integer"):
        # ty: ignore[invalid-argument-type]
        EndpointJetBasis(order, 1.0)


@pytest.mark.parametrize("limit", (1.5, True), ids=("fractional", "boolean"))
def test_endpoint_basis_refuses_noninteger_order_limit(limit: float) -> None:
    with pytest.raises(TypeError, match="maximum_order must be an integer"):
        # ty: ignore[invalid-argument-type]
        EndpointJetBasis(0, 1.0, maximum_order=limit)


@pytest.mark.parametrize("length", (1e-200, 1e200), ids=("underflow", "overflow"))
def test_endpoint_basis_refuses_unrepresentable_interval_scales(length: float) -> None:
    with pytest.raises(ValueError, match="not representable"):
        EndpointJetBasis(2, length)


def test_endpoint_jets_are_bound_to_the_prepared_interval() -> None:
    basis = EndpointJetBasis(2, 2.0)
    with pytest.raises(ValueError, match="prepared interval length"):
        basis.endpoint_jets(4.0)


def test_endpoint_basis_refuses_boolean_coordinates() -> None:
    with pytest.raises(TypeError, match="non-boolean"):
        EndpointJetBasis(1, 1.0).values(jnp.asarray(True))


@pytest.mark.parametrize("length", (1e-100, 1e100), ids=("underflow", "overflow"))
@pytest.mark.strict_jax
def test_endpoint_basis_refuses_unrepresentable_evaluation_precision(
    length: float,
) -> None:
    basis = EndpointJetBasis(1, length)
    evaluate = eqx.filter_jit(basis.values)
    with pytest.raises(eqx.EquinoxRuntimeError, match="coordinate dtype"):
        evaluate(jnp.asarray(0.5, dtype=jnp.float32))


def test_endpoint_coefficients_cannot_underflow_in_adopted_precision() -> None:
    basis = EndpointJetBasis(16, 1.0)
    with pytest.raises(eqx.EquinoxRuntimeError):
        basis.values(jnp.asarray(0.25, dtype=jnp.float16))
