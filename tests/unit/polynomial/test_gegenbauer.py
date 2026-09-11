import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

import phydrax as phx
from phydrax._polynomial._gegenbauer import (
    _gegenbauer_monic_scales,
    _gegenbauer_orthonormal_scales,
    _gegenbauer_standard_scales,
    gauss_gegenbauer_rule_data,
    gegenbauer_connection_data,
    gegenbauer_differentiation_matrix,
)
from phydrax._polynomial._orthogonal import OrthogonalRuleData


def test_standard_monic_and_orthonormal_scales_have_declared_meaning():
    alpha = 0.8
    degree = 8
    standard = np.asarray(_gegenbauer_standard_scales(alpha, degree))
    monic = np.asarray(_gegenbauer_monic_scales(alpha, degree))
    orthonormal = np.asarray(_gegenbauer_orthonormal_scales(alpha, degree))

    expected_leading = np.asarray(
        [
            1.0
            if index == 0
            else 2.0**index * special.poch(alpha, index) / math.factorial(index)
            for index in range(degree + 1)
        ]
    )
    np.testing.assert_allclose(standard, expected_leading, rtol=2e-13, atol=2e-14)
    np.testing.assert_array_equal(monic, np.ones(degree + 1))

    rule = gauss_gegenbauer_rule_data(12, alpha)
    standard_values = np.asarray(phx.special.gegenbauer_vander(alpha, rule.nodes, degree))
    orthonormal_values = standard_values * orthonormal / standard
    gram = (orthonormal_values * np.asarray(rule.weights)[:, None]).T @ (
        orthonormal_values
    )
    np.testing.assert_allclose(gram, np.eye(degree + 1), rtol=3e-12, atol=4e-13)


def test_alpha_zero_scales_preserve_standard_collapse_and_limit_bases():
    degree = 7
    standard = np.asarray(_gegenbauer_standard_scales(0.0, degree))
    orthonormal = np.asarray(_gegenbauer_orthonormal_scales(0.0, degree))
    expected_orthonormal = np.concatenate(
        (
            np.asarray([1.0 / math.sqrt(math.pi)]),
            2.0 ** (np.arange(1, degree + 1) - 0.5) / math.sqrt(math.pi),
        )
    )

    np.testing.assert_array_equal(
        standard, np.concatenate((np.ones(1), np.zeros(degree)))
    )
    np.testing.assert_allclose(orthonormal, expected_orthonormal, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("alpha", (-0.49, 0.0, 0.4, 2.5))
def test_gauss_gegenbauer_rule_has_exact_weighted_moments(alpha):
    count = 9
    rule = gauss_gegenbauer_rule_data(count, alpha)

    assert isinstance(rule, OrthogonalRuleData)
    assert rule.family == "gegenbauer"
    assert rule.node_rule == "gauss"
    assert rule.reference_domain == "minus-one-one"
    assert rule.integration_measure == "gegenbauer-weight"
    assert rule.exact_degree == 2 * count - 1
    for exponent in range(2 * count):
        observed = np.sum(np.asarray(rule.weights) * np.asarray(rule.nodes) ** exponent)
        expected = (
            0.0 if exponent % 2 else special.beta(0.5 * (exponent + 1), alpha + 0.5)
        )
        assert observed == pytest.approx(expected, rel=4e-12, abs=4e-12)


def test_shifted_differentiation_matches_same_standard_series():
    alpha = 0.37
    degree = 9
    order = 2
    coefficients = jnp.asarray([0.4, -0.2, 0.7, 0.0, -0.3, 0.1, 0.6, -0.5, 0.2, 0.9])
    matrix = gegenbauer_differentiation_matrix(alpha, degree, order)
    derivative_coefficients = matrix @ coefficients
    points = jnp.linspace(-0.8, 0.8, 13)
    shifted_values = (
        phx.special.gegenbauer_vander(alpha + order, points, degree)
        @ derivative_coefficients
    )

    def series_value(point):
        return phx.special.gegenbauer_vander(alpha, point, degree) @ coefficients

    direct = jax.vmap(jax.grad(jax.grad(series_value)))(points)
    assert matrix.shape == (degree + 1, degree + 1)
    assert jnp.allclose(shifted_values, direct, rtol=4e-12, atol=5e-13)
    assert jnp.array_equal(
        gegenbauer_differentiation_matrix(alpha, degree, 0),
        jnp.eye(degree + 1),
    )
    assert (
        jnp.count_nonzero(gegenbauer_differentiation_matrix(alpha, degree, degree + 1))
        == 0
    )


def _normalized_vander(alpha, points, degree, normalization):
    standard_values = np.asarray(phx.special.gegenbauer_vander(alpha, points, degree))
    standard_scales = np.asarray(_gegenbauer_standard_scales(alpha, degree))
    if normalization == "standard":
        scales = standard_scales
    elif normalization == "monic":
        scales = np.asarray(_gegenbauer_monic_scales(alpha, degree))
    else:
        scales = np.asarray(_gegenbauer_orthonormal_scales(alpha, degree))
    return standard_values * scales / standard_scales


@pytest.mark.parametrize("normalization", ("standard", "monic", "orthonormal"))
def test_alpha_to_beta_connection_preserves_polynomial_values(normalization):
    alpha = 0.3
    beta = 1.6
    degree = 8
    points = np.linspace(-1.1, 1.1, 23)
    data = gegenbauer_connection_data(alpha, beta, degree, normalization=normalization)
    source_values = _normalized_vander(alpha, points, degree, normalization)
    target_values = _normalized_vander(beta, points, degree, normalization)

    np.testing.assert_allclose(
        target_values @ np.asarray(data.matrix),
        source_values,
        rtol=8e-12,
        atol=8e-13,
    )
    coefficients = jnp.arange(2 * (degree + 1), dtype=float).reshape((degree + 1, 2))
    converted = data.apply(coefficients)
    np.testing.assert_allclose(
        target_values @ np.asarray(converted),
        source_values @ np.asarray(coefficients),
        rtol=8e-12,
        atol=2e-11,
    )


def test_monic_connections_remain_complete_at_alpha_zero():
    degree = 9
    forward = gegenbauer_connection_data(0.0, 0.7, degree, normalization="monic")
    reverse = gegenbauer_connection_data(0.7, 0.0, degree, normalization="monic")

    np.testing.assert_allclose(
        np.asarray(reverse.matrix) @ np.asarray(forward.matrix),
        np.eye(degree + 1),
        rtol=6e-12,
        atol=6e-13,
    )


def test_connection_identity_and_byte_evidence_are_deterministic():
    degree = 10
    count = degree + 1
    data = gegenbauer_connection_data(0.4, 1.2, degree, dtype=jnp.float32)
    replay = gegenbauer_connection_data(0.4, 1.2, degree, dtype=jnp.float32)
    expected_bytes = (3 * count * count + 2 * count) * np.dtype(np.float32).itemsize

    assert data.construction_bytes == expected_bytes
    assert data.construction_bytes >= data.matrix.nbytes
    assert data.data_id == replay.data_id
    assert data.matrix.dtype == jnp.float32
    with pytest.raises(ValueError, match="maximum_construction_bytes"):
        gegenbauer_connection_data(
            0.4,
            1.2,
            degree,
            dtype=jnp.float32,
            maximum_construction_bytes=expected_bytes - 1,
        )


def test_polynomial_construction_boundaries_are_rejected():
    with pytest.raises(TypeError, match="num_nodes must be an integer"):
        gauss_gegenbauer_rule_data(True, 0.3)
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        gauss_gegenbauer_rule_data(0, 0.3)
    with pytest.raises(ValueError, match="greater than -1/2"):
        gauss_gegenbauer_rule_data(4, -0.5)
    with pytest.raises(ValueError, match="degenerate at beta=0"):
        gegenbauer_connection_data(0.5, 0.0, 4)
    with pytest.raises(ValueError, match="normalization"):
        gegenbauer_connection_data(0.5, 1.0, 4, normalization="unknown")
    with pytest.raises(ValueError, match="order must be nonnegative"):
        gegenbauer_differentiation_matrix(0.5, 4, -1)
    with pytest.raises(TypeError, match="real floating dtype"):
        gegenbauer_connection_data(0.5, 1.0, 4, dtype=complex)
