import math

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._polynomial._total_degree import TotalDegreePolynomialFeatures


def test_total_degree_features_standardize_and_evaluate_the_complete_span():
    basis = TotalDegreePolynomialFeatures(2, 2)
    points = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    weights = jnp.asarray([0.25, 0.5, 0.25])

    values, center, scale = basis.evaluate(points, weights)
    standardized = (points - center) / scale
    expected = jnp.prod(
        standardized[:, None, :] ** basis.exponents[None, :, :],
        axis=-1,
    )

    assert basis.feature_count == math.comb(4, 2) - 1
    assert set(map(tuple, basis.exponents.tolist())) == {
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
    }
    assert jnp.allclose(center, jnp.asarray([1.0, 1.0]))
    assert jnp.allclose(scale, jnp.asarray([1.0, jnp.sqrt(3.0)]))
    assert jnp.allclose(values, expected)


def test_total_degree_enumeration_scales_with_feature_count_not_tensor_width():
    basis = TotalDegreePolynomialFeatures(512, 1)

    assert basis.feature_count == 512
    assert basis.exponents.shape == (512, 512)
    assert jnp.array_equal(jnp.sum(basis.exponents, axis=1), jnp.ones((512,)))


def test_total_degree_capacity_and_content_identity_are_explicit():
    first = TotalDegreePolynomialFeatures(3, 3)
    replay = TotalDegreePolynomialFeatures(3, 3)
    alternative = TotalDegreePolynomialFeatures(3, 2)

    assert first.feature_id == replay.feature_id
    assert first.feature_id != alternative.feature_id
    assert first.storage_bytes == first.exponents.size * first.exponents.dtype.itemsize

    with pytest.raises(ValueError, match="exceeding maximum_features"):
        TotalDegreePolynomialFeatures(30, 4, maximum_features=100)
    with pytest.raises(ValueError, match="maximum_feature_bytes"):
        TotalDegreePolynomialFeatures(16, 2, maximum_feature_bytes=100)


def test_polynomial_recombination_prepares_one_static_feature_contract():
    config = phx.solver.PolynomialRecombination(
        3,
        maximum_features=128,
        maximum_feature_bytes=4096,
        maximum_moment_error=1e-8,
    )
    replay = phx.solver.PolynomialRecombination(
        3,
        maximum_features=128,
        maximum_feature_bytes=4096,
        maximum_moment_error=1e-8,
    )
    basis = config.prepare(3)

    assert config.differentiation == "frozen-selection"
    assert config.recombination_id == replay.recombination_id
    assert basis.feature_count == 19

    with pytest.raises(ValueError, match="frozen-selection"):
        phx.solver.PolynomialRecombination(2, differentiation="through-selection")
    with pytest.raises(ValueError, match="maximum_moment_error"):
        phx.solver.PolynomialRecombination(2, maximum_moment_error=-1.0)
