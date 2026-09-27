#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax.numpy as jnp
import pytest

import phydrax as phx


def test_jcgm_contracts() -> None:
    for common_variance, expected_covariance, expected_correlation in [
        (1.0, jnp.asarray([[2.0, 1.0], [1.0, 2.0]]), 0.5),
        (9.0, jnp.asarray([[10.0, 9.0], [9.0, 10.0]]), 0.9),
    ]:
        """Reproduce JCGM 102:2011 sections 9.2.2 and 9.2.4 exactly."""
        input_covariance = jnp.diag(jnp.asarray([1.0, 1.0, common_variance]))
        result = phx.uq.propagate_linearized(
            lambda value: jnp.asarray([value[0] + value[2], value[1] + value[2]]),
            jnp.zeros(3),
            phx.uq.DenseCovariance(input_covariance),
        )
        covariance = result.materialize_covariance().matrix
        standard_uncertainty = jnp.sqrt(jnp.diag(covariance))
        correlation = covariance[0, 1] / (
            standard_uncertainty[0] * standard_uncertainty[1]
        )

        assert jnp.array_equal(result.mean, jnp.zeros(2))
        assert jnp.allclose(covariance, expected_covariance)
        assert jnp.allclose(standard_uncertainty, jnp.sqrt(jnp.diag(expected_covariance)))
        assert correlation == pytest.approx(expected_correlation)
    for x1, correlation, expected_covariance in [
        (0.001, 0.0, jnp.asarray([[1.0e-4, 0.0], [0.0, 100.0]])),
        (0.010, 0.0, jnp.asarray([[1.0e-4, 0.0], [0.0, 1.0]])),
        (0.100, 0.0, jnp.asarray([[1.0e-4, 0.0], [0.0, 0.01]])),
        (0.100, 0.9, jnp.asarray([[1.0e-4, 9.0e-4], [9.0e-4, 0.01]])),
    ]:
        """Reproduce the generalized-GUM rows of JCGM 102:2011 section 9.3."""
        standard_uncertainty = 0.010
        input_covariance = standard_uncertainty**2 * jnp.asarray(
            [[1.0, correlation], [correlation, 1.0]]
        )
        result = phx.uq.propagate_linearized(
            lambda value: jnp.asarray(
                [jnp.hypot(value[0], value[1]), jnp.arctan2(value[1], value[0])]
            ),
            jnp.asarray([x1, 0.0]),
            phx.uq.DenseCovariance(input_covariance),
        )
        covariance = result.materialize_covariance().matrix

        assert jnp.allclose(result.mean, jnp.asarray([x1, 0.0]))
        assert jnp.allclose(covariance, expected_covariance, rtol=1e-12, atol=1e-12)
    """The section 9.3 case must not be disguised as a full polar distribution."""
    result = phx.uq.propagate_linearized(
        lambda value: jnp.asarray(
            [jnp.hypot(value[0], value[1]), jnp.arctan2(value[1], value[0])]
        ),
        jnp.asarray([0.001, 0.0]),
        phx.uq.DiagonalCovariance(jnp.full((2,), 0.010**2)),
    )
    angular_standard_uncertainty = jnp.sqrt(result.exact_variance()[1])

    assert result.approximation == "first_order"
    assert angular_standard_uncertainty == pytest.approx(10.0)
    assert angular_standard_uncertainty > 2.0 * jnp.pi
