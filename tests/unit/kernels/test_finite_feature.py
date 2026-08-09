#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _linear_features(point):
    return jnp.asarray([1.0, point[0]])


def _kernel(factor=None):
    if factor is None:
        factor = jnp.eye(2)
    return phx.kernels.FiniteFeatureKernel(
        _linear_features,
        factor,
        feature_map_id="linear",
        max_derivative_order=None,
    )


def test_finite_feature_capability_reports_rank_and_direct_features():
    points = jnp.asarray([-1.0, 0.5, 2.0])
    kernel = _kernel(jnp.asarray([[1.0, 0.0], [0.25, 0.5]]))

    expected = jax.vmap(_linear_features)(points[:, None]) @ kernel.feature_factor

    assert isinstance(kernel, phx.kernels.AbstractFiniteFeatureKernel)
    assert kernel.feature_rank == 2
    assert phx.kernels.kernel_feature_rank(kernel) == 2
    assert jnp.allclose(phx.kernels.kernel_features(kernel, points), expected)


def test_amplitude_sum_and_input_transform_compose_exact_features():
    points = jnp.asarray([-0.5, 0.25, 1.5])
    first = phx.kernels.AmplitudeKernel(_kernel(), 0.4)
    transformed = phx.kernels.InputTransformedKernel(
        _kernel(jnp.asarray([[0.5], [1.0]])),
        lambda point: 2.0 * point,
        transform_id="double",
        max_derivative_order=None,
    )
    kernel = first + transformed

    first_features = 0.4 * _kernel().features(points)
    transformed_features = _kernel(jnp.asarray([[0.5], [1.0]])).features(2.0 * points)
    expected = jnp.concatenate((first_features, transformed_features), axis=-1)

    assert phx.kernels.kernel_feature_rank(kernel) == 3
    assert jnp.allclose(phx.kernels.kernel_features(kernel, points), expected)
    assert jnp.allclose(kernel.matrix(points, points), expected @ expected.T)
    compiled = jax.jit(lambda values: phx.kernels.kernel_features(kernel, values))
    assert jnp.allclose(compiled(points), expected)


def test_unsupported_kernel_algebra_has_no_feature_representation():
    finite = _kernel()
    stationary = phx.kernels.Matern32Kernel(length_scale=0.5)

    unsupported = (
        phx.kernels.ScaleKernel(finite, 0.5),
        phx.kernels.ProductKernel((finite, finite)),
        phx.kernels.SumKernel((finite, stationary)),
    )
    for kernel in unsupported:
        assert phx.kernels.kernel_feature_rank(kernel) is None
        with pytest.raises(TypeError, match="no exact finite-feature representation"):
            phx.kernels.kernel_features(kernel, jnp.asarray([0.0, 1.0]))


def test_composed_feature_rank_validation_rejects_inconsistent_kernel():
    class BadFeatureKernel(phx.kernels.AbstractFiniteFeatureKernel):
        def features(self, points):
            return jnp.ones((jnp.asarray(points).shape[0], 2))

        @property
        def feature_rank(self):
            return 3

        def pairwise(self, left, right):
            return jnp.asarray(1.0)

        def matrix(self, left, right):
            return jnp.ones((jnp.asarray(left).shape[0], jnp.asarray(right).shape[0]))

        def diagonal(self, points):
            return jnp.ones((jnp.asarray(points).shape[0],))

        @property
        def max_derivative_order(self):
            return 0

        @property
        def is_unit_diagonal(self):
            return True

        @property
        def kernel_id(self):
            return "bad-feature"

    with pytest.raises(ValueError, match="declared rank"):
        phx.kernels.kernel_features(BadFeatureKernel(), jnp.asarray([0.0, 1.0]))
