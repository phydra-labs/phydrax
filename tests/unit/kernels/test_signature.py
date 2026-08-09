#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _feature_gram(paths, depth):
    features = phx.stochastic.SignatureFeatures(
        int(paths.shape[-1]), depth, include_scalar=True
    )
    values = jax.vmap(features)(paths)
    return values @ values.T


def test_one_segment_signature_kernel_has_analytic_picard_series():
    left = jnp.asarray([[0.0, 0.0], [2.0, -1.0]])
    right = jnp.asarray([[1.0, 3.0], [0.5, 5.0]])
    increment_inner_product = jnp.dot(left[1] - left[0], right[1] - right[0])

    for order in (1, 3, 6):
        kernel = phx.kernels.SignaturePDEKernel(
            phx.kernels.LinearKernel(), polynomial_order=order
        )
        expected = sum(
            increment_inner_product**level / factorial(level) ** 2
            for level in range(order + 1)
        )
        assert jnp.allclose(kernel.pairwise(left, right), expected)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    ((jnp.float32, 3e-5), (jnp.float64, 2e-11)),
)
def test_signature_pde_matches_explicit_truncated_signature_features(dtype, tolerance):
    key = jax.random.key(18)
    paths = jnp.cumsum(jax.random.normal(key, (5, 5, 2), dtype=dtype), axis=1)

    for order in (1, 2, 4, 6):
        kernel = phx.kernels.SignaturePDEKernel(
            phx.kernels.LinearKernel(),
            polynomial_order=order,
            pair_block_size=3,
        )
        actual = kernel.matrix(paths, paths)
        expected = _feature_gram(paths, order)

        assert jnp.allclose(actual, expected, rtol=tolerance, atol=tolerance)
        assert jnp.allclose(
            kernel.diagonal(paths),
            jnp.diag(expected),
            rtol=tolerance,
            atol=tolerance,
        )
        assert jnp.allclose(actual, actual.T, rtol=tolerance, atol=tolerance)


def test_signature_pde_supports_rectangular_paths_and_block_sizes():
    left = jnp.asarray([[0.0, 0.0], [1.0, -0.5], [0.5, 1.0], [1.5, 0.5]])
    right = jnp.asarray([[1.0, 0.0], [0.5, 0.0], [0.5, 1.0], [1.0, 1.5], [2.0, 1.0]])
    left_paths = jnp.stack((left, -left, 0.5 * left))
    right_paths = jnp.stack((right, -0.25 * right))

    expected = jax.vmap(
        lambda x: jax.vmap(
            lambda y: (
                phx.stochastic.SignatureFeatures(2, 5, include_scalar=True)(x)
                @ phx.stochastic.SignatureFeatures(2, 5, include_scalar=True)(y)
            )
        )(right_paths)
    )(left_paths)
    for block_size in (1, 2, 4, 17):
        kernel = phx.kernels.SignaturePDEKernel(
            phx.kernels.LinearKernel(),
            polynomial_order=5,
            pair_block_size=block_size,
        )
        assert jnp.allclose(kernel.matrix(left_paths, right_paths), expected)
        assert jnp.allclose(kernel.pairwise(left, right), expected[0, 0])


def test_signature_pde_is_positive_semidefinite_for_linear_and_rbf_lifts():
    paths = jnp.cumsum(
        jax.random.normal(jax.random.key(93), (9, 6, 3), dtype=jnp.float64),
        axis=1,
    )
    kernels = (
        phx.kernels.SignaturePDEKernel(
            phx.kernels.LinearKernel(), polynomial_order=5, pair_block_size=7
        ),
        phx.kernels.SignaturePDEKernel(
            phx.kernels.SquaredExponentialKernel(length_scale=1.3),
            polynomial_order=5,
            pair_block_size=7,
        ),
    )

    for kernel in kernels:
        gram = kernel.matrix(paths, paths)
        eigenvalues = jnp.linalg.eigvalsh(0.5 * (gram + gram.T))
        tolerance = 5e-11 * jnp.max(jnp.diag(gram))
        assert eigenvalues.min() >= -tolerance


def test_signature_pde_is_positive_semidefinite_on_fractional_gaussian_paths():
    process = phx.stochastic.FractionalGaussianProcess(
        0.35,
        jnp.asarray([0.4, 0.7]),
        process_id="signature-kernel-fractional-fixture",
    )
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jax.random.key(94),
        jnp.linspace(0.0, 1.0, 7),
        sample_shape=(8,),
    )
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.LinearKernel(),
        polynomial_order=5,
        pair_block_size=5,
    )
    gram = kernel.matrix(realization.values, realization.values)
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (gram + gram.T))
    tolerance = 5e-11 * jnp.max(jnp.diag(gram))

    assert eigenvalues.min() >= -tolerance


def test_signature_pde_handles_degenerate_segments_and_one_knot_paths():
    path = jnp.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]])
    repeated = jnp.concatenate((path, jnp.broadcast_to(path[-1], (3, 2))), axis=0)
    constant = jnp.broadcast_to(jnp.asarray([2.0, -3.0]), (4, 2))
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.LinearKernel(), polynomial_order=5
    )

    assert jnp.allclose(kernel.pairwise(path, repeated), kernel.pairwise(path, path))
    assert jnp.allclose(kernel.pairwise(constant, path), 1.0)
    assert jnp.allclose(kernel.pairwise(path[:1], repeated), 1.0)
    assert jnp.allclose(kernel.pairwise(path[:1], path[:1]), 1.0)


def test_signature_pde_jit_and_gradients_are_finite_and_correct():
    left = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    right = jnp.asarray([[0.0, 0.0], [-0.5, 1.0], [0.25, 1.5]])
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.LinearKernel(), polynomial_order=4, pair_block_size=2
    )
    compiled = eqx.filter_jit(kernel.pairwise)
    value = compiled(left, right)
    path_gradient = jax.grad(lambda path: kernel.pairwise(path, right))(left)
    epsilon = 1e-5
    direction = jnp.asarray([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]])
    finite_difference = (
        kernel.pairwise(left + epsilon * direction, right)
        - kernel.pairwise(left - epsilon * direction, right)
    ) / (2.0 * epsilon)

    def scaled_value(scale):
        scaled_kernel = phx.kernels.SignaturePDEKernel(
            phx.kernels.ScaleKernel(phx.kernels.LinearKernel(), scale),
            polynomial_order=4,
        )
        return scaled_kernel.pairwise(left, right)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(path_gradient))
    assert jnp.allclose(jnp.sum(path_gradient * direction), finite_difference, rtol=2e-7)
    assert jnp.isfinite(jax.grad(scaled_value)(jnp.asarray(0.8)))


def test_signature_pde_converges_by_exact_nonnegative_self_levels():
    path = jnp.cumsum(
        jax.random.normal(jax.random.key(7), (5, 2), dtype=jnp.float64) * 0.4,
        axis=0,
    )
    reference = phx.kernels.SignaturePDEKernel(
        phx.kernels.LinearKernel(), polynomial_order=10
    ).pairwise(path, path)
    approximations = jnp.asarray(
        [
            phx.kernels.SignaturePDEKernel(
                phx.kernels.LinearKernel(), polynomial_order=order
            ).pairwise(path, path)
            for order in (1, 3, 5, 7)
        ]
    )
    errors = reference - approximations

    assert jnp.all(errors >= 0.0)
    assert jnp.all(errors[1:] < errors[:-1])


def test_signature_pde_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="input_ndim must be 1"):
        phx.kernels.SignaturePDEKernel(
            phx.kernels.InputTransformedKernel(
                phx.kernels.LinearKernel(),
                lambda x: jnp.sum(x, axis=0),
                transform_id="path-sum",
                input_ndim=2,
            )
        )
    with pytest.raises(ValueError, match="polynomial_order must be positive"):
        phx.kernels.SignaturePDEKernel(phx.kernels.LinearKernel(), polynomial_order=0)
    with pytest.raises(ValueError, match="pair_block_size must be positive"):
        phx.kernels.SignaturePDEKernel(phx.kernels.LinearKernel(), pair_block_size=0)

    kernel = phx.kernels.SignaturePDEKernel(phx.kernels.LinearKernel())
    with pytest.raises(ValueError, match="2 nonempty input axes"):
        kernel.pairwise(jnp.ones((2,)), jnp.ones((2,)))
    with pytest.raises(ValueError, match="channel dimensions"):
        kernel.pairwise(jnp.ones((2, 2)), jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="nonempty input axes"):
        kernel.pairwise(jnp.empty((0, 2)), jnp.ones((2, 2)))
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite values"):
        invalid = eqx.filter_jit(kernel.pairwise)(
            jnp.asarray([[0.0, 0.0], [jnp.nan, 1.0]]),
            jnp.ones((2, 2)),
        )
        jax.block_until_ready(invalid)
