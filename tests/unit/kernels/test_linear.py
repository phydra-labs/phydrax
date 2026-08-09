#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_linear_kernel_specialized_operations_agree():
    left = jnp.asarray([[1.0, 2.0], [-1.0, 3.0]])
    right = jnp.asarray([[0.5, -2.0], [4.0, 1.0], [2.0, 2.0]])
    kernel = phx.kernels.LinearKernel()
    matrix = kernel.matrix(left, right)

    assert kernel.input_ndim == 1
    assert kernel.max_derivative_order is None
    assert kernel.kernel_id == "LinearKernel"
    assert jnp.allclose(matrix, left @ right.T)
    assert jnp.allclose(
        matrix,
        jax.vmap(lambda x: jax.vmap(lambda y: kernel.pairwise(x, y))(right))(left),
    )
    assert jnp.allclose(kernel.diagonal(left), jnp.sum(left * left, axis=1))


def test_path_input_transform_builds_exact_truncated_signature_gram():
    paths = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
        ]
    )
    features = phx.stochastic.SignatureFeatures(2, 3, include_scalar=True)
    kernel = phx.kernels.InputTransformedKernel(
        phx.kernels.LinearKernel(),
        features,
        transform_id=features.feature_id,
        input_ndim=2,
        max_derivative_order=0,
    )
    feature_matrix = jax.vmap(features)(paths)

    assert kernel.input_ndim == 2
    assert jnp.allclose(kernel.matrix(paths, paths), feature_matrix @ feature_matrix.T)
    assert jnp.allclose(
        kernel.diagonal(paths),
        jnp.sum(feature_matrix * feature_matrix, axis=1),
    )
    assert jnp.allclose(
        kernel.pairwise(paths[0], paths[1]), feature_matrix[0] @ feature_matrix[1]
    )


def test_normalized_kernel_has_checked_exact_unit_diagonal():
    points = jnp.asarray([[1.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    kernel = phx.kernels.NormalizedKernel(phx.kernels.LinearKernel())
    matrix = kernel.matrix(points, points)

    assert kernel.input_ndim == 1
    assert kernel.is_unit_diagonal
    assert jnp.allclose(jnp.diag(matrix), 1.0)
    assert jnp.allclose(kernel.diagonal(points), jnp.ones((3,)))
    assert kernel.kernel_id == "NormalizedKernel[LinearKernel]"

    with pytest.raises(eqx.EquinoxRuntimeError, match="strictly positive"):
        invalid = eqx.filter_jit(kernel.diagonal)(jnp.zeros((2, 2)))
        jax.block_until_ready(invalid)


def test_kernel_algebra_rejects_mixed_input_ranks_and_propagates_path_rank():
    features = phx.stochastic.SignatureFeatures(2, 2)
    path_kernel = phx.kernels.InputTransformedKernel(
        phx.kernels.LinearKernel(),
        features,
        transform_id=features.feature_id,
        input_ndim=2,
    )

    assert (2.0 * path_kernel).input_ndim == 2
    assert (path_kernel * path_kernel).input_ndim == 2
    assert phx.kernels.NormalizedKernel(path_kernel).input_ndim == 2
    with pytest.raises(ValueError, match="equal input_ndim"):
        path_kernel + phx.kernels.LinearKernel()
    with pytest.raises(ValueError, match="equal input_ndim"):
        path_kernel * phx.kernels.LinearKernel()


def test_input_transform_validates_declared_input_rank():
    transform = phx.kernels.InputTransformedKernel(
        phx.kernels.LinearKernel(),
        lambda path: jnp.sum(path, axis=0),
        transform_id="path-sum",
        input_ndim=2,
    )
    with pytest.raises(ValueError, match="2 nonempty input axes"):
        transform.pairwise(jnp.ones((2,)), jnp.ones((2,)))
    with pytest.raises(ValueError, match="design axis followed by 2"):
        transform.matrix(jnp.ones((3, 2)), jnp.ones((3, 2)))
