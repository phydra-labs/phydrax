#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_affine_pullback_matches_manual_standardized_covariance():
    coordinate = jnp.linspace(-2.0, 2.0, 12)
    points = jnp.stack((coordinate, 0.5 * coordinate**2), axis=1)
    transform = phx.kernels.AffineInputTransform.from_points(points)
    base = phx.kernels.Matern52Kernel(length_scale=jnp.array([0.7, 1.2]))
    kernel = phx.kernels.InputTransformedKernel(
        base,
        transform,
        transform_id="standardize",
        max_derivative_order=None,
    )
    standardized = jax.vmap(transform)(points)

    assert jnp.allclose(jnp.mean(standardized, axis=0), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.std(standardized, axis=0), 1.0, atol=1e-12)
    assert jnp.allclose(
        kernel.matrix(points, points), base.matrix(standardized, standardized)
    )
    assert kernel.max_derivative_order == 2
    assert kernel.is_unit_diagonal


def test_deep_kernel_features_remain_dynamic_differentiable_pytree_leaves():
    points = jnp.stack(
        (jnp.linspace(-1.0, 1.0, 10), jnp.linspace(0.0, 2.0, 10)),
        axis=1,
    )
    feature_map = eqx.nn.MLP(
        in_size=2,
        out_size=3,
        width_size=6,
        depth=2,
        activation=jnp.tanh,
        key=jr.key(3),
    )
    base = phx.kernels.SquaredExponentialKernel(length_scale=jnp.array([0.8, 1.0, 1.2]))
    kernel = phx.kernels.InputTransformedKernel(
        base,
        feature_map,
        transform_id="tanh-mlp",
        max_derivative_order=None,
    )

    value, gradient = eqx.filter_value_and_grad(
        lambda candidate: jnp.sum(candidate.matrix(points, points) ** 2)
    )(kernel)
    gradient_leaves = jax.tree.leaves(
        eqx.filter(gradient.transform_function, eqx.is_array)
    )
    explicit_features = jax.vmap(feature_map)(points)

    assert jnp.isfinite(value)
    assert jnp.allclose(
        kernel.matrix(points, points), base.matrix(explicit_features, explicit_features)
    )
    assert gradient_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in gradient_leaves)
    assert kernel.max_derivative_order is None
