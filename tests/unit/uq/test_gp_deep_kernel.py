#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_deep_kernel_likelihood_and_conditioning_preserve_feature_gradients():
    coordinate = jnp.linspace(-1.0, 1.0, 12)
    points = jnp.stack((coordinate, coordinate**2), axis=1)
    observations = 0.7 * coordinate + 0.1 * jnp.sin(3.0 * coordinate)
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    feature_map = eqx.nn.MLP(
        in_size=2,
        out_size=3,
        width_size=5,
        depth=2,
        activation=jnp.tanh,
        key=jr.key(41),
    )

    def state(candidate):
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.InputTransformedKernel(
                    phx.kernels.SquaredExponentialKernel(length_scale=jnp.ones((3,))),
                    candidate,
                    transform_id="learned-features",
                    max_derivative_order=None,
                ),
                0.2,
            ),
            noise_scale=0.03,
        )

    def objective(candidate):
        return model.log_marginal_likelihood(
            0.7 * coordinate,
            state=state(candidate),
        )

    value, gradient = eqx.filter_value_and_grad(objective)(feature_map)
    compiled = eqx.filter_jit(objective)(feature_map)
    gradient_leaves = jax.tree.leaves(eqx.filter(gradient, eqx.is_array))
    query_coordinate = jnp.linspace(-0.9, 0.9, 7)
    query = jnp.stack((query_coordinate, query_coordinate**2), axis=1)
    condition = model.condition(
        0.7 * coordinate,
        query,
        state=state(feature_map),
        output_dim="query",
    )

    assert jnp.allclose(value, compiled)
    assert gradient_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in gradient_leaves)
    assert condition.mean.shape == (7,)
    assert condition.output_dims == ("query",)
    assert jnp.all(condition.variance >= 0.0)
