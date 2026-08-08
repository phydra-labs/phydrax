#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _coregionalization(output_names):
    return phx.uq.Coregionalization(
        jnp.array([[0.8, 0.0], [-0.3, 0.6]]),
        jnp.array([0.1, 0.2]),
        output_names=output_names,
    )


def test_heterotopic_design_is_exact_subselection_of_dense_icm_covariance():
    output_names = ("temperature", "flux")
    points = jnp.linspace(0.0, 1.0, 6)
    mask = jnp.array(
        [
            [True, True],
            [True, False],
            [False, True],
            [True, True],
            [False, True],
            [True, False],
        ]
    )
    dense_design = phx.uq.MultiOutputDesign.from_dense(
        points,
        output_names=output_names,
    )
    heterotopic = phx.uq.MultiOutputDesign.from_dense(
        points,
        output_names=output_names,
        mask=mask,
    )
    kernel = phx.uq.IntrinsicCoregionalizationKernel(
        phx.kernels.Matern52Kernel(length_scale=0.3),
        _coregionalization(output_names),
    )
    active = jnp.flatnonzero(mask.reshape((-1,)))
    dense_covariance = kernel.matrix(dense_design, dense_design)
    heterotopic_covariance = kernel.matrix(heterotopic, heterotopic)
    values = jnp.arange(12.0).reshape((6, 2))

    assert jnp.allclose(
        heterotopic_covariance,
        dense_covariance[active[:, None], active[None, :]],
    )
    assert jnp.array_equal(heterotopic.flatten(values), values.reshape((-1,))[active])
    reconstructed = heterotopic.dense(heterotopic.flatten(values))
    assert jnp.array_equal(reconstructed[mask], values[mask])
    assert jnp.all(jnp.isnan(reconstructed[~mask]))


def test_lmc_heterotopic_likelihood_supports_observation_noise_and_gradients():
    output_names = ("u", "v")
    points = jnp.linspace(0.0, 1.0, 8)
    observations = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * points),
            0.5 * jnp.cos(2.0 * jnp.pi * points),
        ),
        axis=1,
    )
    mask = jnp.ones_like(observations, dtype=bool)
    mask = mask.at[1, 1].set(False).at[6, 0].set(False)
    model = phx.uq.MultiOutputGaussianProcessDiscrepancy.from_dense(
        points,
        observations,
        output_names=output_names,
        mask=mask,
    )
    query = phx.uq.MultiOutputDesign.from_dense(
        jnp.linspace(0.05, 0.95, 5),
        output_names=output_names,
        mask=jnp.array(
            [[True, True], [True, False], [False, True], [True, True], [True, False]]
        ),
    )

    def objective(weight):
        first = phx.uq.Coregionalization(
            weight,
            jnp.array([0.1, 0.15]),
            output_names=output_names,
        )
        second = _coregionalization(output_names)
        kernel = phx.uq.LinearModelCoregionalizationKernel(
            (
                (phx.kernels.Matern32Kernel(length_scale=0.2), first),
                (
                    phx.kernels.SquaredExponentialKernel(length_scale=0.6),
                    second,
                ),
            )
        )
        state = phx.uq.MultiOutputGaussianProcessLikelihoodState(
            kernel=kernel,
            noise_scale=jnp.linspace(0.02, 0.04, model.design.num_observations),
            noise_layout="observation",
        )
        return model.log_marginal_likelihood(jnp.zeros_like(observations), state=state)

    weight = jnp.array([[0.7, 0.0], [0.2, 0.5]])
    gradient = jax.grad(objective)(weight)
    first = phx.uq.Coregionalization(
        weight,
        jnp.array([0.1, 0.15]),
        output_names=output_names,
    )
    kernel = phx.uq.LinearModelCoregionalizationKernel(
        (
            (phx.kernels.Matern32Kernel(length_scale=0.2), first),
            (
                phx.kernels.SquaredExponentialKernel(length_scale=0.6),
                _coregionalization(output_names),
            ),
        )
    )
    state = phx.uq.MultiOutputGaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=jnp.array([0.02, 0.03]),
    )
    condition = model.condition(jnp.zeros_like(observations), query, state=state)

    assert jnp.isfinite(objective(weight))
    assert jnp.all(jnp.isfinite(gradient))
    assert condition.mean.shape == (query.num_observations,)
    assert condition.covariance.shape == (
        query.num_observations,
        query.num_observations,
    )
    assert jnp.all(condition.variance >= 0.0)
    assert kernel.max_derivative_order == 1
