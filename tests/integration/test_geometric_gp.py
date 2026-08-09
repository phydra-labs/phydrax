#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _cycle_graph(permutation=None):
    undirected = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    directed = np.concatenate((undirected, undirected[:, ::-1]), axis=0)
    if permutation is not None:
        inverse = np.argsort(np.asarray(permutation))
        directed = inverse[directed]
    return phx.graph.GraphIR(
        nodes=jnp.zeros((4, 1)),
        edges={"conductance": jnp.ones((8,))},
        senders=jnp.asarray(directed[:, 0]),
        receivers=jnp.asarray(directed[:, 1]),
        n_node=jnp.asarray([4]),
        n_edge=jnp.asarray([8]),
    )


def _basis(graph):
    complex_ir = phx.graph.graph_to_cochain_complex(
        graph,
        edge_weight_key="conductance",
        node_measure="uniform",
    )
    return phx.graph.cochain_laplacian_eigenbasis(
        complex_ir,
        0,
        num_modes=3,
    )


def _kernel(basis, length_scale=0.7, smoothness=1.4, amplitude=0.8):
    correlation = phx.kernels.SpectralFeatureKernel(
        basis,
        phx.kernels.MaternSpectralMultiplier(length_scale, smoothness),
    )
    return phx.kernels.AmplitudeKernel(correlation, amplitude)


def test_graph_spectral_gp_matches_dense_inference_and_is_permutation_equivariant():
    basis = _basis(_cycle_graph())
    latent = jnp.asarray([0.0, 1.0, 0.0, -1.0])
    observation_entities = jnp.tile(jnp.arange(4), 3)
    noise_scale = 0.08
    noise = noise_scale * jax.random.normal(jax.random.key(17), (12,))
    observations = latent[observation_entities] + noise
    kernel = _kernel(basis)
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=noise_scale,
    )
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        observation_entities,
        observations,
    )
    finite = model.factor(state=state)
    dense = phx.uq.ExactGaussianProcessFactor(observation_entities, state=state)
    residual = model.residual(jnp.zeros_like(observations))
    query = jnp.arange(4)
    finite_condition = finite.condition(residual, query)
    dense_condition = dense.condition(residual, query)

    assert isinstance(finite, phx.uq.FiniteFeatureGaussianProcessFactor)
    assert finite.factor_storage_elements < dense.factor_storage_elements
    assert jnp.allclose(
        finite.log_probability(residual),
        dense.log_probability(residual),
        rtol=1e-9,
        atol=1e-9,
    )
    assert jnp.allclose(finite_condition.mean, dense_condition.mean, atol=1e-9)
    assert jnp.allclose(
        finite_condition.covariance,
        dense_condition.covariance,
        rtol=1e-8,
        atol=1e-8,
    )
    assert jnp.all(finite_condition.variance <= kernel.diagonal(query) + 1e-10)
    assert jnp.mean((finite_condition.mean - latent) ** 2) < jnp.mean(latent**2)

    def likelihood(parameters):
        length_scale, smoothness, amplitude, candidate_noise = parameters
        candidate_state = phx.uq.GaussianProcessLikelihoodState(
            kernel=_kernel(basis, length_scale, smoothness, amplitude),
            noise_scale=candidate_noise,
        )
        return model.log_marginal_likelihood(
            jnp.zeros_like(observations),
            state=candidate_state,
        )

    value, gradient = jax.jit(jax.value_and_grad(likelihood))(
        jnp.asarray([0.7, 1.4, 0.8, noise_scale])
    )
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))

    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted_basis = _basis(_cycle_graph(permutation))
    permuted_entities = jnp.asarray(inverse)[observation_entities]
    permuted_state = phx.uq.GaussianProcessLikelihoodState(
        kernel=_kernel(permuted_basis),
        noise_scale=noise_scale,
    )
    permuted_model = phx.uq.ExactGaussianProcessDiscrepancy(
        permuted_entities,
        observations,
    )
    permuted_condition = permuted_model.factor(state=permuted_state).condition(
        permuted_model.residual(jnp.zeros_like(observations)),
        jnp.arange(4),
    )
    assert jnp.allclose(
        permuted_condition.mean,
        finite_condition.mean[jnp.asarray(permutation)],
        atol=1e-8,
    )
    assert jnp.allclose(
        permuted_condition.covariance,
        finite_condition.covariance[jnp.ix_(permutation, permutation)],
        atol=1e-8,
    )
