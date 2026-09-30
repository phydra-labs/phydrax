#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _path_basis() -> Any:
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((3, 1)),
        edges={"conductance": jnp.ones((4,))},
        senders=jnp.asarray([0, 1, 1, 2]),
        receivers=jnp.asarray([1, 0, 2, 1]),
        n_node=jnp.asarray([3]),
        n_edge=jnp.asarray([4]),
    )
    complex_ir = phx.graph.graph_to_cochain_complex(graph, edge_weight_key="conductance")
    return phx.exterior.hodge_laplacian_eigenbasis(
        complex_ir,
        0,
        count=None,
    )


def test_spectral_scenario_1() -> None:
    basis = _path_basis()
    kernel = phx.kernels.SpectralFeatureKernel(
        basis,
        phx.kernels.MaternSpectralMultiplier(0.7, 1.5),
    )
    entities = jnp.arange(3)
    features = kernel.features(entities)
    matrix = kernel.matrix(entities, entities)

    assert jnp.allclose(matrix, features @ features.T)
    assert jnp.allclose(kernel.diagonal(entities), jnp.diag(matrix))
    assert jnp.allclose(
        jnp.sum(basis.probability_measure * kernel.diagonal(entities)),
        1.0,
    )
    assert kernel.feature_rank == 3
    assert kernel.max_derivative_order == 0
    kernel = phx.kernels.SpectralFeatureKernel(
        _path_basis(), phx.kernels.HeatSpectralMultiplier(0.2)
    )

    for invalid in (
        jnp.asarray([0.5]),
        jnp.asarray([-1.0]),
        jnp.asarray([3.0]),
    ):
        with pytest.raises(Exception):
            kernel.features(invalid)
    with pytest.raises(Exception):
        kernel.features(jnp.asarray([jnp.inf]))
    with pytest.raises(ValueError):
        kernel.pairwise(jnp.asarray([0, 1]), 0)
    basis = _path_basis()
    multiplier = phx.kernels.HeatSpectralMultiplier(0.4)
    kernel = phx.kernels.SpectralFeatureKernel(basis, multiplier, normalize=False)
    expected_mass = jnp.sum(jnp.exp(-0.4 * basis.eigenvalues))

    assert jnp.allclose(
        jnp.sum(basis.probability_measure * kernel.diagonal(jnp.arange(3))),
        expected_mass,
    )
    basis = _path_basis()
    length_scale = 0.8
    matern = phx.kernels.MaternSpectralMultiplier(length_scale, 1e8)
    heat = phx.kernels.HeatSpectralMultiplier(0.5 * length_scale**2)

    assert jnp.allclose(
        matern.log_weights(basis.eigenvalues, basis.spectral_dimension),
        heat.log_weights(basis.eigenvalues, basis.spectral_dimension),
        rtol=1e-7,
        atol=1e-9,
    )
    multiplier = phx.kernels.MaternSpectralMultiplier(1.0, 1.0)
    actual = multiplier.log_weights(jnp.asarray([1], dtype=jnp.int32), 1.5)
    expected = -1.75 * jnp.log(1.5)

    assert jnp.allclose(actual, expected)


def test_spectral_hyperparameter_gradients_and_jit_are_finite() -> None:
    basis = _path_basis()
    entities = jnp.asarray([0, 1, 1, 2])

    def objective(length_scale: Any, smoothness: Any) -> Any:
        kernel = phx.kernels.SpectralFeatureKernel(
            basis,
            phx.kernels.MaternSpectralMultiplier(length_scale, smoothness),
        )
        return jnp.sum(kernel.matrix(entities, entities))

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        jnp.asarray(0.7), jnp.asarray(1.3)
    )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))


def test_spectral_scenario_2() -> None:
    basis = _path_basis()
    multiplier = phx.kernels.HeatSpectralMultiplier(0.3)
    entities = jnp.arange(3)
    reference = phx.kernels.SpectralFeatureKernel(basis, multiplier).matrix(
        entities, entities
    )

    permutation = np.asarray([2, 0, 1])
    permuted_basis = phx.discretization.SpectralDecomposition(
        basis.eigenvalues,
        basis.eigenfunctions[permutation],
        basis.probability_measure[permutation],
        spectral_dimension=basis.spectral_dimension,
        decomposition_id="permuted-path",
    )
    permuted = phx.kernels.SpectralFeatureKernel(permuted_basis, multiplier).matrix(
        entities, entities
    )
    assert jnp.allclose(permuted, reference[jnp.ix_(permutation, permutation)])

    signed_basis = phx.discretization.SpectralDecomposition(
        basis.eigenvalues,
        basis.eigenfunctions * jnp.asarray([1.0, -1.0, -1.0]),
        basis.probability_measure,
        spectral_dimension=basis.spectral_dimension,
        decomposition_id="signed-path",
    )
    signed = phx.kernels.SpectralFeatureKernel(signed_basis, multiplier).matrix(
        entities, entities
    )
    assert jnp.allclose(signed, reference)
    values = jnp.asarray([0.0, 2.0, 2.0, 4.0])
    functions = 2.0 * jnp.asarray(
        [
            [0.5, 1.0 / jnp.sqrt(2.0), 0.0, 0.5],
            [0.5, 0.0, 1.0 / jnp.sqrt(2.0), -0.5],
            [0.5, -1.0 / jnp.sqrt(2.0), 0.0, 0.5],
            [0.5, 0.0, -1.0 / jnp.sqrt(2.0), -0.5],
        ]
    )
    measure = jnp.full((4,), 0.25)
    basis = phx.discretization.SpectralDecomposition(
        values,
        functions,
        measure,
        spectral_dimension=1.0,
        decomposition_id="cycle",
    )
    angle = 0.37
    rotation = jnp.asarray(
        [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
    )
    rotated_functions = functions.at[:, 1:3].set(functions[:, 1:3] @ rotation)
    rotated = phx.discretization.SpectralDecomposition(
        values,
        rotated_functions,
        measure,
        spectral_dimension=1.0,
        decomposition_id="rotated-cycle",
    )
    multiplier = phx.kernels.MaternSpectralMultiplier(0.6, 1.2)

    first = phx.kernels.SpectralFeatureKernel(basis, multiplier).matrix(
        jnp.arange(4), jnp.arange(4)
    )
    second = phx.kernels.SpectralFeatureKernel(rotated, multiplier).matrix(
        jnp.arange(4), jnp.arange(4)
    )
    assert jnp.allclose(first, second, atol=1e-10)
    basis = _path_basis()
    for multiplier in (
        phx.kernels.HeatSpectralMultiplier(1e6),
        phx.kernels.MaternSpectralMultiplier(1e-8, 1e-8),
        phx.kernels.MaternSpectralMultiplier(1e4, 1e8),
    ):
        kernel = phx.kernels.SpectralFeatureKernel(basis, multiplier)
        assert jnp.all(jnp.isfinite(kernel.matrix(jnp.arange(3), jnp.arange(3))))
