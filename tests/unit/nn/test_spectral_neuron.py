#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _svec(matrix):
    value = jnp.asarray(matrix)
    rows, columns = jnp.triu_indices(value.shape[-1])
    scale = jnp.where(rows == columns, 1.0, jnp.sqrt(2.0)).astype(value.dtype)
    return value[..., rows, columns] * scale


def _pack_diagonal_factor(diagonal):
    value = jnp.asarray(diagonal)
    factor = jnp.diag(jnp.sqrt(value))
    rows, columns = jnp.tril_indices(value.shape[-1])
    return factor[rows, columns]


def _deterministic_neuron(eigen_index=1):
    model = phx.nn.layers.SpectralNeuron(
        in_size=3,
        matrix_size=3,
        eigen_index=eigen_index,
        monotonicity=("free", "increasing", "decreasing"),
        initialization_radius=2.0,
        dtype=jnp.float64,
        key=jr.key(0),
    )
    base = jnp.diag(jnp.asarray([-1.0, 0.2, 2.0]))
    free = jnp.asarray([[0.2, 0.1, 0.0], [0.1, -0.1, 0.05], [0.0, 0.05, 0.3]])
    increasing = jnp.diag(jnp.asarray([0.5, 0.2, 0.1]))
    decreasing_magnitude = jnp.diag(jnp.asarray([0.1, 0.4, 0.3]))
    model = eqx.tree_at(
        lambda item: (
            item.free_coordinates,
            item.increasing_factor_coordinates,
            item.decreasing_factor_coordinates,
        ),
        model,
        (
            jnp.stack((_svec(base), _svec(free))),
            _pack_diagonal_factor(jnp.diag(increasing))[None, :],
            _pack_diagonal_factor(jnp.diag(decreasing_magnitude))[None, :],
        ),
    )
    return model, base, jnp.stack((free, increasing, -decreasing_magnitude))


def test_spectral_neuron_matches_dense_reference_and_leading_axes():
    model, base, features = _deterministic_neuron()
    point = jnp.asarray([0.4, -0.2, 0.7])
    expected_matrix = base + jnp.sum(point[:, None, None] * features, axis=0)
    expected_eigenvalues = jnp.linalg.eigvalsh(expected_matrix)

    np.testing.assert_allclose(model.matrix_pencil(point), expected_matrix, atol=2e-14)
    np.testing.assert_allclose(model.eigenvalues(point), expected_eigenvalues, atol=2e-14)
    np.testing.assert_allclose(model(point), expected_eigenvalues[1], atol=2e-14)
    materialized_base, materialized_features = model.materialize_coefficients()
    np.testing.assert_allclose(materialized_base, base, atol=2e-14)
    np.testing.assert_allclose(materialized_features, features, atol=2e-14)

    points = jnp.stack((point, -point, 0.5 * point))
    expected = jnp.stack(
        [
            jnp.linalg.eigvalsh(base + jnp.sum(x[:, None, None] * features, axis=0))[1]
            for x in points
        ]
    )
    np.testing.assert_allclose(model(points), expected, atol=3e-14)
    np.testing.assert_allclose(jax.vmap(model)(points), expected, atol=3e-14)


def test_spectral_neuron_constraints_and_extremal_shape_guarantees():
    model, _, features = _deterministic_neuron()
    for coefficient in features:
        np.testing.assert_allclose(coefficient, coefficient.T, atol=0.0)
    assert float(jnp.min(jnp.linalg.eigvalsh(features[1]))) >= -1e-14
    assert float(jnp.max(jnp.linalg.eigvalsh(features[2]))) <= 1e-14

    point = jnp.asarray([0.2, -0.3, 0.4])
    increasing_point = point.at[1].add(0.7)
    decreasing_point = point.at[2].add(0.7)
    assert float(model(increasing_point)) >= float(model(point)) - 2e-14
    assert float(model(decreasing_point)) <= float(model(point)) + 2e-14

    convex, _, _ = _deterministic_neuron(eigen_index=2)
    concave, _, _ = _deterministic_neuron(eigen_index=0)
    left = jnp.asarray([-0.7, 0.1, -0.2])
    right = jnp.asarray([0.4, 0.8, 0.5])
    weight = 0.35
    mixture = weight * left + (1.0 - weight) * right
    convex_chord = weight * convex(left) + (1.0 - weight) * convex(right)
    concave_chord = weight * concave(left) + (1.0 - weight) * concave(right)
    assert float(convex(mixture)) <= float(convex_chord) + 3e-14
    assert float(concave(mixture)) >= float(concave_chord) - 3e-14
    assert convex.is_convex and not convex.is_concave
    assert concave.is_concave and not concave.is_convex


def test_spectral_neuron_initializer_is_keyed_and_certifies_declared_box():
    arguments = dict(
        in_size=2,
        matrix_size=3,
        eigen_index=1,
        monotonicity=("increasing", "decreasing"),
        initialization_radius=2.0,
        dtype=jnp.float64,
    )
    first = phx.nn.layers.SpectralNeuron(**arguments, key=jr.key(4))
    repeated = phx.nn.layers.SpectralNeuron(**arguments, key=jr.key(4))
    different = phx.nn.layers.SpectralNeuron(**arguments, key=jr.key(5))
    np.testing.assert_allclose(
        first.free_coordinates, repeated.free_coordinates, atol=0.0
    )
    np.testing.assert_allclose(
        first.increasing_factor_coordinates,
        repeated.increasing_factor_coordinates,
        atol=0.0,
    )
    assert not np.array_equal(first.free_coordinates, different.free_coordinates)

    report = first.initialization
    assert report.initialization_radius == 2.0
    assert report.origin_gap > 0.0
    assert report.certified_minimum_gap > 0.0
    minimum_observed = np.inf
    for coordinates in itertools.product((-2.0, 2.0), repeat=2):
        spectrum = np.asarray(first.eigenvalues(jnp.asarray(coordinates)))
        gap = min(spectrum[1] - spectrum[0], spectrum[2] - spectrum[1])
        minimum_observed = min(minimum_observed, float(gap))
    assert minimum_observed + 2e-13 >= report.certified_minimum_gap

    _, features = first.materialize_coefficients()
    assert float(jnp.min(jnp.linalg.eigvalsh(features[0]))) >= -2e-14
    assert float(jnp.max(jnp.linalg.eigvalsh(features[1]))) <= 2e-14


def test_spectral_neuron_jit_vmap_and_grad_are_finite_away_from_crossings():
    model, _, _ = _deterministic_neuron()
    point = jnp.asarray([0.3, -0.25, 0.15])
    points = jnp.stack((point, point + 0.1))
    np.testing.assert_allclose(jax.jit(model)(point), model(point), atol=3e-14)
    np.testing.assert_allclose(jax.vmap(model)(points), model(points), atol=3e-14)
    input_gradient = jax.grad(model)(point)
    assert bool(jnp.all(jnp.isfinite(input_gradient)))

    _, parameter_gradient = eqx.filter_value_and_grad(lambda item: item(point))(model)
    array_gradients = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_array(leaf)
    ]
    assert array_gradients
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in array_gradients)


def test_one_dimensional_spectral_neuron_reduces_to_affine_model():
    model = phx.nn.layers.SpectralNeuron(
        in_size=2,
        matrix_size=1,
        eigen_index=0,
        dtype=jnp.float64,
        key=jr.key(7),
    )
    point = jnp.asarray([0.4, -0.6])
    base, features = model.materialize_coefficients()
    expected = base[0, 0] + jnp.sum(point * features[:, 0, 0])
    np.testing.assert_allclose(model(point), expected, atol=2e-14)
    np.testing.assert_allclose(jax.grad(model)(point), features[:, 0, 0], atol=2e-14)
    assert model.is_convex and model.is_concave
    assert np.isinf(model.initialization.certified_minimum_gap)


def test_spectral_neuron_rejects_invalid_contracts_and_complex_inputs():
    kwargs = dict(in_size=2, matrix_size=3, eigen_index=1, key=jr.key(0))
    with pytest.raises(TypeError, match="matrix_size"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"matrix_size": 2.5}))
    with pytest.raises(TypeError, match="eigen_index"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"eigen_index": True}))
    with pytest.raises(ValueError, match="matrix_size"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"matrix_size": 0}))
    with pytest.raises(ValueError, match="eigen_index"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"eigen_index": -1}))
    with pytest.raises(ValueError, match="eigen_index"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"eigen_index": 3}))
    with pytest.raises(ValueError, match="one entry"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"monotonicity": ("free",)}))
    with pytest.raises(ValueError, match="monotonicity entries"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"monotonicity": ("free", "sideways")}))
    with pytest.raises(ValueError, match="initialization_radius"):
        phx.nn.layers.SpectralNeuron(**(kwargs | {"initialization_radius": 0.0}))

    model = phx.nn.layers.SpectralNeuron(**kwargs)
    with pytest.raises(TypeError, match="real-valued"):
        model(jnp.asarray([1.0 + 0.0j, 2.0 + 0.0j]))
    assert bool(jnp.isfinite(model(jnp.asarray([1, 0], dtype=jnp.int32))))
    assert bool(jnp.isfinite(model(jnp.asarray([True, False]))))
