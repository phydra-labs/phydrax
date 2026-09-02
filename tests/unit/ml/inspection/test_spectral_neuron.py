#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _svec(matrix):
    value = jnp.asarray(matrix)
    rows, columns = jnp.triu_indices(value.shape[-1])
    scale = jnp.where(rows == columns, 1.0, jnp.sqrt(2.0)).astype(value.dtype)
    return value[..., rows, columns] * scale


def _free_neuron(base, features, *, eigen_index=0, in_size=None):
    feature_matrices = jnp.asarray(features)
    feature_count = int(feature_matrices.shape[0])
    model = phx.nn.layers.SpectralNeuron(
        in_size=feature_count if in_size is None else in_size,
        matrix_size=int(base.shape[-1]),
        eigen_index=eigen_index,
        dtype=jnp.float64,
        key=jr.key(0),
    )
    coordinates = jnp.concatenate(
        (_svec(jnp.asarray(base))[None, :], _svec(feature_matrices)), axis=0
    )
    return eqx.tree_at(lambda item: item.free_coordinates, model, coordinates)


def test_inspection_matches_simple_projector_gradient_and_global_bounds():
    base = jnp.diag(jnp.asarray([-1.0, 0.3, 2.0]))
    features = jnp.asarray(
        [
            [[0.2, 0.1, 0.0], [0.1, -0.1, 0.05], [0.0, 0.05, 0.3]],
            [[0.4, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.1]],
        ]
    )
    model = _free_neuron(base, features, eigen_index=1)
    point = jnp.asarray([0.4, -0.2])
    report = phx.ml.inspection.inspect_spectral_neuron(model, point)
    matrix = base + jnp.sum(point[:, None, None] * features, axis=0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    projector = eigenvectors[:, 1, None] * eigenvectors[:, 1, None].T
    expected_bounds = jnp.max(jnp.abs(jnp.linalg.eigvalsh(features)), axis=-1)
    expected_sensitivity = jax.grad(model)(point)

    assert bool(report.valid)
    assert bool(report.selected_is_numerically_simple)
    assert bool(report.local_sensitivity_valid)
    assert int(report.cluster_size) == 1
    np.testing.assert_array_equal(report.cluster_mask, jnp.asarray([False, True, False]))
    np.testing.assert_allclose(report.matrix, matrix, atol=2e-14)
    np.testing.assert_allclose(report.eigenvalues, eigenvalues, atol=2e-14)
    np.testing.assert_allclose(report.selected_eigenvalue, eigenvalues[1], atol=2e-14)
    np.testing.assert_allclose(report.cluster_projector, projector, atol=3e-14)
    np.testing.assert_allclose(
        report.local_sensitivities, expected_sensitivity, atol=3e-14
    )
    np.testing.assert_allclose(report.global_feature_bounds, expected_bounds, atol=2e-14)
    assert float(report.enclosure_lower) <= float(report.selected_eigenvalue) + 2e-14
    assert float(report.enclosure_upper) >= float(report.selected_eigenvalue) - 2e-14
    assert report.eigen_index == 1
    assert not report.convex and not report.concave


def test_inspection_reports_full_repeated_cluster_without_basis_leakage():
    base = jnp.diag(jnp.asarray([0.0, 0.0, 2.0]))
    features = jnp.asarray([jnp.diag(jnp.asarray([1.0, -1.0, 0.2]))])
    model = _free_neuron(base, features, eigen_index=0, in_size="scalar")
    report = phx.ml.inspection.inspect_spectral_neuron(model, jnp.asarray(0.0))

    assert bool(report.valid)
    assert not bool(report.selected_is_numerically_simple)
    assert not bool(report.local_sensitivity_valid)
    assert int(report.cluster_size) == 2
    assert int(report.cluster_lower_index) == 0
    assert int(report.cluster_upper_index) == 1
    np.testing.assert_array_equal(report.cluster_mask, jnp.asarray([True, True, False]))
    np.testing.assert_allclose(
        report.cluster_projector, jnp.diag(jnp.asarray([1.0, 1.0, 0.0])), atol=2e-14
    )
    assert bool(jnp.isnan(report.local_sensitivities[0]))
    np.testing.assert_allclose(
        report.local_sensitivity_bounds, jnp.asarray([1.0]), atol=2e-14
    )
    np.testing.assert_allclose(report.lower_gap, jnp.inf)
    np.testing.assert_allclose(report.upper_gap, 2.0)


def test_inspection_near_repetition_follows_declared_tolerance():
    separation = 5e-9
    base = jnp.diag(jnp.asarray([0.0, separation, 2.0]))
    features = jnp.zeros((1, 3, 3), dtype=jnp.float64)
    model = _free_neuron(base, features, eigen_index=0, in_size="scalar")

    merged = phx.ml.inspection.inspect_spectral_neuron(
        model,
        jnp.asarray(0.0),
        relative_tolerance=0.0,
        absolute_tolerance=1e-8,
    )
    separated = phx.ml.inspection.inspect_spectral_neuron(
        model,
        jnp.asarray(0.0),
        relative_tolerance=0.0,
        absolute_tolerance=1e-10,
    )
    assert int(merged.cluster_size) == 2
    assert not bool(merged.selected_is_numerically_simple)
    assert int(separated.cluster_size) == 1
    assert bool(separated.selected_is_numerically_simple)


def test_inspection_jit_preserves_fixed_shapes_across_variable_clusters():
    base = jnp.diag(jnp.asarray([0.0, 0.0, 2.0]))
    features = jnp.asarray([jnp.diag(jnp.asarray([-1.0, 1.0, 0.0]))])
    model = _free_neuron(base, features, eigen_index=0, in_size="scalar")
    points = jnp.asarray([0.0, 0.2])
    inspect = jax.jit(
        lambda value: phx.ml.inspection.inspect_spectral_neuron(model, value)
    )
    report = inspect(points)

    assert report.cluster_size.shape == (2,)
    assert report.cluster_mask.shape == (2, 3)
    assert report.cluster_projector.shape == (2, 3, 3)
    assert report.local_sensitivities.shape == (2, 1)
    np.testing.assert_array_equal(report.cluster_size, jnp.asarray([2, 1]))
    np.testing.assert_array_equal(
        report.selected_is_numerically_simple, jnp.asarray([False, True])
    )
