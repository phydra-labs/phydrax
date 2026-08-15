#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch, SparseFeatures
from phydrax.ml.manifold import build_neighbor_graph, LocallyLinearEmbeddingRecipe


def test_neighbor_graph_has_fixed_sparse_capacity_deterministic_ties_and_case_diagnostics():
    tied = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    features = jnp.stack((tied, tied + jnp.array([0.0, 1.0])), axis=0)
    active = jnp.array([[True, True, True, True, True], [True, True, False, True, True]])

    first = build_neighbor_graph(features, active, n_neighbors=2)
    second = build_neighbor_graph(features, active, n_neighbors=2)

    assert first.relation.route_shape == (2, 5, 2)
    assert first.relation.target_shape == (5,)
    assert first.relation.width == 2
    assert first.relation.capacity == 20
    assert first.distances.shape == (2, 5, 2)
    assert first.adjacency.shape == (2, 5, 5)
    assert first.components.shape == (2,)
    assert first.minimum_degree.shape == (2,)
    assert first.maximum_degree.shape == (2,)
    assert jnp.array_equal(first.relation.source_indices, second.relation.source_indices)
    assert jnp.array_equal(first.relation.valid, second.relation.valid)
    assert jnp.allclose(first.distances, second.distances)
    assert jnp.any(first.distances[0, :2] == 0.0)
    assert not jnp.any(first.relation.valid[1, 2])
    assert jnp.all(first.distances[1, 2] == 0.0)
    assert jnp.array_equal(first.adjacency, jnp.swapaxes(first.adjacency, -1, -2))
    assert not jnp.any(jnp.diagonal(first.adjacency, axis1=-2, axis2=-1))
    degree = jnp.sum(first.adjacency, axis=-1)
    expected_minimum = jnp.min(jnp.where(active, degree, features.shape[-2]), axis=-1)
    expected_maximum = jnp.max(jnp.where(active, degree, 0), axis=-1)
    assert jnp.array_equal(first.minimum_degree, expected_minimum)
    assert jnp.array_equal(first.maximum_degree, expected_maximum)
    assert first.metric == "euclidean"
    assert "topology" in first.topology_gradient


def test_neighbor_graph_reports_disconnected_components_and_keeps_edge_length_gradients():
    disconnected = jnp.array([[0.0], [1.0], [10.0], [11.0]])
    graph = build_neighbor_graph(disconnected, jnp.ones((4,), dtype=bool), n_neighbors=1)

    assert graph.components == 2
    assert graph.minimum_degree == 1
    assert graph.maximum_degree == 1

    differentiable = jnp.array([[0.0, 0.0], [1.0, 0.2], [2.2, -0.1], [3.7, 0.4]])
    gradient = jax.grad(
        lambda value: jnp.sum(
            build_neighbor_graph(
                value, jnp.ones((4,), dtype=bool), n_neighbors=1
            ).distances
        )
    )(differentiable)
    assert gradient.shape == differentiable.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_neighbor_graph_rejects_precomputed_geometry_and_invalid_capacity():
    coordinates = jnp.eye(4)
    active = jnp.ones((4,), dtype=bool)
    invalid_metric: Any = "precomputed"

    with pytest.raises(ValueError, match="Unsupported metric"):
        build_neighbor_graph(coordinates, active, n_neighbors=2, metric=invalid_metric)
    with pytest.raises(ValueError, match="n_neighbors"):
        build_neighbor_graph(coordinates, active, n_neighbors=4)


def test_sparse_features_fail_explicitly_without_implicit_densification():
    dense = jnp.array(
        [[-2.0, 0.0], [-1.0, 0.5], [0.0, -0.2], [1.0, 0.4], [2.0, -0.1], [3.0, 0.7]]
    )
    sparse = SparseFeatures(
        dense,
        jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), dense.shape),
        feature_count=2,
    )

    with pytest.raises(TypeError, match="requires dense features"):
        LocallyLinearEmbeddingRecipe(1, n_neighbors=2).fit_batch(MLBatch(sparse))
