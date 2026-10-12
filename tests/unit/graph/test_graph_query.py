#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_graph_query_scenario_1() -> None:
    source = jnp.array([[0.0], [1.0], [3.0]])
    target = jnp.array([[0.2], [2.6]])

    bundle = phx.graph.radius_query_graph(
        source,
        target,
        radius=0.5,
        source_features=jnp.array([[1.0], [2.0], [3.0]]),
        weight_kind="hat",
    )
    graph = bundle.graph
    assert graph.senders is not None
    assert graph.receivers is not None

    assert graph.num_nodes == 5
    assert graph.num_edges == 2
    assert jnp.allclose(graph.senders, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.allclose(graph.receivers, jnp.array([3, 4], dtype=jnp.int32))
    assert jnp.allclose(graph.edges["relative"], jnp.array([[0.2], [-0.4]]), atol=1e-7)
    assert jnp.allclose(graph.edges["distance"], jnp.array([[0.2], [0.4]]), atol=1e-7)
    assert jnp.allclose(
        graph.edges["kernel_weight"], jnp.array([[0.6], [0.2]]), atol=1e-7
    )
    assert jnp.allclose(
        graph.nodes["features"][:, 0], jnp.array([1.0, 2.0, 3.0, 0.0, 0.0])
    )
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.9]]),
        jnp.array([[0.1]]),
        radius=0.25,
        periodic_box=1.0,
        weight_kind=None,
    )

    assert bundle.graph.num_edges == 1
    assert jnp.allclose(bundle.graph.edges["relative"], jnp.array([[0.2]]), atol=1e-7)
    assert jnp.allclose(bundle.graph.edges["distance"], jnp.array([[0.2]]), atol=1e-7)
    source = jnp.array([[0.0], [2.0], [5.0]])
    target = jnp.array([[1.0]])
    bundle = phx.graph.knn_query_graph(source, target, k=2, weight_kind=None)
    rebuilt = phx.graph.query_graph_from_edges(
        source,
        target + 1.0,
        bundle.graph.edges["source_index"],
        bundle.graph.edges["target_index"],
        weight_kind=None,
    )
    assert bundle.graph.senders is not None
    assert bundle.graph.receivers is not None

    assert jnp.allclose(bundle.graph.senders, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(bundle.graph.receivers, jnp.array([3, 3], dtype=jnp.int32))
    assert jnp.allclose(rebuilt.graph.edges["relative"], jnp.array([[2.0], [0.0]]))
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )
    domain = phx.domain.GraphDomain(bundle.graph, measure="count")
    structure = phx.domain.SampleLayout((("graph",),))
    sources = domain.component({"graph": bundle.source_nodes_component()})
    targets = domain.component({"graph": bundle.target_nodes_component()})
    query_edges = domain.component({"graph": bundle.query_edges_component()})

    source_batch = sources.sample(phx.domain.PointSampling(2, layout=structure))
    target_batch = targets.sample(phx.domain.PointSampling(1, layout=structure))
    edge_batch = query_edges.sample(phx.domain.PointSampling(2, layout=structure))

    assert jnp.allclose(
        source_batch["graph"]["features"].data[:, 0], jnp.array([1.0, 3.0])
    )
    assert jnp.allclose(target_batch["graph"]["features"].data[:, 0], jnp.array([0.0]))
    assert jnp.allclose(edge_batch["graph"]["distance"].data[:, 0], jnp.array([0.5, 0.5]))
    # ty: ignore[unresolved-attribute]
    assert targets.mass.value == 1.0
    neighborhood = phx.graph.query_neighbors(
        jnp.array([[0.25], [0.75]]),
        jnp.array([[0.0]]),
        max_neighbors=2,
        periodic_lengths=(1.0,),
    )

    assert jnp.array_equal(neighborhood.indices[0, 0], jnp.array([0, 1]))
    assert jnp.allclose(neighborhood.distance[0, 0], jnp.array([0.25, 0.25]))
    assert jnp.allclose(
        neighborhood.relative[0, 0, :, 0],
        jnp.array([-0.25, 0.25]),
    )


def test_graph_query_scenario_2() -> None:
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )
    out = phx.graph.GraphNeuralOperator(
        input_key="features",
        output_key="gno",
        edge_weight_key=None,
        normalize=False,
        target_node_type=bundle.target_type,
    )(bundle.graph)

    assert jnp.allclose(out.nodes["gno"][:, 0], jnp.array([0.0, 0.0, 4.0]))
    source = jnp.array(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
        ]
    )
    target = jnp.array([[[0.2], [1.8]], [[10.2], [11.8]]])
    source_mask = jnp.array([[True, True, True], [True, True, False]])
    target_mask = jnp.array([[True, True], [True, False]])
    query = phx.graph.batched_knn_query_graph(
        source,
        target,
        k=2,
        source_mask=source_mask,
        target_mask=target_mask,
        source_features=jnp.arange(6.0).reshape((2, 3, 1)),
        source_measure=jnp.ones((2, 3)),
    )
    graph = query.graph
    assert graph.senders is not None
    assert graph.receivers is not None
    assert graph.edge_mask is not None
    assert graph.node_mask is not None

    assert jnp.array_equal(graph.n_node, jnp.array([5, 5], dtype=jnp.int32))
    assert jnp.array_equal(graph.n_edge, jnp.array([4, 4], dtype=jnp.int32))
    assert jnp.all(graph.senders[:4] < 5)
    assert jnp.all(graph.receivers[:4] < 5)
    assert jnp.all(graph.senders[4:] >= 5)
    assert jnp.all(graph.receivers[4:] >= 5)
    assert jnp.array_equal(
        graph.edge_mask,
        jnp.array([True, True, True, True, True, True, False, False]),
    )
    assert jnp.array_equal(
        graph.node_mask,
        jnp.array([True, True, True, True, True, True, True, False, True, False]),
    )
    graph = phx.graph.batched_knn_graph(
        jnp.array([[[0.0], [1.0], [3.0]], [[10.0], [11.0], [13.0]]]),
        k=1,
    )
    assert graph.senders is not None
    assert graph.receivers is not None

    assert graph.senders.shape == (6,)
    assert jnp.all(graph.senders != graph.receivers)
    assert jnp.all(graph.senders[:3] < 3)
    assert jnp.all(graph.senders[3:] >= 3)
    points = jnp.array([[0.0], [1.0], [3.0], [6.0]])
    neighborhood = phx.graph.query_neighbors(
        points,
        points,
        max_neighbors=1,
        exclude_self=True,
        target_chunk_size=2,
    )
    assert jnp.array_equal(neighborhood.indices[0, :, 0], jnp.array([1, 0, 1, 2]))


def test_graph_neural_operator_wraps_as_graph_model_on_query_targets() -> None:
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )
    domain = phx.domain.GraphDomain(bundle.graph)
    targets = domain.component({"graph": bundle.target_nodes_component()})
    batch = targets.sample(
        phx.domain.PointSampling(1, layout=phx.domain.SampleLayout((("graph",),)))
    )

    @domain.Function("graph")
    def u(node: Any) -> Any:
        return node.get("features")[0]

    model = domain.GraphModel(
        phx.graph.GraphNeuralOperator(
            input_key="u",
            output_key="gno",
            edge_weight_key=None,
            normalize=False,
            target_node_type=bundle.target_type,
        ),
        input_fn=u,
        input_key="u",
        output_key="gno",
    )

    assert jnp.allclose(jnp.asarray(model(batch).data), jnp.array([4.0]))


def test_batched_query_graph_is_jittable_and_differentiable() -> None:
    source = jnp.array([[[0.0], [0.5], [1.0]]])
    target = jnp.array([[[0.2], [0.8]]])

    def distances(source_points: Any, target_points: Any) -> Any:
        graph = phx.graph.batched_knn_query_graph(
            source_points,
            target_points,
            k=2,
            source_measure=jnp.full((1, 3), 1.0 / 3.0),
            validate=False,
        ).graph
        return graph.edges["distance"]

    result = jax.jit(distances)(source, target)
    gradient = jax.grad(lambda points: jnp.sum(distances(source, points)))(target)

    assert result.shape == (4, 1)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(jnp.isfinite(gradient))


def test_query_neighbors_support_morton_execution_and_geometry_gradients() -> None:
    plan = phx.discretization.spatial.MortonNeighborQueryPlan(
        phx.discretization.spatial.MortonAddressPlan(
            (0.0,),
            (1.0,),
            16,
            periodic_axes=(True,),
        ),
        2,
        1,
        2,
    )

    def distances(target: Any) -> Any:
        return phx.graph.query_neighbors(
            jnp.array([[0.25], [0.75]]),
            target,
            periodic_lengths=(1.0,),
            plan=plan,
        )

    neighborhood = distances(jnp.array([[0.0]]))
    gradient = jax.grad(lambda target: jnp.sum(distances(target).distance))(
        jnp.array([[0.1]])
    )

    assert neighborhood.evidence.backend == "morton"
    assert bool(neighborhood.evidence.successful)
    assert jnp.array_equal(neighborhood.indices[0, 0], jnp.array([0, 1]))
    assert jnp.allclose(neighborhood.distance[0, 0], jnp.array([0.25, 0.25]))
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("radius", [None, 0.5], ids=["knn", "inclusive-radius"])
@pytest.mark.parametrize("source_tile", [1, 3, 8], ids=["singleton", "partial", "whole"])
def test_blocked_neighbors_match_independent_masked_periodic_reference(
    radius: float | None,
    source_tile: int,
) -> None:
    source = np.asarray([[0.0], [0.5], [0.25], [0.75], [0.0]], dtype=np.float64)
    target = source.copy()
    source_mask = np.asarray([True, True, False, True, True], dtype=np.bool_)
    target_mask = np.asarray([True, False, True, True, True], dtype=np.bool_)
    relative = target[:, None, :] - source[None, :, :]
    relative -= np.round(relative)
    squared = np.sum(relative * relative, axis=-1)
    valid = source_mask[None, :] & target_mask[:, None] & ~np.eye(5, dtype=np.bool_)
    if radius is not None:
        valid &= squared <= radius * radius
    indices = np.argsort(np.where(valid, squared, np.inf), axis=-1, kind="stable")[:, :3]
    expected_valid = np.take_along_axis(valid, indices, axis=1)
    expected_squared = np.where(
        expected_valid,
        np.take_along_axis(squared, indices, axis=1),
        0,
    )
    result = phx.graph.query_neighbors(
        source,
        target,
        source_mask=source_mask,
        target_mask=target_mask,
        max_neighbors=3,
        radius=radius,
        periodic_lengths=(1.0,),
        exclude_self=True,
        source_chunk_size=source_tile,
        target_chunk_size=2,
    )
    np.testing.assert_array_equal(result.indices[0], indices)
    np.testing.assert_array_equal(result.mask[0], expected_valid)
    np.testing.assert_array_equal(result.distance_squared[0], expected_squared)
    np.testing.assert_array_equal(result.count[0], expected_valid.sum(axis=-1))
    np.testing.assert_allclose(
        result.relative[0],
        np.where(
            expected_valid[..., None],
            np.take_along_axis(
                relative,
                indices[..., None],
                axis=1,
            ),
            0,
        ),
    )
    assert result.evidence.backend == "blocked"
    assert bool(result.evidence.successful & result.evidence.complete)


@pytest.mark.parametrize(
    "kind", ["radius", "knn"], ids=["inclusive-radius", "stable-knn"]
)
def test_host_query_edges_match_brute_force_across_source_tiles(kind: str) -> None:
    source = np.linspace(-1.0, 1.0, 261, dtype=np.float64)[:, None]
    source[0] = source[260] = 0.5
    target = np.asarray([[0.0], [0.5], [-0.5]], dtype=np.float64)
    distance = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    if kind == "radius":
        target_ids, source_ids = np.nonzero(distance <= 0.5)
        bundle = phx.graph.radius_query_graph(
            source, target, radius=0.5, weight_kind=None
        )
    else:
        source_ids = np.argsort(distance, axis=-1, kind="stable")[:, :4].reshape(-1)
        target_ids = np.repeat(np.arange(3, dtype=np.int32), 4)
        bundle = phx.graph.knn_query_graph(source, target, k=4)
    np.testing.assert_array_equal(bundle.graph.edges["source_index"], source_ids)
    np.testing.assert_array_equal(bundle.graph.edges["target_index"], target_ids)


def test_graph_query_morton_overflow_refuses_neighbors() -> None:
    source = jnp.full((8, 1), 0.25, dtype=jnp.float64)
    plan = phx.discretization.spatial.MortonNeighborQueryPlan(
        phx.discretization.spatial.MortonAddressPlan((0.0,), (1.0,), 8),
        8,
        1,
        1,
        maximum_candidates=2,
    )
    result = phx.graph.query_neighbors(source, jnp.asarray([[0.25]]), plan=plan)
    assert not bool(result.evidence.complete)
    assert not bool(result.evidence.successful)
    assert not bool(jnp.any(result.mask))
    np.testing.assert_array_equal(result.count, 0)


@pytest.mark.parametrize("axis", ["source", "target"])
def test_graph_query_refuses_nonpositive_chunk_size(axis: str) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        if axis == "source":
            phx.graph.query_neighbors([[0.0]], [[0.0]], source_chunk_size=0)
        else:
            phx.graph.query_neighbors([[0.0]], [[0.0]], target_chunk_size=0)


@pytest.mark.parametrize("axis", ["source", "target"])
@pytest.mark.parametrize(
    "tile", [None, True, 1.5, "2"], ids=["none", "boolean", "fractional", "string"]
)
def test_graph_query_refuses_noninteger_chunk_sizes(axis: str, tile: Any) -> None:
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        if axis == "source":
            phx.graph.query_neighbors([[0.0]], [[0.0]], source_chunk_size=tile)
        else:
            phx.graph.query_neighbors([[0.0]], [[0.0]], target_chunk_size=tile)


def test_blocked_neighbors_preserve_nonfinite_failure_routes() -> None:
    source = np.asarray([[np.nan], [1.0], [0.0], [2.0], [np.inf]], dtype=np.float64)
    expected = np.argsort(np.square(source[:, 0]), kind="stable")
    result = phx.graph.query_neighbors(
        source,
        [[0.0]],
        max_neighbors=5,
        source_chunk_size=3,
    )
    np.testing.assert_array_equal(result.indices[0, 0], expected)
    np.testing.assert_array_equal(result.mask, True)
    np.testing.assert_array_equal(result.count, 5)
    assert not bool(result.evidence.finite)
    assert not bool(result.evidence.successful)
