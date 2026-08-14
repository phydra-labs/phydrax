#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_radius_query_graph_builds_weighted_bipartite_geometry():
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


def test_radius_query_graph_uses_periodic_minimum_image():
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


def test_knn_query_graph_and_cached_layout_replay():
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

    assert jnp.allclose(bundle.graph.senders, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(bundle.graph.receivers, jnp.array([3, 3], dtype=jnp.int32))
    assert jnp.allclose(rebuilt.graph.edges["relative"], jnp.array([[2.0], [0.0]]))


def test_query_graph_components_select_source_target_and_edges():
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
    assert targets.mass.value == 1.0


def test_graph_neural_operator_aggregates_query_sources_to_targets():
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


def test_graph_neural_operator_wraps_as_graph_model_on_query_targets():
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
    def u(node):
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

    assert jnp.allclose(model(batch).data, jnp.array([4.0]))


def test_batched_knn_query_graph_is_case_local_and_mask_aware():
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


def test_batched_query_graph_is_jittable_and_differentiable():
    source = jnp.array([[[0.0], [0.5], [1.0]]])
    target = jnp.array([[[0.2], [0.8]]])

    def distances(source_points, target_points):
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


def test_query_neighbors_have_stable_ties_and_periodic_minimum_image():
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


def test_batched_homogeneous_knn_graph_excludes_self_edges():
    graph = phx.graph.batched_knn_graph(
        jnp.array([[[0.0], [1.0], [3.0]], [[10.0], [11.0], [13.0]]]),
        k=1,
    )

    assert graph.senders.shape == (6,)
    assert jnp.all(graph.senders != graph.receivers)
    assert jnp.all(graph.senders[:3] < 3)
    assert jnp.all(graph.senders[3:] >= 3)
