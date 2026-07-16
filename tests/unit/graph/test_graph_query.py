#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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
    assert jnp.allclose(graph.edges["kernel_weight"], jnp.array([[0.6], [0.2]]), atol=1e-7)
    assert jnp.allclose(graph.nodes["features"][:, 0], jnp.array([1.0, 2.0, 3.0, 0.0, 0.0]))


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
    structure = phx.domain.ProductStructure((("graph",),))
    sources = domain.component({"graph": bundle.source_nodes_component()})
    targets = domain.component({"graph": bundle.target_nodes_component()})
    query_edges = domain.component({"graph": bundle.query_edges_component()})

    source_batch = sources.sample(2, structure=structure)
    target_batch = targets.sample(1, structure=structure)
    edge_batch = query_edges.sample(2, structure=structure)

    assert jnp.allclose(source_batch["graph"]["features"].data[:, 0], jnp.array([1.0, 3.0]))
    assert jnp.allclose(target_batch["graph"]["features"].data[:, 0], jnp.array([0.0]))
    assert jnp.allclose(edge_batch["graph"]["distance"].data[:, 0], jnp.array([0.5, 0.5]))
    assert targets.measure() == 1.0


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
    batch = targets.sample(1, structure=phx.domain.ProductStructure((("graph",),)))

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
