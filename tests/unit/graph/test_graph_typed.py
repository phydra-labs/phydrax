#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _typed_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes={
            "features": jnp.array([[1.0], [2.0], [3.0]]),
            "type": jnp.array([0, 1, 0], dtype=jnp.int32),
        },
        edges={
            "features": jnp.array([[0.5], [1.5], [2.5]]),
            "type": jnp.array([0, 1, 0], dtype=jnp.int32),
            "weight": jnp.array([1.0, 2.0, 1.0]),
        },
        senders=jnp.array([0, 2, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 1, 0], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )


def _typed_graphs() -> tuple[phx.graph.GraphIR, phx.graph.GraphIR]:
    graph0 = _typed_graph()
    graph1 = phx.graph.GraphIR(
        nodes={
            "features": jnp.array([[10.0], [20.0]]),
            "type": jnp.array([1, 0], dtype=jnp.int32),
        },
        edges={
            "features": jnp.array([[3.0]]),
            "type": jnp.array([1], dtype=jnp.int32),
            "weight": jnp.array([1.0]),
        },
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    return graph0, graph1


def test_graph_domain_samples_node_and_edge_type_components():
    domain = phx.domain.GraphDomain(_typed_graph(), measure="count")
    structure = phx.domain.SampleLayout((("graph",),))

    node_component = domain.component({"graph": phx.domain.NodeType(1)})
    node_batch = node_component.sample(phx.domain.PointSampling(1, layout=structure))
    edge_component = domain.component({"graph": phx.domain.EdgeType(0)})
    edge_batch = edge_component.sample(phx.domain.PointSampling(2, layout=structure))

    assert node_batch.component_kind == "nodes"
    assert jnp.allclose(node_batch["graph"]["features"].data[:, 0], jnp.array([2.0]))
    assert jnp.allclose(
        jnp.asarray(node_batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data),
        jnp.array([1], dtype=jnp.int32),
    )
    assert edge_batch.component_kind == "edges"
    assert jnp.allclose(edge_batch["graph"]["features"].data[:, 0], jnp.array([0.5, 2.5]))
    assert jnp.allclose(edge_component.mass.value, 2.0)


def test_graph_dataset_domain_resolves_node_types_per_case():
    domain = phx.domain.GraphDatasetDomain(_typed_graphs(), measure="count")
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.NodeType(1),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    assert jnp.allclose(batch["graph"]["features"].data[:, 0], jnp.array([2.0, 10.0]))
    assert jnp.allclose(
        jnp.asarray(batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data),
        jnp.array([1, 3], dtype=jnp.int32),
    )
    assert jnp.allclose(
        jnp.asarray(batch[phx.domain.graph.GRAPH_DATASET_INDEX_KEY].data),
        jnp.array([0, 1], dtype=jnp.int32),
    )
    assert jnp.allclose(
        domain.component({"graph": phx.domain.NodeType(1)}).mass.value, 2.0
    )


def test_typed_graph_helpers_return_indices_and_components():
    graph = _typed_graph()

    assert jnp.allclose(phx.graph.node_type_indices(graph, 0), jnp.array([0, 2]))
    assert jnp.allclose(phx.graph.edge_type_indices(graph, 1), jnp.array([1]))
    assert isinstance(phx.graph.typed_nodes_component(1), phx.domain.NodeType)
    assert isinstance(phx.graph.typed_edges_component(0), phx.domain.EdgeType)


def test_relational_graph_convolution_aggregates_by_edge_type():
    graph = _typed_graph()
    conv = phx.graph.RelationalGraphConvolution(
        jnp.array([10.0, 100.0]),
        input_key="features",
        output_key="updated",
    )

    out = conv(graph)

    assert jnp.allclose(out.nodes["updated"][:, 0], jnp.array([20.0, 310.0, 0.0]))
    assert "features" in out.nodes
    assert "type" in out.nodes


def test_relational_graph_convolution_normalizes_per_receiver_relation():
    graph = phx.graph.GraphIR(
        nodes={
            "features": jnp.array([[1.0], [3.0], [5.0]]),
            "type": jnp.array([0, 0, 1], dtype=jnp.int32),
        },
        edges={"type": jnp.array([0, 0], dtype=jnp.int32)},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([2, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    conv = phx.graph.RelationalGraphConvolution(
        jnp.array([1.0]),
        input_key="features",
        normalize=True,
    )

    out = conv(graph)

    assert jnp.allclose(out.nodes[:, 0], jnp.array([0.0, 0.0, 2.0]))


def test_relational_graph_convolution_wraps_as_graph_model_with_input_key():
    domain = phx.domain.GraphDomain(_typed_graph())
    batch = domain.component({"graph": phx.domain.Nodes()}).sample(
        phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("graph",),)))
    )

    @domain.Function("graph")
    def u(node):
        return node["features"][0]

    model = domain.GraphModel(
        phx.graph.RelationalGraphConvolution(
            jnp.array([1.0, 1.0]),
            input_key="u",
            output_key="updated",
        ),
        input_fn=u,
        input_key="u",
        output_key="updated",
    )

    assert jnp.allclose(model(batch).data[:, 0], jnp.array([2.0, 4.0, 0.0]))
