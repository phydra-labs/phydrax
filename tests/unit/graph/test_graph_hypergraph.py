#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _bundle() -> phx.graph.HypergraphBipartiteGraph:
    return phx.graph.hypergraph_to_bipartite_graph(
        ([0, 1], [1, 2]),
        node_features=jnp.array([[1.0], [2.0], [3.0]]),
    )


def test_hypergraph_to_bipartite_graph_adds_typed_auxiliary_nodes():
    bundle = _bundle()
    graph = bundle.graph

    assert graph.num_nodes == 5
    assert graph.num_edges == 8
    assert jnp.allclose(graph.nodes["type"], jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32))
    assert jnp.allclose(
        graph.edges["type"],
        jnp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32),
    )
    assert jnp.allclose(bundle.original_nodes, jnp.array([0, 1, 2], dtype=jnp.int32))
    assert jnp.allclose(bundle.hyperedge_nodes, jnp.array([3, 4], dtype=jnp.int32))


def test_hypergraph_bundle_components_select_original_and_hyperedge_entities():
    bundle = _bundle()
    domain = phx.domain.GraphDomain(bundle.graph, measure="count")
    structure = phx.domain.SampleLayout((("graph",),))
    original = domain.component({"graph": bundle.original_nodes_component()})
    hyperedges = domain.component({"graph": bundle.hyperedge_nodes_component()})
    incidence = domain.component({"graph": bundle.incidence_edges_component()})

    original_batch = original.sample(phx.domain.PointSampling(3, layout=structure))
    hyperedge_batch = hyperedges.sample(phx.domain.PointSampling(2, layout=structure))
    incidence_batch = incidence.sample(phx.domain.PointSampling(4, layout=structure))

    assert jnp.allclose(
        original_batch["graph"]["features"].data[:, 0], jnp.array([1.0, 2.0, 3.0])
    )
    assert jnp.allclose(
        hyperedge_batch["graph"]["features"].data[:, 0], jnp.array([0.0, 0.0])
    )
    assert jnp.allclose(
        incidence_batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data,
        jnp.array([0, 1, 2, 3], dtype=jnp.int32),
    )
    assert jnp.allclose(hyperedges.mass.value, 2.0)


def test_hypergraph_convolution_computes_two_stage_means():
    out = phx.graph.HypergraphConvolution(output_key="u_next")(_bundle().graph)

    assert jnp.allclose(out.nodes["u_next"][:, 0], jnp.array([1.5, 2.0, 2.5, 1.5, 2.5]))


def test_hypergraph_convolution_wraps_as_graph_model_on_original_nodes():
    bundle = _bundle()
    domain = phx.domain.GraphDomain(bundle.graph)
    component = domain.component({"graph": bundle.original_nodes_component()})
    batch = component.sample(
        phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("graph",),)))
    )

    @domain.Function("graph")
    def u(node):
        return node["features"][0]

    model = domain.GraphModel(
        phx.graph.HypergraphConvolution(input_key="u", output_key="u_next"),
        input_fn=u,
        input_key="u",
        output_key="u_next",
    )

    assert jnp.allclose(model(batch).data[:, 0], jnp.array([1.5, 2.0, 2.5]))


def test_hypergraph_bipartite_graph_batches_in_graph_dataset_domain():
    graph0 = _bundle().graph
    graph1 = phx.graph.hypergraph_to_bipartite_graph(
        ([0, 1, 2],),
        node_features=jnp.array([[2.0], [4.0], [8.0]]),
    ).graph
    domain = phx.domain.GraphDatasetDomain((graph0, graph1))
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.NodeType(0),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    assert batch.graph.num_nodes == 9
    assert jnp.allclose(
        batch["graph"]["features"].data[:, 0], jnp.array([1.0, 2.0, 3.0, 2.0, 4.0, 8.0])
    )
    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_DATASET_INDEX_KEY].data,
        jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32),
    )
