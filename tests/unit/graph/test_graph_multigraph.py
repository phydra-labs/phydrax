#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _source_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes={
            "positions": jnp.array([[0.0], [1.0]]),
            "features": jnp.array([[1.0], [3.0]]),
        },
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )


def _query() -> phx.graph.QueryGraph:
    return phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        weight_kind=None,
    )


class _AddLatentOne:
    def __init__(self, indices):
        self.indices = jnp.asarray(indices, dtype=jnp.int32)

    def __call__(self, graph):
        nodes = dict(graph.nodes)
        nodes["latent"] = nodes["latent"].at[self.indices].add(1.0)
        return graph.replace(nodes=nodes, validate=False)


def test_query_graph_with_source_features_installs_source_side_only():
    graph = phx.graph.query_graph_with_source_features(
        _query(),
        jnp.array([[1.0], [3.0]]),
        input_key="u",
    )

    assert jnp.allclose(graph.nodes["u"][:, 0], jnp.array([1.0, 3.0, 0.0]))


def test_query_graph_operator_can_gather_source_node_subset():
    query = phx.graph.radius_query_graph(
        jnp.array([[0.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        weight_kind=None,
    )
    source = phx.graph.GraphIR(
        nodes={"features": jnp.array([[1.0], [3.0], [10.0]])},
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )
    out = phx.graph.QueryGraphOperator(
        query,
        source_key="features",
        source_indices=jnp.array([1], dtype=jnp.int32),
        input_key="u",
        output_key="out",
        edge_weight_key=None,
        normalize=False,
    )(source)

    assert jnp.allclose(out.nodes["out"][:, 0], jnp.array([0.0, 3.0]))


def test_query_target_features_extracts_target_payload():
    query = _query()
    graph = query.graph.replace(
        nodes={**query.graph.nodes, "out": jnp.array([[0.0], [0.0], [4.0]])},
        validate=False,
    )

    assert jnp.allclose(
        phx.graph.query_target_features(graph, query, "out")[:, 0],
        jnp.array([4.0]),
    )


def test_query_graph_operator_transfers_source_graph_to_target_query_graph():
    query = _query()
    op = phx.graph.QueryGraphOperator(
        query,
        source_key="features",
        input_key="u",
        output_key="out",
        edge_weight_key=None,
        normalize=False,
    )

    out = op(_source_graph())

    assert out.num_nodes == 3
    assert jnp.allclose(out.nodes["out"][:, 0], jnp.array([0.0, 0.0, 4.0]))
    assert jnp.allclose(
        phx.graph.query_target_features(out, query, "out")[:, 0],
        jnp.array([4.0]),
    )


def test_query_graph_operator_result_can_be_used_as_graph_domain():
    query = _query()
    out = phx.graph.QueryGraphOperator(
        query,
        source_key="features",
        input_key="u",
        output_key="out",
        edge_weight_key=None,
        normalize=False,
    )(_source_graph())
    domain = phx.domain.GraphDomain(out)
    targets = domain.component({"graph": query.target_nodes_component()})
    batch = targets.sample(1, structure=phx.domain.ProductStructure((("graph",),)))

    @domain.Function("graph")
    def predicted(node):
        return node.get("out")[0]

    assert jnp.allclose(predicted(batch).data, jnp.array([4.0]))


def test_query_encode_process_decode_transfers_source_to_latent_to_target():
    encoder_query = _query()
    decoder_query = phx.graph.radius_query_graph(
        jnp.array([[0.5]]),
        jnp.array([[0.25]]),
        radius=1.0,
        weight_kind=None,
    )
    pipeline = phx.graph.query_encode_process_decode(
        encoder_query,
        decoder_query,
        processor=_AddLatentOne(encoder_query.target_nodes),
        source_key="features",
        latent_key="latent",
        output_key="out",
        edge_weight_key=None,
        normalize=False,
    )

    out = pipeline(_source_graph())

    assert jnp.allclose(out.nodes["out"][:, 0], jnp.array([0.0, 5.0]))
    assert jnp.allclose(
        phx.graph.query_target_features(out, decoder_query, "out")[:, 0],
        jnp.array([5.0]),
    )


def test_query_encode_process_decode_result_can_be_used_as_graph_domain():
    encoder_query = _query()
    decoder_query = phx.graph.radius_query_graph(
        jnp.array([[0.5]]),
        jnp.array([[0.25]]),
        radius=1.0,
        weight_kind=None,
    )
    out = phx.graph.query_encode_process_decode(
        encoder_query,
        decoder_query,
        source_key="features",
        latent_key="latent",
        output_key="out",
        edge_weight_key=None,
        normalize=False,
    )(_source_graph())
    domain = phx.domain.GraphDomain(out)
    targets = domain.component({"graph": decoder_query.target_nodes_component()})
    batch = targets.sample(1, structure=phx.domain.ProductStructure((("graph",),)))

    @domain.Function("graph")
    def prediction(node):
        return node.get("out")[0]

    assert jnp.allclose(prediction(batch).data, jnp.array([4.0]))
