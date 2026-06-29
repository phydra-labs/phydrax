#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _attention_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes={"features": jnp.array([[1.0], [3.0], [0.0]])},
        edges={"bias": jnp.array([0.0, jnp.log(3.0)])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([2, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def test_graph_attention_operator_uses_edge_bias_softmax():
    graph = _attention_graph()
    model = phx.graph.GraphAttentionOperator(
        logit_fn=lambda edges, sent, recv, globals_: jnp.zeros((sent.shape[0],)),
        input_key="features",
        output_key="attn",
        edge_bias_key="bias",
    )

    out = model(graph)

    assert jnp.allclose(out.nodes["attn"][:, 0], jnp.array([0.0, 0.0, 2.5]))


def test_graph_attention_operator_masks_padding_edges():
    graph = _attention_graph().replace(
        edge_mask=jnp.array([True, False]),
        validate=False,
    )
    model = phx.graph.GraphAttentionOperator(
        logit_fn=lambda edges, sent, recv, globals_: jnp.zeros((sent.shape[0],)),
        input_key="features",
        output_key="attn",
        edge_bias_key="bias",
    )

    out = model(graph)

    assert jnp.allclose(out.nodes["attn"][:, 0], jnp.array([0.0, 0.0, 1.0]))


def test_graph_attention_operator_supports_multihead_logits():
    graph = _attention_graph()

    def logits(edges, sent, recv, globals_):
        del edges, recv, globals_
        zeros = jnp.zeros((sent.shape[0],))
        return jnp.stack([zeros, zeros + 1.0], axis=1)

    out = phx.graph.GraphAttentionOperator(
        logit_fn=logits,
        input_key="features",
        output_key="attn",
        head_reduction="concat",
    )(graph)

    assert out.nodes["attn"].shape == (3, 2)
    assert jnp.allclose(out.nodes["attn"][2], jnp.array([2.0, 2.0]))


def test_graph_attention_operator_can_mask_to_query_targets():
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )

    out = phx.graph.GraphAttentionOperator(
        logit_fn=lambda edges, sent, recv, globals_: jnp.zeros((sent.shape[0],)),
        input_key="features",
        output_key="attn",
        target_node_type=bundle.target_type,
    )(bundle.graph)

    assert jnp.allclose(out.nodes["attn"][:, 0], jnp.array([0.0, 0.0, 2.0]))


def test_graph_attention_operator_wraps_as_graph_model():
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
    def u(point):
        return point.get("features")[0]

    model = domain.GraphModel(
        phx.graph.GraphAttentionOperator(
            logit_fn=lambda edges, sent, recv, globals_: jnp.zeros((sent.shape[0],)),
            input_key="u",
            output_key="attn",
            target_node_type=bundle.target_type,
        ),
        input_fn=u,
        input_key="u",
        output_key="attn",
    )

    assert jnp.allclose(model(batch).data[:, 0], jnp.array([2.0]))
