#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _measure_graph():
    return phx.graph.GraphIR(
        nodes={
            "u": jnp.array([1.0, 3.0, 0.0]),
            "quadrature_weight": jnp.array([0.25, 0.75, 0.0]),
            "type": jnp.array([0, 0, 1], dtype=jnp.int32),
        },
        edges={},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([2, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
        validate=False,
    )


def test_graph_neural_operator_uses_source_quadrature_measure():
    graph = _measure_graph()
    operator = phx.graph.GraphNeuralOperator(
        input_key="u",
        output_key="integral",
        edge_weight_key=None,
        source_measure_key="quadrature_weight",
        normalize=False,
        target_node_type=1,
    )
    output = operator(graph)
    assert jnp.allclose(output.nodes["integral"][2], 2.5)


def test_graph_attention_softmax_is_continuum_measure_aware():
    graph = _measure_graph()
    operator = phx.graph.GraphAttentionOperator(
        query_fn=lambda values: jnp.zeros_like(values),
        key_fn=lambda values: jnp.zeros_like(values),
        input_key="u",
        output_key="attention",
        source_measure_key="quadrature_weight",
        target_node_type=1,
    )
    output = operator(graph)
    assert jnp.allclose(output.nodes["attention"][2, 0], 2.5)


def test_query_graph_installs_source_measure():
    source = jnp.array([[0.0], [0.5], [1.0]])
    target = jnp.array([[0.25], [0.75]])
    measure = jnp.array([0.2, 0.3, 0.5])
    query = phx.graph.radius_query_graph(
        source,
        target,
        radius=2.0,
        source_measure=measure,
    )
    installed = query.graph.nodes["quadrature_weight"]
    assert jnp.allclose(installed[:3], measure)
    assert jnp.allclose(installed[3:], 0.0)


def test_gino_configuration_executes_encode_process_decode():
    source = jnp.linspace(0.0, 1.0, 6)
    latent = jnp.linspace(0.0, 1.0, 4)
    target = jnp.linspace(0.0, 1.0, 5)
    encoder_query = phx.graph.radius_query_graph(
        source[:, None],
        latent[:, None],
        radius=2.0,
        source_measure=jnp.ones((6,)) / 6.0,
    )
    decoder_query = phx.graph.radius_query_graph(
        latent[:, None],
        target[:, None],
        radius=2.0,
        source_measure=jnp.ones((4,)) / 4.0,
    )
    processor = phx.nn.FNO(
        width=4,
        depth=1,
        n_modes=(2,),
        key=jr.key(0),
    )
    operator = phx.graph.gino_operator(
        encoder_query,
        decoder_query,
        processor,
        (4,),
        latent_axes=(latent,),
    )
    source_graph = phx.graph.GraphIR(
        nodes={"features": jnp.sin(2.0 * jnp.pi * source)},
        n_node=jnp.array([6], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
        validate=False,
    )
    output = operator(source_graph)
    prediction = phx.graph.query_target_features(output, decoder_query, "features")
    assert prediction.shape == (5,)
    assert jnp.all(jnp.isfinite(prediction))
