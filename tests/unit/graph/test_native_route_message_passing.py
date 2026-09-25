#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

import phydrax as phx


SENDERS = np.asarray([0, 1, 2, 3, 4, 5, 0, 2, 5, 1], dtype=np.int32)
RECEIVERS = np.asarray([1, 2, 3, 4, 5, 0, 3, 0, 2, 4], dtype=np.int32)
NODE_COUNT = 6
EDGE_TYPES = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)


def _graph(nodes, edges) -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=nodes,
        edges=edges,
        senders=SENDERS,
        receivers=RECEIVERS,
        n_node=[NODE_COUNT],
        n_edge=[SENDERS.size],
    )


def _padded(graph: phx.graph.GraphIR) -> phx.graph.GraphIR:
    """Append masked routes with arbitrary endpoints and large garbage payloads."""
    senders = np.asarray([5, 0, 3], dtype=np.int32)
    receivers = np.asarray([0, 5, 1], dtype=np.int32)
    rng = np.random.default_rng(11)

    def pad(value):
        array = jnp.asarray(value)
        shape = (senders.size,) + array.shape[1:]
        if jnp.issubdtype(array.dtype, jnp.integer):
            extra = jnp.zeros(shape, dtype=array.dtype)
        else:
            extra = jnp.asarray(1.0e3 * rng.normal(size=shape), dtype=array.dtype)
        return jnp.concatenate((array, extra), axis=0)

    return phx.graph.GraphIR(
        nodes=graph.nodes,
        edges=jtu.tree_map(pad, graph.edges),
        senders=np.concatenate((np.asarray(graph.senders), senders)),
        receivers=np.concatenate((np.asarray(graph.receivers), receivers)),
        globals=graph.globals,
        n_node=graph.n_node,
        n_edge=[graph.senders.shape[0] + senders.size],
        edge_mask=np.arange(graph.senders.shape[0] + senders.size)
        < graph.senders.shape[0],
    )


def _random(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape))


def _segment_mean(data, ids, count):
    total = jax.ops.segment_sum(data, ids, count)
    ones = jnp.ones((ids.shape[0],) + (1,) * (data.ndim - 1), dtype=data.dtype)
    return total / jnp.maximum(jax.ops.segment_sum(ones, ids, count), 1.0)


def _inverse(values):
    return jnp.where(values > 0, 1.0 / values, 0.0)


def _mesh_graph_net_block_case():
    block = phx.graph.MeshGraphNetBlock(4, key=jr.key(0))
    graph = _graph(_random((NODE_COUNT, 4), 1), _random((SENDERS.size, 4), 2))

    def reference(graph):
        s, r = graph.senders, graph.receivers
        nodes, edges = graph.nodes, graph.edges
        edges = edges + block.edge_mlp(jnp.concatenate([edges, nodes[s], nodes[r]], -1))
        received = jax.ops.segment_sum(edges, r, NODE_COUNT)
        return nodes + block.node_mlp(jnp.concatenate([nodes, received], -1))

    return graph, lambda g: block(g).nodes, reference


def _attention_case():
    graph = _graph(_random((NODE_COUNT, 3), 3), None)

    def reference(graph):
        x, s, r = graph.nodes, graph.senders, graph.receivers
        logits = jnp.sum(x[s] * x[r], axis=-1) / jnp.sqrt(3.0)
        weights = jnp.exp(logits - jax.ops.segment_max(logits, r, NODE_COUNT)[r])
        weights = weights / jax.ops.segment_sum(weights, r, NODE_COUNT)[r]
        return jax.ops.segment_sum(x[s] * weights[:, None], r, NODE_COUNT)

    operator = phx.graph.GraphAttentionOperator()
    return graph, lambda g: operator(g).nodes, reference


def _kernel_integral_case():
    graph = _graph(_random((NODE_COUNT, 2), 4), _random((SENDERS.size, 1), 5))
    operator = phx.graph.GraphKernelIntegral(lambda edges, s, r, g: edges[:, 0])

    def reference(graph):
        messages = graph.nodes[graph.senders] * graph.edges
        return jax.ops.segment_sum(messages, graph.receivers, NODE_COUNT)

    return graph, lambda g: operator(g).nodes, reference


def _equivariant_case():
    graph = _graph(
        {
            "positions": _random((NODE_COUNT, 2), 6),
            "features": _random((NODE_COUNT, 2), 7),
        },
        {"w": _random((SENDERS.size,), 8)},
    )
    operator = phx.graph.EquivariantGraphConvolution(edge_weight_key="w", normalize=True)

    def apply(graph):
        nodes = operator(graph).nodes
        return jnp.concatenate(
            [nodes["scalar"], nodes["vector"].reshape((NODE_COUNT, -1))], -1
        )

    def reference(graph):
        s, r = graph.senders, graph.receivers
        pos, x, w = graph.nodes["positions"], graph.nodes["features"], graph.edges["w"]
        scale = _inverse(jax.ops.segment_sum(jnp.abs(w), r, NODE_COUNT))
        scalar = jax.ops.segment_sum(x[s] * w[:, None], r, NODE_COUNT)
        relative = pos[r] - pos[s]
        vector = jax.ops.segment_sum(
            relative[:, :, None] * x[s][:, None, :] * w[:, None, None], r, NODE_COUNT
        )
        return jnp.concatenate(
            [
                scalar * scale[:, None],
                (vector * scale[:, None, None]).reshape((NODE_COUNT, -1)),
            ],
            -1,
        )

    return graph, apply, reference


def _relational_case():
    weights = _random((2, 3, 2), 9)
    graph = _graph(
        _random((NODE_COUNT, 3), 10),
        {"type": EDGE_TYPES, "w": jnp.abs(_random((SENDERS.size,), 11)) + 0.1},
    )
    operator = phx.graph.RelationalGraphConvolution(
        weights, edge_weight_key="w", normalize=True
    )

    def reference(graph):
        s, r = graph.senders, graph.receivers
        types, w = graph.edges["type"], graph.edges["w"]
        keys = types * NODE_COUNT + r
        scale = w / jax.ops.segment_sum(w, keys, 2 * NODE_COUNT)[keys]
        messages = jnp.sum(graph.nodes[s][:, :, None] * weights[types], axis=1)
        return jax.ops.segment_sum(messages * scale[:, None], r, NODE_COUNT)

    return graph, lambda g: operator(g).nodes, reference


def _hypergraph_case():
    graph = phx.graph.hypergraph_to_bipartite_graph(
        ([0, 1], [1, 2, 3], [0, 3]),
        node_features=_random((4, 2), 12),
    ).graph
    operator = phx.graph.HypergraphConvolution(output_key="out")

    def reference(graph):
        s, r, n = graph.senders, graph.receivers, graph.nodes["type"].shape[0]
        x, types = graph.nodes["features"], graph.edges["type"]
        weight = graph.edges["incidence_weight"].reshape((-1,))
        incidence = jnp.where(types == 0, weight, 0.0)
        hyper = jax.ops.segment_sum(x[s] * incidence[:, None], r, n)
        hyper = hyper * _inverse(jax.ops.segment_sum(incidence, r, n))[:, None]
        reverse = jnp.where(types == 1, weight, 0.0)
        out = jax.ops.segment_sum(hyper[s] * reverse[:, None], r, n)
        out = out * _inverse(jax.ops.segment_sum(reverse, r, n))[:, None]
        node_types = graph.nodes["type"][:, None]
        return jnp.where(node_types == 0, out, jnp.where(node_types == 1, hyper, 0.0))

    return graph, lambda g: operator(g).nodes["out"], reference


def _pool_case():
    cluster_ids = jnp.asarray([0, 0, 1, 1, 2, -1], dtype=jnp.int32)
    graph = _graph(_random((NODE_COUNT, 2), 13), None)

    def reference(graph):
        valid = np.asarray(cluster_ids) >= 0
        return _segment_mean(graph.nodes[valid], cluster_ids[valid], 3)

    return (
        graph,
        lambda g: (
            phx.graph.pool_graph_by_cluster(g, cluster_ids, reduce_nodes="mean").nodes
        ),
        reference,
    )


def _edge_index_case(layer, reference_fn):
    x = _random((NODE_COUNT, 3), 14)
    edge_index = jnp.asarray(np.stack([SENDERS, RECEIVERS]))
    return (x, edge_index), lambda args: layer(*args), reference_fn


def _gcn_case():
    conv = phx.graph.GCNConv(3, 2, key=jr.key(1), add_self_loops=False)
    weight = jnp.abs(_random((SENDERS.size,), 15)) + 0.1

    def reference(args):
        x, (row, col) = args
        degree = jax.ops.segment_sum(weight, col, NODE_COUNT)
        inv_sqrt = jnp.where(degree > 0, degree**-0.5, 0.0)
        projected = jax.vmap(conv.linear)(x)
        norm = inv_sqrt[row] * weight * inv_sqrt[col]
        return jax.ops.segment_sum(projected[row] * norm[:, None], col, NODE_COUNT)

    return _edge_index_case(lambda x, index: conv(x, index, weight), reference)


def _sage_max_case():
    conv = phx.graph.SAGEConv(3, 2, key=jr.key(2), aggr="max", root_weight=False)

    def reference(args):
        x, (row, col) = args
        projected = jax.vmap(conv.lin_neigh)(x)
        out = jax.ops.segment_max(projected[row], col, NODE_COUNT)
        return jnp.where(jnp.isfinite(out), out, 0.0)

    return _edge_index_case(conv, reference)


def _message_passing_mean_case():
    passing = phx.graph.MessagePassing(
        aggr="mean", message=lambda x_j, x_i, edge_attr: x_j - 2.0 * x_i
    )

    def reference(args):
        x, (row, col) = args
        return _segment_mean(x[row] - 2.0 * x[col], col, NODE_COUNT)

    return _edge_index_case(passing, reference)


PARITY_CASES: dict[str, Callable] = {
    "mesh_graph_net_block": _mesh_graph_net_block_case,
    "graph_attention": _attention_case,
    "graph_kernel_integral": _kernel_integral_case,
    "equivariant_convolution": _equivariant_case,
    "relational_convolution": _relational_case,
    "hypergraph_convolution": _hypergraph_case,
    "cluster_pool": _pool_case,
    "gcn_conv": _gcn_case,
    "sage_conv_max": _sage_max_case,
    "message_passing_mean": _message_passing_mean_case,
}


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_route_reductions_match_segment_reductions(case):
    inputs, apply, reference = PARITY_CASES[case]()

    np.testing.assert_allclose(apply(inputs), reference(inputs), rtol=1e-12, atol=1e-12)


def _typed_graph(nodes, **edges):
    return _graph(nodes, {"type": EDGE_TYPES, **edges})


def _padding_cases() -> dict[str, tuple[phx.graph.GraphIR, Callable]]:
    features = _random((NODE_COUNT, 3), 20)
    mesh_graph_net = phx.graph.MeshGraphNet(
        node_in_size=3,
        edge_in_size=2,
        node_out_size=2,
        latent_size=4,
        processor_steps=2,
        key=jr.key(3),
    )
    kernel = lambda edges, s, r, g: edges[:, 0]
    geometric = {"positions": _random((NODE_COUNT, 2), 21), "features": features}
    return {
        "mesh_graph_net": (
            _graph(features, _random((SENDERS.size, 2), 22)),
            lambda g: mesh_graph_net(g).nodes,
        ),
        "graph_attention": (
            _graph(features, None),
            lambda g: phx.graph.GraphAttentionOperator()(g).nodes,
        ),
        "kernel_integral_max": (
            _graph(features, _random((SENDERS.size, 1), 23)),
            lambda g: phx.graph.GraphKernelIntegral(kernel, reduction="max")(g).nodes,
        ),
        "kernel_integral_mean": (
            _graph(features, _random((SENDERS.size, 1), 24)),
            lambda g: phx.graph.GraphKernelIntegral(kernel, reduction="mean")(g).nodes,
        ),
        "equivariant_convolution": (
            _graph(geometric, {"w": _random((SENDERS.size,), 25)}),
            lambda g: phx.graph.EquivariantGraphConvolution(
                edge_weight_key="w", normalize=True
            )(g).nodes["vector"],
        ),
        "relational_convolution": (
            _typed_graph(features, w=jnp.abs(_random((SENDERS.size,), 26)) + 0.1),
            lambda g: (
                phx.graph.RelationalGraphConvolution(
                    _random((2, 3), 27), edge_weight_key="w", normalize=True
                )(g).nodes
            ),
        ),
    }


@pytest.mark.parametrize("case", sorted(_padding_cases()))
def test_masked_routes_are_inert(case):
    graph, apply = _padding_cases()[case]

    np.testing.assert_allclose(
        apply(_padded(graph)), apply(graph), rtol=1e-14, atol=1e-14
    )


def test_masked_routes_do_not_change_mesh_graph_net_gradients():
    model = phx.graph.MeshGraphNet(
        node_in_size=3,
        edge_in_size=2,
        node_out_size=1,
        latent_size=4,
        processor_steps=2,
        key=jr.key(4),
    )
    graph = _graph(_random((NODE_COUNT, 3), 30), _random((SENDERS.size, 2), 31))

    def loss(model, graph):
        return jnp.sum(model(graph).nodes ** 2)

    grad = eqx.filter_grad(loss)
    reference = jtu.tree_leaves(eqx.filter(grad(model, graph), eqx.is_inexact_array))
    padded = jtu.tree_leaves(
        eqx.filter(grad(model, _padded(graph)), eqx.is_inexact_array)
    )

    for expected, actual in zip(reference, padded, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_graph_view_of_relation_reproduces_routes():
    relation = phx.sparse.EdgeRelation(
        [0, -5, 2, 1],
        [1, 9, 0, 0],
        source_size=3,
        target_size=3,
        valid=[True, False, True, True],
    )
    payload = _random((4, 2), 40)

    graph = phx.graph.GraphIR.from_edge_relation(relation, nodes=_random((3, 1), 41))
    view = graph.edge_relation()

    np.testing.assert_array_equal(view.valid, relation.valid)
    np.testing.assert_allclose(
        phx.sparse.route_reduce(view, payload), phx.sparse.route_reduce(relation, payload)
    )
    np.testing.assert_allclose(
        phx.sparse.gather_routes(view, graph.nodes),
        phx.sparse.gather_routes(relation, graph.nodes),
    )


def test_graph_view_requires_one_node_space():
    rectangular = phx.sparse.EdgeRelation([0], [1], source_size=1, target_size=2)

    with pytest.raises(ValueError, match="one node space"):
        phx.graph.GraphIR.from_edge_relation(rectangular)
    with pytest.raises(TypeError, match="EdgeRelation"):
        phx.graph.GraphIR.from_edge_relation(np.zeros((2, 1), dtype=np.int32))


def test_graph_kernel_integral_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="reduction"):
        phx.graph.GraphKernelIntegral(reduction="median")
