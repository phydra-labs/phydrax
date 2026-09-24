from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._strict import StrictModule

from ..sparse import EdgeRelation, gather_routes, mask_routes, route_reduce
from ._graph import ensure_graph
from ._ir import GraphIR


def _as_2d(name: str, value: Any, /) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-1 or rank-2, got shape {arr.shape!r}.")
    return arr


def _mask_array(value: jnp.ndarray, mask: jnp.ndarray | None, /) -> jnp.ndarray:
    if mask is None:
        return value
    expanded = mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim))
    return jnp.where(expanded, value, jnp.zeros((), dtype=value.dtype))


def _message_relations(
    graph: GraphIR, node_count: int, owner: str, /
) -> tuple[EdgeRelation, EdgeRelation]:
    """Return the sender-to-receiver relation and its receiver-side view."""
    if graph.senders is None or graph.receivers is None:
        raise ValueError(f"{owner} requires explicit senders/receivers.")
    relation = graph.edge_relation(node_count=node_count)
    return relation, relation.transpose()


def _repeat_globals_for_entities(
    globals_: Any | None,
    counts: jnp.ndarray,
    total_length: int,
    /,
) -> jnp.ndarray | None:
    if globals_ is None:
        return None
    arr = _as_2d("globals", globals_)
    real_length = int(jnp.asarray(counts).sum())
    repeated = jnp.repeat(arr, counts, axis=0, total_repeat_length=real_length)
    pad = int(total_length) - repeated.shape[0]
    if pad <= 0:
        return repeated
    return jnp.concatenate(
        [repeated, jnp.zeros((pad, repeated.shape[1]), dtype=repeated.dtype)],
        axis=0,
    )


class RowMLP(StrictModule):
    """Apply an MLP independently to rows of a rank-2 array."""

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable = eqx.field(static=True)
    final_activation: Callable | None = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        /,
        *,
        width_size: int,
        depth: int = 2,
        activation: Callable = jax.nn.silu,
        final_activation: Callable | None = None,
        key: jax.Array,
    ):
        if depth < 1:
            raise ValueError("RowMLP depth must be at least 1.")
        sizes = [int(in_size)]
        if depth > 1:
            sizes.extend([int(width_size)] * (depth - 1))
        sizes.append(int(out_size))
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_, out, key=k)
            for in_, out, k in zip(sizes[:-1], sizes[1:], keys, strict=True)
        )
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: Any) -> jnp.ndarray:
        x = _as_2d("x", x)

        def apply_one(row):
            y = row
            for layer in self.layers[:-1]:
                y = self.activation(layer(y))
            y = self.layers[-1](y)
            if self.final_activation is not None:
                y = self.final_activation(y)
            return y

        return jax.vmap(apply_one)(x)


class MeshGraphNetBlock(StrictModule):
    """Residual message-passing block used by MeshGraphNet-style simulators.

    Edge updates gather sender and receiver latents through
    `graph.edge_relation()`, and receivers reduce edge latents with
    `phydrax.sparse.route_reduce`. Routes with `edge_mask=False` are inert: their
    latents are zero and they contribute nothing to any receiver.
    """

    edge_mlp: RowMLP
    node_mlp: RowMLP
    global_size: int = eqx.field(static=True)
    use_edge_residual: bool = eqx.field(static=True)
    use_node_residual: bool = eqx.field(static=True)

    def __init__(
        self,
        latent_size: int,
        /,
        *,
        hidden_size: int | None = None,
        mlp_depth: int = 2,
        global_size: int = 0,
        activation: Callable = jax.nn.silu,
        use_edge_residual: bool = True,
        use_node_residual: bool = True,
        key: jax.Array,
    ):
        hidden = int(latent_size if hidden_size is None else hidden_size)
        k_edge, k_node = jax.random.split(key, 2)
        self.edge_mlp = RowMLP(
            3 * int(latent_size) + int(global_size),
            int(latent_size),
            width_size=hidden,
            depth=mlp_depth,
            activation=activation,
            key=k_edge,
        )
        self.node_mlp = RowMLP(
            2 * int(latent_size) + int(global_size),
            int(latent_size),
            width_size=hidden,
            depth=mlp_depth,
            activation=activation,
            key=k_node,
        )
        self.global_size = int(global_size)
        self.use_edge_residual = bool(use_edge_residual)
        self.use_node_residual = bool(use_node_residual)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None or graph.edges is None:
            raise ValueError("MeshGraphNetBlock requires node and edge features.")
        node_count = _as_2d("nodes", graph.nodes).shape[0]
        return self._process(
            graph, *_message_relations(graph, node_count, "MeshGraphNetBlock")
        )

    def _process(
        self, graph: GraphIR, relation: EdgeRelation, reverse: EdgeRelation, /
    ) -> GraphIR:
        nodes = _as_2d("nodes", graph.nodes)
        edges = _as_2d("edges", graph.edges)
        glob_edge = None
        if self.global_size > 0:
            glob_edge = _repeat_globals_for_entities(
                graph.globals,
                graph.n_edge,
                relation.capacity,
            )
            if glob_edge is None:
                glob_edge = jnp.zeros(
                    (relation.capacity, self.global_size),
                    dtype=nodes.dtype,
                )

        edge_inputs = [
            edges,
            gather_routes(relation, nodes),
            gather_routes(reverse, nodes),
        ]
        if glob_edge is not None:
            edge_inputs.append(glob_edge)
        edge_delta = self.edge_mlp(jnp.concatenate(edge_inputs, axis=-1))
        edges = edges + edge_delta if self.use_edge_residual else edge_delta
        edges = mask_routes(relation, edges)

        recv_aggr = route_reduce(relation, edges)
        glob_node = None
        if self.global_size > 0:
            glob_node = _repeat_globals_for_entities(
                graph.globals,
                graph.n_node,
                nodes.shape[0],
            )
            if glob_node is None:
                glob_node = jnp.zeros(
                    (nodes.shape[0], self.global_size),
                    dtype=nodes.dtype,
                )
        node_inputs = [nodes, recv_aggr]
        if glob_node is not None:
            node_inputs.append(glob_node)
        node_delta = self.node_mlp(jnp.concatenate(node_inputs, axis=-1))
        nodes = nodes + node_delta if self.use_node_residual else node_delta
        nodes = _mask_array(nodes, graph.node_mask)

        return graph.replace(nodes=nodes, edges=edges, validate=False)


class MeshGraphNet(StrictModule):
    """Encoder-processor-decoder graph simulator architecture.

    This is the canonical mesh-simulation pattern: encode node and edge
    payloads, run residual message-passing processor steps on latent features,
    then decode node outputs. All processor steps share one fixed-topology
    `EdgeRelation` built from the input graph, so a relation prepared by a
    discretization bridge such as `facet_adjacency` is the same relation a
    physical residual on that graph reduces over.
    """

    node_encoder: RowMLP
    edge_encoder: RowMLP
    processors: tuple[MeshGraphNetBlock, ...]
    node_decoder: RowMLP
    edge_decoder: RowMLP | None

    def __init__(
        self,
        *,
        node_in_size: int,
        edge_in_size: int,
        node_out_size: int,
        edge_out_size: int | None = None,
        latent_size: int = 128,
        processor_steps: int = 15,
        hidden_size: int | None = None,
        mlp_depth: int = 2,
        global_size: int = 0,
        activation: Callable = jax.nn.silu,
        key: jax.Array,
    ):
        if processor_steps < 0:
            raise ValueError("processor_steps must be non-negative.")
        hidden = int(latent_size if hidden_size is None else hidden_size)
        key_count = 3 + int(edge_out_size is not None) + int(processor_steps)
        keys = iter(jax.random.split(key, key_count))
        self.node_encoder = RowMLP(
            int(node_in_size),
            int(latent_size),
            width_size=hidden,
            depth=mlp_depth,
            activation=activation,
            key=next(keys),
        )
        self.edge_encoder = RowMLP(
            int(edge_in_size),
            int(latent_size),
            width_size=hidden,
            depth=mlp_depth,
            activation=activation,
            key=next(keys),
        )
        self.processors = tuple(
            MeshGraphNetBlock(
                int(latent_size),
                hidden_size=hidden,
                mlp_depth=mlp_depth,
                global_size=global_size,
                activation=activation,
                key=next(keys),
            )
            for _ in range(int(processor_steps))
        )
        self.node_decoder = RowMLP(
            int(latent_size),
            int(node_out_size),
            width_size=hidden,
            depth=mlp_depth,
            activation=activation,
            key=next(keys),
        )
        self.edge_decoder = None
        if edge_out_size is not None:
            self.edge_decoder = RowMLP(
                int(latent_size),
                int(edge_out_size),
                width_size=hidden,
                depth=mlp_depth,
                activation=activation,
                key=next(keys),
            )

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None or graph.edges is None:
            raise ValueError("MeshGraphNet requires node and edge features.")

        nodes = _mask_array(self.node_encoder(graph.nodes), graph.node_mask)
        edges = _mask_array(self.edge_encoder(graph.edges), graph.edge_mask)
        out = graph.replace(nodes=nodes, edges=edges, validate=False)
        if self.processors:
            # Every processor step shares one fixed-topology relation pair.
            relations = _message_relations(graph, nodes.shape[0], "MeshGraphNet")
            for processor in self.processors:
                out = processor._process(out, *relations)
        nodes = _mask_array(self.node_decoder(out.nodes), out.node_mask)
        edges = out.edges
        if self.edge_decoder is not None:
            edges = _mask_array(self.edge_decoder(out.edges), out.edge_mask)
        return out.replace(nodes=nodes, edges=edges, validate=False)


__all__ = [
    "MeshGraphNet",
    "MeshGraphNetBlock",
    "RowMLP",
]
