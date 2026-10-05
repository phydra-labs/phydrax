from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._strict import StrictModule

from ..sparse._streamed import PreparedStreamedRelation, StreamedRelationPlan
from ._graph import ensure_graph
from ._ir import GraphIR
from ._route_payload import declared_payload, entity_graph_ids, graph_feature_table


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


def _execution_plan(execution: StreamedRelationPlan | None, /) -> StreamedRelationPlan:
    if execution is None:
        return StreamedRelationPlan()
    if not isinstance(execution, StreamedRelationPlan):
        raise TypeError("execution must be a StreamedRelationPlan or None.")
    return execution


def _prepared_messages(
    graph: GraphIR,
    node_count: int,
    execution: StreamedRelationPlan,
    owner: str,
    /,
) -> PreparedStreamedRelation:
    """Prepare the sender-to-receiver schedule shared by every processor step."""
    if graph.senders is None or graph.receivers is None:
        raise ValueError(f"{owner} requires explicit senders/receivers.")
    return execution.prepare(
        graph.edge_relation(node_count=node_count),
        owner_id=f"graph:{owner}",
        source_valid=graph.node_mask,
        receiver_valid=graph.node_mask,
    )


def _graph_features(
    globals_: Any | None,
    counts: Any,
    total_length: int,
    size: int,
    dtype: Any,
    /,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-graph feature table and owning-graph entity ids; zero features when absent."""
    if globals_ is None:
        return (
            jnp.zeros((1, size), dtype=dtype),
            jnp.zeros((total_length,), dtype=jnp.int32),
        )
    return graph_feature_table(_as_2d("globals", globals_)), entity_graph_ids(
        counts, total_length
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
    ) -> None:
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

    def apply_row(self, row: jnp.ndarray, /) -> jnp.ndarray:
        """Apply the MLP to one feature row."""
        y = row
        for layer in self.layers[:-1]:
            y = self.activation(layer(y))
        y = self.layers[-1](y)
        if self.final_activation is not None:
            y = self.final_activation(y)
        return y

    def __call__(self, x: Any) -> jnp.ndarray:
        return jax.vmap(self.apply_row)(_as_2d("x", x))


def _edge_update(
    parameters: tuple[MeshGraphNetBlock, jnp.ndarray, jnp.ndarray],
    sender: jnp.ndarray,
    receiver: tuple[jnp.ndarray, jnp.ndarray],
    edge: tuple[jnp.ndarray, jnp.ndarray],
    /,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Updated edge latent; it is both the receiver message and the edge output."""
    block, edge_table, _node_table = parameters
    latent, graph_id = edge
    inputs = [latent, sender, receiver[0]]
    if block.global_size > 0:
        inputs.append(edge_table[graph_id])
    delta = block.edge_mlp.apply_row(jnp.concatenate(inputs, axis=-1))
    updated = latent + delta if block.use_edge_residual else delta
    return updated, updated


def _node_update(
    parameters: tuple[MeshGraphNetBlock, jnp.ndarray, jnp.ndarray],
    receiver: tuple[jnp.ndarray, jnp.ndarray],
    aggregate: jnp.ndarray,
    /,
) -> jnp.ndarray:
    """Receiver epilogue applied once to the complete incoming edge-latent sum."""
    block, _edge_table, node_table = parameters
    node, graph_id = receiver
    inputs = [node, aggregate]
    if block.global_size > 0:
        inputs.append(node_table[graph_id])
    delta = block.node_mlp.apply_row(jnp.concatenate(inputs, axis=-1))
    return node + delta if block.use_node_residual else delta


class MeshGraphNetBlock(StrictModule):
    """Residual message-passing block used by MeshGraphNet-style simulators.

    Each route evaluates its edge update from sender and receiver latents and
    every receiver applies its node update once to the sum of its incoming
    updated edge latents through the prepared streamed relation selected by
    ``execution``. Updated edge latents are a requested graph-wide output; the
    MLP hidden activations stay bounded by the plan's edge tile. Routes with
    `edge_mask=False` are inert: their latents are zero and they contribute
    nothing to any receiver.
    """

    edge_mlp: RowMLP
    node_mlp: RowMLP
    execution: StreamedRelationPlan
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
        execution: StreamedRelationPlan | None = None,
        key: jax.Array,
    ) -> None:
        hidden = int(latent_size if hidden_size is None else hidden_size)
        plan = _execution_plan(execution)
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
        self.execution = plan
        self.global_size = int(global_size)
        self.use_edge_residual = bool(use_edge_residual)
        self.use_node_residual = bool(use_node_residual)

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None or graph.edges is None:
            raise ValueError("MeshGraphNetBlock requires node and edge features.")
        node_count = _as_2d("nodes", graph.nodes).shape[0]
        return self._process(
            graph,
            _prepared_messages(graph, node_count, self.execution, "MeshGraphNetBlock"),
        )

    def _process(self, graph: GraphIR, prepared: PreparedStreamedRelation, /) -> GraphIR:
        nodes = _as_2d("nodes", graph.nodes)
        edges = _as_2d("edges", graph.edges)
        globals_ = graph.globals if self.global_size > 0 else None
        edge_table, edge_ids = _graph_features(
            globals_, graph.n_edge, edges.shape[0], self.global_size, nodes.dtype
        )
        node_table, node_ids = _graph_features(
            globals_, graph.n_node, nodes.shape[0], self.global_size, nodes.dtype
        )
        parameters = (self, edge_table, node_table)
        receiver_data = (nodes, node_ids)
        edge_data = (edges, edge_ids)
        payload = declared_payload(
            _edge_update,
            _node_update,
            parameters,
            nodes,
            receiver_data,
            edge_data,
            edge_output=True,
        )
        result = prepared.evaluate(
            payload,
            _edge_update,
            _node_update,
            parameters,
            nodes,
            receiver_data,
            edge_data,
        )
        return graph.replace(
            nodes=_mask_array(result.receiver_outputs, graph.node_mask),
            edges=result.edge_outputs,
            validate=False,
        )


class MeshGraphNet(StrictModule):
    """Encoder-processor-decoder graph simulator architecture.

    This is the canonical mesh-simulation pattern: encode node and edge
    payloads, run residual message-passing processor steps on latent features,
    then decode node outputs. All processor steps share one streamed schedule
    prepared from the input graph's fixed-topology `EdgeRelation`, so a relation
    prepared by a discretization bridge such as `facet_adjacency` is the same
    relation a physical residual on that graph reduces over.
    """

    node_encoder: RowMLP
    edge_encoder: RowMLP
    processors: tuple[MeshGraphNetBlock, ...]
    node_decoder: RowMLP
    edge_decoder: RowMLP | None
    execution: StreamedRelationPlan

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
        execution: StreamedRelationPlan | None = None,
        key: jax.Array,
    ) -> None:
        if processor_steps < 0:
            raise ValueError("processor_steps must be non-negative.")
        plan = _execution_plan(execution)
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
                execution=plan,
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
        self.execution = plan

    def __call__(self, graph: GraphIR) -> GraphIR:
        graph = ensure_graph(graph, validate=False)
        if graph.nodes is None or graph.edges is None:
            raise ValueError("MeshGraphNet requires node and edge features.")

        nodes = _mask_array(self.node_encoder(graph.nodes), graph.node_mask)
        edges = _mask_array(self.edge_encoder(graph.edges), graph.edge_mask)
        out = graph.replace(nodes=nodes, edges=edges, validate=False)
        if self.processors:
            # Every processor step shares one prepared fixed-topology schedule.
            prepared = _prepared_messages(
                graph, nodes.shape[0], self.execution, "MeshGraphNet"
            )
            for processor in self.processors:
                out = processor._process(out, prepared)
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
