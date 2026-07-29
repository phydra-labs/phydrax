from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp

from ._ir import batch_graphs, GraphIR, unbatch_graph
from ._kernels import segment_sum


class Data(eqx.Module):
    """Ergonomic single-graph data container.

    This mirrors common `edge_index` workflows while preserving conversion into
    the canonical `GraphIR` execution representation.
    """

    x: jnp.ndarray | None
    edge_index: jnp.ndarray | None
    edge_attr: jnp.ndarray | None
    y: jnp.ndarray | None
    pos: jnp.ndarray | None
    batch: jnp.ndarray | None
    ptr: jnp.ndarray | None

    def __init__(
        self,
        *,
        x: jnp.ndarray | None = None,
        edge_index: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
        y: jnp.ndarray | None = None,
        pos: jnp.ndarray | None = None,
        batch: jnp.ndarray | None = None,
        ptr: jnp.ndarray | None = None,
    ):
        self.x = None if x is None else jnp.asarray(x)
        self.edge_index = None if edge_index is None else jnp.asarray(edge_index)
        self.edge_attr = None if edge_attr is None else jnp.asarray(edge_attr)
        self.y = None if y is None else jnp.asarray(y)
        self.pos = None if pos is None else jnp.asarray(pos)
        self.batch = None if batch is None else jnp.asarray(batch, dtype=jnp.int32)
        self.ptr = None if ptr is None else jnp.asarray(ptr, dtype=jnp.int32)

    @property
    def num_nodes(self) -> int:
        if self.x is not None:
            return int(self.x.shape[0])
        if self.pos is not None:
            return int(self.pos.shape[0])
        if self.edge_index is not None and int(self.edge_index.size) > 0:
            return int(jnp.max(self.edge_index)) + 1
        return 0

    @property
    def num_edges(self) -> int:
        if self.edge_index is None:
            return 0
        return int(self.edge_index.shape[1])

    @property
    def num_node_features(self) -> int:
        if self.x is None or self.x.ndim < 2:
            return 0
        return int(self.x.shape[-1])

    @property
    def num_edge_features(self) -> int:
        if self.edge_attr is None or self.edge_attr.ndim < 2:
            return 0
        return int(self.edge_attr.shape[-1])

    def to_graph_ir(self, *, validate: bool = True) -> GraphIR:
        if self.edge_index is None:
            senders = jnp.zeros((0,), dtype=jnp.int32)
            receivers = jnp.zeros((0,), dtype=jnp.int32)
        else:
            edge_index = jnp.asarray(self.edge_index)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("`edge_index` must have shape (2, num_edges).")
            if not jnp.issubdtype(edge_index.dtype, jnp.integer):
                raise TypeError("`edge_index` must be integer dtype.")
            senders = edge_index[0].astype(jnp.int32)
            receivers = edge_index[1].astype(jnp.int32)

        nodes = self.x if self.x is not None else self.pos
        globals_ = None
        if self.y is not None and self.y.ndim >= 1 and int(self.y.shape[0]) == 1:
            globals_ = self.y

        return GraphIR(
            nodes=nodes,
            edges=self.edge_attr,
            senders=senders,
            receivers=receivers,
            globals=globals_,
            n_node=jnp.asarray([self.num_nodes], dtype=jnp.int32),
            n_edge=jnp.asarray([self.num_edges], dtype=jnp.int32),
            validate=validate,
        )

    @classmethod
    def from_graph_ir(cls, graph: GraphIR, /) -> "Data":
        graph.validate()
        if graph.num_graphs != 1:
            raise ValueError("`Data.from_graph_ir` expects exactly one graph.")
        return cls(
            x=graph.nodes,
            edge_index=graph.edge_index,
            edge_attr=graph.edges,
            y=graph.globals,
            pos=None,
            batch=None,
            ptr=None,
        )


class Batch(Data):
    """Batch of graphs represented as one disconnected sparse graph."""

    @classmethod
    def from_data_list(cls, data_list: Sequence[Data], /) -> "Batch":
        if len(data_list) == 0:
            return cls(
                x=None,
                edge_index=None,
                edge_attr=None,
                y=None,
                pos=None,
                batch=None,
                ptr=None,
            )

        graphs = tuple(data.to_graph_ir(validate=True) for data in data_list)
        batched = batch_graphs(graphs, validate=True)

        n_graphs = int(batched.n_node.shape[0])
        batch = jnp.repeat(
            jnp.arange(n_graphs, dtype=jnp.int32),
            batched.n_node,
            axis=0,
            total_repeat_length=batched.num_nodes,
        )
        ptr = jnp.concatenate(
            [
                jnp.asarray([0], dtype=jnp.int32),
                jnp.cumsum(batched.n_node, axis=0),
            ],
            axis=0,
        )

        has_graph_labels = all(data.y is not None for data in data_list)
        if has_graph_labels:
            y = jnp.stack([jnp.asarray(data.y).reshape(-1)[0] for data in data_list])
        else:
            y = None

        return cls(
            x=batched.nodes,
            edge_index=batched.edge_index,
            edge_attr=batched.edges,
            y=y,
            pos=None,
            batch=batch,
            ptr=ptr,
        )

    def to_data_list(self) -> list[Data]:
        graph = self.to_graph_ir(validate=True)
        pieces = unbatch_graph(graph, validate=True)

        out: list[Data] = []
        for i, piece in enumerate(pieces):
            y = None
            if self.y is not None:
                y = self.y[i : i + 1]
            out.append(
                Data(
                    x=piece.nodes,
                    edge_index=piece.edge_index,
                    edge_attr=piece.edges,
                    y=y,
                )
            )
        return out

    @property
    def num_graphs(self) -> int:
        if self.ptr is not None:
            return int(self.ptr.shape[0]) - 1
        if self.batch is not None and int(self.batch.size) > 0:
            return int(jnp.max(self.batch)) + 1
        return 1

    def to_graph_ir(self, *, validate: bool = True) -> GraphIR:
        if self.edge_index is None:
            senders = jnp.zeros((0,), dtype=jnp.int32)
            receivers = jnp.zeros((0,), dtype=jnp.int32)
        else:
            edge_index = jnp.asarray(self.edge_index, dtype=jnp.int32)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("`edge_index` must have shape (2, num_edges).")
            senders = edge_index[0]
            receivers = edge_index[1]

        if self.ptr is not None:
            ptr = jnp.asarray(self.ptr, dtype=jnp.int32)
            if ptr.ndim != 1 or int(ptr.shape[0]) < 2:
                raise ValueError("`ptr` must be rank-1 with length >= 2.")
            n_node = ptr[1:] - ptr[:-1]
            n_graph = int(n_node.shape[0])
        elif self.batch is not None:
            batch = jnp.asarray(self.batch, dtype=jnp.int32)
            n_graph = int(jnp.max(batch)) + 1 if int(batch.size) > 0 else 1
            n_node = segment_sum(
                jnp.ones((batch.shape[0],), dtype=jnp.int32),
                batch,
                n_graph,
            ).astype(jnp.int32)
        else:
            n_node = jnp.asarray([self.num_nodes], dtype=jnp.int32)
            n_graph = 1

        if int(senders.shape[0]) == 0:
            n_edge = jnp.zeros((n_graph,), dtype=jnp.int32)
        elif self.batch is not None:
            sender_graph = jnp.asarray(self.batch, dtype=jnp.int32)[senders]
            n_edge = segment_sum(
                jnp.ones((senders.shape[0],), dtype=jnp.int32),
                sender_graph,
                n_graph,
            ).astype(jnp.int32)
        elif self.ptr is not None:
            ptr = jnp.asarray(self.ptr, dtype=jnp.int32)
            sender_graph = jnp.searchsorted(ptr[1:], senders, side="right")
            n_edge = segment_sum(
                jnp.ones((senders.shape[0],), dtype=jnp.int32),
                sender_graph,
                n_graph,
            ).astype(jnp.int32)
        else:
            n_edge = jnp.asarray([int(senders.shape[0])], dtype=jnp.int32)

        nodes = self.x if self.x is not None else self.pos
        globals_ = self.y if (self.y is not None and int(self.y.shape[0]) == n_graph) else None

        return GraphIR(
            nodes=nodes,
            edges=self.edge_attr,
            senders=senders,
            receivers=receivers,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge,
            validate=validate,
        )


__all__ = ["Data", "Batch"]
