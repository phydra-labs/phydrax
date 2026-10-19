from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from phydrax._strict import StrictModule
from phydrax._validation import nonnegative_integer

from ..typing import checked
from ._ir import batch_graphs, GraphIR, unbatch_graph

# PyG-style batch conversion counts nodes and edges per graph once while
# building `GraphIR` layout; a one-off segment count needs no prepared relation.
from ._kernels import segment_sum


class Data(StrictModule):
    """Store one graph and its node count for conversion into ``GraphIR``.

    Without ``x`` or ``pos``, ``num_nodes`` is required, including for empty
    graphs. Edges do not determine how many isolated nodes a graph contains.
    """

    x: Array | None
    edge_index: Array | None
    edge_attr: Array | None
    y: Array | None
    pos: Array | None
    batch: Array | None
    ptr: Array | None
    num_nodes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        x: ArrayLike | None = None,
        edge_index: ArrayLike | None = None,
        edge_attr: ArrayLike | None = None,
        y: ArrayLike | None = None,
        pos: ArrayLike | None = None,
        batch: ArrayLike | None = None,
        ptr: ArrayLike | None = None,
        num_nodes: int | None = None,
    ) -> None:
        nodes = None if x is None else jnp.asarray(x)
        indices = None if edge_index is None else jnp.asarray(edge_index)
        edges = None if edge_attr is None else jnp.asarray(edge_attr)
        labels = None if y is None else jnp.asarray(y)
        positions = None if pos is None else jnp.asarray(pos)
        batches = None if batch is None else jnp.asarray(batch, dtype=jnp.int32)
        pointers = None if ptr is None else jnp.asarray(ptr, dtype=jnp.int32)
        node_array = nodes if nodes is not None else positions
        if node_array is not None and node_array.ndim < 1:
            raise ValueError(
                f"Data.nodes must have a node axis; got shape {node_array.shape}"
            )
        if node_array is None and num_nodes is None:
            raise ValueError("Data.num_nodes must be declared without x or pos; got None")
        count = (
            node_array.shape[0]
            if num_nodes is None and node_array is not None
            else nonnegative_integer(num_nodes, "Data.num_nodes")
        )
        for name, value in (("x", nodes), ("pos", positions)):
            if value is not None and (value.ndim < 1 or value.shape[0] != count):
                raise ValueError(
                    f"Data.{name} must have {count} node rows; got shape {value.shape}"
                )
        self.num_nodes = count
        self.x = nodes
        self.edge_index = indices
        self.edge_attr = edges
        self.y = labels
        self.pos = positions
        self.batch = batches
        self.ptr = pointers

    @property
    def num_edges(self) -> int:
        if self.edge_index is None:
            return 0
        return self.edge_index.shape[1]

    @property
    def num_node_features(self) -> int:
        if self.x is None or self.x.ndim < 2:
            return 0
        return self.x.shape[-1]

    @property
    def num_edge_features(self) -> int:
        if self.edge_attr is None or self.edge_attr.ndim < 2:
            return 0
        return self.edge_attr.shape[-1]

    def to_graph_ir(self, *, validate: bool = True) -> GraphIR:
        if self.edge_index is None:
            senders = jnp.zeros((0,), dtype=jnp.int32)
            receivers = jnp.zeros((0,), dtype=jnp.int32)
        else:
            edge_index = jnp.asarray(self.edge_index)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(
                    "Data.edge_index must have shape (2, num_edges); got invalid edge axes"
                )
            if not jnp.issubdtype(edge_index.dtype, jnp.integer):
                raise TypeError(
                    "Data.edge_index must have integer dtype; got noninteger edges"
                )
            senders = edge_index[0].astype(jnp.int32)
            receivers = edge_index[1].astype(jnp.int32)

        nodes = self.x if self.x is not None else self.pos
        globals_ = None
        if self.y is not None and self.y.ndim >= 1 and self.y.shape[0] == 1:
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
    @checked
    def from_graph_ir(cls, graph: GraphIR, /) -> "Data":
        graph.validate()
        if graph.num_graphs != 1:
            raise ValueError(
                "Data.from_graph_ir.graph must contain one graph; got multiple graphs"
            )
        return cls(
            x=graph.nodes,
            edge_index=graph.edge_index,
            edge_attr=graph.edges,
            y=graph.globals,
            pos=None,
            batch=None,
            ptr=None,
            num_nodes=graph.num_nodes,
        )


class Batch(StrictModule):
    """Batch of graphs represented as one disconnected sparse graph."""

    data: Data

    @checked
    def __init__(self, data: Data, /) -> None:
        self.data = data

    @property
    def x(self) -> Array | None:
        return self.data.x

    @property
    def edge_index(self) -> Array | None:
        return self.data.edge_index

    @property
    def edge_attr(self) -> Array | None:
        return self.data.edge_attr

    @property
    def y(self) -> Array | None:
        return self.data.y

    @property
    def pos(self) -> Array | None:
        return self.data.pos

    @property
    def batch(self) -> Array | None:
        return self.data.batch

    @property
    def ptr(self) -> Array | None:
        return self.data.ptr

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes

    @property
    def num_edges(self) -> int:
        return self.data.num_edges

    @property
    def num_node_features(self) -> int:
        return self.data.num_node_features

    @property
    def num_edge_features(self) -> int:
        return self.data.num_edge_features

    @classmethod
    def from_data_list(cls, data_list: Sequence[Data], /) -> "Batch":
        if len(data_list) == 0:
            return cls(
                Data(
                    x=None,
                    edge_index=None,
                    edge_attr=None,
                    y=None,
                    pos=None,
                    batch=None,
                    ptr=None,
                    num_nodes=0,
                )
            )

        graphs = tuple(data.to_graph_ir(validate=True) for data in data_list)
        batched = batch_graphs(graphs, validate=True)

        n_graphs = batched.n_node.shape[0]
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

        labels = tuple(data.y for data in data_list)
        if all(label is not None for label in labels):
            arrays = tuple(jnp.asarray(label) for label in labels)
            if all(array.ndim >= 1 and array.shape[0] == 1 for array in arrays):
                y = jnp.concatenate(arrays, axis=0)
            else:
                y = jnp.stack(arrays, axis=0)
        else:
            y = None

        return cls(
            Data(
                x=batched.nodes,
                edge_index=batched.edge_index,
                edge_attr=batched.edges,
                y=y,
                pos=None,
                batch=batch,
                ptr=ptr,
                num_nodes=batched.num_nodes,
            )
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
                    num_nodes=piece.num_nodes,
                )
            )
        return out

    @property
    def num_graphs(self) -> int:
        if self.ptr is not None:
            return self.ptr.shape[0] - 1
        if self.batch is not None and self.batch.size > 0:
            return int(jnp.max(self.batch)) + 1
        return 1

    def to_graph_ir(self, *, validate: bool = True) -> GraphIR:
        if self.edge_index is None:
            senders = jnp.zeros((0,), dtype=jnp.int32)
            receivers = jnp.zeros((0,), dtype=jnp.int32)
        else:
            edge_index = jnp.asarray(self.edge_index, dtype=jnp.int32)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(
                    "Data.edge_index must have shape (2, num_edges); got invalid edge axes"
                )
            senders = edge_index[0]
            receivers = edge_index[1]

        if self.ptr is not None:
            ptr = jnp.asarray(self.ptr, dtype=jnp.int32)
            if ptr.ndim != 1 or ptr.shape[0] < 2:
                raise ValueError(
                    "Batch.ptr must be rank-1 with length >= 2; got invalid pointer shape"
                )
            n_node = ptr[1:] - ptr[:-1]
            n_graph = n_node.shape[0]
        elif self.batch is not None:
            batch = jnp.asarray(self.batch, dtype=jnp.int32)
            n_graph = int(jnp.max(batch)) + 1 if batch.size > 0 else 1
            n_node = segment_sum(
                jnp.ones((batch.shape[0],), dtype=jnp.int32),
                batch,
                n_graph,
            ).astype(jnp.int32)
        else:
            n_node = jnp.asarray([self.num_nodes], dtype=jnp.int32)
            n_graph = 1

        if senders.shape[0] == 0:
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
            n_edge = jnp.asarray([senders.shape[0]], dtype=jnp.int32)

        nodes = self.x if self.x is not None else self.pos
        globals_ = self.y if (self.y is not None and self.y.shape[0] == n_graph) else None

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
