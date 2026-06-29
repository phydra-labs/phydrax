#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ...domain._function import BatchAwareCallable, DomainFunction
from ...domain.graph._batch import (
    GRAPH_ENTITY_INDEX_KEY,
    GRAPH_GRAPH_INDEX_KEY,
    GraphBatch,
)
from ...graph import segment_max, segment_mean, segment_min, segment_sum


GraphReduce = Literal["sum", "mean", "max", "min"]
GraphComponentKind = Literal["nodes", "edges", "globals"]
GraphDivergenceSign = Literal["in_minus_out", "out_minus_in"]
GraphFlow = Literal["source_to_target", "target_to_source"]


def _graph_axis(batch: GraphBatch, /) -> str:
    axis = batch.structure.axis_for(batch.graph_label)
    if axis is None:
        raise ValueError("GraphBatch is missing its graph sampling axis.")
    return axis


def _require_node_batch(batch: GraphBatch, /) -> None:
    if batch.component_kind != "nodes":
        raise ValueError(
            "This graph operator must be evaluated on a GraphBatch over Nodes(). "
            f"Got {batch.component_kind!r}."
        )


def _require_edge_batch(batch: GraphBatch, /) -> None:
    if batch.component_kind != "edges":
        raise ValueError(
            "This graph operator must be evaluated on a GraphBatch over Edges(). "
            f"Got {batch.component_kind!r}."
        )


def _num_nodes(batch: GraphBatch, /) -> int:
    if batch.graph.node_mask is not None:
        return int(batch.graph.node_mask.shape[0])
    return int(batch.graph.num_nodes)


def _num_edges(batch: GraphBatch, /) -> int:
    if batch.graph.edge_mask is not None:
        return int(batch.graph.edge_mask.shape[0])
    return int(batch.graph.num_edges)


def _num_graphs(batch: GraphBatch, /) -> int:
    if batch.graph.graph_mask is not None:
        return int(batch.graph.graph_mask.shape[0])
    return int(batch.graph.num_graphs)


def _real_num_nodes(batch: GraphBatch, /) -> int:
    return int(jnp.asarray(batch.graph.n_node).sum())


def _real_num_edges(batch: GraphBatch, /) -> int:
    return int(jnp.asarray(batch.graph.n_edge).sum())


def _real_num_graphs(batch: GraphBatch, /) -> int:
    if batch.graph.graph_mask is not None:
        return int(jnp.asarray(batch.graph.graph_mask).astype(jnp.int32).sum())
    return int(batch.graph.num_graphs)


def _component_size(batch: GraphBatch, kind: GraphComponentKind, /) -> int:
    if kind == "nodes":
        return _num_nodes(batch)
    if kind == "edges":
        return _num_edges(batch)
    return _num_graphs(batch)


def _to_axis_fields(tree: Any, axis: str, /) -> Any:
    def _leaf_to_field(value: Any) -> cx.Field:
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            raise ValueError("Graph entity payload leaves must have a leading axis.")
        return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

    return jax.tree_util.tree_map(_leaf_to_field, tree)


def _entity_payload(batch: GraphBatch, kind: GraphComponentKind, /) -> Any:
    if kind == "nodes":
        if batch.graph.nodes is not None:
            return batch.graph.nodes
        return jnp.arange(_num_nodes(batch), dtype=jnp.int32)
    if kind == "edges":
        if batch.graph.edges is not None:
            return batch.graph.edges
        if batch.graph.senders is None or batch.graph.receivers is None:
            return jnp.zeros((0, 2), dtype=jnp.int32)
        return jnp.stack([batch.graph.senders, batch.graph.receivers], axis=-1)
    if batch.graph.globals is not None:
        return batch.graph.globals
    return jnp.arange(_num_graphs(batch), dtype=jnp.int32)


def _pad_ids_to_length(ids: jnp.ndarray, size: int, /) -> jnp.ndarray:
    pad = int(size) - int(ids.shape[0])
    if pad <= 0:
        return ids
    return jnp.concatenate([ids, jnp.full((pad,), -1, dtype=jnp.int32)], axis=0)


def _graph_ids_for_kind(batch: GraphBatch, kind: GraphComponentKind, /) -> jnp.ndarray:
    graph_ids = jnp.arange(batch.graph.n_node.shape[0], dtype=jnp.int32)
    if kind == "nodes":
        real = jnp.repeat(
            graph_ids,
            batch.graph.n_node,
            axis=0,
            total_repeat_length=_real_num_nodes(batch),
        )
        return _pad_ids_to_length(real, _num_nodes(batch))
    if kind == "edges":
        real = jnp.repeat(
            graph_ids,
            batch.graph.n_edge,
            axis=0,
            total_repeat_length=_real_num_edges(batch),
        )
        return _pad_ids_to_length(real, _num_edges(batch))
    real = jnp.arange(_real_num_graphs(batch), dtype=jnp.int32)
    return _pad_ids_to_length(real, _num_graphs(batch))


def _entity_indices(batch: GraphBatch, /) -> jnp.ndarray:
    field = batch.points.get(GRAPH_ENTITY_INDEX_KEY)
    if not isinstance(field, cx.Field):
        return jnp.arange(_component_size(batch, batch.component_kind), dtype=jnp.int32)
    return jnp.asarray(field.data, dtype=jnp.int32)


def _current_graph_ids(batch: GraphBatch, /) -> jnp.ndarray:
    field = batch.points.get(GRAPH_GRAPH_INDEX_KEY)
    if isinstance(field, cx.Field):
        return jnp.asarray(field.data, dtype=jnp.int32)
    return _graph_ids_for_kind(batch, batch.component_kind)[_entity_indices(batch)]


def _remap_graph_axis_field(
    field: cx.Field,
    /,
    *,
    axis: str,
    old_graph_ids: jnp.ndarray,
    new_graph_ids: jnp.ndarray,
    num_graphs: int,
) -> cx.Field:
    if axis not in field.named_dims:
        return field
    axis_pos = field.dims.index(axis)
    data = jnp.moveaxis(jnp.asarray(field.data), axis_pos, 0)
    if int(data.shape[0]) != int(old_graph_ids.shape[0]):
        return field

    valid = old_graph_ids >= 0
    segment_ids = jnp.where(valid, old_graph_ids, 0)
    mask = valid.astype(data.dtype)
    while mask.ndim < data.ndim:
        mask = jnp.expand_dims(mask, axis=-1)
    totals = segment_sum(data * mask, segment_ids, num_graphs)
    counts = segment_sum(valid.astype(float), segment_ids, num_graphs)
    scale = jnp.where(counts > 0, 1.0 / counts, 0.0)
    while scale.ndim < totals.ndim:
        scale = jnp.expand_dims(scale, axis=-1)
    by_graph = totals * scale
    if not jnp.issubdtype(data.dtype, jnp.floating):
        by_graph = jnp.rint(by_graph).astype(data.dtype)

    safe_new = jnp.where(new_graph_ids >= 0, new_graph_ids, 0)
    new_data = by_graph[safe_new]
    valid_new = new_graph_ids >= 0
    valid_new_mask = valid_new.astype(new_data.dtype)
    while valid_new_mask.ndim < new_data.ndim:
        valid_new_mask = jnp.expand_dims(valid_new_mask, axis=-1)
    new_data = new_data * valid_new_mask
    return cx.Field(jnp.moveaxis(new_data, 0, axis_pos), dims=field.dims)


def _remap_graph_axis_tree(
    tree: Any,
    /,
    *,
    axis: str,
    old_graph_ids: jnp.ndarray,
    new_graph_ids: jnp.ndarray,
    num_graphs: int,
) -> Any:
    return jax.tree_util.tree_map(
        lambda x: _remap_graph_axis_field(
            x,
            axis=axis,
            old_graph_ids=old_graph_ids,
            new_graph_ids=new_graph_ids,
            num_graphs=num_graphs,
        )
        if isinstance(x, cx.Field)
        else x,
        tree,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )


def _as_graph_component_batch(
    batch: GraphBatch,
    kind: GraphComponentKind,
    /,
    *,
    full: bool = False,
) -> GraphBatch:
    if batch.component_kind == kind and not full:
        return batch

    axis = _graph_axis(batch)
    size = _component_size(batch, kind)
    old_graph_ids = _current_graph_ids(batch)
    new_graph_ids = _graph_ids_for_kind(batch, kind)
    points = dict(batch.points)
    for key, value in list(points.items()):
        if key in (batch.graph_label, GRAPH_ENTITY_INDEX_KEY, GRAPH_GRAPH_INDEX_KEY):
            continue
        points[key] = _remap_graph_axis_tree(
            value,
            axis=axis,
            old_graph_ids=old_graph_ids,
            new_graph_ids=new_graph_ids,
            num_graphs=_num_graphs(batch),
        )
    points[batch.graph_label] = _to_axis_fields(_entity_payload(batch, kind), axis)
    points[GRAPH_ENTITY_INDEX_KEY] = cx.Field(
        jnp.arange(size, dtype=jnp.int32), dims=(axis,)
    )
    points[GRAPH_GRAPH_INDEX_KEY] = cx.Field(new_graph_ids, dims=(axis,))
    return GraphBatch(
        points=points,
        structure=batch.structure,
        graph=batch.graph,
        graph_label=batch.graph_label,
        component_kind=kind,
    )


def _segment_reduce(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    reduce: GraphReduce,
) -> jnp.ndarray:
    if reduce == "sum":
        return segment_sum(data, segment_ids, num_segments)
    if reduce == "mean":
        return segment_mean(data, segment_ids, num_segments)
    if reduce == "max":
        out = segment_max(data, segment_ids, num_segments)
        return jnp.where(jnp.isfinite(out), out, jnp.zeros_like(out))
    if reduce == "min":
        out = segment_min(data, segment_ids, num_segments)
        return jnp.where(jnp.isfinite(out), out, jnp.zeros_like(out))
    raise ValueError(f"Unsupported graph reduce mode: {reduce!r}.")


def _field_data_on_graph_axis(
    func: DomainFunction,
    batch: GraphBatch,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> tuple[cx.Field, int, jnp.ndarray]:
    y = func(batch, key=key, **kwargs)
    axis = _graph_axis(batch)
    if axis not in y.named_dims:
        raise ValueError(f"Input field is missing graph axis {axis!r}.")
    axis_pos = y.dims.index(axis)
    data = jnp.moveaxis(jnp.asarray(y.data), axis_pos, 0)
    return y, axis_pos, data


def _restore_graph_axis(
    data: jnp.ndarray,
    axis_pos: int,
    dims: tuple[str | None, ...],
    /,
) -> cx.Field:
    return cx.Field(jnp.moveaxis(data, 0, axis_pos), dims=dims)


def _broadcast_over_leading_axis(
    value: jnp.ndarray,
    target: jnp.ndarray,
    /,
) -> jnp.ndarray:
    if value.ndim == 1 and target.ndim > 1:
        return value.reshape((value.shape[0],) + (1,) * (target.ndim - 1))
    while value.ndim < target.ndim:
        value = jnp.expand_dims(value, axis=-1)
    return value


def _optional_edge_weight_data(
    weight: DomainFunction | ArrayLike | None,
    edge_batch: GraphBatch,
    target: jnp.ndarray,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> jnp.ndarray | None:
    if weight is None:
        return None
    if isinstance(weight, DomainFunction):
        _field, _axis_pos, weight_data = _field_data_on_graph_axis(
            weight, edge_batch, key=key, **kwargs
        )
    else:
        weight_data = jnp.asarray(weight)
    return _broadcast_over_leading_axis(jnp.asarray(weight_data), target)


class _GraphDegreeCallable(StrictModule, BatchAwareCallable):
    mode: Literal["in", "out", "total"]

    def __init__(self, mode: Literal["in", "out", "total"]):
        self.mode = mode

    def __call_batch__(
        self,
        batch: GraphBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        _require_node_batch(batch)
        if batch.graph.senders is None or batch.graph.receivers is None:
            raise ValueError("degree requires explicit graph senders/receivers.")

        n = _num_nodes(batch)
        ones = jnp.ones((batch.graph.senders.shape[0],), dtype=float)
        if self.mode == "in":
            deg = segment_sum(ones, batch.graph.receivers, n)
        elif self.mode == "out":
            deg = segment_sum(ones, batch.graph.senders, n)
        else:
            deg = segment_sum(ones, batch.graph.receivers, n) + segment_sum(
                ones, batch.graph.senders, n
            )
        deg = deg[_entity_indices(batch)]
        return cx.Field(deg, dims=(_graph_axis(batch),))


class _NeighborAggregateCallable(StrictModule, BatchAwareCallable):
    u: DomainFunction
    reduce: GraphReduce
    flow: Literal["source_to_target", "target_to_source"]

    def __init__(
        self,
        u: DomainFunction,
        *,
        reduce: GraphReduce,
        flow: Literal["source_to_target", "target_to_source"],
    ):
        self.u = u
        self.reduce = reduce
        self.flow = flow

    def __call_batch__(
        self,
        batch: GraphBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        _require_node_batch(batch)
        if batch.graph.senders is None or batch.graph.receivers is None:
            raise ValueError("neighbor aggregation requires explicit senders/receivers.")

        node_batch = _as_graph_component_batch(batch, "nodes", full=True)
        y, axis_pos, data = _field_data_on_graph_axis(
            self.u, node_batch, key=key, **kwargs
        )

        if self.flow == "source_to_target":
            src = batch.graph.senders
            dst = batch.graph.receivers
        else:
            src = batch.graph.receivers
            dst = batch.graph.senders

        aggregated = _segment_reduce(data[src], dst, _num_nodes(batch), self.reduce)
        aggregated = aggregated[_entity_indices(batch)]
        return _restore_graph_axis(aggregated, axis_pos, y.dims)


class _GraphLaplacianCallable(StrictModule, BatchAwareCallable):
    u: DomainFunction
    normalize: bool

    def __init__(self, u: DomainFunction, *, normalize: bool):
        self.u = u
        self.normalize = bool(normalize)

    def __call_batch__(
        self,
        batch: GraphBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        _require_node_batch(batch)
        if batch.graph.senders is None or batch.graph.receivers is None:
            raise ValueError("graph_laplacian requires explicit senders/receivers.")

        node_batch = _as_graph_component_batch(batch, "nodes", full=True)
        y, axis_pos, data = _field_data_on_graph_axis(
            self.u, node_batch, key=key, **kwargs
        )

        senders = batch.graph.senders
        receivers = batch.graph.receivers
        n = _num_nodes(batch)
        messages = data[receivers] - data[senders]

        if self.normalize:
            ones = jnp.ones((senders.shape[0],), dtype=float)
            deg_in = segment_sum(ones, receivers, n)
            scale = jnp.where(deg_in > 0, 1.0 / deg_in, 0.0)
            messages = messages * scale[receivers].reshape(
                (-1,) + (1,) * (messages.ndim - 1)
            )

        out = segment_sum(messages, receivers, n)
        out = out[_entity_indices(batch)]
        return _restore_graph_axis(out, axis_pos, y.dims)


class _GraphGradientCallable(StrictModule, BatchAwareCallable):
    u: DomainFunction
    weight: DomainFunction | ArrayLike | None
    flow: GraphFlow

    def __init__(
        self,
        u: DomainFunction,
        *,
        weight: DomainFunction | ArrayLike | None,
        flow: GraphFlow,
    ):
        self.u = u
        self.weight = weight
        self.flow = flow

    def __call_batch__(
        self,
        batch: GraphBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        _require_edge_batch(batch)
        if batch.graph.senders is None or batch.graph.receivers is None:
            raise ValueError("graph_gradient requires explicit senders/receivers.")

        node_batch = _as_graph_component_batch(batch, "nodes", full=True)
        edge_batch = _as_graph_component_batch(batch, "edges", full=True)
        field, axis_pos, node_data = _field_data_on_graph_axis(
            self.u, node_batch, key=key, **kwargs
        )

        senders = batch.graph.senders
        receivers = batch.graph.receivers
        if self.flow == "source_to_target":
            edge_data = node_data[receivers] - node_data[senders]
        else:
            edge_data = node_data[senders] - node_data[receivers]

        weight_data = _optional_edge_weight_data(
            self.weight, edge_batch, edge_data, key=key, **kwargs
        )
        if weight_data is not None:
            edge_data = edge_data * weight_data

        edge_data = edge_data[_entity_indices(batch)]
        return _restore_graph_axis(edge_data, axis_pos, field.dims)


class _GraphDivergenceCallable(StrictModule, BatchAwareCallable):
    flux: DomainFunction
    sign: GraphDivergenceSign

    def __init__(
        self,
        flux: DomainFunction,
        *,
        sign: GraphDivergenceSign,
    ):
        self.flux = flux
        self.sign = sign

    def __call_batch__(
        self,
        batch: GraphBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        _require_node_batch(batch)
        if batch.graph.senders is None or batch.graph.receivers is None:
            raise ValueError("graph_divergence requires explicit senders/receivers.")

        edge_batch = _as_graph_component_batch(batch, "edges", full=True)
        field, axis_pos, edge_data = _field_data_on_graph_axis(
            self.flux, edge_batch, key=key, **kwargs
        )

        n = _num_nodes(batch)
        incoming = segment_sum(edge_data, batch.graph.receivers, n)
        outgoing = segment_sum(edge_data, batch.graph.senders, n)
        if self.sign == "in_minus_out":
            node_data = incoming - outgoing
        else:
            node_data = outgoing - incoming
        node_data = node_data[_entity_indices(batch)]
        return _restore_graph_axis(node_data, axis_pos, field.dims)


def degree(
    domain: Any,
    /,
    *,
    mode: Literal["in", "out", "total"] = "in",
) -> DomainFunction:
    """Return node degree as a graph-domain `DomainFunction`."""
    if mode not in ("in", "out", "total"):
        raise ValueError("degree mode must be 'in', 'out', or 'total'.")
    return DomainFunction(
        domain=domain,
        deps=(domain.label,),
        func=_GraphDegreeCallable(mode),
    )


def neighbor_aggregate(
    u: DomainFunction,
    /,
    *,
    reduce: GraphReduce = "sum",
    flow: GraphFlow = "source_to_target",
) -> DomainFunction:
    """Aggregate neighboring node field values along graph edges."""
    return DomainFunction(
        domain=u.domain,
        deps=u.deps,
        func=_NeighborAggregateCallable(u, reduce=reduce, flow=flow),
        metadata=u.metadata,
    )


def graph_gradient(
    u: DomainFunction,
    /,
    *,
    weight: DomainFunction | ArrayLike | None = None,
    flow: GraphFlow = "source_to_target",
) -> DomainFunction:
    """Return the edge-wise graph gradient of a node field.

    For each directed edge sender -> receiver, the default orientation computes
    `u(receiver) - u(sender)` and returns an edge-domain field. Optional edge
    weights are evaluated on the same edge batch and multiplied into the result.
    """
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError("graph_gradient flow must be 'source_to_target' or 'target_to_source'.")
    return DomainFunction(
        domain=u.domain,
        deps=u.deps,
        func=_GraphGradientCallable(u, weight=weight, flow=flow),
        metadata=u.metadata,
    )


def graph_divergence(
    flux: DomainFunction,
    /,
    *,
    sign: GraphDivergenceSign = "in_minus_out",
) -> DomainFunction:
    """Reduce an edge flux to nodes with the graph incidence operator.

    With `sign="in_minus_out"`, each edge contributes `+flux` to its receiver
    and `-flux` to its sender. `sign="out_minus_in"` flips that convention.
    """
    if sign not in ("in_minus_out", "out_minus_in"):
        raise ValueError("graph_divergence sign must be 'in_minus_out' or 'out_minus_in'.")
    return DomainFunction(
        domain=flux.domain,
        deps=flux.deps,
        func=_GraphDivergenceCallable(flux, sign=sign),
        metadata=flux.metadata,
    )


def graph_incidence_laplacian(
    u: DomainFunction,
    /,
    *,
    weight: DomainFunction | ArrayLike | None = None,
    flow: GraphFlow = "source_to_target",
    sign: GraphDivergenceSign = "in_minus_out",
) -> DomainFunction:
    """Return the incidence-form graph Laplacian `div(grad(u))`.

    This is the conservative graph-calculus counterpart to `graph_laplacian`,
    which preserves its existing incoming-neighbor semantics.
    """
    return graph_divergence(
        graph_gradient(u, weight=weight, flow=flow),
        sign=sign,
    )


def graph_laplacian(
    u: DomainFunction,
    /,
    *,
    normalize: bool = False,
) -> DomainFunction:
    """Return the unweighted graph Laplacian of a node field.

    For directed edges sender -> receiver, this computes at each receiver node
    the sum of `u(receiver) - u(sender)` over incoming edges. With
    `normalize=True`, each incoming contribution is divided by the receiver
    in-degree.
    """
    return DomainFunction(
        domain=u.domain,
        deps=u.deps,
        func=_GraphLaplacianCallable(u, normalize=normalize),
        metadata=u.metadata,
    )


graph_grad = graph_gradient
graph_div = graph_divergence
gradient = graph_gradient
divergence = graph_divergence
incidence_laplacian = graph_incidence_laplacian


__all__ = [
    "GraphDivergenceSign",
    "GraphFlow",
    "GraphReduce",
    "degree",
    "divergence",
    "graph_div",
    "graph_divergence",
    "graph_grad",
    "graph_gradient",
    "graph_incidence_laplacian",
    "graph_laplacian",
    "gradient",
    "incidence_laplacian",
    "neighbor_aggregate",
]
