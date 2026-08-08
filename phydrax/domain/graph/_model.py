#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Batch-aware graph-domain model adapters."""

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ...graph import GraphIR, rollout_features, segment_sum
from .._evaluation import BatchEvaluator
from .._function import DomainFunction
from ._batch import GRAPH_ENTITY_INDEX_KEY, GRAPH_GRAPH_INDEX_KEY, GraphBatch
from ._components import GraphComponentKind


GraphModelOutput = Literal["nodes", "edges", "globals"]


def _graph_axis(batch: GraphBatch, /) -> str:
    axis = batch.structure.axis_for(batch.graph_label)
    if axis is None:
        raise ValueError("GraphBatch is missing its graph sampling axis.")
    return axis


def _entity_indices(batch: GraphBatch, /) -> jnp.ndarray:
    field = batch.points.get(GRAPH_ENTITY_INDEX_KEY)
    if not isinstance(field, cx.Field):
        if batch.component_kind == "nodes":
            size = int(batch.graph.num_nodes)
        elif batch.component_kind == "edges":
            size = int(batch.graph.num_edges)
        else:
            size = int(batch.graph.num_graphs)
        return jnp.arange(size, dtype=jnp.int32)
    return jnp.asarray(field.data, dtype=jnp.int32)


def _num_nodes(graph: GraphIR, /) -> int:
    if graph.node_mask is not None:
        return int(graph.node_mask.shape[0])
    return int(graph.num_nodes)


def _num_entities(graph: GraphIR, kind: GraphComponentKind, /) -> int:
    if kind == "nodes":
        return _num_nodes(graph)
    if kind == "edges":
        if graph.edge_mask is not None:
            return int(graph.edge_mask.shape[0])
        return int(graph.num_edges)
    if graph.graph_mask is not None:
        return int(graph.graph_mask.shape[0])
    return int(graph.num_graphs)


def _pad_ids_to_length(ids: jnp.ndarray, size: int, /) -> jnp.ndarray:
    pad = int(size) - int(ids.shape[0])
    if pad <= 0:
        return ids
    return jnp.concatenate([ids, jnp.full((pad,), -1, dtype=jnp.int32)], axis=0)


def _to_axis_fields(tree: Any, axis: str, /) -> Any:
    def _leaf_to_field(value: Any) -> cx.Field:
        arr = jnp.asarray(value)
        return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

    return jax.tree_util.tree_map(_leaf_to_field, tree)


def _node_payload(graph: GraphIR, /) -> Any:
    if graph.nodes is not None:
        return graph.nodes
    return jnp.arange(_num_nodes(graph), dtype=jnp.int32)


def _edge_payload(graph: GraphIR, /) -> Any:
    if graph.edges is not None:
        return graph.edges
    if graph.senders is None or graph.receivers is None:
        return jnp.zeros((0, 2), dtype=jnp.int32)
    return jnp.stack([graph.senders, graph.receivers], axis=-1)


def _global_payload(graph: GraphIR, /) -> Any:
    if graph.globals is not None:
        return graph.globals
    return jnp.arange(_num_entities(graph, "globals"), dtype=jnp.int32)


def _payload_for_kind(graph: GraphIR, kind: GraphComponentKind, /) -> Any:
    if kind == "nodes":
        return _node_payload(graph)
    if kind == "edges":
        return _edge_payload(graph)
    return _global_payload(graph)


def _with_mapping_key(payload: Any, key: str, value: Any, /) -> dict[str, Any]:
    if payload is None:
        out: dict[str, Any] = {}
    elif isinstance(payload, Mapping):
        out = dict(payload)
    else:
        raise TypeError("GraphModel input_key requires mapping-valued graph nodes.")
    out[key] = value
    return out


def _select_mapping_key(payload: Any, key: str | None, label: str, /) -> Any:
    if key is None:
        if isinstance(payload, Mapping):
            raise ValueError(
                f"GraphModel output_key is required for mapping-valued {label}."
            )
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"GraphModel output_key requires mapping-valued {label}.")
    if key not in payload:
        raise KeyError(f"GraphModel {label} payload does not contain output_key {key!r}.")
    return payload[key]


def _graph_ids_from_counts(counts: Array, size: int, /) -> jnp.ndarray:
    positions = jnp.arange(int(size), dtype=jnp.int32)
    counts_array = jnp.asarray(counts, dtype=jnp.int32)
    ends = jnp.cumsum(counts_array)
    ids = jnp.searchsorted(ends, positions, side="right").astype(jnp.int32)
    return jnp.where(positions < jnp.sum(counts_array), ids, -1)


def _node_graph_ids(graph: GraphIR, /) -> jnp.ndarray:
    return _graph_ids_from_counts(graph.n_node, _num_nodes(graph))


def _edge_graph_ids(graph: GraphIR, /) -> jnp.ndarray:
    return _graph_ids_from_counts(
        graph.n_edge,
        _num_entities(graph, "edges"),
    )


def _global_graph_ids(graph: GraphIR, /) -> jnp.ndarray:
    graph_ids = jnp.arange(graph.n_node.shape[0], dtype=jnp.int32)
    return _pad_ids_to_length(graph_ids, _num_entities(graph, "globals"))


def _graph_ids_for_kind(graph: GraphIR, kind: GraphComponentKind, /) -> jnp.ndarray:
    if kind == "nodes":
        return _node_graph_ids(graph)
    if kind == "edges":
        return _edge_graph_ids(graph)
    return _global_graph_ids(graph)


def _current_graph_ids(batch: GraphBatch, /) -> jnp.ndarray:
    field = batch.points.get(GRAPH_GRAPH_INDEX_KEY)
    if isinstance(field, cx.Field):
        return jnp.asarray(field.data, dtype=jnp.int32)
    return _graph_ids_for_kind(batch.graph, batch.component_kind)[_entity_indices(batch)]


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
        lambda x: (
            _remap_graph_axis_field(
                x,
                axis=axis,
                old_graph_ids=old_graph_ids,
                new_graph_ids=new_graph_ids,
                num_graphs=num_graphs,
            )
            if isinstance(x, cx.Field)
            else x
        ),
        tree,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )


def _full_entity_batch(batch: GraphBatch, kind: GraphComponentKind, /) -> GraphBatch:
    axis = _graph_axis(batch)
    n = _num_entities(batch.graph, kind)
    old_graph_ids = _current_graph_ids(batch)
    new_graph_ids = _graph_ids_for_kind(batch.graph, kind)
    points = dict(batch.points)
    for key, value in list(points.items()):
        if key in (batch.graph_label, GRAPH_ENTITY_INDEX_KEY, GRAPH_GRAPH_INDEX_KEY):
            continue
        points[key] = _remap_graph_axis_tree(
            value,
            axis=axis,
            old_graph_ids=old_graph_ids,
            new_graph_ids=new_graph_ids,
            num_graphs=int(batch.graph.n_node.shape[0]),
        )
    points[batch.graph_label] = _to_axis_fields(
        _payload_for_kind(batch.graph, kind), axis
    )
    points[GRAPH_ENTITY_INDEX_KEY] = cx.Field(
        jnp.arange(n, dtype=jnp.int32), dims=(axis,)
    )
    points[GRAPH_GRAPH_INDEX_KEY] = cx.Field(new_graph_ids, dims=(axis,))
    return GraphBatch(
        points=points,
        structure=batch.structure,
        graph=batch.graph,
        graph_label=batch.graph_label,
        component_kind=kind,
    )


def _full_node_batch(batch: GraphBatch, /) -> GraphBatch:
    return _full_entity_batch(batch, "nodes")


def _install_graph_input(
    graph: GraphIR,
    batch: GraphBatch,
    input_fn: DomainFunction | None,
    kind: GraphComponentKind,
    key: str | None,
    /,
    *,
    eval_key: Key[Array, ""] = DOC_KEY0,
    owner: str = "GraphModel",
    **kwargs: Any,
) -> GraphIR:
    if input_fn is None:
        return graph
    values = input_fn(
        _full_entity_batch(batch, kind),
        key=eval_key,
        **kwargs,
    )
    if not isinstance(values, cx.Field):
        raise TypeError(f"{owner} input functions must evaluate to coordax.Field.")
    payload = jnp.asarray(values.data)
    if kind == "nodes":
        if key is None:
            return graph.replace(nodes=payload, validate=False)
        return graph.replace(
            nodes=_with_mapping_key(graph.nodes, key, payload),
            validate=False,
        )
    if kind == "edges":
        if key is None:
            return graph.replace(edges=payload, validate=False)
        return graph.replace(
            edges=_with_mapping_key(graph.edges, key, payload),
            validate=False,
        )
    if key is None:
        return graph.replace(globals=payload, validate=False)
    return graph.replace(
        globals=_with_mapping_key(graph.globals, key, payload),
        validate=False,
    )


class GraphModel(StrictModule, BatchEvaluator):
    """Batch-aware wrapper for `GraphIR -> GraphIR` graph models.

    The wrapper optionally evaluates `input_fn` on a full-node `GraphBatch` and
    installs the result as `graph.nodes` before calling `module`. Edge and global
    side inputs can be supplied with `edge_input_fn` and `global_input_fn`, which
    are evaluated over full edge/global batches and installed into `graph.edges`
    and `graph.globals`.

    When an input key is set, the corresponding input is inserted into the
    mapping-valued graph payload instead, preserving geometry, topology, or case
    metadata. `output_key` selects a named payload from mapping-valued model
    outputs. The selected output is returned as a `coordax.Field` over the
    current graph entity axis, making the result usable as a normal Phydrax
    `DomainFunction`.
    """

    module: Any
    input_fn: DomainFunction | None
    edge_input_fn: DomainFunction | None
    global_input_fn: DomainFunction | None
    output: GraphModelOutput
    input_key: str | None
    edge_input_key: str | None
    global_input_key: str | None
    output_key: str | None

    def __init__(
        self,
        module: Any,
        /,
        *,
        input_fn: DomainFunction | None = None,
        edge_input_fn: DomainFunction | None = None,
        global_input_fn: DomainFunction | None = None,
        output: GraphModelOutput = "nodes",
        input_key: str | None = None,
        edge_input_key: str | None = None,
        global_input_key: str | None = None,
        output_key: str | None = None,
    ):
        if output not in ("nodes", "edges", "globals"):
            raise ValueError("GraphModel output must be 'nodes', 'edges', or 'globals'.")
        self.module = module
        self.input_fn = input_fn
        self.edge_input_fn = edge_input_fn
        self.global_input_fn = global_input_fn
        self.output = output
        self.input_key = input_key
        self.edge_input_key = edge_input_key
        self.global_input_key = global_input_key
        self.output_key = output_key

    def _install_input(
        self,
        graph: GraphIR,
        batch: GraphBatch,
        input_fn: DomainFunction | None,
        kind: GraphComponentKind,
        key: str | None,
        /,
        *,
        eval_key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> GraphIR:
        return _install_graph_input(
            graph,
            batch,
            input_fn,
            kind,
            key,
            eval_key=eval_key,
            owner="GraphModel",
            **kwargs,
        )

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, GraphBatch):
            raise TypeError("GraphModel requires GraphBatch evaluation.")
        graph = batch.graph
        graph = self._install_input(
            graph,
            batch,
            self.input_fn,
            "nodes",
            self.input_key,
            eval_key=key,
            **kwargs,
        )
        graph = self._install_input(
            graph,
            batch,
            self.edge_input_fn,
            "edges",
            self.edge_input_key,
            eval_key=key,
            **kwargs,
        )
        graph = self._install_input(
            graph,
            batch,
            self.global_input_fn,
            "globals",
            self.global_input_key,
            eval_key=key,
            **kwargs,
        )

        out = self.module(graph)
        if not isinstance(out, GraphIR):
            raise TypeError("GraphModel module must return a phydrax.graph.GraphIR.")

        if self.output != batch.component_kind:
            raise ValueError(
                "GraphModel output must match the GraphBatch component kind for now; "
                f"output={self.output!r}, component_kind={batch.component_kind!r}."
            )

        if self.output == "nodes":
            payload = out.nodes
            payload = _select_mapping_key(payload, self.output_key, "nodes")
        elif self.output == "edges":
            payload = out.edges
            payload = _select_mapping_key(payload, self.output_key, "edges")
        else:
            payload = out.globals
            payload = _select_mapping_key(payload, self.output_key, "globals")

        if payload is None:
            raise ValueError(f"GraphModel output graph has no {self.output} payload.")

        arr = jnp.asarray(payload)[_entity_indices(batch)]
        axis = _graph_axis(batch)
        return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))


class GraphRolloutModel(StrictModule, BatchEvaluator):
    """Batch-aware wrapper for autoregressive `GraphIR -> GraphIR` rollouts.

    The wrapper mirrors `GraphModel` input handling, then repeatedly applies a
    graph stepper and returns one stacked feature payload as a `DomainFunction`.
    The graph entity axis is kept first and rollout time is returned as an
    unnamed trailing axis, so rollout outputs can be compared by normal Phydrax
    constraints.
    """

    stepper: Any
    steps: int
    include_initial: bool
    feature: GraphModelOutput
    input_fn: DomainFunction | None
    edge_input_fn: DomainFunction | None
    global_input_fn: DomainFunction | None
    input_key: str | None
    edge_input_key: str | None
    global_input_key: str | None
    output_key: str | None

    def __init__(
        self,
        stepper: Any,
        /,
        *,
        steps: int,
        include_initial: bool = True,
        feature: GraphModelOutput = "nodes",
        input_fn: DomainFunction | None = None,
        edge_input_fn: DomainFunction | None = None,
        global_input_fn: DomainFunction | None = None,
        input_key: str | None = None,
        edge_input_key: str | None = None,
        global_input_key: str | None = None,
        output_key: str | None = None,
    ):
        if int(steps) < 0:
            raise ValueError("GraphRolloutModel steps must be non-negative.")
        if feature not in ("nodes", "edges", "globals"):
            raise ValueError(
                "GraphRolloutModel feature must be 'nodes', 'edges', or 'globals'."
            )
        self.stepper = stepper
        self.steps = int(steps)
        self.include_initial = bool(include_initial)
        self.feature = feature
        self.input_fn = input_fn
        self.edge_input_fn = edge_input_fn
        self.global_input_fn = global_input_fn
        self.input_key = input_key
        self.edge_input_key = edge_input_key
        self.global_input_key = global_input_key
        self.output_key = output_key

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, GraphBatch):
            raise TypeError("GraphRolloutModel requires GraphBatch evaluation.")
        graph = batch.graph
        graph = _install_graph_input(
            graph,
            batch,
            self.input_fn,
            "nodes",
            self.input_key,
            eval_key=key,
            owner="GraphRolloutModel",
            **kwargs,
        )
        graph = _install_graph_input(
            graph,
            batch,
            self.edge_input_fn,
            "edges",
            self.edge_input_key,
            eval_key=key,
            owner="GraphRolloutModel",
            **kwargs,
        )
        graph = _install_graph_input(
            graph,
            batch,
            self.global_input_fn,
            "globals",
            self.global_input_key,
            eval_key=key,
            owner="GraphRolloutModel",
            **kwargs,
        )

        if self.feature != batch.component_kind:
            raise ValueError(
                "GraphRolloutModel feature must match the GraphBatch component kind "
                f"for now; feature={self.feature!r}, "
                f"component_kind={batch.component_kind!r}."
            )

        payload = rollout_features(
            self.stepper,
            graph,
            steps=self.steps,
            feature=self.feature,
            include_initial=self.include_initial,
        )
        payload = _select_mapping_key(payload, self.output_key, self.feature)
        if payload is None:
            raise ValueError(f"GraphRolloutModel rollout has no {self.feature} payload.")

        arr = jnp.asarray(payload)
        if arr.ndim < 2:
            raise ValueError(
                "GraphRolloutModel feature payloads must have a leading rollout axis "
                "and a graph entity axis."
            )
        arr = arr[:, _entity_indices(batch), ...]
        arr = jnp.moveaxis(arr, 0, 1)
        axis = _graph_axis(batch)
        return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))


__all__ = ["GraphModel", "GraphModelOutput", "GraphRolloutModel"]
