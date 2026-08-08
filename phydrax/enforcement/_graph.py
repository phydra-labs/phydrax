#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import (
    BatchEvaluator,
    ComponentSum,
    DomainComponent,
    DomainFunction,
    Interior,
)
from phydrax.domain.graph import (
    cochain_field_spec,
    CochainCells,
    Edges,
    EdgeType,
    Globals,
    graph_component_indices,
    graph_component_kind,
    GRAPH_ENTITY_INDEX_KEY,
    GRAPH_ENTITY_OFFSET_KEY,
    GRAPH_GRAPH_INDEX_KEY,
    GraphBatch,
    has_cochain_field_spec,
    Nodes,
    NodeType,
)

from .._doc import DOC_KEY0
from .._strict import StrictModule


def _graph_label_for_component(
    component: DomainComponent,
    graph_label: str | None,
    /,
) -> str:
    if graph_label is not None:
        label = str(graph_label)
        if label not in component.domain.labels:
            raise KeyError(f"Label {label!r} not in domain {component.domain.labels}.")
        if component.domain.coordinate(label).kind != "graph":
            raise TypeError(f"Label {label!r} is not a graph-domain label.")
        return label

    labels = tuple(
        label
        for label in component.domain.labels
        if component.domain.coordinate(label).kind == "graph"
    )
    if len(labels) != 1:
        raise ValueError(
            "Could not infer a unique graph-domain label; pass graph_label explicitly."
        )
    return labels[0]


def _entity_indices(batch: GraphBatch, /) -> Array:
    field = batch.points.get(GRAPH_ENTITY_INDEX_KEY)
    if isinstance(field, cx.Field):
        return jnp.asarray(field.data, dtype=jnp.int32)
    if batch.component_kind == "nodes":
        size = int(batch.graph.num_nodes)
    elif batch.component_kind == "edges":
        size = int(batch.graph.num_edges)
    else:
        size = int(batch.graph.num_graphs)
    return jnp.arange(size, dtype=jnp.int32)


def _valid_entities(batch: GraphBatch, /) -> Array:
    field = batch.points.get(GRAPH_GRAPH_INDEX_KEY)
    if isinstance(field, cx.Field):
        return jnp.asarray(field.data, dtype=jnp.int32) >= 0
    return jnp.ones((_entity_indices(batch).shape[0],), dtype=bool)


def _local_entity_indices(batch: GraphBatch, /) -> Array:
    indices = _entity_indices(batch)
    offset_field = batch.points.get(GRAPH_ENTITY_OFFSET_KEY)
    if isinstance(offset_field, cx.Field):
        return indices - jnp.asarray(offset_field.data, dtype=jnp.int32)
    return indices


def _isin(values: Array, options: Array, /) -> Array:
    values = jnp.asarray(values, dtype=jnp.int32)
    options = jnp.asarray(options, dtype=jnp.int32)
    if int(options.shape[0]) == 0:
        return jnp.zeros(values.shape, dtype=bool)
    return jnp.any(values[:, None] == options[None, :], axis=1)


def _type_mask(batch: GraphBatch, component: NodeType | EdgeType, /) -> Array:
    payload = batch.points.get(batch.graph_label)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{type(component).__name__} enforcement requires mapping-valued graph payloads."
        )
    if component.type_key not in payload:
        raise KeyError(
            f"Graph payload does not contain type key {component.type_key!r}."
        )
    field = payload[component.type_key]
    if not isinstance(field, cx.Field):
        raise TypeError("Graph type payload must be a coordax.Field.")
    type_ids = jnp.asarray(field.data)
    if type_ids.ndim == 2 and int(type_ids.shape[1]) == 1:
        type_ids = type_ids[:, 0]
    if type_ids.ndim != 1:
        raise ValueError("Graph type payload must have shape (n,) or (n, 1).")
    return _isin(type_ids.astype(jnp.int32), component.type_ids)


def _cochain_mask(batch: GraphBatch, component: CochainCells, /) -> Array:
    payload = batch.points.get(batch.graph_label)
    if not isinstance(payload, Mapping):
        raise TypeError("CochainCells enforcement requires mapping-valued node payloads.")
    degree_field = payload.get("cell_dim")
    if not isinstance(degree_field, cx.Field):
        raise KeyError("CochainCells enforcement requires graph.nodes['cell_dim'].")
    mask = jnp.asarray(degree_field.data, dtype=jnp.int32) == component.degree
    if component.region == "all":
        return mask
    boundary_field = payload.get("boundary")
    if not isinstance(boundary_field, cx.Field):
        raise KeyError(
            "CochainCells boundary regions require graph.nodes['boundary'] metadata."
        )
    boundary = jnp.asarray(boundary_field.data, dtype=bool)
    return mask & (boundary if component.region == "boundary" else ~boundary)


def _component_mask(
    batch: GraphBatch,
    component: DomainComponent,
    graph_label: str,
    /,
) -> Array:
    selector = component.spec.selection_for(graph_label)
    kind = graph_component_kind(selector)
    if kind != batch.component_kind:
        return jnp.zeros((_entity_indices(batch).shape[0],), dtype=bool)

    valid = _valid_entities(batch)
    explicit = graph_component_indices(selector)
    if explicit is not None:
        return valid & _isin(_local_entity_indices(batch), explicit)

    if isinstance(selector, CochainCells):
        return valid & _cochain_mask(batch, selector)
    if isinstance(selector, (NodeType, EdgeType)):
        return valid & _type_mask(batch, selector)

    if isinstance(selector, (Interior, Nodes, Edges, Globals)):
        return valid

    return valid


def _coerce_target(target: DomainFunction | ArrayLike | None, u: DomainFunction, /) -> DomainFunction:
    if target is None:
        return DomainFunction(domain=u.domain, deps=(), func=0.0, metadata={})
    if isinstance(target, DomainFunction):
        if target.domain.labels == u.domain.labels:
            return target
        return target.promote(u.domain)
    return DomainFunction(domain=u.domain, deps=(), func=target, metadata={})


def _broadcast_to_data(value: Array, data: Array, /) -> Array:
    value = jnp.asarray(value)
    if value.ndim == 0:
        return jnp.broadcast_to(value, data.shape)
    try:
        return jnp.broadcast_to(value, data.shape)
    except ValueError:
        while value.ndim < data.ndim:
            value = jnp.expand_dims(value, axis=-1)
        return jnp.broadcast_to(value, data.shape)


class _GraphValueEnforcement(StrictModule, BatchEvaluator):
    u: DomainFunction
    target: DomainFunction
    component: DomainComponent
    graph_label: str

    def __init__(
        self,
        u: DomainFunction,
        target: DomainFunction,
        component: DomainComponent,
        graph_label: str,
        /,
    ):
        self.u = u
        self.target = target
        self.component = component
        self.graph_label = str(graph_label)

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, GraphBatch):
            raise TypeError("enforce_graph_values requires GraphBatch evaluation.")

        base = self.u(batch, key=key, **kwargs)
        target = self.target(batch, key=key, **kwargs)
        if not isinstance(base, cx.Field) or not isinstance(target, cx.Field):
            raise TypeError("Graph value enforcement expects coordax.Field outputs.")
        axis = batch.structure.axis_for(batch.graph_label)
        if axis is None or axis not in base.named_dims:
            raise ValueError("Base graph field is missing the graph sampling axis.")

        axis_pos = base.dims.index(axis)
        data = jnp.moveaxis(jnp.asarray(base.data), axis_pos, 0)
        target_data = jnp.moveaxis(jnp.asarray(target.data), target.dims.index(axis), 0)
        target_data = _broadcast_to_data(target_data, data)

        mask = _component_mask(batch, self.component, self.graph_label).astype(bool)
        while mask.ndim < data.ndim:
            mask = jnp.expand_dims(mask, axis=-1)

        out = jnp.where(mask, target_data, data)
        return cx.Field(jnp.moveaxis(out, 0, axis_pos), dims=base.dims)


def enforce_graph_values(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    target: DomainFunction | ArrayLike | None = None,
    graph_label: str | None = None,
) -> DomainFunction:
    """Return a graph ansatz that exactly overwrites values on a graph subset.

    `component` selects graph nodes, edges, or graph-level entries. During graph
    batch evaluation, values on that selected finite subset are replaced with
    `target`, while values outside the subset remain those of `u`.
    """
    if isinstance(component, ComponentSum):
        raise TypeError(
            "enforce_graph_values requires a DomainComponent, not a ComponentSum."
        )
    if not isinstance(u, DomainFunction):
        raise TypeError("enforce_graph_values expects a DomainFunction.")

    label = _graph_label_for_component(component, graph_label)
    selector = component.spec.selection_for(label)
    graph_component_kind(selector)
    if isinstance(selector, CochainCells):
        field_spec = cochain_field_spec(u)
        if field_spec.degree != selector.degree:
            raise ValueError(
                f"Cannot enforce degree-{selector.degree} cells on a degree-"
                f"{field_spec.degree} cochain field."
            )
    target_fn = _coerce_target(target, u)
    if isinstance(selector, CochainCells):
        if (
            has_cochain_field_spec(target_fn)
            and cochain_field_spec(target_fn) != field_spec
        ):
            raise ValueError(
                "Cochain enforcement targets must have the same degree, side, "
                "orientation, and sampling semantics as the base field."
            )
    deps = tuple(
        lbl
        for lbl in u.domain.labels
        if (lbl in u.deps) or (lbl in target_fn.deps)
    )
    return DomainFunction(
        domain=u.domain,
        deps=deps,
        func=_GraphValueEnforcement(u, target_fn, component, label),
        metadata=u.metadata,
    )

def enforce_cochain_values(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    target: DomainFunction | ArrayLike | None = None,
    graph_label: str | None = None,
) -> DomainFunction:
    """Exactly overwrite a declared cochain field on selected degree cells."""
    label = _graph_label_for_component(component, graph_label)
    selector = component.spec.selection_for(label)
    if not isinstance(selector, CochainCells):
        raise TypeError("enforce_cochain_values requires a CochainCells component.")
    return enforce_graph_values(
        u,
        component,
        target=target,
        graph_label=label,
    )


__all__ = ["enforce_cochain_values", "enforce_graph_values"]
