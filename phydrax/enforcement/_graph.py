#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.conditions._ir import (
    AbstractConditionOperator,
    OperatorCapabilities,
    OperatorLinearization,
)
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
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
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
        raise KeyError(f"Graph payload does not contain type key {component.type_key!r}.")
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


def _coerce_target(
    target: DomainFunction | ArrayLike | None, u: DomainFunction, /
) -> DomainFunction:
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


class GraphRestrictionEvidence(StrictModule):
    """Exact finite restriction/scatter evidence for a graph component."""

    action_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    graph_label: str = eqx.field(static=True)
    component_kind: str = eqx.field(static=True)
    restriction_scope: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    orientation_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        action_id: str,
        provider_id: str,
        graph_label: str,
        component_kind: str,
        topology_id: str,
        orientation_id: str | None,
    ):
        self.action_id = str(action_id)
        self.provider_id = str(provider_id)
        self.graph_label = str(graph_label)
        self.component_kind = str(component_kind)
        self.restriction_scope = "exact_finite_graph_entities"
        self.topology_id = str(topology_id)
        self.orientation_id = (
            None if orientation_id is None else str(orientation_id)
        )


class _GraphRestrictedField(StrictModule, BatchEvaluator):
    value: DomainFunction
    component: DomainComponent
    graph_label: str = eqx.field(static=True)

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, GraphBatch):
            raise TypeError("Graph restriction requires GraphBatch evaluation.")
        if batch.graph_label != self.graph_label:
            raise ValueError(
                f"Graph restriction expects label {self.graph_label!r}, "
                f"got {batch.graph_label!r}."
            )
        field = self.value(batch, key=key, **kwargs)
        if not isinstance(field, cx.Field):
            raise TypeError("Graph restriction expects a coordax.Field output.")
        axis = batch.structure.axis_for(batch.graph_label)
        if axis is None or axis not in field.named_dims:
            raise ValueError("Graph field is missing the graph sampling axis.")
        axis_pos = field.dims.index(axis)
        data = jnp.moveaxis(jnp.asarray(field.data), axis_pos, 0)
        if int(data.shape[0]) != int(_entity_indices(batch).shape[0]):
            raise ValueError(
                "Graph restriction output size does not match the finite graph batch."
            )
        mask = _component_mask(batch, self.component, self.graph_label)
        mask = mask.reshape(mask.shape + (1,) * (data.ndim - 1))
        restricted = jnp.where(mask, data, jnp.zeros((), dtype=data.dtype))
        return cx.Field(jnp.moveaxis(restricted, 0, axis_pos), dims=field.dims)


class GraphRestriction(AbstractConditionOperator):
    """Typed linear restriction to a declared finite graph component."""

    field: str = eqx.field(static=True)
    component: DomainComponent
    graph_label: str = eqx.field(static=True)
    capabilities: OperatorCapabilities = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    orientation_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        /,
        *,
        graph_label: str | None = None,
    ):
        field_ = str(field)
        if not field_:
            raise ValueError("Graph restriction field name must be non-empty.")
        if isinstance(component, ComponentSum):
            raise TypeError("GraphRestriction requires one DomainComponent.")
        label = _graph_label_for_component(component, graph_label)
        selector = component.spec.selection_for(label)
        kind = graph_component_kind(selector)
        factor = component.domain.factor(label)
        graph = factor.graph
        topology_id = canonical_fingerprint(
            {
                "kind": "graph-topology-v1",
                "nodes": int(graph.num_nodes),
                "edges": int(graph.num_edges),
                "graphs": int(graph.num_graphs),
                "senders": array_tree_fingerprint(graph.senders),
                "receivers": array_tree_fingerprint(graph.receivers),
            }
        )
        orientation_id = None
        if isinstance(selector, CochainCells):
            orientation_id = canonical_fingerprint(
                {
                    "kind": "cochain-orientation-v1",
                    "degree": selector.degree,
                    "region": selector.region,
                    "selection": repr(selector),
                }
            )
        action_id = canonical_fingerprint(
            {
                "kind": "graph-restriction-action-v1",
                "field": field_,
                "graph_label": label,
                "component": repr(component.spec),
                "topology": topology_id,
                "orientation": orientation_id,
            }
        )
        self.field = field_
        self.component = component
        self.graph_label = label
        self.capabilities = OperatorCapabilities(is_linear=True)
        self.action_id = action_id
        self.topology_id = topology_id
        self.orientation_id = orientation_id

    def _restricted_field(self, value: DomainFunction, /) -> DomainFunction:
        metadata = dict(value.metadata)
        metadata.update(
            {
                "graph_restriction": True,
                "action_id": self.action_id,
                "exact_scope": "finite_graph_entities",
            }
        )
        return DomainFunction(
            domain=value.domain,
            deps=value.deps,
            func=_GraphRestrictedField(value, self.component, self.graph_label),
            metadata=metadata,
        )

    def _apply(self, values: Mapping[str, Any], /) -> DomainFunction:
        if self.field not in values:
            raise KeyError(f"Missing graph field {self.field!r}.")
        value = values[self.field]
        if not isinstance(value, DomainFunction):
            raise TypeError("GraphRestriction acts on DomainFunction values.")
        if not value.domain.same_support(self.component.domain):
            raise ValueError(
                "Restricted graph field must share the declared component support."
            )
        selector = self.component.spec.selection_for(self.graph_label)
        if isinstance(selector, CochainCells):
            field_spec = cochain_field_spec(value)
            if field_spec.degree != selector.degree:
                raise ValueError(
                    f"Cochain restriction selects degree {selector.degree}, but "
                    f"the field has degree {field_spec.degree}."
                )
        return self._restricted_field(value)

    def apply(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> DomainFunction:
        del key, kwargs
        return self._apply(values)

    def linear_action(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> DomainFunction:
        del key, kwargs
        return self._apply(values)

    def adjoint_action(
        self,
        value: Any,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        del value, key, kwargs
        raise TypeError(
            "GraphRestriction does not assume an unweighted graph pairing; "
            "a graph representation provider must supply the adjoint."
        )

    def linearize(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> OperatorLinearization:
        del values, key, kwargs
        raise TypeError("Globally linear graph restrictions do not linearize.")


class CochainAction(AbstractConditionOperator):
    """Degree/orientation-aware finite cochain restriction action."""

    restriction: GraphRestriction
    capabilities: OperatorCapabilities = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        /,
        *,
        graph_label: str | None = None,
    ):
        label = _graph_label_for_component(component, graph_label)
        if not isinstance(component.spec.selection_for(label), CochainCells):
            raise TypeError("CochainAction requires a CochainCells component.")
        self.restriction = GraphRestriction(field, component, graph_label=label)
        self.capabilities = self.restriction.capabilities

    @property
    def field(self) -> str:
        return self.restriction.field

    @property
    def component(self) -> DomainComponent:
        return self.restriction.component

    @property
    def graph_label(self) -> str:
        return self.restriction.graph_label

    @property
    def action_id(self) -> str:
        return self.restriction.action_id

    @property
    def topology_id(self) -> str:
        return self.restriction.topology_id

    @property
    def orientation_id(self) -> str | None:
        return self.restriction.orientation_id

    def apply(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> DomainFunction:
        return self.restriction.apply(values, key=key, **kwargs)

    def linear_action(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> DomainFunction:
        return self.restriction.linear_action(values, key=key, **kwargs)

    def adjoint_action(
        self,
        value: Any,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        return self.restriction.adjoint_action(value, key=key, **kwargs)

    def linearize(
        self,
        values: Mapping[str, Any],
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> OperatorLinearization:
        return self.restriction.linearize(values, key=key, **kwargs)


class _GraphResidualScatter(StrictModule, BatchEvaluator):
    base: DomainFunction
    target: DomainFunction
    component: DomainComponent
    graph_label: str = eqx.field(static=True)

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        if not isinstance(batch, GraphBatch):
            raise TypeError("Graph correction requires GraphBatch evaluation.")
        if batch.graph_label != self.graph_label:
            raise ValueError(
                f"Graph correction expects label {self.graph_label!r}, "
                f"got {batch.graph_label!r}."
            )
        base = self.base(batch, key=key, **kwargs)
        target = self.target(batch, key=key, **kwargs)
        if not isinstance(base, cx.Field) or not isinstance(target, cx.Field):
            raise TypeError("Graph correction expects coordax.Field outputs.")
        axis = batch.structure.axis_for(batch.graph_label)
        if axis is None or axis not in base.named_dims or axis not in target.named_dims:
            raise ValueError("Graph correction fields are missing the graph sampling axis.")
        axis_pos = base.dims.index(axis)
        data = jnp.moveaxis(jnp.asarray(base.data), axis_pos, 0)
        if int(data.shape[0]) != int(_entity_indices(batch).shape[0]):
            raise ValueError(
                "Graph correction output size does not match the finite graph batch."
            )
        target_data = jnp.moveaxis(
            jnp.asarray(target.data),
            target.dims.index(axis),
            0,
        )
        target_data = _broadcast_to_data(target_data, data)
        mask = _component_mask(batch, self.component, self.graph_label)
        mask = mask.reshape(mask.shape + (1,) * (data.ndim - 1))
        correction = jnp.where(
            mask,
            target_data - data,
            jnp.zeros((), dtype=data.dtype),
        )
        return cx.Field(jnp.moveaxis(correction, 0, axis_pos), dims=base.dims)


class GraphRestrictionCorrectionAction(StrictModule):
    """Exact scatter lift for one finite graph restriction."""

    restriction: GraphRestriction | CochainAction
    evidence: GraphRestrictionEvidence
    field_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, restriction: GraphRestriction | CochainAction, /):
        provider_id = canonical_fingerprint(
            {
                "kind": "graph-restriction-correction-v1",
                "action": restriction.action_id,
            }
        )
        self.restriction = restriction
        self.evidence = GraphRestrictionEvidence(
            action_id=restriction.action_id,
            provider_id=provider_id,
            graph_label=restriction.graph_label,
            component_kind=graph_component_kind(
                restriction.component.spec.selection_for(restriction.graph_label)
            ),
            topology_id=restriction.topology_id,
            orientation_id=restriction.orientation_id,
        )
        self.field_names = (restriction.field,)

    @property
    def provider_id(self) -> str:
        return self.evidence.provider_id

    def lift(self, product_residual: Any, /) -> tuple[DomainFunction, ...]:
        residual = product_residual
        if isinstance(residual, tuple):
            if len(residual) != 1:
                raise ValueError("Graph restriction lift expects one residual block.")
            residual = residual[0]
        if not isinstance(residual, DomainFunction):
            raise TypeError(
                "Graph restriction residuals must remain function-valued until "
                "GraphBatch evaluation."
            )
        restriction = (
            self.restriction.restriction
            if isinstance(self.restriction, CochainAction)
            else self.restriction
        )
        return (restriction._restricted_field(residual),)

    __call__ = lift

    def correction(
        self,
        base: DomainFunction,
        target: DomainFunction,
        /,
    ) -> DomainFunction:
        if not base.domain.same_support(self.restriction.component.domain):
            raise ValueError(
                "Graph correction base must share the restriction component support."
            )
        if not target.domain.same_support(base.domain):
            raise ValueError("Graph correction target must share the base support.")
        deps = tuple(
            label
            for label in base.domain.labels
            if label in base.deps or label in target.deps
        )
        return DomainFunction(
            domain=base.domain,
            deps=deps,
            func=_GraphResidualScatter(
                base,
                target,
                self.restriction.component,
                self.restriction.graph_label,
            ),
            metadata={
                "graph_restriction_correction": True,
                "provider_id": self.provider_id,
                "exact_scope": self.evidence.restriction_scope,
            },
        )


class GraphRestrictionCorrectionProvider(StrictModule):
    """Provider for exact graph restriction/scatter correction actions."""

    action: GraphRestrictionCorrectionAction

    def __init__(self, restriction: GraphRestriction, /):
        if type(restriction) is not GraphRestriction:
            raise TypeError(
                "GraphRestrictionCorrectionProvider requires GraphRestriction; "
                "use CochainCorrectionProvider for cochains."
            )
        self.action = GraphRestrictionCorrectionAction(restriction)

    @property
    def provider_id(self) -> str:
        return self.action.provider_id

    @property
    def evidence(self) -> GraphRestrictionEvidence:
        return self.action.evidence

    def candidate_action(self) -> GraphRestrictionCorrectionAction:
        return self.action


class CochainCorrectionProvider(StrictModule):
    """Provider for degree/orientation-preserving cochain scatter corrections."""

    action: GraphRestrictionCorrectionAction

    def __init__(self, restriction: CochainAction, /):
        if not isinstance(restriction, CochainAction):
            raise TypeError("CochainCorrectionProvider requires CochainAction.")
        self.action = GraphRestrictionCorrectionAction(restriction)

    @property
    def provider_id(self) -> str:
        return self.action.provider_id

    @property
    def evidence(self) -> GraphRestrictionEvidence:
        return self.action.evidence

    def candidate_action(self) -> GraphRestrictionCorrectionAction:
        return self.action


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
    restriction: GraphRestriction | CochainAction
    if isinstance(selector, CochainCells):
        restriction = CochainAction(
            "__legacy_graph_field__",
            component,
            graph_label=label,
        )
        correction_action = CochainCorrectionProvider(
            restriction
        ).candidate_action()
    else:
        restriction = GraphRestriction(
            "__legacy_graph_field__",
            component,
            graph_label=label,
        )
        correction_action = GraphRestrictionCorrectionProvider(
            restriction
        ).candidate_action()
    correction = correction_action.correction(u, target_fn)
    return u + correction


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


__all__ = [
    "CochainAction",
    "CochainCorrectionProvider",
    "GraphRestriction",
    "GraphRestrictionCorrectionAction",
    "GraphRestrictionCorrectionProvider",
    "GraphRestrictionEvidence",
    "enforce_cochain_values",
    "enforce_graph_values",
]
