#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._components import _AbstractVarComponent, Interior


GraphComponentKind = Literal["nodes", "edges", "globals"]


def _entity_indices(indices: ArrayLike, /) -> Array:
    idx = jnp.asarray(indices)
    if idx.ndim != 1:
        raise ValueError(f"Graph entity indices must be rank-1, got shape {idx.shape!r}.")
    if not jnp.issubdtype(idx.dtype, jnp.integer):
        raise TypeError(f"Graph entity indices must be integer dtype, got {idx.dtype!r}.")
    return idx.astype(jnp.int32)


class _AbstractGraphSubset(_AbstractVarComponent):
    """Base marker for explicit graph node/edge subsets."""

    indices: Array
    name: str | None

    def __init__(self, indices: ArrayLike, *, name: str | None = None):
        self.indices = _entity_indices(indices)
        self.name = None if name is None else str(name)


class _AbstractNodeSubset(_AbstractGraphSubset):
    """Base marker for explicit node subsets."""


class _AbstractEdgeSubset(_AbstractGraphSubset):
    """Base marker for explicit edge subsets."""


def _type_ids(type_ids: ArrayLike, /) -> Array:
    arr = jnp.asarray(type_ids)
    if arr.ndim == 0:
        arr = arr.reshape((1,))
    if arr.ndim != 1:
        raise ValueError(f"Graph type ids must be scalar or rank-1, got {arr.shape!r}.")
    if not jnp.issubdtype(arr.dtype, jnp.integer):
        raise TypeError(f"Graph type ids must be integer dtype, got {arr.dtype!r}.")
    return arr.astype(jnp.int32)


class _AbstractGraphTypeSubset(_AbstractVarComponent):
    """Base marker for graph subsets selected by entity type ids."""

    type_ids: Array
    type_key: str
    name: str | None

    def __init__(
        self,
        type_ids: ArrayLike,
        *,
        type_key: str = "type",
        name: str | None = None,
    ):
        self.type_ids = _type_ids(type_ids)
        self.type_key = str(type_key)
        self.name = None if name is None else str(name)


class _AbstractNodeTypeSubset(_AbstractGraphTypeSubset):
    """Base marker for node subsets selected by node type."""


class _AbstractEdgeTypeSubset(_AbstractGraphTypeSubset):
    """Base marker for edge subsets selected by edge type."""


class Nodes(_AbstractVarComponent):
    """Marker selecting all valid nodes of each sampled graph."""

    def __init__(self):
        """Create a node component marker."""


class Edges(_AbstractVarComponent):
    """Marker selecting all valid edges of each sampled graph."""

    def __init__(self):
        """Create an edge component marker."""


class Globals(_AbstractVarComponent):
    """Marker selecting graph-level entries of each sampled graph."""

    def __init__(self):
        """Create a graph-global component marker."""


class NodeSet(_AbstractNodeSubset):
    """Marker selecting explicit local node indices.

    In a graph dataset, the same local node indices are applied independently to
    every sampled graph case.
    """


class EdgeSet(_AbstractEdgeSubset):
    """Marker selecting explicit local edge indices.

    In a graph dataset, the same local edge indices are applied independently to
    every sampled graph case.
    """


class NodeType(_AbstractNodeTypeSubset):
    """Marker selecting nodes whose integer type id is in `type_ids`.

    The graph node payload must be mapping-valued and contain `type_key`, by
    default `graph.nodes["type"]`.
    """


class EdgeType(_AbstractEdgeTypeSubset):
    """Marker selecting edges whose integer type id is in `type_ids`.

    The graph edge payload must be mapping-valued and contain `type_key`, by
    default `graph.edges["type"]`.
    """


class BoundaryNodes(_AbstractNodeSubset):
    """Marker selecting explicit local nodes treated as a boundary set."""


class InteriorNodes(_AbstractNodeSubset):
    """Marker selecting explicit local nodes treated as an interior set."""


class BoundaryEdges(_AbstractEdgeSubset):
    """Marker selecting explicit local edges treated as a boundary set."""


class InterfaceEdges(_AbstractEdgeSubset):
    """Marker selecting explicit local edges treated as an interface set."""


def graph_component_kind(component: _AbstractVarComponent, /) -> GraphComponentKind:
    """Return the graph entity kind selected by a component marker."""
    if isinstance(component, Interior):
        return "nodes"
    if isinstance(component, (Nodes, _AbstractNodeSubset, _AbstractNodeTypeSubset)):
        return "nodes"
    if isinstance(component, (Edges, _AbstractEdgeSubset, _AbstractEdgeTypeSubset)):
        return "edges"
    if isinstance(component, Globals):
        return "globals"
    raise TypeError(
        "GraphDomain components must be Nodes(), Edges(), Globals(), explicit "
        "node/edge sets, or the "
        f"default Interior(); got {type(component).__name__}."
    )


def graph_component_indices(component: _AbstractVarComponent, /) -> Array | None:
    """Return explicit entity indices for graph subset components, if any."""
    if isinstance(component, _AbstractGraphSubset):
        return component.indices
    return None


def _size_for_kind(graph: Any, kind: GraphComponentKind, /) -> int:
    if kind == "nodes":
        return int(graph.num_nodes)
    if kind == "edges":
        return int(graph.num_edges)
    return int(graph.num_graphs)


def _entity_type_payload(graph: Any, component: _AbstractGraphTypeSubset, kind: GraphComponentKind, /) -> Array:
    payload = graph.nodes if kind == "nodes" else graph.edges
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{type(component).__name__} requires mapping-valued graph {kind} "
            f"with key {component.type_key!r}."
        )
    if component.type_key not in payload:
        raise KeyError(
            f"Graph {kind} payload does not contain type key {component.type_key!r}."
        )
    type_arr = jnp.asarray(payload[component.type_key])
    if type_arr.ndim == 2 and int(type_arr.shape[1]) == 1:
        type_arr = type_arr[:, 0]
    if type_arr.ndim != 1:
        raise ValueError(
            f"Graph {kind} type ids must have shape (n,) or (n, 1); got {type_arr.shape!r}."
        )
    if not jnp.issubdtype(type_arr.dtype, jnp.integer):
        raise TypeError(f"Graph {kind} type ids must be integer dtype.")
    return type_arr.astype(jnp.int32)


def _validate_explicit_indices(idx: Array, *, size: int, kind: GraphComponentKind) -> Array:
    idx = jnp.asarray(idx, dtype=jnp.int32)
    idx_np = np.asarray(idx)
    if np.any(idx_np < 0):
        raise ValueError("Graph component indices must be non-negative.")
    if np.any(idx_np >= size):
        raise ValueError(f"Graph component index out of bounds for {kind}: size is {size}.")
    if np.unique(idx_np).shape[0] != idx_np.shape[0]:
        raise ValueError("Graph component indices must be unique.")
    return idx


def graph_component_indices_for_graph(
    graph: Any,
    component: _AbstractVarComponent,
    kind: GraphComponentKind,
    /,
) -> Array:
    """Return graph entity indices for explicit and type-based graph components."""
    size = _size_for_kind(graph, kind)
    explicit = graph_component_indices(component)
    if explicit is not None:
        return _validate_explicit_indices(explicit, size=size, kind=kind)
    if isinstance(component, _AbstractGraphTypeSubset):
        if kind == "globals":
            raise TypeError("Graph type subsets can select only nodes or edges.")
        type_arr = _entity_type_payload(graph, component, kind)
        wanted = jnp.asarray(component.type_ids, dtype=jnp.int32)
        mask = jnp.any(type_arr[:, None] == wanted[None, :], axis=1)
        if kind == "nodes" and graph.node_mask is not None:
            mask = mask & graph.node_mask
        if kind == "edges" and graph.edge_mask is not None:
            mask = mask & graph.edge_mask
        return jnp.asarray(np.nonzero(np.asarray(mask))[0], dtype=jnp.int32)
    return jnp.arange(size, dtype=jnp.int32)


__all__ = [
    "BoundaryEdges",
    "BoundaryNodes",
    "EdgeType",
    "EdgeSet",
    "Edges",
    "Globals",
    "GraphComponentKind",
    "InteriorNodes",
    "InterfaceEdges",
    "NodeType",
    "NodeSet",
    "Nodes",
    "graph_component_indices_for_graph",
    "graph_component_indices",
    "graph_component_kind",
]
