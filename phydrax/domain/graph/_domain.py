#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._frozendict import frozendict
from ...graph import GraphIR
from .._components import _AbstractVarComponent
from .._domain import _AbstractUnaryDomain
from .._structure import ProductStructure
from ._batch import GRAPH_ENTITY_INDEX_KEY, GRAPH_GRAPH_INDEX_KEY, GraphBatch
from ._components import (
    graph_component_indices_for_graph,
    graph_component_kind,
    GraphComponentKind,
)


GraphMeasureMode = Literal["probability", "count"]


def _feature_tree_size(tree: Any, /) -> int | None:
    if tree is None:
        return None
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return None
    return int(jnp.asarray(leaves[0]).shape[0])


def _to_axis_fields(tree: Any, axis: str, /) -> Any:
    def _leaf_to_field(value: Any) -> cx.Field:
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            raise ValueError("GraphDomain feature leaves must have a leading entity axis.")
        return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

    return jax.tree_util.tree_map(_leaf_to_field, tree)


def _take_tree(tree: Any, indices: Array, /) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[indices], tree)


class GraphDomain(_AbstractUnaryDomain):
    """A unary Phydrax domain over a finite sparse graph.

    `GraphDomain` makes nodes, edges, or graph-level entries available as finite
    measure spaces inside the normal Phydrax domain/function/operator pipeline.
    The stored `GraphIR` remains non-trainable domain state.
    """

    graph: GraphIR
    _label: str
    _measure_mode: GraphMeasureMode

    def __init__(
        self,
        graph: GraphIR,
        /,
        *,
        label: str = "graph",
        measure: GraphMeasureMode = "probability",
        validate: bool = True,
    ):
        if not isinstance(graph, GraphIR):
            raise TypeError("GraphDomain expects a phydrax.graph.GraphIR instance.")
        if validate:
            graph.validate()
        if measure not in ("probability", "count"):
            raise ValueError("GraphDomain measure must be 'probability' or 'count'.")
        self.graph = graph
        self._label = str(label)
        self._measure_mode = measure

    @property
    def label(self) -> str:
        return self._label

    @property
    def var_dim(self) -> int:
        return 1

    @property
    def measure_mode(self) -> GraphMeasureMode:
        return self._measure_mode

    @property
    def num_nodes(self) -> int:
        return int(self.graph.num_nodes)

    @property
    def num_edges(self) -> int:
        return int(self.graph.num_edges)

    @property
    def num_graphs(self) -> int:
        return int(self.graph.num_graphs)

    def _size_for_kind(self, kind: GraphComponentKind, /) -> int:
        if kind == "nodes":
            return self.num_nodes
        if kind == "edges":
            return self.num_edges
        return self.num_graphs

    def _component_indices(
        self,
        component: _AbstractVarComponent,
        kind: GraphComponentKind,
        /,
    ) -> Array:
        return graph_component_indices_for_graph(self.graph, component, kind)

    def component_size(self, component: _AbstractVarComponent, /) -> int:
        kind = graph_component_kind(component)
        return int(self._component_indices(component, kind).shape[0])

    def component_measure(self, component: _AbstractVarComponent, /) -> Array:
        if self._measure_mode == "probability":
            return jnp.asarray(1.0, dtype=float)
        return jnp.asarray(float(self.component_size(component)), dtype=float)

    def _graph_ids_for_kind(self, kind: GraphComponentKind, /) -> Array:
        graph_ids = jnp.arange(self.num_graphs, dtype=jnp.int32)
        if kind == "nodes":
            return jnp.repeat(
                graph_ids,
                self.graph.n_node,
                axis=0,
                total_repeat_length=self.num_nodes,
            )
        if kind == "edges":
            return jnp.repeat(
                graph_ids,
                self.graph.n_edge,
                axis=0,
                total_repeat_length=self.num_edges,
            )
        return graph_ids

    def _entity_payload(self, kind: GraphComponentKind, /) -> Any:
        if kind == "nodes":
            if self.graph.nodes is not None:
                return self.graph.nodes
            return jnp.arange(self.num_nodes, dtype=jnp.int32)
        if kind == "edges":
            if self.graph.edges is not None:
                return self.graph.edges
            if self.graph.senders is None or self.graph.receivers is None:
                return jnp.zeros((0, 2), dtype=jnp.int32)
            return jnp.stack([self.graph.senders, self.graph.receivers], axis=-1)
        if self.graph.globals is not None:
            return self.graph.globals
        return jnp.arange(self.num_graphs, dtype=jnp.int32)

    def sample_component(
        self,
        component: _AbstractVarComponent,
        num_points: int,
        *,
        structure: ProductStructure,
        label: str | None = None,
    ) -> GraphBatch:
        """Materialize a full graph entity batch for the selected component."""
        label_out = self.label if label is None else str(label)
        structure_out = structure.canonicalize((label_out,))
        axis = structure_out.axis_for(label_out)
        if axis is None:
            raise ValueError(
                f"GraphDomain sampling requires a sampling axis for label {label_out!r}."
            )

        kind = graph_component_kind(component)
        entity_indices = self._component_indices(component, kind)
        size = int(entity_indices.shape[0])
        n = int(num_points)
        if n != size:
            raise ValueError(
                "GraphDomain currently supports full-entity sampling only; requested "
                f"{n} point(s) for {kind}, but the selected graph component has {size}."
            )

        payload = _take_tree(self._entity_payload(kind), entity_indices)
        graph_ids = self._graph_ids_for_kind(kind)[entity_indices]
        points = {
            label_out: _to_axis_fields(payload, axis),
            GRAPH_ENTITY_INDEX_KEY: cx.Field(entity_indices, dims=(axis,)),
            GRAPH_GRAPH_INDEX_KEY: cx.Field(graph_ids, dims=(axis,)),
        }
        return GraphBatch(
            points=frozendict(points),
            structure=structure_out,
            graph=self.graph,
            graph_label=label_out,
            component_kind=kind,
        )

    def GraphModel(
        self,
        model: Any,
        /,
        *,
        input_fn: Any = None,
        edge_input_fn: Any = None,
        global_input_fn: Any = None,
        output: Literal["nodes", "edges", "globals"] = "nodes",
        input_key: str | None = None,
        edge_input_key: str | None = None,
        global_input_key: str | None = None,
        output_key: str | None = None,
    ):
        """Wrap a `GraphIR -> GraphIR` model as a graph `DomainFunction`."""
        from ...domain._function import DomainFunction
        from ...nn import GraphModel

        return DomainFunction(
            domain=self,
            deps=(self.label,),
            func=GraphModel(
                model,
                input_fn=input_fn,
                edge_input_fn=edge_input_fn,
                global_input_fn=global_input_fn,
                output=output,
                input_key=input_key,
                edge_input_key=edge_input_key,
                global_input_key=global_input_key,
                output_key=output_key,
            ),
        )

    def GraphRolloutModel(
        self,
        stepper: Any,
        /,
        *,
        steps: int,
        include_initial: bool = True,
        feature: Literal["nodes", "edges", "globals"] = "nodes",
        input_fn: Any = None,
        edge_input_fn: Any = None,
        global_input_fn: Any = None,
        input_key: str | None = None,
        edge_input_key: str | None = None,
        global_input_key: str | None = None,
        output_key: str | None = None,
    ):
        """Wrap an autoregressive graph rollout as a graph `DomainFunction`."""
        from ...domain._function import DomainFunction
        from ...nn import GraphRolloutModel

        return DomainFunction(
            domain=self,
            deps=(self.label,),
            func=GraphRolloutModel(
                stepper,
                steps=steps,
                include_initial=include_initial,
                feature=feature,
                input_fn=input_fn,
                edge_input_fn=edge_input_fn,
                global_input_fn=global_input_fn,
                input_key=input_key,
                edge_input_key=edge_input_key,
                global_input_key=global_input_key,
                output_key=output_key,
            ),
        )

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, GraphDomain):
            return False
        if self.label != other.label:
            return False
        if self.measure_mode != other.measure_mode:
            return False
        if self.graph.n_node.shape != other.graph.n_node.shape:
            return False
        if self.graph.n_edge.shape != other.graph.n_edge.shape:
            return False
        if self.graph.edge_index.shape != other.graph.edge_index.shape:
            return False

        for a, b in (
            (self.graph.nodes, other.graph.nodes),
            (self.graph.edges, other.graph.edges),
            (self.graph.globals, other.graph.globals),
        ):
            size_a = _feature_tree_size(a)
            size_b = _feature_tree_size(b)
            if size_a != size_b:
                return False

        return True


__all__ = ["GraphDomain", "GraphMeasureMode"]
