#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._frozendict import frozendict
from ...graph import GraphIR
from .._coordinate import CoordinateSpec
from .._domain import JointFactor
from .._factor_component import FactorComponent
from .._measure import BaseMeasure, ExactMass
from .._selection import Selection
from .._structure import SampleLayout
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
            raise ValueError(
                "GraphDomain feature leaves must have a leading entity axis."
            )
        return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

    return jax.tree_util.tree_map(_leaf_to_field, tree)


def _take_tree(tree: Any, indices: Array, /) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[indices], tree)


class GraphDomain(JointFactor):
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
        """Create a domain over one sparse graph.

        Parameters:
            graph: Graph topology and node, edge, and global payloads.
            label: Domain label used for sampled graph entities.
            measure: Component measure mode. `"probability"` normalizes sampled
                entity reductions; `"count"` scales by the selected entity count.
            validate: Validate the `GraphIR` before storing it.
        """
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
        """Domain label used for graph entity payloads."""
        return self._label

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.label,)

    @property
    def coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        return (CoordinateSpec(None, kind="graph", differentiable=False, dtype=None),)

    def bind_component(
        self,
        selections: Mapping[str, Selection],
        /,
    ) -> FactorComponent:
        if tuple(selections) != self.labels:
            raise ValueError(
                f"Graph factor {self.labels} requires exactly one ordered selection."
            )
        selection = selections[self.label]
        normalized = self.measure_mode == "probability"
        kind = "probability" if normalized else "counting"
        return FactorComponent(
            factor=self,
            selections=selections,
            measure=BaseMeasure(
                kind,
                ExactMass(self.component_measure(selection)),
                normalized=normalized,
            ),
        )

    def _replace_labels(
        self,
        labels: tuple[str, ...],
        /,
    ) -> "GraphDomain":
        return eqx.tree_at(lambda factor: factor._label, self, labels[0])

    @property
    def measure_mode(self) -> GraphMeasureMode:
        """Measure mode used for graph-component reductions."""
        return self._measure_mode

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the stored graph."""
        return int(self.graph.num_nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the stored graph."""
        return int(self.graph.num_edges)

    @property
    def num_graphs(self) -> int:
        """Number of graph-global entries in the stored graph."""
        return int(self.graph.num_graphs)

    def _size_for_kind(self, kind: GraphComponentKind, /) -> int:
        if kind == "nodes":
            return self.num_nodes
        if kind == "edges":
            return self.num_edges
        return self.num_graphs

    def _component_indices(
        self,
        component: Selection,
        kind: GraphComponentKind,
        /,
    ) -> Array:
        return graph_component_indices_for_graph(self.graph, component, kind)

    def component_size(self, component: Selection, /) -> int:
        """Return the number of entities selected by a graph component."""
        kind = graph_component_kind(component)
        return int(self._component_indices(component, kind).shape[0])

    def component_measure(self, component: Selection, /) -> Array:
        """Return the total measure assigned to a graph component."""
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
        component: Selection,
        num_points: int,
        *,
        structure: SampleLayout,
        label: str | None = None,
    ) -> GraphBatch:
        """Materialize all entities selected by `component`.

        `GraphDomain` represents one fixed graph, so sampling is deterministic and
        must request exactly `component_size(component)` entities.
        """
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
        """Wrap a `GraphIR -> GraphIR` model as a graph `DomainFunction`.

        The wrapped model receives the sampled batch topology and can return node,
        edge, or global outputs selected by `output`.
        """
        from phydrax.domain import DomainFunction

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
        """Wrap an autoregressive graph rollout as a graph `DomainFunction`.

        The stepper is applied for `steps` transitions on the sampled graph state,
        and the selected rollout feature is exposed as a graph-domain field.
        """
        from phydrax.domain import DomainFunction

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

    def _same_factor_support(self, other: object, /) -> bool:
        """Return whether another domain has the same public graph-domain shape."""
        if not isinstance(other, GraphDomain):
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
