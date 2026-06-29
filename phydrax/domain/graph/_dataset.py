#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ...graph import batch_graphs, GraphIR, LayoutPlan
from .._components import _AbstractVarComponent
from .._domain import _AbstractUnaryDomain
from .._structure import ProductStructure
from ._batch import GRAPH_ENTITY_INDEX_KEY, GRAPH_GRAPH_INDEX_KEY, GraphBatch
from ._components import (
    graph_component_indices_for_graph,
    graph_component_kind,
    GraphComponentKind,
)


GRAPH_DATASET_INDEX_KEY = "__phydrax_graph_dataset_index__"
GRAPH_SAMPLE_INDEX_KEY = "__phydrax_graph_sample_index__"
GRAPH_ENTITY_OFFSET_KEY = "__phydrax_graph_entity_offset__"
GraphDatasetMeasureMode = Literal["probability", "count"]


def _to_axis_fields(tree: Any, axis: str, /) -> Any:
    def _leaf_to_field(value: Any) -> cx.Field:
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            raise ValueError("GraphDatasetDomain feature leaves must have an entity axis.")
        return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

    return jax.tree_util.tree_map(_leaf_to_field, tree)


def _take_tree(tree: Any, indices: Array, /) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf)[indices], tree)


def _entity_payload(graph: GraphIR, kind: GraphComponentKind, /) -> Any:
    if kind == "nodes":
        if graph.nodes is not None:
            return graph.nodes
        return jnp.arange(graph.num_nodes, dtype=jnp.int32)
    if kind == "edges":
        if graph.edges is not None:
            return graph.edges
        if graph.senders is None or graph.receivers is None:
            return jnp.zeros((0, 2), dtype=jnp.int32)
        return jnp.stack([graph.senders, graph.receivers], axis=-1)
    if graph.globals is not None:
        return graph.globals
    return jnp.arange(graph.num_graphs, dtype=jnp.int32)


def _graph_ids_for_kind(graph: GraphIR, kind: GraphComponentKind, /) -> Array:
    graph_ids = jnp.arange(graph.num_graphs, dtype=jnp.int32)
    if kind == "nodes":
        return jnp.repeat(
            graph_ids,
            graph.n_node,
            axis=0,
            total_repeat_length=graph.num_nodes,
        )
    if kind == "edges":
        return jnp.repeat(
            graph_ids,
            graph.n_edge,
            axis=0,
            total_repeat_length=graph.num_edges,
        )
    return graph_ids


def _size_for_kind(graph: GraphIR, kind: GraphComponentKind, /) -> int:
    if kind == "nodes":
        return int(graph.num_nodes)
    if kind == "edges":
        return int(graph.num_edges)
    return int(graph.num_graphs)


def _component_indices_for_graph(
    graph: GraphIR,
    component: _AbstractVarComponent,
    kind: GraphComponentKind,
    /,
) -> Array:
    return graph_component_indices_for_graph(graph, component, kind)


def _offsets_for_kind(graphs: Sequence[GraphIR], kind: GraphComponentKind, /) -> list[int]:
    offsets: list[int] = []
    running = 0
    for graph in graphs:
        offsets.append(running)
        running += _size_for_kind(graph, kind)
    return offsets


def _feature_tree_size(tree: Any, /) -> int | None:
    if tree is None:
        return None
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return None
    return int(jnp.asarray(leaves[0]).shape[0])


class GraphDatasetDomain(_AbstractUnaryDomain):
    """A finite dataset domain whose samples are sparse graph instances.

    Sampling a graph component draws graph cases, batches their full topology into
    one `GraphIR`, and materializes the selected node/edge/global entities as a
    `GraphBatch`. Whole-graph components (`Nodes`, `Edges`, `Globals`) select all
    entities in each sampled case. Explicit `NodeSet`/`EdgeSet`-style components
    are interpreted as local entity indices applied to every sampled graph.
    """

    graphs: tuple[GraphIR, ...]
    _label: str
    _measure_mode: GraphDatasetMeasureMode
    _layout: LayoutPlan | None

    def __init__(
        self,
        graphs: Sequence[GraphIR],
        /,
        *,
        label: str = "graph",
        measure: GraphDatasetMeasureMode = "probability",
        layout: LayoutPlan | None = None,
        validate: bool = True,
    ):
        if len(graphs) == 0:
            raise ValueError("GraphDatasetDomain requires at least one graph.")
        if measure not in ("probability", "count"):
            raise ValueError("GraphDatasetDomain measure must be 'probability' or 'count'.")
        graphs_tuple = tuple(graphs)
        for graph in graphs_tuple:
            if not isinstance(graph, GraphIR):
                raise TypeError("GraphDatasetDomain expects phydrax.graph.GraphIR values.")
            if validate:
                graph.validate()
        self.graphs = graphs_tuple
        self._label = str(label)
        self._measure_mode = measure
        self._layout = layout

    @property
    def label(self) -> str:
        return self._label

    @property
    def var_dim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self.graphs)

    @property
    def measure_mode(self) -> GraphDatasetMeasureMode:
        return self._measure_mode

    @property
    def layout(self) -> LayoutPlan | None:
        return self._layout

    def layout_for_batch_size(
        self,
        num_cases: int,
        /,
        *,
        multiple: int = 1,
    ) -> LayoutPlan:
        """Return a worst-case static layout for sampling `num_cases` graphs."""
        n = int(num_cases)
        if n <= 0:
            raise ValueError("num_cases must be positive.")
        if multiple <= 0:
            raise ValueError("multiple must be positive.")
        max_nodes = n * max(graph.num_nodes for graph in self.graphs)
        max_edges = n * max(graph.num_edges for graph in self.graphs)
        max_graphs = n * max(graph.num_graphs for graph in self.graphs)

        def _round_up(value: int) -> int:
            if value % multiple == 0:
                return value
            return ((value // multiple) + 1) * multiple

        return LayoutPlan(
            max_nodes=_round_up(max_nodes),
            max_edges=_round_up(max_edges),
            max_graphs=_round_up(max_graphs),
        )

    def with_layout(self, layout: LayoutPlan | None, /) -> "GraphDatasetDomain":
        """Return a copy that packs sampled graph batches with `layout`."""
        return GraphDatasetDomain(
            self.graphs,
            label=self.label,
            measure=self.measure_mode,
            layout=layout,
            validate=False,
        )

    def sample_indices(
        self,
        num_points: int,
        *,
        sampler: str = "uniform",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        del sampler
        n = int(num_points)
        if n < 0:
            raise ValueError("num_points must be non-negative.")
        if n == 0:
            return jnp.zeros((0,), dtype=jnp.int32)
        return jr.randint(
            key,
            shape=(n,),
            minval=0,
            maxval=self.size,
            dtype=jnp.int32,
        )

    def component_size(self, component: _AbstractVarComponent, /) -> int:
        kind = graph_component_kind(component)
        total = 0
        for graph in self.graphs:
            total += int(_component_indices_for_graph(graph, component, kind).shape[0])
        return total

    def component_measure(self, component: _AbstractVarComponent, /) -> Array:
        if self._measure_mode == "probability":
            return jnp.asarray(1.0, dtype=float)
        return jnp.asarray(float(self.component_size(component)), dtype=float)

    def sample_component(
        self,
        component: _AbstractVarComponent,
        num_points: int,
        *,
        structure: ProductStructure,
        label: str | None = None,
        sampler: str = "uniform",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> GraphBatch:
        indices = self.sample_indices(num_points, sampler=sampler, key=key)
        return self.points_from_indices(
            indices,
            component=component,
            structure=structure,
            label=label,
        )

    def points_from_indices(
        self,
        indices: ArrayLike,
        /,
        *,
        component: _AbstractVarComponent,
        structure: ProductStructure | None = None,
        label: str | None = None,
    ) -> GraphBatch:
        label_out = self.label if label is None else str(label)
        structure_in = structure or ProductStructure(((label_out,),))
        structure_out = structure_in.canonicalize((label_out,))
        axis = structure_out.axis_for(label_out)
        if axis is None:
            raise ValueError(
                f"GraphDatasetDomain sampling requires an axis for label {label_out!r}."
            )

        idx = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        idx_np = np.asarray(idx)
        if np.any(idx_np < 0) or np.any(idx_np >= self.size):
            raise ValueError(f"Graph dataset indices must be in [0, {self.size}).")

        selected_graphs = tuple(self.graphs[int(i)] for i in idx_np.tolist())
        if len(selected_graphs) == 0:
            raise ValueError("GraphDatasetDomain graph batches must be non-empty.")

        kind = graph_component_kind(component)
        real_batched = batch_graphs(selected_graphs, validate=True)
        batched = self._layout.pack(real_batched) if self._layout is not None else real_batched
        offsets = _offsets_for_kind(selected_graphs, kind)
        entity_parts: list[Array] = []
        dataset_parts: list[Array] = []
        sample_parts: list[Array] = []
        offset_parts: list[Array] = []
        for sample_index, (case_index, graph, offset) in enumerate(
            zip(idx_np.tolist(), selected_graphs, offsets, strict=True)
        ):
            local = _component_indices_for_graph(graph, component, kind)
            global_indices = local + jnp.asarray(offset, dtype=jnp.int32)
            entity_parts.append(global_indices)
            n_local = int(local.shape[0])
            dataset_parts.append(
                jnp.full(
                    (n_local,),
                    int(case_index),
                    dtype=jnp.int32,
                )
            )
            sample_parts.append(
                jnp.full((n_local,), int(sample_index), dtype=jnp.int32)
            )
            offset_parts.append(jnp.full((n_local,), int(offset), dtype=jnp.int32))
        entity_indices = jnp.concatenate(entity_parts, axis=0)
        dataset_indices = jnp.concatenate(dataset_parts, axis=0)
        sample_indices = jnp.concatenate(sample_parts, axis=0)
        entity_offsets = jnp.concatenate(offset_parts, axis=0)

        payload = _take_tree(_entity_payload(batched, kind), entity_indices)
        graph_ids = _graph_ids_for_kind(real_batched, kind)[entity_indices]
        points = {
            label_out: _to_axis_fields(payload, axis),
            GRAPH_ENTITY_INDEX_KEY: cx.Field(entity_indices, dims=(axis,)),
            GRAPH_GRAPH_INDEX_KEY: cx.Field(graph_ids, dims=(axis,)),
            GRAPH_DATASET_INDEX_KEY: cx.Field(dataset_indices, dims=(axis,)),
            GRAPH_SAMPLE_INDEX_KEY: cx.Field(sample_indices, dims=(axis,)),
            GRAPH_ENTITY_OFFSET_KEY: cx.Field(entity_offsets, dims=(axis,)),
        }
        return GraphBatch(
            points=frozendict(points),
            structure=structure_out,
            graph=batched,
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
        """Wrap a `GraphIR -> GraphIR` model as a graph-family `DomainFunction`."""
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
        """Wrap an autoregressive graph rollout as a graph-family `DomainFunction`."""
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
        if not isinstance(other, GraphDatasetDomain):
            return False
        if self.label != other.label:
            return False
        if self.measure_mode != other.measure_mode:
            return False
        if (self.layout is None) != (other.layout is None):
            return False
        if self.layout is not None and other.layout is not None:
            if self.layout.max_nodes != other.layout.max_nodes:
                return False
            if self.layout.max_edges != other.layout.max_edges:
                return False
            if self.layout.max_graphs != other.layout.max_graphs:
                return False
        if self.size != other.size:
            return False
        for graph_a, graph_b in zip(self.graphs, other.graphs, strict=True):
            if graph_a.n_node.shape != graph_b.n_node.shape:
                return False
            if graph_a.n_edge.shape != graph_b.n_edge.shape:
                return False
            if graph_a.edge_index.shape != graph_b.edge_index.shape:
                return False
            for a, b in (
                (graph_a.nodes, graph_b.nodes),
                (graph_a.edges, graph_b.edges),
                (graph_a.globals, graph_b.globals),
            ):
                if _feature_tree_size(a) != _feature_tree_size(b):
                    return False
        return True


__all__ = [
    "GRAPH_DATASET_INDEX_KEY",
    "GRAPH_ENTITY_OFFSET_KEY",
    "GRAPH_SAMPLE_INDEX_KEY",
    "GraphDatasetDomain",
    "GraphDatasetMeasureMode",
]
