#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native graph and simplicial topology for operator sample sets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.core as jcore
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....graph._ir import batch_graphs, GraphIR, unbatch_graph


if TYPE_CHECKING:
    from ._operator import FunctionSamples


OperatorTopologyEntity: TypeAlias = Literal["node", "edge", "global"]
OperatorTopologyKind: TypeAlias = Literal["graph", "simplicial", "cell_complex"]
OperatorTopologySite: TypeAlias = Literal[
    "node",
    "edge",
    "face",
    "cell",
    "vertex",
    "point",
    "global",
]


def _contains_tracer(tree: Any, /) -> bool:
    return any(isinstance(leaf, jcore.Tracer) for leaf in jax.tree_util.tree_leaves(tree))


def _graph_leading_size(tree: Any, /) -> int | None:
    if tree is None:
        return None
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return None
    return int(jnp.asarray(leaves[0]).shape[0])


def _entity_counts(graph: GraphIR, entity: OperatorTopologyEntity, /) -> Array:
    if entity == "node":
        return graph.n_node
    if entity == "edge":
        return graph.n_edge
    return jnp.ones((graph.num_graphs,), dtype=jnp.int32)


def _entity_payload(graph: GraphIR, entity: OperatorTopologyEntity, /) -> Any:
    if entity == "node":
        return graph.nodes
    if entity == "edge":
        return graph.edges
    return graph.globals


def _entity_mask(graph: GraphIR, entity: OperatorTopologyEntity, /) -> Array | None:
    if entity == "node":
        return graph.node_mask
    if entity == "edge":
        return graph.edge_mask
    return graph.graph_mask


def operator_graph_fingerprint(graph: GraphIR, /) -> str:
    """Return a deterministic digest of one canonical graph representation."""

    digest = hashlib.sha256()
    tree = (
        graph.nodes,
        graph.edges,
        graph.senders,
        graph.receivers,
        graph.globals,
        graph.n_node,
        graph.n_edge,
        graph.node_mask,
        graph.edge_mask,
        graph.graph_mask,
    )
    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest.update(repr(structure).encode("utf-8"))
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
def _derived_graph_fingerprint(
    fingerprint: str,
    operation: tuple[Any, ...],
    /,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(fingerprint).encode("utf-8"))
    digest.update(repr(operation).encode("utf-8"))
    return digest.hexdigest()




class OperatorTopology(StrictModule, NonTrainableState):
    """Graph entities aligned with one sampled operator function.

    ``sample_entities`` stores graph-local entity indices with shape
    ``case_shape + sample_shape``. ``-1`` marks padded samples. ``entity`` says
    whether those indices address ``GraphIR`` nodes, edges, or graph globals;
    ``site`` retains the physical cell semantics (for example, simplicial faces
    are represented as graph nodes).
    """

    graph: GraphIR
    sample_entities: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    kind: OperatorTopologyKind = eqx.field(static=True)
    site: OperatorTopologySite = eqx.field(static=True)
    entity: OperatorTopologyEntity = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    graph_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        graph: GraphIR,
        sample_entities: Any,
        /,
        *,
        case_shape: Sequence[int] = (),
        kind: OperatorTopologyKind = "graph",
        site: OperatorTopologySite = "node",
        entity: OperatorTopologyEntity = "node",
        validate: bool = True,
        _graph_fingerprint: str | None = None,
    ):
        if not isinstance(graph, GraphIR):
            raise TypeError("OperatorTopology graph must be a GraphIR.")
        cases = tuple(int(size) for size in case_shape)
        if any(size <= 0 for size in cases):
            raise ValueError("OperatorTopology case dimensions must be positive.")
        if kind not in ("graph", "simplicial", "cell_complex"):
            raise ValueError(
                "OperatorTopology kind must be 'graph', 'simplicial', or 'cell_complex'."
            )
        if site not in ("node", "edge", "face", "cell", "vertex", "point", "global"):
            raise ValueError("Unknown OperatorTopology site.")
        if entity not in ("node", "edge", "global"):
            raise ValueError("OperatorTopology entity must be 'node', 'edge', or 'global'.")
        mapping = jnp.asarray(sample_entities)
        if not jnp.issubdtype(mapping.dtype, jnp.integer):
            raise TypeError("OperatorTopology sample_entities must have integer dtype.")
        mapping = mapping.astype(jnp.int32)
        if mapping.ndim <= len(cases) or tuple(mapping.shape[: len(cases)]) != cases:
            raise ValueError(
                "OperatorTopology sample_entities must have shape "
                "case_shape + non-empty sample_shape."
            )
        expected_graphs = prod(cases) if cases else 1
        if graph.num_graphs != expected_graphs:
            raise ValueError(
                f"OperatorTopology requires {expected_graphs} graph(s) for case shape "
                f"{cases}; got {graph.num_graphs}."
            )
        counts = _entity_counts(graph, entity)
        payload = _entity_payload(graph, entity)
        leading = _graph_leading_size(payload)
        mask = _entity_mask(graph, entity)
        if leading is None and mask is not None:
            leading = int(mask.shape[0])
        if leading is None:
            if _contains_tracer(counts):
                raise ValueError(
                    "Traced topology graphs without entity payloads require eager "
                    "construction."
                )
            leading = int(np.asarray(counts).sum())
        self.graph = graph
        self.sample_entities = mapping
        self.case_shape = cases
        self.kind = kind
        self.site = site
        self.entity = entity
        self.entity_count = int(leading)
        self.graph_fingerprint = (
            operator_graph_fingerprint(graph)
            if _graph_fingerprint is None
            else str(_graph_fingerprint)
        )
        if validate:
            self.validate()

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.sample_entities.shape[len(self.case_shape) :])

    @classmethod
    def from_graph(
        cls,
        graph: GraphIR,
        sample_entities: Any | None = None,
        /,
        *,
        case_shape: Sequence[int] = (),
        site: OperatorTopologySite = "node",
        entity: OperatorTopologyEntity | None = None,
        validate: bool = True,
    ) -> "OperatorTopology":
        """Bind sampled sites to entities of an existing canonical graph."""

        cases = tuple(int(size) for size in case_shape)
        resolved_entity: OperatorTopologyEntity
        if entity is None:
            resolved_entity = "edge" if site == "edge" else "global" if site == "global" else "node"
        else:
            resolved_entity = entity
        counts_array = _entity_counts(graph, resolved_entity)
        if sample_entities is None:
            if _contains_tracer(counts_array):
                raise ValueError(
                    "Default sample-entity inference requires eager graph counts."
                )
            counts = np.asarray(counts_array, dtype=np.int32)
            if counts.size != (prod(cases) if cases else 1):
                raise ValueError("case_shape does not match the graph count.")
            width = int(counts.max(initial=0))
            inferred = np.full((counts.size, width), -1, dtype=np.int32)
            for index, count in enumerate(counts):
                inferred[index, : int(count)] = np.arange(int(count), dtype=np.int32)
            sample_entities = inferred.reshape(cases + (width,)) if cases else inferred[0]
        return cls(
            graph,
            sample_entities,
            case_shape=cases,
            kind="graph",
            site=site,
            entity=resolved_entity,
            validate=validate,
        )

    @classmethod
    def from_simplicial(
        cls,
        complex_graph: Any,
        /,
        *,
        site: Literal["vertex", "edge", "face"] = "vertex",
        validate: bool = True,
    ) -> "OperatorTopology":
        """Bind a sample set to cells of a canonical simplicial-complex graph."""

        from ....graph._simplicial import SimplicialComplexGraph

        if not isinstance(complex_graph, SimplicialComplexGraph):
            raise TypeError("from_simplicial requires a SimplicialComplexGraph.")
        if site == "vertex":
            sample_nodes = complex_graph.vertex_cells
        elif site == "edge":
            sample_nodes = complex_graph.edge_cells
        elif site == "face":
            sample_nodes = complex_graph.face_cells
        else:
            raise ValueError("Simplicial sample site must be 'vertex', 'edge', or 'face'.")
        return cls(
            complex_graph.graph,
            sample_nodes,
            kind="simplicial",
            site=site,
            validate=validate,
        )
    @classmethod
    def from_cochain(
        cls,
        complex_ir: Any,
        degree: int,
        /,
        *,
        sample_cells: Any | None = None,
        validate: bool = True,
    ) -> "OperatorTopology":
        """Bind samples to one cochain degree of a canonical cell complex."""

        from ....graph._cochain import CochainComplexIR

        if not isinstance(complex_ir, CochainComplexIR):
            raise TypeError("from_cochain requires a CochainComplexIR.")
        resolved_degree = int(degree)
        if resolved_degree < 0 or resolved_degree > complex_ir.max_degree:
            raise ValueError(
                f"Cochain degree must lie in [0, {complex_ir.max_degree}]."
            )
        if sample_cells is None:
            local_cells = np.arange(
                complex_ir.cell_counts[resolved_degree], dtype=np.int32
            )
        else:
            local_cells = np.asarray(sample_cells)
            if local_cells.ndim != 1 or not np.issubdtype(
                local_cells.dtype, np.integer
            ):
                raise TypeError("sample_cells must be a rank-1 integer array.")
            if np.any(local_cells < 0) or np.any(
                local_cells >= complex_ir.cell_counts[resolved_degree]
            ):
                raise ValueError("sample_cells contain out-of-range degree-local indices.")
            local_cells = local_cells.astype(np.int32, copy=False)
        sample_nodes = local_cells + complex_ir.cell_offsets[resolved_degree]
        return cls(
            complex_ir.graph,
            sample_nodes,
            kind="cell_complex",
            site="cell",
            entity="node",
            validate=validate,
        )

    def validate(self) -> None:
        """Validate graph counts, local mappings, uniqueness, and entity masks."""

        self.graph.validate()
        counts_array = _entity_counts(self.graph, self.entity)
        if _contains_tracer((counts_array, self.sample_entities)):
            return
        counts = np.asarray(counts_array, dtype=np.int64)
        mappings = np.asarray(self.sample_entities).reshape((counts.size, -1))
        offsets = np.concatenate((np.zeros((1,), dtype=np.int64), np.cumsum(counts[:-1])))
        graph_mask = (
            np.ones((counts.size,), dtype=bool)
            if self.graph.graph_mask is None
            else np.asarray(self.graph.graph_mask, dtype=bool)
        )
        entity_mask_array = _entity_mask(self.graph, self.entity)
        entity_mask = (
            None
            if entity_mask_array is None
            else np.asarray(entity_mask_array, dtype=bool)
        )
        for case, (mapping, count, offset, graph_valid) in enumerate(
            zip(mappings, counts, offsets, graph_mask, strict=True)
        ):
            valid = mapping >= 0
            if np.any(mapping < -1) or np.any(mapping[valid] >= count):
                raise ValueError(
                    f"OperatorTopology sample_entities for graph {case} must lie in "
                    f"[-1, {int(count)})."
                )
            selected = mapping[valid]
            if np.unique(selected).size != selected.size:
                raise ValueError(
                    f"OperatorTopology sample_entities for graph {case} must be unique."
                )
            if not graph_valid and selected.size:
                raise ValueError("Masked graphs cannot own mapped sample entities.")
            if entity_mask is not None and selected.size and not np.all(
                entity_mask[int(offset) + selected]
            ):
                raise ValueError(
                    "OperatorTopology cannot map samples to masked graph entities."
                )

    def absolute_sample_entities(self) -> Array:
        """Return absolute indices into the batched ``GraphIR`` entity axis."""

        counts = _entity_counts(self.graph, self.entity)
        offsets = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.cumsum(counts[:-1], dtype=jnp.int32),
            )
        ).reshape(self.case_shape + (1,) * len(self.sample_shape))
        return jnp.where(self.sample_entities >= 0, self.sample_entities + offsets, -1)

    def replace_sample_entities(self, sample_entities: Any, /) -> "OperatorTopology":
        """Return the same graph with a different sample-to-entity alignment."""

        return OperatorTopology(
            self.graph,
            sample_entities,
            case_shape=self.case_shape,
            kind=self.kind,
            site=self.site,
            entity=self.entity,
            _graph_fingerprint=self.graph_fingerprint,
        )


def pad_operator_topology(topology: OperatorTopology, size: int, /) -> OperatorTopology:
    """Pad a one-dimensional sample mapping with the ``-1`` sentinel."""

    if len(topology.sample_shape) != 1:
        raise ValueError("Only one-dimensional topology sample mappings can be padded.")
    target = int(size)
    current = topology.sample_shape[0]
    if target < current:
        raise ValueError(f"Cannot pad topology mapping of size {current} to {target}.")
    padding = [(0, 0)] * topology.sample_entities.ndim
    padding[-1] = (0, target - current)
    return topology.replace_sample_entities(jnp.pad(topology.sample_entities, padding, constant_values=-1))


def take_operator_topology(
    topology: OperatorTopology,
    indices: Any,
    /,
) -> OperatorTopology:
    """Select flattened sample sites while retaining the complete relation graph."""

    if topology.case_shape:
        raise ValueError("take_operator_topology expects one unbatched geometry case.")
    flattened = topology.sample_entities.reshape((-1,))
    selected = jnp.take(flattened, jnp.asarray(indices, dtype=jnp.int32), axis=0)
    return topology.replace_sample_entities(selected)


def stack_operator_topologies(
    topologies: Sequence[OperatorTopology],
    /,
) -> OperatorTopology:
    """Stack compatible topologies along a new leading geometry-case axis."""

    items = tuple(topologies)
    if not items:
        raise ValueError("stack_operator_topologies requires at least one topology.")
    first = items[0]
    if any(item.case_shape != first.case_shape for item in items[1:]):
        raise ValueError("Topology case shapes must match when stacking.")
    if any(
        (item.kind, item.site, item.entity) != (first.kind, first.site, first.entity)
        for item in items[1:]
    ):
        raise ValueError("Topology kind, sample site, and entity must match when stacking.")
    shapes = tuple(item.sample_shape for item in items)
    if len(set(shapes)) == 1:
        prepared = items
    elif all(len(shape) == 1 for shape in shapes):
        target = max(shape[0] for shape in shapes)
        prepared = tuple(pad_operator_topology(item, target) for item in items)
    else:
        raise ValueError("Only one-dimensional topology mappings may vary when stacking.")
    return OperatorTopology(
        batch_graphs(tuple(item.graph for item in prepared)),
        jnp.stack(tuple(item.sample_entities for item in prepared), axis=0),
        case_shape=(len(prepared),) + first.case_shape,
        kind=first.kind,
        site=first.site,
        entity=first.entity,
    )


def slice_operator_topology(
    topology: OperatorTopology,
    index: Any,
    axis: int,
    /,
) -> OperatorTopology:
    """Index one geometry case axis and retain the corresponding sparse graphs."""

    position = int(axis)
    if position < 0:
        position += len(topology.case_shape)
    if position < 0 or position >= len(topology.case_shape):
        raise ValueError("Topology case axis index is out of range.")
    if _contains_tracer(index):
        raise ValueError("Topology case slicing requires eager indices.")
    case_ids = np.arange(prod(topology.case_shape), dtype=np.int32).reshape(
        topology.case_shape
    )
    selected_ids = np.take(case_ids, index, axis=position)
    graphs = unbatch_graph(topology.graph)
    selected_graphs = tuple(graphs[int(value)] for value in selected_ids.reshape(-1))
    selected_graph = batch_graphs(selected_graphs)
    selected_entities = jnp.take(topology.sample_entities, index, axis=position)
    if isinstance(index, (int, np.integer)):
        case_shape = topology.case_shape[:position] + topology.case_shape[position + 1 :]
    else:
        case_shape = tuple(
            int(size) for size in selected_entities.shape[: len(topology.case_shape)]
        )
    return OperatorTopology(
        selected_graph,
        selected_entities,
        case_shape=case_shape,
        kind=topology.kind,
        site=topology.site,
        entity=topology.entity,
    )


def _tile_tree(tree: PyTree[Any] | None, repetitions: int, /) -> PyTree[Any] | None:
    if tree is None:
        return None
    return jax.tree_util.tree_map(
        lambda leaf: jnp.tile(
            jnp.asarray(leaf),
            (int(repetitions),) + (1,) * (jnp.asarray(leaf).ndim - 1),
        ),
        tree,
    )


def broadcast_operator_topology(
    topology: OperatorTopology,
    case_shape: Sequence[int],
    /,
) -> OperatorTopology:
    """Broadcast topology across new leading operator case axes."""

    target = tuple(int(size) for size in case_shape)
    if topology.case_shape == target:
        return topology
    source = topology.case_shape
    if source and (
        len(target) < len(source) or target[-len(source) :] != source
    ):
        raise ValueError(
            f"Topology case shape {source} cannot broadcast to {target}; "
            "existing geometry cases must be a trailing suffix."
        )
    prefix = target[: len(target) - len(source)] if source else target
    repetitions = prod(prefix) if prefix else 1
    graph = topology.graph
    if repetitions == 1:
        return OperatorTopology(
            graph,
            jnp.broadcast_to(topology.sample_entities, target + topology.sample_shape),
            case_shape=target,
            kind=topology.kind,
            site=topology.site,
            entity=topology.entity,
            _graph_fingerprint=_derived_graph_fingerprint(
                topology.graph_fingerprint, ("broadcast", target)
            ),
        )
    node_stride = jnp.sum(graph.n_node, dtype=jnp.int32)
    senders = None
    receivers = None
    if graph.senders is not None and graph.receivers is not None:
        offsets = jnp.repeat(
            jnp.arange(repetitions, dtype=jnp.int32) * node_stride,
            int(graph.senders.shape[0]),
        )
        senders = jnp.tile(graph.senders, repetitions) + offsets
        receivers = jnp.tile(graph.receivers, repetitions) + offsets
    repeated = GraphIR(
        nodes=_tile_tree(graph.nodes, repetitions),
        edges=_tile_tree(graph.edges, repetitions),
        senders=senders,
        receivers=receivers,
        globals=_tile_tree(graph.globals, repetitions),
        n_node=jnp.tile(graph.n_node, repetitions),
        n_edge=jnp.tile(graph.n_edge, repetitions),
        node_mask=(
            None if graph.node_mask is None else jnp.tile(graph.node_mask, repetitions)
        ),
        edge_mask=(
            None if graph.edge_mask is None else jnp.tile(graph.edge_mask, repetitions)
        ),
        graph_mask=(
            None if graph.graph_mask is None else jnp.tile(graph.graph_mask, repetitions)
        ),
        validate=False,
    )
    return OperatorTopology(
        repeated,
        jnp.broadcast_to(topology.sample_entities, target + topology.sample_shape),
        case_shape=target,
        kind=topology.kind,
        site=topology.site,
        entity=topology.entity,
        validate=False,
        _graph_fingerprint=_derived_graph_fingerprint(
            topology.graph_fingerprint, ("broadcast", target)
        ),
    )


def _scatter_samples(
    values: PyTree[Any],
    mapping: Array,
    valid: Array,
    entity_count: int,
    leading_shape: tuple[int, ...],
    /,
) -> PyTree[Any]:
    leading_ndim = len(leading_shape)
    flat_mapping = mapping.reshape((-1,))
    flat_valid = valid.reshape((-1,))
    safe_mapping = jnp.maximum(flat_mapping, 0)

    def scatter(leaf: Any) -> Array:
        array = jnp.asarray(leaf)
        if tuple(array.shape[:leading_ndim]) != leading_shape:
            raise ValueError(
                f"Sample value must start with shape {leading_shape}; got {array.shape}."
            )
        trailing = tuple(int(size) for size in array.shape[leading_ndim:])
        flattened = array.reshape((-1,) + trailing)
        mask = flat_valid.reshape((-1,) + (1,) * len(trailing))
        output = jnp.zeros((entity_count,) + trailing, dtype=array.dtype)
        updates = jnp.where(mask, flattened, jnp.zeros((), dtype=array.dtype))
        if jnp.issubdtype(array.dtype, jnp.bool_):
            return output.at[safe_mapping].max(updates)
        return output.at[safe_mapping].add(updates)

    return jax.tree_util.tree_map(scatter, values)


def scatter_operator_graph_entities(
    samples: FunctionSamples,
    sample_values: PyTree[Any],
    /,
    *,
    case_shape: Sequence[int] | None = None,
) -> PyTree[Any]:
    """Scatter topology-aligned sample values to canonical graph entities."""

    if samples.topology is None:
        raise ValueError("FunctionSamples has no native topology.")
    target_cases = (
        samples.topology.case_shape
        if case_shape is None
        else tuple(int(size) for size in case_shape)
    )
    topology = broadcast_operator_topology(samples.topology, target_cases)
    mapping = topology.absolute_sample_entities()
    sample_mask = samples.mask_array(case_shape=target_cases)
    valid = (mapping >= 0) & sample_mask
    return _scatter_samples(
        sample_values,
        mapping,
        valid,
        topology.entity_count,
        target_cases + samples.sample_shape,
    )


def materialize_operator_fields(
    batch: Any,
    fields: Sequence[Any],
    /,
) -> GraphIR:
    """Scatter every named source field onto one shared canonical topology."""

    from ._operator import OperatorBatch
    from ._operator_field import OperatorFieldSpec

    if not isinstance(batch, OperatorBatch):
        raise TypeError("materialize_operator_fields requires an OperatorBatch.")
    specs = tuple(fields)
    if not specs or any(not isinstance(field, OperatorFieldSpec) for field in specs):
        raise TypeError(
            "materialize_operator_fields requires non-empty OperatorFieldSpec fields."
        )
    bound: list[tuple[str, FunctionSamples]] = []
    for field in specs:
        if field.is_source:
            assert field.source_name is not None
            if field.source_name not in batch.inputs:
                if field.required:
                    raise KeyError(f"Missing required source field {field.source_name!r}.")
            else:
                bound.append((field.name, batch.input(field.source_name)))
        if field.is_target:
            assert field.query_name is not None
            if field.query_name not in batch.queries:
                if field.required:
                    raise KeyError(f"Missing required query {field.query_name!r}.")
            else:
                bound.append((f"query:{field.name}", batch.query(field.query_name)))
    if not bound or bound[0][1].topology is None:
        raise ValueError("Named operator fields require attached native topology.")
    first_topology = bound[0][1].topology
    assert first_topology is not None
    if first_topology.entity != "node":
        raise ValueError("Named cochain fields must be represented by graph nodes.")
    for name, samples in bound:
        topology = samples.topology
        if (
            topology is None
            or topology.kind != "cell_complex"
            or topology.entity != "node"
            or topology.graph_fingerprint != first_topology.graph_fingerprint
        ):
            raise ValueError(
                f"Field {name!r} does not share the canonical cell-complex topology."
            )
    topology = broadcast_operator_topology(first_topology, batch.case_shape)
    if not isinstance(topology.graph.nodes, Mapping):
        raise ValueError("Cell-complex topology requires named graph-node metadata.")
    nodes = dict(topology.graph.nodes)
    for field in specs:
        if not field.is_source:
            continue
        assert field.source_name is not None
        if field.source_name not in batch.inputs:
            continue
        samples = batch.input(field.source_name)
        if samples.values is None:
            raise ValueError(f"Source field {field.name!r} has no sampled values.")
        nodes[f"field:{field.name}"] = scatter_operator_graph_entities(
            samples,
            samples.values,
            case_shape=batch.case_shape,
        )
        nodes[f"field_mask:{field.name}"] = scatter_operator_graph_entities(
            samples,
            samples.mask_array(case_shape=batch.case_shape),
            case_shape=batch.case_shape,
        ).astype(bool)
    return topology.graph.replace(nodes=nodes, validate=True)


def operator_graph_from_samples(
    samples: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int] | None = None,
) -> GraphIR:
    """Materialize sample values and geometry on their native canonical graph."""

    if samples.topology is None:
        raise ValueError("FunctionSamples has no native topology.")
    target_cases = (
        samples.topology.case_shape
        if case_shape is None
        else tuple(int(size) for size in case_shape)
    )
    topology = broadcast_operator_topology(samples.topology, target_cases)
    mapping = topology.absolute_sample_entities()
    sample_mask = samples.mask_array(case_shape=target_cases)
    valid = (mapping >= 0) & sample_mask
    leading_shape = target_cases + samples.sample_shape
    entity_count = topology.entity_count
    coordinates = _scatter_samples(
        samples.coordinates_array(case_shape=target_cases),
        mapping,
        valid,
        entity_count,
        leading_shape,
    )
    quadrature = _scatter_samples(
        samples.quadrature(case_shape=target_cases),
        mapping,
        valid,
        entity_count,
        leading_shape,
    )
    payload = _entity_payload(topology.graph, topology.entity)
    entities: dict[str, Any]
    if isinstance(payload, Mapping):
        entities = dict(payload)
    else:
        entities = {}
        if payload is not None:
            entities["topology"] = payload
    entities.update(
        {
            "coordinates": coordinates,
            "quadrature_weights": quadrature,
            "sample_mask": _scatter_samples(
                sample_mask,
                mapping,
                valid,
                entity_count,
                leading_shape,
            ).astype(bool),
        }
    )
    if samples.values is not None:
        entities["features"] = _scatter_samples(
            samples.values,
            mapping,
            valid,
            entity_count,
            leading_shape,
        )
    if topology.entity == "node":
        return topology.graph.replace(nodes=entities, validate=True)
    if topology.entity == "edge":
        return topology.graph.replace(edges=entities, validate=True)
    return topology.graph.replace(globals=entities, validate=True)


def gather_operator_graph_entities(
    samples: FunctionSamples,
    entity_values: PyTree[Any],
    /,
    *,
    case_shape: Sequence[int] | None = None,
) -> PyTree[Any]:
    """Gather graph-entity output into the sample shape and mask padding."""

    if samples.topology is None:
        raise ValueError("FunctionSamples has no native topology.")
    target_cases = (
        samples.topology.case_shape
        if case_shape is None
        else tuple(int(size) for size in case_shape)
    )
    topology = broadcast_operator_topology(samples.topology, target_cases)
    mapping = topology.absolute_sample_entities()
    sample_mask = samples.mask_array(case_shape=target_cases)
    valid = (mapping >= 0) & sample_mask
    flat_mapping = mapping.reshape((-1,))
    safe_mapping = jnp.maximum(flat_mapping, 0)
    output_shape = target_cases + samples.sample_shape

    def gather(leaf: Any) -> Array:
        array = jnp.asarray(leaf)
        if int(array.shape[0]) != topology.entity_count:
            raise ValueError(
                f"Graph-entity values require leading size {topology.entity_count}; "
                f"got {array.shape[0]}."
            )
        trailing = tuple(int(size) for size in array.shape[1:])
        gathered = array[safe_mapping].reshape(output_shape + trailing)
        mask = valid.reshape(output_shape + (1,) * len(trailing))
        return jnp.where(mask, gathered, 0)

    return jax.tree_util.tree_map(gather, entity_values)


def operator_topology_fingerprint(topology: OperatorTopology, /) -> str:
    """Return a deterministic digest of topology structure and relation metadata."""

    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                topology.kind,
                topology.site,
                topology.entity,
                topology.case_shape,
                topology.graph_fingerprint,
            )
        ).encode("utf-8")
    )
    array = np.asarray(jax.device_get(topology.sample_entities))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "OperatorTopology",
    "OperatorTopologyEntity",
    "OperatorTopologyKind",
    "OperatorTopologySite",
    "broadcast_operator_topology",
    "gather_operator_graph_entities",
    "materialize_operator_fields",
    "operator_graph_from_samples",
    "operator_graph_fingerprint",
    "operator_topology_fingerprint",
    "pad_operator_topology",
    "slice_operator_topology",
    "scatter_operator_graph_entities",
    "stack_operator_topologies",
    "take_operator_topology",
]
