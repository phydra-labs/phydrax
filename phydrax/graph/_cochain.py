#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native metric cell-complex lowering to the graph execution carrier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cochain import CochainDiscretization
from ..discretization._cochain_hodge import DiagonalHodge
from ..discretization._topology import CellComplexTopology, EntitySet, OrientedIncidence
from ..exterior._complex import ComplexBoundary
from ..linalg._assembly import assemble_diagonal
from ..linalg._complexes import _coordinate_mass, HarmonicSubspace
from ..sparse import EdgeRelation
from ..typing import checked, parse
from ._cochain_execution import CochainGraphBinding
from ._ir import GraphIR


GraphEdgeSemantics: TypeAlias = Literal["reciprocal", "undirected_once"]
GraphNodeMeasure: TypeAlias = Literal["uniform", "degree"]


def _host_array(name: str, value: Any, /, *, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@final
class CochainComplexIR(StrictModule, NonTrainableState):
    """A lowering, never a second scientific complex or metric owner."""

    graph: GraphIR
    discretization: CochainDiscretization
    harmonic: tuple[HarmonicSubspace | None, ...]
    boundary: ComplexBoundary = eqx.field(static=True)
    cell_offsets: tuple[int, ...] = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: CochainDiscretization,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        harmonic: Sequence[HarmonicSubspace | None] | None = None,
    ) -> None:
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        counts = discretization.cell_counts
        bases = (None,) * len(counts) if harmonic is None else tuple(harmonic)
        if len(bases) != len(counts):
            raise ValueError("harmonic must supply one subspace per degree.")
        hilbert = (
            discretization.hilbert_complex(boundary=boundary_)
            if any(subspace is not None for subspace in bases)
            else None
        )
        for degree, subspace in enumerate(bases):
            if subspace is None:
                continue
            if not isinstance(subspace, HarmonicSubspace):
                raise TypeError("harmonic entries must be HarmonicSubspace or None.")
            if (
                hilbert is None
                or subspace.degree != degree
                or subspace.complex_id != hilbert.complex_id
            ):
                raise ValueError(
                    "Harmonic subspace belongs to a different degree, metric, or boundary."
                )
            if (
                subspace.basis.shape[0]
                != discretization.active_indices(degree, boundary=boundary_).size
            ):
                raise ValueError(
                    "Harmonic basis does not match the active cell coordinates."
                )
        offsets = tuple(np.cumsum((0,) + counts[:-1], dtype=np.int64).tolist())
        graph = _lower_graph(discretization, bases, boundary_, offsets)
        self.discretization = discretization
        self.harmonic = bases
        self.boundary = boundary_
        self.cell_offsets = offsets
        self.fingerprint = discretization.prepared_id
        self.graph = graph

    @property
    def cell_counts(self) -> tuple[int, ...]:
        return self.discretization.cell_counts

    @property
    def max_degree(self) -> int:
        return self.discretization.dimension

    @property
    def num_cells(self) -> int:
        return sum(self.cell_counts)

    def cell_entities(self, degree: int, /) -> Array:
        if degree < 0 or degree > self.max_degree:
            raise ValueError(f"Cochain degree must lie in [0, {self.max_degree}].")
        start = self.cell_offsets[degree]
        return jnp.arange(start, start + self.cell_counts[degree], dtype=jnp.int32)


def _lower_graph(
    discretization: CochainDiscretization,
    harmonic: tuple[HarmonicSubspace | None, ...],
    boundary: ComplexBoundary,
    offsets: tuple[int, ...],
    /,
) -> GraphIR:
    counts = discretization.cell_counts
    dimension = discretization.dimension
    total = sum(counts)
    nodes: dict[str, Array] = {
        "cell_dim": jnp.concatenate(
            tuple(
                jnp.full((count,), degree, dtype=jnp.int32)
                for degree, count in enumerate(counts)
            )
        ),
        "local_index": jnp.concatenate(
            tuple(jnp.arange(count, dtype=jnp.int32) for count in counts)
        ),
        # Exact diagnostic diagonal; full Riesz actions use the native binding.
        "hodge_star": jnp.concatenate(
            tuple(
                assemble_diagonal(
                    _coordinate_mass(discretization.space(degree).vector_space)
                )
                for degree in range(dimension + 1)
            )
        ),
        "primal_measure": jnp.concatenate(discretization.primal_measures),
        "dual_measure": jnp.concatenate(discretization.dual_measures),
        "boundary": jnp.concatenate(discretization.boundary_masks),
    }
    if discretization.coordinates[0] is not None:
        nodes["coordinates"] = jnp.concatenate(
            tuple(point for point in discretization.coordinates if point is not None),
            axis=0,
        )
    ranks = tuple(0 if subspace is None else subspace.dimension for subspace in harmonic)
    has_harmonic = any(subspace is not None for subspace in harmonic)
    if has_harmonic:
        basis_dtype = jnp.result_type(
            nodes["hodge_star"].dtype,
            *(subspace.basis.dtype for subspace in harmonic if subspace is not None),
        )
        packed = jnp.zeros(
            (total, dimension + 1, max(ranks, default=0)), dtype=basis_dtype
        )
        for degree, subspace in enumerate(harmonic):
            if subspace is not None:
                indices = (
                    discretization.active_indices(degree, boundary=boundary)
                    + offsets[degree]
                )
                packed = packed.at[indices, degree, : subspace.dimension].set(
                    subspace.basis
                )
        nodes["harmonic_basis"] = packed
    senders: list[Array] = []
    receivers: list[Array] = []
    signs: list[Array] = []
    directions: list[Array] = []
    degrees: list[Array] = []
    valid: list[Array] = []
    for incidence in discretization.topology.incidences:
        lower = (
            incidence.relation.source_indices.reshape((-1,))
            + offsets[incidence.degree - 1]
        )
        upper = (
            incidence.relation.target_indices.reshape((-1,)) + offsets[incidence.degree]
        )
        count = lower.size
        senders.extend((lower, upper))
        receivers.extend((upper, lower))
        signs.extend((incidence.signs.reshape((-1,)),) * 2)
        directions.extend(
            (jnp.ones((count,), dtype=jnp.int8), -jnp.ones((count,), dtype=jnp.int8))
        )
        degrees.extend((jnp.full((count,), incidence.degree, dtype=jnp.int32),) * 2)
        valid.extend((incidence.relation.valid.reshape((-1,)),) * 2)
    sender_array = (
        jnp.concatenate(senders) if senders else jnp.zeros((0,), dtype=jnp.int32)
    )
    receiver_array = (
        jnp.concatenate(receivers) if receivers else jnp.zeros((0,), dtype=jnp.int32)
    )
    edges = {
        "cochain_incidence": jnp.concatenate(valid)
        if valid
        else jnp.zeros((0,), dtype=jnp.bool_),
        "incidence_degree": jnp.concatenate(degrees)
        if degrees
        else jnp.zeros((0,), dtype=jnp.int32),
        "incidence_direction": jnp.concatenate(directions)
        if directions
        else jnp.zeros((0,), dtype=jnp.int8),
        "incidence_sign": jnp.concatenate(signs)
        if signs
        else jnp.zeros((0,), dtype=nodes["hodge_star"].dtype),
    }
    globals_: dict[str, Array] = {
        "max_degree": jnp.asarray([[dimension]], dtype=jnp.int32),
        "harmonic_rank": jnp.asarray([ranks], dtype=jnp.int32),
        "harmonic_boundary_policy": jnp.asarray(
            [[-1 if not has_harmonic else (0 if boundary == "absolute" else 1)]],
            dtype=jnp.int32,
        ),
        "primal_twist": jnp.asarray(
            [[0 if discretization.primal_twist == "untwisted" else 1]], dtype=jnp.int32
        ),
    }
    return GraphIR(
        nodes=nodes,
        edges=edges,
        senders=sender_array,
        receivers=receiver_array,
        globals=globals_,
        n_node=jnp.asarray([total], dtype=jnp.int32),
        node_mask=jnp.concatenate(
            tuple(
                entities.active_mask for entities in discretization.topology.entity_sets
            )
        ),
        n_edge=jnp.asarray([sender_array.size], dtype=jnp.int32),
        cochain_bindings=(CochainGraphBinding(discretization, harmonic, boundary),),
        validate=False,
    )


def _graph_edge_weight(
    graph: GraphIR,
    edge_weight_key: str | None,
    /,
) -> np.ndarray:
    count = 0 if graph.senders is None else graph.senders.shape[0]
    if edge_weight_key is None:
        if graph.edges is None:
            return np.ones((count,), dtype=np.float64)
        if isinstance(graph.edges, Mapping):
            raise ValueError(
                "edge_weight_key is required when GraphIR.edges is a mapping."
            )
        payload = graph.edges
    else:
        if not isinstance(edge_weight_key, str) or not edge_weight_key:
            raise ValueError("edge_weight_key must be a nonempty string or None.")
        if not isinstance(graph.edges, Mapping) or edge_weight_key not in graph.edges:
            raise ValueError(f"GraphIR.edges has no {edge_weight_key!r} payload.")
        payload = graph.edges[edge_weight_key]
    weight = _host_array("graph edge weights", payload, dtype=jnp.float64)
    if weight.shape != (count,):
        raise ValueError(f"Graph edge weights must have shape ({count},).")
    return weight


def _graph_node_measure(
    node_measure: GraphNodeMeasure | ArrayLike,
    node_count: int,
    senders: np.ndarray,
    receivers: np.ndarray,
    conductances: np.ndarray,
    /,
) -> np.ndarray:
    if isinstance(node_measure, str):
        measure_kind = parse(node_measure, GraphNodeMeasure, "node_measure")
        match measure_kind:
            case "uniform":
                measure = np.ones((node_count,), dtype=np.float64)
            case "degree":
                measure = np.zeros((node_count,), dtype=np.float64)
                np.add.at(measure, senders, conductances)
                np.add.at(measure, receivers, conductances)
            case _:
                assert_never(measure_kind)
    else:
        measure = _host_array("node_measure", node_measure, dtype=jnp.float64)
    if measure.shape != (node_count,):
        raise ValueError(f"node_measure must have shape ({node_count},).")
    if np.any(measure <= 0.0):
        raise ValueError("node_measure must be strictly positive.")
    return measure


def graph_to_cochain_complex(
    graph: GraphIR,
    /,
    *,
    edge_weight_key: str | None = None,
    node_measure: GraphNodeMeasure | ArrayLike = "uniform",
    edge_semantics: GraphEdgeSemantics = "reciprocal",
    reciprocal_rtol: float = 1e-8,
    reciprocal_atol: float = 1e-12,
) -> CochainDiscretization:
    """Convert one unpadded graph into a canonical metric one-complex."""
    if not isinstance(graph, GraphIR):
        raise TypeError("graph_to_cochain_complex requires a GraphIR.")
    graph.validate()
    if graph.num_graphs != 1:
        raise ValueError("Graph-to-cochain conversion requires exactly one graph.")
    if graph.node_mask is not None or graph.graph_mask is not None:
        raise ValueError("Padded node or graph masks are not supported.")
    edge_semantics = parse(edge_semantics, GraphEdgeSemantics, "edge_semantics")
    if float(reciprocal_rtol) < 0.0 or float(reciprocal_atol) < 0.0:
        raise ValueError("Reciprocal tolerances must be nonnegative.")
    node_count = graph.num_nodes
    if node_count <= 0:
        raise ValueError("Graph-to-cochain conversion requires at least one node.")
    if graph.senders is None or graph.receivers is None:
        senders = np.zeros((0,), dtype=np.int32)
        receivers = np.zeros((0,), dtype=np.int32)
    else:
        senders = np.asarray(graph.senders, dtype=np.int32)
        receivers = np.asarray(graph.receivers, dtype=np.int32)
    weights = _graph_edge_weight(graph, edge_weight_key)
    if graph.edge_mask is not None:
        valid = np.asarray(graph.edge_mask, dtype=np.bool_)
        if valid.shape != senders.shape:
            raise ValueError("edge_mask must align with stored graph edges.")
        senders = senders[valid]
        receivers = receivers[valid]
        weights = weights[valid]
    if np.any(senders == receivers):
        raise ValueError("Self-loops do not define one-dimensional cochain cells.")
    if np.any(weights <= 0.0):
        raise ValueError("Graph conductances must be finite and strictly positive.")

    oriented: dict[tuple[int, int], float] = {}
    for source, target, weight in zip(senders, receivers, weights, strict=True):
        pair = (int(source), int(target))
        oriented[pair] = oriented.get(pair, 0.0) + float(weight)
    canonical: dict[tuple[int, int], float] = {}
    if edge_semantics == "reciprocal":
        for (source, target), forward in oriented.items():
            reverse_pair = (target, source)
            if reverse_pair not in oriented:
                raise ValueError("Reciprocal edge semantics require every reverse edge.")
            reverse = oriented[reverse_pair]
            if not np.isclose(
                forward,
                reverse,
                rtol=float(reciprocal_rtol),
                atol=float(reciprocal_atol),
            ):
                raise ValueError("Reciprocal aggregate conductances are inconsistent.")
            pair = (min(source, target), max(source, target))
            canonical[pair] = 0.5 * (forward + reverse)
    else:
        for (source, target), weight in oriented.items():
            pair = (min(source, target), max(source, target))
            canonical[pair] = canonical.get(pair, 0.0) + weight

    ordered = tuple(sorted(canonical))
    edge_count = len(ordered)
    canonical_senders = np.asarray([pair[0] for pair in ordered], dtype=np.int32)
    canonical_receivers = np.asarray([pair[1] for pair in ordered], dtype=np.int32)
    conductances = np.asarray([canonical[pair] for pair in ordered], dtype=np.float64)
    measure = _graph_node_measure(
        node_measure,
        node_count,
        canonical_senders,
        canonical_receivers,
        conductances,
    )
    vertices = EntitySet("vertices", 0, np.arange(node_count, dtype=np.int32))
    if edge_count == 0:
        return CochainDiscretization(
            CellComplexTopology((vertices,), ()), (DiagonalHodge(measure),)
        )
    edges = EntitySet("edges", 1, np.arange(edge_count, dtype=np.int32))
    relation = EdgeRelation(
        np.stack((canonical_senders, canonical_receivers), axis=1).reshape((-1,)),
        np.repeat(np.arange(edge_count, dtype=np.int32), 2),
        source_size=node_count,
        target_size=edge_count,
    )
    incidence = OrientedIncidence(
        1,
        vertices,
        edges,
        relation,
        np.tile(np.asarray((-1.0, 1.0), dtype=np.float64), edge_count),
    )
    return CochainDiscretization(
        CellComplexTopology((vertices, edges), (incidence,)),
        (DiagonalHodge(measure), DiagonalHodge(conductances)),
    )


__all__ = [
    "CochainComplexIR",
    "GraphEdgeSemantics",
    "GraphNodeMeasure",
    "graph_to_cochain_complex",
]
