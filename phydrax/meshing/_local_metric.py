#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native two-dimensional anisotropic metric adaptation by local operations.

Planar triangle meshes are driven toward the unit mesh of a Riemannian metric
(every unprotected edge of metric length in ``[1/sqrt(2), sqrt(2)]``) by passes of
edge splits, edge collapses, edge flips, and vertex relocation. Each operation is
evaluated for every candidate at once; a deterministic greedy selection keeps a
maximal set of candidates whose cavities (cells touched) are pairwise disjoint,
and the selected operations are applied together. Every new or modified triangle
is certified POSITIVE by `orient2d` in the resolved host predicate mode; uncertain
predicates reject the operation. Classification (feature curves, corners, regions,
protected edges) is never crossed, so inherited organization stays unanimous.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._bvh import bvh_overlap_pairs_host, prepare_bvh
from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import orient2d, PredicateMode, PredicateSign
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._periodic_exact import _multiply, exact_periodic_element
from ..discretization._periodic_topology import _entity_vertex_rows
from ..linalg._exact import solve_exact
from ._lineage import EntityLineageKind
from ._metric import _tensor_properties, interpolate_mesh_metric, metric_edge_lengths
from ._periodic import periodic_entity_orbit_map, stage_periodic_topology_edit
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    source_family_blocks,
)


_LOWER = float(1.0 / np.sqrt(2.0))
_UPPER = float(np.sqrt(2.0))
_COLLAPSE_LIMIT = 2.0
# Minimum metric-quality gain of a flip or relocation; excludes rounding cycles.
_IMPROVEMENT = 1.0e-6
# Candidate-pair padding of the source point location; exact signs decide.
_LOCATION_TOLERANCE = 1.0e-9
# Scale lanes batch represented backtracking steps; continuation has no fixed limit.
_RELOCATION_SCALE_LANES = 8
_MINIMUM_CAPACITY = 16
_KEY_SHIFT = np.int64(1 << 32)

_OK, _UNCERTAIN, _INVALID, _INADMISSIBLE = 0, 1, 2, 3
_INTERIOR, _CURVE, _CORNER = 0, 1, 2
# History ranks; a cell's lineage kind is its strongest operation.
_REFINED, _RELOCATED, _SWAPPED, _COLLAPSED = 1, 2, 3, 4
_RANK_KINDS = np.asarray(
    (
        EntityLineageKind.PRESERVED,
        EntityLineageKind.REFINED_FROM,
        EntityLineageKind.RELOCATED,
        EntityLineageKind.SWAPPED_FROM,
        EntityLineageKind.COLLAPSED_INTO,
    ),
    dtype=np.int32,
)


@final
class LocalMetricEvidence(StrictModule, NonTrainableState):
    """Operations, rejections, and unit-mesh measurements of one local adaptation.

    Rejection counts tally candidate evaluations: an operation rejected in several
    sub-rounds counts once per evaluation. Length and quality measurements cover
    the unprotected edges of the returned mesh (all edges when every edge is
    protected); ``converged`` certifies the unit-mesh criterion on them.
    """

    passes: int = eqx.field(static=True)
    splits: int = eqx.field(static=True)
    collapses: int = eqx.field(static=True)
    flips: int = eqx.field(static=True)
    relocations: int = eqx.field(static=True)
    rejected_uncertain: int = eqx.field(static=True)
    rejected_invalid: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    stalled: bool = eqx.field(static=True)
    measured_edges: int = eqx.field(static=True)
    out_of_range_edges: int = eqx.field(static=True)
    minimum_metric_length: float = eqx.field(static=True)
    maximum_metric_length: float = eqx.field(static=True)
    unit_fraction: float = eqx.field(static=True)
    minimum_metric_quality: float = eqx.field(static=True)
    topology_operations: bool = eqx.field(static=True)
    relocation: bool = eqx.field(static=True)
    predicate_mode: PredicateMode = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        passes: int,
        splits: int,
        collapses: int,
        flips: int,
        relocations: int,
        rejected_uncertain: int,
        rejected_invalid: int,
        converged: bool,
        stalled: bool,
        measured_edges: int,
        out_of_range_edges: int,
        minimum_metric_length: float,
        maximum_metric_length: float,
        unit_fraction: float,
        minimum_metric_quality: float,
        topology_operations: bool,
        relocation: bool,
        predicate_mode: PredicateMode,
    ) -> None:
        counts = {
            "passes": passes,
            "splits": splits,
            "collapses": collapses,
            "flips": flips,
            "relocations": relocations,
            "rejected_uncertain": rejected_uncertain,
            "rejected_invalid": rejected_invalid,
            "measured_edges": measured_edges,
            "out_of_range_edges": out_of_range_edges,
        }
        flags = {
            "converged": converged,
            "stalled": stalled,
            "topology_operations": topology_operations,
            "relocation": relocation,
        }
        measures = {
            "minimum_metric_length": minimum_metric_length,
            "maximum_metric_length": maximum_metric_length,
            "unit_fraction": unit_fraction,
            "minimum_metric_quality": minimum_metric_quality,
        }
        for name, value in counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool.")
        for name, value in measures.items():
            if not isinstance(value, float) or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float.")
        if unit_fraction > 1.0:
            raise ValueError("unit_fraction cannot exceed one.")
        if not isinstance(predicate_mode, PredicateMode):
            raise TypeError("predicate_mode must be PredicateMode.")
        if predicate_mode is PredicateMode.FILTERED_DEVICE:
            raise ValueError("Local metric adaptation uses host predicate modes.")
        if converged and stalled:
            raise ValueError("A converged adaptation cannot be stalled.")
        self.passes = passes
        self.splits = splits
        self.collapses = collapses
        self.flips = flips
        self.relocations = relocations
        self.rejected_uncertain = rejected_uncertain
        self.rejected_invalid = rejected_invalid
        self.converged = converged
        self.stalled = stalled
        self.measured_edges = measured_edges
        self.out_of_range_edges = out_of_range_edges
        self.minimum_metric_length = minimum_metric_length
        self.maximum_metric_length = maximum_metric_length
        self.unit_fraction = unit_fraction
        self.minimum_metric_quality = minimum_metric_quality
        self.topology_operations = topology_operations
        self.relocation = relocation
        self.predicate_mode = predicate_mode
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "local-metric-evidence",
                **counts,
                **flags,
                **measures,
                "predicate_mode": str(predicate_mode),
            }
        )


class LocalMetricOutcome(NamedTuple):
    """Target topology edit, its vertex-aligned metric, and the evidence."""

    edit: CellTopologyEdit
    metric: np.ndarray
    evidence: LocalMetricEvidence


type ExactSourceRow = tuple[tuple[int, Fraction], ...]


class _State(NamedTuple):
    """Working mesh; vertex rows are never reused, removed rows stay dead."""

    points: np.ndarray
    metric: np.ndarray
    vertex_ids: np.ndarray
    source_stencil_sources: np.ndarray
    source_stencil_weights: np.ndarray
    source_stencil_valid: np.ndarray
    source_stencil_exact: tuple[ExactSourceRow, ...]
    source_rows: np.ndarray
    fixed: np.ndarray
    moved: np.ndarray
    alive: np.ndarray
    collapse_to: np.ndarray
    cells: np.ndarray
    cell_region: np.ndarray
    cell_block: np.ndarray
    cell_ids: np.ndarray
    cell_rank: np.ndarray
    ancestry_cells: np.ndarray
    ancestry_sources: np.ndarray
    feature_keys: np.ndarray
    feature_class: np.ndarray
    feature_protected: np.ndarray
    lineage_keys: np.ndarray
    lineage_sources: np.ndarray
    lineage_ranks: np.ndarray
    next_vertex_id: int


class _Source(NamedTuple):
    points: np.ndarray
    vertex_ids: np.ndarray
    cells: np.ndarray
    cell_tree: Any
    cell_ids: np.ndarray
    edge_keys: np.ndarray
    block_count: int
    families: tuple[tuple[str, str], ...]


class _Counts(NamedTuple):
    splits: int = 0
    collapses: int = 0
    flips: int = 0
    relocations: int = 0
    rejected_uncertain: int = 0
    rejected_invalid: int = 0


class _Topology(NamedTuple):
    edges: np.ndarray
    codes: np.ndarray
    edge_cells: np.ndarray
    edge_apex: np.ndarray
    cell_edges: np.ndarray
    edge_class: np.ndarray
    edge_protected: np.ndarray
    lengths: np.ndarray
    quality: np.ndarray
    kind: np.ndarray
    curve_class: np.ndarray
    curve_neighbors: np.ndarray
    straight: np.ndarray
    uniform_region: np.ndarray
    ball_offsets: np.ndarray
    ball_cells: np.ndarray
    neighbor_offsets: np.ndarray
    neighbor_values: np.ndarray


class _PeriodicContext(NamedTuple):
    edge_orbits: np.ndarray
    edge_quotient_ids: np.ndarray
    edge_canonical_vertices: np.ndarray
    edge_lengths: np.ndarray
    vertex_orbits: np.ndarray
    vertex_quotient_ids: np.ndarray
    vertex_representatives: np.ndarray
    vertex_exact_isometries: tuple[tuple[tuple[Fraction, ...], ...], ...]
    vertex_stabilizers: tuple[tuple[tuple[tuple[Fraction, ...], ...], ...], ...]
    cell_orbits: np.ndarray
    cell_quotient_ids: np.ndarray
    cell_quality: np.ndarray
    target_mesh_id: str
    vertex_allocator_next_id: int


def _exact_periodic_point(
    isometry: tuple[tuple[Fraction, ...], ...], point: np.ndarray, /
) -> np.ndarray:
    homogeneous = tuple(Fraction.from_float(float(value)) for value in point) + (
        Fraction(1),
    )
    return np.asarray(
        [
            float(
                sum(
                    (
                        isometry[row][column] * homogeneous[column]
                        for column in range(len(homogeneous))
                    ),
                    Fraction(0),
                )
            )
            for row in range(len(point))
        ],
        dtype=np.float64,
    )


def _exact_periodic_metric(
    isometry: tuple[tuple[Fraction, ...], ...], metric: np.ndarray, /
) -> np.ndarray:
    dimension = metric.shape[0]
    return np.asarray(
        [
            [
                float(
                    sum(
                        (
                            isometry[row][left]
                            * Fraction.from_float(float(metric[left, right]))
                            * isometry[column][right]
                            for left in range(dimension)
                            for right in range(dimension)
                        ),
                        Fraction(0),
                    )
                )
                for column in range(dimension)
            ]
            for row in range(dimension)
        ],
        dtype=np.float64,
    )


# ------------------------------------------------------------------ primitives


def _codes(keys: np.ndarray, /) -> np.ndarray:
    """Scalar codes of ascending vertex-row pairs."""
    return keys[:, 0] * _KEY_SHIFT + keys[:, 1]


def _sorted_pairs(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    return np.sort(np.stack((first, second), axis=1), axis=1)


def _lookup(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Position of each query in the ascending unique ``table``, or -1."""
    if table.size == 0:
        return np.full(queries.shape, -1, dtype=np.int64)
    position = np.minimum(np.searchsorted(table, queries), table.size - 1)
    return np.where(table[position] == queries, position, -1)


def _csr(owners: np.ndarray, values: np.ndarray, size: int, /) -> Any:
    order = np.argsort(owners, kind="stable")
    offsets = np.zeros((size + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(owners, minlength=size))
    return offsets, values[order]


def _expand(offsets: np.ndarray, rows: np.ndarray, /) -> Any:
    """``(owner, position)`` pairs enumerating CSR rows ``rows`` in order."""
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    owner = np.repeat(np.arange(rows.size, dtype=np.int64), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    return owner, np.arange(owner.size, dtype=np.int64) - first + starts[owner]


def _distinct(*columns: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of each distinct row, lexicographic order."""
    order = np.lexsort(columns[::-1])
    if order.size == 0:
        return order
    sorted_columns = np.stack([column[order] for column in columns], axis=1)
    first = np.ones((order.size,), dtype=np.bool_)
    first[1:] = np.any(sorted_columns[1:] != sorted_columns[:-1], axis=1)
    return order[first]


def _ranks(*keys: np.ndarray) -> np.ndarray:
    """Unique priority ranks from lexicographic keys (first key most significant)."""
    order = np.lexsort(keys[::-1])
    rank = np.empty((order.size,), dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank


def _capacity(count: int, /) -> int:
    return max(_MINIMUM_CAPACITY, 1 << (max(count, 1) - 1).bit_length())


def _metric_lengths(
    metric: np.ndarray, points: np.ndarray, edges: np.ndarray, /
) -> np.ndarray:
    """Riemannian edge lengths through the compiled owner, padded to 2^k shapes."""
    if edges.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    vertex_capacity = _capacity(points.shape[0])
    edge_capacity = _capacity(edges.shape[0])
    values = np.broadcast_to(np.eye(2), (vertex_capacity, 2, 2)).copy()
    values[: points.shape[0]] = metric
    coordinates = np.zeros((vertex_capacity, 2), dtype=np.float64)
    coordinates[: points.shape[0]] = points
    pairs = np.zeros((edge_capacity, 2), dtype=np.int32)
    pairs[: edges.shape[0]] = edges
    lengths = metric_edge_lengths(values, coordinates, pairs)
    return np.asarray(lengths, dtype=np.float64)[: edges.shape[0]]


def _interpolate_metrics(values: np.ndarray, weights: np.ndarray, /) -> np.ndarray:
    """Log-Euclidean means of ``(k, s, 2, 2)`` samples, padded to 2^k rows."""
    count = values.shape[0]
    if count == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)
    capacity = _capacity(count)
    tensors = np.broadcast_to(np.eye(2), (capacity, *values.shape[1:])).copy()
    tensors[:count] = values
    coefficients = np.zeros((capacity, values.shape[1]), dtype=np.float64)
    coefficients[:, 0] = 1.0
    coefficients[:count] = weights
    result = interpolate_mesh_metric(tensors, coefficients)
    return np.asarray(result, dtype=np.float64)[:count]


def _quality(lengths: np.ndarray, /) -> np.ndarray:
    """Metric shape quality ``4 sqrt(3) area / sum(l^2)`` from edge lengths (Heron)."""
    a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    product = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c)
    return np.sqrt(3.0) * np.sqrt(np.maximum(product, 0.0)) / (a * a + b * b + c * c)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, mode: Any, /) -> np.ndarray:
    """``_OK`` for certified POSITIVE, ``_UNCERTAIN``, else ``_INVALID``."""
    if a.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    result = orient2d(a, b, c, mode=mode)
    signs = np.asarray(result.signs)
    certain = np.asarray(result.certain)
    return np.where(
        ~certain, _UNCERTAIN, np.where(signs == PredicateSign.POSITIVE, _OK, _INVALID)
    )


def _collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray, mode: Any, /) -> np.ndarray:
    """``_OK`` for certified ZERO orientation, ``_UNCERTAIN``, else ``_INVALID``."""
    if a.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    result = orient2d(a, b, c, mode=mode)
    signs = np.asarray(result.signs)
    certain = np.asarray(result.certain)
    return np.where(
        ~certain, _UNCERTAIN, np.where(signs == PredicateSign.ZERO, _OK, _INVALID)
    )


def _barycentric(triangles: np.ndarray, points: np.ndarray, /) -> np.ndarray:
    """Floating barycentric weights of ``points`` in ``(k, 3, 2)`` triangles."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    def cross(origin: Any, first: Any, second: Any) -> Any:
        u = first - origin
        v = second - origin
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    total = cross(a, b, c)
    weights = np.stack(
        (cross(points, b, c), cross(points, c, a), cross(points, a, b)), axis=1
    )
    return weights / total[:, None]


def _normalized(weights: np.ndarray, /) -> np.ndarray:
    clipped = np.maximum(weights, 0.0)
    return clipped / np.sum(clipped, axis=1, keepdims=True)


def _group_maximum(
    values: np.ndarray, owners: np.ndarray, size: int, fill: Any, /
) -> Any:
    result = np.full((size,), fill, dtype=values.dtype)
    np.maximum.at(result, owners, values)
    return result


def _group_minimum(
    values: np.ndarray, owners: np.ndarray, size: int, fill: Any, /
) -> Any:
    result = np.full((size,), fill, dtype=values.dtype)
    np.minimum.at(result, owners, values)
    return result


def _disjoint_selection(
    rank: np.ndarray, pair_ops: np.ndarray, pair_items: np.ndarray, item_count: int, /
) -> np.ndarray:
    """Greedy maximal set of operations with pairwise disjoint item sets.

    Each round selects every undecided operation whose rank is the minimum over
    all of its items, then discards undecided operations touching selected items.
    Unique ranks make simultaneous winners disjoint; the result equals the
    sequential greedy selection in rank order.
    """
    count = rank.size
    undecided = np.ones((count,), dtype=np.bool_)
    selected = np.zeros((count,), dtype=np.bool_)
    while np.any(undecided[pair_ops]):
        live = undecided[pair_ops]
        best = np.full((item_count,), count, dtype=np.int64)
        np.minimum.at(best, pair_items[live], rank[pair_ops[live]])
        losing = np.zeros((count,), dtype=np.bool_)
        losing[pair_ops[live & (best[pair_items] != rank[pair_ops])]] = True
        winners = undecided & ~losing
        if not np.any(winners):
            raise RuntimeError("Disjoint candidate selection made no progress.")
        selected |= winners
        taken = np.zeros((item_count,), dtype=np.bool_)
        taken[pair_items[winners[pair_ops]]] = True
        blocked = np.zeros((count,), dtype=np.bool_)
        blocked[pair_ops[taken[pair_items]]] = True
        undecided &= ~blocked
    return selected


def _complete_periodic_operations(
    candidate_rows: np.ndarray,
    all_orbit_rows: np.ndarray,
    candidate_valid: np.ndarray,
    orbit_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_orbits = all_orbit_rows[candidate_rows]
    member_count = np.bincount(all_orbit_rows, minlength=orbit_count)
    candidate_count = np.bincount(candidate_orbits, minlength=orbit_count)
    valid_count = np.bincount(candidate_orbits[candidate_valid], minlength=orbit_count)
    eligible = np.flatnonzero(
        (member_count > 0)
        & (candidate_count == member_count)
        & (valid_count == member_count)
    )
    operation_of_orbit = np.full((orbit_count,), -1, dtype=np.int64)
    operation_of_orbit[eligible] = np.arange(eligible.size, dtype=np.int64)
    return eligible, operation_of_orbit[candidate_orbits]


def _tally(counts: _Counts, statuses: np.ndarray, /) -> _Counts:
    return counts._replace(
        rejected_uncertain=counts.rejected_uncertain
        + int(np.count_nonzero(statuses == _UNCERTAIN)),
        rejected_invalid=counts.rejected_invalid
        + int(np.count_nonzero(statuses >= _INVALID)),
    )


# ------------------------------------------------------------------ validation


def _integer_array(values: Any, rows: int, name: str, /) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array.")
    if array.shape != (rows,) or np.any(array < 0):
        raise ValueError(f"{name} must be non-negative with shape ({rows},).")
    return array.astype(np.int64)


def _boolean_array(values: Any, rows: int, name: str, /) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must be a boolean array.")
    if array.shape != (rows,):
        raise ValueError(f"{name} must have shape ({rows},).")
    return array


def _validate_mesh(mesh: CellMesh, metric: Any, /) -> np.ndarray:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if mesh.ambient_dimension != 2 or mesh.topological_dimension != 2:
        raise ValueError("Local metric adaptation requires planar triangle meshes.")
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("Local metric adaptation requires triangle blocks only.")
    values = np.asarray(metric, dtype=np.float64)
    if values.shape != (mesh.coordinates.shape[0], 2, 2) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("metric must be finite with shape (vertex_count, 2, 2).")
    properties = _tensor_properties(values)
    if not np.all(properties.hermitian) or not np.all(properties.positive_definite):
        raise ValueError("Metric tensors must be symmetric positive definite.")
    periodic = mesh.periodic_topology
    if periodic is not None:
        orbit_map = periodic_entity_orbit_map(mesh, 0)
        lifted_ids = np.asarray(orbit_map.lifted_entity_ids, dtype=np.int64)
        quotient_rows = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
        representative_rows = np.asarray(
            orbit_map.representative_lifted_rows, dtype=np.int64
        )
        windings = np.asarray(orbit_map.winding_shifts, dtype=np.int64)
        metric_row_by_id = {
            int(identifier): row
            for row, identifier in enumerate(mesh.vertex_global_ids.tolist())
        }
        epsilon = np.finfo(np.float64).eps
        for row, identifier in enumerate(lifted_ids.tolist()):
            quotient_row = int(quotient_rows[row])
            representative_id = int(lifted_ids[representative_rows[quotient_row]])
            representative = values[metric_row_by_id[representative_id]]
            metric_value = values[metric_row_by_id[int(identifier)]]
            transform = np.asarray(
                exact_periodic_element(periodic.cell, windings[row]),
                dtype=np.float64,
            )
            linear = transform[:2, :2]
            try:
                inverse = np.linalg.inv(linear)
            except np.linalg.LinAlgError as error:
                raise ValueError("Periodic metric isometry is singular.") from error
            expected = inverse.T @ representative @ inverse
            scale = max(
                float(np.linalg.norm(expected, ord=np.inf)),
                float(np.linalg.norm(metric_value, ord=np.inf)),
                np.finfo(np.float64).tiny,
            )
            if float(np.linalg.norm(metric_value - expected, ord=np.inf)) > (
                256.0 * epsilon * scale
            ):
                raise ValueError(
                    "SPD metric values must transform equivariantly across periodic vertex orbits."
                )

    return values


def _validate_controls(
    predicate_mode: Any, topology_operations: Any, relocation: Any, /
) -> None:
    if not isinstance(predicate_mode, PredicateMode):
        raise TypeError("predicate_mode must be PredicateMode.")
    match predicate_mode:
        case PredicateMode.EXACT | PredicateMode.FILTERED:
            pass
        case PredicateMode.FILTERED_DEVICE:
            raise ValueError("Local metric adaptation requires a host predicate mode.")
        case _:
            raise ValueError(f"Unsupported predicate mode {predicate_mode!r}.")
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("topology_operations and relocation must be bool.")


def _validate_classification(state: _State, topology: _Topology, /) -> None:
    boundary = topology.edge_cells[:, 1] < 0
    second = np.maximum(topology.edge_cells[:, 1], 0)
    interface = ~boundary & (
        state.cell_region[topology.edge_cells[:, 0]] != state.cell_region[second]
    )
    if np.any((boundary | interface) & (topology.edge_class == 0)):
        raise ValueError(
            "Boundary and region-interface edges must carry a positive edge class."
        )


# ------------------------------------------------------------------ preparation


def _initial_state(
    mesh: CellMesh,
    metric: np.ndarray,
    cell_classes: np.ndarray,
    edge_classes: np.ndarray,
    protected_edges: np.ndarray,
    fixed_vertices: np.ndarray,
    mode: PredicateMode,
    /,
) -> tuple[_State, _Source]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    blocks = np.concatenate(
        [
            np.full((block.vertices.shape[0],), index, dtype=np.int64)
            for index, block in enumerate(mesh.blocks)
        ]
    )
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    classes = cell_classes[key_rows(entity_keys(mesh, 2), cell_ids[:, None])]
    _, region = np.unique(
        np.stack((blocks, classes), axis=1), axis=0, return_inverse=True
    )
    if np.any(
        _orientation(points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]], mode)
        != _OK
    ):
        raise ValueError("Source triangles must be certified positively oriented.")
    # ty: ignore[unresolved-attribute]
    edges = np.sort(np.asarray(mesh.connectivity.edges, dtype=np.int64), axis=1)
    features = np.flatnonzero((edge_classes > 0) | protected_edges)
    count = points.shape[0]
    cell_count = cells.shape[0]
    source_stencil_sources = np.full((count, 3), -1, dtype=np.int64)
    source_stencil_sources[:, 0] = vertex_ids
    source_stencil_weights = np.zeros((count, 3), dtype=np.float64)
    source_stencil_weights[:, 0] = 1.0
    source_stencil_valid = np.zeros((count, 3), dtype=np.bool_)
    source_stencil_valid[:, 0] = True
    state = _State(
        points=points,
        metric=metric,
        vertex_ids=vertex_ids,
        source_stencil_sources=source_stencil_sources,
        source_stencil_weights=source_stencil_weights,
        source_stencil_valid=source_stencil_valid,
        source_stencil_exact=tuple(
            ((int(identifier), Fraction(1)),) for identifier in vertex_ids
        ),
        source_rows=np.arange(count, dtype=np.int64),
        fixed=fixed_vertices.copy(),
        moved=np.zeros((count,), dtype=np.bool_),
        alive=np.ones((count,), dtype=np.bool_),
        collapse_to=np.full((count,), -1, dtype=np.int64),
        cells=cells,
        cell_region=region.reshape((-1,)).astype(np.int64),
        cell_block=blocks,
        cell_ids=cell_ids,
        cell_rank=np.zeros((cell_count,), dtype=np.int64),
        ancestry_cells=np.arange(cell_count, dtype=np.int64),
        ancestry_sources=np.arange(cell_count, dtype=np.int64),
        feature_keys=edges[features],
        feature_class=edge_classes[features],
        feature_protected=protected_edges[features],
        lineage_keys=edges,
        lineage_sources=np.arange(edges.shape[0], dtype=np.int64),
        lineage_ranks=np.zeros((edges.shape[0],), dtype=np.int64),
        next_vertex_id=int(np.max(vertex_ids)) + 1,
    )
    triangles = points[cells]
    source = _Source(
        points=points,
        vertex_ids=vertex_ids,
        cells=cells,
        cell_ids=cell_ids,
        cell_tree=prepare_bvh(
            np.min(triangles, axis=1),
            np.max(triangles, axis=1),
            dtype=jnp.float64,
        ),
        edge_keys=entity_keys(mesh, 1),
        block_count=len(mesh.blocks),
        families=tuple((block.name, block.cell_kind) for block in mesh.blocks),
    )
    return state, source


# ------------------------------------------------------------------ topology


def _edge_table(cells: np.ndarray, /) -> Any:
    count = cells.shape[0]
    first = cells.reshape((-1,))
    second = cells[:, (1, 2, 0)].reshape((-1,))
    apex = cells[:, (2, 0, 1)].reshape((-1,))
    codes, inverse, counts = np.unique(
        _codes(_sorted_pairs(first, second)), return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("Local metric adaptation requires a manifold triangle mesh.")
    inverse = inverse.reshape((-1,))
    owner = np.repeat(np.arange(count, dtype=np.int64), 3)
    order = np.argsort(inverse, kind="stable")
    start = np.cumsum(counts) - counts
    edge_cells = np.full((codes.size, 2), -1, dtype=np.int64)
    edge_apex = np.full((codes.size, 2), -1, dtype=np.int64)
    edge_cells[:, 0] = owner[order[start]]
    edge_apex[:, 0] = apex[order[start]]
    two = np.flatnonzero(counts == 2)
    edge_cells[two, 1] = owner[order[start[two] + 1]]
    edge_apex[two, 1] = apex[order[start[two] + 1]]
    edges = np.stack((codes // _KEY_SHIFT, codes % _KEY_SHIFT), axis=1)
    return edges, codes, edge_cells, edge_apex, inverse.reshape((count, 3))


def _vertex_classes(
    state: _State,
    edges: np.ndarray,
    edge_class: np.ndarray,
    edge_protected: np.ndarray,
    mode: PredicateMode,
    /,
) -> Any:
    count = state.points.shape[0]
    classified = edges[edge_class > 0]
    ends = classified.reshape((-1,))
    others = classified[:, ::-1].reshape((-1,))
    labels = np.repeat(edge_class[edge_class > 0], 2)
    incident = np.bincount(ends, minlength=count)
    low = _group_minimum(labels, ends, count, np.iinfo(np.int64).max)
    high = _group_maximum(labels, ends, count, -1)
    guarded = state.fixed.copy()
    guarded[edges[edge_protected].reshape((-1,))] = True
    curve = (incident == 2) & (low == high) & ~guarded
    kind = np.where(curve, _CURVE, np.where(guarded | (incident > 0), _CORNER, _INTERIOR))
    order = np.argsort(ends, kind="stable")
    first = np.searchsorted(ends[order], np.arange(count))
    rows = np.flatnonzero(curve)
    neighbors = np.full((count, 2), -1, dtype=np.int64)
    neighbors[rows, 0] = others[order][first[rows]]
    neighbors[rows, 1] = others[order][first[rows] + 1]
    straight = np.zeros((count,), dtype=np.bool_)
    points = state.points
    straight[rows] = (
        _collinear(
            points[neighbors[rows, 0]], points[rows], points[neighbors[rows, 1]], mode
        )
        == _OK
    )
    return kind, np.where(curve, low, 0), neighbors, straight


def _topology(state: _State, mode: PredicateMode, /) -> _Topology:
    edges, codes, edge_cells, edge_apex, cell_edges = _edge_table(state.cells)
    feature_rows = _lookup(codes, _codes(state.feature_keys))
    if np.any(feature_rows < 0):
        raise RuntimeError("Local metric adaptation lost a classified edge.")
    edge_class = np.zeros((codes.size,), dtype=np.int64)
    edge_class[feature_rows] = state.feature_class
    edge_protected = np.zeros((codes.size,), dtype=np.bool_)
    edge_protected[feature_rows] = state.feature_protected
    lengths = _metric_lengths(state.metric, state.points, edges)
    kind, curve_class, neighbors, straight = _vertex_classes(
        state, edges, edge_class, edge_protected, mode
    )
    count = state.points.shape[0]
    corners = state.cells.reshape((-1,))
    owners = np.repeat(np.arange(state.cells.shape[0], dtype=np.int64), 3)
    ball_offsets, ball_cells = _csr(corners, owners, count)
    regions = state.cell_region[owners]
    uniform = _group_minimum(regions, corners, count, np.iinfo(np.int64).max) == (
        _group_maximum(regions, corners, count, -1)
    )
    ends = np.concatenate((edges[:, 0], edges[:, 1]))
    neighbor_offsets, neighbor_values = _csr(
        ends, np.concatenate((edges[:, 1], edges[:, 0])), count
    )
    return _Topology(
        edges=edges,
        codes=codes,
        edge_cells=edge_cells,
        edge_apex=edge_apex,
        cell_edges=cell_edges,
        edge_class=edge_class,
        edge_protected=edge_protected,
        lengths=lengths,
        quality=_quality(lengths[cell_edges]),
        kind=kind,
        curve_class=curve_class,
        curve_neighbors=neighbors,
        straight=straight,
        uniform_region=uniform,
        ball_offsets=ball_offsets,
        ball_cells=ball_cells,
        neighbor_offsets=neighbor_offsets,
        neighbor_values=neighbor_values,
    )


def _measured(topology: _Topology, /) -> np.ndarray:
    unprotected = ~topology.edge_protected
    return topology.lengths[unprotected] if np.any(unprotected) else topology.lengths


def _unit(topology: _Topology, /) -> bool:
    lengths = _measured(topology)
    return bool(np.all((lengths >= _LOWER) & (lengths <= _UPPER)))


# ------------------------------------------------------------------ state updates


def _pad_stencil_columns(
    sources: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    width: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sources.shape != weights.shape or sources.shape != valid.shape:
        raise ValueError("Source stencil arrays must have matching shapes.")
    current_width = sources.shape[1]
    target_width = int(width)
    if target_width < current_width:
        raise ValueError("Source stencil padding cannot discard columns.")
    if target_width == current_width:
        return sources, weights, valid
    padding = target_width - current_width
    return (
        np.pad(sources, ((0, 0), (0, padding)), constant_values=-1),
        np.pad(weights, ((0, 0), (0, padding)), constant_values=0.0),
        np.pad(valid, ((0, 0), (0, padding)), constant_values=False),
    )


def _append_vertices(
    state: _State,
    points: np.ndarray,
    metric: np.ndarray,
    source_stencil_sources: np.ndarray,
    source_stencil_weights: np.ndarray,
    source_stencil_valid: np.ndarray,
    source_stencil_exact: tuple[ExactSourceRow, ...],
    /,
) -> _State:
    count = points.shape[0]
    width = max(
        state.source_stencil_weights.shape[1],
        source_stencil_weights.shape[1],
    )
    state_sources, state_weights, state_valid = _pad_stencil_columns(
        state.source_stencil_sources,
        state.source_stencil_weights,
        state.source_stencil_valid,
        width,
    )
    new_sources, new_weights, new_valid = _pad_stencil_columns(
        source_stencil_sources,
        source_stencil_weights,
        source_stencil_valid,
        width,
    )
    return state._replace(
        points=np.concatenate((state.points, points)),
        metric=np.concatenate((state.metric, metric)),
        vertex_ids=np.concatenate(
            (
                state.vertex_ids,
                state.next_vertex_id + np.arange(count, dtype=np.int64),
            )
        ),
        source_stencil_sources=np.concatenate((state_sources, new_sources)),
        source_stencil_weights=np.concatenate((state_weights, new_weights)),
        source_stencil_valid=np.concatenate((state_valid, new_valid)),
        source_stencil_exact=state.source_stencil_exact + source_stencil_exact,
        source_rows=np.concatenate(
            (state.source_rows, np.full((count,), -1, dtype=np.int64))
        ),
        fixed=np.concatenate((state.fixed, np.zeros((count,), dtype=np.bool_))),
        moved=np.concatenate((state.moved, np.zeros((count,), dtype=np.bool_))),
        alive=np.concatenate((state.alive, np.ones((count,), dtype=np.bool_))),
        collapse_to=np.concatenate(
            (state.collapse_to, np.full((count,), -1, dtype=np.int64))
        ),
        next_vertex_id=state.next_vertex_id + count,
    )


def _replace_cells(
    state: _State,
    removed_rows: np.ndarray,
    children: np.ndarray,
    child_of: np.ndarray,
    parent_of: np.ndarray,
    rank: int,
    /,
) -> _State:
    """Replace cells; each child inherits the region and ancestry of its parents.

    ``(child_of, parent_of)`` pairs name the cells whose region a child descends
    from; every child must list its primary parent (same region) first.
    """
    cell_count = state.cells.shape[0]
    removed = np.zeros((cell_count,), dtype=np.bool_)
    removed[removed_rows] = True
    keep = np.flatnonzero(~removed)
    renumber = np.full((cell_count,), -1, dtype=np.int64)
    renumber[keep] = np.arange(keep.size, dtype=np.int64)
    child_count = children.shape[0]
    _, first_listed = np.unique(child_of, return_index=True)
    primary = parent_of[first_listed]
    child_rank = np.full((child_count,), rank, dtype=np.int64)
    np.maximum.at(child_rank, child_of, state.cell_rank[parent_of])
    order = np.argsort(state.ancestry_cells, kind="stable")
    ancestry_cells = state.ancestry_cells[order]
    ancestry_sources = state.ancestry_sources[order]
    offsets = np.searchsorted(ancestry_cells, np.arange(cell_count + 1))
    pair, position = _expand(offsets, parent_of)
    kept = renumber[ancestry_cells] >= 0
    cells = np.concatenate((renumber[ancestry_cells[kept]], keep.size + child_of[pair]))
    sources = np.concatenate((ancestry_sources[kept], ancestry_sources[position]))
    distinct = _distinct(cells, sources)
    return state._replace(
        cells=np.concatenate((state.cells[keep], children)),
        cell_region=np.concatenate((state.cell_region[keep], state.cell_region[primary])),
        cell_block=np.concatenate((state.cell_block[keep], state.cell_block[primary])),
        cell_ids=np.concatenate(
            (state.cell_ids[keep], np.full((child_count,), -1, dtype=np.int64))
        ),
        cell_rank=np.concatenate((state.cell_rank[keep], child_rank)),
        ancestry_cells=cells[distinct],
        ancestry_sources=sources[distinct],
    )


def _split_records(
    keys: np.ndarray,
    split_codes: np.ndarray,
    new_rows: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace records on split edges by both halves.

    Returns the new keys, the record index each new key copies, and a mask of the
    halves (records created by the split).
    """
    order = np.argsort(split_codes)
    found = _lookup(split_codes[order], _codes(keys))
    hit = np.flatnonzero(found >= 0)
    rest = np.flatnonzero(found < 0)
    middle = new_rows[order][found[hit]]
    left = _sorted_pairs(keys[hit, 0], middle)
    right = _sorted_pairs(keys[hit, 1], middle)
    source = np.concatenate((rest, hit, hit))
    halves = np.concatenate(
        (np.zeros(rest.shape, dtype=np.bool_), np.ones((2 * hit.size,), dtype=np.bool_))
    )
    return np.concatenate((keys[rest], left, right)), source, halves


def _raise_touching(state: _State, vertices: np.ndarray, rank: int, /) -> _State:
    """Raise the history rank of lineage records and cells touching ``vertices``."""
    mark = np.zeros((state.points.shape[0],), dtype=np.bool_)
    mark[vertices] = True
    records = np.any(mark[state.lineage_keys], axis=1)
    cells = np.any(mark[state.cells], axis=1)
    return state._replace(
        lineage_ranks=np.where(
            records, np.maximum(state.lineage_ranks, rank), state.lineage_ranks
        ),
        cell_rank=np.where(cells, np.maximum(state.cell_rank, rank), state.cell_rank),
    )


# ------------------------------------------------------------------ split


def _metric_midpoint(state: _State, first: np.ndarray, second: np.ndarray, /) -> Any:
    """Parameter of the metric-length midpoint under geometric size variation."""
    delta = state.points[second] - state.points[first]

    def length(rows: Any) -> Any:
        tensor = state.metric[rows]
        return np.sqrt(
            delta[:, 0] * delta[:, 0] * tensor[:, 0, 0]
            + 2.0 * delta[:, 0] * delta[:, 1] * tensor[:, 0, 1]
            + delta[:, 1] * delta[:, 1] * tensor[:, 1, 1]
        )

    ratio = length(second) / length(first)
    equal = ratio == 1.0
    delta = np.where(equal, 1.0, ratio - 1.0)
    parameter = np.log1p(0.5 * delta) / np.log1p(delta)
    return np.where(equal, 0.5, parameter)


def _cavity_sides(topology: _Topology, edges: np.ndarray, /) -> Any:
    """``(op, cell, local)`` per cell on candidate edges; local edge ``i`` is CCW."""
    sides = topology.edge_cells[edges]
    op, side = np.nonzero(sides >= 0)
    cell = sides[op, side]
    local = np.argmax(topology.cell_edges[cell] == edges[op][:, None], axis=1)
    return op, cell, local


def _split_round(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    mode: PredicateMode,
    periodic_context: _PeriodicContext | None,
    source: _Source,
    /,
) -> tuple[_State, _Counts, int]:
    candidates = np.flatnonzero((topology.lengths > _UPPER) & ~topology.edge_protected)
    if candidates.size == 0:
        return state, counts, 0
    first, second = topology.edges[candidates].T
    t = _metric_midpoint(state, first, second)
    periodic_midpoint_inconsistent: set[int] = set()
    if periodic_context is not None:
        complete_orbits, _ = _complete_periodic_operations(
            candidates,
            periodic_context.edge_orbits,
            np.ones((candidates.size,), dtype=np.bool_),
            periodic_context.edge_quotient_ids.size,
        )
        canonical_first = periodic_context.edge_canonical_vertices[candidates, 0]
        canonical_parameter = np.where(first == canonical_first, t, 1.0 - t)
        candidate_orbits = periodic_context.edge_orbits[candidates]
        for orbit in complete_orbits:
            member_positions = np.flatnonzero(candidate_orbits == orbit)
            parameters = canonical_parameter[member_positions]
            if not np.array_equal(
                parameters,
                np.full(parameters.shape, parameters[0], dtype=np.float64),
            ):
                periodic_midpoint_inconsistent.add(int(orbit))
                continue
            common_parameter = float(parameters[0])
            t[member_positions] = np.where(
                first[member_positions] == canonical_first[member_positions],
                common_parameter,
                1.0 - common_parameter,
            )
    middle = state.points[first] + t[:, None] * (
        state.points[second] - state.points[first]
    )
    op, cell, local = _cavity_sides(topology, candidates)
    rows = state.cells[cell]
    span = np.arange(cell.size)
    u = rows[span, local]
    v = rows[span, (local + 1) % 3]
    apex = rows[span, (local + 2) % 3]
    points = state.points
    checks = np.maximum(
        _orientation(points[u], middle[op], points[apex], mode),
        _orientation(middle[op], points[v], points[apex], mode),
    )
    status = _group_maximum(checks, op, candidates.size, _OK)
    counts = _tally(counts, status)
    valid = np.flatnonzero(status == _OK)
    if valid.size == 0:
        return state, counts, 0
    ids = state.vertex_ids
    if periodic_context is None:
        rank = _ranks(
            -topology.lengths[candidates[valid]],
            ids[first[valid]],
            ids[second[valid]],
        )
        local_of = np.full((candidates.size,), -1, dtype=np.int64)
        local_of[valid] = np.arange(valid.size, dtype=np.int64)
        inside = local_of[op] >= 0
        chosen = valid[
            _disjoint_selection(
                rank, local_of[op[inside]], cell[inside], state.cells.shape[0]
            )
        ]
        chosen = chosen[np.lexsort((ids[second[chosen]], ids[first[chosen]]))]
    else:
        candidate_valid = status == _OK
        for orbit in np.unique(periodic_context.edge_orbits[candidates]):
            members = np.flatnonzero(periodic_context.edge_orbits[candidates] == orbit)
            member_sides = cell[np.isin(op, members)]
            if np.unique(member_sides).size != member_sides.size:
                candidate_valid[members] = False
        eligible, operation_of_candidate = _complete_periodic_operations(
            candidates,
            periodic_context.edge_orbits,
            candidate_valid,
            periodic_context.edge_quotient_ids.size,
        )
        if eligible.size == 0:
            return state, counts, 0
        orbit_lengths = np.full(
            (periodic_context.edge_quotient_ids.size,), np.inf, dtype=np.float64
        )
        np.minimum.at(orbit_lengths, periodic_context.edge_orbits, topology.lengths)
        rank = _ranks(
            -orbit_lengths[eligible],
            periodic_context.edge_quotient_ids[eligible],
        )
        pair_operations = operation_of_candidate[op]
        inside = pair_operations >= 0
        selected_operations = _disjoint_selection(
            rank,
            pair_operations[inside],
            cell[inside],
            state.cells.shape[0],
        )
        selected = np.flatnonzero(selected_operations)
        selected_orbits = eligible[selected]
        for orbit in selected_orbits:
            if int(orbit) in periodic_midpoint_inconsistent:
                raise ValueError("Periodic metric midpoints differ across an edge orbit.")
        chosen = np.flatnonzero(np.isin(operation_of_candidate, selected))
        chosen_edges = candidates[chosen]
        chosen = chosen[
            np.lexsort(
                (
                    ids[second[chosen]],
                    ids[first[chosen]],
                    periodic_context.edge_quotient_ids[
                        periodic_context.edge_orbits[chosen_edges]
                    ],
                )
            )
        ]
    new_rows = np.full((candidates.size,), -1, dtype=np.int64)
    new_rows[chosen] = state.points.shape[0] + np.arange(chosen.size)
    weights = np.stack((1.0 - t[chosen], t[chosen]), axis=1)
    samples = np.stack((state.metric[first[chosen]], state.metric[second[chosen]]), 1)
    split_sources, split_weights, split_valid, split_supported, split_exact = (
        _located_stencil_bank(
            source,
            middle[chosen],
            mode,
            stencil_width=state.source_stencil_weights.shape[1],
        )
    )
    if not np.all(split_supported):
        raise ValueError(
            "Every split midpoint requires exact source interpolation support."
        )
    grown = _append_vertices(
        state,
        middle[chosen],
        _interpolate_metrics(samples, weights),
        split_sources,
        split_weights,
        split_valid,
        split_exact,
    )
    picked = np.flatnonzero(new_rows[op] >= 0)
    m = new_rows[op[picked]]
    children = np.concatenate(
        (
            np.stack((u[picked], m, apex[picked]), axis=1),
            np.stack((m, v[picked], apex[picked]), axis=1),
        )
    )
    parents = np.concatenate((cell[picked], cell[picked]))
    grown = _replace_cells(
        grown,
        cell[picked],
        children,
        np.arange(children.shape[0], dtype=np.int64),
        parents,
        _REFINED,
    )
    split_codes = _codes(topology.edges[candidates[chosen]])
    feature_keys, feature_source, _ = _split_records(
        grown.feature_keys, split_codes, new_rows[chosen]
    )
    lineage_keys, lineage_source, halves = _split_records(
        grown.lineage_keys, split_codes, new_rows[chosen]
    )
    ranks = grown.lineage_ranks[lineage_source]
    grown = grown._replace(
        feature_keys=feature_keys,
        feature_class=grown.feature_class[feature_source],
        feature_protected=grown.feature_protected[feature_source],
        lineage_keys=lineage_keys,
        lineage_sources=grown.lineage_sources[lineage_source],
        lineage_ranks=np.where(halves, np.maximum(ranks, _REFINED), ranks),
    )
    return grown, counts._replace(splits=counts.splits + chosen.size), chosen.size


# ------------------------------------------------------------------ collapse


class _CollapseCandidates(NamedTuple):
    edge: np.ndarray
    removed: np.ndarray
    kept: np.ndarray
    status: np.ndarray
    longest: np.ndarray


def _collapse_admissible(
    topology: _Topology, edge: Any, removed: Any, kept: Any, /
) -> np.ndarray:
    """Classification: interior vertices along unclassified edges inside one region;
    straight curve vertices along an edge of their own curve class."""
    label = topology.edge_class[edge]
    interior = (
        (topology.kind[removed] == _INTERIOR)
        & (label == 0)
        & topology.uniform_region[removed]
    )
    curve = (
        (topology.kind[removed] == _CURVE)
        & (label > 0)
        & (label == topology.curve_class[removed])
        & topology.straight[removed]
    )
    return interior | curve


def _collapse_link(
    topology: _Topology, edge: Any, removed: Any, kept: Any, /
) -> np.ndarray:
    """Link condition: common neighbors are exactly the apexes of the edge."""
    owner, position = _expand(topology.neighbor_offsets, removed)
    other = topology.neighbor_values[position]
    common = _lookup(topology.codes, _codes(_sorted_pairs(kept[owner], other))) >= 0
    common &= other != kept[owner]
    shared = np.bincount(owner[common], minlength=removed.size)
    return shared == 1 + (topology.edge_cells[edge, 1] >= 0)


def _collapse_geometry(
    state: _State, topology: _Topology, removed: Any, kept: Any, mode: PredicateMode, /
) -> Any:
    """Worst orientation of the modified ball cells and the longest new edge."""
    owner, position = _expand(topology.ball_offsets, removed)
    rows = state.cells[topology.ball_cells[position]]
    modified = ~np.any(rows == kept[owner][:, None], axis=1)
    moved = np.where(rows == removed[owner][:, None], kept[owner][:, None], rows)
    moved = moved[modified]
    points = state.points
    orientation = _orientation(
        points[moved[:, 0]], points[moved[:, 1]], points[moved[:, 2]], mode
    )
    status = _group_maximum(orientation, owner[modified], removed.size, _OK)
    neighbor, spot = _expand(topology.neighbor_offsets, removed)
    other = topology.neighbor_values[spot]
    fresh = (other != kept[neighbor]) & (
        _lookup(topology.codes, _codes(_sorted_pairs(kept[neighbor], other))) < 0
    )
    lengths = _metric_lengths(
        state.metric, points, np.stack((kept[neighbor][fresh], other[fresh]), axis=1)
    )
    longest = _group_maximum(lengths, neighbor[fresh], removed.size, 0.0)
    return status, longest


def _collapse_candidates(
    state: _State, topology: _Topology, mode: PredicateMode, /
) -> _CollapseCandidates:
    short = np.flatnonzero((topology.lengths < _LOWER) & ~topology.edge_protected)
    edge = np.concatenate((short, short))
    removed = np.concatenate((topology.edges[short, 0], topology.edges[short, 1]))
    kept = np.concatenate((topology.edges[short, 1], topology.edges[short, 0]))
    admissible = _collapse_admissible(topology, edge, removed, kept)
    link = _collapse_link(topology, edge, removed, kept)
    orientation, longest = _collapse_geometry(state, topology, removed, kept, mode)
    # A -1 source row marks a birth in this execution; the next execution resets
    # every current vertex. Either orientation may remove a birth, but only when
    # the resulting edges stay below the split threshold.
    collapse_limit = np.where(
        (state.source_rows[removed] < 0) | (state.source_rows[kept] < 0),
        _UPPER,
        _COLLAPSE_LIMIT,
    )
    geometric = np.maximum(
        orientation,
        np.where(link & (longest <= collapse_limit), _OK, _INVALID),
    )
    status = np.where(admissible, geometric, _INADMISSIBLE)
    return _CollapseCandidates(edge, removed, kept, status, longest)


def _collapse_cells(
    state: _State, topology: _Topology, removed: Any, kept: Any, /
) -> _State:
    """Remove the edge cells, move the other ball cells onto the kept vertex."""
    owner, position = _expand(topology.ball_offsets, removed)
    cell = topology.ball_cells[position]
    rows = state.cells[cell]
    dropped = np.any(rows == kept[owner][:, None], axis=1)
    slot = np.cumsum(dropped) - 1
    first = np.searchsorted(owner[dropped], np.arange(removed.size))
    lost = np.full((removed.size, 2), -1, dtype=np.int64)
    lost[owner[dropped], slot[dropped] - first[owner[dropped]]] = cell[dropped]
    survivor = np.flatnonzero(~dropped)
    children = np.where(
        rows[survivor] == removed[owner[survivor]][:, None],
        kept[owner[survivor]][:, None],
        rows[survivor],
    )
    absorbed = lost[owner[survivor]]
    same = (absorbed >= 0) & (
        state.cell_region[np.maximum(absorbed, 0)]
        == state.cell_region[cell[survivor]][:, None]
    )
    child, column = np.nonzero(same)
    child_of = np.concatenate((np.arange(survivor.size), child))
    parent_of = np.concatenate((cell[survivor], absorbed[child, column]))
    return _replace_cells(state, cell, children, child_of, parent_of, _COLLAPSED)


def _collapse_records(state: _State, removed: Any, kept: Any, /) -> _State:
    remap = np.arange(state.points.shape[0], dtype=np.int64)
    remap[removed] = kept
    features = np.sort(remap[state.feature_keys], axis=1)
    live = np.flatnonzero(features[:, 0] != features[:, 1])
    codes = _codes(features[live])
    distinct = live[_distinct(codes, -state.feature_class[live])]
    protected = _group_maximum(
        state.feature_protected[live].astype(np.int64),
        np.searchsorted(np.unique(codes), codes),
        distinct.size,
        0,
    )
    touched = np.any(np.isin(state.lineage_keys, removed), axis=1)
    ranks = np.where(
        touched, np.maximum(state.lineage_ranks, _COLLAPSED), state.lineage_ranks
    )
    lineage = np.sort(remap[state.lineage_keys], axis=1)
    alive = np.flatnonzero(lineage[:, 0] != lineage[:, 1])
    keep = alive[
        _distinct(_codes(lineage[alive]), state.lineage_sources[alive], -ranks[alive])
    ]
    return state._replace(
        feature_keys=features[distinct],
        feature_class=state.feature_class[distinct],
        feature_protected=protected.astype(np.bool_),
        lineage_keys=lineage[keep],
        lineage_sources=state.lineage_sources[keep],
        lineage_ranks=ranks[keep],
    )


def _collapse_round(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    mode: PredicateMode,
    periodic_context: _PeriodicContext | None,
    _source: _Source,
    /,
) -> tuple[_State, _Counts, int]:
    found = _collapse_candidates(state, topology, mode)
    if found.edge.size == 0:
        return state, counts, 0
    half = found.edge.size // 2
    best = np.minimum(found.status[:half], found.status[half:])
    counts = _tally(counts, np.where(best == _INADMISSIBLE, _INVALID, best))
    valid = np.flatnonzero(found.status == _OK)
    if valid.size == 0:
        return state, counts, 0

    if periodic_context is None:
        ids = state.vertex_ids
        removed = found.removed[valid]
        kept = found.kept[valid]
        rank = _ranks(
            topology.lengths[found.edge[valid]],
            found.longest[valid],
            ids[removed],
            ids[kept],
        )
        owner_a, position_a = _expand(topology.ball_offsets, removed)
        owner_b, position_b = _expand(topology.ball_offsets, kept)
        selected = _disjoint_selection(
            rank,
            np.concatenate((owner_a, owner_b)),
            np.concatenate(
                (topology.ball_cells[position_a], topology.ball_cells[position_b])
            ),
            state.cells.shape[0],
        )
        removed = removed[selected]
        kept = kept[selected]
    else:
        short_edges = found.edge[:half]
        canonical_removed = periodic_context.edge_canonical_vertices[short_edges, 0]
        first_candidates = np.where(
            found.removed[:half] == canonical_removed,
            np.arange(half),
            np.arange(half) + half,
        )
        second_candidates = np.where(
            found.removed[:half] == canonical_removed,
            np.arange(half) + half,
            np.arange(half),
        )
        candidate_orbits = periodic_context.edge_orbits[short_edges]
        canonical_candidates = first_candidates.copy()
        candidate_valid = np.zeros((half,), dtype=np.bool_)
        for orbit in np.unique(candidate_orbits):
            members = np.flatnonzero(candidate_orbits == orbit)
            first_valid = found.status[first_candidates[members]] == _OK
            second_valid = found.status[second_candidates[members]] == _OK
            if np.all(first_valid):
                candidate_valid[members] = True
            elif np.all(second_valid):
                canonical_candidates[members] = second_candidates[members]
                candidate_valid[members] = True
        removed = found.removed[canonical_candidates]
        kept = found.kept[canonical_candidates]
        for orbit in np.unique(candidate_orbits):
            members = np.flatnonzero(candidate_orbits == orbit)
            endpoints = np.concatenate((removed[members], kept[members]))
            if np.unique(endpoints).size != endpoints.size:
                candidate_valid[members] = False
                continue
            occupied_cells: set[int] = set()
            conflict = False
            for member in members:
                _, positions = _expand(
                    topology.ball_offsets,
                    np.asarray((removed[member], kept[member]), dtype=np.int64),
                )
                cells = np.unique(topology.ball_cells[positions])
                if any(int(cell_id) in occupied_cells for cell_id in cells):
                    conflict = True
                    break
                occupied_cells.update(int(cell_id) for cell_id in cells)
            if conflict:
                candidate_valid[members] = False
        eligible, operation_of_edge = _complete_periodic_operations(
            short_edges,
            periodic_context.edge_orbits,
            candidate_valid,
            periodic_context.edge_quotient_ids.size,
        )
        if eligible.size == 0:
            return state, counts, 0
        orbit_lengths = np.full(
            (periodic_context.edge_quotient_ids.size,), np.inf, dtype=np.float64
        )
        np.minimum.at(orbit_lengths, periodic_context.edge_orbits, topology.lengths)
        rank = _ranks(
            orbit_lengths[eligible],
            periodic_context.edge_quotient_ids[eligible],
        )
        owner_a, position_a = _expand(topology.ball_offsets, removed)
        owner_b, position_b = _expand(topology.ball_offsets, kept)
        pair_operations = np.concatenate(
            (operation_of_edge[owner_a], operation_of_edge[owner_b])
        )
        pair_items = np.concatenate(
            (topology.ball_cells[position_a], topology.ball_cells[position_b])
        )
        active_pairs = pair_operations >= 0
        selected_operations = _disjoint_selection(
            rank,
            pair_operations[active_pairs],
            pair_items[active_pairs],
            state.cells.shape[0],
        )
        active_edges = operation_of_edge >= 0
        selected_edges = np.flatnonzero(
            active_edges & selected_operations[np.maximum(operation_of_edge, 0)]
        )
        removed = removed[selected_edges]
        kept = kept[selected_edges]

    updated = _collapse_records(
        _collapse_cells(state, topology, removed, kept), removed, kept
    )
    alive = updated.alive.copy()
    alive[removed] = False
    collapse_to = updated.collapse_to.copy()
    collapse_to[removed] = kept
    updated = updated._replace(alive=alive, collapse_to=collapse_to)
    return (
        updated,
        counts._replace(collapses=counts.collapses + removed.size),
        removed.size,
    )


# ------------------------------------------------------------------ flip


def _flip_round(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    mode: PredicateMode,
    periodic_context: _PeriodicContext | None,
    _source: _Source,
    /,
) -> tuple[_State, _Counts, int]:
    first_cell = topology.edge_cells[:, 0]
    second_cell = topology.edge_cells[:, 1]
    candidates = np.flatnonzero(
        (second_cell >= 0)
        & (topology.edge_class == 0)
        & ~topology.edge_protected
        & (state.cell_region[first_cell] == state.cell_region[np.maximum(second_cell, 0)])
    )
    if candidates.size == 0:
        return state, counts, 0
    c0 = first_cell[candidates]
    c1 = second_cell[candidates]
    local = np.argmax(topology.cell_edges[c0] == candidates[:, None], axis=1)
    rows = state.cells[c0]
    span = np.arange(candidates.size)
    u = rows[span, local]
    v = rows[span, (local + 1) % 3]
    x0 = rows[span, (local + 2) % 3]
    x1 = topology.edge_apex[candidates, 1]
    left = np.stack((u, x1, x0), axis=1)
    right = np.stack((v, x0, x1), axis=1)
    triangles = np.concatenate((left, right))
    lengths = _metric_lengths(
        state.metric,
        state.points,
        np.stack((triangles, triangles[:, (1, 2, 0)]), axis=2).reshape((-1, 2)),
    ).reshape((-1, 3))
    quality = _quality(lengths)
    after = np.minimum(quality[: candidates.size], quality[candidates.size :])
    before = np.minimum(topology.quality[c0], topology.quality[c1])
    points = state.points
    status = np.maximum(
        _orientation(points[u], points[x1], points[x0], mode),
        _orientation(points[v], points[x0], points[x1], mode),
    )
    duplicate = _lookup(topology.codes, _codes(_sorted_pairs(x0, x1))) >= 0
    status = np.where(duplicate, _INVALID, status)
    diagonal_lengths = lengths[:, 1].reshape((2, candidates.size))
    metric_admissible = np.all(
        (diagonal_lengths >= _LOWER) & (diagonal_lengths <= _UPPER), axis=0
    )
    # A flip may not create an edge the next metric pass must immediately edit.
    improving = (after > before + _IMPROVEMENT) & metric_admissible
    counts = _tally(counts, status[improving])
    valid = np.flatnonzero(improving & (status == _OK))
    if valid.size == 0:
        return state, counts, 0
    if periodic_context is None:
        ids = state.vertex_ids
        edges = topology.edges[candidates[valid]]
        rank = _ranks(before[valid] - after[valid], ids[edges[:, 0]], ids[edges[:, 1]])
        local_ops = np.concatenate((np.arange(valid.size), np.arange(valid.size)))
        selected = valid[
            _disjoint_selection(
                rank,
                local_ops,
                np.concatenate((c0[valid], c1[valid])),
                state.cells.shape[0],
            )
        ]
    else:
        candidate_valid = improving & (status == _OK)
        for orbit in np.unique(periodic_context.edge_orbits[candidates]):
            members = np.flatnonzero(periodic_context.edge_orbits[candidates] == orbit)
            incident_cells = np.concatenate((c0[members], c1[members]))
            if np.unique(incident_cells).size != incident_cells.size:
                candidate_valid[members] = False
        eligible, operation_of_candidate = _complete_periodic_operations(
            candidates,
            periodic_context.edge_orbits,
            candidate_valid,
            periodic_context.edge_quotient_ids.size,
        )
        if eligible.size == 0:
            return state, counts, 0
        orbit_gain = np.full(
            (periodic_context.edge_quotient_ids.size,), np.inf, dtype=np.float64
        )
        np.minimum.at(
            orbit_gain,
            periodic_context.edge_orbits[candidates],
            before - after,
        )
        rank = _ranks(
            orbit_gain[eligible],
            periodic_context.edge_quotient_ids[eligible],
        )
        pair_operations = operation_of_candidate[
            np.concatenate((np.arange(candidates.size), np.arange(candidates.size)))
        ]
        pair_items = np.concatenate((c0, c1))
        inside = pair_operations >= 0
        selected_operations = _disjoint_selection(
            rank,
            pair_operations[inside],
            pair_items[inside],
            state.cells.shape[0],
        )
        selected = np.flatnonzero(
            (operation_of_candidate >= 0)
            & selected_operations[np.maximum(operation_of_candidate, 0)]
        )
    count = selected.size
    children = np.concatenate((left[selected], right[selected]))
    child = np.arange(2 * count, dtype=np.int64)
    updated = _replace_cells(
        state,
        np.concatenate((c0[selected], c1[selected])),
        children,
        np.concatenate((child, child)),
        np.concatenate((c0[selected], c1[selected], c1[selected], c0[selected])),
        _SWAPPED,
    )
    flipped = np.sort(_codes(topology.edges[candidates[selected]]))
    survive = _lookup(flipped, _codes(updated.lineage_keys)) < 0
    updated = updated._replace(
        lineage_keys=updated.lineage_keys[survive],
        lineage_sources=updated.lineage_sources[survive],
        lineage_ranks=updated.lineage_ranks[survive],
    )
    return updated, counts._replace(flips=counts.flips + count), count


# ------------------------------------------------------------------ relocation


def _relocation_targets(state: _State, topology: _Topology, vertices: Any, /) -> Any:
    """Spring displacement toward unit metric lengths and each vertex's badness.

    Interior vertices use every neighbor; curve vertices only their two curve
    neighbors, so the displacement stays on the straight curve line.
    """
    owner, position = _expand(topology.neighbor_offsets, vertices)
    other = topology.neighbor_values[position]
    edge = _lookup(topology.codes, _codes(_sorted_pairs(vertices[owner], other)))
    length = topology.lengths[edge]
    badness = _group_maximum(np.maximum(length, 1.0 / length), owner, vertices.size, 0.0)
    curve = topology.kind[vertices] == _CURVE
    use = ~curve[owner] | np.any(
        topology.curve_neighbors[vertices[owner]] == other[:, None], axis=1
    )
    pull = (1.0 - 1.0 / length[use])[:, None] * (
        state.points[other[use]] - state.points[vertices[owner[use]]]
    )
    total = np.zeros((vertices.size, 2), dtype=np.float64)
    np.add.at(total, owner[use], pull)
    number = np.bincount(owner[use], minlength=vertices.size)
    return total / np.maximum(number, 1)[:, None], badness


def _proposal_metric(
    state: _State, topology: _Topology, vertex: Any, target: Any, /
) -> Any:
    """Log-Euclidean P1 metric at each proposal inside the current vertex ball."""
    owner, position = _expand(topology.ball_offsets, vertex)
    cell = topology.ball_cells[position]
    weights = _barycentric(state.points[state.cells[cell]], target[owner])
    score = np.min(weights, axis=1)
    order = np.lexsort((cell, -score, owner))
    first = order[np.searchsorted(owner[order], np.arange(vertex.size))]
    samples = state.metric[state.cells[cell[first]]]
    return _interpolate_metrics(samples, _normalized(weights[first]))


def _proposal_checks(
    state: _State,
    topology: _Topology,
    vertex: Any,
    target: Any,
    metric: Any,
    mode: Any,
    /,
) -> Any:
    """Worst orientation status and minimum metric quality of each proposal."""
    owner, position = _expand(topology.ball_offsets, vertex)
    rows = state.cells[topology.ball_cells[position]]
    base = state.points.shape[0]
    moved = np.where(rows == vertex[owner][:, None], base + owner[:, None], rows)
    points = np.concatenate((state.points, target))
    orientation = _orientation(
        points[moved[:, 0]], points[moved[:, 1]], points[moved[:, 2]], mode
    )
    status = _group_maximum(orientation, owner, vertex.size, _OK)
    curve = np.flatnonzero(topology.kind[vertex] == _CURVE)
    neighbors = topology.curve_neighbors[vertex[curve]]
    status[curve] = np.maximum(
        status[curve],
        _collinear(
            state.points[neighbors[:, 0]],
            target[curve],
            state.points[neighbors[:, 1]],
            mode,
        ),
    )
    lengths = _metric_lengths(
        np.concatenate((state.metric, metric)),
        points,
        np.stack((moved, moved[:, (1, 2, 0)]), axis=2).reshape((-1, 2)),
    ).reshape((-1, 3))
    quality = _group_minimum(_quality(lengths), owner, vertex.size, np.inf)
    return status, quality


def _relocation_round_nonperiodic(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    done: np.ndarray,
    mode: PredicateMode,
    source: _Source,
    /,
) -> tuple[_State, _Counts, int, np.ndarray]:
    """Backtrack each nonperiodic spring proposal to representability."""
    valence = np.diff(topology.ball_offsets)
    movable = (
        (valence > 0)
        & ~done
        & ((topology.kind == _INTERIOR) | ((topology.kind == _CURVE) & topology.straight))
    )
    vertices = np.flatnonzero(movable)
    displacement, badness = _relocation_targets(state, topology, vertices)
    active = np.any(displacement != 0.0, axis=1)
    vertices, displacement, badness = (
        vertices[active],
        displacement[active],
        badness[active],
    )
    if vertices.size == 0:
        return state, counts, 0, done

    ball_owner, ball_position = _expand(topology.ball_offsets, vertices)
    before = _group_minimum(
        topology.quality[topology.ball_cells[ball_position]],
        ball_owner,
        vertices.size,
        np.inf,
    )
    origin = state.points[vertices]
    target_best = np.zeros_like(origin)
    metric_best = np.zeros((vertices.size, 2, 2), dtype=np.float64)
    stencil_sources_best = state.source_stencil_sources[vertices].copy()
    stencil_weights_best = state.source_stencil_weights[vertices].copy()
    stencil_valid_best = state.source_stencil_valid[vertices].copy()
    source_stencil_exact_best = [
        state.source_stencil_exact[int(vertex)] for vertex in vertices.tolist()
    ]
    success = np.zeros((vertices.size,), dtype=np.bool_)
    rejected_invalid = np.zeros_like(success)
    rejected_uncertain = np.zeros_like(success)
    pending = np.ones_like(success)
    scale = np.ones((vertices.size,), dtype=np.float64)

    while np.any(pending):
        target_bank = np.empty(
            (_RELOCATION_SCALE_LANES, vertices.size, 2), dtype=np.float64
        )
        active_bank = np.zeros((_RELOCATION_SCALE_LANES, vertices.size), dtype=np.bool_)
        scale_bank = scale.copy()
        representable = pending.copy()
        for lane in range(_RELOCATION_SCALE_LANES):
            target = origin + scale_bank[:, None] * displacement
            representable &= np.any(target != origin, axis=1)
            target_bank[lane] = target
            active_bank[lane] = representable
            scale_bank[representable] *= 0.5
        if not np.any(active_bank):
            break

        vertex_bank = np.tile(vertices, _RELOCATION_SCALE_LANES)
        target_rows = target_bank.reshape((-1, 2))
        metric_rows = _proposal_metric(state, topology, vertex_bank, target_rows)
        status_rows, quality_rows = _proposal_checks(
            state, topology, vertex_bank, target_rows, metric_rows, mode
        )
        metric_bank = metric_rows.reshape((_RELOCATION_SCALE_LANES, vertices.size, 2, 2))
        status_bank = status_rows.reshape((_RELOCATION_SCALE_LANES, vertices.size))
        quality_bank = quality_rows.reshape((_RELOCATION_SCALE_LANES, vertices.size))
        improving_bank = (
            active_bank
            & (quality_bank > before[None, :] + _IMPROVEMENT)
            & (status_bank == _OK)
        )
        candidate_indices = np.flatnonzero(improving_bank.reshape((-1,)))
        source_stencil_rows = np.full((improving_bank.size,), -1, dtype=np.int64)
        candidate_support = np.zeros((improving_bank.size,), dtype=np.bool_)
        source_stencils: (
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                tuple[ExactSourceRow, ...],
            ]
            | None
        ) = None
        if candidate_indices.size:
            source_stencils = _located_stencil_bank(
                source,
                target_rows[candidate_indices],
                mode,
                allow_invalid_rows=True,
                stencil_width=state.source_stencil_weights.shape[1],
            )
            if source_stencils[0].shape[1] > stencil_sources_best.shape[1]:
                stencil_sources_best, stencil_weights_best, stencil_valid_best = (
                    _pad_stencil_columns(
                        stencil_sources_best,
                        stencil_weights_best,
                        stencil_valid_best,
                        source_stencils[0].shape[1],
                    )
                )
            source_stencil_rows[candidate_indices] = np.arange(
                candidate_indices.size, dtype=np.int64
            )
            candidate_support[candidate_indices] = source_stencils[3]
        for lane in range(_RELOCATION_SCALE_LANES):
            active = pending & active_bank[lane]
            improving = active & (quality_bank[lane] > before + _IMPROVEMENT)
            accepted = improving & (status_bank[lane] == _OK)
            lane_rows = lane * vertices.size + np.arange(vertices.size)
            has_support = candidate_support[lane_rows]
            unsupported = accepted & ~has_support
            rejected_uncertain |= unsupported
            accepted &= has_support
            chosen = np.flatnonzero(accepted)
            if chosen.size:
                assert source_stencils is not None
                stencil_rows = source_stencil_rows[lane_rows[chosen]]
                stencil_sources_best[chosen] = source_stencils[0][stencil_rows]
                stencil_weights_best[chosen] = source_stencils[1][stencil_rows]
                stencil_valid_best[chosen] = source_stencils[2][stencil_rows]
                for chosen_index, stencil_row in zip(
                    chosen.tolist(), stencil_rows.tolist(), strict=True
                ):
                    source_stencil_exact_best[chosen_index] = source_stencils[4][
                        stencil_row
                    ]
                success[chosen] = True
                target_best[chosen] = target_bank[lane, chosen]
                metric_best[chosen] = metric_bank[lane, chosen]
            rejected_invalid |= improving & (status_bank[lane] >= _INVALID)
            rejected_uncertain |= improving & (status_bank[lane] == _UNCERTAIN)
            pending &= ~accepted
        scale = scale_bank
        pending &= representable

    rejected = np.where(
        rejected_invalid,
        _INVALID,
        np.where(rejected_uncertain, _UNCERTAIN, _OK),
    )
    counts = _tally(counts, rejected[~success])
    winners = np.flatnonzero(success)
    if winners.size == 0:
        return state, counts, 0, done
    rank = _ranks(-badness[winners], state.vertex_ids[vertices[winners]])
    local = np.full((vertices.size,), -1, dtype=np.int64)
    local[winners] = np.arange(winners.size)
    member = local[ball_owner] >= 0
    selected = _disjoint_selection(
        rank,
        local[ball_owner[member]],
        topology.ball_cells[ball_position[member]],
        state.cells.shape[0],
    )
    chosen = winners[selected]
    moved = vertices[chosen]
    points = state.points.copy()
    points[moved] = target_best[chosen]
    tensors = state.metric.copy()
    tensors[moved] = metric_best[chosen]
    flags = state.moved.copy()
    flags[moved] = True
    finished = done.copy()
    finished[moved] = True
    source_stencil_sources, source_stencil_weights, source_stencil_valid = (
        _pad_stencil_columns(
            state.source_stencil_sources,
            state.source_stencil_weights,
            state.source_stencil_valid,
            stencil_sources_best.shape[1],
        )
    )
    source_stencil_sources = source_stencil_sources.copy()
    source_stencil_weights = source_stencil_weights.copy()
    source_stencil_valid = source_stencil_valid.copy()
    source_stencil_sources[moved] = stencil_sources_best[chosen]
    source_stencil_weights[moved] = stencil_weights_best[chosen]
    source_stencil_valid[moved] = stencil_valid_best[chosen]
    source_stencil_exact = list(state.source_stencil_exact)
    for vertex, exact_row in zip(
        moved.tolist(),
        (source_stencil_exact_best[index] for index in chosen.tolist()),
        strict=True,
    ):
        source_stencil_exact[int(vertex)] = exact_row
    updated = _raise_touching(
        state._replace(
            points=points,
            metric=tensors,
            moved=flags,
            source_stencil_sources=source_stencil_sources,
            source_stencil_weights=source_stencil_weights,
            source_stencil_valid=source_stencil_valid,
            source_stencil_exact=tuple(source_stencil_exact),
        ),
        moved,
        _RELOCATED,
    )
    return (
        updated,
        counts._replace(relocations=counts.relocations + moved.size),
        moved.size,
        finished,
    )


def _relocation_round_periodic(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    done: np.ndarray,
    mode: PredicateMode,
    periodic_context: _PeriodicContext,
    source: _Source,
    /,
) -> tuple[_State, _Counts, int, np.ndarray]:
    valence = np.diff(topology.ball_offsets)
    movable = (
        (valence > 0)
        & ~done
        & ((topology.kind == _INTERIOR) | ((topology.kind == _CURVE) & topology.straight))
    )
    vertices = np.flatnonzero(movable)
    displacement, badness = _relocation_targets(state, topology, vertices)
    active = np.any(displacement != 0.0, axis=1)
    vertices, displacement, badness = (
        vertices[active],
        displacement[active],
        badness[active],
    )
    if vertices.size == 0:
        return state, counts, 0, done
    position_by_vertex = {
        int(vertex): row for row, vertex in enumerate(vertices.tolist())
    }
    before_by_vertex = {
        int(vertex): float(
            np.min(
                topology.quality[
                    topology.ball_cells[
                        topology.ball_offsets[int(vertex)] : topology.ball_offsets[
                            int(vertex) + 1
                        ]
                    ]
                ],
                initial=np.inf,
            )
        )
        for vertex in vertices.tolist()
    }
    rejected: list[int] = []
    candidates: list[dict[str, Any]] = []
    orbits = np.asarray(periodic_context.vertex_orbits, dtype=np.int64)
    for orbit in np.unique(orbits[vertices]):
        members = np.flatnonzero(orbits == orbit)
        if not members.size or any(
            int(vertex) not in position_by_vertex for vertex in members
        ):
            continue
        member_positions = np.asarray(
            [position_by_vertex[int(vertex)] for vertex in members], dtype=np.int64
        )
        member_cells = [
            topology.ball_cells[
                topology.ball_offsets[int(vertex)] : topology.ball_offsets[
                    int(vertex) + 1
                ]
            ]
            for vertex in members.tolist()
        ]
        cavity = frozenset(int(cell) for rows in member_cells for cell in rows.tolist())
        if sum(rows.size for rows in member_cells) != len(cavity):
            continue
        before = np.asarray(
            [before_by_vertex[int(vertex)] for vertex in members], dtype=np.float64
        )
        origin = state.points[members]
        member_displacement = displacement[member_positions]
        representative = int(periodic_context.vertex_representatives[members[0]])
        representative_position = position_by_vertex.get(representative)
        if representative_position is None:
            continue
        # ``origin`` and ``member_displacement`` follow member order, while
        # ``representative_position`` indexes the movable-vertex bank.
        representative_member = int(np.flatnonzero(members == representative)[0])
        scale = 1.0
        had_uncertain = False
        had_invalid = False
        accepted: dict[str, Any] | None = None
        while True:
            representative_target = (
                origin[representative_member]
                + scale * member_displacement[representative_member]
            )
            if np.array_equal(representative_target, origin[representative_member]):
                break
            representative_metric = _proposal_metric(
                state,
                topology,
                np.asarray((representative,), dtype=np.int64),
                representative_target[None, :],
            )[0]
            target = np.stack(
                [
                    _exact_periodic_point(
                        periodic_context.vertex_exact_isometries[int(vertex)],
                        representative_target,
                    )
                    for vertex in members.tolist()
                ]
            )
            metric = np.stack(
                [
                    _exact_periodic_metric(
                        periodic_context.vertex_exact_isometries[int(vertex)],
                        representative_metric,
                    )
                    for vertex in members.tolist()
                ]
            )
            if np.any(np.all(target == origin, axis=1)):
                break
            status, after = _proposal_checks(
                state, topology, members, target, metric, mode
            )
            improving = after > before + _IMPROVEMENT
            valid = bool(np.all((status == _OK) & improving))
            geometry_invalid = False
            if valid:
                for member_position, vertex in enumerate(members.tolist()):
                    for stabilizer in periodic_context.vertex_stabilizers[int(vertex)]:
                        if not np.array_equal(
                            target[member_position],
                            _exact_periodic_point(stabilizer, target[member_position]),
                        ) or not np.array_equal(
                            metric[member_position],
                            _exact_periodic_metric(stabilizer, metric[member_position]),
                        ):
                            valid = False
                            geometry_invalid = True
                            break
                    if not valid:
                        break
            source_stencils = None
            if valid:
                source_stencils = _located_stencil_bank(
                    source,
                    target,
                    mode,
                    allow_invalid_rows=True,
                    stencil_width=state.source_stencil_weights.shape[1],
                )
                if not np.all(source_stencils[3]):
                    valid = False
                    had_uncertain = True
            if valid:
                accepted = {
                    "vertices": members.copy(),
                    "target": target.copy(),
                    "metric": metric.copy(),
                    "source_stencils": source_stencils,
                    "cavity": cavity,
                    "quotient_id": int(periodic_context.vertex_quotient_ids[int(orbit)]),
                    "gain": float(np.min(after - before)),
                    "badness": float(badness[representative_position]),
                }
                break
            had_uncertain |= bool(np.any(status == _UNCERTAIN))
            had_invalid |= bool(np.any(status >= _INVALID)) or geometry_invalid
            scale *= 0.5
        if accepted is None:
            if had_uncertain:
                rejected.append(_UNCERTAIN)
            elif had_invalid:
                rejected.append(_INVALID)
            continue
        candidates.append(accepted)
    if rejected:
        counts = _tally(counts, np.asarray(rejected, dtype=np.int8))
    selected: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for candidate in sorted(
        candidates,
        key=lambda value: (
            -value["badness"],
            value["quotient_id"],
            tuple(int(state.vertex_ids[row]) for row in value["vertices"]),
        ),
    ):
        if occupied.isdisjoint(candidate["cavity"]):
            selected.append(candidate)
            occupied.update(candidate["cavity"])
    if not selected:
        return state, counts, 0, done
    moved = np.concatenate([candidate["vertices"] for candidate in selected]).astype(
        np.int64, copy=False
    )
    points = state.points.copy()
    tensors = state.metric.copy()
    for candidate in selected:
        rows = candidate["vertices"]
        points[rows] = candidate["target"]
        tensors[rows] = candidate["metric"]
    flags = state.moved.copy()
    flags[moved] = True
    finished = done.copy()
    finished[moved] = True
    stencil_width = max(
        (
            state.source_stencil_weights.shape[1],
            *(candidate["source_stencils"][0].shape[1] for candidate in selected),
        )
    )
    source_stencil_sources, source_stencil_weights, source_stencil_valid = (
        _pad_stencil_columns(
            state.source_stencil_sources,
            state.source_stencil_weights,
            state.source_stencil_valid,
            stencil_width,
        )
    )
    source_stencil_sources = source_stencil_sources.copy()
    source_stencil_weights = source_stencil_weights.copy()
    source_stencil_valid = source_stencil_valid.copy()
    source_stencil_exact = list(state.source_stencil_exact)
    for candidate in selected:
        rows = candidate["vertices"]
        stencil_sources, stencil_weights, stencil_valid, _, stencil_exact = candidate[
            "source_stencils"
        ]
        stencil_sources, stencil_weights, stencil_valid = _pad_stencil_columns(
            stencil_sources,
            stencil_weights,
            stencil_valid,
            stencil_width,
        )
        source_stencil_sources[rows] = stencil_sources
        source_stencil_weights[rows] = stencil_weights
        source_stencil_valid[rows] = stencil_valid
        for vertex, exact_row in zip(rows.tolist(), stencil_exact, strict=True):
            source_stencil_exact[int(vertex)] = exact_row
    updated = _raise_touching(
        state._replace(
            points=points,
            metric=tensors,
            moved=flags,
            source_stencil_sources=source_stencil_sources,
            source_stencil_weights=source_stencil_weights,
            source_stencil_valid=source_stencil_valid,
            source_stencil_exact=tuple(source_stencil_exact),
        ),
        moved,
        _RELOCATED,
    )
    return (
        updated,
        counts._replace(relocations=counts.relocations + moved.size),
        moved.size,
        finished,
    )


def _relocation_round(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    done: np.ndarray,
    mode: PredicateMode,
    periodic_context: _PeriodicContext | None,
    source: _Source,
    /,
) -> tuple[_State, _Counts, int, np.ndarray]:
    if periodic_context is None:
        return _relocation_round_nonperiodic(state, topology, counts, done, mode, source)
    return _relocation_round_periodic(
        state, topology, counts, done, mode, periodic_context, source
    )


# ------------------------------------------------------------------ passes


def _periodic_topology(
    mesh: CellMesh,
    source: _Source,
    state: _State,
    topology: _Topology,
    mode: PredicateMode,
    /,
) -> tuple[_Topology, _PeriodicContext | None]:
    context = _periodic_state_context(mesh, source, state, topology, mode)
    if context is None:
        return topology, None
    return (
        topology._replace(
            lengths=context.edge_lengths,
            quality=context.cell_quality,
        ),
        context,
    )


def _run_phase(
    state: _State,
    counts: _Counts,
    phase: Any,
    mode: PredicateMode,
    periodic_mesh: CellMesh,
    source: _Source,
    /,
) -> Any:
    total = 0
    while True:
        topology, periodic_context = _periodic_topology(
            periodic_mesh, source, state, _topology(state, mode), mode
        )
        state, counts, count = phase(
            state, topology, counts, mode, periodic_context, source
        )
        if count == 0:
            return state, counts, total
        total += count


def _run_pass(
    state: _State,
    counts: _Counts,
    mode: PredicateMode,
    topology_operations: bool,
    relocation: bool,
    periodic_mesh: CellMesh,
    source: _Source,
    /,
) -> tuple[_State, _Counts, int]:
    applied = 0
    if topology_operations:
        for phase in (_split_round, _collapse_round, _flip_round):
            state, counts, count = _run_phase(
                state, counts, phase, mode, periodic_mesh, source
            )
            applied += count
    if relocation:
        done = np.zeros((state.points.shape[0],), dtype=np.bool_)
        while True:
            topology, periodic_context = _periodic_topology(
                periodic_mesh, source, state, _topology(state, mode), mode
            )
            state, counts, count, done = _relocation_round(
                state, topology, counts, done, mode, periodic_context, source
            )
            applied += count
            if count == 0:
                break
    return state, counts, applied


# ------------------------------------------------------------------ assembly


def _locate(
    source: _Source, queries: np.ndarray, mode: PredicateMode, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Containing source cell and floating P1 weights for each supported query."""
    triangles = source.points[source.cells]
    points_tree = prepare_bvh(queries, queries, dtype=jnp.float64)
    cell, query = bvh_overlap_pairs_host(
        source.cell_tree,
        points_tree,
        include_touching=True,
        relative_tolerance=_LOCATION_TOLERANCE,
    )
    located = np.zeros((queries.shape[0],), dtype=np.bool_)
    located[query] = True
    cells = np.full((queries.shape[0],), -1, dtype=np.int64)
    chosen = np.zeros((queries.shape[0], 3), dtype=np.float64)
    if not np.any(located):
        return cells, chosen, located
    query_rows = np.flatnonzero(located)
    corners = triangles[cell]
    point = queries[query]
    result = orient2d(
        np.stack((corners[:, 1], corners[:, 2], corners[:, 0]), axis=1),
        np.stack((corners[:, 2], corners[:, 0], corners[:, 1]), axis=1),
        np.repeat(point[:, None, :], 3, axis=1),
        mode=mode,
    )
    signs = np.asarray(result.signs)
    certain = np.asarray(result.certain)
    negative = certain & (signs == PredicateSign.NEGATIVE)
    contained = np.all(certain & (signs != PredicateSign.NEGATIVE), axis=1)
    grade = np.where(contained, 0, np.where(np.any(negative, axis=1), 2, 1))
    barycentric = _barycentric(corners, point)
    violation = np.where(grade == 0, 0.0, -np.min(barycentric, axis=1))
    order = np.lexsort((source.cell_ids[cell], violation, grade, query))
    first = order[np.searchsorted(query[order], query_rows)]
    exact_zero = certain[first] & (signs[first] == PredicateSign.ZERO)
    cells[query_rows] = cell[first]
    chosen[query_rows] = _normalized(np.where(exact_zero, 0.0, barycentric[first]))
    return cells, chosen, located


def _located_stencil_bank(
    source: _Source,
    points: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    allow_invalid_rows: bool = False,
    stencil_width: int = 3,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[ExactSourceRow, ...],
]:
    cells, _, _ = _locate(source, points, mode)
    entries_by_row: list[ExactSourceRow] = [()] * points.shape[0]
    supported = np.zeros((points.shape[0],), dtype=np.bool_)
    outside = cells < 0
    underflow = np.zeros((points.shape[0],), dtype=np.bool_)
    width = max(3, int(stencil_width))
    for row, cell in enumerate(cells.tolist()):
        if cell < 0:
            continue
        vertices = source.cells[cell]
        corners = [
            tuple(Fraction(float(value)) for value in source.points[int(vertex)].tolist())
            for vertex in vertices.tolist()
        ]
        origin = corners[0]
        matrix = [
            [corners[column + 1][axis] - origin[axis] for column in range(2)]
            for axis in range(2)
        ]
        query = tuple(Fraction(float(value)) for value in points[row].tolist())
        tail = solve_exact(
            matrix,
            [[query[0] - origin[0]], [query[1] - origin[1]]],
        )
        barycentric = (
            Fraction(1) - tail[0][0] - tail[1][0],
            tail[0][0],
            tail[1][0],
        )
        if any(weight < 0 for weight in barycentric):
            outside[row] = True
            if allow_invalid_rows:
                continue
            raise ValueError("A target vertex lies outside the source domain.")
        entries: list[tuple[int, Fraction]] = []
        for vertex, exact_weight in zip(vertices.tolist(), barycentric, strict=True):
            if exact_weight == 0:
                continue
            projection = float(exact_weight)
            if not np.isfinite(projection) or projection == 0.0:
                underflow[row] = True
                break
            entries.append((int(source.vertex_ids[int(vertex)]), exact_weight))
        entries.sort(key=lambda entry: entry[0])
        if underflow[row] or not entries:
            if allow_invalid_rows:
                continue
            raise FloatingPointError(
                "Exact source coefficient has no nonzero finite binary64 projection."
            )
        if (
            len({source_id for source_id, _ in entries}) != len(entries)
            or sum((weight for _, weight in entries), Fraction(0)) != 1
        ):
            raise ValueError("Exact source interpolation row is not canonical.")
        entries_by_row[row] = tuple(entries)
        supported[row] = True
        width = max(width, len(entries))
    if not allow_invalid_rows and not np.all(supported):
        if np.any(outside & ~supported):
            raise ValueError("A target vertex lies outside the source domain.")
        raise FloatingPointError(
            "Exact source coefficient has no nonzero finite binary64 projection."
        )
    sources = np.full((points.shape[0], width), -1, dtype=np.int64)
    weights = np.zeros((points.shape[0], width), dtype=np.float64)
    valid = np.zeros((points.shape[0], width), dtype=np.bool_)
    for row, source_row in enumerate(entries_by_row):
        for column, (source_id, exact_weight) in enumerate(source_row):
            sources[row, column] = source_id
            weights[row, column] = float(exact_weight)
            valid[row, column] = True
    return (
        sources,
        weights,
        valid,
        supported,
        tuple(entries_by_row),
    )


def _target_order(state: _State, /) -> Any:
    """Target rows: surviving source vertices in source order, then new vertices."""
    rows = np.flatnonzero(state.alive)
    source = rows[state.source_rows[rows] >= 0]
    created = rows[state.source_rows[rows] < 0]
    source = source[np.argsort(state.source_rows[source], kind="stable")]
    created = created[np.argsort(state.vertex_ids[created], kind="stable")]
    order = np.concatenate((source, created))
    first_new = int(np.max(state.vertex_ids[state.source_rows >= 0])) + 1
    identifiers = np.concatenate(
        (
            state.vertex_ids[source],
            first_new + np.arange(created.size, dtype=np.int64),
        )
    )
    final = np.full((state.points.shape[0],), -1, dtype=np.int64)
    final[order] = identifiers
    return order, identifiers, final


def _stencil(
    state: _State, source: _Source, order: Any, mode: PredicateMode, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[Fraction, ...], ...]]:
    order_rows = np.asarray(order, dtype=np.int64)
    exact_rows = [state.source_stencil_exact[int(row)] for row in order_rows]
    missing = [row for row, entries in enumerate(exact_rows) if not entries]
    if missing:
        located = _located_stencil_bank(
            source,
            state.points[order_rows[missing]],
            mode,
            stencil_width=3,
        )
        for row, exact_row in zip(missing, located[4], strict=True):
            exact_rows[row] = exact_row
    source_order = np.argsort(source.vertex_ids, kind="stable")
    source_ids = source.vertex_ids[source_order]
    width = max(3, *(len(row) for row in exact_rows))
    sources = np.full((len(exact_rows), width), -1, dtype=np.int64)
    weights = np.zeros((len(exact_rows), width), dtype=np.float64)
    valid = np.zeros((len(exact_rows), width), dtype=np.bool_)
    exact_weights = [[Fraction(0) for _ in range(width)] for _ in exact_rows]
    for row, entries in enumerate(exact_rows):
        if (
            not entries
            or len({source_id for source_id, _ in entries}) != len(entries)
            or sum((weight for _, weight in entries), Fraction(0)) != 1
            or any(weight < 0 for _, weight in entries)
        ):
            raise ValueError("Target vertex exact source row is not canonical.")
        target = tuple(
            Fraction(float(value)) for value in state.points[int(order_rows[row])]
        )
        reproduced = [Fraction(0) for _ in range(source.points.shape[1])]
        for column, (source_id, exact_weight) in enumerate(entries):
            position = int(np.searchsorted(source_ids, source_id))
            if position >= source_ids.size or source_ids[position] != source_id:
                raise ValueError("Exact source row references an unavailable vertex.")
            source_row = int(source_order[position])
            coordinate = tuple(
                Fraction(float(value)) for value in source.points[source_row]
            )
            for axis, value in enumerate(coordinate):
                reproduced[axis] += exact_weight * value
            projection = float(exact_weight)
            if not np.isfinite(projection) or projection == 0.0:
                raise FloatingPointError(
                    "Exact source coefficient has no nonzero finite binary64 projection."
                )
            sources[row, column] = source_id
            weights[row, column] = projection
            valid[row, column] = True
            exact_weights[row][column] = exact_weight
        if tuple(reproduced) != target:
            raise ValueError(
                "Exact source interpolation row does not reproduce its target."
            )
    return sources, weights, valid, tuple(tuple(row) for row in exact_weights)


def _surviving(state: _State, /) -> np.ndarray:
    """Final surviving vertex row of every vertex row (pointer jumping)."""
    final = np.where(
        state.alive, np.arange(state.points.shape[0], dtype=np.int64), state.collapse_to
    )
    while True:
        following = final[final]
        if np.array_equal(following, final):
            break
        final = following
    return final


def _relation(
    dimension: Any, sources: Any, targets: Any, kinds: Any, /
) -> EntityRelations:
    order = np.lexsort((*targets.T[::-1], *sources.T[::-1]))
    return EntityRelations(
        dimension, sources[order], targets[order], kinds[order].astype(np.int32)
    )


def _vertex_relations(state: Any, final: Any, /) -> Any:
    ids = state.vertex_ids
    rows = np.flatnonzero((state.source_rows >= 0) & state.alive & state.moved)
    collapsed = np.flatnonzero((state.source_rows >= 0) & ~state.alive)
    survivor = _surviving(state)[collapsed]
    sources = np.concatenate((ids[rows], ids[collapsed]))
    targets = np.concatenate((ids[rows], final[survivor]))
    kinds = np.concatenate(
        (
            np.full(rows.shape, EntityLineageKind.RELOCATED, dtype=np.int32),
            np.full(collapsed.shape, EntityLineageKind.COLLAPSED_INTO, dtype=np.int32),
        )
    )
    return _relation(0, sources[:, None], targets[:, None], kinds)


def _edge_relations(state: _State, source: _Source, final: Any, /) -> EntityRelations:
    changed = np.flatnonzero(state.lineage_ranks > 0)
    targets = np.sort(final[state.lineage_keys[changed]], axis=1)
    if np.any(targets < 0):
        raise RuntimeError("Edge lineage references a removed vertex.")
    return _relation(
        1,
        source.edge_keys[state.lineage_sources[changed]],
        targets,
        _RANK_KINDS[state.lineage_ranks[changed]],
    )


def _cell_identifiers(state: _State, source: _Source, final: Any, /) -> np.ndarray:
    identifiers = state.cell_ids.copy()
    new = np.flatnonzero(identifiers < 0)
    keys = np.sort(final[state.cells[new]], axis=1)
    order = np.lexsort(keys.T[::-1])
    identifiers[new[order]] = (
        int(np.max(source.cell_ids)) + 1 + np.arange(new.size, dtype=np.int64)
    )
    return identifiers


def _cell_relations(state: _State, source: _Source, identifiers: Any, /) -> Any:
    related = (state.cell_ids < 0) | (state.cell_rank > 0)
    pairs = np.flatnonzero(related[state.ancestry_cells])
    cells = state.ancestry_cells[pairs]
    return _relation(
        2,
        source.cell_ids[state.ancestry_sources[pairs]][:, None],
        identifiers[cells][:, None],
        _RANK_KINDS[state.cell_rank[cells]],
    )


def _assemble(state: _State, source: _Source, mode: PredicateMode, /) -> Any:
    order, identifiers, final = _target_order(state)
    row_of = np.full((state.points.shape[0],), -1, dtype=np.int64)
    row_of[order] = np.arange(order.size, dtype=np.int64)
    stencil = _stencil(state, source, order, mode)
    cell_identifiers = _cell_identifiers(state, source, final)
    target_cells = row_of[state.cells].astype(np.int32)
    block_cells = tuple(
        target_cells[state.cell_block == block] for block in range(source.block_count)
    )
    block_ids = tuple(
        cell_identifiers[state.cell_block == block] for block in range(source.block_count)
    )
    relations = (
        _vertex_relations(state, final),
        _edge_relations(state, source, final),
        _cell_relations(state, source, cell_identifiers),
    )
    edit = CellTopologyEdit(
        coordinates=state.points[order],
        vertex_global_ids=identifiers,
        blocks=source_family_blocks(source.families, block_cells, block_ids),
        stencil_sources=stencil[0],
        stencil_weights=stencil[1],
        stencil_valid=stencil[2],
        stencil_exact_weights=stencil[3],
        relations=relations,
    )
    return edit, state.metric[order]


def _periodic_state_context(
    mesh: CellMesh,
    source: _Source,
    state: _State,
    topology: _Topology,
    mode: PredicateMode,
    /,
) -> _PeriodicContext | None:
    periodic = mesh.periodic_topology
    if periodic is None:
        return None
    edit, _ = _assemble(state, source, mode)
    bound = stage_periodic_topology_edit(mesh, edit)
    if bound is None:
        raise ValueError("A periodic local operation lost its quotient target.")
    bound_periodic = bound.periodic_topology
    if bound_periodic is None:
        raise ValueError("A periodic local operation lost its bound quotient.")
    edge_map = periodic_entity_orbit_map(bound, 1)
    vertex_map = periodic_entity_orbit_map(bound, 0)
    cell_map = periodic_entity_orbit_map(bound, 2)

    order, _, final = _target_order(state)
    edge_keys = np.sort(final[topology.edges], axis=1)
    edge_target_rows = key_rows(entity_keys(bound, 1), edge_keys)
    if np.any(edge_target_rows < 0):
        raise ValueError("Periodic local topology lost a lifted state edge.")
    edge_orbits = np.asarray(edge_map.lifted_to_quotient, dtype=np.int64)[
        edge_target_rows
    ]
    target_edge_vertices = np.asarray(_entity_vertex_rows(bound)[1], dtype=np.int64)
    permutation_rows = tuple(
        np.asarray(edge_map.corner_permutation_row(int(row)), dtype=np.int64)
        for row in edge_target_rows
    )
    if any(row.shape != (2,) for row in permutation_rows):
        raise ValueError("Periodic metric edge permutations require two corners.")
    edge_permutations = np.stack(permutation_rows, axis=0)
    canonical_target_vertices = target_edge_vertices[
        edge_target_rows[:, None], edge_permutations
    ]
    edge_canonical_vertices = order[canonical_target_vertices]

    vertex_orbits = np.full((state.points.shape[0],), -1, dtype=np.int64)
    vertex_orbits[order] = np.asarray(vertex_map.lifted_to_quotient, dtype=np.int64)
    row_of = np.full((state.points.shape[0],), -1, dtype=np.int64)
    row_of[order] = np.arange(order.size, dtype=np.int64)
    fixed_generators = np.asarray(bound_periodic.vertex_fixed_generators, dtype=np.bool_)
    vertex_windings = np.asarray(vertex_map.winding_shifts, dtype=np.int64)
    vertex_representatives = np.asarray(
        vertex_map.representative_lifted_rows, dtype=np.int64
    )
    representative_state_rows = np.full((state.points.shape[0],), -1, dtype=np.int64)
    identity = tuple(
        tuple(
            Fraction(int(row == column)) for column in range(mesh.ambient_dimension + 1)
        )
        for row in range(mesh.ambient_dimension + 1)
    )
    vertex_exact_isometries = [identity] * state.points.shape[0]
    vertex_stabilizers: list[tuple[tuple[tuple[Fraction, ...], ...], ...]] = [
        ()
    ] * state.points.shape[0]
    for orbit in np.unique(vertex_orbits[order]):
        state_rows = np.flatnonzero(vertex_orbits == orbit)
        target_rows = row_of[state_rows]
        representative_target_row = int(vertex_representatives[orbit])
        representative_winding = vertex_windings[representative_target_row]
        representative_state_row = int(order[representative_target_row])
        representative_state_rows[state_rows] = representative_state_row
        representative_metric = state.metric[representative_state_row]
        representative_point = state.points[representative_state_row]
        for state_row, target_row in zip(state_rows, target_rows, strict=True):
            winding = vertex_windings[target_row] - representative_winding
            exact_isometry = exact_periodic_element(bound_periodic.cell, winding)
            vertex_exact_isometries[state_row] = exact_isometry
            if not np.array_equal(
                state.points[state_row],
                _exact_periodic_point(exact_isometry, representative_point),
            ):
                raise ValueError(
                    "Periodic vertex coordinates must match their exact orbit action."
                )
            if not np.array_equal(
                state.metric[state_row],
                _exact_periodic_metric(exact_isometry, representative_metric),
            ):
                raise ValueError(
                    "Metric tensors must transform exactly over vertex orbits."
                )
            stabilizers = []
            for generator in np.flatnonzero(fixed_generators[target_row]):
                exponents = np.zeros((vertex_windings.shape[1],), dtype=np.int64)
                exponents[generator] = 1
                stabilizer = exact_periodic_element(bound_periodic.cell, exponents)
                stabilizers.append(stabilizer)
                if not np.array_equal(
                    state.points[state_row],
                    _exact_periodic_point(stabilizer, state.points[state_row]),
                ) or not np.array_equal(
                    state.metric[state_row],
                    _exact_periodic_metric(stabilizer, state.metric[state_row]),
                ):
                    raise ValueError(
                        "Periodic vertex coordinates and metrics must respect exact stabilizers."
                    )
            vertex_stabilizers[state_row] = tuple(stabilizers)
    target_cell_ids = np.asarray(cell_map.lifted_entity_ids, dtype=np.int64)
    state_cell_ids = _cell_identifiers(state, source, final)
    cell_order = np.argsort(target_cell_ids, kind="stable")
    cell_positions = np.searchsorted(target_cell_ids[cell_order], state_cell_ids)
    if np.any(cell_positions >= target_cell_ids.size) or not np.array_equal(
        target_cell_ids[cell_order[np.minimum(cell_positions, target_cell_ids.size - 1)]],
        state_cell_ids,
    ):
        raise ValueError("Periodic local topology lost a lifted state cell.")
    cell_target_rows = cell_order[cell_positions]
    cell_orbits = np.asarray(cell_map.lifted_to_quotient, dtype=np.int64)[
        cell_target_rows
    ]

    edge_state_by_target = np.full((target_edge_vertices.shape[0],), -1, dtype=np.int64)
    edge_state_by_target[edge_target_rows] = np.arange(
        topology.edges.shape[0], dtype=np.int64
    )
    edge_representatives = np.asarray(edge_map.representative_lifted_rows, dtype=np.int64)
    edge_windings = np.asarray(edge_map.winding_shifts, dtype=np.int64)
    edge_lengths = topology.lengths.copy()
    # Evaluate each periodic scalar on its exact canonical lift; coordinate and
    # tensor orbit consistency is independently checked above.
    for orbit in np.unique(edge_orbits):
        rows = np.flatnonzero(edge_orbits == orbit)
        if (
            np.unique(topology.edge_class[rows]).size != 1
            or np.unique(topology.edge_protected[rows]).size != 1
        ):
            raise ValueError("Periodic edge controls must agree over each orbit.")
        representative_target_row = int(edge_representatives[orbit])
        representative_row = int(edge_state_by_target[representative_target_row])
        if representative_row < 0:
            raise ValueError("Periodic edge orbit has no state representative.")
        representative_vertices = edge_canonical_vertices[representative_row]
        representative_winding = edge_windings[representative_target_row]
        for edge_row in rows:
            target_row = int(edge_target_rows[edge_row])
            isometry = exact_periodic_element(
                bound_periodic.cell,
                edge_windings[target_row] - representative_winding,
            )
            member_vertices = edge_canonical_vertices[edge_row]
            # Only root-relative lifts are bit-consistent on non-dyadic
            # lattices, so the edge action is checked through the vertex root:
            # the composed exact element must reproduce each member corner.
            for representative_vertex, member_vertex in zip(
                representative_vertices, member_vertices, strict=True
            ):
                root_row = int(representative_state_rows[member_vertex])
                composite = _multiply(
                    isometry, vertex_exact_isometries[representative_vertex]
                )
                if (
                    root_row != int(representative_state_rows[representative_vertex])
                    or not np.array_equal(
                        state.points[member_vertex],
                        _exact_periodic_point(composite, state.points[root_row]),
                    )
                    or not np.array_equal(
                        state.metric[member_vertex],
                        _exact_periodic_metric(composite, state.metric[root_row]),
                    )
                ):
                    raise ValueError(
                        "Periodic edge geometry must respect its exact orbit action."
                    )
        edge_lengths[rows] = topology.lengths[representative_row]

    cell_state_by_target = np.full((target_cell_ids.size,), -1, dtype=np.int64)
    cell_state_by_target[cell_target_rows] = np.arange(
        state_cell_ids.size, dtype=np.int64
    )
    cell_representatives = np.asarray(cell_map.representative_lifted_rows, dtype=np.int64)
    cell_quality = _quality(edge_lengths[topology.cell_edges])
    for orbit in np.unique(cell_orbits):
        rows = np.flatnonzero(cell_orbits == orbit)
        if (
            np.unique(state.cell_block[rows]).size != 1
            or np.unique(state.cell_region[rows]).size != 1
        ):
            raise ValueError("Periodic cell controls must agree over each orbit.")
        representative_row = int(cell_state_by_target[cell_representatives[orbit]])
        if representative_row < 0:
            raise ValueError("Periodic cell orbit has no state representative.")
        cell_quality[rows] = cell_quality[representative_row]
    active_vertices = np.flatnonzero(vertex_orbits >= 0)
    for orbit in np.unique(vertex_orbits[active_vertices]):
        rows = active_vertices[vertex_orbits[active_vertices] == orbit]
        if np.unique(state.fixed[rows]).size != 1:
            raise ValueError("Periodic fixed-vertex controls must agree over each orbit.")
    return _PeriodicContext(
        edge_orbits,
        np.asarray(edge_map.quotient_entity_ids, dtype=np.int64),
        edge_canonical_vertices,
        edge_lengths,
        vertex_orbits,
        np.asarray(vertex_map.quotient_entity_ids, dtype=np.int64),
        representative_state_rows,
        tuple(vertex_exact_isometries),
        tuple(vertex_stabilizers),
        cell_orbits,
        np.asarray(cell_map.quotient_entity_ids, dtype=np.int64),
        cell_quality,
        bound.mesh_id,
        int(bound_periodic.allocator_next_ids[0]),
    )


def _evidence(
    topology: _Topology,
    counts: _Counts,
    passes: int,
    stalled: bool,
    topology_operations: bool,
    relocation: bool,
    mode: PredicateMode,
    /,
) -> LocalMetricEvidence:
    lengths = _measured(topology)
    inside = (lengths >= _LOWER) & (lengths <= _UPPER)
    converged = bool(np.all(inside))
    return LocalMetricEvidence(
        passes=passes,
        splits=counts.splits,
        collapses=counts.collapses,
        flips=counts.flips,
        relocations=counts.relocations,
        rejected_uncertain=counts.rejected_uncertain,
        rejected_invalid=counts.rejected_invalid,
        converged=converged,
        stalled=stalled and not converged,
        measured_edges=lengths.size,
        out_of_range_edges=int(np.count_nonzero(~inside)),
        minimum_metric_length=float(np.min(lengths)),
        maximum_metric_length=float(np.max(lengths)),
        unit_fraction=float(np.count_nonzero(inside) / lengths.size),
        minimum_metric_quality=float(np.min(topology.quality)),
        topology_operations=topology_operations,
        relocation=relocation,
        predicate_mode=mode,
    )


def execute_local_metric_adaptation(
    mesh: CellMesh,
    metric: np.ndarray,
    /,
    *,
    cell_classes: np.ndarray,
    edge_classes: np.ndarray,
    protected_edges: np.ndarray,
    fixed_vertices: np.ndarray,
    predicate_mode: PredicateMode,
    topology_operations: bool,
    relocation: bool,
) -> LocalMetricOutcome:
    """Adapt a planar triangle mesh until it meets the unit-mesh criterion or stalls.

    No candidate or pass quota truncates the finite staged work. Every accepted
    split/collapse/flip phase exhausts its current candidates before proceeding;
    a full pass that cannot change the state terminates as a reported stall.
    Periodic sources group split, collapse, flip, and relocation proposals by
    quotient orbit, execute complete lifted groups, and validate the exact target
    stencil and lineage before returning an immutable candidate edit.
    """
    values = _validate_mesh(mesh, metric)
    _validate_controls(predicate_mode, topology_operations, relocation)
    vertex_count = mesh.coordinates.shape[0]
    edge_count = mesh.entity_set(1).entity_ids.shape[0]
    cell_count = mesh.entity_set(2).entity_ids.shape[0]
    state, source = _initial_state(
        mesh,
        values,
        _integer_array(cell_classes, cell_count, "cell_classes"),
        _integer_array(edge_classes, edge_count, "edge_classes"),
        _boolean_array(protected_edges, edge_count, "protected_edges"),
        _boolean_array(fixed_vertices, vertex_count, "fixed_vertices"),
        predicate_mode,
    )
    topology = _topology(state, predicate_mode)
    _validate_classification(state, topology)
    topology, _ = _periodic_topology(mesh, source, state, topology, predicate_mode)
    counts = _Counts()
    passes = 0
    stalled = False
    periodic_topology_pending = mesh.periodic_topology is not None and topology_operations
    while not _unit(topology) or periodic_topology_pending:
        periodic_topology_pending = False
        passes += 1
        state, counts, applied = _run_pass(
            state,
            counts,
            predicate_mode,
            topology_operations,
            relocation,
            mesh,
            source,
        )
        topology = _topology(state, predicate_mode)
        topology, _ = _periodic_topology(mesh, source, state, topology, predicate_mode)
        if applied == 0:
            stalled = True
            break
    edit, target_metric = _assemble(state, source, predicate_mode)
    stage_periodic_topology_edit(mesh, edit)
    evidence = _evidence(
        topology,
        counts,
        passes,
        stalled,
        topology_operations,
        relocation,
        predicate_mode,
    )
    return LocalMetricOutcome(edit, target_metric, evidence)


__all__ = [
    "LocalMetricEvidence",
    "LocalMetricOutcome",
    "execute_local_metric_adaptation",
]
