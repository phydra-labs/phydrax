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
from ._lineage import EntityLineageKind
from ._metric import interpolate_mesh_metric, metric_edge_lengths
from ._topology_edit import entity_keys, EntityRelations, key_rows, SimplexTopologyEdit


_LOWER = float(1.0 / np.sqrt(2.0))
_UPPER = float(np.sqrt(2.0))
# Longest metric edge a collapse may create. Edges in (sqrt(2), 2] are split by
# the next pass into halves inside the unit range; a strict sqrt(2) bound would
# lock anisotropic coarsening (a removed row always lengthens diagonals).
_COLLAPSE_LIMIT = 2.0
# Sub-rounds of one operation phase per pass and greedy selection rounds per
# sub-round; both only bound work, remaining candidates return next round.
_PHASE_ROUNDS = 8
_SELECTION_ROUNDS = 64
_RELOCATION_STEPS = np.asarray((1.0, 0.5, 0.25), dtype=np.float64)
# Minimum metric-quality gain of a flip or relocation; excludes rounding cycles.
_IMPROVEMENT = 1.0e-6
# Candidate-pair padding of the source point location; exact signs decide.
_LOCATION_TOLERANCE = 1.0e-9
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

    edit: SimplexTopologyEdit
    metric: np.ndarray
    evidence: LocalMetricEvidence


class _State(NamedTuple):
    """Working mesh; vertex rows are never reused, removed rows stay dead."""

    points: np.ndarray
    metric: np.ndarray
    vertex_ids: np.ndarray
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
    cell_ids: np.ndarray
    edge_keys: np.ndarray
    block_count: int


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
    for _ in range(_SELECTION_ROUNDS):
        live = undecided[pair_ops]
        if not np.any(live):
            break
        best = np.full((item_count,), count, dtype=np.int64)
        np.minimum.at(best, pair_items[live], rank[pair_ops[live]])
        losing = np.zeros((count,), dtype=np.bool_)
        losing[pair_ops[live & (best[pair_items] != rank[pair_ops])]] = True
        winners = undecided & ~losing
        selected |= winners
        taken = np.zeros((item_count,), dtype=np.bool_)
        taken[pair_items[winners[pair_ops]]] = True
        blocked = np.zeros((count,), dtype=np.bool_)
        blocked[pair_ops[taken[pair_items]]] = True
        undecided &= ~blocked
    return selected


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
    return values


def _validate_controls(
    predicate_mode: Any, maximum_passes: Any, topology_operations: Any, relocation: Any, /
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
    if not isinstance(maximum_passes, int) or isinstance(maximum_passes, bool):
        raise TypeError("maximum_passes must be an integer.")
    if maximum_passes < 1:
        raise ValueError("maximum_passes must be at least one.")
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
    state = _State(
        points=points,
        metric=metric,
        vertex_ids=vertex_ids,
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
    source = _Source(
        points=points,
        vertex_ids=vertex_ids,
        cells=cells,
        cell_ids=cell_ids,
        edge_keys=entity_keys(mesh, 1),
        block_count=len(mesh.blocks),
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


def _append_vertices(state: _State, points: np.ndarray, metric: np.ndarray, /) -> _State:
    count = points.shape[0]
    return state._replace(
        points=np.concatenate((state.points, points)),
        metric=np.concatenate((state.metric, metric)),
        vertex_ids=np.concatenate(
            (
                state.vertex_ids,
                state.next_vertex_id + np.arange(count, dtype=np.int64),
            )
        ),
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
    nearly_equal = np.abs(ratio - 1.0) <= 1.0e-6
    safe = np.where(nearly_equal, 2.0, ratio)
    return np.where(nearly_equal, 0.5, np.log(0.5 * (1.0 + safe)) / np.log(safe))


def _cavity_sides(topology: _Topology, edges: np.ndarray, /) -> Any:
    """``(op, cell, local)`` per cell on candidate edges; local edge ``i`` is CCW."""
    sides = topology.edge_cells[edges]
    op, side = np.nonzero(sides >= 0)
    cell = sides[op, side]
    local = np.argmax(topology.cell_edges[cell] == edges[op][:, None], axis=1)
    return op, cell, local


def _split_round(
    state: _State, topology: _Topology, counts: _Counts, mode: PredicateMode, /
) -> tuple[_State, _Counts, int]:
    candidates = np.flatnonzero((topology.lengths > _UPPER) & ~topology.edge_protected)
    if candidates.size == 0:
        return state, counts, 0
    first, second = topology.edges[candidates].T
    t = _metric_midpoint(state, first, second)
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
    rank = _ranks(
        -topology.lengths[candidates[valid]], ids[first[valid]], ids[second[valid]]
    )
    local_of = np.full((candidates.size,), -1, dtype=np.int64)
    local_of[valid] = np.arange(valid.size)
    inside = local_of[op] >= 0
    chosen = valid[
        _disjoint_selection(
            rank, local_of[op[inside]], cell[inside], state.cells.shape[0]
        )
    ]
    chosen = chosen[np.lexsort((ids[second[chosen]], ids[first[chosen]]))]
    new_rows = np.full((candidates.size,), -1, dtype=np.int64)
    new_rows[chosen] = state.points.shape[0] + np.arange(chosen.size)
    weights = np.stack((1.0 - t[chosen], t[chosen]), axis=1)
    samples = np.stack((state.metric[first[chosen]], state.metric[second[chosen]]), 1)
    grown = _append_vertices(
        state, middle[chosen], _interpolate_metrics(samples, weights)
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
    geometric = np.maximum(
        orientation, np.where(link & (longest <= _COLLAPSE_LIMIT), _OK, _INVALID)
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
    state: _State, topology: _Topology, counts: _Counts, mode: PredicateMode, /
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
    ids = state.vertex_ids
    removed = found.removed[valid]
    kept = found.kept[valid]
    rank = _ranks(
        topology.lengths[found.edge[valid]], found.longest[valid], ids[removed], ids[kept]
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
    state: _State, topology: _Topology, counts: _Counts, mode: PredicateMode, /
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
    improving = after > before + _IMPROVEMENT
    counts = _tally(counts, status[improving])
    valid = np.flatnonzero(improving & (status == _OK))
    if valid.size == 0:
        return state, counts, 0
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


def _relocation_round(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    done: np.ndarray,
    mode: PredicateMode,
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
    steps = _RELOCATION_STEPS.size
    owner = np.repeat(np.arange(vertices.size), steps)
    vertex = vertices[owner]
    target = state.points[vertex] + (
        np.tile(_RELOCATION_STEPS, vertices.size)[:, None] * displacement[owner]
    )
    metric = _proposal_metric(state, topology, vertex, target)
    status, after = _proposal_checks(state, topology, vertex, target, metric, mode)
    ball, position = _expand(topology.ball_offsets, vertices)
    before = _group_minimum(
        topology.quality[topology.ball_cells[position]], ball, vertices.size, np.inf
    )
    improving = (after > before[owner] + _IMPROVEMENT).reshape((-1, steps))
    grid = status.reshape((-1, steps))
    accepted = improving & (grid == _OK)
    success = np.any(accepted, axis=1)
    rejected = np.where(
        np.any(improving & (grid >= _INVALID), axis=1),
        _INVALID,
        np.where(np.any(improving & (grid == _UNCERTAIN), axis=1), _UNCERTAIN, _OK),
    )
    counts = _tally(counts, rejected[~success])
    winners = np.flatnonzero(success)
    if winners.size == 0:
        return state, counts, 0, done
    proposal = winners * steps + np.argmax(accepted[winners], axis=1)
    rank = _ranks(-badness[winners], state.vertex_ids[vertices[winners]])
    local = np.full((vertices.size,), -1, dtype=np.int64)
    local[winners] = np.arange(winners.size)
    member = local[ball] >= 0
    selected = _disjoint_selection(
        rank,
        local[ball[member]],
        topology.ball_cells[position[member]],
        state.cells.shape[0],
    )
    moved = vertices[winners[selected]]
    chosen = proposal[selected]
    points = state.points.copy()
    points[moved] = target[chosen]
    tensors = state.metric.copy()
    tensors[moved] = metric[chosen]
    flags = state.moved.copy()
    flags[moved] = True
    finished = done.copy()
    finished[moved] = True
    updated = _raise_touching(
        state._replace(points=points, metric=tensors, moved=flags), moved, _RELOCATED
    )
    return (
        updated,
        counts._replace(relocations=counts.relocations + moved.size),
        moved.size,
        finished,
    )


# ------------------------------------------------------------------ passes


def _run_phase(state: Any, counts: Any, phase: Any, mode: Any, /) -> Any:
    applied = 0
    for _ in range(_PHASE_ROUNDS):
        state, counts, count = phase(state, _topology(state, mode), counts, mode)
        applied += count
        if count == 0:
            break
    return state, counts, applied


def _run_pass(
    state: _State,
    counts: _Counts,
    mode: PredicateMode,
    topology_operations: bool,
    relocation: bool,
    /,
) -> tuple[_State, _Counts, int]:
    applied = 0
    if topology_operations:
        for phase in (_split_round, _collapse_round, _flip_round):
            state, counts, count = _run_phase(state, counts, phase, mode)
            applied += count
    if relocation:
        done = np.zeros((state.points.shape[0],), dtype=np.bool_)
        for _ in range(_PHASE_ROUNDS):
            state, counts, count, done = _relocation_round(
                state, _topology(state, mode), counts, done, mode
            )
            applied += count
            if count == 0:
                break
    return state, counts, applied


# ------------------------------------------------------------------ assembly


def _locate(source: _Source, queries: np.ndarray, mode: PredicateMode, /) -> Any:
    """Containing source cell (lowest ID among certified hosts) and P1 weights.

    Points certified inside or on a source triangle take the lowest-ID such cell;
    points within rounding of the domain boundary take the least-violated
    candidate, whose clipped weights are the nearest exact P1 evaluation.
    """
    triangles = source.points[source.cells]
    cells_tree = prepare_bvh(
        np.min(triangles, axis=1), np.max(triangles, axis=1), dtype=jnp.float64
    )
    points_tree = prepare_bvh(queries, queries, dtype=jnp.float64)
    cell, query = bvh_overlap_pairs_host(
        cells_tree,
        points_tree,
        include_touching=True,
        relative_tolerance=_LOCATION_TOLERANCE,
    )
    if np.unique(query).size != queries.shape[0]:
        raise ValueError("A target vertex lies outside the source domain.")
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
    weights = _barycentric(corners, point)
    violation = np.where(grade == 0, 0.0, -np.min(weights, axis=1))
    order = np.lexsort((source.cell_ids[cell], violation, grade, query))
    first = order[np.searchsorted(query[order], np.arange(queries.shape[0]))]
    exact_zero = certain[first] & (signs[first] == PredicateSign.ZERO)
    chosen = _normalized(np.where(exact_zero, 0.0, weights[first]))
    return cell[first], chosen


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


def _stencil(state: _State, source: _Source, order: Any, mode: PredicateMode, /) -> Any:
    count = order.size
    sources = np.full((count, 3), -1, dtype=np.int64)
    weights = np.zeros((count, 3), dtype=np.float64)
    identity = (state.source_rows[order] >= 0) & ~state.moved[order]
    sources[identity, 0] = source.vertex_ids[state.source_rows[order[identity]]]
    weights[identity, 0] = 1.0
    located = np.flatnonzero(~identity)
    if located.size:
        cell, barycentric = _locate(source, state.points[order[located]], mode)
        sources[located] = source.vertex_ids[source.cells[cell]]
        weights[located] = barycentric
    valid = weights > 0.0
    return np.where(valid, sources, -1), np.where(valid, weights, 0.0), valid


def _surviving(state: _State, /) -> np.ndarray:
    """Final surviving vertex row of every vertex row (pointer jumping)."""
    final = np.where(
        state.alive, np.arange(state.points.shape[0], dtype=np.int64), state.collapse_to
    )
    for _ in range(64):
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


def _vertex_relations(
    state: Any, order: Any, identifiers: Any, final: Any, stencil: Any, /
) -> Any:
    ids = state.vertex_ids
    rows = np.flatnonzero((state.source_rows >= 0) & state.alive & state.moved)
    collapsed = np.flatnonzero((state.source_rows >= 0) & ~state.alive)
    survivor = _surviving(state)[collapsed]
    created = np.flatnonzero(state.source_rows[order] < 0)
    stencil_sources, _, valid = stencil
    owner, column = np.nonzero(valid[created])
    sources = np.concatenate(
        (ids[rows], ids[collapsed], stencil_sources[created[owner], column])
    )
    targets = np.concatenate((ids[rows], final[survivor], identifiers[created[owner]]))
    kinds = np.concatenate(
        (
            np.full(rows.shape, EntityLineageKind.RELOCATED, dtype=np.int32),
            np.full(collapsed.shape, EntityLineageKind.COLLAPSED_INTO, dtype=np.int32),
            np.full(owner.shape, EntityLineageKind.SPLIT_FROM, dtype=np.int32),
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
        _vertex_relations(state, order, identifiers, final, stencil),
        _edge_relations(state, source, final),
        _cell_relations(state, source, cell_identifiers),
    )
    edit = SimplexTopologyEdit(
        coordinates=state.points[order],
        vertex_global_ids=identifiers,
        block_cells=block_cells,
        block_cell_ids=block_ids,
        stencil_sources=stencil[0],
        stencil_weights=stencil[1],
        stencil_valid=stencil[2],
        relations=relations,
    )
    return edit, state.metric[order]


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
    maximum_passes: int,
    topology_operations: bool,
    relocation: bool,
) -> LocalMetricOutcome:
    """Adapt a planar triangle mesh toward the unit mesh of ``metric``.

    Passes repeat until every unprotected edge has metric length in
    ``[1/sqrt(2), sqrt(2)]`` (converged), a pass applies nothing (stalled), or
    ``maximum_passes`` is reached; the returned mesh is valid in every case.
    """
    values = _validate_mesh(mesh, metric)
    _validate_controls(predicate_mode, maximum_passes, topology_operations, relocation)
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
    counts = _Counts()
    passes = 0
    stalled = False
    while passes < maximum_passes and not _unit(topology):
        passes += 1
        state, counts, applied = _run_pass(
            state, counts, predicate_mode, topology_operations, relocation
        )
        topology = _topology(state, predicate_mode)
        if applied == 0:
            stalled = True
            break
    edit, target_metric = _assemble(state, source, predicate_mode)
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
