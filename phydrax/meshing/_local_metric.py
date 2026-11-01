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

from .._bvh import _host_boxes_overlap, bvh_overlap_pairs_host, prepare_bvh
from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import orient2d, PredicateMode, PredicateSign
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._periodic_exact import (
    _multiply,
    apply_periodic,
    exact_periodic_element,
    identification_orders,
    reduce_periodic_exponents,
)
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
# Quantum of the |ln L| edge-length deviation scores that order relocations
# which keep their star's minimum quality (see _deviation_improves).
_DEVIATION_QUANTUM = 2.0**-20
# Minimum metric-quality gain of a flip or relocation; excludes rounding cycles.
_IMPROVEMENT = 1.0e-6
# Candidate-pair padding of the source point location; exact signs decide.
_LOCATION_TOLERANCE = 1.0e-9
# Box tests below which point location skips the query tree (identical pairs).
_DIRECT_LOCATION_TESTS = 1 << 16
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
    protected); ``converged`` certifies the unit-mesh criterion on them. A stalled
    adaptation records, per out-of-range edge (one per quotient edge orbit on a
    periodic mesh), its metric length and why no operation could change it,
    e.g. ``collapse=hysteresis+inadmissible;flip=invalid``.
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
    stalled_edges: tuple[tuple[float, str], ...] = eqx.field(static=True)

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
        stalled_edges: tuple[tuple[float, str], ...] = (),
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
        records = tuple((float(length), str(reason)) for length, reason in stalled_edges)
        if records and not stalled:
            raise ValueError("Stalled-edge evidence requires a stalled adaptation.")
        if any(
            not np.isfinite(length) or length <= 0.0 or not reason
            for length, reason in records
        ):
            raise ValueError("Stalled edges require finite positive lengths and reasons.")
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
        self.stalled_edges = records
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "local-metric-evidence",
                **counts,
                **flags,
                **measures,
                "predicate_mode": str(predicate_mode),
                **({"stalled_edges": records} if records else {}),
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
    # Periodic producer-owned vertex orbit identity: one ``(kind, root id)`` key
    # per row (kind 0 keeps a source vertex orbit, kind 1 names the root vertex of
    # an orbit-locked split) and the absolute deck exponents of every member.
    orbit_keys: np.ndarray | None = None
    orbit_windings: np.ndarray | None = None
    # Collapsed chart points (e.g. sphere poles): one group per point, ``-1`` for
    # regular rows. Every copy in a group is one physical vertex; copies are
    # fixed, and their protected incident edges are chart cuts (see ``_cuts``).
    pole: np.ndarray | None = None


class _Source(NamedTuple):
    points: np.ndarray
    exact_points: tuple[tuple[Fraction, ...], ...]
    vertex_ids: np.ndarray
    cells: np.ndarray
    cell_tree: Any
    # Binary64 bounding boxes of the location cells (the ``cell_tree`` items).
    cell_lower: np.ndarray
    cell_upper: np.ndarray
    cell_ids: np.ndarray
    edge_keys: np.ndarray
    block_count: int
    families: tuple[tuple[str, str], ...]
    periodic_cell: Any = None
    # Source location cells: the source cells, then one notch per regular vertex
    # between two cuts of a collapsed chart point (``_notches``). ``None`` means
    # the source cells alone. ``cell_tree`` bounds the location cells.
    locator_cells: np.ndarray | None = None
    locator_ids: np.ndarray | None = None
    # Optional exact metric field: ``(points, source rows (n, 3), weights (n, 3))``
    # to the metric at chart points located in source cells (``_field_metric``).
    metric_field: Any = None


def _locator(source: _Source, /) -> tuple[np.ndarray, np.ndarray]:
    """Source location cells and their tie-break IDs."""
    if source.locator_cells is None or source.locator_ids is None:
        return source.cells, source.cell_ids
    return source.locator_cells, source.locator_ids


def _field_metric(
    source: _Source, points: np.ndarray, fallback: np.ndarray, mode: PredicateMode, /
) -> np.ndarray:
    """Metric of generated chart points from the source's exact metric field.

    Without a field, or for points that round outside the source cells, the
    interpolated ``fallback`` stands.
    """
    if source.metric_field is None or points.shape[0] == 0:
        return fallback
    cells, weights, located = _locate(source, points, mode)
    if not np.any(located):
        return fallback
    result = np.array(fallback, dtype=np.float64, copy=True)
    result[located] = source.metric_field(
        points[located],
        _locator(source)[0][cells[located]],
        weights[located],
    )
    return result


def _notches(
    points: np.ndarray,
    edges: np.ndarray,
    cut: np.ndarray,
    groups: np.ndarray,
    mode: PredicateMode,
    /,
) -> np.ndarray:
    """Chart gaps at collapsed chart points, as positively oriented triangles.

    A regular vertex ``r`` between two cuts ``(a, r)`` and ``(r, b)`` to copies
    ``a`` and ``b`` of one collapsed point bounds the chart gap ``(a, r, b)``.
    The gap is a degenerate physical sliver (its copies are one point), so edits
    of the free regular vertices may place chart points there; locating them in
    the gap gives convex source rows over ``a``, ``r``, ``b``, which are the
    collapsed point and ``r`` on the physical surface.
    """
    cut_edges = edges[cut]
    first_copy = groups[cut_edges[:, 0]] >= 0
    copy = np.where(first_copy, cut_edges[:, 0], cut_edges[:, 1])
    regular = np.where(first_copy, cut_edges[:, 1], cut_edges[:, 0])
    order = np.argsort(regular, kind="stable")
    regular, copy = regular[order], copy[order]
    rows, starts, sizes = np.unique(regular, return_index=True, return_counts=True)
    pair = sizes == 2
    first = copy[starts[pair]]
    second = copy[starts[pair] + 1]
    middle = rows[pair]
    same = groups[first] == groups[second]
    first, middle, second = first[same], middle[same], second[same]
    forward = _orientation(points[first], points[middle], points[second], mode) == _OK
    backward = _orientation(points[second], points[middle], points[first], mode) == _OK
    triangles = np.concatenate(
        (
            np.stack((first, middle, second), axis=1)[forward],
            np.stack((second, middle, first), axis=1)[backward & ~forward],
        )
    )
    return triangles.astype(np.int64).reshape((-1, 3))


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
    # Absolute deck exponents of each state edge and the state edge row that
    # represents each edge orbit; orbit-locked splits are anchored on it.
    edge_member_windings: np.ndarray
    edge_representative_rows: np.ndarray
    # Absolute deck exponents of each live state vertex (zero for dead rows).
    vertex_windings: np.ndarray


def _orbit_groups(keys: np.ndarray, rows: np.ndarray, /) -> list[np.ndarray]:
    """``rows`` grouped by ``keys[rows]``: ascending keys, input order within a group."""
    if rows.size == 0:
        return []
    ordered = rows[np.argsort(keys[rows], kind="stable")]
    starts = np.flatnonzero(np.diff(keys[ordered]) != 0) + 1
    return np.split(ordered, starts)


def _orbit_disagreement(
    orbits: np.ndarray, values: np.ndarray, size: int, /
) -> np.ndarray:
    """Per orbit ``0..size-1``, whether its members' ``values`` are not all equal."""
    reference = np.zeros((size,), dtype=values.dtype)
    reference[orbits] = values
    result = np.zeros((size,), dtype=np.bool_)
    np.logical_or.at(result, orbits, values != reference[orbits])
    return result


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
    """Metric shape quality ``4 sqrt(3) area / sum(l^2)`` from edge lengths (Heron).

    The lengths are sorted first, so a cell's quality does not depend on the
    order in which its edges are listed.
    """
    ordered = np.sort(lengths, axis=-1)
    a, b, c = ordered[..., 0], ordered[..., 1], ordered[..., 2]
    product = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c)
    return np.sqrt(3.0) * np.sqrt(np.maximum(product, 0.0)) / (a * a + b * b + c * c)


def _flip_admissible(
    before: np.ndarray,
    after: np.ndarray,
    new_diagonal: np.ndarray,
    old_codes: np.ndarray,
    /,
) -> np.ndarray:
    """Admission of flips from their ``(k, 2)`` cell qualities and diagonals.

    A flip is admitted when it raises the minimum quality of its two cells by
    ``_IMPROVEMENT`` and its new diagonal lies in the unit band, or when it
    strictly lowers the quantized ``|ln L|`` deviation of the diagonal (the
    only edge a flip changes; ``old_codes`` holds the removed diagonal's
    ``_deviation_codes``) without lowering the cells' quality pair, compared
    minimum first, then maximum.

    Termination: order meshes by their cell-quality multiset (ascending sorted
    vectors, lexicographic) and then by their edge-deviation multiset (worst
    first, see ``_deviation_improves``); both orders are kept when the same
    cells or edges are added to both sides. A quality flip strictly raises the
    first key; a deviation flip keeps or raises it and strictly lowers the
    second. Flips create no vertices, so the triangulations of one vertex set
    are finitely many and no flip sequence repeats one.
    """
    low_before = np.min(before, axis=1)
    low_after = np.min(after, axis=1)
    quality = (low_after > low_before + _IMPROVEMENT) & (
        (new_diagonal >= _LOWER) & (new_diagonal <= _UPPER)
    )
    kept = (low_after > low_before) | (
        (low_after == low_before) & (np.max(after, axis=1) >= np.max(before, axis=1))
    )
    return quality | (kept & (_deviation_codes(new_diagonal) < old_codes))


def _deviation_improves(before: np.ndarray, after: np.ndarray, /) -> bool:
    """Strict improvement of the worst-first quantized metric-length deviations.

    Each affected edge scores ``floor(|ln L| / _DEVIATION_QUANTUM)``. Sorted
    worst first, the first unequal position decides (the smaller deviation is
    better) and a vector that ends first is better: at the largest score whose
    multiplicity differs, the multiset holding fewer copies is better. Adding
    the same unchanged edges to both sides keeps the order, so an edit that
    improves its affected edges improves the whole mesh's multiset.

    Termination: a relocation changes no topology. It is accepted when its
    star's minimum quality rises by ``_IMPROVEMENT`` or, keeping that minimum,
    when its edges strictly improve in this order. Quality never falls under
    either rule, so quality steps are finitely many between topology edits; the
    deviation steps between them strictly decrease the mesh's count vector over
    the finite quantized alphabet, and count vectors under the lexicographic
    order (worst score most significant) are well-ordered, so they are finite
    too.
    """
    worse = np.sort(_deviation_codes(np.asarray(before, dtype=np.float64)))[::-1]
    better = np.sort(_deviation_codes(np.asarray(after, dtype=np.float64)))[::-1]
    for previous, candidate in zip(worse.tolist(), better.tolist(), strict=False):
        if previous != candidate:
            return candidate < previous
    return better.size < worse.size


def _deviation_codes(lengths: np.ndarray, /) -> np.ndarray:
    return np.floor(np.abs(np.log(lengths)) / _DEVIATION_QUANTUM).astype(np.int64)


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
    """Floating barycentric weights of ``points`` in ``(k, 3, 2)`` triangles.

    A triangle whose binary64 area is zero (a sliver that only exact predicates
    orient) has no floating barycentric coordinates; its weights are NaN.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    def cross(origin: Any, first: Any, second: Any) -> Any:
        u = first - origin
        v = second - origin
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    total = cross(a, b, c)
    weights = np.stack(
        (cross(points, b, c), cross(points, c, a), cross(points, a, b)), axis=1
    )
    return np.divide(
        weights,
        total[:, None],
        out=np.full_like(weights, np.nan),
        where=total[:, None] != 0.0,
    )


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


def _validate_classification(
    state: _State, topology: _Topology, periodic_context: _PeriodicContext | None, /
) -> None:
    boundary = topology.edge_cells[:, 1] < 0
    second = np.maximum(topology.edge_cells[:, 1], 0)
    interface = ~boundary & (
        state.cell_region[topology.edge_cells[:, 0]] != state.cell_region[second]
    )
    if periodic_context is not None:
        # A fundamental-domain seam edge has one lifted cell per orbit member;
        # boundary and interface edges are decided by the incident cell orbits.
        incident = topology.edge_cells >= 0
        edge_rows = np.broadcast_to(
            np.arange(topology.edge_cells.shape[0])[:, None], incident.shape
        )[incident]
        cell_rows = topology.edge_cells[incident]
        orbit_count = periodic_context.edge_quotient_ids.size
        pairs = np.unique(
            np.stack(
                (
                    periodic_context.edge_orbits[edge_rows],
                    periodic_context.cell_orbits[cell_rows],
                ),
                axis=1,
            ),
            axis=0,
        )
        region_by_orbit = np.zeros(
            (int(np.max(periodic_context.cell_orbits)) + 1,), dtype=np.int64
        )
        region_by_orbit[periodic_context.cell_orbits] = state.cell_region
        regions = region_by_orbit[pairs[:, 1]]
        low = _group_minimum(regions, pairs[:, 0], orbit_count, np.iinfo(np.int64).max)
        high = _group_maximum(regions, pairs[:, 0], orbit_count, -1)
        cells_per_orbit = np.bincount(pairs[:, 0], minlength=orbit_count)
        boundary = (cells_per_orbit == 1)[periodic_context.edge_orbits]
        interface = (low != high)[periodic_context.edge_orbits]
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
    pole_groups: np.ndarray | None = None,
    metric_field: Any = None,
    exact_source_points: tuple[tuple[Fraction, ...], ...] | None = None,
) -> tuple[_State, _Source]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if exact_source_points is None:
        exact_source_points = tuple(
            tuple(Fraction(float(value)) for value in point) for point in points
        )
    elif (
        len(exact_source_points) != points.shape[0]
        or any(len(point) != points.shape[1] for point in exact_source_points)
        or any(
            not isinstance(value, Fraction)
            for point in exact_source_points
            for value in point
        )
    ):
        raise TypeError("Exact source points must be a mesh-aligned rational bank.")
    if not np.array_equal(np.asarray(exact_source_points, dtype=np.float64), points):
        raise ValueError("Source coordinates must canonically round their exact points.")
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
        pole=(
            None
            if pole_groups is None or not np.any(pole_groups >= 0)
            else pole_groups.copy()
        ),
    )
    periodic = mesh.periodic_topology
    if periodic is not None:
        # Source vertices keep their source orbit: the key names the orbit root
        # and the winding is the root-relative deck exponent, with fixed
        # generators zeroed exactly as the stencil-derived identity does.
        lifted_ids = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
        representatives = np.asarray(periodic.vertex_representatives, dtype=np.int64)
        orders = identification_orders(periodic.cell)
        shifts = reduce_periodic_exponents(
            np.asarray(periodic.vertex_shifts, dtype=np.int64), orders
        )
        shifts[np.asarray(periodic.vertex_fixed_generators, dtype=np.bool_)] = 0
        lifted_order = np.argsort(lifted_ids, kind="stable")
        positions = np.searchsorted(lifted_ids[lifted_order], vertex_ids)
        if np.any(positions >= lifted_ids.size) or not np.array_equal(
            lifted_ids[lifted_order[np.minimum(positions, lifted_ids.size - 1)]],
            vertex_ids,
        ):
            raise ValueError("Periodic vertex orbits do not cover the source rows.")
        rows = lifted_order[positions]
        orbit_keys = np.zeros((count, 2), dtype=np.int64)
        orbit_keys[:, 1] = lifted_ids[representatives[rows]]
        state = state._replace(orbit_keys=orbit_keys, orbit_windings=shifts[rows])
    locator_cells = cells
    locator_ids = cell_ids
    if state.pole is not None:
        notches = _notches(
            points,
            edges,
            protected_edges & np.any(state.pole[edges] >= 0, axis=1),
            state.pole,
            mode,
        )
        locator_cells = np.concatenate((cells, notches))
        locator_ids = np.concatenate(
            (cell_ids, int(np.max(cell_ids)) + 1 + np.arange(notches.shape[0]))
        )
    triangles = points[locator_cells]
    cell_lower = np.min(triangles, axis=1)
    cell_upper = np.max(triangles, axis=1)
    source = _Source(
        points=points,
        exact_points=exact_source_points,
        vertex_ids=vertex_ids,
        cells=cells,
        cell_ids=cell_ids,
        cell_tree=prepare_bvh(cell_lower, cell_upper, dtype=jnp.float64),
        cell_lower=cell_lower,
        cell_upper=cell_upper,
        edge_keys=entity_keys(mesh, 1),
        block_count=len(mesh.blocks),
        families=tuple((block.name, block.cell_kind) for block in mesh.blocks),
        periodic_cell=None if periodic is None else periodic.cell,
        locator_cells=None if state.pole is None else locator_cells,
        locator_ids=None if state.pole is None else locator_ids,
        metric_field=metric_field,
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


def _cuts(state: _State, edges: np.ndarray, edge_protected: np.ndarray, /) -> np.ndarray:
    """Chart cuts: protected edges at a copy of a collapsed chart point.

    A collapsed chart point (a sphere pole) has one fixed copy per incident chart
    cell, so the edges at a copy bound the chart: they are protected (never
    split, collapsed, flipped, or measured). Each such edge is one physical edge
    with its twin at a neighboring copy, so a cut guards only its copy: the
    regular vertices around the point stay free, and the cuts follow their
    edits (the protection records move with collapses and are re-derived from
    the current cells whenever the topology is rebuilt).
    """
    if state.pole is None:
        return np.zeros(edge_protected.shape, dtype=np.bool_)
    return edge_protected & np.any(state.pole[edges] >= 0, axis=1)


def _pole_pairs(state: _State, edges: np.ndarray, /) -> tuple[np.ndarray, int]:
    """Sorted ``vertex * width + group`` codes of rows adjacent to a collapsed point."""
    pole = state.pole
    if pole is None:
        return np.zeros((0,), dtype=np.int64), 1
    width = int(np.max(pole)) + 1
    ends = np.concatenate((edges[:, 0], edges[:, 1]))
    others = np.concatenate((edges[:, 1], edges[:, 0]))
    near = pole[others] >= 0
    return np.unique(ends[near] * width + pole[others[near]]), width


def _adjacent_to_group(
    pairs: tuple[np.ndarray, int], rows: np.ndarray, groups: np.ndarray, /
) -> np.ndarray:
    """Whether each row is adjacent to some copy of its collapsed point group."""
    codes, width = pairs
    result = np.zeros(rows.shape, dtype=np.bool_)
    valid = groups >= 0
    if codes.size == 0 or not np.any(valid):
        return result
    query = rows[valid] * width + groups[valid]
    position = np.minimum(np.searchsorted(codes, query), codes.size - 1)
    result[valid] = codes[position] == query
    return result


def _vertex_classes(
    state: _State,
    edges: np.ndarray,
    edge_class: np.ndarray,
    edge_protected: np.ndarray,
    mode: PredicateMode,
    /,
) -> Any:
    count = state.points.shape[0]
    # Chart cuts classify and guard nothing at their regular end (``_cuts``).
    cut = _cuts(state, edges, edge_protected)
    featured = (edge_class > 0) & ~cut
    classified = edges[featured]
    ends = classified.reshape((-1,))
    others = classified[:, ::-1].reshape((-1,))
    labels = np.repeat(edge_class[featured], 2)
    incident = np.bincount(ends, minlength=count)
    low = _group_minimum(labels, ends, count, np.iinfo(np.int64).max)
    high = _group_maximum(labels, ends, count, -1)
    guarded = state.fixed.copy()
    guarded[edges[edge_protected & ~cut].reshape((-1,))] = True
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


def _band_excess(lengths: np.ndarray, /) -> np.ndarray:
    """Distance of ``ln L`` from the unit band ``[-ln sqrt(2), ln sqrt(2)]``."""
    return np.maximum(np.abs(np.log(lengths)) - np.log(_UPPER), 0.0)


def _stall_potential(topology: _Topology, /) -> tuple[int, int]:
    """Well-founded stall measure: out-of-band edges, then quantized band excess.

    Both components are natural numbers, so the lexicographic order on them
    admits no infinite strictly decreasing sequence (see
    ``execute_local_metric_adaptation``).
    """
    lengths = _measured(topology)
    outside = (lengths < _LOWER) | (lengths > _UPPER)
    excess = np.floor(_band_excess(lengths) / _DEVIATION_QUANTUM).astype(np.int64)
    return int(np.count_nonzero(outside)), int(np.sum(excess))


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
    orbit_keys: np.ndarray | None = None,
    orbit_windings: np.ndarray | None = None,
    /,
) -> _State:
    count = points.shape[0]
    if (state.orbit_keys is None) != (orbit_keys is None) or (
        orbit_keys is not None
        and (
            orbit_windings is None
            or state.orbit_windings is None
            or orbit_keys.shape != (count, 2)
            or orbit_windings.shape != (count, state.orbit_windings.shape[1])
        )
    ):
        raise ValueError("Periodic generated vertices require explicit orbit identity.")
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
        orbit_keys=(
            None
            if state.orbit_keys is None or orbit_keys is None
            else np.concatenate((state.orbit_keys, orbit_keys))
        ),
        orbit_windings=(
            None
            if state.orbit_windings is None or orbit_windings is None
            else np.concatenate((state.orbit_windings, orbit_windings))
        ),
        pole=(
            None
            if state.pole is None
            else np.concatenate((state.pole, np.full((count,), -1, dtype=np.int64)))
        ),
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


def _exact_source_segment_points(
    state: _State,
    source: _Source,
    first: np.ndarray,
    second: np.ndarray,
    parameters: np.ndarray,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    """Interpolate source-authoritative endpoints at binary64 segment parameters."""
    source_points = dict(zip(source.vertex_ids.tolist(), source.exact_points, strict=True))
    endpoints = {}
    for vertex in sorted(set(first.tolist()) | set(second.tolist())):
        stencil = state.source_stencil_exact[vertex]
        if not stencil:
            raise ValueError("A segment endpoint lost its exact source point.")
        endpoints[vertex] = tuple(
            sum(
                (weight * source_points[identifier][axis] for identifier, weight in stencil),
                Fraction(0),
            )
            for axis in range(state.points.shape[1])
        )
    return tuple(
        tuple(
            a + Fraction(float(parameter)) * (b - a)
            for a, b in zip(endpoints[int(start)], endpoints[int(end)], strict=True)
        )
        for start, end, parameter in zip(first, second, parameters, strict=True)
    )


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
    if periodic_context is not None:
        complete_orbits, _ = _complete_periodic_operations(
            candidates,
            periodic_context.edge_orbits,
            np.ones((candidates.size,), dtype=np.bool_),
            periodic_context.edge_quotient_ids.size,
        )
        position_of_edge = {
            int(edge): position for position, edge in enumerate(candidates.tolist())
        }
        canonical_first = periodic_context.edge_canonical_vertices[candidates, 0]
        canonical_parameter = np.where(first == canonical_first, t, 1.0 - t)
        candidate_orbits = periodic_context.edge_orbits[candidates]
        # An orbit is one quotient edge: its members' metric midpoints may differ
        # by rounding of their chart images, so every member takes the canonical
        # split parameter of the orbit's representative edge.
        for orbit in complete_orbits:
            root = position_of_edge.get(
                int(periodic_context.edge_representative_rows[int(orbit)])
            )
            if root is None:
                raise ValueError("A complete split orbit lost its representative edge.")
            member_positions = np.flatnonzero(candidate_orbits == orbit)
            common_parameter = float(canonical_parameter[root])
            t[member_positions] = np.where(
                first[member_positions] == canonical_first[member_positions],
                common_parameter,
                1.0 - common_parameter,
            )
    middle = state.points[first] + t[:, None] * (
        state.points[second] - state.points[first]
    )
    exact_middle: tuple[tuple[Fraction, ...], ...] | None = None
    if periodic_context is None:
        # The authored edge parameter and the endpoints' source rows define
        # the point. Its rounded carrier need not lie on the binary64 source
        # edge (some such edges have no representable interior point at all).
        exact_middle = _exact_source_segment_points(state, source, first, second, t)
        middle = np.asarray(exact_middle, dtype=np.float64)
    split_root = np.arange(candidates.size, dtype=np.int64)
    if periodic_context is not None:
        # Every member of a split orbit is the exact deck image of the midpoint
        # on the orbit's representative edge, so the generated vertex orbit is
        # bit-consistent with its root on non-dyadic lattices.
        cell_value = source.periodic_cell
        if cell_value is None:
            raise ValueError("Periodic splits require the source lattice.")
        member_windings = periodic_context.edge_member_windings[candidates]
        for orbit in complete_orbits:
            root = position_of_edge.get(
                int(periodic_context.edge_representative_rows[int(orbit)])
            )
            if root is None:
                raise ValueError("A complete split orbit lost its representative edge.")
            member_positions = np.flatnonzero(candidate_orbits == orbit)
            split_root[member_positions] = root
            members = member_positions[member_positions != root]
            if members.size:
                middle[members] = apply_periodic(
                    cell_value,
                    np.broadcast_to(middle[root], (members.size, middle.shape[1])),
                    member_windings[members] - member_windings[root],
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
        chosen = np.flatnonzero(np.isin(operation_of_candidate, selected))
        chosen_edges = candidates[chosen]
        # The root member is allocated first so it carries the smallest new
        # vertex ID and therefore represents the generated orbit.
        chosen = chosen[
            np.lexsort(
                (
                    ids[second[chosen]],
                    ids[first[chosen]],
                    (split_root[chosen] != chosen).astype(np.int64),
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
    split_metrics = _field_metric(
        source,
        middle[chosen],
        np.array(_interpolate_metrics(samples, weights), dtype=np.float64),
        mode,
    )
    split_orbit_keys: np.ndarray | None = None
    split_orbit_windings: np.ndarray | None = None
    if periodic_context is not None:
        chosen_position = {
            int(position): row for row, position in enumerate(chosen.tolist())
        }
        root_rows = np.asarray(
            [chosen_position[int(split_root[position])] for position in chosen],
            dtype=np.int64,
        )
        cell_value = source.periodic_cell
        member_windings = periodic_context.edge_member_windings[candidates[chosen]]
        for row, root_row in enumerate(root_rows.tolist()):
            if row != root_row:
                split_metrics[row] = _exact_periodic_metric(
                    exact_periodic_element(
                        cell_value, member_windings[row] - member_windings[root_row]
                    ),
                    split_metrics[root_row],
                )
        split_orbit_keys = np.stack(
            (
                np.ones((chosen.size,), dtype=np.int64),
                state.next_vertex_id + root_rows,
            ),
            axis=1,
        )
        split_orbit_windings = reduce_periodic_exponents(
            member_windings, identification_orders(cell_value)
        )
    split_sources, split_weights, split_valid, split_supported, split_exact = (
        _located_stencil_bank(
            source,
            middle[chosen],
            mode,
            stencil_width=state.source_stencil_weights.shape[1],
            exact_points=None
            if exact_middle is None
            else tuple(exact_middle[index] for index in chosen),
        )
    )
    if not np.all(split_supported):
        raise ValueError(
            "Every split midpoint requires exact source interpolation support."
        )
    grown = _append_vertices(
        state,
        middle[chosen],
        split_metrics,
        split_sources,
        split_weights,
        split_valid,
        split_exact,
        split_orbit_keys,
        split_orbit_windings,
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


def _pole_link(
    state: _State, topology: _Topology, edge: Any, removed: Any, kept: Any, /
) -> np.ndarray:
    """Link condition on the physical surface, where the copies of a collapsed
    chart point are one vertex.

    The physical common neighbors of the two ends (excluding the ends) must be
    exactly the physical apexes of the edge, and a removed vertex may not meet
    another copy of a kept collapsed point (that edge would become a loop).
    Only candidates next to a collapsed point can differ from the chart link.
    """
    ok = np.ones(np.shape(removed), dtype=np.bool_)
    pole = state.pole
    if pole is None or ok.size == 0:
        return ok
    near = pole >= 0
    near[topology.edges[np.any(near[topology.edges], axis=1)].reshape((-1,))] = True
    offsets, values = topology.neighbor_offsets, topology.neighbor_values

    def physical(row: int, /) -> tuple[int, int]:
        group = int(pole[row])
        return (1, group) if group >= 0 else (0, row)

    for index in np.flatnonzero(near[removed] | near[kept]).tolist():
        first, second = int(removed[index]), int(kept[index])
        first_neighbors = values[offsets[first] : offsets[first + 1]].tolist()
        second_neighbors = values[offsets[second] : offsets[second + 1]].tolist()
        if pole[second] >= 0 and any(
            row != second and pole[row] == pole[second] for row in first_neighbors
        ):
            ok[index] = False
            continue
        ends_physical = {physical(first), physical(second)}
        common = (
            {physical(row) for row in first_neighbors if row != second}
            & {physical(row) for row in second_neighbors if row != first}
        ) - ends_physical
        apexes = {
            physical(int(row)) for row in topology.edge_apex[int(edge[index])] if row >= 0
        }
        ok[index] = len(common) == len(apexes)
    return ok


def _band_excess(lengths: np.ndarray, /) -> np.ndarray:
    """Distance of ``ln L`` from the unit band ``[-ln sqrt(2), ln sqrt(2)]``."""
    return np.maximum(np.abs(np.log(lengths)) - np.log(_UPPER), 0.0)


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
    if state.pole is not None:
        # A new chart edge to a copy of a collapsed point the kept vertex already
        # meets is an existing physical edge (a chart cut), not a new edge.
        groups = np.full((points.shape[0],), -1, dtype=np.int64)
        groups[: state.pole.size] = state.pole
        fresh &= ~_adjacent_to_group(
            _pole_pairs(state, topology.edges),
            kept[neighbor],
            groups[other],
        )
    lengths = _metric_lengths(
        state.metric, points, np.stack((kept[neighbor][fresh], other[fresh]), axis=1)
    )
    longest = _group_maximum(lengths, neighbor[fresh], removed.size, 0.0)
    return status, longest


def _collapse_within_limit(births: Any, longest: Any, escape: bool, /) -> Any:
    """Length admission of collapses (see ``execute_local_metric_adaptation``).

    Regular phases: a collapse that removes or keeps a birth (a -1 source row,
    reset by the next execution) must keep every new edge at or below the split
    threshold (the split/collapse hysteresis); other collapses up to
    ``_COLLAPSE_LIMIT``. The stall escape admits every collapse up to
    ``_COLLAPSE_LIMIT``; the next split phase splits its long edges.
    """
    if escape:
        return longest <= _COLLAPSE_LIMIT
    return longest <= np.where(births, _UPPER, _COLLAPSE_LIMIT)


def _collapse_candidates(
    state: _State, topology: _Topology, mode: PredicateMode, escape: bool, /
) -> _CollapseCandidates:
    short = np.flatnonzero((topology.lengths < _LOWER) & ~topology.edge_protected)
    edge = np.concatenate((short, short))
    removed = np.concatenate((topology.edges[short, 0], topology.edges[short, 1]))
    kept = np.concatenate((topology.edges[short, 1], topology.edges[short, 0]))
    admissible = _collapse_admissible(topology, edge, removed, kept)
    link = _collapse_link(topology, edge, removed, kept) & _pole_link(
        state, topology, edge, removed, kept
    )
    orientation, longest = _collapse_geometry(state, topology, removed, kept, mode)
    births = (state.source_rows[removed] < 0) | (state.source_rows[kept] < 0)
    within = _collapse_within_limit(births, longest, escape)
    geometric = np.maximum(orientation, np.where(link & within, _OK, _INVALID))
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


class _PeriodicCollapse(NamedTuple):
    """One complete quotient collapse: every lifted member of the removed orbit.

    ``kept`` holds state rows; values ``>= base`` index ``image_*`` rows that
    must be appended as new lifted images of the kept vertex orbit.
    """

    quotient_id: int
    length: float
    removed: np.ndarray
    kept: np.ndarray
    base: int
    image_points: np.ndarray
    image_metrics: np.ndarray
    image_keys: np.ndarray
    image_windings: np.ndarray
    image_stencils: tuple[Any, ...] | None
    items: np.ndarray
    # ``(kept row, neighbor row, class, protected)`` feature records of new
    # lifted members of the kept orbit's classified quotient edges.
    feature_additions: np.ndarray
    # Lifted member rows the collapse empties and the surviving member of the
    # same vertex orbit that each retires onto (see ``_collapse_orphans``).
    retired: np.ndarray
    retired_to: np.ndarray


class _Orphans(NamedTuple):
    guard_cells: np.ndarray
    retired: np.ndarray
    retired_to: np.ndarray


def _ball_cells(topology: _Topology, rows: np.ndarray, /) -> np.ndarray:
    _, positions = _expand(topology.ball_offsets, np.asarray(rows, dtype=np.int64))
    return np.unique(topology.ball_cells[positions])


def _collapse_orphans(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext,
    removed: np.ndarray,
    kept: np.ndarray,
    /,
) -> _Orphans | None:
    """Guard cells and retired lifted rows of a collapse, or ``None`` if refused.

    A seam vertex keeps one lifted row per member, each with part of its quotient
    ball. Dropping the edge cells may empty such a row although its quotient
    vertex survives. The cut lifted mesh then no longer needs that member: it is
    retired onto another member of its vertex orbit that keeps cells, so the
    quotient vertex, its identity and its lineage are preserved (a source row
    retires only onto a source row, which carries the quotient vertex's source
    ID). Each classified lifted edge of a retired row must keep another lifted
    member of its quotient edge, which carries the feature record on. Otherwise,
    or without such a vertex member, the collapse is refused. The returned guard
    cells (balls of the dropped cells' corners and of the retirement targets,
    and the cells of the surviving classified members) make simultaneously
    selected collapses disjoint there, so they cannot empty one row, vertex
    orbit, or classified quotient edge together.
    """
    empty = np.zeros((0,), dtype=np.int64)
    dropped: set[int] = set()
    gained: dict[int, int] = {}
    for removed_row, kept_row in zip(removed.tolist(), kept.tolist(), strict=True):
        cells = _ball_cells(topology, np.asarray((removed_row,), dtype=np.int64))
        edge_cells = cells[np.any(state.cells[cells] == kept_row, axis=1)]
        dropped.update(int(cell) for cell in edge_cells.tolist())
        gained[int(kept_row)] = gained.get(int(kept_row), 0) + int(
            cells.size - edge_cells.size
        )
    if not dropped:
        return _Orphans(empty, empty, empty)
    removed_set = {int(row) for row in removed.tolist()}
    corners = np.unique(state.cells[np.asarray(sorted(dropped), dtype=np.int64)])

    def remaining(row: int) -> int:
        return gained.get(row, 0) + sum(
            int(cell) not in dropped
            for cell in _ball_cells(topology, np.asarray((row,), dtype=np.int64))
        )

    orphans = [
        row
        for row in corners.tolist()
        if row not in removed_set and remaining(int(row)) == 0
    ]
    orphan_set = set(orphans)
    gone = removed_set | orphan_set
    targets: list[int] = []
    kept_edge_cells: list[int] = []
    pole = state.pole
    for row in orphans:
        twin = pole is not None and bool(pole[row] >= 0)
        # Each classified lifted edge of the row must keep a lifted member of
        # its quotient edge (with its feature record) after the collapse. A copy
        # of a collapsed chart point has only chart cuts, each one physical edge
        # with its twin at a neighboring copy, and retires onto any live copy.
        for keys in (
            () if twin else state.feature_keys[np.any(state.feature_keys == row, axis=1)]
        ):
            edge = int(_lookup(topology.codes, _codes(keys[None, :]))[0])
            survivors = [
                int(member)
                for member in np.flatnonzero(
                    context.edge_orbits == context.edge_orbits[edge]
                ).tolist()
                if member != edge
                and not any(int(end) in gone for end in topology.edges[member])
                and any(
                    int(cell) >= 0 and int(cell) not in dropped
                    for cell in topology.edge_cells[member]
                )
            ]
            if not survivors:
                return None
            kept_edge_cells.extend(
                int(cell)
                for cell in topology.edge_cells[survivors].reshape((-1,))
                if cell >= 0
            )
        candidates = (
            np.flatnonzero(pole == pole[row])
            if pole is not None and twin
            else np.flatnonzero(context.vertex_orbits == context.vertex_orbits[row])
        )
        members = [
            int(member)
            for member in candidates.tolist()
            if member not in gone
            and (state.source_rows[row] < 0 or state.source_rows[member] >= 0)
            and remaining(int(member)) > 0
        ]
        if not members:
            return None
        targets.append(min(members))
    retired = np.asarray(orphans, dtype=np.int64)
    retired_to = np.asarray(targets, dtype=np.int64)
    return _Orphans(
        np.union1d(
            _ball_cells(topology, np.concatenate((corners, retired_to))),
            np.asarray(kept_edge_cells, dtype=np.int64),
        ),
        retired,
        retired_to,
    )


def _cover_labels(
    state: _State,
    topology: _Topology,
    rows: np.ndarray,
    frame_row: int,
    orders: np.ndarray,
    /,
) -> set[tuple[int, ...]]:
    """Cover-vertex labels of every neighbor of an orbit, in one member's frame.

    A neighbor ``x`` of member ``m`` is the cover vertex
    ``(key(x), winding(x) + winding(frame) - winding(m))``.
    """
    keys = state.orbit_keys
    windings = state.orbit_windings
    if keys is None or windings is None:
        raise ValueError("Periodic cover labels require explicit orbit identity.")
    labels: set[tuple[int, ...]] = set()
    for row in np.asarray(rows, dtype=np.int64).tolist():
        neighbors = topology.neighbor_values[
            topology.neighbor_offsets[row] : topology.neighbor_offsets[row + 1]
        ]
        reduced = reduce_periodic_exponents(
            windings[neighbors] + (windings[frame_row] - windings[row]), orders
        )
        labels.update(
            (*(int(value) for value in keys[neighbor]), *(int(value) for value in shift))
            for neighbor, shift in zip(neighbors.tolist(), reduced, strict=True)
        )
    return labels


def _periodic_cover_collapse(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext,
    source: _Source,
    mode: PredicateMode,
    edges: np.ndarray,
    removed: np.ndarray,
    kept: np.ndarray,
    quotient_id: int,
    escape: bool,
    /,
) -> _PeriodicCollapse | str:
    """Collapse a removed vertex orbit whose lifted members are not all edge ends.

    The lifted mesh cuts the quotient along a fundamental-domain seam, so a seam
    vertex has several lifted members, each carrying part of its quotient ball.
    Every member is removed: a member at an edge end collapses onto that edge's
    kept end, and any other member collapses onto the exact deck image of the
    kept vertex in its own frame (an existing lifted member of the kept orbit or
    a new lifted image of it). Admissibility, the link condition, and validity
    are decided on the quotient (the cover around the edge), not on the cut. A
    refused collapse returns its reason (the stall evidence reports it).
    """
    cell_value = source.periodic_cell
    keys = state.orbit_keys
    windings = state.orbit_windings
    if cell_value is None or keys is None or windings is None:
        return "identity"
    orbits = context.vertex_orbits
    removed_orbit = int(orbits[removed[0]])
    kept_orbit = int(orbits[kept[0]])
    if (
        removed_orbit < 0
        or kept_orbit < 0
        or removed_orbit == kept_orbit
        or np.any(orbits[removed] != removed_orbit)
        or np.any(orbits[kept] != kept_orbit)
        or np.any(topology.edge_class[edges] != 0)
        or np.any(topology.edge_protected[edges])
    ):
        return "classified"
    removed_rows = np.flatnonzero(orbits == removed_orbit)
    kept_rows = np.flatnonzero(orbits == kept_orbit)
    if any(
        context.vertex_stabilizers[int(row)]
        for row in np.concatenate((removed_rows, kept_rows)).tolist()
    ):
        return "stabilized"
    if state.pole is not None and np.any(state.pole[kept_rows] >= 0):
        # Deck images of a collapsed chart point copy are not formed here.
        return "pole"
    removed_ball = _ball_cells(topology, removed_rows)
    if (
        np.any(topology.kind[removed_rows] != _INTERIOR)
        or not np.all(topology.uniform_region[removed_rows])
        or np.unique(state.cell_region[removed_ball]).size != 1
    ):
        return "inadmissible"
    orders = identification_orders(cell_value)

    def reduced(values: np.ndarray) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in reduce_periodic_exponents(
                np.asarray(values, dtype=np.int64)[None, :], orders
            )[0]
        )

    root_removed = int(removed[0])
    root_kept = int(kept[0])
    kept_by_removed = dict(zip(removed.tolist(), kept.tolist(), strict=True))
    kept_by_winding = {reduced(windings[row]): int(row) for row in kept_rows.tolist()}
    base = state.points.shape[0]
    targets: list[int] = []
    image_windings: list[tuple[int, ...]] = []
    for row in removed_rows.tolist():
        target_winding = reduced(
            windings[root_kept] + windings[row] - windings[root_removed]
        )
        target = kept_by_removed.get(int(row))
        if target is not None:
            if reduced(windings[target]) != target_winding:
                return "winding"
        else:
            target = kept_by_winding.get(target_winding)
            if target is None:
                target = base + len(image_windings)
                image_windings.append(target_winding)
        targets.append(int(target))
    target_rows = np.asarray(targets, dtype=np.int64)
    orphans = _collapse_orphans(state, topology, context, removed_rows, target_rows)
    if orphans is None:
        return "orphaned"

    removed_label = (*map(int, keys[root_removed]), *reduced(windings[root_removed]))
    kept_label = (*map(int, keys[root_kept]), *reduced(windings[root_kept]))
    removed_neighbors = _cover_labels(state, topology, removed_rows, root_removed, orders)
    kept_neighbors = _cover_labels(state, topology, kept_rows, root_kept, orders)
    key_width = keys.shape[1]
    if any(
        label[:key_width] == kept_label[:key_width] and label != kept_label
        for label in removed_neighbors
    ) or any(
        label[:key_width] == removed_label[:key_width] and label != removed_label
        for label in kept_neighbors
    ):
        # Another image of the kept orbit next to the removed orbit would turn
        # into a quotient loop edge.
        return "loop"
    common = (removed_neighbors & kept_neighbors) - {removed_label, kept_label}
    apex_count = int(np.count_nonzero(topology.edge_cells[edges] >= 0))
    if len(common) != apex_count:
        return "link"
    if state.pole is not None:
        # The same link on the physical surface: copies of a collapsed chart
        # point are one vertex.
        pole = state.pole
        pole_by_key = {
            tuple(int(value) for value in keys[row]): int(pole[row])
            for row in np.flatnonzero(pole >= 0).tolist()
        }

        def physical(label: tuple[int, ...], /) -> tuple[int, ...]:
            group = pole_by_key.get(label[:key_width])
            return label if group is None else (-1, group)

        physical_common = {
            physical(label) for label in removed_neighbors - {kept_label}
        } & {physical(label) for label in kept_neighbors - {removed_label}}
        if len(physical_common) != apex_count:
            return "link"

    # A removed member's edge to a common neighbor becomes a new lifted member
    # of the kept orbit's existing quotient edge; it carries that edge's class.
    classified: dict[tuple[int, ...], tuple[int, int]] = {}
    for row in kept_rows.tolist():
        for neighbor in topology.neighbor_values[
            topology.neighbor_offsets[row] : topology.neighbor_offsets[row + 1]
        ].tolist():
            edge = int(
                _lookup(
                    topology.codes,
                    _codes(_sorted_pairs(np.asarray((row,)), np.asarray((neighbor,)))),
                )[0]
            )
            if topology.edge_class[edge] > 0 or topology.edge_protected[edge]:
                label = (
                    *map(int, keys[neighbor]),
                    *reduced(windings[neighbor] + windings[root_kept] - windings[row]),
                )
                classified[label] = (
                    int(topology.edge_class[edge]),
                    int(topology.edge_protected[edge]),
                )
    removed_set = {int(row) for row in removed_rows.tolist()}
    feature_additions: list[tuple[int, int, int, int]] = []
    for row, target in zip(removed_rows.tolist(), targets, strict=True):
        target_winding = (
            windings[target]
            if target < base
            else np.asarray(image_windings[target - base], dtype=np.int64)
        )
        for neighbor in topology.neighbor_values[
            topology.neighbor_offsets[row] : topology.neighbor_offsets[row + 1]
        ].tolist():
            if neighbor in removed_set or neighbor == target:
                continue
            if (
                target < base
                and _lookup(
                    topology.codes,
                    _codes(_sorted_pairs(np.asarray((target,)), np.asarray((neighbor,)))),
                )[0]
                >= 0
            ):
                continue
            label = (
                *map(int, keys[neighbor]),
                *reduced(windings[neighbor] + windings[root_kept] - target_winding),
            )
            if label in classified:
                feature_additions.append((int(target), int(neighbor), *classified[label]))

    image_count = len(image_windings)
    dimension = state.points.shape[1]
    image_points = np.zeros((image_count, dimension), dtype=np.float64)
    image_metrics = np.zeros((image_count, dimension, dimension), dtype=np.float64)
    representative = int(context.vertex_representatives[root_kept])
    for image, winding in enumerate(image_windings):
        isometry = exact_periodic_element(
            cell_value,
            np.asarray(winding, dtype=np.int64) - windings[representative],
        )
        image_points[image] = _exact_periodic_point(
            isometry, state.points[representative]
        )
        image_metrics[image] = _exact_periodic_metric(
            isometry, state.metric[representative]
        )
    extended = (
        state
        if image_count == 0
        else state._replace(
            points=np.concatenate((state.points, image_points)),
            metric=np.concatenate((state.metric, image_metrics)),
        )
    )
    status, longest = _collapse_geometry(
        extended, topology, removed_rows, target_rows, mode
    )
    if np.any(status != _OK):
        return "invalid" if np.all(status != _UNCERTAIN) else "uncertain"
    # The same length admission as lifted collapses, decided on the whole orbit.
    births = bool(np.any(state.source_rows[removed_rows] < 0)) or bool(
        np.any(state.source_rows[kept_rows] < 0)
    )
    if np.any(longest > _COLLAPSE_LIMIT):
        return "long"
    if not np.all(_collapse_within_limit(births, longest, escape)):
        return "hysteresis"
    stencils = None
    if image_count:
        stencils = _located_stencil_bank(
            source,
            image_points,
            mode,
            allow_invalid_rows=True,
            stencil_width=state.source_stencil_weights.shape[1],
        )
        if not np.all(stencils[3]):
            return "unlocated"
    return _PeriodicCollapse(
        quotient_id=quotient_id,
        length=float(np.min(topology.lengths[edges])),
        removed=removed_rows,
        kept=target_rows,
        base=base,
        image_points=image_points,
        image_metrics=image_metrics,
        image_keys=np.repeat(keys[root_kept][None, :], image_count, axis=0),
        image_windings=np.asarray(image_windings, dtype=np.int64).reshape(
            (image_count, windings.shape[1])
        ),
        image_stencils=stencils,
        feature_additions=np.asarray(feature_additions, dtype=np.int64).reshape((-1, 4)),
        items=np.union1d(
            np.union1d(removed_ball, _ball_cells(topology, kept_rows)),
            orphans.guard_cells,
        ),
        retired=orphans.retired,
        retired_to=orphans.retired_to,
    )


def _periodic_collapse_operations(
    state: _State,
    topology: _Topology,
    found: _CollapseCandidates,
    mode: PredicateMode,
    context: _PeriodicContext,
    source: _Source,
    escape: bool,
    /,
) -> list[_PeriodicCollapse]:
    """Complete quotient collapse operations, one per admissible edge orbit."""
    half = found.edge.size // 2
    short_edges = found.edge[:half]
    canonical_removed = context.edge_canonical_vertices[short_edges, 0]
    forward = found.removed[:half] == canonical_removed
    first_candidates = np.where(forward, np.arange(half), np.arange(half) + half)
    second_candidates = np.where(forward, np.arange(half) + half, np.arange(half))
    candidate_orbits = context.edge_orbits[short_edges]
    member_count = np.bincount(
        context.edge_orbits, minlength=context.edge_quotient_ids.size
    )
    vertex_members = np.bincount(
        context.vertex_orbits[context.vertex_orbits >= 0],
        minlength=context.vertex_quotient_ids.size,
    )
    operations: list[_PeriodicCollapse] = []
    for orbit in np.unique(candidate_orbits).tolist():
        members = np.flatnonzero(candidate_orbits == orbit)
        if members.size != member_count[orbit]:
            continue
        edges = short_edges[members]
        for candidates in (first_candidates[members], second_candidates[members]):
            removed = found.removed[candidates]
            kept = found.kept[candidates]
            removed_orbit = int(context.vertex_orbits[removed[0]])
            kept_orbit = int(context.vertex_orbits[kept[0]])
            # Seam members (one lifted cell, no class) or orbit members away
            # from the edge need the quotient decision; otherwise every lifted
            # member carries its full quotient ball.
            cover = bool(
                np.any(
                    (topology.edge_cells[edges, 1] < 0)
                    & (topology.edge_class[edges] == 0)
                )
                or vertex_members[removed_orbit] != np.unique(removed).size
                or vertex_members[kept_orbit] != np.unique(kept).size
            )
            operation: _PeriodicCollapse | None
            if cover:
                covered = _periodic_cover_collapse(
                    state,
                    topology,
                    context,
                    source,
                    mode,
                    edges,
                    removed,
                    kept,
                    int(context.edge_quotient_ids[orbit]),
                    escape,
                )
                operation = None if isinstance(covered, str) else covered
            else:
                endpoints = np.concatenate((removed, kept))
                operation = None
                if np.all(found.status[candidates] == _OK) and (
                    np.unique(endpoints).size == endpoints.size
                ):
                    balls = [
                        _ball_cells(topology, np.asarray(pair, dtype=np.int64))
                        for pair in zip(removed.tolist(), kept.tolist(), strict=True)
                    ]
                    items = np.concatenate(balls)
                    orphans = _collapse_orphans(state, topology, context, removed, kept)
                    if np.unique(items).size == items.size and orphans is not None:
                        items = np.union1d(items, orphans.guard_cells)
                        operation = _PeriodicCollapse(
                            quotient_id=int(context.edge_quotient_ids[orbit]),
                            length=float(np.min(topology.lengths[edges])),
                            removed=removed,
                            kept=kept,
                            base=state.points.shape[0],
                            image_points=np.zeros((0, state.points.shape[1])),
                            image_metrics=np.zeros(
                                (0, state.points.shape[1], state.points.shape[1])
                            ),
                            image_keys=np.zeros((0, 2), dtype=np.int64),
                            image_windings=np.zeros((0, 0), dtype=np.int64),
                            image_stencils=None,
                            feature_additions=np.zeros((0, 4), dtype=np.int64),
                            items=items,
                            retired=orphans.retired,
                            retired_to=orphans.retired_to,
                        )
            if operation is not None:
                operations.append(operation)
                break
    return operations


def _retired_lineage(
    state: _State,
    retired: np.ndarray,
    source: _Source,
    /,
) -> _State:
    """Move the lineage records of retired lifted rows onto a live lifted member.

    A record ``(r, y)`` of a retired row ``r`` names the quotient edge
    ``(key r, key y, w_y - w_r)`` (orbit keys and reduced deck exponents), which
    survives on another lifted member: the record moves onto the live lifted
    edge with that label (the least such pair), so every lifted member of its
    source edge keeps the same quotient target. A record whose quotient edge
    has no live member is dropped.
    """
    touched = np.any(np.isin(state.lineage_keys, retired), axis=1)
    if retired.size == 0 or not np.any(touched):
        return state
    keys, windings, cell_value = (
        state.orbit_keys,
        state.orbit_windings,
        source.periodic_cell,
    )
    keys_out = state.lineage_keys.copy()
    keep = ~touched
    if keys is not None and windings is not None and cell_value is not None:
        orders = identification_orders(cell_value)

        def label(first: np.ndarray, second: np.ndarray, /) -> list[tuple[int, ...]]:
            shift = reduce_periodic_exponents(windings[second] - windings[first], orders)
            return [
                (*map(int, a), *map(int, b), *map(int, c))
                for a, b, c in zip(keys[first], keys[second], shift, strict=True)
            ]

        edges = np.unique(
            np.sort(state.cells[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1),
            axis=0,
        )
        members: dict[tuple[int, ...], tuple[int, int]] = {}
        for first, second in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            for name, pair in zip(label(first, second), edges.tolist(), strict=True):
                members.setdefault(name, (int(pair[0]), int(pair[1])))
        rows = np.flatnonzero(touched)
        records = state.lineage_keys[rows]
        for record, name in zip(
            rows.tolist(), label(records[:, 0], records[:, 1]), strict=True
        ):
            pair = members.get(name)
            if pair is not None:
                keys_out[record] = pair
                keep[record] = True
    unique = np.flatnonzero(keep)
    unique = unique[
        _distinct(
            _codes(keys_out[unique]),
            state.lineage_sources[unique],
            -state.lineage_ranks[unique],
        )
    ]
    return state._replace(
        lineage_keys=keys_out[unique],
        lineage_sources=state.lineage_sources[unique],
        lineage_ranks=state.lineage_ranks[unique],
    )


def _collapse_round(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    mode: PredicateMode,
    periodic_context: _PeriodicContext | None,
    _source: _Source,
    /,
    *,
    escape: bool = False,
) -> tuple[_State, _Counts, int]:
    found = _collapse_candidates(state, topology, mode, escape)
    if found.edge.size == 0:
        return state, counts, 0
    half = found.edge.size // 2
    best = np.minimum(found.status[:half], found.status[half:])
    counts = _tally(counts, np.where(best == _INADMISSIBLE, _INVALID, best))
    valid = np.flatnonzero(found.status == _OK)
    if periodic_context is None and state.pole is not None:
        # Without orbit retirement, a collapse may not empty a copy of a
        # collapsed chart point (its only cell is an edge cell).
        apexes = topology.edge_apex[found.edge[valid]]
        rows = np.maximum(apexes, 0)
        lone = (
            (apexes >= 0)
            & (state.pole[rows] >= 0)
            & (np.diff(topology.ball_offsets)[rows] == 1)
        )
        valid = valid[~np.any(lone, axis=1)]
    if valid.size == 0 and periodic_context is None:
        return state, counts, 0

    retired = np.zeros((0,), dtype=np.int64)
    retired_to = np.zeros((0,), dtype=np.int64)
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
        operations = _periodic_collapse_operations(
            state, topology, found, mode, periodic_context, _source, escape
        )
        if not operations:
            return state, counts, 0
        rank = _ranks(
            np.asarray([operation.length for operation in operations], dtype=np.float64),
            np.asarray(
                [operation.quotient_id for operation in operations], dtype=np.int64
            ),
        )
        selected = _disjoint_selection(
            rank,
            np.concatenate(
                [
                    np.full((operation.items.size,), index, dtype=np.int64)
                    for index, operation in enumerate(operations)
                ]
            ),
            np.concatenate([operation.items for operation in operations]),
            state.cells.shape[0],
        )
        removed_parts: list[np.ndarray] = []
        kept_parts: list[np.ndarray] = []
        retired_parts: list[np.ndarray] = [retired]
        retired_to_parts: list[np.ndarray] = [retired_to]
        for operation, chosen in zip(operations, selected.tolist(), strict=True):
            if not chosen:
                continue
            kept_rows = operation.kept
            first_image = state.points.shape[0]
            if operation.image_points.shape[0]:
                stencils = operation.image_stencils
                if stencils is None:
                    raise RuntimeError("Periodic collapse images lost their stencils.")
                state = _append_vertices(
                    state,
                    operation.image_points,
                    operation.image_metrics,
                    stencils[0],
                    stencils[1],
                    stencils[2],
                    stencils[4],
                    operation.image_keys,
                    operation.image_windings,
                )
                kept_rows = np.where(
                    kept_rows >= operation.base,
                    kept_rows - operation.base + first_image,
                    kept_rows,
                )
            additions = operation.feature_additions
            if additions.shape[0]:
                owners = np.where(
                    additions[:, 0] >= operation.base,
                    additions[:, 0] - operation.base + first_image,
                    additions[:, 0],
                )
                state = state._replace(
                    feature_keys=np.concatenate(
                        (state.feature_keys, _sorted_pairs(owners, additions[:, 1]))
                    ),
                    feature_class=np.concatenate((state.feature_class, additions[:, 2])),
                    feature_protected=np.concatenate(
                        (state.feature_protected, additions[:, 3].astype(np.bool_))
                    ),
                )
            removed_parts.append(operation.removed)
            kept_parts.append(kept_rows)
            retired_parts.append(operation.retired)
            retired_to_parts.append(operation.retired_to)
        removed = np.concatenate(removed_parts)
        kept = np.concatenate(kept_parts)
        retired = np.concatenate(retired_parts)
        retired_to = np.concatenate(retired_to_parts)

    updated = _collapse_records(
        _collapse_cells(state, topology, removed, kept), removed, kept
    )
    # Retired lifted members have no edges left; their quotient edges keep
    # their lineage and feature records on the surviving members.
    features = ~np.any(np.isin(updated.feature_keys, retired), axis=1)
    alive = updated.alive.copy()
    alive[removed] = False
    alive[retired] = False
    collapse_to = updated.collapse_to.copy()
    collapse_to[removed] = kept
    collapse_to[retired] = retired_to
    updated = _retired_lineage(
        updated._replace(alive=alive, collapse_to=collapse_to), retired, _source
    )
    updated = updated._replace(
        feature_keys=updated.feature_keys[features],
        feature_class=updated.feature_class[features],
        feature_protected=updated.feature_protected[features],
    )
    return (
        updated,
        counts._replace(collapses=counts.collapses + removed.size),
        removed.size,
    )


def _escape_collapse_round(
    state: _State,
    topology: _Topology,
    counts: _Counts,
    mode: PredicateMode,
    periodic_context: _PeriodicContext | None,
    source: _Source,
    /,
) -> tuple[_State, _Counts, int]:
    """Stall escape: every collapse up to ``_COLLAPSE_LIMIT`` (births included)."""
    return _collapse_round(
        state, topology, counts, mode, periodic_context, source, escape=True
    )


# ------------------------------------------------------------------ flip


class _SeamFlip(NamedTuple):
    """Flip of a seam edge orbit whose two lifted members each bound one cell.

    ``children`` holds state rows; values ``>= base`` index ``image_*`` rows
    that must be appended as new lifted images of an apex vertex orbit.
    """

    quotient_id: int
    gain: float
    removed_cells: np.ndarray
    children: np.ndarray
    base: int
    image_points: np.ndarray
    image_metrics: np.ndarray
    image_keys: np.ndarray
    image_windings: np.ndarray
    image_stencils: tuple[Any, ...] | None
    flipped_edges: np.ndarray
    items: np.ndarray
    # ``(row, row, class, protected)`` feature records of new lifted members of
    # classified quotient edges.
    feature_additions: np.ndarray


def _reduced_winding(values: np.ndarray, orders: np.ndarray, /) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in reduce_periodic_exponents(
            np.asarray(values, dtype=np.int64)[None, :], orders
        )[0]
    )


def _cover_edge_exists(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext,
    first: int,
    second: int,
    orders: np.ndarray,
    /,
) -> bool:
    """Whether two lifted rows of one frame are already adjacent in the cover."""
    keys = state.orbit_keys
    windings = state.orbit_windings
    if keys is None or windings is None:
        return False
    rows = np.flatnonzero(context.vertex_orbits == context.vertex_orbits[first])
    label = (*map(int, keys[second]), *_reduced_winding(windings[second], orders))
    return label in _cover_labels(state, topology, rows, first, orders)


def _periodic_seam_flips(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext,
    source: _Source,
    mode: PredicateMode,
    /,
) -> list[_SeamFlip]:
    """Improving flips of seam edge orbits, decided on the quotient.

    A fundamental-domain seam edge has two lifted members, each bounding one
    lifted cell. Its quotient flip joins the two apexes: both new cells are
    placed in one member's frame, with the other apex as its exact deck image
    there (an existing lifted member of that orbit or a new lifted image). The
    frame is chosen so that no lifted vertex row loses its last cell.
    """
    cell_value = source.periodic_cell
    keys = state.orbit_keys
    windings = state.orbit_windings
    if cell_value is None or keys is None or windings is None:
        return []
    seam = (
        (topology.edge_cells[:, 1] < 0)
        & (topology.edge_class == 0)
        & ~topology.edge_protected
    )
    if not np.any(seam):
        return []
    orders = identification_orders(cell_value)
    orbits = context.vertex_orbits
    ball_sizes = np.diff(topology.ball_offsets)
    base = state.points.shape[0]
    dimension = state.points.shape[1]

    def corners(cell: int, edge: int) -> tuple[int, int, int]:
        rows = state.cells[cell]
        local = int(np.flatnonzero(topology.cell_edges[cell] == edge)[0])
        return int(rows[local]), int(rows[(local + 1) % 3]), int(rows[(local + 2) % 3])

    def existing_row(row: int, winding: tuple[int, ...]) -> int | None:
        for member in np.flatnonzero(orbits == orbits[row]).tolist():
            if _reduced_winding(windings[member], orders) == winding:
                return int(member)
        return None

    flips: list[_SeamFlip] = []
    for orbit in np.unique(context.edge_orbits[seam]).tolist():
        members = np.flatnonzero(context.edge_orbits == orbit)
        if members.size != 2 or not np.all(seam[members]):
            continue
        first_edge, second_edge = (int(member) for member in members)
        cell_a = int(topology.edge_cells[first_edge, 0])
        cell_b = int(topology.edge_cells[second_edge, 0])
        if cell_a == cell_b or state.cell_region[cell_a] != state.cell_region[cell_b]:
            continue
        u_a, v_a, x_a = corners(cell_a, first_edge)
        p_b, q_b, x_b = corners(cell_b, second_edge)
        # The second member's cell traverses the quotient edge backwards.
        if orbits[p_b] != orbits[v_a] or orbits[q_b] != orbits[u_a]:
            continue
        shift = windings[u_a] - windings[q_b]
        if _reduced_winding(windings[v_a] - windings[p_b], orders) != _reduced_winding(
            shift, orders
        ) or any(
            context.vertex_stabilizers[row] for row in (u_a, v_a, x_a, p_b, q_b, x_b)
        ):
            continue
        far_b = _reduced_winding(windings[x_b] + shift, orders)
        far_a = _reduced_winding(windings[x_a] - shift, orders)
        if orbits[x_a] == orbits[x_b] and far_b == _reduced_winding(
            windings[x_a], orders
        ):
            continue
        if state.pole is not None and (state.pole[x_a] >= 0 or state.pole[x_b] >= 0):
            # Seam diagonals at a collapsed chart point copy are not formed.
            continue
        diagonal_label = (*map(int, keys[x_b]), *far_b)
        if diagonal_label in _cover_labels(
            state, topology, np.flatnonzero(orbits == orbits[x_a]), x_a, orders
        ):
            continue

        def keeps_lifted_rows(cell: int, edge: int, rows: tuple[int, ...]) -> bool:
            # The cell leaves its frame: no corner row may lose its last cell and
            # no outer lifted edge (with its feature and lineage records) may
            # lose its last cell.
            outer = [int(value) for value in topology.cell_edges[cell] if value != edge]
            return all(ball_sizes[row] > 1 for row in rows) and bool(
                np.all(topology.edge_cells[outer, 1] >= 0)
            )

        options: list[tuple[bool, tuple[int, int, int], int, tuple[int, ...]]] = []
        if keeps_lifted_rows(cell_b, second_edge, (p_b, q_b, x_b)):
            options.append(
                (existing_row(x_b, far_b) is None, (u_a, v_a, x_a), x_b, far_b)
            )
        if keeps_lifted_rows(cell_a, first_edge, (u_a, v_a, x_a)):
            # The second cell traverses the edge p_b -> q_b counterclockwise.
            options.append(
                (existing_row(x_a, far_a) is None, (p_b, q_b, x_b), x_a, far_a)
            )
        if not options:
            continue
        needs_image, (u, v, x0), apex, apex_winding = min(
            options, key=lambda item: item[0]
        )
        image_points = np.zeros((0, dimension), dtype=np.float64)
        image_metrics = np.zeros((0, dimension, dimension), dtype=np.float64)
        image_windings = np.zeros((0, windings.shape[1]), dtype=np.int64)
        x1 = existing_row(apex, apex_winding)
        points = state.points
        tensors = state.metric
        if x1 is None:
            representative = int(context.vertex_representatives[apex])
            isometry = exact_periodic_element(
                cell_value,
                np.asarray(apex_winding, dtype=np.int64) - windings[representative],
            )
            image_points = _exact_periodic_point(isometry, state.points[representative])[
                None, :
            ]
            image_metrics = _exact_periodic_metric(
                isometry, state.metric[representative]
            )[None, :, :]
            image_windings = np.asarray(apex_winding, dtype=np.int64)[None, :]
            points = np.concatenate((points, image_points))
            tensors = np.concatenate((tensors, image_metrics))
            x1 = base
        children = np.asarray(((u, x1, x0), (v, x0, x1)), dtype=np.int64)
        status = max(
            int(
                _orientation(
                    points[children[:, 0]],
                    points[children[:, 1]],
                    points[children[:, 2]],
                    mode,
                ).max()
            ),
            _OK,
        )
        lengths = _metric_lengths(
            tensors,
            points,
            np.sort(
                np.stack((children, children[:, (1, 2, 0)]), axis=2).reshape((-1, 2)),
                axis=1,
            ),
        ).reshape((-1, 3))
        quality = _quality(lengths)
        after = float(np.min(quality))
        before = float(min(topology.quality[cell_a], topology.quality[cell_b]))
        # Both lifted members of the seam orbit must improve on the quotient.
        if status != _OK or not bool(
            _flip_admissible(
                np.asarray(
                    ((topology.quality[cell_a], topology.quality[cell_b]),),
                    dtype=np.float64,
                ),
                quality[None, :],
                lengths[:1, 1],
                np.min(_deviation_codes(topology.lengths[members]), keepdims=True),
            )[0]
        ):
            continue
        # The cell leaving its frame hands each classified outer edge to the new
        # lifted member of that quotient edge in the kept frame.
        if u == u_a:
            lost_cell, lost_edge = cell_b, second_edge
            lost_map = {p_b: v_a, q_b: u_a, x_b: x1}
        else:
            lost_cell, lost_edge = cell_a, first_edge
            lost_map = {u_a: q_b, v_a: p_b, x_a: x1}
        feature_additions: list[tuple[int, int, int, int]] = []
        for outer in topology.cell_edges[lost_cell].tolist():
            if outer == lost_edge or not (
                topology.edge_class[outer] > 0 or topology.edge_protected[outer]
            ):
                continue
            first, second = (lost_map[int(row)] for row in topology.edges[outer])
            if (
                max(first, second) < base
                and _lookup(
                    topology.codes,
                    _codes(_sorted_pairs(np.asarray((first,)), np.asarray((second,)))),
                )[0]
                >= 0
            ):
                continue
            feature_additions.append(
                (
                    first,
                    second,
                    int(topology.edge_class[outer]),
                    int(topology.edge_protected[outer]),
                )
            )
        stencils = None
        # The corners of both cells lose a cell; guarding their balls keeps two
        # selected flips from emptying one lifted row together.
        items = _ball_cells(
            topology, np.asarray((u_a, v_a, x_a, p_b, q_b, x_b), dtype=np.int64)
        )
        if needs_image:
            stencils = _located_stencil_bank(
                source,
                image_points,
                mode,
                allow_invalid_rows=True,
                stencil_width=state.source_stencil_weights.shape[1],
            )
            if not np.all(stencils[3]):
                continue
            # Two flips must not create the same lifted image in one round.
            items = np.union1d(
                items, _ball_cells(topology, np.flatnonzero(orbits == orbits[apex]))
            )
        flips.append(
            _SeamFlip(
                quotient_id=int(context.edge_quotient_ids[orbit]),
                gain=before - after,
                removed_cells=np.asarray((cell_a, cell_b), dtype=np.int64),
                children=children,
                base=base,
                image_points=image_points,
                image_metrics=image_metrics,
                image_keys=np.repeat(keys[apex][None, :], image_points.shape[0], axis=0),
                image_windings=image_windings,
                image_stencils=stencils,
                flipped_edges=members.astype(np.int64),
                items=items,
                feature_additions=np.asarray(feature_additions, dtype=np.int64).reshape(
                    (-1, 4)
                ),
            )
        )
    return flips


class _FlipCandidates(NamedTuple):
    c0: np.ndarray
    c1: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    left: np.ndarray
    right: np.ndarray
    quality: np.ndarray
    status: np.ndarray
    improving: np.ndarray


def _flip_candidates(
    state: _State, topology: _Topology, candidates: np.ndarray, mode: PredicateMode, /
) -> _FlipCandidates:
    """Children, validity, and admission of flipping interior edges ``candidates``."""
    c0 = topology.edge_cells[candidates, 0]
    c1 = topology.edge_cells[candidates, 1]
    local = np.argmax(topology.cell_edges[c0] == candidates[:, None], axis=1)
    rows = state.cells[c0]
    span = np.arange(candidates.size)
    u = rows[span, local]
    v = rows[span, (local + 1) % 3]
    x0 = rows[span, (local + 2) % 3]
    x1 = topology.edge_apex[candidates, 1]

    def outer(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return topology.lengths[
            _lookup(topology.codes, _codes(_sorted_pairs(first, second)))
        ]

    diagonal = _metric_lengths(state.metric, state.points, _sorted_pairs(x0, x1))
    quality = np.stack(
        (
            _quality(np.stack((outer(u, x1), diagonal, outer(x0, u)), axis=1)),
            _quality(np.stack((outer(v, x0), diagonal, outer(x1, v)), axis=1)),
        ),
        axis=1,
    )
    points = state.points
    status = np.maximum(
        _orientation(points[u], points[x1], points[x0], mode),
        _orientation(points[v], points[x0], points[x1], mode),
    )
    duplicate = _lookup(topology.codes, _codes(_sorted_pairs(x0, x1))) >= 0
    if state.pole is not None:
        # On the physical surface the copies of a collapsed chart point are one
        # vertex: a diagonal may not join two copies or re-create a cut.
        pairs = _pole_pairs(state, topology.edges)
        first, second = state.pole[x0], state.pole[x1]
        duplicate |= (
            ((first >= 0) & (second >= 0))
            | _adjacent_to_group(pairs, x1, first)
            | _adjacent_to_group(pairs, x0, second)
        )
    status = np.where(duplicate, _INVALID, status)
    # A quality flip may not create an edge the next metric pass must
    # immediately edit; a deviation flip only shortens the diagonal's deviation.
    improving = _flip_admissible(
        np.stack((topology.quality[c0], topology.quality[c1]), axis=1),
        quality,
        diagonal,
        _deviation_codes(topology.lengths[candidates]),
    )
    return _FlipCandidates(
        c0,
        c1,
        x0,
        x1,
        np.stack((u, x1, x0), axis=1),
        np.stack((v, x0, x1), axis=1),
        quality,
        status,
        improving,
    )


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
    if candidates.size == 0 and periodic_context is None:
        return state, counts, 0
    found = _flip_candidates(state, topology, candidates, mode)
    c0, c1, x0, x1 = found.c0, found.c1, found.x0, found.x1
    left, right, status, improving = (
        found.left,
        found.right,
        found.status,
        found.improving,
    )
    after = np.min(found.quality, axis=1)
    before = np.minimum(topology.quality[c0], topology.quality[c1])
    counts = _tally(counts, status[improving])
    valid = np.flatnonzero(improving & (status == _OK))
    seam_flips: list[_SeamFlip] = []
    if periodic_context is not None:
        seam_flips = _periodic_seam_flips(
            state, topology, periodic_context, _source, mode
        )
    if valid.size == 0 and not seam_flips:
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
        selected_seam: list[_SeamFlip] = []
    else:
        candidate_valid = improving & (status == _OK)
        orders = (
            None
            if _source.periodic_cell is None
            else identification_orders(_source.periodic_cell)
        )
        for orbit in np.unique(periodic_context.edge_orbits[candidates]):
            members = np.flatnonzero(periodic_context.edge_orbits[candidates] == orbit)
            incident_cells = np.concatenate((c0[members], c1[members]))
            if np.unique(incident_cells).size != incident_cells.size:
                candidate_valid[members] = False
            elif (
                orders is not None
                and bool(np.all(candidate_valid[members]))
                and _cover_edge_exists(
                    state,
                    topology,
                    periodic_context,
                    int(x0[members[0]]),
                    int(x1[members[0]]),
                    orders,
                )
            ):
                # The new diagonal already exists through a seam image.
                candidate_valid[members] = False
        eligible, operation_of_candidate = _complete_periodic_operations(
            candidates,
            periodic_context.edge_orbits,
            candidate_valid,
            periodic_context.edge_quotient_ids.size,
        )
        orbit_gain = np.full(
            (periodic_context.edge_quotient_ids.size,), np.inf, dtype=np.float64
        )
        np.minimum.at(
            orbit_gain,
            periodic_context.edge_orbits[candidates],
            before - after,
        )
        operation_count = eligible.size + len(seam_flips)
        if operation_count == 0:
            return state, counts, 0
        rank = _ranks(
            np.concatenate(
                (
                    orbit_gain[eligible],
                    np.asarray([flip.gain for flip in seam_flips], dtype=np.float64),
                )
            ),
            np.concatenate(
                (
                    periodic_context.edge_quotient_ids[eligible],
                    np.asarray([flip.quotient_id for flip in seam_flips], dtype=np.int64),
                )
            ),
        )
        pair_operations = operation_of_candidate[
            np.concatenate((np.arange(candidates.size), np.arange(candidates.size)))
        ]
        pair_items = np.concatenate((c0, c1))
        inside = pair_operations >= 0
        selected_operations = _disjoint_selection(
            rank,
            np.concatenate(
                (
                    pair_operations[inside],
                    *(
                        np.full((flip.items.size,), eligible.size + index, np.int64)
                        for index, flip in enumerate(seam_flips)
                    ),
                )
            ),
            np.concatenate((pair_items[inside], *(flip.items for flip in seam_flips))),
            state.cells.shape[0],
        )
        selected = np.flatnonzero(
            (operation_of_candidate >= 0)
            & selected_operations[np.maximum(operation_of_candidate, 0)]
        )
        selected_seam = [
            flip
            for index, flip in enumerate(seam_flips)
            if selected_operations[eligible.size + index]
        ]
    count = selected.size
    removed_cells = [c0[selected], c1[selected]]
    children_parts = [left[selected], right[selected]]
    child = np.arange(2 * count, dtype=np.int64)
    child_of = [child, child]
    parent_of = [c0[selected], c1[selected], c1[selected], c0[selected]]
    flipped_edges = [candidates[selected]]
    next_child = 2 * count
    for flip in selected_seam:
        flip_children = flip.children
        first_image = state.points.shape[0]
        if flip.image_points.shape[0]:
            stencils = flip.image_stencils
            if stencils is None:
                raise RuntimeError("Periodic flip images lost their stencils.")
            state = _append_vertices(
                state,
                flip.image_points,
                flip.image_metrics,
                stencils[0],
                stencils[1],
                stencils[2],
                stencils[4],
                flip.image_keys,
                flip.image_windings,
            )
            flip_children = np.where(
                flip_children >= flip.base,
                flip_children - flip.base + first_image,
                flip_children,
            )
        additions = flip.feature_additions
        if additions.shape[0]:
            ends = np.where(
                additions[:, :2] >= flip.base,
                additions[:, :2] - flip.base + first_image,
                additions[:, :2],
            )
            state = state._replace(
                feature_keys=np.concatenate((state.feature_keys, np.sort(ends, axis=1))),
                feature_class=np.concatenate((state.feature_class, additions[:, 2])),
                feature_protected=np.concatenate(
                    (state.feature_protected, additions[:, 3].astype(np.bool_))
                ),
            )
        first, second = (int(cell) for cell in flip.removed_cells)
        removed_cells.append(flip.removed_cells)
        children_parts.append(flip_children)
        pair = np.asarray((next_child, next_child + 1), dtype=np.int64)
        child_of.append(np.concatenate((pair, pair)))
        parent_of.append(np.asarray((first, second, second, first), dtype=np.int64))
        flipped_edges.append(flip.flipped_edges)
        next_child += 2
    count += len(selected_seam)
    updated = _replace_cells(
        state,
        np.concatenate(removed_cells),
        np.concatenate(children_parts),
        np.concatenate(child_of),
        np.concatenate(parent_of),
        _SWAPPED,
    )
    flipped = np.sort(_codes(topology.edges[np.concatenate(flipped_edges)]))
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

    Interior vertices use every unprotected neighbor edge (a chart cut at a free
    vertex is not measured); curve vertices only their two curve neighbors, so
    the displacement stays on the straight curve line.
    """
    owner, position = _expand(topology.neighbor_offsets, vertices)
    other = topology.neighbor_values[position]
    edge = _lookup(topology.codes, _codes(_sorted_pairs(vertices[owner], other)))
    measured = ~topology.edge_protected[edge]
    owner, other, edge = owner[measured], other[measured], edge[measured]
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
    cells, weights, _ = _proposal_locations(state, topology, vertex, target)
    return _located_metric(state, cells, weights)


def _proposal_locations(
    state: _State, topology: _Topology, vertex: Any, target: Any, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best ball cell, its barycentric weights, and their minimum for each target.

    A target lies in its vertex ball exactly when its score is nonnegative; a
    degenerate binary64 ball cell yields a NaN score and is ranked last.
    """
    owner, position = _expand(topology.ball_offsets, vertex)
    cell = topology.ball_cells[position]
    weights = _barycentric(state.points[state.cells[cell]], target[owner])
    score = np.min(weights, axis=1)
    order = np.lexsort((cell, -score, owner))
    first = order[np.searchsorted(owner[order], np.arange(vertex.size))]
    return cell[first], weights[first], score[first]


def _located_metric(state: _State, cells: np.ndarray, weights: np.ndarray, /) -> Any:
    """Log-Euclidean P1 metric at located points of ``cells``."""
    return _interpolate_metrics(state.metric[state.cells[cells]], _normalized(weights))


def _periodic_orbit_pull(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext,
    members: np.ndarray,
    /,
) -> np.ndarray:
    """Spring displacement of an interior vertex orbit in its representative frame.

    A seam vertex has several lifted members, each holding part of its quotient
    ball, so the pull sums every member's edges, counting each quotient edge
    orbit once, and maps each member-frame vector back through the orthogonal
    linear part of that member's exact deck element.
    """
    owner, position = _expand(topology.neighbor_offsets, members)
    other = topology.neighbor_values[position]
    edge = _lookup(topology.codes, _codes(_sorted_pairs(members[owner], other)))
    _, first = np.unique(context.edge_orbits[edge], return_index=True)
    owner, other, edge = owner[first], other[first], edge[first]
    length = topology.lengths[edge]
    pull = (1.0 - 1.0 / length)[:, None] * (
        state.points[other] - state.points[members[owner]]
    )
    dimension = state.points.shape[1]
    total = np.zeros((dimension,), dtype=np.float64)
    for row, vector in zip(owner.tolist(), pull, strict=True):
        linear = np.asarray(
            context.vertex_exact_isometries[int(members[row])], dtype=np.float64
        )[:dimension, :dimension]
        total += linear.T @ vector
    return total / max(int(owner.size), 1)


def _relocation_deviation_gains(
    state: _State,
    topology: _Topology,
    vertices: np.ndarray,
    targets: np.ndarray,
    metrics: np.ndarray,
    proposals: np.ndarray,
    edge_orbits: np.ndarray | None,
    /,
) -> np.ndarray:
    """Whether each relocation proposal strictly improves its edges' deviations.

    Row ``i`` moves ``vertices[i]`` to ``targets[i]`` with ``metrics[i]``; rows
    sharing a proposal ID move together (the lifted members of one periodic
    vertex orbit). A proposal's affected edges are those incident to its moved
    rows, one per edge orbit when ``edge_orbits`` is given. Topology does not
    change, so the before and after edge sets coincide.
    """
    count = state.points.shape[0]
    owner, position = _expand(topology.neighbor_offsets, vertices)
    other = topology.neighbor_values[position]
    edges = _lookup(topology.codes, _codes(_sorted_pairs(vertices[owner], other)))
    # A neighbor moved by the same proposal is read at its proposed position.
    mover_keys = proposals * count + vertices
    mover_order = np.argsort(mover_keys, kind="stable")
    lookup_keys = proposals[owner] * count + other
    spot = np.minimum(
        np.searchsorted(mover_keys[mover_order], lookup_keys), mover_keys.size - 1
    )
    moved_other = mover_keys[mover_order[spot]] == lookup_keys
    other_rows = np.where(moved_other, count + mover_order[spot], other)
    after = _metric_lengths(
        np.concatenate((state.metric, metrics)),
        np.concatenate((state.points, targets)),
        np.stack((count + owner, other_rows), axis=1),
    )
    keys = edges if edge_orbits is None else edge_orbits[edges]
    gains = np.zeros((int(np.max(proposals)) + 1,), dtype=np.bool_)
    for proposal in np.unique(proposals).tolist():
        rows = np.flatnonzero(proposals[owner] == proposal)
        _, first = np.unique(keys[rows], return_index=True)
        gains[proposal] = _deviation_improves(
            topology.lengths[edges[rows[first]]], after[rows[first]]
        )
    return gains[proposals]


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


class _Backtracked(NamedTuple):
    """Accepted relocation of each proposal at its largest admissible scale."""

    success: np.ndarray
    target: np.ndarray
    metric: np.ndarray
    stencil_sources: np.ndarray
    stencil_weights: np.ndarray
    stencil_valid: np.ndarray
    stencil_exact: list[ExactSourceRow]
    rejected: np.ndarray
    ball_owner: np.ndarray
    ball_position: np.ndarray


def _backtracked_relocations(
    state: _State,
    topology: _Topology,
    vertices: np.ndarray,
    displacement: np.ndarray,
    mode: PredicateMode,
    source: _Source,
    edge_orbits: np.ndarray | None,
    /,
) -> _Backtracked:
    """Backtrack each single-vertex spring proposal to representability.

    Every proposal halves its step through ``_RELOCATION_SCALE_LANES`` scales per
    batch until a scale keeps all ball orientations valid and raises the star's
    minimum quality, or keeps it while strictly improving the edge deviations
    (counted once per edge orbit when ``edge_orbits`` is given).
    """
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
        metric_rows = _field_metric(
            source,
            target_rows,
            _proposal_metric(state, topology, vertex_bank, target_rows),
            mode,
        )
        status_rows, quality_rows = _proposal_checks(
            state, topology, vertex_bank, target_rows, metric_rows, mode
        )
        metric_bank = metric_rows.reshape((_RELOCATION_SCALE_LANES, vertices.size, 2, 2))
        status_bank = status_rows.reshape((_RELOCATION_SCALE_LANES, vertices.size))
        quality_bank = quality_rows.reshape((_RELOCATION_SCALE_LANES, vertices.size))
        quality_gain = quality_bank > before[None, :] + _IMPROVEMENT
        # A move that keeps its star's minimum quality may also be accepted
        # when it strictly improves the deviation measure of its edges.
        deviation_bank = np.zeros_like(quality_gain)
        lanes, columns = np.nonzero(
            active_bank
            & ~quality_gain
            & (quality_bank >= before[None, :])
            & (status_bank == _OK)
        )
        if lanes.size:
            deviation_bank[lanes, columns] = _relocation_deviation_gains(
                state,
                topology,
                vertices[columns],
                target_bank[lanes, columns],
                metric_bank[lanes, columns],
                np.arange(lanes.size, dtype=np.int64),
                edge_orbits,
            )
        improving_bank = (
            active_bank & (quality_gain | deviation_bank) & (status_bank == _OK)
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
            improving = active & (quality_gain[lane] | deviation_bank[lane])
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
    return _Backtracked(
        success,
        target_best,
        metric_best,
        stencil_sources_best,
        stencil_weights_best,
        stencil_valid_best,
        source_stencil_exact_best,
        rejected,
        ball_owner,
        ball_position,
    )


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
    (
        success,
        target_best,
        metric_best,
        stencil_sources_best,
        stencil_weights_best,
        stencil_valid_best,
        source_stencil_exact_best,
        rejected,
        ball_owner,
        ball_position,
    ) = _backtracked_relocations(
        state, topology, vertices, displacement, mode, source, None
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
    # A vertex that is its orbit's only lifted member, with the identity deck
    # element and no stabilizer, relocates exactly like a planar vertex; those
    # proposals backtrack together in scale lanes.
    orbit_ids, orbit_counts = np.unique(orbits, return_counts=True)
    orbit_sizes = dict(zip(orbit_ids.tolist(), orbit_counts.tolist(), strict=True))
    dimension = state.points.shape[1]
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension + 1))
        for row in range(dimension + 1)
    )
    single = np.asarray(
        [
            orbit_sizes[int(orbits[vertex])] == 1
            and not periodic_context.vertex_stabilizers[vertex]
            and periodic_context.vertex_exact_isometries[vertex] == identity
            for vertex in vertices.tolist()
        ],
        dtype=np.bool_,
    )
    if np.any(single):
        single_vertices = vertices[single]
        relocation = _backtracked_relocations(
            state,
            topology,
            single_vertices,
            displacement[single],
            mode,
            source,
            periodic_context.edge_orbits,
        )
        rejected.extend(relocation.rejected[~relocation.success].tolist())
        single_badness = badness[single]
        for row in np.flatnonzero(relocation.success).tolist():
            vertex = int(single_vertices[row])
            cells = topology.ball_cells[
                topology.ball_offsets[vertex] : topology.ball_offsets[vertex + 1]
            ]
            candidates.append(
                {
                    "vertices": np.asarray((vertex,), dtype=np.int64),
                    "target": relocation.target[row : row + 1].copy(),
                    "metric": relocation.metric[row : row + 1].copy(),
                    "source_stencils": (
                        relocation.stencil_sources[row : row + 1],
                        relocation.stencil_weights[row : row + 1],
                        relocation.stencil_valid[row : row + 1],
                        np.ones((1,), dtype=np.bool_),
                        (relocation.stencil_exact[row],),
                    ),
                    "cavity": frozenset(int(cell) for cell in cells.tolist()),
                    "quotient_id": int(
                        periodic_context.vertex_quotient_ids[int(orbits[vertex])]
                    ),
                    "badness": float(single_badness[row]),
                }
            )
    for orbit in np.unique(orbits[vertices[~single]]):
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
        # A seam orbit splits its quotient ball over several lifted members; its
        # spring pull and proposal metric are taken over the whole quotient ball.
        seam_orbit = members.size > 1 and bool(
            np.all(topology.kind[members] == _INTERIOR)
        )
        representative_displacement = (
            _periodic_orbit_pull(state, topology, periodic_context, members)
            if seam_orbit
            else member_displacement[representative_member]
        )
        scale = 1.0
        had_uncertain = False
        had_invalid = False
        accepted: dict[str, Any] | None = None
        while True:
            representative_target = (
                origin[representative_member] + scale * representative_displacement
            )
            if np.array_equal(representative_target, origin[representative_member]):
                break
            target = np.stack(
                [
                    _exact_periodic_point(
                        periodic_context.vertex_exact_isometries[int(vertex)],
                        representative_target,
                    )
                    for vertex in members.tolist()
                ]
            )
            if seam_orbit:
                # Only a member whose target lies in its own translated ball locates
                # the proposal; the deepest such member supplies the metric. No
                # admissible member means the step leaves the quotient ball.
                cells, weights, scores = _proposal_locations(
                    state, topology, members, target
                )
                admissible = np.isfinite(scores) & (scores >= 0.0)
                if not np.any(admissible):
                    had_invalid = True
                    scale *= 0.5
                    continue
                best = int(np.argmax(np.where(admissible, scores, -np.inf)))
                member_metric = _located_metric(
                    state, cells[best : best + 1], weights[best : best + 1]
                )[0]
                dimension = state.points.shape[1]
                linear = np.asarray(
                    periodic_context.vertex_exact_isometries[int(members[best])],
                    dtype=np.float64,
                )[:dimension, :dimension]
                pulled = linear.T @ member_metric @ linear
                representative_metric = 0.5 * (pulled + pulled.T)
            else:
                representative_metric = _proposal_metric(
                    state,
                    topology,
                    np.asarray((representative,), dtype=np.int64),
                    representative_target[None, :],
                )[0]
            representative_metric = _field_metric(
                source,
                representative_target[None, :],
                representative_metric[None],
                mode,
            )[0]
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
            # The quotient ball is the union of the member balls: a seam member
            # holds only part of it, so the orbit minimum must improve, or keep
            # while the orbit's quotient edges strictly improve their deviations.
            improving = float(np.min(after)) > float(np.min(before)) + _IMPROVEMENT
            if (
                not improving
                and bool(np.all(status == _OK))
                and float(np.min(after)) >= float(np.min(before))
            ):
                improving = bool(
                    _relocation_deviation_gains(
                        state,
                        topology,
                        members,
                        target,
                        metric,
                        np.zeros((members.size,), dtype=np.int64),
                        periodic_context.edge_orbits,
                    )[0]
                )
            valid = bool(np.all(status == _OK) and improving)
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


@final
class _StateTopologies:
    """Topology and periodic context of the most recent state.

    A round that applies nothing returns its input state object, so the next
    phase, the relocation sweep, and the pass loop reuse that state's topology
    and exact periodic context instead of re-staging the same edit.
    """

    def __init__(self, mesh: CellMesh, source: _Source, mode: PredicateMode, /) -> None:
        self._mesh = mesh
        self._source = source
        self._mode = mode
        self._state: _State | None = None
        self._value: tuple[_Topology, _PeriodicContext | None] | None = None

    def __call__(self, state: _State, /) -> tuple[_Topology, _PeriodicContext | None]:
        if self._value is None or state is not self._state:
            self._value = _periodic_topology(
                self._mesh,
                self._source,
                state,
                _topology(state, self._mode),
                self._mode,
            )
            self._state = state
        return self._value


def _run_phase(
    state: _State,
    counts: _Counts,
    phase: Any,
    mode: PredicateMode,
    topologies: _StateTopologies,
    source: _Source,
    /,
) -> Any:
    total = 0
    while True:
        topology, periodic_context = topologies(state)
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
    topologies: _StateTopologies,
    source: _Source,
    /,
) -> tuple[_State, _Counts, int, int]:
    """One pass: exhausted split/collapse/flip phases, then one relocation sweep.

    Returns the topology operations and the relocations applied separately.
    """
    reconnected = 0
    if topology_operations:
        for phase in (_split_round, _collapse_round, _flip_round):
            state, counts, count = _run_phase(
                state, counts, phase, mode, topologies, source
            )
            reconnected += count
    relocated = 0
    if relocation:
        done = np.zeros((state.points.shape[0],), dtype=np.bool_)
        while True:
            topology, periodic_context = topologies(state)
            state, counts, count, done = _relocation_round(
                state, topology, counts, done, mode, periodic_context, source
            )
            relocated += count
            if count == 0:
                break
    return state, counts, reconnected, relocated


def _out_of_range(topology: _Topology, /) -> np.ndarray:
    """Measured edges outside the unit band, by edge row."""
    lengths = topology.lengths
    unprotected = ~topology.edge_protected
    measured = unprotected if np.any(unprotected) else np.ones_like(unprotected)
    return measured & ((lengths < _LOWER) | (lengths > _UPPER))


# ------------------------------------------------------------------ assembly


def _locate(
    source: _Source, queries: np.ndarray, mode: PredicateMode, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Containing source cell and floating P1 weights for each supported query."""
    location_cells, location_ids = _locator(source)
    if queries.shape[0] * source.cell_lower.shape[0] <= _DIRECT_LOCATION_TESTS:
        # Few queries: the tree's item predicate on every (cell, query) box pair
        # gives the same pairs in the same (cell, query) order without building a
        # query tree.
        cell, query = np.nonzero(
            _host_boxes_overlap(
                source.cell_lower[:, None, :],
                source.cell_upper[:, None, :],
                queries[None, :, :],
                queries[None, :, :],
                True,
                0.0,
                _LOCATION_TOLERANCE,
            )
        )
    else:
        cell, query = bvh_overlap_pairs_host(
            source.cell_tree,
            prepare_bvh(queries, queries, dtype=jnp.float64),
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
    corners = source.points[location_cells[cell]]
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
    order = np.lexsort((location_ids[cell], violation, grade, query))
    first = order[np.searchsorted(query[order], query_rows)]
    exact_zero = certain[first] & (signs[first] == PredicateSign.ZERO)
    cells[query_rows] = cell[first]
    chosen[query_rows] = _normalized(np.where(exact_zero, 0.0, barycentric[first]))
    return cells, chosen, located


def _exact_cell_barycentric(
    source: _Source, cell: int, query: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    vertices = _locator(source)[0][cell]
    corners = [source.exact_points[int(vertex)] for vertex in vertices.tolist()]
    origin = corners[0]
    matrix = [
        [corners[column + 1][axis] - origin[axis] for column in range(2)]
        for axis in range(2)
    ]
    tail = solve_exact(
        matrix,
        [[query[0] - origin[0]], [query[1] - origin[1]]],
    )
    return (
        Fraction(1) - tail[0][0] - tail[1][0],
        tail[0][0],
        tail[1][0],
    )


def _periodic_image_exponents(cell_value: Any, /) -> tuple[np.ndarray, ...]:
    """Nearest nonzero deck exponents, ordered by length then lexicographically."""
    from itertools import product

    rank = int(identification_orders(cell_value).size)
    candidates = [
        np.asarray(values, dtype=np.int64)
        for values in product((-1, 0, 1), repeat=rank)
        if any(values)
    ]
    candidates.sort(key=lambda values: (int(np.abs(values).sum()), tuple(values)))
    return tuple(candidates)


def _exact_periodic_image(
    cell_value: Any, exponents: np.ndarray, point: np.ndarray, /
) -> tuple[Fraction, ...]:
    element = exact_periodic_element(cell_value, exponents)
    vector = tuple(Fraction(float(value)) for value in point) + (Fraction(1),)
    return tuple(
        sum(
            (element[row][column] * vector[column] for column in range(len(vector))),
            Fraction(0),
        )
        for row in range(len(point))
    )


def _periodic_image_location(
    source: _Source, point: np.ndarray, mode: PredicateMode, /
) -> tuple[int, tuple[Fraction, ...]] | None:
    """Locate an exact deck image of a query that rounds outside the lifted source.

    A generated vertex on the boundary of the lifted source region can round just
    outside it; one of its exact deck images then lies in a lifted source cell.
    The returned barycentric weights reproduce that exact image, which carries
    the same periodic source data as the query itself.
    """
    if source.periodic_cell is None:
        return None
    for exponents in _periodic_image_exponents(source.periodic_cell):
        image = _exact_periodic_image(source.periodic_cell, exponents, point)
        shadow = np.asarray([[float(value) for value in image]], dtype=np.float64)
        cells, _, _ = _locate(source, shadow, mode)
        if int(cells[0]) < 0:
            continue
        barycentric = _exact_cell_barycentric(source, int(cells[0]), image)
        if all(weight >= 0 for weight in barycentric):
            return int(cells[0]), barycentric
    return None


def _located_stencil_bank(
    source: _Source,
    points: np.ndarray,
    mode: PredicateMode,
    /,
    *,
    allow_invalid_rows: bool = False,
    stencil_width: int = 3,
    exact_points: tuple[tuple[Fraction, ...], ...] | None = None,
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
    for row, located_cell in enumerate(cells.tolist()):
        cell = int(located_cell)
        query = (
            tuple(Fraction(float(value)) for value in points[row].tolist())
            if exact_points is None
            else exact_points[row]
        )
        barycentric: tuple[Fraction, ...] | None = None
        if cell >= 0:
            barycentric = _exact_cell_barycentric(source, cell, query)
        if source.periodic_cell is None and (
            barycentric is None or any(weight < 0 for weight in barycentric)
        ):
            # Canonical rounding is monotone, so an exact point in a source
            # triangle lies inside its rounded-corner AABB. Search all such
            # cells exactly; never clip a negative barycentric coefficient.
            candidates = np.flatnonzero(
                np.all(points[row] >= source.cell_lower, axis=1)
                & np.all(points[row] <= source.cell_upper, axis=1)
            )
            identifiers = _locator(source)[1]
            for candidate in candidates[
                np.argsort(identifiers[candidates], kind="stable")
            ]:
                weights = _exact_cell_barycentric(source, int(candidate), query)
                if all(weight >= 0 for weight in weights):
                    cell, barycentric = int(candidate), weights
                    outside[row] = False
                    break
        if barycentric is None or any(weight < 0 for weight in barycentric):
            image = _periodic_image_location(source, points[row], mode)
            if image is None:
                outside[row] = True
                if allow_invalid_rows:
                    continue
                raise ValueError("A target vertex lies outside the source domain.")
            cell, barycentric = image
            outside[row] = False
        vertices = _locator(source)[0][cell]
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
            coordinate = source.exact_points[source_row]
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
        image = tuple(reproduced)
        if source.periodic_cell is None:
            # The exact point, not its carrier, is the geometry authority.
            # Downstream exact coordinate maps retain this source-stencil image.
            image = tuple(Fraction(float(value)) for value in image)
        if image != target and (
            source.periodic_cell is None
            or not any(
                _exact_periodic_image(
                    source.periodic_cell,
                    exponents,
                    state.points[int(order_rows[row])],
                )
                == tuple(reproduced)
                for exponents in _periodic_image_exponents(source.periodic_cell)
            )
        ):
            # A periodic row may reproduce an exact deck image of its target.
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
    if state.orbit_keys is not None:
        # A retired lifted member points at another member of its own vertex
        # orbit; its quotient vertex survives and is not a collapse.
        retired = np.all(
            state.orbit_keys[collapsed] == state.orbit_keys[state.collapse_to[collapsed]],
            axis=1,
        )
        collapsed = collapsed[~retired]
    if state.pole is not None:
        # A retired copy of a collapsed chart point points at another copy: the
        # physical vertex survives.
        twin = state.pole[collapsed]
        collapsed = collapsed[
            ~((twin >= 0) & (twin == state.pole[state.collapse_to[collapsed]]))
        ]
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
        vertex_orbit_keys=(
            None if state.orbit_keys is None else state.orbit_keys[order].copy()
        ),
        vertex_windings=(
            None if state.orbit_windings is None else state.orbit_windings[order].copy()
        ),
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
    permutation_offsets = np.asarray(edge_map.corner_permutation_offsets, dtype=np.int64)
    permutation_values = np.asarray(edge_map.corner_permutations, dtype=np.int64)
    if np.any(
        permutation_offsets[edge_target_rows + 1] - permutation_offsets[edge_target_rows]
        != 2
    ):
        raise ValueError("Periodic metric edge permutations require two corners.")
    edge_permutations = permutation_values[
        permutation_offsets[edge_target_rows][:, None] + np.arange(2)
    ]
    canonical_target_vertices = target_edge_vertices[
        edge_target_rows[:, None], edge_permutations
    ]
    edge_canonical_vertices = order[canonical_target_vertices]

    count = state.points.shape[0]
    vertex_orbits = np.full((count,), -1, dtype=np.int64)
    vertex_orbits[order] = np.asarray(vertex_map.lifted_to_quotient, dtype=np.int64)
    row_of = np.full((count,), -1, dtype=np.int64)
    row_of[order] = np.arange(order.size, dtype=np.int64)
    fixed_generators = np.asarray(bound_periodic.vertex_fixed_generators, dtype=np.bool_)
    vertex_windings = np.asarray(vertex_map.winding_shifts, dtype=np.int64)
    vertex_representatives = np.asarray(
        vertex_map.representative_lifted_rows, dtype=np.int64
    )
    representative_state_rows = np.full((count,), -1, dtype=np.int64)
    identity = tuple(
        tuple(
            Fraction(int(row == column)) for column in range(mesh.ambient_dimension + 1)
        )
        for row in range(mesh.ambient_dimension + 1)
    )
    vertex_exact_isometries = [identity] * count
    # Rows acted on by the zero winding: their exact action is the identity, so
    # their orbit checks reduce to binary64 equality with the root.
    vertex_identity = np.ones((count,), dtype=np.bool_)
    state_vertex_windings = np.zeros((count, vertex_windings.shape[1]), dtype=np.int64)
    vertex_stabilizers: list[tuple[tuple[tuple[Fraction, ...], ...], ...]] = [()] * count
    live = np.flatnonzero(vertex_orbits >= 0)
    live_orbits = vertex_orbits[live]
    target_rows = row_of[live]
    representative_targets = vertex_representatives[live_orbits]
    root_rows = order[representative_targets]
    representative_state_rows[live] = root_rows
    state_vertex_windings[live] = vertex_windings[target_rows]
    windings = vertex_windings[target_rows] - vertex_windings[representative_targets]
    # Per live row: 1 coordinates, 2 metric, 3 stabilizer; the first failing row
    # in (orbit, row) order names the refusal.
    failure = np.where(
        np.any(state.points[live] != state.points[root_rows], axis=1),
        1,
        np.where(
            np.any(state.metric[live] != state.metric[root_rows], axis=(1, 2)), 2, 0
        ),
    )
    for position in np.flatnonzero(np.any(windings != 0, axis=1)).tolist():
        state_row = int(live[position])
        root_row = int(root_rows[position])
        exact_isometry = exact_periodic_element(bound_periodic.cell, windings[position])
        vertex_exact_isometries[state_row] = exact_isometry
        vertex_identity[state_row] = False
        if not np.array_equal(
            state.points[state_row],
            _exact_periodic_point(exact_isometry, state.points[root_row]),
        ):
            failure[position] = 1
        elif not np.array_equal(
            state.metric[state_row],
            _exact_periodic_metric(exact_isometry, state.metric[root_row]),
        ):
            failure[position] = 2
        else:
            failure[position] = 0
    for position in np.flatnonzero(
        np.any(fixed_generators[target_rows], axis=1)
    ).tolist():
        if failure[position]:
            continue
        state_row = int(live[position])
        stabilizers = []
        for generator in np.flatnonzero(fixed_generators[target_rows[position]]):
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
                failure[position] = 3
                break
        vertex_stabilizers[state_row] = tuple(stabilizers)
    if np.any(failure):
        sequence = np.lexsort((live, live_orbits))
        raise ValueError(
            (
                "Periodic vertex coordinates must match their exact orbit action.",
                "Metric tensors must transform exactly over vertex orbits.",
                "Periodic vertex coordinates and metrics must respect exact stabilizers.",
            )[int(failure[sequence[failure[sequence] > 0][0]]) - 1]
        )
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
    edge_member_windings = edge_windings[edge_target_rows]
    orbit_count = edge_representatives.size
    orbit_rows = edge_state_by_target[edge_representatives]
    edge_representative_rows = np.full((orbit_count,), -1, dtype=np.int64)
    present = np.unique(edge_orbits)
    edge_representative_rows[present] = orbit_rows[present]
    # Per edge orbit: 1 controls disagree, 2 no state representative, 3 a
    # member's corners miss the exact orbit action (first failing orbit raises).
    edge_failure = np.zeros((orbit_count,), dtype=np.int64)
    edge_failure[
        _orbit_disagreement(edge_orbits, topology.edge_class, orbit_count)
        | _orbit_disagreement(edge_orbits, topology.edge_protected, orbit_count)
    ] = 1
    edge_failure[(edge_failure == 0) & (orbit_rows < 0)] = 2
    members = np.flatnonzero(edge_failure[edge_orbits] == 0)
    member_orbits = edge_orbits[members]
    # Only root-relative lifts are bit-consistent on non-dyadic lattices, so
    # the edge action is checked through the vertex root: the composed exact
    # element must reproduce each member corner.
    member_windings = (
        edge_windings[edge_target_rows[members]]
        - edge_windings[edge_representatives[member_orbits]]
    )
    nontrivial = np.any(member_windings != 0, axis=1)
    geometry = np.zeros((members.size,), dtype=np.bool_)
    exact_corners: list[tuple[int, int, int]] = []
    for corner in range(2):
        representative_vertex = edge_canonical_vertices[orbit_rows[member_orbits], corner]
        member_vertex = edge_canonical_vertices[members, corner]
        root = representative_state_rows[member_vertex]
        geometry |= root != representative_state_rows[representative_vertex]
        simple = ~nontrivial & vertex_identity[representative_vertex]
        geometry |= simple & (
            np.any(state.points[member_vertex] != state.points[root], axis=1)
            | np.any(state.metric[member_vertex] != state.metric[root], axis=(1, 2))
        )
        exact_corners.extend(
            (
                int(position),
                int(representative_vertex[position]),
                int(member_vertex[position]),
            )
            for position in np.flatnonzero(~simple & ~geometry)
        )
    for position, representative_vertex, member_vertex in exact_corners:
        if geometry[position]:
            continue
        root_row = int(representative_state_rows[member_vertex])
        composite = (
            _multiply(
                exact_periodic_element(bound_periodic.cell, member_windings[position]),
                vertex_exact_isometries[representative_vertex],
            )
            if nontrivial[position]
            else vertex_exact_isometries[representative_vertex]
        )
        geometry[position] = not (
            np.array_equal(
                state.points[member_vertex],
                _exact_periodic_point(composite, state.points[root_row]),
            )
            and np.array_equal(
                state.metric[member_vertex],
                _exact_periodic_metric(composite, state.metric[root_row]),
            )
        )
    edge_failure[np.unique(member_orbits[geometry])] = 3
    if np.any(edge_failure[present]):
        raise ValueError(
            (
                "Periodic edge controls must agree over each orbit.",
                "Periodic edge orbit has no state representative.",
                "Periodic edge geometry must respect its exact orbit action.",
            )[int(edge_failure[present][edge_failure[present] > 0][0]) - 1]
        )
    # Evaluate each periodic scalar on its exact canonical lift; coordinate and
    # tensor orbit consistency is checked above.
    edge_lengths = topology.lengths[orbit_rows[edge_orbits]]

    cell_state_by_target = np.full((target_cell_ids.size,), -1, dtype=np.int64)
    cell_state_by_target[cell_target_rows] = np.arange(
        state_cell_ids.size, dtype=np.int64
    )
    cell_representatives = np.asarray(cell_map.representative_lifted_rows, dtype=np.int64)
    cell_orbit_rows = cell_state_by_target[cell_representatives]
    cell_failure = np.zeros((cell_representatives.size,), dtype=np.int64)
    cell_failure[
        _orbit_disagreement(cell_orbits, state.cell_block, cell_failure.size)
        | _orbit_disagreement(cell_orbits, state.cell_region, cell_failure.size)
    ] = 1
    cell_failure[(cell_failure == 0) & (cell_orbit_rows < 0)] = 2
    cell_present = np.unique(cell_orbits)
    if np.any(cell_failure[cell_present]):
        raise ValueError(
            (
                "Periodic cell controls must agree over each orbit.",
                "Periodic cell orbit has no state representative.",
            )[int(cell_failure[cell_present][cell_failure[cell_present] > 0][0]) - 1]
        )
    cell_quality = _quality(edge_lengths[topology.cell_edges])[
        cell_orbit_rows[cell_orbits]
    ]
    if np.any(
        _orbit_disagreement(
            live_orbits, state.fixed[live], int(np.max(live_orbits, initial=-1)) + 1
        )
    ):
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
        edge_member_windings,
        edge_representative_rows,
        state_vertex_windings,
    )


def _collapse_refusal(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext | None,
    source: _Source,
    mode: PredicateMode,
    members: np.ndarray,
    removed: np.ndarray,
    kept: np.ndarray,
    /,
) -> str:
    """Why collapsing ``removed`` onto ``kept`` along lifted ``members`` is refused."""
    if not np.all(_collapse_admissible(topology, members, removed, kept)):
        return "inadmissible"
    if context is not None:
        vertex_members = np.bincount(
            context.vertex_orbits[context.vertex_orbits >= 0],
            minlength=context.vertex_quotient_ids.size,
        )
        if (
            np.any(
                (topology.edge_cells[members, 1] < 0)
                & (topology.edge_class[members] == 0)
            )
            or vertex_members[context.vertex_orbits[removed[0]]]
            != np.unique(removed).size
            or vertex_members[context.vertex_orbits[kept[0]]] != np.unique(kept).size
        ):
            refusal = _periodic_cover_collapse(
                state, topology, context, source, mode, members, removed, kept, 0, False
            )
            return refusal if isinstance(refusal, str) else "unselected"
    if not np.all(_collapse_link(topology, members, removed, kept)) or not np.all(
        _pole_link(state, topology, members, removed, kept)
    ):
        return "link"
    status, longest = _collapse_geometry(state, topology, removed, kept, mode)
    if np.any(status != _OK):
        return "invalid" if np.all(status != _UNCERTAIN) else "uncertain"
    if np.any(longest > _COLLAPSE_LIMIT):
        return "long"
    births = (state.source_rows[removed] < 0) | (state.source_rows[kept] < 0)
    if not np.all(_collapse_within_limit(births, longest, False)):
        # Above the split threshold; the stall escape that admits it was rolled
        # back because the next stall did not lower the stall potential.
        return "hysteresis"
    return "unselected"


def _flip_refusal(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext | None,
    edge: int,
    mode: PredicateMode,
    /,
) -> str:
    """Why flipping ``edge`` is refused."""
    first, second = (int(cell) for cell in topology.edge_cells[edge])
    if topology.edge_class[edge] > 0 or topology.edge_protected[edge]:
        return "classified"
    if second < 0:
        return "boundary" if context is None else "seam"
    if state.cell_region[first] != state.cell_region[second]:
        return "interface"
    found = _flip_candidates(state, topology, np.asarray((edge,), dtype=np.int64), mode)
    status = int(found.status[0])
    if status != _OK:
        return "uncertain" if status == _UNCERTAIN else "invalid"
    return "unselected" if bool(found.improving[0]) else "not-improving"


def _stall_findings(
    state: _State,
    topology: _Topology,
    context: _PeriodicContext | None,
    source: _Source,
    mode: PredicateMode,
    /,
) -> tuple[tuple[float, str], ...]:
    """Length and refusal reasons of each out-of-range edge (orbit) at a stall.

    A short edge reports both collapse directions (sorted, joined by ``+``) and
    its flip; a long edge reports its refused split.
    """
    lengths = topology.lengths
    unprotected = ~topology.edge_protected
    measured = unprotected if np.any(unprotected) else np.ones_like(unprotected)
    outside = np.flatnonzero(measured & ((lengths < _LOWER) | (lengths > _UPPER)))
    if context is not None:
        _, first = np.unique(context.edge_orbits[outside], return_index=True)
        outside = outside[np.sort(first)]
    records: list[tuple[float, str]] = []
    for edge in outside.tolist():
        length = float(lengths[edge])
        if length > _UPPER:
            records.append((length, "split=refused"))
            continue
        if context is None:
            members = np.asarray((edge,), dtype=np.int64)
            ends = topology.edges[members]
        else:
            members = np.flatnonzero(context.edge_orbits == context.edge_orbits[edge])
            ends = context.edge_canonical_vertices[members]
        collapse = "+".join(
            sorted(
                _collapse_refusal(
                    state,
                    topology,
                    context,
                    source,
                    mode,
                    members,
                    ends[:, column],
                    ends[:, 1 - column],
                )
                for column in (0, 1)
            )
        )
        flip = _flip_refusal(state, topology, context, edge, mode)
        records.append((length, f"collapse={collapse};flip={flip}"))
    return tuple(sorted(records))


def _evidence(
    topology: _Topology,
    counts: _Counts,
    passes: int,
    stalled: bool,
    topology_operations: bool,
    relocation: bool,
    mode: PredicateMode,
    stalled_edges: tuple[tuple[float, str], ...],
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
        stalled_edges=stalled_edges if stalled and not converged else (),
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
    collapsed_groups: np.ndarray | None = None,
    metric_field: Any = None,
    exact_source_points: tuple[tuple[Fraction, ...], ...] | None = None,
) -> LocalMetricOutcome:
    """Adapt a planar triangle mesh until it meets the unit-mesh criterion or stalls.

    No candidate or pass quota truncates the finite staged work. Every accepted
    split/collapse/flip phase exhausts its current candidates before proceeding.
    A full pass applies nothing when it reconnects nothing and either relocates
    nothing or, with topology operations, its relocation sweep leaves the
    out-of-range edge set unchanged: split and collapse candidates are exactly
    those edges, so further sweeps only polish shape quality by ever smaller
    accepted steps.
    Periodic sources group split, collapse, flip, and relocation proposals by
    quotient orbit, execute complete lifted groups, and validate the exact target
    stencil and lineage before returning an immutable candidate edit.

    Stall escape. Regular collapses keep the split/collapse hysteresis (a
    collapse touching a birth may not create an edge above sqrt(2)), so a run
    can stall on a short edge whose every collapse creates such an edge. At a
    full pass that applies nothing, the run measures the stall potential
    ``(out-of-band edges, quantized band excess)`` (``_stall_potential``) and
    runs one escape phase: every collapse up to ``_COLLAPSE_LIMIT``, births
    included (the next split phase splits its long edges). The escape stands
    when the run then reaches the unit criterion or a stall of strictly lower
    potential, which may escape again; otherwise the run returns the escaped
    stall's state as a reported stall. A stall whose escape phase applies
    nothing is reported as is.

    Termination: the potentials of the escaped stalls strictly decrease in the
    lexicographic order on pairs of natural numbers, which is well-founded, so
    escapes are finitely many; an escape phase removes a vertex (orbit) per
    collapse, so it is finite; between escapes the regular passes run unchanged.

    Collapsed chart points. ``collapsed_groups`` names, per vertex, the
    collapsed chart point it copies (``-1`` for regular vertices), e.g. a sphere
    pole with one copy per incident chart cell. Copies must be fixed, and their
    protected edges are chart cuts (``_cuts``): the regular vertices around the
    point stay free. Collapses and flips are decided on the physical surface,
    where the copies are one vertex (``_pole_link``), and a copy that loses its
    last cell retires onto a live copy.

    ``metric_field`` optionally evaluates the exact metric at generated chart
    points (split midpoints and relocation proposals) from their source
    location, replacing log-Euclidean interpolation of the vertex metrics; a
    singular chart such as a sphere pole needs it, because its pulled-back
    metric is not log-Euclidean linear in the chart.

    ``exact_source_points`` retains rational affine source coordinates across
    epochs; mesh coordinates must be their canonical binary64 roundings.
    Nonperiodic splits author an exact edge point from the split parameter and
    endpoint source stencils, then round its carrier once. Published exact
    stencils define the target geometry even when no binary64 point lies on
    the source edge. The receiving adaptation owner must retain that exact
    geometry rather than reinterpret its rounded carrier as a source point.
    """
    values = _validate_mesh(mesh, metric)
    _validate_controls(predicate_mode, topology_operations, relocation)
    vertex_count = mesh.coordinates.shape[0]
    edge_count = mesh.entity_set(1).entity_ids.shape[0]
    cell_count = mesh.entity_set(2).entity_ids.shape[0]
    fixed = _boolean_array(fixed_vertices, vertex_count, "fixed_vertices")
    groups = None
    if collapsed_groups is not None:
        groups = np.asarray(collapsed_groups)
        if groups.shape != (vertex_count,) or groups.dtype.kind not in "iu":
            raise ValueError(
                f"collapsed_groups must be integer-valued with shape ({vertex_count},)."
            )
        groups = groups.astype(np.int64)
        edges = np.asarray(mesh.connectivity.edges, dtype=np.int64)  # ty: ignore[unresolved-attribute]
        if (
            np.any(groups < -1)
            or np.any(~fixed[groups >= 0])
            or np.any(np.all(groups[edges] >= 0, axis=1))
        ):
            raise ValueError(
                "Collapsed chart point copies must be fixed, grouped by nonnegative "
                "IDs, and never adjacent to one another."
            )
    state, source = _initial_state(
        mesh,
        values,
        _integer_array(cell_classes, cell_count, "cell_classes"),
        _integer_array(edge_classes, edge_count, "edge_classes"),
        _boolean_array(protected_edges, edge_count, "protected_edges"),
        fixed,
        predicate_mode,
        groups,
        metric_field,
        exact_source_points,
    )
    topologies = _StateTopologies(mesh, source, predicate_mode)
    topology, periodic_context = topologies(state)
    _validate_classification(state, topology, periodic_context)
    counts = _Counts()
    passes = 0
    stalled = False
    # The last stall that escaped: its state, counts, passes, and potential.
    escaped_from: tuple[_State, _Counts, int, tuple[int, int]] | None = None
    periodic_topology_pending = mesh.periodic_topology is not None and topology_operations
    while not _unit(topology) or periodic_topology_pending:
        periodic_topology_pending = False
        passes += 1
        outside = _out_of_range(topology)
        state, counts, reconnected, relocated = _run_pass(
            state,
            counts,
            predicate_mode,
            topology_operations,
            relocation,
            topologies,
            source,
        )
        topology, periodic_context = topologies(state)
        if reconnected != 0 or (
            relocated != 0
            and not (
                topology_operations and np.array_equal(outside, _out_of_range(topology))
            )
        ):
            continue
        stalled = True
        if not topology_operations or _unit(topology):
            break
        potential = _stall_potential(topology)
        if escaped_from is not None and potential >= escaped_from[3]:
            # The last escape did not lower the stall potential: return its stall.
            state, counts, passes, _ = escaped_from
            topology, periodic_context = topologies(state)
            break
        stall = (state, counts, passes, potential)
        state, counts, escaped = _run_phase(
            state, counts, _escape_collapse_round, predicate_mode, topologies, source
        )
        if escaped == 0:
            break
        stalled = False
        escaped_from = stall
        topology, periodic_context = topologies(state)
    stalled_edges = (
        _stall_findings(state, topology, periodic_context, source, predicate_mode)
        if stalled and topology_operations
        else ()
    )
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
        stalled_edges,
    )
    return LocalMetricOutcome(edit, target_metric, evidence)


__all__ = [
    "LocalMetricEvidence",
    "LocalMetricOutcome",
    "execute_local_metric_adaptation",
]
