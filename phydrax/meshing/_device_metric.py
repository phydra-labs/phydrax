#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compiled fixed-capacity planar metric adaptation (``DEVICE_METRIC_2D``).

`prepare_device_metric_adaptation` binds one certified planar triangle source and
its vertex metric to a fixed-capacity device state: vertex and cell slots of the
`AdaptiveSimplexPolicy` capacity bucket in global-ID order, the organization
classification of the host route (per-half-edge feature classes, protection,
fixed vertices, cell regions), and the lineage the host edit assembly consumes
(vertex collapse targets and relocation marks, edge-lineage records, and the
cell-to-source-cell ancestry pairs).

`adapt_device_metric` is one module-level compiled entry point per layout. It
runs up to ``maximum_passes`` of the host metric passes (edge splits, collapses,
flips, then vertex relocation) without leaving the device: every candidate is
evaluated at once with FILTERED_DEVICE predicates, a deterministic maximal set
of operations with disjoint cavities is selected by unique static priorities in
bounded rounds of local minima, and the selection is applied in one vectorized
scatter with prefix-sum slot allocation. IDs and vertex slots are never reused
inside the epoch; replaced cell slots are packed in order before each topology
round, so slot order stays (provisional) ID order. An operation whose predicate
is uncertain is not applied and the call reports NEEDS_HOST_RESOLUTION; an
operation whose vertex ball exceeds the static cavity width is not evaluated and
is counted. Capacity overflow or a certified invalid cell refuses the whole
call: the input state is returned with the flag.

`commit_device_metric_adaptation` performs one device-to-host transfer,
reconstructs the host working state, and assembles the target through the host
edit assembly, so target IDs, lineage relations, the vertex stencil, and the P1
transfer follow the conventions of ``NATIVE_METRIC_2D``.
"""

from __future__ import annotations

import math
import time
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    _count,
    _FAILURES,
    _INDEX_LIMIT,
    _select,
    AdaptiveSimplexStatus,
    masked_simplex_facet_neighbors,
    masked_simplex_signature,
    MaskedSimplexMesh,
)
from ..geometry._predicates import (
    orient2d,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
)
from ..linalg import hermitian_exp, hermitian_log
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._lineage import MeshTransitionKind
from ._local_metric import (
    _assemble,
    _boolean_array,
    _codes,
    _COLLAPSE_LIMIT,
    _COLLAPSED,
    _CORNER,
    _CURVE,
    _distinct,
    _IMPROVEMENT,
    _initial_state,
    _integer_array,
    _INTERIOR,
    _lookup,
    _LOWER,
    _measured,
    _PHASE_ROUNDS,
    _REFINED,
    _RELOCATED,
    _RELOCATION_STEPS,
    _SELECTION_ROUNDS,
    _Source,
    _State,
    _SWAPPED,
    _topology as _host_topology,
    _UPPER,
    _validate_classification,
    _validate_mesh,
)
from ._metric import metric_edge_lengths


if TYPE_CHECKING:
    from ._adaptation import (
        _RouteOutcome,
        MeshAdaptationPolicy,
        MeshAdaptationResult,
        MetricMeshAdaptation,
        PreparedMeshAdaptation,
        RelocationMeshAdaptation,
    )
    from ._result import CellMeshingResult


# Static cavity width: incident cells of one vertex ball evaluated by a collapse
# or relocation. Larger balls are not evaluated (``rejected_cavity``).
_BALL_WIDTH = 16
# Edge-lineage records per (vertex + cell) slot: a planar triangulation has at
# most V + C edges; merged collapse records rarely exceed that bound.
_RECORDS_PER_SLOT = 2
# (cell, source cell) ancestry pairs per cell slot.
_PAIRS_PER_CELL = 8

# Operation statuses, worst last; the minimum over the two collapse directions
# of one edge is its tallied outcome.
_OK, _UNCERTAIN, _CAVITY, _INVALID, _INADMISSIBLE = 0, 1, 2, 3, 4
_CODE_SENTINEL = np.iinfo(np.int64).max
_RANK_SENTINEL = np.iinfo(np.int32).max
_NEXT = np.asarray((1, 2, 0), dtype=np.int32)
_PREVIOUS = np.asarray((2, 0, 1), dtype=np.int32)

# cursors: allocated vertex/cell slots, live records/pairs, next provisional IDs.
_VERTICES, _CELLS, _RECORDS, _PAIRS, _NEXT_VERTEX, _NEXT_CELL = range(6)
# controls: pass bound and the enabled phases.
_MAXIMUM_PASSES, _TOPOLOGY, _RELOCATION = range(3)
# counters: totals since preparation.
(
    _PASSES,
    _SPLITS,
    _COLLAPSES,
    _FLIPS,
    _RELOCATIONS,
    _REJECTED_UNCERTAIN,
    _REJECTED_INVALID,
    _REJECTED_CAVITY,
) = range(8)
# flags: union of applied status flags, then convergence and stall of the last call.
_STATUS, _CONVERGED, _STALLED = range(3)


# ------------------------------------------------------------------ public records


@final
class DeviceMetricLayout(StrictModule, NonTrainableState):
    """Static compile identity of one device metric adaptation capacity bucket.

    Vertex and cell capacities come from `AdaptiveSimplexPolicy.capacities`; the
    edge-lineage records hold ``2 (V + C)`` entries, the cell ancestry
    ``8 C`` (cell, source cell) pairs, and every vertex cavity at most
    ``ball_width`` cells. Every field is static: `adapt_device_metric` compiles
    once per layout, never per source topology, pass, or adaptation cycle.
    """

    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    record_capacity: int = eqx.field(static=True)
    ancestry_capacity: int = eqx.field(static=True)
    ball_width: int = eqx.field(static=True)
    mesh_signature_id: str = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(self, *, vertex_capacity: int, cell_capacity: int):
        vertices = _count(vertex_capacity, "vertex_capacity")
        cells = _count(cell_capacity, "cell_capacity")
        records = _RECORDS_PER_SLOT * (vertices + cells)
        pairs = _PAIRS_PER_CELL * cells
        steps = 1 + _RELOCATION_STEPS.size
        if max(vertices * steps, 6 * cells, 2 * records, 2 * pairs) > _INDEX_LIMIT:
            raise ValueError("Device metric capacities must address int32 slots.")
        mesh_signature = masked_simplex_signature(
            "triangle", 2, vertices, cells, np.dtype(np.float64)
        )
        self.vertex_capacity = vertices
        self.cell_capacity = cells
        self.record_capacity = records
        self.ancestry_capacity = pairs
        self.ball_width = _BALL_WIDTH
        self.mesh_signature_id = mesh_signature
        self.signature_id = canonical_fingerprint(
            {
                "kind": "device-metric-layout",
                "mesh": mesh_signature,
                "record_capacity": records,
                "ancestry_capacity": pairs,
                "ball_width": _BALL_WIDTH,
            }
        )


@final
class DeviceMetricState(StrictModule, NonTrainableState):
    """Dynamic arrays of one fixed-capacity device metric adaptation epoch.

    ``mesh`` is the solver-visible masked layout (vertex and cell slot order is
    global-ID order; vertices and cells created on device carry provisional IDs
    that the commit reissues). Per vertex slot: the SPD ``metric``, the source
    vertex row (``-1`` when created), the host ``fixed`` classification, whether
    it was relocated, and the surviving slot it collapsed into (``-1`` while
    alive). Per cell slot: region, source block, lineage rank (refined <
    relocated < swapped < collapsed), and the feature class and protection of
    the edge opposite each local vertex (equal on both halves of an edge).
    ``record_*`` are the live edge-lineage records
    (vertex-slot key, source edge row, rank), ``ancestry`` the sorted
    (cell slot, source cell row) pairs padded by ``(C, 0)``. ``cursors`` hold the
    allocated vertex/cell slots, live record/pair counts and the next provisional
    vertex/cell IDs; ``controls`` the pass bound and enabled phases; ``counters``
    the operation and rejection totals since preparation; ``flags`` the union of
    applied status flags and the convergence and stall of the last call.
    """

    mesh: MaskedSimplexMesh
    metric: Array
    source_rows: Array
    vertex_fixed: Array
    vertex_moved: Array
    collapse_to: Array
    cell_regions: Array
    cell_blocks: Array
    cell_ranks: Array
    edge_classes: Array
    edge_protected: Array
    record_keys: Array
    record_sources: Array
    record_ranks: Array
    ancestry: Array
    cursors: Array
    controls: Array
    counters: Array
    flags: Array

    def __init__(
        self,
        mesh: MaskedSimplexMesh,
        /,
        *,
        metric: ArrayLike,
        source_rows: ArrayLike,
        vertex_fixed: ArrayLike,
        vertex_moved: ArrayLike,
        collapse_to: ArrayLike,
        cell_regions: ArrayLike,
        cell_blocks: ArrayLike,
        cell_ranks: ArrayLike,
        edge_classes: ArrayLike,
        edge_protected: ArrayLike,
        record_keys: ArrayLike,
        record_sources: ArrayLike,
        record_ranks: ArrayLike,
        ancestry: ArrayLike,
        cursors: ArrayLike,
        controls: ArrayLike,
        counters: ArrayLike,
        flags: ArrayLike,
    ):
        if not isinstance(mesh, MaskedSimplexMesh):
            raise TypeError("mesh must be MaskedSimplexMesh.")
        if mesh.cell_kind != "triangle" or mesh.ambient_dimension != 2:
            raise ValueError("Device metric adaptation holds planar triangle meshes.")
        vertices, cells = mesh.vertex_capacity, mesh.cell_capacity
        records = jnp.asarray(record_keys).shape[0]
        pairs = jnp.asarray(ancestry).shape[0]
        specification = {
            "metric": (metric, (vertices, 2, 2), mesh.coordinates.dtype),
            "source_rows": (source_rows, (vertices,), jnp.int32),
            "vertex_fixed": (vertex_fixed, (vertices,), jnp.bool_),
            "vertex_moved": (vertex_moved, (vertices,), jnp.bool_),
            "collapse_to": (collapse_to, (vertices,), jnp.int32),
            "cell_regions": (cell_regions, (cells,), jnp.int32),
            "cell_blocks": (cell_blocks, (cells,), jnp.int32),
            "cell_ranks": (cell_ranks, (cells,), jnp.int32),
            "edge_classes": (edge_classes, (cells, 3), jnp.int32),
            "edge_protected": (edge_protected, (cells, 3), jnp.bool_),
            "record_keys": (record_keys, (records, 2), jnp.int32),
            "record_sources": (record_sources, (records,), jnp.int32),
            "record_ranks": (record_ranks, (records,), jnp.int32),
            "ancestry": (ancestry, (pairs, 2), jnp.int32),
            "cursors": (cursors, (6,), jnp.int64),
            "controls": (controls, (3,), jnp.int32),
            "counters": (counters, (8,), jnp.int64),
            "flags": (flags, (3,), jnp.int32),
        }
        arrays = {}
        for name, (value, shape, dtype) in specification.items():
            array = jnp.asarray(value)
            if array.shape != shape or array.dtype != dtype:
                raise TypeError(f"{name} must be {jnp.dtype(dtype).name} {shape}.")
            arrays[name] = array
        if records < 1 or pairs < 1:
            raise ValueError("Record and ancestry capacities must be positive.")
        self.mesh = mesh
        self.metric = arrays["metric"]
        self.source_rows = arrays["source_rows"]
        self.vertex_fixed = arrays["vertex_fixed"]
        self.vertex_moved = arrays["vertex_moved"]
        self.collapse_to = arrays["collapse_to"]
        self.cell_regions = arrays["cell_regions"]
        self.cell_blocks = arrays["cell_blocks"]
        self.cell_ranks = arrays["cell_ranks"]
        self.edge_classes = arrays["edge_classes"]
        self.edge_protected = arrays["edge_protected"]
        self.record_keys = arrays["record_keys"]
        self.record_sources = arrays["record_sources"]
        self.record_ranks = arrays["record_ranks"]
        self.ancestry = arrays["ancestry"]
        self.cursors = arrays["cursors"]
        self.controls = arrays["controls"]
        self.counters = arrays["counters"]
        self.flags = arrays["flags"]


@final
class DeviceMetricReport(StrictModule):
    """Traceable evidence of one `adapt_device_metric` call.

    ``status`` holds `AdaptiveSimplexStatus` flags: CAPACITY_EXCEEDED and
    INVALID_GEOMETRY refuse the call (the returned state is the input state and
    the operation counts are zero), PASS_LIMIT marks a call stopped at
    ``maximum_passes`` before the unit mesh, NEEDS_HOST_RESOLUTION an uncertain
    FILTERED_DEVICE predicate (the operation was not applied). Operation and
    rejection counts cover this call; rejections tally candidate evaluations
    (``rejected_cavity``: vertex balls wider than the static cavity width).
    Measurements cover the unprotected edges (all edges when every edge is
    protected) and active cells of the returned state.
    """

    status: Array
    passes: Array
    splits: Array
    collapses: Array
    flips: Array
    relocations: Array
    rejected_uncertain: Array
    rejected_invalid: Array
    rejected_cavity: Array
    converged: Array
    stalled: Array
    measured_edges: Array
    out_of_range_edges: Array
    minimum_metric_length: Array
    maximum_metric_length: Array
    unit_fraction: Array
    minimum_metric_quality: Array
    uncertain_cells: Array
    invalid_cells: Array

    @property
    def failed(self) -> Array:
        return (self.status & _FAILURES) != 0


@final
class DeviceMetricUpdate(StrictModule):
    """State and report of one compiled device metric adaptation call."""

    state: DeviceMetricState
    report: DeviceMetricReport


@final
class DeviceMetricEvidence(StrictModule, NonTrainableState):
    """Committed evidence of one device metric adaptation epoch.

    Operation and rejection counts are totals since preparation; ``status`` is
    the union of the applied `AdaptiveSimplexStatus` flags. Length and quality
    measurements cover the unprotected edges of the committed mesh (all edges
    when every edge is protected); ``converged`` certifies the unit-mesh
    criterion on them and ``stalled`` a last pass that applied nothing.
    """

    passes: int = eqx.field(static=True)
    splits: int = eqx.field(static=True)
    collapses: int = eqx.field(static=True)
    flips: int = eqx.field(static=True)
    relocations: int = eqx.field(static=True)
    rejected_uncertain: int = eqx.field(static=True)
    rejected_invalid: int = eqx.field(static=True)
    rejected_cavity: int = eqx.field(static=True)
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
    status: AdaptiveSimplexStatus = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
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
        rejected_cavity: int,
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
        status: AdaptiveSimplexStatus,
        layout_id: str,
    ):
        counts = {
            "passes": passes,
            "splits": splits,
            "collapses": collapses,
            "flips": flips,
            "relocations": relocations,
            "rejected_uncertain": rejected_uncertain,
            "rejected_invalid": rejected_invalid,
            "rejected_cavity": rejected_cavity,
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
        if converged and stalled:
            raise ValueError("A converged adaptation cannot be stalled.")
        if not isinstance(status, AdaptiveSimplexStatus):
            raise TypeError("status must be AdaptiveSimplexStatus.")
        if status & _FAILURES:
            raise ValueError("Committed evidence cannot carry a refused call.")
        if not isinstance(layout_id, str):
            raise TypeError("layout_id must be a str.")
        self.passes = passes
        self.splits = splits
        self.collapses = collapses
        self.flips = flips
        self.relocations = relocations
        self.rejected_uncertain = rejected_uncertain
        self.rejected_invalid = rejected_invalid
        self.rejected_cavity = rejected_cavity
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
        self.status = status
        self.layout_id = layout_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "device-metric-evidence",
                **counts,
                **flags,
                **measures,
                "status": int(status),
                "layout": layout_id,
            }
        )


class _Anchor(NamedTuple):
    """Host source description shared by preparation and commit."""

    source: _Source
    first_cell_id: int


@final
class PreparedDeviceMetricAdaptation(StrictModule, NonTrainableState):
    """One certified planar source bound to a device metric capacity bucket.

    ``adaptation`` is the prepared transaction (source, request, policy,
    resolved protection and organization), ``layout`` the static compile
    identity, and ``state`` the initial device state; `adapt_device_metric` may
    run on it any number of times before `commit_device_metric_adaptation`.
    """

    adaptation: PreparedMeshAdaptation
    layout: DeviceMetricLayout
    state: DeviceMetricState
    anchor: _Anchor
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        adaptation: PreparedMeshAdaptation,
        layout: DeviceMetricLayout,
        state: DeviceMetricState,
        anchor: _Anchor,
        /,
    ):
        self.adaptation = adaptation
        self.layout = layout
        self.state = state
        self.anchor = anchor
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-device-metric-adaptation",
                "adaptation": adaptation.prepared_id,
                "layout": layout.signature_id,
            }
        )


# ------------------------------------------------------------------ kernel records


class _Work(NamedTuple):
    """Array view of one state inside the compiled kernel."""

    coordinates: Array
    vertex_ids: Array
    vertex_active: Array
    cells: Array
    cell_ids: Array
    cell_active: Array
    metric: Array
    source_rows: Array
    vertex_fixed: Array
    vertex_moved: Array
    collapse_to: Array
    cell_regions: Array
    cell_blocks: Array
    cell_ranks: Array
    edge_classes: Array
    edge_protected: Array
    record_keys: Array
    record_sources: Array
    record_ranks: Array
    ancestry: Array
    cursors: Array
    controls: Array
    counters: Array
    flags: Array


def _work(state: DeviceMetricState, /) -> _Work:
    mesh = state.mesh
    return _Work(
        mesh.coordinates,
        mesh.vertex_ids,
        mesh.vertex_active,
        mesh.cells,
        mesh.cell_ids,
        mesh.cell_active,
        state.metric,
        state.source_rows,
        state.vertex_fixed,
        state.vertex_moved,
        state.collapse_to,
        state.cell_regions,
        state.cell_blocks,
        state.cell_ranks,
        state.edge_classes,
        state.edge_protected,
        state.record_keys,
        state.record_sources,
        state.record_ranks,
        state.ancestry,
        state.cursors,
        state.controls,
        state.counters,
        state.flags,
    )


def _state(work: _Work, /) -> DeviceMetricState:
    """Rebuild the public state; half-facet adjacency is recomputed from the cells."""

    mesh = MaskedSimplexMesh(
        work.coordinates,
        work.vertex_ids,
        work.vertex_active,
        work.cells,
        work.cell_ids,
        work.cell_active,
        masked_simplex_facet_neighbors(work.cells, work.cell_active),
    )
    return DeviceMetricState(
        mesh,
        metric=work.metric,
        source_rows=work.source_rows,
        vertex_fixed=work.vertex_fixed,
        vertex_moved=work.vertex_moved,
        collapse_to=work.collapse_to,
        cell_regions=work.cell_regions,
        cell_blocks=work.cell_blocks,
        cell_ranks=work.cell_ranks,
        edge_classes=work.edge_classes,
        edge_protected=work.edge_protected,
        record_keys=work.record_keys,
        record_sources=work.record_sources,
        record_ranks=work.record_ranks,
        ancestry=work.ancestry,
        cursors=work.cursors,
        controls=work.controls,
        counters=work.counters,
        flags=work.flags,
    )


class _Topology(NamedTuple):
    """Derived adjacency and classification of one round (half-edge indexed).

    Half-edge ``3 c + i`` is the edge of cell slot ``c`` opposite local vertex
    ``i``; ``canonical`` selects one half per active edge.
    """

    neighbors: Array
    canonical: Array
    ends: Array
    lengths: Array
    classes: Array
    protected: Array
    quality: Array
    codes: Array
    code_halves: Array
    kind: Array
    curve_class: Array
    curve_neighbors: Array
    straight: Array
    uniform: Array
    valence: Array
    ball: Array
    link: Array
    link_half: Array


class _Children(NamedTuple):
    """Cells created by one round; ``parents[:, 0]`` is the primary parent."""

    mask: Array
    rows: Array
    parents: Array
    rank: int
    classes: Array
    protected: Array


# ------------------------------------------------------------------ primitives


def _take(values: Array, index: Array, /) -> Array:
    return jnp.take_along_axis(values, index[..., None], axis=-1)[..., 0]


def _sorted_pair(first: Array, second: Array, /) -> Array:
    return jnp.stack((jnp.minimum(first, second), jnp.maximum(first, second)), axis=-1)


def _edge_code(first: Array, second: Array, vertex_capacity: int, /) -> Array:
    low = jnp.minimum(first, second).astype(jnp.int64)
    return low * vertex_capacity + jnp.maximum(first, second).astype(jnp.int64)


def _half_ends(cells: Array, /) -> Array:
    """Ascending vertex slots of the edge opposite every local vertex."""
    return _sorted_pair(
        cells[:, _NEXT].reshape((-1,)), cells[:, _PREVIOUS].reshape((-1,))
    )


def _lengths(work: _Work, ends: Array, /) -> Array:
    """Riemannian lengths through the owner; ascending ends make halves identical."""
    return metric_edge_lengths(work.metric, work.coordinates, ends)


def _log_euclidean(tensors: Array, weights: Array, /) -> Array:
    """``exp(sum_k w_k log M_k)`` of SPD samples ``(..., k, 2, 2)``.

    The formula of `interpolate_mesh_metric`; its SPD/convexity guard is a host
    callback (`eqx.error_if`) that compiled passes must not embed. Samples are
    metric tensors of the state and weights are convex by construction.
    """
    logarithm = hermitian_log(tensors).value
    return hermitian_exp(contract("...k,...kij->...ij", weights, logarithm)).value


def _quality(lengths: Array, /) -> Array:
    """Metric shape quality ``4 sqrt(3) area / sum(l^2)`` from edge lengths (Heron)."""
    a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    product = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c)
    return math.sqrt(3.0) * jnp.sqrt(jnp.maximum(product, 0.0)) / (a * a + b * b + c * c)


def _barycentric(triangles: Array, points: Array, /) -> Array:
    """Floating barycentric weights of ``points`` in ``(..., 3, 2)`` triangles."""

    def cross(origin, first, second):
        u = first - origin
        v = second - origin
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    weights = jnp.stack(
        (cross(points, b, c), cross(points, c, a), cross(points, a, b)), axis=-1
    )
    return weights / cross(a, b, c)[..., None]


def _metric_midpoint(metric: Array, points: Array, first: Array, second: Array, /):
    """Parameter of the metric-length midpoint under geometric size variation."""
    delta = points[second] - points[first]

    def length(rows):
        tensor = metric[rows]
        return jnp.sqrt(
            delta[:, 0] * delta[:, 0] * tensor[:, 0, 0]
            + 2.0 * delta[:, 0] * delta[:, 1] * tensor[:, 0, 1]
            + delta[:, 1] * delta[:, 1] * tensor[:, 1, 1]
        )

    ratio = length(second) / length(first)
    nearly_equal = jnp.abs(ratio - 1.0) <= 1.0e-6
    safe = jnp.where(nearly_equal, 2.0, ratio)
    return jnp.where(nearly_equal, 0.5, jnp.log(0.5 * (1.0 + safe)) / jnp.log(safe))


def _orientation(a: Array, b: Array, c: Array, /) -> Array:
    """``_OK`` for certified POSITIVE, ``_UNCERTAIN``, else ``_INVALID``."""
    result = orient2d(a, b, c, mode=PredicateMode.FILTERED_DEVICE)
    positive = result.signs == int(PredicateSign.POSITIVE)
    return jnp.where(
        ~result.certain, _UNCERTAIN, jnp.where(positive, _OK, _INVALID)
    ).astype(jnp.int32)


def _collinear(a: Array, b: Array, c: Array, /) -> Array:
    """``_OK`` for certified ZERO orientation, ``_UNCERTAIN``, else ``_INVALID``."""
    result = orient2d(a, b, c, mode=PredicateMode.FILTERED_DEVICE)
    zero = result.signs == int(PredicateSign.ZERO)
    return jnp.where(~result.certain, _UNCERTAIN, jnp.where(zero, _OK, _INVALID)).astype(
        jnp.int32
    )


def _ranks(valid: Array, *keys: Array) -> Array:
    """Unique priority ranks: valid operations first, then lexicographic keys."""
    count = valid.shape[0]
    index = jnp.arange(count, dtype=jnp.int32)
    ordered = jax.lax.sort(
        (jnp.where(valid, 0, 1).astype(jnp.int32), *keys, index),
        num_keys=len(keys) + 1,
        is_stable=True,
    )
    return jnp.zeros((count,), dtype=jnp.int32).at[ordered[-1]].set(index)


def _independent(rank: Array, valid: Array, items: Array, cell_capacity: int, /):
    """Greedy maximal set of valid operations with pairwise disjoint cell cavities.

    Each round selects every undecided operation whose rank is the minimum over
    all of its cells, then discards undecided operations touching selected
    cells. Unique ranks make simultaneous winners disjoint and the result equals
    the sequential greedy selection in rank order; operations left undecided
    after the round bound return in the next sub-round.
    """
    present = items >= 0
    gather = jnp.where(present, items, 0)
    own = jnp.broadcast_to(rank[:, None], items.shape)

    def undecided_remain(carry):
        undecided, _, rounds = carry
        return jnp.any(undecided) & (rounds < _SELECTION_ROUNDS)

    def select_round(carry):
        undecided, selected, rounds = carry
        live = undecided[:, None] & present
        best = (
            jnp.full((cell_capacity + 1,), _RANK_SENTINEL, dtype=jnp.int32)
            .at[jnp.where(live, items, cell_capacity)]
            .min(own)
        )
        losing = jnp.any(live & (best[gather] != own), axis=1)
        winners = undecided & ~losing
        taken = (
            jnp.zeros((cell_capacity + 1,), dtype=jnp.bool_)
            .at[jnp.where(winners[:, None] & present, items, cell_capacity)]
            .set(True)
        )
        blocked = jnp.any(present & taken[gather], axis=1)
        return undecided & ~winners & ~blocked, selected | winners, rounds + 1

    _, selected, _ = jax.lax.while_loop(
        undecided_remain,
        select_round,
        (valid, jnp.zeros(valid.shape, dtype=jnp.bool_), jnp.int32(0)),
    )
    return selected


def _tally(counters: Array, status, mask: Array, /) -> tuple[Array, Array]:
    """Add the rejections among ``mask``; uncertain ones need host resolution."""

    def count(condition):
        return jnp.sum(mask & condition, dtype=jnp.int64)

    uncertain = count(status == _UNCERTAIN)
    counters = (
        counters.at[_REJECTED_UNCERTAIN]
        .add(uncertain)
        .at[_REJECTED_CAVITY]
        .add(count(status == _CAVITY))
        .at[_REJECTED_INVALID]
        .add(count(status >= _INVALID))
    )
    bits = jnp.where(
        uncertain > 0, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION), 0
    ).astype(jnp.int32)
    return counters, bits


def _overflow(condition: Array, /) -> Array:
    return jnp.where(condition, int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED), 0).astype(
        jnp.int32
    )


def _edge_half(topology: _Topology, first: Array, second: Array, vertex_capacity: int):
    """Canonical half-edge of each queried vertex pair, or -1 when not an edge."""
    code = _edge_code(first, second, vertex_capacity)
    position = jnp.minimum(
        jnp.searchsorted(topology.codes, code), topology.codes.shape[0] - 1
    )
    return jnp.where(topology.codes[position] == code, topology.code_halves[position], -1)


# ------------------------------------------------------------------ topology


def _vertex_classes(work: _Work, canonical, ends, classes, protected, vertex_capacity):
    """Host vertex kinds: curve vertices carry exactly two edges of one class."""
    low, high = ends[:, 0], ends[:, 1]
    classified = canonical & (classes > 0)
    guarded_edge = canonical & protected

    def at(mask, index):
        return jnp.where(mask, index, vertex_capacity)

    shape = (vertex_capacity,)
    incident = (
        jnp.zeros(shape, dtype=jnp.int32)
        .at[at(classified, low)]
        .add(1, mode="drop")
        .at[at(classified, high)]
        .add(1, mode="drop")
    )
    label_low = (
        jnp.full(shape, _RANK_SENTINEL, dtype=jnp.int32)
        .at[at(classified, low)]
        .min(classes, mode="drop")
        .at[at(classified, high)]
        .min(classes, mode="drop")
    )
    label_high = (
        jnp.full(shape, -1, dtype=jnp.int32)
        .at[at(classified, low)]
        .max(classes, mode="drop")
        .at[at(classified, high)]
        .max(classes, mode="drop")
    )
    guarded = (
        work.vertex_fixed.at[at(guarded_edge, low)]
        .set(True, mode="drop")
        .at[at(guarded_edge, high)]
        .set(True, mode="drop")
    )
    curve = (incident == 2) & (label_low == label_high) & ~guarded
    kind = jnp.where(
        curve, _CURVE, jnp.where(guarded | (incident > 0), _CORNER, _INTERIOR)
    ).astype(jnp.int32)
    first = (
        jnp.full(shape, vertex_capacity, dtype=jnp.int32)
        .at[at(classified, low)]
        .min(high, mode="drop")
        .at[at(classified, high)]
        .min(low, mode="drop")
    )
    second = (
        jnp.full(shape, -1, dtype=jnp.int32)
        .at[at(classified, low)]
        .max(high, mode="drop")
        .at[at(classified, high)]
        .max(low, mode="drop")
    )
    neighbors = jnp.where(curve[:, None], jnp.stack((first, second), axis=1), -1)
    points = work.coordinates
    straight = jnp.where(
        curve,
        _collinear(
            points[jnp.maximum(neighbors[:, 0], 0)],
            points,
            points[jnp.maximum(neighbors[:, 1], 0)],
        ),
        _INVALID,
    )
    return kind, jnp.where(curve, label_low, 0), neighbors, straight


def _topology(work: _Work, layout: DeviceMetricLayout, /) -> _Topology:
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    width = layout.ball_width
    halves_count = 3 * cell_capacity
    cells = work.cells
    active = work.cell_active
    neighbors = masked_simplex_facet_neighbors(cells, active)
    other = neighbors.reshape((-1,))
    halves = jnp.arange(halves_count, dtype=jnp.int32)
    half_active = jnp.repeat(active, 3)
    canonical = half_active & ((other < 0) | (halves < other))
    ends = _half_ends(cells)
    lengths = _lengths(work, ends)
    classes = work.edge_classes.reshape((-1,))
    protected = work.edge_protected.reshape((-1,))
    cell_lengths = jnp.where(active[:, None], lengths.reshape((cell_capacity, 3)), 1.0)
    quality = jnp.where(active, _quality(cell_lengths), jnp.inf)
    codes, code_halves = jax.lax.sort(
        (
            jnp.where(
                canonical,
                _edge_code(ends[:, 0], ends[:, 1], vertex_capacity),
                _CODE_SENTINEL,
            ),
            halves,
        ),
        num_keys=1,
    )
    kind, curve_class, curve_neighbors, straight = _vertex_classes(
        work, canonical, ends, classes, protected, vertex_capacity
    )
    corner_vertex = jnp.where(half_active, cells.reshape((-1,)), vertex_capacity)
    corner_region = jnp.repeat(work.cell_regions, 3)
    region_low = (
        jnp.full((vertex_capacity,), _RANK_SENTINEL, dtype=jnp.int32)
        .at[corner_vertex]
        .min(corner_region, mode="drop")
    )
    region_high = (
        jnp.full((vertex_capacity,), -1, dtype=jnp.int32)
        .at[corner_vertex]
        .max(corner_region, mode="drop")
    )
    # Vertex balls: corners sorted by (vertex, corner) list incident cells in
    # ascending slot order.
    ordered_vertex, ordered_corner = jax.lax.sort((corner_vertex, halves), num_keys=2)
    vertices = jnp.arange(vertex_capacity, dtype=jnp.int32)
    starts = jnp.searchsorted(ordered_vertex, vertices, side="left")
    valence = jnp.searchsorted(ordered_vertex, vertices, side="right") - starts
    column = jnp.arange(width, dtype=jnp.int32)
    inside = column[None, :] < valence[:, None]
    corner = ordered_corner[
        jnp.minimum(starts[:, None] + column[None, :], halves_count - 1)
    ]
    ball = jnp.where(inside, corner // 3, -1)
    local = jnp.where(inside, corner % 3, 0)
    safe_ball = jnp.maximum(ball, 0)
    rows = cells[safe_ball]
    # Every neighbor once: the successor in each ball cell, plus the predecessor
    # across a boundary edge (manifold fans).
    following_half = 3 * safe_ball + (local + 2) % 3
    preceding_half = 3 * safe_ball + (local + 1) % 3
    link = jnp.concatenate(
        (
            jnp.where(inside, _take(rows, (local + 1) % 3), -1),
            jnp.where(
                inside & (other[preceding_half] < 0), _take(rows, (local + 2) % 3), -1
            ),
        ),
        axis=1,
    )
    return _Topology(
        neighbors=neighbors,
        canonical=canonical,
        ends=ends,
        lengths=lengths,
        classes=classes,
        protected=protected,
        quality=quality,
        codes=codes,
        code_halves=code_halves,
        kind=kind,
        curve_class=curve_class,
        curve_neighbors=curve_neighbors,
        straight=straight,
        uniform=region_low == region_high,
        valence=valence,
        ball=ball,
        link=link,
        link_half=jnp.concatenate((following_half, preceding_half), axis=1),
    )


# ------------------------------------------------------------------ state updates


def _ancestry(work: _Work, removed: Array, lineage: Array, layout: DeviceMetricLayout):
    """Children inherit the source cells of all listed parents; removed cells drop.

    ``lineage[c]`` lists the parents of new cell slot ``c`` (``-1`` padded). The
    parents' pairs are gathered by prefix sums over the sorted pair table, then
    merged with the surviving pairs, sorted, and deduplicated.
    """
    cell_capacity = layout.cell_capacity
    capacity = layout.ancestry_capacity
    pair_cells, pair_sources = work.ancestry[:, 0], work.ancestry[:, 1]
    slots = jnp.arange(cell_capacity, dtype=jnp.int32)
    starts = jnp.searchsorted(pair_cells, slots, side="left")
    stops = jnp.searchsorted(pair_cells, slots, side="right")
    parent = lineage.reshape((-1,))
    child = jnp.repeat(slots, 3)
    safe = jnp.maximum(parent, 0)
    counts = jnp.where(parent >= 0, stops[safe] - starts[safe], 0)
    ends = jnp.cumsum(counts)
    total = ends[-1]
    position = jnp.arange(capacity, dtype=jnp.int32)
    relation = jnp.minimum(
        jnp.searchsorted(ends, position, side="right"), parent.shape[0] - 1
    )
    offset = position - (ends[relation] - counts[relation])
    fresh = position < total
    source_row = jnp.clip(starts[safe[relation]] + offset, 0, capacity - 1)
    kept = (pair_cells < cell_capacity) & ~removed[
        jnp.minimum(pair_cells, cell_capacity - 1)
    ]
    ordered_cells, ordered_sources = jax.lax.sort(
        (
            jnp.concatenate(
                (
                    jnp.where(kept, pair_cells, cell_capacity),
                    jnp.where(fresh, child[relation], cell_capacity),
                )
            ),
            jnp.concatenate(
                (
                    jnp.where(kept, pair_sources, 0),
                    jnp.where(fresh, pair_sources[source_row], 0),
                )
            ),
        ),
        num_keys=2,
    )
    first = jnp.ones((1,), dtype=jnp.bool_)
    distinct = (ordered_cells < cell_capacity) & jnp.concatenate(
        (
            first,
            (ordered_cells[1:] != ordered_cells[:-1])
            | (ordered_sources[1:] != ordered_sources[:-1]),
        )
    )
    count = jnp.sum(distinct, dtype=jnp.int32)
    target = jnp.where(distinct, jnp.cumsum(distinct, dtype=jnp.int32) - 1, capacity)
    padding = jnp.stack(
        (
            jnp.full((capacity,), cell_capacity, dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=jnp.int32),
        ),
        axis=1,
    )
    pairs = padding.at[target].set(
        jnp.stack((ordered_cells, ordered_sources), axis=1), mode="drop"
    )
    return pairs, count, (total > capacity) | (count > capacity)


def _replace_cells(
    work: _Work, removed: Array, children: _Children, layout: DeviceMetricLayout, /
) -> tuple[_Work, Array]:
    """Deactivate replaced cells and write the children into prefix-sum slots.

    A child takes the region and block of its primary parent, the strongest
    lineage rank of the operation and all its parents, and their ancestry.
    """
    cell_capacity = layout.cell_capacity
    mask = children.mask
    count = jnp.sum(mask, dtype=jnp.int32)
    order = jnp.cumsum(mask, dtype=jnp.int32) - 1
    cursors = work.cursors
    slot = jnp.where(mask, cursors[_CELLS].astype(jnp.int32) + order, cell_capacity)
    parents = children.parents
    safe = jnp.maximum(parents, 0)
    inherited = jnp.max(jnp.where(parents >= 0, work.cell_ranks[safe], 0), axis=1)
    primary = safe[:, 0]
    lineage = (
        jnp.full((cell_capacity, 3), -1, dtype=jnp.int32)
        .at[slot]
        .set(parents, mode="drop")
    )
    pairs, pair_count, pair_overflow = _ancestry(work, removed, lineage, layout)
    work = work._replace(
        cells=work.cells.at[slot].set(children.rows, mode="drop"),
        cell_ids=work.cell_ids.at[slot].set(cursors[_NEXT_CELL] + order, mode="drop"),
        cell_active=(work.cell_active & ~removed).at[slot].set(True, mode="drop"),
        cell_regions=work.cell_regions.at[slot].set(
            work.cell_regions[primary], mode="drop"
        ),
        cell_blocks=work.cell_blocks.at[slot].set(work.cell_blocks[primary], mode="drop"),
        cell_ranks=work.cell_ranks.at[slot].set(
            jnp.maximum(inherited, children.rank), mode="drop"
        ),
        edge_classes=work.edge_classes.at[slot].set(children.classes, mode="drop"),
        edge_protected=work.edge_protected.at[slot].set(children.protected, mode="drop"),
        ancestry=pairs,
        cursors=cursors.at[_CELLS]
        .add(count)
        .at[_NEXT_CELL]
        .add(count)
        .at[_PAIRS]
        .set(pair_count),
    )
    return work, (cursors[_CELLS] + count > cell_capacity) | pair_overflow


def _store_records(
    work: _Work,
    keep: Array,
    keys: Array,
    sources: Array,
    ranks: Array,
    layout: DeviceMetricLayout,
    /,
    appended: tuple[Array, Array, Array, Array] | None = None,
) -> tuple[_Work, Array]:
    """Compact the kept records (then the appended ones) in order by prefix sums."""
    capacity = layout.record_capacity
    if appended is not None:
        extra_keep, extra_keys, extra_sources, extra_ranks = appended
        keep = jnp.concatenate((keep, extra_keep))
        keys = jnp.concatenate((keys, extra_keys))
        sources = jnp.concatenate((sources, extra_sources))
        ranks = jnp.concatenate((ranks, extra_ranks))
    count = jnp.sum(keep, dtype=jnp.int32)
    target = jnp.where(keep, jnp.cumsum(keep, dtype=jnp.int32) - 1, capacity)
    work = work._replace(
        record_keys=jnp.zeros((capacity, 2), dtype=jnp.int32)
        .at[target]
        .set(keys, mode="drop"),
        record_sources=jnp.full((capacity,), -1, dtype=jnp.int32)
        .at[target]
        .set(sources, mode="drop"),
        record_ranks=jnp.zeros((capacity,), dtype=jnp.int32)
        .at[target]
        .set(ranks, mode="drop"),
        cursors=work.cursors.at[_RECORDS].set(count),
    )
    return work, count > capacity


def _compact_cells(work: _Work, layout: DeviceMetricLayout, /) -> _Work:
    """Pack the active cell slots in order; cell IDs are never reused.

    Collapses and flips replace cells, so without compaction the slot demand
    would be every cell ever created. Order preservation keeps slot order equal
    to (provisional) ID order and the ancestry pairs sorted.
    """
    capacity = layout.cell_capacity
    active = work.cell_active
    target = jnp.where(active, jnp.cumsum(active, dtype=jnp.int32) - 1, capacity)

    def packed(values, fill):
        return jnp.full_like(values, fill).at[target].set(values, mode="drop")

    pair_cells = work.ancestry[:, 0]
    moved = jnp.where(
        pair_cells < capacity, target[jnp.minimum(pair_cells, capacity - 1)], capacity
    )
    return work._replace(
        cells=packed(work.cells, 0),
        cell_ids=packed(work.cell_ids, -1),
        cell_active=packed(active, False),
        cell_regions=packed(work.cell_regions, 0),
        cell_blocks=packed(work.cell_blocks, 0),
        cell_ranks=packed(work.cell_ranks, 0),
        edge_classes=packed(work.edge_classes, 0),
        edge_protected=packed(work.edge_protected, False),
        ancestry=work.ancestry.at[:, 0].set(moved),
        cursors=work.cursors.at[_CELLS].set(jnp.sum(active, dtype=jnp.int64)),
    )


def _live_records(work: _Work, /) -> Array:
    return jnp.arange(work.record_keys.shape[0]) < work.cursors[_RECORDS]


# ------------------------------------------------------------------ split


class _Sides(NamedTuple):
    """The (at most two) cells of every half-edge's edge: ``(u, v, apex)`` CCW."""

    cell: Array
    local: Array
    valid: Array
    u: Array
    v: Array
    apex: Array


def _split_sides(work: _Work, topology: _Topology, /) -> _Sides:
    other = topology.neighbors.reshape((-1,))
    halves = jnp.arange(other.shape[0], dtype=jnp.int32)
    half = jnp.stack((halves, jnp.maximum(other, 0)), axis=1)
    local = half % 3
    rows = work.cells[half // 3]
    return _Sides(
        cell=half // 3,
        local=local,
        valid=jnp.stack((jnp.ones(halves.shape, dtype=jnp.bool_), other >= 0), 1),
        u=_take(rows, (local + 1) % 3),
        v=_take(rows, (local + 2) % 3),
        apex=_take(rows, local),
    )


def _split_children(
    work: _Work, sides: _Sides, selected: Array, vertex_slot: Array, capacity: int, /
) -> tuple[_Children, Array]:
    """Children (u, m, apex) and (m, v, apex) of every cavity cell.

    The halves of the split edge keep its class; the edges to the apex are new.
    """
    local = sides.local

    def child_features(values):
        values = values[sides.cell]
        split = _take(values, local)
        following = _take(values, (local + 1) % 3)
        preceding = _take(values, (local + 2) % 3)
        fresh = jnp.zeros_like(split)
        return jnp.stack(
            (
                jnp.stack((fresh, preceding, split), axis=-1),
                jnp.stack((following, fresh, split), axis=-1),
            ),
            axis=2,
        ).reshape((-1, 3))

    middle = jnp.broadcast_to(vertex_slot[:, None], sides.u.shape)
    split_cells = selected[:, None] & sides.valid
    none = jnp.full(sides.cell.shape, -1, dtype=jnp.int32)
    parents = jnp.stack((sides.cell, none, none), axis=-1).reshape((-1, 3))
    children = _Children(
        mask=jnp.repeat(split_cells.reshape((-1,)), 2),
        rows=jnp.stack(
            (
                jnp.stack((sides.u, middle, sides.apex), axis=-1),
                jnp.stack((middle, sides.v, sides.apex), axis=-1),
            ),
            axis=2,
        ).reshape((-1, 3)),
        parents=jnp.repeat(parents, 2, axis=0),
        rank=_REFINED,
        classes=child_features(work.edge_classes),
        protected=child_features(work.edge_protected),
    )
    removed = (
        jnp.zeros((capacity + 1,), dtype=jnp.bool_)
        .at[jnp.where(split_cells, sides.cell, capacity)]
        .set(True)[:capacity]
    )
    return children, removed


def _split_round(work: _Work, done: Array, layout: DeviceMetricLayout, /):
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    candidate = topology.canonical & (topology.lengths > _UPPER) & ~topology.protected
    first, second = topology.ends[:, 0], topology.ends[:, 1]
    points = work.coordinates
    t = jnp.where(candidate, _metric_midpoint(work.metric, points, first, second), 0.5)
    middle = points[first] + t[:, None] * (points[second] - points[first])
    sides = _split_sides(work, topology)
    point = middle[:, None, :]
    checks = jnp.maximum(
        _orientation(points[sides.u], point, points[sides.apex]),
        _orientation(point, points[sides.v], points[sides.apex]),
    )
    status = jnp.max(jnp.where(sides.valid, checks, _OK), axis=1)
    counters, bits = _tally(work.counters, status, candidate)
    valid = candidate & (status == _OK)
    ids = work.vertex_ids
    rank = _ranks(valid, -topology.lengths, ids[first], ids[second])
    selected = _independent(
        rank, valid, jnp.where(sides.valid, sides.cell, -1), layout.cell_capacity
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    order = jnp.cumsum(selected, dtype=jnp.int32) - 1
    cursors = work.cursors
    vertex_slot = jnp.where(
        selected, cursors[_VERTICES].astype(jnp.int32) + order, vertex_capacity
    )
    tensors = _log_euclidean(
        jnp.stack((work.metric[first], work.metric[second]), axis=1),
        jnp.stack((1.0 - t, t), axis=1),
    )
    vertex_overflow = cursors[_VERTICES] + count > vertex_capacity
    work = work._replace(
        coordinates=work.coordinates.at[vertex_slot].set(middle, mode="drop"),
        metric=work.metric.at[vertex_slot].set(tensors, mode="drop"),
        vertex_ids=work.vertex_ids.at[vertex_slot].set(
            cursors[_NEXT_VERTEX] + order, mode="drop"
        ),
        vertex_active=work.vertex_active.at[vertex_slot].set(True, mode="drop"),
        cursors=cursors.at[_VERTICES].add(count).at[_NEXT_VERTEX].add(count),
        counters=counters.at[_SPLITS].add(count),
    )
    children, removed = _split_children(
        work, sides, selected, vertex_slot, layout.cell_capacity
    )
    work, cell_overflow = _replace_cells(work, removed, children, layout)
    # Records on a split edge become both halves.
    keys = work.record_keys
    half = _edge_half(topology, keys[:, 0], keys[:, 1], vertex_capacity)
    hit = _live_records(work) & (half >= 0) & selected[jnp.maximum(half, 0)]
    midpoint = vertex_slot[jnp.maximum(half, 0)]
    ranks = jnp.where(hit, jnp.maximum(work.record_ranks, _REFINED), work.record_ranks)
    work, record_overflow = _store_records(
        work,
        _live_records(work),
        jnp.where(hit[:, None], _sorted_pair(keys[:, 0], midpoint), keys),
        work.record_sources,
        ranks,
        layout,
        appended=(hit, _sorted_pair(keys[:, 1], midpoint), work.record_sources, ranks),
    )
    bits = bits | _overflow(vertex_overflow | cell_overflow | record_overflow)
    return work, done, bits, count


# ------------------------------------------------------------------ collapse


class _CollapseCandidates(NamedTuple):
    """Both directions of every half-edge: ``[removed = low | removed = high]``."""

    edge: Array
    removed: Array
    kept: Array
    ball: Array
    holds_kept: Array
    moved: Array
    longest: Array
    status: Array


def _collapse_candidates(
    work: _Work, topology: _Topology, layout: DeviceMetricLayout, /
) -> _CollapseCandidates:
    """Classification, link condition, moved-ball orientation, and new lengths."""
    vertex_capacity = layout.vertex_capacity
    width = layout.ball_width
    halves = jnp.arange(3 * layout.cell_capacity, dtype=jnp.int32)
    edge = jnp.concatenate((halves, halves))
    removed = jnp.concatenate((topology.ends[:, 0], topology.ends[:, 1]))
    kept = jnp.concatenate((topology.ends[:, 1], topology.ends[:, 0]))
    # Classification: interior vertices along unclassified edges inside one
    # region; straight curve vertices along an edge of their own curve class.
    label = topology.classes[edge]
    kind = topology.kind[removed]
    interior = (kind == _INTERIOR) & (label == 0) & topology.uniform[removed]
    curve = (kind == _CURVE) & (label > 0) & (label == topology.curve_class[removed])
    straight = jnp.where(curve, topology.straight[removed], _OK)
    cavity = (topology.valence[removed] > width) | (topology.valence[kept] > width)
    ball = topology.ball[removed]
    present = ball >= 0
    rows = work.cells[jnp.maximum(ball, 0)]
    holds_kept = jnp.any(rows == kept[:, None, None], axis=2)
    modified = present & ~holds_kept
    moved = jnp.where(rows == removed[:, None, None], kept[:, None, None], rows)
    points = work.coordinates
    orientation = jnp.max(
        jnp.where(
            modified,
            _orientation(
                points[moved[..., 0]], points[moved[..., 1]], points[moved[..., 2]]
            ),
            _OK,
        ),
        axis=1,
    )
    # Link condition: the common neighbors are exactly the apexes of the edge;
    # every other neighbor of the removed vertex forms a new edge.
    link = topology.link[removed]
    listed = (link >= 0) & (link != kept[:, None])
    joined = (
        _edge_half(
            topology,
            jnp.broadcast_to(kept[:, None], link.shape),
            jnp.maximum(link, 0),
            vertex_capacity,
        )
        >= 0
    )
    interior_edge = topology.neighbors.reshape((-1,))[edge] >= 0
    linked = jnp.sum(listed & joined, axis=1) == 1 + interior_edge
    fresh = listed & ~joined
    created = _lengths(
        work,
        _sorted_pair(
            jnp.broadcast_to(kept[:, None], link.shape), jnp.maximum(link, 0)
        ).reshape((-1, 2)),
    ).reshape(link.shape)
    longest = jnp.max(jnp.where(fresh, created, 0.0), axis=1)
    geometric = jnp.maximum(
        jnp.maximum(orientation, straight),
        jnp.where(linked & (longest <= _COLLAPSE_LIMIT), _OK, _INVALID),
    )
    status = jnp.where(
        interior | curve, jnp.where(cavity, _CAVITY, geometric), _INADMISSIBLE
    )
    return _CollapseCandidates(
        edge, removed, kept, ball, holds_kept, moved, longest, status
    )


def _collapse_children(
    work: _Work,
    topology: _Topology,
    found: _CollapseCandidates,
    selected: Array,
    layout: DeviceMetricLayout,
    /,
) -> tuple[_Children, Array]:
    """Survivors of each removed ball move onto the kept vertex as new cells.

    Each survivor absorbs the dropped edge cells of its own region; a survivor
    edge (removed, apex) merges with the dropped cell's (kept, apex) edge and
    keeps the stronger classification.
    """
    width = layout.ball_width
    cell_capacity = layout.cell_capacity
    ball = found.ball
    present = ball >= 0
    lost = selected[:, None] & present & found.holds_kept
    lost_order = jnp.sort(
        jnp.where(lost, jnp.arange(width, dtype=jnp.int32), width), axis=1
    )[:, :2]
    lost_cells = jnp.where(
        lost_order < width,
        jnp.take_along_axis(ball, jnp.minimum(lost_order, width - 1), axis=1),
        -1,
    )
    regions = work.cell_regions
    safe_ball = jnp.maximum(ball, 0)
    absorbed = jnp.where(
        (lost_cells[:, None, :] >= 0)
        & (
            regions[jnp.maximum(lost_cells, 0)][:, None, :]
            == regions[safe_ball][:, :, None]
        ),
        lost_cells[:, None, :],
        -1,
    )
    neighbor = topology.neighbors[safe_ball]
    neighbor_cell = jnp.where(neighbor >= 0, neighbor // 3, -1)
    merging = (neighbor_cell >= 0) & jnp.any(
        (neighbor_cell[..., None] == lost_cells[:, None, None, :])
        & (lost_cells[:, None, None, :] >= 0),
        axis=-1,
    )
    safe_neighbor = jnp.maximum(neighbor_cell, 0)
    position = jnp.argmax(
        work.cells[safe_neighbor] == found.removed[:, None, None, None], axis=-1
    ).astype(jnp.int32)
    own_classes = work.edge_classes[safe_ball]
    own_protected = work.edge_protected[safe_ball]
    children = _Children(
        mask=(selected[:, None] & present & ~found.holds_kept).reshape((-1,)),
        rows=found.moved.reshape((-1, 3)),
        parents=jnp.concatenate((ball[..., None], absorbed), axis=2).reshape((-1, 3)),
        rank=_COLLAPSED,
        classes=jnp.where(
            merging,
            jnp.maximum(own_classes, work.edge_classes[safe_neighbor, position]),
            own_classes,
        ).reshape((-1, 3)),
        protected=(
            own_protected | (merging & work.edge_protected[safe_neighbor, position])
        ).reshape((-1, 3)),
    )
    replaced = (
        jnp.zeros((cell_capacity + 1,), dtype=jnp.bool_)
        .at[jnp.where(selected[:, None] & present, ball, cell_capacity)]
        .set(True)[:cell_capacity]
    )
    return children, replaced


def _collapse_round(work: _Work, done: Array, layout: DeviceMetricLayout, /):
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    halves_count = 3 * layout.cell_capacity
    short = topology.canonical & (topology.lengths < _LOWER) & ~topology.protected
    found = _collapse_candidates(work, topology, layout)
    counters, bits = _tally(
        work.counters,
        jnp.minimum(found.status[:halves_count], found.status[halves_count:]),
        short,
    )
    valid = jnp.concatenate((short, short)) & (found.status == _OK)
    removed, kept = found.removed, found.kept
    ids = work.vertex_ids
    rank = _ranks(
        valid, topology.lengths[found.edge], found.longest, ids[removed], ids[kept]
    )
    selected = _independent(
        rank,
        valid,
        jnp.concatenate((found.ball, topology.ball[kept]), axis=1),
        layout.cell_capacity,
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    work = work._replace(counters=counters.at[_COLLAPSES].add(count))
    children, replaced = _collapse_children(work, topology, found, selected, layout)
    work, cell_overflow = _replace_cells(work, replaced, children, layout)
    target = jnp.where(selected, removed, vertex_capacity)
    work = work._replace(
        vertex_active=work.vertex_active.at[target].set(False, mode="drop"),
        collapse_to=work.collapse_to.at[target].set(kept, mode="drop"),
    )
    # Records follow the removed vertex; the collapsed edge's records vanish.
    remap = jnp.arange(vertex_capacity, dtype=jnp.int32).at[target].set(kept, mode="drop")
    gone = (
        jnp.zeros((vertex_capacity,), dtype=jnp.bool_).at[target].set(True, mode="drop")
    )
    keys = work.record_keys
    remapped = jnp.sort(remap[keys], axis=1)
    touched = jnp.any(gone[keys], axis=1)
    work, record_overflow = _store_records(
        work,
        _live_records(work) & (remapped[:, 0] != remapped[:, 1]),
        remapped,
        work.record_sources,
        jnp.where(touched, jnp.maximum(work.record_ranks, _COLLAPSED), work.record_ranks),
        layout,
    )
    bits = bits | _overflow(cell_overflow | record_overflow)
    return work, done, bits, count


# ------------------------------------------------------------------ flip


def _flip_children(
    work: _Work,
    halves: Array,
    other: Array,
    triangles: Array,
    selected: Array,
    capacity: int,
    /,
) -> tuple[_Children, Array]:
    """Children (u, x1, x0) and (v, x0, x1); the new diagonal is unclassified."""
    first_cell, first_local = halves // 3, halves % 3
    second_cell, second_local = other // 3, other % 3

    def child_features(values):
        first = values[first_cell]
        second = values[second_cell]
        fresh = jnp.zeros_like(first[:, 0])
        return jnp.stack(
            (
                jnp.stack(
                    (
                        fresh,
                        _take(first, (first_local + 2) % 3),
                        _take(second, (second_local + 1) % 3),
                    ),
                    axis=-1,
                ),
                jnp.stack(
                    (
                        fresh,
                        _take(second, (second_local + 2) % 3),
                        _take(first, (first_local + 1) % 3),
                    ),
                    axis=-1,
                ),
            ),
            axis=1,
        ).reshape((-1, 3))

    none = jnp.full(first_cell.shape, -1, dtype=jnp.int32)
    children = _Children(
        mask=jnp.repeat(selected, 2),
        rows=triangles.reshape((-1, 3)),
        parents=jnp.stack(
            (
                jnp.stack((first_cell, second_cell, none), axis=-1),
                jnp.stack((second_cell, first_cell, none), axis=-1),
            ),
            axis=1,
        ).reshape((-1, 3)),
        rank=_SWAPPED,
        classes=child_features(work.edge_classes),
        protected=child_features(work.edge_protected),
    )
    replaced = (
        jnp.zeros((capacity + 1,), dtype=jnp.bool_)
        .at[jnp.where(selected, first_cell, capacity)]
        .set(True)
        .at[jnp.where(selected, second_cell, capacity)]
        .set(True)[:capacity]
    )
    return children, replaced


def _flip_round(work: _Work, done: Array, layout: DeviceMetricLayout, /):
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    halves = jnp.arange(3 * cell_capacity, dtype=jnp.int32)
    other = topology.neighbors.reshape((-1,))
    safe_other = jnp.maximum(other, 0)
    first_cell, first_local = halves // 3, halves % 3
    second_cell, second_local = safe_other // 3, safe_other % 3
    regions = work.cell_regions
    candidate = (
        topology.canonical
        & (other >= 0)
        & (topology.classes == 0)
        & ~topology.protected
        & (regions[first_cell] == regions[second_cell])
    )
    first_rows = work.cells[first_cell]
    u = _take(first_rows, (first_local + 1) % 3)
    v = _take(first_rows, (first_local + 2) % 3)
    x0 = _take(first_rows, first_local)
    x1 = _take(work.cells[second_cell], second_local)
    triangles = jnp.stack(
        (jnp.stack((u, x1, x0), axis=-1), jnp.stack((v, x0, x1), axis=-1)), axis=1
    )
    lengths = _lengths(
        work,
        _sorted_pair(triangles[..., _NEXT], triangles[..., _PREVIOUS]).reshape((-1, 2)),
    ).reshape(triangles.shape)
    quality = _quality(lengths)
    after = jnp.minimum(quality[:, 0], quality[:, 1])
    before = jnp.minimum(topology.quality[first_cell], topology.quality[second_cell])
    points = work.coordinates
    status = jnp.maximum(
        _orientation(points[u], points[x1], points[x0]),
        _orientation(points[v], points[x0], points[x1]),
    )
    duplicate = _edge_half(topology, x0, x1, vertex_capacity) >= 0
    status = jnp.where(duplicate, _INVALID, status)
    improving = candidate & (after > before + _IMPROVEMENT)
    counters, bits = _tally(work.counters, status, improving)
    valid = improving & (status == _OK)
    ids = work.vertex_ids
    rank = _ranks(
        valid, before - after, ids[topology.ends[:, 0]], ids[topology.ends[:, 1]]
    )
    selected = _independent(
        rank, valid, jnp.stack((first_cell, second_cell), axis=1), cell_capacity
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    work = work._replace(counters=counters.at[_FLIPS].add(count))
    children, replaced = _flip_children(
        work, halves, safe_other, triangles, selected, cell_capacity
    )
    work, cell_overflow = _replace_cells(work, replaced, children, layout)
    keys = work.record_keys
    half = _edge_half(topology, keys[:, 0], keys[:, 1], vertex_capacity)
    flipped = (half >= 0) & selected[jnp.maximum(half, 0)]
    work, record_overflow = _store_records(
        work,
        _live_records(work) & ~flipped,
        keys,
        work.record_sources,
        work.record_ranks,
        layout,
    )
    bits = bits | _overflow(cell_overflow | record_overflow)
    return work, done, bits, count


# ------------------------------------------------------------------ relocation


def _relocation_targets(work: _Work, topology: _Topology, /):
    """Spring displacement toward unit metric lengths and each vertex's badness.

    Interior vertices use every neighbor; curve vertices only their two curve
    neighbors, so the displacement stays on the straight curve line.
    """
    points = work.coordinates
    link = topology.link
    listed = link >= 0
    length = jnp.where(listed, topology.lengths[topology.link_half], 1.0)
    badness = jnp.max(jnp.where(listed, jnp.maximum(length, 1.0 / length), 0.0), axis=1)
    curve = topology.kind == _CURVE
    along = (link == topology.curve_neighbors[:, :1]) | (
        link == topology.curve_neighbors[:, 1:]
    )
    use = listed & (~curve[:, None] | along)
    pull = jnp.where(
        use[..., None],
        (1.0 - 1.0 / length)[..., None]
        * (points[jnp.maximum(link, 0)] - points[:, None, :]),
        0.0,
    )
    displacement = jnp.sum(pull, axis=1) / jnp.maximum(jnp.sum(use, axis=1), 1)[:, None]
    return displacement, badness


def _relocation_round(work: _Work, done: Array, layout: DeviceMetricLayout, /):
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    steps = _RELOCATION_STEPS.size
    points = work.coordinates
    kind = topology.kind
    curve = kind == _CURVE
    movable = (
        (topology.valence > 0)
        & ~done
        & ((kind == _INTERIOR) | (curve & (topology.straight != _INVALID)))
    )
    cavity = topology.valence > layout.ball_width
    displacement, badness = _relocation_targets(work, topology)
    active = movable & ~cavity & jnp.any(displacement != 0.0, axis=1)
    target = points[:, None, :] + (
        jnp.asarray(_RELOCATION_STEPS)[None, :, None] * displacement[:, None, :]
    )
    ball = topology.ball
    present = ball >= 0
    rows = work.cells[jnp.maximum(ball, 0)]
    # Log-Euclidean P1 metric at each proposal in the ball cell containing it
    # best (largest minimum barycentric weight, lowest slot on ties).
    weights = _barycentric(points[rows][:, None], target[:, :, None, :])
    score = jnp.where(present[:, None, :], jnp.min(weights, axis=-1), -jnp.inf)
    best = jnp.argmax(score, axis=2)
    host_rows = jnp.take_along_axis(
        jnp.broadcast_to(rows[:, None], (*score.shape, 3)), best[..., None, None], axis=2
    )[:, :, 0]
    host_weights = jnp.take_along_axis(weights, best[..., None, None], axis=2)[:, :, 0]
    clipped = jnp.maximum(host_weights, 0.0)
    normalized = jnp.where(
        active[:, None, None],
        clipped / jnp.sum(clipped, axis=-1, keepdims=True),
        1.0 / 3.0,
    )
    proposal = _log_euclidean(work.metric[host_rows], normalized)
    # Proposal rows: the moved vertex takes row V + steps * v + s.
    image = (
        vertex_capacity
        + steps * jnp.arange(vertex_capacity, dtype=jnp.int32)[:, None]
        + jnp.arange(steps, dtype=jnp.int32)[None, :]
    )
    extended_points = jnp.concatenate((points, target.reshape((-1, 2))))
    extended_metric = jnp.concatenate((work.metric, proposal.reshape((-1, 2, 2))))
    vertex = jnp.arange(vertex_capacity, dtype=jnp.int32)[:, None, None, None]
    moved = jnp.where(rows[:, None] == vertex, image[:, :, None, None], rows[:, None])
    orientation = _orientation(
        extended_points[moved[..., 0]],
        extended_points[moved[..., 1]],
        extended_points[moved[..., 2]],
    )
    status = jnp.max(jnp.where(present[:, None, :], orientation, _OK), axis=2)
    neighbors = jnp.maximum(topology.curve_neighbors, 0)
    status = jnp.where(
        curve[:, None],
        jnp.maximum(
            status,
            _collinear(points[neighbors[:, :1]], target, points[neighbors[:, 1:]]),
        ),
        status,
    )
    lengths = metric_edge_lengths(
        extended_metric,
        extended_points,
        _sorted_pair(moved[..., _NEXT], moved[..., _PREVIOUS]).reshape((-1, 2)),
    ).reshape(moved.shape)
    after = jnp.min(
        jnp.where(
            present[:, None, :],
            _quality(jnp.where(present[:, None, :, None], lengths, 1.0)),
            jnp.inf,
        ),
        axis=2,
    )
    before = jnp.min(
        jnp.where(present, topology.quality[jnp.maximum(ball, 0)], jnp.inf), axis=1
    )
    improving = active[:, None] & (after > before[:, None] + _IMPROVEMENT)
    accepted = improving & (status == _OK)
    success = jnp.any(accepted, axis=1)
    rejected = jnp.where(
        jnp.any(improving & (status >= _INVALID), axis=1),
        _INVALID,
        jnp.where(jnp.any(improving & (status == _UNCERTAIN), axis=1), _UNCERTAIN, _OK),
    )
    counters, bits = _tally(work.counters, rejected, ~success)
    counters, _ = _tally(counters, _CAVITY, movable & cavity)
    rank = _ranks(success, -badness, work.vertex_ids)
    selected = _independent(rank, success, ball, cell_capacity)
    step = jnp.argmax(accepted, axis=1)
    chosen_point = jnp.take_along_axis(target, step[:, None, None], axis=1)[:, 0]
    chosen_metric = jnp.take_along_axis(proposal, step[:, None, None, None], axis=1)[:, 0]
    count = jnp.sum(selected, dtype=jnp.int32)
    touched_cells = jnp.any(selected[work.cells], axis=1)
    touched_records = _live_records(work) & jnp.any(selected[work.record_keys], axis=1)
    work = work._replace(
        coordinates=jnp.where(selected[:, None], chosen_point, points),
        metric=jnp.where(selected[:, None, None], chosen_metric, work.metric),
        vertex_moved=work.vertex_moved | selected,
        cell_ranks=jnp.where(
            touched_cells, jnp.maximum(work.cell_ranks, _RELOCATED), work.cell_ranks
        ),
        record_ranks=jnp.where(
            touched_records,
            jnp.maximum(work.record_ranks, _RELOCATED),
            work.record_ranks,
        ),
        counters=counters.at[_RELOCATIONS].add(count),
    )
    return work, done | selected, bits, count


# ------------------------------------------------------------------ passes


def _unit(work: _Work, /) -> Array:
    """Every measured (unprotected, else every) active edge in the unit range."""
    lengths = _lengths(work, _half_ends(work.cells))
    active = jnp.repeat(work.cell_active, 3)
    unprotected = active & ~work.edge_protected.reshape((-1,))
    measured = jnp.where(jnp.any(unprotected), unprotected, active)
    inside = (lengths >= _LOWER) & (lengths <= _UPPER)
    return jnp.all(~measured | inside)


def _phase(step, work: _Work, done: Array, bits: Array, layout, /):
    """Sub-rounds of one phase until a round applies nothing (bounded)."""

    def proceed(carry):
        _, _, bits, count, rounds, _ = carry
        return (rounds < _PHASE_ROUNDS) & (count > 0) & ((bits & _FAILURES) == 0)

    def sub_round(carry):
        work, done, bits, _, rounds, total = carry
        work, done, round_bits, count = step(work, done, layout)
        return work, done, bits | round_bits, count, rounds + 1, total + count

    work, done, bits, _, _, total = jax.lax.while_loop(
        proceed,
        sub_round,
        (work, done, bits, jnp.int32(1), jnp.int32(0), jnp.int32(0)),
    )
    return work, done, bits, total


def _pass(work: _Work, bits: Array, layout: DeviceMetricLayout, /):
    """Split, collapse, and flip phases, then relocation (each vertex once)."""
    done = jnp.zeros((layout.vertex_capacity,), dtype=jnp.bool_)

    def skipped(operand):
        work, bits = operand
        return work, bits, jnp.int32(0)

    def topological(operand):
        work, bits = operand
        total = jnp.int32(0)
        for step in (_split_round, _collapse_round, _flip_round):
            work, _, bits, count = _phase(step, work, done, bits, layout)
            total = total + count
        return work, bits, total

    def relocating(operand):
        work, bits = operand
        work, _, bits, count = _phase(_relocation_round, work, done, bits, layout)
        return work, bits, count

    work, bits, changed = jax.lax.cond(
        work.controls[_TOPOLOGY] != 0, topological, skipped, (work, bits)
    )
    work, bits, moved = jax.lax.cond(
        work.controls[_RELOCATION] != 0, relocating, skipped, (work, bits)
    )
    work = work._replace(counters=work.counters.at[_PASSES].add(1))
    return work, bits, changed + moved


def _geometry(work: _Work, /) -> tuple[Array, Array]:
    """FILTERED_DEVICE orientation of the active cells: uncertain and invalid counts."""
    corners = work.coordinates[work.cells]
    result = orient2d(
        corners[:, 0], corners[:, 1], corners[:, 2], mode=PredicateMode.FILTERED_DEVICE
    )
    active = work.cell_active
    positive = result.signs == int(PredicateSign.POSITIVE)
    return (
        jnp.sum(active & ~result.certain, dtype=jnp.int32),
        jnp.sum(active & result.certain & ~positive, dtype=jnp.int32),
    )


def _report(
    source: _Work,
    result: _Work,
    status: Array,
    stalled: Array,
    geometry: tuple[Array, Array],
    layout: DeviceMetricLayout,
    /,
) -> DeviceMetricReport:
    cells = result.cells
    active = result.cell_active
    other = masked_simplex_facet_neighbors(cells, active).reshape((-1,))
    halves = jnp.arange(other.shape[0], dtype=jnp.int32)
    canonical = jnp.repeat(active, 3) & ((other < 0) | (halves < other))
    lengths = _lengths(result, _half_ends(cells))
    unprotected = canonical & ~result.edge_protected.reshape((-1,))
    measured = jnp.where(jnp.any(unprotected), unprotected, canonical)
    inside = measured & (lengths >= _LOWER) & (lengths <= _UPPER)
    count = jnp.sum(measured, dtype=jnp.int32)
    within = jnp.sum(inside, dtype=jnp.int32)
    cell_lengths = jnp.where(
        active[:, None], lengths.reshape((layout.cell_capacity, 3)), 1.0
    )
    delta = (result.counters - source.counters).astype(jnp.int32)
    return DeviceMetricReport(
        status=status,
        passes=delta[_PASSES],
        splits=delta[_SPLITS],
        collapses=delta[_COLLAPSES],
        flips=delta[_FLIPS],
        relocations=delta[_RELOCATIONS],
        rejected_uncertain=delta[_REJECTED_UNCERTAIN],
        rejected_invalid=delta[_REJECTED_INVALID],
        rejected_cavity=delta[_REJECTED_CAVITY],
        converged=within == count,
        stalled=stalled,
        measured_edges=count,
        out_of_range_edges=count - within,
        minimum_metric_length=jnp.min(jnp.where(measured, lengths, jnp.inf)),
        maximum_metric_length=jnp.max(jnp.where(measured, lengths, 0.0)),
        unit_fraction=within.astype(jnp.float64) / jnp.maximum(count, 1),
        minimum_metric_quality=jnp.min(
            jnp.where(active, _quality(cell_lengths), jnp.inf)
        ),
        uncertain_cells=geometry[0],
        invalid_cells=geometry[1],
    )


def _adapt(layout: DeviceMetricLayout, state: DeviceMetricState, /) -> DeviceMetricUpdate:
    source = _work(state)
    maximum = source.controls[_MAXIMUM_PASSES]

    def proceed(carry):
        work, bits, passes, stalled = carry
        return (passes < maximum) & ~stalled & ((bits & _FAILURES) == 0) & ~_unit(work)

    def adaptation_pass(carry):
        work, bits, passes, _ = carry
        work, bits, applied = _pass(work, bits, layout)
        return work, bits, passes + 1, applied == 0

    work, bits, passes, stalled = jax.lax.while_loop(
        proceed,
        adaptation_pass,
        (source, jnp.int32(0), jnp.int32(0), jnp.bool_(False)),
    )
    converged = _unit(work)
    stalled = stalled & ~converged
    refused = (bits & _FAILURES) != 0
    uncertain, invalid = _geometry(work)
    bits = (
        bits
        | jnp.where(
            (passes >= maximum) & ~converged & ~stalled & ~refused,
            int(AdaptiveSimplexStatus.PASS_LIMIT),
            0,
        )
        | jnp.where(invalid > 0, int(AdaptiveSimplexStatus.INVALID_GEOMETRY), 0)
        | jnp.where(uncertain > 0, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION), 0)
    ).astype(jnp.int32)
    failed = (bits & _FAILURES) != 0
    work = work._replace(
        flags=work.flags.at[_STATUS]
        .set(work.flags[_STATUS] | bits)
        .at[_CONVERGED]
        .set(converged.astype(jnp.int32))
        .at[_STALLED]
        .set(stalled.astype(jnp.int32))
    )
    result = _select(failed, source, work)
    report = _report(
        source,
        result,
        bits,
        stalled & ~failed,
        _select(failed, _geometry(source), (uncertain, invalid)),
        layout,
    )
    return DeviceMetricUpdate(_state(result), report)


_compiled_adapt = eqx.filter_jit(_adapt)


def _check_state(layout: DeviceMetricLayout, state: DeviceMetricState, /) -> None:
    if not isinstance(layout, DeviceMetricLayout):
        raise TypeError("layout must be DeviceMetricLayout.")
    if not isinstance(state, DeviceMetricState):
        raise TypeError("state must be DeviceMetricState.")
    if (
        state.mesh.signature_id != layout.mesh_signature_id
        or state.record_keys.shape[0] != layout.record_capacity
        or state.ancestry.shape[0] != layout.ancestry_capacity
    ):
        raise ValueError("The state does not belong to this device metric layout.")


def adapt_device_metric(
    layout: DeviceMetricLayout, state: DeviceMetricState, /
) -> DeviceMetricUpdate:
    """Run up to ``maximum_passes`` device metric passes in one compiled call.

    Each pass runs the split, collapse, and flip phases (metric requests) and
    then relocation (when enabled), each in bounded sub-rounds, and stops at the
    unit mesh or a pass that applied nothing. One executable serves every call
    with the same layout; counters in the state accumulate across calls.
    """

    _check_state(layout, state)
    return _compiled_adapt(layout, state)


# ------------------------------------------------------------------ preparation


def _host_mode() -> PredicateMode:
    """Host predicates of preparation (source orientation) and commit (location)."""
    return resolve_host_predicate_mode(PredicateMode.EXACT)


def _edge_features(topology, host_cells: np.ndarray, /):
    """Feature class and protection of the edge opposite every local vertex."""
    first = host_cells[:, _NEXT].reshape((-1,))
    second = host_cells[:, _PREVIOUS].reshape((-1,))
    rows = _lookup(topology.codes, _codes(np.sort(np.stack((first, second), 1), 1)))
    shape = host_cells.shape
    return (
        topology.edge_class[rows].reshape(shape),
        topology.edge_protected[rows].reshape(shape),
    )


def _device_state(
    layout: DeviceMetricLayout, state: _State, topology, controls: np.ndarray, /
) -> DeviceMetricState:
    """Pad the host working state into global-ID-ordered slots of the bucket."""

    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    vertex_order = np.argsort(state.vertex_ids, kind="stable")
    cell_order = np.argsort(state.cell_ids, kind="stable")
    slot_of = np.empty(vertex_order.shape, dtype=np.int64)
    slot_of[vertex_order] = np.arange(vertex_order.size, dtype=np.int64)
    host_cells = state.cells[cell_order]
    vertex_count, cell_count = vertex_order.size, cell_order.size
    record_count = state.lineage_keys.shape[0]
    if record_count > layout.record_capacity or cell_count > layout.ancestry_capacity:
        raise ValueError("The prepared mesh exceeds its lineage capacities.")
    classes, protected = _edge_features(topology, host_cells)

    def padded(values, capacity, fill, dtype):
        array = np.asarray(values, dtype=dtype)
        result = np.full((capacity, *array.shape[1:]), fill, dtype=dtype)
        result[: array.shape[0]] = array
        return jnp.asarray(result)

    metric = np.broadcast_to(np.eye(2), (vertex_capacity, 2, 2)).copy()
    metric[:vertex_count] = state.metric[vertex_order]
    ancestry = np.zeros((layout.ancestry_capacity, 2), dtype=np.int32)
    ancestry[:, 0] = cell_capacity
    ancestry[:cell_count, 0] = np.arange(cell_count)
    ancestry[:cell_count, 1] = cell_order
    records = layout.record_capacity
    work = _Work(
        coordinates=padded(state.points[vertex_order], vertex_capacity, 0.0, np.float64),
        vertex_ids=padded(state.vertex_ids[vertex_order], vertex_capacity, -1, np.int64),
        vertex_active=padded(np.ones(vertex_count), vertex_capacity, False, np.bool_),
        cells=padded(slot_of[host_cells], cell_capacity, 0, np.int32),
        cell_ids=padded(state.cell_ids[cell_order], cell_capacity, -1, np.int64),
        cell_active=padded(np.ones(cell_count), cell_capacity, False, np.bool_),
        metric=jnp.asarray(metric),
        source_rows=padded(vertex_order, vertex_capacity, -1, np.int32),
        vertex_fixed=padded(state.fixed[vertex_order], vertex_capacity, False, np.bool_),
        vertex_moved=jnp.zeros((vertex_capacity,), dtype=jnp.bool_),
        collapse_to=jnp.full((vertex_capacity,), -1, dtype=jnp.int32),
        cell_regions=padded(state.cell_region[cell_order], cell_capacity, 0, np.int32),
        cell_blocks=padded(state.cell_block[cell_order], cell_capacity, 0, np.int32),
        cell_ranks=jnp.zeros((cell_capacity,), dtype=jnp.int32),
        edge_classes=padded(classes, cell_capacity, 0, np.int32),
        edge_protected=padded(protected, cell_capacity, False, np.bool_),
        record_keys=padded(
            np.sort(slot_of[state.lineage_keys], axis=1), records, 0, np.int32
        ),
        record_sources=padded(state.lineage_sources, records, -1, np.int32),
        record_ranks=jnp.zeros((records,), dtype=jnp.int32),
        ancestry=jnp.asarray(ancestry),
        cursors=jnp.asarray(
            np.asarray(
                (
                    vertex_count,
                    cell_count,
                    record_count,
                    cell_count,
                    state.next_vertex_id,
                    int(np.max(state.cell_ids)) + 1,
                ),
                dtype=np.int64,
            )
        ),
        controls=jnp.asarray(controls),
        counters=jnp.zeros((8,), dtype=jnp.int64),
        flags=jnp.zeros((3,), dtype=jnp.int32),
    )
    return _state(work)


def _prepared_metric(adaptation: PreparedMeshAdaptation, /):
    # Lazy: the adaptation transaction imports this module's evidence type.
    from ._adaptation import MetricMeshAdaptation

    policy = adaptation.policy
    constraints = adaptation.constraints
    mesh = adaptation.source.mesh
    mode = _host_mode()
    values = _validate_mesh(mesh, constraints.metric_values)
    vertex_count = mesh.coordinates.shape[0]
    edge_count = mesh.entity_set(1).entity_ids.shape[0]
    cell_count = mesh.entity_set(2).entity_ids.shape[0]
    state, source = _initial_state(
        mesh,
        values,
        _integer_array(constraints.cell_classes, cell_count, "cell_classes"),
        _integer_array(constraints.edge_classes, edge_count, "edge_classes"),
        _boolean_array(constraints.protected_edge_mask, edge_count, "protected_edges"),
        _boolean_array(constraints.fixed_vertex_mask, vertex_count, "fixed_vertices"),
        mode,
    )
    topology = _host_topology(state, mode)
    _validate_classification(state, topology)
    vertex_capacity, cell_capacity = policy.device_policy.capacities(
        vertex_count, cell_count
    )
    layout = DeviceMetricLayout(
        vertex_capacity=vertex_capacity, cell_capacity=cell_capacity
    )
    controls = np.asarray(
        (
            policy.maximum_passes,
            isinstance(adaptation.request, MetricMeshAdaptation),
            policy.relocation,
        ),
        dtype=np.int32,
    )
    device = _device_state(layout, state, topology, controls)
    return PreparedDeviceMetricAdaptation(
        adaptation, layout, device, _Anchor(source, int(np.max(state.cell_ids)) + 1)
    )


def prepare_device_metric_adaptation(
    source: CellMeshingResult,
    request: MetricMeshAdaptation | RelocationMeshAdaptation,
    /,
    *,
    policy: MeshAdaptationPolicy,
) -> PreparedDeviceMetricAdaptation:
    """Validate, classify, and pad one certified planar source into a device state.

    ``policy.route`` must be DEVICE_METRIC_2D; ``policy.device_policy`` fixes the
    capacity bucket. Protection, organization classes, and the metric are
    resolved exactly as for the host NATIVE_METRIC_2D route; metric requests
    enable the topology phases, ``policy.relocation`` the relocation phase.
    """

    from ._adaptation import (
        MeshAdaptationPolicy,
        MeshAdaptationRoute,
        prepare_mesh_adaptation,
    )

    if not isinstance(policy, MeshAdaptationPolicy):
        raise TypeError("policy must be MeshAdaptationPolicy.")
    if policy.route is not MeshAdaptationRoute.DEVICE_METRIC_2D:
        raise ValueError("Device metric adaptation requires the DEVICE_METRIC_2D route.")
    return _prepared_metric(prepare_mesh_adaptation(source, request, policy=policy))


# ------------------------------------------------------------------ commit


def _host_state(prepared: PreparedDeviceMetricAdaptation, host: DeviceMetricState, /):
    """The host working state of the device slots (vertex rows are vertex slots)."""
    mesh = host.mesh
    cursors = np.asarray(host.cursors)
    count = int(cursors[_VERTICES])
    active = np.flatnonzero(np.asarray(mesh.cell_active))
    row_of = np.full((prepared.layout.cell_capacity,), -1, dtype=np.int64)
    row_of[active] = np.arange(active.size, dtype=np.int64)
    cells = np.asarray(mesh.cells, dtype=np.int64)[active]
    classes = np.asarray(host.edge_classes, dtype=np.int64)[active].reshape((-1,))
    protected = np.asarray(host.edge_protected)[active].reshape((-1,))
    halves = np.sort(
        np.stack((cells[:, _NEXT], cells[:, _PREVIOUS]), axis=2).reshape((-1, 2)), axis=1
    )
    flagged = np.flatnonzero((classes > 0) | protected)
    features = flagged[_distinct(_codes(halves[flagged]))]
    records = int(cursors[_RECORDS])
    keys = np.sort(np.asarray(host.record_keys, dtype=np.int64)[:records], axis=1)
    sources = np.asarray(host.record_sources, dtype=np.int64)[:records]
    ranks = np.asarray(host.record_ranks, dtype=np.int64)[:records]
    # Identical records (same key, source edge, and rank) stem from one record
    # and share every later operation; the host deduplicates them eagerly.
    distinct = _distinct(_codes(keys), sources, -ranks)
    pairs = np.asarray(host.ancestry, dtype=np.int64)[: int(cursors[_PAIRS])]
    cell_ids = np.asarray(mesh.cell_ids, dtype=np.int64)[active]
    return _State(
        points=np.asarray(mesh.coordinates, dtype=np.float64)[:count],
        metric=np.asarray(host.metric, dtype=np.float64)[:count],
        vertex_ids=np.asarray(mesh.vertex_ids, dtype=np.int64)[:count],
        source_rows=np.asarray(host.source_rows, dtype=np.int64)[:count],
        fixed=np.asarray(host.vertex_fixed)[:count],
        moved=np.asarray(host.vertex_moved)[:count],
        alive=np.asarray(mesh.vertex_active)[:count],
        collapse_to=np.asarray(host.collapse_to, dtype=np.int64)[:count],
        cells=cells,
        cell_region=np.asarray(host.cell_regions, dtype=np.int64)[active],
        cell_block=np.asarray(host.cell_blocks, dtype=np.int64)[active],
        cell_ids=np.where(cell_ids < prepared.anchor.first_cell_id, cell_ids, -1),
        cell_rank=np.asarray(host.cell_ranks, dtype=np.int64)[active],
        ancestry_cells=row_of[pairs[:, 0]],
        ancestry_sources=pairs[:, 1],
        feature_keys=halves[features],
        feature_class=classes[features],
        feature_protected=protected[features],
        lineage_keys=keys[distinct],
        lineage_sources=sources[distinct],
        lineage_ranks=ranks[distinct],
        next_vertex_id=int(cursors[_NEXT_VERTEX]),
    )


def _evidence(
    prepared: PreparedDeviceMetricAdaptation, host: DeviceMetricState, topology, /
) -> DeviceMetricEvidence:
    counters = np.asarray(host.counters)
    flags = np.asarray(host.flags)
    controls = np.asarray(host.controls)
    lengths = _measured(topology)
    inside = (lengths >= _LOWER) & (lengths <= _UPPER)
    converged = bool(np.all(inside))
    return DeviceMetricEvidence(
        passes=int(counters[_PASSES]),
        splits=int(counters[_SPLITS]),
        collapses=int(counters[_COLLAPSES]),
        flips=int(counters[_FLIPS]),
        relocations=int(counters[_RELOCATIONS]),
        rejected_uncertain=int(counters[_REJECTED_UNCERTAIN]),
        rejected_invalid=int(counters[_REJECTED_INVALID]),
        rejected_cavity=int(counters[_REJECTED_CAVITY]),
        converged=converged,
        stalled=bool(flags[_STALLED]) and not converged,
        measured_edges=lengths.size,
        out_of_range_edges=int(np.count_nonzero(~inside)),
        minimum_metric_length=float(np.min(lengths)),
        maximum_metric_length=float(np.max(lengths)),
        unit_fraction=float(np.count_nonzero(inside) / lengths.size),
        minimum_metric_quality=float(np.min(topology.quality)),
        topology_operations=bool(controls[_TOPOLOGY]),
        relocation=bool(controls[_RELOCATION]),
        status=AdaptiveSimplexStatus(int(flags[_STATUS])),
        layout_id=prepared.layout.signature_id,
    )


def _outcome(
    prepared: PreparedDeviceMetricAdaptation, host: DeviceMetricState, /
) -> _RouteOutcome:
    # Lazy: the adaptation transaction imports this module's evidence type.
    from ._adaptation import (
        _finalize_native,
        _metric_status,
        _RouteOutcome,
        _target_metric,
        _unchanged,
    )

    adaptation = prepared.adaptation
    mode = _host_mode()
    state = _host_state(prepared, host)
    evidence = _evidence(prepared, host, _host_topology(state, mode))
    applied = evidence.splits + evidence.collapses + evidence.flips + evidence.relocations
    if applied == 0:
        # Nothing applied: the target is the source; the evidence carries the
        # convergence, stall, and pass counts.
        return _unchanged(adaptation, evidence, None)
    status = _metric_status(evidence, evidence.topology_operations)
    edit, metric = _assemble(state, prepared.anchor.source, mode)
    native = _finalize_native(
        adaptation, edit, MeshTransitionKind.REMESH, conservative=False
    )
    return _RouteOutcome(
        status,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        _target_metric(adaptation.request.metric, native.target.mesh, metric),
        evidence,
        None,
    )


def commit_device_metric_adaptation(
    prepared: PreparedDeviceMetricAdaptation, state: DeviceMetricState, /
) -> MeshAdaptationResult:
    """Commit one device epoch: one transfer, canonical target, lineage, evidence.

    The device slots are rebuilt into the host working state and assembled by
    the host edit assembly: surviving source vertices keep their IDs, new
    vertices and cells receive IDs in the host order, and the lineage
    (SPLIT_FROM, COLLAPSED_INTO, RELOCATED, REFINED_FROM, SWAPPED_FROM), the
    vertex stencil, organization inheritance, certification, and the P1
    transfer are exactly those of NATIVE_METRIC_2D.
    """

    from ._adaptation import _adaptation_result

    if not isinstance(prepared, PreparedDeviceMetricAdaptation):
        raise TypeError("prepared must be PreparedDeviceMetricAdaptation.")
    _check_state(prepared.layout, state)
    started = time.monotonic()
    host = jax.device_get(state)
    return _adaptation_result(prepared.adaptation, _outcome(prepared, host), started)


def _require_applied(report: DeviceMetricReport, /) -> None:
    """Raise the route failure of a refused device call (host boundary)."""

    status = AdaptiveSimplexStatus(int(report.status))
    if status & AdaptiveSimplexStatus.CAPACITY_EXCEEDED:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Device metric adaptation exceeds its capacity bucket; raise the "
            "AdaptiveSimplexPolicy capacities.",
            stage="device-metric",
        )
    if status & AdaptiveSimplexStatus.INVALID_GEOMETRY:
        raise MeshingFailure(
            MeshingFailureCategory.QUALITY_REJECTED,
            "Device metric adaptation certified an inverted or degenerate cell.",
            stage="device-metric",
        )


def _execute_device_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    """Prepare the device epoch, run the compiled passes once, and commit."""

    metric = _prepared_metric(prepared)
    update = adapt_device_metric(metric.layout, metric.state)
    _require_applied(update.report)
    return _outcome(metric, jax.device_get(update.state))


__all__ = [
    "DeviceMetricEvidence",
    "DeviceMetricLayout",
    "DeviceMetricReport",
    "DeviceMetricState",
    "DeviceMetricUpdate",
    "PreparedDeviceMetricAdaptation",
    "adapt_device_metric",
    "commit_device_metric_adaptation",
    "prepare_device_metric_adaptation",
]
