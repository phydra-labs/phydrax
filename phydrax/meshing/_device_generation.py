#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Process-local hybrid generation stages over native prepared simplex seeds.

The accelerator selects a bounded batch; it does not recover a PLC or query
CAD. Surface insertion uses native constrained reconnection. Tetra insertion
uses interior centroid stars, preserving every existing face and region. This
is a conforming size-refinement stage, not a Delaunay or sliver-removal claim.
No topology is published before the explicit exact host transaction barrier.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._geometry_predicates import orient2d, orient3d, PredicateMode, PredicateSign
from .._meshcore import exact_orient2d, exact_orient3d, MeshcoreStatus, surface_reconnect
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    masked_simplex_facet_neighbors,
    MaskedSimplexMesh,
)


class DeviceGenerationStatus(IntEnum):
    COMPLETE = 0
    CAPACITY_EXCEEDED = 1
    WORK_LIMIT = 2
    INVALID_GEOMETRY = 3


def _positive(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1 or value > np.iinfo(np.int32).max // 8:
        raise ValueError(f"{name} must be positive and addressable by int32.")
    return int(value)


@final
class DeviceGenerationLayout(StrictModule, NonTrainableState):
    """Static capacities; candidate overflow rejects rather than truncates a batch."""

    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    candidate_capacity: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertex_capacity: int,
        cell_capacity: int,
        candidate_capacity: int,
        maximum_work_units: int,
    ) -> None:
        vertices = _positive(vertex_capacity, "vertex_capacity")
        cells = _positive(cell_capacity, "cell_capacity")
        candidates = _positive(candidate_capacity, "candidate_capacity")
        work = _positive(maximum_work_units, "maximum_work_units")
        if candidates > cells:
            raise ValueError("candidate_capacity must not exceed cell_capacity.")
        self.vertex_capacity = vertices
        self.cell_capacity = cells
        self.candidate_capacity = candidates
        self.maximum_work_units = work


@final
class PreparedDeviceGeneration(StrictModule, NonTrainableState):
    """Native seed arrays padded once; semantic IDs are distinct from local slots.

    ``charts`` are meaningful only for surfaces. ``constrained`` flags the
    opposite edge of a triangle. Cell classes are region identities, not slots.
    All numerical data here belongs to the owning process; this API does not
    accept a distributed, non-fully-addressable mesh.
    """

    layout: DeviceGenerationLayout
    mesh: MaskedSimplexMesh
    charts: Array
    normals: Array
    constrained: Array
    cell_classes: Array
    target_sizes: Array


@final
class DeviceGenerationCandidates(StrictModule):
    """Fixed-size candidate batch and device phase evidence."""

    slots: Array
    active: Array
    charts: Array
    points: Array
    spacing: Array
    signs: Array
    count: Array
    uncertain: Array
    invalid: Array
    status: Array


@final
class DeviceGenerationEvidence(StrictModule, NonTrainableState):
    """Observed phase counts, without an all-device or measured-peak-memory claim."""

    status: DeviceGenerationStatus = eqx.field(static=True)
    device_candidates: int = eqx.field(static=True)
    exact_resolution_count: int = eqx.field(static=True)
    native_topology_work: int = eqx.field(static=True)
    geometry_queries: int = eqx.field(static=True)
    accepted_vertices: int = eqx.field(static=True)
    accepted_cells: int = eqx.field(static=True)
    host_barriers: int = eqx.field(static=True)
    transferred_bytes: int = eqx.field(static=True)
    retained_device_bytes: int = eqx.field(static=True)


@final
class DeviceGenerationUpdate(StrictModule, NonTrainableState):
    prepared: PreparedDeviceGeneration
    evidence: DeviceGenerationEvidence


def _pad(values: NDArray, capacity: int, fill: int | float | bool, /) -> NDArray:
    result = np.full((capacity,) + values.shape[1:], fill, dtype=values.dtype)
    result[: values.shape[0]] = values
    return result


def prepare_device_generation(
    layout: DeviceGenerationLayout,
    /,
    *,
    points: NDArray[np.float64],
    cells: NDArray[np.int32],
    vertex_ids: NDArray[np.int64],
    cell_ids: NDArray[np.int64],
    cell_classes: NDArray[np.int64],
    target_sizes: NDArray[np.float64],
    charts: NDArray[np.float64] | None = None,
    normals: NDArray[np.float64] | None = None,
    constrained: NDArray[np.bool_] | None = None,
) -> PreparedDeviceGeneration:
    """Prepare native surface/PLC seed data, never recovering identity from geometry."""
    if not isinstance(layout, DeviceGenerationLayout):
        raise TypeError("layout must be DeviceGenerationLayout.")
    arrays = (points, cells, vertex_ids, cell_ids, cell_classes, target_sizes)
    dtypes = (np.float64, np.int32, np.int64, np.int64, np.int64, np.float64)
    if any(
        not isinstance(a, np.ndarray) or a.dtype != t
        for a, t in zip(arrays, dtypes, strict=True)
    ):
        raise TypeError(
            "Prepared native arrays must be NumPy arrays with explicit canonical dtypes."
        )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or cells.ndim != 2
        or cells.shape[1] not in (3, 4)
    ):
        raise ValueError(
            "Expected physical (vertices, 3) points and triangle/tetrahedron rows."
        )
    nv, nc = points.shape[0], cells.shape[0]
    if not nv or not nc or nv > layout.vertex_capacity or nc > layout.cell_capacity:
        raise ValueError("Native seed arrays exceed the capacity bucket or are empty.")
    if (
        vertex_ids.shape != (nv,)
        or cell_ids.shape != (nc,)
        or cell_classes.shape != (nc,)
        or target_sizes.shape != (nc,)
    ):
        raise ValueError("Native seed metadata must align with its vertex/cell rows.")
    if (
        np.any(vertex_ids < 0)
        or np.any(cell_ids < 0)
        or np.any(np.diff(vertex_ids) <= 0)
        or np.any(np.diff(cell_ids) <= 0)
    ):
        raise ValueError("Native seed IDs must be nonnegative and strictly increasing.")
    if (
        np.max(vertex_ids) == np.iinfo(np.int64).max
        or np.max(cell_ids) == np.iinfo(np.int64).max
    ):
        raise ValueError(
            "Native seed IDs leave no room for deterministic new identities."
        )
    if (
        not np.all(np.isfinite(points))
        or not np.all(np.isfinite(target_sizes))
        or np.any(target_sizes <= 0)
    ):
        raise ValueError(
            "Coordinates must be finite and target sizes finite and positive."
        )
    if (
        np.any(cells < 0)
        or np.any(cells >= nv)
        or np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0)
    ):
        raise ValueError("Cells must reference distinct in-range vertex slots.")
    if cells.shape[1] == 3:
        if charts is None or normals is None or constrained is None:
            raise ValueError(
                "Surface seeds require native charts, normals and constrained edges."
            )
        if (
            charts.shape != (nv, 2)
            or charts.dtype != np.float64
            or normals.shape != (nv, 3)
            or normals.dtype != np.float64
            or constrained.shape != cells.shape
            or constrained.dtype != np.bool_
        ):
            raise TypeError(
                "Surface chart/normal/constraint arrays have incompatible contracts."
            )
        if not np.all(np.isfinite(charts)) or not np.all(np.isfinite(normals)):
            raise ValueError("Surface charts and normals must be finite.")
        corners = charts[cells]
        signs = exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
    else:
        if charts is not None or normals is not None or constrained is not None:
            raise ValueError("Tetrahedral seeds do not carry surface chart data.")
        charts = np.zeros((nv, 2), dtype=np.float64)
        normals = np.zeros((nv, 3), dtype=np.float64)
        constrained = np.zeros(cells.shape, dtype=np.bool_)
        corners = points[cells]
        signs = exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
    if np.any(signs != PredicateSign.POSITIVE):
        raise ValueError("Native seed cells must have exactly positive orientation.")
    return _pack(
        layout,
        points,
        cells,
        vertex_ids,
        cell_ids,
        cell_classes,
        target_sizes,
        charts,
        normals,
        constrained,
    )


def _pack(
    layout: DeviceGenerationLayout,
    points: NDArray[np.float64],
    cells: NDArray[np.int32],
    vertex_ids: NDArray[np.int64],
    cell_ids: NDArray[np.int64],
    cell_classes: NDArray[np.int64],
    target_sizes: NDArray[np.float64],
    charts: NDArray[np.float64],
    normals: NDArray[np.float64],
    constrained: NDArray[np.bool_],
    /,
) -> PreparedDeviceGeneration:
    """Pack already validated transaction arrays without redoing exact work."""
    nv, nc = points.shape[0], cells.shape[0]
    vc, cc = layout.vertex_capacity, layout.cell_capacity
    rows = jnp.asarray(_pad(cells, cc, 0))
    active = jnp.arange(cc, dtype=jnp.int32) < nc
    mesh = MaskedSimplexMesh(
        jnp.asarray(_pad(points, vc, 0.0)),
        jnp.asarray(_pad(vertex_ids, vc, -1)),
        jnp.arange(vc, dtype=jnp.int32) < nv,
        rows,
        jnp.asarray(_pad(cell_ids, cc, -1)),
        active,
        masked_simplex_facet_neighbors(rows, active),
    )
    return PreparedDeviceGeneration(
        layout,
        mesh,
        jnp.asarray(_pad(charts, vc, 0.0)),
        jnp.asarray(_pad(normals, vc, 0.0)),
        jnp.asarray(_pad(constrained, cc, False)),
        jnp.asarray(_pad(cell_classes, cc, -1)),
        jnp.asarray(_pad(target_sizes, cc, 1.0)),
    )


def _candidates(prepared: PreparedDeviceGeneration, /) -> DeviceGenerationCandidates:
    mesh = prepared.mesh
    corners = mesh.coordinates[mesh.cells]
    width = mesh.cells.shape[1]
    # Every local edge is evaluated in a fixed, linear-size workset.
    pairs = np.asarray(
        [(i, j) for i in range(width) for j in range(i + 1, width)], dtype=np.int32
    )
    lengths = jnp.linalg.norm(corners[:, pairs[:, 1]] - corners[:, pairs[:, 0]], axis=-1)
    longest = jnp.max(lengths, axis=1)
    score = longest / prepared.target_sizes
    chart_corners = prepared.charts[mesh.cells]
    if width == 3:
        predicate = orient2d(
            chart_corners[:, 0],
            chart_corners[:, 1],
            chart_corners[:, 2],
            mode=PredicateMode.FILTERED_DEVICE,
        )
    else:
        predicate = orient3d(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            corners[:, 3],
            mode=PredicateMode.FILTERED_DEVICE,
        )
    invalid = (
        mesh.cell_active & predicate.certain & (predicate.signs != PredicateSign.POSITIVE)
    )
    requested = mesh.cell_active & (score > 1.0 + 1.0e-9)
    count = jnp.sum(requested, dtype=jnp.int32)
    # Stable priority: severity, then the seed's semantic cell-ID ordering.
    order = jnp.argsort(jnp.where(requested, -score, jnp.inf), stable=True)
    slots = order[: prepared.layout.candidate_capacity]
    active = requested[slots]
    status = jnp.where(
        jnp.any(invalid),
        int(DeviceGenerationStatus.INVALID_GEOMETRY),
        jnp.where(
            count > prepared.layout.candidate_capacity,
            int(DeviceGenerationStatus.CAPACITY_EXCEEDED),
            0,
        ),
    ).astype(jnp.int32)
    return DeviceGenerationCandidates(
        slots,
        active,
        jnp.mean(chart_corners[slots], axis=1),
        jnp.mean(corners[slots], axis=1),
        0.2 * jnp.minimum(longest[slots], prepared.target_sizes[slots]),
        predicate.signs,
        count,
        jnp.sum(mesh.cell_active & ~predicate.certain, dtype=jnp.int32),
        jnp.sum(invalid, dtype=jnp.int32),
        status,
    )


_compiled_candidates = eqx.filter_jit(_candidates)


def _check_prepared(prepared: PreparedDeviceGeneration, /) -> None:
    if not isinstance(prepared, PreparedDeviceGeneration):
        raise TypeError("prepared must be PreparedDeviceGeneration.")
    if (
        prepared.mesh.vertex_capacity != prepared.layout.vertex_capacity
        or prepared.mesh.cell_capacity != prepared.layout.cell_capacity
    ):
        raise ValueError("Prepared topology does not belong to its capacity bucket.")
    if any(
        isinstance(leaf, Array) and not leaf.is_fully_addressable
        for leaf in jax.tree_util.tree_leaves(prepared)
    ):
        raise ValueError(
            "Generation barriers require process-local, fully addressable arrays."
        )


def evaluate_device_generation_candidates(
    prepared: PreparedDeviceGeneration, /
) -> DeviceGenerationCandidates:
    """One stable compiled candidate evaluation, with no host scalar decisions."""
    _check_prepared(prepared)
    return _compiled_candidates(prepared)


def _bytes(tree: StrictModule, /) -> int:
    return sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, (Array, np.ndarray))
    )


def execute_device_generation_round(
    prepared: PreparedDeviceGeneration,
    /,
    *,
    surface_points: NDArray[np.float64] | None = None,
    surface_normals: NDArray[np.float64] | None = None,
    surface_geometry_queries: int = 0,
    candidates: DeviceGenerationCandidates | None = None,
) -> DeviceGenerationUpdate:
    """Exact process-local barrier, native topology stage, then atomic publication.

    Surface geometry is evaluated by the provider in ONE bounded batch at the
    returned candidate charts; pass arrays of the complete candidate capacity.
    Report the provider's actual query count in ``surface_geometry_queries``;
    precomputed geometry performs zero queries here. A failed
    stage returns the identical source preparation. This does not gather any
    remote mesh, but the local mesh and candidate buffers cross to the host.
    """
    _check_prepared(prepared)
    if isinstance(surface_geometry_queries, bool) or not isinstance(
        surface_geometry_queries, (int, np.integer)
    ):
        raise TypeError("surface_geometry_queries must be an integer.")
    if surface_geometry_queries < 0:
        raise ValueError("surface_geometry_queries must be nonnegative.")
    batch = (
        evaluate_device_generation_candidates(prepared)
        if candidates is None
        else candidates
    )
    expected = prepared.layout.candidate_capacity
    if (
        batch.slots.shape != (expected,)
        or batch.points.shape != (expected, 3)
        or batch.charts.shape != (expected, 2)
        or batch.signs.shape != (prepared.layout.cell_capacity,)
    ):
        raise ValueError("Candidates do not belong to the prepared capacity bucket.")
    # One explicit transfer of this owning process's bounded transaction data.
    host, candidate = jax.device_get((prepared, batch))
    transferred = _bytes(prepared) + _bytes(batch)
    count = int(np.asarray(candidate.count))
    status = DeviceGenerationStatus(int(np.asarray(candidate.status)))
    exact, work, inserted, created = 0, 0, 0, 0
    target = prepared
    if status is DeviceGenerationStatus.COMPLETE and count:
        target, status, exact, work, inserted, created = _host_round(
            prepared,
            host,
            candidate,
            surface_points,
            surface_normals,
        )
    queries = int(surface_geometry_queries)
    evidence = DeviceGenerationEvidence(
        status,
        count,
        exact,
        work,
        queries,
        inserted,
        created,
        1,
        transferred,
        _bytes(target),
    )
    return DeviceGenerationUpdate(target, evidence)


@eqx.filter_jit
def _initial_surface_signs(charts: Array, /) -> Array:
    return orient2d(
        charts[:, 0],
        charts[:, 1],
        charts[:, 2],
        mode=PredicateMode.FILTERED_DEVICE,
    ).signs


def certify_device_surface_initial(
    layout: DeviceGenerationLayout,
    vertices: NDArray[np.float64],
    triangle_charts: NDArray[np.float64],
    expected_signs: NDArray[np.int32],
    /,
) -> DeviceGenerationEvidence:
    """Resolve bounded initial chart predicates before collective publication.

    Native constrained construction owns physical/source fidelity. This stage
    owns the filtered-device/exact-host orientation barrier, not another
    triangulation or an all-device construction claim.
    """
    if not isinstance(layout, DeviceGenerationLayout):
        raise TypeError("layout must be DeviceGenerationLayout.")
    if (
        vertices.dtype != np.float64
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or triangle_charts.dtype != np.float64
        or triangle_charts.ndim != 3
        or triangle_charts.shape[1:] != (3, 2)
        or expected_signs.dtype != np.int32
        or expected_signs.shape != (triangle_charts.shape[0],)
    ):
        raise ValueError("Initial surface arrays have incompatible canonical contracts.")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(triangle_charts)):
        raise ValueError("Initial surface coordinates and charts must be finite.")
    if np.any(
        (expected_signs != PredicateSign.POSITIVE)
        & (expected_signs != PredicateSign.NEGATIVE)
    ):
        raise ValueError("Each initial triangle requires its authored chart winding.")
    count = triangle_charts.shape[0]
    status = DeviceGenerationStatus.COMPLETE
    if vertices.shape[0] > layout.vertex_capacity or count > layout.cell_capacity:
        status = DeviceGenerationStatus.CAPACITY_EXCEEDED
    elif count > layout.maximum_work_units:
        status = DeviceGenerationStatus.WORK_LIMIT
    if status is not DeviceGenerationStatus.COMPLETE:
        return DeviceGenerationEvidence(status, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    exact, barriers, transferred, retained = 0, 0, 0, 0
    for first in range(0, count, layout.candidate_capacity):
        charts = triangle_charts[first : first + layout.candidate_capacity]
        padded = _pad(charts, layout.candidate_capacity, 0.0)
        device = jnp.asarray(padded)
        signs = np.asarray(jax.device_get(_initial_surface_signs(device)))[
            : charts.shape[0]
        ].copy()
        barriers += 1
        transferred += padded.nbytes + layout.candidate_capacity * signs.dtype.itemsize
        retained = max(
            retained, padded.nbytes + layout.candidate_capacity * signs.dtype.itemsize
        )
        uncertain = signs == PredicateSign.UNCERTAIN
        exact += int(np.sum(uncertain))
        if count + exact > layout.maximum_work_units:
            status = DeviceGenerationStatus.WORK_LIMIT
            break
        if np.any(uncertain):
            corners = charts[uncertain]
            signs[uncertain] = exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
        if np.any(signs != expected_signs[first : first + charts.shape[0]]):
            status = DeviceGenerationStatus.INVALID_GEOMETRY
            break
    accepted = status is DeviceGenerationStatus.COMPLETE
    return DeviceGenerationEvidence(
        status,
        count,
        exact,
        0,
        0,
        vertices.shape[0] if accepted else 0,
        count if accepted else 0,
        barriers,
        transferred,
        retained,
    )


def _surface_cell_ids(
    source_rows: NDArray[np.int32],
    source_ids: NDArray[np.int64],
    vertex_ids: NDArray[np.int64],
    target_rows: NDArray[np.int32],
    /,
) -> NDArray[np.int64]:
    """Retain unchanged scientific cells; issue IDs only for new vertex sets."""
    keys = np.sort(vertex_ids[source_rows], axis=1)
    if source_ids.shape != (keys.shape[0],):
        raise ValueError(
            "Scientific source cell IDs must match their exact source connectivity rows."
        )
    lookup = {
        tuple(keys[index, column] for column in range(keys.shape[1])): source_ids[index]
        for index in range(keys.shape[0])
    }
    target_keys = np.sort(vertex_ids[target_rows], axis=1)
    identifiers = np.asarray(
        [
            lookup.get(
                tuple(
                    target_keys[index, column] for column in range(target_keys.shape[1])
                ),
                np.int64(-1),
            )
            for index in range(target_keys.shape[0])
        ],
        dtype=np.int64,
    )
    new = identifiers < 0
    identifiers[new] = np.max(source_ids) + 1 + np.arange(np.sum(new), dtype=np.int64)
    return identifiers


def _host_round(
    source: PreparedDeviceGeneration,
    host: PreparedDeviceGeneration,
    candidate: DeviceGenerationCandidates,
    surface_points: NDArray[np.float64] | None,
    surface_normals: NDArray[np.float64] | None,
    /,
) -> tuple[PreparedDeviceGeneration, DeviceGenerationStatus, int, int, int, int]:
    mesh = host.mesh
    vertex_live = np.asarray(mesh.vertex_active)
    cell_live = np.asarray(mesh.cell_active)
    nv = int(np.sum(vertex_live))
    active_slots = np.flatnonzero(cell_live)
    points = np.asarray(mesh.coordinates)[:nv]
    rows = np.asarray(mesh.cells)[cell_live]
    vertex_ids = np.asarray(mesh.vertex_ids)[:nv]
    cell_ids = np.asarray(mesh.cell_ids)[cell_live]
    classes = np.asarray(host.cell_classes)[cell_live]
    sizes = np.asarray(host.target_sizes)[cell_live]
    selected = np.asarray(candidate.slots)[np.asarray(candidate.active)]
    hints = np.searchsorted(active_slots, selected).astype(np.int32)
    if (
        np.unique(selected).size != selected.size
        or np.any(hints >= active_slots.size)
        or not np.array_equal(
            active_slots[np.minimum(hints, active_slots.size - 1)], selected
        )
    ):
        raise ValueError("Candidates must reference distinct active source cell slots.")
    live_candidates = np.asarray(candidate.active)
    tolerance = 16 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(points))))
    if not np.allclose(
        np.asarray(candidate.points)[live_candidates],
        np.mean(points[rows[hints]], axis=1),
        rtol=0,
        atol=tolerance,
    ) or not np.allclose(
        np.asarray(candidate.charts)[live_candidates],
        np.mean(np.asarray(host.charts)[rows[hints]], axis=1),
        rtol=16 * np.finfo(np.float64).eps,
        atol=tolerance,
    ):
        raise ValueError("Candidate coordinates do not belong to the prepared source.")
    count = hints.size
    layout = source.layout
    width = rows.shape[1]
    created_bound = rows.shape[0] + (width - 1) * count
    if nv + count > layout.vertex_capacity or created_bound > layout.cell_capacity:
        return source, DeviceGenerationStatus.CAPACITY_EXCEEDED, 0, 0, 0, 0
    if (
        np.max(vertex_ids) > np.iinfo(np.int64).max - count
        or np.max(cell_ids) > np.iinfo(np.int64).max - width * count - rows.shape[0]
    ):
        return source, DeviceGenerationStatus.CAPACITY_EXCEEDED, 0, 0, 0, 0
    # Resolve uncertain SOURCE decisions in one exact batch before topology.
    unresolved = np.asarray(candidate.signs)[cell_live] == PredicateSign.UNCERTAIN
    exact = int(np.sum(unresolved))
    reserved_exact_work = exact + created_bound
    if reserved_exact_work >= layout.maximum_work_units:
        return source, DeviceGenerationStatus.WORK_LIMIT, 0, 0, 0, 0
    if exact:
        corners = (np.asarray(host.charts) if width == 3 else points)[rows[unresolved]]
        signs = (
            exact_orient2d(corners[:, 0], corners[:, 1], corners[:, 2])
            if width == 3
            else exact_orient3d(
                corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
            )
        )
        if np.any(signs != PredicateSign.POSITIVE):
            return source, DeviceGenerationStatus.INVALID_GEOMETRY, exact, 0, 0, 0
    charts_out = normals_out = flags_out = None
    if width == 3:
        if not np.all(classes == classes[0]) or not np.all(sizes == sizes[0]):
            raise ValueError(
                "Surface reconnection requires one patch region and uniform target size."
            )
        if surface_points is None or surface_normals is None:
            raise ValueError(
                "Surface candidates require one explicit batched geometry evaluation."
            )
        shape = (layout.candidate_capacity, 3)
        if (
            surface_points.shape != shape
            or surface_normals.shape != shape
            or surface_points.dtype != np.float64
            or surface_normals.dtype != np.float64
        ):
            raise TypeError(
                "Surface geometry arrays must be float64 candidate-capacity arrays."
            )
        if not np.all(np.isfinite(surface_points)) or not np.all(
            np.isfinite(surface_normals)
        ):
            raise ValueError("Surface candidate geometry must be finite.")
        live = np.asarray(candidate.active)
        new_points = surface_points[live]
        new_charts = np.asarray(candidate.charts)[live]
        new_normals = surface_normals[live]
        result, flags_out, issued, statuses, counters = surface_reconnect(
            np.asarray(host.charts)[:nv],
            points,
            np.asarray(host.normals)[:nv],
            rows,
            np.asarray(host.constrained)[cell_live],
            new_charts,
            new_points,
            new_normals,
            np.asarray(candidate.spacing)[live],
            hints,
            sweep=False,
            max_triangles=layout.cell_capacity,
            work_limit=layout.maximum_work_units - reserved_exact_work,
            insert_edges=None,
        )
        work = int(counters[4])
        if counters[5] or np.any(statuses == MeshcoreStatus.CAPACITY_EXCEEDED):
            return source, DeviceGenerationStatus.WORK_LIMIT, exact, work, 0, 0
        accepted = issued >= 0
        if not np.any(accepted):
            return source, DeviceGenerationStatus.COMPLETE, exact, work, 0, 0
        charts_out = np.concatenate((np.asarray(host.charts)[:nv], new_charts[accepted]))
        normals_out = np.concatenate(
            (np.asarray(host.normals)[:nv], new_normals[accepted])
        )
        points_out = np.concatenate((points, new_points[accepted]))
        classes_out = np.full((result.shape[0],), classes[0], dtype=np.int64)
        sizes_out = np.full((result.shape[0],), sizes[0], dtype=np.float64)
        all_vertex_ids = np.concatenate(
            (
                vertex_ids,
                np.max(vertex_ids) + 1 + np.arange(np.sum(accepted), dtype=np.int64),
            )
        )
        cell_ids_out = _surface_cell_ids(rows, cell_ids, all_vertex_ids, result)
        order = np.argsort(cell_ids_out, kind="stable")
        cell_ids_out, result, flags_out = (
            cell_ids_out[order],
            result[order],
            flags_out[order],
        )
        exact += result.shape[0]
        final = charts_out[result]
        signs = exact_orient2d(final[:, 0], final[:, 1], final[:, 2])
    else:
        work = rows.shape[0] + 4 * count
        if work + reserved_exact_work > layout.maximum_work_units:
            return source, DeviceGenerationStatus.WORK_LIMIT, exact, work, 0, 0
        new_points = np.asarray(candidate.points)[np.asarray(candidate.active)]
        points_out = np.concatenate((points, new_points))
        children = np.repeat(rows[hints, None, :], 4, axis=1)
        for local in range(4):
            children[:, local, local] = nv + np.arange(count, dtype=np.int32)
        keep = np.ones((rows.shape[0],), dtype=np.bool_)
        keep[hints] = False
        result = np.concatenate((rows[keep], children.reshape((-1, 4))))
        cell_ids_out = np.concatenate(
            (cell_ids[keep], np.arange(4 * count, dtype=np.int64) + np.max(cell_ids) + 1)
        )
        classes_out = np.concatenate((classes[keep], np.repeat(classes[hints], 4)))
        sizes_out = np.concatenate((sizes[keep], np.repeat(sizes[hints], 4)))
        final = points_out[result]
        exact += result.shape[0]
        signs = exact_orient3d(final[:, 0], final[:, 1], final[:, 2], final[:, 3])
    if np.any(signs != PredicateSign.POSITIVE):
        return source, DeviceGenerationStatus.INVALID_GEOMETRY, exact, work, 0, 0
    inserted = points_out.shape[0] - nv
    vertex_ids_out = np.concatenate(
        (vertex_ids, np.arange(inserted, dtype=np.int64) + np.max(vertex_ids) + 1)
    )
    if width == 4:
        charts_out = np.zeros((points_out.shape[0], 2), dtype=np.float64)
        normals_out = np.zeros(points_out.shape, dtype=np.float64)
        flags_out = np.zeros(result.shape, dtype=np.bool_)
    if charts_out is None or normals_out is None or flags_out is None:
        raise RuntimeError("Topology stage did not produce its surface metadata.")
    target = _pack(
        layout,
        points_out,
        result.astype(np.int32),
        vertex_ids_out,
        cell_ids_out,
        classes_out,
        sizes_out,
        charts_out,
        normals_out,
        flags_out,
    )
    return target, DeviceGenerationStatus.COMPLETE, exact, work, inserted, result.shape[0]


__all__ = [
    "DeviceGenerationLayout",
    "PreparedDeviceGeneration",
    "DeviceGenerationCandidates",
    "DeviceGenerationStatus",
    "DeviceGenerationEvidence",
    "DeviceGenerationUpdate",
    "prepare_device_generation",
    "evaluate_device_generation_candidates",
    "execute_device_generation_round",
    "certify_device_surface_initial",
]
