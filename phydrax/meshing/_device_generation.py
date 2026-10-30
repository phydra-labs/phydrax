#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device-selected simplex seed refinement with an exact host commit barrier.

The compiled device stage selects cells whose longest edge exceeds their target
size. Host topology changes are staged transactionally: uncertain source
orientations are resolved together before insertion, and all new simplices are
exactly oriented before the target mesh and semantic IDs are published. Curved
surface points and normals are supplied as one host geometry batch; this route
is therefore hybrid host/device, not fully device execution.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import orient2d, orient3d, PredicateMode, PredicateSign
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    _INDEX_LIMIT,
    adaptive_simplex_bucket,
    masked_simplex_facet_neighbors,
    MaskedSimplexMesh,
)
from ..typing import checked


class DeviceGenerationStatus(IntEnum):
    COMPLETE = 0
    INVALID_GEOMETRY = 1
    ADDRESSABILITY_EXCEEDED = 2
    IDENTITY_EXHAUSTED = 3


@final
class DeviceGenerationLayout(StrictModule, NonTrainableState):
    """Initial device allocation hints; actual topology growth is unbounded."""

    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(self, *, vertex_capacity: int, cell_capacity: int) -> None:
        vertices = _positive_capacity(vertex_capacity, "vertex_capacity")
        cells = _positive_capacity(cell_capacity, "cell_capacity")
        self.vertex_capacity = vertices
        self.cell_capacity = cells
        self.signature_id = canonical_fingerprint(
            {
                "kind": "device-generation-layout",
                "vertex_capacity": vertices,
                "cell_capacity": cells,
            }
        )


@final
class PreparedDeviceGeneration(StrictModule, NonTrainableState):
    """Capacity-bucketed seed mesh and its source-bound semantic metadata."""

    layout: DeviceGenerationLayout
    mesh: MaskedSimplexMesh
    charts: Array
    normals: Array
    constrained: Array
    cell_classes: Array
    target_sizes: Array
    generation_id: str = eqx.field(static=True)


@final
class DeviceGenerationCandidates(StrictModule):
    """Compiled, full-cell candidate batch in semantic cell-ID slot order."""

    slots: Array
    active: Array
    charts: Array
    points: Array
    spacing: Array
    signs: Array
    certain: Array
    count: Array
    uncertain: Array
    invalid: Array
    status: Array
    generation_id: str = eqx.field(static=True)


@final
class DeviceGenerationEvidence(StrictModule, NonTrainableState):
    """Accepted-state result; uncertainty is resolved on the host before commit."""

    status: DeviceGenerationStatus = eqx.field(static=True)
    exact_predicate_count: int = eqx.field(static=True)
    inserted_vertices: int = eqx.field(static=True)
    accepted_cells: int = eqx.field(static=True)


@final
class DeviceGenerationUpdate(StrictModule, NonTrainableState):
    prepared: PreparedDeviceGeneration
    evidence: DeviceGenerationEvidence


class _GenerationSeed(NamedTuple):
    points: NDArray[np.float64]
    cells: NDArray[np.int32]
    vertex_ids: NDArray[np.int64]
    cell_ids: NDArray[np.int64]
    cell_classes: NDArray[np.int64]
    target_sizes: NDArray[np.float64]
    charts: NDArray[np.float64]
    normals: NDArray[np.float64]
    constrained: NDArray[np.bool_]


def _positive_capacity(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"DeviceGenerationLayout.{name} must be an integer.")
    if value < 1:
        raise ValueError(f"DeviceGenerationLayout.{name} must be positive.")
    return int(value)


def _generation_id(seed: _GenerationSeed, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "device-generation-source",
            "arrays": array_tree_fingerprint(
                (
                    seed.points,
                    seed.cells,
                    seed.vertex_ids,
                    seed.cell_ids,
                    seed.cell_classes,
                    seed.target_sizes,
                    seed.charts,
                    seed.normals,
                    seed.constrained,
                )
            ),
        }
    )


def _admit_generation_seed(
    points: NDArray[np.float64],
    cells: NDArray[np.int32],
    vertex_ids: NDArray[np.int64],
    cell_ids: NDArray[np.int64],
    cell_classes: NDArray[np.int64],
    target_sizes: NDArray[np.float64],
    charts: NDArray[np.float64] | None,
    normals: NDArray[np.float64] | None,
    constrained: NDArray[np.bool_] | None,
    /,
) -> _GenerationSeed:
    arrays = (points, cells, vertex_ids, cell_ids, cell_classes, target_sizes)
    dtypes = (np.float64, np.int32, np.int64, np.int64, np.int64, np.float64)
    if any(
        not isinstance(array, np.ndarray) or array.dtype != dtype
        for array, dtype in zip(arrays, dtypes, strict=True)
    ):
        raise TypeError(
            "prepare_device_generation arrays must be NumPy arrays with explicit float64, int32, and int64 dtypes."
        )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or cells.ndim != 2
        or cells.shape[1] not in (3, 4)
    ):
        raise ValueError(
            "prepare_device_generation requires physical (vertices, 3) coordinates and triangle or tetrahedron rows."
        )
    vertex_count, cell_count = points.shape[0], cells.shape[0]
    if not vertex_count or not cell_count:
        raise ValueError("prepare_device_generation requires a nonempty seed mesh.")
    if (
        vertex_ids.shape != (vertex_count,)
        or cell_ids.shape != (cell_count,)
        or cell_classes.shape != (cell_count,)
        or target_sizes.shape != (cell_count,)
    ):
        raise ValueError(
            "prepare_device_generation semantic IDs, classes, and target sizes must align with seed rows."
        )
    if (
        np.any(vertex_ids < 0)
        or np.any(cell_ids < 0)
        or np.any(np.diff(vertex_ids) <= 0)
        or np.any(np.diff(cell_ids) <= 0)
    ):
        raise ValueError(
            "prepare_device_generation vertex and cell IDs must be nonnegative, unique, and in strictly increasing order."
        )
    if (
        not np.all(np.isfinite(points))
        or not np.all(np.isfinite(target_sizes))
        or np.any(target_sizes <= 0.0)
    ):
        raise ValueError(
            "prepare_device_generation coordinates must be finite and target sizes finite and positive."
        )
    if (
        np.any(cells < 0)
        or np.any(cells >= vertex_count)
        or np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0)
    ):
        raise ValueError(
            "prepare_device_generation cells must reference distinct in-range vertex slots."
        )

    if cells.shape[1] == 3:
        if charts is None or normals is None or constrained is None:
            raise ValueError(
                "Triangle seeds require chart coordinates, vertex normals, and constrained-edge flags."
            )
        if (
            not isinstance(charts, np.ndarray)
            or charts.dtype != np.float64
            or charts.shape != (vertex_count, 2)
            or not isinstance(normals, np.ndarray)
            or normals.dtype != np.float64
            or normals.shape != (vertex_count, 3)
            or not isinstance(constrained, np.ndarray)
            or constrained.dtype != np.bool_
            or constrained.shape != cells.shape
        ):
            raise TypeError(
                "Triangle chart coordinates, normals, and constraints must be float64 (vertices, 2), float64 (vertices, 3), and bool (cells, 3) arrays."
            )
        if not np.all(np.isfinite(charts)) or not np.all(np.isfinite(normals)):
            raise ValueError("Triangle charts and normals must be finite.")
        chart_bank = charts
        normal_bank = normals
        constraint_bank = constrained
        corners = chart_bank[cells]
        orientation = orient2d(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            mode=PredicateMode.EXACT,
        ).signs
    else:
        if charts is not None or normals is not None or constrained is not None:
            raise ValueError("Tetrahedral seeds do not accept surface metadata.")
        chart_bank = np.zeros((vertex_count, 2), dtype=np.float64)
        normal_bank = np.zeros((vertex_count, 3), dtype=np.float64)
        constraint_bank = np.zeros(cells.shape, dtype=np.bool_)
        corners = points[cells]
        orientation = orient3d(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            corners[:, 3],
            mode=PredicateMode.EXACT,
        ).signs
    if np.any(orientation != PredicateSign.POSITIVE):
        raise ValueError(
            "prepare_device_generation seed simplices must have positive exact orientation."
        )
    return _GenerationSeed(
        points,
        cells,
        vertex_ids,
        cell_ids,
        cell_classes,
        target_sizes,
        chart_bank,
        normal_bank,
        constraint_bank,
    )


def _grown_capacity(required: int, limit: int, /) -> int:
    if required > limit:
        raise OverflowError("Device generation exceeds addressable int32 simplex slots.")
    bucket = adaptive_simplex_bucket(required, 2.0)
    return bucket if bucket <= limit else required


def _fit_layout(
    layout: DeviceGenerationLayout,
    vertex_count: int,
    cell_count: int,
    width: int,
    /,
) -> DeviceGenerationLayout:
    vertex_limit = _INDEX_LIMIT
    cell_limit = _INDEX_LIMIT // width
    if layout.vertex_capacity > vertex_limit or layout.cell_capacity > cell_limit:
        raise OverflowError(
            "Device generation capacities exceed int32 simplex addressability."
        )
    vertices = (
        layout.vertex_capacity
        if vertex_count <= layout.vertex_capacity
        else _grown_capacity(vertex_count, vertex_limit)
    )
    cells = (
        layout.cell_capacity
        if cell_count <= layout.cell_capacity
        else _grown_capacity(cell_count, cell_limit)
    )
    if vertices == layout.vertex_capacity and cells == layout.cell_capacity:
        return layout
    return DeviceGenerationLayout(vertex_capacity=vertices, cell_capacity=cells)


def _pad(values: NDArray, capacity: int, fill: int | float | bool, /) -> NDArray:
    result = np.full((capacity,) + values.shape[1:], fill, dtype=values.dtype)
    result[: values.shape[0]] = values
    return result


def _pack(
    layout: DeviceGenerationLayout, seed: _GenerationSeed, /
) -> PreparedDeviceGeneration:
    vertex_count, cell_count = seed.points.shape[0], seed.cells.shape[0]
    width = seed.cells.shape[1]
    if (
        vertex_count > layout.vertex_capacity
        or cell_count > layout.cell_capacity
        or layout.vertex_capacity > _INDEX_LIMIT
        or layout.cell_capacity > _INDEX_LIMIT // width
    ):
        raise ValueError(
            "The admitted seed does not fit its addressable generation layout."
        )
    vertex_capacity, cell_capacity = layout.vertex_capacity, layout.cell_capacity
    rows = jnp.asarray(_pad(seed.cells, cell_capacity, 0))
    vertex_active = jnp.arange(vertex_capacity, dtype=jnp.int32) < vertex_count
    cell_active = jnp.arange(cell_capacity, dtype=jnp.int32) < cell_count
    mesh = MaskedSimplexMesh(
        jnp.asarray(_pad(seed.points, vertex_capacity, 0.0)),
        jnp.asarray(_pad(seed.vertex_ids, vertex_capacity, -1)),
        vertex_active,
        rows,
        jnp.asarray(_pad(seed.cell_ids, cell_capacity, -1)),
        cell_active,
        masked_simplex_facet_neighbors(rows, cell_active),
    )
    return PreparedDeviceGeneration(
        layout,
        mesh,
        jnp.asarray(_pad(seed.charts, vertex_capacity, 0.0)),
        jnp.asarray(_pad(seed.normals, vertex_capacity, 0.0)),
        jnp.asarray(_pad(seed.constrained, cell_capacity, False)),
        jnp.asarray(_pad(seed.cell_classes, cell_capacity, -1)),
        jnp.asarray(_pad(seed.target_sizes, cell_capacity, 1.0)),
        _generation_id(seed),
    )


@checked
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
    """Validate exact seed geometry and pack it into an expandable device bucket."""
    seed = _admit_generation_seed(
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
    fitted = _fit_layout(
        layout, seed.points.shape[0], seed.cells.shape[0], seed.cells.shape[1]
    )
    return _pack(fitted, seed)


def _candidates(prepared: PreparedDeviceGeneration, /) -> DeviceGenerationCandidates:
    mesh = prepared.mesh
    corners = mesh.coordinates[mesh.cells]
    width = mesh.cells.shape[1]
    pairs = np.asarray(
        [(first, second) for first in range(width) for second in range(first + 1, width)],
        dtype=np.int32,
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
    return DeviceGenerationCandidates(
        jnp.arange(mesh.cell_capacity, dtype=jnp.int32),
        requested,
        jnp.mean(chart_corners, axis=1),
        jnp.mean(corners, axis=1),
        0.2 * jnp.minimum(longest, prepared.target_sizes),
        predicate.signs,
        predicate.certain,
        jnp.sum(requested, dtype=jnp.int32),
        jnp.sum(mesh.cell_active & ~predicate.certain, dtype=jnp.int32),
        jnp.sum(invalid, dtype=jnp.int32),
        jnp.where(
            jnp.any(invalid),
            int(DeviceGenerationStatus.INVALID_GEOMETRY),
            int(DeviceGenerationStatus.COMPLETE),
        ).astype(jnp.int32),
        prepared.generation_id,
    )


_compiled_candidates = eqx.filter_jit(_candidates)


def _check_prepared(prepared: PreparedDeviceGeneration, /) -> None:
    if not isinstance(prepared, PreparedDeviceGeneration):
        raise TypeError("prepared must be PreparedDeviceGeneration.")
    if not isinstance(prepared.layout, DeviceGenerationLayout) or not isinstance(
        prepared.mesh, MaskedSimplexMesh
    ):
        raise TypeError(
            "PreparedDeviceGeneration must contain its layout and simplex mesh."
        )
    mesh, layout = prepared.mesh, prepared.layout
    width = mesh.cells.shape[1]
    if (
        mesh.vertex_capacity != layout.vertex_capacity
        or mesh.cell_capacity != layout.cell_capacity
        or width not in (3, 4)
        or mesh.coordinates.shape[1] != 3
        or mesh.coordinates.dtype != jnp.float64
        or mesh.vertex_ids.dtype != jnp.int64
        or mesh.vertex_active.dtype != jnp.bool_
        or mesh.cells.dtype != jnp.int32
        or mesh.cell_ids.dtype != jnp.int64
        or mesh.cell_active.dtype != jnp.bool_
        or prepared.charts.shape != (layout.vertex_capacity, 2)
        or prepared.charts.dtype != jnp.float64
        or prepared.normals.shape != (layout.vertex_capacity, 3)
        or prepared.normals.dtype != jnp.float64
        or prepared.constrained.shape != (layout.cell_capacity, width)
        or prepared.constrained.dtype != jnp.bool_
        or prepared.cell_classes.shape != (layout.cell_capacity,)
        or prepared.cell_classes.dtype != jnp.int64
        or prepared.target_sizes.shape != (layout.cell_capacity,)
        or prepared.target_sizes.dtype != jnp.float64
    ):
        raise ValueError("PreparedDeviceGeneration arrays do not belong to its layout.")
    if any(
        isinstance(leaf, Array) and not leaf.is_fully_addressable
        for leaf in jax.tree_util.tree_leaves(prepared)
    ):
        raise ValueError("Device generation requires process-local, addressable arrays.")


@checked
def evaluate_device_generation_candidates(
    prepared: PreparedDeviceGeneration, /
) -> DeviceGenerationCandidates:
    """Evaluate the complete deterministic candidate batch on the device."""
    _check_prepared(prepared)
    return _compiled_candidates(prepared)


def _check_candidates(
    prepared: PreparedDeviceGeneration,
    candidates: DeviceGenerationCandidates,
    /,
) -> None:
    if not isinstance(candidates, DeviceGenerationCandidates):
        raise TypeError("candidates must be DeviceGenerationCandidates.")
    if candidates.generation_id != prepared.generation_id:
        raise ValueError("Candidates do not belong to this prepared generation source.")
    arrays = (
        candidates.slots,
        candidates.active,
        candidates.charts,
        candidates.points,
        candidates.spacing,
        candidates.signs,
        candidates.certain,
        candidates.count,
        candidates.uncertain,
        candidates.invalid,
        candidates.status,
    )
    if any(not isinstance(array, Array) for array in arrays):
        raise TypeError("Candidate state must contain device arrays.")
    vertex_capacity = prepared.layout.vertex_capacity
    cell_capacity = prepared.layout.cell_capacity
    shapes = (
        (cell_capacity,),
        (cell_capacity,),
        (cell_capacity, 2),
        (cell_capacity, 3),
        (cell_capacity,),
        (cell_capacity,),
        (cell_capacity,),
        (),
        (),
        (),
        (),
    )
    dtypes = (
        jnp.int32,
        jnp.bool_,
        jnp.float64,
        jnp.float64,
        jnp.float64,
        jnp.int8,
        jnp.bool_,
        jnp.int32,
        jnp.int32,
        jnp.int32,
        jnp.int32,
    )
    if any(
        array.shape != shape or array.dtype != dtype
        for array, shape, dtype in zip(arrays, shapes, dtypes, strict=True)
    ) or prepared.charts.shape != (vertex_capacity, 2):
        raise ValueError("Candidate arrays do not match the prepared generation layout.")


def _host_seed(prepared: PreparedDeviceGeneration, /) -> _GenerationSeed:
    mesh = prepared.mesh
    vertex_active = np.asarray(mesh.vertex_active, dtype=np.bool_)
    cell_active = np.asarray(mesh.cell_active, dtype=np.bool_)
    vertex_count = int(np.count_nonzero(vertex_active))
    cell_count = int(np.count_nonzero(cell_active))
    if (
        vertex_count == 0
        or cell_count == 0
        or not np.all(vertex_active[:vertex_count])
        or np.any(vertex_active[vertex_count:])
        or not np.all(cell_active[:cell_count])
        or np.any(cell_active[cell_count:])
    ):
        raise ValueError(
            "Prepared generation activity masks must be nonempty ID-ordered prefixes."
        )
    seed = _GenerationSeed(
        np.asarray(mesh.coordinates, dtype=np.float64)[:vertex_count],
        np.asarray(mesh.cells, dtype=np.int32)[:cell_count],
        np.asarray(mesh.vertex_ids, dtype=np.int64)[:vertex_count],
        np.asarray(mesh.cell_ids, dtype=np.int64)[:cell_count],
        np.asarray(prepared.cell_classes, dtype=np.int64)[:cell_count],
        np.asarray(prepared.target_sizes, dtype=np.float64)[:cell_count],
        np.asarray(prepared.charts, dtype=np.float64)[:vertex_count],
        np.asarray(prepared.normals, dtype=np.float64)[:vertex_count],
        np.asarray(prepared.constrained, dtype=np.bool_)[:cell_count],
    )
    if (
        np.any(np.diff(seed.vertex_ids) <= 0)
        or np.any(np.diff(seed.cell_ids) <= 0)
        or np.any(seed.cells < 0)
        or np.any(seed.cells >= vertex_count)
        or _generation_id(seed) != prepared.generation_id
    ):
        raise ValueError(
            "Prepared generation semantic IDs or source identity are inconsistent."
        )
    return seed


def _host_candidates(
    seed: _GenerationSeed,
    candidates: DeviceGenerationCandidates,
    /,
) -> tuple[
    NDArray[np.bool_], NDArray[np.int8], NDArray[np.bool_], int, DeviceGenerationStatus
]:
    active = np.asarray(candidates.active, dtype=np.bool_)
    slots = np.asarray(candidates.slots, dtype=np.int32)
    signs = np.asarray(candidates.signs, dtype=np.int8)
    certain = np.asarray(candidates.certain, dtype=np.bool_)
    count = int(np.asarray(candidates.count))
    uncertain_count = int(np.asarray(candidates.uncertain))
    invalid_count = int(np.asarray(candidates.invalid))
    status = DeviceGenerationStatus(int(np.asarray(candidates.status)))
    cell_capacity = slots.size
    expected_slots = np.arange(cell_capacity, dtype=np.int32)
    cell_count = seed.cells.shape[0]
    expected_uncertain = int(np.count_nonzero(~certain[:cell_count]))
    expected_invalid = int(
        np.count_nonzero(
            certain[:cell_count] & (signs[:cell_count] != PredicateSign.POSITIVE)
        )
    )
    expected_status = (
        DeviceGenerationStatus.INVALID_GEOMETRY
        if expected_invalid
        else DeviceGenerationStatus.COMPLETE
    )
    if (
        not np.array_equal(slots, expected_slots)
        or active.shape != (cell_capacity,)
        or signs.shape != (cell_capacity,)
        or certain.shape != (cell_capacity,)
        or count != int(np.count_nonzero(active))
        or np.any(active[cell_count:])
        or uncertain_count != expected_uncertain
        or invalid_count != expected_invalid
        or status is not expected_status
    ):
        raise ValueError(
            "Candidate slots and phase evidence do not match active source cells."
        )
    selected = np.flatnonzero(active)
    if selected.size:
        rows = seed.cells[selected]
        expected_points = np.mean(seed.points[rows], axis=1)
        expected_charts = np.mean(seed.charts[rows], axis=1)
        tolerance = (
            16.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(seed.points))))
        )
        if not np.allclose(
            np.asarray(candidates.points)[selected],
            expected_points,
            rtol=0.0,
            atol=tolerance,
        ) or not np.allclose(
            np.asarray(candidates.charts)[selected],
            expected_charts,
            rtol=16.0 * np.finfo(np.float64).eps,
            atol=tolerance,
        ):
            raise ValueError(
                "Candidate coordinates do not match their semantic source cells."
            )
    return active, signs, certain, count, status


def _exact_cell_signs(
    seed: _GenerationSeed, rows: NDArray[np.int32], /
) -> NDArray[np.int8]:
    if rows.shape[0] == 0:
        return np.empty((0,), dtype=np.int8)
    if rows.shape[1] == 3:
        corners = seed.charts[rows]
        result = orient2d(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            mode=PredicateMode.EXACT,
        )
    else:
        corners = seed.points[rows]
        result = orient3d(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            corners[:, 3],
            mode=PredicateMode.EXACT,
        )
    return np.asarray(result.signs, dtype=np.int8)


def _issued_ids(first: int, count: int, name: str, /) -> NDArray[np.int64]:
    maximum = np.iinfo(np.int64).max
    if count < 0 or first < 0 or count > maximum - first + 1:
        raise OverflowError(
            f"Device generation exhausted the int64 {name} identity range."
        )
    if count == 0:
        return np.empty((0,), dtype=np.int64)
    return np.arange(count, dtype=np.int64) + np.int64(first)


def _candidate_geometry(
    candidates: DeviceGenerationCandidates,
    active: NDArray[np.bool_],
    width: int,
    surface_points: NDArray[np.float64] | None,
    surface_normals: NDArray[np.float64] | None,
    /,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    count = int(np.count_nonzero(active))
    selected_points = np.asarray(candidates.points, dtype=np.float64)[active]
    selected_charts = np.asarray(candidates.charts, dtype=np.float64)[active]
    if width == 3:
        if surface_points is None or surface_normals is None:
            raise ValueError(
                "Surface insertion requires one geometry evaluation for every candidate chart."
            )
        if (
            not isinstance(surface_points, np.ndarray)
            or surface_points.dtype != np.float64
            or surface_points.shape != (count, 3)
            or not isinstance(surface_normals, np.ndarray)
            or surface_normals.dtype != np.float64
            or surface_normals.shape != (count, 3)
        ):
            raise TypeError(
                "Surface geometry batches must be float64 (candidate, 3) arrays."
            )
        if not np.all(np.isfinite(surface_points)) or not np.all(
            np.isfinite(surface_normals)
        ):
            raise ValueError("Surface geometry batches must be finite.")
        return surface_points, selected_charts, surface_normals
    if surface_points is not None or surface_normals is not None:
        raise ValueError(
            "Tetrahedral insertion does not accept surface geometry batches."
        )
    return selected_points, selected_charts, np.zeros((count, 3), dtype=np.float64)


def _stage_target(
    seed: _GenerationSeed,
    layout: DeviceGenerationLayout,
    active: NDArray[np.bool_],
    candidate_points: NDArray[np.float64],
    candidate_charts: NDArray[np.float64],
    candidate_normals: NDArray[np.float64],
    /,
) -> tuple[PreparedDeviceGeneration | None, int, DeviceGenerationStatus]:
    rows = seed.cells
    width = rows.shape[1]
    selected = np.flatnonzero(active)
    count = selected.size
    child_count = width * count
    child_cells = np.repeat(rows[selected, None, :], width, axis=1)
    new_vertex_slots = seed.points.shape[0] + np.arange(count, dtype=np.int32)
    for local in range(width):
        child_cells[:, local, local] = new_vertex_slots
    children = child_cells.reshape((-1, width))

    vertex_ids = _issued_ids(int(seed.vertex_ids[-1]) + 1, count, "vertex")
    cell_ids = _issued_ids(int(seed.cell_ids[-1]) + 1, child_count, "cell")
    keep = np.ones((rows.shape[0],), dtype=np.bool_)
    keep[selected] = False
    cells_out = np.concatenate((rows[keep], children), axis=0).astype(
        np.int32, copy=False
    )
    cell_ids_out = np.concatenate((seed.cell_ids[keep], cell_ids))
    classes_out = np.concatenate(
        (seed.cell_classes[keep], np.repeat(seed.cell_classes[selected], width))
    )
    sizes_out = np.concatenate(
        (seed.target_sizes[keep], np.repeat(seed.target_sizes[selected], width))
    )
    points_out = np.concatenate((seed.points, candidate_points), axis=0)
    vertex_ids_out = np.concatenate((seed.vertex_ids, vertex_ids))
    if width == 3:
        charts_out = np.concatenate((seed.charts, candidate_charts), axis=0)
        normals_out = np.concatenate((seed.normals, candidate_normals), axis=0)
        constrained_out = np.zeros((cells_out.shape[0], 3), dtype=np.bool_)
        constrained_out[: np.count_nonzero(keep)] = seed.constrained[keep]
        child_constraints = constrained_out[np.count_nonzero(keep) :]
        for local in range(width):
            child_constraints.reshape((count, width, width))[:, local, local] = (
                seed.constrained[selected, local]
            )
    else:
        charts_out = np.zeros((points_out.shape[0], 2), dtype=np.float64)
        normals_out = np.zeros(points_out.shape, dtype=np.float64)
        constrained_out = np.zeros((cells_out.shape[0], width), dtype=np.bool_)

    target_seed = _GenerationSeed(
        points_out,
        cells_out,
        vertex_ids_out,
        cell_ids_out,
        classes_out,
        sizes_out,
        charts_out,
        normals_out,
        constrained_out,
    )
    child_signs = _exact_cell_signs(target_seed, children)
    if np.any(child_signs != PredicateSign.POSITIVE):
        return None, child_count, DeviceGenerationStatus.INVALID_GEOMETRY
    fitted = _fit_layout(layout, points_out.shape[0], cells_out.shape[0], width)
    return _pack(fitted, target_seed), child_count, DeviceGenerationStatus.COMPLETE


@checked
def execute_device_generation_round(
    prepared: PreparedDeviceGeneration,
    /,
    *,
    surface_points: NDArray[np.float64] | None = None,
    surface_normals: NDArray[np.float64] | None = None,
    candidates: DeviceGenerationCandidates | None = None,
) -> DeviceGenerationUpdate:
    """Resolve uncertainty, stage all candidate insertions, then publish atomically.

    The optional surface geometry arrays are the host evaluation of the active
    candidate charts, in their semantic cell-ID order. Uncertain device
    orientations are resolved in one exact host batch before topology staging;
    target children are exactly certified before a new preparation is returned.
    """
    _check_prepared(prepared)
    batch = (
        evaluate_device_generation_candidates(prepared)
        if candidates is None
        else candidates
    )
    _check_candidates(prepared, batch)
    host_prepared, host_batch = jax.device_get((prepared, batch))
    seed = _host_seed(host_prepared)
    current = prepared
    active, signs, certain, count, status = _host_candidates(seed, host_batch)
    source_cell_count = seed.cells.shape[0]
    if status is DeviceGenerationStatus.INVALID_GEOMETRY:
        evidence = DeviceGenerationEvidence(status, 0, 0, source_cell_count)
        return DeviceGenerationUpdate(prepared, evidence)
    if not count:
        if surface_points is not None or surface_normals is not None:
            raise ValueError(
                "Surface geometry was supplied for an empty candidate batch."
            )
        evidence = DeviceGenerationEvidence(
            DeviceGenerationStatus.COMPLETE, 0, 0, source_cell_count
        )
        return DeviceGenerationUpdate(prepared, evidence)

    width = seed.cells.shape[1]
    required_vertices = seed.points.shape[0] + count
    required_cells = source_cell_count + (width - 1) * count
    try:
        fitted = _fit_layout(current.layout, required_vertices, required_cells, width)
    except OverflowError:
        evidence = DeviceGenerationEvidence(
            DeviceGenerationStatus.ADDRESSABILITY_EXCEEDED, 0, 0, source_cell_count
        )
        return DeviceGenerationUpdate(prepared, evidence)
    if fitted.signature_id != current.layout.signature_id:
        current = _pack(fitted, seed)

    selected = np.flatnonzero(active)
    unresolved = np.flatnonzero(~certain[:source_cell_count])
    exact_count = unresolved.size
    if exact_count:
        exact_signs = _exact_cell_signs(seed, seed.cells[unresolved])
        if np.any(exact_signs != PredicateSign.POSITIVE):
            evidence = DeviceGenerationEvidence(
                DeviceGenerationStatus.INVALID_GEOMETRY, exact_count, 0, source_cell_count
            )
            return DeviceGenerationUpdate(prepared, evidence)
    selected_signs = signs[selected]
    if np.any(certain[selected] & (selected_signs != PredicateSign.POSITIVE)):
        evidence = DeviceGenerationEvidence(
            DeviceGenerationStatus.INVALID_GEOMETRY, exact_count, 0, source_cell_count
        )
        return DeviceGenerationUpdate(prepared, evidence)
    child_count = width * count
    maximum_id = np.iinfo(np.int64).max
    if count > maximum_id - int(seed.vertex_ids[-1]) or child_count > maximum_id - int(
        seed.cell_ids[-1]
    ):
        evidence = DeviceGenerationEvidence(
            DeviceGenerationStatus.IDENTITY_EXHAUSTED, exact_count, 0, source_cell_count
        )
        return DeviceGenerationUpdate(prepared, evidence)

    points_out, charts_out, normals_out = _candidate_geometry(
        host_batch,
        active,
        seed.cells.shape[1],
        surface_points,
        surface_normals,
    )
    target, exact_children, target_status = _stage_target(
        seed,
        current.layout,
        active,
        points_out,
        charts_out,
        normals_out,
    )
    if target_status is not DeviceGenerationStatus.COMPLETE or target is None:
        evidence = DeviceGenerationEvidence(
            DeviceGenerationStatus.INVALID_GEOMETRY,
            exact_count + exact_children,
            0,
            source_cell_count,
        )
        return DeviceGenerationUpdate(prepared, evidence)
    target_cell_count = seed.cells.shape[0] + (seed.cells.shape[1] - 1) * count
    evidence = DeviceGenerationEvidence(
        DeviceGenerationStatus.COMPLETE,
        exact_count + exact_children,
        count,
        target_cell_count,
    )
    return DeviceGenerationUpdate(target, evidence)


__all__ = [
    "DeviceGenerationCandidates",
    "DeviceGenerationEvidence",
    "DeviceGenerationLayout",
    "DeviceGenerationStatus",
    "DeviceGenerationUpdate",
    "PreparedDeviceGeneration",
    "evaluate_device_generation_candidates",
    "execute_device_generation_round",
    "prepare_device_generation",
]
