#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.geometry.implicit._realization as realization_owner

from ..._bvh import bvh_overlap_pair_blocks, PackedBVH
from ..._resource_ticket import BatchTicket
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._tensor_support import PreparedTensorGrid
from ...ein import contract
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...nonlinear import ITP, NonlinearStatus, ScalarRootResult
from ...typing import checked
from .._certificate import FieldRegularity, SignReliability, ZeroSetAccuracy
from .._contracts import CompiledGeometry, GeometryKernel, GeometryKind
from ..design._schema import DesignState
from ..simplicial import TriangleTopology
from ._enclosure import consume_queries, query_budget
from ._policy import ImplicitSurfacePolicy
from ._projection import field_and_gradient, ImplicitPointProjectionPlan


_EDGE_ROOT_METHOD = ITP()
# One shared ITP batch runs at most this many iterations over every lane.
_EDGE_ROOT_MAXIMUM_STEPS = 51

_DEFAULT_SURFACE_POLICY = ImplicitSurfacePolicy()

# Cube corners in the dual-contouring convention: corner c has lattice offset
# _CORNER_OFFSETS[c] relative to the cell's lower lattice point.
CORNER_OFFSETS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    dtype=np.int64,
)
_CUBE_EDGES = np.asarray(
    (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
    dtype=np.int64,
)
_CORNER_LOOKUP = np.zeros((2, 2, 2), dtype=np.int64)
_CORNER_LOOKUP[tuple(CORNER_OFFSETS.T)] = np.arange(8, dtype=np.int64)
EDGE_AXIS = np.argmax(
    CORNER_OFFSETS[_CUBE_EDGES[:, 0]] != CORNER_OFFSETS[_CUBE_EDGES[:, 1]],
    axis=1,
)
EDGE_LOWER = np.minimum(
    CORNER_OFFSETS[_CUBE_EDGES[:, 0]],
    CORNER_OFFSETS[_CUBE_EDGES[:, 1]],
)
# Four cells around a lattice edge in cyclic order; row `axis` of this table
# lists cell offsets relative to the edge's lower lattice point.
INCIDENT_CELL_OFFSETS = np.asarray(
    (
        ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)),
        ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)),
        ((-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)),
    ),
    dtype=np.int64,
)
_QEF_REGULARIZATION_LEVELS = 12
_QEF_SOLVE_PLAN = SmallLinearSolvePlan(3)


def _corner_component_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inside-corner components for every one of the 256 cube sign configurations.

    Components are numbered by increasing minimum corner index. Each crossing cube
    edge is owned by the component containing its inside endpoint.
    """

    component_of = np.full((256, 8), -1, dtype=np.int64)
    edge_component = np.full((256, 12), -1, dtype=np.int64)
    component_count = np.zeros((256,), dtype=np.int64)
    for configuration in range(256):
        inside = (configuration >> np.arange(8)) & 1 == 1
        label = np.where(inside, np.arange(8), 8)
        internal = inside[_CUBE_EDGES[:, 0]] & inside[_CUBE_EDGES[:, 1]]
        for _ in range(8):
            joined = np.minimum(label[_CUBE_EDGES[:, 0]], label[_CUBE_EDGES[:, 1]])
            np.minimum.at(label, _CUBE_EDGES[internal, 0], joined[internal])
            np.minimum.at(label, _CUBE_EDGES[internal, 1], joined[internal])
        roots = np.unique(label[inside])
        component_of[configuration, inside] = np.searchsorted(roots, label[inside])
        component_count[configuration] = roots.size
        crossing = inside[_CUBE_EDGES[:, 0]] != inside[_CUBE_EDGES[:, 1]]
        inside_endpoint = np.where(
            inside[_CUBE_EDGES[:, 0]], _CUBE_EDGES[:, 0], _CUBE_EDGES[:, 1]
        )
        edge_component[configuration, crossing] = component_of[
            configuration, inside_endpoint[crossing]
        ]
    return component_of, edge_component, component_count


_COMPONENT_OF, _EDGE_COMPONENT, _COMPONENT_COUNT = _corner_component_tables()


@dataclass(frozen=True, slots=True)
class _Crossings:
    axis: np.ndarray
    lower: np.ndarray
    edge_index: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class _DualVertices:
    cell: np.ndarray
    first_vertex: np.ndarray
    mixed_index: np.ndarray
    configuration: np.ndarray
    anchor_indices: np.ndarray
    anchor_mask: np.ndarray


def _validate_discovery_inputs(
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    policy: ImplicitSurfacePolicy,
    source_id: str,
) -> None:
    if not source_id:
        raise ValueError(
            f"ImplicitDiscovery.source_id must be non-empty; got {source_id!r}"
        )
    if geometry.ambient_dimension != 3 or geometry.kind is not GeometryKind.REGION:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.geometry must be a three-dimensional region; got dimension {geometry.ambient_dimension} and kind {geometry.kind}"
        )
    if len(grid.structured_axes) != 3:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.grid must have three structured axes; got {len(grid.structured_axes)}"
        )
    if any(axis.periodic for axis in grid.structured_axes):
        raise ValueError(
            "ImplicitSurfaceDiscovery.grid.axes must be nonperiodic; got periodic axes"
        )
    if not bool(np.asarray(geometry.validity().accepted)):
        raise ValueError(
            "ImplicitSurfaceDiscovery.geometry must have accepted validity evidence; got a rejected state"
        )
    certificate = geometry.field_certificate
    if certificate.sign_reliability is not SignReliability.RELIABLE:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.field.sign_reliability must be reliable; got {certificate.sign_reliability.value}"
        )
    if (
        certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
        and not policy.allow_approximate_zero_set
    ):
        raise ValueError(
            "ImplicitDiscovery.policy.allow_approximate_zero_set must permit approximate zero sets; got an approximate source with approval disabled"
        )
    if (
        certificate.regularity is FieldRegularity.NONSMOOTH
        and not policy.allow_nonsmooth_field
    ):
        raise ValueError(
            "ImplicitDiscovery.policy.allow_nonsmooth_field must approve selected branches of nonsmooth fields; got a nonsmooth source with approval disabled"
        )
    if not certificate.parameter_differentiable:
        raise ValueError(
            "ImplicitSurfaceDiscovery.field.parameter_differentiable must be true; got a nondifferentiable field declaration"
        )


def _lattice_crossings(inside: np.ndarray, policy: ImplicitSurfacePolicy) -> _Crossings:
    """Return sign-changing lattice edges ordered by (axis, i, j, k)."""

    axes: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    edge_index: list[np.ndarray] = []
    offset = 0
    for axis in range(3):
        changed = np.diff(inside, axis=axis)
        lower = np.argwhere(changed)
        index = np.full(changed.shape, -1, dtype=np.int64)
        index[changed] = offset + np.arange(lower.shape[0], dtype=np.int64)
        offset += lower.shape[0]
        axes.append(np.full((lower.shape[0],), axis, dtype=np.int64))
        lowers.append(lower)
        edge_index.append(index)
    if offset == 0:
        raise ValueError(
            "ImplicitSurfaceDiscovery.grid.crossings must contain a surface crossing; got zero crossings"
        )
    if offset > policy.maximum_crossings:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.crossings must fit maximum_crossings {policy.maximum_crossings}; got {offset}"
        )
    return _Crossings(
        np.concatenate(axes),
        np.concatenate(lowers),
        (edge_index[0], edge_index[1], edge_index[2]),
    )


@final
class RootIsolationResult(StrictModule, NonTrainableState):
    """Edge coordinates, residuals, and native root status and work evidence."""

    points: Array
    residuals: Array
    native_result: ScalarRootResult

    def require_converged(self) -> None:
        """Synchronize lane status at the host publication boundary and refuse failures."""
        statuses = np.asarray(jax.device_get(self.native_result.status), dtype=np.int32)
        if np.any(statuses != int(NonlinearStatus.SUCCESS)):
            summary = tuple(
                NonlinearStatus(value).name for value in np.unique(statuses).tolist()
            )
            diagnostics = self.native_result.nonlinear_result.diagnostics
            iterations, evaluations = jax.device_get(
                (diagnostics.iterations, diagnostics.residual_evaluations)
            )
            raise ValueError(
                "RootIsolationResult.native_result.status must be SUCCESS; "
                f"got {summary}, iterations={int(iterations)}, "
                f"residual_evaluations={int(evaluations)}"
            )


@eqx.filter_jit
def isolate_roots(
    kernel: GeometryKernel,
    state: DesignState,
    lower_points: Array,
    upper_points: Array,
    lower_values: Array,
    upper_values: Array,
    tolerance: Array,
) -> RootIsolationResult:
    """Isolate lattice-edge roots with the prepared native ITP method.

    Each lane solves f(lower + t (upper - lower)) = 0 on t in [0, 1].
    The edge parameter, callback arithmetic, and lane-freezing order are fixed.
    """
    direction = upper_points - lower_points

    def field(parameter: Array) -> Array:
        return kernel.boundary_field(state, lower_points + parameter[:, None] * direction)

    result = _EDGE_ROOT_METHOD.solve_batched(
        field,
        lower_values,
        upper_values,
        tolerance,
        lower_points,
        maximum_steps=_EDGE_ROOT_MAXIMUM_STEPS,
    )
    points = lower_points + result.root[:, None] * direction
    return RootIsolationResult(points, jnp.abs(result.value), result)


def _dual_vertices(
    inside: np.ndarray,
    crossings: _Crossings,
    policy: ImplicitSurfacePolicy,
) -> _DualVertices:
    """One dual vertex per inside-corner component of every mixed cell.

    Vertices are numbered by row-major cell order, then component order.
    """

    configuration = np.zeros(tuple(size - 1 for size in inside.shape), dtype=np.int64)
    for corner, (dx, dy, dz) in enumerate(CORNER_OFFSETS):
        corner_inside = inside[
            dx : inside.shape[0] - 1 + dx,
            dy : inside.shape[1] - 1 + dy,
            dz : inside.shape[2] - 1 + dz,
        ]
        configuration |= corner_inside.astype(np.int64) << corner
    mixed = (configuration != 0) & (configuration != 255)
    mixed_cells = np.argwhere(mixed)
    mixed_configuration = configuration[mixed]
    counts = _COMPONENT_COUNT[mixed_configuration]
    vertex_count = int(np.sum(counts))
    if vertex_count == 0:
        raise ValueError(
            "ImplicitSurfaceDiscovery.dual_vertices must be nonempty; got zero vertices"
        )
    if vertex_count > policy.maximum_vertices:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.dual_vertices must fit maximum_vertices {policy.maximum_vertices}; got {vertex_count}"
        )
    first_vertex = np.concatenate(((0,), np.cumsum(counts)[:-1])).astype(np.int64)
    mixed_index = np.full(configuration.shape, -1, dtype=np.int64)
    mixed_index[mixed] = np.arange(mixed_cells.shape[0], dtype=np.int64)
    owner = np.repeat(np.arange(mixed_cells.shape[0], dtype=np.int64), counts)
    component = np.arange(vertex_count, dtype=np.int64) - first_vertex[owner]
    vertex_cell = mixed_cells[owner]
    vertex_configuration = mixed_configuration[owner]

    edge_cells = vertex_cell[:, None, :] + EDGE_LOWER[None, :, :]
    candidates = np.full((vertex_count, 12), -1, dtype=np.int64)
    for axis in range(3):
        selected = EDGE_AXIS == axis
        cells = edge_cells[:, selected]
        candidates[:, selected] = crossings.edge_index[axis][
            cells[..., 0], cells[..., 1], cells[..., 2]
        ]
    member = _EDGE_COMPONENT[vertex_configuration] == component[:, None]
    if np.any(member & (candidates < 0)):
        raise ValueError(
            "ImplicitSurfaceDiscovery.incidence must represent every required dual vertex and component; got missing incidences"
        )
    anchor_count = np.sum(member, axis=1)
    width = int(np.max(anchor_count))
    ordered = np.sort(np.where(member, candidates, np.iinfo(np.int64).max), axis=1)
    anchor_mask = np.arange(width)[None, :] < anchor_count[:, None]
    anchor_indices = np.where(anchor_mask, ordered[:, :width], 0)
    return _DualVertices(
        vertex_cell,
        first_vertex,
        mixed_index,
        configuration,
        anchor_indices.astype(np.int32),
        anchor_mask,
    )


def qef_vertices(
    anchors: np.ndarray,
    gradients: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_mask: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    regularization: float,
    tolerance: float,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched regularized QEF vertices with bounded regularization escalation.

    The ``_QEF_REGULARIZATION_LEVELS`` levels are admitted up front; each level
    that runs charges one solve per vertex before it is dispatched.
    """

    norms = np.linalg.norm(gradients, axis=-1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError(
            "ImplicitQEF.anchor_gradients must be finite and nonzero; got nonfinite or zero gradient norms"
        )
    unit = gradients / norms[:, None]
    mask = anchor_mask.astype(np.float64)
    points = anchors[anchor_indices]
    normals = unit[anchor_indices]
    count = np.sum(mask, axis=1)
    mass = np.sum(points * mask[..., None], axis=1) / count[:, None]
    relative = points - mass[:, None, :]
    normal_matrix = np.asarray(contract("vki,vkj,vk->vij", normals, normals, mask))
    right_hand_side = np.asarray(
        contract(
            "vki,vk,vk->vi",
            normals,
            np.asarray(contract("vki,vki->vk", normals, relative)),
            mask,
        )
    )
    vertex_count = int(cell_lower.shape[0])
    vertices = np.zeros_like(cell_lower)
    selected_regularization = np.full((vertex_count,), np.nan, dtype=np.float64)
    done = np.zeros((vertex_count,), dtype=np.bool_)
    identity = np.eye(3, dtype=np.float64)
    # Every level solves every vertex system so one batch shape serves all
    # levels; each vertex keeps its first in-cell solution.
    with _admitted_batch(budget, vertex_count * _QEF_REGULARIZATION_LEVELS, 0) as admitted:
        for level in range(_QEF_REGULARIZATION_LEVELS):
            admitted.consume(vertex_count)
            weight = regularization * 10.0**level
            solved = solve_small_linear(
                _QEF_SOLVE_PLAN,
                normal_matrix + (weight * count)[:, None, None] * identity[None, :, :],
                right_hand_side,
            )
            value = mass + np.asarray(solved.value)
            accepted = (
                ~done
                & np.asarray(solved.successful)
                & np.all(np.isfinite(value), axis=1)
                & np.all(value >= cell_lower - tolerance, axis=1)
                & np.all(value <= cell_upper + tolerance, axis=1)
            )
            vertices[accepted] = value[accepted]
            selected_regularization[accepted] = weight
            done |= accepted
            if np.all(done):
                return vertices, selected_regularization
    raise ValueError(
        "ImplicitQEF.vertex must remain inside its discovery cell under the bounded regularization schedule; got an out-of-cell vertex"
    )


def _dual_faces(
    geometry: CompiledGeometry,
    inside: np.ndarray,
    crossings: _Crossings,
    dual: _DualVertices,
    base_vertices: np.ndarray,
    policy: ImplicitSurfacePolicy,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Split every crossing's dual quad into two field-oriented triangles."""

    cells = crossings.lower[:, None, :] + INCIDENT_CELL_OFFSETS[crossings.axis]
    cell_shape = np.asarray(dual.mixed_index.shape)
    if np.any(cells < 0) or np.any(cells >= cell_shape):
        raise ValueError(
            "ImplicitSurfaceDiscovery.cells must remain inside the grid boundary; got exterior incident cells"
        )
    upper = crossings.lower + np.eye(3, dtype=np.int64)[crossings.axis]
    lower_inside = inside[tuple(crossings.lower.T)]
    inside_point = np.where(lower_inside[:, None], crossings.lower, upper)
    corner_offset = inside_point[:, None, :] - cells
    corner = _CORNER_LOOKUP[
        corner_offset[..., 0], corner_offset[..., 1], corner_offset[..., 2]
    ]
    cell_key = tuple(np.moveaxis(cells, -1, 0))
    mixed = dual.mixed_index[cell_key]
    component = _COMPONENT_OF[dual.configuration[cell_key], corner]
    if np.any(mixed < 0) or np.any(component < 0):
        raise ValueError(
            "ImplicitSurfaceDiscovery.incidence must represent every required dual vertex and component; got missing incidences"
        )
    quads = dual.first_vertex[mixed] + component
    ordered = np.sort(quads, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError(
            "ImplicitSurfaceDiscovery.dual_faces must have distinct ordered vertices; got a collapsed dual face"
        )
    diagonal_02 = np.linalg.norm(
        base_vertices[quads[:, 0]] - base_vertices[quads[:, 2]], axis=1
    )
    diagonal_13 = np.linalg.norm(
        base_vertices[quads[:, 1]] - base_vertices[quads[:, 3]], axis=1
    )
    short_02 = (diagonal_02 <= diagonal_13)[:, None]
    first = np.where(short_02, quads[:, (0, 1, 2)], quads[:, (0, 1, 3)])
    second = np.where(short_02, quads[:, (0, 2, 3)], quads[:, (1, 2, 3)])
    faces = np.stack((first, second), axis=1).reshape((-1, 3))
    if faces.shape[0] > policy.maximum_faces:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.faces must fit maximum_faces {policy.maximum_faces}; got {faces.shape[0]}"
        )
    triangles = base_vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    area = 0.5 * np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(area)) or np.any(area <= policy.minimum_face_area):
        raise ValueError(
            f"ImplicitSurfaceDiscovery.triangle_areas must be finite and exceed minimum_face_area {policy.minimum_face_area}; got a nonfinite or undersized triangle"
        )
    # One fixed-shape field-and-gradient query per face centroid.
    consume_queries(budget, faces.shape[0])
    _, gradients = field_and_gradient(
        geometry.kernel,
        geometry.state,
        jnp.asarray(np.mean(triangles, axis=1)),
    )
    gradients = np.asarray(gradients)
    if not np.all(np.isfinite(gradients)) or np.any(
        np.linalg.norm(gradients, axis=1) == 0.0
    ):
        raise ValueError(
            "ImplicitSurfaceDiscovery.face_gradients must be finite and nonzero; got nonfinite or zero gradients"
        )
    flip = np.sum(normals * gradients, axis=1) < 0.0
    return np.where(flip[:, None], faces[:, (0, 2, 1)], faces).astype(np.int32)


def metered_self_overlap_pairs(
    hierarchy: PackedBVH,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    absolute_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Touching self-overlap pairs of ``hierarchy``, sorted as `bvh_overlap_pairs_host`.

    Every node and item AABB-test batch, rejected tests included, is charged
    to ``budget`` as work before it is evaluated. A self query on ``F`` items
    tests at most ``(2F - 1)**2`` node pairs and ``F**2`` item pairs.
    """
    blocks = list(
        bvh_overlap_pair_blocks(
            hierarchy,
            hierarchy,
            include_touching=True,
            absolute_tolerance=absolute_tolerance,
            visit=lambda count: consume_queries(budget, 0, work_units=count),
        )
    )
    if not blocks:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy()
    first = np.concatenate([block[0] for block in blocks])
    second = np.concatenate([block[1] for block in blocks])
    order = np.lexsort((second, first))
    return first[order], second[order]


def _lattice_overlap_pairs(
    lower_cell: np.ndarray,
    upper_cell: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    box_min: np.ndarray,
    box_max: np.ndarray,
    padding: float,
    tolerance: float,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Unordered touching overlaps of lattice-cell boxes, sorted by (first, second).

    Box ``i`` spans lattice cells ``lower_cell[i] .. upper_cell[i]`` widened by
    ``tolerance``; overlap is the `bvh_overlap_pairs_host` touching predicate
    with absolute tolerance ``padding``. Boxes separated on an axis by at least
    ``reach`` whole cells are disjoint under that predicate with rounding
    margin, so only bucketed lattice-window candidates are tested exactly; the
    result equals the BVH self query restricted to ``first < second``.

    Charged work is one unit per exact box-overlap predicate, charged before
    the tests run, matching the per-AABB-test convention of the BVH query;
    bucketing and window lookups are uncharged host passes.
    """
    count = int(lower_cell.shape[0])
    shape = tuple(int(coordinates.size) - 1 for coordinates in axes)
    reach = []
    for coordinates, cells in zip(axes, shape, strict=True):
        unit = np.finfo(np.float64).eps * (
            1.0 + float(np.max(np.abs(coordinates))) + tolerance
        )
        separation = 2.0 * (padding + 2.0 * tolerance) + 4.0 * unit
        k = 1
        while k < cells and float(np.min(coordinates[k:] - coordinates[:-k])) < separation:
            k += 1
        reach.append(k)
    span = upper_cell - lower_cell
    span_max = np.max(span, axis=0) if count else np.zeros((3,), dtype=np.int64)
    cell_count = shape[0] * shape[1] * shape[2]
    key = (lower_cell[:, 0] * shape[1] + lower_cell[:, 1]) * shape[2] + lower_cell[:, 2]
    offsets = np.zeros((cell_count + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(key, minlength=cell_count))
    order = np.argsort(key, kind="stable")
    position = np.empty((count,), dtype=np.int64)
    position[order] = np.arange(count, dtype=np.int64)

    # Each unordered pair is found once, from the box whose bucket comes first:
    # rows (dx, dy) with dx > 0, or dx == 0 and dy >= 0, over the z window.
    dx = np.arange(span_max[0] + reach[0] + 1, dtype=np.int64)
    dy = np.arange(-(span_max[1] + reach[1]), span_max[1] + reach[1] + 1, dtype=np.int64)
    row_dx, row_dy = (value.reshape((-1,)) for value in np.meshgrid(dx, dy, indexing="ij"))
    box = np.repeat(np.arange(count, dtype=np.int64), row_dx.size)
    row_dx = np.tile(row_dx, count)
    row_dy = np.tile(row_dy, count)
    x = lower_cell[box, 0] + row_dx
    y = lower_cell[box, 1] + row_dy
    visited = (
        (row_dx <= span[box, 0] + reach[0])
        & (row_dy <= span[box, 1] + reach[1])
        & ((row_dx > 0) | (row_dy >= 0))
        & (x < shape[0])
        & (y >= 0)
        & (y < shape[1])
    )
    box, row_dx, row_dy, x, y = (
        value[visited] for value in (box, row_dx, row_dy, x, y)
    )
    z_lower = np.maximum(lower_cell[box, 2] - span_max[2] - reach[2], 0)
    z_upper = np.minimum(upper_cell[box, 2] + reach[2], shape[2] - 1)
    column = (x * shape[1] + y) * shape[2]
    start = offsets[column + z_lower]
    end = offsets[column + z_upper + 1]
    own_row = (row_dx == 0) & (row_dy == 0)
    start = np.where(own_row, np.maximum(start, position[box] + 1), start)
    lengths = np.maximum(end - start, 0)
    total = int(np.sum(lengths))
    consume_queries(budget, 0, work_units=total)
    first_slot = np.cumsum(lengths) - lengths
    slots = (
        np.arange(total, dtype=np.int64)
        - np.repeat(first_slot, lengths)
        + np.repeat(start, lengths)
    )
    first = np.repeat(box, lengths)
    second = order[slots]
    extent = np.minimum(box_max[first], box_max[second]) - np.maximum(
        box_min[first], box_min[second]
    )
    hit = np.all(extent >= -padding, axis=-1)
    first, second = first[hit], second[hit]
    first, second = np.minimum(first, second), np.maximum(first, second)
    sorted_rows = np.lexsort((second, first))
    return first[sorted_rows], second[sorted_rows]


def _intersection_candidates(
    faces: np.ndarray,
    vertex_cell: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    policy: ImplicitSurfacePolicy,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Enumerate all self-intersection candidates for every accepted realization.

    An accepted realization keeps each vertex inside its discovery cell expanded
    by root_tolerance, so a triangle stays inside the union box of its vertices'
    cells. The intersection predicate accepts parametric slack root_tolerance,
    which reaches at most root_tolerance times twice that box diagonal beyond the
    triangle. Face pairs whose padded boxes are disjoint therefore can never be
    reported as intersecting, and the box overlap set contains every intersecting pair for this test.
    """

    tolerance = policy.projection.root_tolerance
    box_min = np.min(cell_lower[faces], axis=1) - tolerance
    box_max = np.max(cell_upper[faces], axis=1) + tolerance
    diagonal = float(np.max(np.linalg.norm(box_max - box_min, axis=1)))
    padding = tolerance * (1.0 + 2.0 * diagonal)
    corner_cells = vertex_cell[faces]
    first, second = _lattice_overlap_pairs(
        np.min(corner_cells, axis=1),
        np.max(corner_cells, axis=1),
        axes,
        box_min,
        box_max,
        padding,
        tolerance,
        budget,
    )
    # Pairs arrive with first < second.
    shares_vertex = np.any(
        faces[first][:, :, None] == faces[second][:, None, :], axis=(1, 2)
    )
    pairs = np.stack((first[~shares_vertex], second[~shares_vertex]), axis=1)
    if pairs.shape[0] > policy.maximum_intersection_pairs:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.intersection_pairs must fit maximum_intersection_pairs {policy.maximum_intersection_pairs}; got {pairs.shape[0]}"
        )
    return pairs.astype(np.int32)


def _realization_counts(
    lattice: int, anchors: int, steps: int, vertices: int, faces: int, pairs: int, /
) -> tuple[int, int]:
    """Work and geometry queries of one fixed-shape realization of these sizes."""
    work = anchors * steps + vertices + faces + pairs
    return work, lattice + anchors * (steps + 1)


def implicit_realization_operations(
    plan: realization_owner.ImplicitSurfacePlan, /, *, retained_proof: bool = False
) -> tuple[int, int]:
    """Return the exact work and geometry queries of one fixed-shape realization.

    ``ImplicitSurfacePlan.realize`` has no data-dependent control flow: the
    projection runs ``maximum_steps`` iterations over every anchor, and every
    vertex, face and candidate pair is evaluated. Queries: one lattice field
    sample per grid point and one field-and-gradient evaluation per anchor for
    every projection step plus the final evaluation. Work: one projection step
    per anchor and step, one dense QEF solve per vertex, one area/orientation
    evaluation per face and one triangle test per candidate pair.
    With the exact-state retained proof, only projection and QEF coordinates
    are recomputed; no lattice, face or intersection predicate is repeated.
    """
    return _realization_counts(
        0 if retained_proof else int(plan.grid_points.shape[0]),
        int(plan.projection.anchors.shape[0]),
        int(plan.policy.projection.maximum_steps),
        int(plan.base_vertices.shape[0]),
        0 if retained_proof else int(plan.faces.shape[0]),
        0 if retained_proof else int(plan.intersection_pairs.shape[0]),
    )


def _discovery_upper(policy: ImplicitSurfacePolicy, /) -> tuple[int, int]:
    """Policy-capacity work and query bound of one fixed-lattice discovery.

    Lattice samples, at most ``_EDGE_ROOT_MAXIMUM_STEPS`` shared ITP iterations
    plus the final residual per crossing, one anchor and one face gradient, at
    most ``_QEF_REGULARIZATION_LEVELS`` QEF levels per vertex, at most
    ``F (F - 1) / 2`` exact candidate box tests, and the base realization of
    `implicit_realization_operations` at policy capacity.
    """
    lattice, crossings = policy.maximum_lattice_points, policy.maximum_crossings
    vertices, faces = policy.maximum_vertices, policy.maximum_faces
    steps = policy.projection.maximum_steps
    work = (
        crossings * _EDGE_ROOT_MAXIMUM_STEPS
        + vertices * _QEF_REGULARIZATION_LEVELS
        + crossings * steps
        + vertices
        + faces
        + policy.maximum_intersection_pairs
        + faces * (faces - 1) // 2
    )
    queries = (
        lattice
        + crossings * (_EDGE_ROOT_MAXIMUM_STEPS + 1)
        + crossings
        + faces
        + lattice
        + crossings * (steps + 1)
    )
    return work, queries


def _admitted_batch(
    budget: CoordinateEnclosureBudget, work_upper: int, queries_upper: int, /
) -> BatchTicket:
    """Admit a conditional batch at its upper bound; the caller consumes actuals."""
    ticket = budget.ticket
    if (
        work_upper > ticket.remaining_work_units
        or queries_upper > ticket.remaining_geometry_queries
    ):
        ticket.refuse(work_upper, queries_upper=queries_upper)
    return ticket.batch(work_upper, queries_upper=queries_upper)


@checked
def discover_implicit_surface(
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    /,
    *,
    policy: ImplicitSurfacePolicy = _DEFAULT_SURFACE_POLICY,
    source_id: str,
    budget: CoordinateEnclosureBudget | None = None,
) -> realization_owner.ImplicitSurfacePlan:
    """Discover a closed manifold dual surface and freeze its topology.

    Every lattice sample, ITP root iteration, anchor and face gradient, QEF
    regularization level and the base realization is charged to ``budget``.
    Fixed-shape stages are charged before dispatch; the conditional ITP and
    QEF stages are admitted at their upper bound and charge the realized
    iteration and level counts. Without a budget an input-bounded host request
    admits the policy capacity.
    """
    work_upper, queries_upper = _discovery_upper(policy)
    with query_budget(
        budget, work_upper=work_upper, queries_upper=queries_upper
    ) as ledger:
        return _discover_implicit_surface(
            geometry, grid, policy=policy, source_id=source_id, budget=ledger
        )


def _discover_implicit_surface(
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    /,
    *,
    policy: ImplicitSurfacePolicy,
    source_id: str,
    budget: CoordinateEnclosureBudget,
) -> realization_owner.ImplicitSurfacePlan:
    _validate_discovery_inputs(geometry, grid, policy, source_id)
    axes = tuple(
        np.asarray(axis.point_coordinates, dtype=np.float64)
        for axis in grid.structured_axes
    )
    if any(axis.size < 2 or np.any(np.diff(axis) <= 0.0) for axis in axes):
        raise ValueError(
            "ImplicitSurfaceDiscovery.grid.axes must contain at least two strictly increasing coordinates; got an undersized or unordered axis"
        )
    shape = tuple(axis.size for axis in axes)
    if np.prod(shape, dtype=np.int64) > policy.maximum_lattice_points:
        raise ValueError(
            f"ImplicitSurfaceDiscovery.lattice must fit maximum_lattice_points {policy.maximum_lattice_points}; got shape {shape}"
        )
    lattice_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    consume_queries(budget, int(np.prod(shape, dtype=np.int64)))
    values = np.asarray(
        geometry.boundary_field(jnp.asarray(lattice_points.reshape((-1, 3))))
    ).reshape(shape)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "ImplicitSurfaceDiscovery.lattice.values must be finite; got nonfinite values"
        )
    if np.any(np.abs(values) <= policy.lattice_zero_tolerance):
        raise ValueError(
            f"ImplicitSurfaceDiscovery.lattice.values must exclude ambiguous zero samples; got samples within tolerance {policy.lattice_zero_tolerance}"
        )
    inside = values < 0.0
    crossings = _lattice_crossings(inside, policy)

    lower_key = tuple(crossings.lower.T)
    upper_key = tuple((crossings.lower + np.eye(3, dtype=np.int64)[crossings.axis]).T)
    lower_points = lattice_points[lower_key]
    upper_points = lattice_points[upper_key]
    lanes = int(crossings.lower.shape[0])
    # The shared batch stops at a data-dependent iteration; every iteration and
    # the final residual evaluate all lanes. Admit the bound, charge the actual.
    with _admitted_batch(
        budget,
        lanes * _EDGE_ROOT_MAXIMUM_STEPS,
        lanes * (_EDGE_ROOT_MAXIMUM_STEPS + 1),
    ) as admitted:
        isolation = isolate_roots(
            geometry.kernel,
            geometry.state,
            jnp.asarray(lower_points),
            jnp.asarray(upper_points),
            jnp.asarray(values[lower_key]),
            jnp.asarray(values[upper_key]),
            jnp.asarray(policy.projection.root_tolerance, dtype=jnp.float64),
        )
        diagnostics = isolation.native_result.nonlinear_result.diagnostics
        iterations = int(jax.device_get(diagnostics.iterations))
        admitted.consume(
            lanes * iterations, geometry_queries=lanes * (iterations + 1)
        )
    anchors = np.asarray(isolation.points)
    residuals = isolation.residuals
    if np.any(np.asarray(residuals) > policy.projection.root_tolerance):
        raise ValueError(
            "ImplicitDiscovery.root_residuals must meet root_tolerance; got residuals above the declared tolerance"
        )
    isolation.require_converged()
    trust_radii = policy.projection.trust_fraction * np.linalg.norm(
        upper_points - lower_points, axis=1
    )

    dual = _dual_vertices(inside, crossings, policy)
    axis_coordinates = (axes[0], axes[1], axes[2])
    cell_lower = np.stack(
        [axis_coordinates[axis][dual.cell[:, axis]] for axis in range(3)], axis=1
    )
    cell_upper = np.stack(
        [axis_coordinates[axis][dual.cell[:, axis] + 1] for axis in range(3)], axis=1
    )
    consume_queries(budget, int(anchors.shape[0]))
    _, anchor_gradients = field_and_gradient(
        geometry.kernel,
        geometry.state,
        jnp.asarray(anchors),
    )
    anchor_gradients = np.asarray(anchor_gradients)
    base_vertices, qef_regularization = qef_vertices(
        anchors,
        np.asarray(anchor_gradients),
        dual.anchor_indices,
        dual.anchor_mask,
        cell_lower,
        cell_upper,
        policy.qef_regularization,
        policy.projection.root_tolerance,
        budget=budget,
    )
    faces = _dual_faces(geometry, inside, crossings, dual, base_vertices, policy, budget)
    # ty: ignore[invalid-argument-type]
    topology = TriangleTopology(faces, num_vertices=base_vertices.shape[0])
    if not topology.watertight:
        raise ValueError(
            "ImplicitSurfaceDiscovery.topology must be watertight; got an open surface"
        )
    pairs = _intersection_candidates(
        faces, dual.cell, axis_coordinates, cell_lower, cell_upper, policy, budget
    )
    base_triangles = base_vertices[faces]
    base_face_normals = np.cross(
        base_triangles[:, 1] - base_triangles[:, 0],
        base_triangles[:, 2] - base_triangles[:, 0],
    )

    projection = ImplicitPointProjectionPlan(
        geometry,
        anchors,
        trust_radii,
        policy=policy.projection,
        source_id=f"{source_id}:anchors",
    )
    # The plan realizes and retains the discovery state once at construction.
    work, queries = _realization_counts(
        int(lattice_points.size // 3),
        int(anchors.shape[0]),
        int(policy.projection.maximum_steps),
        int(base_vertices.shape[0]),
        int(faces.shape[0]),
        int(pairs.shape[0]),
    )
    consume_queries(budget, queries, work_units=work)
    return realization_owner.ImplicitSurfacePlan(
        geometry=geometry,
        grid_points=lattice_points.reshape((-1, 3)),
        inside_pattern=inside.reshape((-1,)),
        projection=projection,
        vertex_anchor_indices=dual.anchor_indices,
        vertex_anchor_mask=dual.anchor_mask,
        qef_regularization=qef_regularization,
        cell_lower=cell_lower,
        cell_upper=cell_upper,
        base_vertices=base_vertices,
        faces=faces,
        base_face_normals=base_face_normals,
        intersection_pairs=pairs,
        policy=policy,
        source_id=source_id,
        topology_id=topology.cell_complex_topology().topology_id,
        initial_state=geometry.state,
    )


__all__ = ["discover_implicit_surface"]
