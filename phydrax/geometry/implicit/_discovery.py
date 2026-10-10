#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._bvh import bvh_overlap_pairs_host, prepare_bvh
from ...discretization._tensor_support import PreparedTensorGrid
from ...ein import contract
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from .._certificate import FieldRegularity, SignReliability, ZeroSetAccuracy
from .._contracts import CompiledGeometry, GeometryKernel, GeometryKind
from ..design._schema import DesignState
from ..simplicial import TriangleTopology
from ._policy import ImplicitSurfacePolicy
from ._projection import _field_and_gradient, ImplicitPointProjectionPlan


_DEFAULT_SURFACE_POLICY = ImplicitSurfacePolicy()

# Cube corners in the dual-contouring convention: corner c has lattice offset
# _CORNER_OFFSETS[c] relative to the cell's lower lattice point.
_CORNER_OFFSETS = np.asarray(
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
_CORNER_LOOKUP[tuple(_CORNER_OFFSETS.T)] = np.arange(8, dtype=np.int64)
_EDGE_AXIS = np.argmax(
    _CORNER_OFFSETS[_CUBE_EDGES[:, 0]] != _CORNER_OFFSETS[_CUBE_EDGES[:, 1]],
    axis=1,
)
_EDGE_LOWER = np.minimum(
    _CORNER_OFFSETS[_CUBE_EDGES[:, 0]],
    _CORNER_OFFSETS[_CUBE_EDGES[:, 1]],
)
# Four cells around a lattice edge in cyclic order; row `axis` of this table
# lists cell offsets relative to the edge's lower lattice point.
_INCIDENT_CELL_OFFSETS = np.asarray(
    (
        ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)),
        ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)),
        ((-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)),
    ),
    dtype=np.int64,
)
_QEF_REGULARIZATION_LEVELS = 12
_QEF_SOLVE_PLAN = SmallLinearSolvePlan(3)
# ITP root isolation (Oliveira & Takahashi 2020) on the unit edge parameter.
_ITP_BRACKET_TOLERANCE = 2.0**-50
_ITP_MAXIMUM_ITERATIONS = 51
_ITP_K1 = 0.2
_ITP_K2 = 2.0


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
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    if not isinstance(grid, PreparedTensorGrid):
        raise TypeError("grid must be PreparedTensorGrid.")
    if not isinstance(policy, ImplicitSurfacePolicy):
        raise TypeError("policy must be ImplicitSurfacePolicy.")
    if not source_id:
        raise ValueError("source_id must be non-empty.")
    if geometry.ambient_dimension != 3 or geometry.kind is not GeometryKind.REGION:
        raise ValueError(
            "Implicit surface discovery requires a three-dimensional region."
        )
    if len(grid.structured_axes) != 3:
        raise ValueError("Implicit surface discovery requires a three-dimensional grid.")
    if any(axis.periodic for axis in grid.structured_axes):
        raise ValueError("Implicit surface discovery requires nonperiodic axes.")
    if not bool(np.asarray(geometry.validity().accepted)):
        raise ValueError("Implicit surface discovery geometry must be valid.")
    certificate = geometry.field_certificate
    if certificate.sign_reliability is not SignReliability.RELIABLE:
        raise ValueError("Implicit surface discovery requires reliable field sign.")
    if (
        certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
        and not policy.allow_approximate_zero_set
    ):
        raise ValueError("Approximate zero sets require explicit policy approval.")
    if (
        certificate.regularity is FieldRegularity.NONSMOOTH
        and not policy.allow_nonsmooth_field
    ):
        raise ValueError("Nonsmooth fields require explicit selected-branch approval.")
    if not certificate.parameter_differentiable:
        raise ValueError("Implicit realization requires parameter-differentiable fields.")


def _lattice_crossings(inside: np.ndarray, policy: ImplicitSurfacePolicy) -> _Crossings:
    """Sign-change lattice edges in canonical (axis, i, j, k) order."""

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
        raise ValueError("Implicit grid contains no surface crossings.")
    if offset > policy.maximum_crossings:
        raise ValueError("Implicit surface exceeds maximum_crossings.")
    return _Crossings(
        np.concatenate(axes),
        np.concatenate(lowers),
        (edge_index[0], edge_index[1], edge_index[2]),
    )


@eqx.filter_jit
def _isolate_roots(
    kernel: GeometryKernel,
    state: DesignState,
    lower_points: Array,
    upper_points: Array,
    lower_values: Array,
    upper_values: Array,
    tolerance: Array,
) -> tuple[Array, Array]:
    """Bracketed ITP root isolation over every lattice crossing in one program.

    Each lane solves f(lower + t (upper - lower)) = 0 on t in [0, 1]. ITP keeps
    bisection's worst-case iteration bound while converging superlinearly on
    smooth fields. Lanes freeze once |f| <= tolerance.
    """

    direction = upper_points - lower_points

    def field(parameter: Any) -> Any:
        return kernel.boundary_field(state, lower_points + parameter[:, None] * direction)

    count = lower_points.shape[0]
    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((count,), dtype=lower_points.dtype),
        jnp.ones((count,), dtype=lower_points.dtype),
        lower_values,
        upper_values,
        jnp.zeros((count,), dtype=lower_points.dtype),
        jnp.zeros((count,), dtype=jnp.bool_),
    )

    def active(carry: Any) -> Any:
        _, left, right, _, _, _, converged = carry
        return ~converged & (right - left > 2.0 * _ITP_BRACKET_TOLERANCE)

    def condition(carry: Any) -> Any:
        return (carry[0] < _ITP_MAXIMUM_ITERATIONS) & jnp.any(active(carry))

    def body(carry: Any) -> Any:
        iteration, left, right, left_value, right_value, root, converged = carry
        running = active(carry)
        width = right - left
        middle = 0.5 * (left + right)
        radius = jnp.maximum(
            _ITP_BRACKET_TOLERANCE
            * jnp.exp2(
                (_ITP_MAXIMUM_ITERATIONS - 1 - iteration).astype(lower_points.dtype)
            )
            - 0.5 * width,
            0.0,
        )
        truncation = _ITP_K1 * width**_ITP_K2
        falsi = (right_value * left - left_value * right) / (right_value - left_value)
        falsi = jnp.where(jnp.isfinite(falsi), falsi, middle)
        sigma = jnp.sign(middle - falsi)
        truncated = jnp.where(
            truncation <= jnp.abs(middle - falsi), falsi + sigma * truncation, middle
        )
        candidate = jnp.where(
            jnp.abs(truncated - middle) <= radius, truncated, middle - sigma * radius
        )
        value = field(candidate)
        same_as_left = (value < 0.0) == (left_value < 0.0)
        move_left = running & same_as_left
        move_right = running & ~same_as_left
        hit = running & (jnp.abs(value) <= tolerance)
        return (
            iteration + 1,
            jnp.where(move_left, candidate, left),
            jnp.where(move_right, candidate, right),
            jnp.where(move_left, value, left_value),
            jnp.where(move_right, value, right_value),
            jnp.where(hit, candidate, root),
            converged | hit,
        )

    _, left, right, _, _, root, converged = jax.lax.while_loop(condition, body, initial)
    parameter = jnp.where(converged, root, 0.5 * (left + right))
    points = lower_points + parameter[:, None] * direction
    return points, jnp.abs(kernel.boundary_field(state, points))


def _dual_vertices(
    inside: np.ndarray,
    crossings: _Crossings,
    policy: ImplicitSurfacePolicy,
) -> _DualVertices:
    """One dual vertex per inside-corner component of every mixed cell.

    Vertices are numbered by row-major cell order, then component order.
    """

    configuration = np.zeros(tuple(size - 1 for size in inside.shape), dtype=np.int64)
    for corner, (dx, dy, dz) in enumerate(_CORNER_OFFSETS):
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
        raise ValueError("Implicit surface discovery produced no dual vertices.")
    if vertex_count > policy.maximum_vertices:
        raise ValueError("Implicit surface exceeds maximum_vertices.")
    first_vertex = np.concatenate(((0,), np.cumsum(counts)[:-1])).astype(np.int64)
    mixed_index = np.full(configuration.shape, -1, dtype=np.int64)
    mixed_index[mixed] = np.arange(mixed_cells.shape[0], dtype=np.int64)
    owner = np.repeat(np.arange(mixed_cells.shape[0], dtype=np.int64), counts)
    component = np.arange(vertex_count, dtype=np.int64) - first_vertex[owner]
    vertex_cell = mixed_cells[owner]
    vertex_configuration = mixed_configuration[owner]

    edge_cells = vertex_cell[:, None, :] + _EDGE_LOWER[None, :, :]
    candidates = np.full((vertex_count, 12), -1, dtype=np.int64)
    for axis in range(3):
        selected = _EDGE_AXIS == axis
        cells = edge_cells[:, selected]
        candidates[:, selected] = crossings.edge_index[axis][
            cells[..., 0], cells[..., 1], cells[..., 2]
        ]
    member = _EDGE_COMPONENT[vertex_configuration] == component[:, None]
    if np.any(member & (candidates < 0)):
        raise ValueError("Implicit manifold incidence is incomplete.")
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


def _qef_vertices(
    anchors: np.ndarray,
    gradients: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_mask: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    regularization: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched regularized QEF vertices with bounded regularization escalation."""

    norms = np.linalg.norm(gradients, axis=-1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("Implicit QEF anchors require finite nonzero gradients.")
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
    vertices = np.zeros_like(cell_lower)
    selected_regularization = np.full((cell_lower.shape[0],), np.nan, dtype=np.float64)
    done = np.zeros((cell_lower.shape[0],), dtype=np.bool_)
    identity = np.eye(3, dtype=np.float64)
    # Every level solves every vertex system so one batch shape serves all
    # levels; each vertex keeps its first in-cell solution.
    for level in range(_QEF_REGULARIZATION_LEVELS):
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
        "Implicit QEF vertex left its discovery cell even after bounded regularization adaptation."
    )


def _dual_faces(
    geometry: CompiledGeometry,
    inside: np.ndarray,
    crossings: _Crossings,
    dual: _DualVertices,
    base_vertices: np.ndarray,
    policy: ImplicitSurfacePolicy,
) -> np.ndarray:
    """Split every crossing's dual quad into two field-oriented triangles."""

    cells = crossings.lower[:, None, :] + _INCIDENT_CELL_OFFSETS[crossings.axis]
    cell_shape = np.asarray(dual.mixed_index.shape)
    if np.any(cells < 0) or np.any(cells >= cell_shape):
        raise ValueError("Implicit surface intersects the outer grid boundary.")
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
        raise ValueError("Implicit manifold incidence is incomplete.")
    quads = dual.first_vertex[mixed] + component
    ordered = np.sort(quads, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("Implicit manifold incidence produced a collapsed dual face.")
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
        raise ValueError("Implicit surface face count is empty or exceeds policy.")
    triangles = base_vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    area = 0.5 * np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(area)) or np.any(area <= policy.minimum_face_area):
        raise ValueError("Implicit surface discovery produced a degenerate triangle.")
    _, gradients = _field_and_gradient(
        geometry.kernel,
        geometry.state,
        jnp.asarray(np.mean(triangles, axis=1)),
    )
    gradients = np.asarray(gradients)
    if not np.all(np.isfinite(gradients)) or np.any(
        np.linalg.norm(gradients, axis=1) == 0.0
    ):
        raise ValueError("Implicit face orientation requires a regular field gradient.")
    flip = np.sum(normals * gradients, axis=1) < 0.0
    return np.where(flip[:, None], faces[:, (0, 2, 1)], faces).astype(np.int32)


def _intersection_candidates(
    faces: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    policy: ImplicitSurfacePolicy,
) -> np.ndarray:
    """Complete self-intersection candidates for every accepted realization.

    An accepted realization keeps each vertex inside its discovery cell expanded
    by root_tolerance, so a triangle stays inside the union box of its vertices'
    cells. The intersection predicate accepts parametric slack root_tolerance,
    which reaches at most root_tolerance times twice that box diagonal beyond the
    triangle. Face pairs whose padded boxes are disjoint therefore can never be
    reported as intersecting, and the BVH overlap set is exact for this test.
    """

    tolerance = policy.projection.root_tolerance
    box_min = np.min(cell_lower[faces], axis=1) - tolerance
    box_max = np.max(cell_upper[faces], axis=1) + tolerance
    diagonal = float(np.max(np.linalg.norm(box_max - box_min, axis=1)))
    padding = tolerance * (1.0 + 2.0 * diagonal)
    hierarchy = prepare_bvh(box_min, box_max, dtype=jnp.float64)
    first, second = bvh_overlap_pairs_host(
        hierarchy,
        hierarchy,
        include_touching=True,
        absolute_tolerance=padding,
    )
    ordered = first < second
    first = first[ordered]
    second = second[ordered]
    shares_vertex = np.any(
        faces[first][:, :, None] == faces[second][:, None, :], axis=(1, 2)
    )
    pairs = np.stack((first[~shares_vertex], second[~shares_vertex]), axis=1)
    if pairs.shape[0] > policy.maximum_intersection_pairs:
        raise ValueError("Implicit surface exceeds maximum_intersection_pairs.")
    return pairs.astype(np.int32)


def discover_implicit_surface(
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    /,
    *,
    policy: ImplicitSurfacePolicy = _DEFAULT_SURFACE_POLICY,
    source_id: str,
) -> Any:
    """Discover a closed manifold dual surface and freeze its topology."""

    _validate_discovery_inputs(geometry, grid, policy, source_id)
    axes = tuple(
        np.asarray(axis.point_coordinates, dtype=np.float64)
        for axis in grid.structured_axes
    )
    if any(axis.size < 2 or np.any(np.diff(axis) <= 0.0) for axis in axes):
        raise ValueError("Implicit grid axes must contain increasing point coordinates.")
    shape = tuple(axis.size for axis in axes)
    if np.prod(shape, dtype=np.int64) > policy.maximum_lattice_points:
        raise ValueError("Implicit lattice exceeds maximum_lattice_points.")
    lattice_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = np.asarray(
        geometry.boundary_field(jnp.asarray(lattice_points.reshape((-1, 3))))
    ).reshape(shape)
    if not np.all(np.isfinite(values)):
        raise ValueError("Implicit lattice field contains nonfinite values.")
    if np.any(np.abs(values) <= policy.lattice_zero_tolerance):
        raise ValueError(
            "Implicit lattice contains an ambiguous zero; shift or refine the grid."
        )
    inside = values < 0.0
    crossings = _lattice_crossings(inside, policy)

    lower_key = tuple(crossings.lower.T)
    upper_key = tuple((crossings.lower + np.eye(3, dtype=np.int64)[crossings.axis]).T)
    lower_points = lattice_points[lower_key]
    upper_points = lattice_points[upper_key]
    anchors, residuals = _isolate_roots(
        geometry.kernel,
        geometry.state,
        jnp.asarray(lower_points),
        jnp.asarray(upper_points),
        jnp.asarray(values[lower_key]),
        jnp.asarray(values[upper_key]),
        jnp.asarray(policy.projection.root_tolerance, dtype=jnp.float64),
    )
    anchors = np.asarray(anchors)
    if np.any(np.asarray(residuals) > policy.projection.root_tolerance):
        raise ValueError("Implicit root isolation did not meet root_tolerance.")
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
    _, anchor_gradients = _field_and_gradient(
        geometry.kernel,
        geometry.state,
        jnp.asarray(anchors),
    )
    base_vertices, qef_regularization = _qef_vertices(
        anchors,
        np.asarray(anchor_gradients),
        dual.anchor_indices,
        dual.anchor_mask,
        cell_lower,
        cell_upper,
        policy.qef_regularization,
        policy.projection.root_tolerance,
    )
    faces = _dual_faces(geometry, inside, crossings, dual, base_vertices, policy)
    # ty: ignore[invalid-argument-type]
    topology = TriangleTopology(faces, num_vertices=base_vertices.shape[0])
    if not topology.watertight:
        raise ValueError("Implicit surface discovery did not produce a closed surface.")
    pairs = _intersection_candidates(faces, cell_lower, cell_upper, policy)
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
    from ._realization import ImplicitSurfacePlan

    plan = ImplicitSurfacePlan(
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
    )
    base = plan.realize(geometry.state)
    if not bool(np.asarray(base.accepted)):
        raise ValueError("Implicit surface base realization failed runtime evidence.")
    return plan


__all__ = ["discover_implicit_surface"]
