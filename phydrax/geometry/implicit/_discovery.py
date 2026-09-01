#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

from ...discretization import PreparedTensorGrid
from .._certificate import FieldRegularity, SignReliability, ZeroSetAccuracy
from .._contracts import CompiledGeometry, GeometryKind
from ..simplicial import TriangleTopology
from ._policy import ImplicitSurfacePolicy
from ._projection import _field_and_gradient, ImplicitPointProjectionPlan


_DEFAULT_SURFACE_POLICY = ImplicitSurfacePolicy()


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
    dtype=np.int32,
)
_CORNER_INDEX = {tuple(offset): index for index, offset in enumerate(_CORNER_OFFSETS)}
_CUBE_EDGES = (
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
)


def _edge_key(cell: tuple[int, int, int], first: int, second: int):
    first_offset = _CORNER_OFFSETS[first]
    second_offset = _CORNER_OFFSETS[second]
    axis = int(np.flatnonzero(first_offset != second_offset)[0])
    lower_offset = np.minimum(first_offset, second_offset)
    lower = np.asarray(cell, dtype=np.int32) + lower_offset
    return axis, int(lower[0]), int(lower[1]), int(lower[2])


def _inside_components(corner_inside: np.ndarray) -> tuple[tuple[int, ...], ...]:
    adjacency = {index: [] for index in np.flatnonzero(corner_inside).tolist()}
    for first, second in _CUBE_EDGES:
        if corner_inside[first] and corner_inside[second]:
            adjacency[first].append(second)
            adjacency[second].append(first)
    remaining = set(adjacency)
    components: list[tuple[int, ...]] = []
    while remaining:
        first = min(remaining)
        pending = deque((first,))
        component: list[int] = []
        remaining.remove(first)
        while pending:
            current = pending.popleft()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    pending.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _incident_cells(key: tuple[int, int, int, int]):
    axis, i, j, k = key
    if axis == 0:
        return (
            (i, j - 1, k - 1),
            (i, j, k - 1),
            (i, j, k),
            (i, j - 1, k),
        )
    if axis == 1:
        return (
            (i - 1, j, k - 1),
            (i - 1, j, k),
            (i, j, k),
            (i, j, k - 1),
        )
    return (
        (i - 1, j - 1, k),
        (i, j - 1, k),
        (i, j, k),
        (i - 1, j, k),
    )


def _bisect_root(
    geometry: CompiledGeometry,
    first: np.ndarray,
    second: np.ndarray,
    first_value: float,
    second_value: float,
    tolerance: float,
) -> np.ndarray:
    left = first.copy()
    right = second.copy()
    left_value = float(first_value)
    right_value = float(second_value)
    if (left_value < 0.0) == (right_value < 0.0):
        raise ValueError("Implicit root bracket must change sign.")
    for _ in range(64):
        middle = 0.5 * (left + right)
        value = float(np.asarray(geometry.boundary_field(jnp.asarray(middle))))
        if abs(value) <= tolerance:
            return middle
        if (value < 0.0) == (left_value < 0.0):
            left, left_value = middle, value
        else:
            right, right_value = middle, value
    root = 0.5 * (left + right)
    residual = abs(float(np.asarray(geometry.boundary_field(jnp.asarray(root)))))
    if residual > tolerance:
        raise ValueError("Implicit root bisection did not meet root_tolerance.")
    return root


def _base_qef_vertices(
    anchors: np.ndarray,
    gradients: np.ndarray,
    vertex_anchors: tuple[tuple[int, ...], ...],
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    regularization: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.zeros_like(cell_lower, dtype=float)
    regularizations = np.zeros((cell_lower.shape[0],), dtype=float)
    identity = np.eye(3)
    for vertex, anchor_indices in enumerate(vertex_anchors):
        points = anchors[np.asarray(anchor_indices, dtype=np.int32)]
        normals = gradients[np.asarray(anchor_indices, dtype=np.int32)]
        normal_norm = np.linalg.norm(normals, axis=-1)
        if np.any(normal_norm <= 0.0) or not np.all(np.isfinite(normal_norm)):
            raise ValueError("Implicit QEF anchors require finite nonzero gradients.")
        normals = normals / normal_norm[:, None]
        mass = np.mean(points, axis=0)
        count = float(len(anchor_indices))
        relative = points - mass
        right_hand_side = np.sum(
            normals * np.sum(normals * relative, axis=-1)[:, None],
            axis=0,
        )
        selected_regularization = float(regularization)
        value = mass
        accepted = False
        for _ in range(12):
            matrix = normals.T @ normals + selected_regularization * count * identity
            value = mass + np.linalg.solve(matrix, right_hand_side)
            finite = np.all(np.isfinite(value))
            in_cell = np.all(value >= cell_lower[vertex] - tolerance) and np.all(
                value <= cell_upper[vertex] + tolerance
            )
            if finite and in_cell:
                accepted = True
                break
            selected_regularization *= 10.0
        if not accepted:
            raise ValueError(
                "Implicit QEF vertex left its discovery cell even after bounded "
                "regularization adaptation."
            )
        vertices[vertex] = value
        regularizations[vertex] = selected_regularization
    return vertices, regularizations


def _oriented_triangle(
    geometry: CompiledGeometry,
    vertices: np.ndarray,
    indices: tuple[int, int, int],
    minimum_area: float,
) -> tuple[int, int, int]:
    triangle = vertices[np.asarray(indices, dtype=np.int32)]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    area = 0.5 * float(np.linalg.norm(normal))
    if not np.isfinite(area) or area <= minimum_area:
        raise ValueError("Implicit surface discovery produced a degenerate triangle.")
    centroid = np.mean(triangle, axis=0)

    def field(point):
        return geometry.boundary_field(point)

    gradient = np.asarray(jax.grad(field)(jnp.asarray(centroid)))
    if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) == 0.0:
        raise ValueError("Implicit face orientation requires a regular field gradient.")
    if float(np.dot(normal, gradient)) < 0.0:
        return indices[0], indices[2], indices[1]
    return indices


def discover_implicit_surface(
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    /,
    *,
    policy: ImplicitSurfacePolicy = _DEFAULT_SURFACE_POLICY,
    source_id: str,
):
    """Discover a closed manifold dual surface and freeze its topology."""

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

    axes = tuple(
        np.asarray(axis.point_coordinates, dtype=float) for axis in grid.structured_axes
    )
    if any(axis.size < 2 or np.any(np.diff(axis) <= 0.0) for axis in axes):
        raise ValueError("Implicit grid axes must contain increasing point coordinates.")
    mesh = np.meshgrid(*axes, indexing="ij")
    lattice_points = np.stack(mesh, axis=-1)
    point_count = int(np.prod(lattice_points.shape[:-1]))
    if point_count > policy.maximum_lattice_points:
        raise ValueError("Implicit lattice exceeds maximum_lattice_points.")
    values = np.asarray(
        geometry.boundary_field(jnp.asarray(lattice_points.reshape((-1, 3))))
    ).reshape(lattice_points.shape[:-1])
    if not np.all(np.isfinite(values)):
        raise ValueError("Implicit lattice field contains nonfinite values.")
    if np.any(np.abs(values) <= policy.lattice_zero_tolerance):
        raise ValueError(
            "Implicit lattice contains an ambiguous zero; shift or refine the grid."
        )
    inside = values < 0.0
    shape = values.shape

    crossing_keys: list[tuple[int, int, int, int]] = []
    for axis in range(3):
        first_slice = [slice(None)] * 3
        second_slice = [slice(None)] * 3
        first_slice[axis] = slice(0, shape[axis] - 1)
        second_slice[axis] = slice(1, shape[axis])
        changed = inside[tuple(first_slice)] != inside[tuple(second_slice)]
        for index in np.argwhere(changed):
            crossing_keys.append((axis, int(index[0]), int(index[1]), int(index[2])))
    crossing_keys.sort()
    if not crossing_keys:
        raise ValueError("Implicit grid contains no surface crossings.")
    if len(crossing_keys) > policy.maximum_crossings:
        raise ValueError("Implicit surface exceeds maximum_crossings.")

    crossing_index = {key: index for index, key in enumerate(crossing_keys)}
    anchors = np.zeros((len(crossing_keys), 3), dtype=float)
    trust_radii = np.zeros((len(crossing_keys),), dtype=float)
    inside_lattice: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
    for index, key in enumerate(crossing_keys):
        axis, i, j, k = key
        lower_index = np.asarray((i, j, k), dtype=np.int32)
        upper_index = lower_index.copy()
        upper_index[axis] += 1
        lower_tuple = tuple(int(value) for value in lower_index)
        upper_tuple = tuple(int(value) for value in upper_index)
        lower_point = lattice_points[lower_tuple]
        upper_point = lattice_points[upper_tuple]
        anchors[index] = _bisect_root(
            geometry,
            lower_point,
            upper_point,
            values[lower_tuple],
            values[upper_tuple],
            policy.projection.root_tolerance,
        )
        trust_radii[index] = policy.projection.trust_fraction * float(
            np.linalg.norm(upper_point - lower_point)
        )
        inside_lattice[key] = lower_tuple if inside[lower_tuple] else upper_tuple

    vertex_anchors: list[tuple[int, ...]] = []
    cell_lower: list[np.ndarray] = []
    cell_upper: list[np.ndarray] = []
    corner_vertex: dict[tuple[int, int, int, int], int] = {}
    cell_shape = tuple(size - 1 for size in shape)
    for i in range(cell_shape[0]):
        for j in range(cell_shape[1]):
            for k in range(cell_shape[2]):
                cell = (i, j, k)
                corner_inside = np.asarray(
                    [
                        inside[tuple(np.asarray(cell) + offset)]
                        for offset in _CORNER_OFFSETS
                    ]
                )
                if np.all(corner_inside) or not np.any(corner_inside):
                    continue
                for component in _inside_components(corner_inside):
                    component_set = set(component)
                    component_anchors = {
                        crossing_index[_edge_key(cell, first, second)]
                        for first, second in _CUBE_EDGES
                        if corner_inside[first] != corner_inside[second]
                        and (
                            (first in component_set and corner_inside[first])
                            or (second in component_set and corner_inside[second])
                        )
                    }
                    if not component_anchors:
                        continue
                    vertex = len(vertex_anchors)
                    if vertex >= policy.maximum_vertices:
                        raise ValueError("Implicit surface exceeds maximum_vertices.")
                    vertex_anchors.append(tuple(sorted(component_anchors)))
                    cell_lower.append(np.asarray((axes[0][i], axes[1][j], axes[2][k])))
                    cell_upper.append(
                        np.asarray((axes[0][i + 1], axes[1][j + 1], axes[2][k + 1]))
                    )
                    for corner in component:
                        corner_vertex[(i, j, k, corner)] = vertex

    if not vertex_anchors:
        raise ValueError("Implicit surface discovery produced no dual vertices.")
    cell_lower_array = np.asarray(cell_lower)
    cell_upper_array = np.asarray(cell_upper)
    _, anchor_gradients = _field_and_gradient(
        geometry.kernel,
        geometry.state,
        jnp.asarray(anchors),
    )
    base_vertices, qef_regularization = _base_qef_vertices(
        anchors,
        np.asarray(anchor_gradients),
        tuple(vertex_anchors),
        cell_lower_array,
        cell_upper_array,
        policy.qef_regularization,
        policy.projection.root_tolerance,
    )

    faces: list[tuple[int, int, int]] = []
    for key in crossing_keys:
        selected: list[int] = []
        inside_point = np.asarray(inside_lattice[key], dtype=np.int32)
        for cell in _incident_cells(key):
            if any(cell[axis] < 0 or cell[axis] >= cell_shape[axis] for axis in range(3)):
                raise ValueError("Implicit surface intersects the outer grid boundary.")
            offset = inside_point - np.asarray(cell, dtype=np.int32)
            corner = _CORNER_INDEX.get(tuple(int(value) for value in offset))
            if corner is None or (*cell, corner) not in corner_vertex:
                raise ValueError("Implicit manifold incidence is incomplete.")
            selected.append(corner_vertex[(*cell, corner)])
        if len(set(selected)) != 4:
            raise ValueError(
                "Implicit manifold incidence produced a collapsed dual face."
            )
        diagonal_02 = np.linalg.norm(
            base_vertices[selected[0]] - base_vertices[selected[2]]
        )
        diagonal_13 = np.linalg.norm(
            base_vertices[selected[1]] - base_vertices[selected[3]]
        )
        if diagonal_02 <= diagonal_13:
            candidates = (
                (selected[0], selected[1], selected[2]),
                (selected[0], selected[2], selected[3]),
            )
        else:
            candidates = (
                (selected[0], selected[1], selected[3]),
                (selected[1], selected[2], selected[3]),
            )
        faces.extend(
            _oriented_triangle(
                geometry,
                base_vertices,
                candidate,
                policy.minimum_face_area,
            )
            for candidate in candidates
        )
    if not faces or len(faces) > policy.maximum_faces:
        raise ValueError("Implicit surface face count is empty or exceeds policy.")
    faces_array = np.asarray(faces, dtype=np.int32)
    topology = TriangleTopology(faces_array, num_vertices=base_vertices.shape[0])
    if not topology.watertight:
        raise ValueError("Implicit surface discovery did not produce a closed surface.")

    pairs = [
        pair
        for pair in combinations(range(faces_array.shape[0]), 2)
        if not set(faces_array[pair[0]]).intersection(faces_array[pair[1]])
    ]
    if len(pairs) > policy.maximum_intersection_pairs:
        raise ValueError("Implicit surface exceeds maximum_intersection_pairs.")
    pair_array = np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
    maximum_anchors = max(len(indices) for indices in vertex_anchors)
    padded = np.zeros((len(vertex_anchors), maximum_anchors), dtype=np.int32)
    mask = np.zeros_like(padded, dtype=bool)
    for vertex, indices in enumerate(vertex_anchors):
        padded[vertex, : len(indices)] = indices
        mask[vertex, : len(indices)] = True
    base_triangles = base_vertices[faces_array]
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
        vertex_anchor_indices=padded,
        vertex_anchor_mask=mask,
        qef_regularization=qef_regularization,
        cell_lower=cell_lower_array,
        cell_upper=cell_upper_array,
        base_vertices=base_vertices,
        faces=faces_array,
        base_face_normals=base_face_normals,
        intersection_pairs=pair_array,
        policy=policy,
        source_id=source_id,
        topology_id=topology.cell_complex_topology().topology_id,
    )
    base = plan.realize(geometry.state)
    if not bool(np.asarray(base.accepted)):
        raise ValueError("Implicit surface base realization failed runtime evidence.")
    return plan


__all__ = ["discover_implicit_surface"]
