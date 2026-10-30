# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Source-guided tetrahedral frame blocks and exact integer chart layouts."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from fractions import Fraction
from itertools import permutations, product
from typing import TypeAlias


_ExactAxis3: TypeAlias = tuple[Fraction, Fraction, Fraction]
_ExactFrame3: TypeAlias = tuple[_ExactAxis3, _ExactAxis3, _ExactAxis3]
_ExactKnotAxis: TypeAlias = tuple[Fraction, ...]
_ExactKnots3: TypeAlias = tuple[_ExactKnotAxis, _ExactKnotAxis, _ExactKnotAxis]


import jax.numpy as jnp
import numpy as np

from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    SimplicialConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import (
    coordinate_effective_basis,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    compose_expression,
    constant,
    derivative,
    evaluate_expression,
    Expression,
    expression_equal,
    linear_combinations,
    multiply,
    polynomial_bounds,
    scale,
    sum_polynomials,
)
from ..discretization._reference_cell import reference_cell_topology
from ..ein import contract
from ..optim import (
    Bounds,
    MinimizationResult,
    minimize,
    OptimizationStatus,
    OptimizationTermination,
    ProjectedLBFGS,
)
from ..typing import checked, ConvertibleToArray
from ._hex_cut_generation import BalancedCutHexConstruction, generate_balanced_cut_hexes
from ._hex_cut_preparation import _generic_axis_knots, prepare_balanced_cut_grid
from ._hex_frame_records import (
    _exact_frame_transition,
    FrameFieldObstructionRecord,
    FrameFieldRefusal,
    HexFrameFieldSolution,
    HexFrameIntegerLayout,
    PreparedFrameHexGrid,
    PreparedSourceHexFrame,
    SourceBlockConnection,
)
from ._hex_schedule import NativeHexGridSchedule
from ._plc_source import PiecewiseLinearComplex


def _frame_objective(
    angles: jnp.ndarray,
    arguments: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    /,
) -> jnp.ndarray:
    """Align each tetrahedral cross frame and smooth its exact cube-symmetry class."""
    directions, weights, left_cells, right_cells, cube_group = arguments
    roll, pitch, yaw = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx = jnp.cos(roll), jnp.sin(roll)
    cy, sy = jnp.cos(pitch), jnp.sin(pitch)
    cz, sz = jnp.cos(yaw), jnp.sin(yaw)
    rx = jnp.stack(
        (
            jnp.stack(
                (jnp.ones_like(cx), jnp.zeros_like(cx), jnp.zeros_like(cx)), axis=-1
            ),
            jnp.stack((jnp.zeros_like(cx), cx, -sx), axis=-1),
            jnp.stack((jnp.zeros_like(cx), sx, cx), axis=-1),
        ),
        axis=-2,
    )
    ry = jnp.stack(
        (
            jnp.stack((cy, jnp.zeros_like(cy), sy), axis=-1),
            jnp.stack(
                (jnp.zeros_like(cy), jnp.ones_like(cy), jnp.zeros_like(cy)), axis=-1
            ),
            jnp.stack((-sy, jnp.zeros_like(cy), cy), axis=-1),
        ),
        axis=-2,
    )
    rz = jnp.stack(
        (
            jnp.stack((cz, -sz, jnp.zeros_like(cz)), axis=-1),
            jnp.stack((sz, cz, jnp.zeros_like(cz)), axis=-1),
            jnp.stack(
                (jnp.zeros_like(cz), jnp.zeros_like(cz), jnp.ones_like(cz)), axis=-1
            ),
        ),
        axis=-2,
    )
    frames = rz @ ry @ rx
    projections = contract("nkd,ndj->nkj", directions, frames, backend="jax")
    alignment = jnp.sum(projections**4, axis=-1)
    local_energy = jnp.sum(weights * (1.0 - alignment))
    if left_cells.shape[0] == 0:
        return local_energy
    relative = jnp.swapaxes(frames[left_cells], -1, -2) @ frames[right_cells]
    scores = contract("eij,gji->eg", relative, cube_group, backend="jax")
    connection_energy = jnp.mean(1.0 - jnp.max(scores, axis=1) / 3.0)
    return local_energy + 0.25 * connection_energy


def _proper_cube_rotations() -> np.ndarray:
    rotations = []
    for order in permutations(range(3)):
        permutation_sign = (
            -1
            if sum(
                order[left] > order[right]
                for left in range(3)
                for right in range(left + 1, 3)
            )
            % 2
            else 1
        )
        for signs in product((-1, 1), repeat=3):
            if permutation_sign * int(np.prod(signs)) != 1:
                continue
            matrix = np.zeros((3, 3), dtype=np.float64)
            matrix[np.arange(3), np.asarray(order, dtype=np.int64)] = signs
            rotations.append(matrix)
    return np.asarray(rotations, dtype=np.float64)


def _frame_primitives(
    source: PiecewiseLinearComplex,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(source.vertices, dtype=np.float64)
    offsets = np.asarray(source.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(source.polygon_vertices, dtype=np.int64)
    directions: list[np.ndarray] = []
    weights: list[float] = []
    centers: list[np.ndarray] = []
    for row in range(len(offsets) - 1):
        loop = polygon_vertices[offsets[row] : offsets[row + 1]]
        polygon = points[loop]
        vector_area = np.zeros((3,), dtype=np.float64)
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            vector_area += np.cross(first, second)
        area_twice = float(np.linalg.norm(vector_area))
        if not np.isfinite(area_twice) or area_twice == 0.0:
            raise ValueError("Frame-field preparation requires nondegenerate PLC facets.")
        directions.append(vector_area / area_twice)
        weights.append(area_twice / 2.0)
        centers.append(np.mean(polygon, axis=0))
    for first, second in source.plc_edge_vertices:
        direction = points[int(second)] - points[int(first)]
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length == 0.0:
            raise ValueError(
                "Frame-field preparation requires nondegenerate PLC features."
            )
        directions.append(direction / length)
        weights.append(length)
        centers.append((points[int(first)] + points[int(second)]) / 2.0)
    if not directions:
        raise ValueError(
            "Frame-field preparation requires source facet or feature directions."
        )
    return (
        np.asarray(directions, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
    )


def _mesh_cell_components(
    mesh: CellMesh,
    root_cells: np.ndarray,
    root_cell_ids: np.ndarray,
    /,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return face-connected tetra blocks with exact orientation and boundary parity."""
    connectivity = mesh.connectivity
    if isinstance(connectivity, TetrahedralConnectivity):
        incidence = np.asarray(connectivity.cell_faces, dtype=np.int64)
        signs = np.asarray(connectivity.cell_face_signs, dtype=np.int64)
        face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
        edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
    elif isinstance(connectivity, SimplicialConnectivity) and connectivity.dimension == 3:
        if not np.array_equal(np.asarray(connectivity.cells, dtype=np.int64), root_cells):
            raise ValueError(
                "Simplicial connectivity must retain exact tetrahedron block row order."
            )
        incidence = np.asarray(connectivity.cell_entities[2], dtype=np.int64)
        face_vertices = np.asarray(connectivity.entities[2], dtype=np.int64)
        edge_vertices = np.asarray(connectivity.entities[1], dtype=np.int64)
        signs = np.zeros(incidence.shape, dtype=np.int64)
        seen = np.zeros(incidence.shape, dtype=np.bool_)
        cell_entity_ids = np.asarray(
            mesh.topology.entity_sets[3].entity_ids,
            dtype=np.int64,
        )
        if cell_entity_ids.shape != root_cell_ids.shape:
            raise ValueError(
                "Simplicial top-cell IDs must align with tetrahedron block order."
            )
        root_row_by_id = {
            int(identifier): row for row, identifier in enumerate(root_cell_ids.tolist())
        }
        face_cell_incidence = mesh.topology.incidences[2]
        relation = face_cell_incidence.relation
        valid_routes = np.asarray(relation.valid, dtype=np.bool_).reshape((-1,))
        source_routes = np.asarray(relation.source_indices, dtype=np.int64).reshape((-1,))
        target_routes = np.asarray(relation.target_indices, dtype=np.int64).reshape((-1,))
        route_signs = np.asarray(face_cell_incidence.signs, dtype=np.int64).reshape((-1,))
        if not (
            valid_routes.shape
            == source_routes.shape
            == target_routes.shape
            == route_signs.shape
        ):
            raise ValueError(
                "Simplicial oriented face-cell routes must have matching extents."
            )
        for route in range(valid_routes.size):
            if not valid_routes[route]:
                continue
            cell_id = int(cell_entity_ids[int(target_routes[route])])
            cell_row = root_row_by_id[cell_id]
            face = int(source_routes[route])
            local_faces = np.flatnonzero(incidence[cell_row] == face)
            if local_faces.shape != (1,):
                raise ValueError(
                    "Each simplicial face-cell incidence must match one local tetrahedron face."
                )
            local_face = int(local_faces[0])
            if seen[cell_row, local_face]:
                raise ValueError(
                    "Simplicial face-cell incidence repeats a local tetrahedron face."
                )
            signs[cell_row, local_face] = int(route_signs[route])
            seen[cell_row, local_face] = True
        if not np.all(seen):
            raise ValueError(
                "Simplicial connectivity must provide every oriented tetrahedron face incidence."
            )
    else:
        raise TypeError(
            "Frame-field components require tetrahedral or three-dimensional simplicial connectivity."
        )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if (
        incidence.shape != (root_cell_ids.size, 4)
        or signs.shape != incidence.shape
        or face_vertices.ndim != 2
        or face_vertices.shape[1] != 3
        or face_ids.shape != (face_vertices.shape[0],)
        or edge_vertices.ndim != 2
        or edge_vertices.shape[1] != 2
        or edge_ids.shape != (edge_vertices.shape[0],)
        or np.unique(root_cell_ids).size != root_cell_ids.size
    ):
        raise ValueError(
            "Frame-field tetrahedral incidences must preserve every source-cell identity."
        )
    face_owners: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for cell_row in range(incidence.shape[0]):
        for local_face in range(4):
            face = int(incidence[cell_row, local_face])
            face_owners[face].append(
                (cell_row, local_face, int(signs[cell_row, local_face]))
            )
    neighbors: list[list[int]] = [[] for _ in range(root_cell_ids.size)]
    connections: list[tuple[int, int, int, int]] = []
    boundary: list[tuple[int, int, int, int]] = []
    for face, owners in face_owners.items():
        if len(owners) == 1:
            cell, local_face, sign = owners[0]
            boundary.append((cell, local_face, face, sign))
        elif len(owners) == 2:
            (left, left_face, left_sign), (right, right_face, right_sign) = owners
            if left_sign != -right_sign:
                raise FrameFieldRefusal(
                    "boundary-parity",
                    "Shared root tetrahedron faces have inconsistent orientation parity.",
                    (int(face_ids[face]),),
                )
            if root_cell_ids[left] > root_cell_ids[right]:
                left, right = right, left
                left_face, right_face = right_face, left_face
            connections.append((left, left_face, right, right_face))
            neighbors[left].append(right)
            neighbors[right].append(left)
        else:
            raise FrameFieldRefusal(
                "nonmanifold-root-face",
                "A frame-field root face has more than two tetrahedron incidences.",
                (int(face_ids[face]),),
            )
    components: list[tuple[int, ...]] = []
    assigned = np.zeros((root_cell_ids.size,), dtype=np.bool_)
    for start in sorted(
        range(root_cell_ids.size), key=lambda row: int(root_cell_ids[row])
    ):
        if assigned[start]:
            continue
        assigned[start] = True
        queue = deque((start,))
        members: list[int] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in neighbors[current]:
                if not assigned[neighbor]:
                    assigned[neighbor] = True
                    queue.append(neighbor)
        components.append(tuple(sorted(members, key=lambda row: int(root_cell_ids[row]))))
    components.sort(key=lambda rows: int(root_cell_ids[rows[0]]))
    block_rows = np.empty((root_cell_ids.size,), dtype=np.int64)
    boundary_edges: list[dict[tuple[int, int], list[tuple[int, int]]]] = [
        defaultdict(list) for _ in components
    ]
    for block, rows in enumerate(components):
        block_rows[np.asarray(rows, dtype=np.int64)] = block
    for cell, _, face, sign in boundary:
        block = int(block_rows[cell])
        loop = face_vertices[face]
        if sign < 0:
            loop = loop[::-1]
        global_loop: tuple[int, int, int] = (
            int(vertex_ids[int(loop[0])]),
            int(vertex_ids[int(loop[1])]),
            int(vertex_ids[int(loop[2])]),
        )
        for local_edge in range(3):
            first = global_loop[local_edge]
            second = global_loop[(local_edge + 1) % 3]
            edge_key = (min(first, second), max(first, second))
            boundary_edges[block][edge_key].append((first, second))
    edge_id_by_vertices: dict[tuple[int, int], int] = {}
    for edge_row in range(edge_vertices.shape[0]):
        first = int(vertex_ids[int(edge_vertices[edge_row, 0])])
        second = int(vertex_ids[int(edge_vertices[edge_row, 1])])
        edge_id_by_vertices[(min(first, second), max(first, second))] = int(
            edge_ids[edge_row]
        )
    for edges in boundary_edges:
        for edge, directions in edges.items():
            if len(directions) != 2 or directions[0] != directions[1][::-1]:
                edge_id = edge_id_by_vertices.get(edge)
                raise FrameFieldRefusal(
                    "boundary-parity",
                    "A frame-block boundary edge does not have two oppositely oriented incidences.",
                    edge if edge_id is None else (edge_id,),
                )
    return (
        tuple(components),
        block_rows,
        np.asarray(connections, dtype=np.int64).reshape((-1, 4)),
        np.asarray(boundary, dtype=np.int64).reshape((-1, 4)),
        np.ones((len(components),), dtype=np.int8),
    )


def _rotation_matrix(angles: np.ndarray, /) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in angles)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    ry = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rz = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def _solve_frame_field(
    arguments: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    root_cell_ids: np.ndarray,
    /,
) -> MinimizationResult:
    affected = tuple(int(value) for value in root_cell_ids.tolist())
    parameters = jnp.zeros((root_cell_ids.shape[0], 3), dtype=jnp.float64)
    initial_objective = float(np.asarray(_frame_objective(parameters, arguments)))
    if not math.isfinite(initial_objective):
        raise FrameFieldRefusal(
            "field-nonfinite",
            "The initial per-cell frame objective is nonfinite.",
            affected,
        )
    bounds = Bounds(
        jnp.full(parameters.shape, -np.pi, dtype=jnp.float64),
        jnp.full(parameters.shape, np.pi, dtype=jnp.float64),
    )
    result = minimize(
        _frame_objective,
        parameters,
        method=ProjectedLBFGS(),
        args=arguments,
        bounds=bounds,
        termination=OptimizationTermination(),
    )
    final_parameters = np.asarray(result.parameters, dtype=np.float64)
    final_objective = float(np.asarray(result.objective))
    if not np.isfinite(final_parameters).all() or not math.isfinite(final_objective):
        raise FrameFieldRefusal(
            "field-nonfinite",
            "The per-cell frame optimizer produced a nonfinite state or objective.",
            affected,
        )
    if bool(np.asarray(result.successful)):
        return result
    status = int(np.asarray(result.status))
    status_name = OptimizationStatus(status).name
    category = "field-nonfinite" if "NONFINITE" in status_name else "field-stationarity"
    raise FrameFieldRefusal(
        category,
        "The per-cell source-guided frame field stopped without convergence "
        f"(optimizer status {status_name}).",
        affected,
    )


@checked
def prepare_frame_hex_grid(
    source: PiecewiseLinearComplex,
    source_mesh: CellMesh,
    schedule: NativeHexGridSchedule,
    target_size: float,
    /,
    *,
    source_id: str,
    source_revision: str,
) -> PreparedFrameHexGrid:
    """Solve per-tetrahedron cross frames and extract exact block chart transitions."""
    if schedule.route != "frame_grid":
        raise ValueError("Frame-field preparation requires a frame_grid schedule.")
    if (
        source.geometry is not None
        or source.boundary != "conforming"
        or source_mesh.topological_dimension != 3
        or source_mesh.ambient_dimension != 3
        or not source_mesh.blocks
        or any(
            reference_cell_topology(block.cell_kind).family != "tetrahedron"
            for block in source_mesh.blocks
        )
    ):
        raise ValueError(
            "Frame-field PLC layouts require a conforming piecewise-linear tetrahedral-family fill."
        )
    target = float(target_size)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("Frame-field target_size must be positive and finite.")
    root_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in source_mesh.blocks],
        axis=0,
    )
    root_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks],
    )
    components, component_rows, face_connections, boundary_faces, component_parity = (
        _mesh_cell_components(source_mesh, root_cells, root_cell_ids)
    )
    if boundary_faces.shape[0] == 0:
        raise FrameFieldRefusal(
            "missing-boundary-parity",
            "Frame-field roots require explicit oriented source boundary faces.",
            tuple(int(value) for value in root_cell_ids.tolist()),
        )
    points = np.asarray(source_mesh.coordinates, dtype=np.float64)
    cell_centers = np.mean(points[root_cells], axis=1)
    component_origins = np.asarray(
        [
            np.mean(cell_centers[np.asarray(rows, dtype=np.int64)], axis=0)
            for rows in components
        ],
        dtype=np.float64,
    )
    cell_origins = component_origins[component_rows]
    directions, primitive_weights, primitive_centers = _frame_primitives(source)
    distances = np.linalg.norm(
        cell_centers[:, None, :] - primitive_centers[None, :, :],
        axis=-1,
    )
    cell_scales = np.maximum(
        np.max(
            np.linalg.norm(points[root_cells] - cell_centers[:, None, :], axis=-1), axis=1
        ),
        target,
    )
    relative_distance = distances - np.min(distances, axis=1, keepdims=True)
    proximities = np.exp(-relative_distance / cell_scales[:, None])
    local_weights = primitive_weights[None, :] * proximities
    weight_sums = np.sum(local_weights, axis=1, keepdims=True)
    if (
        not np.all(np.isfinite(weight_sums))
        or np.any(weight_sums <= 0.0)
        or not np.all(np.isfinite(cell_scales))
        or np.any(cell_scales <= 0.0)
    ):
        raise FrameFieldRefusal(
            "unresolved-cell-directions",
            "Every root tetrahedron must have finite source-guided facet and feature directions.",
            tuple(int(value) for value in root_cell_ids.tolist()),
        )
    local_weights /= weight_sums
    left_cells = face_connections[:, 0]
    right_cells = face_connections[:, 2]
    cube_group = _proper_cube_rotations()
    arguments = (
        jnp.asarray(
            np.broadcast_to(directions, (root_cells.shape[0], *directions.shape)),
            dtype=jnp.float64,
        ),
        jnp.asarray(local_weights, dtype=jnp.float64),
        jnp.asarray(left_cells, dtype=jnp.int32),
        jnp.asarray(right_cells, dtype=jnp.int32),
        jnp.asarray(cube_group, dtype=jnp.float64),
    )
    result = _solve_frame_field(arguments, root_cell_ids)
    cell_frames = np.empty((root_cells.shape[0], 3, 3), dtype=np.float64)
    for row, angles in enumerate(np.asarray(result.parameters, dtype=np.float64)):
        rotation = _rotation_matrix(angles)
        left, _, right = np.linalg.svd(rotation)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        cell_frames[row] = rotation.T
    integer_block_rows = np.arange(root_cells.shape[0], dtype=np.int64)
    spacing = Fraction.from_float(target)
    exact_axes: tuple[_ExactFrame3, ...] = tuple(
        (
            (
                Fraction.from_float(float(frame[0, 0])),
                Fraction.from_float(float(frame[0, 1])),
                Fraction.from_float(float(frame[0, 2])),
            ),
            (
                Fraction.from_float(float(frame[1, 0])),
                Fraction.from_float(float(frame[1, 1])),
                Fraction.from_float(float(frame[1, 2])),
            ),
            (
                Fraction.from_float(float(frame[2, 0])),
                Fraction.from_float(float(frame[2, 1])),
                Fraction.from_float(float(frame[2, 2])),
            ),
        )
        for frame in cell_frames
    )
    exact_origins: tuple[_ExactAxis3, ...] = tuple(
        (
            Fraction.from_float(float(origin[0])),
            Fraction.from_float(float(origin[1])),
            Fraction.from_float(float(origin[2])),
        )
        for origin in cell_origins
    )
    component_vertex_rows = tuple(
        np.unique(root_cells[np.asarray(rows, dtype=np.int64)]) for rows in components
    )
    block_knots: list[_ExactKnots3] = []
    for row in range(root_cells.shape[0]):
        vertices = component_vertex_rows[int(component_rows[row])]
        exact_points = tuple(
            tuple(Fraction.from_float(float(value)) for value in points[int(vertex)])
            for vertex in vertices
        )
        chart_values = tuple(
            tuple(
                sum(
                    (
                        exact_axes[row][axis][coordinate]
                        * (point[coordinate] - exact_origins[row][coordinate])
                        for coordinate in range(3)
                    ),
                    Fraction(0),
                )
                for point in exact_points
            )
            for axis in range(3)
        )
        block_knots.append(
            (
                _generic_axis_knots(chart_values[0], spacing, 1),
                _generic_axis_knots(chart_values[1], spacing, 2),
                _generic_axis_knots(chart_values[2], spacing, 3),
            )
        )
    integer_coordinates = np.empty((root_cells.shape[0], 3), dtype=np.int64)
    for row, cell in enumerate(root_cells):
        center = tuple(
            sum(
                (
                    Fraction.from_float(float(points[int(vertex), axis]))
                    for vertex in cell
                ),
                Fraction(0),
            )
            / 4
            for axis in range(3)
        )
        local = tuple(
            sum(
                (
                    exact_axes[row][axis][coordinate]
                    * (center[coordinate] - exact_origins[row][coordinate])
                    for coordinate in range(3)
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        integer_coordinates[row] = tuple(
            int((value - block_knots[row][axis][0]) // spacing)
            for axis, value in enumerate(local)
        )
    cell_parity = component_parity[component_rows]
    field = HexFrameFieldSolution(
        cell_frames,
        cell_origins,
        root_cell_ids,
        integer_block_rows,
        integer_coordinates,
        float(np.asarray(result.objective)),
        result,
        source_id,
        source_revision,
        source.complex_id,
        source_mesh.mesh_id,
        source_mesh.topology_id,
        schedule.schedule_id,
        tuple(zip(source.vertex_ids, source.vertex_occurrence_paths, strict=True)),
        tuple(zip(source.facet_ids, source.facet_occurrence_paths, strict=True)),
        tuple(zip(source.plc_edge_ids, source.plc_edge_occurrence_paths, strict=True)),
        tuple(zip(source.region_ids, source.region_occurrence_paths, strict=True)),
        source.facet_regions,
    )
    transitions = np.empty((face_connections.shape[0], 3, 3), dtype=np.int64)
    offsets = np.empty((face_connections.shape[0], 3), dtype=np.int64)
    try:
        for edge_row, (left_cell, _, right_cell, _) in enumerate(
            face_connections.tolist()
        ):
            left_row, right_row = int(left_cell), int(right_cell)
            transition, offset = _exact_frame_transition(
                exact_axes[left_row],
                exact_axes[right_row],
                exact_origins[left_row],
                exact_origins[right_row],
                block_knots[left_row],
                block_knots[right_row],
                (int(root_cell_ids[left_row]), int(root_cell_ids[right_row])),
            )
            transitions[edge_row] = transition
            offsets[edge_row] = offset
        layout = HexFrameIntegerLayout(
            root_cell_ids,
            integer_block_rows,
            integer_coordinates,
            cell_frames,
            cell_origins,
            tuple(block_knots),
            face_connections,
            transitions,
            offsets,
            cell_parity,
            source.complex_id,
            source_mesh.mesh_id,
            source_mesh.topology_id,
            schedule.schedule_id,
            field.frame_id,
        )
    except FrameFieldRefusal as obstruction:
        record = FrameFieldObstructionRecord(
            obstruction.kind,
            obstruction.entity_ids,
            str(obstruction),
            field.frame_id,
        )
        return PreparedFrameHexGrid(field, None, None, record)
    grid = prepare_balanced_cut_grid(
        source,
        source_mesh,
        schedule,
        target,
        source_id=source_id,
        source_revision=source_revision,
        frame_axes=cell_frames[0],
        frame_origin=cell_origins[0],
        frame_id=field.frame_id,
        frame_layout=layout,
    )
    return PreparedFrameHexGrid(field, layout, grid, None)


@checked
def generate_frame_hexes(
    source: PiecewiseLinearComplex,
    source_mesh: CellMesh,
    cell_regions: ConvertibleToArray,
    prepared: PreparedFrameHexGrid,
    schedule: NativeHexGridSchedule,
    /,
    *,
    source_geometry: CellGeometrySpec | None = None,
) -> BalancedCutHexConstruction:
    if not isinstance(prepared, PreparedFrameHexGrid):
        raise TypeError("prepared must be PreparedFrameHexGrid.")
    if prepared.obstruction is not None:
        raise FrameFieldRefusal(
            prepared.obstruction.kind,
            prepared.obstruction.message,
            prepared.obstruction.entity_ids,
        )
    if prepared.layout is None or prepared.grid is None:
        raise FrameFieldRefusal(
            "missing-field-integer-layout",
            "Native frame generation requires an exact block-integer layout and its bound cut plan.",
            tuple(int(value) for value in prepared.frame.cell_global_ids.tolist()),
        )
    if (
        prepared.frame.source_complex_id != source.complex_id
        or prepared.frame.source_mesh_id != source_mesh.mesh_id
        or prepared.frame.source_topology_id != source_mesh.topology_id
        or prepared.frame.schedule_id != schedule.schedule_id
        or prepared.layout.frame_id != prepared.frame.frame_id
        or prepared.layout.source_complex_id != source.complex_id
        or prepared.layout.source_mesh_id != source_mesh.mesh_id
        or prepared.layout.source_topology_id != source_mesh.topology_id
        or prepared.layout.schedule_id != schedule.schedule_id
        or prepared.grid.source_complex_id != source.complex_id
        or prepared.grid.source_mesh_id != source_mesh.mesh_id
        or prepared.grid.source_topology_id != source_mesh.topology_id
        or prepared.grid.schedule_id != schedule.schedule_id
        or schedule.route != "frame_grid"
    ):
        raise ValueError(
            "Prepared frame layout must bind the exact PLC, root mesh, and schedule."
        )
    if prepared.grid.frame_layout_id != prepared.layout.layout_id:
        raise ValueError(
            "Frame cut plan must retain the exact solved block integer layout."
        )
    return generate_balanced_cut_hexes(
        source,
        source_mesh,
        cell_regions,
        prepared.grid,
        schedule,
        source_geometry=source_geometry,
    )


def _source_face_permutation(
    left: tuple[int, ...], right: tuple[int, ...], /
) -> tuple[int, ...]:
    if len(left) != 4 or len(right) != 4 or set(left) != set(right):
        raise ValueError("Mapped source roots must share exact four-vertex tensor faces.")
    permutation = tuple(right.index(vertex) for vertex in left)
    if any(
        (permutation[(index + 1) % 4] - permutation[index]) % 4 != 3 for index in range(4)
    ):
        raise ValueError(
            "Mapped source face cycles must have opposite boundary orientation."
        )
    return permutation


def _hex_face_chart_arguments(
    corners: tuple[tuple[Fraction, ...], ...], /
) -> tuple[Expression, ...]:
    u, v = axes(2)
    one = constant(Fraction(1), 2)
    one_minus_u = add(one, scale(u, -1))
    one_minus_v = add(one, scale(v, -1))
    weights = (
        multiply(one_minus_u, one_minus_v),
        multiply(u, one_minus_v),
        multiply(u, v),
        multiply(one_minus_u, v),
    )
    return tuple(
        sum_polynomials(
            tuple(scale(weights[index], corners[index][axis]) for index in range(4)),
            2,
        )
        for axis in range(3)
    )


@checked
def prepare_source_hex_frame(
    reference_mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> PreparedSourceHexFrame:
    """Prepare exact source-map Jacobian bounds and root-face connectivity."""
    if not isinstance(reference_mesh, CellMesh) or not isinstance(
        geometry, CellGeometrySpec
    ):
        raise TypeError(
            "Source frame preparation requires a CellMesh and CellGeometrySpec."
        )
    if (
        reference_mesh.topological_dimension != 3
        or reference_mesh.ambient_dimension != 3
        or not reference_mesh.blocks
        or any(block.cell_kind != "hexahedron" for block in reference_mesh.blocks)
    ):
        raise ValueError(
            "Mapped source frames require a three-dimensional all-hexahedral root mesh."
        )
    validity = certify_cell_geometry_validity(geometry, mesh=reference_mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise ValueError(
            "Mapped source root coordinate maps must have certified positive Jacobians."
        )
    elements, routes, _ = geometry.resolve(reference_mesh)
    source_values = geometry.source_coordinates()
    root_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in reference_mesh.blocks]
    )
    root_block_indices = np.concatenate(
        [
            np.full((block.cell_count,), index, dtype=np.int64)
            for index, block in enumerate(reference_mesh.blocks)
        ]
    )
    root_block_rows = np.concatenate(
        [np.arange(block.cell_count, dtype=np.int64) for block in reference_mesh.blocks]
    )
    root_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in reference_mesh.blocks]
    )
    root_count = root_cells.shape[0]
    corner_points = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    center_point = (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    corner_jacobians = np.empty((root_count, 8, 3, 3), dtype=np.float64)
    center_frames = np.empty((root_count, 3, 3), dtype=np.float64)
    axis_bounds = np.empty((root_count, 3), dtype=np.float64)
    host_routes = tuple(np.asarray(route, dtype=np.int64) for route in routes)
    expressions_by_root: list[tuple[Expression, ...]] = []
    root = 0
    for block_index, (block, element, block_routes) in enumerate(
        zip(reference_mesh.blocks, elements, host_routes, strict=True)
    ):
        if (
            not isinstance(element, ScalarCoordinateElement)
            or element.cell_kind != "hexahedron"
        ):
            raise ValueError(
                "Mapped source roots require scalar hexahedral coordinate elements."
            )
        if block_routes.shape != (block.cell_count, element.local_dof_count):
            raise ValueError(
                "Mapped source root DOF routes must align with root hexahedra."
            )
        for row in range(block.cell_count):
            route = block_routes[row]
            local = tuple(
                source_values[int(route[dof])] for dof in range(element.local_dof_count)
            )
            basis = coordinate_effective_basis(element, cell_row=row)
            components = tuple(
                tuple(local[dof][axis] for dof in range(len(local))) for axis in range(3)
            )
            expressions = linear_combinations(basis, components)
            expressions_by_root.append(expressions)
            derivatives = tuple(
                tuple(derivative(expression, axis) for axis in range(3))
                for expression in expressions
            )
            for axis in range(3):
                upper = []
                for physical_axis in range(3):
                    lower, upper_value = polynomial_bounds(
                        derivatives[physical_axis][axis], "box", 3
                    )
                    upper.append(max(abs(lower), abs(upper_value)))
                axis_bounds[root, axis] = np.nextafter(
                    math.sqrt(sum(value * value for value in upper)), np.inf
                )
            for corner_index, point in enumerate(corner_points):
                jacobian = np.asarray(
                    [
                        [
                            float(evaluate_expression(derivatives[physical][axis], point))
                            for axis in range(3)
                        ]
                        for physical in range(3)
                    ],
                    dtype=np.float64,
                )
                corner_jacobians[root, corner_index] = jacobian
            center_jacobian = np.asarray(
                [
                    [
                        float(
                            evaluate_expression(derivatives[physical][axis], center_point)
                        )
                        for axis in range(3)
                    ]
                    for physical in range(3)
                ],
                dtype=np.float64,
            )
            left, singular, right = np.linalg.svd(center_jacobian, full_matrices=False)
            if np.any(singular <= 0.0):
                raise ValueError("Mapped source center Jacobians must have full rank.")
            frame = left @ right
            if np.linalg.det(frame) < 0.0:
                left[:, -1] *= -1.0
                frame = left @ right
            center_frames[root] = frame
            if not np.all(np.isfinite(axis_bounds[root])) or np.any(
                axis_bounds[root] <= 0.0
            ):
                raise ValueError(
                    "Mapped source physical Jacobian bounds must be positive and finite."
                )
            root += 1
    topology = reference_cell_topology("hexahedron")
    faces = topology.entities[2]
    vertex_ids = np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64)
    occurrences: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
    for root_row, cell in enumerate(root_cells):
        for face_index, local in enumerate(faces):
            cycle = tuple(int(vertex_ids[int(cell[index])]) for index in local)
            occurrences.setdefault(tuple(sorted(cycle)), []).append(
                (root_row, face_index, cycle)
            )
    connections = []
    permutations = []
    boundary = []
    for rows in occurrences.values():
        if len(rows) == 1:
            boundary.append(rows[0][:2])
        elif len(rows) == 2:
            left_row, left_face, left_cycle = rows[0]
            right_row, right_face, right_cycle = rows[1]
            connections.append((left_row, left_face, right_row, right_face))
            permutations.append(_source_face_permutation(left_cycle, right_cycle))
        else:
            raise ValueError(
                "Mapped root faces must have one boundary or two conforming incidences."
            )
    local_vertices = tuple(
        tuple(
            tuple(Fraction(int(value)) for value in topology.vertices[int(index)])
            for index in face
        )
        for face in faces
    )
    for connection_row, (left, left_face, right, right_face) in enumerate(connections):
        left_arguments = _hex_face_chart_arguments(local_vertices[left_face])
        right_corners = tuple(
            local_vertices[right_face][int(index)]
            for index in permutations[connection_row]
        )
        right_arguments = _hex_face_chart_arguments(right_corners)
        left_map = tuple(
            compose_expression(value, left_arguments)
            for value in expressions_by_root[left]
        )
        right_map = tuple(
            compose_expression(value, right_arguments)
            for value in expressions_by_root[right]
        )
        if any(
            not expression_equal(left_value, right_value)
            for left_value, right_value in zip(left_map, right_map, strict=True)
        ):
            raise ValueError(
                "Adjacent mapped root charts must agree exactly across shared faces."
            )

    return PreparedSourceHexFrame(
        reference_mesh,
        geometry,
        root_cells,
        root_block_indices,
        root_block_rows,
        root_cell_ids,
        corner_jacobians,
        center_frames,
        axis_bounds,
        np.asarray(connections, dtype=np.int64).reshape((-1, 4)),
        np.asarray(permutations, dtype=np.int64).reshape((-1, 4)),
        np.asarray(boundary, dtype=np.int64).reshape((-1, 2)),
    )


@checked
def prepare_source_block_connection(
    prepared: PreparedSourceHexFrame, /
) -> SourceBlockConnection:
    """Derive exact tensor-axis transitions from shared root face vertices."""
    if not isinstance(prepared, PreparedSourceHexFrame):
        raise TypeError("prepared must be PreparedSourceHexFrame.")
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    faces = reference_cell_topology("hexahedron").entities[2]
    transports = np.zeros((prepared.face_connections.shape[0], 3, 3), dtype=np.int64)
    offsets = np.zeros((prepared.face_connections.shape[0], 3), dtype=np.int64)
    for row, (left, left_face, right, right_face) in enumerate(
        prepared.face_connections.tolist()
    ):
        left_points = corners[np.asarray(faces[left_face], dtype=np.int64)]
        right_points = corners[np.asarray(faces[right_face], dtype=np.int64)][
            prepared.face_permutations[row]
        ]
        left_normal = int(
            np.flatnonzero(np.all(left_points == left_points[:1], axis=0))[0]
        )
        right_normal = int(
            np.flatnonzero(np.all(right_points == right_points[:1], axis=0))[0]
        )
        action = np.zeros((3, 3), dtype=np.int64)
        for axis in range(3):
            if axis == left_normal:
                continue
            for corner in range(1, 4):
                direction = left_points[corner] - left_points[0]
                if np.count_nonzero(direction) == 1 and direction[axis]:
                    action[:, axis] = (
                        right_points[corner] - right_points[0]
                    ) * direction[axis]
                    break
            else:
                raise ValueError(
                    "Mapped root face transport lacks a tangent tensor axis."
                )
        action[right_normal, left_normal] = (
            1 if left_points[0, left_normal] != right_points[0, right_normal] else -1
        )
        offset = right_points[0] - action @ left_points[0]
        if not np.array_equal(left_points @ action.T + offset, right_points):
            raise ValueError(
                "Mapped root face transport does not preserve its exact corner action."
            )
        transports[row] = action
        offsets[row] = offset
    return SourceBlockConnection(
        transports,
        offsets,
        prepared.face_connections,
        prepared.frame_id,
    )


__all__ = [
    "generate_frame_hexes",
    "prepare_frame_hex_grid",
    "prepare_source_block_connection",
    "prepare_source_hex_frame",
]
