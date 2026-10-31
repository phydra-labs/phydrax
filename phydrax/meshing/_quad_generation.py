# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Feature-constrained quad layouts and honest triangle-dual extraction."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from typing import cast, final, Literal

import equinox as eqx
import numpy as np
from jax import Array
from scipy.optimize import Bounds as MilpBounds, LinearConstraint, milp
from scipy.sparse import coo_array
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import orient2d, orient3d, PredicateMode, PredicateSign
from .._meshcore import triangle_dual_quads
from .._strict import StrictModule
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_complex import PolygonalConnectivity
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._coordinate_enclosure import (
    compose_expression,
    CoordinateSourceBank,
    ExactScalar,
    Expression,
    expression_sequence_equal,
    ExpressionInput,
    linear_combinations,
    Polynomial,
    RationalPolynomial,
)
from ..discretization._exact_algebraic_geometry import RadicalElement
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked, ConvertibleToArray
from ._dual_extraction import (
    _affine_geometry_matches,
    _RestrictedCoordinateMap,
    DualExtraction,
    fresh_vertex_global_ids,
    QuadLayoutObstruction,
    SurfaceCrossField,
    SurfaceQuadLayout,
)


type QuadReferencePoint = tuple[Fraction, Fraction]
type QuadChart = tuple[Polynomial, Polynomial]


def _exact_scalar(value: ExactScalar | int | float | np.generic, /) -> ExactScalar:
    if isinstance(value, (Fraction, RadicalElement)):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        return Fraction.from_float(float(value))
    raise TypeError("Exact quad charts require rational or radical coefficients.")


def _exact_expression(value: ExpressionInput, /) -> Expression:
    if isinstance(value, RationalPolynomial):
        return value
    polynomial: Polynomial = {}
    for powers, coefficient in value.items():
        polynomial[tuple(powers)] = _exact_scalar(coefficient)
    return polynomial


def _exact_source_coordinate_bank(
    rows: Sequence[Sequence[ExactScalar | int | float | np.generic]], /
) -> CoordinateSourceBank:
    return tuple(tuple(_exact_scalar(value) for value in row) for row in rows)


def _reference_points_2d(cell_kind: str, /) -> tuple[QuadReferencePoint, ...]:
    topology = reference_cell_topology(cell_kind)
    if topology.dimension != 2:
        raise ValueError("Quad source charts require two-dimensional reference cells.")
    points: list[QuadReferencePoint] = []
    for point in topology.vertices:
        if len(point) != 2:
            raise ValueError("Quad reference vertices must have two coordinates.")
        points.append((Fraction(float(point[0])), Fraction(float(point[1]))))
    return tuple(points)


def _edge_key(first: int, second: int, /) -> tuple[int, int]:
    return (min(first, second), max(first, second))


# Layout refusals that only say no complete triangle pairing exists; the
# subdividing extraction answers them with a partial pairing instead.
_PAIRING_OBSTRUCTIONS = frozenset(
    (
        "patch_parity",
        "boundary_parity",
        "patch_boundary",
        "field_alignment",
        "source_map",
        "patch_matching",
        "singularity_valence",
    )
)


@final
class _TriangleData(StrictModule):
    cells: np.ndarray
    edges: np.ndarray
    edge_owners: dict[tuple[int, int], tuple[int, ...]] = eqx.field(static=True)


def _triangle_data(mesh: CellMesh, /) -> _TriangleData:
    if mesh.topological_dimension != 2 or mesh.ambient_dimension not in (2, 3):
        raise ValueError("Quad extraction requires a two-dimensional mesh in 2D or 3D.")
    if len(mesh.blocks) != 1:
        raise ValueError("Quad extraction requires one homogeneous triangle block.")
    block = mesh.blocks[0]
    topology = reference_cell_topology(block.cell_kind)
    if topology.family != "triangle" or topology.dimension != 2:
        raise ValueError("Quad extraction requires triangle reference cells.")
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolygonalConnectivity):
        edges = np.asarray(connectivity.edges, dtype=np.int64)
        routed_edges = np.asarray(connectivity.cell_edges, dtype=np.int64)
        valid_edges = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        boundary_edges = np.asarray(connectivity.boundary_edges, dtype=np.bool_)
        if (
            routed_edges.ndim != 2
            or routed_edges.shape[0] != block.cell_count
            or routed_edges.shape[1] < 3
            or valid_edges.shape != routed_edges.shape
            or not np.all(valid_edges[:, :3])
            or np.any(valid_edges[:, 3:])
        ):
            raise ValueError(
                "Triangle source connectivity must contain exactly three edge routes per cell."
            )
        cell_edges = routed_edges[:, :3]
    elif isinstance(connectivity, SimplicialConnectivity) and connectivity.dimension == 2:
        edges = np.asarray(connectivity.entities[1], dtype=np.int64)
        cell_edges = np.asarray(connectivity.cell_entities[1], dtype=np.int64)
        boundary_edges = np.asarray(connectivity.boundary_masks[1], dtype=np.bool_)
    else:
        raise TypeError("Quad extraction requires polygonal or two-simplex connectivity.")
    cells = np.asarray(block.vertices, dtype=np.int64).copy()
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or cell_edges.shape != (cells.shape[0], 3)
        or boundary_edges.shape != (edges.shape[0],)
        or (
            cell_edges.size
            and (np.any(cell_edges < 0) or np.any(cell_edges >= edges.shape[0]))
        )
    ):
        raise ValueError(
            "Triangle source edge, cell-route, and boundary banks must align."
        )
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    directed: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for row, triangle in enumerate(cells.tolist()):
        expected_edges = {
            _edge_key(triangle[index], triangle[(index + 1) % 3]) for index in range(3)
        }
        routed_edges = {
            _edge_key(int(edges[edge_row, 0]), int(edges[edge_row, 1]))
            for edge_row in cell_edges[row].tolist()
        }
        if expected_edges != routed_edges:
            raise QuadLayoutObstruction(
                "source_identity",
                "Triangle cell-edge routes do not preserve the source vertex edges.",
                source_cell_rows=(row,),
            )
        for edge_row in cell_edges[row].tolist():
            key = _edge_key(int(edges[edge_row, 0]), int(edges[edge_row, 1]))
            owners[key].append(row)
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = _edge_key(a, b)
            directed[key].append((row, 1 if (a, b) == key else -1))
    owner_counts = np.bincount(cell_edges.reshape(-1), minlength=edges.shape[0])
    if (
        np.any(owner_counts > 2)
        or not np.array_equal(boundary_edges, owner_counts == 1)
        or any(len(rows) not in (1, 2) for rows in owners.values())
    ):
        raise ValueError("Quad extraction requires a manifold triangle surface.")
    orientation = np.zeros((cells.shape[0],), dtype=np.int8)
    for start in range(cells.shape[0]):
        if orientation[start] != 0:
            continue
        orientation[start] = 1
        pending = [start]
        while pending:
            face = pending.pop()
            triangle = cells[face].tolist()
            for a, b in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                adjacent = directed[(min(a, b), max(a, b))]
                if len(adjacent) != 2:
                    continue
                own_direction = next(sign for row, sign in adjacent if row == face)
                neighbour, neighbour_direction = next(
                    (row, sign) for row, sign in adjacent if row != face
                )
                required = -int(orientation[face]) * own_direction * neighbour_direction
                if orientation[neighbour] == 0:
                    orientation[neighbour] = required
                    pending.append(neighbour)
                elif orientation[neighbour] != required:
                    raise QuadLayoutObstruction(
                        "nonorientable_surface",
                        "The source triangle surface has no globally consistent orientation.",
                        source_cell_rows=(face, neighbour),
                    )
    reversed_rows = np.flatnonzero(orientation < 0)
    cells[reversed_rows, 1], cells[reversed_rows, 2] = (
        cells[reversed_rows, 2].copy(),
        cells[reversed_rows, 1].copy(),
    )
    vertex_faces: list[list[int]] = [[] for _ in range(mesh.coordinates.shape[0])]
    vertex_links: list[dict[int, set[int]]] = [
        defaultdict(set) for _ in range(mesh.coordinates.shape[0])
    ]
    boundary_vertices: set[int] = set()
    for face, triangle in enumerate(cells.tolist()):
        for vertex in triangle:
            vertex_faces[vertex].append(face)
    for edge, adjacent in owners.items():
        if len(adjacent) == 1:
            boundary_vertices.update(edge)
            continue
        first, second = adjacent
        for vertex in edge:
            vertex_links[vertex][first].add(second)
            vertex_links[vertex][second].add(first)
    for vertex, star in enumerate(vertex_faces):
        if not star:
            raise QuadLayoutObstruction(
                "vertex_link",
                "The source triangle surface contains an unsupported isolated vertex.",
                source_vertex_rows=(vertex,),
            )
        reachable = {star[0]}
        pending = [star[0]]
        while pending:
            face = pending.pop()
            for neighbour in vertex_links[vertex].get(face, ()):
                if neighbour not in reachable:
                    reachable.add(neighbour)
                    pending.append(neighbour)
        degrees = [len(vertex_links[vertex].get(face, ())) for face in star]
        if (
            reachable != set(star)
            or (
                vertex in boundary_vertices
                and not (
                    (len(star) == 1 and degrees == [0])
                    or (
                        len(star) > 1
                        and degrees.count(1) == 2
                        and all(degree in (1, 2) for degree in degrees)
                    )
                )
            )
            or (
                vertex not in boundary_vertices and any(degree != 2 for degree in degrees)
            )
        ):
            raise QuadLayoutObstruction(
                "vertex_link",
                "The source triangle vertex link is not one manifold cycle or boundary path.",
                source_cell_rows=star,
                source_vertex_rows=(vertex,),
            )
    return _TriangleData(cells, edges, {key: tuple(rows) for key, rows in owners.items()})


def _surface_patch_ids(face_count: int, adjacent: np.ndarray, /) -> np.ndarray:
    neighbours: list[list[int]] = [[] for _ in range(face_count)]
    for left, right in adjacent.tolist():
        neighbours[left].append(right)
        neighbours[right].append(left)
    labels = np.full((face_count,), -1, dtype=np.int32)
    patch = 0
    for start in range(face_count):
        if labels[start] >= 0:
            continue
        labels[start] = patch
        pending = [start]
        while pending:
            row = pending.pop()
            for neighbour in neighbours[row]:
                if labels[neighbour] < 0:
                    labels[neighbour] = patch
                    pending.append(neighbour)
        patch += 1
    return labels


def _triangle_frames(mesh: CellMesh, cells: np.ndarray, /) -> np.ndarray:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    local = padded[cells]
    first = local[:, 1] - local[:, 0]
    second = local[:, 2] - local[:, 0]
    normals = np.cross(first, second)
    lengths = np.linalg.norm(first, axis=1)
    areas = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0.0) or np.any(areas == 0.0):
        raise ValueError("Cross-field frames require nondegenerate source triangles.")
    tangent = first / lengths[:, None]
    normal = normals / areas[:, None]
    frames = np.stack((tangent, np.cross(normal, tangent)), axis=1)
    if not np.all(np.isfinite(frames)):
        raise ValueError("Cross-field frames must be finite.")
    return frames


def _edge_angle(frame: np.ndarray, direction: np.ndarray, /) -> float:
    local = np.asarray(frame) @ np.asarray(direction, dtype=np.float64)
    if not np.all(np.isfinite(local)) or float(np.linalg.norm(local)) == 0.0:
        raise ValueError("A cross-field edge direction must be finite and nonzero.")
    return float(np.arctan2(local[1], local[0]))


def _dihedral_feature_rows(
    mesh: CellMesh,
    cells: np.ndarray,
    edges: np.ndarray,
    owners: dict[tuple[int, int], tuple[int, ...]],
    threshold: float | None,
    /,
) -> set[int]:
    edge_rows = {
        _edge_key(int(edge[0]), int(edge[1])): row
        for row, edge in enumerate(edges.tolist())
    }
    result = {edge_rows[key] for key, adjacent in owners.items() if len(adjacent) == 1}
    if threshold is None:
        return result
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    local = padded[cells]
    normals = np.cross(local[:, 1] - local[:, 0], local[:, 2] - local[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(
            "Feature classification requires nondegenerate source triangles."
        )
    normals /= norms[:, None]
    for key, adjacent in owners.items():
        if len(adjacent) != 2:
            continue
        cosine = float(
            np.clip(np.dot(normals[adjacent[0]], normals[adjacent[1]]), -1.0, 1.0)
        )
        if math.acos(cosine) >= threshold:
            result.add(edge_rows[key])
    return result


def _feature_inputs(value: ConvertibleToArray | None, edge_count: int) -> set[int]:
    if value is None:
        return set()
    edges = np.asarray(value)
    if edges.ndim != 1 or edges.dtype.kind not in "iu":
        raise TypeError("feature_edges must be a vector of source edge rows.")
    result = {int(row) for row in edges.tolist()}
    if any(row < 0 or row >= edge_count for row in result):
        raise ValueError("feature_edges must index the source mesh's edge entities.")
    return result


def _sparse_cross_field_solution(
    face_count: int,
    links: np.ndarray,
    transports: np.ndarray,
    feature_rows: np.ndarray,
    feature_angles: np.ndarray,
    /,
) -> tuple[np.ndarray, float, float, float]:
    diagonal = np.zeros((face_count,), dtype=np.complex128)
    right = np.zeros((face_count,), dtype=np.complex128)
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for (left, right_face), transport in zip(
        links.tolist(), transports.tolist(), strict=True
    ):
        phase = np.exp(4.0j * float(transport))
        diagonal[left] += 1.0
        diagonal[right_face] += 1.0
        rows.extend((left, right_face))
        columns.extend((right_face, left))
        values.extend((-np.conjugate(phase), -phase))
    constrained_faces = set()
    for face, angle in zip(feature_rows.tolist(), feature_angles.tolist(), strict=True):
        diagonal[face] += 1.0
        right[face] += np.exp(4.0j * float(angle))
        constrained_faces.add(face)
    patches = _surface_patch_ids(face_count, links)
    for patch in np.unique(patches).tolist():
        members = np.flatnonzero(patches == patch)
        if not any(int(face) in constrained_faces for face in members.tolist()):
            anchor = int(members[0])
            diagonal[anchor] += 1.0
            right[anchor] += 1.0
    rows.extend(range(face_count))
    columns.extend(range(face_count))
    values.extend(diagonal.tolist())
    matrix = coo_array(
        (
            np.asarray(values, dtype=np.complex128),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(face_count, face_count),
    ).tocsc()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MatrixRankWarning)
            solution = np.asarray(spsolve(matrix, right), dtype=np.complex128)
    except (MatrixRankWarning, RuntimeError, ValueError) as error:
        raise QuadLayoutObstruction(
            "cross_field_solve",
            "The sparse cross-field connection system is not representable.",
        ) from error
    if any(issubclass(item.category, MatrixRankWarning) for item in caught):
        raise QuadLayoutObstruction(
            "cross_field_solve",
            "The sparse cross-field connection system is singular.",
        )
    residual = matrix @ solution - right
    residual_norm = float(np.linalg.norm(residual))
    matrix_norm = float(np.linalg.norm(matrix.data))
    denominator = max(
        1.0,
        float(np.linalg.norm(right)),
        matrix_norm * float(np.linalg.norm(solution)),
    )
    relative_residual = residual_norm / denominator
    if (
        not np.all(np.isfinite(solution))
        or not math.isfinite(residual_norm)
        or not math.isfinite(relative_residual)
        or relative_residual > 1.0e-10
    ):
        raise QuadLayoutObstruction(
            "cross_field_solve",
            "The sparse cross-field connection solve did not meet its residual certificate.",
        )
    magnitudes = np.abs(solution)
    if np.any(magnitudes <= np.finfo(np.float64).tiny):
        faces = np.flatnonzero(magnitudes <= np.finfo(np.float64).tiny)
        raise QuadLayoutObstruction(
            "cross_field_representability",
            "The solved fourfold field vanishes on a source triangle.",
            source_cell_rows=faces.tolist(),
        )
    unit = solution / magnitudes
    angles = np.angle(unit) / 4.0
    angles = np.remainder(angles + np.pi / 4.0, np.pi / 2.0) - np.pi / 4.0
    link_mismatch = angles[links[:, 1]] - angles[links[:, 0]] - transports
    feature_mismatch = angles[feature_rows] - feature_angles
    energy = float(
        np.sum(1.0 - np.cos(4.0 * link_mismatch))
        + 4.0 * np.sum(1.0 - np.cos(4.0 * feature_mismatch))
    )
    if not math.isfinite(energy):
        raise QuadLayoutObstruction(
            "cross_field_representability",
            "The solved fourfold field has non-finite energy.",
        )
    return angles, residual_norm, relative_residual, energy


def _rotate_between_normals(
    vector: np.ndarray,
    source_normal: np.ndarray,
    target_normal: np.ndarray,
    vertex: int,
    /,
) -> np.ndarray:
    cosine = float(np.clip(np.dot(source_normal, target_normal), -1.0, 1.0))
    axis = np.cross(source_normal, target_normal)
    sine = float(np.linalg.norm(axis))
    if sine <= 1.0e-14:
        if cosine < 0.0:
            raise QuadLayoutObstruction(
                "singularity_transport",
                "Cross-field index transport is ambiguous across opposed source normals.",
                source_vertex_rows=(vertex,),
            )
        return vector.copy()
    axis /= sine
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.dot(axis, vector) * (1.0 - cosine)
    )


def _vertex_cross_field_indices(
    mesh: CellMesh,
    data: _TriangleData,
    frames: np.ndarray,
    angles: np.ndarray,
    feature_edges: set[int],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    local = padded[data.cells]
    raw_normals = np.cross(local[:, 1] - local[:, 0], local[:, 2] - local[:, 0])
    areas = np.linalg.norm(raw_normals, axis=1)
    if np.any(areas == 0.0):
        raise QuadLayoutObstruction(
            "singularity_transport",
            "Cross-field vertex indices require nondegenerate oriented source triangles.",
        )
    face_normals = raw_normals / areas[:, None]
    vertex_faces: list[list[int]] = [[] for _ in range(padded.shape[0])]
    for face, triangle in enumerate(data.cells.tolist()):
        for vertex in triangle:
            vertex_faces[vertex].append(face)
    boundary_vertices = {
        vertex
        for key, owners in data.edge_owners.items()
        if len(owners) == 1
        for vertex in key
    }
    feature_vertices = {
        int(vertex)
        for edge_row_value in feature_edges
        for vertex in data.edges[edge_row_value].tolist()
    }
    indexed_vertices: list[int] = []
    indices: list[int] = []
    quarter_turn = np.pi / 2.0
    for vertex, star in enumerate(vertex_faces):
        if vertex in boundary_vertices or vertex in feature_vertices or not star:
            continue
        next_face: dict[int, int] = {}
        for face in star:
            triangle = data.cells[face].tolist()
            position = triangle.index(vertex)
            next_vertex = triangle[(position + 2) % 3]
            key = _edge_key(vertex, next_vertex)
            incident = data.edge_owners[key]
            if len(incident) != 2:
                raise QuadLayoutObstruction(
                    "vertex_link",
                    "An interior cross-field vertex link is not a closed manifold cycle.",
                    source_vertex_rows=(vertex,),
                )
            next_face[face] = incident[0] if incident[1] == face else incident[1]
        start = min(star)
        order: list[int] = []
        current = start
        while current not in order:
            order.append(current)
            current = next_face.get(current, -1)
            if current < 0:
                break
        if current != start or len(order) != len(star):
            raise QuadLayoutObstruction(
                "vertex_link",
                "An interior cross-field vertex link is disconnected or inconsistently oriented.",
                source_vertex_rows=(vertex,),
            )
        normal_sum = np.sum(
            face_normals[np.asarray(star, dtype=np.int64)]
            * areas[np.asarray(star, dtype=np.int64), None],
            axis=0,
        )
        normal_length = float(np.linalg.norm(normal_sum))
        if normal_length <= np.finfo(np.float64).eps * float(
            np.sum(areas[np.asarray(star, dtype=np.int64)])
        ):
            raise QuadLayoutObstruction(
                "singularity_transport",
                "A source vertex has no stable tangent plane for cross-field index transport.",
                source_vertex_rows=(vertex,),
            )
        vertex_normal = normal_sum / normal_length
        tangent = None
        for face in order:
            triangle = data.cells[face].tolist()
            for other in triangle:
                if other == vertex:
                    continue
                direction = padded[other] - padded[vertex]
                direction -= np.dot(direction, vertex_normal) * vertex_normal
                direction_length = float(np.linalg.norm(direction))
                if direction_length > 1.0e-14:
                    tangent = direction / direction_length
                    break
            if tangent is not None:
                break
        if tangent is None:
            raise QuadLayoutObstruction(
                "singularity_transport",
                "A source vertex has no stable local tangent direction.",
                source_vertex_rows=(vertex,),
            )
        second_tangent = np.cross(vertex_normal, tangent)
        phases: dict[int, float] = {}
        for face in order:
            direction = (
                math.cos(float(angles[face])) * frames[face, 0]
                + math.sin(float(angles[face])) * frames[face, 1]
            )
            transported = _rotate_between_normals(
                direction, face_normals[face], vertex_normal, vertex
            )
            phases[face] = math.atan2(
                float(np.dot(transported, second_tangent)),
                float(np.dot(transported, tangent)),
            )
        circulation = 0.0
        for index, face in enumerate(order):
            following = order[(index + 1) % len(order)]
            delta = phases[following] - phases[face]
            circulation += (delta + np.pi / 4.0) % quarter_turn - np.pi / 4.0
        charge = int(np.rint(circulation / quarter_turn))
        if abs(circulation - charge * quarter_turn) > 1.0e-8:
            raise QuadLayoutObstruction(
                "singularity_index",
                "The cross-field vertex circulation is not representable as a fourfold index.",
                source_vertex_rows=(vertex,),
            )
        if charge != 0:
            indexed_vertices.append(vertex)
            indices.append(charge)
    return (
        np.asarray(indexed_vertices, dtype=np.int64),
        np.asarray(indices, dtype=np.int32),
    )


@checked
def prepare_surface_cross_field(
    mesh: CellMesh,
    /,
    *,
    feature_edges: ConvertibleToArray | None = None,
    feature_angle: float | None = None,
) -> SurfaceCrossField:
    """Solve a sparse transported fourfold field and expose its vertex indices."""
    data = _triangle_data(mesh)
    if data.cells.shape[0] == 0:
        raise ValueError("Cross-field preparation requires at least one source triangle.")
    threshold = None if feature_angle is None else float(feature_angle)
    if threshold is not None and (
        not math.isfinite(threshold) or not 0.0 <= threshold <= math.pi
    ):
        raise ValueError("feature_angle must lie in [0, pi] or be None.")
    frames = _triangle_frames(mesh, data.cells)
    explicit = _feature_inputs(feature_edges, data.edges.shape[0])
    features = _dihedral_feature_rows(
        mesh, data.cells, data.edges, data.edge_owners, threshold
    )
    features.update(explicit)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    adjacent: list[tuple[int, int]] = []
    transport: list[float] = []
    feature_rows: list[int] = []
    feature_angles: list[float] = []
    for edge_row, edge in enumerate(data.edges.tolist()):
        key = _edge_key(int(edge[0]), int(edge[1]))
        faces = data.edge_owners[key]
        direction = padded[key[1]] - padded[key[0]]
        local_angles = [_edge_angle(frames[face], direction) for face in faces]
        if edge_row in features:
            for face, angle in zip(faces, local_angles, strict=True):
                feature_rows.append(face)
                feature_angles.append(angle)
        elif len(faces) == 2:
            adjacent.append((faces[0], faces[1]))
            transport.append(local_angles[1] - local_angles[0])
        else:
            raise AssertionError("Every boundary edge is a field feature.")
    links = np.asarray(adjacent, dtype=np.int32).reshape((-1, 2))
    transports = np.asarray(transport, dtype=np.float64)
    feature_face_rows = np.asarray(feature_rows, dtype=np.int32)
    feature_directions = np.asarray(feature_angles, dtype=np.float64)
    angles, solve_residual, relative_residual, energy = _sparse_cross_field_solution(
        data.cells.shape[0],
        links,
        transports,
        feature_face_rows,
        feature_directions,
    )
    residuals = (
        np.abs(
            np.arctan2(
                np.sin(4.0 * (angles[feature_face_rows] - feature_directions)),
                np.cos(4.0 * (angles[feature_face_rows] - feature_directions)),
            )
        )
        / 4.0
    )
    singular_vertices, singularity_indices = _vertex_cross_field_indices(
        mesh, data, frames, angles, features
    )
    return SurfaceCrossField(
        frames,
        angles,
        links,
        _surface_patch_ids(data.cells.shape[0], links),
        np.asarray(sorted(features), dtype=np.int64),
        transports,
        feature_face_rows,
        feature_directions,
        residuals,
        solve_residual,
        relative_residual,
        energy,
        singular_vertices,
        singularity_indices,
        mesh,
    )


def _exact_orientation(
    cycle: tuple[int, ...], points: np.ndarray, /
) -> tuple[np.ndarray, int] | None:
    coordinates = points[np.asarray(cycle, dtype=np.int64)]
    if points.shape[1] == 2:
        projected = coordinates
    else:
        p0, p1, p2 = coordinates[:3]
        normal = np.cross(p1 - p0, p2 - p0)
        if float(np.linalg.norm(normal)) == 0.0:
            return None
        if int(
            np.asarray(orient3d(*coordinates, mode=PredicateMode.EXACT).signs).item()
        ) != int(PredicateSign.ZERO):
            return None
        dropped = int(np.argmax(np.abs(normal)))
        projected = coordinates[:, [axis for axis in range(3) if axis != dropped]]
    signs = []
    for index in range(4):
        sign = int(
            np.asarray(
                orient2d(
                    projected[index],
                    projected[(index + 1) % 4],
                    projected[(index + 2) % 4],
                    mode=PredicateMode.EXACT,
                ).signs
            ).item()
        )
        if sign in (int(PredicateSign.ZERO), int(PredicateSign.UNCERTAIN)):
            return None
        signs.append(sign)
    if len(set(signs)) != 1:
        return None
    return projected, signs[0]


def _oriented_pair_cycle(
    faces: tuple[int, int],
    edge: tuple[int, int],
    cells: np.ndarray,
    /,
) -> tuple[int, int, int, int] | None:
    boundary: list[tuple[int, int]] = []
    for face in faces:
        triangle = cells[face].tolist()
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            if _edge_key(a, b) != edge:
                boundary.append((a, b))
    if len(boundary) != 4:
        return None
    outgoing: dict[int, int] = {}
    for a, b in boundary:
        if a in outgoing:
            return None
        outgoing[a] = b
    if len(outgoing) != 4 or set(outgoing) != set(outgoing.values()):
        return None
    start = min(outgoing)
    cycle = [start]
    for _ in range(3):
        cycle.append(outgoing[cycle[-1]])
    if outgoing[cycle[-1]] != start or len(set(cycle)) != 4:
        return None
    return cycle[0], cycle[1], cycle[2], cycle[3]


def _edge_misalignment(
    field: SurfaceCrossField,
    face: int,
    edge: tuple[int, int],
    points: np.ndarray,
    /,
) -> float:
    direction = points[edge[1]] - points[edge[0]]
    frame = np.asarray(field.frames[face], dtype=np.float64)
    local = frame @ direction
    if float(np.linalg.norm(local)) == 0.0:
        return math.inf
    direction_angle = math.atan2(float(local[1]), float(local[0]))
    residual = 4.0 * (direction_angle - float(field.angles[face]))
    return 1.0 - math.cos(residual)


@final
class _PairCandidate(StrictModule):
    edge_row: int
    faces: tuple[int, int] = eqx.field(static=True)
    vertices: tuple[int, int, int, int] = eqx.field(static=True)
    alignment_error: float = eqx.field(static=True)
    quad_quality: float = eqx.field(static=True)


def _quad_patch_reference_chart(
    source_cells: np.ndarray,
    parent: int,
    cycle: tuple[int, int, int, int],
    reference_triangles: tuple[QuadReferencePoint, ...],
    reference_quads: tuple[QuadReferencePoint, ...],
    /,
) -> QuadChart:
    row = source_cells[parent]
    triangle = (int(row[0]), int(row[1]), int(row[2]))
    parameters: dict[int, QuadReferencePoint] = {
        vertex: reference_quads[index] for index, vertex in enumerate(cycle)
    }
    references: dict[int, QuadReferencePoint] = {
        vertex: reference_triangles[index] for index, vertex in enumerate(triangle)
    }
    first, second, third = (
        parameters[triangle[0]],
        parameters[triangle[1]],
        parameters[triangle[2]],
    )
    determinant = (second[0] - first[0]) * (third[1] - first[1]) - (
        third[0] - first[0]
    ) * (second[1] - first[1])
    if determinant == 0:
        raise ValueError("A quad patch chart has a degenerate source triangle.")
    charts: list[Polynomial] = []
    for axis in range(2):
        first_value = references[triangle[0]][axis]
        second_value = references[triangle[1]][axis]
        third_value = references[triangle[2]][axis]
        slope_first = (
            (second_value - first_value) * (third[1] - first[1])
            - (third_value - first_value) * (second[1] - first[1])
        ) / determinant
        slope_second = (
            (second[0] - first[0]) * (third_value - first_value)
            - (third[0] - first[0]) * (second_value - first_value)
        ) / determinant
        offset = first_value - slope_first * first[0] - slope_second * first[1]
        chart: Polynomial = {
            (0, 0): offset,
            (1, 0): slope_first,
            (0, 1): slope_second,
        }
        charts.append(chart)
    return charts[0], charts[1]


def _pair_source_chart_expressions(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    parents: tuple[int, int],
    cycle: tuple[int, int, int, int],
    elements: tuple[ScalarCoordinateElement, ...],
    routes: tuple[Array, ...],
    source_coordinates: CoordinateSourceBank,
    reference_triangles: tuple[QuadReferencePoint, ...],
    reference_quads: tuple[QuadReferencePoint, ...],
    /,
) -> tuple[tuple[Expression, ...], tuple[Expression, ...]]:
    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    return (
        _source_quad_coordinate_expressions(
            source,
            source_geometry,
            parents[0],
            _quad_patch_reference_chart(
                source_cells,
                parents[0],
                cycle,
                reference_triangles,
                reference_quads,
            ),
            elements,
            routes,
            source_coordinates,
        ),
        _source_quad_coordinate_expressions(
            source,
            source_geometry,
            parents[1],
            _quad_patch_reference_chart(
                source_cells,
                parents[1],
                cycle,
                reference_triangles,
                reference_quads,
            ),
            elements,
            routes,
            source_coordinates,
        ),
    )


def _quad_shape_quality(
    cycle: tuple[int, int, int, int],
    points: np.ndarray,
    /,
) -> float | None:
    oriented = _exact_orientation(cycle, points)
    if points.shape[1] == 2 and oriented is None:
        return None
    coordinates = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))[
        np.asarray(cycle, dtype=np.int64)
    ]
    if points.shape[1] == 3:
        coplanar = int(
            np.asarray(orient3d(*coordinates, mode=PredicateMode.EXACT).signs).item()
        ) == int(PredicateSign.ZERO)
        if coplanar and oriented is None:
            return None
    turns = []
    for index in range(4):
        previous = coordinates[index - 1] - coordinates[index]
        following = coordinates[(index + 1) % 4] - coordinates[index]
        first_length = float(np.linalg.norm(previous))
        second_length = float(np.linalg.norm(following))
        if first_length == 0.0 or second_length == 0.0:
            return None
        turn = np.cross(following, previous)
        magnitude = float(np.linalg.norm(turn))
        if magnitude == 0.0:
            return None
        turns.append(turn)
        if oriented is not None:
            projected, sign = oriented
            exact_turn = int(
                np.asarray(
                    orient2d(
                        projected[index],
                        projected[(index + 1) % 4],
                        projected[(index + 2) % 4],
                        mode=PredicateMode.EXACT,
                    ).signs
                ).item()
            )
            if exact_turn != sign:
                return None
    if points.shape[1] == 3 and oriented is None:
        reference = np.sum(turns, axis=0)
        if float(np.linalg.norm(reference)) == 0.0 or any(
            float(np.dot(value, reference)) <= 0.0 for value in turns
        ):
            return None
    return min(
        float(np.linalg.norm(value))
        / (
            float(np.linalg.norm(coordinates[(index - 1) % 4] - coordinates[index]))
            * float(np.linalg.norm(coordinates[(index + 1) % 4] - coordinates[index]))
        )
        for index, value in enumerate(turns)
    )


def _pair_candidate(
    edge_row: int,
    faces: tuple[int, int],
    edge: tuple[int, int],
    data: _TriangleData,
    field: SurfaceCrossField,
    points: np.ndarray,
    /,
) -> _PairCandidate | None:
    cycle = _oriented_pair_cycle(faces, edge, data.cells)
    if cycle is None:
        return None
    quality = _quad_shape_quality(cycle, points)
    if quality is None or quality <= 0.0:
        return None
    members = set(faces)
    alignment_weight = 0.0
    total_length = 0.0
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    for index in range(4):
        a, b = cycle[index], cycle[(index + 1) % 4]
        key = _edge_key(a, b)
        incident = [face for face in data.edge_owners[key] if face in members]
        if len(incident) != 1:
            return None
        length = float(np.linalg.norm(padded[b] - padded[a]))
        alignment_weight += length * _edge_misalignment(field, incident[0], key, padded)
        total_length += length
    if total_length == 0.0:
        return None
    return _PairCandidate(
        edge_row,
        faces,
        cycle,
        alignment_weight / total_length,
        quality,
    )


def _patch_boundary_obstruction(
    patch_faces: list[int],
    patch_id: int,
    data: _TriangleData,
    patch_ids: np.ndarray,
    edge_rows: dict[tuple[int, int], int],
    /,
) -> None:
    boundary_edges = {
        key
        for key, owners in data.edge_owners.items()
        if sum(int(patch_ids[face]) == patch_id for face in owners) == 1
    }
    if not boundary_edges:
        return
    incident: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for edge in boundary_edges:
        incident[edge[0]].add(edge)
        incident[edge[1]].add(edge)
    invalid_vertices = tuple(
        vertex for vertex, edges in incident.items() if len(edges) != 2
    )
    if invalid_vertices:
        raise QuadLayoutObstruction(
            "patch_boundary",
            "A feature-cut patch boundary is not a disjoint union of cycles.",
            source_cell_rows=patch_faces,
            source_edge_rows=(edge_rows[edge] for edge in boundary_edges),
            source_vertex_rows=invalid_vertices,
        )
    unseen_vertices = set(incident)
    while unseen_vertices:
        start = min(unseen_vertices)
        pending = [start]
        component_vertices: set[int] = set()
        component_edges: set[tuple[int, int]] = set()
        while pending:
            vertex = pending.pop()
            if vertex in component_vertices:
                continue
            component_vertices.add(vertex)
            unseen_vertices.discard(vertex)
            for edge in incident[vertex]:
                component_edges.add(edge)
                other = edge[1] if edge[0] == vertex else edge[0]
                if other not in component_vertices:
                    pending.append(other)
        if len(component_edges) % 2:
            raise QuadLayoutObstruction(
                "boundary_parity",
                "A feature-cut patch has an odd boundary cycle and cannot be recombined into quads.",
                source_cell_rows=patch_faces,
                source_edge_rows=(edge_rows[edge] for edge in component_edges),
            )


@checked
def prepare_surface_quad_layout(
    mesh: CellMesh,
    cross_field: SurfaceCrossField,
    /,
    *,
    maximum_field_error: float = 2.0,
    source_geometry: CellGeometrySpec | None = None,
    complete: bool = True,
) -> SurfaceQuadLayout:
    """Extract an exact, field-guided quad patching with parity and valence constraints.

    ``complete=False`` instead returns the maximal positive pairing; triangles
    it leaves unpaired are subdivided by the quad mesh builder.
    """
    data = _triangle_data(mesh)
    if (
        cross_field.source_topology_id != mesh.topology_id
        or cross_field.source_geometry_id != mesh.geometry_id
    ):
        raise ValueError("cross_field must bind the source mesh topology and geometry.")
    tolerance = float(maximum_field_error)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 2.0:
        raise ValueError("maximum_field_error must lie in [0, 2].")
    if source_geometry is None:
        source_geometry = CellGeometrySpec.affine(mesh)
    elif not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be CellGeometrySpec or None.")
    affine_source = _affine_geometry_matches(mesh, source_geometry)
    feature_edges = set(np.asarray(cross_field.feature_edges, dtype=np.int64).tolist())
    if any(row >= data.edges.shape[0] for row in feature_edges):
        raise ValueError("cross_field contains an undeclared source feature edge.")
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    expected_frames = _triangle_frames(mesh, data.cells)
    if not np.allclose(
        np.asarray(cross_field.frames), expected_frames, rtol=0.0, atol=1.0e-12
    ):
        raise QuadLayoutObstruction(
            "field_evidence",
            "Cross-field frame evidence does not match the bound source triangles.",
        )
    expected_links: list[tuple[int, int]] = []
    expected_transport: list[float] = []
    expected_feature_rows: list[int] = []
    expected_feature_angles: list[float] = []
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    for edge_row, edge_values in enumerate(data.edges.tolist()):
        edge = _edge_key(int(edge_values[0]), int(edge_values[1]))
        faces = data.edge_owners[edge]
        local_angles = [
            _edge_angle(expected_frames[face], padded[edge[1]] - padded[edge[0]])
            for face in faces
        ]
        if edge_row in feature_edges:
            if len(faces) == 2:
                for face, angle in zip(faces, local_angles, strict=True):
                    expected_feature_rows.append(face)
                    expected_feature_angles.append(angle)
            elif len(faces) == 1:
                expected_feature_rows.append(faces[0])
                expected_feature_angles.append(local_angles[0])
        elif len(faces) == 2:
            expected_links.append((faces[0], faces[1]))
            expected_transport.append(local_angles[1] - local_angles[0])
        else:
            raise QuadLayoutObstruction(
                "field_feature_integrity",
                "Every source boundary edge must be retained as a cross-field feature.",
                source_edge_rows=(edge_row,),
            )
    links = np.asarray(expected_links, dtype=np.int32).reshape((-1, 2))
    transports = np.asarray(expected_transport, dtype=np.float64)
    feature_rows = np.asarray(expected_feature_rows, dtype=np.int32)
    feature_angles = np.asarray(expected_feature_angles, dtype=np.float64)
    if not np.array_equal(
        np.asarray(cross_field.transport_edges), links
    ) or not np.allclose(
        np.asarray(cross_field.transport_angles),
        transports,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise QuadLayoutObstruction(
            "field_evidence",
            "Cross-field transport evidence does not match the source edge graph.",
        )
    if not np.array_equal(
        np.asarray(cross_field.feature_rows), feature_rows
    ) or not np.allclose(
        np.asarray(cross_field.feature_angles),
        feature_angles,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise QuadLayoutObstruction(
            "field_evidence",
            "Cross-field feature evidence does not match source feature directions.",
        )
    patches = _surface_patch_ids(data.cells.shape[0], links)
    if not np.array_equal(np.asarray(cross_field.patch_ids), patches):
        raise QuadLayoutObstruction(
            "field_evidence",
            "Cross-field patch labels do not match its cut transport graph.",
        )
    solved_angles, solve_residual, relative_residual, field_energy = (
        _sparse_cross_field_solution(
            data.cells.shape[0], links, transports, feature_rows, feature_angles
        )
    )
    angle_difference = (
        np.abs(
            np.arctan2(
                np.sin(4.0 * (np.asarray(cross_field.angles) - solved_angles)),
                np.cos(4.0 * (np.asarray(cross_field.angles) - solved_angles)),
            )
        )
        / 4.0
    )
    expected_feature_residuals = (
        np.abs(
            np.arctan2(
                np.sin(4.0 * (solved_angles[feature_rows] - feature_angles)),
                np.cos(4.0 * (solved_angles[feature_rows] - feature_angles)),
            )
        )
        / 4.0
    )
    if (
        np.any(angle_difference > 1.0e-10)
        or not np.allclose(
            np.asarray(cross_field.feature_residuals),
            expected_feature_residuals,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not math.isclose(
            float(cross_field.field_solve_residual),
            solve_residual,
            rel_tol=1.0e-8,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(cross_field.field_solve_relative_residual),
            relative_residual,
            rel_tol=1.0e-8,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(cross_field.field_energy),
            field_energy,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    ):
        raise QuadLayoutObstruction(
            "field_evidence",
            "Cross-field angles or sparse-solve evidence do not match the source-bound system.",
        )
    singular_vertices, singularity_indices = _vertex_cross_field_indices(
        mesh, data, expected_frames, solved_angles, feature_edges
    )
    if not np.array_equal(
        np.asarray(cross_field.singular_vertices), singular_vertices
    ) or not np.array_equal(
        np.asarray(cross_field.singularity_indices), singularity_indices
    ):
        raise QuadLayoutObstruction(
            "field_evidence",
            "Cross-field vertex-index evidence does not match the solved field.",
        )
    edge_row = {
        _edge_key(int(edge[0]), int(edge[1])): row
        for row, edge in enumerate(data.edges.tolist())
    }
    patch_faces = {
        int(patch): np.flatnonzero(patches == patch).astype(np.int64).tolist()
        for patch in np.unique(patches).tolist()
    }
    if complete:
        for patch, faces in patch_faces.items():
            if len(faces) % 2:
                raise QuadLayoutObstruction(
                    "patch_parity",
                    "A feature-constrained patch has odd triangle parity.",
                    source_cell_rows=faces,
                    source_edge_rows=(
                        edge_row[key]
                        for key, owners in data.edge_owners.items()
                        if sum(int(patches[face]) == patch for face in owners) == 1
                    ),
                )
            _patch_boundary_obstruction(faces, patch, data, patches, edge_row)
    elements: tuple[ScalarCoordinateElement, ...] = ()
    routes: tuple[Array, ...] = ()
    source_coordinate_bank: CoordinateSourceBank = ()
    if not affine_source:
        resolved_elements, routes, _ = source_geometry.resolve(mesh)
        if len(resolved_elements) != 1:
            raise QuadLayoutObstruction(
                "source_map",
                "Quad patch recombination requires one exact scalar source coordinate element.",
            )
        source_element = resolved_elements[0]
        if not isinstance(source_element, ScalarCoordinateElement):
            raise QuadLayoutObstruction(
                "source_map",
                "Quad patch recombination requires one exact scalar source coordinate element.",
            )
        elements = (source_element,)
        source_coordinate_bank = _exact_source_coordinate_bank(
            source_geometry.source_coordinates()
        )
    reference_triangles = _reference_points_2d(mesh.blocks[0].cell_kind)
    reference_quads = _reference_points_2d("quadrilateral")
    candidates: list[_PairCandidate] = []
    rejected_source_map_edges: list[int] = []
    considered_edges: list[int] = []
    for edge_row_value, edge_values in enumerate(data.edges.tolist()):
        edge = _edge_key(int(edge_values[0]), int(edge_values[1]))
        faces = data.edge_owners[edge]
        if (
            len(faces) != 2
            or edge_row_value in feature_edges
            or patches[faces[0]] != patches[faces[1]]
        ):
            continue
        considered_edges.append(edge_row_value)
        face_pair = (faces[0], faces[1])
        candidate = _pair_candidate(
            edge_row_value,
            face_pair,
            edge,
            data,
            cross_field,
            points,
        )
        if candidate is None:
            continue
        if candidate.alignment_error > tolerance:
            continue
        if not affine_source:
            expressions = _pair_source_chart_expressions(
                mesh,
                source_geometry,
                candidate.faces,
                candidate.vertices,
                elements,
                routes,
                source_coordinate_bank,
                reference_triangles,
                reference_quads,
            )
            if not expression_sequence_equal(expressions[0], expressions[1]):
                rejected_source_map_edges.append(edge_row_value)
                continue
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item.edge_row, item.faces))
    if not candidates and complete:
        if rejected_source_map_edges:
            raise QuadLayoutObstruction(
                "source_map",
                "No field-compatible patch pair shares one exact source map.",
                source_edge_rows=rejected_source_map_edges,
            )
        raise QuadLayoutObstruction(
            "field_alignment",
            "No feature-compatible, positive-quality, field-aligned source patch pair exists.",
            source_cell_rows=range(data.cells.shape[0]),
            source_edge_rows=considered_edges,
        )
    if not complete:
        return _surface_quad_layout(
            mesh,
            source_geometry,
            cross_field,
            patches,
            singular_vertices,
            singularity_indices,
            _maximum_partial_pairing(candidates, data.cells.shape[0]),
            tolerance,
            complete=False,
        )
    candidate_count = len(candidates)
    face_count = data.cells.shape[0]
    face_rows = np.asarray(
        [face for candidate in candidates for face in candidate.faces],
        dtype=np.int64,
    )
    candidate_columns = np.repeat(np.arange(candidate_count, dtype=np.int64), 2)
    face_incidence = coo_array(
        (
            np.ones(face_rows.size, dtype=np.float64),
            (face_rows, candidate_columns),
        ),
        shape=(face_count, candidate_count),
    ).tocsc()
    integrality = np.ones((candidate_count,), dtype=np.int8)
    bounds = MilpBounds(
        np.zeros((candidate_count,), dtype=np.float64),
        np.ones((candidate_count,), dtype=np.float64),
    )
    face_target = np.ones((face_count,), dtype=np.float64)
    face_constraints = LinearConstraint(face_incidence, face_target, face_target)
    face_matching = milp(
        np.zeros((candidate_count,), dtype=np.float64),
        integrality=integrality,
        bounds=bounds,
        constraints=face_constraints,
    )
    if not face_matching.success or face_matching.x is None:
        raise QuadLayoutObstruction(
            "patch_matching",
            "The feature-constrained source patches have no complete pair decomposition.",
            source_cell_rows=range(face_count),
        )
    boundary_vertices = {
        vertex
        for key, owners in data.edge_owners.items()
        if len(owners) == 1
        for vertex in key
    }
    feature_vertices = {
        int(vertex) for row in feature_edges for vertex in data.edges[row].tolist()
    }
    charge_by_vertex = dict(
        zip(singular_vertices.tolist(), singularity_indices.tolist(), strict=True)
    )
    incident_faces: list[list[int]] = [[] for _ in range(mesh.coordinates.shape[0])]
    for face, triangle in enumerate(data.cells.tolist()):
        for vertex in triangle:
            incident_faces[vertex].append(face)
    constrained_vertices = [
        vertex
        for vertex in range(mesh.coordinates.shape[0])
        if vertex not in boundary_vertices and vertex not in feature_vertices
    ]
    target_valences = np.asarray(
        [4 - charge_by_vertex.get(vertex, 0) for vertex in constrained_vertices],
        dtype=np.float64,
    )
    for vertex, target_valence in zip(
        constrained_vertices, target_valences.tolist(), strict=True
    ):
        if target_valence < 1.0 or target_valence > len(incident_faces[vertex]):
            raise QuadLayoutObstruction(
                "singularity_valence",
                "A cross-field index requests an impossible interior quad valence.",
                source_vertex_rows=(vertex,),
            )
    vertex_rows = {
        vertex: face_count + row for row, vertex in enumerate(constrained_vertices)
    }
    full_rows = face_rows.tolist()
    full_columns = candidate_columns.tolist()
    full_values = [1.0] * len(full_rows)
    for column, candidate in enumerate(candidates):
        for vertex in candidate.vertices:
            row = vertex_rows.get(vertex)
            if row is not None:
                full_rows.append(row)
                full_columns.append(column)
                full_values.append(1.0)
    full_incidence = coo_array(
        (
            np.asarray(full_values, dtype=np.float64),
            (
                np.asarray(full_rows, dtype=np.int64),
                np.asarray(full_columns, dtype=np.int64),
            ),
        ),
        shape=(face_count + len(constrained_vertices), candidate_count),
    ).tocsc()
    target = np.concatenate((face_target, target_valences))
    constraints = LinearConstraint(full_incidence, target, target)
    costs = np.asarray(
        [
            candidate.alignment_error
            + 0.25 * (1.0 - candidate.quad_quality)
            + 1.0e-12 * (candidate.edge_row + 1)
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    optimized = milp(
        costs,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    if not optimized.success or optimized.x is None:
        raise QuadLayoutObstruction(
            "singularity_valence",
            "No complete source patch decomposition satisfies the cross-field valences.",
            source_cell_rows=range(face_count),
            source_vertex_rows=constrained_vertices,
        )
    selection = np.rint(optimized.x).astype(np.int64)
    realized = np.asarray(full_incidence @ selection, dtype=np.float64)
    if not np.allclose(
        optimized.x, selection, rtol=0.0, atol=1.0e-8
    ) or not np.array_equal(realized, target):
        raise QuadLayoutObstruction(
            "patch_solver",
            "The sparse integer patch solver returned a nonintegral or infeasible decomposition.",
            source_cell_rows=range(face_count),
            source_vertex_rows=constrained_vertices,
        )
    selected = sorted(
        candidate.edge_row
        for candidate, selected_row in zip(candidates, selection.tolist(), strict=True)
        if selected_row == 1
    )
    return _surface_quad_layout(
        mesh,
        source_geometry,
        cross_field,
        patches,
        singular_vertices,
        singularity_indices,
        selected,
        tolerance,
        complete=True,
    )


def _maximum_partial_pairing(
    candidates: Sequence[_PairCandidate], face_count: int, /
) -> list[int]:
    """Select the most positive field-aligned pairs; leftovers stay triangles."""
    if not candidates:
        return []
    face_rows = np.asarray(
        [face for candidate in candidates for face in candidate.faces], dtype=np.int64
    )
    columns = np.repeat(np.arange(len(candidates), dtype=np.int64), 2)
    incidence = coo_array(
        (np.ones(face_rows.size, dtype=np.float64), (face_rows, columns)),
        shape=(face_count, len(candidates)),
    ).tocsc()
    # Each pair saves one subdivided cell; alignment and shape only break ties.
    costs = np.asarray(
        [
            -1.0
            + 1.0e-3 * (candidate.alignment_error + 0.25 * (1.0 - candidate.quad_quality))
            + 1.0e-12 * (candidate.edge_row + 1)
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    matching = milp(
        costs,
        integrality=np.ones((len(candidates),), dtype=np.int8),
        bounds=MilpBounds(
            np.zeros((len(candidates),), dtype=np.float64),
            np.ones((len(candidates),), dtype=np.float64),
        ),
        constraints=LinearConstraint(
            incidence,
            np.zeros((face_count,), dtype=np.float64),
            np.ones((face_count,), dtype=np.float64),
        ),
    )
    if not matching.success or matching.x is None:
        raise QuadLayoutObstruction(
            "patch_solver",
            "The sparse integer pair solver returned no partial matching.",
            source_cell_rows=range(face_count),
        )
    selection = np.rint(matching.x).astype(np.int64)
    if not np.allclose(matching.x, selection, rtol=0.0, atol=1.0e-8) or np.any(
        np.asarray(incidence @ selection) > 1
    ):
        raise QuadLayoutObstruction(
            "patch_solver",
            "The sparse integer pair solver returned a nonintegral or overlapping matching.",
            source_cell_rows=range(face_count),
        )
    return sorted(
        candidate.edge_row
        for candidate, chosen in zip(candidates, selection.tolist(), strict=True)
        if chosen == 1
    )


def _surface_quad_layout(
    mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    cross_field: SurfaceCrossField,
    patches: np.ndarray,
    singular_vertices: np.ndarray,
    singularity_indices: np.ndarray,
    selected: list[int],
    tolerance: float,
    /,
    *,
    complete: bool,
) -> SurfaceQuadLayout:
    layout_id = canonical_fingerprint(
        {
            "kind": "feature-guided-quad-recombination"
            if complete
            else "feature-guided-partial-quad-recombination",
            "source_topology": mesh.topology_id,
            "source_geometry": mesh.geometry_id,
            "source_coordinate_geometry": source_geometry.geometry_layout_id,
            "cross_field": {
                "angles": np.asarray(cross_field.angles),
                "patches": patches,
                "features": np.asarray(cross_field.feature_edges),
                "singular_vertices": singular_vertices,
                "singularity_indices": singularity_indices,
            },
            "paired_edge_rows": selected,
            "maximum_field_error": tolerance,
        }
    )
    count = mesh.blocks[0].cell_count
    return SurfaceQuadLayout(
        mesh,
        np.arange(count + 1, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        np.asarray(selected, dtype=np.int64),
        mesh.topology_id,
        layout_id,
    )


def _source_edge_ancestry(
    source: CellMesh,
    target: CellMesh,
    supports: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_connectivity = source.connectivity
    target_connectivity = target.connectivity
    if not isinstance(target_connectivity, PolygonalConnectivity):
        raise TypeError(
            "Surface quad edge ancestry requires polygonal target connectivity."
        )
    target_edge_rows = np.asarray(target_connectivity.edges, dtype=np.int64)
    target_edge_ids = np.asarray(target.entity_set(1).entity_ids, dtype=np.int64)
    if isinstance(source_connectivity, PolygonalConnectivity):
        source_edge_rows = np.asarray(source_connectivity.edges, dtype=np.int64)
    elif (
        isinstance(source_connectivity, SimplicialConnectivity)
        and source_connectivity.dimension == 2
    ):
        source_edge_rows = np.asarray(source_connectivity.entities[1], dtype=np.int64)
    else:
        raise TypeError(
            "Surface quad edge ancestry requires triangle source connectivity."
        )
    source_edge_entity_ids = np.asarray(source.entity_set(1).entity_ids, dtype=np.int64)
    if target_edge_rows.shape != (target_edge_ids.size, 2) or source_edge_rows.shape != (
        source_edge_entity_ids.size,
        2,
    ):
        raise ValueError(
            "Quad source-edge ancestry must align with canonical edge banks."
        )
    source_edges_by_key = {
        _edge_key(int(edge[0]), int(edge[1])): (
            int(source_edge_entity_ids[row]),
            (int(edge[0]), int(edge[1])),
        )
        for row, edge in enumerate(source_edge_rows.tolist())
    }
    source_edge_ids = np.full(target_edge_ids.shape, -1, dtype=np.int64)
    source_edge_parameter_numerators = np.zeros((target_edge_ids.size, 2), dtype=np.int64)
    source_edge_parameter_denominators = np.ones(
        (target_edge_ids.size, 2), dtype=np.int64
    )
    for target_edge_row, target_edge in enumerate(target_edge_rows.tolist()):
        endpoint_supports = tuple(
            tuple(int(value) for value in supports[int(vertex)] if value >= 0)
            for vertex in target_edge
        )
        source_endpoints = tuple(
            sorted(
                {
                    source_vertex
                    for support in endpoint_supports
                    for source_vertex in support
                }
            )
        )
        if len(source_endpoints) != 2:
            continue
        source_edge = source_edges_by_key.get(
            _edge_key(source_endpoints[0], source_endpoints[1])
        )
        if source_edge is None:
            continue
        source_edge_id, oriented_source_endpoints = source_edge
        parameters: list[tuple[int, int]] = []
        for support in endpoint_supports:
            if len(support) == 1 and support[0] in oriented_source_endpoints:
                parameters.append((oriented_source_endpoints.index(support[0]), 1))
            elif len(support) == 2 and set(support) == set(oriented_source_endpoints):
                parameters.append((1, 2))
            else:
                parameters = []
                break
        if len(parameters) != 2:
            continue
        source_edge_ids[target_edge_row] = source_edge_id
        source_edge_parameter_numerators[target_edge_row] = np.asarray(
            [value[0] for value in parameters], dtype=np.int64
        )
        source_edge_parameter_denominators[target_edge_row] = np.asarray(
            [value[1] for value in parameters], dtype=np.int64
        )
    return (
        source_edge_ids,
        source_edge_parameter_numerators,
        source_edge_parameter_denominators,
    )


def _build_recombined_quad_mesh(
    source: CellMesh,
    layout: SurfaceQuadLayout,
    /,
) -> tuple[
    CellMesh,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    data = _triangle_data(source)
    if (
        layout.source_topology_id != source.topology_id
        or layout.mesh.topology_id != source.topology_id
    ):
        raise ValueError("Quad layout must bind the unchanged source triangle topology.")
    selected = np.asarray(layout.paired_edge_rows, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected >= data.edges.shape[0]):
        raise QuadLayoutObstruction(
            "source_identity",
            "Quad layout contains an undeclared source edge row.",
            source_edge_rows=selected.tolist(),
        )
    points = np.asarray(source.coordinates, dtype=np.float64)
    pairs: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []
    consumed_faces: set[int] = set()
    for edge_row in selected.tolist():
        edge = _edge_key(
            int(data.edges[int(edge_row), 0]), int(data.edges[int(edge_row), 1])
        )
        faces = data.edge_owners[edge]
        if len(faces) != 2:
            raise QuadLayoutObstruction(
                "patch_decomposition",
                "A selected source pair no longer owns exactly two triangles.",
                source_edge_rows=(int(edge_row),),
                source_cell_rows=faces,
            )
        face_pair = (faces[0], faces[1])
        cycle = _oriented_pair_cycle(face_pair, edge, data.cells)
        if (
            cycle is None
            or _quad_shape_quality(cycle, points) is None
            or any(face in consumed_faces for face in face_pair)
        ):
            raise QuadLayoutObstruction(
                "patch_decomposition",
                "A selected source pair is no longer a positive, unique quad patch.",
                source_edge_rows=(int(edge_row),),
                source_cell_rows=face_pair,
            )
        pairs.append((cycle, face_pair))
        consumed_faces.update(face_pair)
    if len(consumed_faces) != data.cells.shape[0]:
        return _build_subdivided_pair_mesh(source, data, pairs)
    pairs.sort(key=lambda item: item[1])
    quad_rows = np.asarray([cycle for cycle, _ in pairs], dtype=np.int32).reshape((-1, 4))
    target_points = np.asarray(source.coordinates, dtype=np.float64).copy()
    block = CellBlock(source.blocks[0].name, "quadrilateral", quad_rows)
    target = CellMesh(
        target_points,
        (block,),
        vertex_global_ids=np.asarray(source.vertex_global_ids, dtype=np.int64),
        numeric_version=source.numeric_version,
    )
    supports = np.full((target_points.shape[0], 4), -1, dtype=np.int64)
    supports[:, 0] = np.arange(target_points.shape[0], dtype=np.int64)
    parent_cells = np.asarray(
        [parent for _, parents in pairs for parent in parents], dtype=np.int64
    )
    parent_offsets = np.arange(0, parent_cells.size + 1, 2, dtype=np.int64)
    valences = np.bincount(
        quad_rows.reshape(-1), minlength=target_points.shape[0]
    ).astype(np.int64)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for quad in quad_rows.tolist():
        for a, b in zip(quad, (*quad[1:], quad[0]), strict=True):
            edge_counts[(min(a, b), max(a, b))] += 1
    if any(count not in (1, 2) for count in edge_counts.values()):
        raise QuadLayoutObstruction(
            "patch_topology",
            "Recombined quad patches do not form a two-manifold cell complex.",
        )
    boundary = np.zeros((target_points.shape[0],), dtype=np.bool_)
    for edge, count in edge_counts.items():
        if count == 1:
            boundary[list(edge)] = True
    singular = np.flatnonzero((~boundary) & (valences != 4)).astype(np.int64)
    source_edge_ids, numerators, denominators = _source_edge_ancestry(
        source, target, supports
    )
    return (
        target,
        supports,
        parent_offsets,
        parent_cells,
        valences,
        boundary,
        singular,
        source_edge_ids,
        numerators,
        denominators,
    )


def _build_quad_mesh(
    source: CellMesh,
    layout: SurfaceQuadLayout | None,
    /,
    *,
    native_dual: bool = True,
) -> tuple[
    CellMesh,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if layout is not None:
        return _build_recombined_quad_mesh(source, layout)
    data = _triangle_data(source)
    points = np.asarray(source.coordinates, dtype=np.float64)
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    edge_row = {
        _edge_key(int(edge[0]), int(edge[1])): row
        for row, edge in enumerate(data.edges.tolist())
    }
    patches = [
        (
            tuple(triangle),
            (face,),
            tuple(
                edge_row[_edge_key(triangle[index], triangle[(index + 1) % 3])]
                for index in range(3)
            ),
        )
        for face, triangle in enumerate(data.cells.tolist())
    ]
    patches.sort(key=lambda value: (min(value[1]), len(value[0]), value[1]))
    used_edges = sorted({edge for _, _, edge_rows in patches for edge in edge_rows})
    midpoint_row = {
        row: source.coordinates.shape[0] + index for index, row in enumerate(used_edges)
    }
    midpoint_coordinates = np.asarray(
        [
            0.5 * (padded[edge[0]] + padded[edge[1]])
            for edge in (data.edges[row] for row in used_edges)
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    centers: list[np.ndarray] = []
    quads: list[list[int]] = []
    support_rows: list[list[int]] = [
        [vertex, -1, -1, -1] for vertex in range(source.coordinates.shape[0])
    ]
    support_rows.extend(
        [*sorted(map(int, data.edges[row].tolist())), -1, -1] for row in used_edges
    )
    parent_offsets = [0]
    parent_cells: list[int] = []
    center_base = source.coordinates.shape[0] + len(used_edges)
    for patch_rows, parents, patch_edge_rows in patches:
        patch_vertices = tuple(patch_rows)
        center = np.mean(padded[np.asarray(patch_vertices, dtype=np.int64)], axis=0)
        center_row = center_base + len(centers)
        centers.append(center)
        support = list(sorted(set(patch_vertices)))
        support_rows.append([*support, *([-1] * (4 - len(support)))])
        if len(patch_vertices) == 3:
            a, b, c = patch_vertices
            edge_ab = midpoint_row[edge_row[_edge_key(a, b)]]
            edge_bc = midpoint_row[edge_row[_edge_key(b, c)]]
            edge_ca = midpoint_row[edge_row[_edge_key(c, a)]]
            quads.extend(
                (
                    [a, edge_ab, center_row, edge_ca],
                    [b, edge_bc, center_row, edge_ab],
                    [c, edge_ca, center_row, edge_bc],
                )
            )
            parent_cells.extend((parents[0],) * 3)
            parent_offsets.extend(
                (parent_offsets[-1] + 1, parent_offsets[-1] + 2, parent_offsets[-1] + 3)
            )
        else:
            for index, vertex in enumerate(patch_vertices):
                edge = patch_edge_rows[index]
                previous_edge = patch_edge_rows[index - 1]
                quads.append(
                    [
                        vertex,
                        midpoint_row[edge],
                        center_row,
                        midpoint_row[previous_edge],
                    ]
                )
                parent_cells.extend(parents)
                parent_offsets.append(parent_offsets[-1] + len(parents))
    target_points3 = np.concatenate(
        (
            padded,
            midpoint_coordinates,
            np.asarray(centers, dtype=np.float64).reshape((-1, 3)),
        )
    )
    target_points = target_points3[:, : source.ambient_dimension]
    quad_rows = np.asarray(quads, dtype=np.int32)
    if layout is None and native_dual:
        nodes = np.empty((data.cells.shape[0], 7), dtype=np.int64)
        nodes[:, :3] = data.cells
        for row, triangle in enumerate(data.cells.tolist()):
            nodes[row, 3:6] = [
                midpoint_row[edge_row[_edge_key(triangle[a], triangle[b])]]
                for a, b in ((0, 1), (1, 2), (2, 0))
            ]
        nodes[:, 6] = center_base + np.arange(data.cells.shape[0], dtype=np.int64)
        native_quads = triangle_dual_quads(nodes, target_points.shape[0])
        if native_quads.shape != (3 * data.cells.shape[0], 4):
            raise ValueError(
                "Native triangle dual extraction returned an invalid cell extent."
            )
        quad_rows = np.asarray(native_quads, dtype=np.int32)
    return _quad_mesh_record(
        source, target_points, quad_rows, support_rows, parent_offsets, parent_cells
    )


def _quad_mesh_record(
    source: CellMesh,
    target_points: np.ndarray,
    quad_rows: np.ndarray,
    support_rows: Sequence[Sequence[int]],
    parent_offsets: Sequence[int],
    parent_cells: Sequence[int],
    /,
) -> tuple[
    CellMesh,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    block = CellBlock(source.blocks[0].name, "quadrilateral", quad_rows)
    vertex_ids = fresh_vertex_global_ids(
        source, target_points.shape[0] - source.coordinates.shape[0]
    )
    target = CellMesh(
        target_points,
        (block,),
        vertex_global_ids=vertex_ids,
        numeric_version=source.numeric_version,
    )
    supports = np.asarray(support_rows, dtype=np.int64)
    parents = np.asarray(parent_cells, dtype=np.int64)
    offsets = np.asarray(parent_offsets, dtype=np.int64)
    valences = np.bincount(
        quad_rows.reshape(-1), minlength=target_points.shape[0]
    ).astype(np.int64)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for quad in quad_rows.tolist():
        for a, b in zip(quad, (*quad[1:], quad[0]), strict=True):
            edge_counts[(min(a, b), max(a, b))] += 1
    boundary = np.zeros((target_points.shape[0],), dtype=np.bool_)
    for edge, count in edge_counts.items():
        if count == 1:
            boundary[list(edge)] = True
    (
        source_edge_ids,
        source_edge_parameter_numerators,
        source_edge_parameter_denominators,
    ) = _source_edge_ancestry(source, target, supports)
    singular = np.flatnonzero((~boundary) & (valences != 4)).astype(np.int64)
    return (
        target,
        supports,
        offsets,
        parents,
        valences,
        boundary,
        singular,
        source_edge_ids,
        source_edge_parameter_numerators,
        source_edge_parameter_denominators,
    )


def _build_subdivided_pair_mesh(
    source: CellMesh,
    data: _TriangleData,
    pairs: Sequence[tuple[tuple[int, int, int, int], tuple[int, int]]],
    /,
) -> tuple[
    CellMesh,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Quad-mesh a partial pairing by conforming midpoint subdivision.

    Every unpaired triangle splits into three quads at its edge midpoints and
    centroid. A split edge forces the opposite edge of each incident pair quad
    to split as well, so splits travel along quad strips until they close or
    leave the surface. Each pair then stays one quad, splits into two along
    one strip, or into four about its centroid; every split edge is split on
    both sides, and each new vertex keeps its exact source-vertex average.
    """
    paired_faces = {face for _, faces in pairs for face in faces}
    unpaired = [face for face in range(data.cells.shape[0]) if face not in paired_faces]

    def cycle_edge(cycle: Sequence[int], position: int, /) -> tuple[int, int]:
        return _edge_key(
            int(cycle[position % len(cycle)]), int(cycle[(position + 1) % len(cycle)])
        )

    quads_by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (cycle, _) in enumerate(pairs):
        for position in range(4):
            quads_by_edge[cycle_edge(cycle, position)].append(index)
    split: set[tuple[int, int]] = set()
    pending: list[tuple[int, int]] = []
    for face in unpaired:
        triangle = data.cells[face].tolist()
        for position in range(3):
            key = cycle_edge(triangle, position)
            if key not in split:
                split.add(key)
                pending.append(key)
    while pending:
        key = pending.pop()
        for index in quads_by_edge.get(key, ()):
            cycle = pairs[index][0]
            position = next(
                value for value in range(4) if cycle_edge(cycle, value) == key
            )
            opposite = cycle_edge(cycle, position + 2)
            if opposite not in split:
                split.add(opposite)
                pending.append(opposite)

    points = np.asarray(source.coordinates, dtype=np.float64)
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    vertex_count = points.shape[0]
    new_points: list[np.ndarray] = []
    support_rows: list[list[int]] = [
        [vertex, -1, -1, -1] for vertex in range(vertex_count)
    ]
    midpoint: dict[tuple[int, int], int] = {}
    for key in sorted(split):
        midpoint[key] = vertex_count + len(new_points)
        new_points.append(0.5 * (padded[key[0]] + padded[key[1]]))
        support_rows.append([key[0], key[1], -1, -1])

    def add_center(vertices: Sequence[int], /) -> int:
        row = vertex_count + len(new_points)
        new_points.append(np.mean(padded[np.asarray(vertices, dtype=np.int64)], axis=0))
        ordered = sorted(int(vertex) for vertex in vertices)
        support_rows.append([*ordered, *([-1] * (4 - len(ordered)))])
        return row

    quads: list[list[int]] = []
    parent_cells: list[int] = []
    parent_offsets = [0]

    def emit(quad: list[int], parents: Sequence[int], /) -> None:
        quads.append(quad)
        parent_cells.extend(parents)
        parent_offsets.append(parent_offsets[-1] + len(parents))

    entries = sorted(
        [(min(faces), 1, index) for index, (_, faces) in enumerate(pairs)]
        + [(face, 0, face) for face in unpaired]
    )
    for _, is_pair, value in entries:
        if not is_pair:
            triangle = tuple(int(vertex) for vertex in data.cells[value].tolist())
            center = add_center(triangle)
            for position in range(3):
                emit(
                    [
                        triangle[position],
                        midpoint[cycle_edge(triangle, position)],
                        center,
                        midpoint[cycle_edge(triangle, position - 1)],
                    ],
                    (value,),
                )
            continue
        cycle, faces = pairs[value]
        split_positions = [
            position for position in range(4) if cycle_edge(cycle, position) in split
        ]
        if not split_positions:
            emit(list(cycle), faces)
        elif len(split_positions) == 4:
            center = add_center(cycle)
            for position in range(4):
                emit(
                    [
                        cycle[position],
                        midpoint[cycle_edge(cycle, position)],
                        center,
                        midpoint[cycle_edge(cycle, position - 1)],
                    ],
                    faces,
                )
        elif split_positions in ([0, 2], [1, 3]):
            start = split_positions[0]
            rotated = (*cycle[start:], *cycle[:start])
            first = midpoint[cycle_edge(rotated, 0)]
            second = midpoint[cycle_edge(rotated, 2)]
            emit([rotated[0], first, second, rotated[3]], faces)
            emit([first, rotated[1], rotated[2], second], faces)
        else:
            raise QuadLayoutObstruction(
                "patch_decomposition",
                "Strip-propagated midpoint splits left a pair quad nonconforming.",
                source_cell_rows=faces,
            )
    target_points = np.concatenate(
        (padded, np.asarray(new_points, dtype=np.float64).reshape((-1, 3)))
    )[:, : source.ambient_dimension]
    return _quad_mesh_record(
        source,
        target_points,
        np.asarray(quads, dtype=np.int32),
        support_rows,
        parent_offsets,
        parent_cells,
    )


def _bilinear_triangle_chart(
    corners: Sequence[QuadReferencePoint],
    /,
) -> QuadChart:
    if len(corners) != 4:
        raise ValueError("A quad chart requires four source-reference corners.")
    first, second, third, fourth = corners
    result: list[Polynomial] = []
    for axis in range(2):
        terms = {
            (0, 0): first[axis],
            (1, 0): second[axis] - first[axis],
            (0, 1): fourth[axis] - first[axis],
            (1, 1): first[axis] - second[axis] + third[axis] - fourth[axis],
        }
        polynomial: Polynomial = {}
        for powers, coefficient in terms.items():
            if coefficient:
                polynomial[powers] = coefficient
        result.append(polynomial)
    return result[0], result[1]


def _chart_value(
    polynomial: Polynomial,
    point: QuadReferencePoint,
    /,
) -> ExactScalar:
    return sum(
        (
            coefficient * point[0] ** powers[0] * point[1] ** powers[1]
            for powers, coefficient in polynomial.items()
        ),
        Fraction(0),
    )


def _source_quad_coordinate_expressions(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    parent_row: int,
    chart: QuadChart,
    elements: tuple[ScalarCoordinateElement, ...],
    routes: tuple[Array, ...],
    source_coordinates: CoordinateSourceBank,
    /,
) -> tuple[Expression, ...]:
    element = elements[0]
    basis = tuple(
        _exact_expression(value)
        for value in coordinate_effective_basis(element, cell_row=parent_row)
    )
    local_route = np.asarray(routes[0], dtype=np.int64)[parent_row]
    coefficients = tuple(source_coordinates[int(index)] for index in local_route.tolist())
    coordinate_coefficients: tuple[tuple[ExactScalar, ...], ...] = tuple(
        tuple(coefficients[dof][axis] for dof in range(len(coefficients)))
        for axis in range(source.ambient_dimension)
    )
    physical = linear_combinations(basis, coordinate_coefficients)
    return tuple(compose_expression(value, chart) for value in physical)


def _compose_quad_source_geometry(
    source: CellMesh,
    target: CellMesh,
    supports: np.ndarray,
    parent_offsets: np.ndarray,
    parent_cells: np.ndarray,
    source_geometry: CellGeometrySpec,
    /,
) -> tuple[CellMesh, CellGeometrySpec]:
    """Compose each output quad through its exact source chart or shared patch map."""
    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    source_cell_ids = np.asarray(source.blocks[0].global_ids, dtype=np.int64)
    if (
        len(source.blocks) != 1
        or target.topological_dimension != 2
        or len(target.blocks) != 1
    ):
        raise ValueError(
            "Curved quad chart composition requires one source and target block."
        )
    resolved_elements, source_routes, _ = source_geometry.resolve(source)
    if len(resolved_elements) != 1:
        raise ValueError(
            "Curved quad chart composition requires one exact source coordinate element."
        )
    source_element = resolved_elements[0]
    if not isinstance(source_element, ScalarCoordinateElement):
        raise ValueError(
            "Curved quad chart composition requires one exact scalar coordinate element."
        )
    elements = (source_element,)
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    reference_triangles = _reference_points_2d(source.blocks[0].cell_kind)
    reference_quads = _reference_points_2d("quadrilateral")
    block = target.blocks[0]
    quad_rows = np.asarray(block.vertices, dtype=np.int64)
    target_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    source_coordinate_bank = _exact_source_coordinate_bank(
        source_geometry.source_coordinates()
    )
    selected_parent_rows = np.empty((quad_rows.shape[0],), dtype=np.int64)
    parent_vertex_ids = np.empty((quad_rows.shape[0], 3), dtype=np.int64)
    charts: list[QuadChart] = []
    exact_edge_expressions: dict[tuple[int, int], tuple[Expression, ...]] = {}
    for row, quad in enumerate(quad_rows.tolist()):
        start, stop = int(parent_offsets[row]), int(parent_offsets[row + 1])
        candidate_parents = tuple(int(value) for value in parent_cells[start:stop])
        if not candidate_parents or len(candidate_parents) > 2:
            raise ValueError(
                "Curved quad charts require one parent or an exactly shared two-triangle patch."
            )
        if any(not 0 <= parent < source_cells.shape[0] for parent in candidate_parents):
            raise ValueError("Quad chart ancestry references an absent source triangle.")
        chart_by_parent: list[QuadChart] = []
        expressions_by_parent: list[tuple[Expression, ...]] = []
        patch_cycle: tuple[int, int, int, int] | None = None
        patch_corners: list[QuadReferencePoint] = []
        if len(candidate_parents) == 2:
            shared_edge = sorted(
                set(map(int, source_cells[candidate_parents[0]].tolist()))
                & set(map(int, source_cells[candidate_parents[1]].tolist()))
            )
            if len(shared_edge) != 2:
                raise ValueError(
                    "A two-triangle quad patch must share exactly one source edge."
                )
            patch_cycle = _oriented_pair_cycle(
                (candidate_parents[0], candidate_parents[1]),
                _edge_key(shared_edge[0], shared_edge[1]),
                source_cells,
            )
            if patch_cycle is None:
                raise ValueError("A two-triangle quad patch has no oriented cycle.")
            # Target corners are exact source-vertex averages inside the patch;
            # a whole recombined pair is the identity corner assignment.
            for target_vertex in quad:
                support = tuple(
                    int(value)
                    for value in supports[int(target_vertex)].tolist()
                    if int(value) >= 0
                )
                if not support or any(vertex not in patch_cycle for vertex in support):
                    raise ValueError(
                        "A patch quad vertex lost its two-triangle source support."
                    )
                positions = tuple(patch_cycle.index(vertex) for vertex in support)
                patch_corners.append(
                    (
                        sum(
                            (reference_quads[index][0] for index in positions),
                            Fraction(0),
                        )
                        / len(positions),
                        sum(
                            (reference_quads[index][1] for index in positions),
                            Fraction(0),
                        )
                        / len(positions),
                    )
                )
        for parent in candidate_parents:
            if patch_cycle is not None:
                patch_chart = _quad_patch_reference_chart(
                    source_cells,
                    parent,
                    patch_cycle,
                    reference_triangles,
                    reference_quads,
                )
                corner_chart = _bilinear_triangle_chart(patch_corners)
                chart = (
                    cast(Polynomial, compose_expression(patch_chart[0], corner_chart)),
                    cast(Polynomial, compose_expression(patch_chart[1], corner_chart)),
                )
            else:
                triangle_row = source_cells[parent]
                vertices = (
                    int(triangle_row[0]),
                    int(triangle_row[1]),
                    int(triangle_row[2]),
                )
                reference_by_vertex: dict[int, QuadReferencePoint] = {
                    vertices[index]: reference_triangles[index] for index in range(3)
                }
                corners: list[QuadReferencePoint] = []
                for target_vertex in quad:
                    support = tuple(
                        int(value)
                        for value in supports[int(target_vertex)].tolist()
                        if int(value) >= 0
                    )
                    if not support or any(
                        vertex not in reference_by_vertex for vertex in support
                    ):
                        raise ValueError(
                            "A source-chart quad vertex lost its parent-triangle support."
                        )
                    corners.append(
                        (
                            sum(
                                (reference_by_vertex[vertex][0] for vertex in support),
                                Fraction(0),
                            )
                            / len(support),
                            sum(
                                (reference_by_vertex[vertex][1] for vertex in support),
                                Fraction(0),
                            )
                            / len(support),
                        )
                    )
                chart = _bilinear_triangle_chart(corners)
            chart_by_parent.append(chart)
            expressions_by_parent.append(
                _source_quad_coordinate_expressions(
                    source,
                    source_geometry,
                    parent,
                    chart,
                    elements,
                    source_routes,
                    source_coordinate_bank,
                )
            )
        if len(expressions_by_parent) == 2 and not expression_sequence_equal(
            expressions_by_parent[0], expressions_by_parent[1]
        ):
            raise ValueError(
                "A curved field-paired quad crosses source charts without one exact common patch map."
            )
        for index, start_vertex in enumerate(quad):
            stop_vertex = quad[(index + 1) % 4]
            start_reference = reference_quads[index]
            stop_reference = reference_quads[(index + 1) % 4]
            if start_vertex > stop_vertex:
                start_reference, stop_reference = stop_reference, start_reference
            first_edge_chart: Polynomial = {
                (0, 0): start_reference[0],
                (1, 0): stop_reference[0] - start_reference[0],
            }
            second_edge_chart: Polynomial = {
                (0, 0): start_reference[1],
                (1, 0): stop_reference[1] - start_reference[1],
            }
            edge_chart: QuadChart = (first_edge_chart, second_edge_chart)
            edge_expression: tuple[Expression, ...] = tuple(
                compose_expression(value, edge_chart)
                for value in expressions_by_parent[0]
            )
            edge_key = _edge_key(start_vertex, stop_vertex)
            edge_expression = tuple(
                compose_expression(value, edge_chart)
                for value in expressions_by_parent[0]
            )
            edge_key = (min(start_vertex, stop_vertex), max(start_vertex, stop_vertex))
            previous_expression = exact_edge_expressions.get(edge_key)
            if previous_expression is not None and not expression_sequence_equal(
                previous_expression, edge_expression
            ):
                raise ValueError(
                    "Adjacent source charts disagree on an exact shared quad-edge map."
                )
            exact_edge_expressions[edge_key] = edge_expression
        parent = candidate_parents[0]
        selected_parent_rows[row] = parent
        parent_vertex_ids[row] = source_vertex_ids[source_cells[parent]]
        charts.append(chart_by_parent[0])

    target_points = np.asarray(target.coordinates, dtype=np.float64).copy()
    exact_at_vertex: dict[int, tuple[ExactScalar, ...]] = {}
    source_cell_reference_ids = source_cell_ids[selected_parent_rows]
    for row, quad in enumerate(quad_rows.tolist()):
        chart = charts[row]
        source_references = tuple(
            tuple(_chart_value(polynomial, point) for polynomial in chart)
            for point in reference_quads
        )
        evaluated_coordinates = source_geometry.evaluate_exact(
            source,
            np.asarray((source_cell_reference_ids[row],), dtype=np.int64),
            (source_references,),
        )[0]
        exact_coordinates = _exact_source_coordinate_bank(evaluated_coordinates)
        for vertex, exact_coordinate in zip(quad, exact_coordinates, strict=True):
            previous = exact_at_vertex.get(int(vertex))
            if previous is not None and previous != exact_coordinate:
                raise ValueError(
                    "Adjacent source charts disagree on an exact shared quad vertex."
                )
            exact_at_vertex[int(vertex)] = exact_coordinate
            target_points[int(vertex)] = np.asarray(
                [float(value) for value in exact_coordinate], dtype=np.float64
            )
    target = CellMesh(
        target_points,
        target.blocks,
        vertex_global_ids=np.asarray(target.vertex_global_ids, dtype=np.int64),
        numeric_version=target.numeric_version,
    )
    restricted_element = PerCellRestrictedCellGeometryElement(
        elements[0],
        "quadrilateral",
        tuple(int(value) for value in target_cell_ids.tolist()),
        tuple(int(value) for value in source_cell_reference_ids.tolist()),
        tuple(charts),
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source.topology_id,
        {block.name: source_cell_reference_ids},
        {block.name: parent_vertex_ids},
        block_source_blocks={block.name: source.blocks[0].name},
    )
    restricted_map = (
        None
        if source_geometry.source_map is None
        else _RestrictedCoordinateMap(source, source_geometry, target)
    )
    geometry = CellGeometrySpec(
        {block.name: restricted_element},
        {block.name: np.asarray(source_routes[0], dtype=np.int64)[selected_parent_rows]},
        source_geometry.coordinates,
        source_map=restricted_map,
        source_mesh=target if restricted_map is not None else None,
        restriction_source=restriction,
    )
    return target, geometry


@checked
def extract_surface_quads(
    source: CellMesh,
    /,
    *,
    cross_field: SurfaceCrossField | None = None,
    source_geometry: CellGeometrySpec | None = None,
    maximum_field_error: float = 2.0,
    unpaired_triangles: Literal["refuse", "subdivide"] = "refuse",
) -> DualExtraction:
    """Extract a triangle dual or an exact field-guided patch recombination.

    Field-guided patch cells retain exact source-parent and feature-edge ancestry.
    Non-affine source maps must agree exactly across each recombined pair; every
    returned target map is independently certified for continuous positivity.
    When no complete pairing exists, ``unpaired_triangles="subdivide"`` keeps
    the maximal positive pairing and subdivides the rest conformingly at edge
    midpoints instead of refusing.
    """
    if unpaired_triangles not in ("refuse", "subdivide"):
        raise ValueError("unpaired_triangles must be 'refuse' or 'subdivide'.")
    _triangle_data(source)
    if source_geometry is None:
        source_geometry = CellGeometrySpec.affine(source)
    elif not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be CellGeometrySpec or None.")
    affine_source = _affine_geometry_matches(source, source_geometry)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    if source_validity.invalid_count or source_validity.unresolved_count:
        if cross_field is not None:
            source_status = np.asarray(source_validity.status)
            raise QuadLayoutObstruction(
                "source_map",
                "Source triangles lack certified positive coordinate maps.",
                source_cell_rows=np.flatnonzero(
                    source_status != int(CellValidityStatus.CERTIFIED_VALID)
                ).tolist(),
            )
        raise ValueError(
            "Quad extraction source triangles must have certified positive cell maps."
        )
    layout = None
    if cross_field is not None:
        try:
            layout = prepare_surface_quad_layout(
                source,
                cross_field,
                maximum_field_error=maximum_field_error,
                source_geometry=source_geometry,
            )
        except QuadLayoutObstruction as obstruction:
            if (
                unpaired_triangles != "subdivide"
                or obstruction.category not in _PAIRING_OBSTRUCTIONS
            ):
                raise
            layout = prepare_surface_quad_layout(
                source,
                cross_field,
                maximum_field_error=maximum_field_error,
                source_geometry=source_geometry,
                complete=False,
            )
    (
        target,
        supports,
        offsets,
        parents,
        valences,
        boundary,
        singular,
        source_edge_ids,
        source_edge_parameter_numerators,
        source_edge_parameter_denominators,
    ) = _build_quad_mesh(source, layout, native_dual=affine_source)
    if affine_source:
        geometry = CellGeometrySpec.affine(target)
    else:
        try:
            target, geometry = _compose_quad_source_geometry(
                source, target, supports, offsets, parents, source_geometry
            )
        except ValueError as error:
            if layout is None:
                raise
            raise QuadLayoutObstruction(
                "source_map",
                "A recombined source chart could not be composed exactly.",
                source_cell_rows=parents.tolist(),
            ) from error
    validity = certify_cell_geometry_validity(geometry, mesh=target)
    status = np.asarray(validity.status)
    if np.any(status != int(CellValidityStatus.CERTIFIED_VALID)):
        if layout is not None:
            invalid_rows = np.flatnonzero(
                status != int(CellValidityStatus.CERTIFIED_VALID)
            )
            obstructed_parents = tuple(
                int(parent)
                for row in invalid_rows.tolist()
                for parent in parents[int(offsets[row]) : int(offsets[row + 1])].tolist()
            )
            raise QuadLayoutObstruction(
                "map_inversion",
                "A recombined quad map lacks a continuous positive-Jacobian certificate.",
                source_cell_rows=obstructed_parents,
            )
        raise ValueError(
            "Quad extraction produced a cell without certified positive validity."
        )
    return DualExtraction(
        target,
        source,
        supports,
        offsets,
        parents,
        validity,
        valences,
        boundary,
        singular,
        cross_field,
        geometry,
        source_geometry,
        source_edge_ids=source_edge_ids,
        source_edge_parameter_numerators=source_edge_parameter_numerators,
        source_edge_parameter_denominators=source_edge_parameter_denominators,
    )


__all__ = [
    "QuadLayoutObstruction",
    "prepare_surface_cross_field",
    "prepare_surface_quad_layout",
    "extract_surface_quads",
]
