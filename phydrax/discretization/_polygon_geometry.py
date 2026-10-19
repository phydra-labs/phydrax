#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from contextlib import ExitStack
from fractions import Fraction
from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .. import integration as imported_integration
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import bigint_bytes
from .._meshcore import (
    exact_orient2d,
    NativeExecutionLimits,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._polygon import (
    orientation as _orientation,
    signed_area2 as _signed_area2,
    validate_simple_polygon as _validate_simple_polygon,
)
from ..geometry._triangulation import ConstrainedDelaunayTriangulation
from ..typing import checked
from . import (
    _exact_power_consumers as imported_exact_power_consumers,
    _exact_rational_geometry as exact_rational_geometry_owner,
)
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry import CellGeometrySpec, owned_coordinate_preparation
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget
from ._exact_algebraic_geometry import ExactAlgebraicCellGeometrySource
from ._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ._polygon_types import PolyhedralFaceTriangulation


def _face_fraction_proof_bounds(count: int, /) -> tuple[int, int]:
    """Bound the unchanged shoelace and reciprocal partition arithmetic."""
    triangles = count - 2
    # Each shoelace term lifts four scalars, multiplies twice, subtracts and
    # accumulates. Each partition term lifts eight scalars, takes four
    # differences, multiplies twice, subtracts and accumulates.
    work = 8 * count + 16 * triangles
    # Binary64 Fractions have at most 1075 input bits. Unreduced shoelace
    # differences need 4*b+1 bits; differences-of-products of input
    # differences need 8*b+5 bits. Each summed term adds one further bit.
    bits = max(count * (4 * 1075 + 2) + 1, triangles * (8 * 1075 + 6) + 1)
    integer = bigint_bytes(bits)
    return work, work * (2 * integer + 64) + 7 * bigint_bytes(2 * bits + 1)


def _face_numeric_preparation_memory(sizes: np.ndarray, /) -> int:
    """Bound output and per-face float/index/evidence buffer shapes."""
    face_count = sizes.size
    triangle_count = int(np.sum(sizes - 2))
    memory = (
        (face_count + 1) * 8
        + triangle_count * 3 * 4
        + face_count * (7 * 8 + 9 * 8 + 6 * 8)
    )
    for size in sizes:
        count = int(size)
        memory += (
            count * (3 * 8 + 3 * 8 + 2 * 8 + 2 * 4)
            + (count - 2) * (3 * 4 + 3 * 4 + 9 * 8 + 3 * 8 + 8)
            + count * (2 * 8 + 2 * 4 + 4 + 8)
            + (2 * count + 4) * 6 * 4
        )
    return memory


@checked
def polyhedral_face_preparation_bounds(
    mesh: CellMesh,
    /,
    *,
    maximum_entries: int,
) -> tuple[int, int, int]:
    """Bound host face proofs and buffers, not the independent native CDT."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "PolyhedralFaceTriangulation.connectivity must be polyhedral; got another connectivity kind"
        )
    offsets = np.asarray(connectivity.face_vertex_offsets)
    sizes = np.diff(offsets)
    counts = sizes - 2
    if np.any(counts < 1) or 3 * int(np.sum(counts)) > maximum_entries:
        raise ValueError(
            "PolyhedralFaceTriangulation.maximum_entries must admit triangulation capacity; got an exceeded capacity"
        )
    memory = _face_numeric_preparation_memory(sizes)
    work = 0
    for size in sizes:
        proof_work, proof_memory = _face_fraction_proof_bounds(int(size))
        work += proof_work
        memory += proof_memory
    # Raw native orientation queries remain primitive telemetry, not a new
    # source-query or host work debit. CDT owns its preexisting native controls.
    return work, memory, 0


@checked
def prepare_polyhedral_face_triangulation(
    mesh: CellMesh,
    /,
    *,
    planarity_tolerance: float = 1e-10,
    maximum_entries: int = 100_000_000,
    cell_geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PolyhedralFaceTriangulation:
    """Triangulate every face exactly in projection, retaining all boundary nodes.

    No fan center is assumed to lie inside a nonconvex face. The native CDT
    uses its closed constrained boundary to classify the domain. Triangle
    incidence and projected area are independently checked here; zero-area
    triangles, discarded boundary points and unresolved native status refuse.
    """

    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "PolyhedralFaceTriangulation.connectivity must be polyhedral; got another connectivity kind"
        )
    tolerance = float(planarity_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            f"PolyhedralFaceTriangulation.planarity_tolerance must be finite and positive; got {tolerance!r}"
        )
    if (
        isinstance(maximum_entries, bool)
        or not isinstance(maximum_entries, int)
        or maximum_entries < 1
    ):
        raise ValueError(
            f"PolyhedralFaceTriangulation.maximum_entries must be a positive integer; got {maximum_entries!r}"
        )
    if cell_geometry is not None:
        match cell_geometry.exact_source:
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
            ):
                if budget is not None:
                    return imported_exact_power_consumers.exact_power_face_triangulation(
                        mesh,
                        cell_geometry,
                        maximum_entries=maximum_entries,
                        budget=budget,
                    )
                work, memory, queries = (
                    imported_exact_power_consumers.exact_power_finite_volume_preparation_bounds(
                        mesh, cell_geometry
                    )
                )
                with owned_coordinate_preparation(
                    work, memory, maximum_geometry_queries=queries
                ) as budget:
                    return imported_exact_power_consumers.exact_power_face_triangulation(
                        mesh,
                        cell_geometry,
                        maximum_entries=maximum_entries,
                        budget=budget,
                    )
            case ExactAlgebraicCellGeometrySource():
                raise ValueError(
                    "PolyhedralFaceTriangulation.cell_geometry.exact_source must be a power source; got an algebraic nodal source"
                )
            case (
                None
                | ExactPlcCellGeometrySource()
                | ExactPlcCellGeometryConvexSource()
                | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
            ):
                raise ValueError(
                    "PolyhedralFaceTriangulation.cell_geometry.exact_source must be a power source; got another coordinate source"
                )
            case invalid:
                assert_never(invalid)
    with ExitStack() as resources:
        if budget is None:
            work, memory, queries = polyhedral_face_preparation_bounds(
                mesh, maximum_entries=maximum_entries
            )
            budget = resources.enter_context(
                owned_coordinate_preparation(
                    work, memory, maximum_geometry_queries=queries
                )
            )
        resources.enter_context(budget.temporary_scope())
        return _prepare_raw_polyhedral_face_triangulation(
            mesh, connectivity, tolerance, maximum_entries, budget=budget
        )


class _ProjectedFace(StrictModule):
    """A planar physical face and its rational projected area."""

    normal: np.ndarray
    projected: np.ndarray
    area2: Fraction = eqx.field(static=True)
    boundary: np.ndarray


class _CertifiedFaceTriangles(StrictModule):
    """Boundary-preserving triangles with native work and memory evidence."""

    local: np.ndarray
    work: np.ndarray
    memory_counters: NativeExecutionMemoryCounters
    evidence_id: str = eqx.field(static=True)


def _project_polyhedral_face(
    polygon: np.ndarray, face: int, tolerance: float, budget: CoordinateEnclosureBudget, /
) -> _ProjectedFace:
    shifted = polygon - polygon[0]
    vector = np.sum(np.cross(shifted, np.roll(shifted, -1, axis=0)), axis=0) / 2
    magnitude = float(np.linalg.norm(vector))
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError(
            f"PolyhedralFaceTriangulation.faces must have resolved positive area; got degenerate face {face}"
        )
    normal = vector / magnitude
    if np.max(np.abs(shifted @ normal)) > tolerance * max(
        1.0, float(np.max(np.linalg.norm(shifted, axis=1)))
    ):
        raise ValueError(
            f"PolyhedralFaceTriangulation.faces must satisfy planarity admission; got nonplanar face {face}"
        )
    axis = int(np.argmax(np.abs(vector)))
    axes = ((axis + 1) % 3, (axis + 2) % 3)
    projected = polygon[:, axes]
    count = polygon.shape[0]
    _, proof_memory = _face_fraction_proof_bounds(count)
    budget.reserve(8 * count, proof_memory)
    area2 = sum(
        Fraction(float(a[0])) * Fraction(float(b[1]))
        - Fraction(float(a[1])) * Fraction(float(b[0]))
        for a, b in zip(projected, np.roll(projected, -1, axis=0), strict=True)
    )
    if area2 == 0:
        raise ValueError(
            f"PolyhedralFaceTriangulation.faces must have nonzero projected area; got zero area at face {face}"
        )
    boundary = np.column_stack((np.arange(count), np.roll(np.arange(count), -1)))
    return _ProjectedFace(normal, projected, area2, boundary)


def _triangulate_projected_face(
    projection: _ProjectedFace, face: int, budget: CoordinateEnclosureBudget, /
) -> _CertifiedFaceTriangles:
    projected, area2, boundary = (
        projection.projected,
        projection.area2,
        projection.boundary,
    )
    count = projected.shape[0]

    cavity = max(1, 2 * count + 4)
    # CDT retains the primitive's preexisting work and scratch limits.
    limits = NativeExecutionLimits(
        (1 << 63) - 1,
        (1 << 64) - 1,
        cavity,
        (1 << 63) - 1,
        math.inf,
    )
    with NativeExecutionScope(limits):
        prepared = ConstrainedDelaunayTriangulation(
            projected,
            boundary,
            max_steiner=0,
            max_triangles=2 * count + 4,
            max_cavity_cells=cavity,
        )
    if (
        prepared.evidence.status != "ok"
        or prepared.points.shape != projected.shape
        or not np.array_equal(prepared.points, projected)
    ):
        raise ValueError(
            f"PolyhedralFaceTriangulation.faces must retain resolved boundary-preserving native triangles; got unresolved triangulation at face {face}"
        )
    local = np.asarray(prepared.triangles, dtype=np.int32).copy()
    if local.shape != (count - 2, 3) or not np.array_equal(
        np.unique(local), np.arange(count)
    ):
        raise ValueError(
            f"PolyhedralFaceTriangulation.faces must retain boundary topology; got missing boundary incidence at face {face}"
        )
    if area2 < 0:
        local[:, [1, 2]] = local[:, [2, 1]]

    signs = exact_orient2d(
        projected[local[:, 0]], projected[local[:, 1]], projected[local[:, 2]]
    )
    if np.any(signs != (1 if area2 > 0 else -1)):
        raise ValueError(
            f"PolyhedralFaceTriangulation.faces must contain positive triangles; got nonpositive triangles at face {face}"
        )
    directed: dict[tuple[int, int], int] = {}
    partition = Fraction(0)
    budget.reserve(16 * local.shape[0])
    for triangle in local:
        corners = projected[triangle]
        partition += (Fraction(float(corners[1, 0])) - Fraction(float(corners[0, 0]))) * (
            Fraction(float(corners[2, 1])) - Fraction(float(corners[0, 1]))
        ) - (Fraction(float(corners[1, 1])) - Fraction(float(corners[0, 1]))) * (
            Fraction(float(corners[2, 0])) - Fraction(float(corners[0, 0]))
        )
        for a, b in zip(triangle, np.roll(triangle, -1), strict=True):
            a, b = int(a), int(b)
            edge = (min(a, b), max(a, b))
            directed[edge] = directed.get(edge, 0) + (1 if a < b else -1)
    expected = {
        (min(int(a), int(b)), max(int(a), int(b))): (1 if a < b else -1)
        for a, b in boundary
    }
    actual = {edge: sign for edge, sign in directed.items() if sign}
    if actual != expected or partition != area2:
        raise ValueError(
            f"PolyhedralFaceTriangulation.faces must close reciprocal boundary and projected area; got inconsistent closure at face {face}"
        )
    return _CertifiedFaceTriangles(
        local,
        prepared.work_evidence,
        prepared.memory_counters,
        prepared.evidence.evidence_id,
    )


def _prepare_raw_polyhedral_face_triangulation(
    mesh: CellMesh,
    connectivity: PolyhedralConnectivity,
    tolerance: float,
    maximum_entries: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PolyhedralFaceTriangulation:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    c = connectivity
    offsets, values = np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values)
    counts = np.diff(offsets) - 2
    if np.any(counts < 1) or 3 * int(np.sum(counts)) > maximum_entries:
        raise ValueError(
            "PolyhedralFaceTriangulation.maximum_entries must admit triangulation capacity; got an exceeded capacity"
        )
    budget.reserve(0, _face_numeric_preparation_memory(np.diff(offsets)))
    triangle_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    triangles = np.empty((int(triangle_offsets[-1]), 3), dtype=np.int32)
    vectors = np.empty((c.face_count, 3), dtype=np.float64)
    measures = np.empty(c.face_count, dtype=np.float64)
    centers = np.empty((c.face_count, 3), dtype=np.float64)
    work, memory, evidence = [], [], []
    for face, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        loop = values[start:stop]
        projection = _project_polyhedral_face(points[loop], face, tolerance, budget)
        prepared = _triangulate_projected_face(projection, face, budget)
        local, normal = prepared.local, projection.normal
        global_triangles = loop[local]
        triangle_points = points[global_triangles]
        triangle_vectors = (
            np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            )
            / 2
        )
        weights = triangle_vectors @ normal
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(
                f"PolyhedralFaceTriangulation.faces must have resolved physical triangle measures; got invalid measures at face {face}"
            )
        triangles[triangle_offsets[face] : triangle_offsets[face + 1]] = global_triangles
        vectors[face] = np.sum(triangle_vectors, axis=0)
        measures[face] = np.sum(weights)
        centers[face] = (
            np.sum(weights[:, None] * np.mean(triangle_points, axis=1), axis=0)
            / measures[face]
        )
        work.append(prepared.work)
        memory.append(prepared.memory_counters)
        evidence.append(prepared.evidence_id)
    arrays = (
        triangle_offsets,
        triangles,
        vectors,
        measures,
        centers,
        np.asarray(work),
    )
    for array in arrays:
        array.setflags(write=False)
    return PolyhedralFaceTriangulation(
        *arrays,
        tuple(memory),
        tuple(evidence),
        canonical_fingerprint(
            {
                "kind": "polyhedral-face-triangulation",
                "mesh": mesh.mesh_id,
                "triangles": array_tree_fingerprint(triangles),
                "planarity_tolerance": tolerance,
                "native": evidence,
            }
        ),
    )


def _cross_2d(left: Array, right: Array) -> Array:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _remove_collinear(points: np.ndarray, indices: list[int], /) -> list[int]:
    changed = True
    scale = max(float(np.max(np.abs(points))), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
    result = list(indices)
    while changed and len(result) > 3:
        changed = False
        for position in range(len(result)):
            left = result[position - 1]
            center = result[position]
            right = result[(position + 1) % len(result)]
            if (
                abs(_orientation(points[left], points[center], points[right]))
                <= tolerance
            ):
                result.pop(position)
                changed = True
                break
    return result


def _inside_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tolerance: float | np.floating,
    /,
) -> bool | np.bool_:
    return (
        _orientation(a, b, point) >= -tolerance
        and _orientation(b, c, point) >= -tolerance
        and _orientation(c, a, point) >= -tolerance
    )


def _ear_clip(points: np.ndarray, /) -> tuple[tuple[int, int, int], ...]:
    remaining = _remove_collinear(points, list(range(points.shape[0])))
    triangles: list[tuple[int, int, int]] = []
    scale = max(float(np.max(np.abs(points))), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
    while len(remaining) > 3:
        clipped = False
        for position in range(len(remaining)):
            left = remaining[position - 1]
            center = remaining[position]
            right = remaining[(position + 1) % len(remaining)]
            if _orientation(points[left], points[center], points[right]) <= tolerance:
                continue
            if any(
                candidate not in (left, center, right)
                and _inside_triangle(
                    points[candidate],
                    points[left],
                    points[center],
                    points[right],
                    tolerance,
                )
                for candidate in remaining
            ):
                continue
            triangles.append((left, center, right))
            remaining.pop(position)
            clipped = True
            break
        if not clipped:
            raise ValueError(
                "PolygonTriangulation.coordinates must admit a valid ear; got unresolved polygon triangulation"
            )
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    if not triangles:
        raise ValueError(
            "PolygonTriangulation.coordinates must produce positive triangles; got no positive triangles"
        )
    return tuple(triangles)


def _clip_half_plane(
    polygon: list[np.ndarray], start: np.ndarray, stop: np.ndarray, /
) -> list[np.ndarray]:
    if not polygon:
        return []
    edge = stop - start

    def signed(point: np.ndarray) -> float:
        return _orientation(start, stop, point)

    result: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = signed(previous)
    for current in polygon:
        current_value = signed(current)
        previous_inside = previous_value >= 0.0
        current_inside = current_value >= 0.0
        if previous_inside != current_inside:
            direction = current - previous
            denominator = edge[0] * direction[1] - edge[1] * direction[0]
            if denominator != 0.0:
                numerator = edge[0] * (start[1] - previous[1]) - edge[1] * (
                    start[0] - previous[0]
                )
                result.append(previous + (numerator / denominator) * direction)
        if current_inside:
            result.append(current)
        previous = current
        previous_value = current_value
    return result


def _kernel_witness(points: np.ndarray, /) -> tuple[np.ndarray, float]:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    extent = max(float(np.max(upper - lower)), 1.0)
    lower = lower - extent
    upper = upper + extent
    kernel = [
        np.asarray((lower[0], lower[1])),
        np.asarray((upper[0], lower[1])),
        np.asarray((upper[0], upper[1])),
        np.asarray((lower[0], upper[1])),
    ]
    for index in range(points.shape[0]):
        kernel = _clip_half_plane(
            kernel, points[index], points[(index + 1) % points.shape[0]]
        )
        if not kernel:
            raise ValueError(
                "PolygonTriangulation.coordinates must be star-shaped; got an empty star kernel"
            )
    witness = np.mean(np.stack(kernel), axis=0)
    edge = np.roll(points, -1, axis=0) - points
    lengths = np.sqrt(np.sum(edge * edge, axis=1))
    margins = (
        np.asarray(
            [
                _orientation(points[i], points[(i + 1) % points.shape[0]], witness)
                for i in range(points.shape[0])
            ]
        )
        / lengths
    )
    area = 0.5 * _signed_area2(points)
    return witness, float(np.min(margins) / math.sqrt(area))


class PolygonAdmissibilityPolicy(StrictModule, NonTrainableState):
    minimum_star_margin: float = eqx.field(static=True)
    minimum_edge_ratio: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_star_margin: float = 1.0e-8,
        minimum_edge_ratio: float = 1.0e-10,
    ) -> None:
        star = float(minimum_star_margin)
        edge = float(minimum_edge_ratio)
        if star < 0.0 or edge < 0.0:
            raise ValueError(
                "PolygonAdmissibilityPolicy.thresholds must be nonnegative; got a negative threshold"
            )
        self.minimum_star_margin = star
        self.minimum_edge_ratio = edge
        self.policy_id = canonical_fingerprint(
            {"kind": "polygon-admissibility", "star": star, "edge": edge}
        )


class PolygonTriangulation(StrictModule, NonTrainableState):
    local_triangles: Array
    triangle_valid: Array
    witness_weights: Array
    star_margin: Array
    triangulation_id: str = eqx.field(static=True)


class PolygonGeometryEvidence(StrictModule):
    valid: Array
    minimum_edge_ratio: Array
    star_margin: Array
    area_partition_error: Array
    minimum_triangle_measure: Array

    @property
    def passed(self) -> Array:
        return jnp.all(self.valid)


class PolygonGeometry(StrictModule):
    vertices: Array
    edge_vectors: Array
    edge_lengths: Array
    outward_normals: Array
    areas: Array
    centroids: Array
    characteristic_lengths: Array
    diameters: Array
    triangle_points: Array
    triangle_measures: Array
    evidence: PolygonGeometryEvidence
    geometry_id: str = eqx.field(static=True)


class PolygonCubature(StrictModule):
    points: Array
    weights: Array
    degree: int = eqx.field(static=True)
    cubature_id: str = eqx.field(static=True)


def prepare_polygon_triangulation(
    coordinates: ArrayLike,
    cells: ArrayLike,
    /,
    *,
    policy: PolygonAdmissibilityPolicy | None = None,
) -> PolygonTriangulation:
    points = np.asarray(coordinates, dtype=np.float64)
    cells_ = np.asarray(cells, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "PolygonTriangulation.coordinates must have shape (vertices, 2); got incompatible coordinate shape"
        )
    if cells_.ndim != 2 or cells_.shape[1] < 3:
        raise ValueError(
            "PolygonTriangulation.cells must have shape (cells, arity >= 3); got incompatible connectivity shape"
        )
    selected = points[cells_]
    capacity = cells_.shape[1] - 2
    triangles = np.zeros((cells_.shape[0], capacity, 3), dtype=np.int32)
    valid = np.zeros((cells_.shape[0], capacity), dtype=np.bool_)
    witness_weights = np.zeros((cells_.shape[0], cells_.shape[1]), dtype=np.float64)
    star_margins = np.zeros((cells_.shape[0],), dtype=np.float64)
    policy_ = PolygonAdmissibilityPolicy() if policy is None else policy
    for cell in range(selected.shape[0]):
        polygon = selected[cell]
        _validate_simple_polygon(polygon)
        local_triangles = _ear_clip(polygon)
        for index, triangle in enumerate(local_triangles):
            triangles[cell, index] = triangle
            valid[cell, index] = True
        witness, star_margin = _kernel_witness(polygon)
        if star_margin < policy_.minimum_star_margin:
            raise ValueError(
                "PolygonTriangulation.coordinates must satisfy the star-kernel margin policy; got a margin below policy"
            )
        augmented = np.concatenate((polygon.T, np.ones((1, polygon.shape[0]))), axis=0)
        target = np.concatenate((witness, np.ones((1,))))
        weights = np.linalg.lstsq(augmented, target, rcond=None)[0]
        witness_weights[cell] = weights
        star_margins[cell] = star_margin
    return PolygonTriangulation(
        local_triangles=jnp.asarray(triangles),
        triangle_valid=jnp.asarray(valid),
        witness_weights=jnp.asarray(witness_weights),
        star_margin=jnp.asarray(star_margins),
        triangulation_id=canonical_fingerprint(
            {
                "kind": "polygon-triangulation",
                "cells": array_tree_fingerprint(cells_),
                "triangles": array_tree_fingerprint(triangles),
                "valid": array_tree_fingerprint(valid),
                "witness_weights": array_tree_fingerprint(witness_weights),
                "policy": policy_.policy_id,
            }
        ),
    )


def evaluate_polygon_geometry(
    coordinates: ArrayLike,
    cells: ArrayLike,
    triangulation: PolygonTriangulation,
    /,
    *,
    policy: PolygonAdmissibilityPolicy | None = None,
    geometry_id: str = "polygon-geometry",
) -> PolygonGeometry:
    points = jnp.asarray(coordinates)
    cells_ = jnp.asarray(cells, dtype=jnp.int32)
    vertices = points[cells_]
    following = jnp.roll(vertices, -1, axis=1)
    edge_vectors = following - vertices
    edge_lengths = jnp.sqrt(jnp.sum(edge_vectors * edge_vectors, axis=-1))
    area2 = jnp.sum(_cross_2d(vertices, following), axis=1)
    areas = 0.5 * area2
    safe_area2 = jnp.where(area2 != 0.0, area2, 1.0)
    centroid_factor = _cross_2d(vertices, following)[..., None]
    centroids = jnp.sum((vertices + following) * centroid_factor, axis=1) / (
        3.0 * safe_area2[:, None]
    )
    characteristic = jnp.sqrt(jnp.maximum(areas, jnp.finfo(points.dtype).tiny))
    safe_lengths = jnp.maximum(edge_lengths, jnp.finfo(points.dtype).tiny)
    outward_normals = (
        jnp.stack((edge_vectors[..., 1], -edge_vectors[..., 0]), axis=-1)
        / safe_lengths[..., None]
    )
    pair_difference = vertices[:, :, None, :] - vertices[:, None, :, :]
    diameters = jnp.sqrt(
        jnp.max(jnp.sum(pair_difference * pair_difference, axis=-1), axis=(1, 2))
    )
    local_triangles = triangulation.local_triangles
    safe_triangles = jnp.where(
        triangulation.triangle_valid[..., None], local_triangles, 0
    )
    cell_indices = jnp.arange(vertices.shape[0])[:, None, None]
    triangle_points = vertices[cell_indices, safe_triangles]
    first = triangle_points[:, :, 1] - triangle_points[:, :, 0]
    second = triangle_points[:, :, 2] - triangle_points[:, :, 0]
    triangle_measures = 0.5 * _cross_2d(first, second)
    triangle_measures = jnp.where(triangulation.triangle_valid, triangle_measures, 0.0)
    witness = ein.contract("cv,cvd->cd", triangulation.witness_weights, vertices)
    witness_delta = witness[:, None, :] - vertices
    inward = _cross_2d(edge_vectors, witness_delta) / safe_lengths
    runtime_star_margin = jnp.min(inward, axis=1) / characteristic
    edge_ratio = jnp.min(edge_lengths, axis=1) / jnp.maximum(
        diameters, jnp.finfo(points.dtype).tiny
    )
    area_error = jnp.abs(jnp.sum(triangle_measures, axis=1) - areas)
    active_triangle_measure = jnp.where(
        triangulation.triangle_valid,
        triangle_measures,
        jnp.asarray(jnp.inf, dtype=points.dtype),
    )
    minimum_triangle = jnp.min(active_triangle_measure, axis=1)
    policy_ = PolygonAdmissibilityPolicy() if policy is None else policy
    tolerance = 512.0 * jnp.finfo(points.dtype).eps * jnp.maximum(areas, 1.0)
    valid = (
        (areas > 0.0)
        & jnp.all(edge_lengths > 0.0, axis=1)
        & (minimum_triangle > 0.0)
        & (area_error <= tolerance)
        & (runtime_star_margin >= policy_.minimum_star_margin)
        & (edge_ratio >= policy_.minimum_edge_ratio)
    )
    evidence = PolygonGeometryEvidence(
        valid=valid,
        minimum_edge_ratio=edge_ratio,
        star_margin=runtime_star_margin,
        area_partition_error=area_error,
        minimum_triangle_measure=minimum_triangle,
    )
    return PolygonGeometry(
        vertices=vertices,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        outward_normals=outward_normals,
        areas=areas,
        centroids=centroids,
        characteristic_lengths=characteristic,
        diameters=diameters,
        triangle_points=triangle_points,
        triangle_measures=triangle_measures,
        evidence=evidence,
        geometry_id=canonical_fingerprint(
            {
                "kind": str(geometry_id),
                "triangulation": triangulation.triangulation_id,
                "coordinate_shape": list(points.shape),
                "coordinate_dtype": str(points.dtype),
            }
        ),
    )


def polygon_cubature(
    geometry: PolygonGeometry,
    triangulation: PolygonTriangulation,
    degree: int,
    /,
) -> PolygonCubature:

    degree_ = int(degree)
    if degree_ < 0:
        raise ValueError(
            f"PolygonTriangulation.cubature_degree must be nonnegative; got {degree}"
        )
    order = max(2, (degree_ + 3) // 2)
    data = imported_integration.reference_rule_data(
        imported_integration.ReferenceTriangleRule(
            imported_integration.GaussLegendreRule(order)
        )
    )
    reference = jnp.asarray(data.points)
    reference_weights = jnp.asarray(data.weights)
    triangles = geometry.triangle_points
    first = triangles[:, :, 0]
    axis_one = triangles[:, :, 1] - first
    axis_two = triangles[:, :, 2] - first
    points = (
        first[:, :, None, :]
        + reference[None, None, :, 0, None] * axis_one[:, :, None, :]
        + reference[None, None, :, 1, None] * axis_two[:, :, None, :]
    )
    jacobian = 2.0 * geometry.triangle_measures
    weights = jacobian[:, :, None] * reference_weights[None, None, :]
    weights = jnp.where(triangulation.triangle_valid[..., None], weights, 0.0)
    return PolygonCubature(
        points=points.reshape((points.shape[0], -1, 2)),
        weights=weights.reshape((weights.shape[0], -1)),
        degree=degree_,
        cubature_id=canonical_fingerprint(
            {
                "kind": "polygon-cubature",
                "geometry": geometry.geometry_id,
                "degree": degree_,
                "rule": type(data).__name__,
                "point_count": points.shape[1] * points.shape[2],
            }
        ),
    )


__all__ = [
    "PolygonAdmissibilityPolicy",
    "PolygonCubature",
    "PolygonGeometry",
    "PolygonGeometryEvidence",
    "PolygonTriangulation",
    "PolyhedralFaceTriangulation",
    "evaluate_polygon_geometry",
    "polygon_cubature",
    "prepare_polygon_triangulation",
    "prepare_polyhedral_face_triangulation",
]
