#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Source-stratified native CDT images with explicit quotient construction witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import ceil, floor

import numpy as np

from ..._meshcore import (
    constrained_delaunay_2d,
    MeshcoreStatus,
    NativeResourceFailure,
    PlanarExecutionEvidence,
)
from ...discretization import CellMesh, PeriodicCell
from ...geometry._mesh_certificates import _dyadic_integers
from .._contracts import MeshingFailure, MeshingFailureCategory, MeshingLimits
from .._controls import ProtectedFeature
from .._periodic import (
    _edge_key,
    _interiors_overlap,
    _quotient_simplices,
    _require_periodic_topology,
    _simplex_entity_corners,
)


@dataclass(frozen=True, slots=True)
class PeriodicSourceStrata:
    points: np.ndarray
    cells: np.ndarray
    shifts: np.ndarray
    regions: np.ndarray
    edges: np.ndarray
    cell: PeriodicCell
    block_name: str
    protected_vertices: np.ndarray


def prepare_periodic_source_strata(
    mesh: CellMesh, regions: np.ndarray, features: tuple[ProtectedFeature, ...], /
) -> PeriodicSourceStrata:
    """Protect declared source edges and every material-interface edge orbit."""
    points, cells, shifts, cell, name = _quotient_simplices(mesh)
    if mesh.topological_dimension != 2:
        raise ValueError("Source-stratified periodic CDT requires triangles.")
    periodic = _require_periodic_topology(mesh)
    roots = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    _, local = np.unique(roots, return_inverse=True)
    rows = _simplex_entity_corners(mesh, 1)
    images = np.asarray(periodic.vertex_shifts, dtype=np.int64)
    keys = [
        _edge_key(int(local[first]), images[first], int(local[second]), images[second])[0]
        for first, second in rows
    ]
    selected: set[tuple[int, ...]] = set()
    orbit = np.asarray(periodic.orbits(1)[0])
    protected_vertices = np.zeros(points.shape[0], dtype=np.bool_)
    for feature in features:
        if feature.scope.entity_dimension == 1:
            selected.update(
                key
                for key, member in zip(keys, orbit, strict=True)
                if member in np.asarray(feature.scope.entity_ids)
            )
        elif feature.scope.entity_dimension == 0:
            leaders = np.asarray(periodic.orbit_representatives(0), dtype=np.int64)
            source_rows = leaders[np.asarray(feature.scope.entity_ids, dtype=np.int64)]
            protected_vertices[local[source_rows]] = True
    material: dict[tuple[int, ...], set[int]] = {}
    for vertices, corner_shifts, region in zip(cells, shifts, regions, strict=True):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            key = _edge_key(
                int(vertices[first]),
                corner_shifts[first],
                int(vertices[second]),
                corner_shifts[second],
            )[0]
            material.setdefault(key, set()).add(int(region))
    selected.update(key for key, members in material.items() if len(members) > 1)
    edges = np.asarray(sorted(selected), dtype=np.int64).reshape((-1, 4))
    return PeriodicSourceStrata(
        points, cells, shifts, regions, edges, cell, name, protected_vertices
    )


def periodic_source_seed_points(
    strata: PeriodicSourceStrata,
    target_size: float,
    minimum_size: float,
    maximum_vertices: int,
    /,
) -> np.ndarray:
    """Subdivide source edge orbits without moving any authored source vertex."""
    vectors = np.asarray(strata.cell.vectors, dtype=np.float64)
    points = list(strata.points)
    edges = {tuple(int(value) for value in edge) for edge in strata.edges}
    for first, second, dx, dy in sorted(edges):
        start = strata.points[first]
        stop = strata.points[second] + np.asarray((dx, dy), dtype=np.int64) @ vectors
        length = float(np.linalg.norm(stop - start))
        pieces = 1
        while length / pieces > target_size and length / (2 * pieces) >= minimum_size:
            pieces *= 2
            if len(points) + pieces - 1 > maximum_vertices:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Periodic source-feature subdivision exceeds its vertex budget.",
                )
        for numerator in range(1, pieces):
            parameter = numerator / pieces
            points.append((1.0 - parameter) * start + parameter * stop)
    return np.asarray(points, dtype=np.float64)


def _rational_dual(vectors: np.ndarray, /) -> tuple[tuple[Fraction, Fraction], ...]:
    a, b = (Fraction(float(value)) for value in vectors[0])
    c, d = (Fraction(float(value)) for value in vectors[1])
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("Periodic source lattice is singular.")
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def _fractional(
    point: tuple[Fraction, Fraction], dual: tuple[tuple[Fraction, Fraction], ...], /
) -> tuple[Fraction, Fraction]:
    return (
        point[0] * dual[0][0] + point[1] * dual[1][0],
        point[0] * dual[0][1] + point[1] * dual[1][1],
    )


class PeriodicConstructionImageBudgetError(MeshingFailure):
    """An individual image neighborhood has a proved unadmitted cardinality."""

    def __init__(self, maximum: int, required: int, /) -> None:
        self.work_units = 0
        super().__init__(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Periodic constrained construction exceeds its image budget.",
            requested=(("maximum_images", maximum),),
            achieved=(("required_images_lower_bound", required),),
        )


def _constraint_images(
    points: np.ndarray,
    strata: PeriodicSourceStrata,
    margin: Fraction,
    maximum_images: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Include every site and constraint intersecting the exact fractional box."""
    vectors = np.asarray(strata.cell.vectors, dtype=np.float64)
    dual = _rational_dual(vectors)
    fractional = [
        _fractional((Fraction(float(point[0])), Fraction(float(point[1]))), dual)
        for point in points
    ]
    lower, upper = -margin, 1 + margin
    sites: list[tuple[int, int, int]] = []
    lookup: dict[tuple[int, int, int], int] = {}

    def add(key: tuple[int, int, int], /) -> int:
        if key not in lookup:
            if len(sites) >= maximum_images:
                raise PeriodicConstructionImageBudgetError(maximum_images, len(sites) + 1)
            lookup[key] = len(sites)
            sites.append(key)
        return lookup[key]

    for root, point in enumerate(fractional):
        bounds = tuple(
            range(ceil(lower - value), floor(upper - value) + 1) for value in point
        )
        count = len(bounds[0]) * len(bounds[1])
        if len(sites) + count > maximum_images:
            raise PeriodicConstructionImageBudgetError(maximum_images, len(sites) + count)
        for x, y in product(bounds[0], bounds[1]):
            add((root, x, y))
    segments: list[tuple[int, int]] = []
    for first, second, dx, dy in strata.edges:
        first_, second_ = int(first), int(second)
        displacement = (int(dx), int(dy))
        a = fractional[first_]
        b = tuple(
            value + shift
            for value, shift in zip(fractional[second_], displacement, strict=True)
        )
        bounds = tuple(
            range(ceil(lower - max(x, y)), floor(upper - min(x, y)) + 1)
            for x, y in zip(a, b, strict=True)
        )
        if len(bounds[0]) * len(bounds[1]) > maximum_images:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic constraint images exceed their bounded neighborhood.",
            )
        for x, y in product(*bounds):
            segments.append(
                (
                    add((first_, x, y)),
                    add((second_, x + displacement[0], y + displacement[1])),
                )
            )
    packed = np.asarray(sites, dtype=np.int64)
    roots, shifts = packed[:, 0], packed[:, 1:]
    lifted = points[roots] + shifts @ vectors
    return lifted, roots, shifts, np.asarray(segments, dtype=np.int32).reshape((-1, 2))


def _quotient_image_cells(
    points: np.ndarray,
    native_cells: np.ndarray,
    roots: np.ndarray,
    images: np.ndarray,
    vectors: np.ndarray,
    margin: Fraction,
    /,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Select one local lift and prove its exact circumball clears the image box."""
    dual = _rational_dual(vectors)
    point_values = [
        (Fraction(float(point[0])), Fraction(float(point[1]))) for point in points
    ]
    lattice = [
        (Fraction(float(vector[0])), Fraction(float(vector[1]))) for vector in vectors
    ]
    cells: list[np.ndarray] = []
    shifts: list[np.ndarray] = []
    complete = True
    for native in native_cells:
        vertices, exponents = roots[native], images[native]
        corners = [
            (
                point_values[root][0]
                + sum(
                    Fraction(int(shift)) * lattice[row][0]
                    for row, shift in enumerate(image)
                ),
                point_values[root][1]
                + sum(
                    Fraction(int(shift)) * lattice[row][1]
                    for row, shift in enumerate(image)
                ),
            )
            for root, image in zip(vertices, exponents, strict=True)
        ]
        center_of_mass = (
            sum(point[0] for point in corners) / 3,
            sum(point[1] for point in corners) / 3,
        )
        fractional_center = _fractional(center_of_mass, dual)
        if any(value < 0 or value >= 1 for value in fractional_center):
            continue
        a, b, c = corners
        u, v = tuple(b[i] - a[i] for i in range(2)), tuple(c[i] - a[i] for i in range(2))
        determinant = u[0] * v[1] - u[1] * v[0]
        if determinant <= 0:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "Native constrained cell lacks a positive scientific quotient lift.",
            )
        first, second = (
            sum(value * value for value in u) / 2,
            sum(value * value for value in v) / 2,
        )
        offset = (
            (first * v[1] - second * u[1]) / determinant,
            (u[0] * second - v[0] * first) / determinant,
        )
        circle = _fractional((a[0] + offset[0], a[1] + offset[1]), dual)
        radius_squared = sum(value * value for value in offset)
        for axis, coordinate in enumerate(circle):
            support_squared = radius_squared * sum(
                dual[row][axis] ** 2 for row in range(2)
            )
            distance = min(coordinate + margin, 1 + margin - coordinate)
            complete &= distance > 0 and distance * distance > support_squared
        cells.append(vertices)
        shifts.append(exponents)
    if not cells:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Constrained periodic construction has no fundamental-cell representative.",
        )
    return np.asarray(cells, dtype=np.int64), np.asarray(shifts, dtype=np.int64), complete


@dataclass(frozen=True, slots=True)
class PeriodicStratifiedTriangulation:
    simplices: np.ndarray
    simplex_shifts: np.ndarray
    image_count: int
    work_units: int
    protected_vertices: np.ndarray
    protected_edges: tuple[tuple[int, ...], ...]


def bind_periodic_source_strata(
    points: np.ndarray,
    cells: np.ndarray,
    shifts: np.ndarray,
    strata: PeriodicSourceStrata,
    maximum_work: int,
    /,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...], int]:
    """Exact positive-area overlaps bind every target to its authored material stratum."""
    vectors = np.asarray(strata.cell.vectors, dtype=np.float64)
    integers, _ = _dyadic_integers(np.concatenate((points, strata.points, vectors)))
    point_count = points.shape[0]
    first = integers[:point_count]
    original = integers[point_count:-2]
    lattice = integers[-2:]
    target = first[cells] + shifts @ lattice
    source = original[strata.cells] + strata.shifts @ lattice
    dual = _rational_dual(vectors)
    actual_target = points[cells] + shifts @ vectors
    actual_source = strata.points[strata.cells] + strata.shifts @ vectors
    fractional_target = actual_target @ np.asarray(
        [[float(value) for value in row] for row in dual]
    )
    fractional_source = actual_source @ np.asarray(
        [[float(value) for value in row] for row in dual]
    )
    ancestry: list[tuple[int, ...]] = []
    regions = []
    work = 0
    for triangle, fractional in zip(target, fractional_target, strict=True):
        members: set[int] = set()
        for parent, (source_triangle, source_fractional) in enumerate(
            zip(source, fractional_source, strict=True)
        ):
            lower = (
                np.floor(
                    np.min(fractional, axis=0) - np.max(source_fractional, axis=0)
                ).astype(np.int64)
                - 1
            )
            upper = (
                np.ceil(
                    np.max(fractional, axis=0) - np.min(source_fractional, axis=0)
                ).astype(np.int64)
                + 1
            )
            for image in product(
                *(range(int(a), int(b) + 1) for a, b in zip(lower, upper, strict=True))
            ):
                work += 1
                if work > maximum_work:
                    raise MeshingFailure(
                        MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        "Periodic source-stratum certification exceeds its work budget.",
                    )
                if _interiors_overlap(
                    triangle, source_triangle + np.asarray(image, dtype=object) @ lattice
                ):
                    members.add(parent)
        material = np.unique(strata.regions[sorted(members)])
        if material.size != 1:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A constrained periodic cell crosses or loses an authored material stratum.",
            )
        regions.append(material[0])
        ancestry.append(tuple(sorted(members)))
    return np.asarray(regions, dtype=np.int64), tuple(ancestry), work


def triangulate_periodic_source_strata(
    points: np.ndarray,
    strata: PeriodicSourceStrata,
    limits: MeshingLimits,
    maximum_images: int,
    maximum_work: int,
    /,
) -> PeriodicStratifiedTriangulation:
    """Native constrained triangulation; no scientific identity is reconstructed from coordinates."""
    vectors = np.asarray(strata.cell.vectors, dtype=np.float64)
    margin, work = Fraction(1, 4), 0
    while True:
        if work >= maximum_work:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic constrained construction exhausts its shared work budget.",
            )
        try:
            lifted, roots, images, segments = _constraint_images(
                points, strata, margin, maximum_images
            )
        except PeriodicConstructionImageBudgetError as error:
            error.work_units = work
            raise
        evidence: list[PlanarExecutionEvidence] = []
        try:
            _, native_cells, vertex_map, cell_segments, status = constrained_delaunay_2d(
                lifted,
                segments,
                keep_convex_hull=True,
                max_steiner=0,
                max_triangles=limits.maximum_cells,
                max_cavity_cells=limits.maximum_cavity_cells,
                max_work=maximum_work - work,
                max_scratch_bytes=limits.maximum_scratch_bytes,
                record_native_resources=evidence.append,
            )
        except NativeResourceFailure as error:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED, str(error)
            ) from error
        work += int(evidence[0].work_evidence[0])
        if status is not MeshcoreStatus.OK or not np.array_equal(
            np.sort(vertex_map), np.arange(roots.size)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "Native constrained source images lack a complete input-identity witness.",
            )
        inverse = np.empty_like(vertex_map)
        inverse[vertex_map] = np.arange(vertex_map.size)
        cells, shifts, complete = _quotient_image_cells(
            points,
            inverse[native_cells],
            roots,
            images,
            vectors,
            margin,
        )
        work += native_cells.shape[0]
        if work > maximum_work:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic constrained image certification exceeds its work budget.",
            )
        if complete:
            protected = np.zeros(points.shape[0], dtype=np.bool_)
            protected[: strata.points.shape[0]] = strata.protected_vertices
            protected_edges: set[tuple[int, ...]] = set()
            for opposite in range(3):
                rows = native_cells[cell_segments[:, opposite] >= 0]
                corners = ((opposite + 1) % 3, (opposite + 2) % 3)
                protected[roots[inverse[rows[:, corners]]]] = True
                for row in inverse[rows]:
                    first, second = row[corners[0]], row[corners[1]]
                    protected_edges.add(
                        _edge_key(
                            int(roots[first]),
                            images[first],
                            int(roots[second]),
                            images[second],
                        )[0]
                    )
            return PeriodicStratifiedTriangulation(
                cells,
                shifts,
                roots.size,
                work,
                protected,
                tuple(sorted(protected_edges)),
            )
        margin *= 2
