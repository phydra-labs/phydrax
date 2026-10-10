#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent degree-one tubular projection proof for qualified analytic sources.

The actual affine boundary must be a closed oriented two-manifold. Every face
is covered by an exhaustive dyadic reference-triangle worklist; outward affine
coordinate boxes and source-expression intervals establish the complete tube
and strictly positive normal derivative premises. No rounded subdivision mesh
is substituted for the original coordinate map.

For the nominal source's positive reach, nearest projection on this tube is
well-defined. Positive face normals, manifold vertex links and source topology
establish an orientation-preserving local cover. An exact rational regular
normal fiber with one interior crossing establishes degree one, hence every
source point has one preimage. The complete reverse distance bound is therefore
the complete forward bound, not a sampled-nearest-point estimate. Volume cell
validity and global interior embedding remain separate mandatory certificates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from ...discretization import CellGeometrySpec, CellMesh, TetrahedralConnectivity
from ...discretization._coordinate_enclosure import (
    coordinate_corner_images,
    coordinate_enclosure_budget,
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    prepared_coordinate_source_bank,
    RationalPolynomial,
    source_expressions,
)
from .._mesh_certificates import (
    _coordinate_scope,
    ImplicitProjectionCoverageEvidence,
    MeshCertificateBinding,
    MeshCertificateFinding,
    MeshCertificateLimits,
    MeshFindingStatus,
    ProjectionFiberCrossingKind,
)
from ._analytic_profile import AnalyticImplicitProfile


type _RationalPoint = tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class _Boundary:
    triangles: np.ndarray
    local_vertices: np.ndarray
    global_vertices: np.ndarray
    identifiers: np.ndarray
    components: np.ndarray
    findings: tuple[MeshCertificateFinding, ...]
    exact_triangles: (
        tuple[tuple[_RationalPoint, _RationalPoint, _RationalPoint], ...] | None
    ) = None
    coordinate_bounds: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _Pieces:
    triangles: np.ndarray
    parameters: np.ndarray
    coordinates: np.ndarray
    owners: np.ndarray
    fields: np.ndarray
    gradients: np.ndarray
    normals: np.ndarray
    upper: float
    lower: float
    queries: int
    findings: tuple[MeshCertificateFinding, ...]


@dataclass(frozen=True, slots=True)
class _Fiber:
    endpoints: tuple[tuple[tuple[int, int], ...], ...]
    values: np.ndarray
    gradient: np.ndarray
    candidates: np.ndarray
    classes: tuple[ProjectionFiberCrossingKind, ...]
    parameters: tuple[tuple[int, int] | None, ...]
    queries: int
    tests: int
    findings: tuple[MeshCertificateFinding, ...]


def _product_bounds(
    a_lower: np.ndarray,
    a_upper: np.ndarray,
    b_lower: np.ndarray,
    b_upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    products = np.stack(
        (a_lower * b_lower, a_lower * b_upper, a_upper * b_lower, a_upper * b_upper)
    )
    return (
        np.nextafter(np.min(products, axis=0), -np.inf),
        np.nextafter(np.max(products, axis=0), np.inf),
    )


def _normal_bounds(
    triangles: np.ndarray,
    coordinate_bounds: np.ndarray | None = None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lower = triangles if coordinate_bounds is None else coordinate_bounds[:, 0]
    upper = triangles if coordinate_bounds is None else coordinate_bounds[:, 1]
    low = np.nextafter(lower[:, 1:] - upper[:, :1], -np.inf)
    high = np.nextafter(upper[:, 1:] - lower[:, :1], np.inf)
    normal_low = np.empty((triangles.shape[0], 3), dtype=np.float64)
    normal_high = np.empty_like(normal_low)
    for axis in range(3):
        first, second = (axis + 1) % 3, (axis + 2) % 3
        positive_low, positive_high = _product_bounds(
            low[:, 0, first], high[:, 0, first], low[:, 1, second], high[:, 1, second]
        )
        negative_low, negative_high = _product_bounds(
            low[:, 0, second], high[:, 0, second], low[:, 1, first], high[:, 1, first]
        )
        normal_low[:, axis] = np.nextafter(positive_low - negative_high, -np.inf)
        normal_high[:, axis] = np.nextafter(positive_high - negative_low, np.inf)
    return normal_low, normal_high


def _dot_bounds(
    normal_low: np.ndarray,
    normal_high: np.ndarray,
    gradient_low: np.ndarray,
    gradient_high: np.ndarray,
    /,
) -> np.ndarray:
    lower = np.zeros(normal_low.shape[0], dtype=np.float64)
    upper = np.zeros_like(lower)
    for axis in range(3):
        low, high = _product_bounds(
            normal_low[:, axis],
            normal_high[:, axis],
            gradient_low[:, axis],
            gradient_high[:, axis],
        )
        lower = np.nextafter(lower + low, -np.inf)
        upper = np.nextafter(upper + high, np.inf)
    return np.stack((lower, upper), axis=1)


def _manifold_components(
    vertices: np.ndarray, identifiers: np.ndarray, profile: AnalyticImplicitProfile, /
) -> tuple[np.ndarray, tuple[MeshCertificateFinding, ...]]:
    """Establish the closed-manifold premise, not a source-distance violation."""
    count = vertices.shape[0]
    components = np.full(count, -1, dtype=np.int64)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    links: dict[int, list[tuple[int, int]]] = {}
    for row, face in enumerate(vertices.tolist()):
        for local in range(3):
            first, second = face[local], face[(local + 1) % 3]
            key = (min(first, second), max(first, second))
            edges.setdefault(key, []).append((row, 1 if first < second else -1))
            links.setdefault(face[local], []).append(
                (face[(local + 1) % 3], face[(local + 2) % 3])
            )
    bad = {
        row
        for incidences in edges.values()
        if len(incidences) != 2 or sum(sign for _, sign in incidences) != 0
        for row, _ in incidences
    }
    findings: list[MeshCertificateFinding] = []
    if bad:
        findings.append(
            MeshCertificateFinding(
                "projection_closed_oriented_edges",
                "unresolved",
                "facet",
                tuple(int(identifiers[row]) for row in sorted(bad)),
            )
        )
        return components, tuple(findings)
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for incidences in edges.values():
        first, second = incidences[0][0], incidences[1][0]
        adjacency[first].append(second)
        adjacency[second].append(first)
    component = 0
    for seed in range(count):
        if components[seed] >= 0:
            continue
        pending = [seed]
        components[seed] = component
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if components[neighbor] < 0:
                    components[neighbor] = component
                    pending.append(neighbor)
        component += 1
    for vertex, pairs in links.items():
        graph: dict[int, list[int]] = {}
        for first, second in pairs:
            graph.setdefault(first, []).append(second)
            graph.setdefault(second, []).append(first)
        seen: set[int] = set()
        pending = [next(iter(graph))]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(graph[current])
        if len(seen) != len(graph) or any(
            len(neighbors) != 2 for neighbors in graph.values()
        ):
            affected = np.flatnonzero(np.any(vertices == vertex, axis=1))
            findings.append(
                MeshCertificateFinding(
                    "projection_manifold_vertex_link",
                    "unresolved",
                    "facet",
                    tuple(int(value) for value in identifiers[affected]),
                )
            )
    euler = len(links) - len(edges) + count
    if (
        component != profile.component_count
        or euler != profile.boundary_euler_characteristic
    ):
        findings.append(
            MeshCertificateFinding("projection_source_topology", "unresolved", "mesh")
        )
    return components, tuple(findings)


def _boundary(mesh: CellMesh, profile: AnalyticImplicitProfile, /) -> _Boundary:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if mesh.topological_dimension == 3 and isinstance(
        mesh.connectivity, TetrahedralConnectivity
    ):
        connectivity = mesh.connectivity
        selected = np.flatnonzero(np.asarray(connectivity.boundary_faces))
        vertices = np.asarray(connectivity.faces, dtype=np.int64)[selected].copy()
        orientation = np.zeros(connectivity.faces.shape[0], dtype=np.float64)
        np.add.at(
            orientation,
            np.asarray(connectivity.cell_faces).reshape((-1,)),
            np.asarray(connectivity.cell_face_signs).reshape((-1,)),
        )
        reversed_ = orientation[selected] < 0.0
        vertices[reversed_] = vertices[reversed_][:, (0, 2, 1)]
        ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)[selected]
    elif mesh.topological_dimension == 2 and all(
        block.cell_kind == "triangle" for block in mesh.blocks
    ):
        vertices = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
    else:
        return _Boundary(
            np.empty((0, 3, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            (
                MeshCertificateFinding(
                    "projection_affine_triangle_carrier", "unresolved", "mesh"
                ),
            ),
        )
    if not vertices.size:
        return _Boundary(
            np.empty((0, 3, 3), dtype=np.float64),
            vertices,
            vertices,
            ids,
            np.empty((0,), dtype=np.int64),
            (MeshCertificateFinding("projection_empty_boundary", "unresolved", "mesh"),),
        )
    components, findings = _manifold_components(vertices, ids, profile)
    return _Boundary(
        points[vertices],
        vertices,
        np.asarray(mesh.vertex_global_ids, dtype=np.int64)[vertices],
        ids,
        components,
        findings,
    )


def _source_affine_boundary(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    boundary: _Boundary,
    budget: CoordinateEnclosureBudget,
    /,
) -> _Boundary:
    """Prove full source-affine laws and retain their exact, nonbinary corners."""
    bank = prepared_coordinate_source_bank(geometry)
    elements, routes, _ = geometry._resolve(mesh, exact_source_prepared=True)
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    budget.reserve(0, 8 * carrier.shape[0] + 144 * boundary.triangles.shape[0])
    vertices: list[_RationalPoint | None] = [None] * carrier.shape[0]
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        expressions = source_expressions(element)
        if (
            expressions is None
            or len(expressions) != 3
            or any(
                isinstance(value, RationalPolynomial)
                or any(sum(exponent) > 1 for exponent in value)
                for value in expressions
            )
        ):
            return replace(
                boundary,
                findings=(
                    *boundary.findings,
                    MeshCertificateFinding(
                        "projection_actual_affine_maps",
                        "unresolved",
                        "mesh",
                    ),
                ),
            )
        for row, corners in zip(
            np.asarray(route), np.asarray(block.vertices), strict=True
        ):
            local = tuple(bank[int(index)] for index in row if index >= 0)
            images = coordinate_corner_images(element, local)
            if images is None:
                return replace(
                    boundary,
                    findings=(
                        *boundary.findings,
                        MeshCertificateFinding(
                            "projection_actual_affine_maps",
                            "unresolved",
                            "mesh",
                        ),
                    ),
                )
            actual = corners[corners >= 0]
            if len(images) != actual.size:
                raise ValueError(
                    "Projection source corners do not match their actual SCI cell."
                )
            for vertex, image in zip(actual, images, strict=True):
                budget.reserve(3)
                point: _RationalPoint = (image[0], image[1], image[2])
                rounded = np.asarray(
                    tuple(float(value) for value in point), dtype=np.float64
                )
                if not np.array_equal(
                    rounded.view(np.uint64), carrier[vertex].view(np.uint64)
                ):
                    raise ValueError(
                        "Projection carrier is not the RNE image of its actual source law."
                    )
                previous = vertices[int(vertex)]
                if previous is not None and previous != point:
                    return replace(
                        boundary,
                        findings=(
                            *boundary.findings,
                            MeshCertificateFinding(
                                "projection_source_vertex_continuity",
                                "unresolved",
                                "mesh",
                            ),
                        ),
                    )
                vertices[int(vertex)] = point
    triangles = []
    bounds = np.empty((boundary.local_vertices.shape[0], 2, 3, 3), dtype=np.float64)
    for row, face in enumerate(boundary.local_vertices):
        points = tuple(vertices[int(vertex)] for vertex in face)
        if any(point is None for point in points):
            raise ValueError("Projection boundary has an uncovered source vertex.")
        first, second, third = points
        if first is None or second is None or third is None:
            raise ValueError("Projection boundary has an uncovered source vertex.")
        triangle = (first, second, third)
        triangles.append(triangle)
        for corner, point in enumerate(triangle):
            for axis, value in enumerate(point):
                bounds[row, :, corner, axis] = _fraction_interval(value)
    exact = tuple(triangles)
    budget.retain_basis(exact)
    return replace(boundary, exact_triangles=exact, coordinate_bounds=bounds)


def _affine_piece_boxes(
    facets: np.ndarray,
    owners: np.ndarray,
    parameters: np.ndarray,
    coordinate_bounds: np.ndarray | None = None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = facets[owners]
    original_low = original if coordinate_bounds is None else coordinate_bounds[owners, 0]
    original_high = (
        original if coordinate_bounds is None else coordinate_bounds[owners, 1]
    )
    low = np.zeros((owners.size, 3, 3), dtype=np.float64)
    high = np.zeros_like(low)
    representative = np.zeros_like(low)
    for vertex in range(3):
        weight = parameters[:, :, vertex, None]
        source = original[:, None, vertex, :]
        product_low = np.nextafter(weight * original_low[:, None, vertex, :], -np.inf)
        product_high = np.nextafter(weight * original_high[:, None, vertex, :], np.inf)
        low = np.nextafter(low + product_low, -np.inf)
        high = np.nextafter(high + product_high, np.inf)
        representative += weight * source
    coordinates = np.stack((np.min(low, axis=1), np.max(high, axis=1)), axis=1)
    return representative, coordinates[:, 0], coordinates[:, 1]


def _split_reference_triangles(parameters: np.ndarray, /) -> np.ndarray:
    first, second, third = parameters[:, 0], parameters[:, 1], parameters[:, 2]
    ab, bc, ca = 0.5 * (first + second), 0.5 * (second + third), 0.5 * (third + first)
    # Dyadic barycentric midpoints are exact through the bounded subdivision
    # depth; only their affine physical images need outward enclosures.
    return np.stack(
        (
            np.stack((first, ab, ca), axis=1),
            np.stack((ab, second, bc), axis=1),
            np.stack((ca, bc, third), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ),
        axis=1,
    ).reshape((-1, 3, 3))


def _pieces(
    boundary: _Boundary,
    profile: AnalyticImplicitProfile,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
) -> _Pieces:
    owners = np.arange(boundary.triangles.shape[0], dtype=np.int64)
    parameters = np.tile(np.eye(3, dtype=np.float64), (owners.size, 1, 1))
    depth = np.zeros(owners.size, dtype=np.int64)
    normal_low, normal_high = _normal_bounds(
        boundary.triangles, boundary.coordinate_bounds
    )
    terminals: list[tuple[np.ndarray, ...]] = []
    findings: list[MeshCertificateFinding] = []
    queries = 1  # Nominal profile reconstruction establishes its interior witness.
    created = owners.size
    while owners.size:
        available = (
            min(limits.maximum_source_samples, limits.maximum_distance_evaluations)
            - queries
        )
        if owners.size > available:
            triangles, low, high = _affine_piece_boxes(
                boundary.triangles,
                owners,
                parameters,
                boundary.coordinate_bounds,
            )
            terminals.append(
                (
                    triangles,
                    parameters,
                    np.stack((low, high), axis=1),
                    owners,
                    np.tile(
                        np.asarray((-np.inf, np.inf), dtype=np.float64), (owners.size, 1)
                    ),
                    np.stack(
                        (
                            np.full((owners.size, 3), -np.inf, dtype=np.float64),
                            np.full((owners.size, 3), np.inf, dtype=np.float64),
                        ),
                        axis=1,
                    ),
                    np.tile(
                        np.asarray((-np.inf, np.inf), dtype=np.float64), (owners.size, 1)
                    ),
                )
            )
            findings.append(
                MeshCertificateFinding(
                    "projection_source_query_capacity", "unresolved", "mesh"
                )
            )
            break
        triangles, low, high = _affine_piece_boxes(
            boundary.triangles,
            owners,
            parameters,
            boundary.coordinate_bounds,
        )
        source = profile.field_bounds.boxes(low, high)
        queries += owners.size
        fields = np.stack((source.value_lower, source.value_upper), axis=1)
        gradients = np.stack((source.gradient_lower, source.gradient_upper), axis=1)
        normals = _dot_bounds(
            normal_low[owners],
            normal_high[owners],
            source.gradient_lower,
            source.gradient_upper,
        )
        deviation = np.maximum(np.abs(fields[:, 0]), np.abs(fields[:, 1]))
        good = (
            (deviation <= tolerance)
            & (deviation <= profile.tube_radius)
            & (normals[:, 0] > 0.0)
        )
        violated = (
            (normals[:, 1] <= 0.0)
            | (fields[:, 0] > tolerance)
            | (fields[:, 1] < -tolerance)
        )
        stopped = good | violated | (depth >= limits.maximum_subdivision_depth)
        refusals: tuple[tuple[np.ndarray, str, MeshFindingStatus], ...] = (
            (violated, "projection_source_orientation_or_distance", "violated"),
            (stopped & ~good & ~violated, "projection_source_piece_depth", "unresolved"),
        )
        for mask, name, status in refusals:
            if np.any(mask):
                findings.append(
                    MeshCertificateFinding(
                        name,
                        status,
                        "facet",
                        tuple(
                            int(value)
                            for value in boundary.identifiers[np.unique(owners[mask])]
                        ),
                    )
                )
        if np.any(stopped):
            terminals.append(
                (
                    triangles[stopped],
                    parameters[stopped],
                    np.stack((low[stopped], high[stopped]), axis=1),
                    owners[stopped],
                    fields[stopped],
                    gradients[stopped],
                    normals[stopped],
                )
            )
        selected = ~stopped
        if not np.any(selected):
            break
        if (
            created + 4 * int(np.count_nonzero(selected))
            > limits.maximum_subdivision_pieces
        ):
            terminals.append(
                (
                    triangles[selected],
                    parameters[selected],
                    np.stack((low[selected], high[selected]), axis=1),
                    owners[selected],
                    fields[selected],
                    gradients[selected],
                    normals[selected],
                )
            )
            findings.append(
                MeshCertificateFinding(
                    "projection_source_piece_capacity", "unresolved", "mesh"
                )
            )
            break
        parameters = _split_reference_triangles(parameters[selected])
        owners = np.repeat(owners[selected], 4)
        depth = np.repeat(depth[selected] + 1, 4)
        created += owners.size
    shapes = ((0, 3, 3), (0, 3, 3), (0, 2, 3), (0,), (0, 2), (0, 2, 3), (0, 2))
    arrays = tuple(
        np.concatenate([terminal[column] for terminal in terminals])
        if terminals
        else np.empty(shape, dtype=np.int64 if column == 3 else np.float64)
        for column, shape in enumerate(shapes)
    )
    fields = arrays[4]
    upper = float(np.max(np.abs(fields))) if fields.size else math.inf
    excludes = (fields[:, 0] > 0.0) | (fields[:, 1] < 0.0)
    lower = (
        float(
            np.max(
                np.where(
                    excludes, np.minimum(np.abs(fields[:, 0]), np.abs(fields[:, 1])), 0.0
                )
            )
        )
        if fields.size
        else 0.0
    )
    return _Pieces(
        triangles=arrays[0],
        parameters=arrays[1],
        coordinates=arrays[2],
        owners=arrays[3],
        fields=arrays[4],
        gradients=arrays[5],
        normals=arrays[6],
        upper=upper,
        lower=lower,
        queries=queries,
        findings=tuple(findings),
    )


def _fraction_point(values: np.ndarray, /) -> _RationalPoint:
    return (
        Fraction(float(values[0])),
        Fraction(float(values[1])),
        Fraction(float(values[2])),
    )


def _determinant(a: _RationalPoint, b: _RationalPoint, c: _RationalPoint, /) -> Fraction:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _difference(a: _RationalPoint, b: _RationalPoint, /) -> _RationalPoint:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _exact_crossing(
    start: _RationalPoint,
    end: _RationalPoint,
    face: tuple[_RationalPoint, _RationalPoint, _RationalPoint],
    /,
) -> tuple[ProjectionFiberCrossingKind, Fraction | None, int]:
    direction = _difference(end, start)
    first = _difference(face[0], face[1])
    second = _difference(face[0], face[2])
    right = _difference(face[0], start)
    denominator = _determinant(direction, first, second)
    if denominator == 0:
        coplanar = _determinant(right, first, second) == 0
        return ("coplanar" if coplanar else "disjoint"), None, 0
    parameter = _determinant(right, first, second) / denominator
    u = _determinant(direction, right, second) / denominator
    v = _determinant(direction, first, right) / denominator
    weights = (1 - u - v, u, v)
    if parameter < 0 or parameter > 1 or any(weight < 0 for weight in weights):
        return "disjoint", None, 0
    zeros = sum(weight == 0 for weight in weights)
    kind: ProjectionFiberCrossingKind = (
        "vertex" if zeros > 1 else ("edge" if zeros else "interior")
    )
    return kind, parameter, 1 if denominator > 0 else -1


def _fraction_interval(value: Fraction, /) -> tuple[float, float]:
    approximate = float(value)
    lower = (
        float(np.nextafter(approximate, -np.inf))
        if Fraction(approximate) > value
        else approximate
    )
    upper = (
        float(np.nextafter(approximate, np.inf))
        if Fraction(approximate) < value
        else approximate
    )
    return lower, upper


def _sqrt_interval(value: Fraction, /) -> np.ndarray:
    approximate = math.sqrt(float(value))
    lower, upper = approximate, approximate
    while Fraction(lower) ** 2 > value:
        lower = float(np.nextafter(lower, -np.inf))
    while Fraction(upper) ** 2 < value:
        upper = float(np.nextafter(upper, np.inf))
    return np.asarray((lower, upper), dtype=np.float64)


def _source_fiber(
    profile: AnalyticImplicitProfile, index: int, /
) -> tuple[_RationalPoint, _RationalPoint]:
    center = _fraction_point(profile.center)
    radius = Fraction(profile.radius)
    match profile.family:
        case "sphere":
            start = center
            direction: _RationalPoint = (
                Fraction(1),
                Fraction(index),
                Fraction(index * index + 1),
            )
        case "ring-torus":
            # Rational unit-circle coordinates establish the exact represented
            # torus core point. Positive radial motion keeps rho strictly > 0;
            # phi(t)=t*||end-start||-radius throughout this normal ray.
            t = Fraction(index, index + 1)
            cosine, sine = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
            major = Fraction(profile.major_radius)
            start = (center[0] + major * cosine, center[1] + major * sine, center[2])
            direction = (cosine, sine, Fraction(index * index + 1))
        case _:
            raise ValueError("The nominal analytic source has no qualified normal fiber.")
    end = tuple(start[axis] + 2 * radius * direction[axis] for axis in range(3))
    return start, (end[0], end[1], end[2])


def _fiber(
    boundary: _Boundary,
    profile: AnalyticImplicitProfile,
    forward: float,
    limits: MeshCertificateLimits,
    remaining_queries: int,
    /,
) -> _Fiber:
    empty = np.empty((0,), dtype=np.int64)
    uncomputed_values = np.asarray(
        ((-np.inf, np.inf), (-np.inf, np.inf)), dtype=np.float64
    )
    uncomputed_gradient = np.asarray((-np.inf, np.inf), dtype=np.float64)
    if remaining_queries < 2:
        return _Fiber(
            (),
            uncomputed_values,
            uncomputed_gradient,
            empty,
            (),
            (),
            0,
            0,
            (
                MeshCertificateFinding(
                    "projection_fiber_query_capacity", "unresolved", "mesh"
                ),
            ),
        )
    lower = (
        boundary.triangles
        if boundary.coordinate_bounds is None
        else boundary.coordinate_bounds[:, 0]
    )
    upper = (
        boundary.triangles
        if boundary.coordinate_bounds is None
        else boundary.coordinate_bounds[:, 1]
    )
    boxes_low = np.min(lower, axis=1)
    boxes_high = np.max(upper, axis=1)
    tests = 0
    queries = 0
    cache: dict[int, tuple[_RationalPoint, _RationalPoint, _RationalPoint]] = {}
    for index in range(1, limits.maximum_ray_tests + 1):
        start, end = _source_fiber(profile, index)
        endpoint_boxes = np.asarray(
            [[_fraction_interval(value) for value in point] for point in (start, end)],
            dtype=np.float64,
        )
        segment_low = np.min(endpoint_boxes[:, :, 0], axis=0)
        segment_high = np.max(endpoint_boxes[:, :, 1], axis=0)
        candidates = np.flatnonzero(
            np.all((boxes_high >= segment_low) & (boxes_low <= segment_high), axis=1)
        )
        if (
            tests + candidates.size > limits.maximum_ray_tests
            or queries + 2 > remaining_queries
        ):
            return _Fiber(
                (),
                uncomputed_values,
                uncomputed_gradient,
                empty,
                (),
                (),
                queries,
                tests,
                (
                    MeshCertificateFinding(
                        "projection_fiber_capacity", "unresolved", "mesh"
                    ),
                ),
            )
        classes: list[ProjectionFiberCrossingKind] = []
        parameters: list[tuple[int, int] | None] = []
        orientation: list[int] = []
        for row in candidates.tolist():
            if row not in cache:
                if boundary.exact_triangles is None:
                    values = boundary.triangles[row]
                    cache[row] = (
                        _fraction_point(values[0]),
                        _fraction_point(values[1]),
                        _fraction_point(values[2]),
                    )
                else:
                    cache[row] = boundary.exact_triangles[row]
            kind, parameter, sign = _exact_crossing(start, end, cache[row])
            classes.append(kind)
            parameters.append(
                None
                if parameter is None
                else (parameter.numerator, parameter.denominator)
            )
            orientation.append(sign)
        tests += candidates.size
        if any(kind in ("edge", "vertex", "coplanar") for kind in classes):
            # Generic rational directions avoid every finite exceptional edge
            # and plane set; the work limit remains authoritative.
            continue
        endpoint_source = profile.field_bounds.boxes(
            endpoint_boxes[:, :, 0], endpoint_boxes[:, :, 1]
        )
        queries += 2
        value_bounds = np.stack(
            (endpoint_source.value_lower, endpoint_source.value_upper), axis=1
        )
        direction = _difference(end, start)
        gradient = _sqrt_interval(
            sum((value * value for value in direction), Fraction(0))
        )
        serialized = tuple(
            tuple((value.numerator, value.denominator) for value in point)
            for point in (start, end)
        )
        crossings = [row for row, kind in enumerate(classes) if kind == "interior"]
        findings: list[MeshCertificateFinding] = []
        if not (
            value_bounds[0, 1] < -forward
            and value_bounds[1, 0] > forward
            and gradient[0] > 0.0
        ):
            findings.append(
                MeshCertificateFinding(
                    "projection_regular_source_fiber", "unresolved", "mesh"
                )
            )
        if len(crossings) != 1 or any(orientation[row] != 1 for row in crossings):
            findings.append(
                MeshCertificateFinding("projection_fiber_degree_one", "violated", "mesh")
            )
        return _Fiber(
            serialized,
            value_bounds,
            gradient,
            candidates,
            tuple(classes),
            tuple(parameters),
            queries,
            tests,
            tuple(findings),
        )
    return _Fiber(
        (),
        uncomputed_values,
        uncomputed_gradient,
        empty,
        (),
        (),
        queries,
        tests,
        (
            MeshCertificateFinding(
                "projection_generic_fiber_capacity", "unresolved", "mesh"
            ),
        ),
    )


def compute_implicit_projection_cover(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    profile: AnalyticImplicitProfile,
    /,
    *,
    tolerance: float,
    limits: MeshCertificateLimits,
) -> ImplicitProjectionCoverageEvidence:
    """Recompute all source-projection premises from actual maps and nominal facts."""
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Projection coverage requires CellMesh and CellGeometrySpec.")
    if not isinstance(profile, AnalyticImplicitProfile):
        raise TypeError(
            "Projection coverage requires an established AnalyticImplicitProfile."
        )
    if not isinstance(limits, MeshCertificateLimits):
        raise TypeError("limits must be MeshCertificateLimits.")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    profile.require_bound(profile.geometry, profile.coordinate_contract)
    established = AnalyticImplicitProfile(
        profile.geometry,
        profile.coordinate_contract,
        tube_radius=profile.tube_radius,
        cover_radius=profile.cover_radius,
        source_id=profile.source_id,
        source_revision=profile.source_revision,
    )
    if established.profile_id != profile.profile_id:
        raise ValueError(
            "The analytic profile's cached premises are not source-established."
        )
    budget = coordinate_enclosure_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    with budget.activate():
        scope_resource: CoordinateEnclosureResourceError | None = None
        try:
            scope, _, _ = _coordinate_scope(mesh, geometry, budget)
        except CoordinateEnclosureResourceError as error:
            scope, scope_resource = "mapped", error
        binding = MeshCertificateBinding(
            mesh,
            geometry,
            scope,
            limits,
            source_id=established.source_id,
            source_revision=established.source_revision,
        )
        boundary = _boundary(mesh, established)
        if (
            scope_resource is None
            and scope != "affine"
            and not boundary.findings
            and boundary.triangles.shape[0] <= limits.maximum_subdivision_pieces
        ):
            try:
                boundary = _source_affine_boundary(mesh, geometry, boundary, budget)
            except CoordinateEnclosureResourceError as error:
                scope_resource = error
        findings = list(boundary.findings)
        if scope_resource is not None:
            findings.append(
                MeshCertificateFinding(
                    "projection_coordinate_resource_budget",
                    "unresolved",
                    "mesh",
                    resource=scope_resource.resource,
                    requested=(
                        ("limit", scope_resource.limit),
                        ("requested", scope_resource.requested),
                    ),
                    achieved=(
                        ("completed", scope_resource.completed),
                        ("source_expression_work_units", budget.work_units),
                        ("source_expression_peak_bytes", budget.peak_bytes_upper),
                        (
                            "source_expression_native_charged_work_units",
                            budget.native_charged_work_units,
                        ),
                        (
                            "source_expression_retained_basis_bytes_at_catch",
                            budget.retained_basis_bytes,
                        ),
                        (
                            "source_expression_temporary_bytes_upper_at_catch",
                            budget.temporary_bytes_upper,
                        ),
                    ),
                )
            )
        if (
            scope != "affine" and boundary.exact_triangles is None
        ) or mesh.ambient_dimension != 3:
            if not any(
                finding.check == "projection_actual_affine_maps" for finding in findings
            ):
                findings.append(
                    MeshCertificateFinding(
                        "projection_actual_affine_maps", "unresolved", "mesh"
                    )
                )
        if boundary.triangles.shape[0] > limits.maximum_subdivision_pieces:
            findings.append(
                MeshCertificateFinding(
                    "projection_boundary_capacity", "unresolved", "mesh"
                )
            )
        if findings:
            pieces = _Pieces(
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0, 2, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                math.inf,
                0.0,
                1,
                (),
            )
        else:
            pieces = _pieces(boundary, established, tolerance, limits)
            findings.extend(pieces.findings)
        if findings:
            fiber = _Fiber(
                (),
                np.asarray(((-np.inf, np.inf), (-np.inf, np.inf)), dtype=np.float64),
                np.asarray((-np.inf, np.inf), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                (),
                (),
                0,
                0,
                (),
            )
        else:
            remaining = (
                min(limits.maximum_source_samples, limits.maximum_distance_evaluations)
                - pieces.queries
            )
            fiber = _fiber(boundary, established, pieces.upper, limits, remaining)
            findings.extend(fiber.findings)
        return ImplicitProjectionCoverageEvidence(
            binding,
            established,
            tuple(findings),
            facets=boundary.triangles,
            facet_vertices=boundary.global_vertices,
            facet_ids=boundary.identifiers,
            facet_components=boundary.components,
            piece_triangles=pieces.triangles,
            piece_barycentric_vertices=pieces.parameters,
            piece_coordinate_bounds=pieces.coordinates,
            piece_owners=pieces.owners,
            field_bounds=pieces.fields,
            gradient_bounds=pieces.gradients,
            normal_bounds=pieces.normals,
            fiber_endpoints=fiber.endpoints,
            fiber_value_bounds=fiber.values,
            fiber_gradient_bounds=fiber.gradient,
            fiber_candidate_facets=fiber.candidates,
            fiber_crossing_classes=fiber.classes,
            fiber_crossing_parameters=fiber.parameters,
            forward_upper=pieces.upper,
            forward_lower=pieces.lower,
            query_count=pieces.queries + fiber.queries,
            ray_test_count=fiber.tests,
        )


__all__ = ["compute_implicit_projection_cover"]
