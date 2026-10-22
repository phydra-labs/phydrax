#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify degree-one tubular projection onto a qualified analytic source.

The affine boundary is a closed oriented two-manifold. Dyadic reference
triangles cover every face; outward coordinate and source-field intervals
bound its distance and strictly positive normal derivative in the reach tube.
Oriented vertex links and source topology establish a local cover. A rational
regular normal fiber with one positively oriented interior crossing fixes
degree one, so the forward distance bound also bounds the reverse distance.
Volume-cell validity and interior embedding are certified separately.

Mesh certificate checks
-----------------------
[
  {
    "checks": ["projection_closed_oriented_edges"],
    "premises": [
      {"tag": "X", "statement": "Every undirected boundary edge has exactly two incident faces whose directed incidence signs sum to zero."}
    ],
    "conclusion": "The boundary is an oriented two-cycle with paired edge incidences; an unpaired or inconsistently oriented edge leaves projection unresolved.",
    "citation": {"kind": "definition", "reference": "Projection boundary incidence", "locator": "phydrax.geometry.implicit._projection_cover._manifold_components", "url": ""},
    "failure_statuses": ["unresolved"]
  },
  {
    "checks": ["projection_manifold_vertex_link"],
    "premises": [
      {"tag": "X", "statement": "Each boundary vertex link graph is connected and every link vertex has degree two."}
    ],
    "conclusion": "Each boundary vertex has one circular link; a disconnected or branched link leaves the local surface condition unresolved.",
    "citation": {"kind": "definition", "reference": "Simplicial surface vertex link", "locator": "phydrax.geometry.implicit._projection_cover._manifold_components", "url": ""},
    "failure_statuses": ["unresolved"]
  },
  {
    "checks": ["projection_source_topology"],
    "premises": [
      {"tag": "X", "statement": "Boundary connected-component count and V-E+F agree with the qualified source profile component count and Euler characteristic."}
    ],
    "conclusion": "The discrete boundary satisfies the component and Euler-characteristic prerequisites used by the projection proof.",
    "citation": {"kind": "definition", "reference": "Source topology comparison", "locator": "phydrax.geometry.implicit._projection_cover._manifold_components", "url": ""},
    "failure_statuses": ["unresolved"]
  },
  {
    "checks": ["projection_affine_triangle_carrier", "projection_empty_boundary"],
    "premises": [
      {"tag": "X", "statement": "Boundary extraction yields a nonempty set of affine triangular facets in three spatial dimensions."}
    ],
    "conclusion": "The boundary admits the exhaustive affine reference-triangle route; unsupported or absent facets leave projection unresolved.",
    "citation": {"kind": "definition", "reference": "Projection facet admission", "locator": "phydrax.geometry.implicit._projection_cover._boundary", "url": ""},
    "failure_statuses": ["unresolved"]
  },
  {
    "checks": ["projection_actual_affine_maps", "projection_source_vertex_continuity"],
    "premises": [
      {"tag": "X", "statement": "Coordinate-source expressions are affine on each triangular cell, their corners round to the published coordinates, and every shared vertex has one rational source image."}
    ],
    "conclusion": "Rational triangle laws describe one continuous source boundary, preserving its cell and shared-vertex identities.",
    "citation": {"kind": "definition", "reference": "Affine coordinate source admission", "locator": "phydrax.geometry.implicit._projection_cover._source_affine_boundary", "url": ""},
    "failure_statuses": ["unresolved"]
  },
  {
    "checks": ["projection_source_orientation_or_distance"],
    "premises": [
      {"tag": "N", "statement": "Outward field and gradient intervals cover each reference-triangle image; its distance bound is at most the requested tolerance and reach-tube radius, and its normal derivative lower bound is strictly positive."},
      {"tag": "T", "statement": "For a sphere source f(y) = |y - c| - R, convexity of the Euclidean norm bounds f on an exact affine piece below by the center's distance to its plane and above by its largest corner distance; field intervals may intersect this with the coordinate-box enclosure."},
      {"tag": "T", "statement": "For the current ring-torus f=|(rho-R,z)|-r, 0<L=r-band<U=r+band<R makes B_a=|p|^2+R^2-a^2 strictly positive. Thus F_U=B_U^2-4R^2*rho^2<=0 and F_L>=0 are equivalent to the full closed band L<=|(rho-R,z)|<=U; exact simplex Bernstein controls may prove both inequalities over an entire affine piece. This tightens field intervals only and does not establish normal orientation."},
      {"tag": "N", "statement": "The source profile recomputes signed-distance and positive-reach bounds from the current sphere or ring-torus state; its tube radius is strictly below that reach."},
      {"tag": "T", "statement": "Within positive reach, nearest-point projection is unique and continuous; the qualified smooth source tube has normal fibers and positive-normal surface maps give local covers."}
    ],
    "conclusion": "Admitted facets lie in the projection tube with positive orientation; a nonpositive normal upper bound or a field interval wholly outside tolerance proves violation.",
    "citation": {"kind": "mathematical", "reference": "Federer, Curvature Measures (1959)", "locator": "Section 4.8, nearest-point projection and reach", "url": "https://doi.org/10.1090/S0002-9947-1959-0110078-1"},
    "failure_statuses": ["violated"]
  },
  {
    "checks": ["projection_source_piece_depth"],
    "premises": [
      {"tag": "X", "statement": "The subdivision depth equals the configured maximum depth."},
      {"tag": "N", "statement": "Terminal field and normal intervals neither establish all distance/tube/positive-normal inequalities nor disprove the distance or orientation condition."}
    ],
    "conclusion": "The unresolved terminal retains its interval bounds and source facet identity.",
    "citation": {"kind": "definition", "reference": "Projection subdivision stopping condition", "locator": "phydrax.geometry.implicit._projection_cover._pieces", "url": ""},
    "failure_statuses": ["unresolved"]
  },
  {
    "checks": ["projection_regular_source_fiber"],
    "premises": [
      {"tag": "N", "statement": "The rational fiber endpoint field bounds straddle the forward-distance band and the normal-direction length lower bound is positive."},
      {"tag": "X", "statement": "The candidate normal fiber avoids all boundary edges, vertices, and coplanar triangles under rational crossing predicates."}
    ],
    "conclusion": "The selected fiber is a regular degree-counting fiber across the qualified source tube; undecided endpoint separation leaves the count unresolved.",
    "citation": {"kind": "mathematical", "reference": "Hatcher, Algebraic Topology (2002)", "locator": "Proposition 2.30, degree as the sum of local degrees at preimages", "url": "https://pi.math.cornell.edu/~hatcher/AT/AT.pdf"},
    "failure_statuses": ["unresolved"]
  },
  {
    "checks": ["projection_fiber_degree_one"],
    "premises": [
      {"tag": "X", "statement": "Rational predicates find exactly one interior triangle crossing of the regular normal fiber, with orientation sign +1."},
      {"tag": "N", "statement": "All facets satisfy the certified distance and positive-normal tube bounds."},
      {"tag": "X", "statement": "All boundary vertex links are connected degree-two graphs and the oriented boundary edges are paired."},
      {"tag": "T", "statement": "The resulting map between closed oriented source and boundary surfaces is an orientation-preserving local cover; local degrees are positive."}
    ],
    "conclusion": "The fiber degree is one, hence the local cover has one sheet and the forward deviation bound also bounds reverse deviation; any other crossing count or orientation violates this degree requirement.",
    "citation": {"kind": "mathematical", "reference": "Hatcher, Algebraic Topology (2002)", "locator": "Proposition 2.30 and Proposition 1.32, local degree sum and number of covering sheets", "url": "https://pi.math.cornell.edu/~hatcher/AT/AT.pdf"},
    "failure_statuses": ["violated"]
  },
  {
    "checks": ["projection_source_query_capacity", "projection_source_piece_capacity", "projection_fiber_query_capacity", "projection_fiber_capacity", "projection_generic_fiber_capacity", "projection_coordinate_resource_budget", "projection_boundary_capacity"],
    "premises": [
      {"tag": "X", "statement": "The next source-query, reference-piece, rational ray-test, coordinate-expression, or input-boundary admission exceeds its declared remaining query, work, memory, depth, or piece limit."}
    ],
    "conclusion": "Projection remains unresolved with the capacity category and completed work retained; capacity refusal proves no geometric violation.",
    "citation": {"kind": "definition", "reference": "Projection resource admission", "locator": "phydrax.geometry.implicit._projection_cover.compute_implicit_projection_cover, _pieces, and _fiber", "url": ""},
    "failure_statuses": ["unresolved"]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from phydrax.geometry._embedding_state import (
    coordinate_certificate_request,
    MeshCertificateFinding,
    MeshCertificateLimits,
    MeshFindingStatus,
)

from ...discretization import CellGeometrySpec, CellMesh, TetrahedralConnectivity
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from ...discretization._coordinate_enclosure import (
    _scalar_hash_storage,
    add as polynomial_add,
    axes as polynomial_axes,
    bernstein_coefficients,
    constant as polynomial_constant,
    linear_combinations as polynomial_linear_combinations,
    multiply as polynomial_multiply,
    scale as polynomial_scale,
    prepared_coordinate_source_bank,
    prepared_coordinate_vertex_images,
    RationalPolynomial,
    source_expressions,
)
from ...discretization._exact_algebraic_geometry import rational_coefficient
from ...typing import checked
from .._global_embedding import coordinate_scope
from .._mesh_certificates import MeshCertificateBinding
from .._source_fidelity import (
    ImplicitProjectionCoverageEvidence,
    ProjectionFacetEvidence,
    ProjectionFiberCrossingKind,
    ProjectionFiberEvidence,
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
    """Establish closed-manifold incidence and oriented vertex links."""
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
    match (mesh.topological_dimension, mesh.connectivity):
        case (3, TetrahedralConnectivity() as connectivity):
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
        case (2, _) if all(block.cell_kind == "triangle" for block in mesh.blocks):
            vertices = np.concatenate(
                [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
            )
            ids = np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            )
        case _:
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
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Boundary:
    """Prove full source-affine laws and retain their rational nonbinary corners."""
    prepared_coordinate_source_bank(geometry, budget=budget)
    elements, _, _ = geometry._resolve(
        mesh, exact_source_prepared=True, budget=budget
    )
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    budget.reserve(0, 8 * carrier.shape[0] + 144 * boundary.triangles.shape[0])
    vertices: list[_RationalPoint | None] = [None] * carrier.shape[0]
    for element in elements:
        expressions = source_expressions(element, budget=budget)
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
    # This existing preparation owner binds the complete live source bank,
    # element definitions, reference actions, routes and mesh layout. Equal
    # global vertex IDs or rounded carrier points alone never authorize reuse.
    # Distinct operations reaching one vertex still earn exact continuity;
    # only an identical routed corner law shares its already computed image.
    mismatch = "projection_cover.source_vertex_continuity"
    try:
        images = prepared_coordinate_vertex_images(
            mesh, geometry, mismatch=mismatch, budget=budget
        )
    except ValueError as error:
        if str(error) != mismatch:
            raise
        return replace(
            boundary,
            findings=(
                *boundary.findings,
                MeshCertificateFinding(
                    "projection_source_vertex_continuity", "unresolved", "mesh"
                ),
            ),
        )
    if images is None:
        return replace(
            boundary,
            findings=(
                *boundary.findings,
                MeshCertificateFinding(
                    "projection_actual_affine_maps", "unresolved", "mesh"
                ),
            ),
        )
    for vertex, image in enumerate(images):
        if image is None:
            continue
        # Rational conversion and carrier comparison execute once per proved
        # shared image, not once per incident cell. No seam check is omitted.
        budget.reserve(3)
        point: _RationalPoint = (
            rational_coefficient(image[0], role="projection_cover.source_corner"),
            rational_coefficient(image[1], role="projection_cover.source_corner"),
            rational_coefficient(image[2], role="projection_cover.source_corner"),
        )
        rounded = np.asarray(tuple(float(value) for value in point), dtype=np.float64)
        if not np.array_equal(rounded.view(np.uint64), carrier[vertex].view(np.uint64)):
            raise ValueError(
                "projection_cover.carrier must equal the round-to-nearest source image; got mismatched coordinates"
            )
        vertices[vertex] = point
    triangles = []
    bounds = np.empty((boundary.local_vertices.shape[0], 2, 3, 3), dtype=np.float64)
    for row, face in enumerate(boundary.local_vertices):
        points = tuple(vertices[int(vertex)] for vertex in face)
        if any(point is None for point in points):
            raise ValueError(
                "ImplicitProjectionCover.source_vertices must have one represented rational point per boundary vertex; got an uncovered source vertex"
            )
        first, second, third = points
        if first is None or second is None or third is None:
            raise ValueError(
                "ImplicitProjectionCover.source_vertices must have one represented rational point per boundary vertex; got an uncovered source vertex"
            )
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
    # Dyadic barycentric midpoints remain representable through the bounded
    # subdivision depth; only affine physical images need outward enclosures.
    return np.stack(
        (
            np.stack((first, ab, ca), axis=1),
            np.stack((ab, second, bc), axis=1),
            np.stack((ca, bc, third), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ),
        axis=1,
    ).reshape((-1, 3, 3))


def _piece_center_offset(
    weights: np.ndarray,
    facet: tuple[_RationalPoint, ...],
    center: _RationalPoint,
    /,
) -> _RationalPoint:
    """Exact offset of one dyadic barycentric piece corner from a rational center."""
    x, y, z = (
        sum(
            (
                Fraction(float(weight)) * point[axis]
                for weight, point in zip(weights, facet, strict=True)
            ),
            Fraction(0),
        )
        - center[axis]
        for axis in range(3)
    )
    return x, y, z


def _sphere_piece_fields(
    boundary: _Boundary,
    profile: AnalyticImplicitProfile,
    owners: np.ndarray,
    parameters: np.ndarray,
    fields: np.ndarray,
    tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> np.ndarray:
    """Intersect box intervals with each exact piece's sphere distance range.

    The Euclidean norm is convex, so its maximum over a triangle is attained at
    a corner, and its minimum is at least the center's distance to the
    triangle's plane. With ``f = |y - c| - R`` this encloses ``f`` over the
    exact affine piece itself rather than over its coordinate box.
    """
    # Pieces whose box interval already meets the distance bounds need no
    # second enclosure; only the remaining rows earn the exact sphere range.
    deviation = np.maximum(np.abs(fields[:, 0]), np.abs(fields[:, 1]))
    rows = np.flatnonzero(deviation > min(tolerance, profile.tube_radius))
    if budget is not None:
        # Rational corners, center offsets, corner norms, plane normal,
        # triple product and the four square-root enclosures of each piece.
        budget.reserve(113 * rows.size)
    center = _fraction_point(profile.center)
    radius = Fraction(profile.radius)
    tightened = fields.copy()
    for row in rows.tolist():
        owner, weights = int(owners[row]), parameters[row]
        facet = (
            boundary.exact_triangles[owner]
            if boundary.exact_triangles is not None
            else tuple(_fraction_point(point) for point in boundary.triangles[owner])
        )
        first, second, third = (
            _piece_center_offset(weights[corner], facet, center) for corner in range(3)
        )
        ceiling = max(
            float(_sqrt_interval(sum((value * value for value in point), Fraction(0)))[1])
            for point in (first, second, third)
        )
        edge, other = _difference(second, first), _difference(third, first)
        normal = (
            edge[1] * other[2] - edge[2] * other[1],
            edge[2] * other[0] - edge[0] * other[2],
            edge[0] * other[1] - edge[1] * other[0],
        )
        area = sum((value * value for value in normal), Fraction(0))
        floor = (
            0.0
            if area == 0
            else float(_sqrt_interval(_determinant(first, second, third) ** 2 / area)[0])
        )
        tightened[row, 0] = max(
            tightened[row, 0], _fraction_interval(Fraction(floor) - radius)[0]
        )
        tightened[row, 1] = min(
            tightened[row, 1], _fraction_interval(Fraction(ceiling) - radius)[1]
        )
    return tightened


# One cold torus piece row has three identical 40-unit exact corner
# computations and a separate 120-unit gradient/curvature/slope stage.
# Memo hits omit only the corner computation that is not executed.
_TORUS_CORNER_WORK = 120
_TORUS_SHAPE_WORK = 120
_TORUS_SINGLE_CORNER_WORK = _TORUS_CORNER_WORK // 3

_TorusCorner = tuple[
    _RationalPoint,
    tuple[Fraction, Fraction] | None,
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
]
type _TorusCornerKey = tuple[int, bytes] | tuple[str, _RationalPoint | bytes]


def _torus_corner_bounds(
    weights: np.ndarray,
    facet: tuple[_RationalPoint, ...],
    center: _RationalPoint,
    major: Fraction,
    minor: Fraction,
    /,
) -> _TorusCorner:
    point = _piece_center_offset(weights, facet, center)
    x, y, z = point
    rho = _sqrt_interval(x * x + y * y)
    axis = (Fraction(float(rho[0])), Fraction(float(rho[1])))
    radial = (axis[0] - major, axis[1] - major)
    nearest = (
        Fraction(0)
        if radial[0] <= 0 <= radial[1]
        else min(abs(radial[0]), abs(radial[1]))
    )
    farthest = max(abs(radial[0]), abs(radial[1]))
    tube = (
        Fraction(float(_sqrt_interval(nearest * nearest + z * z)[0])),
        Fraction(float(_sqrt_interval(farthest * farthest + z * z)[1])),
    )
    value = None if tube[0] <= 0 or axis[0] <= 0 else (tube[0] - minor, tube[1] - minor)
    return point, value, tube, axis


class _TorusCornerMemo:
    """Own exact corner laws for one immutable boundary/profile subdivision.

    One-hot corners use the complete exact input-point law: tagged rational
    source coordinates, or tagged binary64 source bytes. Their pure point
    function values can agree across facets without identifying those facets.
    Signed-zero binary bytes remain distinct, and nonbinary rational inputs
    never use their rounded carrier as a key. Other corners keep owner row
    plus exact dyadic barycentric bytes, fixing their full affine construction.
    The call fixes all source laws, center/radii and current profile; neither
    entries nor this owner escape _pieces. Facet normals and shape proofs are
    still earned per owner, and every new call independently earns its values.
    """

    def __init__(
        self,
        boundary: _Boundary,
        profile: AnalyticImplicitProfile,
        maximum_depth: int,
        budget: CoordinateEnclosureBudget | None,
        storage: CoordinateLiveStorage | None,
    ) -> None:
        self.boundary = boundary
        self.center = _fraction_point(profile.center)
        self.major = Fraction(profile.major_radius)
        self.minor = Fraction(profile.radius)
        self.maximum_depth = maximum_depth
        self.budget = budget
        self.storage = storage
        self.corners: dict[_TorusCornerKey, _TorusCorner] = {}

    def corner(self, owner: int, weights: np.ndarray, /) -> _TorusCorner:
        if self.budget is not None:
            # Key construction, dictionary lookup and (on a miss) insertion.
            self.budget.reserve(1)
        source_corner = None
        if weights.shape == (3,) and weights.dtype == np.float64:
            if weights[0] == 1.0 and weights[1] == 0.0 and weights[2] == 0.0:
                source_corner = 0
            elif weights[0] == 0.0 and weights[1] == 1.0 and weights[2] == 0.0:
                source_corner = 1
            elif weights[0] == 0.0 and weights[1] == 0.0 and weights[2] == 1.0:
                source_corner = 2
        key: _TorusCornerKey | None = None
        hash_storage = 0
        if source_corner is not None:
            if self.boundary.exact_triangles is not None:
                point = self.boundary.exact_triangles[owner][source_corner]
                if len(point) == 3 and all(type(value) is Fraction for value in point):
                    key = ("rational", point)
                    if self.budget is not None:
                        # The existing rational-key convention admits integer
                        # hashing scratch before either lookup or insertion.
                        hash_storage = 256 + 4 * max(
                            _scalar_hash_storage(value) for value in point
                        )
            elif self.boundary.triangles.dtype == np.float64:
                key = ("binary64", self.boundary.triangles[owner, source_corner].tobytes())
        if key is None:
            key = (owner, weights.tobytes())
        if self.budget is not None and hash_storage:
            with self.budget.temporary_scope():
                self.budget.reserve(0, hash_storage)
                return self._corner_value(owner, weights, key)
        return self._corner_value(owner, weights, key)

    def _corner_value(
        self, owner: int, weights: np.ndarray, key: _TorusCornerKey, /
    ) -> _TorusCorner:
        cached = self.corners.get(key)
        if cached is not None:
            return cached
        facet = (
            self.boundary.exact_triangles[owner]
            if self.boundary.exact_triangles is not None
            else None
        )
        if self.storage is not None:
            # Three dyadic-weighted source atoms plus center: multiplying
            # denominators bounds either coordinate integer by 4B+3D+4 bits.
            # Rounded sqrt endpoints have at most 1075 bits; subtraction of
            # the source radius adds at most B+1077. Nine retained Fractions,
            # their integers, tuples, byte key, dict growth and spare lookup
            # storage are admitted before constructing the owned corner.
            bits = max(
                max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                for value in (*self.center, self.major, self.minor)
            )
            if facet is None:
                for point in self.boundary.triangles[owner]:
                    for value in point:
                        numerator, denominator = float(value).as_integer_ratio()
                        bits = max(
                            bits, abs(numerator).bit_length(), denominator.bit_length()
                        )
            else:
                for point in facet:
                    for value in point:
                        bits = max(
                            bits, abs(value.numerator).bit_length(),
                            value.denominator.bit_length(),
                        )
            bound = max(4 * bits + 3 * min(self.maximum_depth + 1, 1075) + 4, bits + 1077)
            entry_bytes = 4096 + 9 * (96 + 2 * (32 + 4 * ((bound + 29) // 30)))
            self.storage.set_bound(
                self.storage.bound + entry_bytes, work=_TORUS_SINGLE_CORNER_WORK
            )
        elif self.budget is not None:
            self.budget.reserve(_TORUS_SINGLE_CORNER_WORK)
        if facet is None:
            facet = tuple(
                _fraction_point(point) for point in self.boundary.triangles[owner]
            )
        result = _torus_corner_bounds(weights, facet, self.center, self.major, self.minor)
        self.corners[key] = result
        return result


def _torus_quartic_band(
    corners: tuple[_RationalPoint, ...],
    major: Fraction,
    minor: Fraction,
    band: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction] | None:
    """Prove the same torus's entire closed band, or leave it unresolved.

    For L=r-band and U=r+band, require 0<L<U<R. Then
    B_a=|p|^2+R^2-a^2 >= R^2-U^2 > 0 for a in {L,U}.
    Consequently squaring preserves both comparisons with 2*R*rho:
    t<=U iff F_U<=0, and t>=L iff F_L>=0, where
    F_a=B_a^2-4*R^2*(x^2+y^2) and t=|(rho-R,z)|.
    In particular t<=U<R implies rho>=R-U>0; no axis or
    medial-circle branch is silently admitted.

    The owning polynomial operations construct three affine coordinates
    (at most 3 terms each), three quadratic squares (at most 6 each),
    and two quartics (at most 15 each). The nonconstant squares require
    at most 3*6+2*21=60 distinct symmetric product pairs. A degree-four
    two-simplex net has 15 controls and at most 55 nonconstant
    term-to-control entries: its owner charges at most 15+15+55=85
    initialization/support/accumulation visits per net, in addition to
    profiles, reference-table preparation and sign comparisons. Products,
    scales and sums use their actual existing owner rates, not shape120.
    At dimension two each polynomial admission reserves
    256+n*(576+2*declared_integer_storage_bytes(bits)) bytes, using its
    earned coefficient-bit profile; overlapping stages accumulate these
    bounds. The existing owners also admit key/profile/reference scratch
    and retained controls rather than treating the 15 coefficients as free.
    Local preparations and all arithmetic scratch die with this scope;
    no source/profile identity or numerical proof escapes in a cache.
    """
    with budget.polynomial_preparation_scope():
        # Use the owning arithmetic even for the side-condition constants.
        radius = polynomial_constant(minor, 2, budget=budget)
        width = polynomial_constant(band, 2, budget=budget)
        lower = polynomial_add(
            radius, polynomial_scale(width, -1, budget=budget), budget=budget
        )
        upper = polynomial_add(radius, width, budget=budget)
        low = lower.get((0, 0), Fraction(0))
        high = upper.get((0, 0), Fraction(0))
        if not 0 < low < high < major:
            return None
        u, v = polynomial_axes(2, budget=budget)
        last = polynomial_add(
            polynomial_constant(1, 2, budget=budget),
            polynomial_scale(polynomial_add(u, v, budget=budget), -1, budget=budget),
            budget=budget,
        )
        coordinates = polynomial_linear_combinations(
            (u, v, last),
            tuple(tuple(point[axis] for point in corners) for axis in range(3)),
            budget=budget,
        )
        squares = tuple(
            polynomial_multiply(value, value, budget=budget) for value in coordinates
        )
        planar = polynomial_add(squares[0], squares[1], budget=budget)
        squared = polynomial_add(planar, squares[2], budget=budget)
        major_polynomial = polynomial_constant(major, 2, budget=budget)
        major_squared = polynomial_multiply(
            major_polynomial, major_polynomial, budget=budget
        )
        base = polynomial_add(squared, major_squared, budget=budget)
        radial = polynomial_multiply(
            polynomial_scale(major_squared, -4, budget=budget), planar, budget=budget
        )
        for endpoint, outside in ((upper, False), (lower, True)):
            b = polynomial_add(
                base,
                polynomial_scale(
                    polynomial_multiply(endpoint, endpoint, budget=budget),
                    -1,
                    budget=budget,
                ),
                budget=budget,
            )
            quartic = polynomial_add(
                polynomial_multiply(b, b, budget=budget), radial, budget=budget
            )
            controls = bernstein_coefficients(quartic, "simplex", 2, budget=budget)
            budget.reserve(len(controls))
            if any(value < 0 if outside else value > 0 for value in controls):
                return None
        return -band, band


def _torus_piece_bounds(
    boundary: _Boundary,
    profile: AnalyticImplicitProfile,
    owners: np.ndarray,
    parameters: np.ndarray,
    fields: np.ndarray,
    normals: np.ndarray,
    normal_low: np.ndarray,
    normal_high: np.ndarray,
    tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
    corner_memo: _TorusCornerMemo | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect box intervals with each exact piece's torus value and slope range.

    The full torus field ``f = |(rho - R, z)| - r`` is smooth off its axis and
    core circle, where its Hessian eigenvalues are ``0``, ``1/t`` and
    ``(rho - R)/(t rho)``. They lie between ``-1/rho`` and ``1/t``.
    An exact enclosing-ball squared radius ``v`` bounds the barycentric
    variance ``sum(w_i |p_i - x|^2)`` and puts each piece point within
    ``sqrt(v)`` of some corner. Corner enclosures therefore give positive
    floors on ``t`` and ``rho`` throughout the piece. Taylor's formula about
    ``x`` bounds its field below the corner interpolant by ``v/(2 t_floor)``
    and above it by ``v/(2 rho_floor)``; outside the core cylinder the latter
    error is zero. The same exact gradient's normal component is bounded
    directly as ``(n.p - R n_xy.(p_xy/rho))/t``; planar direction extrema
    occur at corner rays or an aligned/antialigned normal ray in their cone.

    Corner value enclosures come first: a corner beyond the tolerance already
    witnesses a violating facet point, so that row stops before its gradient,
    curvature and slope stage and keeps its coordinate-box intervals.

    If those field bounds remain inconclusive, the metered exact quartic
    band proof can intersect the entire piece with the unchanged requested
    band. It does not replace any corner, source or normal-orientation proof.
    Without a ledger the original enclosure path is retained.
    """
    deviation = np.maximum(np.abs(fields[:, 0]), np.abs(fields[:, 1]))
    rows = np.flatnonzero(
        (deviation > min(tolerance, profile.tube_radius)) | (normals[:, 0] <= 0.0)
    )
    center = _fraction_point(profile.center)
    major = Fraction(profile.major_radius)
    minor = Fraction(profile.radius)
    band = Fraction(min(tolerance, profile.tube_radius))
    tightened, oriented = fields.copy(), normals.copy()
    # A piece corner whose exact field enclosure lies beyond the tolerance is a
    # point of the facet farther than the tolerance from the source.
    witnessed = np.zeros(fields.shape[0], dtype=np.bool_)
    corner_rows: list[
        tuple[
            int,
            int,
            tuple[_RationalPoint, ...],
            list[tuple[Fraction, Fraction]],
            list[tuple[Fraction, Fraction]],
            list[tuple[Fraction, Fraction]],
        ]
    ] = []
    for row in rows.tolist():
        owner, weights = int(owners[row]), parameters[row]
        corners: list[_RationalPoint] = []
        values: list[tuple[Fraction, Fraction]] = []
        tubes: list[tuple[Fraction, Fraction]] = []
        axes: list[tuple[Fraction, Fraction]] = []
        for weights_corner in weights:
            if corner_memo is None:
                if budget is not None:
                    budget.reserve(_TORUS_SINGLE_CORNER_WORK)
                facet = (
                    boundary.exact_triangles[owner]
                    if boundary.exact_triangles is not None
                    else tuple(_fraction_point(point) for point in boundary.triangles[owner])
                )
                point, value, tube, axis = _torus_corner_bounds(
                    weights_corner, facet, center, major, minor
                )
            else:
                point, value, tube, axis = corner_memo.corner(owner, weights_corner)
            if value is None:
                break
            corners.append(point)
            values.append(value)
            if values[-1][0] > tolerance or values[-1][1] < -tolerance:
                witnessed[row] = True
            tubes.append(tube)
            axes.append(axis)
        if len(values) == 3 and not witnessed[row]:
            corner_rows.append((row, owner, tuple(corners), values, tubes, axes))
    if budget is not None:
        # Exact triangle radius, curvature floors and the closed-form
        # whole-piece normal derivative of each unwitnessed row.
        budget.reserve(_TORUS_SHAPE_WORK * len(corner_rows))
    for row, owner, corners, values, tubes, axes in corner_rows:
        variance = _triangle_variance_upper(corners)
        reach = Fraction(float(_sqrt_interval(variance)[1]))
        tube_floor = max(
            min(tube[0] for tube in tubes) - reach,
            _torus_tube_floor(corners, tubes, axes, major),
        )
        axis_floor = min(axis[0] for axis in axes) - reach
        if tube_floor <= 0 or axis_floor <= 0:
            continue
        lowest = min(value[0] for value in values)
        highest = max(value[1] for value in values)
        # Valid floors give valid field bounds; since t = f + r and
        # |rho - R| <= t on the piece, those bounds raise both floors again.
        for _ in range(3):
            lower_error = variance / (2 * tube_floor)
            negative_curvature = max(
                Fraction(0),
                min(1 / axis_floor, (major - axis_floor) / (tube_floor * axis_floor)),
            )
            upper_error = negative_curvature * variance / 2
            field_low = max(lowest - lower_error, tube_floor - minor)
            field_high = highest + upper_error
            tube_floor = max(tube_floor, minor + field_low)
            axis_floor = max(axis_floor, major - minor - field_high)
        lower_error = variance / (2 * tube_floor)
        negative_curvature = max(
            Fraction(0),
            min(1 / axis_floor, (major - axis_floor) / (tube_floor * axis_floor)),
        )
        upper_error = negative_curvature * variance / 2
        tightened[row, 0] = max(
            tightened[row, 0],
            _fraction_interval(max(lowest - lower_error, tube_floor - minor))[0],
        )
        tightened[row, 1] = min(
            tightened[row, 1], _fraction_interval(highest + upper_error)[1]
        )
        slope = _torus_normal_interval(
            corners, axes, major, tube_floor, minor + highest + upper_error,
            normal_low[owner], normal_high[owner],
        )
        oriented[row, 0] = max(oriented[row, 0], _fraction_interval(slope[0])[0])
        oriented[row, 1] = min(oriented[row, 1], _fraction_interval(slope[1])[1])
    if budget is not None:
        for row, _owner, corners, _values, _tubes, _axes in corner_rows:
            if tightened[row, 0] >= -band and tightened[row, 1] <= band:
                continue
            exact_band = _torus_quartic_band(
                corners, major, minor, band, budget=budget
            )
            if exact_band is not None:
                tightened[row, 0] = max(
                    tightened[row, 0], _fraction_interval(exact_band[0])[0]
                )
                tightened[row, 1] = min(
                    tightened[row, 1], _fraction_interval(exact_band[1])[1]
                )
    return tightened, oriented, witnessed


def _interval_product(
    first: tuple[Fraction, Fraction], second: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    products = tuple(left * right for left in first for right in second)
    return min(products), max(products)


def _length_upper(vector: _RationalPoint, /) -> Fraction:
    squared = sum((value * value for value in vector), Fraction(0))
    return Fraction(float(_sqrt_interval(squared)[1]))


def _triangle_variance_upper(corners: tuple[_RationalPoint, ...], /) -> Fraction:
    """Exact minimum enclosing-ball radius squared for three rational points.

    For any barycentric x, sum(w_i |p_i-x|^2) equals
    sum(w_i |p_i-c|^2)-|x-c|^2, hence is at most this radius squared.
    A right/obtuse triangle's longest-edge ball contains all three corners;
    an acute triangle uses its circumcircle, with radius^2=abc/(16 area^2).
    The obtuse branch also covers every collinear/zero-area triangle.
    """
    squared = tuple(
        sum((value * value for value in _difference(corners[i], corners[j])), Fraction(0))
        for i, j in ((0, 1), (1, 2), (0, 2))
    )
    return _barycentric_pair_upper(*squared)


def _barycentric_pair_upper(a: Fraction, b: Fraction, c: Fraction, /) -> Fraction:
    """Maximize a*w0*w1 + b*w1*w2 + c*w0*w2 on the closed simplex.

    A boundary maximum is max(0,a,b,c)/4. An interior maximum exists only
    when all three coefficients are positive and twice their maximum is
    below their sum; solving the two stationary equations then gives abc/D.
    This also covers signed coefficients, whose maxima lie on the boundary.
    """
    longest = max(a, b, c)
    if 2 * longest >= a + b + c:
        return max(Fraction(0), longest) / 4
    return a * b * c / (4 * a * b - (a + b - c) ** 2)


def _barycentric_quadratic_floor(
    values: tuple[Fraction, Fraction, Fraction],
    edges: list[Fraction],
    /,
) -> Fraction:
    """Minimize sum(w_i*values_i)-sum(w_i*w_j*edges_ij) exactly.

    A minimum on the compact triangle is at a vertex, at a stationary point
    of a convex edge restriction, or at an interior stationary point with
    positive-definite Hessian. A singular positive-semidefinite interior
    minimum extends to an edge, so it needs no separate candidate.
    """
    floor = min(values)
    for (i, j), edge in zip(((0, 1), (1, 2), (0, 2)), edges, strict=True):
        along = values[j] + edge - values[i]
        if edge > 0 and 0 < along < 2 * edge:
            floor = min(floor, values[j] - along * along / (4 * edge))
    a, b, c = edges
    cross = b + c - a
    determinant = 4 * b * c - cross * cross
    if b > 0 and c > 0 and determinant > 0:
        u, v = values[2] + c - values[0], values[2] + b - values[1]
        x = (2 * b * u - cross * v) / determinant
        y = (2 * c * v - cross * u) / determinant
        if x > 0 and y > 0 and x + y < 1:
            floor = min(floor, values[2] - (u * x + v * y) / 2)
    return floor


def _torus_tube_floor(
    corners: tuple[_RationalPoint, ...],
    tubes: list[tuple[Fraction, Fraction]],
    axes: list[tuple[Fraction, Fraction]],
    major: Fraction,
    /,
) -> Fraction:
    """Anisotropic lower bound on the same exact distance to the core circle.

    Put s=sum(w_i*rho_i). The norm identity gives
    s-rho(x)=sum(w_i*w_j*(|p_i,xy-p_j,xy|^2-(rho_i-rho_j)^2))/(s+rho(x)).
    Since the numerator terms are nonnegative and s+rho(x)<=2*rho_max,
    t(x)^2 >= sum(w_i*t_i^2) - sum(w_i*w_j*edge_error_ij), where each
    edge_error is |p_i-p_j|^2 - (R/rho_max) times that angular term.
    Interval endpoints only enlarge each error; the exact quadratic minimum
    retains all three corner values before a downward square root gives the
    whole-triangle floor.
    Unlike an isotropic Hessian error, this keeps toroidal curvature small.
    """
    rho_max = max(axis[1] for axis in axes)
    errors: list[Fraction] = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        dx, dy, dz = _difference(corners[i], corners[j])
        planar = dx * dx + dy * dy
        radial_difference = max(
            abs(axes[i][0] - axes[j][1]), abs(axes[i][1] - axes[j][0])
        )
        angular = max(Fraction(0), planar - radial_difference * radial_difference)
        errors.append(planar + dz * dz - major * angular / rho_max)
    squared = _barycentric_quadratic_floor(
        tuple(tube[0] * tube[0] for tube in tubes), errors
    )
    return (
        Fraction(0) if squared <= 0 else Fraction(float(_sqrt_interval(squared)[0]))
    )


def _corner_cone_contains(
    corners: tuple[_RationalPoint, ...], ray: tuple[Fraction, Fraction], /
) -> bool:
    """Test membership in the positive planar cone of the three corner rays."""
    x, y = ray
    for i, j in ((0, 1), (1, 2), (0, 2)):
        ax, ay = corners[i][:2]
        bx, by = corners[j][:2]
        cross = ax * by - ay * bx
        first, second = ax * y - ay * x, x * by - y * bx
        if cross > 0 and first >= 0 and second >= 0:
            return True
        if cross < 0 and first <= 0 and second <= 0:
            return True
        if cross == 0 and first == 0 and ax * x + ay * y >= 0:
            return True
    return False


def _torus_normal_interval(
    corners: tuple[_RationalPoint, ...],
    axes: list[tuple[Fraction, Fraction]],
    major: Fraction,
    tube_floor: Fraction,
    tube_ceiling: Fraction,
    normal_low: np.ndarray,
    normal_high: np.ndarray,
    /,
) -> tuple[Fraction, Fraction]:
    """Enclose the original exact torus gradient dotted with the facet normal.

    A linear form on a triangle has its extrema at corners. For a fixed
    planar normal m, m.p_xy/rho has extrema at corner rays or at +/-m if
    that ray is in their positive cone. The positive axis floor established
    by the caller excludes a cone spanning opposite rays through the axis.
    Use the midpoint of the full normal interval for that direction problem;
    its componentwise half-widths bound the remaining unit-direction error.
    Division by the positive whole-piece tube interval finishes the same
    closed-form derivative, without a Hessian-norm widening of corner slopes.
    """
    low = tuple(Fraction(float(value)) for value in normal_low)
    high = tuple(Fraction(float(value)) for value in normal_high)
    plane_low: list[Fraction] = []
    plane_high: list[Fraction] = []
    direction_low: list[Fraction] = []
    direction_high: list[Fraction] = []
    midpoint = ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2)
    for point, axis in zip(corners, axes, strict=True):
        products = tuple(
            (min(a * p, b * p), max(a * p, b * p))
            for a, b, p in zip(low, high, point, strict=True)
        )
        plane_low.append(sum((value[0] for value in products), Fraction(0)))
        plane_high.append(sum((value[1] for value in products), Fraction(0)))
        dot = midpoint[0] * point[0] + midpoint[1] * point[1]
        interval = _interval_product((dot, dot), (1 / axis[1], 1 / axis[0]))
        direction_low.append(interval[0])
        direction_high.append(interval[1])
    norm = Fraction(float(_sqrt_interval(midpoint[0] ** 2 + midpoint[1] ** 2)[1]))
    if _corner_cone_contains(corners, midpoint):
        direction_high.append(norm)
    if _corner_cone_contains(corners, (-midpoint[0], -midpoint[1])):
        direction_low.append(-norm)
    error = (high[0] - low[0] + high[1] - low[1]) / 2
    numerator = (
        min(plane_low) - major * (max(direction_high) + error),
        max(plane_high) - major * (min(direction_low) - error),
    )
    return _interval_product(numerator, (1 / tube_ceiling, 1 / tube_floor))


def _pieces(
    boundary: _Boundary,
    profile: AnalyticImplicitProfile,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
    retire_violated_facets: bool = False,
) -> _Pieces:
    """Subdivide facets into pieces until each is proved, violated or exhausted.

    ``retire_violated_facets`` stops subdividing every remaining piece of a facet
    once one of its pieces is violated. Per-facet decisions are unchanged, but
    the retired pieces no longer cover their facet, so only the facet-level
    bounds of ``implicit_facet_projection_bounds`` use it.
    """
    storage_scope = budget.live_storage() if budget is not None else nullcontext(None)
    with storage_scope as storage:
        if profile.family == "ring-torus":
            if storage is not None:
                # Memo owner, empty dict, profile atoms and one transient key.
                storage.set_bound(4096)
            memo = _TorusCornerMemo(
                boundary, profile, limits.maximum_subdivision_depth, budget, storage
            )
        else:
            memo = None
        try:
            return _pieces_impl(
                boundary,
                profile,
                tolerance,
                limits,
                budget=budget,
                retire_violated_facets=retire_violated_facets,
                corner_memo=memo,
            )
        finally:
            if memo is not None:
                memo.corners.clear()


def _pieces_impl(
    boundary: _Boundary,
    profile: AnalyticImplicitProfile,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
    retire_violated_facets: bool,
    corner_memo: _TorusCornerMemo | None,
) -> _Pieces:
    owners = np.arange(boundary.triangles.shape[0], dtype=np.int64)
    parameters = np.tile(np.eye(3, dtype=np.float64), (owners.size, 1, 1))
    depth = np.zeros(owners.size, dtype=np.int64)
    retired = np.zeros(owners.size, dtype=np.bool_)
    normal_low, normal_high = _normal_bounds(
        boundary.triangles, boundary.coordinate_bounds
    )
    terminals: list[tuple[np.ndarray, ...]] = []
    findings: list[MeshCertificateFinding] = []
    queries = 1  # Nominal profile reconstruction establishes an interior point.
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
        source = profile.field_bounds.boxes(low, high, budget=budget)
        queries += owners.size
        fields = np.stack((source.value_lower, source.value_upper), axis=1)
        if profile.family == "sphere":
            fields = _sphere_piece_fields(
                boundary, profile, owners, parameters, fields, tolerance, budget=budget
            )
        gradients = np.stack((source.gradient_lower, source.gradient_upper), axis=1)
        normals = _dot_bounds(
            normal_low[owners],
            normal_high[owners],
            source.gradient_lower,
            source.gradient_upper,
        )
        witnessed = np.zeros(owners.size, dtype=np.bool_)
        if profile.family == "ring-torus":
            fields, normals, witnessed = _torus_piece_bounds(
                boundary,
                profile,
                owners,
                parameters,
                fields,
                normals,
                normal_low,
                normal_high,
                tolerance,
                budget=budget,
                corner_memo=corner_memo,
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
            | witnessed
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
        if retire_violated_facets:
            retired[owners[violated]] = True
            selected &= ~retired[owners]
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


def implicit_facet_projection_bounds(
    triangles: np.ndarray,
    profile: AnalyticImplicitProfile,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Bound whole affine facets without changing their physical triangulation.

    This is preparation evidence, not a closed-manifold or degree certificate.
    The final projection independently recomputes those additional obligations.
    """
    count = triangles.shape[0]
    identifiers = np.arange(count, dtype=np.int64)
    vertices = np.arange(3 * count, dtype=np.int64).reshape((-1, 3))
    boundary = _Boundary(
        triangles, vertices, vertices, identifiers, np.zeros(count, dtype=np.int64), ()
    )
    pieces = _pieces(
        boundary, profile, tolerance, limits, budget=budget, retire_violated_facets=True
    )
    deviation = np.zeros(count, dtype=np.float64)
    np.maximum.at(deviation, pieces.owners, np.max(np.abs(pieces.fields), axis=1))
    oriented = np.ones(count, dtype=np.bool_)
    np.logical_and.at(oriented, pieces.owners, pieces.normals[:, 0] > 0.0)
    return deviation, oriented, pieces.queries - 1


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
            # Rational unit-circle coordinates fix the represented
            # torus core point. Positive radial motion keeps rho strictly > 0;
            # phi(t)=t*||end-start||-radius throughout this normal ray.
            t = Fraction(index, index + 1)
            cosine, sine = (1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)
            major = Fraction(profile.major_radius)
            start = (center[0] + major * cosine, center[1] + major * sine, center[2])
            direction = (cosine, sine, Fraction(index * index + 1))
        case _:
            raise ValueError(
                f"ImplicitProjectionCover.profile.family must provide a supported normal fiber; got {profile.family!r}"
            )
    end = tuple(start[axis] + 2 * radius * direction[axis] for axis in range(3))
    return start, (end[0], end[1], end[2])


def _fiber(
    boundary: _Boundary,
    profile: AnalyticImplicitProfile,
    forward: float,
    limits: MeshCertificateLimits,
    remaining_queries: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
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
            # Rational directions avoid the finite exceptional edge and plane
            # sets; every trial remains subject to the declared work limit.
            continue
        endpoint_source = profile.field_bounds.boxes(
            endpoint_boxes[:, :, 0], endpoint_boxes[:, :, 1], budget=budget
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


@checked
def compute_implicit_projection_cover(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    profile: AnalyticImplicitProfile,
    /,
    *,
    tolerance: float,
    limits: MeshCertificateLimits,
    budget: CoordinateEnclosureBudget | None = None,
) -> ImplicitProjectionCoverageEvidence:
    """Recompute source-projection bounds from coordinate maps and nominal source facts."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"ImplicitProjectionCover.tolerance must be finite and nonnegative; got {tolerance}"
        )
    with coordinate_certificate_request(limits, budget=budget) as budget:
        with budget.activate():
            profile.require_bound(
                profile.geometry, profile.coordinate_contract, budget=budget
            )
            established = AnalyticImplicitProfile(
                profile.geometry,
                profile.coordinate_contract,
                tube_radius=profile.tube_radius,
                cover_radius=profile.cover_radius,
                source_id=profile.source_id,
                source_revision=profile.source_revision,
                budget=budget,
            )
            if established.profile_id != profile.profile_id:
                raise ValueError(
                    "projection_cover.profile must match recomputed source facts; got stale cached facts"
                )
            scope_resource: CoordinateEnclosureResourceError | None = None
            try:
                with budget.measurement() as scope_measurement:
                    scope = coordinate_scope(
                        mesh, geometry, budget=budget
                    ).coordinate_scope
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
                    with budget.measurement() as scope_measurement:
                        boundary = _source_affine_boundary(
                            mesh, geometry, boundary, budget=budget
                        )
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
                            ("limit", int(scope_resource.limit)),
                            ("requested", int(scope_resource.requested)),
                        ),
                        achieved=(
                            ("completed", int(scope_resource.completed)),
                            (
                                "source_expression_work_units",
                                scope_measurement.work_units,
                            ),
                            (
                                "source_expression_peak_bytes",
                                scope_measurement.peak_bytes_upper,
                            ),
                            (
                                "source_expression_native_charged_work_units",
                                scope_measurement.native_charged_work_units,
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
                    finding.check == "projection_actual_affine_maps"
                    for finding in findings
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
                pieces = _pieces(boundary, established, tolerance, limits, budget=budget)
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
                    min(
                        limits.maximum_source_samples, limits.maximum_distance_evaluations
                    )
                    - pieces.queries
                )
                fiber = _fiber(
                    boundary, established, pieces.upper, limits, remaining, budget=budget
                )
                findings.extend(fiber.findings)
            return ImplicitProjectionCoverageEvidence(
                binding,
                established,
                tuple(findings),
                facet_evidence=ProjectionFacetEvidence(
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
                ),
                boundary_fiber=ProjectionFiberEvidence(
                    fiber_endpoints=fiber.endpoints,
                    fiber_value_bounds=fiber.values,
                    fiber_gradient_bounds=fiber.gradient,
                    fiber_candidate_facets=fiber.candidates,
                    fiber_crossing_classes=fiber.classes,
                    fiber_crossing_parameters=fiber.parameters,
                ),
                forward_upper=pieces.upper,
                forward_lower=pieces.lower,
                query_count=pieces.queries + fiber.queries,
                ray_test_count=fiber.tests,
            )


__all__ = ["compute_implicit_projection_cover"]
