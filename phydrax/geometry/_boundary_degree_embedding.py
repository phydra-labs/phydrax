#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Boundary-degree embedding of mapped volume meshes.

The theorem, its premises and their evidence are documented on
``MappedBoundaryDegreeEvidence``. This module proves the premises exactly or
with outward-rounded Bernstein enclosures, records them, and reports whether
the theorem decided the mesh. An unproven premise never accepts the mesh: the
caller falls back to pairwise cell contacts. All polynomial work is charged to
the active coordinate-enclosure ledger and every subdivision piece, candidate
pair and ray test to the embedding state's counters.
"""

from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
from math import factorial
from typing import TYPE_CHECKING

import numpy as np

from ..discretization._coordinate_enclosure import (
    _COORDINATE_BUDGET,
    add,
    affine_arguments,
    axes,
    constant,
    derivative,
    Expression,
    expression_reference_evaluate,
    ExpressionComposition as ArgumentComposition,
    multiply,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    restrict_chart_expressions,
    scale,
    sum_polynomials,
)
from ..discretization._reference_cell import reference_cell_topology


if TYPE_CHECKING:
    from ..discretization._cell_mesh import CellMesh
    from ._mapped_embedding import _Cell
    from ._mesh_certificates import (
        _EmbeddingState,
        MappedBoundaryDegreeStatus,
        MeshCertificateLimits,
    )


type _Point = tuple[Fraction, ...]

_VOLUME_KINDS = {
    2: ("triangle", "quadrilateral"),
    3: ("tetrahedron", "prism", "hexahedron"),
}
_FACE_KINDS = {2: "interval", 3: "triangle", 4: "quadrilateral"}
# Triangles of the quadrilateral chart split along its 0-2 diagonal.
_QUAD_HALVES = (
    ((0, 1, 2), np.asarray(((1.0, 1.0), (0.0, 1.0)), dtype=np.float64)),
    ((0, 2, 3), np.asarray(((1.0, 0.0), (1.0, 1.0)), dtype=np.float64)),
)


def _shared_entity_conformity(
    cells: list[_Cell], /
) -> tuple[int, dict[int, _Point], tuple[int, int] | None]:
    """Exact identity of every shared vertex image and edge restriction.

    Facet traces are proved by ``_trace_continuity``; vertices and (in 3-D)
    edges may be shared by cells that share no facet, so they are compared
    across all incident cells in one canonical chart (smaller global id first).
    """
    budget = _COORDINATE_BUDGET.get()
    images: dict[int, tuple[_Point, int]] = {}
    edges: dict[tuple[int, int], tuple[tuple[Expression, ...], int]] = {}
    matched = 0
    for index, cell in enumerate(cells):
        for vertex, point in zip(cell.vertices, cell.reference_vertices, strict=True):
            image = tuple(
                expression_reference_evaluate(value, point, cell.domain)
                for value in cell.coordinates
            )
            prior = images.get(vertex)
            if prior is None:
                if budget is not None:
                    budget.reserve(1, 256 + 128 * len(image))
                images[vertex] = (image, index)
            elif prior[0] != image:
                return matched, {}, (prior[1], index)
            else:
                matched += 1
        if cell.dimension != 3:
            continue
        reference = np.asarray(
            reference_cell_topology(cell.kind).vertices, dtype=np.float64
        )
        for first, second in reference_cell_topology(cell.kind).entities[1]:
            if cell.vertices[first] > cell.vertices[second]:
                first, second = second, first
            key = (cell.vertices[first], cell.vertices[second])
            restricted = restrict_chart_expressions(
                cell.coordinates,
                cell.kind,
                "interval",
                reference[first],
                (reference[second] - reference[first])[:, None],
            )
            prior_edge = edges.get(key)
            if prior_edge is None:
                if budget is not None:
                    budget.reserve(1, 256)
                edges[key] = (restricted, index)
            elif prior_edge[0] != restricted:
                return matched, {}, (prior_edge[1], index)
            else:
                matched += 1
    return matched, {vertex: value[0] for vertex, value in images.items()}, None


def _boundary_face(cell: _Cell, vertices: np.ndarray, /) -> _Cell | None:
    """Exact outward-oriented restriction of a cell map to one boundary facet."""
    from ._mapped_embedding import _Cell, _domain, _vertices

    kind = _FACE_KINDS[vertices.size]
    reference = np.asarray(reference_cell_topology(cell.kind).vertices, dtype=np.float64)
    corners = reference[[cell.vertices.index(int(vertex)) for vertex in vertices]]
    columns = (1,) if vertices.size == 2 else (1, 2) if vertices.size == 3 else (1, 3)
    coordinates = restrict_chart_expressions(
        cell.coordinates,
        cell.kind,
        kind,
        corners[0],
        (corners[list(columns)] - corners[0]).T,
    )
    if any(isinstance(value, RationalPolynomial) for value in coordinates):
        return None
    return _Cell(
        coordinates,
        _domain(kind),
        kind,
        cell.dimension - 1,
        tuple(int(vertex) for vertex in vertices),
        _vertices(kind),
    )


def _integral(polynomial: Polynomial, domain: str, /) -> Fraction:
    """Exact reference-domain integral of a polynomial in one or two variables."""
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(4 * len(polynomial))
    total = Fraction(0)
    for index, value in polynomial.items():
        if domain == "simplex":
            a, b = index
            total += value * Fraction(factorial(a) * factorial(b), factorial(a + b + 2))
        else:
            weight = Fraction(1)
            for exponent in index:
                weight /= exponent + 1
            total += value * weight
    return total


def _face_measure(face: _Cell, /) -> Fraction:
    """Exact ``d`` times the signed volume contribution of one oriented face."""
    coordinates = tuple(value for value in face.coordinates if isinstance(value, dict))
    if face.dimension == 1:
        x, y = coordinates
        integrand = add(
            multiply(x, derivative(y, 0)), scale(multiply(y, derivative(x, 0)), -1)
        )
    else:
        u = tuple(derivative(value, 0) for value in coordinates)
        v = tuple(derivative(value, 1) for value in coordinates)
        integrand = sum_polynomials(
            tuple(
                multiply(
                    coordinates[axis],
                    add(
                        multiply(u[(axis + 1) % 3], v[(axis + 2) % 3]),
                        scale(multiply(u[(axis + 2) % 3], v[(axis + 1) % 3]), -1),
                    ),
                )
                for axis in range(3)
            )
        )
    return _integral(integrand, face.domain)


def _affine_deviation(
    coordinates: tuple[Polynomial, ...], corners: tuple[_Point, ...], domain: str, /
) -> Fraction:
    """Outward bound of ``max |F - L|`` for the affine interpolant ``L`` of corners."""
    dimension = len(corners) - 1
    variables = axes(dimension)
    bound = Fraction(0)
    for axis, value in enumerate(coordinates):
        interpolant = sum_polynomials(
            (
                constant(corners[0][axis], dimension),
                *(
                    scale(variables[k], corners[k + 1][axis] - corners[0][axis])
                    for k in range(dimension)
                ),
            )
        )
        lower, upper = polynomial_bounds(
            add(value, scale(interpolant, -1)), domain, dimension
        )
        bound = max(bound, Fraction(-lower), Fraction(upper))
    return bound


def _proxy_simplices(
    face: _Cell, images: dict[int, _Point], /
) -> tuple[list[tuple[int, ...]], Fraction]:
    """Affine proxy simplices (global vertex rows) and their certified deviation."""
    coordinates = tuple(value for value in face.coordinates if isinstance(value, dict))
    if face.kind != "quadrilateral":
        corners = tuple(images[vertex] for vertex in face.vertices)
        return [face.vertices], _affine_deviation(coordinates, corners, face.domain)
    rows: list[tuple[int, ...]] = []
    deviation = Fraction(0)
    for local, matrix in _QUAD_HALVES:
        composition = ArgumentComposition(
            affine_arguments(np.zeros((2,), dtype=np.float64), matrix)
        )
        half = tuple(composition(value) for value in coordinates)
        row = tuple(face.vertices[k] for k in local)
        rows.append(row)
        deviation = max(
            deviation,
            _affine_deviation(
                tuple(value for value in half if isinstance(value, dict)),
                tuple(images[vertex] for vertex in row),
                "simplex",
            ),
        )
    return rows, deviation


def _outside_tube(point: _Point, simplex: tuple[_Point, ...], radius: Fraction) -> bool:
    """Exact ``||point - x||_inf > radius`` for every ``x`` in the simplex hull box."""
    return any(
        value > max(corner[axis] for corner in simplex) + radius
        or value < min(corner[axis] for corner in simplex) - radius
        for axis, value in enumerate(point)
    )


def _shell_degrees(
    state: _EmbeddingState,
    faces: list[_Cell],
    rows: np.ndarray,
    labels: np.ndarray,
    images: dict[int, _Point],
    limits: MeshCertificateLimits,
    /,
) -> tuple[int, ...] | str:
    """Exterior degree ``a_j + c_j - 1`` of every shell, or the unproven premise."""
    from ._exact_polyhedral_geometry import coordinate_integers
    from ._mesh_certificates import (
        _ray_crossings_2d,
        _ray_crossings_3d,
        _shell_representatives,
    )

    shells = int(np.max(labels)) + 1
    representative = _shell_representatives(rows, labels, shells)
    owner: dict[int, int] = {}
    for face, shell in zip(faces, labels.tolist(), strict=True):
        for vertex in face.vertices:
            if owner.setdefault(vertex, shell) != shell:
                return "shell_vertex_disjointness"
    if np.any(representative < 0):
        return "shell_vertex_disjointness"
    simplices: list[list[tuple[int, ...]]] = [[] for _ in range(shells)]
    deviations = [Fraction(0)] * shells
    measures = [Fraction(0)] * shells
    for face, shell in zip(faces, labels.tolist(), strict=True):
        proxies, deviation = _proxy_simplices(face, images)
        simplices[shell].extend(proxies)
        deviations[shell] = max(deviations[shell], deviation)
        measures[shell] += _face_measure(face)
    if any(measure == 0 for measure in measures):
        return "degenerate_shell"
    pieces = sum(len(value) for value in simplices)
    if shells * pieces > limits.maximum_ray_tests - state.ray_tests:
        return "exterior_degree_capacity"
    vertices = sorted({vertex for group in simplices for row in group for vertex in row})
    position = {vertex: index for index, vertex in enumerate(vertices)}
    integers, _ = coordinate_integers(
        np.asarray([images[vertex] for vertex in vertices], dtype=object)
    )
    budget = _COORDINATE_BUDGET.get()
    degrees = []
    for shell in range(shells):
        point = images[int(representative[shell])]
        origin = integers[position[int(representative[shell])]]
        winding = 0
        for other in range(shells):
            if other == shell:
                continue
            count = len(simplices[other])
            if budget is not None:
                budget.reserve(27 * count)
            # The straight homotopy from the affine proxy to the curved shell
            # moves every point by at most the certified deviation, so it never
            # meets ``point`` when every proxy simplex stays strictly farther.
            if not all(
                _outside_tube(
                    point, tuple(images[vertex] for vertex in row), deviations[other]
                )
                for row in simplices[other]
            ):
                return "shell_homotopy_separation"
            corners = integers[
                np.asarray(
                    [[position[vertex] for vertex in row] for row in simplices[other]],
                    dtype=np.int64,
                )
            ]
            state.ray_tests += count
            crossings, decided = (
                _ray_crossings_3d(origin, corners)
                if corners.shape[1] == 3
                else _ray_crossings_2d(origin, corners)
            )
            if not decided:
                return "shell_winding_predicates"
            winding += crossings
        degrees.append(winding + (1 if measures[shell] > 0 else 0) - 1)
    return tuple(degrees)


type _Outcome = tuple[MappedBoundaryDegreeStatus, str | None, tuple[int, ...], int | None]


def certify_boundary_degree_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    cells: list[_Cell],
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
) -> bool:
    """Decide a mapped volume mesh by the boundary-degree theorem.

    Returns ``True`` when the theorem decided the mesh (embedded, or a proved
    violation recorded as a finding) and ``False`` when a premise is unproven
    and the caller must run its pairwise cell-contact proof.
    """
    from ._mesh_certificates import MappedBoundaryDegreeEvidence

    dimension = mesh.topological_dimension
    if (
        dimension != mesh.ambient_dimension
        or dimension not in _VOLUME_KINDS
        or not state.clean
        or not state.closed_cell_orientation
    ):
        return False
    record = {
        "oriented_cell_count": len(cells),
        "conforming_entity_count": 0,
        "boundary_facet_count": 0,
        "boundary_pair_count": 0,
        "tangent_cone_contact_count": 0,
        "shell_count": 0,
    }
    status, premise, degrees, maximum = _decide(
        state, mesh, cells, cell_ids, limits, record
    )
    state.boundary_degree = MappedBoundaryDegreeEvidence(
        status,
        premise,
        **record,
        shell_exterior_degrees=degrees,
        maximum_degree=maximum,
    )
    if status == "premise_unproven":
        return False
    state.checks.append("mapped_boundary_degree")
    return True


def _decide(
    state: _EmbeddingState,
    mesh: CellMesh,
    cells: list[_Cell],
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    record: dict[str, int],
    /,
) -> _Outcome:
    from ._mesh_certificates import _Boundary, _facet_entity_ids, _mesh_facets, _shells

    dimension = mesh.topological_dimension
    if any(cell.kind not in _VOLUME_KINDS[dimension] for cell in cells):
        return "premise_unproven", "boundary_chart_kind", (), None
    # (b) Exact conformity of every shared vertex and edge; facets were proved
    # by the caller's trace continuity.
    matched, images, mismatch = _shared_entity_conformity(cells)
    record["conforming_entity_count"] = matched
    if mismatch is not None:
        state.add("mapped_trace_mismatch", "violated", "cell", cell_ids[list(mismatch)])
        return "violated", "mapped_shared_entity_conformity", (), None
    # (c) Boundary facets: exact outward restrictions and ridge-linked shells.
    facets = _mesh_facets(mesh)
    occurrences = np.flatnonzero(facets.boundary)
    rows = facets.rows[occurrences]
    rows = rows[:, : int(np.max(np.sum(rows >= 0, axis=1), initial=1))]
    owners = facets.cells[occurrences]
    state.boundary_facets = rows.shape[0]
    record["boundary_facet_count"] = rows.shape[0]
    if not rows.shape[0]:
        return "premise_unproven", "boundary_facets", (), None
    faces = []
    for owner, row in zip(owners.tolist(), rows, strict=True):
        face = _boundary_face(cells[owner], row[row >= 0])
        if face is None:
            return "premise_unproven", "boundary_rational_expression", (), None
        faces.append(face)
    entities = _facet_entity_ids(mesh, rows)
    labels, unbalanced = _shells(
        _Boundary(rows, entities, rows, np.arange(rows.shape[0])), dimension
    )
    state.shells = unbalanced.size
    record["shell_count"] = unbalanced.size
    if np.any(unbalanced):
        return "premise_unproven", "boundary_shell_manifold", (), None
    contact = _boundary_injectivity(state, faces, owners, cell_ids, limits, record)
    if contact is not None:
        return contact
    # (d) Exterior shell degrees. One shell has no other winding and its
    # covered side carries degree at least one, so its exterior degree is 0.
    shell_degrees = (
        (0,)
        if unbalanced.size == 1
        else _shell_degrees(state, faces, rows, labels, images, limits)
    )
    if isinstance(shell_degrees, str):
        return "premise_unproven", shell_degrees, (), None
    state.checks.extend(
        (
            "mapped_closed_cell_orientation",
            "mapped_shared_entity_conformity",
            "mapped_boundary_injectivity",
            "mapped_shell_exterior_degree",
        )
    )
    if any(shell_degrees):
        state.add(
            "mapped_boundary_degree",
            "violated",
            "facet",
            entities[np.isin(labels, np.flatnonzero(np.asarray(shell_degrees)))],
        )
        return (
            "violated",
            "mapped_shell_exterior_degree",
            shell_degrees,
            max(0, 1 + max(shell_degrees)),
        )
    return "embedded", None, shell_degrees, 1


def _boundary_injectivity(
    state: _EmbeddingState,
    faces: list[_Cell],
    owners: np.ndarray,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    record: dict[str, int],
    /,
) -> _Outcome | None:
    """Premise (c): boundary facets of different cells meet only where shared."""
    from ._mapped_embedding import _bounds, _pair_decision
    from ._mesh_certificates import _candidate_pairs
    from ._tangent_cone_contact import tangent_cone_contact

    budget = _COORDINATE_BUDGET.get()
    boxes = [
        _bounds(
            face,
            np.zeros((face.dimension,), dtype=np.float64),
            np.eye(face.dimension, dtype=np.float64),
        )
        for face in faces
    ]
    lower = np.stack([box[0] for box in boxes])
    upper = np.stack([box[1] for box in boxes])
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return "premise_unproven", "boundary_bound_range", (), None
    first, second, exceeded = _candidate_pairs(
        lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
    )
    state.candidate_pairs += first.size
    if exceeded:
        return "premise_unproven", "boundary_candidate_pair_budget", (), None
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        # The proved closed-cell injectivity of the owner separates its own
        # facets beyond their shared entities.
        if owners[a] == owners[b]:
            continue
        record["boundary_pair_count"] += 1
        remaining = limits.maximum_subdivision_pieces - state.subdivision_pieces
        with nullcontext() if budget is None else budget.temporary_scope():
            contact = (
                tangent_cone_contact(faces[a], faces[b], limits, remaining)
                if set(faces[a].vertices).intersection(faces[b].vertices)
                else None
            )
            decision, work = "undecided", 0
            if contact is not None:
                proved, work = contact
                if proved:
                    record["tangent_cone_contact_count"] += 1
                    decision = "separated"
            if decision != "separated":
                decision, extra = _pair_decision(
                    faces[a], faces[b], limits, remaining - work, tangent_cone=False
                )
                work += extra
        state.subdivision_pieces += work
        if decision == "separated":
            continue
        if decision in ("overlap", "mapped_trace_mismatch"):
            # Codimension-one overlap witnesses are transverse crossings, so the
            # two owning closed cells overlap in an open set.
            state.add(
                "mapped_cell_overlap" if decision == "overlap" else decision,
                "violated",
                "cell",
                cell_ids[[int(owners[a]), int(owners[b])]],
            )
            return "violated", "mapped_boundary_injectivity", (), None
        return "premise_unproven", decision, (), None
    return None
