#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Boundary-degree embedding of mapped volume meshes.

The theorem, its premises and their evidence are documented on
``MappedBoundaryDegreeEvidence``. This module proves the premises exactly or
with outward-rounded Bernstein enclosures, records them, and reports whether
the theorem decided the mesh. An unproven hypothesis never accepts the mesh: the
caller falls back to pairwise cell contacts. All polynomial work is charged to
the active coordinate-enclosure ledger and every subdivision piece, candidate
pair and ray test to the embedding state's counters.

Mesh certificate checks
-----------------------
[
{"checks":["mapped_boundary_degree"],"owner":"phydrax.geometry._boundary_degree_embedding.certify_boundary_degree_embedding","premises":[{"tag":"N","statement":"The bound validity certificate proves positive orientation and each closed cell map is injective; boundary pair proofs admit only shared contacts."},{"tag":"X","statement":"Every shared entity has one exact restriction and every boundary shell is ridge-linked and balanced."},{"tag":"T","statement":"Cancellation of identical interior traces makes the positive cell preimage count equal to the oriented boundary winding degree."}],"conclusion":"Zero shell exterior degrees prove global injectivity; a nonzero certified shell exterior degree contradicts degree-one embedding.","citation":{"kind":"mathematical","reference":"Lipman, Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing","locator":"Theorems 1 and 4; Section 3, cycle degree","url":"https://arxiv.org/abs/1310.0955"},"failure_statuses":["violated"]},
{"checks":["mapped_closed_cell_orientation"],"owner":"phydrax.geometry._boundary_degree_embedding._decide","premises":[{"tag":"N","statement":"The bound closed-cell validity certificate proves a strictly positive determinant on each full source cell."}],"conclusion":"The positive local-degree orientation hypothesis is recorded for the boundary-degree route.","citation":{"kind":"mathematical","reference":"Hatcher, Algebraic Topology","locator":"Section 2.2, Proposition 2.30 (sum of local degrees)","url":"https://pi.math.cornell.edu/~hatcher/AT/AT.pdf"},"failure_statuses":[]},
{"checks":["mapped_shared_entity_conformity"],"owner":"phydrax.geometry._boundary_degree_embedding._shared_entity_conformity","premises":[{"tag":"X","statement":"Every repeated vertex image and every repeated oriented edge restriction is identical across its incident cells; facet identity was checked by trace_continuity."}],"conclusion":"The source coordinate action is well-defined on the shared vertex, edge and facet skeleton.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._boundary_degree_embedding._shared_entity_conformity","url":""},"failure_statuses":[]},
{"checks":["mapped_shared_entity_conformity"],"owner":"phydrax.geometry._boundary_degree_embedding._decide","premises":[{"tag":"X","statement":"A repeated source vertex or edge has different exact images or restrictions."}],"conclusion":"The boundary-degree route reports violated shared-entity conformity and emits mapped_trace_mismatch.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._boundary_degree_embedding._decide","url":""},"failure_statuses":["violated"]},
{"checks":["mapped_trace_mismatch"],"owner":"phydrax.geometry._boundary_degree_embedding._shared_entity_conformity","premises":[{"tag":"X","statement":"Two incident cells disagree on an exact shared vertex image or oriented edge expression."}],"conclusion":"The mapped mesh has a nonconforming shared coordinate action.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._boundary_degree_embedding._shared_entity_conformity","url":""},"failure_statuses":["violated"]},
{"checks":["mapped_boundary_injectivity"],"owner":"phydrax.geometry._boundary_degree_embedding._boundary_injectivity","premises":[{"tag":"N","statement":"All boundary candidate pairs are separated by certified full-image or tangent-cone contact proofs."},{"tag":"X","statement":"Facets of one owner inherit its closed-cell injectivity and facets of different owners admit only their shared entities."}],"conclusion":"The union of mapped boundary facets is injective modulo its shared mesh entities.","citation":{"kind":"mathematical","reference":"Rockafellar, Convex Analysis","locator":"Section 11, separation theorems","url":"https://doi.org/10.1515/9781400873173"},"failure_statuses":[]},
{"checks":["mapped_boundary_injectivity"],"owner":"phydrax.geometry._boundary_degree_embedding._decide","premises":[{"tag":"N","statement":"A boundary pair has a certified transverse interior crossing or an exact shared-trace mismatch."}],"conclusion":"The boundary-degree route reports a violated boundary-injectivity hypothesis.","citation":{"kind":"mathematical","reference":"R. E. Moore, A Test for Existence of Solutions to Nonlinear Systems (1977)","locator":"Krawczyk existence test, pp. 611-615","url":"https://doi.org/10.1137/0714040"},"failure_statuses":["violated"]},
{"checks":["mapped_cell_overlap"],"owner":"phydrax.geometry._boundary_degree_embedding._boundary_injectivity","premises":[{"tag":"N","statement":"A strict inclusion proves a transverse interior crossing of two boundary facets."},{"tag":"X","statement":"The facets have different closed-cell owners."},{"tag":"T","statement":"The locally embedded oriented cells meet in an open physical set across the transverse boundary crossing."}],"conclusion":"The two owning mapped cells overlap beyond shared reference entities.","citation":{"kind":"mathematical","reference":"R. E. Moore, A Test for Existence of Solutions to Nonlinear Systems (1977)","locator":"Krawczyk existence test, pp. 611-615","url":"https://doi.org/10.1137/0714040"},"failure_statuses":["violated"]},
{"checks":["mapped_shell_exterior_degree"],"owner":"phydrax.geometry._boundary_degree_embedding._shell_degrees","premises":[{"tag":"N","statement":"Each affine shell proxy stays strictly outside the certified curved-to-affine deviation tube of every shell representative."},{"tag":"X","statement":"Exact ray crossings give the winding of the other shells; the exact signed shell measure selects the covered side."},{"tag":"T","statement":"Winding degree is invariant under a homotopy that avoids the query point."}],"conclusion":"Every recorded exterior degree equals other-shell winding plus the shell covered-side orientation minus one.","citation":{"kind":"mathematical","reference":"Lipman, Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing","locator":"Theorems 1 and 4; Section 3, cycle degree","url":"https://arxiv.org/abs/1310.0955"},"failure_statuses":[]},
{"checks":["mapped_shell_exterior_degree"],"owner":"phydrax.geometry._boundary_degree_embedding._decide","premises":[{"tag":"N","statement":"At least one certified shell exterior degree is nonzero."},{"tag":"T","statement":"An injective positively oriented closed-cell mesh has degree zero on each shell exterior side."}],"conclusion":"The boundary-degree route records a violated degree-one embedding hypothesis.","citation":{"kind":"mathematical","reference":"Lipman, Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing","locator":"Theorems 1 and 4; Section 3, cycle degree","url":"https://arxiv.org/abs/1310.0955"},"failure_statuses":["violated"]},
{"checks":["degenerate_shell"],"owner":"phydrax.geometry._boundary_degree_embedding._shell_degrees","premises":[{"tag":"N","statement":"The exact signed measure of a boundary shell is zero."}],"conclusion":"The shell orientation hypothesis is premise_unproven; another contact route may still certify the mesh.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._boundary_degree_embedding._shell_degrees","url":""},"failure_statuses":["premise_unproven"]},
{"checks":["exterior_degree_capacity"],"owner":"phydrax.geometry._boundary_degree_embedding._shell_degrees","premises":[{"tag":"X","statement":"The number of shell-to-proxy ray tests exceeds the remaining maximum_ray_tests allowance."}],"conclusion":"The exterior-degree hypothesis is premise_unproven; another contact route may still certify the mesh.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._boundary_degree_embedding._shell_degrees","url":""},"failure_statuses":["premise_unproven"]}
]
End mesh certificate checks
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import assert_never, final

import equinox as eqx
import numpy as np

import phydrax.geometry._affine_mapped_contact as affine_mapped_contact_owner
import phydrax.geometry._embedding_cells as embedding_cells_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._exact_polyhedral_geometry as exact_polyhedral_geometry_owner
import phydrax.geometry._global_embedding as mesh_certificates_owner
import phydrax.geometry._mapped_embedding as mapped_embedding_owner

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    add,
    affine_arguments,
    axes,
    BernsteinDomain,
    constant,
    derivative,
    Expression,
    expression_reference_evaluate,
    expression_sequence_equal,
    ExpressionComposition as ArgumentComposition,
    multiply,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    restrict_chart_expressions,
    scale,
    sum_polynomials,
)
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked
from ._embedding_cells import Cell
from ._embedding_state import ContactDecision, EmbeddingState, MeshCertificateLimits
from ._global_embedding import MappedBoundaryDegreeStatus


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


@final
class _IncidentCells(StrictModule, NonTrainableState):
    first: int = eqx.field(static=True)
    second: int = eqx.field(static=True)


@final
class _EntityConformity(StrictModule, NonTrainableState):
    count: int = eqx.field(static=True)
    images: dict[int, _Point]
    mismatch: _IncidentCells | None


def _shared_entity_conformity(
    cells: list[Cell], /, *, budget: CoordinateEnclosureBudget
) -> _EntityConformity:
    """Compare shared vertex images and edge coefficient restrictions across cells.

    Facet traces are proved by ``_trace_continuity``; vertices and (in 3-D)
    edges may be shared by cells that share no facet, so they are compared
    across all incident cells in one standard reference chart (smaller global id first).
    """

    images: dict[int, tuple[_Point, int]] = {}
    edges: dict[tuple[int, int], tuple[tuple[Expression, ...], int]] = {}
    matched = 0
    for index, cell in enumerate(cells):
        for vertex, point in zip(cell.vertices, cell.reference_vertices, strict=True):
            image = tuple(
                rational_coefficient(
                    expression_reference_evaluate(
                        value, point, cell.domain, budget=budget
                    ),
                    role="boundary_degree.vertex_image",
                )
                for value in cell.coordinates
            )
            prior = images.get(vertex)
            if prior is None:
                budget.reserve(1, 256 + 128 * len(image))
                images[vertex] = (image, index)
            elif prior[0] != image:
                return _EntityConformity(matched, {}, _IncidentCells(prior[1], index))
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
                budget=budget,
            )
            prior_edge = edges.get(key)
            if prior_edge is None:
                budget.reserve(1, 256)
                edges[key] = (restricted, index)
            elif not expression_sequence_equal(prior_edge[0], restricted, budget=budget):
                return _EntityConformity(
                    matched, {}, _IncidentCells(prior_edge[1], index)
                )
            else:
                matched += 1
    return _EntityConformity(
        matched, {vertex: value[0] for vertex, value in images.items()}, None
    )


def _boundary_face(
    cell: embedding_cells_owner.Cell,
    vertices: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> embedding_cells_owner.Cell | None:
    """Restrict a cell map to its outward-oriented facet using rational coefficients."""

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
        budget=budget,
    )
    if any(isinstance(value, RationalPolynomial) for value in coordinates):
        return None
    return embedding_cells_owner.Cell(
        coordinates,
        embedding_cells_owner.domain(kind),
        kind,
        cell.dimension - 1,
        tuple(int(vertex) for vertex in vertices),
        embedding_cells_owner.vertices(kind),
    )


def _integral(
    polynomial: Polynomial,
    domain: BernsteinDomain,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Integrate a reference-domain polynomial using Fraction arithmetic."""

    budget.reserve(4 * len(polynomial))
    total = Fraction(0)
    for index, value in polynomial.items():
        value = rational_coefficient(value, role="boundary_degree.integrand")
        if domain == "simplex":
            a, b = index
            total += value * Fraction(factorial(a) * factorial(b), factorial(a + b + 2))
        else:
            weight = Fraction(1)
            for exponent in index:
                weight /= exponent + 1
            total += value * weight
    return total


def _face_measure(face: Cell, /, *, budget: CoordinateEnclosureBudget) -> Fraction:
    """Return ``d`` times a face's signed volume contribution using Fraction arithmetic."""
    coordinates = tuple(value for value in face.coordinates if isinstance(value, dict))
    if face.dimension == 1:
        x, y = coordinates
        integrand = add(
            multiply(x, derivative(y, 0, budget=budget), budget=budget),
            scale(
                multiply(y, derivative(x, 0, budget=budget), budget=budget),
                -1,
                budget=budget,
            ),
            budget=budget,
        )
    else:
        u = tuple(derivative(value, 0, budget=budget) for value in coordinates)
        v = tuple(derivative(value, 1, budget=budget) for value in coordinates)
        integrand = sum_polynomials(
            tuple(
                multiply(
                    coordinates[axis],
                    add(
                        multiply(u[(axis + 1) % 3], v[(axis + 2) % 3], budget=budget),
                        scale(
                            multiply(u[(axis + 2) % 3], v[(axis + 1) % 3], budget=budget),
                            -1,
                            budget=budget,
                        ),
                        budget=budget,
                    ),
                    budget=budget,
                )
                for axis in range(3)
            ),
            budget=budget,
        )
    return _integral(integrand, face.domain, budget=budget)


def _affine_deviation(
    coordinates: tuple[Polynomial, ...],
    corners: tuple[_Point, ...],
    domain: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Outward bound of ``max |F - L|`` for the affine interpolant ``L`` of corners."""
    dimension = len(corners) - 1
    variables = axes(dimension, budget=budget)
    bound = Fraction(0)
    for axis, value in enumerate(coordinates):
        interpolant = sum_polynomials(
            (
                constant(corners[0][axis], dimension, budget=budget),
                *(
                    scale(
                        variables[k],
                        corners[k + 1][axis] - corners[0][axis],
                        budget=budget,
                    )
                    for k in range(dimension)
                ),
            ),
            budget=budget,
        )
        lower, upper = polynomial_bounds(
            add(value, scale(interpolant, -1, budget=budget), budget=budget),
            domain,
            dimension,
            budget=budget,
        )
        bound = max(bound, Fraction(-lower), Fraction(upper))
    return bound


def _proxy_simplices(
    face: Cell, images: dict[int, _Point], /, *, budget: CoordinateEnclosureBudget
) -> tuple[list[tuple[int, ...]], Fraction]:
    """Affine proxy simplices (global vertex rows) and their certified deviation."""
    coordinates = tuple(value for value in face.coordinates if isinstance(value, dict))
    if face.kind != "quadrilateral":
        corners = tuple(images[vertex] for vertex in face.vertices)
        return [face.vertices], _affine_deviation(
            coordinates, corners, face.domain, budget=budget
        )
    rows: list[tuple[int, ...]] = []
    deviation = Fraction(0)
    for local, matrix in _QUAD_HALVES:
        composition = ArgumentComposition(
            affine_arguments(np.zeros((2,), dtype=np.float64), matrix, budget=budget),
            budget=budget,
        )
        half = tuple(composition(value, budget=budget) for value in coordinates)
        row = tuple(face.vertices[k] for k in local)
        rows.append(row)
        deviation = max(
            deviation,
            _affine_deviation(
                tuple(value for value in half if isinstance(value, dict)),
                tuple(images[vertex] for vertex in row),
                "simplex",
                budget=budget,
            ),
        )
    return rows, deviation


def _outside_tube(point: _Point, simplex: tuple[_Point, ...], radius: Fraction) -> bool:
    """Prove ``||point - x||_inf > radius`` throughout the simplex hull box."""
    return any(
        value > max(corner[axis] for corner in simplex) + radius
        or value < min(corner[axis] for corner in simplex) - radius
        for axis, value in enumerate(point)
    )


@final
class _ShellDegrees(StrictModule, NonTrainableState):
    values: tuple[int, ...] = eqx.field(static=True)
    failed_premise: str | None = eqx.field(static=True)


def _shell_degrees(
    state: EmbeddingState,
    faces: list[Cell],
    rows: np.ndarray,
    labels: np.ndarray,
    images: dict[int, _Point],
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _ShellDegrees:
    """Exterior degree ``a_j + c_j - 1`` of every shell, or the unproven hypothesis."""

    shells = int(np.max(labels)) + 1
    representative = mesh_certificates_owner.shell_representatives(rows, labels, shells)
    owner: dict[int, int] = {}
    for face, shell in zip(faces, labels.tolist(), strict=True):
        for vertex in face.vertices:
            if owner.setdefault(vertex, shell) != shell:
                return _ShellDegrees((), "shell_vertex_disjointness")
    if np.any(representative < 0):
        return _ShellDegrees((), "shell_vertex_disjointness")
    simplices: list[list[tuple[int, ...]]] = [[] for _ in range(shells)]
    deviations = [Fraction(0)] * shells
    measures = [Fraction(0)] * shells
    for face, shell in zip(faces, labels.tolist(), strict=True):
        proxies, deviation = _proxy_simplices(face, images, budget=budget)
        simplices[shell].extend(proxies)
        deviations[shell] = max(deviations[shell], deviation)
        measures[shell] += _face_measure(face, budget=budget)
    if any(measure == 0 for measure in measures):
        return _ShellDegrees((), "degenerate_shell")
    pieces = sum(len(value) for value in simplices)
    if shells * pieces > limits.maximum_ray_tests - state.ray_tests:
        return _ShellDegrees((), "exterior_degree_capacity")
    vertices = sorted({vertex for group in simplices for row in group for vertex in row})
    position = {vertex: index for index, vertex in enumerate(vertices)}
    integers, _ = exact_polyhedral_geometry_owner.coordinate_integers(
        np.asarray([images[vertex] for vertex in vertices], dtype=object), budget=budget
    )

    degrees = []
    for shell in range(shells):
        point = images[int(representative[shell])]
        origin = integers[position[int(representative[shell])]]
        winding = 0
        for other in range(shells):
            if other == shell:
                continue
            count = len(simplices[other])
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
                return _ShellDegrees((), "shell_homotopy_separation")
            corners = integers[
                np.asarray(
                    [[position[vertex] for vertex in row] for row in simplices[other]],
                    dtype=np.int64,
                )
            ]
            state.ray_tests += count
            crossings, decided = (
                mesh_certificates_owner.ray_crossings_3d(origin, corners, budget=budget)
                if corners.shape[1] == 3
                else mesh_certificates_owner.ray_crossings_2d(
                    origin, corners, budget=budget
                )
            )
            if not decided:
                return _ShellDegrees((), "shell_winding_predicates")
            winding += crossings
        degrees.append(winding + (1 if measures[shell] > 0 else 0) - 1)
    return _ShellDegrees(tuple(degrees), None)


@dataclass
class _DegreeCounts:
    oriented_cell_count: int
    conforming_entity_count: int = 0
    boundary_facet_count: int = 0
    boundary_pair_count: int = 0
    tangent_cone_contact_count: int = 0
    shell_count: int = 0


@final
class _DegreeOutcome(StrictModule, NonTrainableState):
    status: MappedBoundaryDegreeStatus = eqx.field(static=True)
    failed_premise: str | None = eqx.field(static=True)
    shell_degrees: tuple[int, ...] = eqx.field(static=True)
    maximum_degree: int | None = eqx.field(static=True)


@checked
def certify_boundary_degree_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    cells: list[Cell],
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Decide a mapped volume mesh by the boundary-degree theorem.

    Returns ``True`` when the theorem decided the mesh (embedded, or a proved
    violation recorded as a finding) and ``False`` when a hypothesis is unproven
    and the caller must run its pairwise cell-contact proof.
    """

    dimension = mesh.topological_dimension
    if (
        dimension != mesh.ambient_dimension
        or dimension not in _VOLUME_KINDS
        or not state.clean
        or not state.closed_cell_orientation
    ):
        return False
    record = _DegreeCounts(len(cells))
    outcome = _decide(state, mesh, cells, cell_ids, limits, record, budget=budget)
    state.boundary_degree = mesh_certificates_owner.MappedBoundaryDegreeEvidence(
        outcome.status,
        outcome.failed_premise,
        oriented_cell_count=record.oriented_cell_count,
        conforming_entity_count=record.conforming_entity_count,
        boundary_facet_count=record.boundary_facet_count,
        boundary_pair_count=record.boundary_pair_count,
        tangent_cone_contact_count=record.tangent_cone_contact_count,
        shell_count=record.shell_count,
        shell_exterior_degrees=outcome.shell_degrees,
        maximum_degree=outcome.maximum_degree,
    )
    if outcome.status == "premise_unproven":
        return False
    state.checks.append("mapped_boundary_degree")
    return True


def _decide(
    state: EmbeddingState,
    mesh: CellMesh,
    cells: list[Cell],
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    record: _DegreeCounts,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _DegreeOutcome:

    dimension = mesh.topological_dimension
    if any(cell.kind not in _VOLUME_KINDS[dimension] for cell in cells):
        return _DegreeOutcome("premise_unproven", "boundary_chart_kind", (), None)
    # (b) Shared vertex images and edge coefficients agree; facets were proved
    # by the caller's trace continuity.
    conformity = _shared_entity_conformity(cells, budget=budget)
    images, mismatch = conformity.images, conformity.mismatch
    record.conforming_entity_count = conformity.count
    if mismatch is not None:
        state.add(
            "mapped_trace_mismatch",
            "violated",
            "cell",
            cell_ids[[mismatch.first, mismatch.second]],
        )
        return _DegreeOutcome("violated", "mapped_shared_entity_conformity", (), None)
    # (c) Boundary facets: rational outward restrictions and ridge-linked shells.
    facets = embedding_state_owner.mesh_facets(mesh, budget=budget)
    occurrences = np.flatnonzero(facets.boundary)
    rows = facets.rows[occurrences]
    rows = rows[:, : int(np.max(np.sum(rows >= 0, axis=1), initial=1))]
    owners = facets.cells[occurrences]
    state.boundary_facets = rows.shape[0]
    record.boundary_facet_count = rows.shape[0]
    if not rows.shape[0]:
        return _DegreeOutcome("premise_unproven", "boundary_facets", (), None)
    faces = []
    for owner, row in zip(owners.tolist(), rows, strict=True):
        face = _boundary_face(cells[owner], row[row >= 0], budget=budget)
        if face is None:
            return _DegreeOutcome(
                "premise_unproven", "boundary_rational_expression", (), None
            )
        faces.append(face)
    entities = embedding_state_owner.facet_entity_ids(mesh, rows)
    labels, unbalanced = mesh_certificates_owner.boundary_shells(
        mesh_certificates_owner.Boundary(rows, entities, rows, np.arange(rows.shape[0])),
        dimension,
    )
    state.shells = unbalanced.size
    record.shell_count = unbalanced.size
    if np.any(unbalanced):
        return _DegreeOutcome("premise_unproven", "boundary_shell_manifold", (), None)
    contact = _boundary_injectivity(
        state, faces, owners, cell_ids, limits, record, budget=budget
    )
    if contact is not None:
        return contact
    # (d) Exterior shell degrees. One shell has no other winding and its
    # covered side carries degree at least one, so its exterior degree is 0.
    shell_degrees = (
        _ShellDegrees((0,), None)
        if unbalanced.size == 1
        else _shell_degrees(state, faces, rows, labels, images, limits, budget=budget)
    )
    if shell_degrees.failed_premise is not None:
        return _DegreeOutcome("premise_unproven", shell_degrees.failed_premise, (), None)
    degrees = shell_degrees.values
    state.checks.extend(
        (
            "mapped_closed_cell_orientation",
            "mapped_shared_entity_conformity",
            "mapped_boundary_injectivity",
            "mapped_shell_exterior_degree",
        )
    )
    if any(degrees):
        state.add(
            "mapped_boundary_degree",
            "violated",
            "facet",
            entities[np.isin(labels, np.flatnonzero(np.asarray(degrees)))],
        )
        return _DegreeOutcome(
            "violated", "mapped_shell_exterior_degree", degrees, max(0, 1 + max(degrees))
        )
    return _DegreeOutcome("embedded", None, degrees, 1)


def _boundary_contact(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    remaining: int,
    record: _DegreeCounts,
    *,
    budget: CoordinateEnclosureBudget,
) -> ContactDecision:
    """Try tangent cones before the generic pair proof, retaining both work charges."""
    with budget.temporary_scope():
        contact = (
            affine_mapped_contact_owner.contact_decision(
                first,
                second,
                remaining,
                method="tangent_cone",
                limits=limits,
                budget=budget,
            )
            if set(first.vertices).intersection(second.vertices)
            else None
        )
        work = 0 if contact is None else contact.work
        if contact is not None and contact.status == "separated":
            record.tangent_cone_contact_count += 1
            return contact
        pair = mapped_embedding_owner.pair_decision(
            first, second, limits, remaining - work, tangent_cone=False, budget=budget
        )
        return ContactDecision(pair.status, work + pair.work, check=pair.check)


def _boundary_injectivity(
    state: EmbeddingState,
    faces: list[Cell],
    owners: np.ndarray,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    record: _DegreeCounts,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _DegreeOutcome | None:
    """Check that boundary facets of different cells meet only where shared."""

    boxes = [
        embedding_cells_owner.bounds(
            face,
            np.zeros((face.dimension,), dtype=np.float64),
            np.eye(face.dimension, dtype=np.float64),
            budget=budget,
        )
        for face in faces
    ]
    lower = np.stack([box[0] for box in boxes])
    upper = np.stack([box[1] for box in boxes])
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return _DegreeOutcome("premise_unproven", "boundary_bound_range", (), None)
    first, second, exceeded = embedding_state_owner.candidate_pairs(
        lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
    )
    state.candidate_pairs += first.size
    if exceeded:
        return _DegreeOutcome(
            "premise_unproven", "boundary_candidate_pair_budget", (), None
        )
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        # The proved closed-cell injectivity of the owner separates its own
        # facets beyond their shared entities.
        if owners[a] == owners[b]:
            continue
        record.boundary_pair_count += 1
        remaining = limits.maximum_subdivision_pieces - state.subdivision_pieces
        contact = _boundary_contact(
            faces[a], faces[b], limits, remaining, record, budget=budget
        )
        state.subdivision_pieces += contact.work
        match contact.status:
            case "separated":
                continue
            case "overlap":
                state.add(
                    "mapped_cell_overlap",
                    "violated",
                    "cell",
                    cell_ids[[int(owners[a]), int(owners[b])]],
                )
                return _DegreeOutcome("violated", "mapped_boundary_injectivity", (), None)
            case "unresolved":
                check = contact.unresolved_check()
                if check == "mapped_trace_mismatch":
                    state.add(
                        check,
                        "violated",
                        "cell",
                        cell_ids[[int(owners[a]), int(owners[b])]],
                    )
                    return _DegreeOutcome(
                        "violated", "mapped_boundary_injectivity", (), None
                    )
                return _DegreeOutcome("premise_unproven", check, (), None)
            case _:
                assert_never(contact.status)
    return None
